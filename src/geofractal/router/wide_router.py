"""
geofractal.router.wide_router
====================================

WideRouter - Execution coordinator for wide models with multiple towers.

When multiple towers process the same input, WideRouter provides:
- Automatic tower discovery and registration
- Structure analysis for alignment detection
- Grouped execution (torch.compile handles kernel fusion)
- Cache management across tower tree

Current Execution Path (actively used):
    1. wide_forward() calls _execute_aligned()
    2. Towers grouped by structural signature
    3. Sequential execution per group (torch.compile fuses)
    4. Results collected and returned

Reserved Infrastructure (built, not yet active):
    - StackTracker: Call depth monitoring (for future profiling)
    - TensorPool: Batched accumulation (for manual batching)
    - CacheRegistry: Alignment heuristics (for adaptive batching)
    - WideForwardHook: Forward interception (for transparent batching)
    - BatchedLinearGroup: Operation-level batching (experimental)

Architecture:
    WideRouter (BaseRouter)
    ├── tracker: StackTracker       [reserved]
    ├── pool: TensorPool            [reserved, cleared on reset()]
    ├── registry: CacheRegistry     [reserved]
    ├── analyzer: StructureAnalyzer [used for alignment]
    └── towers                      [the actual tower modules]

Usage:
    class MyCollective(WideRouter):
        def __init__(self, name: str, num_towers: int, dim: int):
            super().__init__(name)

            for i in range(num_towers):
                tower = MyTower(f'tower_{i}', dim)
                self.attach(f'tower_{i}', tower)
                self.register_tower(f'tower_{i}')  # Enable wide execution

            self.attach('fusion', AdaptiveFusion(...))

        def forward(self, x: Tensor) -> Tensor:
            # Automatic batched execution across registered towers
            opinions = self.wide_forward(x)

            # Clear tower caches if they store intermediates
            self.clear_tower_caches()

            return self['fusion'](*opinions.values())

    # Or even simpler - auto-detect towers:
    class AutoCollective(WideRouter):
        def forward(self, x: Tensor) -> Tensor:
            opinions = self.wide_forward(x)  # Auto-finds all towers
            return self['fusion'](*opinions.values())

Cache Management:
    WideRouter inherits BaseRouter's cache system:
    - self._cache: Ephemeral dict for intermediates
    - cache_set/cache_get/cache_clear: Managed lifecycle
    - reset(): Clears cache + pool + tower caches
    - clear_tower_caches(): Just tower caches

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Optional, Dict, List, Tuple, Any, Union,
    Callable, Set, Iterator, TypeVar, Generic
)
from collections import defaultdict
from functools import wraps
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call, vmap, stack_module_state

from geofractal.router.base_router import BaseRouter
from geofractal.router.base_tower import BaseTower


# =============================================================================
# TYPE ALIASES
# =============================================================================

T = TypeVar('T')
OpSignature = str  # e.g., "tower.block_0.attn.qkv"
TensorShape = Tuple[int, ...]


# =============================================================================
# STACK TRACKER
# =============================================================================

@dataclass
class StackFrame:
    """Single frame in the call stack."""
    name: str
    depth: int
    module_type: str
    timestamp: float = field(default_factory=time.perf_counter)


class StackTracker:
    """
    Track call depth across tower traversals.

    Thread-local storage ensures concurrent forwards don't interfere.
    """

    _local = threading.local()

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        """Initialize or reset thread-local state."""
        self._local.depth = 0
        self._local.path = []
        self._local.frames = []
        self._local.active = False

    @property
    def depth(self) -> int:
        return getattr(self._local, 'depth', 0)

    @property
    def path(self) -> List[str]:
        return getattr(self._local, 'path', [])

    @property
    def active(self) -> bool:
        return getattr(self._local, 'active', False)

    @contextmanager
    def track(self):
        """Context manager for tracking a forward pass."""
        self._reset_state()
        self._local.active = True
        try:
            yield self
        finally:
            self._local.active = False

    @contextmanager
    def frame(self, name: str, module: nn.Module = None):
        """Push a frame onto the stack."""
        if not self.active:
            yield
            return

        module_type = type(module).__name__ if module else 'unknown'
        frame = StackFrame(
            name=name,
            depth=self._local.depth,
            module_type=module_type,
        )

        self._local.depth += 1
        self._local.path.append(name)
        self._local.frames.append(frame)

        try:
            yield frame
        finally:
            self._local.depth -= 1
            self._local.path.pop()

    def signature(self) -> OpSignature:
        """Current operation signature based on path."""
        return '.'.join(self._local.path)

    def structural_signature(self) -> OpSignature:
        """
        Signature based on structure, not names.

        Replaces tower-specific names with generic placeholders
        so structurally identical operations align.
        """
        if not self._local.frames:
            return ''

        # Use module types and depths, not specific names
        parts = []
        for frame in self._local.frames:
            parts.append(f"{frame.module_type}@{frame.depth}")
        return '.'.join(parts)


# =============================================================================
# TENSOR POOL
# =============================================================================

@dataclass
class PendingOp:
    """An operation waiting to be batched."""
    input: Tensor
    callback: Callable[[Tensor], Tensor]
    source_id: str  # Which tower this came from
    shape: TensorShape


class TensorPool:
    """
    Pre-allocated tensor buffers for batched execution.

    Accumulates operations with matching signatures, then executes
    them in a single batched call.
    """

    def __init__(
        self,
        max_pool_bytes: int = 1024 * 1024 * 512,  # 512MB default
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.max_pool_bytes = max_pool_bytes
        self.device = device
        self.dtype = dtype

        # Signature -> list of pending operations
        self.pending: Dict[OpSignature, List[PendingOp]] = defaultdict(list)

        # Signature -> pre-allocated buffer
        self.pools: Dict[OpSignature, Tensor] = {}

        # Results waiting to be collected
        self.results: Dict[str, Dict[OpSignature, Tensor]] = defaultdict(dict)

        # Accumulation mode flag
        self._accumulating = False

        # Statistics
        self.stats = {
            'ops_accumulated': 0,
            'ops_batched': 0,
            'ops_sequential': 0,
            'batches_executed': 0,
            'bytes_pooled': 0,
        }

    @property
    def accumulating(self) -> bool:
        return self._accumulating

    @contextmanager
    def accumulation_mode(self):
        """Context manager for accumulation phase."""
        self._accumulating = True
        self.pending.clear()
        self.results.clear()
        try:
            yield
        finally:
            self._accumulating = False

    def accumulate(
        self,
        signature: OpSignature,
        source_id: str,
        input: Tensor,
        callback: Callable[[Tensor], Tensor],
    ) -> None:
        """
        Accumulate an operation for batched execution.

        Args:
            signature: Operation signature for grouping
            source_id: Which tower/source this came from
            input: Input tensor
            callback: Function to apply (e.g., module.forward)
        """
        if not self._accumulating:
            raise RuntimeError("accumulate() called outside accumulation_mode()")

        op = PendingOp(
            input=input,
            callback=callback,
            source_id=source_id,
            shape=tuple(input.shape),
        )
        self.pending[signature].append(op)
        self.stats['ops_accumulated'] += 1

    def can_batch(self, signature: OpSignature) -> bool:
        """Check if operations for this signature can be batched."""
        ops = self.pending.get(signature, [])
        if len(ops) < 2:
            return False

        # All shapes must match
        shapes = {op.shape for op in ops}
        return len(shapes) == 1

    def flush_signature(self, signature: OpSignature) -> Dict[str, Tensor]:
        """
        Execute all pending ops for a signature.

        Returns dict mapping source_id -> result tensor.
        """
        ops = self.pending.pop(signature, [])
        if not ops:
            return {}

        results = {}

        if len(ops) == 1:
            # Single op - just execute
            op = ops[0]
            results[op.source_id] = op.callback(op.input)
            self.stats['ops_sequential'] += 1

        elif self.can_batch(signature):
            # Batch execute
            # Stack inputs: [N, ...] where N = num ops
            inputs = torch.stack([op.input for op in ops], dim=0)

            # Get a representative callback (they should all be the same op)
            # Execute on batched input
            callback = ops[0].callback

            # For batched execution, we need to handle the batch dimension
            N = len(ops)
            B = ops[0].input.shape[0]

            # Reshape: [N, B, ...] -> [N*B, ...]
            flat_shape = (N * B,) + ops[0].shape[1:]
            inputs_flat = inputs.reshape(flat_shape)

            # Execute
            outputs_flat = callback(inputs_flat)

            # Reshape back: [N*B, ...] -> [N, B, ...]
            out_shape = (N, B) + outputs_flat.shape[1:]
            outputs = outputs_flat.reshape(out_shape)

            # Scatter results
            for i, op in enumerate(ops):
                results[op.source_id] = outputs[i]

            self.stats['ops_batched'] += N
            self.stats['batches_executed'] += 1

        else:
            # Shapes don't match - sequential fallback
            for op in ops:
                results[op.source_id] = op.callback(op.input)
                self.stats['ops_sequential'] += 1

        return results

    def flush_all(self) -> Dict[str, Dict[OpSignature, Tensor]]:
        """
        Execute all pending operations.

        Returns nested dict: source_id -> signature -> result
        """
        all_results: Dict[str, Dict[OpSignature, Tensor]] = defaultdict(dict)

        for signature in list(self.pending.keys()):
            sig_results = self.flush_signature(signature)
            for source_id, result in sig_results.items():
                all_results[source_id][signature] = result

        return dict(all_results)

    def clear(self):
        """Clear all pending operations and pools."""
        self.pending.clear()
        self.results.clear()
        self.pools.clear()


# =============================================================================
# CACHE REGISTRY
# =============================================================================

@dataclass
class OpSpec:
    """Specification for an operation type."""
    signature: OpSignature
    module_type: str
    input_shape: Optional[TensorShape] = None
    output_shape: Optional[TensorShape] = None
    param_count: int = 0
    avg_time_ms: float = 0.0
    call_count: int = 0


class CacheRegistry:
    """
    Registry of cacheable operations and their alignments.

    Tracks which operations can be batched together and maintains
    heuristics for when batching is beneficial.
    """

    def __init__(self, min_batch_benefit_ms: float = 0.1):
        # Signature -> OpSpec
        self.specs: Dict[OpSignature, OpSpec] = {}

        # Groups of alignable signatures
        self.alignment_groups: Dict[str, Set[OpSignature]] = defaultdict(set)

        # Timing benchmarks
        self.benchmarks: Dict[OpSignature, List[float]] = defaultdict(list)

        # Minimum time saved to justify batching overhead
        self.min_batch_benefit_ms = min_batch_benefit_ms

        # Cache of should_batch decisions
        self._batch_decisions: Dict[OpSignature, bool] = {}

    def register(
        self,
        signature: OpSignature,
        module: nn.Module,
        input_shape: TensorShape = None,
    ) -> OpSpec:
        """Register an operation specification."""
        spec = OpSpec(
            signature=signature,
            module_type=type(module).__name__,
            input_shape=input_shape,
            param_count=sum(p.numel() for p in module.parameters()),
        )
        self.specs[signature] = spec

        # Auto-group by structural similarity
        structural_key = self._structural_key(signature, module)
        self.alignment_groups[structural_key].add(signature)

        return spec

    def _structural_key(self, signature: OpSignature, module: nn.Module) -> str:
        """Generate key for structural grouping."""
        # Group by module type and parameter count
        param_count = sum(p.numel() for p in module.parameters())
        return f"{type(module).__name__}_{param_count}"

    def find_alignable(self, signature: OpSignature) -> Set[OpSignature]:
        """Find signatures that can be batched with this one."""
        for group in self.alignment_groups.values():
            if signature in group:
                return group - {signature}
        return set()

    def record_timing(self, signature: OpSignature, time_ms: float):
        """Record execution time for heuristics."""
        self.benchmarks[signature].append(time_ms)

        # Update spec
        if signature in self.specs:
            spec = self.specs[signature]
            spec.call_count += 1
            # Running average
            n = spec.call_count
            spec.avg_time_ms = spec.avg_time_ms * (n - 1) / n + time_ms / n

        # Invalidate cached decision
        self._batch_decisions.pop(signature, None)

    def should_batch(self, signature: OpSignature, num_ops: int = 2) -> bool:
        """
        Heuristic: should we batch this operation?

        Considers:
        - Number of operations to batch
        - Historical timing data
        - Overhead of batching
        """
        # Check cache
        cache_key = f"{signature}_{num_ops}"
        if cache_key in self._batch_decisions:
            return self._batch_decisions[cache_key]

        # Default: batch if we have multiple ops
        decision = num_ops >= 2

        # Refine with timing data if available
        if signature in self.specs:
            spec = self.specs[signature]
            if spec.call_count > 10:
                # Estimate time saved
                sequential_time = spec.avg_time_ms * num_ops
                # Assume batched is ~40% faster (conservative)
                batched_time = spec.avg_time_ms * num_ops * 0.6
                benefit = sequential_time - batched_time
                decision = benefit > self.min_batch_benefit_ms

        self._batch_decisions[cache_key] = decision
        return decision

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            'registered_ops': len(self.specs),
            'alignment_groups': len(self.alignment_groups),
            'avg_group_size': (
                sum(len(g) for g in self.alignment_groups.values()) /
                max(len(self.alignment_groups), 1)
            ),
        }


# =============================================================================
# STRUCTURE ANALYZER
# =============================================================================

class StructureAnalyzer:
    """
    Analyze tower structures to discover alignment opportunities.

    Walks through towers and identifies operations that can be
    batched together based on structural similarity.
    """

    def __init__(self):
        self.structures: Dict[str, List[Tuple[str, str, int]]] = {}

    def analyze(self, name: str, module: nn.Module, depth: int = 0) -> List[Tuple[str, str, int]]:
        """
        Recursively analyze module structure.

        Returns list of (path, module_type, param_count) tuples.
        """
        structure = []

        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        structure.append((name, type(module).__name__, param_count))

        for child_name, child in module.named_children():
            child_path = f"{name}.{child_name}"
            structure.extend(self.analyze(child_path, child, depth + 1))

        self.structures[name] = structure
        return structure

    def find_alignments(self, tower_names: List[str]) -> Dict[int, List[List[str]]]:
        """
        Find aligned operations across towers.

        Returns dict mapping depth -> list of aligned tower.path signatures.
        """
        if not tower_names or not all(n in self.structures for n in tower_names):
            return {}

        # Use regular dict, not defaultdict (compile-safe)
        alignments: Dict[int, List[List[str]]] = {}

        # Get structures
        structures = [self.structures[n] for n in tower_names]

        # Find common structure by position
        min_len = min(len(s) for s in structures)

        for i in range(min_len):
            # Check if all towers have same structure at position i
            entries = [s[i] for s in structures]
            types = {e[1] for e in entries}
            params = {e[2] for e in entries}

            if len(types) == 1 and len(params) == 1:
                # Aligned! Collect the paths
                paths = [e[0] for e in entries]
                if i not in alignments:
                    alignments[i] = []
                alignments[i].append(paths)

        return alignments


# =============================================================================
# BATCHED OPERATION GROUPS
# =============================================================================

class BatchedLinearGroup(nn.Module):
    """
    Executes multiple Linear layers in parallel via stacked weights.

    Instead of N separate matmuls, does ONE batched matmul.

    This is where actual speedup happens.
    """

    def __init__(self, linears: List[nn.Linear], names: List[str]):
        super().__init__()

        self.n_layers = len(linears)
        self.names = names

        # Verify all have same shape
        shapes = {(l.in_features, l.out_features) for l in linears}
        if len(shapes) != 1:
            raise ValueError(f"Linear layers must have same shape, got {shapes}")

        self.in_features = linears[0].in_features
        self.out_features = linears[0].out_features
        self.has_bias = linears[0].bias is not None

        # Stack weights: [N, out, in]
        weights = torch.stack([l.weight.data for l in linears], dim=0)
        self.register_buffer('_stacked_weights', weights)

        # Stack biases if present: [N, out]
        if self.has_bias:
            biases = torch.stack([l.bias.data for l in linears], dim=0)
            self.register_buffer('_stacked_biases', biases)

    def forward(self, x: Tensor) -> Tensor:
        """
        Batched forward.

        Args:
            x: [N, B, ..., in_features] - N copies of input for N layers

        Returns:
            [N, B, ..., out_features] - N outputs from N layers
        """
        # x: [N, B, ..., in]
        # W: [N, out, in]
        # out = x @ W.T -> [N, B, ..., out]

        # Use einsum for clarity and efficiency
        out = torch.einsum('n...i,noi->n...o', x, self._stacked_weights)

        if self.has_bias:
            # Expand bias for broadcasting: [N, 1, ..., out]
            bias_shape = [self.n_layers] + [1] * (x.dim() - 2) + [self.out_features]
            out = out + self._stacked_biases.view(*bias_shape)

        return out

    def forward_and_scatter(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward and return dict mapping name -> output.

        Args:
            x: [B, ..., in_features] - single input

        Returns:
            Dict[name, Tensor] where each tensor is [B, ..., out_features]
        """
        # Expand input: [B, ..., in] -> [N, B, ..., in]
        x_expanded = x.unsqueeze(0).expand(self.n_layers, *x.shape)

        # Batched forward
        out = self.forward(x_expanded)  # [N, B, ..., out]

        # Scatter to dict
        return {name: out[i] for i, name in enumerate(self.names)}


class BatchedModuleGroup(nn.Module):
    """
    Wraps multiple identical modules for batched execution.

    For modules that aren't simple Linear layers, we use a different
    strategy: execute in parallel using the batch dimension.
    """

    def __init__(self, modules: List[nn.Module], names: List[str]):
        super().__init__()

        self.n_modules = len(modules)
        self.names = names
        self.modules_list = nn.ModuleList(modules)

        # Check if we can do true batched execution (all same type)
        types = {type(m).__name__ for m in modules}
        self.homogeneous = len(types) == 1
        self.module_type = list(types)[0] if self.homogeneous else 'mixed'

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward with batch dimension encoding module index.

        Args:
            x: [N*B, ...] where N is number of modules, B is batch size

        Returns:
            [N*B, ...] outputs
        """
        B_per_module = x.shape[0] // self.n_modules

        outputs = []
        for i, module in enumerate(self.modules_list):
            xi = x[i * B_per_module:(i + 1) * B_per_module]
            outputs.append(module(xi))

        return torch.cat(outputs, dim=0)

    def forward_parallel(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Execute all modules on same input in parallel.

        This uses CUDA's ability to overlap independent operations.

        Args:
            x: [B, ...] single input

        Returns:
            Dict[name, output]
        """
        # For truly parallel execution, we'd use CUDA streams
        # For now, sequential but GPU can overlap
        results = {}
        for name, module in zip(self.names, self.modules_list):
            results[name] = module(x)
        return results


class VMapTowerGroup(nn.Module):
    """
    Vectorized tower execution using torch.func.vmap.

    Instead of N sequential tower calls, executes all towers in ONE
    vectorized operation by stacking parameters and using vmap.

    This provides true batching - not just a for loop.
    """

    def __init__(self, towers: List[nn.Module], names: List[str]):
        super().__init__()
        self.names = names
        self.n = len(towers)

        # Store towers for parameter tracking
        self.towers = nn.ModuleList(towers)

        # Cached stacked state (use different names to avoid nn.Module collision)
        self._stacked_params: Optional[Dict[str, Tensor]] = None
        self._stacked_buffs: Optional[Dict[str, Tensor]] = None
        self._stale = True

    def _stack_state(self) -> None:
        """Stack parameters and buffers from all towers."""
        if not self._stale:
            return
        self._stacked_params, self._stacked_buffs = stack_module_state(list(self.towers))
        self._stale = False

    def _invalidate(self) -> None:
        """Mark cached state as stale."""
        self._stale = True
        self._stacked_params = None
        self._stacked_buffs = None

    def train(self, mode: bool = True):
        result = super().train(mode)
        self._invalidate()
        return result

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self._invalidate()
        return result

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Vectorized forward through all towers.

        Args:
            x: Input tensor [B, ...] - same input to all towers

        Returns:
            Dict mapping name -> output tensor [B, ...]
        """
        if self.n == 0:
            return {}

        if self.n == 1:
            out = self.towers[0](x)
            return {self.names[0]: out[0] if isinstance(out, tuple) else out}

        # Ensure state is stacked
        self._stack_state()

        # Reference model for functional_call
        base = self.towers[0]

        # Single-tower forward via functional_call
        def single_forward(params, buffers, data):
            return functional_call(base, (params, buffers), (data,))

        # vmap over params/buffers (dim 0), broadcast input
        vmapped_forward = vmap(single_forward, in_dims=(0, 0, None))

        # Execute all towers at once: [N, B, ...]
        outputs = vmapped_forward(self._stacked_params, self._stacked_buffs, x)

        # Handle tuple outputs (e.g., (opinion, features))
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Scatter to dict
        return {name: outputs[i] for i, name in enumerate(self.names)}


# =============================================================================

class WideForwardHook:
    """
    Hook that intercepts forward calls for pooled execution.

    Installed on tower modules to capture their forward calls
    during wide execution mode.
    """

    def __init__(
        self,
        pool: TensorPool,
        tracker: StackTracker,
        registry: CacheRegistry,
        source_id: str,
    ):
        self.pool = pool
        self.tracker = tracker
        self.registry = registry
        self.source_id = source_id
        self._handles: List[Any] = []

    def install(self, module: nn.Module, name: str = ''):
        """Install hooks on module and children."""
        # Register pre-forward hook
        handle = module.register_forward_pre_hook(
            lambda m, inputs: self._pre_forward(m, inputs, name)
        )
        self._handles.append(handle)

        # Recurse to children
        for child_name, child in module.named_children():
            child_path = f"{name}.{child_name}" if name else child_name
            self.install(child, child_path)

    def _pre_forward(self, module: nn.Module, inputs: Tuple, name: str):
        """Pre-forward hook - potentially accumulate for batching."""
        if not self.pool.accumulating:
            return None

        # Get structural signature
        with self.tracker.frame(name, module):
            sig = self.tracker.structural_signature()

        # For now, we don't modify inputs - just track
        # Full interception would require more complex hook management
        return None

    def remove(self):
        """Remove all installed hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# =============================================================================
# WIDE ROUTER
# =============================================================================

class WideRouter(BaseRouter):
    """
    Router with autonomous caching for wide model execution.

    Automatically discovers structural alignment across registered
    towers and batches aligned operations for efficiency.

    Attributes:
        tracker: Stack depth tracker
        pool: Tensor pool for batched execution
        registry: Cache registry for alignment tracking
        analyzer: Structure analyzer for alignment discovery
    """

    def __init__(
        self,
        name: str,
        uuid: Optional[str] = None,
        strict: bool = False,
        auto_discover: bool = True,
        pool_size_mb: int = 512,
        **kwargs,
    ):
        """
        Initialize WideRouter.

        Args:
            name: Router name
            uuid: Unique identifier
            strict: Hardware control strictness
            auto_discover: Auto-discover towers on first wide_forward
            pool_size_mb: Tensor pool size in MB
        """
        super().__init__(name, uuid, strict=strict, **kwargs)

        # Core infrastructure
        self.tracker = StackTracker()
        self.pool = TensorPool(max_pool_bytes=pool_size_mb * 1024 * 1024)
        self.registry = CacheRegistry()
        self.analyzer = StructureAnalyzer()

        # Tower tracking
        self.objects['_tower_names'] = []
        self.objects['_tower_signatures'] = {}  # name -> cached signature
        self.objects['_alignments'] = {}
        self.objects['_analyzed'] = False
        self.objects['_auto_discover'] = auto_discover

        # Wide execution stats
        self.objects['wide_stats'] = {
            'wide_forwards': 0,
            'towers_executed': 0,
            'alignment_hits': 0,
        }

        # vmap tower groups: signature -> VMapTowerGroup
        self.objects['_vmap_groups'] = {}

    # =========================================================================
    # TOWER REGISTRATION
    # =========================================================================

    @property
    def tower_names(self) -> List[str]:
        """Names of registered towers."""
        # Return cached list directly - discovery happens at init or explicit call
        return self.objects['_tower_names']

    def discover_towers(self) -> List[str]:
        """
        Explicitly discover and register towers.

        Call this before compile() to ensure all towers are registered.
        """
        if self.objects.get('_auto_discover', True):
            self._auto_discover_towers()
        return self.objects['_tower_names']

    @property
    def towers(self) -> Dict[str, BaseTower]:
        """Dict of registered towers."""
        return {name: self[name] for name in self.tower_names}

    def register_tower(self, name: str) -> None:
        """
        Register a tower for wide execution.

        Args:
            name: Name of attached tower component
        """
        if name not in self.components:
            raise KeyError(f"No component '{name}' attached")

        if not isinstance(self.components[name], (BaseTower, nn.Module)):
            raise TypeError(f"Component '{name}' is not a tower/module")

        if name not in self.objects['_tower_names']:
            self.objects['_tower_names'].append(name)
            # Cache signature at registration time
            self.objects['_tower_signatures'][name] = self._compute_tower_signature(self.components[name])
            self.objects['_analyzed'] = False  # Require re-analysis
            self.objects['_vmap_groups'].clear()  # Invalidate vmap cache

    def unregister_tower(self, name: str) -> None:
        """Remove tower from wide execution."""
        if name in self.objects['_tower_names']:
            self.objects['_tower_names'].remove(name)
            self.objects['_tower_signatures'].pop(name, None)
            self.objects['_analyzed'] = False
            self.objects['_vmap_groups'].clear()  # Invalidate vmap cache

    @torch.compiler.disable
    def _auto_discover_towers(self) -> None:
        """Auto-discover towers from attached components."""
        for name, component in self.components.items():
            if isinstance(component, BaseTower):
                if name not in self.objects['_tower_names']:
                    self.objects['_tower_names'].append(name)
                    # Cache signature at discovery time
                    self.objects['_tower_signatures'][name] = self._compute_tower_signature(component)

    # =========================================================================
    # STRUCTURE ANALYSIS (isolated from compile path)
    # =========================================================================

    @torch.compiler.disable
    def analyze_structure(self) -> Dict[int, List[List[str]]]:
        """
        Analyze tower structures and find alignments.

        Decorated with @torch.compiler.disable to prevent dynamo issues
        with defaultdict and other Python constructs.

        Returns alignment map: depth -> list of aligned path groups
        """
        # Ensure towers are discovered
        if self.objects.get('_auto_discover', True) and not self.objects['_tower_names']:
            self._auto_discover_towers()

        if not self.tower_names:
            return {}

        # Analyze each tower
        for name in self.tower_names:
            tower = self[name]
            self.analyzer.analyze(name, tower)

        # Find alignments
        alignments = self.analyzer.find_alignments(self.tower_names)
        self.objects['_alignments'] = alignments
        self.objects['_analyzed'] = True

        return alignments

    # =========================================================================
    # WIDE FORWARD
    # =========================================================================

    def wide_forward(
        self,
        x: Tensor,
        tower_names: List[str] = None,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Execute towers with automatic batching of aligned operations.

        Args:
            x: Input tensor [B, ...] - same input to all towers
            tower_names: Specific towers to execute (None = all registered)
            mask: Optional attention mask

        Returns:
            Dict mapping tower name -> output tensor
        """
        # Determine which towers to execute
        names = tower_names or self.tower_names
        if not names:
            return {}

        # Analyze structure if needed (compile-disabled)
        if not self.objects.get('_analyzed', False):
            self.analyze_structure()

        # Execute - this is the hot path
        outputs = self._execute_aligned(x, names, mask)

        return outputs

    @torch.compiler.disable
    def _update_wide_stats(self, towers_executed: int, alignment_hits: int):
        """Update stats outside compile path."""
        self.objects['wide_stats']['wide_forwards'] += 1
        self.objects['wide_stats']['towers_executed'] += towers_executed
        self.objects['wide_stats']['alignment_hits'] += alignment_hits

    def _execute_aligned(
        self,
        x: Tensor,
        tower_names: List[str],
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Execute towers with alignment-aware batching.

        For towers with identical structure, we batch at the tensor level.
        """
        outputs = {}

        # Group towers by structure signature (use regular dict)
        structure_groups: Dict[str, List[str]] = {}

        for name in tower_names:
            sig = self._tower_signature(name)  # Uses cached signature
            if sig not in structure_groups:
                structure_groups[sig] = []
            structure_groups[sig].append(name)

        # Process each structure group
        for sig, group_names in structure_groups.items():
            if len(group_names) == 1:
                # Single tower - direct execution
                name = group_names[0]
                tower = self[name]
                if mask is not None:
                    out = tower(x, mask=mask)
                else:
                    out = tower(x)

                # Handle tuple returns (opinion, features)
                if isinstance(out, tuple):
                    out = out[0]
                outputs[name] = out

            else:
                # Multiple aligned towers - batched execution
                group_outputs = self._batched_tower_forward(
                    x, group_names, mask
                )
                outputs.update(group_outputs)

        return outputs

    def _tower_signature(self, name: str) -> str:
        """Get cached structural signature for a tower by name."""
        # Fast path: lookup cached signature
        sig = self.objects['_tower_signatures'].get(name)
        if sig is not None:
            return sig
        # Fallback: compute and cache (shouldn't happen if properly registered)
        tower = self[name]
        sig = self._compute_tower_signature(tower)
        self.objects['_tower_signatures'][name] = sig
        return sig

    @torch.compiler.disable
    def _compute_tower_signature(self, tower: nn.Module) -> str:
        """Compute structural signature for a tower (cached at registration)."""
        # Use class name + total params as signature
        param_count = sum(p.numel() for p in tower.parameters())

        # Also include stage count if it's a BaseTower
        if isinstance(tower, BaseTower):
            return f"{type(tower).__name__}_{len(tower.stages)}_{param_count}"

        return f"{type(tower).__name__}_{param_count}"

    def _batched_tower_forward(
        self,
        x: Tensor,
        tower_names: List[str],
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Execute multiple aligned towers using vmap.

        Uses torch.func.vmap to truly vectorize execution across towers,
        replacing N sequential calls with ONE vectorized operation.

        VMapTowerGroup instances are cached by signature for reuse.
        """
        if not tower_names:
            return {}

        # Get signature for this group (all towers in group have same sig)
        sig = self._tower_signature(tower_names[0])

        # Get or create VMapTowerGroup for this signature
        vmap_groups = self.objects['_vmap_groups']

        # Use frozenset of names as cache key (order-independent)
        cache_key = (sig, frozenset(tower_names))

        if cache_key not in vmap_groups:
            # Build new VMapTowerGroup
            towers = [self[name] for name in tower_names]
            vmap_group = VMapTowerGroup(towers, list(tower_names))

            # Move to same device as input
            vmap_group = vmap_group.to(x.device)
            vmap_groups[cache_key] = vmap_group

        # Execute via vmap
        return vmap_groups[cache_key](x)

    # =========================================================================
    # COMPILATION
    # =========================================================================

    def compile(self, prepare_and_compile=True, **kwargs) -> 'WideRouter':
        """
        Compile the router for optimized execution.

        This is the primary optimization path. torch.compile automatically
        discovers aligned operations and fuses kernels.

        Note: Call analyze_structure() before compile() if you want
        alignment data, or use prepare_and_compile() for convenience.

        Args:
            prepare_and_compile: If True, analyze structure before compiling.
            **kwargs: Passed to torch.compile (mode, fullgraph, etc.)

        Returns:
            Compiled router
        """
        if prepare_and_compile:
            self.analyze_structure()
        return torch.compile(self, **kwargs)

    def prepare_and_compile(self, **kwargs) -> 'WideRouter':
        """
        Analyze structure then compile.

        Convenience method that ensures analysis happens before
        torch.compile traces the forward pass.

        Args:
            **kwargs: Passed to torch.compile

        Returns:
            Compiled router with structure pre-analyzed
        """
        self.analyze_structure()
        return self.compile(**kwargs)

    # =========================================================================
    # BATCHED LINEAR EXECUTION (Operation-level batching)
    # =========================================================================

    def build_batched_linears(self) -> Dict[str, BatchedLinearGroup]:
        """
        Build BatchedLinearGroups from aligned Linear layers.

        Call this after structure analysis to enable operation-level batching.
        Returns dict of path -> BatchedLinearGroup.
        """
        if not self.objects['_analyzed']:
            self.analyze_structure()

        alignments = self.objects.get('_alignments', {})
        batched_groups = {}

        for depth, path_groups in alignments.items():
            for paths in path_groups:
                # Try to extract Linear layers at these paths
                linears = []
                names = []

                for path in paths:
                    try:
                        module = self._get_module_by_path(path)
                        if isinstance(module, nn.Linear):
                            linears.append(module)
                            names.append(path)
                    except (KeyError, AttributeError):
                        continue

                if len(linears) >= 2:
                    # Check all same shape
                    shapes = {(l.in_features, l.out_features) for l in linears}
                    if len(shapes) == 1:
                        group_name = f"batched_linear_{depth}"
                        try:
                            batched_groups[group_name] = BatchedLinearGroup(linears, names)
                        except ValueError:
                            pass

        self.objects['_batched_linears'] = batched_groups
        return batched_groups

    def _get_module_by_path(self, path: str) -> nn.Module:
        """Get module by dot-separated path."""
        parts = path.split('.')
        module = self[parts[0]]  # First part is tower name

        for part in parts[1:]:
            if hasattr(module, part):
                module = getattr(module, part)
            elif hasattr(module, 'components') and part in module.components:
                module = module.components[part]
            elif hasattr(module, 'stages') and part.isdigit():
                module = module.stages[int(part)]
            else:
                raise AttributeError(f"Cannot resolve path: {path} at {part}")

        return module

    # =========================================================================
    # FORWARD (ABSTRACT - subclass must implement)
    # =========================================================================

    def forward(self, *args, **kwargs):
        """
        Subclass must implement forward.

        Typical pattern:
            def forward(self, x):
                opinions = self.wide_forward(x)
                return self['fusion'](*opinions.values())
        """
        raise NotImplementedError(
            "WideRouter subclass must implement forward(). "
            "Use self.wide_forward(x) to execute registered towers."
        )

    # =========================================================================
    # CLEANUP AND LIFECYCLE
    # =========================================================================

    def reset(self) -> None:
        """
        Clear transient state including tower caches.

        Clears:
        - Base router cache (via super)
        - TensorPool pending/results
        - Tower caches (for all registered towers)
        - VMap tower groups cache
        """
        super().reset()  # Clears self._cache and recurses to components
        self.pool.clear()
        self.objects['_vmap_groups'].clear()

    def clear_tower_caches(self) -> None:
        """
        Clear ephemeral caches on all registered towers.

        Call this after wide_forward if towers cache intermediate tensors.
        Note: ConfigurableCollective and ConvTowerCollective do this automatically.
        """
        for name in self.tower_names:
            tower = self.get(name)
            if tower is not None and hasattr(tower, 'cache_clear'):
                tower.cache_clear()

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_wide_stats(self) -> Dict[str, Any]:
        """Get wide execution statistics."""
        return {
            **self.objects['wide_stats'],
            'pool_stats': self.pool.stats,
            'registry_stats': self.registry.get_stats(),
            'registered_towers': len(self.tower_names),
            'alignments': self.objects.get('_alignments', {}),
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.objects['wide_stats'] = {
            'wide_forwards': 0,
            'towers_executed': 0,
            'alignment_hits': 0,
        }
        self.pool.stats = {k: 0 for k in self.pool.stats}

    def __repr__(self) -> str:
        tower_count = len(self.tower_names)
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"towers={tower_count}, "
            f"analyzed={self.objects['_analyzed']}"
            f")"
        )


# =============================================================================
# CONVENIENCE DECORATORS
# =============================================================================

def cache_similar(group: str = None):
    """
    Decorator to mark a method for cache-similar execution.

    When multiple towers call methods with the same cache_similar group,
    their executions can be batched together.

    Args:
        group: Cache group name. Auto-generated if None.

    Usage:
        class MyBlock(TorchComponent):
            @cache_similar(group='qkv_proj')
            def forward(self, x):
                return self.qkv(x)
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            # For now, just execute normally
            # Full implementation would check if we're in wide mode
            # and accumulate to pool if so
            return fn(self, *args, **kwargs)

        wrapper._cache_group = group or f"{fn.__module__}.{fn.__qualname__}"
        return wrapper

    return decorator


def cache_barrier(fn):
    """
    Decorator to mark a synchronization point.

    Forces flush of all pending cached operations before executing.

    Usage:
        class MyTower(BaseTower):
            @cache_barrier
            def forward(self, x):
                # All cached ops from stages will be flushed first
                return self.process(x)
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # For now, just execute normally
        # Full implementation would flush pool here
        return fn(self, *args, **kwargs)

    wrapper._is_cache_barrier = True
    return wrapper


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    import time

    torch.set_float32_matmul_precision('high')

    print("=" * 60)
    print("WideRouter Test - Compile First")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Test Towers
    # -------------------------------------------------------------------------

    class SimpleTower(BaseTower):
        def __init__(self, name: str, dim: int, depth: int = 2):
            super().__init__(name, strict=False)

            for i in range(depth):
                self.append(nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Linear(dim * 2, dim),
                ))

            self.attach('norm', nn.LayerNorm(dim))

        def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
            for stage in self.stages:
                x = x + stage(x)
            return self['norm'](x)

    # -------------------------------------------------------------------------
    # Collectives
    # -------------------------------------------------------------------------

    class TestCollective(WideRouter):
        def __init__(self, name: str, num_towers: int, dim: int):
            super().__init__(name, strict=False, auto_discover=True)

            for i in range(num_towers):
                self.attach(f'tower_{i}', SimpleTower(f'tower_{i}', dim, depth=2))

            # Discover towers now (for compile safety)
            self.discover_towers()

        def forward(self, x: Tensor) -> Tensor:
            opinions = self.wide_forward(x)
            return torch.stack(list(opinions.values()), dim=0).mean(dim=0)

    class SequentialCollective(BaseRouter):
        def __init__(self, name: str, num_towers: int, dim: int):
            super().__init__(name, strict=False)

            self._tower_names = []
            for i in range(num_towers):
                self.attach(f'tower_{i}', SimpleTower(f'tower_{i}', dim, depth=2))
                self._tower_names.append(f'tower_{i}')

        def forward(self, x: Tensor) -> Tensor:
            opinions = [self[n](x) for n in self._tower_names]
            return torch.stack(opinions, dim=0).mean(dim=0)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    NUM_TOWERS = 8
    DIM = 256
    B, L = 32, 64

    baseline = SequentialCollective('baseline', NUM_TOWERS, DIM)
    wide = TestCollective('wide', NUM_TOWERS, DIM)

    baseline.network_to(device=device)
    wide.network_to(device=device)

    x = torch.randn(B, L, DIM, device=device)

    print(f"\nConfig: {NUM_TOWERS} towers, dim={DIM}, batch={B}x{L}")
    print(f"Params: {sum(p.numel() for p in wide.parameters()):,}")

    # -------------------------------------------------------------------------
    # Compile
    # -------------------------------------------------------------------------

    print("\nPre-analyzing and compiling...")

    # Pre-analyze WideRouter before compile
    wide.analyze_structure()

    compiled_baseline = torch.compile(baseline)
    compiled_wide = wide.compile()

    # Warmup
    for _ in range(5):
        _ = baseline(x)
        _ = wide(x)
        _ = compiled_baseline(x)
        _ = compiled_wide(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # -------------------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------------------

    def bench(fn, x, iters=100):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fn(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    ms_baseline = bench(baseline, x)
    ms_wide = bench(wide, x)
    ms_compiled_baseline = bench(compiled_baseline, x)
    ms_compiled_wide = bench(compiled_wide, x)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Method':<25} {'Time (ms)':<12} {'vs Baseline':<12}")
    print("-" * 49)
    print(f"{'Baseline (eager)':<25} {ms_baseline:<12.2f} {'1.00x':<12}")
    print(f"{'WideRouter (eager)':<25} {ms_wide:<12.2f} {ms_baseline/ms_wide:.2f}x")
    print(f"{'Baseline (compiled)':<25} {ms_compiled_baseline:<12.2f} {ms_baseline/ms_compiled_baseline:.2f}x")
    print(f"{'WideRouter (compiled)':<25} {ms_compiled_wide:<12.2f} {ms_baseline/ms_compiled_wide:.2f}x")

    # -------------------------------------------------------------------------
    # Scaling test
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("SCALING (compiled WideRouter)")
    print("=" * 60)

    for n_towers in [4, 8, 16, 32]:
        w = TestCollective(f'scale_{n_towers}', n_towers, DIM)
        w.network_to(device=device)

        # Pre-analyze BEFORE compiling (avoids dynamo issues)
        w.analyze_structure()

        wc = w.compile()

        # Warmup
        for _ in range(5):
            _ = wc(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        ms = bench(wc, x, iters=50)
        per_tower = ms / n_towers
        print(f"{n_towers:>3} towers: {ms:.2f}ms total, {per_tower:.3f}ms/tower")

    print("\n✓ WideRouter ready (use .compile() for best performance)")