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
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    Optional, Dict, List, Tuple, Any, Union,
    Callable, Set, Iterator, TypeVar, Generic
)
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call, vmap, stack_module_state

from geofractal.router.base_router import BaseRouter
from geofractal.router.base_tower import BaseTower

# Wide compiler integration (optional dependency)
from wide_compiler.core.registry import get_registry as get_wide_registry, build_wide
_HAS_WIDE_COMPILER = True


# =============================================================================
# EXECUTION STRATEGY
# =============================================================================

class ExecutionStrategy(Enum):
    """Execution strategy for batched tower forward."""
    VMAP = auto()           # torch.func.vmap (current default)
    WIDE_COMPILER = auto()  # wide_compiler operation-level fusion via Wide primitives
    SEQUENTIAL = auto()     # plain for-loop fallback
    AUTO = auto()           # training: vmap, eval: wide_compiler -> vmap -> sequential


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

    IMPORTANT: Call prepare_for_forward() before torch.compile() to avoid
    graph breaks from stack_module_state() which uses requires_grad_().
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
        self._prepared = False
        self._current_device: Optional[torch.device] = None

    @torch.compiler.disable
    def _stack_state(self) -> None:
        """Stack parameters and buffers from all towers.

        Decorated with @torch.compiler.disable because stack_module_state
        uses requires_grad_() which breaks dynamo tracing.
        """
        if not self._stale:
            return
        self._stacked_params, self._stacked_buffs = stack_module_state(list(self.towers))
        self._stale = False

    @torch.compiler.disable
    def prepare_for_forward(self, device: torch.device = None) -> 'VMapTowerGroup':
        """
        Prepare state for forward pass. Call BEFORE torch.compile().

        This stacks parameters and buffers and optionally moves to device.
        By doing it here outside the forward path, we avoid graph breaks.

        Args:
            device: Target device (optional)

        Returns:
            self for chaining
        """
        if self._prepared and self._current_device == device:
            return self

        # Stack params and buffers
        self._stacked_params, self._stacked_buffs = stack_module_state(list(self.towers))
        self._stale = False

        # Move to device if specified
        if device is not None:
            self._stacked_params = {
                k: v.to(device) for k, v in self._stacked_params.items()
            }
            self._stacked_buffs = {
                k: v.to(device) for k, v in self._stacked_buffs.items()
            }
            self._current_device = device

        self._prepared = True
        return self

    def _invalidate(self) -> None:
        """Mark cached state as stale."""
        self._stale = True
        self._prepared = False
        self._stacked_params = None
        self._stacked_buffs = None

    def train(self, mode: bool = True):
        result = super().train(mode)
        self._invalidate()
        return result

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)

        # Extract device from args
        device = None
        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
                break
            elif isinstance(arg, str):
                device = torch.device(arg)
                break
        if device is None:
            device = kwargs.get('device')
            if isinstance(device, str):
                device = torch.device(device)

        # Re-prepare with new device if we were previously prepared
        if device is not None and self._prepared:
            self._invalidate()
            self.prepare_for_forward(device)
        else:
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

        # Ensure state is stacked (will cause graph break if not pre-prepared)
        if not self._prepared:
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
# WIDE PRIMITIVE TOWER GROUP (wide_compiler integration)
# =============================================================================

@dataclass
class OpPlan:
    """Execution plan for one operation within a stage."""
    path: str            # e.g. '0' or '0.linear' within the stage
    execution: str       # 'wide_kernel', 'vmap', 'sequential'
    wide_type: str       # e.g. 'WideLinear' or '' if not fused
    module_type: str     # e.g. 'Linear', 'LayerNorm', etc.


@dataclass
class StagePlan:
    """Execution plan for one stage position across N towers."""
    stage_index: int
    classification: str  # 'fully_fused', 'partially_fused', 'opaque'
    ops: List[OpPlan] = field(default_factory=list)


@dataclass
class FusionCoverage:
    """Tracks what percentage of ops actually fused."""
    total_ops: int = 0
    fused_ops: int = 0         # using Wide kernels
    vmap_ops: int = 0          # using vmap fallback
    sequential_ops: int = 0    # using sequential fallback

    @property
    def fused_pct(self) -> float:
        return self.fused_ops / max(self.total_ops, 1) * 100

    @property
    def summary(self) -> str:
        return (
            f"{self.fused_ops}/{self.total_ops} ops fused ({self.fused_pct:.0f}%), "
            f"{self.vmap_ops} vmap, {self.sequential_ops} sequential"
        )


class WidePrimitiveTowerGroup(nn.Module):
    """
    Batched tower execution using wide_compiler primitives (einsum, grouped conv).

    Like VMapTowerGroup, towers own their parameters. This group stacks them
    at prepare time and feeds the stacked tensors into Wide primitive kernels
    instead of vmap(functional_call).

    Architecture:
        - Walks tower stages and classifies each as fully_fused, partially_fused,
          or opaque based on Wide registry coverage.
        - Fully fused stages: all ops executed through Wide kernels (einsum etc.)
        - Partially fused stages: mix of Wide kernels and vmap for unregistered ops
        - Opaque stages (custom forward with residuals): vmap(functional_call)

    Parameter ownership:
        Towers ALWAYS own their parameters. This group stacks views at prepare time
        via torch.stack. Gradients flow back through the stack to original tower params.
        During training, re-stacking happens every forward to pick up optimizer updates.
        During eval, cached stacked params are used (no graph break).

    No sample_input needed: operation-level fusion reads weight shapes from tower
    parameters directly, unlike FX-trace-based TracedWideModel.
    """

    def __init__(self, towers: List[nn.Module], names: List[str]):
        super().__init__()
        self.names = names
        self.n = len(towers)

        # Towers own the parameters
        self.towers = nn.ModuleList(towers)

        # Stage execution plans (built at prepare time)
        self._stage_plans: List[StagePlan] = []
        self._fusion_coverage = FusionCoverage()

        # Stacked parameters: path -> stacked tensor [N, ...]
        self._stacked_params: Dict[str, Tensor] = {}
        self._stale = True
        self._prepared = False
        self._current_device: Optional[torch.device] = None

        # For opaque/partial stages: stacked state for vmap fallback
        self._vmap_stage_params: Dict[int, Dict[str, Tensor]] = {}
        self._vmap_stage_buffs: Dict[int, Dict[str, Tensor]] = {}

    # =========================================================================
    # STAGE CLASSIFICATION
    # =========================================================================

    @torch.compiler.disable
    def _classify_stages(self) -> Tuple[List[StagePlan], FusionCoverage]:
        """
        Walk tower stages and classify each for execution strategy.

        Only stages where ALL towers have identical structure at that position
        are considered for fusion. Structural matching uses module type names
        and parameter shapes.
        """
        coverage = FusionCoverage()
        plans: List[StagePlan] = []

        if not hasattr(self.towers[0], 'stages'):
            # Tower has no stages — treat entire tower as one opaque stage
            plan = StagePlan(
                stage_index=0,
                classification='opaque',
                ops=[OpPlan(
                    path='',
                    execution='vmap',
                    wide_type='',
                    module_type=type(self.towers[0]).__name__,
                )],
            )
            coverage.total_ops += 1
            coverage.vmap_ops += 1
            plans.append(plan)
            return plans, coverage

        num_stages = len(self.towers[0].stages)

        # Verify all towers have same number of stages
        if not all(
            hasattr(t, 'stages') and len(t.stages) == num_stages
            for t in self.towers
        ):
            # Mismatched stage counts — fall back to one opaque group
            plan = StagePlan(
                stage_index=0,
                classification='opaque',
                ops=[OpPlan(path='', execution='vmap', wide_type='', module_type='mixed')],
            )
            coverage.total_ops += 1
            coverage.vmap_ops += 1
            plans.append(plan)
            return plans, coverage

        registry = get_wide_registry() if _HAS_WIDE_COMPILER else None

        for si in range(num_stages):
            stages = [self.towers[ti].stages[si] for ti in range(self.n)]

            # Check if this stage is nn.Sequential (decomposable)
            if all(isinstance(s, nn.Sequential) for s in stages):
                plan = self._classify_sequential_stage(si, stages, registry, coverage)
            else:
                # Opaque stage: custom forward (may have residuals, branching)
                plan = StagePlan(
                    stage_index=si,
                    classification='opaque',
                    ops=[OpPlan(
                        path='',
                        execution='vmap',
                        wide_type='',
                        module_type=type(stages[0]).__name__,
                    )],
                )
                coverage.total_ops += 1
                coverage.vmap_ops += 1

            plans.append(plan)

        return plans, coverage

    @staticmethod
    def _classify_sequential_stage(
        si: int,
        stages: List[nn.Sequential],
        registry,
        coverage: FusionCoverage,
    ) -> StagePlan:
        """Classify a Sequential stage by checking each child against the registry."""
        ops: List[OpPlan] = []
        all_fused = True

        # All Sequential stages must have same number of children
        num_children = len(list(stages[0].children()))
        if not all(len(list(s.children())) == num_children for s in stages):
            coverage.total_ops += 1
            coverage.vmap_ops += 1
            return StagePlan(
                stage_index=si,
                classification='opaque',
                ops=[OpPlan(path='', execution='vmap', wide_type='', module_type='Sequential')],
            )

        for ci, (name, child) in enumerate(stages[0].named_children()):
            module_type = type(child).__name__
            coverage.total_ops += 1

            # Check all N towers have same module type at this position
            children_match = True
            for s in stages[1:]:
                other_child = list(s.children())[ci]
                if type(other_child).__name__ != module_type:
                    children_match = False
                    break

            can_fuse = (
                children_match
                and registry is not None
                and registry.has(module_type)
            )

            if can_fuse:
                # Verify parameter shapes match across towers
                child_params = {n: p.shape for n, p in child.named_parameters(recurse=False)}
                for s in stages[1:]:
                    other_child = list(s.children())[ci]
                    other_params = {n: p.shape for n, p in other_child.named_parameters(recurse=False)}
                    if child_params != other_params:
                        can_fuse = False
                        break

            if can_fuse:
                wide_cls = registry.get_class(module_type)
                ops.append(OpPlan(
                    path=str(ci),
                    execution='wide_kernel',
                    wide_type=wide_cls.__name__ if wide_cls else '',
                    module_type=module_type,
                ))
                coverage.fused_ops += 1
            else:
                ops.append(OpPlan(
                    path=str(ci),
                    execution='vmap',
                    wide_type='',
                    module_type=module_type,
                ))
                coverage.vmap_ops += 1
                all_fused = False

        classification = 'fully_fused' if all_fused else 'partially_fused'
        return StagePlan(stage_index=si, classification=classification, ops=ops)

    # =========================================================================
    # PARAMETER STACKING
    # =========================================================================

    @torch.compiler.disable
    def _stack_params(self) -> None:
        """
        Stack tower parameters into N-first format for Wide kernels.

        Each stacked tensor is [N, ...original_shape...] where dim 0 indexes towers.
        Gradients flow back through torch.stack to original tower parameters.

        Also stacks params/buffers for vmap-executed stages.
        """
        self._stacked_params.clear()
        self._vmap_stage_params.clear()
        self._vmap_stage_buffs.clear()

        for plan in self._stage_plans:
            si = plan.stage_index

            if plan.classification == 'opaque':
                # Stack entire stage for vmap(functional_call)
                stage_modules = [self.towers[ti].stages[si] for ti in range(self.n)]
                params, buffs = stack_module_state(stage_modules)
                self._vmap_stage_params[si] = params
                self._vmap_stage_buffs[si] = buffs

            elif plan.classification in ('fully_fused', 'partially_fused'):
                for op in plan.ops:
                    ci = int(op.path) if op.path.isdigit() else 0

                    if op.execution == 'wide_kernel':
                        # Stack leaf module parameters for Wide kernel
                        children = [
                            list(self.towers[ti].stages[si].children())[ci]
                            for ti in range(self.n)
                        ]
                        key_prefix = f"stage_{si}.op_{ci}"
                        for pname, param in children[0].named_parameters(recurse=False):
                            stacked = torch.stack(
                                [list(self.towers[ti].stages[si].children())[ci]
                                 .__dict__.get('_parameters', {})[pname]
                                 if pname in dict(list(self.towers[ti].stages[si].children())[ci].named_parameters(recurse=False))
                                 else getattr(list(self.towers[ti].stages[si].children())[ci], pname)
                                 for ti in range(self.n)],
                                dim=0,
                            )
                            self._stacked_params[f"{key_prefix}.{pname}"] = stacked

                    elif op.execution == 'vmap':
                        # Stack this specific child for vmap
                        children = [
                            list(self.towers[ti].stages[si].children())[ci]
                            for ti in range(self.n)
                        ]
                        params, buffs = stack_module_state(children)
                        vmap_key = f"stage_{si}.op_{ci}"
                        self._vmap_stage_params[hash(vmap_key)] = params
                        self._vmap_stage_buffs[hash(vmap_key)] = buffs

        self._stale = False

    # =========================================================================
    # PREPARATION
    # =========================================================================

    @torch.compiler.disable
    def prepare_for_forward(self, device: torch.device = None) -> 'WidePrimitiveTowerGroup':
        """
        Classify stages and stack parameters. Call BEFORE torch.compile().

        Args:
            device: Target device (optional)

        Returns:
            self for chaining
        """
        if self._prepared and self._current_device == device:
            return self

        # Classify stages
        self._stage_plans, self._fusion_coverage = self._classify_stages()

        # Stack parameters
        self._stack_params()

        # Move stacked params to device if specified
        if device is not None:
            self._stacked_params = {
                k: v.to(device) for k, v in self._stacked_params.items()
            }
            for si in list(self._vmap_stage_params.keys()):
                self._vmap_stage_params[si] = {
                    k: v.to(device) for k, v in self._vmap_stage_params[si].items()
                }
                self._vmap_stage_buffs[si] = {
                    k: v.to(device) for k, v in self._vmap_stage_buffs[si].items()
                }
            self._current_device = device

        self._prepared = True
        return self

    def _invalidate(self) -> None:
        """Mark cached state as stale."""
        self._stale = True
        self._prepared = False
        self._stacked_params.clear()
        self._vmap_stage_params.clear()
        self._vmap_stage_buffs.clear()

    def train(self, mode: bool = True):
        result = super().train(mode)
        self._invalidate()
        return result

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)

        device = None
        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
                break
            elif isinstance(arg, str):
                device = torch.device(arg)
                break
        if device is None:
            device = kwargs.get('device')
            if isinstance(device, str):
                device = torch.device(device)

        if device is not None and self._prepared:
            self._invalidate()
            self.prepare_for_forward(device)
        else:
            self._invalidate()

        return result

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Execute towers through Wide primitive kernels where possible.

        Args:
            x: Input tensor [B, ...] — same input to all towers

        Returns:
            Dict mapping tower name -> output tensor [B, ...]
        """
        if self.n == 0:
            return {}

        if self.n == 1:
            out = self.towers[0](x)
            return {self.names[0]: out[0] if isinstance(out, tuple) else out}

        # During training: re-stack every forward to pick up optimizer updates.
        # During eval: use cached stacked params (no graph break).
        if self.training or self._stale:
            self._stack_params()

        if not self._stage_plans:
            # No stage plans — shouldn't happen if prepared, fall back
            return self._fallback_sequential(x)

        # Stack input: [B, ...] -> [N, B, ...]
        x_wide = x.unsqueeze(0).expand(self.n, *x.shape)

        # Execute stage by stage
        for plan in self._stage_plans:
            x_wide = self._execute_stage(x_wide, plan)

        # Handle tuple outputs
        if isinstance(x_wide, tuple):
            x_wide = x_wide[0]

        # Scatter to dict: [N, B, ...] -> {name: [B, ...]}
        return {name: x_wide[i] for i, name in enumerate(self.names)}

    def _execute_stage(self, x: Tensor, plan: StagePlan) -> Tensor:
        """
        Execute one stage according to its plan.

        Args:
            x: [N, B, ...] N-first format
            plan: StagePlan describing how to execute

        Returns:
            [N, B, ...] stage output
        """
        si = plan.stage_index

        if plan.classification == 'opaque':
            return self._execute_opaque_stage(x, si)

        # Sequential stage (fully or partially fused)
        for op in plan.ops:
            ci = int(op.path) if op.path.isdigit() else 0

            if op.execution == 'wide_kernel':
                x = self._execute_wide_kernel(x, si, ci, op.module_type)
            elif op.execution == 'vmap':
                x = self._execute_vmap_op(x, si, ci)
            else:
                # sequential fallback
                x = self._execute_sequential_op(x, si, ci)

        return x

    def _execute_opaque_stage(self, x: Tensor, si: int) -> Tensor:
        """Execute opaque stage via vmap(functional_call)."""
        params = self._vmap_stage_params.get(si)
        buffs = self._vmap_stage_buffs.get(si)

        if params is None:
            # Fallback: sequential
            return self._execute_sequential_stage(x, si)

        base = self.towers[0].stages[si]

        def single_forward(p, b, data):
            return functional_call(base, (p, b), (data,))

        vmapped = vmap(single_forward, in_dims=(0, 0, 0))
        out = vmapped(params, buffs, x)

        if isinstance(out, tuple):
            out = out[0]
        return out

    def _execute_wide_kernel(
        self, x: Tensor, si: int, ci: int, module_type: str
    ) -> Tensor:
        """
        Execute a fused op through a Wide primitive kernel.

        Uses stacked parameters and the Wide primitive's forward logic
        (einsum for Linear, grouped conv for Conv2d, etc.)
        """
        key_prefix = f"stage_{si}.op_{ci}"

        weight = self._stacked_params.get(f"{key_prefix}.weight")
        bias = self._stacked_params.get(f"{key_prefix}.bias")

        if weight is None:
            # Missing stacked params — fall back
            return self._execute_sequential_op(x, si, ci)

        # Dispatch to appropriate Wide kernel based on module type
        if module_type == 'Linear':
            return self._wide_linear_forward(x, weight, bias)
        elif module_type == 'LayerNorm':
            ref_module = list(self.towers[0].stages[si].children())[ci]
            eps = getattr(ref_module, 'eps', 1e-5)
            return self._wide_layernorm_forward(x, weight, bias, eps)
        elif module_type in ('Conv1d', 'Conv2d', 'Conv3d'):
            ref_module = list(self.towers[0].stages[si].children())[ci]
            return self._wide_conv_forward(x, weight, bias, ref_module, module_type)
        else:
            # Registered type but no kernel implementation — use vmap fallback
            return self._execute_vmap_op(x, si, ci)

    @staticmethod
    def _wide_linear_forward(x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        """
        N-first batched linear via einsum.

        x: [N, B, ..., in_features]
        weight: [N, out_features, in_features]
        bias: [N, out_features] or None
        Returns: [N, B, ..., out_features]
        """
        N = weight.shape[0]
        in_features = weight.shape[2]
        out_features = weight.shape[1]
        orig_shape = x.shape  # [N, B, ..., in]
        batch_shape = orig_shape[1:-1]  # [B, ...]

        # Flatten batch dims: [N, B, ..., in] -> [N, B*, in]
        x_flat = x.reshape(N, -1, in_features)

        # Einsum: [N, out, in] @ [N, B*, in] -> [N, B*, out]
        out = torch.einsum('noi,nbi->nbo', weight, x_flat)

        if bias is not None:
            out = out + bias.unsqueeze(1)  # [N, 1, out]

        # Restore batch dims: [N, B*, out] -> [N, B, ..., out]
        return out.reshape(N, *batch_shape, out_features)

    @staticmethod
    def _wide_layernorm_forward(
        x: Tensor, weight: Tensor, bias: Optional[Tensor], eps: float
    ) -> Tensor:
        """
        N-first batched layer norm.

        x: [N, B, ..., D]
        weight: [N, D]
        bias: [N, D] or None
        Returns: [N, B, ..., D]
        """
        # Expand weight/bias to match x dimensions
        w = weight
        b = bias
        for _ in range(x.dim() - 2):
            w = w.unsqueeze(1)
            if b is not None:
                b = b.unsqueeze(1)

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + eps)

        x = x * w
        if b is not None:
            x = x + b
        return x

    @staticmethod
    def _wide_conv_forward(
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        ref_module: nn.Module,
        module_type: str,
    ) -> Tensor:
        """
        N-first batched convolution via grouped conv.

        Reshapes [N, B, C, ...] -> [B, N*C, ...], applies grouped conv, reshapes back.
        """
        N = weight.shape[0]
        B = x.shape[1]
        spatial = x.shape[3:]  # after [N, B, C, ...]

        # Merge N into channel dim: [N, B, C, ...] -> [B, N*C, ...]
        C = x.shape[2]
        x_merged = x.permute(1, 0, 2, *range(3, x.dim())).reshape(B, N * C, *spatial)

        # Stack weights: [N, C_out, C_in, ...] -> [N*C_out, C_in, ...]
        w_shape = weight.shape  # [N, C_out, C_in/groups, ...]
        w_merged = weight.reshape(N * w_shape[1], *w_shape[2:])

        b_merged = None
        if bias is not None:
            b_merged = bias.reshape(N * bias.shape[1])

        # Grouped conv with N groups
        conv_fn = {
            'Conv1d': F.conv1d,
            'Conv2d': F.conv2d,
            'Conv3d': F.conv3d,
        }.get(module_type, F.conv2d)

        stride = ref_module.stride if hasattr(ref_module, 'stride') else 1
        padding = ref_module.padding if hasattr(ref_module, 'padding') else 0
        dilation = ref_module.dilation if hasattr(ref_module, 'dilation') else 1

        out = conv_fn(
            x_merged, w_merged, b_merged,
            stride=stride, padding=padding, dilation=dilation,
            groups=N,
        )

        # Unmerge: [B, N*C_out, ...] -> [N, B, C_out, ...]
        C_out = w_shape[1]
        out_spatial = out.shape[2:]
        out = out.reshape(B, N, C_out, *out_spatial).permute(1, 0, 2, *range(3, 3 + len(out_spatial)))

        return out

    def _execute_vmap_op(self, x: Tensor, si: int, ci: int) -> Tensor:
        """Execute a single op within a sequential stage via vmap."""
        vmap_key = hash(f"stage_{si}.op_{ci}")
        params = self._vmap_stage_params.get(vmap_key)
        buffs = self._vmap_stage_buffs.get(vmap_key)

        if params is None:
            return self._execute_sequential_op(x, si, ci)

        base = list(self.towers[0].stages[si].children())[ci]

        def single_forward(p, b, data):
            return functional_call(base, (p, b), (data,))

        vmapped = vmap(single_forward, in_dims=(0, 0, 0))
        out = vmapped(params, buffs, x)

        if isinstance(out, tuple):
            out = out[0]
        return out

    def _execute_sequential_op(self, x: Tensor, si: int, ci: int) -> Tensor:
        """Sequential fallback for a single op."""
        outputs = []
        for ti in range(self.n):
            child = list(self.towers[ti].stages[si].children())[ci]
            outputs.append(child(x[ti]))
        return torch.stack(outputs, dim=0)

    def _execute_sequential_stage(self, x: Tensor, si: int) -> Tensor:
        """Sequential fallback for an entire stage."""
        outputs = []
        for ti in range(self.n):
            out = self.towers[ti].stages[si](x[ti])
            if isinstance(out, tuple):
                out = out[0]
            outputs.append(out)
        return torch.stack(outputs, dim=0)

    def _fallback_sequential(self, x: Tensor) -> Dict[str, Tensor]:
        """Full sequential fallback when nothing else works."""
        outputs = {}
        for i, (name, tower) in enumerate(zip(self.names, self.towers)):
            out = tower(x)
            if isinstance(out, tuple):
                out = out[0]
            outputs[name] = out
        return outputs

    def __repr__(self) -> str:
        return (
            f"WidePrimitiveTowerGroup("
            f"n={self.n}, "
            f"stages={len(self._stage_plans)}, "
            f"coverage={self._fusion_coverage.summary}, "
            f"prepared={self._prepared}"
            f")"
        )


# =============================================================================
# SUB-ENSEMBLE GROUP (pooled gradient interpolation)
# =============================================================================

class SubEnsembleGroup(nn.Module):
    """
    Multiple execution paths over same tower group for pooled gradient interpolation.

    Each sub-ensemble provides a different computational vantage point
    (VMap vs Wide primitives). Learnable interpolation weights discover
    which path produces the most useful gradient signal, while ALL paths
    receive gradient updates through the weighted sum.
    """

    def __init__(self, name: str, dim: int):
        super().__init__()
        self._name = name
        self.dim = dim
        self.groups = nn.ModuleDict()
        self._interpolation_weights: Optional[nn.Parameter] = None
        self._gradient_norms: Dict[str, List[float]] = defaultdict(list)

    def add_group(
        self,
        name: str,
        group: Union[VMapTowerGroup, WidePrimitiveTowerGroup],
    ) -> None:
        """Add an execution group."""
        self.groups[name] = group
        # Rebuild interpolation weights
        n_groups = len(self.groups)
        self._interpolation_weights = nn.Parameter(
            torch.ones(n_groups) / n_groups
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Execute all groups and pool outputs via learned interpolation.

        Args:
            x: [B, ...] input tensor

        Returns:
            Dict[tower_name, Tensor] — pooled outputs with gradient paths
            to all execution groups.
        """
        if len(self.groups) == 0:
            return {}

        if len(self.groups) == 1:
            return list(self.groups.values())[0](x)

        # Execute each group
        all_outputs: List[Dict[str, Tensor]] = []
        for group in self.groups.values():
            all_outputs.append(group(x))

        # Compute interpolation weights
        weights = torch.softmax(self._interpolation_weights, dim=0)

        # Pool: weighted sum of outputs per tower name
        # Collect all tower names across groups
        all_names: Set[str] = set()
        for out in all_outputs:
            all_names.update(out.keys())

        pooled: Dict[str, Tensor] = {}
        for tower_name in all_names:
            contributions = []
            contrib_weights = []
            for gi, out in enumerate(all_outputs):
                if tower_name in out:
                    contributions.append(out[tower_name])
                    contrib_weights.append(weights[gi])

            if len(contributions) == 1:
                pooled[tower_name] = contributions[0]
            else:
                # Weighted sum: gradients flow to all contributing groups
                stacked = torch.stack(contributions, dim=0)  # [G, B, ...]
                w = torch.stack(contrib_weights, dim=0)      # [G]
                # Expand weights for broadcasting
                for _ in range(stacked.dim() - 1):
                    w = w.unsqueeze(-1)
                pooled[tower_name] = (stacked * w).sum(dim=0)

        return pooled

    def get_gradient_diagnostics(self) -> Dict[str, Any]:
        """Per-group gradient norms and interpolation weight distribution."""
        diag = {
            'num_groups': len(self.groups),
            'group_names': list(self.groups.keys()),
        }
        if self._interpolation_weights is not None:
            weights = torch.softmax(self._interpolation_weights, dim=0)
            diag['interpolation_weights'] = {
                name: weights[i].item()
                for i, name in enumerate(self.groups.keys())
            }
            diag['weight_entropy'] = -(weights * weights.log()).sum().item()
        diag['gradient_norms'] = dict(self._gradient_norms)
        return diag

    def __repr__(self) -> str:
        return (
            f"SubEnsembleGroup(name='{self._name}', "
            f"groups={list(self.groups.keys())})"
        )


# =============================================================================
# CACHE DEVICE CONTROLLER
# =============================================================================

class CacheDeviceController:
    """
    Manages cache lifecycle across device movements.

    Ensures caches respect the geofractal structure's device movements,
    especially when Wide primitives and sub-ensembles span devices.

    Policies:
        'clear':       Clear all caches before device move (default, current behavior)
        'migrate':     Move cache tensors to new device after move
        'reconstruct': Clear caches and mark for lazy rebuild
    """

    def __init__(self, router: 'WideRouter', policy: str = 'clear'):
        self._router = weakref.ref(router)
        self._policy = policy
        self._cache_needs_rebuild = False

    @property
    def policy(self) -> str:
        return self._policy

    @policy.setter
    def policy(self, value: str):
        if value not in ('clear', 'migrate', 'reconstruct'):
            raise ValueError(f"Unknown cache policy: {value}")
        self._policy = value

    def pre_device_move(self, target_device) -> None:
        """Called before network_to(). Records cache state based on policy."""
        if self._policy == 'clear':
            router = self._router()
            if router:
                router.cache_clear_recursive()

    def post_device_move(self, target_device) -> None:
        """Called after network_to(). Finalizes cache migration."""
        router = self._router()
        if router is None:
            return

        if self._policy == 'migrate' and target_device is not None:
            router.cache_to_recursive(target_device)
        elif self._policy == 'reconstruct':
            router.cache_clear_recursive()
            self._cache_needs_rebuild = True

    def validate_cache_devices(self) -> List[str]:
        """
        Walk all caches and report device mismatches.

        Returns list of issue strings (empty = healthy).
        """
        issues = []
        router = self._router()
        if router is None:
            return issues

        # Determine expected device from parameters
        expected_device = None
        for p in router.parameters():
            expected_device = p.device
            break

        if expected_device is None:
            return issues

        # Check router cache
        for key, val in getattr(router, '_cache', {}).items():
            if isinstance(val, Tensor) and val.device != expected_device:
                issues.append(
                    f"Cache '{key}' on {val.device}, expected {expected_device}"
                )

        # Check Wide primitive groups
        for cache_key, group in router.objects.get('_wide_primitive_groups', {}).items():
            for pkey, ptensor in group._stacked_params.items():
                if ptensor.device != expected_device:
                    issues.append(
                        f"Wide group stacked param '{pkey}' on {ptensor.device}, "
                        f"expected {expected_device}"
                    )

        return issues

    @property
    def needs_rebuild(self) -> bool:
        return self._cache_needs_rebuild

    def mark_rebuilt(self) -> None:
        self._cache_needs_rebuild = False


# =============================================================================
# GRADIENT DEBUGGER
# =============================================================================

@dataclass
class GradientAnomaly:
    """A detected gradient anomaly."""
    location: str
    anomaly_type: str  # 'nan', 'inf', 'dead', 'exploding', 'flow_break'
    value: float
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class GradientSnapshot:
    """Gradient state captured after backward."""
    tower_norms: Dict[str, float] = field(default_factory=dict)
    stage_norms: Dict[str, Dict[int, float]] = field(default_factory=dict)
    anomalies: List[GradientAnomaly] = field(default_factory=list)
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class FusionDiagnostic:
    """Health of Wide primitive fusion gradient flow."""
    fused_vs_source_diff: Dict[str, float] = field(default_factory=dict)
    interpolation_entropy: float = 0.0
    starving_groups: List[str] = field(default_factory=list)


@dataclass
class CompilationDiagnostic:
    """Eager vs compiled gradient comparison."""
    matches: bool = True
    max_diff: float = 0.0
    mismatched_params: List[str] = field(default_factory=list)
    graph_breaks: int = 0


class GradientDebugger:
    """
    Multi-level gradient monitoring for the wide execution tree.

    Levels:
        0: Off (no overhead)
        1: Per-tower gradient norms after backward
        2: Per-stage gradient norms + fusion vs unfused comparison
        3: Per-parameter gradients + anomaly detection + compilation diagnostics
    """

    # Thresholds for anomaly detection
    DEAD_THRESHOLD = 1e-10
    EXPLODING_THRESHOLD = 1e6

    def __init__(self, router: 'WideRouter', level: int = 0):
        self._router = weakref.ref(router)
        self._level = level
        self._hooks: List[Any] = []
        self._history: deque = deque(maxlen=100)
        self._attached = False

    @property
    def level(self) -> int:
        return self._level

    @level.setter
    def level(self, value: int):
        if value != self._level:
            self.detach()
            self._level = value
            if value > 0:
                self.attach()

    def attach(self) -> None:
        """Register backward hooks based on level."""
        if self._attached or self._level == 0:
            return

        router = self._router()
        if router is None:
            return

        if self._level >= 1:
            # Hook on each tower's output
            for name in router.tower_names:
                tower = router.get(name)
                if tower is not None:
                    handle = tower.register_full_backward_hook(
                        self._make_tower_hook(name)
                    )
                    self._hooks.append(handle)

        self._attached = True

    def detach(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attached = False

    def _make_tower_hook(self, tower_name: str):
        """Create a backward hook for a tower."""
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                norm = grad_output[0].norm().item()
                # Store in latest snapshot or create new one
                if not self._history or (time.perf_counter() - self._history[-1].timestamp > 0.001):
                    self._history.append(GradientSnapshot())
                self._history[-1].tower_norms[tower_name] = norm

                # Anomaly detection
                if norm != norm:  # NaN
                    self._history[-1].anomalies.append(
                        GradientAnomaly(tower_name, 'nan', norm)
                    )
                elif norm == float('inf'):
                    self._history[-1].anomalies.append(
                        GradientAnomaly(tower_name, 'inf', norm)
                    )
                elif norm < self.DEAD_THRESHOLD:
                    self._history[-1].anomalies.append(
                        GradientAnomaly(tower_name, 'dead', norm)
                    )
                elif norm > self.EXPLODING_THRESHOLD:
                    self._history[-1].anomalies.append(
                        GradientAnomaly(tower_name, 'exploding', norm)
                    )

        return hook

    def snapshot(self) -> Optional[GradientSnapshot]:
        """Get most recent gradient snapshot."""
        return self._history[-1] if self._history else None

    def check_fusion_health(self) -> FusionDiagnostic:
        """Check gradient health of Wide primitive fusion."""
        router = self._router()
        diag = FusionDiagnostic()

        if router is None:
            return diag

        # Check sub-ensemble interpolation weights
        for name, se in router.objects.get('_sub_ensembles', {}).items():
            if isinstance(se, SubEnsembleGroup):
                se_diag = se.get_gradient_diagnostics()
                diag.interpolation_entropy = se_diag.get('weight_entropy', 0.0)

                # Detect starving groups
                weights = se_diag.get('interpolation_weights', {})
                for gname, w in weights.items():
                    if w < 0.01:  # less than 1% weight
                        diag.starving_groups.append(f"{name}/{gname}")

        return diag

    def report(self) -> str:
        """Human-readable gradient health report."""
        if not self._history:
            return "No gradient data (run backward first, or set gradient_debug_level > 0)"

        snap = self._history[-1]
        lines = ["Gradient Report", "=" * 40]

        if snap.tower_norms:
            lines.append("\nTower gradient norms:")
            for name, norm in sorted(snap.tower_norms.items()):
                status = ""
                if norm < self.DEAD_THRESHOLD:
                    status = " [DEAD]"
                elif norm > self.EXPLODING_THRESHOLD:
                    status = " [EXPLODING]"
                lines.append(f"  {name}: {norm:.6f}{status}")

        if snap.anomalies:
            lines.append(f"\nAnomalies ({len(snap.anomalies)}):")
            for a in snap.anomalies:
                lines.append(f"  [{a.anomaly_type}] {a.location}: {a.value:.6f}")

        return '\n'.join(lines)

    def __del__(self):
        self.detach()


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
        execution_strategy: ExecutionStrategy = ExecutionStrategy.VMAP,
        gradient_debug_level: int = 0,
        cache_preservation_policy: str = 'clear',
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
            execution_strategy: How to batch aligned towers (VMAP, WIDE_COMPILER, AUTO, SEQUENTIAL)
            gradient_debug_level: 0=off, 1=tower norms, 2=stage norms, 3=full
            cache_preservation_policy: 'clear', 'migrate', or 'reconstruct'
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

        # Compile preparation tracking
        self.objects['_prepared_for_compile'] = False
        self.objects['_vmap_device'] = None
        self.objects['_non_vmappable_groups'] = set()

        # Wide compiler integration
        self.objects['_execution_strategy'] = execution_strategy
        self.objects['_wide_primitive_groups'] = {}
        self.objects['_non_wide_fusable_groups'] = set()
        self.objects['_sub_ensembles'] = {}

        # Pre-resolved strategy for compile-safe dispatch (ints, not Enums)
        # 0=VMAP, 1=WIDE_COMPILER, 2=SEQUENTIAL
        self.objects['_resolved_strategy'] = self._resolve_strategy_int(execution_strategy, False)
        self.objects['_auto_fallback'] = (execution_strategy == ExecutionStrategy.AUTO)

        # Diagnostic infrastructure (Phase A)
        self._cache_controller = CacheDeviceController(self, policy=cache_preservation_policy)
        self._gradient_debugger = GradientDebugger(self, level=gradient_debug_level)

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
    # COMPILE PREPARATION (pre-build vmap groups)
    # =========================================================================

    @torch.compiler.disable
    def _prebuild_vmap_groups(self, device: torch.device = None) -> None:
        """
        Pre-build all VMapTowerGroups for registered towers.

        This MUST be called before torch.compile() to avoid graph breaks.
        Groups are built and their state is pre-stacked.

        Args:
            device: Target device for computation
        """
        if not self.objects.get('_analyzed', False):
            self.analyze_structure()

        # Clear existing groups
        self.objects['_vmap_groups'].clear()
        if '_non_vmappable_groups' not in self.objects:
            self.objects['_non_vmappable_groups'] = set()
        else:
            self.objects['_non_vmappable_groups'].clear()

        # Group towers by signature
        sig_to_names: Dict[str, List[str]] = {}
        for name in self.tower_names:
            sig = self._tower_signature(name)
            if sig not in sig_to_names:
                sig_to_names[sig] = []
            sig_to_names[sig].append(name)

        # Build VMapTowerGroup for each signature group with 2+ towers
        for sig, names in sig_to_names.items():
            if len(names) < 2:
                continue

            try:
                towers = [self[name] for name in names]
                group = VMapTowerGroup(towers, names)

                # Pre-prepare the group (stacks state and moves to device)
                group.prepare_for_forward(device)

                # Cache by (signature, frozenset of names)
                cache_key = (sig, frozenset(names))
                self.objects['_vmap_groups'][cache_key] = group

            except Exception as e:
                # Mark as non-vmappable
                cache_key = (sig, frozenset(names))
                self.objects['_non_vmappable_groups'].add(cache_key)

        self.objects['_prepared_for_compile'] = True
        self.objects['_vmap_device'] = device

    def prepare_for_compile(
        self,
        device: torch.device = None,
        execution_strategy: Optional[ExecutionStrategy] = None,
        build_sub_ensembles: bool = False,
    ) -> 'WideRouter':
        """
        Prepare router for torch.compile().

        Call this method BEFORE torch.compile() to ensure execution groups
        are pre-built and their state is pre-stacked, avoiding graph breaks.

        Args:
            device: Target device for computation. If None, inferred from parameters.
            execution_strategy: Override the router's execution strategy.
            build_sub_ensembles: If True, build sub-ensembles for multi-vantage
                gradient interpolation (creates both vmap and wide_compiler groups).

        Returns:
            self for chaining

        Example:
            router = WideRouter('my_router', execution_strategy=ExecutionStrategy.AUTO)
            # ... attach towers ...
            router.prepare_for_compile()
            compiled = torch.compile(router)
        """
        # Infer device from parameters if not specified
        if device is None:
            for p in self.parameters():
                device = p.device
                break

        # Update strategy if overridden
        if execution_strategy is not None:
            self.objects['_execution_strategy'] = execution_strategy
            self.objects['_auto_fallback'] = (execution_strategy == ExecutionStrategy.AUTO)

        strategy = self.objects['_execution_strategy']

        # Pre-resolve to int for compile-safe dispatch in forward path
        self.objects['_resolved_strategy'] = self._resolve_strategy_int(strategy, self.training)

        if strategy == ExecutionStrategy.VMAP:
            self._prebuild_vmap_groups(device)
        elif strategy == ExecutionStrategy.WIDE_COMPILER:
            self._prebuild_wide_primitive_groups(device)
        elif strategy == ExecutionStrategy.AUTO:
            # Build both for fallback chain
            self._prebuild_vmap_groups(device)
            if _HAS_WIDE_COMPILER:
                self._prebuild_wide_primitive_groups(device)
        # SEQUENTIAL: nothing to pre-build

        if build_sub_ensembles:
            self.build_sub_ensembles()

        # Attach gradient debugger if configured
        if self._gradient_debugger.level > 0:
            self._gradient_debugger.attach()

        return self

    @torch.compiler.disable
    def _prebuild_wide_primitive_groups(self, device: torch.device = None) -> None:
        """
        Pre-build WidePrimitiveTowerGroups for registered towers.

        Parallel to _prebuild_vmap_groups. Walks aligned tower groups and
        builds Wide primitive execution plans using the wide_compiler registry.

        Args:
            device: Target device for computation
        """
        if not _HAS_WIDE_COMPILER:
            return

        if not self.objects.get('_analyzed', False):
            self.analyze_structure()

        # Clear existing groups
        self.objects['_wide_primitive_groups'].clear()
        self.objects['_non_wide_fusable_groups'].clear()

        # Group towers by signature
        sig_to_names: Dict[str, List[str]] = {}
        for name in self.tower_names:
            sig = self._tower_signature(name)
            if sig not in sig_to_names:
                sig_to_names[sig] = []
            sig_to_names[sig].append(name)

        # Build WidePrimitiveTowerGroup for each group with 2+ towers
        for sig, names in sig_to_names.items():
            if len(names) < 2:
                continue

            cache_key = (sig, frozenset(names))

            try:
                towers = [self[name] for name in names]
                group = WidePrimitiveTowerGroup(towers, names)
                group.prepare_for_forward(device)

                # Only store if we actually fused something
                if group._fusion_coverage.fused_ops > 0:
                    self.objects['_wide_primitive_groups'][cache_key] = group
                else:
                    # No ops fused — not worth the overhead
                    self.objects['_non_wide_fusable_groups'].add(cache_key)

            except Exception:
                self.objects['_non_wide_fusable_groups'].add(cache_key)

    @torch.compiler.disable
    def build_sub_ensembles(self) -> Dict[str, SubEnsembleGroup]:
        """
        Build sub-ensembles for multi-vantage gradient interpolation.

        For each group of 2+ aligned towers, creates both a VMapTowerGroup
        and a WidePrimitiveTowerGroup (if possible), wrapped in a
        SubEnsembleGroup with learnable interpolation weights.

        Returns:
            Dict mapping group key -> SubEnsembleGroup
        """
        if not self.objects.get('_analyzed', False):
            self.analyze_structure()

        sub_ensembles = {}

        # Group towers by signature
        sig_to_names: Dict[str, List[str]] = {}
        for name in self.tower_names:
            sig = self._tower_signature(name)
            if sig not in sig_to_names:
                sig_to_names[sig] = []
            sig_to_names[sig].append(name)

        # Infer device and dim
        device = None
        dim = 0
        for p in self.parameters():
            device = p.device
            dim = p.shape[-1] if p.dim() > 0 else 0
            break

        for sig, names in sig_to_names.items():
            if len(names) < 2:
                continue

            cache_key = (sig, frozenset(names))
            towers = [self[name] for name in names]
            se = SubEnsembleGroup(name=f"se_{sig[:20]}", dim=dim)

            # Always add vmap group
            try:
                vmap_group = VMapTowerGroup(towers, names)
                vmap_group.prepare_for_forward(device)
                se.add_group('vmap', vmap_group)
            except Exception:
                pass

            # Add wide_compiler group if available
            if _HAS_WIDE_COMPILER:
                try:
                    wide_group = WidePrimitiveTowerGroup(towers, names)
                    wide_group.prepare_for_forward(device)
                    if wide_group._fusion_coverage.fused_ops > 0:
                        se.add_group('wide', wide_group)
                except Exception:
                    pass

            if len(se.groups) > 0:
                sub_ensembles[str(cache_key)] = se

        self.objects['_sub_ensembles'] = sub_ensembles
        return sub_ensembles

    def wide_forward_ensemble(
        self,
        x: Tensor,
        tower_names: List[str] = None,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Execute towers using sub-ensemble pooling for multi-vantage gradients.

        Falls back to wide_forward() if sub-ensembles not built.

        Args:
            x: Input tensor [B, ...]
            tower_names: Specific towers (None = all registered)
            mask: Optional attention mask

        Returns:
            Dict mapping tower name -> pooled output tensor
        """
        sub_ensembles = self.objects.get('_sub_ensembles', {})
        if not sub_ensembles:
            return self.wide_forward(x, tower_names, mask)

        outputs = {}
        for se in sub_ensembles.values():
            se_outputs = se(x)
            outputs.update(se_outputs)

        # Execute any towers not covered by sub-ensembles
        names = tower_names or self.tower_names
        uncovered = [n for n in names if n not in outputs]
        if uncovered:
            for name in uncovered:
                tower = self[name]
                out = tower(x, mask=mask) if mask is not None else tower(x)
                if isinstance(out, tuple):
                    out = out[0]
                outputs[name] = out

        return outputs

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

        # Lazy-attach gradient debugger (towers must be registered first)
        self._maybe_attach_gradient_debugger()

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

    @staticmethod
    def _resolve_strategy_int(strategy: ExecutionStrategy, training: bool) -> int:
        """Resolve ExecutionStrategy enum to compile-safe int.

        Returns: 0=VMAP, 1=WIDE_COMPILER, 2=SEQUENTIAL
        """
        if strategy == ExecutionStrategy.AUTO:
            return 0 if training else 1  # VMAP for training, WIDE for eval
        elif strategy == ExecutionStrategy.VMAP:
            return 0
        elif strategy == ExecutionStrategy.WIDE_COMPILER:
            return 1
        else:
            return 2

    def _batched_tower_forward(
        self,
        x: Tensor,
        tower_names: List[str],
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Execute multiple aligned towers using the configured execution strategy.

        Strategy dispatch order:
        - WIDE_COMPILER: wide_compiler primitives -> sequential
        - VMAP: vmap(functional_call) -> sequential
        - AUTO: training=vmap, eval=wide_compiler -> vmap -> sequential
        - SEQUENTIAL: direct sequential execution

        Execution groups should be PRE-BUILT via prepare_for_compile().
        Falls back to sequential if no pre-built group or execution fails.
        """
        if not tower_names:
            return {}

        # Read pre-resolved strategy int (set at prepare_for_compile time)
        # 0=VMAP, 1=WIDE_COMPILER, 2=SEQUENTIAL
        # Falls back to 0 (VMAP) if not set — backwards compatible
        strategy_int = self.objects.get('_resolved_strategy', 0)
        sig = self._tower_signature(tower_names[0])
        cache_key = (sig, frozenset(tower_names))

        # Try wide_compiler primitives first (strategy_int == 1)
        if strategy_int == 1:
            wp_groups = self.objects.get('_wide_primitive_groups', {})
            non_wf = self.objects.get('_non_wide_fusable_groups', set())
            if cache_key not in non_wf and cache_key in wp_groups:
                try:
                    return wp_groups[cache_key](x)
                except (RuntimeError, Exception):
                    self._mark_non_wide_fusable(cache_key)

            # Fall through: AUTO has vmap as fallback, WIDE_COMPILER falls to sequential
            auto_fallback = self.objects.get('_auto_fallback', False)
            if not auto_fallback:
                return self._sequential_tower_forward(x, tower_names, mask)
            # else: fall through to vmap below

        # Try vmap (strategy_int == 0 or AUTO fallback from above)
        if strategy_int == 0 or (strategy_int == 1 and self.objects.get('_auto_fallback', False)):
            non_vmappable = self.objects.get('_non_vmappable_groups', set())
            if cache_key not in non_vmappable:
                vmap_groups = self.objects['_vmap_groups']
                if cache_key in vmap_groups:
                    try:
                        return vmap_groups[cache_key](x)
                    except (KeyError, RuntimeError):
                        if '_non_vmappable_groups' not in self.objects:
                            self.objects['_non_vmappable_groups'] = set()
                        self.objects['_non_vmappable_groups'].add(cache_key)
                        return self._sequential_tower_forward(x, tower_names, mask)

        # SEQUENTIAL fallback
        return self._sequential_tower_forward(x, tower_names, mask)

    @torch.compiler.disable
    def _maybe_attach_gradient_debugger(self) -> None:
        """Lazy-attach gradient debugger hooks. Called from wide_forward."""
        if self._gradient_debugger.level > 0 and not self._gradient_debugger._attached:
            self._gradient_debugger.attach()

    @torch.compiler.disable
    def _mark_non_wide_fusable(self, cache_key: Tuple) -> None:
        """Mark a group as non-fusable (outside compile path)."""
        self.objects['_non_wide_fusable_groups'].add(cache_key)

    def _sequential_tower_forward(
        self,
        x: Tensor,
        tower_names: List[str],
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Sequential fallback when vmap is not possible."""
        outputs = {}
        for name in tower_names:
            tower = self[name]
            if mask is not None:
                out = tower(x, mask=mask)
            else:
                out = tower(x)
            if isinstance(out, tuple):
                out = out[0]
            outputs[name] = out
        return outputs

    # =========================================================================
    # COMPILATION
    # =========================================================================

    def compile(
        self,
        prepare_and_compile=True,
        build_sub_ensembles: bool = False,
        **kwargs,
    ) -> 'WideRouter':
        """
        Compile the router for optimized execution.

        This is the primary optimization path. torch.compile automatically
        discovers aligned operations and fuses kernels.

        Args:
            prepare_and_compile: If True, call prepare_for_compile() before compiling.
            build_sub_ensembles: If True, build sub-ensembles for gradient interpolation.
            **kwargs: Passed to torch.compile (mode, fullgraph, etc.)

        Returns:
            Compiled router
        """
        if prepare_and_compile:
            self.prepare_for_compile(build_sub_ensembles=build_sub_ensembles)
        return torch.compile(self, **kwargs)

    def prepare_and_compile(self, build_sub_ensembles: bool = False, **kwargs) -> 'WideRouter':
        """
        Prepare and compile in one call.

        Convenience method that ensures execution groups are pre-built before
        torch.compile traces the forward pass. This minimizes graph breaks.

        Args:
            build_sub_ensembles: If True, build sub-ensembles.
            **kwargs: Passed to torch.compile

        Returns:
            Compiled router with execution groups pre-built
        """
        self.prepare_for_compile(build_sub_ensembles=build_sub_ensembles)
        return torch.compile(self, **kwargs)

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
        - Wide primitive groups cache
        - Sub-ensemble gradient tracking
        """
        super().reset()  # Clears self._cache and recurses to components
        self.pool.clear()
        self.objects['_vmap_groups'].clear()
        self.objects.get('_wide_primitive_groups', {}).clear()
        for se in self.objects.get('_sub_ensembles', {}).values():
            if hasattr(se, '_gradient_norms'):
                se._gradient_norms.clear()

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
        stats = {
            **self.objects['wide_stats'],
            'pool_stats': self.pool.stats,
            'registry_stats': self.registry.get_stats(),
            'registered_towers': len(self.tower_names),
            'alignments': self.objects.get('_alignments', {}),
            'execution_strategy': self.objects.get('_execution_strategy', ExecutionStrategy.VMAP).name,
            'vmap_groups': len(self.objects.get('_vmap_groups', {})),
            'wide_primitive_groups': len(self.objects.get('_wide_primitive_groups', {})),
            'sub_ensembles': len(self.objects.get('_sub_ensembles', {})),
        }

        # Include fusion coverage per wide group
        fusion_coverages = {}
        for key, group in self.objects.get('_wide_primitive_groups', {}).items():
            if hasattr(group, '_fusion_coverage'):
                fusion_coverages[str(key)] = group._fusion_coverage.summary
        if fusion_coverages:
            stats['fusion_coverage'] = fusion_coverages

        return stats

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.objects['wide_stats'] = {
            'wide_forwards': 0,
            'towers_executed': 0,
            'alignment_hits': 0,
        }
        self.pool.stats = {k: 0 for k in self.pool.stats}

    # =========================================================================
    # DIAGNOSTIC METHODS (Phase A)
    # =========================================================================

    def gradient_report(self) -> str:
        """Human-readable gradient health report across all execution paths."""
        return self._gradient_debugger.report()

    def check_fusion_health(self) -> FusionDiagnostic:
        """Check Wide primitive fusion gradient health."""
        return self._gradient_debugger.check_fusion_health()

    def get_fusion_coverage(self) -> Dict[str, FusionCoverage]:
        """Per-group fusion coverage stats (what % of ops fused)."""
        return {
            str(k): g._fusion_coverage
            for k, g in self.objects.get('_wide_primitive_groups', {}).items()
            if hasattr(g, '_fusion_coverage')
        }

    def validate_cache_devices(self) -> List[str]:
        """Check all caches for device mismatches. Returns issues list."""
        return self._cache_controller.validate_cache_devices()

    # =========================================================================
    # DEVICE MANAGEMENT OVERRIDE
    # =========================================================================

    def network_to(self, device=None, dtype=None, **kwargs):
        """
        Move router to device with cache controller integration.

        Brackets the base network_to with cache controller pre/post hooks
        to ensure caches respect the device movement policy.
        """
        self._cache_controller.pre_device_move(device)
        result = super().network_to(device=device, dtype=dtype, **kwargs)
        self._cache_controller.post_device_move(device)
        return result

    def __repr__(self) -> str:
        tower_count = len(self.tower_names)
        prepared = self.objects.get('_prepared_for_compile', False)
        strategy = self.objects.get('_execution_strategy', ExecutionStrategy.VMAP).name
        wp_groups = len(self.objects.get('_wide_primitive_groups', {}))
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"towers={tower_count}, "
            f"strategy={strategy}, "
            f"analyzed={self.objects['_analyzed']}, "
            f"prepared={prepared}, "
            f"wide_groups={wp_groups}"
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

    print("\nPreparing and compiling...")

    # compile() now auto-prepares (builds vmap groups, stacks state)
    compiled_baseline = torch.compile(baseline)
    compiled_wide = wide.compile()  # Calls prepare_for_compile() internally

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

        # compile() now calls prepare_for_compile() automatically
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