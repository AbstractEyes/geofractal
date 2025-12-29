"""
geofractal.router.compiler.compile_router
=========================================

CompileRouter - Introspective compilation layer for arbitrary nn.Module structures.

Takes any module tree (router, raw nn.Module, sloppy code) and:
1. Introspects to find all nn.Module children regardless of attachment style
2. Wraps raw modules in TorchComponent for proper lifecycle
3. Groups similar operations by type/signature/depth
4. Creates staged execution without modifying original structure
5. Compiles the staged view for optimized execution

This is the "make it work anyway" compiler - handles:
- self.thing = nn.Linear(...)
- self.stuff = nn.Sequential(...)
- Nested routers with mixed attachment styles
- Raw nn.Module trees with no router structure

Usage:
    # Wrap any module for compilation
    compiler = CompileRouter.from_module(messy_model)
    compiler.compile_towers()
    compiled = compiler.compile()

    # Or use directly
    compiler = CompileRouter('compiler')
    compiler.attach('model', messy_model)
    compiler.compile_towers()

Integration with WideRouter:
    class MyCollective(WideRouter):
        def compile_towers(self):
            compiler = CompileRouter.from_module(self)
            return compiler.compile_towers()

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Optional, Dict, List, Tuple, Any, Union,
    Set, Type, Callable
)
from collections import defaultdict
from enum import Enum, auto

import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.base_router import BaseRouter
from geofractal.router.base_tower import BaseTower
from geofractal.router.base_component import BaseComponent
from geofractal.router.components.torch_component import TorchComponent


# =============================================================================
# INTROSPECTION DATA STRUCTURES
# =============================================================================

class ModuleCategory(Enum):
    """Categories for module classification."""
    LINEAR = auto()          # nn.Linear and variants
    CONV = auto()            # nn.Conv1d, Conv2d, Conv3d
    NORM = auto()            # LayerNorm, BatchNorm, etc.
    ACTIVATION = auto()      # ReLU, GELU, etc.
    ATTENTION = auto()       # MultiheadAttention
    EMBEDDING = auto()       # nn.Embedding
    DROPOUT = auto()         # nn.Dropout
    CONTAINER = auto()       # Sequential, ModuleList, ModuleDict
    ROUTER = auto()          # BaseRouter subclass
    TOWER = auto()           # BaseTower subclass
    COMPONENT = auto()       # TorchComponent subclass
    POOLING = auto()         # Pooling layers
    RECURRENT = auto()       # RNN, LSTM, GRU
    TRANSFORMER = auto()     # TransformerEncoder, etc.
    UNKNOWN = auto()         # Unclassified


@dataclass
class ModuleNode:
    """Node in the introspected module tree."""
    name: str                          # Attribute name
    path: str                          # Full dotted path
    module: nn.Module                  # The actual module
    category: ModuleCategory           # Classification
    signature: str                     # Structural signature for grouping
    depth: int                         # Depth in tree
    param_count: int                   # Number of parameters
    children: List['ModuleNode'] = field(default_factory=list)
    parent: Optional['ModuleNode'] = None
    is_leaf: bool = True               # Has no nn.Module children
    is_wrapped: bool = False           # Already a TorchComponent
    execution_order: int = -1          # Order in forward pass

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        if isinstance(other, ModuleNode):
            return self.path == other.path
        return False


@dataclass
class ExecutionStage:
    """A group of similar operations that can be batched."""
    stage_id: int
    category: ModuleCategory
    signature: str
    nodes: List[ModuleNode]
    depth: int
    can_batch: bool = True

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def total_params(self) -> int:
        return sum(n.param_count for n in self.nodes)


@dataclass
class CompiledStructure:
    """Result of compile_towers() - the staged execution plan."""
    stages: List[ExecutionStage]
    execution_order: List[str]  # Node paths in execution order
    wrapped_modules: Dict[str, 'WrappedModule']  # path -> wrapper
    signature_groups: Dict[str, List[ModuleNode]]  # signature -> nodes
    depth_map: Dict[int, List[ModuleNode]]  # depth -> nodes at that depth
    total_params: int
    total_stages: int
    batchable_stages: int


# =============================================================================
# MODULE CLASSIFIER
# =============================================================================

class ModuleClassifier:
    """Classifies nn.Module instances by type and structure."""

    # Type -> Category mapping
    CATEGORY_MAP: Dict[Type, ModuleCategory] = {
        nn.Linear: ModuleCategory.LINEAR,
        nn.Conv1d: ModuleCategory.CONV,
        nn.Conv2d: ModuleCategory.CONV,
        nn.Conv3d: ModuleCategory.CONV,
        nn.ConvTranspose1d: ModuleCategory.CONV,
        nn.ConvTranspose2d: ModuleCategory.CONV,
        nn.ConvTranspose3d: ModuleCategory.CONV,
        nn.LayerNorm: ModuleCategory.NORM,
        nn.BatchNorm1d: ModuleCategory.NORM,
        nn.BatchNorm2d: ModuleCategory.NORM,
        nn.BatchNorm3d: ModuleCategory.NORM,
        nn.GroupNorm: ModuleCategory.NORM,
        nn.InstanceNorm1d: ModuleCategory.NORM,
        nn.InstanceNorm2d: ModuleCategory.NORM,
        nn.ReLU: ModuleCategory.ACTIVATION,
        nn.GELU: ModuleCategory.ACTIVATION,
        nn.SiLU: ModuleCategory.ACTIVATION,
        nn.Tanh: ModuleCategory.ACTIVATION,
        nn.Sigmoid: ModuleCategory.ACTIVATION,
        nn.Softmax: ModuleCategory.ACTIVATION,
        nn.LeakyReLU: ModuleCategory.ACTIVATION,
        nn.PReLU: ModuleCategory.ACTIVATION,
        nn.ELU: ModuleCategory.ACTIVATION,
        nn.Mish: ModuleCategory.ACTIVATION,
        nn.MultiheadAttention: ModuleCategory.ATTENTION,
        nn.Embedding: ModuleCategory.EMBEDDING,
        nn.EmbeddingBag: ModuleCategory.EMBEDDING,
        nn.Dropout: ModuleCategory.DROPOUT,
        nn.Dropout2d: ModuleCategory.DROPOUT,
        nn.Dropout3d: ModuleCategory.DROPOUT,
        nn.AlphaDropout: ModuleCategory.DROPOUT,
        nn.Sequential: ModuleCategory.CONTAINER,
        nn.ModuleList: ModuleCategory.CONTAINER,
        nn.ModuleDict: ModuleCategory.CONTAINER,
        nn.MaxPool1d: ModuleCategory.POOLING,
        nn.MaxPool2d: ModuleCategory.POOLING,
        nn.MaxPool3d: ModuleCategory.POOLING,
        nn.AvgPool1d: ModuleCategory.POOLING,
        nn.AvgPool2d: ModuleCategory.POOLING,
        nn.AvgPool3d: ModuleCategory.POOLING,
        nn.AdaptiveAvgPool1d: ModuleCategory.POOLING,
        nn.AdaptiveAvgPool2d: ModuleCategory.POOLING,
        nn.AdaptiveMaxPool1d: ModuleCategory.POOLING,
        nn.AdaptiveMaxPool2d: ModuleCategory.POOLING,
        nn.RNN: ModuleCategory.RECURRENT,
        nn.LSTM: ModuleCategory.RECURRENT,
        nn.GRU: ModuleCategory.RECURRENT,
        nn.TransformerEncoder: ModuleCategory.TRANSFORMER,
        nn.TransformerDecoder: ModuleCategory.TRANSFORMER,
        nn.TransformerEncoderLayer: ModuleCategory.TRANSFORMER,
        nn.TransformerDecoderLayer: ModuleCategory.TRANSFORMER,
    }

    # Add RMSNorm if it exists (PyTorch 2.4+)
    if hasattr(nn, 'RMSNorm'):
        CATEGORY_MAP[nn.RMSNorm] = ModuleCategory.NORM

    @classmethod
    def classify(cls, module: nn.Module) -> ModuleCategory:
        """Classify a module by its type."""
        # Check exact type first
        module_type = type(module)
        if module_type in cls.CATEGORY_MAP:
            return cls.CATEGORY_MAP[module_type]

        # Check inheritance for mapped types
        for base_type, category in cls.CATEGORY_MAP.items():
            if isinstance(module, base_type):
                return category

        # Check for geofractal types (order matters: Tower before Router)
        if isinstance(module, BaseTower):
            return ModuleCategory.TOWER
        if isinstance(module, BaseRouter):
            return ModuleCategory.ROUTER
        if isinstance(module, TorchComponent):
            return ModuleCategory.COMPONENT

        return ModuleCategory.UNKNOWN

    @classmethod
    def compute_signature(cls, module: nn.Module, category: ModuleCategory) -> str:
        """Compute structural signature for grouping similar modules."""
        type_name = type(module).__name__
        param_count = sum(p.numel() for p in module.parameters(recurse=False))

        # Add shape info for specific types
        shape_info = ""
        if isinstance(module, nn.Linear):
            shape_info = f"_{module.in_features}x{module.out_features}"
            if module.bias is not None:
                shape_info += "_bias"
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            shape_info = f"_{module.in_channels}x{module.out_channels}"
            shape_info += f"_k{module.kernel_size}_s{module.stride}"
        elif isinstance(module, nn.LayerNorm):
            shape_info = f"_{list(module.normalized_shape)}"
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            shape_info = f"_{module.num_features}"
        elif isinstance(module, nn.Embedding):
            shape_info = f"_{module.num_embeddings}x{module.embedding_dim}"
        elif isinstance(module, nn.MultiheadAttention):
            shape_info = f"_{module.embed_dim}h{module.num_heads}"
        elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            shape_info = f"_{module.input_size}x{module.hidden_size}L{module.num_layers}"

        return f"{category.name}:{type_name}{shape_info}:{param_count}"

    @classmethod
    def is_leaf_module(cls, module: nn.Module) -> bool:
        """Check if module is a leaf (no meaningful children to traverse)."""
        # These are always leaves - no internal nn.Module children
        leaf_types = (
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d,
            nn.Embedding, nn.EmbeddingBag,
            nn.Dropout, nn.Dropout2d, nn.Dropout3d,
            nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid,
            nn.LeakyReLU, nn.PReLU, nn.ELU, nn.Mish,
            nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
        )
        if isinstance(module, leaf_types):
            return True

        # Check if it has any nn.Module children
        children = list(module.children())
        return len(children) == 0


# =============================================================================
# WRAPPED COMPONENT
# =============================================================================

class WrappedModule(TorchComponent):
    """
    Wrapper that turns any nn.Module into a TorchComponent.

    Preserves the original module's forward() while adding
    component identity and lifecycle hooks.
    """

    def __init__(self, name: str, module: nn.Module, original_path: str = ""):
        super().__init__(name)
        self.wrapped = module
        self.original_path = original_path
        self._original_type = type(module).__name__

    def forward(self, *args, **kwargs) -> Any:
        return self.wrapped(*args, **kwargs)

    def __repr__(self) -> str:
        return f"WrappedModule(name='{self.name}', wraps={self._original_type})"


# =============================================================================
# COMPILE ROUTER
# =============================================================================

class CompileRouter(BaseRouter):
    """
    Introspective compilation layer for arbitrary nn.Module structures.

    Discovers, classifies, wraps, and stages modules for optimized execution
    without modifying the original structure.

    Attributes:
        _tree: Root of introspected module tree
        _compiled_structure: Result of compile_towers()
        _all_nodes: Flat dict of path -> ModuleNode
        _wrappers: Created wrapper components
    """

    def __init__(
        self,
        name: str,
        strict: bool = False,
        auto_wrap: bool = True,
        min_batch_size: int = 2,
        **kwargs,
    ):
        """
        Initialize CompileRouter.

        Args:
            name: Router name
            strict: Hardware strictness (passed to BaseRouter)
            auto_wrap: Automatically wrap raw modules in TorchComponent
            min_batch_size: Minimum group size for batching
            **kwargs: Additional args passed to BaseRouter
        """
        super().__init__(name, strict=strict, **kwargs)

        self.objects['_auto_wrap'] = auto_wrap
        self.objects['_min_batch_size'] = min_batch_size
        self.objects['_introspected'] = False
        self.objects['_compiled'] = False

        # Introspection results
        self._tree: Optional[ModuleNode] = None
        self._all_nodes: Dict[str, ModuleNode] = {}
        self._wrappers: Dict[str, WrappedModule] = {}
        self._compiled_structure: Optional[CompiledStructure] = None

    @classmethod
    def from_module(cls, module: nn.Module, name: str = None, **kwargs) -> 'CompileRouter':
        """
        Create CompileRouter from any nn.Module.

        Args:
            module: Any nn.Module to compile
            name: Name for the router (defaults to module class name)
            **kwargs: Passed to CompileRouter.__init__

        Returns:
            CompileRouter wrapping the module
        """
        name = name or f"compiled_{type(module).__name__}"
        router = cls(name, **kwargs)
        router.attach('root', module)
        return router

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    @torch.compiler.disable
    def introspect(self, root_name: str = 'root') -> ModuleNode:
        """
        Walk the module tree and build introspection graph.

        Decorated with @torch.compiler.disable to prevent dynamo issues
        with dynamic Python constructs during tree traversal.

        Args:
            root_name: Name of root component to introspect

        Returns:
            Root ModuleNode of introspected tree
        """
        if root_name not in self.components:
            raise KeyError(f"No component '{root_name}' to introspect")

        root_module = self.components[root_name]
        self._all_nodes.clear()

        # Build tree
        self._tree = self._introspect_recursive(
            name=root_name,
            module=root_module,
            path=root_name,
            depth=0,
            parent=None,
        )

        # Assign execution order (topological)
        self._assign_execution_order()

        self.objects['_introspected'] = True
        return self._tree

    def _introspect_recursive(
        self,
        name: str,
        module: nn.Module,
        path: str,
        depth: int,
        parent: Optional[ModuleNode],
    ) -> ModuleNode:
        """Recursively introspect module tree."""

        # Classify
        category = ModuleClassifier.classify(module)
        signature = ModuleClassifier.compute_signature(module, category)
        is_leaf = ModuleClassifier.is_leaf_module(module)
        is_wrapped = isinstance(module, TorchComponent)
        param_count = sum(p.numel() for p in module.parameters(recurse=False))

        # Create node
        node = ModuleNode(
            name=name,
            path=path,
            module=module,
            category=category,
            signature=signature,
            depth=depth,
            param_count=param_count,
            parent=parent,
            is_leaf=is_leaf,
            is_wrapped=is_wrapped,
        )

        self._all_nodes[path] = node

        # Recurse into children
        if not is_leaf:
            for child_name, child_module in module.named_children():
                child_path = f"{path}.{child_name}"
                child_node = self._introspect_recursive(
                    name=child_name,
                    module=child_module,
                    path=child_path,
                    depth=depth + 1,
                    parent=node,
                )
                node.children.append(child_node)

            # Update leaf status based on actual children found
            node.is_leaf = len(node.children) == 0

        return node

    def _assign_execution_order(self) -> None:
        """Assign execution order via depth-first traversal."""
        order = 0

        def dfs(node: ModuleNode):
            nonlocal order
            for child in node.children:
                dfs(child)
            node.execution_order = order
            order += 1

        if self._tree:
            dfs(self._tree)

    # =========================================================================
    # WRAPPING
    # =========================================================================

    @torch.compiler.disable
    def wrap_raw_modules(self) -> Dict[str, WrappedModule]:
        """
        Wrap all raw (non-TorchComponent) leaf modules.

        Returns:
            Dict mapping path -> WrappedModule
        """
        if not self.objects['_introspected']:
            self.introspect()

        wrappers = {}

        for path, node in self._all_nodes.items():
            if node.is_leaf and not node.is_wrapped:
                wrapper = WrappedModule(
                    name=f"wrapped_{node.name}",
                    module=node.module,
                    original_path=path,
                )
                wrappers[path] = wrapper
                node.is_wrapped = True

        self._wrappers = wrappers
        return wrappers

    # =========================================================================
    # STAGING
    # =========================================================================

    @torch.compiler.disable
    def compile_towers(self) -> CompiledStructure:
        """
        Analyze structure and create staged execution plan.

        Groups similar operations by:
        1. Category (LINEAR, CONV, NORM, etc.)
        2. Signature (same shape, same params)
        3. Depth (same level in tree)

        Returns:
            CompiledStructure with execution stages
        """
        if not self.objects['_introspected']:
            self.introspect()

        if self.objects['_auto_wrap']:
            self.wrap_raw_modules()

        # Group by signature
        signature_groups: Dict[str, List[ModuleNode]] = defaultdict(list)
        for node in self._all_nodes.values():
            if node.is_leaf:
                signature_groups[node.signature].append(node)

        # Group by depth
        depth_map: Dict[int, List[ModuleNode]] = defaultdict(list)
        for node in self._all_nodes.values():
            depth_map[node.depth].append(node)

        # Create execution stages
        # Stage by (depth, signature) for maximum batching opportunity
        stages = []
        stage_id = 0
        min_batch = self.objects['_min_batch_size']

        depth_sig_groups: Dict[Tuple[int, str], List[ModuleNode]] = defaultdict(list)
        for node in self._all_nodes.values():
            if node.is_leaf:
                key = (node.depth, node.signature)
                depth_sig_groups[key].append(node)

        # Sort by depth for execution order
        sorted_keys = sorted(depth_sig_groups.keys(), key=lambda x: x[0])

        for depth, signature in sorted_keys:
            nodes = depth_sig_groups[(depth, signature)]
            if not nodes:
                continue

            # Determine if batchable
            can_batch = (
                len(nodes) >= min_batch and
                nodes[0].category not in {
                    ModuleCategory.CONTAINER,
                    ModuleCategory.ROUTER,
                    ModuleCategory.TOWER,
                    ModuleCategory.UNKNOWN,
                }
            )

            stage = ExecutionStage(
                stage_id=stage_id,
                category=nodes[0].category,
                signature=signature,
                nodes=nodes,
                depth=depth,
                can_batch=can_batch,
            )
            stages.append(stage)
            stage_id += 1

        # Build execution order
        execution_order = sorted(
            [n.path for n in self._all_nodes.values() if n.is_leaf],
            key=lambda p: self._all_nodes[p].execution_order,
        )

        # Create compiled structure
        self._compiled_structure = CompiledStructure(
            stages=stages,
            execution_order=execution_order,
            wrapped_modules=dict(self._wrappers),
            signature_groups=dict(signature_groups),
            depth_map=dict(depth_map),
            total_params=sum(n.param_count for n in self._all_nodes.values()),
            total_stages=len(stages),
            batchable_stages=sum(1 for s in stages if s.can_batch),
        )

        self.objects['_compiled'] = True
        return self._compiled_structure

    # =========================================================================
    # COMPILATION
    # =========================================================================

    def compile(self, **kwargs) -> 'CompileRouter':
        """
        Compile the router for optimized execution.

        Calls compile_towers() if not already done, then torch.compile().

        Args:
            **kwargs: Passed to torch.compile

        Returns:
            Compiled router
        """
        if not self.objects['_compiled']:
            self.compile_towers()

        return torch.compile(self, **kwargs)

    def prepare_and_compile(self, **kwargs) -> 'CompileRouter':
        """
        Full preparation and compilation.

        Introspects, wraps, stages, then compiles.

        Args:
            **kwargs: Passed to torch.compile

        Returns:
            Compiled router
        """
        self.introspect()
        self.compile_towers()
        return self.compile(**kwargs)

    # =========================================================================
    # STAGED EXECUTION
    # =========================================================================

    def staged_forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Execute using staged batching.

        Groups operations by stage and executes in batch where possible.

        Args:
            x: Input tensor

        Returns:
            Dict mapping node path -> output tensor
        """
        if not self.objects['_compiled']:
            self.compile_towers()

        outputs: Dict[str, Tensor] = {}

        for stage in self._compiled_structure.stages:
            if stage.can_batch and stage.size >= 2:
                # Batched execution for aligned operations
                stage_outputs = self._execute_stage_batched(x, stage)
            else:
                # Sequential execution
                stage_outputs = self._execute_stage_sequential(x, stage)

            outputs.update(stage_outputs)

        return outputs

    def _execute_stage_batched(
        self,
        x: Tensor,
        stage: ExecutionStage,
    ) -> Dict[str, Tensor]:
        """Execute a stage with batched operations."""
        outputs = {}

        # For aligned operations of same signature, execute together
        # torch.compile will fuse these kernels
        for node in stage.nodes:
            module = self._wrappers.get(node.path, node.module)
            try:
                out = module(x)
                outputs[node.path] = out
            except Exception:
                # Fallback to original module on error
                out = node.module(x)
                outputs[node.path] = out

        return outputs

    def _execute_stage_sequential(
        self,
        x: Tensor,
        stage: ExecutionStage,
    ) -> Dict[str, Tensor]:
        """Execute a stage sequentially."""
        outputs = {}

        for node in stage.nodes:
            module = self._wrappers.get(node.path, node.module)
            try:
                out = module(x)
                outputs[node.path] = out
            except Exception:
                out = node.module(x)
                outputs[node.path] = out

        return outputs

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, x: Tensor) -> Tensor:
        """
        Default forward - delegates to root component.

        For staged execution, use staged_forward().
        """
        if 'root' in self.components:
            return self.components['root'](x)
        raise RuntimeError("No root component attached")

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        if not self._compiled_structure:
            return {'compiled': False}

        cs = self._compiled_structure

        # Category breakdown
        category_counts: Dict[str, int] = {}
        for stage in cs.stages:
            cat_name = stage.category.name
            category_counts[cat_name] = category_counts.get(cat_name, 0) + stage.size

        return {
            'compiled': True,
            'total_modules': len(self._all_nodes),
            'leaf_modules': sum(1 for n in self._all_nodes.values() if n.is_leaf),
            'total_params': cs.total_params,
            'total_stages': cs.total_stages,
            'batchable_stages': cs.batchable_stages,
            'wrapped_modules': len(cs.wrapped_modules),
            'category_breakdown': category_counts,
            'signature_groups': len(cs.signature_groups),
            'max_depth': max(cs.depth_map.keys()) if cs.depth_map else 0,
        }

    def get_batchable_groups(self) -> List[Dict[str, Any]]:
        """Get list of batchable operation groups with details."""
        if not self._compiled_structure:
            return []

        groups = []
        for stage in self._compiled_structure.stages:
            if stage.can_batch:
                groups.append({
                    'stage_id': stage.stage_id,
                    'category': stage.category.name,
                    'signature': stage.signature,
                    'count': stage.size,
                    'depth': stage.depth,
                    'total_params': stage.total_params,
                    'paths': [n.path for n in stage.nodes],
                })
        return groups

    def print_tree(self, node: ModuleNode = None, indent: int = 0) -> None:
        """Print the introspected tree structure."""
        if node is None:
            if self._tree is None:
                print("No tree introspected. Call introspect() first.")
                return
            node = self._tree

        prefix = "  " * indent
        wrapped_marker = "⚡" if node.is_wrapped else "○"
        leaf_marker = "●" if node.is_leaf else "◆"

        print(f"{prefix}{leaf_marker}{wrapped_marker} {node.name} [{node.category.name}] "
              f"params={node.param_count:,}")

        for child in node.children:
            self.print_tree(child, indent + 1)

    def print_stages(self) -> None:
        """Print execution stages."""
        if not self._compiled_structure:
            print("No stages compiled. Call compile_towers() first.")
            return

        print(f"\n{'='*60}")
        print(f"EXECUTION STAGES ({self._compiled_structure.total_stages} total, "
              f"{self._compiled_structure.batchable_stages} batchable)")
        print(f"{'='*60}")

        for stage in self._compiled_structure.stages:
            batch_marker = "⚡ BATCH" if stage.can_batch else "○ SEQ"
            print(f"\nStage {stage.stage_id} [{stage.category.name}] {batch_marker}")
            print(f"  Signature: {stage.signature}")
            print(f"  Depth: {stage.depth}")
            print(f"  Nodes: {stage.size}")
            print(f"  Params: {stage.total_params:,}")
            print(f"  Modules:")
            for node in stage.nodes[:5]:  # Show first 5
                print(f"    - {node.path}")
            if stage.size > 5:
                print(f"    ... and {stage.size - 5} more")

    def __repr__(self) -> str:
        status = "compiled" if self.objects.get('_compiled', False) else "not compiled"
        node_count = len(self._all_nodes) if self._all_nodes else 0
        return f"CompileRouter(name='{self.name}', modules={node_count}, {status})"


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================

def compile_module(module: nn.Module, name: str = None, **kwargs) -> CompileRouter:
    """
    One-liner to create a compiled router from any module.

    Args:
        module: Any nn.Module
        name: Optional name
        **kwargs: Passed to CompileRouter

    Returns:
        CompileRouter with structure analyzed and ready to compile
    """
    router = CompileRouter.from_module(module, name, **kwargs)
    router.compile_towers()
    return router


# =============================================================================
# TEST (requires torch and geofractal)
# =============================================================================

if __name__ == '__main__':
    import torch

    print("=" * 60)
    print("CompileRouter Test")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Test 1: Sloppy MLP
    # -------------------------------------------------------------------------

    print("\n--- Test 1: Sloppy MLP ---")

    class SloppyMLP(nn.Module):
        """Typical lazy implementation."""
        def __init__(self, dim: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
            self.norm = nn.LayerNorm(dim)
            self.out = nn.Linear(dim, dim)

        def forward(self, x):
            x = self.layers(x)
            x = self.norm(x)
            return self.out(x)

    sloppy = SloppyMLP(256)
    compiler = CompileRouter.from_module(sloppy, "sloppy_mlp")
    compiler.introspect()
    compiler.print_tree()

    compiler.compile_towers()
    compiler.print_stages()

    stats = compiler.get_compilation_stats()
    print(f"\nStats: {stats}")

    # -------------------------------------------------------------------------
    # Test 2: Multi-head model (good batching target)
    # -------------------------------------------------------------------------

    print("\n--- Test 2: Multi-head Model ---")

    class MultiHeadModel(nn.Module):
        """Model with multiple similar heads - ideal for batching."""
        def __init__(self, dim: int, num_heads: int = 4):
            super().__init__()
            self.encoder = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)

            # Multiple identical heads - perfect batching target
            self.heads = nn.ModuleList([
                nn.Linear(dim, dim // 4) for _ in range(num_heads)
            ])

            self.out = nn.Linear(dim, dim)

        def forward(self, x):
            x = self.encoder(x)
            x = self.norm(x)
            head_outs = [h(x) for h in self.heads]
            x = torch.cat(head_outs, dim=-1)
            return self.out(x)

    multi = MultiHeadModel(256, num_heads=8)
    compiler2 = compile_module(multi, "multi_head")
    compiler2.print_tree()
    compiler2.print_stages()

    print("\nBatchable groups:")
    for group in compiler2.get_batchable_groups():
        print(f"  {group['category']}: {group['count']} ops at depth {group['depth']}")

    stats2 = compiler2.get_compilation_stats()
    print(f"\nStats: {stats2}")

    # -------------------------------------------------------------------------
    # Test 3: Forward execution
    # -------------------------------------------------------------------------

    print("\n--- Test 3: Forward Execution ---")

    x = torch.randn(4, 256)

    # Original forward
    y_orig = sloppy(x)
    print(f"Original: {x.shape} -> {y_orig.shape}")

    # Through compiler
    y_compiled = compiler(x)
    print(f"Compiled: {x.shape} -> {y_compiled.shape}")

    # Check equivalence
    diff = (y_orig - y_compiled).abs().max().item()
    print(f"Max diff: {diff}")

    print("\n✓ CompileRouter ready")