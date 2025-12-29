"""
geofractal.router.base_router
=============================

Abstract base for all routers in the geofractal system.

A router is a named module container with unique identity that holds
components as submodules. It provides minimal structure with maximal
flexibility - specialization comes entirely from subclassing and
component attachment.

Design Principles:
    - Everything is an nn.Module (the router itself)
    - Components are ENCOURAGED to be nn.Module but not required
    - Module components get automatic parameter registration
    - Non-module components are stored separately
    - No dimension constraints (models know their own shapes)
    - No enforced data flow (subclass decides topology)
    - Override what you need, ignore the rest

Storage Types:
    Routers have three storage mechanisms for different use cases:

    1. components (ModuleDict): nn.Module submodules with parameter tracking
       - Use for learnable layers, nested routers
       - Automatically included in state_dict, .to(), optimizer

    2. objects (dict): Persistent non-module data
       - Use for configs, metadata, constants
       - NOT moved by .to() or network_to()
       - NOT included in state_dict (serialize manually if needed)

    3. _cache (dict): Ephemeral tensor storage
       - Use for intermediate tensors during forward pass
       - Cleared automatically via reset() or cache_clear()
       - NOT moved by .to() - use cache_to() for explicit movement
       - NOT included in state_dict (intentionally transient)

Cache System:
    The cache provides managed lifecycle for intermediate tensors to
    prevent memory leaks from stale references. Unlike objects[], cache
    entries are expected to be cleared after each forward pass.

    # Store intermediate tensor
    self.cache_set('features', features)

    # Retrieve (returns None if not found)
    features = self.cache_get('features')

    # Clear after forward pass
    self.cache_clear()

    # Clear entire router tree
    self.cache_clear_recursive()

    # Debug: inspect cache across all nested routers
    debug_info = router.cache_debug()

    # Move cache tensors to device (explicit, not automatic)
    router.cache_to(device='cuda:0')

Hardware Control:
    Routers provide optional strict controls for device and dtype
    management across complex multi-model systems.

    - strict: Warns when using raw torch.to() which bypasses safeguards
    - strict_dtype: Locks router to specific dtype, validates on attach
    - strict_device: Locks router to specific device, validates on attach
    - network_to: Moves entire router subnetwork with configurable behavior

Accelerate Compatibility:
    The cache is intentionally NOT managed by PyTorch's parameter/buffer
    system. This means:

    - accelerate's device_map won't move cache tensors
    - .to() and network_to() won't move cache tensors
    - For multi-GPU: clear cache before device changes, or use cache_to()

    Recommended pattern with accelerate:
        model.reset()  # Clears cache
        model = accelerate.prepare(model)

Implications:
    - Routers can hold any object as a component
    - nn.Module components register parameters automatically
    - Non-module components (configs, buffers, callables) stored separately
    - Multiple routers can share components by reference
    - Routers can be nested (router as component of another router)
    - Networks emerge from router composition, not inheritance
    - Serialization works automatically via PyTorch state_dict for modules

Usage:
    class MyRouter(BaseRouter):
        def __init__(self, name: str):
            super().__init__(name)
            self.attach('encoder', nn.Linear(512, 256))
            self.attach('decoder', nn.Linear(256, 512))
            self.attach('config', {'dropout': 0.1})  # Non-module OK

        def forward(self, x: Tensor) -> Tensor:
            x = self['encoder'](x)

            # Cache intermediate for later retrieval
            self.cache_set('encoded', x)

            x = self['decoder'](x)

            # Clear cache at end of forward
            self.cache_clear()
            return x

    # Pythonic access
    router.has('encoder')       # True
    'encoder' in router         # True
    router['encoder']           # nn.Linear(512, 256)
    router['config']            # {'dropout': 0.1}
    router.device               # torch.device('cuda:0')
    router.dtype                # torch.float32

    # Hardware-controlled router
    router = MyRouter('gpu_router', strict_device=torch.device('cuda:0'))
    router.network_to(device='cuda:0', dtype=torch.float16)

    # Cache management
    router.cache_set('temp', tensor)
    router.cache_get('temp')
    router.cache_clear()
    router.cache_to(device='cuda:0')  # Explicit cache movement

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import uuid as uuid_lib
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union, Any

import torch
import torch.nn as nn
from torch import Tensor


class BaseRouter(nn.Module, ABC):
    """
    Abstract base for routers.

    A router is an addressable module container. It has a human-readable
    name, a unique identifier, and holds components.

    The only requirement is implementing `forward()`. Everything else
    is optional convenience.

    Components can be any object:
        - nn.Module: Registered as submodule (parameters tracked)
        - Any other object: Stored in auxiliary dict (no parameter tracking)

    Attributes:
        name: Human-readable identifier.
        uuid: Unique identifier for machine addressing.
        components: ModuleDict of nn.Module components.
        objects: Dict of non-module components.
        strict: If True, warn on unbounded to() usage.
        strict_dtype: If set, enforce dtype on attach.
        strict_device: If set, enforce device on attach.
    """

    def __init__(
        self,
        name: str,
        uuid: Optional[str] = None,
        strict: bool = True,
        strict_dtype: Optional[torch.dtype] = None,
        strict_device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize router.

        Args:
            name: Human-readable name for this router.
            uuid: Unique identifier. Generated if not provided.
            strict: Warn on raw to() usage if True.
            strict_dtype: Lock dtype. Validates module components on attach.
            strict_device: Lock device. Validates module components on attach.
        """
        super().__init__()
        self.name = name
        self.uuid = uuid or str(uuid_lib.uuid4())
        self.components = nn.ModuleDict()
        self.objects: dict = {}

        # Ephemeral cache - for intermediate tensors during forward pass
        # Use cache_set/cache_get/cache_clear for managed lifecycle
        self._cache: dict = {}

        # Hardware control
        self.strict = strict
        self.strict_dtype = strict_dtype
        self.strict_device = torch.device(strict_device) if strict_device else None
        self._to_warned = False

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Core routing operation.

        Override this with your routing logic and concrete signature.
        This is the only method you must implement.

        Subclasses define their own signatures:
            def forward(self, x: Tensor) -> Tensor
            def forward(self, src: Tensor, tgt: Tensor) -> Tensor
            def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor
        """
        ...

    def attach(self, name: str, component: Any) -> None:
        """
        Attach a component to this router.

        If component is nn.Module, it becomes a submodule with automatic
        parameter registration. Validates against strict_dtype and
        strict_device if set.

        If component is not nn.Module, it is stored in auxiliary dict
        without parameter tracking.

        Args:
            name: Name to register component under.
            component: Any object. nn.Module recommended.

        Raises:
            ValueError: If module component violates strict_dtype or strict_device.
        """
        if isinstance(component, nn.Module):
            if self.strict_dtype is not None or self.strict_device is not None:
                self._validate_component(name, component)
            self.components[name] = component
        else:
            self.objects[name] = component

    def _validate_component(self, name: str, component: nn.Module) -> None:
        """
        Validate component against strict hardware constraints.

        Args:
            name: Component name (for error messages).
            component: Module to validate.

        Raises:
            ValueError: If component violates constraints.
        """
        for param_name, param in component.named_parameters():
            if self.strict_dtype is not None and param.dtype != self.strict_dtype:
                raise ValueError(
                    f"Component '{name}' parameter '{param_name}' has dtype "
                    f"{param.dtype}, router requires {self.strict_dtype}"
                )

            if self.strict_device is not None and param.device != self.strict_device:
                raise ValueError(
                    f"Component '{name}' parameter '{param_name}' on device "
                    f"{param.device}, router requires {self.strict_device}"
                )

    def get(self, name: str) -> Optional[Any]:
        """
        Retrieve a component by name.

        Checks module components first, then non-module objects.

        Args:
            name: Component name.

        Returns:
            The component, or None if not found.
        """
        if name in self.components:
            return self.components[name]
        return self.objects.get(name)

    def has(self, name: str) -> bool:
        """
        Check if component exists.

        Args:
            name: Component name.

        Returns:
            True if component exists in either storage.
        """
        return name in self.components or name in self.objects

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def cache_set(self, key: str, value: Any) -> None:
        """
        Store a value in ephemeral cache.

        Use for intermediate tensors that should be cleared after forward pass.
        Unlike objects[], cache is explicitly managed and can be bulk-cleared.

        Args:
            key: Cache key.
            value: Value to cache (typically a Tensor).
        """
        self._cache[key] = value

    def cache_get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from ephemeral cache.

        Args:
            key: Cache key.
            default: Value to return if key not found.

        Returns:
            Cached value or default.
        """
        return self._cache.get(key, default)

    def cache_clear(self) -> None:
        """
        Clear all ephemeral cache entries.

        Call this at the end of forward() to prevent memory leaks.
        """
        self._cache.clear()

    def cache_clear_recursive(self) -> None:
        """
        Clear cache on this router and all nested routers.

        Traverses component tree to clear all _cache dicts.
        """
        self._cache.clear()
        for component in self.components.values():
            if isinstance(component, BaseRouter):
                component.cache_clear_recursive()
            elif hasattr(component, '_cache'):
                component._cache.clear()

    def cache_keys(self) -> list:
        """Return list of current cache keys."""
        return list(self._cache.keys())

    def cache_size_bytes(self) -> int:
        """
        Estimate total bytes held in cache.

        Returns:
            Total bytes for tensor values, 0 for non-tensors.
        """
        total = 0
        for v in self._cache.values():
            if isinstance(v, Tensor):
                total += v.numel() * v.element_size()
        return total

    def cache_to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> 'BaseRouter':
        """
        Move all cached tensors to device/dtype.

        This is EXPLICIT - not called by .to() or network_to().
        Use when you need to preserve cache across device changes.

        For most use cases, prefer cache_clear() before device changes
        to avoid stale tensor references.

        Args:
            device: Target device.
            dtype: Target dtype.
            non_blocking: If True, use async transfer.

        Returns:
            Self for chaining.
        """
        device = torch.device(device) if device else None

        for key in list(self._cache.keys()):
            val = self._cache[key]
            if isinstance(val, Tensor):
                if device is not None and dtype is not None:
                    self._cache[key] = val.to(device=device, dtype=dtype, non_blocking=non_blocking)
                elif device is not None:
                    self._cache[key] = val.to(device=device, non_blocking=non_blocking)
                elif dtype is not None:
                    self._cache[key] = val.to(dtype=dtype, non_blocking=non_blocking)

        return self

    def cache_to_recursive(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> 'BaseRouter':
        """
        Move cache tensors on this router and all nested routers.

        Args:
            device: Target device.
            dtype: Target dtype.
            non_blocking: If True, use async transfer.

        Returns:
            Self for chaining.
        """
        self.cache_to(device=device, dtype=dtype, non_blocking=non_blocking)

        for component in self.components.values():
            if isinstance(component, BaseRouter):
                component.cache_to_recursive(device=device, dtype=dtype, non_blocking=non_blocking)
            elif hasattr(component, 'cache_to'):
                component.cache_to(device=device, dtype=dtype, non_blocking=non_blocking)

        return self

    def detach(self, name: str) -> Optional[Any]:
        """
        Remove and return a component.

        Args:
            name: Component name.

        Returns:
            The removed component, or None if not found.
        """
        if name in self.components:
            component = self.components[name]
            del self.components[name]
            return component
        return self.objects.pop(name, None)

    def reset(self) -> None:
        """
        Clear transient state.

        Clears ephemeral cache and calls reset() on nested routers.
        Override to clear additional state that shouldn't persist between
        forward passes (buffers, message queues, etc).
        """
        self._cache.clear()
        for component in self.components.values():
            if isinstance(component, BaseRouter):
                component.reset()

    def to(self, *args, **kwargs) -> 'BaseRouter':
        """
        Standard PyTorch device/dtype movement.

        This bypasses router hardware safeguards. Emits a warning
        on first use if strict=True. Use network_to() for controlled
        movement across router networks.

        Returns:
            Self for chaining.
        """
        if self.strict and not self._to_warned:
            warnings.warn(
                f"Router '{self.name}': Using torch.to() bypasses router "
                "hardware safeguards. Use network_to() for full cohesion, "
                "or initialize with strict=False to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
            self._to_warned = True

        return super().to(*args, **kwargs)

    def network_to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        strict: bool = False,
        silent: bool = True,
        clear_cache: bool = True,
    ) -> 'BaseRouter':
        """
        Move router and all nested routers to device/dtype.

        Traverses the component tree and moves all modules. Handles
        incompatible components based on strict and silent flags.
        Non-module objects are unaffected.

        Cache tensors are NOT automatically moved (they're not registered
        with PyTorch). By default, cache is cleared to prevent device
        mismatch errors. Set clear_cache=False and use cache_to_recursive()
        if you need to preserve cache contents.

        Args:
            device: Target device.
            dtype: Target dtype.
            strict: If True, raise on incompatible components.
            silent: If True and not strict, skip incompatible quietly.
            clear_cache: If True (default), clear cache before moving.
                        Set False if using cache_to_recursive() manually.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If strict=True and component is incompatible.
        """
        # Clear cache first to prevent stale device references
        if clear_cache:
            self.cache_clear_recursive()

        device = torch.device(device) if device else None

        self._network_to_recursive(self, device, dtype, strict, silent)

        # Update strict constraints if move succeeded
        if device is not None:
            self.strict_device = device
        if dtype is not None:
            self.strict_dtype = dtype

        return self

    def _network_to_recursive(
        self,
        module: nn.Module,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        strict: bool,
        silent: bool,
    ) -> None:
        """
        Recursively move modules in network.

        Args:
            module: Current module to process.
            device: Target device.
            dtype: Target dtype.
            strict: Raise on incompatible.
            silent: Suppress warnings if not strict.
        """
        for name, child in module.named_children():
            if isinstance(child, BaseRouter):
                child.network_to(device=device, dtype=dtype, strict=strict, silent=silent)
            else:
                try:
                    if device is not None and dtype is not None:
                        child.to(device=device, dtype=dtype)
                    elif device is not None:
                        child.to(device=device)
                    elif dtype is not None:
                        child.to(dtype=dtype)
                except Exception as e:
                    if strict:
                        raise RuntimeError(
                            f"Cannot move component '{name}' to "
                            f"device={device}, dtype={dtype}: {e}"
                        ) from e
                    elif not silent:
                        warnings.warn(
                            f"Router '{self.name}': Skipping component '{name}' - "
                            f"incompatible with device={device}, dtype={dtype}: {e}",
                            UserWarning,
                        )


    @property
    def device(self) -> Optional[torch.device]:
        """Dominant device of first parameter, or None if empty."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return None

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Dominant dtype of first parameter, or None if empty."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return None

    def __contains__(self, name: str) -> bool:
        """Enable 'name in router' syntax."""
        return name in self.components or name in self.objects

    def __getitem__(self, name: str) -> Any:
        """Enable router['name'] syntax."""
        if name in self.components:
            return self.components[name]
        if name in self.objects:
            return self.objects[name]
        raise KeyError(f"Component '{name}' not found")

    def cache_debug(self, prefix: str = '') -> dict:
        """
        Debug cache across entire router tree.

        Returns dict with cache info for this router and all nested routers.
        Useful for finding memory leaks from uncached tensors.

        Args:
            prefix: Path prefix for nested routers.

        Returns:
            Dict mapping router paths to cache info.
        """
        path = f"{prefix}/{self.name}" if prefix else self.name
        result = {}

        # This router's cache
        if self._cache:
            cache_info = {}
            for k, v in self._cache.items():
                if isinstance(v, Tensor):
                    cache_info[k] = {
                        'shape': list(v.shape),
                        'dtype': str(v.dtype),
                        'device': str(v.device),
                        'bytes': v.numel() * v.element_size(),
                    }
                else:
                    cache_info[k] = {'type': type(v).__name__}
            result[path] = cache_info

        # Recurse into components
        for comp_name, comp in self.components.items():
            if isinstance(comp, BaseRouter):
                result.update(comp.cache_debug(prefix=path))
            elif hasattr(comp, '_cache') and comp._cache:
                comp_path = f"{path}/{comp_name}"
                cache_info = {}
                for k, v in comp._cache.items():
                    if isinstance(v, Tensor):
                        cache_info[k] = {
                            'shape': list(v.shape),
                            'bytes': v.numel() * v.element_size(),
                        }
                    else:
                        cache_info[k] = {'type': type(v).__name__}
                result[comp_path] = cache_info

        return result

    def __repr__(self) -> str:
        module_keys = list(self.components.keys())
        object_keys = list(self.objects.keys())
        return f"{self.__class__.__name__}(name='{self.name}', modules={module_keys}, objects={object_keys})"


# =============================================================================
# TEST ROUTERS
# =============================================================================

class MLPRouter(BaseRouter):
    """Simple MLP router."""

    def __init__(self, name: str, dims: list):
        super().__init__(name, strict=False)
        self.attach('dims', dims)

        for i in range(len(dims) - 1):
            self.attach(f'layer_{i}', nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.attach(f'norm_{i}', nn.LayerNorm(dims[i + 1]))

    def forward(self, x: Tensor) -> Tensor:
        dims = self['dims']
        for i in range(len(dims) - 1):
            x = self[f'layer_{i}'](x)
            if i < len(dims) - 2:
                x = self[f'norm_{i}'](x)
                x = torch.relu(x)
        return x


class AttentionRouter(BaseRouter):
    """Self-attention router."""

    def __init__(self, name: str, dim: int, heads: int = 8):
        super().__init__(name, strict=False)
        self.attach('dim', dim)
        self.attach('norm', nn.LayerNorm(dim))
        self.attach('attn', nn.MultiheadAttention(dim, heads, batch_first=True))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self['norm'](x)
        x, _ = self['attn'](x, x, x)
        return x + residual


class NestedRouter(BaseRouter):
    """Router containing other routers."""

    def __init__(self, name: str, dim: int):
        super().__init__(name, strict=False)
        self.attach('pre', MLPRouter(f'{name}_pre', [dim, dim * 2, dim]))
        self.attach('attn', AttentionRouter(f'{name}_attn', dim))
        self.attach('post', MLPRouter(f'{name}_post', [dim, dim * 2, dim]))
        self.attach('metadata', {'depth': 3, 'type': 'nested'})

    def forward(self, x: Tensor) -> Tensor:
        x = self['pre'](x)
        x = self['attn'](x)
        x = self['post'](x)
        return x


class DeepRouter(BaseRouter):
    """Stack of nested routers."""

    def __init__(self, name: str, dim: int, depth: int):
        super().__init__(name, strict=False)
        self.attach('depth', depth)

        for i in range(depth):
            self.attach(f'block_{i}', NestedRouter(f'{name}_block_{i}', dim))

        self.attach('final', nn.LayerNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        depth = self['depth']
        for i in range(depth):
            x = self[f'block_{i}'](x)
        return self['final'](x)


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == '__main__':
    import time

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    def test_section(title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------------------------------------------
    test_section("BASIC ROUTER")
    # -------------------------------------------------------------------------

    mlp = MLPRouter('mlp', [512, 1024, 512, 256])
    print(f"Router: {mlp}")
    print(f"Params: {count_params(mlp):,}")
    print(f"Has 'layer_0': {mlp.has('layer_0')}")
    print(f"Has 'dims': {mlp.has('dims')}")
    print(f"'layer_0' in mlp: {'layer_0' in mlp}")
    print(f"Dims config: {mlp['dims']}")

    x = torch.randn(4, 512)
    y = mlp(x)
    print(f"Forward: {x.shape} -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("NESTED ROUTER")
    # -------------------------------------------------------------------------

    nested = NestedRouter('nested', 256)
    print(f"Router: {nested}")
    print(f"Params: {count_params(nested):,}")
    print(f"Metadata: {nested['metadata']}")

    x = torch.randn(4, 16, 256)
    y = nested(x)
    print(f"Forward: {x.shape} -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("DEEP ROUTER")
    # -------------------------------------------------------------------------

    deep = DeepRouter('deep', 512, depth=6)
    print(f"Router: {deep}")
    print(f"Params: {count_params(deep):,}")
    print(f"Depth config: {deep['depth']}")

    x = torch.randn(4, 32, 512)
    y = deep(x)
    print(f"Forward: {x.shape} -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("DEVICE MOVEMENT")
    # -------------------------------------------------------------------------

    print(f"Target device: {device}")
    deep.network_to(device=device)
    print(f"Router device: {deep.device}")
    print(f"Router dtype: {deep.dtype}")

    x = torch.randn(4, 32, 512, device=device)
    y = deep(x)
    print(f"Forward on {device}: {x.shape} -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("COMPILATION")
    # -------------------------------------------------------------------------

    print("Compiling...")
    t0 = time.perf_counter()
    compiled = torch.compile(deep)

    # Warmup
    with torch.no_grad():
        _ = compiled(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    compile_time = time.perf_counter() - t0
    print(f"Compile time: {compile_time:.2f}s")

    # Benchmark
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            y1 = deep(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        eager_time = (time.perf_counter() - t0) / 20

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            y2 = compiled(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        compiled_time = (time.perf_counter() - t0) / 20

    print(f"Eager: {eager_time*1000:.2f}ms")
    print(f"Compiled: {compiled_time*1000:.2f}ms")
    print(f"Speedup: {eager_time/compiled_time:.2f}x")

    # -------------------------------------------------------------------------
    test_section("DETACH / REATTACH")
    # -------------------------------------------------------------------------

    mlp2 = MLPRouter('mlp2', [128, 256, 128])
    print(f"Before: {mlp2}")

    layer = mlp2.detach('layer_0')
    print(f"Detached layer_0: {layer}")
    print(f"After detach: {mlp2}")

    mlp2.attach('layer_0', nn.Linear(128, 256))
    print(f"After reattach: {mlp2}")

    # -------------------------------------------------------------------------
    test_section("UUID UNIQUENESS")
    # -------------------------------------------------------------------------

    r1 = MLPRouter('same_name', [64, 64])
    r2 = MLPRouter('same_name', [64, 64])
    print(f"Router 1: name='{r1.name}', uuid='{r1.uuid[:8]}...'")
    print(f"Router 2: name='{r2.name}', uuid='{r2.uuid[:8]}...'")
    print(f"Same name: {r1.name == r2.name}")
    print(f"Same uuid: {r1.uuid == r2.uuid}")

    # -------------------------------------------------------------------------
    test_section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nBaseRouter is ready.")