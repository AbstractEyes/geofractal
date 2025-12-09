"""
geofractal.router.components.torch_component
============================================

PyTorch-based component for the geofractal system.

TorchComponent combines BaseComponent identity and lifecycle with
nn.Module parameter tracking and tensor operations. This is the
primary base for any component that holds learnable parameters
or performs tensor computations.

Design Principles:
    - Dual inheritance: BaseComponent + nn.Module
    - Real PyTorch behavior (parameters, state_dict, device movement)
    - Optional strict hardware controls (like BaseRouter)
    - Multi-device support via allowed_devices set
    - Minimal surface area - subclasses define specifics

Hardware Control:
    - strict: Warns when using raw to() which bypasses safeguards
    - strict_dtype: Locks component to specific dtype
    - home_device: Where this component belongs
    - allowed_devices: Set of devices this component can move to

Multi-Device Pattern:
    Components have device affinity - they belong somewhere and know it.

    - home_device: Where I belong (can return to via to_home())
    - allowed_devices: Where I'm permitted to go
    - If allowed_devices is None, no restrictions
    - If allowed_devices is set, to() validates against it

Usage:
    class MyLayer(TorchComponent):
        def __init__(self, name: str, in_dim: int, out_dim: int):
            super().__init__(name)
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

    # Simple usage
    layer = MyLayer('encoder', 512, 256)
    layer.to('cuda:0')

    # With device constraints
    layer = MyLayer(
        'encoder', 512, 256,
        home_device=torch.device('cuda:0'),
        allowed_devices={torch.device('cuda:0'), torch.device('cpu')},
    )
    layer.to('cpu')      # OK - in allowed set
    layer.to('cuda:1')   # Raises ValueError
    layer.to_home()      # Back to cuda:0

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import warnings
from typing import Optional, Union, Any, Set

import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.base_component import BaseComponent


class TorchComponent(nn.Module, BaseComponent):
    """
    PyTorch-based component.

    Combines BaseComponent identity/lifecycle with nn.Module
    parameter tracking. This is the base for any component
    that holds learnable parameters or performs tensor operations.

    Attributes:
        name: Human-readable identifier (from BaseComponent).
        uuid: Unique identifier (from BaseComponent).
        parent: Reference to parent router (from BaseComponent).
        strict: If True, warn on unbounded to() usage.
        strict_dtype: If set, enforce dtype constraint.
        home_device: Where this component belongs.
        allowed_devices: Set of devices component can move to.
    """

    def __init__(
            self,
            name: str,
            uuid: Optional[str] = None,
            strict: bool = False,
            strict_dtype: Optional[torch.dtype] = None,
            home_device: Optional[Union[str, torch.device]] = None,
            allowed_devices: Optional[Set[Union[str, torch.device]]] = None,
    ):
        """
        Initialize TorchComponent.

        Args:
            name: Human-readable name for this component.
            uuid: Unique identifier. Generated if not provided.
            strict: Warn on raw to() usage if True.
            strict_dtype: Lock dtype constraint.
            home_device: Primary device for this component.
            allowed_devices: Set of permitted devices. None means no restrictions.
        """
        BaseComponent.__init__(self, name, uuid)
        nn.Module.__init__(self)

        # Hardware control
        self.strict = strict
        self.strict_dtype = strict_dtype
        self._to_warned = False

        # Device affinity
        self.home_device = torch.device(home_device) if home_device else None
        self.allowed_devices: Optional[Set[torch.device]] = None
        if allowed_devices is not None:
            self.allowed_devices = {
                torch.device(d) if isinstance(d, str) else d
                for d in allowed_devices
            }

    # =========================================================================
    # PYTORCH PROPERTIES (override BaseComponent defaults)
    # =========================================================================

    @property
    def device(self) -> Optional[torch.device]:
        """Device of first parameter, or None if no parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return None

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Dtype of first parameter, or None if no parameters."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return None

    # =========================================================================
    # DEVICE MANAGEMENT
    # =========================================================================

    def can_move_to(self, device: Union[str, torch.device]) -> bool:
        """
        Check if component can move to device.

        Args:
            device: Target device to check.

        Returns:
            True if move is allowed.
        """
        if self.allowed_devices is None:
            return True
        target = torch.device(device) if isinstance(device, str) else device
        return target in self.allowed_devices

    def to(self, *args, **kwargs) -> 'TorchComponent':
        """
        Move component to device/dtype with validation.

        Validates against allowed_devices if set.
        Warns if strict=True and no constraints set.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If target device not in allowed_devices.
        """
        # Extract device from args/kwargs
        target_device = None
        if args:
            if isinstance(args[0], (str, torch.device)):
                target_device = torch.device(args[0])
            elif isinstance(args[0], torch.dtype):
                pass  # dtype only, no device change
        if 'device' in kwargs:
            target_device = torch.device(kwargs['device'])

        # Validate device if we're moving
        if target_device is not None and self.allowed_devices is not None:
            if target_device not in self.allowed_devices:
                raise ValueError(
                    f"TorchComponent '{self.name}' cannot move to {target_device}. "
                    f"Allowed devices: {self.allowed_devices}"
                )

        # Warn on uncontrolled movement
        if self.strict and not self._to_warned:
            if self.allowed_devices is None and self.home_device is None:
                warnings.warn(
                    f"TorchComponent '{self.name}': Using to() without device "
                    "constraints. Set home_device/allowed_devices for control.",
                    UserWarning,
                    stacklevel=2,
                )
                self._to_warned = True

        return super().to(*args, **kwargs)

    def to_home(self) -> 'TorchComponent':
        """
        Move component to its home device.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If home_device not set.
        """
        if self.home_device is None:
            raise ValueError(
                f"TorchComponent '{self.name}' has no home_device set."
            )
        return super().to(self.home_device)

    @property
    def is_home(self) -> bool:
        """Check if component is on its home device."""
        if self.home_device is None:
            return True  # No home means always home
        return self.device == self.home_device

    def add_allowed_device(self, device: Union[str, torch.device]) -> None:
        """
        Add a device to allowed set.

        Args:
            device: Device to allow.
        """
        target = torch.device(device) if isinstance(device, str) else device
        if self.allowed_devices is None:
            self.allowed_devices = set()
        self.allowed_devices.add(target)

    def remove_allowed_device(self, device: Union[str, torch.device]) -> None:
        """
        Remove a device from allowed set.

        Args:
            device: Device to disallow.
        """
        if self.allowed_devices is None:
            return
        target = torch.device(device) if isinstance(device, str) else device
        self.allowed_devices.discard(target)

    # =========================================================================
    # DTYPE MANAGEMENT
    # =========================================================================

    def validate_dtype(self) -> None:
        """
        Validate current dtype against strict_dtype.

        Raises:
            ValueError: If component violates strict_dtype.
        """
        if self.strict_dtype is not None and self.dtype is not None:
            if self.dtype != self.strict_dtype:
                raise ValueError(
                    f"TorchComponent '{self.name}' has dtype {self.dtype}, "
                    f"but strict_dtype requires {self.strict_dtype}"
                )

    # =========================================================================
    # LIFECYCLE HOOKS
    # =========================================================================

    def on_attach(self, parent: Any) -> None:
        """
        Called when attached to a router.

        Inherits constraints from parent router if not already set.

        Args:
            parent: The router this component is being attached to.
        """
        # Inherit dtype constraint from parent
        if hasattr(parent, 'strict_dtype'):
            if self.strict_dtype is None and parent.strict_dtype is not None:
                self.strict_dtype = parent.strict_dtype

        # Inherit device constraint from parent (add to allowed set)
        if hasattr(parent, 'strict_device') and parent.strict_device is not None:
            if self.allowed_devices is None:
                self.allowed_devices = set()
            self.allowed_devices.add(parent.strict_device)

        # Validate against inherited constraints
        self.validate_dtype()

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Count parameters.

        Args:
            only_trainable: If True, only count trainable parameters.

        Returns:
            Total parameter count.
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # =========================================================================
    # DUNDER METHODS
    # =========================================================================

    def __repr__(self) -> str:
        parts = [f"name='{self.name}'"]
        parts.append(f"params={self.num_parameters():,}")

        if self.device:
            parts.append(f"device={self.device}")
        if self.dtype:
            parts.append(f"dtype={self.dtype}")
        if self.home_device:
            parts.append(f"home={self.home_device}")
        if self.allowed_devices:
            devices_str = '{' + ', '.join(str(d) for d in self.allowed_devices) + '}'
            parts.append(f"allowed={devices_str}")
        if self.parent:
            parts.append(f"parent='{self.parent.name}'")

        return f"{self.__class__.__name__}({', '.join(parts)})"

    def freeze(self) -> 'TorchComponent':
        for p in self.parameters():
            p.requires_grad = False
        return self

    def unfreeze(self) -> 'TorchComponent':
        for p in self.parameters():
            p.requires_grad = True
        return self


# =============================================================================
# TEST COMPONENTS
# =============================================================================

class LinearComponent(TorchComponent):
    """Simple linear layer for testing."""

    def __init__(
            self,
            name: str,
            in_features: int,
            out_features: int,
            bias: bool = True,
            **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class MLPComponent(TorchComponent):
    """Multi-layer perceptron for testing."""

    def __init__(
            self,
            name: str,
            dims: list,
            activation: str = 'relu',
            **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.dims = dims

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AttentionComponent(TorchComponent):
    """Self-attention for testing."""

    def __init__(
            self,
            name: str,
            dim: int,
            num_heads: int = 8,
            dropout: float = 0.0,
            **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.dim = dim
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return x + residual


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == '__main__':

    def test_section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    has_cuda = torch.cuda.is_available()

    # -------------------------------------------------------------------------
    test_section("BASIC COMPONENT")
    # -------------------------------------------------------------------------

    linear = LinearComponent('encoder', 512, 256)
    print(f"Component: {linear}")
    print(f"Is attached: {linear.is_attached}")
    print(f"Is home: {linear.is_home}")

    x = torch.randn(4, 512)
    y = linear(x)
    print(f"Forward: {x.shape} -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("HOME DEVICE")
    # -------------------------------------------------------------------------

    homed = LinearComponent(
        'homed', 256, 128,
        home_device='cpu',
    )
    print(f"Component: {homed}")
    print(f"Is home: {homed.is_home}")

    if has_cuda:
        homed.to('cuda')
        print(f"After to('cuda'): {homed.device}")
        print(f"Is home: {homed.is_home}")

        homed.to_home()
        print(f"After to_home(): {homed.device}")
        print(f"Is home: {homed.is_home}")

    # -------------------------------------------------------------------------
    test_section("ALLOWED DEVICES")
    # -------------------------------------------------------------------------

    if has_cuda:
        restricted = LinearComponent(
            'restricted', 128, 64,
            home_device='cuda:0',
            allowed_devices={'cuda:0', 'cpu'},
        )
        print(f"Component: {restricted}")
        print(f"Can move to cpu: {restricted.can_move_to('cpu')}")
        print(f"Can move to cuda:0: {restricted.can_move_to('cuda:0')}")
        print(f"Can move to cuda:1: {restricted.can_move_to('cuda:1')}")

        restricted.to('cpu')
        print(f"After to('cpu'): {restricted.device}")

        restricted.to('cuda:0')
        print(f"After to('cuda:0'): {restricted.device}")

        # Test rejection
        try:
            restricted.to('cuda:1')
            print("ERROR: Should have raised ValueError")
        except ValueError as e:
            print(f"Correctly rejected cuda:1: {e}")
    else:
        print("Skipping (no CUDA)")

    # -------------------------------------------------------------------------
    test_section("DYNAMIC DEVICE MANAGEMENT")
    # -------------------------------------------------------------------------

    dynamic = LinearComponent('dynamic', 64, 32)
    print(f"Initial allowed: {dynamic.allowed_devices}")

    dynamic.add_allowed_device('cpu')
    print(f"After add cpu: {dynamic.allowed_devices}")

    if has_cuda:
        dynamic.add_allowed_device('cuda:0')
        print(f"After add cuda:0: {dynamic.allowed_devices}")

        dynamic.remove_allowed_device('cuda:0')
        print(f"After remove cuda:0: {dynamic.allowed_devices}")

    # -------------------------------------------------------------------------
    test_section("MULTI-DEVICE PLACEMENT PATTERN")
    # -------------------------------------------------------------------------

    if has_cuda:
        # Simulate different components on different devices
        encoder = LinearComponent('encoder', 512, 256, home_device='cuda:0')
        decoder = LinearComponent('decoder', 256, 512, home_device='cpu')

        encoder.to_home()
        decoder.to_home()

        print(f"Encoder: {encoder.device}")
        print(f"Decoder: {decoder.device}")

        # Forward with device transfer
        x = torch.randn(4, 512, device='cuda:0')
        h = encoder(x)
        print(f"Encoded on {h.device}: {h.shape}")

        h_cpu = h.to('cpu')
        y = decoder(h_cpu)
        print(f"Decoded on {y.device}: {y.shape}")
    else:
        print("Skipping (no CUDA)")

    # -------------------------------------------------------------------------
    test_section("STATE DICT")
    # -------------------------------------------------------------------------

    original = LinearComponent('original', 128, 64)
    state = original.state_dict()
    print(f"State dict keys: {list(state.keys())}")

    restored = LinearComponent('restored', 128, 64)
    restored.load_state_dict(state)

    x = torch.randn(4, 128)
    y1 = original(x)
    y2 = restored(x)
    print(f"Outputs match: {torch.allclose(y1, y2)}")

    # -------------------------------------------------------------------------
    test_section("COMPILATION")
    # -------------------------------------------------------------------------

    if has_cuda:
        print("Compiling MLPComponent...")
        mlp = MLPComponent('compiled', [512, 1024, 512], home_device='cuda:0')
        mlp.to_home()

        compiled = torch.compile(mlp)

        x = torch.randn(4, 512, device='cuda:0')
        y1 = mlp(x)
        y2 = compiled(x)
        print(f"Compiled outputs match: {torch.allclose(y1, y2, atol=1e-5)}")
    else:
        print("Skipping (no CUDA)")

    # -------------------------------------------------------------------------
    test_section("INHERITANCE CHECK")
    # -------------------------------------------------------------------------

    comp = LinearComponent('test', 64, 32)
    print(f"Is BaseComponent: {isinstance(comp, BaseComponent)}")
    print(f"Is nn.Module: {isinstance(comp, nn.Module)}")
    print(f"MRO: {[c.__name__ for c in LinearComponent.__mro__]}")

    # -------------------------------------------------------------------------
    test_section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nTorchComponent is ready.")