"""
geofractal.router.components.projection_component
=================================================

Learned projections for the geofractal system.

ProjectionComponent is the base for any learned transform.
Defines input/output contract and project() method.

Chain Pattern:
    DataComponent -> ProjectionComponent -> LearningModule
                  -> ProjectionComponent -> LearningModule
                  -> ...

Design Principles:
    - Inherits TorchComponent (has parameters)
    - Clear in/out contract
    - Override project() for custom behavior
    - Chainable by design

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.components.torch_component import TorchComponent


class ProjectionComponent(TorchComponent):
    """
    Base for learned projections.

    Defines in_features -> out_features contract.
    Override project() for custom transforms.

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
    """

    def __init__(
            self,
            name: str,
            in_features: int,
            out_features: int,
            uuid: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)
        self.in_features = in_features
        self.out_features = out_features

    def project(self, x: Tensor) -> Tensor:
        """
        Apply projection.

        Override in subclass.

        Args:
            x: Input tensor [..., in_features].

        Returns:
            Projected tensor [..., out_features].
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass calls project."""
        return self.project(x)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"{self.in_features} -> {self.out_features}, "
            f"params={self.num_parameters():,})"
        )


# =============================================================================
# LINEAR
# =============================================================================

class LinearProjection(ProjectionComponent):
    """
    Simple linear projection.

    x @ W.T + b
    """

    def __init__(
            self,
            name: str,
            in_features: int,
            out_features: int,
            bias: bool = True,
            **kwargs,
    ):
        super().__init__(name, in_features, out_features, **kwargs)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def project(self, x: Tensor) -> Tensor:
        return self.linear(x)


# =============================================================================
# SLOT EXPANSION
# =============================================================================

class SlotProjection(ProjectionComponent):
    """
    Slot expansion with learnable embeddings.

    [B, D] -> [B, num_slots, D]

    Each slot is a learned viewpoint on the input.
    The mechanism that enabled 0.1% -> 84.68% emergence.
    """

    def __init__(
            self,
            name: str,
            features: int,
            num_slots: int,
            **kwargs,
    ):
        super().__init__(name, features, features, **kwargs)
        self.num_slots = num_slots
        self.slot_embed = nn.Parameter(torch.randn(num_slots, features) * 0.02)

    def project(self, x: Tensor) -> Tensor:
        """
        Expand input to slots with learned embeddings.

        Args:
            x: [B, D] input features.

        Returns:
            [B, num_slots, D] slot-expanded features.
        """
        # [B, D] -> [B, 1, D] -> [B, num_slots, D]
        expanded = x.unsqueeze(1).expand(-1, self.num_slots, -1)
        return expanded + self.slot_embed

    def collapse(self, x: Tensor, mode: str = 'mean') -> Tensor:
        """
        Collapse slots back to single vector.

        Args:
            x: [B, num_slots, D] slot features.
            mode: 'mean', 'sum', 'max', 'first', 'last'.

        Returns:
            [B, D] collapsed features.
        """
        if mode == 'mean':
            return x.mean(dim=1)
        elif mode == 'sum':
            return x.sum(dim=1)
        elif mode == 'max':
            return x.max(dim=1).values
        elif mode == 'first':
            return x[:, 0]
        elif mode == 'last':
            return x[:, -1]
        else:
            raise ValueError(f"Unknown collapse mode: {mode}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"features={self.in_features}, slots={self.num_slots}, "
            f"params={self.num_parameters():,})"
        )


# =============================================================================
# MULTI-HEAD
# =============================================================================

class MultiHeadProjection(ProjectionComponent):
    """
    Split into multiple heads.

    [B, L, D] -> [B, L, num_heads, head_dim]

    For attention mechanisms.
    """

    def __init__(
            self,
            name: str,
            features: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            bias: bool = True,
            **kwargs,
    ):
        head_dim = head_dim or features // num_heads
        out_features = num_heads * head_dim
        super().__init__(name, features, out_features, **kwargs)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj = nn.Linear(features, out_features, bias=bias)

    def project(self, x: Tensor) -> Tensor:
        """
        Project and reshape to heads.

        Args:
            x: [B, L, D] input.

        Returns:
            [B, L, num_heads, head_dim] multi-head output.
        """
        B, L, _ = x.shape
        x = self.proj(x)
        return x.view(B, L, self.num_heads, self.head_dim)

    def merge(self, x: Tensor) -> Tensor:
        """
        Merge heads back.

        Args:
            x: [B, L, num_heads, head_dim] multi-head input.

        Returns:
            [B, L, num_heads * head_dim] merged output.
        """
        B, L, H, D = x.shape
        return x.view(B, L, H * D)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"features={self.in_features}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, params={self.num_parameters():,})"
        )


# =============================================================================
# BOTTLENECK
# =============================================================================

class BottleneckProjection(ProjectionComponent):
    """
    Compress then expand.

    in -> narrow -> out

    Forces information bottleneck.
    """

    def __init__(
            self,
            name: str,
            in_features: int,
            out_features: int,
            bottleneck_features: int,
            bias: bool = True,
            activation: str = 'gelu',
            **kwargs,
    ):
        super().__init__(name, in_features, out_features, **kwargs)

        self.bottleneck_features = bottleneck_features
        self.down = nn.Linear(in_features, bottleneck_features, bias=bias)
        self.up = nn.Linear(bottleneck_features, out_features, bias=bias)

        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()

    def project(self, x: Tensor) -> Tensor:
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"{self.in_features} -> {self.bottleneck_features} -> {self.out_features}, "
            f"params={self.num_parameters():,})"
        )


# =============================================================================
# SCALE (multiply by learnable scalar)
# =============================================================================

class ScaleProjection(ProjectionComponent):
    """
    Learnable scale factor.

    x * scale

    Simple but useful for gating/modulation.
    """

    def __init__(
            self,
            name: str,
            features: int,
            init_scale: float = 1.0,
            per_feature: bool = False,
            **kwargs,
    ):
        super().__init__(name, features, features, **kwargs)

        self.per_feature = per_feature
        if per_feature:
            self.scale = nn.Parameter(torch.full((features,), init_scale))
        else:
            self.scale = nn.Parameter(torch.tensor(init_scale))

    def project(self, x: Tensor) -> Tensor:
        return x * self.scale


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == '__main__':
    def test_section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------------------------------------------
    test_section("LINEAR PROJECTION")
    # -------------------------------------------------------------------------

    proj = LinearProjection('linear', 512, 256)
    print(f"Component: {proj}")

    x = torch.randn(4, 512)
    y = proj(x)
    print(f"Project: {x.shape} -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("SLOT PROJECTION")
    # -------------------------------------------------------------------------

    slot = SlotProjection('slots', features=512, num_slots=16)
    print(f"Component: {slot}")

    x = torch.randn(4, 512)
    y = slot(x)
    print(f"Expand: {x.shape} -> {y.shape}")

    # Each slot is different due to learned embeddings
    print(f"Slot 0 == Slot 1: {torch.allclose(y[:, 0], y[:, 1])}")

    # Collapse back
    z = slot.collapse(y, mode='mean')
    print(f"Collapse: {y.shape} -> {z.shape}")

    # -------------------------------------------------------------------------
    test_section("MULTI-HEAD PROJECTION")
    # -------------------------------------------------------------------------

    mh = MultiHeadProjection('heads', features=512, num_heads=8)
    print(f"Component: {mh}")

    x = torch.randn(4, 32, 512)
    y = mh(x)
    print(f"Project: {x.shape} -> {y.shape}")

    z = mh.merge(y)
    print(f"Merge: {y.shape} -> {z.shape}")

    # -------------------------------------------------------------------------
    test_section("BOTTLENECK PROJECTION")
    # -------------------------------------------------------------------------

    bn = BottleneckProjection('bottleneck', 512, 512, bottleneck_features=64)
    print(f"Component: {bn}")

    x = torch.randn(4, 512)
    y = bn(x)
    print(f"Project: {x.shape} -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("SCALE PROJECTION")
    # -------------------------------------------------------------------------

    scale = ScaleProjection('scale', 256, init_scale=0.5, per_feature=True)
    print(f"Component: {scale}")

    x = torch.randn(4, 256)
    y = scale(x)
    print(f"Scale: {x.shape} -> {y.shape}")
    print(f"Scale factor sample: {scale.scale[:5].tolist()}")

    # -------------------------------------------------------------------------
    test_section("CHAINING")
    # -------------------------------------------------------------------------

    # Data -> Slot -> Bottleneck -> Scale
    slot = SlotProjection('expand', features=256, num_slots=8)
    bn = BottleneckProjection('compress', 256, 256, bottleneck_features=32)
    scale = ScaleProjection('gate', 256, init_scale=1.0)

    x = torch.randn(4, 256)
    print(f"Input: {x.shape}")

    x = slot(x)
    print(f"After slot: {x.shape}")

    x = bn(x)
    print(f"After bottleneck: {x.shape}")

    x = scale(x)
    print(f"After scale: {x.shape}")

    # Collapse
    x = slot.collapse(x)
    print(f"After collapse: {x.shape}")

    # -------------------------------------------------------------------------
    test_section("DEVICE MOVEMENT")
    # -------------------------------------------------------------------------

    proj = LinearProjection('gpu_proj', 512, 256, home_device=device)
    proj.to_home()
    print(f"Component: {proj}")
    print(f"Device: {proj.device}")

    x = torch.randn(4, 512, device=device)
    y = proj(x)
    print(f"Project on {device}: {y.shape}")

    # -------------------------------------------------------------------------
    test_section("COMPILATION")
    # -------------------------------------------------------------------------

    proj = LinearProjection('compiled', 512, 256).to(device)
    compiled = torch.compile(proj)

    x = torch.randn(4, 512, device=device)
    y1 = proj(x)
    y2 = compiled(x)
    print(f"Compiled match: {torch.allclose(y1, y2, atol=1e-5)}")

    # -------------------------------------------------------------------------
    test_section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nProjectionComponent is ready.")