"""
geofractal.router.components.fusion_component
=============================================

Learned fusion for the geofractal system.

FusionComponent combines multiple signals into one. This is where
emergence happens - the mechanism that enabled 0.1% -> 84.68%
collective accuracy from individually random streams.

Design Principles:
    - Inherits TorchComponent (has parameters)
    - Multiple inputs, single output
    - Override fuse() for custom strategy
    - Chainable with Projection/Data components

Chain Pattern:
    Data -> Projection -> Module_A ──┐
    Data -> Projection -> Module_B ──┼─> Fusion -> Output
    Data -> Projection -> Module_C ──┘

Fusion Strategies:
    - ConcatFusion: Concatenate then project
    - SumFusion: Weighted sum
    - GatedFusion: Learned gates per input
    - AttentionFusion: Cross-attention between inputs
    - AdaptiveFusion: Content-dependent weights (Lyra pattern)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.components.torch_component import TorchComponent


class FusionComponent(TorchComponent):
    """
    Base for learned fusion.

    Combines multiple inputs into single output.
    Override fuse() for custom strategy.

    Attributes:
        num_inputs: Expected number of inputs.
        in_features: Feature dimension per input.
        out_features: Output feature dimension.
    """

    def __init__(
            self,
            name: str,
            num_inputs: int,
            in_features: int,
            out_features: Optional[int] = None,
            uuid: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)
        self.num_inputs = num_inputs
        self.in_features = in_features
        self.out_features = out_features or in_features

    def fuse(self, *inputs: Tensor) -> Tensor:
        """
        Combine inputs.

        Override in subclass.

        Args:
            *inputs: Input tensors, each [..., in_features].

        Returns:
            Fused tensor [..., out_features].
        """
        raise NotImplementedError

    def forward(self, *inputs: Tensor) -> Tensor:
        """Forward pass calls fuse."""
        return self.fuse(*inputs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"inputs={self.num_inputs}, {self.in_features} -> {self.out_features}, "
            f"params={self.num_parameters():,})"
        )


# =============================================================================
# CONCAT FUSION
# =============================================================================

class ConcatFusion(FusionComponent):
    """
    Concatenate then project.

    [x1, x2, ...] -> cat(dim=-1) -> linear -> out
    """

    def __init__(
            self,
            name: str,
            num_inputs: int,
            in_features: int,
            out_features: Optional[int] = None,
            bias: bool = True,
            **kwargs,
    ):
        out_features = out_features or in_features
        super().__init__(name, num_inputs, in_features, out_features, **kwargs)

        self.proj = nn.Linear(num_inputs * in_features, out_features, bias=bias)

    def fuse(self, *inputs: Tensor) -> Tensor:
        x = torch.cat(inputs, dim=-1)
        return self.proj(x)


# =============================================================================
# SUM FUSION
# =============================================================================

class SumFusion(FusionComponent):
    """
    Weighted sum with learnable weights.

    w1*x1 + w2*x2 + ...

    Weights are softmax-normalized by default.
    """

    def __init__(
            self,
            name: str,
            num_inputs: int,
            in_features: int,
            normalize: bool = True,
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, in_features, **kwargs)

        self.normalize = normalize
        self.weights = nn.Parameter(torch.ones(num_inputs))

    def fuse(self, *inputs: Tensor) -> Tensor:
        weights = F.softmax(self.weights, dim=0) if self.normalize else self.weights

        # Stack inputs: [num_inputs, B, ..., D]
        stacked = torch.stack(inputs, dim=0)

        # Weight and sum: [B, ..., D]
        # Reshape weights for broadcasting
        w_shape = [self.num_inputs] + [1] * (stacked.dim() - 1)
        weighted = stacked * weights.view(*w_shape)

        return weighted.sum(dim=0)


# =============================================================================
# GATED FUSION
# =============================================================================

class GatedFusion(FusionComponent):
    """
    Learned gate per input.

    sigmoid(g1)*x1 + sigmoid(g2)*x2 + ...

    Gates computed from input content.
    """

    def __init__(
            self,
            name: str,
            num_inputs: int,
            in_features: int,
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, in_features, **kwargs)

        # Gate projection for each input
        self.gate_proj = nn.Linear(in_features, num_inputs)

    def fuse(self, *inputs: Tensor) -> Tensor:
        # Stack to [B, N, D] - compile-friendly, no movedim
        stacked = torch.stack(inputs, dim=1)  # [B, N, D]
        pooled = stacked.mean(dim=1)  # [B, D]

        gates = torch.sigmoid(self.gate_proj(pooled))  # [B, N]
        gates = gates.unsqueeze(-1)  # [B, N, 1]

        gated = stacked * gates  # [B, N, D]
        return gated.sum(dim=1)  # [B, D]


# =============================================================================
# ATTENTION FUSION
# =============================================================================

class AttentionFusion(FusionComponent):
    """
    Cross-attention between inputs.

    First input provides query, all inputs provide keys/values.
    """

    def __init__(
            self,
            name: str,
            num_inputs: int,
            in_features: int,
            num_heads: int = 8,
            dropout: float = 0.0,
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, in_features, **kwargs)

        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        self.q_proj = nn.Linear(in_features, in_features)
        self.k_proj = nn.Linear(in_features, in_features)
        self.v_proj = nn.Linear(in_features, in_features)
        self.out_proj = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def fuse(self, *inputs: Tensor) -> Tensor:
        # Query from first input
        q = self.q_proj(inputs[0])  # [B, ..., D]

        # Keys and values from all inputs concatenated
        kv_input = torch.cat(inputs, dim=-2)  # [B, ..., N*L, D] or [B, N*L, D]
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        # Reshape for multi-head attention
        # Assuming [B, L, D] input shape
        B = q.shape[0]
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.in_features)

        return self.out_proj(out)


# =============================================================================
# ADAPTIVE FUSION (Lyra pattern)
# =============================================================================

class AdaptiveFusion(FusionComponent):
    """
    Content-dependent fusion weights.

    Weights computed dynamically from inputs.
    The Lyra pattern - each input influences its own weight.
    """

    def __init__(
            self,
            name: str,
            num_inputs: int,
            in_features: int,
            hidden_features: Optional[int] = None,
            temperature: float = 1.0,
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, in_features, **kwargs)

        hidden = hidden_features or in_features // 4
        self.temperature = temperature

        # Per-input weight predictor
        self.weight_net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def fuse(self, *inputs: Tensor) -> Tensor:
        # Stack inputs: [N, B, ..., D]
        stacked = torch.stack(inputs, dim=0)

        # Compute weight for each input: [N, B, ..., 1]
        weights = self.weight_net(stacked)

        # Softmax over input dimension
        weights = F.softmax(weights / self.temperature, dim=0)

        # Weighted sum
        fused = (stacked * weights).sum(dim=0)

        return fused


# =============================================================================
# BILINEAR FUSION
# =============================================================================

class BilinearFusion(FusionComponent):
    """
    Bilinear interaction between two inputs.

    x1^T W x2 + b

    For exactly 2 inputs.
    """

    def __init__(
            self,
            name: str,
            in_features: int,
            out_features: Optional[int] = None,
            bias: bool = True,
            **kwargs,
    ):
        out_features = out_features or in_features
        super().__init__(name, 2, in_features, out_features, **kwargs)

        self.bilinear = nn.Bilinear(in_features, in_features, out_features, bias=bias)

    def fuse(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.bilinear(x1, x2)


# =============================================================================
# RESIDUAL FUSION
# =============================================================================

class ResidualFusion(FusionComponent):
    """
    Fuse with residual connection.

    x1 + fuse(x1, x2, ...)

    First input is residual.
    """

    def __init__(
            self,
            name: str,
            num_inputs: int,
            in_features: int,
            fusion_type: str = 'concat',
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, in_features, **kwargs)

        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            self.inner = nn.Linear(num_inputs * in_features, in_features)
        elif fusion_type == 'sum':
            self.inner = nn.Parameter(torch.ones(num_inputs))
        elif fusion_type == 'gated':
            self.inner = nn.Linear(in_features, num_inputs)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def fuse(self, *inputs: Tensor) -> Tensor:
        residual = inputs[0]

        if self.fusion_type == 'concat':
            x = torch.cat(inputs, dim=-1)
            x = self.inner(x)
        elif self.fusion_type == 'sum':
            weights = F.softmax(self.inner, dim=0)
            stacked = torch.stack(inputs, dim=0)
            w_shape = [self.num_inputs] + [1] * (stacked.dim() - 1)
            x = (stacked * weights.view(*w_shape)).sum(dim=0)
        elif self.fusion_type == 'gated':
            stacked = torch.stack(inputs, dim=1)  # [B, N, D]
            gates = torch.sigmoid(self.inner(stacked.mean(dim=1)))  # [B, N]
            gates = gates.unsqueeze(-1)  # [B, N, 1]
            x = (stacked * gates).sum(dim=1)  # [B, D]

        return residual + x


# =============================================================================
# SLOT FUSION
# =============================================================================

class SlotFusion(FusionComponent):
    """
    Fuse across slot dimension.

    Specialized for [B, num_slots, D] inputs.
    Combines information across slots.
    """

    def __init__(
            self,
            name: str,
            num_slots: int,
            in_features: int,
            mode: str = 'attention',
            num_heads: int = 8,
            **kwargs,
    ):
        super().__init__(name, 1, in_features, in_features, **kwargs)

        self.num_slots = num_slots
        self.mode = mode

        if mode == 'attention':
            self.attn = nn.MultiheadAttention(
                in_features, num_heads, batch_first=True
            )
        elif mode == 'linear':
            self.proj = nn.Linear(num_slots * in_features, in_features)
        elif mode == 'weighted':
            self.weights = nn.Parameter(torch.ones(num_slots))

    def fuse(self, x: Tensor) -> Tensor:
        """
        Fuse slots.

        Args:
            x: [B, num_slots, D] slot features.

        Returns:
            [B, D] fused features.
        """
        B, S, D = x.shape

        if self.mode == 'attention':
            # Self-attention across slots, then mean pool
            x, _ = self.attn(x, x, x)
            return x.mean(dim=1)

        elif self.mode == 'linear':
            # Flatten and project
            x = x.view(B, S * D)
            return self.proj(x)

        elif self.mode == 'weighted':
            # Learned weighted mean
            weights = F.softmax(self.weights, dim=0)
            return (x * weights.view(1, S, 1)).sum(dim=1)

        elif self.mode == 'mean':
            return x.mean(dim=1)

        elif self.mode == 'max':
            return x.max(dim=1).values

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"slots={self.num_slots}, mode='{self.mode}', "
            f"features={self.in_features}, params={self.num_parameters():,})"
        )


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
    test_section("CONCAT FUSION")
    # -------------------------------------------------------------------------

    fusion = ConcatFusion('concat', num_inputs=3, in_features=256, out_features=256)
    print(f"Component: {fusion}")

    x1 = torch.randn(4, 256)
    x2 = torch.randn(4, 256)
    x3 = torch.randn(4, 256)

    y = fusion(x1, x2, x3)
    print(f"Fuse: 3x[4, 256] -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("SUM FUSION")
    # -------------------------------------------------------------------------

    fusion = SumFusion('sum', num_inputs=3, in_features=256)
    print(f"Component: {fusion}")
    print(f"Weights: {F.softmax(fusion.weights, dim=0).tolist()}")

    y = fusion(x1, x2, x3)
    print(f"Fuse: 3x[4, 256] -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("GATED FUSION")
    # -------------------------------------------------------------------------

    fusion = GatedFusion('gated', num_inputs=3, in_features=256)
    print(f"Component: {fusion}")

    y = fusion(x1, x2, x3)
    print(f"Fuse: 3x[4, 256] -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("ADAPTIVE FUSION")
    # -------------------------------------------------------------------------

    fusion = AdaptiveFusion('adaptive', num_inputs=3, in_features=256)
    print(f"Component: {fusion}")

    y = fusion(x1, x2, x3)
    print(f"Fuse: 3x[4, 256] -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("BILINEAR FUSION")
    # -------------------------------------------------------------------------

    fusion = BilinearFusion('bilinear', in_features=256, out_features=128)
    print(f"Component: {fusion}")

    y = fusion(x1, x2)
    print(f"Fuse: 2x[4, 256] -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("RESIDUAL FUSION")
    # -------------------------------------------------------------------------

    fusion = ResidualFusion('residual', num_inputs=3, in_features=256, fusion_type='concat')
    print(f"Component: {fusion}")

    y = fusion(x1, x2, x3)
    print(f"Fuse: 3x[4, 256] -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("SLOT FUSION")
    # -------------------------------------------------------------------------

    # Simulating slot-expanded input
    x_slots = torch.randn(4, 16, 256)

    for mode in ['attention', 'linear', 'weighted', 'mean', 'max']:
        fusion = SlotFusion('slot', num_slots=16, in_features=256, mode=mode)
        y = fusion(x_slots)
        print(f"SlotFusion ({mode}): {x_slots.shape} -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("ATTENTION FUSION")
    # -------------------------------------------------------------------------

    # Sequence inputs
    x1 = torch.randn(4, 32, 256)
    x2 = torch.randn(4, 32, 256)
    x3 = torch.randn(4, 32, 256)

    fusion = AttentionFusion('attention', num_inputs=3, in_features=256, num_heads=8)
    print(f"Component: {fusion}")

    y = fusion(x1, x2, x3)
    print(f"Fuse: 3x[4, 32, 256] -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("CHAINING: PROJECT -> FUSE")
    # -------------------------------------------------------------------------

    from geofractal.router.components.projection_component import (
        SlotProjection, LinearProjection
    )

    # Three streams, slot expand, fuse
    proj_a = SlotProjection('stream_a', features=256, num_slots=16)
    proj_b = SlotProjection('stream_b', features=256, num_slots=16)
    proj_c = SlotProjection('stream_c', features=256, num_slots=16)

    fusion = AdaptiveFusion('fuse', num_inputs=3, in_features=256)
    collapse = SlotFusion('collapse', num_slots=16, in_features=256, mode='attention')

    # Input
    a = torch.randn(4, 256)
    b = torch.randn(4, 256)
    c = torch.randn(4, 256)
    print(f"Inputs: {a.shape}, {b.shape}, {c.shape}")

    # Expand
    a = proj_a(a)
    b = proj_b(b)
    c = proj_c(c)
    print(f"After slot expand: {a.shape}")

    # Fuse
    fused = fusion(a, b, c)
    print(f"After fusion: {fused.shape}")

    # Collapse
    out = collapse(fused)
    print(f"After collapse: {out.shape}")

    # -------------------------------------------------------------------------
    test_section("DEVICE + COMPILE")
    # -------------------------------------------------------------------------

    fusion = AdaptiveFusion('gpu_fusion', num_inputs=3, in_features=256).to(device)
    compiled = torch.compile(fusion)

    x1 = torch.randn(4, 256, device=device)
    x2 = torch.randn(4, 256, device=device)
    x3 = torch.randn(4, 256, device=device)

    y1 = fusion(x1, x2, x3)
    y2 = compiled(x1, x2, x3)
    print(f"Compiled match: {torch.allclose(y1, y2, atol=1e-5)}")

    # -------------------------------------------------------------------------
    test_section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nFusionComponent is ready.")