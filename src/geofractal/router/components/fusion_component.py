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
    Basic:
        - ConcatFusion: Concatenate then project
        - SumFusion: Weighted sum
        - GatedFusion: Learned gates per input
        - AttentionFusion: Cross-attention between inputs
        - AdaptiveFusion: Content-dependent weights (Lyra pattern)
        - BilinearFusion: Bilinear interaction (2 inputs)
        - ResidualFusion: Fuse with residual connection
        - SlotFusion: Fuse across slot dimension

    Geometric (from David):
        - GeometricAttentionGate: Cayley-Menger + Angular + MHA combined
        - CantorScaleFusion: Fractal geometry routing
        - HierarchicalTreeGating: Binary tree decisions

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, List, Tuple, Union
import math

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
            out_features: Optional[int] = None,
            normalize: bool = True,
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, out_features, **kwargs)

        self.normalize = normalize
        self.weights = nn.Parameter(torch.ones(num_inputs))

        # Output projection if dimensions differ
        if self.out_features != in_features:
            self.out_proj = nn.Linear(in_features, self.out_features)
        else:
            self.out_proj = None

    def fuse(self, *inputs: Tensor) -> Tensor:
        weights = F.softmax(self.weights, dim=0) if self.normalize else self.weights

        # Stack inputs: [num_inputs, B, ..., D]
        stacked = torch.stack(inputs, dim=0)

        # Weight and sum: [B, ..., D]
        # Reshape weights for broadcasting
        w_shape = [self.num_inputs] + [1] * (stacked.dim() - 1)
        weighted = stacked * weights.view(*w_shape)

        fused = weighted.sum(dim=0)

        if self.out_proj is not None:
            fused = self.out_proj(fused)

        return fused


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
            out_features: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, out_features, **kwargs)

        # Gate projection for each input
        self.gate_proj = nn.Linear(in_features, num_inputs)

        # Output projection if dimensions differ
        if self.out_features != in_features:
            self.out_proj = nn.Linear(in_features, self.out_features)
        else:
            self.out_proj = None

    def fuse(self, *inputs: Tensor) -> Tensor:
        # Stack to [B, N, D] - compile-friendly, no movedim
        stacked = torch.stack(inputs, dim=1)  # [B, N, D]
        pooled = stacked.mean(dim=1)  # [B, D]

        gates = torch.sigmoid(self.gate_proj(pooled))  # [B, N]
        gates = gates.unsqueeze(-1)  # [B, N, 1]

        gated = stacked * gates  # [B, N, D]
        fused = gated.sum(dim=1)  # [B, D]

        if self.out_proj is not None:
            fused = self.out_proj(fused)

        return fused


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
            out_features: Optional[int] = None,
            num_heads: int = 8,
            dropout: float = 0.0,
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, out_features, **kwargs)

        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        self.q_proj = nn.Linear(in_features, in_features)
        self.k_proj = nn.Linear(in_features, in_features)
        self.v_proj = nn.Linear(in_features, in_features)
        self.out_proj = nn.Linear(in_features, self.out_features)
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
            out_features: Optional[int] = None,
            hidden_features: Optional[int] = None,
            temperature: float = 1.0,
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, out_features, **kwargs)

        hidden = hidden_features or in_features // 4
        self.temperature = temperature

        # Per-input weight predictor
        self.weight_net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Output projection if dimensions differ
        if self.out_features != in_features:
            self.out_proj = nn.Linear(in_features, self.out_features)
        else:
            self.out_proj = None

    def fuse(self, *inputs: Tensor) -> Tensor:
        # Stack inputs: [N, B, ..., D]
        stacked = torch.stack(inputs, dim=0)

        # Compute weight for each input: [N, B, ..., 1]
        weights = self.weight_net(stacked)

        # Softmax over input dimension
        weights = F.softmax(weights / self.temperature, dim=0)

        # Weighted sum
        fused = (stacked * weights).sum(dim=0)

        # Project if needed
        if self.out_proj is not None:
            fused = self.out_proj(fused)

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
# GEOMETRIC ATTENTION GATE (from David)
# =============================================================================

class GeometricAttentionGate(FusionComponent):
    """
    Geometric attention gate using pentachoron-inspired multi-scale fusion.

    Combines three attention mechanisms:
    1. Multi-head attention (standard)
    2. Angular attention (cosine similarity -> angle -> exponential decay)
    3. Cayley-Menger attention (5-simplex volume as importance signal)

    The Cayley-Menger piece creates a pentachoron (5-cell) from feature pairs
    and uses the simplex volume as a geometric routing signal.

    From David: This is the core geometric fusion mechanism.
    """

    def __init__(
            self,
            name: str,
            num_inputs: int,
            in_features: int,
            out_features: Optional[int] = None,
            num_heads: int = 4,
            use_cayley_attention: bool = True,
            use_angular_attention: bool = True,
            temperature: float = 0.07,
            dropout: float = 0.1,
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, out_features, **kwargs)

        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.use_cayley = use_cayley_attention
        self.use_angular = use_angular_attention
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Query/Key/Value projections
        self.q_proj = nn.Linear(in_features, in_features)
        self.k_proj = nn.Linear(in_features, in_features)
        self.v_proj = nn.Linear(in_features, in_features)

        # Per-input embeddings
        self.input_embeddings = nn.Parameter(
            torch.randn(num_inputs, in_features) * 0.02
        )

        # Pentachoron role weights for angular attention
        # anchor=1.0, need=-0.75, relation=0.75, purpose=0.75, observer=-0.75
        if self.use_angular:
            role_weights = torch.tensor([1.0, -0.75, 0.75, 0.75, -0.75])
            # Pad or truncate to num_inputs
            if num_inputs <= 5:
                role_weights = role_weights[:num_inputs]
            else:
                role_weights = F.pad(role_weights, (0, num_inputs - 5), value=0.5)
            self.register_buffer("role_weights", role_weights)

        # Output projection (supports out_features)
        self.out_proj = nn.Sequential(
            nn.Linear(in_features, self.out_features),
            nn.LayerNorm(self.out_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Learnable combination weights for attention types
        num_attn_types = 1 + int(self.use_angular) + int(self.use_cayley)
        self.attention_weights = nn.Parameter(torch.ones(num_attn_types) / num_attn_types)

        self.dropout = nn.Dropout(dropout)

    def _compute_geometric_attention(self, features: Tensor, input_features: List[Tensor]) -> Tensor:
        """Compute attention based on angular relationships."""
        features_norm = F.normalize(features, dim=-1)

        geometric_scores = []
        for i, inp_feat in enumerate(input_features):
            inp_feat_norm = F.normalize(inp_feat, dim=-1)

            # Cosine similarity -> angle
            cos_sim = (features_norm * inp_feat_norm).sum(dim=-1, keepdim=True)
            angles = torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))

            # Apply role weight
            if hasattr(self, 'role_weights'):
                angles = angles * self.role_weights[i].abs()

            geometric_scores.append(angles)

        angles_stack = torch.cat(geometric_scores, dim=-1)
        # Exponential decay from angle
        attention = torch.exp(-angles_stack / self.temperature.abs())

        return attention

    def _compute_cayley_attention(self, features: Tensor, input_features: List[Tensor]) -> Tensor:
        """
        Compute attention based on Cayley-Menger volumes.

        Creates a 5-point simplex (pentachoron) from feature pairs
        and uses volume as importance signal.
        """
        volume_scores = []

        for inp_feat in input_features:
            # Build 5 points for simplex
            points = [features, inp_feat]

            # Add 3 interpolated points at different angles
            for j in range(3):
                angle = (j + 1) * math.pi / 4
                rot_feat = features * math.cos(angle) + inp_feat * math.sin(angle)
                points.append(rot_feat)

            # Stack to [B, 5, D]
            simplex = torch.stack(points, dim=1)

            # Pairwise squared distances
            diff = simplex.unsqueeze(2) - simplex.unsqueeze(1)  # [B, 5, 5, D]
            distsq = (diff * diff).sum(dim=-1)  # [B, 5, 5]

            # Volume proxy (mean of squared distances)
            # Full Cayley-Menger determinant is expensive, this is a fast approximation
            volume_proxy = distsq.mean(dim=(1, 2))
            volume_scores.append(volume_proxy.unsqueeze(1))

        volumes = torch.cat(volume_scores, dim=1)  # [B, N]
        attention = F.softmax(volumes / self.temperature.abs(), dim=-1)

        return attention

    def _compute_multihead_attention(self, features: Tensor, input_features: List[Tensor]) -> Tensor:
        """Standard multi-head attention over inputs."""
        B = features.shape[0]

        Q = self.q_proj(features)
        Q = Q.view(B, self.num_heads, self.head_dim)

        K_list, V_list = [], []
        for i, inp_feat in enumerate(input_features):
            # Add input embedding
            inp_feat_emb = inp_feat + self.input_embeddings[i]

            K = self.k_proj(inp_feat_emb).view(B, self.num_heads, self.head_dim)
            V = self.v_proj(inp_feat_emb).view(B, self.num_heads, self.head_dim)

            K_list.append(K.unsqueeze(2))
            V_list.append(V.unsqueeze(2))

        K = torch.cat(K_list, dim=2)  # [B, H, N, D]
        V = torch.cat(V_list, dim=2)  # [B, H, N, D]

        # Attention scores
        scores = torch.einsum('bhd,bhnd->bhn', Q, K) / math.sqrt(self.head_dim)
        attn = F.softmax(scores / self.temperature.abs(), dim=-1)
        attn = self.dropout(attn)

        # Average over heads
        attn_avg = attn.mean(dim=1)  # [B, N]

        return attn_avg

    def fuse(self, *inputs: Tensor) -> Tensor:
        """
        Fuse inputs using geometric attention.

        Args:
            *inputs: Input tensors [B, D] or [B, L, D]

        Returns:
            Fused tensor, same shape as inputs
        """
        # Use mean of inputs as query features
        input_list = list(inputs)
        features = torch.stack(input_list, dim=0).mean(dim=0)

        attention_types = []

        # 1. Standard multi-head attention
        mha_attention = self._compute_multihead_attention(features, input_list)
        attention_types.append(mha_attention)

        # 2. Angular/geometric attention
        if self.use_angular:
            geo_attention = self._compute_geometric_attention(features, input_list)
            geo_attention = F.softmax(geo_attention, dim=-1)
            attention_types.append(geo_attention)

        # 3. Cayley-Menger volume attention
        if self.use_cayley:
            cayley_attention = self._compute_cayley_attention(features, input_list)
            attention_types.append(cayley_attention)

        # Combine attention types with learnable weights
        attn_weights = F.softmax(self.attention_weights[:len(attention_types)], dim=0)
        combined_attention = sum(
            w * attn for w, attn in zip(attn_weights, attention_types)
        )

        # Normalize
        combined_attention = combined_attention / combined_attention.sum(dim=-1, keepdim=True)

        # Apply attention to inputs
        stacked = torch.stack(input_list, dim=1)  # [B, N, D]
        fused = (stacked * combined_attention.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # Project to output dimension
        fused = self.out_proj(fused)

        return fused

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"inputs={self.num_inputs}, features={self.in_features}, "
            f"cayley={self.use_cayley}, angular={self.use_angular}, "
            f"params={self.num_parameters():,})"
        )


# =============================================================================
# CANTOR SCALE FUSION (from David)
# =============================================================================

class CantorScaleFusion(FusionComponent):
    """
    Cantor-based multi-scale fusion using fractal geometry for routing.

    Maps each input to a Cantor set coordinate, then uses sparse attention
    based on fractal proximity. Inputs that are "close" in Cantor space
    attend to each other more strongly.

    The Cantor coordinate is computed by iteratively dividing [0,1] into
    thirds and tracking which third each position falls into.

    From David: 70% Cantor routing, 30% learned gate.
    """

    def __init__(
            self,
            name: str,
            num_inputs: int,
            in_features: int,
            out_features: Optional[int] = None,
            num_heads: int = 4,
            cantor_depth: int = 8,
            local_window: int = 3,
            temperature: float = 0.07,
            dropout: float = 0.1,
            cantor_weight: float = 0.7,
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, out_features, **kwargs)

        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.cantor_depth = cantor_depth
        self.local_window = min(local_window, num_inputs)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.cantor_weight = cantor_weight

        # QKV projections
        self.q_proj = nn.Linear(in_features, in_features)
        self.k_proj = nn.Linear(in_features, in_features)
        self.v_proj = nn.Linear(in_features, in_features)

        # Input embeddings
        self.input_embeddings = nn.Parameter(
            torch.randn(num_inputs, in_features) * 0.02
        )

        # Pre-compute Cantor coordinates
        self.register_buffer(
            'cantor_coords',
            self._compute_cantor_coordinates()
        )

        # Pre-compute routing table
        self.register_buffer(
            'routes',
            self._build_routes()
        )

        # Output projection (supports out_features)
        self.out_proj = nn.Linear(in_features, self.out_features)
        self.dropout = nn.Dropout(dropout)

        # Learned gate network
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, num_inputs)
        )

    def _cantor_coordinate(self, position: int, max_len: int, depth: int) -> float:
        """Compute Cantor set coordinate for a position."""
        x = position / max(1, max_len - 1) if max_len > 1 else 0.5
        x = max(1e-6, min(x, 1.0 - 1e-6))

        cantor_val = 0.0
        factor = 0.5

        for _ in range(depth):
            x *= 3.0
            digit = int(x)
            x -= digit

            if digit == 2:
                cantor_val += factor

            factor *= 0.5

        return cantor_val

    def _compute_cantor_coordinates(self) -> Tensor:
        """Map each input position to a Cantor coordinate."""
        coords = torch.tensor([
            self._cantor_coordinate(i, self.num_inputs, self.cantor_depth)
            for i in range(self.num_inputs)
        ], dtype=torch.float32)

        return coords

    def _build_routes(self) -> Tensor:
        """Build routing table: which inputs attend to which."""
        routes = torch.zeros(self.num_inputs, self.local_window, dtype=torch.long)

        for i in range(self.num_inputs):
            distances = torch.abs(self.cantor_coords - self.cantor_coords[i])
            _, nearest = torch.topk(distances, self.local_window, largest=False)
            routes[i] = nearest

        return routes

    def _sparse_cantor_attention(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor
    ) -> Tensor:
        """Sparse attention using Cantor routing."""
        B, H, N, D = k.shape
        device = q.device

        routes = self.routes.to(device)  # [N, W]
        routes_exp = routes.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)  # [B, H, N, W]

        # Gather keys and values according to routes
        batch_idx = torch.arange(B, device=device).view(-1, 1, 1, 1).expand(B, H, N, self.local_window)
        head_idx = torch.arange(H, device=device).view(1, -1, 1, 1).expand(B, H, N, self.local_window)

        k_gathered = k[batch_idx, head_idx, routes_exp, :]  # [B, H, N, W, D]
        v_gathered = v[batch_idx, head_idx, routes_exp, :]  # [B, H, N, W, D]

        # Attention scores
        q_exp = q.unsqueeze(3)  # [B, H, N, 1, D]
        scores = (q_exp * k_gathered).sum(dim=-1) / math.sqrt(D)  # [B, H, N, W]

        attn_weights = F.softmax(scores / self.temperature.abs(), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        output = (attn_weights.unsqueeze(-1) * v_gathered).sum(dim=3)  # [B, H, N, D]

        # Compute importance as norm
        importance = output.norm(dim=-1).mean(dim=1)  # [B, N]

        return importance

    def fuse(self, *inputs: Tensor) -> Tensor:
        """
        Fuse inputs using Cantor-based routing.

        Args:
            *inputs: Input tensors [B, D]

        Returns:
            Fused tensor [B, D]
        """
        input_list = list(inputs)
        B = input_list[0].shape[0]

        # Mean as query features
        features = torch.stack(input_list, dim=0).mean(dim=0)

        # Add embeddings to inputs
        input_features = [
            inp + self.input_embeddings[i]
            for i, inp in enumerate(input_list)
        ]

        # Project to Q, K, V
        Q = self.q_proj(features).view(B, self.num_heads, 1, self.head_dim)
        Q = Q.expand(-1, -1, self.num_inputs, -1)  # [B, H, N, D]

        K_list = [self.k_proj(f).view(B, self.num_heads, 1, self.head_dim) for f in input_features]
        V_list = [self.v_proj(f).view(B, self.num_heads, 1, self.head_dim) for f in input_features]

        K = torch.cat(K_list, dim=2)  # [B, H, N, D]
        V = torch.cat(V_list, dim=2)  # [B, H, N, D]

        # Sparse Cantor attention
        cantor_importance = self._sparse_cantor_attention(Q, K, V)

        # Learned gate
        gate_logits = self.gate_net(features)

        # Combine: cantor_weight * cantor + (1 - cantor_weight) * learned
        combined_scores = (
            self.cantor_weight * cantor_importance +
            (1 - self.cantor_weight) * gate_logits
        )

        # Final attention weights
        attention_weights = F.softmax(combined_scores, dim=-1)

        # Apply to inputs
        stacked = torch.stack(input_list, dim=1)  # [B, N, D]
        fused = (stacked * attention_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # Project to output dimension
        fused = self.out_proj(fused)

        return fused

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"inputs={self.num_inputs}, features={self.in_features}, "
            f"cantor_depth={self.cantor_depth}, window={self.local_window}, "
            f"params={self.num_parameters():,})"
        )


# =============================================================================
# HIERARCHICAL TREE GATING (from David)
# =============================================================================

class HierarchicalTreeGating(FusionComponent):
    """
    Tree-based hierarchical gating.

    Binary tree where each node makes a left/right decision.
    Path probabilities multiply down the tree to produce leaf weights.
    Leaf weights are mapped to input weights.

    Combines with a direct gate via learnable mixing weight.

    From David: Tree structure provides hierarchical reasoning over inputs.
    """

    def __init__(
            self,
            name: str,
            num_inputs: int,
            in_features: int,
            out_features: Optional[int] = None,
            depth: int = 3,
            node_hidden: int = 64,
            combine_weight: float = 0.7,
            **kwargs,
    ):
        super().__init__(name, num_inputs, in_features, out_features, **kwargs)

        self.depth = depth
        self.num_leaves = 2 ** depth

        # Tree nodes: each level has 2^level nodes
        self.tree_nodes = nn.ModuleList()
        for level in range(depth):
            num_nodes = 2 ** level
            level_nodes = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features, node_hidden),
                    nn.LayerNorm(node_hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(node_hidden, 2)  # Binary decision
                ) for _ in range(num_nodes)
            ])
            self.tree_nodes.append(level_nodes)

        # Map leaves to input weights
        self.leaf_to_input = nn.Sequential(
            nn.Linear(self.num_leaves, num_inputs * 2),
            nn.GELU(),
            nn.Linear(num_inputs * 2, num_inputs)
        )

        # Direct gate (bypass tree)
        self.direct_gate = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.GELU(),
            nn.Linear(in_features // 4, num_inputs)
        )

        # Learnable combination weight
        self.combine_weight = nn.Parameter(torch.tensor(combine_weight))

        # Output projection if dimensions differ
        if self.out_features != in_features:
            self.out_proj = nn.Linear(in_features, self.out_features)
        else:
            self.out_proj = None

    def fuse(self, *inputs: Tensor) -> Tensor:
        """
        Fuse inputs using hierarchical tree gating.

        Args:
            *inputs: Input tensors [B, D]

        Returns:
            Fused tensor [B, D]
        """
        input_list = list(inputs)
        features = torch.stack(input_list, dim=0).mean(dim=0)

        B = features.shape[0]
        device = features.device

        # Traverse tree to get leaf probabilities
        path_probs = [torch.ones(B, 1, device=device)]

        for level in range(self.depth):
            next_probs = []
            num_nodes = 2 ** level

            for node_idx in range(num_nodes):
                parent_prob = path_probs[0] if level == 0 else path_probs[node_idx]

                # Binary decision at this node
                node_logits = self.tree_nodes[level][node_idx](features)
                node_probs = F.softmax(node_logits / 0.5, dim=-1)

                # Split probability to children
                left_prob = parent_prob * node_probs[:, 0:1]
                right_prob = parent_prob * node_probs[:, 1:2]
                next_probs.extend([left_prob, right_prob])

            path_probs = next_probs

        # Concatenate leaf probabilities
        leaf_probs = torch.cat(path_probs, dim=1)  # [B, num_leaves]

        # Map to input weights
        tree_gates = F.softmax(self.leaf_to_input(leaf_probs), dim=-1)

        # Direct gate
        direct_gates = F.softmax(self.direct_gate(features), dim=-1)

        # Combine tree and direct gates
        alpha = torch.sigmoid(self.combine_weight)
        gates = alpha * tree_gates + (1 - alpha) * direct_gates
        gates = gates / gates.sum(dim=-1, keepdim=True)

        # Apply gates to inputs
        stacked = torch.stack(input_list, dim=1)  # [B, N, D]
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # Project to output dimension if needed
        if self.out_proj is not None:
            fused = self.out_proj(fused)

        return fused

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"inputs={self.num_inputs}, features={self.in_features}, "
            f"depth={self.depth}, leaves={self.num_leaves}, "
            f"params={self.num_parameters():,})"
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
    test_section("GEOMETRIC ATTENTION GATE")
    # -------------------------------------------------------------------------

    fusion = GeometricAttentionGate(
        'geometric', num_inputs=5, in_features=256,
        use_cayley_attention=True, use_angular_attention=True
    )
    print(f"Component: {fusion}")

    inputs = [torch.randn(4, 256) for _ in range(5)]
    y = fusion(*inputs)
    print(f"Fuse: 5x[4, 256] -> {y.shape}")

    # Without Cayley (faster)
    fusion_fast = GeometricAttentionGate(
        'geometric_fast', num_inputs=5, in_features=256,
        use_cayley_attention=False, use_angular_attention=True
    )
    y = fusion_fast(*inputs)
    print(f"Fuse (no Cayley): 5x[4, 256] -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("CANTOR SCALE FUSION")
    # -------------------------------------------------------------------------

    fusion = CantorScaleFusion(
        'cantor', num_inputs=5, in_features=256,
        cantor_depth=8, local_window=3
    )
    print(f"Component: {fusion}")
    print(f"Cantor coords: {fusion.cantor_coords.tolist()}")
    print(f"Routes:\n{fusion.routes}")

    y = fusion(*inputs)
    print(f"Fuse: 5x[4, 256] -> {y.shape}")

    # -------------------------------------------------------------------------
    test_section("HIERARCHICAL TREE GATING")
    # -------------------------------------------------------------------------

    fusion = HierarchicalTreeGating(
        'tree', num_inputs=5, in_features=256, depth=3
    )
    print(f"Component: {fusion}")

    y = fusion(*inputs)
    print(f"Fuse: 5x[4, 256] -> {y.shape}")

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

    fusion = GeometricAttentionGate(
        'gpu_geometric', num_inputs=5, in_features=256
    ).to(device)

    inputs = [torch.randn(4, 256, device=device) for _ in range(5)]

    y1 = fusion(*inputs)
    print(f"GeometricAttentionGate on {device}: {y1.shape}")

    # Compile test
    fusion_compiled = torch.compile(AdaptiveFusion('compiled', num_inputs=3, in_features=256).to(device))
    x1 = torch.randn(4, 256, device=device)
    x2 = torch.randn(4, 256, device=device)
    x3 = torch.randn(4, 256, device=device)

    y2 = fusion_compiled(x1, x2, x3)
    print(f"Compiled AdaptiveFusion: {y2.shape}")

    # -------------------------------------------------------------------------
    test_section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nFusionComponent is ready.")
    print("\nNew geometric fusion strategies:")
    print("  - GeometricAttentionGate: Cayley-Menger + Angular + MHA")
    print("  - CantorScaleFusion: Fractal geometry routing")
    print("  - HierarchicalTreeGating: Binary tree decisions")