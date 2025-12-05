"""
geofractal.router.head.components
=================================
Swappable head components with gradient-healthy implementations.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math

from .head_protocols import (
    BaseAttention,
    BaseRouter,
    BaseAnchorBank,
    BaseGate,
    BaseCombiner,
    BaseRefinement,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HeadConfig:
    """Configuration for head components."""
    feature_dim: int = 512
    fingerprint_dim: int = 64
    num_heads: int = 8
    num_anchors: int = 16
    num_routes: int = 4
    dropout: float = 0.1
    use_cantor: bool = True
    ffn_expansion: int = 4
    temperature: float = 1.0

    @property
    def head_dim(self) -> int:
        return self.feature_dim // self.num_heads


# =============================================================================
# ATTENTION COMPONENTS
# =============================================================================

class StandardAttention(BaseAttention):
    """Standard multi-head attention."""

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.k_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.v_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.out_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attn_out = torch.matmul(weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(attn_out)

        if return_weights:
            return output, weights
        return output, None


class CantorAttention(BaseAttention):
    """
    Multi-head attention with Cantor pairing positional bias.

    The Cantor pairing function creates a unique diagonal structure
    that encodes spatial relationships geometrically.
    """

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.k_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.v_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.out_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Per-head learnable Cantor scale
        self.cantor_scale = nn.Parameter(torch.ones(config.num_heads) * 0.1)

    def _cantor_pair(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Cantor pairing: π(i,j) = ((i+j)(i+j+1))/2 + j"""
        return ((i + j) * (i + j + 1)) // 2 + j

    def _build_cantor_bias(self, S: int, device: torch.device) -> torch.Tensor:
        """Build [S, S] Cantor bias matrix."""
        idx = torch.arange(S, device=device)
        i, j = torch.meshgrid(idx, idx, indexing='ij')
        paired = self._cantor_pair(i, j).float()
        return torch.log1p(paired) / math.log(S + 1)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add Cantor bias (per-head scaling)
        cantor_bias = self._build_cantor_bias(S, x.device)
        scaled_bias = self.cantor_scale.view(-1, 1, 1) * cantor_bias.unsqueeze(0)
        scores = scores + scaled_bias.unsqueeze(0)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attn_out = torch.matmul(weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(attn_out)

        if return_weights:
            return output, weights
        return output, None


# =============================================================================
# ROUTER COMPONENTS
# =============================================================================

class TopKRouter(BaseRouter):
    """
    Top-K sparse router with fingerprint-guided key biasing.

    GRADIENT FIX: Fingerprint bias applies PER-KEY (not per-query).
    This ensures the bias affects relative key rankings within softmax,
    allowing gradients to flow back through fp_to_bias.
    """

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        self.K = config.num_routes

        self.score_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.fp_to_bias = nn.Linear(config.fingerprint_dim, config.feature_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, D = q.shape
        K = min(self.K, S)

        # Compute base scores
        q_proj = self.score_proj(q)
        scores = torch.bmm(q_proj, k.transpose(-2, -1)) / (D ** 0.5)

        # FIX: Fingerprint bias applied per-KEY for softmax differentiation
        fp_bias = self.fp_to_bias(fingerprint)  # [D]
        key_bias = torch.einsum('bsd,d->bs', k, fp_bias)  # [B, S] per-key bias
        scores = scores + key_bias.unsqueeze(1) * 0.1

        # Top-K selection
        topk_scores, routes = torch.topk(scores / self.config.temperature, K, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        # Gather values
        routes_exp = routes.unsqueeze(-1).expand(-1, -1, -1, D)
        v_exp = v.unsqueeze(1).expand(-1, S, -1, -1)
        gathered = torch.gather(v_exp, 2, routes_exp)

        output = (gathered * weights.unsqueeze(-1)).sum(dim=2)

        return routes, weights, output


class SoftRouter(BaseRouter):
    """
    Soft attention-based router (no hard selection).
    All positions contribute, weighted by learned scores.
    """

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config

        self.score_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.fp_to_bias = nn.Linear(config.fingerprint_dim, config.feature_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, D = q.shape

        q_proj = self.score_proj(q)
        scores = torch.bmm(q_proj, k.transpose(-2, -1)) / (D ** 0.5)

        # Fingerprint bias per-key
        fp_bias = self.fp_to_bias(fingerprint)
        key_bias = torch.einsum('bsd,d->bs', k, fp_bias)
        scores = scores + key_bias.unsqueeze(1) * 0.1

        weights = F.softmax(scores / self.config.temperature, dim=-1)
        output = torch.bmm(weights, v)

        routes = torch.arange(S, device=q.device).unsqueeze(0).unsqueeze(0).expand(B, S, -1)

        return routes, weights, output


# =============================================================================
# ANCHOR COMPONENTS
# =============================================================================

class ConstitutiveAnchorBank(BaseAnchorBank):
    """
    Anchor bank that constitutively contributes to output.

    GRADIENT FIX: LayerNorm receives properly shaped tensor via unsqueeze/squeeze.
    """

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config

        self.anchors = nn.Parameter(
            torch.randn(config.num_anchors, config.feature_dim) * 0.02
        )

        self.fp_to_anchor = nn.Sequential(
            nn.Linear(config.fingerprint_dim, config.num_anchors * 2),
            nn.GELU(),
            nn.Linear(config.num_anchors * 2, config.num_anchors),
        )

        self.anchor_out = nn.Linear(config.feature_dim, config.feature_dim)
        self.norm = nn.LayerNorm(config.feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape

        affinities = torch.sigmoid(self.fp_to_anchor(fingerprint))
        weighted = (self.anchors * affinities.unsqueeze(-1)).sum(dim=0)  # [D]

        anchor_out = self.anchor_out(weighted)  # [D]

        # FIX: Unsqueeze before norm, squeeze after (LayerNorm needs 2D+)
        anchor_out = anchor_out.unsqueeze(0)  # [1, D]
        anchor_out = self.norm(anchor_out)     # [1, D]
        anchor_out = anchor_out.squeeze(0)     # [D]

        anchor_out = anchor_out.unsqueeze(0).unsqueeze(0).expand(B, S, -1)

        return anchor_out, affinities


class AttentiveAnchorBank(BaseAnchorBank):
    """
    Anchor bank with input-dependent selection.
    """

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config

        self.anchors = nn.Parameter(
            torch.randn(config.num_anchors, config.feature_dim) * 0.02
        )

        self.fp_to_anchor = nn.Sequential(
            nn.Linear(config.fingerprint_dim, config.num_anchors * 2),
            nn.GELU(),
            nn.Linear(config.num_anchors * 2, config.num_anchors),
        )

        self.input_to_anchor = nn.Linear(config.feature_dim, config.num_anchors)
        self.anchor_out = nn.Linear(config.feature_dim, config.feature_dim)
        self.norm = nn.LayerNorm(config.feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape

        fp_affinity = torch.sigmoid(self.fp_to_anchor(fingerprint))
        input_affinity = torch.sigmoid(self.input_to_anchor(x))
        combined_affinity = fp_affinity.unsqueeze(0).unsqueeze(0) * input_affinity

        anchor_out = torch.einsum('bsn,nd->bsd', combined_affinity, self.anchors)
        anchor_out = self.anchor_out(anchor_out)
        anchor_out = self.norm(anchor_out)

        return anchor_out, combined_affinity.mean(dim=1)


# =============================================================================
# GATE COMPONENTS
# =============================================================================

class FingerprintGate(BaseGate):
    """
    Gating mechanism based on fingerprint.

    GRADIENT FIX: Proper forward() method implemented.
    """

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config

        self.fp_compare = nn.Sequential(
            nn.Linear(config.fingerprint_dim * 2, config.fingerprint_dim),
            nn.GELU(),
            nn.Linear(config.fingerprint_dim, 1),
            nn.Sigmoid(),
        )

        self.fp_to_gate = nn.Sequential(
            nn.Linear(config.fingerprint_dim, config.feature_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        fingerprint: torch.Tensor,
        target_fingerprint: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward pass."""
        gated = self.gate_values(x, fingerprint)

        if target_fingerprint is not None:
            similarity = self.compute_similarity(fingerprint, target_fingerprint)
            gated = gated * similarity

        return gated

    def gate_values(
        self,
        x: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        """Gate values based on fingerprint."""
        gate = self.fp_to_gate(fingerprint)
        return x * gate.unsqueeze(0).unsqueeze(0)

    def compute_similarity(
        self,
        fp_a: torch.Tensor,
        fp_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gating value from two fingerprints."""
        combined = torch.cat([fp_a, fp_b], dim=-1)
        return self.fp_compare(combined)


class ChannelGate(BaseGate):
    """Squeeze-and-excitation style channel gating."""

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config

        reduction = 4
        hidden = config.feature_dim // reduction

        self.excite = nn.Sequential(
            nn.Linear(config.feature_dim + config.fingerprint_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.feature_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        fingerprint: torch.Tensor,
        target_fingerprint: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        squeezed = x.mean(dim=1)
        fp_expanded = fingerprint.unsqueeze(0).expand(B, -1)
        combined = torch.cat([squeezed, fp_expanded], dim=-1)
        gate = self.excite(combined)
        return x * gate.unsqueeze(1)

    def gate_values(self, x: torch.Tensor, fingerprint: torch.Tensor) -> torch.Tensor:
        return self.forward(x, fingerprint)

    def compute_similarity(self, fp_a: torch.Tensor, fp_b: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(fp_a.unsqueeze(0), fp_b.unsqueeze(0))


# =============================================================================
# COMBINER COMPONENTS
# =============================================================================

class LearnableWeightCombiner(BaseCombiner):
    """Combine signals with learnable softmax weights."""

    def __init__(self, config: HeadConfig, **kwargs):
        super().__init__()
        self.config = config
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.1]))

    def forward(self, signals: Dict[str, torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.weights, dim=0)
        return (
                weights[0] * signals['attention'] +
                weights[1] * signals['routing'] +
                weights[2] * signals['anchors']
        )


class GatedCombiner(BaseCombiner):
    """Combine signals with input-dependent gating."""

    def __init__(self, config: HeadConfig, **kwargs):
        super().__init__()
        self.config = config

        self.gate_net = nn.Sequential(
            nn.Linear(config.feature_dim * 3, config.feature_dim),
            nn.GELU(),
            nn.Linear(config.feature_dim, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, signals: Dict[str, torch.Tensor]) -> torch.Tensor:
        concat = torch.cat([signals['attention'], signals['routing'], signals['anchors']], dim=-1)
        gates = self.gate_net(concat)
        return (
                gates[..., 0:1] * signals['attention'] +
                gates[..., 1:2] * signals['routing'] +
                gates[..., 2:3] * signals['anchors']
        )


# =============================================================================
# REFINEMENT COMPONENTS
# =============================================================================

class FFNRefinement(BaseRefinement):
    """Standard FFN refinement with residual."""

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config

        hidden = config.feature_dim * config.ffn_expansion

        self.norm = nn.LayerNorm(config.feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.feature_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class MixtureOfExpertsRefinement(BaseRefinement):
    """MoE-style refinement with multiple expert FFNs."""

    def __init__(self, config: HeadConfig, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k

        hidden = config.feature_dim * config.ffn_expansion

        self.norm = nn.LayerNorm(config.feature_dim)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.feature_dim),
            )
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(config.feature_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_norm = self.norm(x)

        logits = self.router(x_norm)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        expert_outputs = torch.stack([expert(x_norm) for expert in self.experts], dim=-2)
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, -1, D)
        selected = torch.gather(expert_outputs, 2, indices_exp)
        output = (selected * weights.unsqueeze(-1)).sum(dim=2)

        return x + output

# =============================================================================
# CANTOR PAIRING FUNCTIONS
# =============================================================================

def cantor_pair(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Cantor pairing function - maps 2D coordinates to unique 1D index.

    The Cantor pairing creates a self-similar diagonal structure
    that encodes spatial relationships geometrically.

    Args:
        x, y: Coordinate tensors (same shape)

    Returns:
        z: Paired indices with same shape
    """
    return ((x + y) * (x + y + 1)) // 2 + y


def cantor_unpair(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse Cantor pairing - recovers coordinates from paired index.

    Args:
        z: Paired indices

    Returns:
        x, y: Original coordinates
    """
    w = torch.floor((torch.sqrt(8 * z.float() + 1) - 1) / 2)
    t = (w * w + w) / 2
    y = (z.float() - t).long()
    x = (w - y.float()).long()
    return x, y


def build_cantor_bias(
    height: int,
    width: int,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Build Cantor pairing bias matrix for attention.

    Creates a self-similar structure where positions along
    diagonals have related attention patterns.

    Args:
        height, width: Grid dimensions
        device: Target device
        normalize: Whether to normalize to [0, 1]

    Returns:
        bias: [height * width, height * width] attention bias
    """
    # Create coordinate grids
    y_coords = torch.arange(height, device=device)
    x_coords = torch.arange(width, device=device)

    # Meshgrid for all positions
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    positions = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # [S, 2]

    S = height * width

    # Compute Cantor pair for each position
    cantor_indices = cantor_pair(positions[:, 0], positions[:, 1])  # [S]

    # Build pairwise relationship matrix
    # Positions with similar Cantor indices attend to each other
    diff = torch.abs(
        cantor_indices.unsqueeze(0) - cantor_indices.unsqueeze(1)
    ).float()  # [S, S]

    # Convert to similarity (closer Cantor indices = higher attention)
    if normalize:
        max_diff = diff.max() + 1
        bias = 1.0 - (diff / max_diff)
    else:
        bias = -diff

    return bias

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'HeadConfig',
    'StandardAttention',
    'CantorAttention',
    'TopKRouter',
    'SoftRouter',
    'ConstitutiveAnchorBank',
    'AttentiveAnchorBank',
    'FingerprintGate',
    'ChannelGate',
    'LearnableWeightCombiner',
    'GatedCombiner',
    'FFNRefinement',
    'MixtureOfExpertsRefinement',
    'cantor_pair',
    'cantor_unpair',
    'build_cantor_bias',
]