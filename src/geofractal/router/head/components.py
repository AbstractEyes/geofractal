"""
geofractal.router.head.components
=================================
Concrete implementations of head components.

Each component is a standalone module that can be:
- Used directly
- Subclassed for customization
- Replaced entirely with a compatible implementation

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from .protocols import (
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
    """
    Configuration for head components.

    This is the minimal config needed to construct any head component.
    """
    feature_dim: int = 512
    fingerprint_dim: int = 64
    num_heads: int = 8
    num_anchors: int = 16
    num_routes: int = 4
    dropout: float = 0.1
    temperature: float = 1.0

    # Cantor configuration
    use_cantor: bool = True
    grid_height: int = 16
    grid_width: int = 1

    @property
    def head_dim(self) -> int:
        return self.feature_dim // self.num_heads

    @property
    def num_slots(self) -> int:
        return self.grid_height * self.grid_width


# =============================================================================
# CANTOR UTILITIES
# =============================================================================

def cantor_pair(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cantor pairing: ℕ² → ℕ"""
    return ((x + y) * (x + y + 1)) // 2 + y


def build_cantor_bias(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Build Cantor bias matrix for attention."""
    y = torch.arange(height, device=device)
    x = torch.arange(width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    positions = torch.stack([yy.flatten(), xx.flatten()], dim=-1)

    cantor_idx = cantor_pair(positions[:, 0], positions[:, 1])
    diff = torch.abs(cantor_idx.unsqueeze(0) - cantor_idx.unsqueeze(1)).float()

    return 1.0 - (diff / (diff.max() + 1))


# =============================================================================
# ATTENTION COMPONENTS
# =============================================================================

class StandardAttention(BaseAttention):
    """Standard multi-head attention without geometric structure."""

    def __init__(self, config: HeadConfig):
        super().__init__()
        D, H = config.feature_dim, config.num_heads
        head_dim = config.head_dim

        self.num_heads = H
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(D, H * head_dim)
        self.k_proj = nn.Linear(D, H * head_dim)
        self.v_proj = nn.Linear(D, H * head_dim)
        self.out_proj = nn.Linear(H * head_dim, D)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, d).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, H * d)
        out = self.out_proj(out)

        return out, weights if return_weights else None


class CantorAttention(BaseAttention):
    """Multi-head attention with Cantor geometric structure."""

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        D, H = config.feature_dim, config.num_heads
        head_dim = config.head_dim

        self.num_heads = H
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(D, H * head_dim)
        self.k_proj = nn.Linear(D, H * head_dim)
        self.v_proj = nn.Linear(D, H * head_dim)
        self.out_proj = nn.Linear(H * head_dim, D)

        # Per-head Cantor scale
        self.cantor_scale = nn.Parameter(torch.ones(H) * 0.1)

        self.register_buffer('cantor_bias', None)
        self.dropout = nn.Dropout(config.dropout)

    def _ensure_bias(self, S: int, device: torch.device):
        if self.cantor_bias is None or self.cantor_bias.shape[0] != S:
            H, W = self.config.grid_height, self.config.grid_width
            if H * W != S:
                H, W = S, 1
            self.cantor_bias = build_cantor_bias(H, W, device)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, d).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add Cantor structure
        if self.config.use_cantor:
            self._ensure_bias(S, x.device)
            cantor = self.cantor_bias.unsqueeze(0).unsqueeze(0)
            cantor = cantor * self.cantor_scale.view(1, H, 1, 1)
            scores = scores + cantor

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, H * d)
        out = self.out_proj(out)

        return out, weights if return_weights else None


# =============================================================================
# ROUTER COMPONENTS
# =============================================================================

class TopKRouter(BaseRouter):
    """Top-K sparse routing with fingerprint modulation."""

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        D, F = config.feature_dim, config.fingerprint_dim

        self.K = config.num_routes
        self.score_proj = nn.Linear(D, D)
        self.fp_to_bias = nn.Linear(F, D)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, D = q.shape
        K = min(self.K, S)

        # Compute scores
        q_proj = self.score_proj(q)
        scores = torch.bmm(q_proj, k.transpose(-2, -1)) / (D ** 0.5)

        # Fingerprint bias
        fp_bias = self.fp_to_bias(fingerprint)
        bias = torch.einsum('bsd,d->bs', q, fp_bias)
        scores = scores + bias.unsqueeze(-1) * 0.1

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
    """Soft routing without top-K selection (full attention-like)."""

    def __init__(self, config: HeadConfig):
        super().__init__()
        D, F = config.feature_dim, config.fingerprint_dim

        self.score_proj = nn.Linear(D, D)
        self.fp_to_bias = nn.Linear(F, D)

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

        fp_bias = self.fp_to_bias(fingerprint)
        bias = torch.einsum('bsd,d->bs', q, fp_bias)
        scores = scores + bias.unsqueeze(-1) * 0.1

        weights = F.softmax(scores, dim=-1)
        output = torch.bmm(weights, v)

        # For compatibility, return "routes" as all indices
        routes = torch.arange(S, device=q.device).unsqueeze(0).unsqueeze(0).expand(B, S, -1)

        return routes, weights, output


# =============================================================================
# ANCHOR COMPONENTS
# =============================================================================

class ConstitutiveAnchorBank(BaseAnchorBank):
    """
    Anchor bank with constitutive (not additive) contribution.

    This is the correct implementation - anchors contribute
    directly to output, ensuring gradient flow.
    """

    def __init__(self, config: HeadConfig):
        super().__init__()
        D, A, F = config.feature_dim, config.num_anchors, config.fingerprint_dim

        self.anchors = nn.Parameter(torch.randn(A, D) * 0.02)

        self.fp_to_anchor = nn.Sequential(
            nn.Linear(F, A * 2),
            nn.GELU(),
            nn.Linear(A * 2, A),
        )

        # CONSTITUTIVE: Projects to output space
        self.anchor_out = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)

    def forward(
            self,
            x: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape

        affinities = torch.sigmoid(self.fp_to_anchor(fingerprint))
        weighted = (self.anchors * affinities.unsqueeze(-1)).sum(dim=0)
        anchor_out = self.anchor_out(weighted)
        anchor_out = anchor_out.unsqueeze(0).unsqueeze(0).expand(B, S, -1)

        return anchor_out, affinities


class AttentiveAnchorBank(BaseAnchorBank):
    """
    Anchor bank with attention-based selection.

    Input features attend to anchors, allowing input-dependent
    anchor selection (not just fingerprint-based).
    """

    def __init__(self, config: HeadConfig):
        super().__init__()
        D, A, F = config.feature_dim, config.num_anchors, config.fingerprint_dim

        self.anchors = nn.Parameter(torch.randn(A, D) * 0.02)

        # Input-based attention to anchors
        self.query_proj = nn.Linear(D, D)
        self.key_proj = nn.Linear(D, D)

        # Fingerprint modulation
        self.fp_to_scale = nn.Linear(F, A)

        self.anchor_out = nn.Linear(D, D)

    def forward(
            self,
            x: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        A = self.anchors.shape[0]

        # Input attends to anchors
        q = self.query_proj(x)  # [B, S, D]
        k = self.key_proj(self.anchors)  # [A, D]

        scores = torch.einsum('bsd,ad->bsa', q, k) / (D ** 0.5)  # [B, S, A]

        # Fingerprint modulates anchor importance
        fp_scale = torch.sigmoid(self.fp_to_scale(fingerprint))  # [A]
        scores = scores * fp_scale.unsqueeze(0).unsqueeze(0)

        affinities = F.softmax(scores, dim=-1)  # [B, S, A]

        # Weighted anchor combination per position
        attended = torch.einsum('bsa,ad->bsd', affinities, self.anchors)
        anchor_out = self.anchor_out(attended)

        # Return mean affinities for logging
        mean_affinities = affinities.mean(dim=(0, 1))

        return anchor_out, mean_affinities


# =============================================================================
# GATE COMPONENTS
# =============================================================================

class FingerprintGate(BaseGate):
    """Standard fingerprint-based gating."""

    def __init__(self, config: HeadConfig):
        super().__init__()
        D, F = config.feature_dim, config.fingerprint_dim

        self.fp_compare = nn.Sequential(
            nn.Linear(F * 2, F),
            nn.GELU(),
            nn.Linear(F, 1),
            nn.Sigmoid(),
        )

        self.fp_to_gate = nn.Sequential(
            nn.Linear(F, D),
            nn.Sigmoid(),
        )

    def gate_values(
            self,
            v: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.fp_to_gate(fingerprint)
        return v * gate.unsqueeze(0).unsqueeze(0)

    def compute_similarity(
            self,
            fp_self: torch.Tensor,
            fp_target: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([fp_self, fp_target], dim=-1)
        return self.fp_compare(combined)


class ChannelGate(BaseGate):
    """Channel attention-style gating with fingerprint conditioning."""

    def __init__(self, config: HeadConfig):
        super().__init__()
        D, F = config.feature_dim, config.fingerprint_dim

        # Squeeze-and-excitation style
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(D + F, D // 4),
            nn.GELU(),
            nn.Linear(D // 4, D),
            nn.Sigmoid(),
        )

        self.fp_compare = nn.Sequential(
            nn.Linear(F * 2, F),
            nn.GELU(),
            nn.Linear(F, 1),
            nn.Sigmoid(),
        )

    def gate_values(
            self,
            v: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        B, S, D = v.shape

        # Global context
        squeezed = v.transpose(1, 2)  # [B, D, S]
        squeezed = self.squeeze(squeezed).squeeze(-1)  # [B, D]

        # Combine with fingerprint
        combined = torch.cat([squeezed, fingerprint.unsqueeze(0).expand(B, -1)], dim=-1)
        gate = self.excite(combined)  # [B, D]

        return v * gate.unsqueeze(1)

    def compute_similarity(
            self,
            fp_self: torch.Tensor,
            fp_target: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([fp_self, fp_target], dim=-1)
        return self.fp_compare(combined)


# =============================================================================
# COMBINER COMPONENTS
# =============================================================================

class LearnableWeightCombiner(BaseCombiner):
    """Combine signals with learnable softmax weights."""

    def __init__(self, signal_names: list, init_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.signal_names = signal_names

        # Initialize weights
        if init_weights is None:
            init_weights = {name: 1.0 for name in signal_names}

        weights = [init_weights.get(name, 1.0) for name in signal_names]
        self.weights = nn.Parameter(torch.tensor(weights))

    def forward(self, signals: Dict[str, torch.Tensor]) -> torch.Tensor:
        normalized = F.softmax(self.weights, dim=0)

        combined = None
        for i, name in enumerate(self.signal_names):
            if name in signals:
                contrib = normalized[i] * signals[name]
                combined = contrib if combined is None else combined + contrib

        return combined


class GatedCombiner(BaseCombiner):
    """Combine signals with learned gating per position."""

    def __init__(self, feature_dim: int, signal_names: list):
        super().__init__()
        self.signal_names = signal_names
        num_signals = len(signal_names)

        # Gate network: input features → per-signal weights
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * num_signals, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_signals),
            nn.Softmax(dim=-1),
        )

    def forward(self, signals: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Stack signals
        stacked = torch.stack([signals[name] for name in self.signal_names], dim=-1)
        B, S, D, N = stacked.shape

        # Compute gates from concatenated signals
        concat = stacked.view(B, S, D * N)
        gates = self.gate_net(concat)  # [B, S, N]

        # Apply gates
        combined = (stacked * gates.unsqueeze(2)).sum(dim=-1)

        return combined


# =============================================================================
# REFINEMENT COMPONENTS
# =============================================================================

class FFNRefinement(BaseRefinement):
    """Standard FFN refinement with residual."""

    def __init__(self, config: HeadConfig, expansion: int = 4):
        super().__init__()
        D = config.feature_dim

        self.norm = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, D * expansion),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(D * expansion, D),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class MixtureOfExpertsRefinement(BaseRefinement):
    """MoE-style refinement with multiple expert FFNs."""

    def __init__(self, config: HeadConfig, num_experts: int = 4):
        super().__init__()
        D = config.feature_dim

        self.norm = nn.LayerNorm(D)

        # Router
        self.router = nn.Linear(D, num_experts)

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, D * 4),
                nn.GELU(),
                nn.Linear(D * 4, D),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        normed = self.norm(x)

        # Route to experts
        gates = F.softmax(self.router(normed), dim=-1)  # [B, S, E]

        # Apply experts
        expert_outputs = torch.stack([e(normed) for e in self.experts], dim=-1)  # [B, S, D, E]

        # Weighted combination
        refined = (expert_outputs * gates.unsqueeze(2)).sum(dim=-1)

        return x + refined


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'HeadConfig',
    # Attention
    'StandardAttention',
    'CantorAttention',
    # Routing
    'TopKRouter',
    'SoftRouter',
    # Anchors
    'ConstitutiveAnchorBank',
    'AttentiveAnchorBank',
    # Gates
    'FingerprintGate',
    'ChannelGate',
    # Combiners
    'LearnableWeightCombiner',
    'GatedCombiner',
    # Refinement
    'FFNRefinement',
    'MixtureOfExpertsRefinement',
    # Utilities
    'cantor_pair',
    'build_cantor_bias',
]