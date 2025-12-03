"""
geofractal.router.head.components
=================================
Swappable head components with gradient-healthy implementations.

FIXES APPLIED:
- ConstitutiveAnchorBank: norm now in forward path
- TopKRouter: fp_bias applies per-key (not per-query) for softmax differentiation
- FingerprintGate: proper forward() method implemented

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math

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
        # Normalize to reasonable range
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
        
        # Standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add Cantor bias (per-head scaling)
        cantor_bias = self._build_cantor_bias(S, x.device)  # [S, S]
        # Scale per head: [num_heads, 1, 1] * [S, S] -> [num_heads, S, S]
        scaled_bias = self.cantor_scale.view(-1, 1, 1) * cantor_bias.unsqueeze(0)
        scores = scores + scaled_bias.unsqueeze(0)  # [B, H, S, S]
        
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

    FIX: Fingerprint bias now applies PER-KEY (not per-query).
    This ensures the bias affects relative key rankings within softmax,
    allowing gradients to flow back through fp_to_bias.

    Old (broken): bias per query -> uniform shift -> softmax invariant -> grad ≈ 0
    New (fixed):  bias per key   -> differential shift -> softmax varies -> grad flows
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
        # Broadcast: [B, 1, S] adds to [B, S, S] -> varies across key dimension
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
    Good for comparison / ablation against TopK.
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
        
        # Compute scores
        q_proj = self.score_proj(q)
        scores = torch.bmm(q_proj, k.transpose(-2, -1)) / (D ** 0.5)
        
        # Fingerprint bias per-key
        fp_bias = self.fp_to_bias(fingerprint)
        key_bias = torch.einsum('bsd,d->bs', k, fp_bias)
        scores = scores + key_bias.unsqueeze(1) * 0.1
        
        # Soft weights (all positions)
        weights = F.softmax(scores / self.config.temperature, dim=-1)
        
        # Weighted combination
        output = torch.bmm(weights, v)
        
        # Routes are all indices (for compatibility)
        routes = torch.arange(S, device=q.device).unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        
        return routes, weights, output


# =============================================================================
# ANCHOR COMPONENTS
# =============================================================================

class ConstitutiveAnchorBank(BaseAnchorBank):
    """
    Anchor bank that constitutively contributes to output.

    Anchors are behavioral prototypes selected by fingerprint affinity.
    MUST be constitutive (directly contribute) not additive (bias scores).

    FIX: LayerNorm now applied in forward path for gradient flow.
    """
    
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        
        # Learnable anchor prototypes
        self.anchors = nn.Parameter(
            torch.randn(config.num_anchors, config.feature_dim) * 0.02
        )
        
        # Fingerprint -> anchor affinity
        self.fp_to_anchor = nn.Sequential(
            nn.Linear(config.fingerprint_dim, config.num_anchors * 2),
            nn.GELU(),
            nn.Linear(config.num_anchors * 2, config.num_anchors),
        )
        
        # Output projection
        self.anchor_out = nn.Linear(config.feature_dim, config.feature_dim)
        
        # FIX: Norm is now properly used in forward
        self.norm = nn.LayerNorm(config.feature_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        
        # Fingerprint -> anchor affinities
        affinities = torch.sigmoid(self.fp_to_anchor(fingerprint))  # [num_anchors]
        
        # Weighted anchor combination
        weighted = (self.anchors * affinities.unsqueeze(-1)).sum(dim=0)  # [D]
        
        # Project and normalize
        anchor_out = self.anchor_out(weighted)
        anchor_out = self.norm(anchor_out)  # FIX: Apply norm!
        
        # Expand to sequence
        anchor_out = anchor_out.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        
        return anchor_out, affinities


class AttentiveAnchorBank(BaseAnchorBank):
    """
    Anchor bank with input-dependent selection.

    Unlike ConstitutiveAnchorBank which uses only fingerprint,
    this also attends over input to select anchors dynamically.
    """
    
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        
        # Learnable anchor prototypes
        self.anchors = nn.Parameter(
            torch.randn(config.num_anchors, config.feature_dim) * 0.02
        )
        
        # Fingerprint -> base affinity
        self.fp_to_anchor = nn.Sequential(
            nn.Linear(config.fingerprint_dim, config.num_anchors * 2),
            nn.GELU(),
            nn.Linear(config.num_anchors * 2, config.num_anchors),
        )
        
        # Input -> anchor attention
        self.input_to_anchor = nn.Linear(config.feature_dim, config.num_anchors)
        
        # Output projection
        self.anchor_out = nn.Linear(config.feature_dim, config.feature_dim)
        self.norm = nn.LayerNorm(config.feature_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        
        # Fingerprint base affinity
        fp_affinity = torch.sigmoid(self.fp_to_anchor(fingerprint))  # [num_anchors]
        
        # Input-dependent attention (per position)
        input_affinity = torch.sigmoid(self.input_to_anchor(x))  # [B, S, num_anchors]
        
        # Combine: fingerprint gates which anchors, input selects strength
        combined_affinity = fp_affinity.unsqueeze(0).unsqueeze(0) * input_affinity  # [B, S, num_anchors]
        
        # Weighted anchor per position
        anchor_out = torch.einsum('bsn,nd->bsd', combined_affinity, self.anchors)
        anchor_out = self.anchor_out(anchor_out)
        anchor_out = self.norm(anchor_out)
        
        return anchor_out, combined_affinity.mean(dim=1)  # Return mean affinity for logging


# =============================================================================
# GATE COMPONENTS
# =============================================================================

class FingerprintGate(BaseGate):
    """
    Gating mechanism based on fingerprint.

    Two modes:
    1. Value gating: Gate input values based on own fingerprint
    2. Adjacent gating: Gate based on compatibility with another fingerprint

    FIX: Proper forward() method implemented.
    """
    
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        
        # Cross-fingerprint comparison (for adjacent gating)
        self.fp_compare = nn.Sequential(
            nn.Linear(config.fingerprint_dim * 2, config.fingerprint_dim),
            nn.GELU(),
            nn.Linear(config.fingerprint_dim, 1),
            nn.Sigmoid(),
        )
        
        # Fingerprint -> per-dimension gate
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
        """
        Full forward pass.

        Args:
            x: [B, S, D] input values
            fingerprint: Own fingerprint
            target_fingerprint: Optional adjacent fingerprint for cross-gating

        Returns:
            Gated values [B, S, D]
        """
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
        gate = self.fp_to_gate(fingerprint)  # [D]
        return x * gate.unsqueeze(0).unsqueeze(0)  # [B, S, D]
    
    def compute_similarity(
        self,
        fp_a: torch.Tensor,
        fp_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gating value from two fingerprints."""
        combined = torch.cat([fp_a, fp_b], dim=-1)
        return self.fp_compare(combined)


class ChannelGate(BaseGate):
    """
    Squeeze-and-excitation style channel gating.

    Uses input statistics + fingerprint for gating.
    """
    
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        
        reduction = 4
        hidden = config.feature_dim // reduction
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)
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
        
        # Squeeze: pool over sequence
        squeezed = x.mean(dim=1)  # [B, D]
        
        # Expand fingerprint to batch
        fp_expanded = fingerprint.unsqueeze(0).expand(B, -1)
        
        # Combine and excite
        combined = torch.cat([squeezed, fp_expanded], dim=-1)
        gate = self.excite(combined)  # [B, D]
        
        return x * gate.unsqueeze(1)
    
    def gate_values(self, x: torch.Tensor, fingerprint: torch.Tensor) -> torch.Tensor:
        return self.forward(x, fingerprint)
    
    def compute_similarity(self, fp_a: torch.Tensor, fp_b: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(fp_a.unsqueeze(0), fp_b.unsqueeze(0))


# =============================================================================
# COMBINER COMPONENTS
# =============================================================================

class LearnableWeightCombiner(BaseCombiner):
    """
    Combine signals with learnable weights.

    Weights are softmax-normalized for stability.
    Default initialization: attention=1, routing=1, anchors=0.1
    """
    
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        
        # Learnable combination weights [attention, routing, anchors]
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.1]))
    
    def forward(self, signals: Dict[str, torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.weights, dim=0)
        
        combined = (
            weights[0] * signals['attention'] +
            weights[1] * signals['routing'] +
            weights[2] * signals['anchors']
        )
        
        return combined


class GatedCombiner(BaseCombiner):
    """
    Combine signals with input-dependent gating.

    Learns to weight signals based on input content.
    """
    
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        
        # Input -> 3 gate values
        self.gate_net = nn.Sequential(
            nn.Linear(config.feature_dim * 3, config.feature_dim),
            nn.GELU(),
            nn.Linear(config.feature_dim, 3),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, signals: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Concatenate signals for gate computation
        concat = torch.cat([
            signals['attention'],
            signals['routing'],
            signals['anchors'],
        ], dim=-1)
        
        # Compute per-position gates
        gates = self.gate_net(concat)  # [B, S, 3]
        
        combined = (
            gates[..., 0:1] * signals['attention'] +
            gates[..., 1:2] * signals['routing'] +
            gates[..., 2:3] * signals['anchors']
        )
        
        return combined


# =============================================================================
# REFINEMENT COMPONENTS
# =============================================================================

class FFNRefinement(BaseRefinement):
    """
    Standard FFN refinement with residual.

    output = input + FFN(norm(input))
    """
    
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
    """
    MoE-style refinement with multiple expert FFNs.

    Router selects which experts to use per position.
    """
    
    def __init__(self, config: HeadConfig, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.top_k = top_k
        
        hidden = config.feature_dim * config.ffn_expansion
        
        self.norm = nn.LayerNorm(config.feature_dim)
        
        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.feature_dim),
            )
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(config.feature_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_norm = self.norm(x)
        
        # Route
        logits = self.router(x_norm)  # [B, S, num_experts]
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Compute expert outputs (simplified - full impl would be more efficient)
        expert_outputs = torch.stack([expert(x_norm) for expert in self.experts], dim=-2)  # [B, S, E, D]
        
        # Gather selected experts
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, S, K, D]
        selected = torch.gather(expert_outputs, 2, indices_exp)  # [B, S, K, D]
        
        # Weighted combination
        output = (selected * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, D]
        
        return x + output


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    'HeadConfig',
    # Attention
    'StandardAttention',
    'CantorAttention',
    # Router
    'TopKRouter',
    'SoftRouter',
    # Anchors
    'ConstitutiveAnchorBank',
    'AttentiveAnchorBank',
    # Gate
    'FingerprintGate',
    'ChannelGate',
    # Combiner
    'LearnableWeightCombiner',
    'GatedCombiner',
    # Refinement
    'FFNRefinement',
    'MixtureOfExpertsRefinement',
]