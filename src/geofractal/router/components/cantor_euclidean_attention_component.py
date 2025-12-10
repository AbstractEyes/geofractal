"""
geofractal.router.components.cantor_euclidean_attention
=======================================================

Hybrid Cantor-Euclidean Attention Component

PHILOSOPHICAL FOUNDATION:

    Euclidean mathematics has 300+ years of refinement. Gradients, backprop,
    optimization - these are mature tools. Cantor topology is deterministic
    mathematical truth, but we've only just begun exploring its integration
    with learned systems.

    This component BRIDGES the two geometries:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  EUCLIDEAN (Learned)           │  CANTOR (Fixed/Deterministic)     │
    │  - Q·K content similarity      │  - Branch path topology           │
    │  - Gradients flow here         │  - Guaranteed recursive structure │
    │  - 300+ years refined          │  - Mathematical truth, not helper │
    └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    LEARNED GATE THRESHOLD
                    (When to trust the wormhole)
                              │
                              ▼
                    ┌─────────────────────┐
                    │ If gate closed:     │
                    │   Pure Euclidean    │
                    │                     │
                    │ If gate open:       │
                    │   Fractal shortcut  │
                    └─────────────────────┘

WHAT THIS IS:
    - Euclidean attention learns WHAT to attend to (content)
    - Cantor topology provides WHERE shortcuts exist (structure)
    - Learned gates decide WHEN to use the shortcuts

WHAT THIS IS NOT:
    - Not replacing Euclidean with Cantor
    - Not "Cantor attention" - it's hybrid attention
    - Not claiming Cantor is "better" - it's DIFFERENT geometry

The mesh between π-based rotations (continuous, transcendental) and
Cantor dust (totally disconnected, measure zero) is inherently difficult.
The gate threshold is where we learn which fractal shortcuts actually
help the Euclidean optimization process.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Dict, Set, Union
from dataclasses import dataclass
import math

from geofractal.router.components.torch_component import TorchComponent


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CantorEuclideanConfig:
    """Configuration for hybrid attention."""
    feature_dim: int = 512
    num_heads: int = 8
    levels: int = 5
    branch_tau: float = 1.0
    gate_sharpness: float = 5.0
    gate_threshold: float = 0.3
    dropout: float = 0.1
    use_wormhole_skip: bool = True
    wormhole_strength: float = 0.3

    @property
    def head_dim(self) -> int:
        return self.feature_dim // self.num_heads


# =============================================================================
# SUB-COMPONENTS (as TorchComponents)
# =============================================================================

class BranchEncoder(TorchComponent):
    """
    Encode content embeddings into soft branch assignments.

    Unlike positional Cantor encoding, this assigns branches by CONTENT,
    enabling content-driven wormhole formation.

    Output: Soft probability over 3 branches at each of L levels.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        levels: int = 5,
        tau: float = 1.0,
        hidden_mult: int = 2,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.dim = dim
        self.levels = levels
        self.tau = tau

        # Two-layer projection for richer branch assignment
        hidden = dim * hidden_mult
        self.branch_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, levels * 3),
        )

        # Hierarchical level weights (coarse matters more)
        weights = 0.5 ** torch.arange(1, levels + 1, dtype=torch.float32)
        weights = weights / weights.sum()
        self.register_buffer('level_weights', weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, S, D) content embeddings
        Returns:
            branch_probs: (B, S, levels, 3) soft branch assignments
        """
        B, S, D = x.shape

        # Project to branch logits
        logits = self.branch_net(x)  # (B, S, levels * 3)

        # Reshape and normalize per level
        logits = logits.view(B, S, self.levels, 3)
        branch_probs = F.softmax(logits / self.tau, dim=-1)

        return branch_probs


class BranchAlignment(TorchComponent):
    """
    Compute branch path alignment between all token pairs.

    This is NOT a distance metric. It's a probability of path matching:

        align(i,j) = Σ_k w_k · P(same_branch_at_level_k)
                   = Σ_k w_k · Σ_b P_i(b,k) · P_j(b,k)

    High alignment = wormhole connection possible.
    """

    def __init__(self, name: str, levels: int = 5, **kwargs):
        super().__init__(name, **kwargs)
        self.levels = levels

        # Hierarchical weights
        weights = 0.5 ** torch.arange(1, levels + 1, dtype=torch.float32)
        weights = weights / weights.sum()
        self.register_buffer('level_weights', weights)

    def forward(self, branch_probs: Tensor) -> Tensor:
        """
        Args:
            branch_probs: (B, S, L, 3) soft branch assignments
        Returns:
            alignment: (B, S, S) pairwise branch alignment in [0, 1]
        """
        B, S, L, _ = branch_probs.shape

        # Ensure weights are on same device as input
        level_weights = self.level_weights.to(branch_probs.device)

        # P(same branch) = Σ_b P_i(b) · P_j(b) at each level
        # Expand: (B, S, 1, L, 3) × (B, 1, S, L, 3) → (B, S, S, L, 3)
        p_i = branch_probs.unsqueeze(2)
        p_j = branch_probs.unsqueeze(1)

        # Dot product over branches gives P(same branch at level k)
        same_branch_prob = (p_i * p_j).sum(dim=-1)  # (B, S, S, L)

        # Weighted sum over levels
        alignment = (same_branch_prob * level_weights).sum(dim=-1)  # (B, S, S)

        return alignment


class WormholeGate(TorchComponent):
    """
    Convert branch alignment to attention gates.

    Creates the wormhole topology:
        - High alignment → gate opens → information flows
        - Low alignment → gate closes → no connection

    The gate is topology, not content. It determines WHO can attend,
    not WHETHER they will (that's content's job).
    """

    def __init__(
        self,
        name: str,
        sharpness: float = 5.0,
        threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        # Learnable sharpness and threshold
        self.sharpness = nn.Parameter(torch.tensor(sharpness))
        self.threshold = nn.Parameter(torch.tensor(threshold))

    def forward(self, alignment: Tensor) -> Tensor:
        """
        Args:
            alignment: (B, S, S) branch alignment scores in [0, 1]
        Returns:
            gate: (B, S, S) wormhole gates in [0, 1]
        """
        # Sigmoid centered at threshold
        gate = torch.sigmoid(self.sharpness * (alignment - self.threshold))
        return gate

    @property
    def effective_threshold(self) -> float:
        return self.threshold.item()

    @property
    def effective_sharpness(self) -> float:
        return self.sharpness.item()


# =============================================================================
# MAIN COMPONENT
# =============================================================================

class CantorEuclideanAttention(TorchComponent):
    """
    Hybrid Cantor-Euclidean Attention

    Combines:
        1. Content-based branch encoding (Cantor topology)
        2. Branch alignment for wormhole gating (Cantor)
        3. Standard Q·K attention (Euclidean content)
        4. Gated value aggregation (Hybrid)

    The key insight:
        attention = wormhole_gate ⊙ softmax(Q·K / √d)

    Wormholes PERMIT attention based on topology.
    Content DIRECTS attention based on similarity.

    This allows long-range information flow between topologically
    aligned positions, even if sequentially distant.

    Inherits from TorchComponent:
        - name, uuid for identity
        - on_attach/on_detach lifecycle
        - device/dtype management
        - freeze/unfreeze
    """

    def __init__(
        self,
        name: str,
        config: CantorEuclideanConfig,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.config = config
        self.dim = config.feature_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.levels = config.levels
        self.scale = self.head_dim ** -0.5

        # ═══════════════════════════════════════════════════════════
        # Euclidean Components (standard attention)
        # ═══════════════════════════════════════════════════════════

        self.q_proj = nn.Linear(config.feature_dim, config.feature_dim, bias=False)
        self.k_proj = nn.Linear(config.feature_dim, config.feature_dim, bias=False)
        self.v_proj = nn.Linear(config.feature_dim, config.feature_dim, bias=False)
        self.out_proj = nn.Linear(config.feature_dim, config.feature_dim, bias=True)

        # ═══════════════════════════════════════════════════════════
        # Cantor Components (topology) - as child TorchComponents
        # ═══════════════════════════════════════════════════════════

        self.branch_encoder = BranchEncoder(
            name=f'{name}_branch_encoder',
            dim=config.feature_dim,
            levels=config.levels,
            tau=config.branch_tau,
        )

        self.branch_alignment = BranchAlignment(
            name=f'{name}_branch_alignment',
            levels=config.levels,
        )

        self.wormhole_gate = WormholeGate(
            name=f'{name}_wormhole_gate',
            sharpness=config.gate_sharpness,
            threshold=config.gate_threshold,
        )

        # ═══════════════════════════════════════════════════════════
        # Hybrid Components
        # ═══════════════════════════════════════════════════════════

        # Learnable balance between gated content and direct wormhole
        if config.use_wormhole_skip:
            self.wormhole_strength = nn.Parameter(
                torch.tensor(config.wormhole_strength)
            )
        else:
            self.register_buffer(
                'wormhole_strength',
                torch.tensor(0.0)
            )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Layer norms for stability
        self.norm_q = nn.LayerNorm(config.feature_dim)
        self.norm_k = nn.LayerNorm(config.feature_dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Forward pass implementing hybrid attention.

        Args:
            x: (B, S, D) input embeddings
            mask: Optional (B, S) or (B, S, S) attention mask
            return_info: Whether to return topology info

        Returns:
            output: (B, S, D) attended output
            info: Optional dict with topology tensors if return_info=True
        """
        B, S, D = x.shape
        H, head_dim = self.num_heads, self.head_dim

        # ═══════════════════════════════════════════════════════════
        # CANTOR: Compute Branch Topology
        # ═══════════════════════════════════════════════════════════

        # Encode content into branch assignments
        branch_probs = self.branch_encoder(x)  # (B, S, L, 3)

        # Compute pairwise branch alignment
        alignment = self.branch_alignment(branch_probs)  # (B, S, S)

        # Convert to wormhole gates
        wormhole_gates = self.wormhole_gate(alignment)  # (B, S, S)

        # ═══════════════════════════════════════════════════════════
        # EUCLIDEAN: Standard Q·K·V Attention
        # ═══════════════════════════════════════════════════════════

        # Normalized projections
        q = self.q_proj(self.norm_q(x))
        k = self.k_proj(self.norm_k(x))
        v = self.v_proj(x)

        # Reshape for multi-head: (B, S, D) → (B, H, S, head_dim)
        q = q.view(B, S, H, head_dim).transpose(1, 2)
        k = k.view(B, S, H, head_dim).transpose(1, 2)
        v = v.view(B, S, H, head_dim).transpose(1, 2)

        # Content similarity scores
        content_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S, S)

        # ═══════════════════════════════════════════════════════════
        # HYBRID: Combine Topology and Content
        # ═══════════════════════════════════════════════════════════

        # Expand gates for heads: (B, S, S) → (B, 1, S, S)
        gates_expanded = wormhole_gates.unsqueeze(1)

        # Additive log-gating (soft topology prior)
        # Allows some attention through low gates if content is very aligned
        gate_bias = torch.log(gates_expanded.clamp(min=1e-8))
        gated_scores = content_scores + gate_bias

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:  # (B, S) → (B, 1, 1, S)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:  # (B, S, S) → (B, 1, S, S)
                mask = mask.unsqueeze(1)
            gated_scores = gated_scores.masked_fill(~mask.bool(), float('-inf'))

        # Softmax over keys
        attention_weights = F.softmax(gated_scores, dim=-1)  # (B, H, S, S)
        attention_weights = self.dropout(attention_weights)

        # ═══════════════════════════════════════════════════════════
        # VALUE AGGREGATION
        # ═══════════════════════════════════════════════════════════

        # Standard attention output
        attn_out = torch.matmul(attention_weights, v)  # (B, H, S, head_dim)

        # Optional: Direct wormhole pathway
        if self.config.use_wormhole_skip:
            α = torch.sigmoid(self.wormhole_strength)

            # Wormhole attention (topology only, uniform over open gates)
            wormhole_weights = gates_expanded / (gates_expanded.sum(dim=-1, keepdim=True) + 1e-8)
            wormhole_out = torch.matmul(wormhole_weights, v)  # (B, H, S, head_dim)

            # Blend
            attn_out = (1 - α) * attn_out + α * wormhole_out

        # Reshape: (B, H, S, head_dim) → (B, S, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection
        output = self.out_proj(attn_out)

        if return_info:
            info = {
                'branch_probs': branch_probs,
                'alignment': alignment,
                'wormhole_gates': wormhole_gates,
                'attention_weights': attention_weights,
                'gate_threshold': self.wormhole_gate.effective_threshold,
                'gate_sharpness': self.wormhole_gate.effective_sharpness,
                'wormhole_strength': torch.sigmoid(self.wormhole_strength).item(),
            }
            return output, info

        return output, None

    def on_attach(self, parent) -> None:
        """Lifecycle hook when attached to router."""
        # Propagate to child components
        self.branch_encoder.parent = parent
        self.branch_alignment.parent = parent
        self.wormhole_gate.parent = parent
        super().on_attach(parent)

    def on_detach(self) -> None:
        """Lifecycle hook when detached from router."""
        self.branch_encoder.parent = None
        self.branch_alignment.parent = None
        self.wormhole_gate.parent = None
        super().on_detach()

    def __repr__(self) -> str:
        return (
            f"CantorEuclideanAttention("
            f"name='{self.name}', "
            f"dim={self.dim}, "
            f"heads={self.num_heads}, "
            f"levels={self.levels}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
            f")"
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_cantor_euclidean_attention(
    name: str,
    feature_dim: int = 512,
    num_heads: int = 8,
    levels: int = 5,
    dropout: float = 0.1,
    **kwargs
) -> CantorEuclideanAttention:
    """Factory function for router attachment."""
    config = CantorEuclideanConfig(
        feature_dim=feature_dim,
        num_heads=num_heads,
        levels=levels,
        dropout=dropout,
        **{k: v for k, v in kwargs.items() if hasattr(CantorEuclideanConfig, k)}
    )
    component_kwargs = {k: v for k, v in kwargs.items() if not hasattr(CantorEuclideanConfig, k)}
    return CantorEuclideanAttention(name, config, **component_kwargs)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  CantorEuclideanAttention TorchComponent Test")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create via factory
    attn = create_cantor_euclidean_attention(
        name='hybrid_attn',
        feature_dim=256,
        num_heads=8,
        levels=5,
        dropout=0.1,
    )

    print(f"Component: {attn}")
    print(f"UUID: {attn.uuid[:12]}...")
    print(f"Is attached: {attn.is_attached}")
    print()

    # Test forward
    B, S, D = 2, 64, 256
    x = torch.randn(B, S, D, device=device)
    attn = attn.to(device)

    # Without info
    output, _ = attn(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {output.shape}")

    # With info
    output, info = attn(x, return_info=True)
    print(f"\nTopology Info:")
    print(f"  branch_probs:     {info['branch_probs'].shape}")
    print(f"  alignment:        {info['alignment'].shape}")
    print(f"  wormhole_gates:   {info['wormhole_gates'].shape}")
    print(f"  attention_weights: {info['attention_weights'].shape}")
    print(f"  gate_threshold:   {info['gate_threshold']:.3f}")
    print(f"  wormhole_strength: {info['wormhole_strength']:.3f}")

    # Gate statistics
    gates = info['wormhole_gates'][0].cpu()
    print(f"\nGate Statistics:")
    print(f"  min={gates.min():.3f}, max={gates.max():.3f}, mean={gates.mean():.3f}")
    print(f"  % open (>0.5): {(gates > 0.5).float().mean():.1%}")

    # Gradient check
    print(f"\nGradient Check:")
    x_grad = torch.randn(B, S, D, device=device, requires_grad=True)
    out, _ = attn(x_grad)
    loss = out.sum()
    loss.backward()
    print(f"  grad norm: {x_grad.grad.norm():.4f}")
    print(f"  grad finite: {torch.isfinite(x_grad.grad).all()}")

    # Freeze/unfreeze
    print(f"\nFreeze/Unfreeze:")
    attn.freeze()
    trainable = sum(p.numel() for p in attn.parameters() if p.requires_grad)
    print(f"  After freeze - trainable params: {trainable}")

    attn.unfreeze()
    trainable = sum(p.numel() for p in attn.parameters() if p.requires_grad)
    print(f"  After unfreeze - trainable params: {trainable:,}")

    # Simulate router attachment
    print(f"\nLifecycle Simulation:")

    class FakeRouter:
        name = 'test_router'

    fake_parent = FakeRouter()
    attn.parent = fake_parent
    attn.on_attach(fake_parent)
    print(f"  After attach: parent='{attn.parent.name}'")
    print(f"  Child branch_encoder parent: {attn.branch_encoder.parent.name}")

    attn.on_detach()
    attn.parent = None
    print(f"  After detach: parent={attn.parent}")

    print("\n" + "=" * 70)
    print("  ✓ TorchComponent tests passed")
    print("=" * 70)