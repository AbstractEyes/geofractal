"""
Constitutive Fingerprint Router - Sparse & Full Versions
=========================================================
Fingerprint is not a bias - it shapes the projections themselves.
Cantor pairing defines behavioral identity, not just geometric prior.

Sparse: Fingerprint gates values, Cantor masks connectivity
Full: Fingerprint shapes Q/K/V projections, anchors contribute to output

Author: AbstractPhil
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# CANTOR BEHAVIORAL SYSTEM
# =============================================================================

class CantorBehavior(nn.Module):
    """
    Cantor pairing as behavioral identity, not just geometric bias.

    Each module gets a unique Cantor "lens" - how it perceives spatial relationships.
    The fingerprint IS the Cantor parameters, making it constitutive.
    """

    def __init__(
            self,
            num_positions: int,
            grid_size: int,
            fingerprint_dim: int = 64,
            num_basis: int = 32,
    ):
        super().__init__()
        self.num_positions = num_positions
        self.grid_size = grid_size
        self.fingerprint_dim = fingerprint_dim
        self.num_basis = num_basis

        # Build base Cantor indices for grid
        self.register_buffer('base_cantor', self._build_cantor_indices())

        # Learnable basis vectors that Cantor selects from
        self.basis = nn.Parameter(torch.randn(num_basis, fingerprint_dim) * 0.02)

        # Fingerprint-to-offset: converts fingerprint to Cantor offset
        self.fp_to_offset = nn.Linear(fingerprint_dim, 3)  # scale, shift_x, shift_y

    def _build_cantor_indices(self) -> torch.Tensor:
        """Build Cantor pairing indices for grid positions."""
        P, G = self.num_positions, self.grid_size

        pos = torch.arange(P)
        x = pos % G
        y = pos // G

        # Cantor pairing: z = ((x+y)*(x+y+1))/2 + y
        z = ((x + y) * (x + y + 1)) // 2 + y

        return z.float()

    def get_behavioral_indices(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """
        Get Cantor indices offset by fingerprint.
        Each fingerprint creates a unique "lens" on spatial relationships.
        """
        # Fingerprint → offset parameters
        params = self.fp_to_offset(fingerprint)  # [3]
        scale = params[0].sigmoid() * 2 + 0.5  # [0.5, 2.5]
        shift = params[1:2].tanh() * self.grid_size  # [-G, G]

        # Apply offset to base Cantor
        offset_cantor = self.base_cantor * scale + shift.sum()

        return offset_cantor

    def get_basis_for_positions(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """
        Select basis vectors for each position based on Cantor + fingerprint.
        Returns: [P, fingerprint_dim]
        """
        cantor_idx = self.get_behavioral_indices(fingerprint)

        # Map to basis indices
        basis_idx = (cantor_idx.long() % self.num_basis)

        # Gather basis vectors
        return self.basis[basis_idx]  # [P, fingerprint_dim]

    def get_affinity_matrix(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """
        Cantor-derived affinity matrix shaped by fingerprint.
        """
        cantor_idx = self.get_behavioral_indices(fingerprint)
        P = cantor_idx.shape[0]

        # Normalized distance in Cantor space
        cantor_norm = cantor_idx / cantor_idx.max().clamp(min=1)
        dist = (cantor_norm.unsqueeze(0) - cantor_norm.unsqueeze(1)).abs()

        # Affinity (closer in Cantor space = higher affinity)
        affinity = 1.0 - dist

        # Mask diagonal
        affinity = affinity.masked_fill(torch.eye(P, device=affinity.device).bool(), -1e9)

        return affinity


# =============================================================================
# CONSTITUTIVE ANCHOR BANK
# =============================================================================

class ConstitutiveAnchorBank(nn.Module):
    """
    Anchors that MUST contribute to output, not just compute affinities.

    Each anchor is a behavioral mode. The output is a weighted combination
    of anchor-transformed features, forcing specialization.
    """

    def __init__(
            self,
            num_anchors: int,
            feature_dim: int,
            fingerprint_dim: int,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.feature_dim = feature_dim

        # Anchor embeddings (behavioral modes)
        self.anchor_embeds = nn.Parameter(torch.randn(num_anchors, feature_dim) * 0.02)

        # Each anchor has a transformation it applies to features
        self.anchor_transforms = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim, bias=False)
            for _ in range(num_anchors)
        ])

        # Fingerprint influences anchor selection
        self.fp_to_anchor_bias = nn.Linear(fingerprint_dim, num_anchors)

        # Feature projection for affinity computation
        self.feature_proj = nn.Linear(feature_dim, feature_dim)

    def forward(
            self,
            features: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, D] pooled features
            fingerprint: [fingerprint_dim] router fingerprint

        Returns:
            output: [B, D] anchor-transformed features (CONTRIBUTES TO OUTPUT)
            affinities: [B, num_anchors] anchor selection weights
            anchor_predictions: [B, num_anchors, D] per-anchor outputs
        """
        B, D = features.shape

        # Compute affinities
        feat_proj = F.normalize(self.feature_proj(features), dim=-1)
        anchors_norm = F.normalize(self.anchor_embeds, dim=-1)

        affinities = torch.matmul(feat_proj, anchors_norm.T)  # [B, num_anchors]

        # Add fingerprint bias
        fp_bias = self.fp_to_anchor_bias(fingerprint)  # [num_anchors]
        affinities = affinities + 0.3 * fp_bias

        affinities = F.softmax(affinities, dim=-1)

        # Apply each anchor's transformation
        anchor_outputs = []
        for i, transform in enumerate(self.anchor_transforms):
            anchor_out = transform(features)  # [B, D]
            anchor_outputs.append(anchor_out)

        anchor_predictions = torch.stack(anchor_outputs, dim=1)  # [B, num_anchors, D]

        # Weighted combination - THIS IS THE OUTPUT, not discarded
        output = torch.einsum('ba,bad->bd', affinities, anchor_predictions)

        return output, affinities, anchor_predictions


# =============================================================================
# SPARSE CONSTITUTIVE ROUTER
# =============================================================================

class SparseConstitutiveRouter(nn.Module):
    """
    Sparse version: Fingerprint gates values, Cantor masks connectivity.

    - Cantor behavioral identity (can't be ignored - shapes mask)
    - Fingerprint gates V projection (constitutive)
    - Minimal overhead
    """

    def __init__(
            self,
            dim: int,
            num_positions: int,
            grid_size: int,
            num_routes: int = 8,
            fingerprint_dim: int = 64,
            temperature: float = 0.1,
            cantor_sparsity: float = 0.3,  # Keep top 30% of Cantor affinities
    ):
        super().__init__()
        self.dim = dim
        self.num_positions = num_positions
        self.num_routes = min(num_routes, num_positions - 1)
        self.temperature = temperature
        self.cantor_sparsity = cantor_sparsity

        # Cantor behavioral system
        self.cantor = CantorBehavior(num_positions, grid_size, fingerprint_dim)

        # Standard projections
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # CONSTITUTIVE: Fingerprint gates values
        self.fp_value_gate = nn.Sequential(
            nn.Linear(fingerprint_dim, dim),
            nn.Sigmoid(),
        )

        # Learnable fingerprint (behavioral identity)
        self.fingerprint = nn.Parameter(torch.randn(fingerprint_dim) * 0.02)

    def forward(
            self,
            x: torch.Tensor,
            skip_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            routes: [B, P, K]
            weights: [B, P, K]
            output: [B, S, D]
            info: metrics
        """
        if skip_first:
            cls_token = x[:, :1, :]
            x = x[:, 1:, :]
        else:
            cls_token = None

        B, P, D = x.shape
        device = x.device

        # Ensure fingerprint on device
        fingerprint = self.fingerprint.to(device)

        # Q/K computation (standard)
        q = F.normalize(self.query_proj(x), dim=-1)
        k = F.normalize(self.key_proj(x), dim=-1)

        # Content scores
        scores = torch.bmm(q, k.transpose(1, 2))  # [B, P, P]

        # Cantor sparsity mask - CONSTITUTIVE (can't route outside mask)
        cantor_affinity = self.cantor.get_affinity_matrix(fingerprint)[:P, :P]

        # Keep only top-k% of Cantor affinities
        k_sparse = max(1, int(P * self.cantor_sparsity))
        topk_vals, _ = cantor_affinity.topk(k_sparse, dim=-1)
        threshold = topk_vals[:, -1:].expand(-1, P)
        cantor_mask = cantor_affinity < threshold

        # Apply Cantor mask - positions outside mask are unreachable
        scores = scores.masked_fill(cantor_mask.unsqueeze(0), -1e9)

        # Mask self
        eye_mask = torch.eye(P, device=device, dtype=torch.bool)
        scores = scores.masked_fill(eye_mask.unsqueeze(0), -1e9)

        # Top-K selection
        topk_scores, routes = torch.topk(scores / self.temperature, self.num_routes, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        # CONSTITUTIVE: Fingerprint gates values
        v = self.value_proj(x)
        value_gate = self.fp_value_gate(fingerprint)  # [D]
        v = v * value_gate  # Fingerprint shapes what information flows

        # Gather and combine
        v_gathered = self._gather(v, routes)
        routed = torch.einsum('bpk,bpkd->bpd', weights, v_gathered)

        # Output
        output = x + self.out_proj(routed)

        if cls_token is not None:
            output = torch.cat([cls_token, output], dim=1)

        info = {
            'route_entropy': -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item(),
            'value_gate_mean': value_gate.mean().item(),
            'cantor_mask_density': (~cantor_mask).float().mean().item(),
        }

        return routes, weights, output, info

    @staticmethod
    def _gather(x: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        B, P, D = x.shape
        K = routes.shape[-1]
        routes_flat = routes.reshape(B, P * K).unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(x, 1, routes_flat).view(B, P, K, D)


# =============================================================================
# FULL CONSTITUTIVE ROUTER
# =============================================================================

class FullConstitutiveRouter(nn.Module):
    """
    Full version: Fingerprint shapes Q/K/V projections, anchors contribute to output.

    - Cantor behavioral identity defines projection basis
    - Fingerprint modulates ALL projections (truly constitutive)
    - Anchors transform features and contribute to output
    - Mailbox-ready with meaningful state
    """

    def __init__(
            self,
            dim: int,
            num_positions: int,
            grid_size: int,
            num_routes: int = 8,
            fingerprint_dim: int = 64,
            num_anchors: int = 16,
            temperature: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_positions = num_positions
        self.num_routes = min(num_routes, num_positions - 1)
        self.temperature = temperature
        self.fingerprint_dim = fingerprint_dim

        # Cantor behavioral system
        self.cantor = CantorBehavior(num_positions, grid_size, fingerprint_dim)

        # Base projections
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # CONSTITUTIVE: Fingerprint modulates projection weights
        # Instead of W*x, we do (W + fp_mod)*x where fp_mod depends on fingerprint
        self.fp_query_mod = nn.Linear(fingerprint_dim, dim * dim)
        self.fp_key_mod = nn.Linear(fingerprint_dim, dim * dim)
        self.fp_value_mod = nn.Linear(fingerprint_dim, dim * dim)

        # Scaling for modulation (keep it bounded)
        self.mod_scale = 0.1

        # Constitutive anchor bank
        self.anchor_bank = ConstitutiveAnchorBank(num_anchors, dim, fingerprint_dim)

        # Anchor contribution weight (learnable)
        self.anchor_weight = nn.Parameter(torch.tensor(0.3))

        # Learnable fingerprint
        self.fingerprint = nn.Parameter(torch.randn(fingerprint_dim) * 0.02)

        # Communication state encoder (for mailbox)
        self.state_encoder = nn.Linear(dim + fingerprint_dim + num_anchors, dim)

    def _modulated_projection(
            self,
            x: torch.Tensor,
            base_proj: nn.Linear,
            fp_mod: nn.Linear,
            fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        """Apply fingerprint-modulated projection."""
        B, P, D = x.shape

        # Base projection
        base_out = base_proj(x)

        # Fingerprint modulation matrix
        mod_weights = fp_mod(fingerprint).view(D, D) * self.mod_scale

        # Modulated projection: base + fingerprint-shaped component
        mod_out = torch.einsum('bpd,de->bpe', x, mod_weights)

        return base_out + mod_out

    def forward(
            self,
            x: torch.Tensor,
            skip_first: bool = True,
            return_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            routes: [B, P, K]
            weights: [B, P, K]
            output: [B, S, D]
            info: metrics and optional state
        """
        if skip_first:
            cls_token = x[:, :1, :]
            patches = x[:, 1:, :]
        else:
            cls_token = None
            patches = x

        B, P, D = patches.shape
        device = patches.device

        fingerprint = self.fingerprint.to(device)

        # CONSTITUTIVE: Fingerprint-modulated Q/K/V
        q = F.normalize(
            self._modulated_projection(patches, self.query_proj, self.fp_query_mod, fingerprint),
            dim=-1
        )
        k = F.normalize(
            self._modulated_projection(patches, self.key_proj, self.fp_key_mod, fingerprint),
            dim=-1
        )
        v = self._modulated_projection(patches, self.value_proj, self.fp_value_mod, fingerprint)

        # Cantor affinity
        cantor_affinity = self.cantor.get_affinity_matrix(fingerprint)[:P, :P]

        # Content + Cantor scores
        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores + 0.3 * cantor_affinity.unsqueeze(0)

        # Mask self
        eye_mask = torch.eye(P, device=device, dtype=torch.bool)
        scores = scores.masked_fill(eye_mask.unsqueeze(0), -1e9)

        # Top-K
        topk_scores, routes = torch.topk(scores / self.temperature, self.num_routes, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        # Gather and combine
        v_gathered = self._gather(v, routes)
        routed = torch.einsum('bpk,bpkd->bpd', weights, v_gathered)

        # CONSTITUTIVE: Anchor bank contribution
        pooled = patches.mean(dim=1)  # [B, D]
        anchor_out, anchor_affinities, anchor_preds = self.anchor_bank(pooled, fingerprint)

        # Combine routing output with anchor output
        anchor_contribution = self.anchor_weight.sigmoid()

        # Project routed features
        routed_out = self.out_proj(routed)

        # Add anchor contribution (expanded to match spatial dims)
        anchor_expanded = anchor_out.unsqueeze(1).expand(-1, P, -1)

        output = patches + routed_out + anchor_contribution * anchor_expanded

        if cls_token is not None:
            output = torch.cat([cls_token, output], dim=1)

        # Build info
        info = {
            'route_entropy': -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item(),
            'anchor_entropy': -(anchor_affinities * (anchor_affinities + 1e-8).log()).sum(dim=-1).mean().item(),
            'anchor_top': anchor_affinities.argmax(dim=-1).float().mean().item(),
            'anchor_weight': anchor_contribution.item(),
        }

        # Mailbox state (meaningful representation of router behavior)
        if return_state:
            state_input = torch.cat([
                pooled.mean(dim=0),  # [D]
                fingerprint,  # [fp_dim]
                anchor_affinities.mean(dim=0),  # [num_anchors]
            ])
            info['state'] = self.state_encoder(state_input)
            info['anchor_predictions'] = anchor_preds

        return routes, weights, output, info

    @staticmethod
    def _gather(x: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        B, P, D = x.shape
        K = routes.shape[-1]
        routes_flat = routes.reshape(B, P * K).unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(x, 1, routes_flat).view(B, P, K, D)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ConstitutiveRouterConfig:
    """Configuration for constitutive routers."""

    # Dimensions
    dim: int = 128
    num_positions: int = 49  # 7x7 for 28/4 patches
    grid_size: int = 7
    fingerprint_dim: int = 64

    # Routing
    num_routes: int = 4
    temperature: float = 0.1

    # Sparse-specific
    cantor_sparsity: float = 0.3

    # Full-specific
    num_anchors: int = 16

    # Mode
    mode: str = "full"  # "sparse" or "full"


def create_router(config: ConstitutiveRouterConfig) -> nn.Module:
    """Factory function for constitutive routers."""
    if config.mode == "sparse":
        return SparseConstitutiveRouter(
            dim=config.dim,
            num_positions=config.num_positions,
            grid_size=config.grid_size,
            num_routes=config.num_routes,
            fingerprint_dim=config.fingerprint_dim,
            temperature=config.temperature,
            cantor_sparsity=config.cantor_sparsity,
        )
    elif config.mode == "full":
        return FullConstitutiveRouter(
            dim=config.dim,
            num_positions=config.num_positions,
            grid_size=config.grid_size,
            num_routes=config.num_routes,
            fingerprint_dim=config.fingerprint_dim,
            num_anchors=config.num_anchors,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


# =============================================================================
# TEST
# =============================================================================

def test_routers():
    print("=" * 60)
    print("Testing Constitutive Routers")
    print("=" * 60)

    B, S, D = 2, 50, 128  # [batch, 1+patches, dim]
    x = torch.randn(B, S, D)

    # Test sparse
    print("\n[Sparse Router]")
    config = ConstitutiveRouterConfig(mode="sparse")
    router = create_router(config)
    routes, weights, output, info = router(x, skip_first=True)
    print(f"  Routes: {routes.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Route entropy: {info['route_entropy']:.4f}")
    print(f"  Value gate mean: {info['value_gate_mean']:.4f}")
    print(f"  Cantor mask density: {info['cantor_mask_density']:.4f}")
    print(f"  Parameters: {sum(p.numel() for p in router.parameters()):,}")

    # Test full
    print("\n[Full Router]")
    config = ConstitutiveRouterConfig(mode="full")
    router = create_router(config)
    routes, weights, output, info = router(x, skip_first=True, return_state=True)
    print(f"  Routes: {routes.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Route entropy: {info['route_entropy']:.4f}")
    print(f"  Anchor entropy: {info['anchor_entropy']:.4f}")
    print(f"  Anchor top: {info['anchor_top']:.4f}")
    print(f"  Anchor weight: {info['anchor_weight']:.4f}")
    print(f"  State shape: {info['state'].shape}")
    print(f"  Parameters: {sum(p.numel() for p in router.parameters()):,}")

    # Verify gradients flow through anchor path
    print("\n[Gradient Check]")
    output.sum().backward()
    anchor_grad = router.anchor_bank.anchor_transforms[0].weight.grad
    print(f"  Anchor transform grad norm: {anchor_grad.norm().item():.6f}")
    assert anchor_grad.norm() > 0, "Anchors should receive gradients!"
    print("  ✓ Anchors are constitutive (gradients flow)")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_routers()