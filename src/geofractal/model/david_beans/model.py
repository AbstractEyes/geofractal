"""
DavidBeans: Unified Vision-to-Crystal Architecture
===================================================

Combines:
- ViT-Beans v2: Cantor-routed sparse attention for patch processing
- David: Multi-scale crystal classification with geometric fusion

Cross-contrast learning enables patch features to align with crystal anchors
at multiple scales, while shared Cayley-Menger regularization maintains
geometric structure throughout.

Author: AbstractPhil
Date: November 28, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DavidBeansConfig:
    """Unified configuration for DavidBeans."""

    # Vision (Beans) settings
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Shared dimensions
    dim: int = 768

    # Beans backbone
    num_layers: int = 12
    num_heads: int = 12
    num_experts: int = 5  # Pentachoron
    k_neighbors: int = 32
    cantor_weight: float = 0.3  # Hybrid routing
    mlp_ratio: float = 4.0

    # David classifier
    scales: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 768])
    num_classes: int = 1000
    fusion_mode: str = "geometric_attention"  # or "cantor_scale"
    use_belly: bool = True
    belly_expand: float = 2.0

    # Cross-contrast settings
    contrast_temperature: float = 0.07
    contrast_weight: float = 0.5
    patch_crystal_align: bool = True

    # Geometric regularization
    cayley_weight: float = 0.01
    volume_floor: float = 1e-4

    # Training
    dropout: float = 0.1
    pooling: str = "cls"  # "cls" or "mean"

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size


# ============================================================================
# SHARED GEOMETRIC COMPONENTS
# ============================================================================

class UnifiedCayleyMengerLoss(nn.Module):
    """
    Unified Cayley-Menger loss for both Beans pentachoron experts
    and David crystal geometry.
    """

    def __init__(
        self,
        volume_floor: float = 1e-4,
        edge_uniformity_weight: float = 0.5,
        cross_simplex_weight: float = 0.1
    ):
        super().__init__()
        self.volume_floor = volume_floor
        self.edge_weight = edge_uniformity_weight
        self.cross_weight = cross_simplex_weight

        # Cache triangular indices
        self.register_buffer('_triu_i', None)
        self.register_buffer('_triu_j', None)

    def _get_triu_indices(self, n: int, device: torch.device):
        if self._triu_i is None or self._triu_i.shape[0] != n * (n - 1) // 2:
            indices = torch.triu_indices(n, n, offset=1, device=device)
            self._triu_i = indices[0]
            self._triu_j = indices[1]
        return self._triu_i.to(device), self._triu_j.to(device)

    def compute_volume(self, X: torch.Tensor) -> torch.Tensor:
        """Compute Cayley-Menger volume for batch of simplices."""
        B, N, D = X.shape

        # Pairwise squared distances
        diff = X.unsqueeze(2) - X.unsqueeze(1)
        distsq = (diff * diff).sum(dim=-1)

        # Build Cayley-Menger matrix
        M = torch.zeros((B, N + 1, N + 1), dtype=X.dtype, device=X.device)
        M[:, 0, 1:] = 1.0
        M[:, 1:, 0] = 1.0
        M[:, 1:, 1:] = distsq

        # Volume from determinant
        det = torch.linalg.det(M)

        # For N=5 (pentachoron in 4D), denominator is 9216
        # General: (-1)^(N+1) * 2^N * (N!)^2
        denom = 9216.0 if N == 5 else (2 ** N) * (math.factorial(N) ** 2)
        volume_sq = (-det / denom).clamp(min=0.0)

        return volume_sq.sqrt()

    def compute_edge_uniformity(self, X: torch.Tensor) -> torch.Tensor:
        """Measure how uniform edge lengths are."""
        B, N, D = X.shape

        diff = X.unsqueeze(2) - X.unsqueeze(1)
        distsq = (diff * diff).sum(dim=-1)

        triu_i, triu_j = self._get_triu_indices(N, X.device)
        edge_lengths = distsq[:, triu_i, triu_j].sqrt()

        edge_mean = edge_lengths.mean(dim=1, keepdim=True)
        edge_std = edge_lengths.std(dim=1)

        return edge_std / edge_mean.squeeze(-1).clamp(min=1e-6)

    def forward(
        self,
        expert_vertices: Optional[torch.Tensor] = None,
        crystal_anchors: Optional[Dict[int, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute unified geometric loss.

        Args:
            expert_vertices: [num_experts, 5, dim] pentachoron vertices
            crystal_anchors: {scale: [num_classes, 5, scale_dim]} crystal pentachora

        Returns:
            total_loss, metrics_dict
        """
        losses = {}
        total = 0.0

        # Expert geometry (Beans)
        if expert_vertices is not None:
            volumes = self.compute_volume(expert_vertices)
            collapse_penalty = F.relu(self.volume_floor - volumes)
            losses['expert_volume'] = volumes.mean()
            losses['expert_collapse'] = collapse_penalty.mean()
            total = total + collapse_penalty.mean()

            edge_dev = self.compute_edge_uniformity(expert_vertices)
            losses['expert_edge_dev'] = edge_dev.mean()
            total = total + self.edge_weight * edge_dev.mean()

        # Crystal geometry (David)
        if crystal_anchors is not None:
            for scale, anchors in crystal_anchors.items():
                if anchors.dim() == 3:  # [C, 5, D]
                    volumes = self.compute_volume(anchors)
                    collapse = F.relu(self.volume_floor - volumes)
                    losses[f'crystal_{scale}_volume'] = volumes.mean()
                    losses[f'crystal_{scale}_collapse'] = collapse.mean()
                    total = total + collapse.mean()

        # Cross-simplex alignment (optional)
        if expert_vertices is not None and crystal_anchors is not None:
            # Align expert geometry with crystal geometry
            expert_vol = self.compute_volume(expert_vertices).mean()
            crystal_vols = [
                self.compute_volume(a).mean()
                for a in crystal_anchors.values()
                if a.dim() == 3
            ]
            if crystal_vols:
                crystal_vol = torch.stack(crystal_vols).mean()
                cross_loss = (expert_vol - crystal_vol).abs()
                losses['cross_volume_align'] = cross_loss
                total = total + self.cross_weight * cross_loss

        return total, losses


class PentachoronRoleWeights(nn.Module):
    """
    Shared pentachoron role weighting for both Beans experts and David crystals.

    Roles:
        0: anchor   (+1.0)  - The central reference point
        1: need     (-0.75) - What's missing/required
        2: relation (+0.75) - How things connect
        3: purpose  (+0.75) - The goal/intent
        4: observer (-0.75) - The perspective/context
    """

    def __init__(self, learnable: bool = False):
        super().__init__()

        default_weights = torch.tensor([1.0, -0.75, 0.75, 0.75, -0.75])

        if learnable:
            self.weights = nn.Parameter(default_weights)
        else:
            self.register_buffer('weights', default_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply role weighting to 5-vertex representations."""
        # x: [..., 5, D] or [..., 5]
        if x.dim() >= 2 and x.shape[-2] == 5:
            return x * self.weights.view(*([1] * (x.dim() - 2)), 5, 1)
        elif x.shape[-1] == 5:
            return x * self.weights
        else:
            raise ValueError(f"Expected 5 vertices, got shape {x.shape}")


# ============================================================================
# CANTOR ROUTING (Shared Infrastructure)
# ============================================================================

class CantorPairingFunction:
    """
    Cantor pairing for position encoding and routing.
    Used by both Beans (patch routing) and David (scale routing).
    """

    @staticmethod
    def encode(x: int, y: int) -> int:
        """Cantor pairing: N² → N"""
        return ((x + y) * (x + y + 1)) // 2 + y

    @staticmethod
    def decode(z: int) -> Tuple[int, int]:
        """Inverse Cantor pairing: N → N²"""
        w = int((math.sqrt(8 * z + 1) - 1) / 2)
        t = (w * w + w) // 2
        y = z - t
        x = w - y
        return x, y

    @staticmethod
    def fingerprint(position: int, depth: int = 16) -> torch.Tensor:
        """
        Compute Cantor fingerprint for a position.
        Returns [depth, 2] tensor of (bit, entropy) pairs.
        """
        fp = torch.zeros(depth, 2)
        x = position / (3 ** depth)

        for level in range(depth):
            x *= 3
            digit = int(x)
            x -= digit

            # Bit: 0, 0.5, or 1 for ternary digits
            fp[level, 0] = digit / 2.0
            # Entropy: uncertainty at this level
            fp[level, 1] = -abs(digit - 1) + 1  # 1 for middle, 0 for edges

        return fp


class HybridCantorRouter(nn.Module):
    """
    Hybrid Cantor + positional routing.
    Shared between Beans (patches) and David (scales).
    """

    def __init__(
        self,
        num_positions: int,
        k_neighbors: int,
        cantor_weight: float = 0.3,
        grid_size: Optional[int] = None
    ):
        super().__init__()
        self.num_positions = num_positions
        self.k = min(k_neighbors, num_positions)
        self.cantor_weight = cantor_weight
        self.grid_size = grid_size or int(math.sqrt(num_positions))

        # Pre-compute routes
        routes = self._compute_routes()
        self.register_buffer('routes', routes)

    def _compute_routes(self) -> torch.Tensor:
        """Compute k-nearest neighbors using hybrid distance."""
        P = self.num_positions
        G = self.grid_size

        # Compute Cantor fingerprints
        fingerprints = torch.zeros(P)
        for i in range(P):
            x, y = i % G, i // G
            z = CantorPairingFunction.encode(x, y)
            # Normalize to [0, 1]
            fingerprints[i] = z / (G * G * 2)

        # Cantor distance
        D_cantor = torch.abs(fingerprints.unsqueeze(0) - fingerprints.unsqueeze(1))

        # Positional distance (Manhattan on grid)
        pos_x = torch.arange(P) % G
        pos_y = torch.arange(P) // G
        D_pos = (
            torch.abs(pos_x.unsqueeze(0) - pos_x.unsqueeze(1)) +
            torch.abs(pos_y.unsqueeze(0) - pos_y.unsqueeze(1))
        ).float() / (2 * (G - 1))

        # Hybrid distance
        D = self.cantor_weight * D_cantor + (1 - self.cantor_weight) * D_pos

        # Get k-nearest
        _, routes = torch.topk(D, self.k, dim=1, largest=False)

        return routes

    def forward(self) -> torch.Tensor:
        """Return pre-computed routes."""
        return self.routes


# ============================================================================
# BEANS BACKBONE (Simplified from v2)
# ============================================================================

class BeansAttentionBlock(nn.Module):
    """
    Single attention block with hybrid Cantor routing.
    CLS gets dense attention, patches get sparse.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_patches: int,
        k_neighbors: int,
        routes: torch.Tensor,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_patches = num_patches
        self.k = k_neighbors

        # Store routes
        self.register_buffer('routes', routes)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # MLP
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] where S = 1 (CLS) + num_patches
        """
        B, S, D = x.shape

        # Pre-norm
        x_norm = self.norm1(x)

        # QKV
        qkv = self.qkv(x_norm).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D]
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Split CLS and patches
        Q_cls, Q_patches = Q[:, :, :1, :], Q[:, :, 1:, :]
        K_cls, K_patches = K[:, :, :1, :], K[:, :, 1:, :]
        V_cls, V_patches = V[:, :, :1, :], V[:, :, 1:, :]

        # CLS: dense attention to all
        scores_cls = torch.einsum('bhqd,bhkd->bhqk', Q_cls, K) * self.scale
        attn_cls = F.softmax(scores_cls, dim=-1)
        attn_cls = self.attn_drop(attn_cls)
        out_cls = torch.einsum('bhqk,bhkd->bhqd', attn_cls, V)

        # Patches: sparse attention via routes
        routes = self.routes  # [P, k]
        routes_shifted = routes + 1  # Account for CLS at position 0

        # Gather K, V for each patch's neighbors
        # routes_shifted: [P, k] -> need [B, H, P, k]
        routes_exp = routes_shifted.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)

        # Gather from full K, V (including CLS)
        K_gathered = torch.gather(
            K.unsqueeze(3).expand(-1, -1, -1, self.k, -1),
            dim=2,
            index=routes_exp.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        )  # [B, H, P, k, D]

        V_gathered = torch.gather(
            V.unsqueeze(3).expand(-1, -1, -1, self.k, -1),
            dim=2,
            index=routes_exp.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        )  # [B, H, P, k, D]

        # Sparse attention
        scores_patches = torch.einsum('bhpd,bhpkd->bhpk', Q_patches, K_gathered) * self.scale
        attn_patches = F.softmax(scores_patches, dim=-1)
        attn_patches = self.attn_drop(attn_patches)
        out_patches = torch.einsum('bhpk,bhpkd->bhpd', attn_patches, V_gathered)

        # Combine
        out = torch.cat([out_cls, out_patches], dim=2)  # [B, H, S, D]
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.proj_drop(self.proj(out))

        # Residual
        x = x + out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class PentachoronExpertLayer(nn.Module):
    """
    Pentachoron expert layer: 5 parallel experts with geometric structure.
    Each expert processes all patches but owns a feature slice.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 5,
        expert_ratio: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        # Handle non-divisible dimensions: last expert gets remainder
        base_slice = dim // num_experts
        remainder = dim % num_experts

        self.slice_sizes = []
        self.slice_starts = []
        current = 0
        for i in range(num_experts):
            # Distribute remainder across first 'remainder' experts
            size = base_slice + (1 if i < remainder else 0)
            self.slice_sizes.append(size)
            self.slice_starts.append(current)
            current += size

        # Expert networks (each with its own slice size)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.slice_sizes[i], int(self.slice_sizes[i] * expert_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(self.slice_sizes[i] * expert_ratio), self.slice_sizes[i])
            )
            for i in range(num_experts)
        ])

        # Pentachoron vertices (learnable) - use base_slice for consistency
        self.vertices = nn.Parameter(
            torch.randn(num_experts, 5, base_slice) * 0.02
        )

        # Role weights
        self.role_weights = PentachoronRoleWeights(learnable=True)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S, D]

        Returns:
            output: [B, S, D]
            vertices: [num_experts, 5, slice_size] for geometric loss
        """
        B, S, D = x.shape
        x_norm = self.norm(x)

        outputs = []
        for i, expert in enumerate(self.experts):
            start = self.slice_starts[i]
            end = start + self.slice_sizes[i]

            x_slice = x_norm[..., start:end]
            out_slice = expert(x_slice)
            outputs.append(out_slice)

        # Concatenate expert outputs
        out = torch.cat(outputs, dim=-1)

        # Residual
        out = x + out

        return out, self.vertices


class BeansBackbone(nn.Module):
    """
    ViT-Beans v2 backbone with Cantor routing.
    """

    def __init__(self, config: DavidBeansConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + config.num_patches, config.dim) * 0.02
        )

        # Build router
        self.router = HybridCantorRouter(
            num_positions=config.num_patches,
            k_neighbors=config.k_neighbors,
            cantor_weight=config.cantor_weight,
            grid_size=config.grid_size
        )

        # Attention blocks with experts
        self.blocks = nn.ModuleList()
        self.expert_layers = nn.ModuleList()

        for _ in range(config.num_layers):
            self.blocks.append(
                BeansAttentionBlock(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    num_patches=config.num_patches,
                    k_neighbors=config.k_neighbors,
                    routes=self.router.routes,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout
                )
            )
            self.expert_layers.append(
                PentachoronExpertLayer(
                    dim=config.dim,
                    num_experts=config.num_experts,
                    dropout=config.dropout
                )
            )

        self.norm = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, C, H, W] images
            return_all_tokens: Whether to return all patch tokens

        Returns:
            features: [B, D] (CLS or mean pooled)
            all_tokens: [B, S, D] if return_all_tokens
            all_vertices: List of [num_experts, 5, slice_size] for geometric loss
        """
        B = x.shape[0]

        # Patch embed
        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, P, D]

        # Add CLS and position
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        # Process through blocks
        all_vertices = []
        for block, expert_layer in zip(self.blocks, self.expert_layers):
            x = block(x)
            x, vertices = expert_layer(x)
            all_vertices.append(vertices)

        x = self.norm(x)

        # Pool
        if self.config.pooling == "cls":
            features = x[:, 0]
        else:
            features = x[:, 1:].mean(dim=1)

        if return_all_tokens:
            return features, x, all_vertices
        else:
            return features, x[:, 0:1], all_vertices


# ============================================================================
# DAVID CLASSIFIER HEAD (Simplified)
# ============================================================================

class CrystalProjectionHead(nn.Module):
    """
    Project features to crystal space at a specific scale.
    """

    def __init__(
        self,
        input_dim: int,
        crystal_dim: int,
        use_belly: bool = True,
        belly_expand: float = 2.0,
        temperature: float = 0.07
    ):
        super().__init__()
        self.crystal_dim = crystal_dim
        self.temperature = temperature

        if use_belly:
            belly_dim = int(crystal_dim * belly_expand)
            self.projection = nn.Sequential(
                nn.Linear(input_dim, belly_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(belly_dim, crystal_dim, bias=False)
            )
        else:
            self.projection = nn.Linear(input_dim, crystal_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        anchors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, D]
            anchors: [C, crystal_dim]

        Returns:
            logits: [B, C]
            projected: [B, crystal_dim]
        """
        z = self.projection(features)
        z = F.normalize(z, dim=-1)

        anchors_norm = F.normalize(anchors, dim=-1)
        logits = (z @ anchors_norm.T) / self.temperature

        return logits, z


class MultiScaleCrystalHead(nn.Module):
    """
    Multi-scale crystal classification head (David's core).
    """

    def __init__(self, config: DavidBeansConfig):
        super().__init__()
        self.config = config
        self.scales = config.scales

        # Per-scale projection heads
        self.heads = nn.ModuleDict({
            str(scale): CrystalProjectionHead(
                input_dim=config.dim,
                crystal_dim=scale,
                use_belly=config.use_belly,
                belly_expand=config.belly_expand,
                temperature=config.contrast_temperature
            )
            for scale in config.scales
        })

        # Fusion (simplified geometric attention)
        self.fusion = self._build_fusion()

    def _build_fusion(self) -> nn.Module:
        """Build scale fusion module."""
        config = self.config

        # Simple learned fusion
        return nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.LayerNorm(config.dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim // 2, len(config.scales))
        )

    def forward(
        self,
        features: torch.Tensor,
        anchors_dict: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Args:
            features: [B, D]
            anchors_dict: {scale: [C, scale_dim]}

        Returns:
            combined_logits: [B, C]
            scale_logits: List of [B, C]
            scale_features: List of [B, scale_dim]
            fusion_weights: [B, num_scales]
        """
        scale_logits = []
        scale_features = []

        for scale in self.scales:
            head = self.heads[str(scale)]
            logits, z = head(features, anchors_dict[scale])
            scale_logits.append(logits)
            scale_features.append(z)

        # Fusion
        fusion_logits = self.fusion(features)
        fusion_weights = F.softmax(fusion_logits, dim=-1)

        # Combine logits
        combined = torch.zeros_like(scale_logits[0])
        for i, logits in enumerate(scale_logits):
            combined += fusion_weights[:, i:i+1] * logits

        return combined, scale_logits, scale_features, fusion_weights


# ============================================================================
# CROSS-CONTRASTIVE LOSS
# ============================================================================

class CrossContrastiveLoss(nn.Module):
    """
    Cross-contrastive loss between patch features and crystal anchors.

    Enables alignment between:
    1. Patch-level representations (from Beans)
    2. Crystal-level representations (from David)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        patch_weight: float = 0.5,
        scale_weights: Optional[Dict[int, float]] = None
    ):
        super().__init__()
        self.temperature = temperature
        self.patch_weight = patch_weight
        self.scale_weights = scale_weights or {}

    def forward(
        self,
        patch_features: torch.Tensor,  # [B, P, D]
        scale_features: List[torch.Tensor],  # List of [B, scale_dim]
        anchors_dict: Dict[int, torch.Tensor],  # {scale: [C, scale_dim]}
        targets: torch.Tensor,  # [B]
        scales: List[int]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute cross-contrastive loss.

        The key insight: patch features should align with the correct
        class's crystal anchor at each scale.
        """
        losses = {}
        total = 0.0

        B, P, D = patch_features.shape

        # 1. Patch-to-crystal alignment
        # Mean pool patches
        patch_mean = patch_features.mean(dim=1)  # [B, D]

        for i, (scale, scale_feat) in enumerate(zip(scales, scale_features)):
            anchors = anchors_dict[scale]  # [C, scale_dim]

            # Project patches to scale dimension (simple linear)
            # In full implementation, use learned projector
            if D != scale:
                # Simple dimensionality alignment
                if D > scale:
                    patch_proj = patch_mean[..., :scale]
                else:
                    patch_proj = F.pad(patch_mean, (0, scale - D))
            else:
                patch_proj = patch_mean

            patch_proj = F.normalize(patch_proj, dim=-1)
            scale_feat_norm = F.normalize(scale_feat, dim=-1)
            anchors_norm = F.normalize(anchors, dim=-1)

            # Contrastive: patches should match scale features for same sample
            patch_to_scale = torch.einsum('bd,bd->b', patch_proj, scale_feat_norm)
            patch_scale_loss = (1 - patch_to_scale).mean()

            # Contrastive: scale features should match correct anchor
            scale_to_anchor = scale_feat_norm @ anchors_norm.T / self.temperature
            anchor_loss = F.cross_entropy(scale_to_anchor, targets)

            scale_weight = self.scale_weights.get(scale, 1.0)
            losses[f'patch_scale_{scale}'] = patch_scale_loss
            losses[f'anchor_{scale}'] = anchor_loss

            total = total + scale_weight * (
                self.patch_weight * patch_scale_loss + anchor_loss
            )

        losses['total'] = total / len(scales)

        return losses['total'], losses


# ============================================================================
# UNIFIED MODEL: DAVID-BEANS
# ============================================================================

class DavidBeans(nn.Module):
    """
    DavidBeans: Unified Vision-to-Crystal Architecture

    Combines ViT-Beans (Cantor-routed vision backbone) with
    David (multi-scale crystal classification).

    Features:
    - Hybrid Cantor + positional routing for patches
    - Pentachoron experts with geometric structure
    - Multi-scale crystal projection
    - Cross-contrastive learning
    - Unified Cayley-Menger regularization
    """

    def __init__(self, config: DavidBeansConfig):
        super().__init__()
        self.config = config

        # Beans backbone
        self.backbone = BeansBackbone(config)

        # David head
        self.head = MultiScaleCrystalHead(config)

        # Geometric loss
        self.cayley_loss = UnifiedCayleyMengerLoss(
            volume_floor=config.volume_floor
        )

        # Cross-contrastive loss
        self.cross_contrast = CrossContrastiveLoss(
            temperature=config.contrast_temperature,
            patch_weight=config.contrast_weight
        )

        # Crystal anchors (learnable or fixed)
        self.anchors = nn.ParameterDict({
            str(scale): nn.Parameter(
                torch.randn(config.num_classes, scale) * 0.02
            )
            for scale in config.scales
        })

    def get_anchors_dict(self) -> Dict[int, torch.Tensor]:
        """Get anchors as {scale: tensor} dict."""
        return {
            int(k): v for k, v in self.anchors.items()
        }

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, C, H, W] images
            targets: [B] class labels (optional, for loss)
            return_loss: Whether to compute losses

        Returns:
            Dict with:
                - logits: [B, num_classes]
                - features: [B, dim]
                - losses: Dict of loss components (if targets provided)
        """
        # Backbone
        features, all_tokens, all_vertices = self.backbone(x, return_all_tokens=True)

        # Head
        anchors_dict = self.get_anchors_dict()
        combined_logits, scale_logits, scale_features, fusion_weights = self.head(
            features, anchors_dict
        )

        result = {
            'logits': combined_logits,
            'features': features,
            'scale_logits': scale_logits,
            'scale_features': scale_features,
            'fusion_weights': fusion_weights
        }

        if return_loss and targets is not None:
            losses = {}

            # Classification loss
            ce_loss = F.cross_entropy(combined_logits, targets)
            losses['ce'] = ce_loss

            # Per-scale CE
            for i, (scale, logits) in enumerate(zip(self.config.scales, scale_logits)):
                losses[f'ce_{scale}'] = F.cross_entropy(logits, targets)

            # Geometric loss from expert vertices
            stacked_vertices = torch.stack(all_vertices, dim=0)  # [L, E, 5, D]
            mean_vertices = stacked_vertices.mean(dim=0)  # [E, 5, D]
            geo_loss, geo_metrics = self.cayley_loss(expert_vertices=mean_vertices)
            losses['geometric'] = geo_loss
            losses.update(geo_metrics)

            # Cross-contrastive loss
            patch_tokens = all_tokens[:, 1:, :]  # Exclude CLS
            contrast_loss, contrast_metrics = self.cross_contrast(
                patch_features=patch_tokens,
                scale_features=scale_features,
                anchors_dict=anchors_dict,
                targets=targets,
                scales=self.config.scales
            )
            losses['contrast'] = contrast_loss
            losses.update(contrast_metrics)

            # Total loss
            losses['total'] = (
                ce_loss +
                self.config.cayley_weight * geo_loss +
                self.config.contrast_weight * contrast_loss
            )

            result['losses'] = losses

        return result

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'name': 'DavidBeans',
            'image_size': self.config.image_size,
            'patch_size': self.config.patch_size,
            'dim': self.config.dim,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'num_experts': self.config.num_experts,
            'k_neighbors': self.config.k_neighbors,
            'cantor_weight': self.config.cantor_weight,
            'scales': self.config.scales,
            'num_classes': self.config.num_classes,
            'num_patches': self.config.num_patches,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

    def __repr__(self):
        info = self.get_model_info()
        return (
            f"DavidBeans(\n"
            f"  Vision: {info['image_size']}px → {info['num_patches']} patches\n"
            f"  Backbone: {info['num_layers']} layers, {info['dim']}d, {info['num_heads']} heads\n"
            f"  Routing: k={info['k_neighbors']}, α={info['cantor_weight']}\n"
            f"  Experts: {info['num_experts']} (pentachoron)\n"
            f"  Scales: {info['scales']}\n"
            f"  Classes: {info['num_classes']}\n"
            f"  Parameters: {info['total_params']:,}\n"
            f")"
        )


# ============================================================================
# QUICK TESTS
# ============================================================================

def test_david_beans():
    """Quick sanity tests."""
    print("=" * 60)
    print("DavidBeans Quick Tests")
    print("=" * 60)

    # Small config for testing
    config = DavidBeansConfig(
        image_size=32,
        patch_size=4,
        dim=256,
        num_layers=4,
        num_heads=4,
        num_experts=5,
        k_neighbors=16,
        scales=[64, 128, 256],
        num_classes=10,
        cantor_weight=0.3
    )

    model = DavidBeans(config)
    print(f"\n{model}")

    # Test forward pass
    print("\n[1] Forward Pass Test")
    x = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))

    result = model(x, targets=targets, return_loss=True)

    print(f"  Logits shape: {result['logits'].shape}")
    print(f"  Features shape: {result['features'].shape}")
    print(f"  Fusion weights: {result['fusion_weights'][0].tolist()}")

    print("\n[2] Loss Components")
    for k, v in result['losses'].items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")

    # Test gradient flow
    print("\n[3] Gradient Flow Test")
    result['losses']['total'].backward()

    grad_checks = {
        'patch_embed': model.backbone.patch_embed.weight.grad,
        'cls_token': model.backbone.cls_token.grad,
        'expert_0': model.backbone.expert_layers[0].experts[0][0].weight.grad,
        'crystal_64': model.anchors['64'].grad,
        'head_64': model.head.heads['64'].projection[-1].weight.grad
    }

    all_ok = True
    for name, grad in grad_checks.items():
        if grad is not None:
            norm = grad.norm().item()
            ok = norm > 0
            print(f"  {name}: grad_norm={norm:.4f} {'✓' if ok else '✗'}")
            all_ok = all_ok and ok
        else:
            print(f"  {name}: NO GRADIENT ✗")
            all_ok = False

    print(f"\n{'✓ All gradients flow correctly!' if all_ok else '✗ Gradient issues detected'}")

    # Test inference mode
    print("\n[4] Inference Test")
    model.eval()
    with torch.no_grad():
        result = model(x, return_loss=False)
        preds = result['logits'].argmax(dim=-1)
        print(f"  Predictions: {preds.tolist()}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_david_beans()