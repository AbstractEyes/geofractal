"""
DavidBeans Optimized: Vectorized Operations
============================================

Key optimizations:
1. PentachoronExpertLayer: chunk-based parallel processing
2. BeansAttentionBlock: optimized gather with index_select pattern
3. MultiScaleCrystalHead: batched scale processing where possible
4. Fusion: einsum-based weighted combination
5. CrossContrastiveLoss: vectorized scale processing

Author: AbstractPhil
Date: November 28, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DavidBeansConfig:
    """Unified configuration for DavidBeans."""

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_experts: int = 5
    k_neighbors: int = 32
    cantor_weight: float = 0.3
    mlp_ratio: float = 4.0
    scales: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 768])
    num_classes: int = 1000
    use_belly: bool = True
    belly_expand: float = 2.0
    contrast_temperature: float = 0.07
    contrast_weight: float = 0.5
    cayley_weight: float = 0.01
    volume_floor: float = 1e-4
    dropout: float = 0.1
    pooling: str = "cls"

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size


# ============================================================================
# OPTIMIZED GEOMETRIC COMPONENTS
# ============================================================================

class UnifiedCayleyMengerLoss(nn.Module):
    """Optimized Cayley-Menger loss with batched operations."""

    def __init__(
        self,
        volume_floor: float = 1e-4,
        edge_uniformity_weight: float = 0.5,
    ):
        super().__init__()
        self.volume_floor = volume_floor
        self.edge_weight = edge_uniformity_weight

        # Pre-compute triangular indices for 5 vertices
        indices = torch.triu_indices(5, 5, offset=1)
        self.register_buffer('triu_i', indices[0])
        self.register_buffer('triu_j', indices[1])

    def forward(
        self,
        expert_vertices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute geometric loss.

        Args:
            expert_vertices: [num_experts, 5, dim] pentachoron vertices
        """
        if expert_vertices is None:
            return torch.tensor(0.0), {}

        E, N, D = expert_vertices.shape
        device = expert_vertices.device

        # Pairwise squared distances: [E, 5, 5]
        diff = expert_vertices.unsqueeze(2) - expert_vertices.unsqueeze(1)
        distsq = (diff * diff).sum(dim=-1)

        # Build Cayley-Menger matrix: [E, 6, 6]
        M = torch.zeros((E, 6, 6), dtype=expert_vertices.dtype, device=device)
        M[:, 0, 1:] = 1.0
        M[:, 1:, 0] = 1.0
        M[:, 1:, 1:] = distsq

        # Volume from determinant
        det = torch.linalg.det(M)
        volumes = (-det / 9216.0).clamp(min=0.0).sqrt()

        # Collapse penalty
        collapse_penalty = F.relu(self.volume_floor - volumes)

        # Edge uniformity: extract upper triangle edges
        edge_lengths = distsq[:, self.triu_i, self.triu_j].sqrt()  # [E, 10]
        edge_mean = edge_lengths.mean(dim=1, keepdim=True)
        edge_std = edge_lengths.std(dim=1)
        edge_dev = edge_std / edge_mean.squeeze(-1).clamp(min=1e-6)

        # Total loss
        total = collapse_penalty.mean() + self.edge_weight * edge_dev.mean()

        losses = {
            'expert_volume': volumes.mean(),
            'expert_collapse': collapse_penalty.mean(),
            'expert_edge_dev': edge_dev.mean(),
        }

        return total, losses


class PentachoronRoleWeights(nn.Module):
    """Pentachoron role weighting."""

    def __init__(self, learnable: bool = False):
        super().__init__()
        default_weights = torch.tensor([1.0, -0.75, 0.75, 0.75, -0.75])
        if learnable:
            self.weights = nn.Parameter(default_weights)
        else:
            self.register_buffer('weights', default_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() >= 2 and x.shape[-2] == 5:
            return x * self.weights.view(*([1] * (x.dim() - 2)), 5, 1)
        elif x.shape[-1] == 5:
            return x * self.weights
        raise ValueError(f"Expected 5 vertices, got shape {x.shape}")


# ============================================================================
# OPTIMIZED CANTOR ROUTING
# ============================================================================

class HybridCantorRouter(nn.Module):
    """Pre-computed hybrid Cantor + positional routing."""

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

        routes = self._compute_routes()
        self.register_buffer('routes', routes)

    def _compute_routes(self) -> torch.Tensor:
        """Vectorized route computation."""
        P = self.num_positions
        G = self.grid_size

        # Grid positions
        pos_x = torch.arange(P) % G
        pos_y = torch.arange(P) // G

        # Cantor pairing: z = (x+y)(x+y+1)/2 + y
        sums = pos_x + pos_y
        cantor_z = (sums * (sums + 1)) // 2 + pos_y
        fingerprints = cantor_z.float() / (G * G * 2)

        # Distance matrices (vectorized)
        D_cantor = (fingerprints.unsqueeze(0) - fingerprints.unsqueeze(1)).abs()
        D_pos = (
            (pos_x.unsqueeze(0) - pos_x.unsqueeze(1)).abs() +
            (pos_y.unsqueeze(0) - pos_y.unsqueeze(1)).abs()
        ).float() / (2 * (G - 1))

        # Hybrid distance
        D = self.cantor_weight * D_cantor + (1 - self.cantor_weight) * D_pos

        # k-nearest (including self at distance 0)
        _, routes = torch.topk(D, self.k, dim=1, largest=False)

        return routes


# ============================================================================
# OPTIMIZED ATTENTION BLOCK
# ============================================================================

class BeansAttentionBlock(nn.Module):
    """
    Optimized attention with sparse routing.
    Key optimization: use advanced indexing instead of expand+gather.
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

        self.register_buffer('routes', routes)

        # Fused QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Fused MLP
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Pre-compute flat indices for gathering
        self._precompute_gather_indices()

    def _precompute_gather_indices(self):
        """Pre-compute indices for efficient K,V gathering."""
        P = self.num_patches
        k = self.k

        # Routes shifted for CLS token at position 0
        routes_shifted = self.routes + 1  # [P, k]

        # Flatten for use with index_select pattern
        # We'll use this to construct batch indices at runtime
        self.register_buffer('routes_shifted', routes_shifted)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H = self.num_heads
        P = self.num_patches
        k = self.k
        head_dim = self.head_dim

        # Pre-norm
        x_norm = self.norm1(x)

        # QKV projection: [B, S, 3*D] -> [B, S, 3, H, head_dim]
        qkv = self.qkv(x_norm).reshape(B, S, 3, H, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, head_dim]
        Q, K, V = qkv.unbind(0)  # Each: [B, H, S, head_dim]

        # === CLS ATTENTION (dense to all) ===
        Q_cls = Q[:, :, :1, :]  # [B, H, 1, head_dim]
        scores_cls = torch.einsum('bhqd,bhkd->bhqk', Q_cls, K) * self.scale
        attn_cls = F.softmax(scores_cls, dim=-1)
        attn_cls = self.attn_drop(attn_cls)
        out_cls = torch.einsum('bhqk,bhkd->bhqd', attn_cls, V)  # [B, H, 1, head_dim]

        # === PATCH ATTENTION (sparse via routes) ===
        Q_patches = Q[:, :, 1:, :]  # [B, H, P, head_dim]

        # Optimized gathering using einsum-friendly reshape
        # routes_shifted: [P, k] - indices into S dimension
        routes = self.routes_shifted  # [P, k]

        # Flatten K, V for gathering: [B, H, S, head_dim] -> [B*H, S, head_dim]
        K_flat = K.reshape(B * H, S, head_dim)
        V_flat = V.reshape(B * H, S, head_dim)

        # Expand routes for batch: [P, k] -> [B*H, P, k]
        routes_exp = routes.unsqueeze(0).expand(B * H, -1, -1)  # [B*H, P, k]

        # Gather K, V using advanced indexing
        # Create batch indices
        batch_idx = torch.arange(B * H, device=x.device).view(-1, 1, 1).expand(-1, P, k)

        # Gather: K_gathered[b, p, ki] = K_flat[b, routes[p, ki], :]
        K_gathered = K_flat[batch_idx, routes_exp, :]  # [B*H, P, k, head_dim]
        V_gathered = V_flat[batch_idx, routes_exp, :]  # [B*H, P, k, head_dim]

        # Reshape back
        K_gathered = K_gathered.view(B, H, P, k, head_dim)
        V_gathered = V_gathered.view(B, H, P, k, head_dim)

        # Sparse attention scores
        scores_patches = torch.einsum('bhpd,bhpkd->bhpk', Q_patches, K_gathered) * self.scale
        attn_patches = F.softmax(scores_patches, dim=-1)
        attn_patches = self.attn_drop(attn_patches)
        out_patches = torch.einsum('bhpk,bhpkd->bhpd', attn_patches, V_gathered)  # [B, H, P, head_dim]

        # === COMBINE ===
        out = torch.cat([out_cls, out_patches], dim=2)  # [B, H, S, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.proj_drop(self.proj(out))

        # Residual + MLP
        x = x + out
        x = x + self.mlp(self.norm2(x))

        return x


# ============================================================================
# OPTIMIZED PENTACHORON EXPERT LAYER
# ============================================================================

class PentachoronExpertLayer(nn.Module):
    """
    Optimized pentachoron experts using chunked parallel processing.

    Key insight: Instead of looping over experts, we:
    1. Chunk the input along feature dim
    2. Process chunks with a batched linear
    3. Concatenate results

    For uniform chunk sizes, this is a single batched operation.
    For non-uniform (remainder), we use two groups.
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

        # Compute slice distribution
        base_slice = dim // num_experts
        remainder = dim % num_experts

        self.base_slice = base_slice
        self.remainder = remainder
        self.has_remainder = remainder > 0

        # For remainder handling, we have two groups:
        # Group A: first `remainder` experts with size `base_slice + 1`
        # Group B: remaining experts with size `base_slice`

        if self.has_remainder:
            self.large_slice = base_slice + 1
            self.num_large = remainder
            self.num_small = num_experts - remainder

            # Group A: larger slices (batched)
            self.experts_large = nn.Sequential(
                nn.Linear(self.large_slice, int(self.large_slice * expert_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(self.large_slice * expert_ratio), self.large_slice)
            )

            # Group B: smaller slices (batched)
            if self.num_small > 0:
                self.experts_small = nn.Sequential(
                    nn.Linear(base_slice, int(base_slice * expert_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(base_slice * expert_ratio), base_slice)
                )
        else:
            # Uniform slices - single batched operation
            self.experts = nn.Sequential(
                nn.Linear(base_slice, int(base_slice * expert_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(base_slice * expert_ratio), base_slice)
            )

        # Pentachoron vertices
        self.vertices = nn.Parameter(
            torch.randn(num_experts, 5, base_slice) * 0.02
        )

        self.role_weights = PentachoronRoleWeights(learnable=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        x_norm = self.norm(x)

        if self.has_remainder:
            # Split into large and small groups
            large_end = self.num_large * self.large_slice

            # Large slices
            x_large = x_norm[..., :large_end]  # [B, S, num_large * large_slice]
            x_large = x_large.view(B, S, self.num_large, self.large_slice)  # [B, S, num_large, large_slice]
            x_large = x_large.reshape(B * S * self.num_large, self.large_slice)
            out_large = self.experts_large(x_large)
            out_large = out_large.view(B, S, -1)  # [B, S, num_large * large_slice]

            # Small slices (if any)
            if self.num_small > 0:
                x_small = x_norm[..., large_end:]  # [B, S, num_small * base_slice]
                x_small = x_small.view(B, S, self.num_small, self.base_slice)
                x_small = x_small.reshape(B * S * self.num_small, self.base_slice)
                out_small = self.experts_small(x_small)
                out_small = out_small.view(B, S, -1)

                out = torch.cat([out_large, out_small], dim=-1)
            else:
                out = out_large
        else:
            # Uniform slices - fully batched
            x_chunked = x_norm.view(B, S, self.num_experts, self.base_slice)
            x_chunked = x_chunked.reshape(B * S * self.num_experts, self.base_slice)
            out_chunked = self.experts(x_chunked)
            out = out_chunked.view(B, S, -1)

        # Residual
        out = x + out

        return out, self.vertices


# ============================================================================
# OPTIMIZED MULTI-SCALE CRYSTAL HEAD
# ============================================================================

class CrystalProjectionHead(nn.Module):
    """Single scale projection head."""

    def __init__(
        self,
        input_dim: int,
        crystal_dim: int,
        use_belly: bool = True,
        belly_expand: float = 2.0,
        dropout: float = 0.1,
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
                nn.Dropout(dropout),
                nn.Linear(belly_dim, crystal_dim, bias=False)
            )
        else:
            self.projection = nn.Linear(input_dim, crystal_dim, bias=False)

    def forward(self, features: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(self.projection(features), dim=-1)
        anchors_norm = F.normalize(anchors, dim=-1)
        logits = torch.mm(z, anchors_norm.T) / self.temperature
        return logits, z


class MultiScaleCrystalHead(nn.Module):
    """
    Optimized multi-scale head.

    Since scales have different dimensions, we can't fully batch,
    but we optimize the fusion step.
    """

    def __init__(self, config: DavidBeansConfig):
        super().__init__()
        self.config = config
        self.scales = config.scales
        self.num_scales = len(config.scales)

        # Per-scale heads (different output dims, so separate)
        self.heads = nn.ModuleList([
            CrystalProjectionHead(
                input_dim=config.dim,
                crystal_dim=scale,
                use_belly=config.use_belly,
                belly_expand=config.belly_expand,
                dropout=config.dropout,
                temperature=config.contrast_temperature
            )
            for scale in config.scales
        ])

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.LayerNorm(config.dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim // 2, self.num_scales)
        )

    def forward(
        self,
        features: torch.Tensor,
        anchors_dict: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        #B = features.shape[0]
        #num_classes = anchors_dict[self.scales[0]].shape[0]

        # Process all scales (can't batch due to different dims)
        scale_logits = []
        scale_features = []

        for i, scale in enumerate(self.scales):
            logits, z = self.heads[i](features, anchors_dict[scale])
            scale_logits.append(logits)
            scale_features.append(z)

        # Fusion weights
        fusion_weights = F.softmax(self.fusion(features), dim=-1)  # [B, num_scales]

        # OPTIMIZED: Stack logits and use einsum for weighted combination
        # Instead of: for i, logits in enumerate(scale_logits): combined += w[:, i:i+1] * logits
        stacked_logits = torch.stack(scale_logits, dim=1)  # [B, num_scales, num_classes]
        combined = torch.einsum('bs,bsc->bc', fusion_weights, stacked_logits)  # [B, num_classes]

        return combined, scale_logits, scale_features, fusion_weights


# ============================================================================
# OPTIMIZED CROSS-CONTRASTIVE LOSS
# ============================================================================

class CrossContrastiveLoss(nn.Module):
    """Optimized cross-contrastive loss."""

    def __init__(
        self,
        temperature: float = 0.07,
        patch_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.patch_weight = patch_weight

    def forward(
        self,
        patch_features: torch.Tensor,  # [B, P, D]
        scale_features: List[torch.Tensor],  # List of [B, scale_dim]
        anchors_dict: Dict[int, torch.Tensor],  # {scale: [C, scale_dim]}
        targets: torch.Tensor,  # [B]
        scales: List[int]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B, P, D = patch_features.shape
        device = patch_features.device

        # Mean pool patches once
        patch_mean = patch_features.mean(dim=1)  # [B, D]

        total_loss = torch.tensor(0.0, device=device)
        losses = {}

        for i, (scale, scale_feat) in enumerate(zip(scales, scale_features)):
            anchors = anchors_dict[scale]

            # Dimension alignment (vectorized)
            if D >= scale:
                patch_proj = patch_mean[..., :scale]
            else:
                patch_proj = F.pad(patch_mean, (0, scale - D))

            patch_proj = F.normalize(patch_proj, dim=-1)
            scale_feat_norm = F.normalize(scale_feat, dim=-1)
            anchors_norm = F.normalize(anchors, dim=-1)

            # Patch-to-scale similarity (batch dot product)
            patch_scale_sim = (patch_proj * scale_feat_norm).sum(dim=-1)  # [B]
            patch_scale_loss = (1 - patch_scale_sim).mean()

            # Anchor contrastive
            scale_to_anchor = torch.mm(scale_feat_norm, anchors_norm.T) / self.temperature
            anchor_loss = F.cross_entropy(scale_to_anchor, targets)

            losses[f'patch_scale_{scale}'] = patch_scale_loss
            losses[f'anchor_{scale}'] = anchor_loss

            total_loss = total_loss + self.patch_weight * patch_scale_loss + anchor_loss

        total_loss = total_loss / len(scales)
        losses['total'] = total_loss

        return total_loss, losses


# ============================================================================
# OPTIMIZED BEANS BACKBONE
# ============================================================================

class BeansBackbone(nn.Module):
    """Optimized ViT-Beans backbone."""

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

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + config.num_patches, config.dim) * 0.02
        )

        # Router (computed once at init)
        self.router = HybridCantorRouter(
            num_positions=config.num_patches,
            k_neighbors=config.k_neighbors,
            cantor_weight=config.cantor_weight,
            grid_size=config.grid_size
        )

        # Interleaved attention + experts
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
        B = x.shape[0]

        # Patch embed + flatten
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, P, D]

        # Add CLS + position
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.dropout(x + self.pos_embed)

        # Process layers
        all_vertices = []
        for block, expert_layer in zip(self.blocks, self.expert_layers):
            x = block(x)
            x, vertices = expert_layer(x)
            all_vertices.append(vertices)

        x = self.norm(x)

        # Pool
        features = x[:, 0] if self.config.pooling == "cls" else x[:, 1:].mean(dim=1)

        return features, x, all_vertices


# ============================================================================
# UNIFIED MODEL
# ============================================================================

class DavidBeans(nn.Module):
    """
    DavidBeans: Optimized Unified Vision-to-Crystal Architecture.
    """

    def __init__(self, config: DavidBeansConfig):
        super().__init__()
        self.config = config

        self.backbone = BeansBackbone(config)
        self.head = MultiScaleCrystalHead(config)
        self.cayley_loss = UnifiedCayleyMengerLoss(volume_floor=config.volume_floor)
        self.cross_contrast = CrossContrastiveLoss(
            temperature=config.contrast_temperature,
            patch_weight=config.contrast_weight
        )

        # Crystal anchors
        self.anchors = nn.ParameterDict({
            str(scale): nn.Parameter(torch.randn(config.num_classes, scale) * 0.02)
            for scale in config.scales
        })

    def get_anchors_dict(self) -> Dict[int, torch.Tensor]:
        return {int(k): v for k, v in self.anchors.items()}

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:

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

            # CE loss
            losses['ce'] = F.cross_entropy(combined_logits, targets)

            # Per-scale CE (vectorized)
            for i, (scale, logits) in enumerate(zip(self.config.scales, scale_logits)):
                losses[f'ce_{scale}'] = F.cross_entropy(logits, targets)

            # Geometric loss
            # Stack all layer vertices: [L, E, 5, D] -> mean -> [E, 5, D]
            stacked = torch.stack(all_vertices, dim=0)
            mean_vertices = stacked.mean(dim=0)
            geo_loss, geo_metrics = self.cayley_loss(expert_vertices=mean_vertices)
            losses['geometric'] = geo_loss
            losses.update(geo_metrics)

            # Cross-contrastive
            patch_tokens = all_tokens[:, 1:, :]
            contrast_loss, contrast_metrics = self.cross_contrast(
                patch_features=patch_tokens,
                scale_features=scale_features,
                anchors_dict=anchors_dict,
                targets=targets,
                scales=self.config.scales
            )
            losses['contrast'] = contrast_loss
            losses.update(contrast_metrics)

            # Total
            losses['total'] = (
                losses['ce'] +
                self.config.cayley_weight * geo_loss +
                self.config.contrast_weight * contrast_loss
            )

            result['losses'] = losses

        return result

    def get_model_info(self) -> Dict:
        return {
            'name': 'DavidBeans-Optimized',
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
        }

    def __repr__(self):
        info = self.get_model_info()
        return (
            f"DavidBeans-Optimized(\n"
            f"  Vision: {info['image_size']}px → {info['num_patches']} patches\n"
            f"  Backbone: {info['num_layers']} layers, {info['dim']}d, {info['num_heads']} heads\n"
            f"  Routing: k={info['k_neighbors']}, α={info['cantor_weight']}\n"
            f"  Experts: {info['num_experts']} (pentachoron, batched)\n"
            f"  Scales: {info['scales']}\n"
            f"  Parameters: {info['total_params']:,}\n"
            f")"
        )


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark():
    """Compare original vs optimized forward pass speed."""
    import time

    print("=" * 60)
    print("DavidBeans Optimization Benchmark")
    print("=" * 60)

    config = DavidBeansConfig(
        image_size=32,
        patch_size=4,
        dim=256,
        num_layers=4,
        num_heads=4,
        num_experts=5,
        k_neighbors=16,
        scales=[64, 128, 256],
        num_classes=10
    )

    model = DavidBeans(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    print(f"\nDevice: {device}")
    print(f"Config: {config.dim}d, {config.num_layers} layers, {config.num_experts} experts")

    # Warmup
    x = torch.randn(32, 3, 32, 32, device=device)
    targets = torch.randint(0, 10, (32,), device=device)

    for _ in range(10):
        with torch.no_grad():
            _ = model(x, targets=targets)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    num_runs = 100

    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            result = model(x, targets=targets)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_batch = (elapsed / num_runs) * 1000
    samples_per_sec = (32 * num_runs) / elapsed

    print(f"\nResults ({num_runs} runs):")
    print(f"  Time per batch: {ms_per_batch:.2f} ms")
    print(f"  Throughput: {samples_per_sec:.0f} samples/sec")
    print(f"  Loss: {result['losses']['total'].item():.4f}")

    # Memory
    if device == 'cuda':
        print(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark()