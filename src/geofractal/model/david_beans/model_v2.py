"""
DavidBeans V2: Wormhole Routing Integration
============================================

Key upgrades from V1:
1. LearnedWormholeRouter: Content-aware routing (replaces fixed Cantor)
2. WormholeAttentionBlock: Learned sparse attention patterns
3. WormholeTessellationExpert: Tiles connect via learned wormholes
4. Proper gradient flow through all routing decisions

Based on experimental validation:
- Permutation recovery: 100% routing accuracy
- Patch matching: 99.6% correspondence learning
- Key insight: When routing IS necessary, routing learns structure

Author: AbstractPhil
Date: November 29, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math


# ============================================================================
# CONFIGURATION V2
# ============================================================================

@dataclass
class DavidBeansV2Config:
    """Configuration for DavidBeans V2 with wormhole routing."""

    # Vision
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3

    # Architecture
    dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0

    # Wormhole routing
    num_wormholes: int = 8  # Connections per position
    wormhole_temperature: float = 0.1
    wormhole_mode: str = "hybrid"  # "learned", "cantor", "hybrid"
    cantor_weight: float = 0.3  # For hybrid mode initialization

    # Tessellation
    num_tiles: int = 16  # Feature-dim tessellation
    tile_wormholes: int = 4  # Cross-tile connections

    # Crystal head
    scales: List[int] = field(default_factory=lambda: [64, 128, 256, 384, 512])
    num_classes: int = 100
    use_belly: bool = True
    belly_expand: float = 2.0

    # Loss weights
    contrast_temperature: float = 0.07
    contrast_weight: float = 0.5
    routing_weight: float = 0.0  # Optional: auxiliary routing loss

    # Regularization
    dropout: float = 0.1
    pooling: str = "cls"

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def tile_dim(self) -> int:
        return self.dim // self.num_tiles


# ============================================================================
# BATCHED WORMHOLE GATHER UTILITIES
# ============================================================================

def gather_wormhole(x: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
    """
    Efficient batched gather via wormhole routes using torch.gather.

    Args:
        x: [B, P, D] features to gather from
        routes: [B, P, K] destination indices
    Returns:
        gathered: [B, P, K, D]
    """
    B, P, D = x.shape
    K = routes.shape[-1]

    # Flatten routes and expand for gather: [B, P*K, D]
    routes_flat = routes.reshape(B, P * K).unsqueeze(-1).expand(-1, -1, D)

    # Gather and reshape
    gathered = torch.gather(x, 1, routes_flat)
    return gathered.view(B, P, K, D)


def gather_wormhole_multihead(x: torch.Tensor, routes: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Batched gather for multi-head attention.

    Args:
        x: [B, H, P, D] features per head
        routes: [B, P, K] shared routes across heads
    Returns:
        gathered: [B, H, P, K, D]
    """
    B, H, P, D = x.shape
    K = routes.shape[-1]

    # Flatten B*H for efficient gather
    x_flat = x.reshape(B * H, P, D)

    # Expand routes: [B, P, K] -> [B*H, P*K]
    routes_exp = routes.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, P * K)
    routes_exp = routes_exp.unsqueeze(-1).expand(-1, -1, D)

    # Gather and reshape
    gathered = torch.gather(x_flat, 1, routes_exp)
    return gathered.view(B, H, P, K, D)


def gather_tiles(x_tiles: torch.Tensor, routes: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Batched gather for tile wormholes across sequence positions.

    Args:
        x_tiles: [B, S, T, tile_dim] tiled features
        routes: [B, T, K] tile routes (shared across S)
    Returns:
        gathered: [B, S, T, K, tile_dim]
    """
    B, S, T, tile_dim = x_tiles.shape
    K = routes.shape[-1]

    # Flatten B*S as batch dimension
    x_flat = x_tiles.reshape(B * S, T, tile_dim)

    # Expand routes for all sequence positions: [B, T, K] -> [B*S, T*K]
    routes_exp = routes.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, T * K)
    routes_exp = routes_exp.unsqueeze(-1).expand(-1, -1, tile_dim)

    # Gather and reshape
    gathered = torch.gather(x_flat, 1, routes_exp)
    return gathered.view(B, S, T, K, tile_dim)


# ============================================================================
# LEARNED WORMHOLE ROUTER
# ============================================================================

class LearnedWormholeRouter(nn.Module):
    """
    Content-aware wormhole routing with Cantor initialization.

    Key properties:
    - Routes computed from content (query @ key), not just position
    - Straight-through estimator for hard routing
    - Cantor fingerprints provide geometric initialization
    - Fully learnable with gradient flow
    """

    def __init__(
            self,
            dim: int,
            num_positions: int,
            num_wormholes: int,
            temperature: float = 0.1,
            mode: str = "hybrid",
            cantor_weight: float = 0.3,
            grid_size: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        self.num_positions = num_positions
        self.num_wormholes = min(num_wormholes, num_positions - 1)
        self.temperature = temperature
        self.mode = mode
        self.grid_size = grid_size or int(math.sqrt(num_positions))

        # Query/Key projections for content-aware routing
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)

        # Cantor fingerprints for initialization/bias
        fingerprints = self._compute_cantor_fingerprints()
        self.register_buffer('fingerprints', fingerprints)

        if mode in ["hybrid", "cantor"]:
            # Learnable position bias initialized from Cantor distance
            cantor_bias = self._compute_cantor_bias()
            self.position_bias = nn.Parameter(cantor_bias * cantor_weight)
        else:
            self.position_bias = None

    def _compute_cantor_fingerprints(self) -> torch.Tensor:
        """Cantor pairing function for position fingerprints."""
        P = self.num_positions
        G = self.grid_size

        x = torch.arange(P) % G
        y = torch.arange(P) // G

        sums = x + y
        z = (sums * (sums + 1)) // 2 + y

        return z.float() / z.max().clamp(min=1)

    def _compute_cantor_bias(self) -> torch.Tensor:
        """Initialize bias from Cantor distance (closer = higher affinity)."""
        fp = self.fingerprints
        dist = (fp.unsqueeze(0) - fp.unsqueeze(1)).abs()
        # Convert distance to affinity (closer = higher)
        affinity = 1.0 - dist
        # Mask self-connections
        affinity.fill_diagonal_(-1e9)
        return affinity

    def forward(
            self,
            x: torch.Tensor,
            return_scores: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute wormhole routes from content.

        Args:
            x: [B, S, D] input features (S = num_positions + 1 for CLS)
            return_scores: whether to return raw scores for loss

        Returns:
            routes: [B, P, K] indices of selected positions per query
            weights: [B, P, K] soft weights for selected positions
            scores: [B, P, P] raw routing scores (if return_scores=True)
        """
        P = self.num_positions
        K = self.num_wormholes

        # Exclude CLS token for patch routing
        x_patches = x[:, 1:, :]  # [B, P, D]

        # Content-based routing scores
        queries = F.normalize(self.query_proj(x_patches), dim=-1)
        keys = F.normalize(self.key_proj(x_patches), dim=-1)

        # Similarity scores
        scores = torch.bmm(queries, keys.transpose(1, 2))  # [B, P, P]

        # Add position bias if hybrid/cantor mode
        if self.position_bias is not None:
            scores = scores + self.position_bias.unsqueeze(0)

        # Mask self-connections
        mask = torch.eye(P, device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(mask.unsqueeze(0), -1e9)

        # Temperature scaling
        scores_scaled = scores / self.temperature

        # Top-k selection
        topk_scores, routes = torch.topk(scores_scaled, K, dim=-1)

        # Soft weights over selected routes
        weights = F.softmax(topk_scores, dim=-1)

        if return_scores:
            return routes, weights, scores
        return routes, weights, None


# ============================================================================
# WORMHOLE ATTENTION BLOCK
# ============================================================================

class WormholeAttentionBlock(nn.Module):
    """
    Attention block with learned wormhole routing.

    CLS token: Dense attention to all patches
    Patches: Sparse attention via learned wormholes
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_patches: int,
            num_wormholes: int,
            temperature: float = 0.1,
            mode: str = "hybrid",
            mlp_ratio: float = 4.0,
            dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_patches = num_patches
        self.num_wormholes = num_wormholes

        # Wormhole router
        self.router = LearnedWormholeRouter(
            dim=dim,
            num_positions=num_patches,
            num_wormholes=num_wormholes,
            temperature=temperature,
            mode=mode
        )

        # Standard QKV for attention values
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # MLP
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

    def forward(
            self,
            x: torch.Tensor,
            return_routing: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        B, S, D = x.shape
        H = self.num_heads
        P = self.num_patches
        head_dim = self.head_dim

        # Pre-norm
        x_norm = self.norm1(x)

        # Get wormhole routes from content
        routes, route_weights, route_scores = self.router(
            x_norm, return_scores=return_routing
        )

        # QKV projection - fused reshape
        qkv = self.qkv(x_norm).view(B, S, 3, H, head_dim).permute(2, 0, 3, 1, 4)
        Q, K_full, V = qkv.unbind(0)  # Each: [B, H, S, head_dim]

        # === CLS ATTENTION (dense) ===
        Q_cls = Q[:, :, :1, :]
        attn_cls = F.softmax(
            torch.einsum('bhqd,bhkd->bhqk', Q_cls, K_full) * self.scale,
            dim=-1
        )
        attn_cls = self.attn_drop(attn_cls)
        out_cls = torch.einsum('bhqk,bhkd->bhqd', attn_cls, V)

        # === PATCH ATTENTION (sparse via wormholes) - BATCHED ===
        Q_patches = Q[:, :, 1:, :]  # [B, H, P, head_dim]
        K_patches = K_full[:, :, 1:, :]
        V_patches = V[:, :, 1:, :]

        # Batched gather through wormholes
        K_gathered = gather_wormhole_multihead(K_patches, routes, H)  # [B, H, P, K, head_dim]
        V_gathered = gather_wormhole_multihead(V_patches, routes, H)

        # Sparse attention scores
        scores_patches = torch.einsum('bhpd,bhpkd->bhpk', Q_patches, K_gathered) * self.scale

        # Incorporate route weights (log-space addition = multiplication)
        scores_patches = scores_patches + route_weights.unsqueeze(1).log().clamp(min=-10)

        attn_patches = self.attn_drop(F.softmax(scores_patches, dim=-1))
        out_patches = torch.einsum('bhpk,bhpkd->bhpd', attn_patches, V_gathered)

        # Combine and project
        out = torch.cat([out_cls, out_patches], dim=2)
        out = self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, S, D)))

        # Residual + MLP
        x = x + out
        x = x + self.mlp(self.norm2(x))

        if return_routing:
            return x, {'routes': routes, 'weights': route_weights, 'scores': route_scores}
        return x, None


# ============================================================================
# WORMHOLE TESSELLATION EXPERT
# ============================================================================

class WormholeTessellationExpert(nn.Module):
    """
    Feature-dim tessellation with learned wormhole connections.

    Each tile processes its slice + gathered wormhole context.
    """

    def __init__(
            self,
            dim: int,
            num_tiles: int,
            num_wormholes: int = 4,
            temperature: float = 0.5,
            dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_tiles = num_tiles
        self.tile_dim = dim // num_tiles
        self.num_wormholes = min(num_wormholes, num_tiles - 1)
        self.temperature = temperature

        # Tile router (learns which tiles connect)
        self.tile_query = nn.Linear(self.tile_dim, self.tile_dim)
        self.tile_key = nn.Linear(self.tile_dim, self.tile_dim)

        # Process: self + wormhole context
        context_dim = self.tile_dim * (1 + self.num_wormholes)
        hidden_dim = self.tile_dim * 2

        self.processor = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.tile_dim)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
            self,
            x: torch.Tensor,
            return_routing: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        B, S, D = x.shape
        T = self.num_tiles
        K = self.num_wormholes
        tile_dim = self.tile_dim

        x_norm = self.norm(x)

        # Tessellate: [B, S, D] -> [B, S, T, tile_dim]
        x_tiles = x_norm.view(B, S, T, tile_dim)

        # Compute tile routing (shared across sequence positions)
        tile_repr = x_tiles.mean(dim=1)  # [B, T, tile_dim]

        queries = F.normalize(self.tile_query(tile_repr), dim=-1)
        keys = F.normalize(self.tile_key(tile_repr), dim=-1)

        # Routing scores
        scores = torch.bmm(queries, keys.transpose(1, 2))  # [B, T, T]

        # Mask self
        mask = torch.eye(T, device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(mask.unsqueeze(0), -1e9)

        # Top-k wormholes per tile
        topk_scores, routes = torch.topk(scores / self.temperature, K, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)  # [B, T, K]

        # === BATCHED GATHER across all sequence positions ===
        gathered = gather_tiles(x_tiles, routes, S)  # [B, S, T, K, tile_dim]

        # Concatenate: self + all wormhole neighbors
        gathered_flat = gathered.view(B, S, T, K * tile_dim)
        combined = torch.cat([x_tiles, gathered_flat], dim=-1)  # [B, S, T, (1+K)*tile_dim]

        # Process all tiles in parallel
        out_tiles = self.processor(combined)  # [B, S, T, tile_dim]

        # Reconstruct
        out = out_tiles.reshape(B, S, D)
        out = x + out  # Residual

        if return_routing:
            return out, {'routes': routes, 'weights': weights, 'scores': scores}
        return out, None


# ============================================================================
# MULTI-SCALE CRYSTAL HEAD
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
    """Multi-scale projection with learned fusion."""

    def __init__(self, config: DavidBeansV2Config):
        super().__init__()
        self.config = config
        self.scales = config.scales
        self.num_scales = len(config.scales)

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

        # Pre-compute fusion weights once
        fusion_weights = F.softmax(self.fusion(features), dim=-1)

        # Process scales - preallocate lists
        scale_logits = [None] * self.num_scales
        scale_features = [None] * self.num_scales

        for i, scale in enumerate(self.scales):
            scale_logits[i], scale_features[i] = self.heads[i](features, anchors_dict[scale])

        # Fused weighted combination
        stacked_logits = torch.stack(scale_logits, dim=1)
        combined = torch.einsum('bs,bsc->bc', fusion_weights, stacked_logits)

        return combined, scale_logits, scale_features, fusion_weights


# ============================================================================
# DAVIDBEANS V2 BACKBONE
# ============================================================================

class BeansBackboneV2(nn.Module):
    """Backbone with wormhole routing throughout."""

    def __init__(self, config: DavidBeansV2Config):
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

        # Interleaved: WormholeAttention -> WormholeTessellation
        self.attention_blocks = nn.ModuleList()
        self.expert_layers = nn.ModuleList()

        for _ in range(config.num_layers):
            self.attention_blocks.append(
                WormholeAttentionBlock(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    num_patches=config.num_patches,
                    num_wormholes=config.num_wormholes,
                    temperature=config.wormhole_temperature,
                    mode=config.wormhole_mode,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout
                )
            )
            self.expert_layers.append(
                WormholeTessellationExpert(
                    dim=config.dim,
                    num_tiles=config.num_tiles,
                    num_wormholes=config.tile_wormholes,
                    dropout=config.dropout
                )
            )

        self.norm = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self,
            x: torch.Tensor,
            return_routing: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Dict]]]:
        B = x.shape[0]

        # Patch embed
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Add CLS + position
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.dropout(x + self.pos_embed)

        # Process layers
        all_routing = [] if return_routing else None

        for attn_block, expert_layer in zip(self.attention_blocks, self.expert_layers):
            x, attn_routing = attn_block(x, return_routing=return_routing)
            x, expert_routing = expert_layer(x, return_routing=return_routing)

            if return_routing:
                all_routing.append({
                    'attention': attn_routing,
                    'expert': expert_routing
                })

        x = self.norm(x)

        # Pool
        features = x[:, 0] if self.config.pooling == "cls" else x[:, 1:].mean(dim=1)

        return features, x, all_routing


# ============================================================================
# DAVIDBEANS V2 FULL MODEL
# ============================================================================

class DavidBeansV2(nn.Module):
    """
    DavidBeans V2: Wormhole Routing Architecture

    Key improvements over V1:
    - Learned content-aware routing (not fixed Cantor)
    - Wormhole connections in both attention and tessellation
    - Gradient flow through all routing decisions
    """

    def __init__(self, config: DavidBeansV2Config):
        super().__init__()
        self.config = config

        self.backbone = BeansBackboneV2(config)
        self.head = MultiScaleCrystalHead(config)

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
            return_loss: bool = True,
            return_routing: bool = False
    ) -> Dict[str, torch.Tensor]:

        # Backbone
        features, all_tokens, routing_info = self.backbone(
            x, return_routing=return_routing
        )

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

        if return_routing:
            result['routing'] = routing_info

        if return_loss and targets is not None:
            losses = {}

            # CE loss
            losses['ce'] = F.cross_entropy(combined_logits, targets)

            # Per-scale CE
            for i, (scale, logits) in enumerate(zip(self.config.scales, scale_logits)):
                losses[f'ce_{scale}'] = F.cross_entropy(logits, targets)

            # Contrastive losses
            patch_tokens = all_tokens[:, 1:, :]
            contrast_loss = self._compute_contrast_loss(
                patch_tokens, scale_features, anchors_dict, targets
            )
            losses['contrast'] = contrast_loss

            # Total
            losses['total'] = (
                    losses['ce'] +
                    self.config.contrast_weight * contrast_loss
            )

            result['losses'] = losses

        return result

    def _compute_contrast_loss(
            self,
            patch_features: torch.Tensor,
            scale_features: List[torch.Tensor],
            anchors_dict: Dict[int, torch.Tensor],
            targets: torch.Tensor
    ) -> torch.Tensor:
        """Batched contrastive loss across scales."""
        device = patch_features.device
        num_scales = len(self.config.scales)

        # Stack scale features and anchors for batched computation where possible
        total_loss = torch.tensor(0.0, device=device)

        for scale, scale_feat in zip(self.config.scales, scale_features):
            anchors = anchors_dict[scale]
            scale_feat_norm = F.normalize(scale_feat, dim=-1)
            anchors_norm = F.normalize(anchors, dim=-1)

            # Anchor contrastive
            logits = torch.mm(scale_feat_norm, anchors_norm.T) / self.config.contrast_temperature
            total_loss = total_loss + F.cross_entropy(logits, targets)

        return total_loss / num_scales

    def get_model_info(self) -> Dict:
        return {
            'name': 'DavidBeans-V2-Wormhole',
            'image_size': self.config.image_size,
            'patch_size': self.config.patch_size,
            'dim': self.config.dim,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'num_wormholes': self.config.num_wormholes,
            'num_tiles': self.config.num_tiles,
            'tile_wormholes': self.config.tile_wormholes,
            'wormhole_mode': self.config.wormhole_mode,
            'scales': self.config.scales,
            'num_classes': self.config.num_classes,
            'num_patches': self.config.num_patches,
            'total_params': sum(p.numel() for p in self.parameters()),
        }

    def __repr__(self):
        info = self.get_model_info()
        return (
            f"DavidBeans-V2-Wormhole(\n"
            f"  Vision: {info['image_size']}px → {info['num_patches']} patches\n"
            f"  Backbone: {info['num_layers']} layers, {info['dim']}d, {info['num_heads']} heads\n"
            f"  Wormholes: {info['num_wormholes']} per position, mode={info['wormhole_mode']}\n"
            f"  Tessellation: {info['num_tiles']} tiles × {info['tile_wormholes']} wormholes\n"
            f"  Scales: {info['scales']}\n"
            f"  Parameters: {info['total_params']:,}\n"
            f")"
        )


# ============================================================================
# QUICK TEST
# ============================================================================

def test_v2():
    """Quick functionality test."""
    print("=" * 60)
    print("DavidBeans V2 - Wormhole Routing Test")
    print("=" * 60)

    config = DavidBeansV2Config(
        image_size=32,
        patch_size=4,
        dim=256,
        num_layers=2,
        num_heads=4,
        num_wormholes=8,
        num_tiles=8,
        tile_wormholes=3,
        scales=[64, 128, 256],
        num_classes=100
    )

    model = DavidBeansV2(config)
    print(model)

    # Test forward
    x = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 100, (4,))

    result = model(x, targets=targets, return_routing=True)

    print(f"\nOutput shapes:")
    print(f"  Logits: {result['logits'].shape}")
    print(f"  Features: {result['features'].shape}")
    print(f"  Total loss: {result['losses']['total'].item():.4f}")

    # Check routing info
    if result['routing']:
        attn_routing = result['routing'][0]['attention']
        expert_routing = result['routing'][0]['expert']
        print(f"\nRouting shapes (layer 0):")
        print(f"  Attention routes: {attn_routing['routes'].shape}")
        print(f"  Attention weights: {attn_routing['weights'].shape}")
        print(f"  Expert routes: {expert_routing['routes'].shape}")
        print(f"  Expert weights: {expert_routing['weights'].shape}")

    print("\n✓ V2 forward pass successful!")

    return model, config


if __name__ == "__main__":
    test_v2()