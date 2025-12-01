"""
DavidBeans V2.3: Wormhole Routing with Topological Dropout
==========================================================

Key upgrades from V2.2:
- Removed FractalRegularizer (CantorGELU mothballed - didn't improve convergence)
- Added TopologicalDropout variants based on Fashion-MNIST experiments:
  - ScheduledTopologicalDropout: Best accuracy (warmup aligns with crystallization)
  - SpatialTopologicalDropout: Best generalization (0.62% gap)
  - WormholeDropout: Combined strategy for attention blocks

Experimental Results (Fashion-MNIST):
- topo_scheduled: 93.01% (1.66% gap) - WINNER
- standard:       92.82% (1.23% gap) - baseline
- spatial:        92.78% (0.62% gap) - best generalization
- topo_importance: 90.48% (5.17% gap) - HARMFUL, not included

Author: AbstractPhil
Date: December 1, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math

# Import topological dropout - adjust path as needed
try:
    from geofractal.model.layers.dropout.topological_dropout import (
        TopologicalDropout,
        ScheduledTopologicalDropout,
        SpatialTopologicalDropout,
        WormholeDropout,
    )
    TOPO_DROPOUT_AVAILABLE = True
except ImportError:
    TOPO_DROPOUT_AVAILABLE = False
    print("Warning: Topological dropout not available, falling back to standard dropout")


# ============================================================================
# CONFIGURATION V2.3
# ============================================================================

@dataclass
class DavidBeansV2Config:
    """Configuration for DavidBeans V2.3 with wormhole routing and topological dropout."""

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
    num_wormholes: int = 8
    wormhole_temperature: float = 0.1
    wormhole_mode: str = "hybrid"
    cantor_weight: float = 0.3

    # Tessellation
    num_tiles: int = 16
    tile_wormholes: int = 4

    # Crystal head - scales
    scales: List[int] = field(default_factory=lambda: [64, 128, 256, 384, 512])
    num_classes: int = 100

    # Crystal head - belly configuration
    use_belly: bool = True
    belly_expand: float = 2.0
    belly_layers: int = 2
    belly_residual: bool = False

    # Crystal head - scale weighting
    weighting_mode: str = "learned"
    scale_weight_floor: float = 0.1

    # Crystal head - redundant scales
    scale_copies: Optional[List[int]] = None
    copy_theta_step: float = 0.15

    # Crystal head - collective mode
    use_collective: bool = False
    collective_temperature: float = 0.07

    # Conv spine
    use_spine: bool = False
    spine_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    spine_cross_attn: bool = True
    spine_gate_init: float = 0.0

    # Loss weights
    contrast_temperature: float = 0.07
    contrast_weight: float = 0.5
    routing_weight: float = 0.0

    # === DROPOUT CONFIGURATION (V2.3) ===
    dropout: float = 0.1  # Base dropout rate (used if topo disabled)

    # Topological dropout (structure-preserving)
    use_topo_dropout: bool = True
    topo_drop_prob: float = 0.15        # Route dropout probability
    topo_warmup_epochs: int = 35        # Warmup before full dropout (align with crystallization)
    topo_min_routes_keep: int = 2       # Minimum routes to preserve
    topo_steps_per_epoch: int = 391     # Steps per epoch (CIFAR-100 with batch 128)

    # Spatial dropout (for patch embeddings)
    use_spatial_dropout: bool = True
    spatial_drop_prob: float = 0.1
    spatial_patch_size: int = 2

    # Route dimension in tensors
    route_dim_attention: int = -2       # For [B, S, K, D] gathered values
    route_dim_tiles: int = 2            # For [B, S, T, tile_dim] tessellation

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

    def __post_init__(self):
        """Validate configuration."""
        assert self.dim % self.num_tiles == 0, \
            f"dim ({self.dim}) must be divisible by num_tiles ({self.num_tiles})"
        assert self.dim % self.num_heads == 0, \
            f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.weighting_mode in ["balanced", "geometric", "top_heavy", "learned"], \
            f"Invalid weighting_mode: {self.weighting_mode}"
        assert 2 <= self.belly_layers <= 4, \
            f"belly_layers must be 2-4, got {self.belly_layers}"

        # Validate topo dropout settings
        if self.use_topo_dropout and not TOPO_DROPOUT_AVAILABLE:
            print("Warning: use_topo_dropout=True but TopologicalDropout not available, disabling")
            self.use_topo_dropout = False


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

    routes_flat = routes.reshape(B, P * K).unsqueeze(-1).expand(-1, -1, D)
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

    x_flat = x.reshape(B * H, P, D)
    routes_exp = routes.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, P * K)
    routes_exp = routes_exp.unsqueeze(-1).expand(-1, -1, D)

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

    x_flat = x_tiles.reshape(B * S, T, tile_dim)
    routes_exp = routes.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, T * K)
    routes_exp = routes_exp.unsqueeze(-1).expand(-1, -1, tile_dim)

    gathered = torch.gather(x_flat, 1, routes_exp)
    return gathered.view(B, S, T, K, tile_dim)


# ============================================================================
# LEARNED WORMHOLE ROUTER
# ============================================================================

class LearnedWormholeRouter(nn.Module):
    """
    Content-aware wormhole routing with Cantor initialization.
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

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)

        fingerprints = self._compute_cantor_fingerprints()
        self.register_buffer('fingerprints', fingerprints)

        if mode in ["hybrid", "cantor"]:
            cantor_bias = self._compute_cantor_bias()
            self.position_bias = nn.Parameter(cantor_bias * cantor_weight)
        else:
            self.position_bias = None

    def _compute_cantor_fingerprints(self) -> torch.Tensor:
        P = self.num_positions
        G = self.grid_size

        x = torch.arange(P) % G
        y = torch.arange(P) // G

        sums = x + y
        z = (sums * (sums + 1)) // 2 + y

        return z.float() / z.max().clamp(min=1)

    def _compute_cantor_bias(self) -> torch.Tensor:
        fp = self.fingerprints
        dist = (fp.unsqueeze(0) - fp.unsqueeze(1)).abs()
        affinity = 1.0 - dist
        affinity.fill_diagonal_(-1e9)
        return affinity

    def forward(
            self,
            x: torch.Tensor,
            return_scores: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        P = self.num_positions
        K = self.num_wormholes

        x_patches = x[:, 1:, :]

        queries = F.normalize(self.query_proj(x_patches), dim=-1)
        keys = F.normalize(self.key_proj(x_patches), dim=-1)

        scores = torch.bmm(queries, keys.transpose(1, 2))

        if self.position_bias is not None:
            scores = scores + self.position_bias.unsqueeze(0)

        mask = torch.eye(P, device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(mask.unsqueeze(0), -1e9)

        scores_scaled = scores / self.temperature

        topk_scores, routes = torch.topk(scores_scaled, K, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        if return_scores:
            return routes, weights, scores
        return routes, weights, None


# ============================================================================
# CONV SPINE
# ============================================================================

class ConvSpine(nn.Module):
    """Persistent wide conv pathway for spatial coherence."""

    def __init__(
        self,
        in_channels: int = 3,
        spine_channels: List[int] = [64, 128, 256],
        output_dim: int = 512,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.stages = nn.ModuleList()
        ch_in = in_channels

        for ch_out in spine_channels:
            self.stages.append(nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.GELU(),
                nn.Conv2d(ch_out, ch_out, 3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.GELU(),
            ))
            ch_in = ch_out

        self.proj = nn.Conv2d(spine_channels[-1], output_dim, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for stage in self.stages:
            x = stage(x)
            x = F.max_pool2d(x, 2)

        x = self.proj(x)

        spine_global = self.pool(x).flatten(1)

        B, D, H, W = x.shape
        spine_tokens = x.flatten(2).transpose(1, 2)

        return spine_global, spine_tokens


# ============================================================================
# DROPOUT FACTORY
# ============================================================================

def create_dropout(
    config: DavidBeansV2Config,
    dropout_type: str = "standard",
    route_dim: int = 1,
    num_routes: Optional[int] = None,
) -> nn.Module:
    """
    Factory for creating appropriate dropout based on config and type.

    Args:
        config: Model configuration
        dropout_type: "standard", "topo", "topo_scheduled", "spatial", "wormhole"
        route_dim: Which dimension contains routes
        num_routes: Number of routes (for min_keep calculation)

    Returns:
        Dropout module
    """
    if not config.use_topo_dropout or not TOPO_DROPOUT_AVAILABLE:
        # Fallback to standard dropout
        return nn.Dropout(config.dropout)

    if dropout_type == "standard":
        return nn.Dropout(config.dropout)

    elif dropout_type == "topo":
        return TopologicalDropout(
            drop_prob=config.topo_drop_prob,
            min_keep=config.topo_min_routes_keep,
            route_dim=route_dim,
        )

    elif dropout_type == "topo_scheduled":
        warmup_steps = config.topo_warmup_epochs * config.topo_steps_per_epoch
        return ScheduledTopologicalDropout(
            drop_prob=config.topo_drop_prob,
            min_keep=config.topo_min_routes_keep,
            route_dim=route_dim,
            warmup_steps=warmup_steps,
        )

    elif dropout_type == "spatial":
        if config.use_spatial_dropout:
            return SpatialTopologicalDropout(
                drop_prob=config.spatial_drop_prob,
                patch_size=config.spatial_patch_size,
            )
        else:
            return nn.Identity()

    elif dropout_type == "wormhole":
        return WormholeDropout(
            route_drop_prob=config.topo_drop_prob,
            spatial_drop_prob=config.spatial_drop_prob if config.use_spatial_dropout else 0.0,
            num_routes=num_routes or config.num_wormholes,
            min_routes_keep=config.topo_min_routes_keep,
            warmup_epochs=config.topo_warmup_epochs,
            steps_per_epoch=config.topo_steps_per_epoch,
            patch_size=config.spatial_patch_size,
        )

    else:
        raise ValueError(f"Unknown dropout_type: {dropout_type}")


# ============================================================================
# WORMHOLE ATTENTION BLOCK
# ============================================================================

class WormholeAttentionBlock(nn.Module):
    """
    Attention block with learned wormhole routing and topological dropout.

    V2.3: Uses ScheduledTopologicalDropout for route regularization
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
            config: Optional[DavidBeansV2Config] = None,
            # Legacy params (used if config not provided)
            dropout: float = 0.1,
            use_spine_cross_attn: bool = False,
            spine_gate_init: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_patches = num_patches
        self.num_wormholes = num_wormholes

        # Use config if provided, else legacy params
        self.config = config
        use_topo = config.use_topo_dropout if config else False
        base_dropout = config.dropout if config else dropout
        use_spine_cross_attn = config.spine_cross_attn if config and config.use_spine else use_spine_cross_attn
        spine_gate_init = config.spine_gate_init if config else spine_gate_init
        self.use_spine_cross_attn = use_spine_cross_attn

        # Wormhole router
        self.router = LearnedWormholeRouter(
            dim=dim,
            num_positions=num_patches,
            num_wormholes=num_wormholes,
            temperature=temperature,
            mode=mode
        )

        # QKV and projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # MLP with standard dropout (not route-based)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(base_dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(base_dropout)
        )

        # === TOPOLOGICAL DROPOUT (V2.3) ===
        if use_topo and TOPO_DROPOUT_AVAILABLE:
            # Scheduled dropout for attention (warms up with crystallization)
            self.attn_drop = create_dropout(config, "topo_scheduled", route_dim=-1)

            # Route dropout for gathered values [B, H, P, K, D] -> route_dim=3
            self.route_drop = create_dropout(config, "topo_scheduled", route_dim=3)

            self.proj_drop = nn.Dropout(base_dropout)  # Standard for projection
        else:
            self.attn_drop = nn.Dropout(base_dropout)
            self.route_drop = nn.Identity()
            self.proj_drop = nn.Dropout(base_dropout)

        # Spine cross-attention (optional)
        if use_spine_cross_attn:
            self.spine_cross_attn = nn.MultiheadAttention(
                dim, num_heads=max(1, num_heads // 2),
                batch_first=True, dropout=base_dropout
            )
            self.spine_norm = nn.LayerNorm(dim)
            self.spine_gate = nn.Parameter(torch.ones(1) * spine_gate_init)

    def forward(
            self,
            x: torch.Tensor,
            spine_tokens: Optional[torch.Tensor] = None,
            return_routing: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        B, S, D = x.shape
        H = self.num_heads
        P = self.num_patches
        head_dim = self.head_dim

        x_norm = self.norm1(x)

        # Get wormhole routes
        routes, route_weights, route_scores = self.router(
            x_norm, return_scores=return_routing
        )

        # QKV projection
        qkv = self.qkv(x_norm).view(B, S, 3, H, head_dim).permute(2, 0, 3, 1, 4)
        Q, K_full, V = qkv.unbind(0)

        # === CLS ATTENTION (dense) ===
        Q_cls = Q[:, :, :1, :]
        attn_cls = F.softmax(
            torch.einsum('bhqd,bhkd->bhqk', Q_cls, K_full) * self.scale,
            dim=-1
        )
        attn_cls = self.attn_drop(attn_cls)
        out_cls = torch.einsum('bhqk,bhkd->bhqd', attn_cls, V)

        # === PATCH ATTENTION (sparse via wormholes) ===
        Q_patches = Q[:, :, 1:, :]
        K_patches = K_full[:, :, 1:, :]
        V_patches = V[:, :, 1:, :]

        # Batched gather through wormholes
        K_gathered = gather_wormhole_multihead(K_patches, routes, H)
        V_gathered = gather_wormhole_multihead(V_patches, routes, H)

        # Apply topological dropout to gathered values (drops entire routes)
        V_gathered = self.route_drop(V_gathered)

        # Sparse attention scores
        scores_patches = torch.einsum('bhpd,bhpkd->bhpk', Q_patches, K_gathered) * self.scale
        scores_patches = scores_patches + route_weights.unsqueeze(1).log().clamp(min=-10)

        attn_patches = F.softmax(scores_patches, dim=-1)
        attn_patches = self.attn_drop(attn_patches)

        out_patches = torch.einsum('bhpk,bhpkd->bhpd', attn_patches, V_gathered)

        # Combine and project
        out = torch.cat([out_cls, out_patches], dim=2)
        out = self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, S, D)))

        x = x + out

        # Spine cross-attention (optional)
        if self.use_spine_cross_attn and spine_tokens is not None:
            x_norm_spine = self.spine_norm(x)
            spine_context, _ = self.spine_cross_attn(
                x_norm_spine, spine_tokens, spine_tokens
            )
            x = x + torch.sigmoid(self.spine_gate) * spine_context

        # MLP
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

    V2.3: Topological dropout for tile routes
    """

    def __init__(
            self,
            dim: int,
            num_tiles: int,
            num_wormholes: int = 4,
            temperature: float = 0.5,
            config: Optional[DavidBeansV2Config] = None,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_tiles = num_tiles
        self.tile_dim = dim // num_tiles
        self.num_wormholes = min(num_wormholes, num_tiles - 1)
        self.temperature = temperature
        self.config = config

        use_topo = config.use_topo_dropout if config else False
        base_dropout = config.dropout if config else dropout

        self.tile_query = nn.Linear(self.tile_dim, self.tile_dim)
        self.tile_key = nn.Linear(self.tile_dim, self.tile_dim)

        context_dim = self.tile_dim * (1 + self.num_wormholes)
        hidden_dim = self.tile_dim * 2

        self.processor = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(base_dropout),
            nn.Linear(hidden_dim, self.tile_dim)
        )

        # Topological dropout for tiles [B, S, T, tile_dim] -> route_dim=2
        if use_topo and TOPO_DROPOUT_AVAILABLE:
            self.tile_drop = create_dropout(config, "topo", route_dim=2)
        else:
            self.tile_drop = nn.Identity()

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

        # Tessellate
        x_tiles = x_norm.view(B, S, T, tile_dim)

        # Apply tile dropout
        x_tiles = self.tile_drop(x_tiles)

        # Compute tile routing
        tile_repr = x_tiles.mean(dim=1)

        queries = F.normalize(self.tile_query(tile_repr), dim=-1)
        keys = F.normalize(self.tile_key(tile_repr), dim=-1)

        scores = torch.bmm(queries, keys.transpose(1, 2))

        mask = torch.eye(T, device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(mask.unsqueeze(0), -1e9)

        topk_scores, routes = torch.topk(scores / self.temperature, K, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        # Batched gather
        gathered = gather_tiles(x_tiles, routes, S)

        # Concatenate and process
        gathered_flat = gathered.view(B, S, T, K * tile_dim)
        combined = torch.cat([x_tiles, gathered_flat], dim=-1)

        out_tiles = self.processor(combined)

        out = out_tiles.reshape(B, S, D)
        out = x + out

        if return_routing:
            return out, {'routes': routes, 'weights': weights, 'scores': scores}
        return out, None


# ============================================================================
# CONFIGURABLE BELLY PROJECTION
# ============================================================================

class ConfigurableBelly(nn.Module):
    """Multi-layer belly projection with optional residual connections."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expand: float = 2.0,
        num_layers: int = 2,
        use_residual: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_residual = use_residual

        belly_dim = int(output_dim * expand)

        def make_block(in_d, out_d, is_last=False):
            if is_last:
                return nn.Linear(in_d, out_d, bias=False)
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        if num_layers == 2:
            self.layers = nn.Sequential(
                make_block(input_dim, belly_dim),
                make_block(belly_dim, output_dim, is_last=True)
            )
        elif num_layers == 3:
            self.layers = nn.Sequential(
                make_block(input_dim, belly_dim),
                make_block(belly_dim, belly_dim),
                make_block(belly_dim, output_dim, is_last=True)
            )
        elif num_layers == 4:
            self.layers = nn.Sequential(
                make_block(input_dim, belly_dim),
                make_block(belly_dim, belly_dim),
                make_block(belly_dim, belly_dim),
                make_block(belly_dim, output_dim, is_last=True)
            )

        if use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)

        if self.use_residual:
            if self.residual_proj is not None:
                out = out + self.residual_proj(x)
            elif self.input_dim == self.output_dim:
                out = out + x

        return out


# ============================================================================
# CANTOR FINGERPRINT UTILITIES
# ============================================================================

def compute_cantor_fingerprint(
    active_dim: int,
    max_dim: int,
    theta: float = 0.0
) -> torch.Tensor:
    """Compute unique fingerprint mask with theta rotation."""
    positions = torch.arange(max_dim).float()

    grid_size = int(max_dim ** 0.5) + 1
    x = positions % grid_size
    y = positions // grid_size
    cantor_z = ((x + y) * (x + y + 1)) / 2 + y
    cantor_z = cantor_z / cantor_z.max().clamp(min=1)

    rotated = cantor_z * math.cos(theta) + (positions / max_dim) * math.sin(theta)

    mask = torch.zeros(max_dim)
    mask[:active_dim] = 1.0

    fingerprint = mask * (0.5 + 0.5 * torch.tanh(rotated))

    return fingerprint


# ============================================================================
# CRYSTAL PROJECTION HEADS
# ============================================================================

class CrystalProjectionHead(nn.Module):
    """Single scale projection head with configurable belly."""

    def __init__(
            self,
            input_dim: int,
            crystal_dim: int,
            use_belly: bool = True,
            belly_expand: float = 2.0,
            belly_layers: int = 2,
            belly_residual: bool = False,
            dropout: float = 0.1,
            temperature: float = 0.07,
            theta: float = 0.0,
            copy_index: int = 0,
    ):
        super().__init__()
        self.crystal_dim = crystal_dim
        self.temperature = temperature
        self.theta = theta
        self.copy_index = copy_index

        if use_belly:
            self.projection = ConfigurableBelly(
                input_dim=input_dim,
                output_dim=crystal_dim,
                expand=belly_expand,
                num_layers=belly_layers,
                use_residual=belly_residual,
                dropout=dropout,
            )
        else:
            self.projection = nn.Linear(input_dim, crystal_dim, bias=False)

        if theta != 0.0:
            fingerprint = compute_cantor_fingerprint(crystal_dim, crystal_dim, theta)
            self.register_buffer('fingerprint', fingerprint)
        else:
            self.fingerprint = None

    def forward(self, features: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.projection(features)

        if self.fingerprint is not None:
            z = z * self.fingerprint.unsqueeze(0)

        z = F.normalize(z, dim=-1)
        anchors_norm = F.normalize(anchors, dim=-1)
        logits = torch.mm(z, anchors_norm.T) / self.temperature
        return logits, z


class MultiScaleCrystalHead(nn.Module):
    """Multi-scale projection with learned fusion."""

    def __init__(self, config: DavidBeansV2Config):
        super().__init__()
        self.config = config
        self.scales = config.scales
        self.num_base_scales = len(config.scales)
        self.weighting_mode = config.weighting_mode
        self.scale_weight_floor = config.scale_weight_floor

        scale_copies = config.scale_copies or [1] * len(config.scales)
        assert len(scale_copies) == len(config.scales)

        self.heads = nn.ModuleList()
        self.head_scale_map = []

        for scale_idx, scale in enumerate(config.scales):
            num_copies = scale_copies[scale_idx]
            for copy_idx in range(num_copies):
                theta = copy_idx * config.copy_theta_step

                head = CrystalProjectionHead(
                    input_dim=config.dim,
                    crystal_dim=scale,
                    use_belly=config.use_belly,
                    belly_expand=config.belly_expand,
                    belly_layers=config.belly_layers,
                    belly_residual=config.belly_residual,
                    dropout=config.dropout,
                    temperature=config.contrast_temperature,
                    theta=theta,
                    copy_index=copy_idx,
                )
                self.heads.append(head)
                self.head_scale_map.append((scale, copy_idx))

        self.num_heads = len(self.heads)

        if config.weighting_mode == "learned":
            self.fusion = nn.Sequential(
                nn.Linear(config.dim, config.dim // 2),
                nn.LayerNorm(config.dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dim // 2, self.num_heads)
            )
            self.fixed_weights = None
        else:
            self.fusion = None
            self.fixed_weights = self._compute_fixed_weights(config.weighting_mode, scale_copies)

    def _compute_fixed_weights(self, mode: str, scale_copies: List[int]) -> torch.Tensor:
        weights = []

        for scale_idx, scale in enumerate(self.scales):
            num_copies = scale_copies[scale_idx]

            if mode == "balanced":
                base_weight = 1.0
            elif mode == "geometric":
                base_weight = 2.0 ** (scale_idx / (self.num_base_scales - 1))
            elif mode == "top_heavy":
                if scale_idx >= self.num_base_scales - 2:
                    base_weight = 1.5
                else:
                    base_weight = 0.5
            else:
                base_weight = 1.0

            copy_weight = base_weight / num_copies
            for _ in range(num_copies):
                weights.append(copy_weight)

        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights.clamp(min=self.scale_weight_floor)
        weights = weights / weights.sum()

        return weights

    def forward(
            self,
            features: torch.Tensor,
            anchors_dict: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:

        if self.fusion is not None:
            fusion_weights = F.softmax(self.fusion(features), dim=-1)
            fusion_weights = fusion_weights.clamp(min=self.scale_weight_floor)
            fusion_weights = fusion_weights / fusion_weights.sum(dim=-1, keepdim=True)
        else:
            fusion_weights = self.fixed_weights.to(features.device).unsqueeze(0).expand(features.shape[0], -1)

        scale_logits = [None] * self.num_heads
        scale_features = [None] * self.num_heads

        for i, head in enumerate(self.heads):
            scale, copy_idx = self.head_scale_map[i]
            scale_logits[i], scale_features[i] = head(features, anchors_dict[scale])

        stacked_logits = torch.stack(scale_logits, dim=1)
        combined = torch.einsum('bs,bsc->bc', fusion_weights, stacked_logits)

        return combined, scale_logits, scale_features, fusion_weights


class CollectiveCrystalHead(nn.Module):
    """Collective multi-scale head with shared anchor space."""

    def __init__(self, config: DavidBeansV2Config):
        super().__init__()
        self.config = config
        self.scales = config.scales
        self.max_scale = max(config.scales)
        self.num_scales = len(config.scales)

        self.projections = nn.ModuleList()
        self.fingerprints = nn.ParameterList()
        self.weight_biases = nn.ParameterList()

        for i, scale in enumerate(config.scales):
            theta = i * config.copy_theta_step

            if config.use_belly:
                proj = ConfigurableBelly(
                    input_dim=config.dim,
                    output_dim=scale,
                    expand=config.belly_expand,
                    num_layers=config.belly_layers,
                    use_residual=config.belly_residual,
                    dropout=config.dropout,
                )
            else:
                proj = nn.Linear(config.dim, scale, bias=False)

            self.projections.append(proj)

            fingerprint = compute_cantor_fingerprint(scale, self.max_scale, theta)
            self.fingerprints.append(nn.Parameter(fingerprint, requires_grad=True))

            self.weight_biases.append(nn.Parameter(torch.ones(1)))

        self.fusion_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)

    def forward(
        self,
        features: torch.Tensor,
        anchors: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        B = features.shape[0]

        scale_outputs = []
        scale_logits = []

        for i, (proj, fingerprint, bias) in enumerate(
            zip(self.projections, self.fingerprints, self.weight_biases)
        ):
            z_active = proj(features)
            z_active = F.normalize(z_active, dim=-1)

            z_padded = F.pad(z_active, (0, self.max_scale - z_active.shape[-1]))
            z_masked = z_padded * fingerprint.unsqueeze(0)
            z_biased = z_masked * bias

            scale_outputs.append(z_biased)

            anchors_norm = F.normalize(anchors, dim=-1)
            logits = torch.mm(z_biased, anchors_norm.T) / self.config.collective_temperature
            scale_logits.append(logits)

        fusion = F.softmax(self.fusion_weights, dim=0)
        fusion_expanded = fusion.unsqueeze(0).expand(B, -1)

        stacked_logits = torch.stack(scale_logits, dim=1)
        combined = torch.einsum('bs,bsc->bc', fusion_expanded, stacked_logits)

        return combined, scale_logits, scale_outputs, fusion_expanded


# ============================================================================
# DAVIDBEANS V2.3 BACKBONE
# ============================================================================

class BeansBackboneV2(nn.Module):
    """Backbone with wormhole routing and topological dropout."""

    def __init__(self, config: DavidBeansV2Config):
        super().__init__()
        self.config = config

        # Conv spine (optional)
        if config.use_spine:
            self.spine = ConvSpine(
                in_channels=config.in_channels,
                spine_channels=config.spine_channels,
                output_dim=config.dim
            )
        else:
            self.spine = None

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # Spatial dropout after patch embedding (optional)
        if config.use_topo_dropout and config.use_spatial_dropout and TOPO_DROPOUT_AVAILABLE:
            self.spatial_drop = SpatialTopologicalDropout(
                drop_prob=config.spatial_drop_prob,
                patch_size=config.spatial_patch_size,
            )
        else:
            self.spatial_drop = nn.Identity()

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
                    config=config,
                )
            )
            self.expert_layers.append(
                WormholeTessellationExpert(
                    dim=config.dim,
                    num_tiles=config.num_tiles,
                    num_wormholes=config.tile_wormholes,
                    config=config,
                )
            )

        self.norm = nn.LayerNorm(config.dim)

    def forward(
            self,
            x: torch.Tensor,
            return_routing: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[torch.Tensor]]:
        B = x.shape[0]

        # Get spine features if enabled
        spine_tokens = None
        spine_global = None
        if self.spine is not None:
            spine_global, spine_tokens = self.spine(x)

        # Patch embedding with optional spatial dropout
        x = self.patch_embed(x)  # [B, D, H, W]
        x = self.spatial_drop(x)  # Spatial topological dropout
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed

        routing_info = {'attention': [], 'expert': []} if return_routing else None

        for attn, expert in zip(self.attention_blocks, self.expert_layers):
            x, attn_routing = attn(x, spine_tokens=spine_tokens, return_routing=return_routing)
            if return_routing:
                routing_info['attention'].append(attn_routing)

            x, expert_routing = expert(x, return_routing=return_routing)
            if return_routing:
                routing_info['expert'].append(expert_routing)

        x = self.norm(x)

        return x, routing_info, spine_global


# ============================================================================
# DAVIDBEANS V2.3 FULL MODEL
# ============================================================================

class DavidBeansV2(nn.Module):
    """
    DavidBeans V2.3: Multi-scale classification with wormhole routing
    and structure-preserving topological dropout.
    """

    def __init__(self, config: DavidBeansV2Config):
        super().__init__()
        self.config = config

        self.backbone = BeansBackboneV2(config)

        if config.use_collective:
            self.head = CollectiveCrystalHead(config)
        else:
            self.head = MultiScaleCrystalHead(config)

        if config.use_collective:
            max_scale = max(config.scales)
            self.anchors = nn.Parameter(torch.randn(config.num_classes, max_scale) * 0.02)
        else:
            self.anchors = nn.ParameterDict({
                str(s): nn.Parameter(torch.randn(config.num_classes, s) * 0.02)
                for s in config.scales
            })

    def get_anchors_dict(self) -> Dict[int, torch.Tensor]:
        return {int(k): v for k, v in self.anchors.items()}

    def set_epoch(self, epoch: int):
        """
        Set current epoch for scheduled dropout warmup.
        Call this at the start of each epoch.
        """
        for block in self.backbone.attention_blocks:
            if hasattr(block.attn_drop, 'set_step'):
                step = epoch * self.config.topo_steps_per_epoch
                block.attn_drop.set_step(step)
            if hasattr(block.route_drop, 'set_step'):
                step = epoch * self.config.topo_steps_per_epoch
                block.route_drop.set_step(step)

    def reset_dropout_schedule(self):
        """Reset dropout schedules (call at start of training)."""
        for block in self.backbone.attention_blocks:
            if hasattr(block.attn_drop, 'reset_schedule'):
                block.attn_drop.reset_schedule()
            if hasattr(block.route_drop, 'reset_schedule'):
                block.route_drop.reset_schedule()

    def forward(
            self,
            x: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            return_routing: bool = False,
            return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:

        all_tokens, routing_info, spine_features = self.backbone(x, return_routing=return_routing)

        cls_token = all_tokens[:, 0, :]

        if self.config.use_collective:
            combined_logits, scale_logits, scale_features, fusion_weights = self.head(
                cls_token, self.anchors
            )
            anchors = {'collective': self.anchors}
        else:
            anchors = self.get_anchors_dict()
            combined_logits, scale_logits, scale_features, fusion_weights = self.head(
                cls_token, anchors
            )

        result = {
            'logits': combined_logits,
            'scale_logits': scale_logits,
            'scale_features': scale_features,
            'fusion_weights': fusion_weights,
            'features': cls_token
        }

        if return_routing and routing_info is not None:
            result['routing_info'] = routing_info

        if spine_features is not None:
            result['spine_features'] = spine_features

        if return_loss and targets is not None:
            losses = self._compute_losses(
                combined_logits, scale_logits, scale_features,
                all_tokens, anchors, targets
            )
            result['losses'] = losses

        return result

    def _compute_losses(
        self,
        combined_logits: torch.Tensor,
        scale_logits: List[torch.Tensor],
        scale_features: List[torch.Tensor],
        all_tokens: torch.Tensor,
        anchors: Dict[int, torch.Tensor],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        losses['ce'] = F.cross_entropy(combined_logits, targets)

        for i, logits in enumerate(scale_logits):
            scale, copy_idx = self.head.head_scale_map[i] if hasattr(self.head, 'head_scale_map') else (self.config.scales[i], 0)
            key = f'ce_{scale}' if copy_idx == 0 else f'ce_{scale}_c{copy_idx}'
            losses[key] = F.cross_entropy(logits, targets)

        if not self.config.use_collective:
            patch_tokens = all_tokens[:, 1:, :]
            contrast_loss = self._compute_contrast_loss(
                patch_tokens, scale_features, anchors, targets
            )
            losses['contrast'] = contrast_loss
        else:
            losses['contrast'] = torch.tensor(0.0, device=combined_logits.device)

        losses['total'] = (
            losses['ce'] +
            self.config.contrast_weight * losses['contrast']
        )

        return losses

    def _compute_contrast_loss(
            self,
            patch_features: torch.Tensor,
            scale_features: List[torch.Tensor],
            anchors_dict: Dict[int, torch.Tensor],
            targets: torch.Tensor
    ) -> torch.Tensor:
        device = patch_features.device

        total_loss = torch.tensor(0.0, device=device)
        num_primary_scales = 0

        for i, scale_feat in enumerate(scale_features):
            scale, copy_idx = self.head.head_scale_map[i] if hasattr(self.head, 'head_scale_map') else (self.config.scales[i], 0)

            if copy_idx == 0:
                anchors = anchors_dict[scale]
                scale_feat_norm = F.normalize(scale_feat, dim=-1)
                anchors_norm = F.normalize(anchors, dim=-1)

                logits = torch.mm(scale_feat_norm, anchors_norm.T) / self.config.contrast_temperature
                total_loss = total_loss + F.cross_entropy(logits, targets)
                num_primary_scales += 1

        return total_loss / max(num_primary_scales, 1)

    def get_model_info(self) -> Dict:
        num_heads = len(self.head.heads) if hasattr(self.head, 'heads') else len(self.config.scales)

        # Get current dropout prob if scheduled
        current_drop = self.config.topo_drop_prob
        if self.config.use_topo_dropout:
            for block in self.backbone.attention_blocks:
                if hasattr(block.attn_drop, 'current_drop_prob'):
                    current_drop = block.attn_drop.current_drop_prob
                    break

        return {
            'name': 'DavidBeans-V2-Wormhole',
            'version': '2.3',
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
            'scale_copies': self.config.scale_copies,
            'num_scale_heads': num_heads,
            'weighting_mode': self.config.weighting_mode,
            'belly_layers': self.config.belly_layers,
            'belly_residual': self.config.belly_residual,
            'use_spine': self.config.use_spine,
            'use_collective': self.config.use_collective,
            'use_topo_dropout': self.config.use_topo_dropout,
            'topo_drop_prob': self.config.topo_drop_prob,
            'topo_warmup_epochs': self.config.topo_warmup_epochs,
            'current_drop_prob': current_drop,
            'use_spatial_dropout': self.config.use_spatial_dropout,
            'num_classes': self.config.num_classes,
            'num_patches': self.config.num_patches,
            'total_params': sum(p.numel() for p in self.parameters()),
        }

    def __repr__(self):
        info = self.get_model_info()
        spine_str = f"\n  Spine: {self.config.spine_channels}" if info['use_spine'] else ""
        collective_str = " (collective)" if info['use_collective'] else ""
        copies_str = f", copies={info['scale_copies']}" if info['scale_copies'] else ""

        if info['use_topo_dropout']:
            topo_str = f" [TopoDrop p={info['topo_drop_prob']}, warmup={info['topo_warmup_epochs']}ep]"
        else:
            topo_str = ""

        spatial_str = " [SpatialDrop]" if info['use_spatial_dropout'] else ""

        return (
            f"DavidBeans-V2.3-Wormhole(\n"
            f"  Vision: {info['image_size']}px → {info['num_patches']} patches\n"
            f"  Backbone: {info['num_layers']} layers, {info['dim']}d, {info['num_heads']} heads\n"
            f"  Wormholes: {info['num_wormholes']} per position, mode={info['wormhole_mode']}\n"
            f"  Tessellation: {info['num_tiles']} tiles × {info['tile_wormholes']} wormholes\n"
            f"  Scales: {info['scales']}{copies_str}{collective_str}\n"
            f"  Weighting: {info['weighting_mode']}, belly={info['belly_layers']}L{topo_str}{spatial_str}\n"
            f"  Parameters: {info['total_params']:,}{spine_str}\n"
            f")"
        )


# ============================================================================
# QUICK TEST
# ============================================================================

def test_v2():
    """Quick functionality test."""
    print("=" * 60)
    print("DavidBeans V2.3 - Topological Dropout Test")
    print("=" * 60)

    # Test with topological dropout
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
        num_classes=100,
        belly_layers=2,
        # Topological dropout settings
        use_topo_dropout=True,
        topo_drop_prob=0.15,
        topo_warmup_epochs=35,
        topo_min_routes_keep=2,
        use_spatial_dropout=True,
        spatial_drop_prob=0.1,
    )

    model = DavidBeansV2(config)
    print(model)

    # Test forward
    x = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 100, (4,))

    # Test training mode
    model.train()
    result = model(x, targets=targets, return_routing=True)

    print(f"\nTraining mode:")
    print(f"  Logits: {result['logits'].shape}")
    print(f"  Total loss: {result['losses']['total'].item():.4f}")

    # Check dropout schedule
    info = model.get_model_info()
    print(f"  Current drop prob: {info['current_drop_prob']:.4f}")

    # Simulate epoch progression
    print(f"\nDropout schedule progression:")
    for epoch in [0, 10, 20, 35, 50]:
        model.set_epoch(epoch)
        info = model.get_model_info()
        print(f"  Epoch {epoch:2d}: drop_prob = {info['current_drop_prob']:.4f}")

    # Test eval mode
    model.eval()
    result_eval = model(x, targets=targets)
    print(f"\nEval mode:")
    print(f"  Total loss: {result_eval['losses']['total'].item():.4f}")

    # Test without topological dropout
    print("\n" + "=" * 60)
    print("Testing without topological dropout...")

    config_standard = DavidBeansV2Config(
        image_size=32,
        patch_size=4,
        dim=256,
        num_layers=2,
        num_heads=4,
        num_wormholes=8,
        num_tiles=8,
        tile_wormholes=3,
        scales=[64, 128, 256],
        num_classes=100,
        use_topo_dropout=False,
        use_spatial_dropout=False,
    )

    model_standard = DavidBeansV2(config_standard)
    print(model_standard)

    result_standard = model_standard(x, targets=targets)
    print(f"  Total loss: {result_standard['losses']['total'].item():.4f}")

    print("\n✓ All V2.3 tests passed!")

    return model, config


if __name__ == "__main__":
    test_v2()