"""
DavidBeans V2: Wormhole Routing Integration
============================================

Key upgrades from V1:
1. LearnedWormholeRouter: Content-aware routing (replaces fixed Cantor)
2. WormholeAttentionBlock: Learned sparse attention patterns
3. WormholeTessellationExpert: Tiles connect via learned wormholes
4. Proper gradient flow through all routing decisions

V2.1 Additions:
- Configurable belly depth with residual connections
- Scale weighting modes (balanced, geometric, top_heavy, learned)
- Redundant multiscale with theta differentiation
- Collective shared space option with fingerprint masks
- Conv spine for persistent spatial coherence

V2.2 Additions:
- FractalRegularizer: CantorGate activation + TopologicalDropout
- Router-aware regularization (importance-weighted dropout)
- Structure-preserving dropout that drops routes, not neurons

Based on experimental validation:
- Permutation recovery: 100% routing accuracy
- Patch matching: 99.6% correspondence learning
- Key insight: When routing IS necessary, routing learns structure

Author: AbstractPhil
Date: November 30, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except:
    TRITON_AVAILABLE = False

# Import FractalRegularizer - adjust path as needed
try:
    if not TRITON_AVAILABLE:
        print("Triton not available, attempting to use slower regularizer")
        from geofractal.function.fractal_regularizer import FractalRegularizer, CantorGate, TopologicalDropout
    else:
        print("Triton available, using optimized FractalRegularizer")
        from geofractal.function.fractal_regularizer_triton import FractalRegularizer, CantorGate, TopologicalDropout
    FRACTAL_AVAILABLE = True

except ImportError:
    FRACTAL_AVAILABLE = False
    print("Warning: FractalRegularizer not available, falling back to standard dropout")


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

    # Crystal head - scales
    scales: List[int] = field(default_factory=lambda: [64, 128, 256, 384, 512])
    num_classes: int = 100

    # Crystal head - belly configuration
    use_belly: bool = True
    belly_expand: float = 2.0
    belly_layers: int = 2  # Number of layers in belly (2-4)
    belly_residual: bool = False  # Use residual connections in deep belly

    # Crystal head - scale weighting
    weighting_mode: str = "learned"  # "balanced", "geometric", "top_heavy", "learned"
    scale_weight_floor: float = 0.1  # Minimum weight per scale (prevents collapse)

    # Crystal head - redundant scales
    scale_copies: Optional[List[int]] = None  # e.g., [2, 1, 1, 1, 1] for 2 copies of first scale
    copy_theta_step: float = 0.15  # Theta rotation between copies

    # Crystal head - collective mode
    use_collective: bool = False  # Use shared anchor space with pad/mask
    collective_temperature: float = 0.07

    # Conv spine
    use_spine: bool = False
    spine_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    spine_cross_attn: bool = True  # Per-layer cross-attention to spine
    spine_gate_init: float = 0.0  # Initial gate value (0 = start gated off)

    # Loss weights
    contrast_temperature: float = 0.07
    contrast_weight: float = 0.5
    routing_weight: float = 0.0  # Optional: auxiliary routing loss

    # Regularization
    dropout: float = 0.1
    pooling: str = "cls"

    # Fractal Regularizer (V2.2)
    use_fractal_reg: bool = False  # Use FractalRegularizer instead of dropout
    fractal_num_levels: int = 4    # Cantor staircase levels (2^n stairs)
    fractal_drop_prob: float = 0.1 # Topological dropout probability
    fractal_min_routes: int = 2    # Minimum routes to keep during dropout

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

        # Validate fractal reg settings
        if self.use_fractal_reg and not FRACTAL_AVAILABLE:
            print("Warning: use_fractal_reg=True but FractalRegularizer not available, disabling")
            self.use_fractal_reg = False


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
# CONV SPINE - Persistent Spatial Coherence
# ============================================================================

class ConvSpine(nn.Module):
    """
    Persistent wide conv pathway that runs parallel to transformer.
    Provides spatial coherence that patches can't forget.
    """

    def __init__(
        self,
        in_channels: int = 3,
        spine_channels: List[int] = [64, 128, 256],
        output_dim: int = 512,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Progressive conv stages (maintains spatial structure)
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

        # Final projection to match transformer dim
        self.proj = nn.Conv2d(spine_channels[-1], output_dim, 1)

        # Learnable spatial pooling (keeps gradient flow)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            spine_global: [B, D] global conv features
            spine_tokens: [B, H*W, D] flattened spatial features for cross-attn
        """
        for stage in self.stages:
            x = stage(x)
            x = F.max_pool2d(x, 2)  # Downsample

        # Final projection
        x = self.proj(x)  # [B, D, H', W']

        # Global feature
        spine_global = self.pool(x).flatten(1)  # [B, D]

        # Flatten for cross-attention: [B, D, H, W] -> [B, H*W, D]
        B, D, H, W = x.shape
        spine_tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, D]

        return spine_global, spine_tokens


# ============================================================================
# WORMHOLE ATTENTION BLOCK
# ============================================================================

class WormholeAttentionBlock(nn.Module):
    """
    Attention block with learned wormhole routing.

    CLS token: Dense attention to all patches
    Patches: Sparse attention via learned wormholes
    Optional: Cross-attention to conv spine features

    V2.2: Optional FractalRegularizer for structure-preserving dropout
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
            dropout: float = 0.1,
            use_spine_cross_attn: bool = False,
            spine_gate_init: float = 0.0,
            # Fractal regularizer options
            use_fractal_reg: bool = False,
            fractal_num_levels: int = 4,
            fractal_drop_prob: float = 0.1,
            fractal_min_routes: int = 2
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_patches = num_patches
        self.num_wormholes = num_wormholes
        self.use_spine_cross_attn = use_spine_cross_attn
        self.use_fractal_reg = use_fractal_reg and FRACTAL_AVAILABLE

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

        if self.use_fractal_reg:
            # Use CantorGate instead of GELU, no dropout in MLP
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                CantorGate(dim=mlp_dim, num_levels=fractal_num_levels),
                nn.Linear(mlp_dim, dim),
            )

            # FractalRegularizer for post-routing regularization
            self.fractal_reg = FractalRegularizer(
                dim=self.head_dim,  # Per-head dimension
                num_routes=num_wormholes,
                num_levels=fractal_num_levels,
                drop_prob=fractal_drop_prob,
                min_routes_keep=fractal_min_routes
            )

            # No separate dropout layers needed
            self.attn_drop = nn.Identity()
            self.proj_drop = nn.Identity()
        else:
            # Standard MLP with dropout
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(dropout)
            )

            self.fractal_reg = None
            self.attn_drop = nn.Dropout(dropout)
            self.proj_drop = nn.Dropout(dropout)

        # Spine cross-attention (optional)
        if use_spine_cross_attn:
            self.spine_cross_attn = nn.MultiheadAttention(
                dim, num_heads=max(1, num_heads // 2),
                batch_first=True, dropout=dropout
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

        attn_patches = F.softmax(scores_patches, dim=-1)

        # Apply fractal regularization if enabled
        if self.use_fractal_reg and self.fractal_reg is not None:
            # Apply FractalRegularizer to gathered values before weighted sum
            # V_gathered: [B, H, P, K, head_dim]
            # Reshape for fractal reg: [B*H*P, K, head_dim]
            V_shape = V_gathered.shape
            V_flat = V_gathered.reshape(-1, V_shape[-2], V_shape[-1])

            # Get importance from attention weights for this batch
            # attn_patches: [B, H, P, K] -> mean over B, H, P to get per-route importance
            importance = attn_patches.mean(dim=(0, 1, 2))  # [K]

            V_reg = self.fractal_reg(V_flat, route_dim=-2, routing_weights=importance.unsqueeze(0))
            V_gathered = V_reg.reshape(V_shape)
        else:
            attn_patches = self.attn_drop(attn_patches)

        out_patches = torch.einsum('bhpk,bhpkd->bhpd', attn_patches, V_gathered)

        # Combine and project
        out = torch.cat([out_cls, out_patches], dim=2)
        out = self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, S, D)))

        # Residual
        x = x + out

        # === SPINE CROSS-ATTENTION (optional) ===
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

    Each tile processes its slice + gathered wormhole context.

    V2.2: Optional CantorGate activation
    """

    def __init__(
            self,
            dim: int,
            num_tiles: int,
            num_wormholes: int = 4,
            temperature: float = 0.5,
            dropout: float = 0.1,
            use_fractal_reg: bool = False,
            fractal_num_levels: int = 4
    ):
        super().__init__()
        self.dim = dim
        self.num_tiles = num_tiles
        self.tile_dim = dim // num_tiles
        self.num_wormholes = min(num_wormholes, num_tiles - 1)
        self.temperature = temperature
        self.use_fractal_reg = use_fractal_reg and FRACTAL_AVAILABLE

        # Tile router (learns which tiles connect)
        self.tile_query = nn.Linear(self.tile_dim, self.tile_dim)
        self.tile_key = nn.Linear(self.tile_dim, self.tile_dim)

        # Process: self + wormhole context
        context_dim = self.tile_dim * (1 + self.num_wormholes)
        hidden_dim = self.tile_dim * 2

        if self.use_fractal_reg:
            self.processor = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                CantorGate(dim=hidden_dim, num_levels=fractal_num_levels),
                nn.Linear(hidden_dim, self.tile_dim)
            )
        else:
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
# CONFIGURABLE BELLY PROJECTION
# ============================================================================

class ConfigurableBelly(nn.Module):
    """
    Multi-layer belly projection with optional residual connections.

    Supports 2-4 layers with configurable expansion and residual.

    V2.2: Optional CantorGate activation
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expand: float = 2.0,
        num_layers: int = 2,
        use_residual: bool = False,
        dropout: float = 0.1,
        use_fractal_reg: bool = False,
        fractal_num_levels: int = 4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_fractal_reg = use_fractal_reg and FRACTAL_AVAILABLE

        belly_dim = int(output_dim * expand)

        layers = []

        # Choose activation
        if self.use_fractal_reg:
            def make_activation(dim):
                return CantorGate(dim=dim, num_levels=fractal_num_levels)
        else:
            def make_activation(dim):
                return nn.Sequential(nn.GELU(), nn.Dropout(dropout))

        if num_layers == 2:
            layers = [
                nn.Linear(input_dim, belly_dim),
                make_activation(belly_dim),
                nn.Linear(belly_dim, output_dim, bias=False)
            ]
        elif num_layers == 3:
            layers = [
                nn.Linear(input_dim, belly_dim),
                make_activation(belly_dim),
                nn.Linear(belly_dim, belly_dim),
                make_activation(belly_dim),
                nn.Linear(belly_dim, output_dim, bias=False)
            ]
        elif num_layers == 4:
            layers = [
                nn.Linear(input_dim, belly_dim),
                make_activation(belly_dim),
                nn.Linear(belly_dim, belly_dim),
                make_activation(belly_dim),
                nn.Linear(belly_dim, belly_dim),
                make_activation(belly_dim),
                nn.Linear(belly_dim, output_dim, bias=False)
            ]

        self.layers = nn.Sequential(*layers)

        # Residual projection if dimensions differ and residual enabled
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
    """
    Compute unique fingerprint mask with theta rotation.
    Prevents redundant scales from learning identical representations.

    Args:
        active_dim: Active dimension for this scale
        max_dim: Maximum scale dimension (for padding)
        theta: Rotation angle for differentiation

    Returns:
        fingerprint: [max_dim] soft mask
    """
    positions = torch.arange(max_dim).float()

    # Cantor pairing for base pattern
    grid_size = int(max_dim ** 0.5) + 1
    x = positions % grid_size
    y = positions // grid_size
    cantor_z = ((x + y) * (x + y + 1)) / 2 + y
    cantor_z = cantor_z / cantor_z.max().clamp(min=1)

    # Theta rotation (unique per copy)
    rotated = cantor_z * math.cos(theta) + (positions / max_dim) * math.sin(theta)

    # Create soft mask: active region = 1, padded = decay
    mask = torch.zeros(max_dim)
    mask[:active_dim] = 1.0

    # Blend: fingerprint modulates the mask
    fingerprint = mask * (0.5 + 0.5 * torch.tanh(rotated))

    return fingerprint


# ============================================================================
# MULTI-SCALE CRYSTAL HEAD
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
            theta: float = 0.0,  # For redundant scale differentiation
            copy_index: int = 0,  # Which copy this is (0 = primary)
            use_fractal_reg: bool = False,
            fractal_num_levels: int = 4
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
                use_fractal_reg=use_fractal_reg,
                fractal_num_levels=fractal_num_levels
            )
        else:
            self.projection = nn.Linear(input_dim, crystal_dim, bias=False)

        # Learnable fingerprint for redundant copies
        if theta != 0.0:
            fingerprint = compute_cantor_fingerprint(crystal_dim, crystal_dim, theta)
            self.register_buffer('fingerprint', fingerprint)
        else:
            self.fingerprint = None

    def forward(self, features: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.projection(features)

        # Apply fingerprint modulation for redundant copies
        if self.fingerprint is not None:
            z = z * self.fingerprint.unsqueeze(0)

        z = F.normalize(z, dim=-1)
        anchors_norm = F.normalize(anchors, dim=-1)
        logits = torch.mm(z, anchors_norm.T) / self.temperature
        return logits, z


class MultiScaleCrystalHead(nn.Module):
    """
    Multi-scale projection with learned fusion.

    Supports:
    - Multiple weighting modes (balanced, geometric, top_heavy, learned)
    - Redundant scales with theta differentiation
    - Collective shared space option
    """

    def __init__(self, config: DavidBeansV2Config):
        super().__init__()
        self.config = config
        self.scales = config.scales
        self.num_base_scales = len(config.scales)
        self.weighting_mode = config.weighting_mode
        self.scale_weight_floor = config.scale_weight_floor

        # Handle redundant scale copies
        scale_copies = config.scale_copies or [1] * len(config.scales)
        assert len(scale_copies) == len(config.scales), \
            f"scale_copies length ({len(scale_copies)}) must match scales length ({len(config.scales)})"

        # Build heads (including copies)
        self.heads = nn.ModuleList()
        self.head_scale_map = []  # Maps head index to (scale, copy_index)

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
                    use_fractal_reg=config.use_fractal_reg,
                    fractal_num_levels=config.fractal_num_levels
                )
                self.heads.append(head)
                self.head_scale_map.append((scale, copy_idx))

        self.num_heads = len(self.heads)

        # Compute fixed scale weights based on mode
        if config.weighting_mode == "learned":
            # Learned fusion network
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

    def _compute_fixed_weights(
        self,
        mode: str,
        scale_copies: List[int]
    ) -> torch.Tensor:
        """Compute fixed scale weights based on weighting mode."""
        weights = []

        for scale_idx, scale in enumerate(self.scales):
            num_copies = scale_copies[scale_idx]

            if mode == "balanced":
                base_weight = 1.0
            elif mode == "geometric":
                # Larger scales get more weight
                base_weight = 2.0 ** (scale_idx / (self.num_base_scales - 1))
            elif mode == "top_heavy":
                # Only top scales get significant weight
                if scale_idx >= self.num_base_scales - 2:
                    base_weight = 1.5
                else:
                    base_weight = 0.5
            else:
                base_weight = 1.0

            # Distribute weight among copies
            copy_weight = base_weight / num_copies
            for _ in range(num_copies):
                weights.append(copy_weight)

        weights = torch.tensor(weights, dtype=torch.float32)

        # Apply floor and normalize
        weights = weights.clamp(min=self.scale_weight_floor)
        weights = weights / weights.sum()

        return weights

    def forward(
            self,
            features: torch.Tensor,
            anchors_dict: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:

        # Compute fusion weights
        if self.fusion is not None:
            fusion_weights = F.softmax(self.fusion(features), dim=-1)
            # Apply floor
            fusion_weights = fusion_weights.clamp(min=self.scale_weight_floor)
            fusion_weights = fusion_weights / fusion_weights.sum(dim=-1, keepdim=True)
        else:
            # Use fixed weights, expand for batch
            fusion_weights = self.fixed_weights.to(features.device).unsqueeze(0).expand(features.shape[0], -1)

        # Process all heads
        scale_logits = [None] * self.num_heads
        scale_features = [None] * self.num_heads

        for i, head in enumerate(self.heads):
            scale, copy_idx = self.head_scale_map[i]
            scale_logits[i], scale_features[i] = head(features, anchors_dict[scale])

        # Fused weighted combination
        stacked_logits = torch.stack(scale_logits, dim=1)
        combined = torch.einsum('bs,bsc->bc', fusion_weights, stacked_logits)

        return combined, scale_logits, scale_features, fusion_weights


# ============================================================================
# COLLECTIVE CRYSTAL HEAD (Alternative Mode)
# ============================================================================

class CollectiveCrystalHead(nn.Module):
    """
    Collective multi-scale head with shared anchor space.

    All scales project to max_scale dimension with pad/mask,
    competing in shared space with fingerprint differentiation.
    """

    def __init__(self, config: DavidBeansV2Config):
        super().__init__()
        self.config = config
        self.scales = config.scales
        self.max_scale = max(config.scales)
        self.num_scales = len(config.scales)

        # Per-scale projections (to active subspace)
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
                    use_fractal_reg=config.use_fractal_reg,
                    fractal_num_levels=config.fractal_num_levels
                )
            else:
                proj = nn.Linear(config.dim, scale, bias=False)

            self.projections.append(proj)

            # Cantor fingerprint mask
            fingerprint = compute_cantor_fingerprint(scale, self.max_scale, theta)
            self.fingerprints.append(nn.Parameter(fingerprint, requires_grad=True))

            # Weight bias scalar (like GeoDavid)
            self.weight_biases.append(nn.Parameter(torch.ones(1)))

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)

    def forward(
        self,
        features: torch.Tensor,
        anchors: torch.Tensor  # Single shared anchor: [num_classes, max_scale]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        B = features.shape[0]

        scale_outputs = []
        scale_logits = []

        for i, (proj, fingerprint, bias) in enumerate(
            zip(self.projections, self.fingerprints, self.weight_biases)
        ):
            # Project to active subspace
            z_active = proj(features)  # [B, active_dim]
            z_active = F.normalize(z_active, dim=-1)

            # Pad to max_scale
            z_padded = F.pad(z_active, (0, self.max_scale - z_active.shape[-1]))

            # Apply fingerprint mask
            z_masked = z_padded * fingerprint.unsqueeze(0)

            # Apply weight bias
            z_biased = z_masked * bias

            scale_outputs.append(z_biased)

            # Compute logits against shared anchors
            anchors_norm = F.normalize(anchors, dim=-1)
            logits = torch.mm(z_biased, anchors_norm.T) / self.config.collective_temperature
            scale_logits.append(logits)

        # Stack for collective processing
        stacked = torch.stack(scale_outputs, dim=1)  # [B, num_scales, max_scale]

        # Fusion with learned weights
        fusion = F.softmax(self.fusion_weights, dim=0)
        fusion_expanded = fusion.unsqueeze(0).expand(B, -1)

        # Combined logits
        stacked_logits = torch.stack(scale_logits, dim=1)  # [B, num_scales, num_classes]
        combined = torch.einsum('bs,bsc->bc', fusion_expanded, stacked_logits)

        return combined, scale_logits, scale_outputs, fusion_expanded


# ============================================================================
# DAVIDBEANS V2 BACKBONE
# ============================================================================

class BeansBackboneV2(nn.Module):
    """Backbone with wormhole routing throughout and optional conv spine."""

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
                    dropout=config.dropout,
                    use_spine_cross_attn=config.use_spine and config.spine_cross_attn,
                    spine_gate_init=config.spine_gate_init,
                    use_fractal_reg=config.use_fractal_reg,
                    fractal_num_levels=config.fractal_num_levels,
                    fractal_drop_prob=config.fractal_drop_prob,
                    fractal_min_routes=config.fractal_min_routes
                )
            )
            self.expert_layers.append(
                WormholeTessellationExpert(
                    dim=config.dim,
                    num_tiles=config.num_tiles,
                    num_wormholes=config.tile_wormholes,
                    dropout=config.dropout,
                    use_fractal_reg=config.use_fractal_reg,
                    fractal_num_levels=config.fractal_num_levels
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

        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed

        routing_info = {'attention': [], 'expert': []} if return_routing else None

        # Interleaved attention + expert layers
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
# DAVIDBEANS V2 FULL MODEL
# ============================================================================

class DavidBeansV2(nn.Module):
    """
    DavidBeans V2: Multi-scale classification with wormhole routing.

    V2.2: Optional FractalRegularizer for structure-preserving dropout
    """

    def __init__(self, config: DavidBeansV2Config):
        super().__init__()
        self.config = config

        # Backbone
        self.backbone = BeansBackboneV2(config)

        # Crystal head (multi-scale or collective)
        if config.use_collective:
            self.head = CollectiveCrystalHead(config)
        else:
            self.head = MultiScaleCrystalHead(config)

        # Class anchors per scale
        if config.use_collective:
            # Single shared anchor for collective mode
            max_scale = max(config.scales)
            self.anchors = nn.Parameter(torch.randn(config.num_classes, max_scale) * 0.02)
        else:
            # Per-scale anchors
            self.anchors = nn.ParameterDict({
                str(s): nn.Parameter(torch.randn(config.num_classes, s) * 0.02)
                for s in config.scales
            })

    def get_anchors_dict(self) -> Dict[int, torch.Tensor]:
        """Get anchors dictionary for non-collective mode."""
        return {int(k): v for k, v in self.anchors.items()}

    def forward(
            self,
            x: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            return_routing: bool = False,
            return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:

        # Backbone forward
        all_tokens, routing_info, spine_features = self.backbone(x, return_routing=return_routing)

        # Get CLS token
        cls_token = all_tokens[:, 0, :]

        # Crystal head forward
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
        """Compute all loss components."""
        losses = {}

        # CE loss on combined logits
        losses['ce'] = F.cross_entropy(combined_logits, targets)

        # Per-scale CE (for logging)
        for i, logits in enumerate(scale_logits):
            scale, copy_idx = self.head.head_scale_map[i] if hasattr(self.head, 'head_scale_map') else (self.config.scales[i], 0)
            key = f'ce_{scale}' if copy_idx == 0 else f'ce_{scale}_c{copy_idx}'
            losses[key] = F.cross_entropy(logits, targets)

        # Contrastive loss
        if not self.config.use_collective:
            patch_tokens = all_tokens[:, 1:, :]
            contrast_loss = self._compute_contrast_loss(
                patch_tokens, scale_features, anchors, targets
            )
            losses['contrast'] = contrast_loss
        else:
            losses['contrast'] = torch.tensor(0.0, device=combined_logits.device)

        # Total loss
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
        """Batched contrastive loss across scales."""
        device = patch_features.device

        total_loss = torch.tensor(0.0, device=device)
        num_primary_scales = 0

        for i, scale_feat in enumerate(scale_features):
            scale, copy_idx = self.head.head_scale_map[i] if hasattr(self.head, 'head_scale_map') else (self.config.scales[i], 0)

            # Only compute contrast for primary copies
            if copy_idx == 0:
                anchors = anchors_dict[scale]
                scale_feat_norm = F.normalize(scale_feat, dim=-1)
                anchors_norm = F.normalize(anchors, dim=-1)

                logits = torch.mm(scale_feat_norm, anchors_norm.T) / self.config.contrast_temperature
                total_loss = total_loss + F.cross_entropy(logits, targets)
                num_primary_scales += 1

        return total_loss / max(num_primary_scales, 1)

    def get_model_info(self) -> Dict:
        # Count total heads including copies
        num_heads = len(self.head.heads) if hasattr(self.head, 'heads') else len(self.config.scales)

        return {
            'name': 'DavidBeans-V2-Wormhole',
            'version': '2.2',
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
            'use_fractal_reg': self.config.use_fractal_reg,
            'num_classes': self.config.num_classes,
            'num_patches': self.config.num_patches,
            'total_params': sum(p.numel() for p in self.parameters()),
        }

    def __repr__(self):
        info = self.get_model_info()
        spine_str = f"\n  Spine: {self.config.spine_channels}" if info['use_spine'] else ""
        collective_str = " (collective)" if info['use_collective'] else ""
        copies_str = f", copies={info['scale_copies']}" if info['scale_copies'] else ""
        fractal_str = " [FractalReg]" if info['use_fractal_reg'] else ""

        return (
            f"DavidBeans-V2.2-Wormhole(\n"
            f"  Vision: {info['image_size']}px → {info['num_patches']} patches\n"
            f"  Backbone: {info['num_layers']} layers, {info['dim']}d, {info['num_heads']} heads\n"
            f"  Wormholes: {info['num_wormholes']} per position, mode={info['wormhole_mode']}\n"
            f"  Tessellation: {info['num_tiles']} tiles × {info['tile_wormholes']} wormholes\n"
            f"  Scales: {info['scales']}{copies_str}{collective_str}\n"
            f"  Weighting: {info['weighting_mode']}, belly={info['belly_layers']}L{fractal_str}\n"
            f"  Parameters: {info['total_params']:,}{spine_str}\n"
            f")"
        )


# ============================================================================
# QUICK TEST
# ============================================================================

def test_v2():
    """Quick functionality test."""
    print("=" * 60)
    print("DavidBeans V2.2 - Wormhole Routing Test")
    print("=" * 60)

    # Test basic config
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
        belly_layers=3,
        belly_residual=True,
        weighting_mode="geometric"
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
    print(f"  Fusion weights: {result['fusion_weights'][0]}")

    # Test with FractalRegularizer if available
    if FRACTAL_AVAILABLE:
        print("\n" + "=" * 60)
        print("Testing FractalRegularizer...")

        config_fractal = DavidBeansV2Config(
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
            use_fractal_reg=True,
            fractal_num_levels=4,
            fractal_drop_prob=0.1,
            fractal_min_routes=2
        )

        model_fractal = DavidBeansV2(config_fractal)
        print(model_fractal)

        result_fractal = model_fractal(x, targets=targets)
        print(f"  With FractalReg - Loss: {result_fractal['losses']['total'].item():.4f}")
    else:
        print("\n  FractalRegularizer not available, skipping test")

    # Test with spine
    print("\n" + "=" * 60)
    print("Testing conv spine...")

    config3 = DavidBeansV2Config(
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
        use_spine=True,
        spine_channels=[32, 64, 128],
        spine_cross_attn=True
    )

    model3 = DavidBeansV2(config3)
    print(model3)

    result3 = model3(x, targets=targets)
    print(f"  Has spine features: {'spine_features' in result3}")
    if 'spine_features' in result3:
        print(f"  Spine features shape: {result3['spine_features'].shape}")

    print("\n✓ All V2.2 tests passed!")

    return model, config


if __name__ == "__main__":
    test_v2()