"""
geofractal.router.prefab.geometric_conv_tower_builder
=====================================================

Convolutional tower variants with BATCHED collective processing.
Now with WalkerFusion support for static and learnable fusion.

Each tower is a TorchComponent-based stage system.
ConvTowerCollective groups towers by type for batched forward.

Quick Construction:
    tower = quick_conv_tower('wide_resnet')
    pair = quick_conv_pair('wide_resnet', 'frequency', use_inception=True)
    collective = quick_conv_collective(['wide_resnet', 'frequency', 'squeeze_excite'])

Multi-Channel Latent Support:
    Designed for VAE latent processing (Flux, SD, etc.):

    - FlexibleInputComponent: Handles [B, C, H, W] spatial OR [B, L, D] sequence
    - ChannelMixerBlock: Cross-channel attention for multi-channel latents
    - MultiScaleConvBlock: Local/regional/global feature extraction
    - SpatialToOpinionComponent: Configurable pooling (adaptive/attention/multiscale)

    Example for Flux VAE (16 channels):
        configs = preset_flux_vae_towers()
        collective = build_conv_collective(configs, dim=256, spatial_size=32)

        # Direct spatial input
        latents = vae.encode(images)  # [B, 16, 32, 32]
        fused, opinions = collective(latents)

Tower types:
    - DepthTower: Multi-scale dilated convolutions
    - FrequencyTower: FFT-based frequency domain processing
    - ColorPatternTower: Interpolated IN/BN normalization
    - CoarseFineTower: Parallel resolution streams
    - WideResNetTower: Wide residual blocks
    - BottleneckTower: ResNet-style bottleneck blocks
    - SqueezeExciteTower: SE blocks with channel attention
    - InvertedBottleneckTower: MobileNetV2-style
    - SpatialAttentionTower: Spatial self-attention

Config Options:
    - in_channels: Input channels for spatial mode (e.g., 16 for Flux, 4 for SD)
    - input_mode: 'spatial', 'sequence', 'sequence_pooled', or 'auto'
    - pool_mode: 'adaptive', 'attention', or 'multiscale'
    - use_channel_mixer: Enable cross-channel attention
    - use_multiscale: Enable MultiScaleConvBlock between stages
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union
from enum import Enum
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.base_router import BaseRouter
from geofractal.router.base_tower import BaseTower
from geofractal.router.wide_router import WideRouter
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.components.address_component import AddressComponent
from geofractal.router.components.fusion_component import AdaptiveFusion
from geofractal.router.components.walker_component import (
    WalkerFusion,
    WalkerInception,
)


# =============================================================================
# CONV TOWER TYPE ENUM
# =============================================================================

class ConvTowerType(str, Enum):
    DEPTH = "depth"              # Multi-scale dilated (legacy alias)
    DILATED = "dilated"          # Explicit dilated convolutions
    FREQUENCY = "frequency"      # Spatial frequency filtering
    COLOR_PATTERN = "color_pattern"  # Interpolated IN/BN normalization
    COARSE_FINE = "coarse_fine"  # Parallel resolution streams
    WIDE_RESNET = "wide_resnet"  # Wide residual blocks
    BOTTLENECK = "bottleneck"    # Bottleneck residual blocks (ResNet-style)
    SQUEEZE_EXCITE = "squeeze_excite"  # SE blocks with channel attention
    INVERTED_BOTTLENECK = "inverted_bottleneck"  # MobileNetV2-style
    SPATIAL_ATTENTION = "spatial_attention"  # Spatial self-attention


class FusionType(str, Enum):
    """Fusion types for collectives."""
    # Legacy
    ADAPTIVE = "adaptive"
    # Walker fusion types
    WALKER_STATIC = "walker_static"
    WALKER_INCEPTION = "walker_inception"
    WALKER_SHIVA = "walker_shiva"
    WALKER_SLERP = "walker_slerp"
    WALKER_LERP = "walker_lerp"
    WALKER_ZEUS = "walker_zeus"


WALKER_PRESET_MAP = {
    FusionType.WALKER_STATIC: 'shiva',
    FusionType.WALKER_INCEPTION: 'shiva',
    FusionType.WALKER_SHIVA: 'shiva',
    FusionType.WALKER_SLERP: 'slerp',
    FusionType.WALKER_LERP: 'lerp',
    FusionType.WALKER_ZEUS: 'zeus',
}


# =============================================================================
# CONV TOWER CONFIG
# =============================================================================

@dataclass
class ConvTowerConfig:
    """Configuration for a convolutional tower."""

    name: str
    tower_type: Union[ConvTowerType, str] = ConvTowerType.WIDE_RESNET

    dim: Optional[int] = None
    depth: Optional[int] = None
    fingerprint_dim: Optional[int] = None

    inverted: bool = False
    tower_params: Dict[str, Any] = field(default_factory=dict)

    # Multi-channel latent support (e.g., Flux VAE)
    in_channels: Optional[int] = None      # Input channels (e.g., 16 for Flux)
    input_mode: str = 'auto'               # 'spatial', 'sequence', 'sequence_pooled', 'auto'
    pool_mode: str = 'adaptive'            # 'adaptive', 'attention', 'multiscale'
    use_channel_mixer: bool = False        # Cross-channel attention
    use_multiscale: bool = True            # MultiScaleConvBlock vs standard

    def __post_init__(self):
        if isinstance(self.tower_type, str):
            self.tower_type = ConvTowerType(self.tower_type)

    @property
    def structure_signature(self) -> str:
        """Towers with same signature can be batched."""
        return self.tower_type.value


# =============================================================================
# CONV STAGE COMPONENTS (TorchComponent subclasses)
# =============================================================================

class DilatedConvComponent(TorchComponent):
    """Dilated convolution stage for multi-scale receptive fields."""

    def __init__(self, name: str, channels: int, dilation: int, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        self.dilation = dilation

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn2(self.conv2(out)))
        return out + x


class FrequencyComponent(TorchComponent):
    """
    Spatial-domain frequency filtering (compile-safe).

    Approximates FFT-based filtering using multi-scale convolutions:
    - Low freq: large receptive field (pooled)
    - Mid freq: medium kernel
    - High freq: residual (fine details)

    This is compile-safe (no complex tensors).
    """

    def __init__(self, name: str, channels: int, spatial_size: int, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        self.spatial_size = spatial_size

        # Multi-scale frequency approximation
        # Low frequency path (large receptive field)
        self.low_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.low_conv = nn.Conv2d(channels, channels, 1)

        # Mid frequency path
        self.mid_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.mid_mix = nn.Conv2d(channels, channels, 1)

        # High frequency is residual (original - smoothed)
        self.high_conv = nn.Conv2d(channels, channels, 1)

        # Learnable frequency mixing weights
        self.freq_weights = nn.Parameter(torch.ones(3) / 3)

        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        # Normalize mixing weights
        w = F.softmax(self.freq_weights, dim=0)

        # Low frequency (smoothed)
        low = self.low_conv(self.low_pool(x))

        # Mid frequency (edges/textures)
        mid = self.mid_mix(self.mid_conv(x))

        # High frequency (fine detail = original - smoothed)
        high = self.high_conv(x - self.low_pool(x))

        # Weighted combination
        out = w[0] * low + w[1] * mid + w[2] * high

        return F.gelu(self.norm(out)) + x


class InterpolatedNormComponent(TorchComponent):
    """Interpolated IN/BN normalization for texture patterns."""

    def __init__(self, name: str, channels: int, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels

        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.instance_norm = nn.InstanceNorm2d(channels, affine=True)
        self.batch_norm = nn.BatchNorm2d(channels, affine=True)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        alpha = torch.sigmoid(self.alpha)
        normed = alpha * self.instance_norm(x) + (1 - alpha) * self.batch_norm(x)
        return F.gelu(self.conv(normed)) + x


class CoarseFineComponent(TorchComponent):
    """Parallel coarse/fine resolution processing with cross-gating."""

    def __init__(self, name: str, channels: int, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels

        self.fine_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.fine_bn = nn.BatchNorm2d(channels)

        self.coarse_down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.coarse_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.coarse_bn = nn.BatchNorm2d(channels)

        self.fine_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels), nn.Sigmoid()
        )
        self.coarse_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        fine = F.gelu(self.fine_bn(self.fine_conv(x)))
        coarse = F.gelu(self.coarse_down(x))
        coarse = F.gelu(self.coarse_bn(self.coarse_conv(coarse)))
        coarse = F.interpolate(coarse, size=x.shape[2:], mode='bilinear', align_corners=False)

        fg = self.fine_gate(fine).unsqueeze(-1).unsqueeze(-1)
        cg = self.coarse_gate(coarse).unsqueeze(-1).unsqueeze(-1)

        return fine * cg + coarse * fg + x


class WideResComponent(TorchComponent):
    """Wide residual block component."""

    def __init__(self, name: str, channels: int, dropout: float = 0.1, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = F.gelu(self.bn1(x))
        out = self.conv1(out)
        out = F.gelu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + x


class BottleneckComponent(TorchComponent):
    """Bottleneck residual block (ResNet-style)."""

    def __init__(self, name: str, channels: int, reduction: int = 4, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        mid_channels = channels // reduction

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)  # Reduce
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)  # Process
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, channels, 1, bias=False)  # Expand

    def forward(self, x: Tensor) -> Tensor:
        out = F.gelu(self.bn1(x))
        out = self.conv1(out)
        out = F.gelu(self.bn2(out))
        out = self.conv2(out)
        out = F.gelu(self.bn3(out))
        out = self.conv3(out)
        return out + x


class SqueezeExciteComponent(TorchComponent):
    """Squeeze-and-Excitation block with channel attention."""

    def __init__(self, name: str, channels: int, reduction: int = 16, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        mid_channels = max(channels // reduction, 8)

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE module
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Linear(channels, mid_channels)
        self.se_fc2 = nn.Linear(mid_channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # SE attention
        B, C, _, _ = out.shape
        se = self.se_pool(out).view(B, C)
        se = F.gelu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se)).view(B, C, 1, 1)

        return F.gelu(out * se) + x


class InvertedBottleneckComponent(TorchComponent):
    """Inverted bottleneck (MobileNetV2-style): expand -> depthwise -> project."""

    def __init__(self, name: str, channels: int, expansion: int = 4, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        exp_channels = channels * expansion

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, exp_channels, 1, bias=False)  # Expand
        self.bn2 = nn.BatchNorm2d(exp_channels)
        self.conv2 = nn.Conv2d(exp_channels, exp_channels, 3, padding=1, groups=exp_channels, bias=False)  # Depthwise
        self.bn3 = nn.BatchNorm2d(exp_channels)
        self.conv3 = nn.Conv2d(exp_channels, channels, 1, bias=False)  # Project

    def forward(self, x: Tensor) -> Tensor:
        out = F.gelu(self.bn1(x))
        out = self.conv1(out)
        out = F.gelu(self.bn2(out))
        out = self.conv2(out)
        out = F.gelu(self.bn3(out))
        out = self.conv3(out)
        return out + x


class SpatialAttentionComponent(TorchComponent):
    """Spatial self-attention for conv features."""

    def __init__(self, name: str, channels: int, num_heads: int = 4, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.bn1 = nn.BatchNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        L = H * W

        out = self.bn1(x)
        qkv = self.qkv(out).view(B, 3, self.num_heads, self.head_dim, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, heads, head_dim, L]

        # Attention
        q = q.transpose(-2, -1)  # [B, heads, L, head_dim]
        k = k  # [B, heads, head_dim, L]
        v = v.transpose(-2, -1)  # [B, heads, L, head_dim]

        attn = torch.matmul(q, k) * self.scale  # [B, heads, L, L]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, heads, L, head_dim]
        out = out.transpose(-2, -1).reshape(B, C, H, W)

        out = self.bn2(self.proj(out))
        return out + x


# =============================================================================
# PROJECTION COMPONENTS
# =============================================================================

class FlexibleInputComponent(TorchComponent):
    """
    Flexible input projection supporting multiple formats.

    Supported input modes:
    - 'spatial': [B, C, H, W] - direct spatial input (e.g., VAE latents)
    - 'sequence': [B, L, D] where L = H*W - reshape to spatial
    - 'sequence_pooled': [B, L, D] - mean pool then expand (legacy, lossy)

    For multi-channel inputs like Flux VAE (16 channels):
    - Set in_channels=16, spatial_size=32
    - Input [B, 16, 32, 32] is projected to [B, out_channels, H, W]
    """

    def __init__(
        self,
        name: str,
        in_features: int,          # D for sequence, C for spatial
        out_channels: int,         # Output channels for conv stages
        spatial_size: int,         # Target H=W
        mode: str = 'auto',        # 'spatial', 'sequence', 'sequence_pooled', 'auto'
        in_channels: int = None,   # For spatial mode (e.g., 16 for Flux)
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.spatial_size = spatial_size
        self.out_channels = out_channels
        self.mode = mode
        self.in_features = in_features
        self.in_channels = in_channels or in_features

        # Spatial input: [B, C_in, H, W] -> [B, C_out, H, W]
        self.spatial_proj = nn.Conv2d(self.in_channels, out_channels, 1)

        # Sequence input: [B, L, D] -> [B, C_out, H, W]
        # Project each token, reshape to spatial
        self.seq_proj = nn.Linear(in_features, out_channels)

        # Sequence pooled (legacy): [B, D] -> [B, C_out, H, W]
        self.pooled_proj = nn.Linear(in_features, out_channels * spatial_size * spatial_size)

        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)

    def forward(self, x: Tensor) -> Tensor:
        mode = self.mode

        # Auto-detect mode based on input shape
        if mode == 'auto':
            if x.dim() == 4:
                mode = 'spatial'
            elif x.dim() == 3:
                B, L, D = x.shape
                if L == self.spatial_size * self.spatial_size:
                    mode = 'sequence'
                else:
                    mode = 'sequence_pooled'
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

        if mode == 'spatial':
            # [B, C, H, W] -> [B, C_out, H, W]
            h = self.spatial_proj(x)
            # Handle size mismatch
            if h.shape[-1] != self.spatial_size:
                h = F.interpolate(h, size=self.spatial_size, mode='bilinear', align_corners=False)

        elif mode == 'sequence':
            # [B, L, D] -> [B, H*W, C_out] -> [B, C_out, H, W]
            B, L, D = x.shape
            h = self.seq_proj(x)  # [B, L, C_out]
            h = h.permute(0, 2, 1)  # [B, C_out, L]
            h = h.view(B, self.out_channels, self.spatial_size, self.spatial_size)

        elif mode == 'sequence_pooled':
            # [B, L, D] -> [B, D] -> [B, C_out*H*W] -> [B, C_out, H, W]
            B = x.shape[0]
            pooled = x.mean(dim=1)
            h = self.pooled_proj(pooled)
            h = h.view(B, self.out_channels, self.spatial_size, self.spatial_size)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return F.gelu(self.norm(h))


class MultiScaleConvBlock(TorchComponent):
    """
    Multi-scale convolution block for processing multi-channel latents.

    Processes at multiple scales simultaneously:
    - Local: 3×3 conv for fine details
    - Regional: 5×5 or dilated 3×3 for medium context
    - Global: pooled features for full-image context

    Channel-wise attention gates which scale contributes where.
    """

    def __init__(
        self,
        name: str,
        channels: int,
        expansion: int = 2,
        use_se: bool = True,  # Squeeze-excite attention
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.channels = channels
        mid_channels = channels * expansion

        # Local path (fine details)
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 3, padding=1),
            nn.GroupNorm(min(32, mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels, 3, padding=1),
        )

        # Regional path (medium context)
        self.regional_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 3, padding=2, dilation=2),
            nn.GroupNorm(min(32, mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels, 3, padding=2, dilation=2),
        )

        # Global path (full context)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, channels),
        )

        # Scale mixing
        self.scale_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.Sigmoid(),
        )

        # Optional squeeze-excite
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels // 4),
                nn.GELU(),
                nn.Linear(channels // 4, channels),
                nn.Sigmoid(),
            )

        self.norm = nn.GroupNorm(min(32, channels), channels)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        # Multi-scale features
        local_feat = self.local_conv(x)
        regional_feat = self.regional_conv(x)

        global_feat = self.global_pool(x).flatten(1)  # [B, C]
        global_feat = self.global_fc(global_feat)  # [B, C]
        global_feat = global_feat.view(B, C, 1, 1).expand(-1, -1, H, W)  # [B, C, H, W]

        # Concatenate and gate
        concat = torch.cat([local_feat, regional_feat, global_feat], dim=1)  # [B, 3C, H, W]
        gate = self.scale_gate(concat)  # [B, C, H, W]

        # Weighted combination
        out = gate * local_feat + (1 - gate) * 0.5 * (regional_feat + global_feat)

        # Squeeze-excite channel attention
        if self.use_se:
            se_weight = self.se(out).view(B, C, 1, 1)
            out = out * se_weight

        return self.norm(out + x)


class ChannelMixerBlock(TorchComponent):
    """
    Channel mixing block for multi-channel latents (e.g., Flux's 16 channels).

    Cross-channel attention allows channels to share information,
    important for VAE latents where channels encode different aspects.
    """

    def __init__(
        self,
        name: str,
        channels: int,
        spatial_size: int,
        num_heads: int = 4,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.channels = channels
        self.spatial_size = spatial_size
        self.num_heads = num_heads

        # Spatial pooling to reduce computation
        self.spatial_pool = nn.AdaptiveAvgPool2d(8)  # Always 8×8 for attention

        # Channel attention via linear layers (treat channels as sequence)
        self.channel_norm = nn.LayerNorm(64)  # 8*8 = 64
        self.channel_q = nn.Linear(64, 64)
        self.channel_k = nn.Linear(64, 64)
        self.channel_v = nn.Linear(64, 64)
        self.channel_proj = nn.Linear(64, 64)

        # Upsample back
        self.upsample = nn.Upsample(size=spatial_size, mode='bilinear', align_corners=False)

        # Residual mixing
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        # Pool to fixed size for attention
        h = self.spatial_pool(x)  # [B, C, 8, 8]
        h = h.view(B, C, -1)  # [B, C, 64]

        # Channel attention (treat C as sequence length)
        h_norm = self.channel_norm(h)
        q = self.channel_q(h_norm)  # [B, C, 64]
        k = self.channel_k(h_norm)  # [B, C, 64]
        v = self.channel_v(h_norm)  # [B, C, 64]

        # Attention: [B, C, 64] × [B, 64, C] -> [B, C, C]
        attn = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(64)
        attn = F.softmax(attn, dim=-1)

        # Apply: [B, C, C] × [B, C, 64] -> [B, C, 64]
        out = torch.bmm(attn, v)
        out = self.channel_proj(out)

        # Reshape and upsample
        out = out.view(B, C, 8, 8)
        out = self.upsample(out)  # [B, C, H, W]

        # Gated residual
        return x + torch.sigmoid(self.gate) * out


class SpatialToOpinionComponent(TorchComponent):
    """
    Project spatial to opinion: [B, C, H, W] -> [B, D]

    Supports multiple pooling strategies:
    - 'adaptive': AdaptiveAvgPool2d (default)
    - 'attention': Learned spatial attention weights
    - 'multiscale': Pool at multiple scales, concatenate
    """

    def __init__(
        self,
        name: str,
        channels: int,
        out_dim: int,
        pool_mode: str = 'adaptive',
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.pool_mode = pool_mode
        self.channels = channels
        self.out_dim = out_dim

        if pool_mode == 'adaptive':
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(channels, out_dim)

        elif pool_mode == 'attention':
            self.attn_conv = nn.Conv2d(channels, 1, 1)
            self.proj = nn.Linear(channels, out_dim)

        elif pool_mode == 'multiscale':
            # Pool at 1×1, 2×2, 4×4 and concatenate
            self.pool1 = nn.AdaptiveAvgPool2d(1)
            self.pool2 = nn.AdaptiveAvgPool2d(2)
            self.pool4 = nn.AdaptiveAvgPool2d(4)
            # 1 + 4 + 16 = 21 spatial positions
            self.proj = nn.Linear(channels * 21, out_dim)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        if self.pool_mode == 'adaptive':
            h = self.pool(x).flatten(1)  # [B, C]

        elif self.pool_mode == 'attention':
            # Learn spatial attention weights
            attn = self.attn_conv(x)  # [B, 1, H, W]
            attn = F.softmax(attn.view(B, 1, -1), dim=-1)  # [B, 1, H*W]
            h = x.view(B, C, -1)  # [B, C, H*W]
            h = torch.bmm(h, attn.transpose(-1, -2)).squeeze(-1)  # [B, C]

        elif self.pool_mode == 'multiscale':
            p1 = self.pool1(x).flatten(1)  # [B, C]
            p2 = self.pool2(x).flatten(1)  # [B, C*4]
            p4 = self.pool4(x).flatten(1)  # [B, C*16]
            h = torch.cat([p1, p2, p4], dim=1)  # [B, C*21]

        return self.norm(self.proj(h))


# Legacy alias for backward compatibility
class SeqToSpatialComponent(FlexibleInputComponent):
    """Legacy wrapper - prefer FlexibleInputComponent directly."""

    def __init__(self, name: str, seq_dim: int, channels: int, spatial_size: int, **kwargs):
        super().__init__(
            name=name,
            in_features=seq_dim,
            out_channels=channels,
            spatial_size=spatial_size,
            mode='sequence_pooled',  # Legacy behavior
            **kwargs
        )


# =============================================================================
# CONFIGURABLE CONV TOWER
# =============================================================================

class ConfigurableConvTower(BaseTower):
    """
    Conv tower built from ConvTowerConfig.

    Supports multi-channel latent inputs (e.g., Flux VAE with 16 channels):
    - FlexibleInputComponent handles [B, C, H, W] or [B, L, D] inputs
    - Optional ChannelMixerBlock for cross-channel attention
    - MultiScaleConvBlock for multi-scale processing
    - Configurable output pooling (adaptive, attention, multiscale)
    """

    def __init__(
        self,
        config: ConvTowerConfig,
        default_dim: int = 256,
        default_depth: int = 2,
        default_fingerprint_dim: int = 64,
        spatial_size: int = 16,
    ):
        super().__init__(config.name, strict=False)

        dim = config.dim or default_dim
        depth = config.depth or default_depth
        fingerprint_dim = config.fingerprint_dim or default_fingerprint_dim
        params = config.tower_params

        self.dim = dim
        self.depth = depth
        self.config = config
        self._fingerprint_dim = fingerprint_dim
        self._inverted = config.inverted

        self.objects['tower_config'] = {
            'name': config.name,
            'tower_type': config.tower_type.value,
            'inverted': config.inverted,
            'dim': dim,
            'depth': depth,
            'spatial_size': spatial_size,
            'in_channels': config.in_channels,
            'input_mode': config.input_mode,
        }
        self.objects['spatial_size'] = spatial_size

        base_channels = params.get('base_channels', 64)

        # Input projection - now supports multi-channel inputs
        self.attach('input_proj', FlexibleInputComponent(
            f'{config.name}_input',
            in_features=dim,
            out_channels=base_channels,
            spatial_size=spatial_size,
            mode=config.input_mode,
            in_channels=config.in_channels,
        ))

        # Optional channel mixer at start (for multi-channel latents)
        if config.use_channel_mixer and config.in_channels and config.in_channels > 1:
            self.attach('channel_mixer', ChannelMixerBlock(
                f'{config.name}_channel_mix',
                channels=base_channels,
                spatial_size=spatial_size,
            ))

        # Main processing stages
        self._build_stages(
            config.tower_type, config.name, base_channels, depth,
            spatial_size, params, config.use_multiscale
        )

        # Output projection with configurable pooling
        self.attach('output_proj', SpatialToOpinionComponent(
            f'{config.name}_output', base_channels, dim,
            pool_mode=config.pool_mode,
        ))

        self.attach('address', AddressComponent(
            f'{config.name}_address', fingerprint_dim=fingerprint_dim
        ))

        self.attach('opinion_proj', nn.Linear(dim, dim))

    def _build_stages(self, tower_type, tower_name, channels, depth, spatial_size, params, use_multiscale):
        # Optionally inject MultiScaleConvBlock between regular stages
        if use_multiscale and depth >= 2:
            multiscale_interval = max(1, depth // 2)
        else:
            multiscale_interval = depth + 1  # Never insert

        stage_idx = 0

        if tower_type == ConvTowerType.DEPTH:
            dilations = [1, 2, 4, 8]
            for i in range(depth):
                dilation = dilations[i % len(dilations)]
                self.append(DilatedConvComponent(f'{tower_name}_dilated_{i}', channels, dilation))
                stage_idx += 1
                if stage_idx % multiscale_interval == 0 and stage_idx < depth:
                    self.append(MultiScaleConvBlock(f'{tower_name}_ms_{stage_idx}', channels))

        elif tower_type == ConvTowerType.DILATED:
            dilations = params.get('dilations', [1, 2, 4, 8])
            for i in range(depth):
                dilation = dilations[i % len(dilations)]
                self.append(DilatedConvComponent(f'{tower_name}_dil_{i}', channels, dilation))
                stage_idx += 1
                if stage_idx % multiscale_interval == 0 and stage_idx < depth:
                    self.append(MultiScaleConvBlock(f'{tower_name}_ms_{stage_idx}', channels))

        elif tower_type == ConvTowerType.FREQUENCY:
            for i in range(depth):
                self.append(FrequencyComponent(f'{tower_name}_freq_{i}', channels, spatial_size))
                stage_idx += 1
                if stage_idx % multiscale_interval == 0 and stage_idx < depth:
                    self.append(MultiScaleConvBlock(f'{tower_name}_ms_{stage_idx}', channels))

        elif tower_type == ConvTowerType.COLOR_PATTERN:
            for i in range(depth):
                self.append(InterpolatedNormComponent(f'{tower_name}_pattern_{i}', channels))
                stage_idx += 1
                if stage_idx % multiscale_interval == 0 and stage_idx < depth:
                    self.append(MultiScaleConvBlock(f'{tower_name}_ms_{stage_idx}', channels))

        elif tower_type == ConvTowerType.COARSE_FINE:
            for i in range(depth):
                self.append(CoarseFineComponent(f'{tower_name}_cf_{i}', channels))
                stage_idx += 1
                if stage_idx % multiscale_interval == 0 and stage_idx < depth:
                    self.append(MultiScaleConvBlock(f'{tower_name}_ms_{stage_idx}', channels))

        elif tower_type == ConvTowerType.WIDE_RESNET:
            dropout = params.get('dropout', 0.1)
            for i in range(depth):
                self.append(WideResComponent(f'{tower_name}_wrn_{i}', channels, dropout))
                stage_idx += 1
                if stage_idx % multiscale_interval == 0 and stage_idx < depth:
                    self.append(MultiScaleConvBlock(f'{tower_name}_ms_{stage_idx}', channels))

        elif tower_type == ConvTowerType.BOTTLENECK:
            reduction = params.get('reduction', 4)
            for i in range(depth):
                self.append(BottleneckComponent(f'{tower_name}_bn_{i}', channels, reduction))

        elif tower_type == ConvTowerType.SQUEEZE_EXCITE:
            reduction = params.get('se_reduction', 16)
            for i in range(depth):
                self.append(SqueezeExciteComponent(f'{tower_name}_se_{i}', channels, reduction))

        elif tower_type == ConvTowerType.INVERTED_BOTTLENECK:
            expansion = params.get('expansion', 4)
            for i in range(depth):
                self.append(InvertedBottleneckComponent(f'{tower_name}_inv_{i}', channels, expansion))

        elif tower_type == ConvTowerType.SPATIAL_ATTENTION:
            num_heads = params.get('num_heads', 4)
            for i in range(depth):
                self.append(SpatialAttentionComponent(f'{tower_name}_sa_{i}', channels, num_heads))

    @property
    def fingerprint(self) -> Tensor:
        fp = F.normalize(self['address'].fingerprint, dim=-1)
        if self._inverted:
            fp = 1.0 - fp
        return fp

    @property
    def is_inverted(self) -> bool:
        return self._inverted

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Process through conv tower.

        Supports multiple input formats:
        - [B, L, D]: Sequence input (L = H*W tokens or arbitrary)
        - [B, C, H, W]: Spatial input (e.g., VAE latents)

        Returns:
            opinion: [B, dim] tower opinion
            features: [B, L, dim] or [B, H*W, dim] expanded features
        """
        # Determine input format and get sequence length for feature expansion
        if x.dim() == 4:
            # Spatial input [B, C, H, W]
            B, C, H, W = x.shape
            seq_len = H * W
        elif x.dim() == 3:
            # Sequence input [B, L, D]
            B, seq_len, D = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Input projection (handles both formats)
        h = self['input_proj'](x)

        # Optional channel mixer (early cross-channel attention)
        if 'channel_mixer' in self.components:
            h = self['channel_mixer'](h)

        # Main conv stages
        for stage in self.stages:
            h = stage(h)

        # Output projection
        pooled = self['output_proj'](h)
        opinion = self['opinion_proj'](pooled)

        # Expand features to match input sequence length
        features = pooled.unsqueeze(1).expand(-1, seq_len, -1)

        # Cache features for WideRouter retrieval
        # Uses ephemeral cache that can be bulk-cleared
        self.cache_set('features', features)

        return opinion, features

    @property
    def cached_features(self) -> Optional[Tensor]:
        """Features from last forward pass (for WideRouter integration)."""
        return self.cache_get('features')


# =============================================================================
# WALKER CONV PAIR
# =============================================================================

class WalkerConvPair(nn.Module):
    """
    Two conv towers fused with WalkerFusion.

    The simplest collective unit for geometric conv fusion.
    """

    def __init__(
        self,
        name: str,
        config_a: ConvTowerConfig,
        config_b: ConvTowerConfig,
        dim: int = 256,
        depth: int = 2,
        spatial_size: int = 16,
        use_inception: bool = False,
        walker_preset: str = 'shiva',
        inception_aux_type: str = 'geometric',
    ):
        super().__init__()
        self.name = name

        self.tower_a = ConfigurableConvTower(
            config=config_a, default_dim=dim, default_depth=depth, spatial_size=spatial_size
        )
        self.tower_b = ConfigurableConvTower(
            config=config_b, default_dim=dim, default_depth=depth, spatial_size=spatial_size
        )

        inception = None
        if use_inception:
            inception = WalkerInception(
                f'{name}_inception', in_features=dim,
                num_steps=8, num_inputs=2, aux_type=inception_aux_type
            )

        self.fusion = WalkerFusion(
            f'{name}_fusion', in_features=dim,
            preset=walker_preset, num_steps=8, inception=inception
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        opinion_a, _ = self.tower_a(x, mask)
        opinion_b, _ = self.tower_b(x, mask)
        return self.fusion(opinion_a, opinion_b)

    @property
    def fusion_params(self) -> int:
        return sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)


# =============================================================================
# CONV TOWER COLLECTIVE (using WideRouter)
# =============================================================================

class ConvTowerCollective(WideRouter):
    """
    Conv tower collective using WideRouter for optimized execution.

    Groups towers by type for batched processing via torch.compile.
    Now supports WalkerFusion as an alternative to AdaptiveFusion.
    """

    def __init__(
        self,
        name: str,
        tower_configs: List[ConvTowerConfig],
        dim: int = 256,
        default_depth: int = 2,
        fingerprint_dim: int = 64,
        spatial_size: int = 16,
        fusion_type: Union[FusionType, str] = FusionType.ADAPTIVE,
        fusion_params: Dict[str, Any] = None,
    ):
        # Initialize WideRouter with auto_discover=False (we register manually)
        super().__init__(name, strict=False, auto_discover=False)

        self.dim = dim
        self._fingerprint_dim = fingerprint_dim
        self._num_towers = len(tower_configs)

        if isinstance(fusion_type, str):
            fusion_type = FusionType(fusion_type)

        self.objects['config'] = {
            'dim': dim,
            'num_towers': self._num_towers,
            'fusion_type': fusion_type.value,
        }

        # Build and attach towers
        for cfg in tower_configs:
            tower = ConfigurableConvTower(
                config=cfg,
                default_dim=dim,
                default_depth=default_depth,
                default_fingerprint_dim=fingerprint_dim,
                spatial_size=spatial_size,
            )
            self.attach(cfg.name, tower)
            # Register with WideRouter for optimized execution
            self.register_tower(cfg.name)

        # Build fusion
        fusion_params = fusion_params or {}
        fusion = self._build_fusion(f'{name}_fusion', fusion_type, dim, fusion_params)
        self.attach('fusion', fusion)

        # Fingerprint projection
        self.attach('fp_proj', nn.Linear(fingerprint_dim * self._num_towers, fingerprint_dim))

    def _build_fusion(self, name: str, fusion_type: FusionType, dim: int, params: Dict) -> nn.Module:
        """Build fusion module based on type."""
        if fusion_type in WALKER_PRESET_MAP:
            preset = WALKER_PRESET_MAP[fusion_type]
            inception = None
            if fusion_type == FusionType.WALKER_INCEPTION:
                inception = WalkerInception(
                    f'{name}_inception', in_features=dim,
                    num_steps=params.get('num_steps', 8), num_inputs=2,
                    aux_type=params.get('inception_aux_type', 'geometric')
                )
            return WalkerFusion(
                name, in_features=dim, preset=preset,
                num_steps=params.get('num_steps', 8), inception=inception
            )
        else:
            # Legacy AdaptiveFusion
            return AdaptiveFusion(name, num_inputs=self._num_towers, in_features=dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward using WideRouter's optimized execution.

        Returns:
            fused: [B, D] fused opinion
            opinions: Dict of tower_name -> opinion tensor
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Use WideRouter's optimized execution
        opinions_dict = self.wide_forward(x, mask=mask)

        # Collect fingerprints and opinion tensors in order
        all_fingerprints = []
        opinion_tensors = []

        for name in self.tower_names:
            opinion_tensors.append(opinions_dict[name])
            all_fingerprints.append(self[name].fingerprint)

        # Fuse
        fused = self['fusion'](*opinion_tensors)

        # Collective fingerprint
        fp_cat = torch.cat(all_fingerprints, dim=-1)
        collective_fp = F.normalize(self['fp_proj'](fp_cat), dim=-1)

        # Clear tower caches to prevent memory leaks
        for name in self.tower_names:
            self[name].cache_clear()

        return fused, opinions_dict


# =============================================================================
# QUICK CONSTRUCTION FUNCTIONS
# =============================================================================

def get_conv_config(
    tower_type: str,
    name: Optional[str] = None,
    inverted: bool = False,
    **kwargs
) -> ConvTowerConfig:
    """Get a conv tower config by type name."""
    return ConvTowerConfig(
        name=name or tower_type,
        tower_type=tower_type,
        inverted=inverted,
        **kwargs
    )


def quick_conv_tower(
    tower_type: str,
    dim: int = 256,
    depth: int = 2,
    name: Optional[str] = None,
    inverted: bool = False,
    spatial_size: int = 16,
    **kwargs
) -> ConfigurableConvTower:
    """
    Quickly create a single conv tower by type name.

    Example:
        tower = quick_conv_tower('wide_resnet', dim=256, depth=3)
    """
    config = get_conv_config(tower_type, name=name, inverted=inverted, **kwargs)
    return ConfigurableConvTower(
        config=config,
        default_dim=dim,
        default_depth=depth,
        spatial_size=spatial_size,
    )


def quick_conv_pair(
    tower_type_a: str,
    tower_type_b: str,
    dim: int = 256,
    depth: int = 2,
    spatial_size: int = 16,
    use_inception: bool = False,
    walker_preset: str = 'shiva',
    inception_aux_type: str = 'geometric',
    name: Optional[str] = None,
    **kwargs
) -> WalkerConvPair:
    """
    Quickly create a pair of conv towers with WalkerFusion.

    Example:
        pair = quick_conv_pair('wide_resnet', 'frequency', use_inception=True)
        fused = pair(x)  # [B, D]
    """
    name = name or f'{tower_type_a}_{tower_type_b}'
    config_a = get_conv_config(tower_type_a, **kwargs)
    config_b = get_conv_config(tower_type_b, **kwargs)
    return WalkerConvPair(
        name=name, config_a=config_a, config_b=config_b,
        dim=dim, depth=depth, spatial_size=spatial_size,
        use_inception=use_inception, walker_preset=walker_preset,
        inception_aux_type=inception_aux_type,
    )


def quick_conv_collective(
    tower_types: List[str],
    dim: int = 256,
    depth: int = 2,
    spatial_size: int = 16,
    fusion_type: Union[FusionType, str] = FusionType.WALKER_STATIC,
    name: str = 'conv_collective',
    **kwargs
) -> ConvTowerCollective:
    """
    Quickly create a collective of conv towers.

    Example:
        collective = quick_conv_collective(
            ['wide_resnet', 'frequency', 'squeeze_excite'],
            fusion_type='walker_inception'
        )
    """
    configs = [get_conv_config(t, **kwargs) for t in tower_types]
    return ConvTowerCollective(
        name=name, tower_configs=configs, dim=dim, default_depth=depth,
        spatial_size=spatial_size, fusion_type=fusion_type,
        fusion_params=kwargs.get('fusion_params', {}),
    )


# =============================================================================
# BUILDER FUNCTIONS
# =============================================================================

def build_conv_tower(
    config: ConvTowerConfig,
    default_dim: int = 256,
    default_depth: int = 2,
    default_fingerprint_dim: int = 64,
    spatial_size: int = 16,
) -> ConfigurableConvTower:
    """Build a single conv tower."""
    return ConfigurableConvTower(
        config=config,
        default_dim=default_dim,
        default_depth=default_depth,
        default_fingerprint_dim=default_fingerprint_dim,
        spatial_size=spatial_size,
    )


def build_conv_collective(
    configs: List[ConvTowerConfig],
    dim: int = 256,
    default_depth: int = 2,
    fingerprint_dim: int = 64,
    spatial_size: int = 16,
    name: str = 'conv_collective',
    fusion_type: Union[FusionType, str] = FusionType.ADAPTIVE,
) -> ConvTowerCollective:
    """Build a conv tower collective with batched forward."""
    return ConvTowerCollective(
        name=name,
        tower_configs=configs,
        dim=dim,
        default_depth=default_depth,
        fingerprint_dim=fingerprint_dim,
        spatial_size=spatial_size,
        fusion_type=fusion_type,
    )


# =============================================================================
# PRESETS
# =============================================================================

def preset_conv_towers() -> List[ConvTowerConfig]:
    """One tower of each type."""
    return [
        ConvTowerConfig('depth', tower_type='depth'),
        ConvTowerConfig('dilated', tower_type='dilated'),
        ConvTowerConfig('frequency', tower_type='frequency'),
        ConvTowerConfig('color_pattern', tower_type='color_pattern'),
        ConvTowerConfig('coarse_fine', tower_type='coarse_fine'),
        ConvTowerConfig('wide_resnet', tower_type='wide_resnet'),
        ConvTowerConfig('bottleneck', tower_type='bottleneck'),
        ConvTowerConfig('squeeze_excite', tower_type='squeeze_excite'),
        ConvTowerConfig('inverted_bottleneck', tower_type='inverted_bottleneck'),
        ConvTowerConfig('spatial_attention', tower_type='spatial_attention'),
    ]


def preset_conv_pos_neg(
    tower_types: List[str] = ['wide_resnet', 'frequency', 'bottleneck', 'squeeze_excite']
) -> List[ConvTowerConfig]:
    """Pos/neg pairs for specified conv towers."""
    configs = []
    for tower_type in tower_types:
        configs.append(ConvTowerConfig(f'{tower_type}_pos', tower_type=tower_type, inverted=False))
        configs.append(ConvTowerConfig(f'{tower_type}_neg', tower_type=tower_type, inverted=True))
    return configs


def preset_conv_full_stack() -> List[ConvTowerConfig]:
    """Full dual stack: all conv types × pos/neg = 20 towers."""
    return preset_conv_pos_neg([
        'wide_resnet', 'frequency', 'bottleneck', 'squeeze_excite',
        'dilated', 'coarse_fine', 'inverted_bottleneck', 'spatial_attention',
        'depth', 'color_pattern'
    ])


def preset_flux_vae_towers(
    tower_types: List[str] = ['wide_resnet', 'frequency', 'squeeze_excite', 'spatial_attention']
) -> List[ConvTowerConfig]:
    """
    Conv tower configs optimized for Flux VAE latents.

    Flux VAE produces 16-channel latents at 8x spatial compression:
    - 256×256 image → 32×32×16 latent
    - 512×512 image → 64×64×16 latent

    These configs enable:
    - Direct spatial input (no sequence conversion)
    - Cross-channel attention
    - Multi-scale processing
    - Attention pooling for output
    """
    configs = []
    for tower_type in tower_types:
        configs.append(ConvTowerConfig(
            f'{tower_type}_pos',
            tower_type=tower_type,
            inverted=False,
            in_channels=16,           # Flux VAE channels
            input_mode='spatial',     # Direct [B, C, H, W] input
            pool_mode='attention',    # Learned spatial attention for output
            use_channel_mixer=True,   # Cross-channel attention
            use_multiscale=True,      # Multi-scale conv blocks
        ))
        configs.append(ConvTowerConfig(
            f'{tower_type}_neg',
            tower_type=tower_type,
            inverted=True,
            in_channels=16,
            input_mode='spatial',
            pool_mode='attention',
            use_channel_mixer=True,
            use_multiscale=True,
        ))
    return configs


def preset_sd_vae_towers(
    tower_types: List[str] = ['wide_resnet', 'frequency', 'squeeze_excite']
) -> List[ConvTowerConfig]:
    """
    Conv tower configs for Stable Diffusion VAE latents.

    SD VAE produces 4-channel latents at 8x spatial compression:
    - 512×512 image → 64×64×4 latent
    """
    configs = []
    for tower_type in tower_types:
        configs.append(ConvTowerConfig(
            f'{tower_type}_pos',
            tower_type=tower_type,
            inverted=False,
            in_channels=4,            # SD VAE channels
            input_mode='spatial',
            pool_mode='adaptive',
            use_channel_mixer=False,  # Only 4 channels, less benefit
            use_multiscale=True,
        ))
        configs.append(ConvTowerConfig(
            f'{tower_type}_neg',
            tower_type=tower_type,
            inverted=True,
            in_channels=4,
            input_mode='spatial',
            pool_mode='adaptive',
            use_channel_mixer=False,
            use_multiscale=True,
        ))
    return configs


def preset_sequence_towers(
    tower_types: List[str] = ['wide_resnet', 'frequency']
) -> List[ConvTowerConfig]:
    """
    Conv tower configs for sequence inputs (e.g., from transformers).

    Expects [B, L, D] input where L = H*W spatial tokens.
    Reshapes to spatial for conv processing.
    """
    configs = []
    for tower_type in tower_types:
        configs.append(ConvTowerConfig(
            f'{tower_type}_pos',
            tower_type=tower_type,
            inverted=False,
            input_mode='sequence',    # [B, L, D] with L = H*W
            pool_mode='adaptive',
            use_channel_mixer=False,
            use_multiscale=True,
        ))
        configs.append(ConvTowerConfig(
            f'{tower_type}_neg',
            tower_type=tower_type,
            inverted=True,
            input_mode='sequence',
            pool_mode='adaptive',
            use_channel_mixer=False,
            use_multiscale=True,
        ))
    return configs


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    import time

    print("=" * 60)
    print("Conv Tower Builder - Quick Construction Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test quick_conv_tower
    print("\n--- quick_conv_tower ---")
    tower = quick_conv_tower('wide_resnet', dim=256, depth=2)
    tower.network_to(device=device)
    x = torch.randn(4, 256, 256, device=device)
    opinion, features = tower(x)
    print(f"Tower output: opinion {opinion.shape}, features {features.shape}")

    # Test quick_conv_pair
    print("\n--- quick_conv_pair (static) ---")
    pair = quick_conv_pair('wide_resnet', 'frequency', dim=256, depth=1)
    pair.to(device)
    fused = pair(x)
    print(f"Pair output: {fused.shape}")
    print(f"Pair params: {sum(p.numel() for p in pair.parameters()):,}")
    print(f"Fusion params: {pair.fusion_params}")

    print("\n--- quick_conv_pair (inception) ---")
    pair_inc = quick_conv_pair('wide_resnet', 'frequency', dim=256, depth=1, use_inception=True)
    pair_inc.to(device)
    fused = pair_inc(x)
    print(f"Pair output: {fused.shape}")
    print(f"Pair params: {sum(p.numel() for p in pair_inc.parameters()):,}")
    print(f"Fusion params: {pair_inc.fusion_params}")

    # Test quick_conv_collective with walker fusion
    print("\n--- quick_conv_collective (walker_static) ---")
    collective = quick_conv_collective(
        ['wide_resnet', 'frequency', 'squeeze_excite'],
        dim=256, depth=1, fusion_type='walker_static'
    )
    collective.network_to(device=device)
    collective.discover_towers()
    fused, opinions = collective(x)
    print(f"Collective output: {fused.shape}")
    print(f"Collective params: {sum(p.numel() for p in collective.parameters()):,}")

    # Legacy test
    print("\n--- Legacy build_conv_collective ---")
    configs = preset_conv_pos_neg()
    print(f"Towers: {len(configs)}")

    collective = build_conv_collective(configs, dim=256, spatial_size=16)
    collective.network_to(device=device)

    print(f"Tower names: {collective.tower_names}")
    print(f"Params: {sum(p.numel() for p in collective.parameters()):,}")

    B, L, D = 32, 256, 256
    x = torch.randn(B, L, D, device=device)

    # Warmup
    collective.prepare_and_compile()
    for _ in range(5):
        _ = collective(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(50):
        fused, opinions = collective(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / 50 * 1000

    print(f"\nInput: {x.shape}")
    print(f"Fused: {fused.shape}")
    print(f"Opinions: {len(opinions)}")
    print(f"Eager forward: {eager_ms:.2f}ms ({B / (eager_ms/1000):.0f} samples/sec)")

    print("\n✓ All conv tower tests passed")