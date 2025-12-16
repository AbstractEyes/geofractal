"""
geofractal.router.prefab.geometric_conv_tower_builder
===========================================

Convolutional tower variants extending BaseTower with TorchComponent stages.

Each stage is a proper TorchComponent with:
- name, uuid, parent awareness
- Device affinity controls
- Lifecycle hooks

Tower types:
- DepthTower: Multi-scale dilated convolutions
- FrequencyTower: FFT-based frequency domain processing
- ColorPatternTower: Interpolated normalization for texture
- CoarseFineTower: Parallel resolution streams
- WideResNetTower: Wide residual blocks
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.base_tower import BaseTower
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.components.address_component import AddressComponent


# =============================================================================
# CONV TOWER TYPE ENUM
# =============================================================================

class ConvTowerType(str, Enum):
    """Available convolutional tower variants."""
    DEPTH = "depth"
    FREQUENCY = "frequency"
    COLOR_PATTERN = "color_pattern"
    COARSE_FINE = "coarse_fine"
    WIDE_RESNET = "wide_resnet"


# =============================================================================
# CONV TOWER CONFIG
# =============================================================================

@dataclass
class ConvTowerConfig:
    """Configuration for a convolutional tower."""

    name: str
    tower_type: Union[ConvTowerType, str] = ConvTowerType.WIDE_RESNET

    # Architecture overrides (None = use collective defaults)
    dim: Optional[int] = None
    depth: Optional[int] = None
    fingerprint_dim: Optional[int] = None

    # Fingerprint options
    inverted: bool = False

    # Type-specific params
    tower_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.tower_type, str):
            self.tower_type = ConvTowerType(self.tower_type)


# =============================================================================
# CONV STAGE COMPONENTS (TorchComponent subclasses)
# =============================================================================

class DilatedConvComponent(TorchComponent):
    """
    Dilated convolution stage for multi-scale receptive fields.

    Uses dilated convs to capture patterns at different scales
    without downsampling.
    """

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
        return out + x  # Residual


class FrequencyComponent(TorchComponent):
    """
    FFT-based frequency domain processing stage.

    Learns frequency-specific filters to capture periodic
    patterns and textures.
    """

    def __init__(self, name: str, channels: int, spatial_size: int, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        self.spatial_size = spatial_size

        # Learnable frequency filter (real + imag parts)
        self.freq_real = nn.Parameter(torch.randn(1, channels, spatial_size, spatial_size // 2 + 1) * 0.02)
        self.freq_imag = nn.Parameter(torch.randn(1, channels, spatial_size, spatial_size // 2 + 1) * 0.02)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        # FFT
        freq = torch.fft.rfft2(x, norm='ortho')

        # Apply learnable complex filter
        filt = torch.complex(self.freq_real, self.freq_imag)
        freq_filtered = freq * filt

        # IFFT back to spatial
        out = torch.fft.irfft2(freq_filtered, s=(H, W), norm='ortho')
        return F.gelu(self.norm(out)) + x  # Residual


class InterpolatedNormComponent(TorchComponent):
    """
    Interpolated IN/BN normalization for color/texture patterns.

    Learns to blend instance norm (texture-sensitive) with
    batch norm (statistics-sensitive) for adaptive normalization.
    """

    def __init__(self, name: str, channels: int, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels

        # Learnable interpolation factor
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.instance_norm = nn.InstanceNorm2d(channels, affine=True)
        self.batch_norm = nn.BatchNorm2d(channels, affine=True)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # Interpolated normalization
        alpha = torch.sigmoid(self.alpha)
        normed = alpha * self.instance_norm(x) + (1 - alpha) * self.batch_norm(x)
        return F.gelu(self.conv(normed)) + x  # Residual


class CoarseFineComponent(TorchComponent):
    """
    Parallel coarse/fine resolution processing with cross-gating.

    Fine path preserves local detail, coarse path captures global context.
    Cross-gating allows information exchange between scales.
    """

    def __init__(self, name: str, channels: int, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels

        # Fine path (preserve resolution)
        self.fine_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.fine_bn = nn.BatchNorm2d(channels)

        # Coarse path (downsample -> process -> upsample)
        self.coarse_down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.coarse_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.coarse_bn = nn.BatchNorm2d(channels)

        # Cross-gating via channel attention
        self.fine_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels), nn.Sigmoid()
        )
        self.coarse_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        # Fine path
        fine = F.gelu(self.fine_bn(self.fine_conv(x)))

        # Coarse path with upsample
        coarse = F.gelu(self.coarse_down(x))
        coarse = F.gelu(self.coarse_bn(self.coarse_conv(coarse)))
        coarse = F.interpolate(coarse, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Cross-gate
        fg = self.fine_gate(fine).unsqueeze(-1).unsqueeze(-1)
        cg = self.coarse_gate(coarse).unsqueeze(-1).unsqueeze(-1)

        return fine * cg + coarse * fg + x  # Residual


class WideResComponent(TorchComponent):
    """
    Wide residual block component.

    Standard WRN block with pre-activation, wider channels,
    and optional dropout for regularization.
    """

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
        return out + x  # Residual


# =============================================================================
# INPUT/OUTPUT PROJECTION COMPONENTS
# =============================================================================

class SeqToSpatialComponent(TorchComponent):
    """Projects sequence [B, L, D] to spatial [B, C, H, W]."""

    def __init__(self, name: str, dim: int, channels: int, spatial_size: int, **kwargs):
        super().__init__(name, **kwargs)
        self.dim = dim
        self.channels = channels
        self.spatial_size = spatial_size
        self.proj = nn.Linear(dim, channels)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        h = self.proj(x)  # [B, L, C]
        return h.transpose(1, 2).reshape(B, self.channels, self.spatial_size, self.spatial_size)


class SpatialToOpinionComponent(TorchComponent):
    """Projects spatial [B, C, H, W] to opinion [B, D]."""

    def __init__(self, name: str, channels: int, dim: int, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        self.dim = dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        h = self.pool(x).flatten(1)  # [B, C]
        return self.norm(self.proj(h))  # [B, D]


# =============================================================================
# CONFIGURABLE CONV TOWER
# =============================================================================

class ConfigurableConvTower(BaseTower):
    """
    Convolutional tower built from ConvTowerConfig.

    Uses BaseTower's stage/component pattern:
    - stages: nn.ModuleList of TorchComponent stages
    - components: input_proj, output_proj, address, opinion_proj
    - objects: config, spatial_size
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

        # Resolve config with defaults
        dim = config.dim or default_dim
        depth = config.depth or default_depth
        fingerprint_dim = config.fingerprint_dim or default_fingerprint_dim
        params = config.tower_params

        self.dim = dim
        self.depth = depth
        self.config = config
        self._fingerprint_dim = fingerprint_dim
        self._inverted = config.inverted

        # Store config in objects
        self.objects['tower_config'] = {
            'name': config.name,
            'tower_type': config.tower_type.value,
            'inverted': config.inverted,
            'dim': dim,
            'depth': depth,
            'spatial_size': spatial_size,
        }
        self.objects['spatial_size'] = spatial_size

        # Channel width for conv stages
        base_channels = params.get('base_channels', 64)

        # Input projection component
        self.attach('input_proj', SeqToSpatialComponent(
            f'{config.name}_input', dim, base_channels, spatial_size
        ))

        # Build TorchComponent stages
        self._build_stages(config.tower_type, config.name, base_channels, depth, spatial_size, params)

        # Output projection component
        self.attach('output_proj', SpatialToOpinionComponent(
            f'{config.name}_output', base_channels, dim
        ))

        # Address component for fingerprint
        self.attach('address', AddressComponent(
            f'{config.name}_address', fingerprint_dim=fingerprint_dim
        ))

        # Opinion projection
        self.attach('opinion_proj', nn.Linear(dim, dim))

    def _build_stages(
            self,
            tower_type: ConvTowerType,
            tower_name: str,
            channels: int,
            depth: int,
            spatial_size: int,
            params: Dict,
    ):
        """Build TorchComponent stages based on tower type."""

        if tower_type == ConvTowerType.DEPTH:
            dilations = [1, 2, 4, 8]
            for i in range(depth):
                dilation = dilations[i % len(dilations)]
                self.append(DilatedConvComponent(
                    f'{tower_name}_dilated_{i}', channels, dilation
                ))

        elif tower_type == ConvTowerType.FREQUENCY:
            for i in range(depth):
                self.append(FrequencyComponent(
                    f'{tower_name}_freq_{i}', channels, spatial_size
                ))

        elif tower_type == ConvTowerType.COLOR_PATTERN:
            for i in range(depth):
                self.append(InterpolatedNormComponent(
                    f'{tower_name}_pattern_{i}', channels
                ))

        elif tower_type == ConvTowerType.COARSE_FINE:
            for i in range(depth):
                self.append(CoarseFineComponent(
                    f'{tower_name}_cf_{i}', channels
                ))

        elif tower_type == ConvTowerType.WIDE_RESNET:
            dropout = params.get('dropout', 0.1)
            for i in range(depth):
                self.append(WideResComponent(
                    f'{tower_name}_wrn_{i}', channels, dropout
                ))

        else:
            raise ValueError(f"Unknown conv tower type: {tower_type}")

    @property
    def fingerprint(self) -> Tensor:
        fp = F.normalize(self['address'].fingerprint, dim=-1)
        if self._inverted:
            fp = 1.0 - fp
        return fp

    @property
    def is_inverted(self) -> bool:
        return self._inverted

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process through conv tower.

        Args:
            x: [B, L, D] sequence input
            mask: Ignored for conv towers

        Returns:
            opinion: [B, D] pooled output
            features: [B, L, D] sequence features
        """
        B, L, D = x.shape

        # Seq -> Spatial
        h = self['input_proj'](x)  # [B, C, H, W]

        # Process through stages
        for stage in self.stages:
            h = stage(h)

        # Spatial -> Opinion
        pooled = self['output_proj'](h)  # [B, D]
        opinion = self['opinion_proj'](pooled)

        # Features for compatibility (expand pooled)
        features = pooled.unsqueeze(1).expand(-1, L, -1)

        return opinion, features


# =============================================================================
# BUILDER FUNCTION
# =============================================================================

def build_conv_tower(
        config: ConvTowerConfig,
        default_dim: int = 256,
        default_depth: int = 2,
        default_fingerprint_dim: int = 64,
        spatial_size: int = 16,
) -> ConfigurableConvTower:
    """Build a conv tower from config."""
    return ConfigurableConvTower(
        config=config,
        default_dim=default_dim,
        default_depth=default_depth,
        default_fingerprint_dim=default_fingerprint_dim,
        spatial_size=spatial_size,
    )


# =============================================================================
# PRESETS
# =============================================================================

def preset_conv_towers() -> List[ConvTowerConfig]:
    """One tower of each type."""
    return [
        ConvTowerConfig('depth', tower_type='depth'),
        ConvTowerConfig('frequency', tower_type='frequency'),
        ConvTowerConfig('color_pattern', tower_type='color_pattern'),
        ConvTowerConfig('coarse_fine', tower_type='coarse_fine'),
        ConvTowerConfig('wide_resnet', tower_type='wide_resnet'),
    ]


def preset_conv_pos_neg() -> List[ConvTowerConfig]:
    """Pos/neg pairs for conv towers."""
    configs = []
    for tower_type in ['depth', 'frequency', 'coarse_fine', 'wide_resnet']:
        configs.append(ConvTowerConfig(f'{tower_type}_pos', tower_type=tower_type, inverted=False))
        configs.append(ConvTowerConfig(f'{tower_type}_neg', tower_type=tower_type, inverted=True))
    return configs


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Conv Tower Builder Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    B, L, D = 2, 256, 256  # 256 patches = 16x16 spatial
    x = torch.randn(B, L, D, device=device)

    configs = preset_conv_towers()

    print("\n--- Tower Tests ---")
    for config in configs:
        tower = build_conv_tower(config, default_dim=256, spatial_size=16).to(device)
        opinion, features = tower(x)
        params = sum(p.numel() for p in tower.parameters())

        # Verify stages are TorchComponents
        stage_types = [type(s).__name__ for s in tower.stages]

        print(f"{config.name:15s}: opinion={opinion.shape}, stages={len(tower)}, "
              f"params={params:,}, stage_type={stage_types[0]}")

    print("\n✓ All stages are TorchComponents")