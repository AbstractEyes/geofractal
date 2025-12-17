"""
geofractal.router.prefab.geometric_conv_tower_builder
=====================================================

Convolutional tower variants with BATCHED collective processing.

Each tower is a TorchComponent-based stage system.
ConvTowerCollective groups towers by type for batched forward.

Tower types:
- DepthTower: Multi-scale dilated convolutions
- FrequencyTower: FFT-based frequency domain processing
- ColorPatternTower: Interpolated IN/BN normalization
- CoarseFineTower: Parallel resolution streams
- WideResNetTower: Wide residual blocks
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


# =============================================================================
# CONV TOWER TYPE ENUM
# =============================================================================

class ConvTowerType(str, Enum):
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

    dim: Optional[int] = None
    depth: Optional[int] = None
    fingerprint_dim: Optional[int] = None

    inverted: bool = False
    tower_params: Dict[str, Any] = field(default_factory=dict)

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
    """FFT-based frequency domain processing stage."""

    def __init__(self, name: str, channels: int, spatial_size: int, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        self.spatial_size = spatial_size

        self.freq_real = nn.Parameter(torch.randn(1, channels, spatial_size, spatial_size // 2 + 1) * 0.02)
        self.freq_imag = nn.Parameter(torch.randn(1, channels, spatial_size, spatial_size // 2 + 1) * 0.02)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        freq = torch.fft.rfft2(x, norm='ortho')
        filt = torch.complex(self.freq_real, self.freq_imag)
        freq_filtered = freq * filt
        out = torch.fft.irfft2(freq_filtered, s=(H, W), norm='ortho')
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


# =============================================================================
# PROJECTION COMPONENTS
# =============================================================================

class SeqToSpatialComponent(TorchComponent):
    """Project sequence to spatial: [B, L, D] -> [B, C, H, W]"""

    def __init__(self, name: str, seq_dim: int, channels: int, spatial_size: int, **kwargs):
        super().__init__(name, **kwargs)
        self.spatial_size = spatial_size
        self.channels = channels
        self.proj = nn.Linear(seq_dim, channels * spatial_size * spatial_size)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        h = self.proj(x.mean(dim=1))
        h = h.view(B, self.channels, self.spatial_size, self.spatial_size)
        return self.norm(h)


class SpatialToOpinionComponent(TorchComponent):
    """Project spatial to opinion: [B, C, H, W] -> [B, D]"""

    def __init__(self, name: str, channels: int, out_dim: int, **kwargs):
        super().__init__(name, **kwargs)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.pool(x).flatten(1)
        return self.norm(self.proj(h))


# =============================================================================
# CONFIGURABLE CONV TOWER
# =============================================================================

class ConfigurableConvTower(BaseTower):
    """Conv tower built from ConvTowerConfig."""

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
        }
        self.objects['spatial_size'] = spatial_size

        base_channels = params.get('base_channels', 64)

        self.attach('input_proj', SeqToSpatialComponent(
            f'{config.name}_input', dim, base_channels, spatial_size
        ))

        self._build_stages(config.tower_type, config.name, base_channels, depth, spatial_size, params)

        self.attach('output_proj', SpatialToOpinionComponent(
            f'{config.name}_output', base_channels, dim
        ))

        self.attach('address', AddressComponent(
            f'{config.name}_address', fingerprint_dim=fingerprint_dim
        ))

        self.attach('opinion_proj', nn.Linear(dim, dim))

    def _build_stages(self, tower_type, tower_name, channels, depth, spatial_size, params):
        if tower_type == ConvTowerType.DEPTH:
            dilations = [1, 2, 4, 8]
            for i in range(depth):
                dilation = dilations[i % len(dilations)]
                self.append(DilatedConvComponent(f'{tower_name}_dilated_{i}', channels, dilation))

        elif tower_type == ConvTowerType.FREQUENCY:
            for i in range(depth):
                self.append(FrequencyComponent(f'{tower_name}_freq_{i}', channels, spatial_size))

        elif tower_type == ConvTowerType.COLOR_PATTERN:
            for i in range(depth):
                self.append(InterpolatedNormComponent(f'{tower_name}_pattern_{i}', channels))

        elif tower_type == ConvTowerType.COARSE_FINE:
            for i in range(depth):
                self.append(CoarseFineComponent(f'{tower_name}_cf_{i}', channels))

        elif tower_type == ConvTowerType.WIDE_RESNET:
            dropout = params.get('dropout', 0.1)
            for i in range(depth):
                self.append(WideResComponent(f'{tower_name}_wrn_{i}', channels, dropout))

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
        """Process through conv tower."""
        B, L, D = x.shape

        h = self['input_proj'](x)

        for stage in self.stages:
            h = stage(h)

        pooled = self['output_proj'](h)
        opinion = self['opinion_proj'](pooled)

        features = pooled.unsqueeze(1).expand(-1, L, -1)

        # Cache features for retrieval after wide_forward
        self.objects['_cached_features'] = features

        return opinion, features

    @property
    def cached_features(self) -> Optional[Tensor]:
        """Features from last forward pass (for WideRouter integration)."""
        return self.objects.get('_cached_features')


# =============================================================================
# CONV TOWER COLLECTIVE (using WideRouter)
# =============================================================================

class ConvTowerCollective(WideRouter):
    """
    Conv tower collective using WideRouter for optimized execution.

    Groups towers by type for batched processing via torch.compile.
    """

    def __init__(
        self,
        name: str,
        tower_configs: List[ConvTowerConfig],
        dim: int = 256,
        default_depth: int = 2,
        fingerprint_dim: int = 64,
        spatial_size: int = 16,
    ):
        # Initialize WideRouter with auto_discover=False (we'll register manually)
        super().__init__(name, strict=False, auto_discover=False)

        self.dim = dim
        self._fingerprint_dim = fingerprint_dim
        self._num_towers = len(tower_configs)

        self.objects['config'] = {
            'dim': dim,
            'num_towers': self._num_towers,
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

        # Fusion (attached after towers so it's not auto-discovered)
        self.attach('fusion', AdaptiveFusion(
            f'{name}_fusion',
            num_inputs=self._num_towers,
            in_features=dim,
        ))

        # Fingerprint projection
        self.attach('fp_proj', nn.Linear(fingerprint_dim * self._num_towers, fingerprint_dim))

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

        return fused, opinions_dict


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
) -> ConvTowerCollective:
    """Build a conv tower collective with batched forward."""
    return ConvTowerCollective(
        name=name,
        tower_configs=configs,
        dim=dim,
        default_depth=default_depth,
        fingerprint_dim=fingerprint_dim,
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
    import time

    print("=" * 60)
    print("WideRouter Conv Collective Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    configs = preset_conv_pos_neg()
    print(f"Towers: {len(configs)}")

    collective = build_conv_collective(configs, dim=256, spatial_size=16)
    collective.network_to(device=device)

    print(f"Tower names: {collective.tower_names}")
    print(f"Params: {sum(p.numel() for p in collective.parameters()):,}")

    B, L, D = 32, 256, 256
    x = torch.randn(B, L, D, device=device)

    # Warmup (eager)
    print("\nEager warmup...")
    for _ in range(5):
        _ = collective(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark eager
    t0 = time.perf_counter()
    for _ in range(50):
        fused, opinions = collective(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / 50 * 1000

    print(f"Eager forward: {eager_ms:.2f}ms")

    # Compile using WideRouter's method
    print("\nCompiling with WideRouter.prepare_and_compile()...")
    compiled = collective.prepare_and_compile()

    # Warmup compiled
    for _ in range(5):
        _ = compiled(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark compiled
    t0 = time.perf_counter()
    for _ in range(50):
        fused, opinions = compiled(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    compiled_ms = (time.perf_counter() - t0) / 50 * 1000

    print(f"Compiled forward: {compiled_ms:.2f}ms")
    print(f"Speedup: {eager_ms/compiled_ms:.2f}x")

    print(f"\nInput: {x.shape}")
    print(f"Fused: {fused.shape}")
    print(f"Opinions: {len(opinions)}")
    print(f"Throughput: {B / (compiled_ms/1000):.0f} samples/sec")

    print("\n✓ WideRouter conv collective ready")