"""
BEATRIX COLLECTIVE (WideRouter)
===============================

Tower ensemble for the Beatrix oscillator, built on WideRouter
for automatic batched execution and torch.compile optimization.

Integrates:
- geometric_tower_builder.py (ConfigurableCollective, RoPE variants)
- geometric_conv_tower_builder.py (ConvTowerCollective, WideResNet etc.)

Architecture:
    Head Router → Fused Mail [B, D]
                     ↓
              BeatrixCollective (WideRouter)
         ┌──────────┴──────────┐
         ↓                     ↓
    Geometric Towers      Conv Towers
    - cantor ±           - wide_resnet ±
    - beatrix ±          - frequency ±
    - simplex ±          - squeeze_excite ±
    - helix ±            - bottleneck ±
         └──────────┬──────────┘
                    ↓
              + θ probes
                    ↓
              Tower Forces
                    ↓
            Beatrix Oscillator

Author: AbstractPhil + Claude
Date: December 2024
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Base classes
from geofractal.router.wide_router import WideRouter
from geofractal.router.base_tower import BaseTower

# Tower builders
from geofractal.router.prefab.geometric_tower_builder import (
    TowerConfig,
    RoPEType,
    AddressType,
    ConfigurableTower,
    build_rope,
    build_address,
    preset_pos_neg_pairs,
)
from geofractal.router.prefab.geometric_conv_tower_builder import (
    ConvTowerConfig,
    ConvTowerType,
    ConfigurableConvTower,
    preset_conv_pos_neg,
)
from geofractal.router.components.fusion_component import AdaptiveFusion
from geofractal.router.components.address_component import AddressComponent


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BeatrixCollectiveConfig:
    """Configuration for the Beatrix tower collective."""

    # Dimensions
    dim: int = 256
    fingerprint_dim: int = 64

    # Geometric tower config
    geometric_depth: int = 2
    geometric_num_heads: int = 4
    geometric_ffn_mult: float = 4.0

    # Conv tower config
    conv_depth: int = 2
    conv_spatial_size: int = 16

    # Which towers to include
    geometric_types: List[str] = field(default_factory=lambda: [
        'cantor', 'beatrix', 'simplex', 'helix'
    ])
    conv_types: List[str] = field(default_factory=lambda: [
        'wide_resnet', 'frequency', 'squeeze_excite', 'bottleneck'
    ])

    # Whether to use pos/neg pairs
    use_signed_pairs: bool = True

    # Theta probe config
    num_theta_probes: int = 4
    theta_hidden_dim: int = 128

    # Fusion
    fusion_type: str = 'adaptive'

    @property
    def num_geometric_towers(self) -> int:
        mult = 2 if self.use_signed_pairs else 1
        return len(self.geometric_types) * mult

    @property
    def num_conv_towers(self) -> int:
        mult = 2 if self.use_signed_pairs else 1
        return len(self.conv_types) * mult

    @property
    def total_towers(self) -> int:
        return self.num_geometric_towers + self.num_conv_towers + self.num_theta_probes


# =============================================================================
# THETA PROBE TOWER
# =============================================================================

class ThetaProbeTower(BaseTower):
    """
    Lightweight theta probe as a proper BaseTower.

    Simple MLP for fine control signals.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        hidden_dim: int = 128,
        fingerprint_dim: int = 64,
    ):
        super().__init__(name, strict=False)

        self.dim = dim
        self._fingerprint_dim = fingerprint_dim

        # MLP stages
        self.append(nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ))
        self.append(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        ))
        self.append(nn.Linear(hidden_dim, dim))

        # Address component for fingerprint
        self.attach('address', AddressComponent(
            f'{name}_addr',
            fingerprint_dim=fingerprint_dim,
        ))

        # Cache for WideRouter integration
        self.objects['_cached_features'] = None

    @property
    def fingerprint(self) -> Tensor:
        return self['address'].fingerprint

    @property
    def cached_features(self) -> Optional[Tensor]:
        return self.objects.get('_cached_features')

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Pool if sequence
        if x.dim() == 3:
            x = x.mean(dim=1)

        h = x
        for stage in self.stages:
            h = stage(h)

        self.objects['_cached_features'] = h
        return h


# =============================================================================
# BEATRIX COLLECTIVE (WideRouter)
# =============================================================================

class BeatrixCollective(WideRouter):
    """
    Complete tower ensemble for the Beatrix oscillator.

    Extends WideRouter for automatic batched execution and
    torch.compile optimization.

    Usage:
        collective = BeatrixCollective(config)
        collective.network_to(device)
        collective.prepare_and_compile()

        outputs = collective.get_tower_forces(fused)  # List[Tensor]
    """

    def __init__(self, config: BeatrixCollectiveConfig):
        super().__init__('beatrix_collective', strict=False, auto_discover=False)

        self.config = config
        self._tower_order = []  # Track tower order for consistent output

        # Store config in objects
        self.objects['collective_config'] = config

        # Build geometric towers
        self._build_geometric_towers(config)

        # Build conv towers
        self._build_conv_towers(config)

        # Build theta probes
        self._build_theta_probes(config)

        # Fusion for cross-tower aggregation
        num_towers = len(self._tower_order)
        self.attach('fusion', AdaptiveFusion(
            'beatrix_fusion',
            num_inputs=num_towers,
            in_features=config.dim,
        ))

        # Input projection
        self.attach('input_proj', nn.Linear(config.dim, config.dim))

        # Fingerprint aggregation
        self.attach('fp_proj', nn.Linear(
            config.fingerprint_dim * num_towers,
            config.fingerprint_dim
        ))

        # Discover towers for WideRouter
        self.discover_towers()

    def _build_geometric_towers(self, config: BeatrixCollectiveConfig):
        """Build and attach geometric (transformer) towers."""
        if config.use_signed_pairs:
            tower_configs = preset_pos_neg_pairs(config.geometric_types)
        else:
            tower_configs = [
                TowerConfig(name, rope=name, address=name)
                for name in config.geometric_types
            ]

        for cfg in tower_configs:
            tower = ConfigurableTower(
                config=cfg,
                default_dim=config.dim,
                default_depth=config.geometric_depth,
                default_num_heads=config.geometric_num_heads,
                default_ffn_mult=config.geometric_ffn_mult,
                default_fingerprint_dim=config.fingerprint_dim,
            )
            self.attach(cfg.name, tower)
            self.register_tower(cfg.name)
            self._tower_order.append(cfg.name)

    def _build_conv_towers(self, config: BeatrixCollectiveConfig):
        """Build and attach convolutional towers."""
        if config.use_signed_pairs:
            tower_configs = preset_conv_pos_neg(config.conv_types)
        else:
            tower_configs = [
                ConvTowerConfig(name, tower_type=name)
                for name in config.conv_types
            ]

        for cfg in tower_configs:
            tower = ConfigurableConvTower(
                config=cfg,
                default_dim=config.dim,
                default_depth=config.conv_depth,
                default_fingerprint_dim=config.fingerprint_dim,
                spatial_size=config.conv_spatial_size,
            )
            self.attach(cfg.name, tower)
            self.register_tower(cfg.name)
            self._tower_order.append(cfg.name)

    def _build_theta_probes(self, config: BeatrixCollectiveConfig):
        """Build and attach theta probe towers."""
        for i in range(config.num_theta_probes):
            name = f'theta_{i}'
            tower = ThetaProbeTower(
                name=name,
                dim=config.dim,
                hidden_dim=config.theta_hidden_dim,
                fingerprint_dim=config.fingerprint_dim,
            )
            self.attach(name, tower)
            self.register_tower(name)
            self._tower_order.append(name)

    @property
    def tower_names(self) -> List[str]:
        """Ordered list of tower names."""
        return self._tower_order.copy()

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_all: bool = False,
    ) -> Union[List[Tensor], Dict[str, Any]]:
        """
        Forward pass through all towers.

        Args:
            x: Input [B, D] or [B, L, D]
            mask: Optional attention mask
            return_all: If True, return diagnostics dict

        Returns:
            If return_all=False: List of tower output tensors
            If return_all=True: Dict with outputs, fused, fingerprint, etc.
        """
        # Ensure 3D for tower processing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]

        B = x.shape[0]

        # Input projection
        x = self['input_proj'](x)

        # Execute all towers via WideRouter
        opinions_dict = self.wide_forward(x, mask=mask)

        # Collect outputs in consistent order
        outputs = []
        fingerprints = []

        for name in self._tower_order:
            opinion = opinions_dict[name]

            # Pool if sequence
            if opinion.dim() == 3:
                opinion = opinion.mean(dim=1)

            outputs.append(opinion)

            # Get fingerprint
            tower = self[name]
            fp = tower.fingerprint
            if fp.dim() == 1:
                fp = fp.unsqueeze(0).expand(B, -1)
            fingerprints.append(fp)

        if not return_all:
            return outputs

        # Fuse all opinions
        fused = self['fusion'](*outputs)

        # Aggregate fingerprints
        fp_cat = torch.cat(fingerprints, dim=-1)
        collective_fp = F.normalize(self['fp_proj'](fp_cat), dim=-1)

        return {
            'outputs': outputs,
            'fused': fused,
            'fingerprint': collective_fp,
            'opinions': opinions_dict,
            'num_towers': len(outputs),
        }

    def get_tower_forces(self, x: Tensor, mask: Optional[Tensor] = None) -> List[Tensor]:
        """Get tower outputs as force vectors for oscillator."""
        return self.forward(x, mask=mask, return_all=False)

    def get_fingerprint(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Get collective fingerprint for routing."""
        result = self.forward(x, mask=mask, return_all=True)
        return result['fingerprint']


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_beatrix_collective(
    dim: int = 256,
    fingerprint_dim: int = 64,
    geometric_types: List[str] = None,
    conv_types: List[str] = None,
    num_theta_probes: int = 4,
    use_signed_pairs: bool = True,
) -> BeatrixCollective:
    """
    Factory function for creating a Beatrix collective.
    """
    config = BeatrixCollectiveConfig(
        dim=dim,
        fingerprint_dim=fingerprint_dim,
        geometric_types=geometric_types or ['cantor', 'beatrix', 'simplex', 'helix'],
        conv_types=conv_types or ['wide_resnet', 'frequency', 'squeeze_excite', 'bottleneck'],
        num_theta_probes=num_theta_probes,
        use_signed_pairs=use_signed_pairs,
    )
    return BeatrixCollective(config)


def create_minimal_collective(dim: int = 256) -> BeatrixCollective:
    """Minimal collective for testing."""
    config = BeatrixCollectiveConfig(
        dim=dim,
        geometric_types=['cantor', 'beatrix'],
        conv_types=['wide_resnet'],
        num_theta_probes=2,
        use_signed_pairs=True,
    )
    return BeatrixCollective(config)


def create_full_collective(dim: int = 256) -> BeatrixCollective:
    """Full collective with all tower types."""
    config = BeatrixCollectiveConfig(
        dim=dim,
        geometric_types=[
            'cantor', 'beatrix', 'simplex', 'helix',
            'golden', 'fibonacci', 'sinusoidal', 'fractal'
        ],
        conv_types=[
            'wide_resnet', 'frequency', 'squeeze_excite', 'bottleneck',
            'dilated', 'coarse_fine', 'inverted_bottleneck', 'spatial_attention'
        ],
        num_theta_probes=8,
        use_signed_pairs=True,
    )
    return BeatrixCollective(config)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    import time

    print("=" * 60)
    print("  BEATRIX COLLECTIVE (WideRouter)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create collective
    config = BeatrixCollectiveConfig(
        dim=256,
        geometric_types=['cantor', 'beatrix', 'simplex', 'helix'],
        conv_types=['wide_resnet', 'frequency'],
        num_theta_probes=4,
        use_signed_pairs=True,
    )

    collective = BeatrixCollective(config)
    collective.network_to(device=device)

    print(f"\nConfig:")
    print(f"  Geometric towers: {config.num_geometric_towers}")
    print(f"  Conv towers: {config.num_conv_towers}")
    print(f"  Theta probes: {config.num_theta_probes}")
    print(f"  Total: {config.total_towers}")

    print(f"\nTower names: {collective.tower_names}")
    print(f"Parameters: {sum(p.numel() for p in collective.parameters()):,}")

    # Test input
    B = 2
    x = torch.randn(B, 256, device=device)

    print("\n--- Tower Outputs ---")
    outputs = collective.get_tower_forces(x)
    print(f"Number of outputs: {len(outputs)}")
    for name, out in zip(collective.tower_names, outputs):
        print(f"  {name}: {out.shape}, norm={out.norm():.4f}")

    print("\n--- Full Diagnostics ---")
    result = collective(x, return_all=True)
    print(f"Fused: {result['fused'].shape}")
    print(f"Fingerprint: {result['fingerprint'].shape}")
    print(f"Num towers: {result['num_towers']}")

    # Compile and benchmark
    print("\n--- Compile & Benchmark ---")
    collective.analyze_structure()
    compiled = collective.prepare_and_compile()

    # Warmup
    for _ in range(5):
        _ = compiled(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(50):
        outputs = compiled.get_tower_forces(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 50 * 1000

    print(f"Compiled forward: {elapsed:.2f}ms")
    print(f"Throughput: {B / (elapsed/1000):.0f} samples/sec")

    print("\n" + "=" * 60)
    print("  ✓ BEATRIX COLLECTIVE READY")
    print("=" * 60)