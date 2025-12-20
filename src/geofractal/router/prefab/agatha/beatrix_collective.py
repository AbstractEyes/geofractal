"""
BEATRIX COLLECTIVE (Using Existing Tower Builders)
==================================================

Integrates with:
- geometric_tower_builder.py (ConfigurableCollective, RoPE variants)
- geometric_conv_tower_builder.py (ConvTowerCollective, WideResNet etc.)

Architecture:
    Head Router → Fused Mail [B, D]
                     ↓
         ┌──────────┴──────────┐
         ↓                     ↓
    Geometric Towers      Conv Towers
    (Transformer-based)   (CNN-based)
    - cantor ±           - wide_resnet ±
    - beatrix ±          - frequency ±
    - simplex ±          - squeeze_excite ±
    - helix ±            - bottleneck ±
         │                     │
         └──────────┬──────────┘
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
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import existing tower builders
from geofractal.router.prefab.geometric_tower_builder import (
    TowerConfig,
    RoPEType,
    AddressType,
    FusionType,
    ConfigurableCollective,
    build_tower_collective,
    preset_pos_neg_pairs,
    preset_all_eight,
)
from geofractal.router.prefab.geometric_conv_tower_builder import (
    ConvTowerConfig,
    ConvTowerType,
    ConvTowerCollective,
    build_conv_collective,
    preset_conv_pos_neg,
)
from geofractal.router.components.fusion_component import AdaptiveFusion


# =============================================================================
# BEATRIX COLLECTIVE CONFIG
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

    # Theta probe config (lightweight MLPs)
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
# THETA PROBE (Lightweight MLP)
# =============================================================================

class ThetaProbe(nn.Module):
    """
    Lightweight auxiliary probe for fine control signals.

    These don't have geometric structure - just simple MLPs
    that provide additional degrees of freedom for the oscillator.
    """

    def __init__(
        self,
        name: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        fingerprint_dim: int = 64,
    ):
        super().__init__()
        self.name = name
        self._fingerprint_dim = fingerprint_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Learnable fingerprint
        self._fingerprint = nn.Parameter(
            torch.randn(fingerprint_dim) * 0.02
        )

    @property
    def fingerprint(self) -> Tensor:
        return F.normalize(self._fingerprint, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# =============================================================================
# BEATRIX COLLECTIVE
# =============================================================================

class BeatrixCollective(nn.Module):
    """
    Complete tower ensemble for the Beatrix oscillator.

    Combines:
        - Geometric towers (transformer-based with RoPE variants)
        - Conv towers (CNN-based with various architectures)
        - Theta probes (lightweight MLPs)

    All towers produce forces that the oscillator integrates.
    """

    def __init__(self, config: BeatrixCollectiveConfig):
        super().__init__()
        self.config = config

        # Build geometric tower configs
        if config.use_signed_pairs:
            geom_configs = preset_pos_neg_pairs(config.geometric_types)
        else:
            geom_configs = [
                TowerConfig(name, rope=name, address=name)
                for name in config.geometric_types
            ]

        # Build geometric collective
        self.geometric_collective = build_tower_collective(
            configs=geom_configs,
            dim=config.dim,
            default_depth=config.geometric_depth,
            num_heads=config.geometric_num_heads,
            ffn_mult=config.geometric_ffn_mult,
            fingerprint_dim=config.fingerprint_dim,
            fusion_type=config.fusion_type,
            name='beatrix_geometric',
        )

        # Build conv tower configs
        if config.use_signed_pairs:
            conv_configs = preset_conv_pos_neg(config.conv_types)
        else:
            conv_configs = [
                ConvTowerConfig(name, tower_type=name)
                for name in config.conv_types
            ]

        # Build conv collective
        self.conv_collective = build_conv_collective(
            configs=conv_configs,
            dim=config.dim,
            default_depth=config.conv_depth,
            fingerprint_dim=config.fingerprint_dim,
            spatial_size=config.conv_spatial_size,
            name='beatrix_conv',
        )

        # Theta probes
        self.theta_probes = nn.ModuleList([
            ThetaProbe(
                name=f'theta_{i}',
                input_dim=config.dim,
                hidden_dim=config.theta_hidden_dim,
                output_dim=config.dim,
                fingerprint_dim=config.fingerprint_dim,
            )
            for i in range(config.num_theta_probes)
        ])

        # Cross-collective fusion (3 inputs: geom_fused, conv_fused, theta_combined)
        self.cross_fusion = AdaptiveFusion(
            'beatrix_cross_fusion',
            num_inputs=3,
            in_features=config.dim,
        )

        # Fingerprint aggregation (3 fingerprints: geom, conv, theta)
        self.fp_proj = nn.Linear(
            config.fingerprint_dim * 3,
            config.fingerprint_dim
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_all: bool = False,
    ) -> Union[List[Tensor], Dict[str, Any]]:
        """
        Compute tower outputs for oscillator.

        Args:
            x: Input tensor [B, D] or [B, L, D]
            mask: Optional attention mask
            return_all: If True, return full diagnostics

        Returns:
            If return_all=False: List of tower force tensors
            If return_all=True: Dict with outputs, fingerprints, diagnostics
        """
        # Ensure 3D for collectives
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)  # [B, 1, D]
        else:
            x_seq = x

        B = x_seq.shape[0]

        # Run geometric collective
        geom_result = self.geometric_collective(x_seq, mask=mask)
        geom_fused = geom_result.fused  # [B, L, D] or [B, D]
        geom_opinions = geom_result.opinions  # Dict[name, TowerOpinion]
        geom_fp = geom_result.collective_fingerprint

        # Run conv collective
        conv_fused, conv_opinions = self.conv_collective(x_seq, mask=mask)
        # Get conv fingerprints
        conv_fps = [self.conv_collective[name].fingerprint
                    for name in self.conv_collective.tower_names]
        conv_fp = torch.stack(conv_fps, dim=0).mean(dim=0)  # Aggregate

        # Run theta probes
        x_pooled = x_seq.mean(dim=1) if x_seq.dim() == 3 else x_seq  # [B, D]
        theta_outputs = [probe(x_pooled) for probe in self.theta_probes]
        theta_fps = [probe.fingerprint for probe in self.theta_probes]
        theta_fp = torch.stack(theta_fps, dim=0).mean(dim=0)

        # Collect all tower outputs for oscillator
        all_outputs = []

        # Geometric tower outputs (in order)
        for name in self.geometric_collective.tower_names:
            opinion = geom_opinions[name].opinion
            # Pool if sequence
            if opinion.dim() == 3:
                opinion = opinion.mean(dim=1)
            all_outputs.append(opinion)

        # Conv tower outputs (in order)
        for name in self.conv_collective.tower_names:
            opinion = conv_opinions[name]
            if opinion.dim() == 3:
                opinion = opinion.mean(dim=1)
            all_outputs.append(opinion)

        # Theta outputs
        all_outputs.extend(theta_outputs)

        if not return_all:
            return all_outputs

        # Cross-collective fusion
        geom_pooled = geom_fused.mean(dim=1) if geom_fused.dim() == 3 else geom_fused
        conv_pooled = conv_fused.mean(dim=1) if conv_fused.dim() == 3 else conv_fused
        theta_combined = torch.stack(theta_outputs, dim=0).mean(dim=0)

        cross_fused = self.cross_fusion(geom_pooled, conv_pooled, theta_combined)

        # Aggregate fingerprints
        # Handle different fingerprint shapes
        if geom_fp.dim() == 1:
            geom_fp = geom_fp.unsqueeze(0).expand(B, -1)
        if conv_fp.dim() == 1:
            conv_fp = conv_fp.unsqueeze(0).expand(B, -1)
        if theta_fp.dim() == 1:
            theta_fp = theta_fp.unsqueeze(0).expand(B, -1)

        fp_cat = torch.cat([geom_fp, conv_fp, theta_fp], dim=-1)
        collective_fp = F.normalize(self.fp_proj(fp_cat), dim=-1)

        return {
            'outputs': all_outputs,
            'fused': cross_fused,
            'fingerprint': collective_fp,
            'geometric': {
                'fused': geom_fused,
                'opinions': geom_opinions,
                'fingerprint': geom_fp,
            },
            'conv': {
                'fused': conv_fused,
                'opinions': conv_opinions,
                'fingerprint': conv_fp,
            },
            'theta': {
                'outputs': theta_outputs,
                'fingerprint': theta_fp,
            },
            'num_towers': len(all_outputs),
        }

    def get_tower_forces(self, x: Tensor, mask: Optional[Tensor] = None) -> List[Tensor]:
        """Alias for forward() with return_all=False."""
        return self.forward(x, mask=mask, return_all=False)

    def get_fingerprint(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Get collective fingerprint for routing."""
        result = self.forward(x, mask=mask, return_all=True)
        return result['fingerprint']

    @property
    def tower_names(self) -> List[str]:
        """List of all tower names in output order."""
        names = []
        names.extend(self.geometric_collective.tower_names)
        names.extend(self.conv_collective.tower_names)
        names.extend([f'theta_{i}' for i in range(len(self.theta_probes))])
        return names

    def network_to(self, device: torch.device):
        """Move entire network to device (for WideRouter compatibility)."""
        self.to(device)
        self.geometric_collective.network_to(device=device)
        self.conv_collective.network_to(device=device)
        return self

    def prepare_and_compile(self):
        """Compile sub-collectives for optimized execution."""
        self.geometric_collective.prepare_and_compile()
        self.conv_collective.prepare_and_compile()
        return self


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

    Args:
        dim: Embedding dimension
        fingerprint_dim: Fingerprint dimension
        geometric_types: List of geometric tower types (default: cantor, beatrix, simplex, helix)
        conv_types: List of conv tower types (default: wide_resnet, frequency, squeeze_excite, bottleneck)
        num_theta_probes: Number of theta probes
        use_signed_pairs: Whether to use pos/neg pairs
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


def create_minimal_beatrix_collective(dim: int = 256) -> BeatrixCollective:
    """Minimal collective for testing (4 geometric + 4 conv + 2 theta = 10 towers)."""
    config = BeatrixCollectiveConfig(
        dim=dim,
        geometric_types=['cantor', 'beatrix'],
        conv_types=['wide_resnet', 'frequency'],
        num_theta_probes=2,
        use_signed_pairs=True,
    )
    return BeatrixCollective(config)


def create_full_beatrix_collective(dim: int = 256) -> BeatrixCollective:
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
    print("=" * 60)
    print("  BEATRIX COLLECTIVE TEST (Using Tower Builders)")
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
    collective.network_to(device)

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
    for i, (name, out) in enumerate(zip(collective.tower_names, outputs)):
        print(f"  {name}: {out.shape}, norm={out.norm():.4f}")

    print("\n--- Full Diagnostics ---")
    result = collective(x, return_all=True)
    print(f"Fused: {result['fused'].shape}")
    print(f"Fingerprint: {result['fingerprint'].shape}")
    print(f"Num towers: {result['num_towers']}")

    # Compile test
    print("\n--- Compile Test ---")
    collective.prepare_and_compile()

    # Warmup
    for _ in range(3):
        _ = collective(x)

    import time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()
    for _ in range(20):
        outputs = collective.get_tower_forces(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = (time.perf_counter() - t0) / 20 * 1000

    print(f"Forward time: {elapsed:.2f}ms")
    print(f"Throughput: {B / (elapsed/1000):.0f} samples/sec")

    print("\n" + "=" * 60)
    print("  ✓ BEATRIX COLLECTIVE (TOWER BUILDERS) READY")
    print("=" * 60)