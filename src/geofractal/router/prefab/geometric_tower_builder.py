"""
geofractal.router.prefab.geometric_tower_builder
================================================

Flexible tower builder with BATCHED forward for tower groups.

Uses actual component signatures from:
- rope_component.py (RoPE, DualRoPE, TriRoPE, QuadRoPE, BeatrixRoPE, CantorRoPE)
- address_component.py (AddressComponent, SimplexAddressComponent, etc.)
- fusion_component.py (AdaptiveFusion, GatedFusion, etc.)
"""

from __future__ import annotations

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
from geofractal.router.components.transformer_component import (
    TransformerConfig,
    TransformerVariant,
    ActivationType,
    PreNormBlock,
)
from geofractal.router.components.rope_component import (
    RoPE,
    DualRoPE,
    TriRoPE,
    QuadRoPE,
    BeatrixRoPE,
    CantorRoPE,
    SinusoidalRoPE,
)
from geofractal.router.components.address_component import (
    AddressComponent,
    SimplexAddressComponent,
    SphericalAddressComponent,
    FractalAddressComponent,
    CantorAddressComponent,
)
from geofractal.router.components.fusion_component import (
    AdaptiveFusion,
    GatedFusion,
    AttentionFusion,
    ConcatFusion,
    SumFusion,
)


# =============================================================================
# TYPE ENUMS
# =============================================================================

class RoPEType(str, Enum):
    STANDARD = "standard"
    DUAL = "dual"
    TRI = "tri"
    QUAD = "quad"
    CANTOR = "cantor"
    BEATRIX = "beatrix"
    HELIX = "helix"      # -> TriRoPE
    SIMPLEX = "simplex"  # -> QuadRoPE
    FRACTAL = "fractal"  # -> DualRoPE
    SINUSOIDAL = "sinusoidal"  # -> SinusoidalRoPE (pure harmonic)


class AddressType(str, Enum):
    STANDARD = "standard"
    SIMPLEX = "simplex"
    SPHERICAL = "spherical"
    FRACTAL = "fractal"
    CANTOR = "cantor"
    BEATRIX = "beatrix"  # -> CantorAddressComponent with mode='staircase'
    HELIX = "helix"      # -> SphericalAddressComponent
    SINUSOIDAL = "sinusoidal"  # -> AddressComponent (standard, pairs with SinusoidalRoPE)


class FusionType(str, Enum):
    ADAPTIVE = "adaptive"
    GATED = "gated"
    ATTENTION = "attention"
    MEAN = "mean"
    CONCAT = "concat"
    SUM = "sum"


# =============================================================================
# TOWER CONFIG
# =============================================================================

@dataclass
class TowerConfig:
    """Configuration for a single tower."""

    name: str
    rope: Union[RoPEType, str] = RoPEType.STANDARD
    address: Union[AddressType, str] = AddressType.STANDARD
    inverted: bool = False

    dim: Optional[int] = None
    depth: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    ffn_mult: Optional[float] = None
    dropout: Optional[float] = None
    fingerprint_dim: Optional[int] = None

    rope_params: Dict[str, Any] = field(default_factory=dict)
    address_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.rope, str):
            self.rope = RoPEType(self.rope)
        if isinstance(self.address, str):
            self.address = AddressType(self.address)

    @property
    def structure_signature(self) -> str:
        """Towers with same signature can be batched together."""
        return f"{self.rope.value}_{self.address.value}"


# =============================================================================
# INVERTED ADDRESS WRAPPER
# =============================================================================

class InvertedAddressWrapper(nn.Module):
    def __init__(self, base_address: nn.Module):
        super().__init__()
        self.base_address = base_address

    @property
    def fingerprint(self) -> Tensor:
        return 1.0 - F.normalize(self.base_address.fingerprint, dim=-1)

    def forward(self, *args, **kwargs):
        return self.base_address(*args, **kwargs)


# =============================================================================
# COMPONENT FACTORIES (using actual signatures)
# =============================================================================

def build_rope(name: str, rope_type: RoPEType, head_dim: int, **kwargs) -> nn.Module:
    """Build RoPE component using actual signatures from rope_component.py."""

    if rope_type == RoPEType.CANTOR:
        # CantorRoPE(name, head_dim, theta, levels, tau, mode, blend_alpha, ...)
        return CantorRoPE(
            f'{name}_rope',
            head_dim=head_dim,
            theta=kwargs.get('theta', 10000.0),
            levels=kwargs.get('levels', 5),
            tau=kwargs.get('tau', 0.25),
            mode=kwargs.get('mode', 'hybrid'),
        )

    elif rope_type == RoPEType.BEATRIX:
        # BeatrixRoPE(name, head_dim, theta, levels, alpha, tau, ...)
        return BeatrixRoPE(
            f'{name}_rope',
            head_dim=head_dim,
            theta=kwargs.get('theta', 10000.0),
            levels=kwargs.get('levels', 5),
            alpha=kwargs.get('alpha', 0.5),
            tau=kwargs.get('tau', 0.25),
        )

    elif rope_type in (RoPEType.HELIX, RoPEType.TRI):
        # TriRoPE(name, head_dim, theta_alpha, theta_beta, theta_gamma, augmentation, ...)
        return TriRoPE(
            f'{name}_rope',
            head_dim=head_dim,
            theta_alpha=kwargs.get('theta_alpha', 10000.0),
            theta_beta=kwargs.get('theta_beta', 5000.0),
            theta_gamma=kwargs.get('theta_gamma', 2500.0),
            augmentation=kwargs.get('augmentation', 'barycentric'),
        )

    elif rope_type in (RoPEType.SIMPLEX, RoPEType.QUAD):
        # QuadRoPE(name, head_dim, theta_w, theta_x, theta_y, theta_z, augmentation, ...)
        return QuadRoPE(
            f'{name}_rope',
            head_dim=head_dim,
            theta_w=kwargs.get('theta_w', 10000.0),
            theta_x=kwargs.get('theta_x', 5000.0),
            theta_y=kwargs.get('theta_y', 2500.0),
            theta_z=kwargs.get('theta_z', 1000.0),
            augmentation=kwargs.get('augmentation', 'simplex'),
        )

    elif rope_type in (RoPEType.FRACTAL, RoPEType.DUAL):
        # DualRoPE(name, head_dim, theta_primary, theta_secondary, augmentation, ...)
        return DualRoPE(
            f'{name}_rope',
            head_dim=head_dim,
            theta_primary=kwargs.get('theta_primary', 10000.0),
            theta_secondary=kwargs.get('theta_secondary', 3000.0),
            augmentation=kwargs.get('augmentation', 'lerp'),
        )

    elif rope_type == RoPEType.SINUSOIDAL:
        # SinusoidalRoPE - pure harmonic control with learnable phase/amplitude
        return SinusoidalRoPE(
            f'{name}_rope',
            head_dim=head_dim,
            base_freq=kwargs.get('base_freq', 1.0),
            num_harmonics=kwargs.get('num_harmonics', None),
            learnable_phase=kwargs.get('learnable_phase', True),
            learnable_amplitude=kwargs.get('learnable_amplitude', True),
        )

    else:  # STANDARD
        # RoPE(name, head_dim, theta, theta_scale, ...)
        return RoPE(
            f'{name}_rope',
            head_dim=head_dim,
            theta=kwargs.get('theta', 10000.0),
            theta_scale=kwargs.get('theta_scale', 1.0),
        )


def build_address(name: str, address_type: AddressType, fingerprint_dim: int, **kwargs) -> nn.Module:
    """
    Build Address component using actual signatures from address_component.py.

    Signatures:
        AddressComponent(name, fingerprint_dim, init_scale=0.02)
        SimplexAddressComponent(name, k=4, embed_dim=64, method='regular', learnable=True)
        SphericalAddressComponent(name, fingerprint_dim)
        FractalAddressComponent(name, region='seahorse', orbit_length=64, learnable=True)
        CantorAddressComponent(name, k_simplex=4, fingerprint_dim=64, mode='staircase', tau=0.25)
    """

    if address_type == AddressType.CANTOR:
        return CantorAddressComponent(
            f'{name}_addr',
            k_simplex=kwargs.get('k_simplex', 4),
            fingerprint_dim=fingerprint_dim,
            mode=kwargs.get('mode', 'staircase'),
            tau=kwargs.get('tau', 0.25),
        )

    elif address_type == AddressType.BEATRIX:
        # Beatrix RoPE already has Devil's Staircase - use learned address for unique identity
        return AddressComponent(
            f'{name}_addr',
            fingerprint_dim=fingerprint_dim,
            init_scale=kwargs.get('init_scale', 0.02),
        )

    elif address_type == AddressType.SIMPLEX:
        return SimplexAddressComponent(
            f'{name}_addr',
            k=kwargs.get('k', 4),
            embed_dim=fingerprint_dim,
            method=kwargs.get('method', 'regular'),
            learnable=kwargs.get('learnable', True),
        )

    elif address_type in (AddressType.SPHERICAL, AddressType.HELIX):
        return SphericalAddressComponent(
            f'{name}_addr',
            fingerprint_dim=fingerprint_dim,
        )

    elif address_type == AddressType.FRACTAL:
        return FractalAddressComponent(
            f'{name}_addr',
            region=kwargs.get('region', 'seahorse'),
            orbit_length=fingerprint_dim,
            learnable=kwargs.get('learnable', True),
        )

    elif address_type == AddressType.SINUSOIDAL:
        # Sinusoidal uses standard learned address
        return AddressComponent(
            f'{name}_addr',
            fingerprint_dim=fingerprint_dim,
            init_scale=kwargs.get('init_scale', 0.02),
        )

    else:  # STANDARD
        return AddressComponent(
            f'{name}_addr',
            fingerprint_dim=fingerprint_dim,
            init_scale=kwargs.get('init_scale', 0.02),
        )


def build_fusion(name: str, fusion_type: FusionType, num_inputs: int, in_features: int, **kwargs) -> nn.Module:
    """Build Fusion component using actual signatures from fusion_component.py."""

    if fusion_type == FusionType.ADAPTIVE:
        # AdaptiveFusion(name, num_inputs, in_features, hidden_features, temperature, ...)
        return AdaptiveFusion(
            name,
            num_inputs=num_inputs,
            in_features=in_features,
            hidden_features=kwargs.get('hidden_features'),
            temperature=kwargs.get('temperature', 1.0),
        )

    elif fusion_type == FusionType.GATED:
        # GatedFusion(name, num_inputs, in_features, ...)
        return GatedFusion(name, num_inputs=num_inputs, in_features=in_features)

    elif fusion_type == FusionType.ATTENTION:
        # AttentionFusion(name, num_inputs, in_features, num_heads, dropout, ...)
        return AttentionFusion(
            name,
            num_inputs=num_inputs,
            in_features=in_features,
            num_heads=kwargs.get('num_heads', 4),
        )

    elif fusion_type == FusionType.CONCAT:
        # ConcatFusion(name, num_inputs, in_features, out_features, ...)
        return ConcatFusion(
            name,
            num_inputs=num_inputs,
            in_features=in_features,
            out_features=in_features,
        )

    elif fusion_type == FusionType.SUM:
        # SumFusion(name, num_inputs, in_features, normalize, ...)
        return SumFusion(
            name,
            num_inputs=num_inputs,
            in_features=in_features,
            normalize=kwargs.get('normalize', True),
        )

    else:  # MEAN - use SumFusion with normalize
        return SumFusion(name, num_inputs=num_inputs, in_features=in_features, normalize=True)


# =============================================================================
# SAFE ATTACH MIXIN
# =============================================================================

class SafeAttachMixin:
    def attach(self, name: str, component: Any) -> None:
        if isinstance(component, nn.Module):
            self.components[name] = component
        else:
            self.objects[name] = component

        if hasattr(component, 'parent') and hasattr(component, 'on_attach'):
            object.__setattr__(component, 'parent', self)
            component.on_attach(self)


# =============================================================================
# CONFIGURABLE TOWER
# =============================================================================

class ConfigurableTower(SafeAttachMixin, BaseTower):
    """Tower built from TowerConfig."""

    def __init__(
        self,
        config: TowerConfig,
        default_dim: int = 256,
        default_depth: int = 2,
        default_num_heads: int = 4,
        default_head_dim: int = 64,
        default_ffn_mult: float = 4.0,
        default_dropout: float = 0.1,
        default_fingerprint_dim: int = 64,
    ):
        super().__init__(config.name, strict=False)

        dim = config.dim or default_dim
        depth = config.depth or default_depth
        num_heads = config.num_heads or default_num_heads
        head_dim = config.head_dim or default_head_dim
        ffn_mult = config.ffn_mult or default_ffn_mult
        dropout = config.dropout or default_dropout
        fingerprint_dim = config.fingerprint_dim or default_fingerprint_dim

        self.dim = dim
        self.depth = depth
        self.config = config
        self._fingerprint_dim = fingerprint_dim
        self._inverted = config.inverted

        self.objects['tower_config'] = {
            'name': config.name,
            'rope': config.rope.value,
            'address': config.address.value,
            'inverted': config.inverted,
            'dim': dim,
            'depth': depth,
        }

        tf_config = TransformerConfig(
            dim=dim,
            num_heads=num_heads,
            ffn_mult=ffn_mult,
            variant=TransformerVariant.PRENORM,
            activation=ActivationType.GELU,
            dropout=dropout,
            depth=depth,
        )

        # Build RoPE
        rope = build_rope(config.name, config.rope, head_dim, **config.rope_params)
        self.attach('rope', rope)

        # Build Address
        address = build_address(config.name, config.address, fingerprint_dim, **config.address_params)
        if config.inverted:
            address = InvertedAddressWrapper(address)
        self.attach('address', address)

        # Transformer blocks
        for i in range(depth):
            block = PreNormBlock(
                name=f'{config.name}_block_{i}',
                config=tf_config,
                block_idx=i,
            )
            self.append(block)

        self.attach('final_norm', nn.LayerNorm(dim))
        self.attach('opinion_proj', nn.Linear(dim, dim))

    @property
    def fingerprint(self) -> Tensor:
        return F.normalize(self['address'].fingerprint, dim=-1)

    @property
    def is_inverted(self) -> bool:
        return self._inverted

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        for block in self.stages:
            x, _ = block(x, mask=mask)

        features = self['final_norm'](x)
        pooled = features.mean(dim=1)
        opinion = self['opinion_proj'](pooled)

        # Cache features for retrieval after wide_forward
        self.objects['_cached_features'] = features

        return opinion, features

    @property
    def cached_features(self) -> Optional[Tensor]:
        """Features from last forward pass (for WideRouter integration)."""
        return self.objects.get('_cached_features')


# =============================================================================
# TOWER OPINION CONTAINERS
# =============================================================================

@dataclass
class TowerOpinion:
    name: str
    opinion: Tensor
    features: Tensor
    fingerprint: Tensor
    config: TowerConfig


@dataclass
class CollectiveOpinion:
    fused: Tensor
    opinions: Dict[str, TowerOpinion]
    weights: Optional[Tensor]
    collective_fingerprint: Tensor


# =============================================================================
# CONFIGURABLE COLLECTIVE (using WideRouter)
# =============================================================================

class ConfigurableCollective(SafeAttachMixin, WideRouter):
    """Tower collective using WideRouter for optimized execution."""

    def __init__(
        self,
        name: str,
        tower_configs: List[TowerConfig],
        dim: int = 256,
        default_depth: int = 2,
        num_heads: int = 4,
        head_dim: int = 64,
        ffn_mult: float = 4.0,
        dropout: float = 0.1,
        fingerprint_dim: int = 64,
        fusion_type: Union[FusionType, str] = FusionType.ADAPTIVE,
        fusion_params: Dict[str, Any] = None,
    ):
        # Initialize WideRouter with auto_discover=False (we'll register manually)
        super().__init__(name, strict=False, auto_discover=False)

        self.dim = dim
        self._fingerprint_dim = fingerprint_dim
        self._num_towers = len(tower_configs)

        if isinstance(fusion_type, str):
            fusion_type = FusionType(fusion_type)

        self.objects['tower_configs'] = tower_configs
        self.objects['collective_config'] = {
            'dim': dim,
            'num_towers': self._num_towers,
            'fusion_type': fusion_type.value,
        }

        # Build and attach towers
        for config in tower_configs:
            tower = ConfigurableTower(
                config=config,
                default_dim=dim,
                default_depth=default_depth,
                default_num_heads=num_heads,
                default_head_dim=head_dim,
                default_ffn_mult=ffn_mult,
                default_dropout=dropout,
                default_fingerprint_dim=fingerprint_dim,
            )
            self.attach(config.name, tower)
            # Register with WideRouter for optimized execution
            self.register_tower(config.name)

        # Fusion (attached after towers so it's not auto-discovered)
        fusion_params = fusion_params or {}
        fusion = build_fusion(f'{name}_fusion', fusion_type,
                              num_inputs=self._num_towers, in_features=dim, **fusion_params)
        self.attach('fusion', fusion)

        # Input projection
        self.attach('input_proj', nn.Linear(dim, dim))

        # Fingerprint aggregation
        self.attach('fp_proj', nn.Linear(fingerprint_dim * self._num_towers, fingerprint_dim))

        self.objects['debug'] = False
        self.objects['debug_info'] = {}

    @property
    def towers(self) -> Dict[str, ConfigurableTower]:
        """Dict of tower name -> ConfigurableTower."""
        return {name: self[name] for name in self.tower_names}

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> CollectiveOpinion:
        """Forward using WideRouter's optimized wide_forward."""
        debug = self.objects['debug']
        debug_info = self.objects['debug_info']

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self['input_proj'](x)

        # Use WideRouter's optimized execution
        # Returns Dict[tower_name, opinion_tensor]
        opinions_dict = self.wide_forward(x, mask=mask)

        # Build TowerOpinion objects with cached features
        all_opinions = []
        opinion_tensors = []

        for tower_name in self.tower_names:
            tower = self[tower_name]
            opinion = opinions_dict[tower_name]
            features = tower.cached_features  # Retrieved from tower's cache

            tower_opinion = TowerOpinion(
                name=tower_name,
                opinion=opinion,
                features=features,
                fingerprint=tower.fingerprint,
                config=tower.config,
            )
            all_opinions.append(tower_opinion)
            opinion_tensors.append(opinion)

            if debug:
                debug_info[f'{tower_name}_opinion_norm'] = opinion.norm(dim=-1).mean().item()

        # Fuse
        fused = self['fusion'](*opinion_tensors)

        # Aggregate fingerprints
        fp_stack = torch.cat([op.fingerprint for op in all_opinions], dim=-1)
        collective_fp = F.normalize(self['fp_proj'](fp_stack), dim=-1)

        if debug:
            debug_info['fused_norm'] = fused.norm(dim=-1).mean().item()

        return CollectiveOpinion(
            fused=fused,
            opinions={op.name: op for op in all_opinions},
            weights=None,
            collective_fingerprint=collective_fp,
        )

    def tower_fingerprints(self) -> Dict[str, Tensor]:
        return {name: self[name].fingerprint for name in self.tower_names}

    def debug_on(self):
        self.objects['debug'] = True

    def debug_off(self):
        self.objects['debug'] = False


# =============================================================================
# BUILDER FUNCTION
# =============================================================================

def build_tower_collective(
    configs: List[TowerConfig],
    dim: int = 256,
    default_depth: int = 2,
    num_heads: int = 4,
    ffn_mult: float = 4.0,
    dropout: float = 0.1,
    fingerprint_dim: int = 64,
    fusion_type: str = 'adaptive',
    name: str = 'tower_collective',
) -> ConfigurableCollective:
    """Build a tower collective with batched forward."""
    return ConfigurableCollective(
        name=name,
        tower_configs=configs,
        dim=dim,
        default_depth=default_depth,
        num_heads=num_heads,
        head_dim=dim // num_heads,
        ffn_mult=ffn_mult,
        dropout=dropout,
        fingerprint_dim=fingerprint_dim,
        fusion_type=fusion_type,
    )


# =============================================================================
# PRESET CONFIGS
# =============================================================================

def preset_pos_neg_pairs(
    geometries: List[str] = ['cantor', 'beatrix', 'helix', 'simplex', 'sinusoidal'],
) -> List[TowerConfig]:
    """Create pos/neg pairs for each geometry."""
    configs = []
    for geom in geometries:
        configs.append(TowerConfig(f'{geom}_pos', rope=geom, address=geom, inverted=False))
        configs.append(TowerConfig(f'{geom}_neg', rope=geom, address=geom, inverted=True))
    return configs


def preset_all_six() -> List[TowerConfig]:
    """Original 6-tower config."""
    return [
        TowerConfig('cantor', rope='cantor', address='cantor'),
        TowerConfig('beatrix', rope='beatrix', address='beatrix'),
        TowerConfig('helix', rope='helix', address='helix'),
        TowerConfig('simplex', rope='simplex', address='simplex'),
        TowerConfig('fractal', rope='fractal', address='fractal'),
        TowerConfig('standard', rope='standard', address='standard'),
    ]


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    import time

    print("=" * 60)
    print("WideRouter Tower Collective Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    configs = preset_pos_neg_pairs(['cantor', 'beatrix', 'helix', 'simplex'])
    print(f"Towers: {len(configs)}")

    collective = build_tower_collective(configs, dim=256, default_depth=1)
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
        out = collective(x)
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
        out = compiled(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    compiled_ms = (time.perf_counter() - t0) / 50 * 1000

    print(f"Compiled forward: {compiled_ms:.2f}ms")
    print(f"Speedup: {eager_ms/compiled_ms:.2f}x")

    print(f"\nInput: {x.shape}")
    print(f"Fused: {out.fused.shape}")
    print(f"Throughput: {B / (compiled_ms/1000):.0f} samples/sec")

    print("\n✓ WideRouter collective ready")