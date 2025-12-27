"""
geofractal.router.prefab.geometric_tower_builder
================================================

Flexible tower builder with BATCHED forward for tower groups.
Now with WalkerFusion support for static and learnable fusion.

Quick Construction:
    tower = quick_tower('cantor')
    pair = quick_pair('cantor', 'beatrix', use_inception=True)
    collective = quick_collective(['cantor', 'beatrix', 'helix'], fusion='walker_shiva')

Walker Fusion Options:
    - 'walker_static' / 'walker_shiva': Pure formula (0 learnable params)
    - 'walker_inception': Learnable modulation (~20k params)
    - 'walker_slerp', 'walker_lerp', 'walker_zeus': Other presets
    - Legacy: 'adaptive', 'gated', 'attention', 'concat', 'sum', 'mean'
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
from geofractal.router.components.walker_component import (
    WalkerFusion,
    WalkerInception,
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
    HELIX = "helix"
    SIMPLEX = "simplex"
    FRACTAL = "fractal"
    SINUSOIDAL = "sinusoidal"
    GOLDEN = "golden"
    FIBONACCI = "fibonacci"


class AddressType(str, Enum):
    STANDARD = "standard"
    SIMPLEX = "simplex"
    SPHERICAL = "spherical"
    FRACTAL = "fractal"
    CANTOR = "cantor"
    BEATRIX = "beatrix"
    HELIX = "helix"
    SINUSOIDAL = "sinusoidal"
    GOLDEN = "golden"
    FIBONACCI = "fibonacci"


class FusionType(str, Enum):
    # Legacy
    ADAPTIVE = "adaptive"
    GATED = "gated"
    ATTENTION = "attention"
    MEAN = "mean"
    CONCAT = "concat"
    SUM = "sum"
    # Walker
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
        return f"{self.rope.value}_{self.address.value}"


# =============================================================================
# GEOMETRY PRESETS
# =============================================================================

GEOMETRY_CONFIGS = {
    'cantor': ('cantor', 'cantor'),
    'beatrix': ('beatrix', 'beatrix'),
    'helix': ('helix', 'helix'),
    'simplex': ('simplex', 'simplex'),
    'fractal': ('fractal', 'fractal'),
    'standard': ('standard', 'standard'),
    'golden': ('golden', 'golden'),
    'fibonacci': ('fibonacci', 'fibonacci'),
    'sinusoidal': ('sinusoidal', 'sinusoidal'),
    'spherical': ('standard', 'spherical'),
}


def get_tower_config(geometry: str, name: Optional[str] = None, inverted: bool = False) -> TowerConfig:
    """Get a tower config by geometry name."""
    if geometry not in GEOMETRY_CONFIGS:
        raise ValueError(f"Unknown geometry: {geometry}. Available: {list(GEOMETRY_CONFIGS.keys())}")
    rope, addr = GEOMETRY_CONFIGS[geometry]
    return TowerConfig(name=name or geometry, rope=rope, address=addr, inverted=inverted)


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
# BUILDER FUNCTIONS
# =============================================================================

def build_rope(name: str, rope_type: RoPEType, head_dim: int, **kwargs) -> nn.Module:
    """Build RoPE component by type."""
    if rope_type == RoPEType.CANTOR:
        return CantorRoPE(f'{name}_rope', head_dim=head_dim,
                         theta_primary=kwargs.get('theta_primary', 10000.0),
                         cantor_depth=kwargs.get('cantor_depth', 5))
    elif rope_type == RoPEType.BEATRIX:
        return BeatrixRoPE(f'{name}_rope', head_dim=head_dim,
                          theta_base=kwargs.get('theta_base', 10000.0),
                          staircase_resolution=kwargs.get('staircase_resolution', 64))
    elif rope_type in (RoPEType.QUAD, RoPEType.SIMPLEX):
        return QuadRoPE(f'{name}_rope', head_dim=head_dim,
                        theta_w=kwargs.get('theta_w', 10000.0),
                        theta_x=kwargs.get('theta_x', 5000.0),
                        theta_y=kwargs.get('theta_y', 2500.0),
                        theta_z=kwargs.get('theta_z', 1000.0),
                        augmentation=kwargs.get('augmentation', 'simplex'))
    elif rope_type in (RoPEType.TRI, RoPEType.HELIX):
        return TriRoPE(f'{name}_rope', head_dim=head_dim,
                       theta_alpha=kwargs.get('theta_alpha', 10000.0),
                       theta_beta=kwargs.get('theta_beta', 5000.0),
                       theta_gamma=kwargs.get('theta_gamma', 2500.0),
                       augmentation=kwargs.get('augmentation', 'barycentric'))
    elif rope_type in (RoPEType.FRACTAL, RoPEType.DUAL):
        return DualRoPE(f'{name}_rope', head_dim=head_dim,
                        theta_primary=kwargs.get('theta_primary', 10000.0),
                        theta_secondary=kwargs.get('theta_secondary', 3000.0),
                        augmentation=kwargs.get('augmentation', 'lerp'))
    elif rope_type == RoPEType.SINUSOIDAL:
        return SinusoidalRoPE(f'{name}_rope', head_dim=head_dim,
                              base_freq=kwargs.get('base_freq', 1.0),
                              num_harmonics=kwargs.get('num_harmonics', None),
                              learnable_phase=kwargs.get('learnable_phase', True),
                              learnable_amplitude=kwargs.get('learnable_amplitude', True))
    elif rope_type == RoPEType.GOLDEN:
        PHI = 1.618033988749895
        theta_primary = kwargs.get('theta_primary', 10000.0)
        return DualRoPE(f'{name}_rope', head_dim=head_dim,
                        theta_primary=theta_primary,
                        theta_secondary=theta_primary / PHI,
                        augmentation=kwargs.get('augmentation', 'lerp'))
    elif rope_type == RoPEType.FIBONACCI:
        PHI = 1.618033988749895
        theta_base = kwargs.get('theta_base', 10000.0)
        return TriRoPE(f'{name}_rope', head_dim=head_dim,
                       theta_alpha=theta_base,
                       theta_beta=theta_base / PHI,
                       theta_gamma=theta_base / (PHI * PHI),
                       augmentation=kwargs.get('augmentation', 'barycentric'))
    else:
        return RoPE(f'{name}_rope', head_dim=head_dim,
                    theta=kwargs.get('theta', 10000.0),
                    theta_scale=kwargs.get('theta_scale', 1.0))


def build_address(name: str, address_type: AddressType, fingerprint_dim: int, **kwargs) -> nn.Module:
    """Build Address component by type."""
    if address_type == AddressType.CANTOR:
        return CantorAddressComponent(f'{name}_addr', k_simplex=kwargs.get('k_simplex', 4),
                                      fingerprint_dim=fingerprint_dim,
                                      mode=kwargs.get('mode', 'staircase'),
                                      tau=kwargs.get('tau', 0.25))
    elif address_type == AddressType.SIMPLEX:
        return SimplexAddressComponent(f'{name}_addr', k=kwargs.get('k', 4),
                                       embed_dim=fingerprint_dim,
                                       method=kwargs.get('method', 'regular'),
                                       learnable=kwargs.get('learnable', True))
    elif address_type in (AddressType.SPHERICAL, AddressType.HELIX):
        return SphericalAddressComponent(f'{name}_addr', fingerprint_dim=fingerprint_dim)
    elif address_type == AddressType.FRACTAL:
        if fingerprint_dim % 2 != 0:
            raise ValueError(f"Fractal address requires even fingerprint_dim (got {fingerprint_dim})")
        return FractalAddressComponent(f'{name}_addr', region=kwargs.get('region', 'seahorse'),
                                       orbit_length=fingerprint_dim // 2,
                                       learnable=kwargs.get('learnable', True))
    elif address_type == AddressType.GOLDEN:
        return SimplexAddressComponent(f'{name}_addr', k=kwargs.get('k', 5),
                                       embed_dim=fingerprint_dim, method='regular',
                                       learnable=kwargs.get('learnable', True))
    elif address_type == AddressType.FIBONACCI:
        return SphericalAddressComponent(f'{name}_addr', fingerprint_dim=fingerprint_dim)
    else:
        return AddressComponent(f'{name}_addr', fingerprint_dim=fingerprint_dim,
                                init_scale=kwargs.get('init_scale', 0.02))


def build_fusion(
    name: str,
    fusion_type: Union[FusionType, str],
    num_inputs: int,
    in_features: int,
    num_steps: int = 8,
    inception_aux_type: str = 'geometric',
    walker_preset: str = 'shiva',
    **kwargs
) -> nn.Module:
    """Build Fusion component by type."""
    if isinstance(fusion_type, str):
        fusion_type = FusionType(fusion_type)

    # Walker fusion types
    if fusion_type in WALKER_PRESET_MAP:
        preset = WALKER_PRESET_MAP.get(fusion_type, walker_preset)
        inception = None
        if fusion_type == FusionType.WALKER_INCEPTION:
            inception = WalkerInception(f'{name}_inception', in_features=in_features,
                                        num_steps=num_steps, num_inputs=2,
                                        aux_type=inception_aux_type)
        return WalkerFusion(name, in_features=in_features, preset=preset,
                           num_steps=num_steps, inception=inception)

    # Legacy fusion types
    if fusion_type == FusionType.ADAPTIVE:
        return AdaptiveFusion(name, num_inputs=num_inputs, in_features=in_features,
                              hidden_features=kwargs.get('hidden_features'),
                              temperature=kwargs.get('temperature', 1.0))
    elif fusion_type == FusionType.GATED:
        return GatedFusion(name, num_inputs=num_inputs, in_features=in_features)
    elif fusion_type == FusionType.ATTENTION:
        return AttentionFusion(name, num_inputs=num_inputs, in_features=in_features,
                               num_heads=kwargs.get('num_heads', 4))
    elif fusion_type == FusionType.CONCAT:
        return ConcatFusion(name, num_inputs=num_inputs, in_features=in_features,
                            out_features=in_features)
    elif fusion_type == FusionType.SUM:
        return SumFusion(name, num_inputs=num_inputs, in_features=in_features,
                         normalize=kwargs.get('normalize', True))
    else:
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
            'name': config.name, 'rope': config.rope.value,
            'address': config.address.value, 'inverted': config.inverted,
            'dim': dim, 'depth': depth,
        }

        tf_config = TransformerConfig(
            dim=dim, num_heads=num_heads, ffn_mult=ffn_mult,
            variant=TransformerVariant.PRENORM, activation=ActivationType.GELU,
            dropout=dropout, depth=depth,
        )

        rope = build_rope(config.name, config.rope, head_dim, **config.rope_params)
        self.attach('rope', rope)

        address = build_address(config.name, config.address, fingerprint_dim, **config.address_params)
        if config.inverted:
            address = InvertedAddressWrapper(address)
        self.attach('address', address)

        for i in range(depth):
            block = PreNormBlock(name=f'{config.name}_block_{i}', config=tf_config, block_idx=i)
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
        self.cache_set('features', features)
        return opinion, features

    @property
    def cached_features(self) -> Optional[Tensor]:
        return self.cache_get('features')


# =============================================================================
# OPINION CONTAINERS
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
# WALKER PAIR - Two towers with WalkerFusion
# =============================================================================

class WalkerPair(nn.Module):
    """Two towers fused with WalkerFusion. Simplest collective unit."""

    def __init__(
        self,
        name: str,
        config_a: TowerConfig,
        config_b: TowerConfig,
        dim: int = 256,
        depth: int = 2,
        num_heads: int = 4,
        fingerprint_dim: int = 64,
        use_inception: bool = False,
        walker_preset: str = 'shiva',
        inception_aux_type: str = 'geometric',
    ):
        super().__init__()
        self.name = name
        head_dim = dim // num_heads

        self.tower_a = ConfigurableTower(config=config_a, default_dim=dim, default_depth=depth,
                                         default_num_heads=num_heads, default_head_dim=head_dim,
                                         default_fingerprint_dim=fingerprint_dim)
        self.tower_b = ConfigurableTower(config=config_b, default_dim=dim, default_depth=depth,
                                         default_num_heads=num_heads, default_head_dim=head_dim,
                                         default_fingerprint_dim=fingerprint_dim)

        inception = None
        if use_inception:
            inception = WalkerInception(f'{name}_inception', in_features=dim,
                                        num_steps=8, num_inputs=2, aux_type=inception_aux_type)

        self.fusion = WalkerFusion(f'{name}_fusion', in_features=dim,
                                   preset=walker_preset, num_steps=8, inception=inception)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        opinion_a, _ = self.tower_a(x, mask)
        opinion_b, _ = self.tower_b(x, mask)
        return self.fusion(opinion_a, opinion_b)

    @property
    def fusion_params(self) -> int:
        return sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)


# =============================================================================
# CONFIGURABLE COLLECTIVE (WideRouter + WalkerFusion)
# =============================================================================

class ConfigurableCollective(SafeAttachMixin, WideRouter):
    """Tower collective with WideRouter optimization and WalkerFusion support."""

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
        fusion_type: Union[FusionType, str] = FusionType.WALKER_STATIC,
        fusion_params: Dict[str, Any] = None,
    ):
        super().__init__(name, strict=False, auto_discover=False)

        self.dim = dim
        self._fingerprint_dim = fingerprint_dim
        self._num_towers = len(tower_configs)

        if isinstance(fusion_type, str):
            fusion_type = FusionType(fusion_type)

        self.objects['tower_configs'] = tower_configs
        self.objects['collective_config'] = {
            'dim': dim, 'num_towers': self._num_towers, 'fusion_type': fusion_type.value,
        }

        for config in tower_configs:
            tower = ConfigurableTower(config=config, default_dim=dim, default_depth=default_depth,
                                      default_num_heads=num_heads, default_head_dim=head_dim,
                                      default_ffn_mult=ffn_mult, default_dropout=dropout,
                                      default_fingerprint_dim=fingerprint_dim)
            self.attach(config.name, tower)
            self.register_tower(config.name)

        fusion_params = fusion_params or {}
        fusion = build_fusion(f'{name}_fusion', fusion_type, num_inputs=self._num_towers,
                              in_features=dim, **fusion_params)
        self.attach('fusion', fusion)
        self.attach('input_proj', nn.Linear(dim, dim))
        self.attach('fp_proj', nn.Linear(fingerprint_dim * self._num_towers, fingerprint_dim))

        self.objects['debug'] = False
        self.objects['debug_info'] = {}

    @property
    def towers(self) -> Dict[str, ConfigurableTower]:
        return {name: self[name] for name in self.tower_names}

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> CollectiveOpinion:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self['input_proj'](x)
        opinions_dict = self.wide_forward(x, mask=mask)

        all_opinions = []
        opinion_tensors = []
        for tower_name in self.tower_names:
            tower = self[tower_name]
            opinion = opinions_dict[tower_name]
            features = tower.cached_features
            tower_opinion = TowerOpinion(name=tower_name, opinion=opinion, features=features,
                                         fingerprint=tower.fingerprint, config=tower.config)
            all_opinions.append(tower_opinion)
            opinion_tensors.append(opinion)

        fused = self['fusion'](*opinion_tensors)
        fp_stack = torch.cat([op.fingerprint for op in all_opinions], dim=-1)
        collective_fp = F.normalize(self['fp_proj'](fp_stack), dim=-1)

        for tower_name in self.tower_names:
            self[tower_name].cache_clear()

        return CollectiveOpinion(fused=fused, opinions={op.name: op for op in all_opinions},
                                 weights=None, collective_fingerprint=collective_fp)

    def tower_fingerprints(self) -> Dict[str, Tensor]:
        return {name: self[name].fingerprint for name in self.tower_names}


# =============================================================================
# QUICK CONSTRUCTION
# =============================================================================

def quick_tower(
    geometry: str,
    dim: int = 256,
    depth: int = 2,
    name: Optional[str] = None,
    inverted: bool = False,
    **kwargs
) -> ConfigurableTower:
    """
    Quick single tower by geometry name.

    Example:
        tower = quick_tower('cantor', dim=512, depth=3)
    """
    config = get_tower_config(geometry, name=name, inverted=inverted)
    return ConfigurableTower(
        config=config, default_dim=dim, default_depth=depth,
        default_num_heads=kwargs.get('num_heads', 4),
        default_head_dim=kwargs.get('head_dim', dim // 4),
        default_ffn_mult=kwargs.get('ffn_mult', 4.0),
        default_dropout=kwargs.get('dropout', 0.1),
        default_fingerprint_dim=kwargs.get('fingerprint_dim', 64),
    )


def quick_pair(
    geometry_a: str,
    geometry_b: str,
    dim: int = 256,
    depth: int = 2,
    use_inception: bool = False,
    walker_preset: str = 'shiva',
    inception_aux_type: str = 'geometric',
    name: Optional[str] = None,
    **kwargs
) -> WalkerPair:
    """
    Quick tower pair with WalkerFusion.

    Example:
        pair = quick_pair('cantor', 'beatrix', use_inception=True)
        fused = pair(x)  # [B, D]
    """
    name = name or f'{geometry_a}_{geometry_b}'
    config_a = get_tower_config(geometry_a)
    config_b = get_tower_config(geometry_b)
    return WalkerPair(
        name=name, config_a=config_a, config_b=config_b,
        dim=dim, depth=depth, num_heads=kwargs.get('num_heads', 4),
        fingerprint_dim=kwargs.get('fingerprint_dim', 64),
        use_inception=use_inception, walker_preset=walker_preset,
        inception_aux_type=inception_aux_type,
    )


def quick_collective(
    geometries: List[str],
    dim: int = 256,
    depth: int = 2,
    fusion_type: Union[FusionType, str] = FusionType.WALKER_STATIC,
    name: str = 'collective',
    **kwargs
) -> ConfigurableCollective:
    """
    Quick tower collective.

    Example:
        collective = quick_collective(
            ['cantor', 'beatrix', 'helix', 'simplex'],
            fusion_type='walker_inception'
        )
    """
    configs = [get_tower_config(g) for g in geometries]
    return ConfigurableCollective(
        name=name, tower_configs=configs, dim=dim, default_depth=depth,
        num_heads=kwargs.get('num_heads', 4),
        head_dim=kwargs.get('head_dim', dim // 4),
        ffn_mult=kwargs.get('ffn_mult', 4.0),
        dropout=kwargs.get('dropout', 0.1),
        fingerprint_dim=kwargs.get('fingerprint_dim', 64),
        fusion_type=fusion_type,
        fusion_params=kwargs.get('fusion_params', {}),
    )


# =============================================================================
# PRESETS
# =============================================================================

def preset_pos_neg_pairs(geometries: List[str] = None) -> List[TowerConfig]:
    """Create pos/neg pairs for each geometry."""
    geometries = geometries or ['cantor', 'beatrix', 'helix', 'simplex', 'sinusoidal', 'golden', 'fibonacci']
    configs = []
    for geom in geometries:
        configs.append(get_tower_config(geom, f'{geom}_pos', inverted=False))
        configs.append(get_tower_config(geom, f'{geom}_neg', inverted=True))
    return configs


def preset_all_six() -> List[TowerConfig]:
    return [get_tower_config(g) for g in ['cantor', 'beatrix', 'helix', 'simplex', 'fractal', 'standard']]


def preset_all_eight() -> List[TowerConfig]:
    return [get_tower_config(g) for g in [
        'cantor', 'beatrix', 'helix', 'simplex', 'fractal', 'standard', 'golden', 'fibonacci'
    ]]


def preset_dual_stack_full() -> List[TowerConfig]:
    return preset_pos_neg_pairs([
        'cantor', 'beatrix', 'helix', 'simplex', 'golden', 'fibonacci', 'sinusoidal', 'fractal'
    ])


# Legacy
def build_tower_collective(
    configs: List[TowerConfig],
    dim: int = 256,
    default_depth: int = 2,
    num_heads: int = 4,
    ffn_mult: float = 4.0,
    dropout: float = 0.1,
    fingerprint_dim: int = 64,
    fusion_type: str = 'walker_static',
    name: str = 'tower_collective',
) -> ConfigurableCollective:
    return ConfigurableCollective(
        name=name, tower_configs=configs, dim=dim, default_depth=default_depth,
        num_heads=num_heads, head_dim=dim // num_heads, ffn_mult=ffn_mult,
        dropout=dropout, fingerprint_dim=fingerprint_dim, fusion_type=fusion_type,
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Geometric Tower Builder - Quick Construction Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, L, D = 4, 16, 256
    x = torch.randn(B, L, D, device=device)

    # Single tower
    tower = quick_tower('cantor', dim=D, depth=2).to(device)
    opinion, features = tower(x)
    print(f"quick_tower: opinion {opinion.shape}, features {features.shape}")

    # Pair (static)
    pair = quick_pair('cantor', 'beatrix', dim=D, depth=1).to(device)
    fused = pair(x)
    print(f"quick_pair (static): {fused.shape}, fusion_params={pair.fusion_params}")

    # Pair (inception)
    pair_inc = quick_pair('cantor', 'beatrix', dim=D, depth=1, use_inception=True).to(device)
    fused = pair_inc(x)
    print(f"quick_pair (inception): {fused.shape}, fusion_params={pair_inc.fusion_params}")

    # Collective
    collective = quick_collective(['cantor', 'beatrix', 'helix', 'simplex'],
                                   dim=D, depth=1, fusion_type='walker_shiva').to(device)
    collective.discover_towers()
    result = collective(x)
    print(f"quick_collective: fused {result.fused.shape}")

    print("\n✓ All quick construction tests passed")