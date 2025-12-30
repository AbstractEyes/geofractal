"""
geofractal.router.prefab.geometric_tower_builder
================================================

Flexible tower builder with BATCHED forward for tower groups.
Now with WalkerFusion support for static and learnable fusion.

v1.1.1: RoPE injection fix - rope now passed to attention layers.

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
from geofractal.router.components.torch_component import TorchComponent
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
        # Negate the base fingerprint (negation survives normalization)
        return -self.base_address.fingerprint

    def forward(self, *args, **kwargs):
        return self.base_address(*args, **kwargs)


# =============================================================================
# BUILDER FUNCTIONS
# =============================================================================

def build_rope(name: str, rope_type: RoPEType, head_dim: int, **kwargs) -> nn.Module:
    """Build RoPE component by type."""
    if rope_type == RoPEType.CANTOR:
        return CantorRoPE(f'{name}_rope', head_dim=head_dim,
                         theta=kwargs.get('theta', 10000.0),
                         levels=kwargs.get('levels', 5),
                         tau=kwargs.get('tau', 0.25),
                         mode=kwargs.get('mode', 'hybrid'),
                         blend_alpha=kwargs.get('blend_alpha', 0.5))
    elif rope_type == RoPEType.BEATRIX:
        return BeatrixRoPE(f'{name}_rope', head_dim=head_dim,
                          theta=kwargs.get('theta', 10000.0),
                          levels=kwargs.get('levels', 5),
                          alpha=kwargs.get('alpha', 0.5),
                          tau=kwargs.get('tau', 0.25))
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
    fingerprint_dim: Optional[int] = None,
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
                           num_steps=num_steps, inception=inception,
                           fingerprint_dim=fingerprint_dim)

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
        # Get rope component for position encoding (v1.1.1 fix)
        rope = self['rope']

        for block in self.stages:
            x, _ = block(x, mask=mask, rope=rope)
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

# =============================================================================
# HETEROGENEOUS FUSION (for multi-encoder with different dims)
# =============================================================================

class HeterogeneousFusion(TorchComponent):
    """
    WalkerFusion wrapper that projects heterogeneous inputs to common dimension.

    Use when fusing encoder outputs with different dimensions:
        fusion = HeterogeneousFusion('meta', in_dims=[1536, 1280, 768], out_dim=512)
        fused = fusion(conv_out, bigg_out, maxvit_out)  # All projected to 512, then fused
    """

    def __init__(
        self,
        name: str,
        in_dims: List[int],
        out_dim: int,
        use_inception: bool = False,
        walker_preset: str = 'shiva',
        inception_aux_type: str = 'geometric',
        num_steps: int = 8,
    ):
        super().__init__(name)
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.num_inputs = len(in_dims)

        # Input projections
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            ) for in_dim in in_dims
        ])

        # Walker fusion
        inception = None
        if use_inception:
            inception = WalkerInception(
                f'{name}_inception', in_features=out_dim,
                num_steps=num_steps, num_inputs=2, aux_type=inception_aux_type
            )

        self.fusion = WalkerFusion(
            f'{name}_fusion', in_features=out_dim,
            preset=walker_preset, num_steps=num_steps, inception=inception
        )

    def forward(self, *inputs: Tensor) -> Tensor:
        """
        Args:
            *inputs: Variable number of tensors, one per input dimension
                     Each should be [B, in_dims[i]] or [B, L, in_dims[i]]
        Returns:
            Fused output [B, out_dim] or [B, L, out_dim]
        """
        assert len(inputs) == self.num_inputs, \
            f"Expected {self.num_inputs} inputs, got {len(inputs)}"

        # Project all inputs to common dimension
        projected = [proj(x) for proj, x in zip(self.projections, inputs)]

        # Fuse pairwise (WalkerFusion handles N inputs hierarchically)
        return self.fusion(*projected)

    @property
    def fusion_params(self) -> int:
        """Learnable params in fusion (excluding projections)."""
        return sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        """All learnable params including projections."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HeterogeneousPair(TorchComponent):
    """
    Fuse two inputs with different dimensions.

    Example:
        pair = HeterogeneousPair('bigg_conv', in_dim_a=1280, in_dim_b=1536, out_dim=512)
        fused = pair(bigg_features, conv_features)
    """

    def __init__(
        self,
        name: str,
        in_dim_a: int,
        in_dim_b: int,
        out_dim: int,
        use_inception: bool = False,
        walker_preset: str = 'shiva',
        inception_aux_type: str = 'geometric',
        num_steps: int = 8,
    ):
        super().__init__(name)

        # Input projections
        self.proj_a = nn.Sequential(
            nn.LayerNorm(in_dim_a),
            nn.Linear(in_dim_a, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.proj_b = nn.Sequential(
            nn.LayerNorm(in_dim_b),
            nn.Linear(in_dim_b, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

        # Walker fusion
        inception = None
        if use_inception:
            inception = WalkerInception(
                f'{name}_inception', in_features=out_dim,
                num_steps=num_steps, num_inputs=2, aux_type=inception_aux_type
            )

        self.fusion = WalkerFusion(
            f'{name}_fusion', in_features=out_dim,
            preset=walker_preset, num_steps=num_steps, inception=inception
        )

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Fuse two inputs with different dimensions."""
        pa = self.proj_a(a)
        pb = self.proj_b(b)
        return self.fusion(pa, pb)

    @property
    def fusion_params(self) -> int:
        return sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)


class WalkerPair(TorchComponent):
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
        super().__init__(name)
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
                              in_features=dim, fingerprint_dim=fingerprint_dim, **fusion_params)
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
        B = x.shape[0]
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

        # Compute collective fingerprint BEFORE fusion (v1.1.1)
        # Fingerprints are per-tower [fp_dim], expand to batch [B, fp_dim * num_towers]
        fp_stack = torch.cat([op.fingerprint for op in all_opinions], dim=-1)
        fp_stack = fp_stack.unsqueeze(0).expand(B, -1)
        collective_fp = F.normalize(self['fp_proj'](fp_stack), dim=-1)

        # Pass fingerprint to fusion for modulation (WalkerFusion only)
        fusion = self['fusion']
        if hasattr(fusion, 'fingerprint_gate'):
            fused = fusion(*opinion_tensors, fingerprint=collective_fp)
        else:
            fused = fusion(*opinion_tensors)

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
    num_heads = kwargs.get('num_heads', 4)
    head_dim = kwargs.get('head_dim', dim // num_heads)

    config = get_tower_config(geometry, name=name, inverted=inverted)
    return ConfigurableTower(
        config=config, default_dim=dim, default_depth=depth,
        default_num_heads=num_heads,
        default_head_dim=head_dim,
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

    Supports:
    - 'simplex' → single simplex tower
    - ['cantor', 3] → 3 inverse pairs (6 towers: 3 pos, 3 neg)

    Example:
        collective = quick_collective(
            ['simplex', ['cantor', 2], 'helix'],
            fusion_type='walker_inception'
        )
        # Creates: simplex, cantor_0_pos, cantor_0_neg, cantor_1_pos, cantor_1_neg, helix
        # = 6 towers total
    """
    # Generate configs, expanding inverse pairs
    name_counts = {}
    configs = []

    for item in geometries:
        # Check for inverse pairs: ['geometry', count]
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], int):
            g, num_pairs = item
            for i in range(num_pairs):
                count = name_counts.get(g, 0)
                pos_name = f'{g}_{count}_pos'
                neg_name = f'{g}_{count}_neg'
                name_counts[g] = count + 1

                configs.append(get_tower_config(g, name=pos_name, inverted=False))
                configs.append(get_tower_config(g, name=neg_name, inverted=True))
        else:
            # Single tower
            g = item
            count = name_counts.get(g, 0)
            tower_name = f'{g}_{count}' if count > 0 or sum(1 for x in geometries if (x == g if isinstance(x, str) else x[0] == g)) > 1 else g
            name_counts[g] = count + 1
            configs.append(get_tower_config(g, name=tower_name))

    num_heads = kwargs.get('num_heads', 4)
    head_dim = kwargs.get('head_dim', dim // num_heads)

    return ConfigurableCollective(
        name=name, tower_configs=configs, dim=dim, default_depth=depth,
        num_heads=num_heads,
        head_dim=head_dim,
        ffn_mult=kwargs.get('ffn_mult', 4.0),
        dropout=kwargs.get('dropout', 0.1),
        fingerprint_dim=kwargs.get('fingerprint_dim', 64),
        fusion_type=fusion_type,
    )


class InversePairModule(TorchComponent):
    """
    Wraps two towers (pos/neg) with AdaptiveGate fusion.

    Acts as a single "expert" with internal inverse diversity.
    The gate learns to blend the positive and negative perspectives.
    """

    def __init__(
        self,
        name: str,
        geometry: str,
        dim: int = 256,
        depth: int = 2,
        num_heads: int = 4,
        fingerprint_dim: int = 64,
        **kwargs
    ):
        super().__init__(name)
        self.dim = dim
        self._fingerprint_dim = fingerprint_dim
        head_dim = dim // num_heads

        # Create positive and negative towers
        config_pos = get_tower_config(geometry, name=f'{name}_pos', inverted=False)
        config_neg = get_tower_config(geometry, name=f'{name}_neg', inverted=True)

        tower_pos = ConfigurableTower(
            config=config_pos, default_dim=dim, default_depth=depth,
            default_num_heads=num_heads, default_head_dim=head_dim,
            default_fingerprint_dim=fingerprint_dim,
            **{k: v for k, v in kwargs.items() if k.startswith('default_')}
        )
        tower_neg = ConfigurableTower(
            config=config_neg, default_dim=dim, default_depth=depth,
            default_num_heads=num_heads, default_head_dim=head_dim,
            default_fingerprint_dim=fingerprint_dim,
            **{k: v for k, v in kwargs.items() if k.startswith('default_')}
        )
        self.attach('tower_pos', tower_pos)
        self.attach('tower_neg', tower_neg)

        # AdaptiveGate for blending
        self.attach('gate', nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        ))

        # Combined fingerprint projection
        self.attach('fp_proj', nn.Linear(fingerprint_dim * 2, fingerprint_dim))

    @property
    def fingerprint(self) -> Tensor:
        """Combined fingerprint from both towers."""
        fp_pos = self['tower_pos'].fingerprint
        fp_neg = self['tower_neg'].fingerprint
        combined = torch.cat([fp_pos, fp_neg], dim=-1)
        return F.normalize(self['fp_proj'](combined), dim=-1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward through both towers, gate-fused output."""
        opinion_pos, _ = self['tower_pos'](x, mask)
        opinion_neg, _ = self['tower_neg'](x, mask)

        # Adaptive gating
        combined = torch.cat([opinion_pos, opinion_neg], dim=-1)
        gate = self['gate'](combined)

        # Blend: gate * pos + (1-gate) * neg
        return gate * opinion_pos + (1 - gate) * opinion_neg


class HybridCollective(TorchComponent):
    """
    Collective supporting both regular towers and inverse pair modules.

    Usage:
        collective = quick_hybrid_collective(
            ['simplex', ['cantor', 'inverse_pair']],
            fusion_type='walker_inception'
        )
    """

    def __init__(
        self,
        name: str,
        units: List[nn.Module],  # Mix of ConfigurableTower and InversePairModule
        unit_names: List[str],
        dim: int = 256,
        fingerprint_dim: int = 64,
        fusion_type: Union[FusionType, str] = FusionType.WALKER_STATIC,
        fusion_params: Dict[str, Any] = None,
    ):
        super().__init__(name)
        self.dim = dim
        self._fingerprint_dim = fingerprint_dim
        self._num_units = len(units)
        self.unit_names = unit_names

        # Attach all units
        for unit_name, unit in zip(unit_names, units):
            self.attach(unit_name, unit)

        # Build fusion
        fusion_params = fusion_params or {}
        fusion = build_fusion(
            f'{name}_fusion', fusion_type, num_inputs=self._num_units,
            in_features=dim, fingerprint_dim=fingerprint_dim, **fusion_params
        )
        self.attach('fusion', fusion)

        # Fingerprint projection
        self.attach('fp_proj', nn.Linear(fingerprint_dim * self._num_units, fingerprint_dim))

    @property
    def units(self) -> Dict[str, nn.Module]:
        return {name: self[name] for name in self.unit_names}

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> CollectiveOpinion:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B = x.shape[0]

        opinions = []
        fingerprints = []

        for unit_name in self.unit_names:
            unit = self[unit_name]

            # Get opinion (both tower types return tensor from forward)
            if isinstance(unit, ConfigurableTower):
                opinion, _ = unit(x, mask)
            else:
                opinion = unit(x, mask)

            opinions.append(opinion)
            fingerprints.append(unit.fingerprint)

        # Build collective fingerprint
        fp_stack = torch.cat(fingerprints, dim=-1)
        fp_stack = fp_stack.unsqueeze(0).expand(B, -1)
        collective_fp = F.normalize(self['fp_proj'](fp_stack), dim=-1)

        # Fuse opinions
        fusion = self['fusion']
        if hasattr(fusion, 'fingerprint_gate'):
            fused = fusion(*opinions, fingerprint=collective_fp)
        else:
            fused = fusion(*opinions)

        return CollectiveOpinion(
            fused=fused,
            opinions={name: TowerOpinion(name=name, opinion=op, features=None,
                                         fingerprint=fp, config=None)
                      for name, op, fp in zip(self.unit_names, opinions, fingerprints)},
            weights=None,
            collective_fingerprint=collective_fp,
        )


def quick_hybrid_collective(
    geometries: List[Union[str, List]],
    dim: int = 256,
    depth: int = 2,
    fusion_type: Union[FusionType, str] = FusionType.WALKER_STATIC,
    name: str = 'hybrid_collective',
    **kwargs
) -> HybridCollective:
    """
    Quick hybrid collective supporting inverse pairs.

    Syntax:
        - 'simplex' → single simplex tower
        - ['cantor', 'inverse_pair'] → InversePairModule with cantor+/cantor-

    Example:
        collective = quick_hybrid_collective(
            ['simplex', ['cantor', 'inverse_pair'], 'helix'],
            fusion_type='walker_inception'
        )

        # Creates:
        # - simplex tower
        # - cantor inverse pair (pos + neg with AdaptiveGate)
        # - helix tower
        # All fused with WalkerInception
    """
    fingerprint_dim = kwargs.get('fingerprint_dim', 64)
    num_heads = kwargs.get('num_heads', 4)
    head_dim = dim // num_heads

    units = []
    unit_names = []
    name_counts = {}

    for item in geometries:
        # Check for inverse pair: ['geometry', 'inverse_pair']
        if isinstance(item, (list, tuple)) and len(item) == 2 and item[1] == 'inverse_pair':
            g = item[0]
            count = name_counts.get(g, 0)
            pair_name = f'{g}_pair_{count}' if count > 0 else f'{g}_pair'
            name_counts[g] = count + 1

            pair = InversePairModule(
                name=pair_name,
                geometry=g,
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                fingerprint_dim=fingerprint_dim,
                default_ffn_mult=kwargs.get('ffn_mult', 4.0),
                default_dropout=kwargs.get('dropout', 0.1),
            )
            units.append(pair)
            unit_names.append(pair_name)

        else:
            # Regular tower
            g = item
            count = name_counts.get(g, 0)
            tower_name = f'{g}_{count}' if count > 0 else g
            name_counts[g] = count + 1

            config = get_tower_config(g, name=tower_name)
            tower = ConfigurableTower(
                config=config,
                default_dim=dim,
                default_depth=depth,
                default_num_heads=num_heads,
                default_head_dim=head_dim,
                default_fingerprint_dim=fingerprint_dim,
                default_ffn_mult=kwargs.get('ffn_mult', 4.0),
                default_dropout=kwargs.get('dropout', 0.1),
            )
            units.append(tower)
            unit_names.append(tower_name)

    return HybridCollective(
        name=name,
        units=units,
        unit_names=unit_names,
        dim=dim,
        fingerprint_dim=fingerprint_dim,
        fusion_type=fusion_type,
        fusion_params=kwargs.get('fusion_params'),
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
    import time

    print("=" * 70)
    print("Geometric Tower Builder - Comprehensive Forward Test")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    B, L, D = 4, 16, 256
    x = torch.randn(B, L, D, device=device)

    # =========================================================================
    # 1. Test all RoPE types
    # =========================================================================
    print("\n" + "-" * 70)
    print("1. RoPE Types")
    print("-" * 70)

    rope_types = list(RoPEType)
    for rope_type in rope_types:
        try:
            config = TowerConfig(f'test_{rope_type.value}', rope=rope_type, address='standard')
            tower = ConfigurableTower(config, default_dim=D, default_depth=1).to(device)
            opinion, features = tower(x)
            params = sum(p.numel() for p in tower.parameters())
            print(f"  {rope_type.value:12s} ✓  opinion {opinion.shape}, params {params:,}")
        except Exception as e:
            print(f"  {rope_type.value:12s} ✗  {e}")

    # =========================================================================
    # 2. Test all Address types
    # =========================================================================
    print("\n" + "-" * 70)
    print("2. Address Types")
    print("-" * 70)

    address_types = list(AddressType)
    for addr_type in address_types:
        try:
            config = TowerConfig(f'test_{addr_type.value}', rope='standard', address=addr_type)
            tower = ConfigurableTower(config, default_dim=D, default_depth=1).to(device)
            opinion, features = tower(x)
            fp = tower.fingerprint
            print(f"  {addr_type.value:12s} ✓  fingerprint {fp.shape}")
        except Exception as e:
            print(f"  {addr_type.value:12s} ✗  {e}")

    # =========================================================================
    # 3. Test all geometry presets
    # =========================================================================
    print("\n" + "-" * 70)
    print("3. Geometry Presets (via quick_tower)")
    print("-" * 70)

    geometries = list(GEOMETRY_CONFIGS.keys())
    for geom in geometries:
        try:
            tower = quick_tower(geom, dim=D, depth=1).to(device)
            opinion, features = tower(x)
            params = sum(p.numel() for p in tower.parameters())
            print(f"  {geom:12s} ✓  params {params:,}")
        except Exception as e:
            print(f"  {geom:12s} ✗  {e}")

    # =========================================================================
    # 4. Test all fusion types
    # =========================================================================
    print("\n" + "-" * 70)
    print("4. Fusion Types (via quick_collective)")
    print("-" * 70)

    fusion_types = list(FusionType)
    for fusion_type in fusion_types:
        try:
            collective = quick_collective(
                ['cantor', 'beatrix'], dim=D, depth=1, fusion_type=fusion_type.value
            ).to(device)
            collective.discover_towers()
            result = collective(x)
            fusion_params = sum(p.numel() for p in collective['fusion'].parameters())
            print(f"  {fusion_type.value:18s} ✓  fusion_params {fusion_params:,}")
        except Exception as e:
            print(f"  {fusion_type.value:18s} ✗  {e}")

    # =========================================================================
    # 5. Test WalkerPair with all presets
    # =========================================================================
    print("\n" + "-" * 70)
    print("5. WalkerPair Presets")
    print("-" * 70)

    walker_presets = ['shiva', 'slerp', 'lerp', 'zeus']
    for preset in walker_presets:
        try:
            pair = quick_pair('cantor', 'helix', dim=D, depth=1,
                              walker_preset=preset, use_inception=False).to(device)
            fused = pair(x)
            print(f"  {preset:12s} (static)    ✓  output {fused.shape}")

            pair_inc = quick_pair('cantor', 'helix', dim=D, depth=1,
                                  walker_preset=preset, use_inception=True).to(device)
            fused_inc = pair_inc(x)
            print(f"  {preset:12s} (inception) ✓  output {fused_inc.shape}, params {pair_inc.fusion_params:,}")
        except Exception as e:
            print(f"  {preset:12s} ✗  {e}")

    # =========================================================================
    # 6. Test inception aux types
    # =========================================================================
    print("\n" + "-" * 70)
    print("6. Inception Aux Types")
    print("-" * 70)

    aux_types = ['geometric', 'cosine', 'learned', 'walker_path']
    for aux_type in aux_types:
        try:
            pair = quick_pair('simplex', 'fractal', dim=D, depth=1,
                              use_inception=True, inception_aux_type=aux_type).to(device)
            fused = pair(x)
            print(f"  {aux_type:12s} ✓  fusion_params {pair.fusion_params:,}")
        except Exception as e:
            print(f"  {aux_type:12s} ✗  {e}")

    # =========================================================================
    # 7. Large collective test
    # =========================================================================
    print("\n" + "-" * 70)
    print("7. Large Collective (all geometries)")
    print("-" * 70)

    try:
        all_geoms = list(GEOMETRY_CONFIGS.keys())
        collective = quick_collective(all_geoms, dim=D, depth=2, fusion_type='walker_inception').to(device)
        collective.discover_towers()

        # Warmup
        _ = collective(x)

        # Timed forward
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            result = collective(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 10 * 1000

        total_params = sum(p.numel() for p in collective.parameters())
        print(f"  Towers: {len(all_geoms)}")
        print(f"  Total params: {total_params:,}")
        print(f"  Forward: {elapsed:.2f}ms")
        print(f"  Output: fused {result.fused.shape}")
        print(f"  Opinions: {list(result.opinions.keys())}")
    except Exception as e:
        print(f"  ✗ {e}")

    # =========================================================================
    # 8. Inverted tower test
    # =========================================================================
    print("\n" + "-" * 70)
    print("8. Inverted Towers")
    print("-" * 70)

    try:
        tower_pos = quick_tower('cantor', dim=D, depth=1, inverted=False).to(device)
        tower_neg = quick_tower('cantor', dim=D, depth=1, inverted=True).to(device)

        op_pos, _ = tower_pos(x)
        op_neg, _ = tower_neg(x)

        fp_pos = tower_pos.fingerprint
        fp_neg = tower_neg.fingerprint

        # Fingerprints should be negated (fp_pos + fp_neg ≈ 0)
        fp_sum = (fp_pos + fp_neg).abs().mean().item()
        print(f"  Pos fingerprint: {fp_pos[:3].tolist()}")
        print(f"  Neg fingerprint: {fp_neg[:3].tolist()}")
        print(f"  Inversion check (should be ~0): {fp_sum:.6f}")
    except Exception as e:
        print(f"  ✗ {e}")

    # =========================================================================
    # 9. Direct TowerConfig construction
    # =========================================================================
    print("\n" + "-" * 70)
    print("9. Direct TowerConfig Construction")
    print("-" * 70)

    try:
        configs = [
            TowerConfig('custom_a', rope='cantor', address='cantor',
                        rope_params={'levels': 6, 'mode': 'hybrid'}),
            TowerConfig('custom_b', rope='quad', address='simplex',
                        rope_params={'theta_w': 8000.0}),
            TowerConfig('custom_c', rope='sinusoidal', address='spherical',
                        rope_params={'base_freq': 2.0}),
        ]
        collective = build_tower_collective(configs, dim=D, default_depth=2,
                                             fusion_type='walker_static')
        collective.to(device)
        collective.discover_towers()
        result = collective(x)
        print(f"  Custom configs: {[c.name for c in configs]}")
        print(f"  Output: {result.fused.shape}")
    except Exception as e:
        print(f"  ✗ {e}")

    # =========================================================================
    # 10. Stress test: deeper towers
    # =========================================================================
    print("\n" + "-" * 70)
    print("10. Depth Scaling Test")
    print("-" * 70)

    for depth in [1, 2, 4, 6]:
        try:
            tower = quick_tower('helix', dim=D, depth=depth).to(device)
            t0 = time.perf_counter()
            opinion, _ = tower(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            params = sum(p.numel() for p in tower.parameters())
            print(f"  depth={depth}: params {params:,}, forward {elapsed:.2f}ms")
        except Exception as e:
            print(f"  depth={depth}: ✗ {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  RoPE types tested:     {len(rope_types)}")
    print(f"  Address types tested:  {len(address_types)}")
    print(f"  Geometry presets:      {len(geometries)}")
    print(f"  Fusion types:          {len(fusion_types)}")
    print(f"  Walker presets:        {len(walker_presets)}")
    print(f"  Inception aux types:   {len(aux_types)}")
    print("\n✓ Comprehensive forward test complete")