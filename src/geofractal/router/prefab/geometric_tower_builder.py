"""
geofractal.router.prefab.tower_builder
======================================

Flexible tower builder that creates towers from config sequences.
Allows mix-and-match of RoPE types, Address types, depths, and more.

Usage:
    configs = [
        TowerConfig(name='cantor_pos', rope='cantor', address='cantor'),
        TowerConfig(name='cantor_neg', rope='cantor', address='cantor', inverted=True),
        TowerConfig(name='beatrix_deep', rope='beatrix', address='beatrix', depth=4),
        TowerConfig(name='hybrid', rope='cantor', address='simplex'),  # mix!
    ]
    collective = build_tower_collective(configs, dim=256)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Literal, Any, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.base_router import BaseRouter
from geofractal.router.base_tower import BaseTower
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
)


# =============================================================================
# TYPE ENUMS
# =============================================================================

class RoPEType(str, Enum):
    """Available RoPE variants."""
    STANDARD = "standard"
    DUAL = "dual"
    TRI = "tri"
    QUAD = "quad"
    CANTOR = "cantor"
    BEATRIX = "beatrix"
    # Aliases for geometry names
    HELIX = "helix"      # -> TriRoPE
    SIMPLEX = "simplex"  # -> QuadRoPE
    FRACTAL = "fractal"  # -> DualRoPE


class AddressType(str, Enum):
    """Available Address variants."""
    STANDARD = "standard"
    SIMPLEX = "simplex"
    SPHERICAL = "spherical"
    FRACTAL = "fractal"
    CANTOR = "cantor"
    CANTOR_STAIRCASE = "cantor_staircase"
    CANTOR_FEATURES = "cantor_features"
    # Aliases
    HELIX = "helix"      # -> Spherical
    BEATRIX = "beatrix"  # -> Cantor with staircase_features


class FusionType(str, Enum):
    """Available Fusion variants."""
    ADAPTIVE = "adaptive"
    GATED = "gated"
    ATTENTION = "attention"
    MEAN = "mean"
    CONCAT = "concat"


# =============================================================================
# TOWER CONFIG
# =============================================================================

@dataclass
class TowerConfig:
    """Configuration for a single tower."""

    name: str

    # Geometry components (can mix and match!)
    rope: Union[RoPEType, str] = RoPEType.STANDARD
    address: Union[AddressType, str] = AddressType.STANDARD

    # Fingerprint options
    inverted: bool = False  # Invert the fingerprint (1 - fp)

    # Architecture overrides (None = use collective defaults)
    dim: Optional[int] = None
    depth: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    ffn_mult: Optional[float] = None
    dropout: Optional[float] = None
    fingerprint_dim: Optional[int] = None

    # RoPE-specific params
    rope_params: Dict[str, Any] = field(default_factory=dict)

    # Address-specific params
    address_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Convert strings to enums
        if isinstance(self.rope, str):
            self.rope = RoPEType(self.rope)
        if isinstance(self.address, str):
            self.address = AddressType(self.address)


# =============================================================================
# INVERTED ADDRESS WRAPPER
# =============================================================================

class InvertedAddressWrapper(nn.Module):
    """Wraps an address component and inverts its fingerprint."""

    def __init__(self, base_address: nn.Module):
        super().__init__()
        self.base_address = base_address

    @property
    def fingerprint(self) -> Tensor:
        return 1.0 - self.base_address.fingerprint

    def forward(self, *args, **kwargs):
        return self.base_address(*args, **kwargs)


# =============================================================================
# COMPONENT FACTORIES
# =============================================================================

def build_rope(
    name: str,
    rope_type: RoPEType,
    head_dim: int,
    **kwargs
) -> nn.Module:
    """Build RoPE component from type."""

    if rope_type in (RoPEType.CANTOR,):
        levels = kwargs.get('levels', 5)
        mode = kwargs.get('mode', 'hybrid')
        return CantorRoPE(f'{name}_rope', head_dim=head_dim, levels=levels, mode=mode)

    elif rope_type in (RoPEType.BEATRIX,):
        levels = kwargs.get('levels', 5)
        return BeatrixRoPE(f'{name}_rope', head_dim=head_dim, levels=levels)

    elif rope_type in (RoPEType.HELIX, RoPEType.TRI):
        theta_alpha = kwargs.get('theta_alpha', 10000.0)
        theta_beta = kwargs.get('theta_beta', 5000.0)
        theta_gamma = kwargs.get('theta_gamma', 2500.0)
        augmentation = kwargs.get('augmentation', 'barycentric')
        return TriRoPE(
            f'{name}_rope', head_dim=head_dim,
            theta_alpha=theta_alpha, theta_beta=theta_beta, theta_gamma=theta_gamma,
            augmentation=augmentation
        )

    elif rope_type in (RoPEType.SIMPLEX, RoPEType.QUAD):
        theta_w = kwargs.get('theta_w', 10000.0)
        theta_x = kwargs.get('theta_x', 7500.0)
        theta_y = kwargs.get('theta_y', 5000.0)
        theta_z = kwargs.get('theta_z', 2500.0)
        augmentation = kwargs.get('augmentation', 'simplex')
        return QuadRoPE(
            f'{name}_rope', head_dim=head_dim,
            theta_w=theta_w, theta_x=theta_x, theta_y=theta_y, theta_z=theta_z,
            augmentation=augmentation
        )

    elif rope_type in (RoPEType.FRACTAL, RoPEType.DUAL):
        theta_primary = kwargs.get('theta_primary', 10000.0)
        theta_secondary = kwargs.get('theta_secondary', 3000.0)
        augmentation = kwargs.get('augmentation', 'lerp')
        return DualRoPE(
            f'{name}_rope', head_dim=head_dim,
            theta_primary=theta_primary, theta_secondary=theta_secondary,
            augmentation=augmentation
        )

    else:  # STANDARD
        theta = kwargs.get('theta', 10000.0)
        return RoPE(f'{name}_rope', head_dim=head_dim, theta=theta)


def build_address(
    name: str,
    address_type: AddressType,
    fingerprint_dim: int,
    **kwargs
) -> nn.Module:
    """Build Address component from type."""

    if address_type == AddressType.CANTOR:
        k_simplex = kwargs.get('k_simplex', 4)
        # Use valid RouteMode: 'staircase' for standard Cantor
        return CantorAddressComponent(
            f'{name}_address', k_simplex=k_simplex,
            fingerprint_dim=fingerprint_dim, mode='staircase'
        )

    elif address_type == AddressType.CANTOR_STAIRCASE:
        k_simplex = kwargs.get('k_simplex', 4)
        return CantorAddressComponent(
            f'{name}_address', k_simplex=k_simplex,
            fingerprint_dim=fingerprint_dim, mode='staircase'
        )

    elif address_type in (AddressType.BEATRIX, AddressType.CANTOR_FEATURES):
        k_simplex = kwargs.get('k_simplex', 4)
        return CantorAddressComponent(
            f'{name}_address', k_simplex=k_simplex,
            fingerprint_dim=fingerprint_dim, mode='staircase_features'
        )

    elif address_type in (AddressType.HELIX, AddressType.SPHERICAL):
        return SphericalAddressComponent(
            f'{name}_address', fingerprint_dim=fingerprint_dim
        )

    elif address_type == AddressType.SIMPLEX:
        k = kwargs.get('k', 4)
        return SimplexAddressComponent(
            f'{name}_address', k=k, embed_dim=fingerprint_dim
        )

    elif address_type == AddressType.FRACTAL:
        region = kwargs.get('region', 'seahorse')
        orbit_length = kwargs.get('orbit_length', fingerprint_dim // 2)
        return FractalAddressComponent(
            f'{name}_address', region=region, orbit_length=orbit_length
        )

    else:  # STANDARD
        return AddressComponent(
            f'{name}_address', fingerprint_dim=fingerprint_dim
        )


def build_fusion(
    name: str,
    fusion_type: FusionType,
    num_inputs: int,
    in_features: int,
    **kwargs
) -> nn.Module:
    """Build Fusion component from type."""

    if fusion_type == FusionType.ADAPTIVE:
        return AdaptiveFusion(name, num_inputs=num_inputs, in_features=in_features)

    elif fusion_type == FusionType.GATED:
        return GatedFusion(name, num_inputs=num_inputs, in_features=in_features)

    elif fusion_type == FusionType.ATTENTION:
        num_heads = kwargs.get('num_heads', 8)
        return AttentionFusion(
            name, num_inputs=num_inputs, in_features=in_features, num_heads=num_heads
        )

    elif fusion_type == FusionType.MEAN:
        return MeanFusion(name, num_inputs=num_inputs)

    elif fusion_type == FusionType.CONCAT:
        return ConcatFusion(name, num_inputs=num_inputs, in_features=in_features)

    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


# =============================================================================
# SIMPLE FUSION VARIANTS
# =============================================================================

class MeanFusion(nn.Module):
    """Simple mean fusion."""
    def __init__(self, name: str, num_inputs: int):
        super().__init__()
        self.name = name
        self.num_inputs = num_inputs

    def forward(self, *inputs: Tensor) -> Tensor:
        return torch.stack(inputs, dim=0).mean(dim=0)


class ConcatFusion(nn.Module):
    """Concatenate and project."""
    def __init__(self, name: str, num_inputs: int, in_features: int):
        super().__init__()
        self.name = name
        self.proj = nn.Linear(num_inputs * in_features, in_features)

    def forward(self, *inputs: Tensor) -> Tensor:
        concatenated = torch.cat(inputs, dim=-1)
        return self.proj(concatenated)


# =============================================================================
# SAFE ATTACH MIXIN
# =============================================================================

class SafeAttachMixin:
    """Mixin that overrides attach() to avoid nn.Module parent cycle."""

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
    """
    Tower built from TowerConfig.

    Allows any combination of RoPE and Address types.
    """

    def __init__(
        self,
        config: TowerConfig,
        # Defaults (can be overridden in config)
        default_dim: int = 256,
        default_depth: int = 2,
        default_num_heads: int = 4,
        default_head_dim: int = 64,
        default_ffn_mult: float = 4.0,
        default_dropout: float = 0.1,
        default_fingerprint_dim: int = 64,
    ):
        super().__init__(config.name, strict=False)

        # Resolve config with defaults
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

        # Store config
        self.objects['tower_config'] = {
            'name': config.name,
            'rope': config.rope.value,
            'address': config.address.value,
            'inverted': config.inverted,
            'dim': dim,
            'depth': depth,
        }

        # Transformer config
        tf_config = TransformerConfig(
            dim=dim,
            num_heads=num_heads,
            ffn_mult=ffn_mult,
            variant=TransformerVariant.PRENORM,
            activation=ActivationType.GELU,
            dropout=dropout,
            depth=depth,
        )

        # Build and attach RoPE
        rope = build_rope(
            config.name, config.rope, head_dim,
            **config.rope_params
        )
        self.attach('rope', rope)

        # Build and attach Address (with optional inversion)
        address = build_address(
            config.name, config.address, fingerprint_dim,
            **config.address_params
        )
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

        # Output components
        self.attach('final_norm', nn.LayerNorm(dim))
        self.attach('opinion_proj', nn.Linear(dim, dim))

    @property
    def fingerprint(self) -> Tensor:
        return F.normalize(self['address'].fingerprint, dim=-1)

    @property
    def is_inverted(self) -> bool:
        return self._inverted

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process through tower.

        Returns:
            opinion: [B, D] pooled output
            features: [B, L, D] full sequence
        """
        for block in self.stages:
            x, _ = block(x, mask=mask)

        features = self['final_norm'](x)
        pooled = features.mean(dim=1)
        opinion = self['opinion_proj'](pooled)

        return opinion, features


# =============================================================================
# TOWER OPINION CONTAINERS
# =============================================================================

@dataclass
class TowerOpinion:
    """Output from a single tower."""
    name: str
    opinion: Tensor
    features: Tensor
    fingerprint: Tensor
    config: TowerConfig


@dataclass
class CollectiveOpinion:
    """Aggregated output from all towers."""
    fused: Tensor
    opinions: Dict[str, TowerOpinion]
    weights: Optional[Tensor]
    collective_fingerprint: Tensor


# =============================================================================
# CONFIGURABLE COLLECTIVE
# =============================================================================

class ConfigurableCollective(SafeAttachMixin, BaseRouter):
    """
    Tower collective built from list of TowerConfigs.

    Allows complete flexibility in tower composition.
    """

    def __init__(
        self,
        name: str,
        tower_configs: List[TowerConfig],
        # Collective-level defaults
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
        super().__init__(name, strict=False)

        self.dim = dim
        self._fingerprint_dim = fingerprint_dim
        self._num_towers = len(tower_configs)

        if isinstance(fusion_type, str):
            fusion_type = FusionType(fusion_type)

        # Store metadata
        self.objects['tower_names'] = [c.name for c in tower_configs]
        self.objects['tower_configs'] = tower_configs
        self.objects['collective_config'] = {
            'dim': dim,
            'num_towers': self._num_towers,
            'fusion_type': fusion_type.value,
        }

        # Build and attach each tower
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

        # Build and attach fusion
        fusion_params = fusion_params or {}
        fusion = build_fusion(
            f'{name}_fusion',
            fusion_type,
            num_inputs=self._num_towers,
            in_features=dim,
            **fusion_params
        )
        self.attach('fusion', fusion)

        # Input projection
        self.attach('input_proj', nn.Linear(dim, dim))

        # Fingerprint aggregation
        total_fp_dim = sum(
            self[n].fingerprint.shape[-1] for n in self.objects['tower_names']
        )
        self.attach('fp_proj', nn.Linear(total_fp_dim, fingerprint_dim))

        # Debug state
        self.objects['debug'] = False
        self.objects['debug_info'] = {}

    @property
    def tower_names(self) -> List[str]:
        return self.objects['tower_names']

    @property
    def towers(self) -> Dict[str, ConfigurableTower]:
        return {name: self[name] for name in self.tower_names}

    def debug_on(self):
        self.objects['debug'] = True

    def debug_off(self):
        self.objects['debug'] = False

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> CollectiveOpinion:
        """Process through tower collective."""
        debug = self.objects['debug']
        debug_info = self.objects['debug_info']

        # Handle [B, D] input
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Project input
        x = self['input_proj'](x)

        # Process through towers
        opinions = []
        opinion_tensors = []

        for tower_name in self.tower_names:
            tower = self[tower_name]
            opinion, features = tower(x, mask=mask)

            tower_opinion = TowerOpinion(
                name=tower_name,
                opinion=opinion,
                features=features,
                fingerprint=tower.fingerprint,
                config=tower.config,
            )
            opinions.append(tower_opinion)
            opinion_tensors.append(opinion)

            if debug:
                debug_info[f'{tower_name}_opinion_norm'] = opinion.norm(dim=-1).mean().item()

        # Fuse opinions
        fused = self['fusion'](*opinion_tensors)

        # Extract weights if available
        fusion = self['fusion']
        weights = None
        if hasattr(fusion, 'weight_net'):
            stacked = torch.stack(opinion_tensors, dim=0)
            weights = fusion.weight_net(stacked)
            weights = F.softmax(weights / fusion.temperature, dim=0)
            weights = weights.squeeze(-1).T

        # Aggregate fingerprints
        fp_stack = torch.cat([op.fingerprint for op in opinions], dim=-1)
        collective_fp = self['fp_proj'](fp_stack)
        collective_fp = F.normalize(collective_fp, dim=-1)

        if debug:
            debug_info['fused_norm'] = fused.norm(dim=-1).mean().item()
            if weights is not None:
                debug_info['weights'] = weights.detach().cpu()

        return CollectiveOpinion(
            fused=fused,
            opinions={op.name: op for op in opinions},
            weights=weights,
            collective_fingerprint=collective_fp,
        )

    def tower_fingerprints(self) -> Dict[str, Tensor]:
        return {name: self[name].fingerprint for name in self.tower_names}

    def fingerprint_similarity_matrix(self) -> Tensor:
        fps = list(self.tower_fingerprints().values())
        max_dim = max(fp.shape[-1] for fp in fps)
        padded = []
        for fp in fps:
            if fp.shape[-1] < max_dim:
                pad = torch.zeros(max_dim - fp.shape[-1], device=fp.device, dtype=fp.dtype)
                fp = torch.cat([fp, pad], dim=-1)
            padded.append(fp)
        fp_stack = torch.stack(padded)
        fp_norm = F.normalize(fp_stack, dim=-1)
        return torch.mm(fp_norm, fp_norm.T)

    def get_debug_info(self) -> Dict:
        return dict(self.objects['debug_info'])


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
    """
    Build a tower collective from a list of configs.

    Example:
        configs = [
            TowerConfig('cantor_pos', rope='cantor', address='cantor'),
            TowerConfig('cantor_neg', rope='cantor', address='cantor', inverted=True),
            TowerConfig('beatrix_pos', rope='beatrix', address='beatrix'),
            TowerConfig('beatrix_neg', rope='beatrix', address='beatrix', inverted=True),
        ]
        collective = build_tower_collective(configs, dim=256, default_depth=1)
    """
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
    geometries: List[str] = ['cantor', 'beatrix', 'helix', 'simplex'],
) -> List[TowerConfig]:
    """Create pos/neg pairs for each geometry."""
    configs = []
    for geom in geometries:
        configs.append(TowerConfig(
            name=f'{geom}_pos',
            rope=geom,
            address=geom,
            inverted=False,
        ))
        configs.append(TowerConfig(
            name=f'{geom}_neg',
            rope=geom,
            address=geom,
            inverted=True,
        ))
    return configs


def preset_all_six() -> List[TowerConfig]:
    """Original 6-tower config from AgathaTowerCollective."""
    return [
        TowerConfig('cantor', rope='cantor', address='cantor'),
        TowerConfig('beatrix', rope='beatrix', address='beatrix'),
        TowerConfig('helix', rope='helix', address='helix'),
        TowerConfig('simplex', rope='simplex', address='simplex'),
        TowerConfig('fractal', rope='fractal', address='fractal'),
        TowerConfig('standard', rope='standard', address='standard'),
    ]


def preset_hybrid_experiments() -> List[TowerConfig]:
    """Experimental hybrid configs - mix rope and address types."""
    return [
        # Cantor RoPE with different addresses
        TowerConfig('cantor_cantor', rope='cantor', address='cantor'),
        TowerConfig('cantor_simplex', rope='cantor', address='simplex'),
        TowerConfig('cantor_spherical', rope='cantor', address='spherical'),
        # Beatrix RoPE with different addresses
        TowerConfig('beatrix_beatrix', rope='beatrix', address='beatrix'),
        TowerConfig('beatrix_fractal', rope='beatrix', address='fractal'),
        # Inverted variants
        TowerConfig('cantor_inv', rope='cantor', address='cantor', inverted=True),
        TowerConfig('beatrix_inv', rope='beatrix', address='beatrix', inverted=True),
    ]


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Configurable Tower Builder Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test 1: Pos/Neg pairs
    print("\n--- Test 1: Pos/Neg Pairs ---")
    configs = preset_pos_neg_pairs(['cantor', 'beatrix'])
    collective = build_tower_collective(configs, dim=256, default_depth=1).to(device)

    print(f"Towers: {collective.tower_names}")
    for name in collective.tower_names:
        tower = collective[name]
        cfg = tower.config
        print(f"  {name}: rope={cfg.rope.value}, addr={cfg.address.value}, inv={cfg.inverted}")

    # Forward pass
    B, L, D = 2, 64, 256
    x = torch.randn(B, L, D, device=device)
    result = collective(x)
    print(f"Fused shape: {result.fused.shape}")

    # Test 2: Custom hybrid
    print("\n--- Test 2: Custom Hybrid ---")
    configs = [
        TowerConfig('hybrid1', rope='cantor', address='simplex', depth=2),
        TowerConfig('hybrid2', rope='beatrix', address='fractal', depth=1),
        TowerConfig('deep', rope='helix', address='helix', depth=4),
    ]
    collective2 = build_tower_collective(configs, dim=256).to(device)

    for name in collective2.tower_names:
        tower = collective2[name]
        print(f"  {name}: depth={tower.depth}, params={sum(p.numel() for p in tower.parameters()):,}")

    result2 = collective2(x)
    print(f"Fused shape: {result2.fused.shape}")

    # Test 3: Fingerprint similarity
    print("\n--- Test 3: Fingerprint Similarity ---")
    sim = collective.fingerprint_similarity_matrix()
    print(f"Similarity matrix:\n{sim}")

    print("\n✓ Tower Builder Ready")