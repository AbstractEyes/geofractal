"""
geofractal.router.prefab.agatha.tower_collective
=================================================

Agatha Block 2: Six Tower Collective

Six specialized towers process HeadMail and produce teaching signals.
Each tower has distinct geometric configuration via RoPE + Address.

NOTE: This file overrides attach() to use object.__setattr__ for parent
assignment, avoiding the nn.Module cycle that occurs when parent is set
directly on TorchComponents.

Copyright 2025 AbstractPhil
Licensed under Apache License 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Literal, Any
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
# GEOMETRY TYPES
# =============================================================================

class GeometryType(str, Enum):
    """Tower geometric personalities."""
    CANTOR = "cantor"
    BEATRIX = "beatrix"
    HELIX = "helix"
    SIMPLEX = "simplex"
    FRACTAL = "fractal"
    STANDARD = "standard"


# =============================================================================
# OUTPUT CONTAINERS
# =============================================================================

@dataclass
class TowerOpinion:
    """Output from a single tower."""
    name: str
    opinion: Tensor
    features: Tensor
    fingerprint: Tensor
    geometry: GeometryType


@dataclass
class CollectiveOpinion:
    """Aggregated output from all towers."""
    fused: Tensor
    opinions: Dict[str, TowerOpinion]
    weights: Tensor
    collective_fingerprint: Tensor


# =============================================================================
# SAFE ATTACH MIXIN
# =============================================================================

#lass SafeAttachMixin:
#   """
#   Mixin that overrides attach() to avoid nn.Module parent cycle.

#   The issue: When BaseRouter.attach() sets component.parent = self,
#   and both are nn.Module, PyTorch's __setattr__ registers self as
#   a submodule of component, creating a cycle.

#   Fix: Use object.__setattr__ to bypass nn.Module tracking.
#   """

#   def attach(self, name: str, component: Any) -> None:
#       """Attach component without creating parent cycle."""
#       if isinstance(component, nn.Module):
#           self.components[name] = component
#       else:
#           self.objects[name] = component

#       # Set parent using object.__setattr__ to bypass nn.Module tracking
#       if hasattr(component, 'parent') and hasattr(component, 'on_attach'):
#           object.__setattr__(component, 'parent', self)
#           component.on_attach(self)


# =============================================================================
# ROPE FACTORY
# =============================================================================

def build_rope(name: str, geometry: GeometryType, head_dim: int) -> nn.Module:
    """Build geometry-specific RoPE."""
    if geometry == GeometryType.CANTOR:
        return CantorRoPE(f'{name}_rope', head_dim=head_dim, levels=5, mode='hybrid')
    elif geometry == GeometryType.BEATRIX:
        return BeatrixRoPE(f'{name}_rope', head_dim=head_dim, levels=5)
    elif geometry == GeometryType.HELIX:
        return TriRoPE(
            f'{name}_rope', head_dim=head_dim,
            theta_alpha=10000.0, theta_beta=5000.0, theta_gamma=2500.0,
            augmentation='barycentric'
        )
    elif geometry == GeometryType.SIMPLEX:
        return QuadRoPE(
            f'{name}_rope', head_dim=head_dim,
            theta_w=10000.0, theta_x=7500.0, theta_y=5000.0, theta_z=2500.0,
            augmentation='simplex'
        )
    elif geometry == GeometryType.FRACTAL:
        return DualRoPE(
            f'{name}_rope', head_dim=head_dim,
            theta_primary=10000.0, theta_secondary=3000.0,
            augmentation='lerp'
        )
    else:  # STANDARD
        return RoPE(f'{name}_rope', head_dim=head_dim, theta=10000.0)


# =============================================================================
# ADDRESS FACTORY
# =============================================================================

def build_address(name: str, geometry: GeometryType, fingerprint_dim: int) -> nn.Module:
    """Build geometry-specific Address."""
    if geometry == GeometryType.CANTOR:
        return CantorAddressComponent(
            f'{name}_address', k_simplex=4,
            fingerprint_dim=fingerprint_dim, mode='staircase'
        )
    elif geometry == GeometryType.BEATRIX:
        return CantorAddressComponent(
            f'{name}_address', k_simplex=4,
            fingerprint_dim=fingerprint_dim, mode='staircase_features'
        )
    elif geometry == GeometryType.HELIX:
        return SphericalAddressComponent(
            f'{name}_address', fingerprint_dim=fingerprint_dim
        )
    elif geometry == GeometryType.SIMPLEX:
        return SimplexAddressComponent(
            f'{name}_address', k=4, embed_dim=fingerprint_dim
        )
    elif geometry == GeometryType.FRACTAL:
        return FractalAddressComponent(
            f'{name}_address', region='seahorse',
            orbit_length=fingerprint_dim // 2
        )
    else:  # STANDARD
        return AddressComponent(
            f'{name}_address', fingerprint_dim=fingerprint_dim
        )


# =============================================================================
# GEOMETRIC TOWER (BaseTower + SafeAttach)
# =============================================================================

class GeometricTower(BaseTower):
    """
    Single tower with geometric personality.

    Extends BaseTower with SafeAttachMixin to avoid parent cycle.
    Uses attach() for components, append() for stages.
    """

    def __init__(
        self,
        name: str,
        dim: int = 1024,
        depth: int = 4,
        num_heads: int = 8,
        geometry: GeometryType = GeometryType.STANDARD,
        fingerprint_dim: int = 64,
        head_dim: int = 64,
        ffn_mult: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        uuid: Optional[str] = None,
    ):
        super().__init__(name, uuid=uuid, strict=False)

        self.dim = dim
        self.depth = depth
        self.geometry = geometry
        self._fingerprint_dim = fingerprint_dim

        # Store config in objects (non-module storage)
        self.objects['tower_config'] = {
            'dim': dim,
            'depth': depth,
            'num_heads': num_heads,
            'geometry': geometry.value,
        }

        # Transformer config
        config = TransformerConfig(
            dim=dim,
            num_heads=num_heads,
            ffn_mult=ffn_mult,
            variant=TransformerVariant.PRENORM,
            activation=ActivationType(activation),
            dropout=dropout,
            depth=depth,
        )

        # Attach RoPE component
        self.attach('rope', build_rope(name, geometry, head_dim))

        # Attach Address component
        self.attach('address', build_address(name, geometry, fingerprint_dim))

        # Append transformer blocks to stages
        for i in range(depth):
            block = PreNormBlock(
                name=f'{name}_block_{i}',
                config=config,
                block_idx=i,
            )
            self.append(block)

        # Attach output components
        self.attach('final_norm', nn.LayerNorm(dim))
        self.attach('opinion_proj', nn.Linear(dim, dim))

    @property
    def fingerprint(self) -> Tensor:
        """Get normalized tower fingerprint."""
        return F.normalize(self['address'].fingerprint, dim=-1)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process input through tower stages.

        Args:
            x: [B, L, D] input sequence
            mask: Optional attention mask

        Returns:
            opinion: [B, D] pooled output
            features: [B, L, D] full sequence
        """
        # Process through transformer blocks (stages)
        for block in self.stages:
            x, _ = block(x, mask=mask)

        # Final norm
        features = self['final_norm'](x)

        # Pool to opinion
        pooled = features.mean(dim=1)
        opinion = self['opinion_proj'](pooled)

        return opinion, features


# =============================================================================
# TOWER COLLECTIVE (BaseRouter + SafeAttach)
# =============================================================================

class AgathaTowerCollective(BaseRouter):
    """
    Agatha Block 2: Six Tower Collective.

    Router containing six GeometricTowers with distinct personalities.
    All components attached via attach() pattern.
    """

    TOWER_CONFIGS: List[Tuple[str, GeometryType]] = [
        ('cantor', GeometryType.CANTOR),
        ('beatrix', GeometryType.BEATRIX),
        ('helix', GeometryType.HELIX),
        ('simplex', GeometryType.SIMPLEX),
        ('fractal', GeometryType.FRACTAL),
        ('standard', GeometryType.STANDARD),
    ]

    def __init__(
        self,
        name: str = 'tower_collective',
        dim: int = 1024,
        tower_depth: int = 4,
        num_heads: int = 8,
        head_dim: int = 64,
        ffn_mult: float = 4.0,
        dropout: float = 0.1,
        fingerprint_dim: int = 64,
        activation: str = "gelu",
        fusion_type: Literal['adaptive', 'gated', 'attention'] = 'adaptive',
        tower_configs: Optional[List[Tuple[str, GeometryType]]] = None,
        uuid: Optional[str] = None,
    ):
        super().__init__(name, uuid=uuid, strict=False)

        self.dim = dim
        self._fingerprint_dim = fingerprint_dim

        configs = tower_configs or self.TOWER_CONFIGS
        self._num_towers = len(configs)

        # Store tower names in objects (non-module storage)
        self.objects['tower_names'] = [c[0] for c in configs]
        self.objects['collective_config'] = {
            'dim': dim,
            'tower_depth': tower_depth,
            'num_towers': self._num_towers,
            'fusion_type': fusion_type,
        }

        # Attach each tower
        for tower_name, geometry in configs:
            tower = GeometricTower(
                name=tower_name,
                dim=dim,
                depth=tower_depth,
                num_heads=num_heads,
                head_dim=head_dim,
                geometry=geometry,
                fingerprint_dim=fingerprint_dim,
                ffn_mult=ffn_mult,
                dropout=dropout,
                activation=activation,
            )
            self.attach(tower_name, tower)

        # Attach fusion
        if fusion_type == 'adaptive':
            self.attach('fusion', AdaptiveFusion(
                'tower_fusion', num_inputs=self._num_towers, in_features=dim
            ))
        elif fusion_type == 'gated':
            self.attach('fusion', GatedFusion(
                'tower_fusion', num_inputs=self._num_towers, in_features=dim
            ))
        elif fusion_type == 'attention':
            self.attach('fusion', AttentionFusion(
                'tower_fusion', num_inputs=self._num_towers,
                in_features=dim, num_heads=8
            ))
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Attach input projection
        self.attach('input_proj', nn.Linear(dim, dim))

        # Compute actual combined fingerprint dimension from towers
        total_fp_dim = sum(
            self[name].fingerprint.shape[-1] for name in self.objects['tower_names']
        )
        self.attach('fp_proj', nn.Linear(total_fp_dim, fingerprint_dim))

        # Debug state in objects
        self.objects['debug'] = False
        self.objects['debug_info'] = {}

    def debug_on(self):
        self.objects['debug'] = True

    def debug_off(self):
        self.objects['debug'] = False
        self.objects['debug_info'] = {}

    @property
    def tower_names(self) -> List[str]:
        return self.objects['tower_names']

    @property
    def towers(self) -> Dict[str, GeometricTower]:
        """Get towers by name."""
        return {name: self[name] for name in self.tower_names}

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> CollectiveOpinion:
        """
        Process through tower collective.

        Args:
            x: [B, L, D] or [B, D] input
            mask: Optional attention mask

        Returns:
            CollectiveOpinion with fused result
        """
        debug = self.objects['debug']
        debug_info = self.objects['debug_info']

        # Handle [B, D] input
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Project input
        x = self['input_proj'](x)

        # Process through each tower
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
                geometry=tower.geometry,
            )
            opinions.append(tower_opinion)
            opinion_tensors.append(opinion)

            if debug:
                debug_info[f'{tower_name}_opinion_norm'] = opinion.norm(dim=-1).mean().item()

        # Fuse opinions
        fused = self['fusion'](*opinion_tensors)

        # Extract weights
        fusion = self['fusion']
        if hasattr(fusion, 'weight_net'):
            stacked = torch.stack(opinion_tensors, dim=0)
            weights = fusion.weight_net(stacked)
            weights = F.softmax(weights / fusion.temperature, dim=0)
            weights = weights.squeeze(-1).T
        else:
            B = opinion_tensors[0].shape[0]
            weights = torch.ones(B, self._num_towers, device=fused.device) / self._num_towers

        # Aggregate fingerprints
        fp_stack = torch.cat([op.fingerprint for op in opinions], dim=-1)
        collective_fp = self['fp_proj'](fp_stack)
        collective_fp = F.normalize(collective_fp, dim=-1)

        if debug:
            debug_info['fused_norm'] = fused.norm(dim=-1).mean().item()
            debug_info['weights'] = weights.detach().cpu()

        return CollectiveOpinion(
            fused=fused,
            opinions={op.name: op for op in opinions},
            weights=weights,
            collective_fingerprint=collective_fp,
        )

    def tower_fingerprints(self) -> Dict[str, Tensor]:
        """Get all tower fingerprints."""
        return {name: self[name].fingerprint for name in self.tower_names}

    def fingerprint_similarity_matrix(self) -> Tensor:
        """Compute pairwise fingerprint similarities (projects to common dim)."""
        fps = list(self.tower_fingerprints().values())

        # Find max dimension and pad all to that size
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
# FACTORY
# =============================================================================

def create_tower_collective(
    dim: int = 1024,
    tower_depth: int = 4,
    num_heads: int = 8,
    head_dim: int = 64,
    ffn_mult: float = 4.0,
    dropout: float = 0.1,
    fingerprint_dim: int = 64,
    activation: str = "gelu",
    fusion_type: str = 'adaptive',
    subset: Optional[List[str]] = None,
) -> AgathaTowerCollective:
    """Create tower collective."""
    if subset is not None:
        name_to_geometry = dict(AgathaTowerCollective.TOWER_CONFIGS)
        configs = [(name, name_to_geometry[name]) for name in subset]
    else:
        configs = None

    return AgathaTowerCollective(
        name='tower_collective',
        dim=dim,
        tower_depth=tower_depth,
        num_heads=num_heads,
        head_dim=head_dim,
        ffn_mult=ffn_mult,
        dropout=dropout,
        fingerprint_dim=fingerprint_dim,
        activation=activation,
        fusion_type=fusion_type,
        tower_configs=configs,
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Agatha Tower Collective Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    collective = create_tower_collective(
        dim=1024,
        tower_depth=4,
        num_heads=8,
        head_dim=64,
        fingerprint_dim=64,
        fusion_type='adaptive',
    ).to(device)

    print(f"\nTowers: {collective.tower_names}")

    # Debug fingerprint dimensions
    print("\nFingerprint dimensions:")
    for name in collective.tower_names:
        fp = collective[name].fingerprint
        print(f"  {name}: {fp.shape}")

    total_params = sum(p.numel() for p in collective.parameters())
    print(f"Total parameters: {total_params:,}")

    for name in collective.tower_names:
        tower = collective[name]
        tower_params = sum(p.numel() for p in tower.parameters())
        print(f"  {name}: {tower_params:,} params, geometry={tower.geometry.value}")

    print("\n" + "-" * 40)
    print("Forward Pass")
    print("-" * 40)

    collective.debug_on()

    B, L, D = 2, 16, 1024
    x = torch.randn(B, L, D, device=device)

    print(f"Input: {list(x.shape)}")

    with torch.no_grad():
        result = collective(x)

    print(f"Fused output: {list(result.fused.shape)}")
    print(f"Collective fingerprint: {list(result.collective_fingerprint.shape)}")
    print(f"Fusion weights: {list(result.weights.shape)}")

    print("\nTower opinions:")
    for name, op in result.opinions.items():
        print(f"  {name}: opinion={list(op.opinion.shape)}, geometry={op.geometry.value}")

    print("\nFusion weights (sample 0):")
    weights = result.weights[0].cpu().numpy()
    for i, name in enumerate(collective.tower_names):
        bar = "█" * int(weights[i] * 30)
        print(f"  {name:10s}: {weights[i]:.3f} {bar}")

    print("\nFingerprint similarity matrix:")
    sim_matrix = collective.fingerprint_similarity_matrix()
    tower_names = collective.tower_names

    print("           " + " ".join(f"{n[:7]:>8s}" for n in tower_names))
    for i, name in enumerate(tower_names):
        row = sim_matrix[i].cpu().detach().numpy()
        row_str = " ".join(f"{v:8.3f}" for v in row)
        print(f"{name:10s} {row_str}")

    if device.type == 'cuda':
        print(f"\nGPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print("\n" + "-" * 40)
    print("Gradient Check")
    print("-" * 40)

    x_grad = torch.randn(B, L, D, device=device, requires_grad=True)
    result = collective(x_grad)
    loss = result.fused.sum()
    loss.backward()

    print(f"Input grad norm: {x_grad.grad.norm():.4f}")
    print(f"Grad finite: {torch.isfinite(x_grad.grad).all()}")

    print("\n" + "=" * 60)
    print("✓ Tower Collective Ready")
    print("=" * 60)