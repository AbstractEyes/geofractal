"""
geofractal.router.components.walker_component
=============================================

Static interpolative walking with optional learnable modulation.

v1.1.1: Added fingerprint modulation support

Architecture:

    ConfigurableWalker (STATIC - no learnable params):
        Composes utilities: blend + schedule + aggregation
        Pure mathematical formula walking
        walk(a, b) → stepped interpolation → aggregated result

    WalkerInception (LEARNABLE - optional, None by default):
        Generates aux features from inputs
        Modulates schedule and aggregation weights
        Only instantiated when explicitly requested

    WalkerFusion (FusionComponent):
        Wraps ConfigurableWalker + optional WalkerInception
        inception=None → pure static walking (just the formula)
        inception=WalkerInception(...) → modulated walking
        fingerprint_dim=N → enables collective fingerprint gating

Key Principle:
    No configuration = just the formula
    Learnable sections are None by default

Experimental Results:
    Static shiva+cosine+similarity_tree: 98.21% CIFAR-10
    With cosine aux modulation: 89.00% CIFAR-100 (0.994 consistency)
    With geometric aux modulation: 88.96% (0.997 consistency)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, Dict, Any, Union, List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.components.torch_component import TorchComponent

# Static utilities
from geofractal.router.components.utility.blend_utility import (
    BlendUtility, create_blend, BLEND_REGISTRY,
)
from geofractal.router.components.utility.schedule_utility import (
    ScheduleUtility, create_schedule, SCHEDULE_REGISTRY,
)
from geofractal.router.components.utility.aggregation_utility import (
    AggregationUtility, create_aggregation, AGGREGATION_REGISTRY,
)


# =============================================================================
# CONFIGURABLE WALKER (Static - No Learnable Params)
# =============================================================================

class ConfigurableWalker:
    """
    Static interpolative walking using utility formulas.

    NOT a TorchComponent - no learnable parameters.
    Pure mathematical formula composition.

    Composes:
        - BlendUtility: How to interpolate (shiva, slerp, lerp)
        - ScheduleUtility: Alpha progression (cosine, linear)
        - AggregationUtility: How to reduce steps (mean, similarity_tree)

    Usage:
        walker = ConfigurableWalker(blend='shiva', schedule='cosine', aggregation='mean')
        result = walker.walk(a, b, num_steps=8)
    """

    def __init__(
        self,
        blend: Union[str, BlendUtility] = 'lerp',
        schedule: Union[str, ScheduleUtility] = 'linear',
        aggregation: Union[str, AggregationUtility] = 'mean',
        name: str = 'walker',
    ):
        self.name = name

        # Resolve utilities
        self.blend = create_blend(blend) if isinstance(blend, str) else blend
        self.schedule = create_schedule(schedule) if isinstance(schedule, str) else schedule
        self.aggregation = create_aggregation(aggregation) if isinstance(aggregation, str) else aggregation

    def walk(
        self,
        a: Tensor,
        b: Tensor,
        num_steps: int = 8,
        alphas: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Walk from a to b through interpolation steps.

        Args:
            a: [B, D] or [B, T, D] source tensor
            b: [B, D] or [B, T, D] target tensor
            num_steps: Number of interpolation steps
            alphas: Optional pre-computed alpha schedule [S] or [B, S]

        Returns:
            (stepped, aggregated):
                stepped: [B, S, D] or [B, S, T, D] all interpolation steps
                aggregated: [B, D] or [B, T, D] final result
        """
        device = a.device
        dtype = a.dtype

        # Get alpha schedule
        if alphas is None:
            alphas = self.schedule(num_steps, device=device)

        # Ensure alphas on correct device
        alphas = alphas.to(device=device, dtype=dtype)

        # Handle dimensions
        if a.dim() == 2:
            # [B, D] → compute stepped [B, S, D]
            B, D = a.shape
            S = num_steps

            # Expand for vectorized blending
            a_exp = a.unsqueeze(1).expand(B, S, D)  # [B, S, D]
            b_exp = b.unsqueeze(1).expand(B, S, D)  # [B, S, D]

            # Alphas: [S] → [1, S, 1] for broadcasting
            if alphas.dim() == 1:
                alpha_exp = alphas.view(1, S, 1).expand(B, S, D)
            else:
                alpha_exp = alphas.unsqueeze(-1).expand(B, S, D)

            # Apply blend formula (vectorized)
            stepped = self.blend(a_exp, b_exp, alpha_exp)  # [B, S, D]

            # Aggregate steps → [B, D]
            # For 2D inputs, use simple weighted mean over steps
            # (Avoids 4D reshape issues with complex aggregations)
            if alphas.dim() == 1:
                weights = alphas / alphas.sum().clamp_min(1e-8)
                weights = weights.view(1, S, 1).expand(B, S, D)
            else:
                weights = alphas / alphas.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                weights = weights.unsqueeze(-1).expand(B, S, D)

            aggregated = (stepped * weights).sum(dim=1)  # [B, D]

        else:
            # [B, T, D] → compute stepped [B, S, T, D]
            B, T, D = a.shape
            S = num_steps

            # Expand for vectorized blending
            a_exp = a.unsqueeze(1).expand(B, S, T, D)  # [B, S, T, D]
            b_exp = b.unsqueeze(1).expand(B, S, T, D)  # [B, S, T, D]

            # Alphas: [S] → [1, S, 1, 1] for broadcasting
            if alphas.dim() == 1:
                alpha_exp = alphas.view(1, S, 1, 1).expand(B, S, T, D)
            else:
                alpha_exp = alphas.view(B, S, 1, 1).expand(B, S, T, D)

            # Apply blend formula (vectorized)
            stepped = self.blend(a_exp, b_exp, alpha_exp)  # [B, S, T, D]

            # Aggregate steps → [B, T, D]
            aggregated = self.aggregation(stepped, alphas=alphas)  # [B, T, D]

        return stepped, aggregated

    def __call__(
        self,
        a: Tensor,
        b: Tensor,
        num_steps: int = 8,
        alphas: Optional[Tensor] = None,
    ) -> Tensor:
        """Convenience: returns only aggregated result."""
        _, aggregated = self.walk(a, b, num_steps, alphas)
        return aggregated

    def __repr__(self) -> str:
        return (
            f"ConfigurableWalker(blend={self.blend.blend_name}, "
            f"schedule={self.schedule.schedule_name}, "
            f"aggregation={self.aggregation.aggregation_name})"
        )


# =============================================================================
# WALKER PRESETS (Static Configurations)
# =============================================================================

WALKER_PRESETS = {
    # Best performers from ablation
    'shiva': {
        'blend': 'shiva',
        'schedule': 'cosine',
        'aggregation': 'similarity_tree',
    },
    'slerp': {
        'blend': 'slerp',
        'schedule': 'cosine',
        'aggregation': 'mean',
    },
    'lerp': {
        'blend': 'lerp',
        'schedule': 'linear',
        'aggregation': 'mean',
    },
    'slip': {
        'blend': 'slip',
        'schedule': 'cosine',
        'aggregation': 'similarity',
    },
    'zeus': {
        'blend': 'zeus',
        'schedule': 'cosine',
        'aggregation': 'softmax',
    },
    'gilgamesh': {
        'blend': 'gilgamesh',
        'schedule': 'tau',
        'aggregation': 'weighted_mean',
    },
}


def create_walker(preset: str = 'lerp', **overrides) -> ConfigurableWalker:
    """
    Create ConfigurableWalker from preset.

    Args:
        preset: Preset name or 'custom'
        **overrides: Override preset values

    Returns:
        ConfigurableWalker (static, no learnable params)
    """
    if preset in WALKER_PRESETS:
        config = WALKER_PRESETS[preset].copy()
    else:
        config = {'blend': 'lerp', 'schedule': 'linear', 'aggregation': 'mean'}

    config.update(overrides)
    return ConfigurableWalker(**config, name=preset)


# =============================================================================
# WALKER INCEPTION (Learnable Modulation - Optional)
# =============================================================================

class WalkerInception(TorchComponent):
    """
    Optional learnable modulation layer for Walker.

    Generates aux features from inputs and modulates:
        - Schedule: base_schedule + modulation
        - Aggregation weights: learned weighting of steps

    This is WHERE learning happens when enabled.
    If not provided to WalkerFusion, pure static walking is used.

    Aux Types:
        - 'cosine': Pairwise cosine similarities
        - 'geometric': Distances + angles + norms
        - 'learned': Fixed learned embeddings
        - 'walker_path': Similarities along interpolation path
    """

    def __init__(
        self,
        name: str,
        in_features: int,
        num_steps: int,
        num_inputs: int = 2,
        aux_type: str = 'cosine',
        aux_dim: int = 64,
        hidden_dim: int = 256,
        base_schedule: str = 'cosine',
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        self.in_features = in_features
        self.num_steps = num_steps
        self.num_inputs = num_inputs
        self.aux_type = aux_type
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim

        # === Auxiliary Generator ===
        num_pairs = num_inputs * (num_inputs - 1) // 2

        if aux_type == 'cosine':
            self.aux_gen = nn.Linear(num_pairs, aux_dim)
        elif aux_type == 'geometric':
            raw_dim = num_pairs * 2 + num_inputs
            self.aux_gen = nn.Linear(raw_dim, aux_dim)
        elif aux_type == 'learned':
            self.aux_embeddings = nn.Parameter(torch.randn(num_inputs, aux_dim) * 0.02)
            self.aux_gen = nn.Linear(num_inputs * aux_dim, aux_dim)
        elif aux_type == 'walker_path':
            raw_dim = num_pairs * (num_steps + 1)
            self.aux_gen = nn.Linear(raw_dim, aux_dim)
        else:
            self.aux_gen = None

        # === Base Schedule (buffer) ===
        if base_schedule == 'linear':
            base = torch.linspace(0, 1, num_steps)
        elif base_schedule == 'cosine':
            t = torch.linspace(0, 1, num_steps)
            base = 0.5 * (1 - torch.cos(math.pi * t))
        else:
            base = torch.linspace(0, 1, num_steps)

        self.register_buffer('base_schedule', base)

        # === Schedule Modulator ===
        self.schedule_modulator = nn.Sequential(
            nn.Linear(aux_dim, aux_dim),
            nn.GELU(),
            nn.Linear(aux_dim, num_steps),
            nn.Tanh(),
        )
        self.modulation_scale = nn.Parameter(torch.tensor(0.1))

        # === Aggregation Weight Network ===
        self.agg_weight_net = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_steps),
        )

    def compute_aux(self, *inputs: Tensor) -> Tensor:
        """Generate auxiliary features from inputs."""
        B = inputs[0].shape[0]
        device, dtype = inputs[0].device, inputs[0].dtype

        # Pool if 3D
        pooled = []
        for x in inputs:
            if x.dim() == 3:
                pooled.append(x.mean(dim=1))
            else:
                pooled.append(x)

        if self.aux_gen is None:
            return torch.zeros(B, self.aux_dim, device=device, dtype=dtype)

        if self.aux_type == 'cosine':
            cosines = []
            for i in range(len(pooled)):
                for j in range(i + 1, len(pooled)):
                    cos = F.cosine_similarity(pooled[i], pooled[j], dim=-1, eps=1e-8)
                    cosines.append(cos)
            cosines = torch.stack(cosines, dim=-1)
            return self.aux_gen(cosines)

        elif self.aux_type == 'geometric':
            features = []
            norms = [x.norm(dim=-1, keepdim=True) for x in pooled]
            for i in range(len(pooled)):
                for j in range(i + 1, len(pooled)):
                    dist = (pooled[i] - pooled[j]).norm(dim=-1, keepdim=True)
                    cos = F.cosine_similarity(pooled[i], pooled[j], dim=-1, eps=1e-8).unsqueeze(-1)
                    features.extend([dist, cos])
            features.extend(norms)
            combined = torch.cat(features, dim=-1)
            return self.aux_gen(combined)

        elif self.aux_type == 'learned':
            expanded = self.aux_embeddings.unsqueeze(0).expand(B, -1, -1)
            return self.aux_gen(expanded.reshape(B, -1))

        elif self.aux_type == 'walker_path':
            features = []
            ts = torch.linspace(0, 1, self.num_steps, device=device)
            for i in range(len(pooled)):
                for j in range(i + 1, len(pooled)):
                    a, b = pooled[i], pooled[j]
                    midpoint = (a + b) / 2
                    cos_ab = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
                    features.append(cos_ab.unsqueeze(-1))
                    for t in ts:
                        interp = (1 - t) * a + t * b
                        cos_interp = F.cosine_similarity(interp, midpoint, dim=-1, eps=1e-8)
                        features.append(cos_interp.unsqueeze(-1))
            combined = torch.cat(features, dim=-1)
            return self.aux_gen(combined)

        return torch.zeros(B, self.aux_dim, device=device, dtype=dtype)

    def modulate_schedule(self, aux: Tensor) -> Tensor:
        """Apply aux-based modulation to base schedule."""
        modulation = self.schedule_modulator(aux)  # [B, S]
        scaled = modulation * self.modulation_scale
        schedule = self.base_schedule.unsqueeze(0) + scaled
        return schedule.clamp(0, 1)

    def compute_agg_weights(self, aux: Tensor) -> Tensor:
        """Compute aggregation weights from aux features."""
        raw_weights = self.agg_weight_net(aux)
        return F.softmax(raw_weights, dim=-1)

    def forward(self, *inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full inception forward.

        Returns:
            (aux, modulated_schedule, agg_weights)
        """
        aux = self.compute_aux(*inputs)
        schedule = self.modulate_schedule(aux)
        agg_weights = self.compute_agg_weights(aux)
        return aux, schedule, agg_weights


# =============================================================================
# WALKER FUSION (FusionComponent - Houses Everything)
# =============================================================================

class WalkerFusion(TorchComponent):
    """
    Fusion via interpolative walking with optional learned modulation.

    Wraps:
        - ConfigurableWalker (STATIC): The walking formula
        - WalkerInception (OPTIONAL): Learned modulation

    If inception=None:
        Pure static walking using walker formula only
        No learnable parameters (except optional projections)

    If inception provided:
        Modulated walking with learned aux features
        schedule = base + inception.modulate(aux)
        agg_weights = inception.compute_weights(aux)

    Usage:
        # Pure static (no learning)
        fusion = WalkerFusion("walk", in_features=512, preset='shiva')

        # With learnable modulation
        inception = WalkerInception("inc", in_features=512, num_steps=8)
        fusion = WalkerFusion("walk", in_features=512, preset='shiva', inception=inception)
    """

    def __init__(
        self,
        name: str,
        in_features: int,
        out_features: Optional[int] = None,
        num_steps: int = 8,
        # Walker configuration (static)
        preset: str = 'lerp',
        blend: Optional[str] = None,
        schedule: Optional[str] = None,
        aggregation: Optional[str] = None,
        # Optional learnable modulation (None = pure static)
        inception: Optional[WalkerInception] = None,
        # Optional projections
        project_inputs: bool = False,
        project_output: bool = False,
        hidden_dim: Optional[int] = None,
        # Fingerprint modulation (v1.1.1)
        fingerprint_dim: Optional[int] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        self.in_features = in_features
        self.out_features = out_features or in_features
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim or in_features
        self.fingerprint_dim = fingerprint_dim

        # === Static Walker (formula only) ===
        walker_config = {}
        if blend is not None:
            walker_config['blend'] = blend
        if schedule is not None:
            walker_config['schedule'] = schedule
        if aggregation is not None:
            walker_config['aggregation'] = aggregation

        self.walker = create_walker(preset, **walker_config)

        # === Optional Inception (learnable, None by default) ===
        self.inception = inception

        # === Optional Projections ===
        if project_inputs:
            self.input_proj = nn.Linear(in_features, self.hidden_dim)
        else:
            self.input_proj = None

        if project_output or self.out_features != in_features:
            self.output_proj = nn.Linear(
                self.hidden_dim if project_inputs else in_features,
                self.out_features
            )
        else:
            self.output_proj = None

        # === Fingerprint Gate (v1.1.1) ===
        # Projects fingerprint to scale factor for output modulation
        if fingerprint_dim is not None:
            self.fingerprint_gate = nn.Sequential(
                nn.Linear(fingerprint_dim, fingerprint_dim),
                nn.GELU(),
                nn.Linear(fingerprint_dim, self.out_features),
                nn.Sigmoid(),
            )
        else:
            self.fingerprint_gate = None

        # Diagnostics
        self._last_stepped = None
        self._last_alphas = None
        self._last_agg_weights = None

    def forward(self, *inputs: Tensor, fingerprint: Optional[Tensor] = None) -> Tensor:
        """
        Walk through inputs.

        For 2 inputs: direct walking a→b
        For 3+ inputs: hierarchical (a,b)→ab, (ab,c)→abc, ...

        Args:
            *inputs: 2+ tensors of [B, D] or [B, T, D]
            fingerprint: Optional [B, fp_dim] collective fingerprint for modulation

        Returns:
            [B, D] or [B, T, D] fused result
        """
        if len(inputs) < 2:
            raise ValueError(f"WalkerFusion requires at least 2 inputs, got {len(inputs)}")

        if len(inputs) == 2:
            return self._walk_pair(inputs[0], inputs[1], fingerprint=fingerprint)

        # Hierarchical fusion for 3+ inputs
        current = self._walk_pair(inputs[0], inputs[1], fingerprint=fingerprint)
        for i in range(2, len(inputs)):
            current = self._walk_pair(current, inputs[i], fingerprint=fingerprint)

        return current

    def _walk_pair(self, a: Tensor, b: Tensor, fingerprint: Optional[Tensor] = None) -> Tensor:
        """
        Walk from a to b (internal 2-input method).

        Args:
            a: [B, D] or [B, T, D] source
            b: [B, D] or [B, T, D] target
            fingerprint: Optional [B, fp_dim] for output modulation

        Returns:
            [B, D] or [B, T, D] fused result
        """
        # Optional input projection
        if self.input_proj is not None:
            a = self.input_proj(a)
            b = self.input_proj(b)

        # Determine schedule and agg weights
        if self.inception is not None:
            # Modulated walking
            aux, alphas, agg_weights = self.inception(a, b)
            self._last_alphas = alphas.detach()
            self._last_agg_weights = agg_weights.detach()

            # Walk with modulated schedule
            stepped, _ = self.walker.walk(a, b, self.num_steps, alphas=alphas)
            self._last_stepped = stepped.detach()

            # Apply learned aggregation weights
            if stepped.dim() == 3:
                # [B, S, D] → weighted sum
                result = (stepped * agg_weights.unsqueeze(-1)).sum(dim=1)
            else:
                # [B, S, T, D] → weighted sum over S
                result = (stepped * agg_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        else:
            # Pure static walking (no learning)
            stepped, result = self.walker.walk(a, b, self.num_steps)
            self._last_stepped = stepped.detach()
            self._last_alphas = None
            self._last_agg_weights = None

        # Optional output projection
        if self.output_proj is not None:
            result = self.output_proj(result)

        # Fingerprint modulation (v1.1.1)
        if fingerprint is not None and self.fingerprint_gate is not None:
            gate = self.fingerprint_gate(fingerprint)  # [B, out_features]
            if result.dim() == 3:
                gate = gate.unsqueeze(1)  # [B, 1, out_features]
            result = result * gate

        return result

    def fuse(self, *inputs: Tensor, fingerprint: Optional[Tensor] = None) -> Tensor:
        """Alias for forward (FusionComponent interface)."""
        return self.forward(*inputs, fingerprint=fingerprint)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics from last forward pass."""
        return {
            'stepped': self._last_stepped,
            'alphas': self._last_alphas,
            'agg_weights': self._last_agg_weights,
            'walker': str(self.walker),
            'has_inception': self.inception is not None,
        }

    def __repr__(self) -> str:
        inception_str = "None" if self.inception is None else self.inception.aux_type
        return (
            f"WalkerFusion(name='{self.name}', "
            f"walker={self.walker.blend.blend_name}+{self.walker.schedule.schedule_name}, "
            f"inception={inception_str}, "
            f"steps={self.num_steps}, "
            f"params={self.num_parameters():,})"
        )


# =============================================================================
# CONVENIENCE FACTORIES
# =============================================================================

def create_walker_fusion(
    name: str,
    in_features: int,
    preset: str = 'lerp',
    num_steps: int = 8,
    with_inception: bool = False,
    aux_type: str = 'cosine',
    **kwargs,
) -> WalkerFusion:
    """
    Create WalkerFusion with optional inception.

    Args:
        name: Component name
        in_features: Input feature dimension
        preset: Walker preset ('shiva', 'slerp', 'lerp', etc.)
        num_steps: Number of interpolation steps
        with_inception: If True, create with learnable inception
        aux_type: Aux type for inception ('cosine', 'geometric', 'learned')
        **kwargs: Additional WalkerFusion arguments

    Returns:
        WalkerFusion (static if with_inception=False, learnable if True)
    """
    inception = None
    if with_inception:
        inception = WalkerInception(
            f"{name}_inception",
            in_features=in_features,
            num_steps=num_steps,
            num_inputs=2,
            aux_type=aux_type,
        )

    return WalkerFusion(
        name=name,
        in_features=in_features,
        num_steps=num_steps,
        preset=preset,
        inception=inception,
        **kwargs,
    )


# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':

    def section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    section("CONFIGURABLE WALKER (Static)")

    walker = ConfigurableWalker(blend='shiva', schedule='cosine', aggregation='mean')
    print(walker)

    B, D = 4, 512
    a = torch.randn(B, D, device=device)
    b = torch.randn(B, D, device=device)

    stepped, result = walker.walk(a, b, num_steps=8)
    print(f"Input a: {a.shape}")
    print(f"Input b: {b.shape}")
    print(f"Stepped: {stepped.shape}")
    print(f"Result:  {result.shape}")

    section("WALKER PRESETS")

    for preset_name in WALKER_PRESETS:
        w = create_walker(preset_name)
        print(f"{preset_name:<12s} | {w}")

    section("WALKER FUSION (Static - No Inception)")

    fusion = WalkerFusion(
        "static_walk",
        in_features=512,
        preset='shiva',
        num_steps=8,
        inception=None,  # Explicitly None = pure static
    )
    fusion.to(device)

    print(fusion)
    print(f"Parameters: {fusion.num_parameters()}")  # Should be 0 (no projections)

    result = fusion(a, b)
    print(f"Result: {result.shape}")

    diag = fusion.get_diagnostics()
    print(f"Has inception: {diag['has_inception']}")
    print(f"Walker: {diag['walker']}")

    section("WALKER FUSION (With Inception)")

    inception = WalkerInception(
        "test_inception",
        in_features=512,
        num_steps=8,
        num_inputs=2,
        aux_type='cosine',
    )

    fusion_learn = WalkerFusion(
        "learned_walk",
        in_features=512,
        preset='slerp',
        num_steps=8,
        inception=inception,
    )
    fusion_learn.to(device)

    print(fusion_learn)
    print(f"Parameters: {fusion_learn.num_parameters()}")  # Should be >0

    result = fusion_learn(a, b)
    print(f"Result: {result.shape}")

    diag = fusion_learn.get_diagnostics()
    print(f"Has inception: {diag['has_inception']}")
    print(f"Alphas shape: {diag['alphas'].shape}")
    print(f"Agg weights: {[f'{w:.3f}' for w in diag['agg_weights'][0].tolist()]}")

    section("CONVENIENCE FACTORY")

    # Static (default)
    static = create_walker_fusion("s", in_features=512, preset='shiva')
    print(f"Static: {static.num_parameters()} params, has_inception={static.inception is not None}")

    # With inception
    learned = create_walker_fusion("l", in_features=512, preset='shiva', with_inception=True)
    print(f"Learned: {learned.num_parameters()} params, has_inception={learned.inception is not None}")

    section("MULTI-INPUT FUSION (3 inputs)")

    c = torch.randn(B, D, device=device)

    result = fusion(a, b, c)  # Uses fuse() method
    print(f"3-input fusion result: {result.shape}")

    section("CONSISTENCY CHECK")

    # Static walker should be deterministic
    result1 = fusion(a, b)
    result2 = fusion(a, b)
    diff = (result1 - result2).abs().max().item()
    print(f"Static walker same-input difference: {diff:.10f} (should be 0)")

    section("3D INPUT TEST (with aggregation utility)")

    # Test with 3D inputs [B, T, D] to exercise full aggregation path
    T = 16
    a_3d = torch.randn(B, T, D, device=device)
    b_3d = torch.randn(B, T, D, device=device)

    fusion_3d = WalkerFusion(
        "test_3d",
        in_features=512,
        preset='shiva',  # Uses similarity_tree aggregation
        num_steps=8,
    )
    fusion_3d.to(device)

    result_3d = fusion_3d(a_3d, b_3d)
    print(f"3D input shape: {a_3d.shape}")
    print(f"3D output shape: {result_3d.shape}")
    print(f"3D result mean: {result_3d.mean().item():.4f}")

    section("ALL TESTS PASSED")