"""
geofractal.router.prefab.fusion_builder
=======================================

Configurable fusion system with builder pattern.

Like ConfigurableTower but for fusion operations:
- Multiple topology options (parallel, sequential, hierarchical, blend)
- Pluggable fusion strategies
- α/β/γ controllers for fine-grained control
- Cache support for intermediate features

Design:
    ConfigurableFusion wraps fusion strategies with controllers.
    FusionBuilder provides factory methods for easy construction.
    FusionCollective coordinates multiple fusion pipelines.

Topologies:
    PARALLEL:     All inputs → single fusion → output
    SEQUENTIAL:   Inputs fused pairwise in sequence
    HIERARCHICAL: Tree-structured fusion
    BLEND:        Two pathways with β-controlled mixing

Controllers:
    Alpha (α): Neighbor bleed-over / interpolation strength
    Beta (β):  Balance between two pathways (geometric vs learned)
    Gamma (γ): Per-stage or per-scale importance weights

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, List, Dict, Tuple, Union, Literal, Any
from dataclasses import dataclass, field
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.base_router import BaseRouter
from geofractal.router.wide_router import WideRouter
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.components.fusion_component import (
    FusionComponent,
    AdaptiveFusion,
    GatedFusion,
    AttentionFusion,
    ConcatFusion,
    SumFusion,
    BilinearFusion,
    GeometricAttentionGate,
    CantorScaleFusion,
    HierarchicalTreeGating,
    AdaptiveBindingFusion,
    InceptiveFusion,
)


# =============================================================================
# ENUMS AND CONFIGS
# =============================================================================

class FusionTopology(Enum):
    """How inputs flow through fusion stages."""
    PARALLEL = "parallel"           # All inputs → single fusion
    SEQUENTIAL = "sequential"       # Pairwise fusion in sequence
    HIERARCHICAL = "hierarchical"   # Tree-structured fusion
    BLEND = "blend"                 # Two pathways with β mixing


class FusionStrategy(Enum):
    """Available fusion algorithms."""
    ADAPTIVE = "adaptive"
    GATED = "gated"
    ATTENTION = "attention"
    CONCAT = "concat"
    SUM = "sum"
    BILINEAR = "bilinear"
    GEOMETRIC = "geometric"
    CANTOR = "cantor"
    TREE = "tree"
    BINDING = "binding"  # Lyra's AdaptiveBindingFusion
    INCEPTIVE = "inceptive"  # Auxiliary feature injection


@dataclass
class ControllerConfig:
    """Configuration for fusion controllers."""

    # Alpha: neighbor bleed-over / interpolation
    use_alpha: bool = False
    alpha_init: float = 0.1
    alpha_learnable: bool = True
    alpha_per_stage: bool = False
    alpha_min: float = 0.0
    alpha_max: float = 0.5

    # Beta: pathway balance (geometric vs learned)
    use_beta: bool = False
    beta_init: float = 0.5
    beta_learnable: bool = True
    beta_per_stage: bool = False
    beta_min: float = 0.0
    beta_max: float = 1.0

    # Gamma: stage/input importance weights
    use_gamma: bool = False
    gamma_learnable: bool = True
    gamma_temperature: float = 1.0

    # Binding (for AdaptiveBindingFusion / Lyra pattern)
    # binding_config: Dict mapping input_idx -> list of allowed input_idxs
    # binding_pairs: List of (source_idx, target_idx) pairs to boost
    binding_config: Optional[Dict[int, List[int]]] = None
    binding_pairs: Optional[List[Tuple[int, int]]] = None
    visibility_alpha_init: float = 1.0  # Lyra's alpha (per-input visibility)
    binding_beta_init: float = 0.3  # Lyra's beta (per-pair boost)

    # Inceptive (for InceptiveFusion / auxiliary feature injection)
    aux_features: int = 0  # Dimension of auxiliary features (0 = not used)


@dataclass
class FusionStageSpec:
    """Specification for a single fusion stage."""
    strategy: Union[str, FusionStrategy]
    num_inputs: Optional[int] = None  # None = inherit from previous
    in_features: Optional[int] = None  # None = inherit
    out_features: Optional[int] = None  # None = same as in_features
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.strategy, str):
            self.strategy = FusionStrategy(self.strategy)


@dataclass
class FusionConfig:
    """Complete configuration for ConfigurableFusion."""

    # Core dimensions
    num_inputs: int
    in_features: int
    out_features: Optional[int] = None

    # Topology
    topology: Union[str, FusionTopology] = FusionTopology.PARALLEL

    # Stages (for multi-stage fusion)
    stages: List[FusionStageSpec] = field(default_factory=list)

    # Single-stage shortcut
    strategy: Optional[Union[str, FusionStrategy]] = None
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # Controllers
    controllers: ControllerConfig = field(default_factory=ControllerConfig)

    # Options
    expose_intermediates: bool = False
    use_residual: bool = False
    dropout: float = 0.0

    def __post_init__(self):
        if isinstance(self.topology, str):
            self.topology = FusionTopology(self.topology)
        if isinstance(self.strategy, str):
            self.strategy = FusionStrategy(self.strategy)
        if self.out_features is None:
            self.out_features = self.in_features

        # Convert single strategy to stages list
        if self.strategy is not None and len(self.stages) == 0:
            self.stages = [FusionStageSpec(
                strategy=self.strategy,
                num_inputs=self.num_inputs,
                in_features=self.in_features,
                out_features=self.out_features,
                params=self.strategy_params
            )]


# =============================================================================
# CONTROLLERS
# =============================================================================

class AlphaController(TorchComponent):
    """
    Controls neighbor bleed-over / interpolation strength.

    Used for smooth transitions between discrete attention targets,
    allowing gradient flow to adjacent positions.

    α = 0: Hard attention to exact positions
    α = 0.5: 50% bleed to neighbors
    """

    def __init__(
        self,
        name: str,
        num_stages: int = 1,
        config: Optional[ControllerConfig] = None,
    ):
        super().__init__(name)
        config = config or ControllerConfig(use_alpha=True)

        self.num_stages = num_stages
        self.config = config
        self.per_stage = config.alpha_per_stage

        shape = (num_stages,) if self.per_stage else (1,)
        alpha_init = torch.full(shape, config.alpha_init)

        if config.alpha_learnable:
            self.alpha_raw = nn.Parameter(alpha_init)
        else:
            self.register_buffer('alpha_raw', alpha_init)

    def forward(self, stage_idx: Optional[int] = None) -> Tensor:
        """Get alpha value, optionally for specific stage."""
        # Sigmoid to [0, 1], then scale to [min, max]
        alpha = torch.sigmoid(self.alpha_raw)
        alpha = self.config.alpha_min + alpha * (self.config.alpha_max - self.config.alpha_min)

        if self.per_stage and stage_idx is not None:
            return alpha[stage_idx]
        return alpha.squeeze() if alpha.numel() == 1 else alpha

    def get_all(self) -> Tensor:
        """Get all alpha values."""
        alpha = torch.sigmoid(self.alpha_raw)
        return self.config.alpha_min + alpha * (self.config.alpha_max - self.config.alpha_min)


class BetaController(TorchComponent):
    """
    Controls balance between two pathways.

    Typically: geometric features vs learned projection

    β = 0: Fully learned pathway
    β = 1: Fully geometric pathway
    β = 0.5: Equal blend
    """

    def __init__(
        self,
        name: str,
        num_stages: int = 1,
        config: Optional[ControllerConfig] = None,
    ):
        super().__init__(name)
        config = config or ControllerConfig(use_beta=True)

        self.num_stages = num_stages
        self.config = config
        self.per_stage = config.beta_per_stage

        shape = (num_stages,) if self.per_stage else (1,)
        beta_init = torch.full(shape, config.beta_init)

        if config.beta_learnable:
            self.beta_raw = nn.Parameter(beta_init)
        else:
            self.register_buffer('beta_raw', beta_init)

    def forward(self, stage_idx: Optional[int] = None) -> Tensor:
        """Get beta value, optionally for specific stage."""
        beta = torch.sigmoid(self.beta_raw)
        beta = self.config.beta_min + beta * (self.config.beta_max - self.config.beta_min)

        if self.per_stage and stage_idx is not None:
            return beta[stage_idx]
        return beta.squeeze() if beta.numel() == 1 else beta

    def get_all(self) -> Tensor:
        """Get all beta values."""
        beta = torch.sigmoid(self.beta_raw)
        return self.config.beta_min + beta * (self.config.beta_max - self.config.beta_min)

    def blend(self, pathway_a: Tensor, pathway_b: Tensor, stage_idx: Optional[int] = None) -> Tensor:
        """Blend two pathways using beta."""
        beta = self.forward(stage_idx)
        return pathway_a * (1 - beta) + pathway_b * beta


class GammaController(TorchComponent):
    """
    Controls per-input or per-stage importance weights.

    Produces softmax-normalized weights that sum to 1.
    Can be used for:
    - Weighting expert opinions
    - Weighting scale outputs
    - Weighting fusion stages
    """

    def __init__(
        self,
        name: str,
        num_weights: int,
        config: Optional[ControllerConfig] = None,
    ):
        super().__init__(name)
        config = config or ControllerConfig(use_gamma=True)

        self.num_weights = num_weights
        self.config = config

        # Initialize to uniform
        gamma_init = torch.ones(num_weights) / num_weights

        if config.gamma_learnable:
            self.gamma_raw = nn.Parameter(gamma_init)
        else:
            self.register_buffer('gamma_raw', gamma_init)

    def forward(self) -> Tensor:
        """Get normalized weights."""
        return F.softmax(self.gamma_raw / self.config.gamma_temperature, dim=0)

    def weighted_sum(self, inputs: List[Tensor]) -> Tensor:
        """Compute weighted sum of inputs."""
        weights = self.forward()
        stacked = torch.stack(inputs, dim=0)

        # Reshape weights for broadcasting
        w_shape = [self.num_weights] + [1] * (stacked.dim() - 1)
        return (stacked * weights.view(*w_shape)).sum(dim=0)


class FusionControllerBundle(TorchComponent):
    """
    Bundle of all controllers for convenient access.

    Provides unified interface to α/β/γ controllers.
    """

    def __init__(
        self,
        name: str,
        num_stages: int,
        num_inputs: int,
        config: ControllerConfig,
    ):
        super().__init__(name)

        self.config = config
        self.num_stages = num_stages
        self.num_inputs = num_inputs

        # Create controllers based on config
        if config.use_alpha:
            self.alpha = AlphaController(f"{name}_alpha", num_stages, config)
        else:
            self.alpha = None

        if config.use_beta:
            self.beta = BetaController(f"{name}_beta", num_stages, config)
        else:
            self.beta = None

        if config.use_gamma:
            self.gamma = GammaController(f"{name}_gamma", num_inputs, config)
        else:
            self.gamma = None

    def get_alpha(self, stage_idx: Optional[int] = None) -> Optional[Tensor]:
        """Get alpha value if controller exists."""
        if self.alpha is not None:
            return self.alpha(stage_idx)
        return None

    def get_beta(self, stage_idx: Optional[int] = None) -> Optional[Tensor]:
        """Get beta value if controller exists."""
        if self.beta is not None:
            return self.beta(stage_idx)
        return None

    def get_gamma(self) -> Optional[Tensor]:
        """Get gamma weights if controller exists."""
        if self.gamma is not None:
            return self.gamma()
        return None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get all controller values for diagnostics."""
        diag = {}

        if self.alpha is not None:
            diag['alpha'] = self.alpha.get_all().detach().cpu().tolist()
        if self.beta is not None:
            diag['beta'] = self.beta.get_all().detach().cpu().tolist()
        if self.gamma is not None:
            diag['gamma'] = self.gamma().detach().cpu().tolist()

        return diag

    def forward(self) -> Dict[str, Any]:
        """Return all controller values."""
        return self.get_diagnostics()


# =============================================================================
# FUSION BUILDER
# =============================================================================

class FusionBuilder:
    """
    Factory for creating fusion components.

    Provides:
    - build(): Create from FusionStageSpec
    - from_config(): Create full ConfigurableFusion from FusionConfig
    - preset(): Create from named preset
    """

    # Strategy registry
    STRATEGIES = {
        FusionStrategy.ADAPTIVE: AdaptiveFusion,
        FusionStrategy.GATED: GatedFusion,
        FusionStrategy.ATTENTION: AttentionFusion,
        FusionStrategy.CONCAT: ConcatFusion,
        FusionStrategy.SUM: SumFusion,
        FusionStrategy.BILINEAR: BilinearFusion,
        FusionStrategy.GEOMETRIC: GeometricAttentionGate,
        FusionStrategy.CANTOR: CantorScaleFusion,
        FusionStrategy.TREE: HierarchicalTreeGating,
        FusionStrategy.BINDING: AdaptiveBindingFusion,
        FusionStrategy.INCEPTIVE: InceptiveFusion,
    }

    # Presets for common patterns
    PRESETS = {
        'simple': {
            'strategy': FusionStrategy.ADAPTIVE,
            'topology': FusionTopology.PARALLEL,
        },
        'geometric': {
            'strategy': FusionStrategy.GEOMETRIC,
            'topology': FusionTopology.PARALLEL,
            'controllers': ControllerConfig(use_beta=True, beta_init=0.5),
        },
        'cantor': {
            'strategy': FusionStrategy.CANTOR,
            'topology': FusionTopology.PARALLEL,
            'controllers': ControllerConfig(use_alpha=True, alpha_init=0.1),
        },
        'hierarchical': {
            'strategy': FusionStrategy.TREE,
            'topology': FusionTopology.PARALLEL,
            'controllers': ControllerConfig(use_gamma=True),
        },
        'david': {
            'strategy': FusionStrategy.GEOMETRIC,
            'topology': FusionTopology.BLEND,
            'controllers': ControllerConfig(
                use_alpha=True, alpha_init=0.1,
                use_beta=True, beta_init=0.7,
                use_gamma=True
            ),
        },
        'liminal': {
            'strategy': FusionStrategy.ADAPTIVE,
            'topology': FusionTopology.HIERARCHICAL,
            'controllers': ControllerConfig(
                use_alpha=True, alpha_init=0.1, alpha_per_stage=True,
                use_beta=True, beta_init=0.5, beta_per_stage=True,
                use_gamma=True
            ),
        },
        'lyra': {
            'strategy': FusionStrategy.BINDING,
            'topology': FusionTopology.PARALLEL,
            'controllers': ControllerConfig(
                visibility_alpha_init=1.0,
                binding_beta_init=0.3,
            ),
        },
        'inceptive': {
            'strategy': FusionStrategy.INCEPTIVE,
            'topology': FusionTopology.PARALLEL,
            'controllers': ControllerConfig(
                aux_features=40,  # Default: 20 levels * 2 features
            ),
        },
    }

    @classmethod
    def build_stage(
        cls,
        name: str,
        spec: FusionStageSpec,
        num_inputs: int,
        in_features: int,
    ) -> FusionComponent:
        """Build a single fusion stage from spec."""

        strategy = spec.strategy
        if isinstance(strategy, str):
            strategy = FusionStrategy(strategy)

        fusion_cls = cls.STRATEGIES.get(strategy)
        if fusion_cls is None:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Determine dimensions
        n_in = spec.num_inputs or num_inputs
        d_in = spec.in_features or in_features
        d_out = spec.out_features or d_in

        # Build with strategy-specific params
        params = spec.params.copy()

        # Bilinear is special - only takes 2 inputs, different signature
        if strategy == FusionStrategy.BILINEAR:
            return fusion_cls(name, in_features=d_in, out_features=d_out, **params)

        # Binding (Lyra's AdaptiveBindingFusion) has special signature
        if strategy == FusionStrategy.BINDING:
            return fusion_cls(
                name,
                num_inputs=n_in,
                in_features=d_in,
                out_features=d_out,
                binding_config=params.pop('binding_config', None),
                binding_pairs=params.pop('binding_pairs', None),
                alpha_init=params.pop('alpha_init', 1.0),
                beta_init=params.pop('beta_init', 0.3),
                **params
            )

        # Inceptive (auxiliary feature injection) requires aux_features
        if strategy == FusionStrategy.INCEPTIVE:
            aux_features = params.pop('aux_features', 40)  # Default: 20 levels * 2
            return fusion_cls(
                name,
                num_inputs=n_in,
                in_features=d_in,
                aux_features=aux_features,
                out_features=d_out,
                **params
            )

        # All other strategies have uniform signature
        return fusion_cls(name, num_inputs=n_in, in_features=d_in, out_features=d_out, **params)

    @classmethod
    def from_config(cls, name: str, config: FusionConfig) -> 'ConfigurableFusion':
        """Build ConfigurableFusion from config."""
        return ConfigurableFusion(name, config)

    @classmethod
    def preset(
        cls,
        name: str,
        preset_name: str,
        num_inputs: int,
        in_features: int,
        out_features: Optional[int] = None,
        **overrides
    ) -> 'ConfigurableFusion':
        """Build from named preset with optional overrides."""

        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(cls.PRESETS.keys())}")

        preset = cls.PRESETS[preset_name].copy()

        # Apply overrides
        for key, value in overrides.items():
            if key == 'controllers' and isinstance(value, dict):
                # Merge controller config
                base_ctrl = preset.get('controllers', ControllerConfig())
                if isinstance(base_ctrl, ControllerConfig):
                    for k, v in value.items():
                        setattr(base_ctrl, k, v)
                    preset['controllers'] = base_ctrl
            else:
                preset[key] = value

        # Build config
        config = FusionConfig(
            num_inputs=num_inputs,
            in_features=in_features,
            out_features=out_features,
            **preset
        )

        return cls.from_config(name, config)

    @classmethod
    def simple(
        cls,
        name: str,
        num_inputs: int,
        in_features: int,
        strategy: Union[str, FusionStrategy] = FusionStrategy.ADAPTIVE,
        **params
    ) -> FusionComponent:
        """Quick builder for single-strategy fusion."""

        spec = FusionStageSpec(
            strategy=strategy,
            num_inputs=num_inputs,
            in_features=in_features,
            params=params
        )

        return cls.build_stage(name, spec, num_inputs, in_features)


# =============================================================================
# CONFIGURABLE FUSION
# =============================================================================

class ConfigurableFusion(BaseRouter):
    """
    Multi-stage configurable fusion with controllers.

    Like ConfigurableTower but for fusion operations.
    Supports multiple topologies and α/β/γ control.

    Topologies:
        PARALLEL: All inputs → single fusion → output
        SEQUENTIAL: Progressive pairwise fusion
        HIERARCHICAL: Tree-structured fusion
        BLEND: Two pathways with β mixing

    Example:
        config = FusionConfig(
            num_inputs=5,
            in_features=256,
            strategy=FusionStrategy.GEOMETRIC,
            controllers=ControllerConfig(use_beta=True)
        )
        fusion = ConfigurableFusion('my_fusion', config)
        output = fusion(x1, x2, x3, x4, x5)
    """

    def __init__(self, name: str, config: FusionConfig):
        super().__init__(name, strict=False)

        self.config = config
        self.topology = config.topology
        self.num_inputs = config.num_inputs
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.expose_intermediates = config.expose_intermediates

        # Build stages
        self._build_stages()

        # Build controllers
        if any([config.controllers.use_alpha, config.controllers.use_beta, config.controllers.use_gamma]):
            self.controllers = FusionControllerBundle(
                f"{name}_controllers",
                num_stages=len(self.stages),
                num_inputs=config.num_inputs,
                config=config.controllers
            )
            self.attach('controllers', self.controllers)
        else:
            self.controllers = None

        # Optional components
        if config.use_residual:
            self.residual_proj = nn.Linear(config.in_features, config.out_features)
            self.attach('residual_proj', self.residual_proj)

        if config.dropout > 0:
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.dropout = None

    def _build_stages(self):
        """Build fusion stages based on topology."""
        self.stages = nn.ModuleList()

        if self.topology == FusionTopology.PARALLEL:
            self._build_parallel_stages()
        elif self.topology == FusionTopology.SEQUENTIAL:
            self._build_sequential_stages()
        elif self.topology == FusionTopology.HIERARCHICAL:
            self._build_hierarchical_stages()
        elif self.topology == FusionTopology.BLEND:
            self._build_blend_stages()

    def _build_parallel_stages(self):
        """All inputs → single fusion."""
        for i, spec in enumerate(self.config.stages):
            stage = FusionBuilder.build_stage(
                f"{self.name}_stage_{i}",
                spec,
                self.num_inputs,
                self.in_features
            )
            self.stages.append(stage)

    def _build_sequential_stages(self):
        """Progressive pairwise fusion."""
        # First stage fuses first 2 inputs
        # Each subsequent stage fuses result with next input
        current_dim = self.in_features

        for i in range(self.num_inputs - 1):
            spec = self.config.stages[i] if i < len(self.config.stages) else self.config.stages[-1]

            # Override to 2 inputs for sequential
            stage = FusionBuilder.build_stage(
                f"{self.name}_stage_{i}",
                FusionStageSpec(
                    strategy=spec.strategy,
                    num_inputs=2,
                    in_features=current_dim,
                    out_features=spec.out_features or current_dim,
                    params=spec.params
                ),
                2,
                current_dim
            )
            self.stages.append(stage)
            current_dim = spec.out_features or current_dim

    def _build_hierarchical_stages(self):
        """Tree-structured fusion."""
        # Binary tree: pairs at each level, then combine
        num_levels = math.ceil(math.log2(self.num_inputs))
        current_dim = self.in_features

        for level in range(num_levels):
            spec = self.config.stages[level] if level < len(self.config.stages) else self.config.stages[-1]

            stage = FusionBuilder.build_stage(
                f"{self.name}_level_{level}",
                FusionStageSpec(
                    strategy=spec.strategy,
                    num_inputs=2,
                    in_features=current_dim,
                    out_features=spec.out_features or current_dim,
                    params=spec.params
                ),
                2,
                current_dim
            )
            self.stages.append(stage)
            current_dim = spec.out_features or current_dim

    def _build_blend_stages(self):
        """Two pathways with β mixing."""
        # Need exactly 2 stages for blend
        if len(self.config.stages) < 2:
            # Default: adaptive + geometric
            self.config.stages = [
                FusionStageSpec(strategy=FusionStrategy.ADAPTIVE),
                FusionStageSpec(strategy=FusionStrategy.GEOMETRIC),
            ]

        for i, spec in enumerate(self.config.stages[:2]):
            stage = FusionBuilder.build_stage(
                f"{self.name}_pathway_{i}",
                spec,
                self.num_inputs,
                self.in_features
            )
            self.stages.append(stage)

    def forward(self, *inputs: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through fusion.

        Args:
            *inputs: Input tensors to fuse

        Returns:
            Fused output tensor, or dict with intermediates if expose_intermediates=True
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")

        # Route based on topology
        if self.topology == FusionTopology.PARALLEL:
            output, intermediates = self._forward_parallel(inputs)
        elif self.topology == FusionTopology.SEQUENTIAL:
            output, intermediates = self._forward_sequential(inputs)
        elif self.topology == FusionTopology.HIERARCHICAL:
            output, intermediates = self._forward_hierarchical(inputs)
        elif self.topology == FusionTopology.BLEND:
            output, intermediates = self._forward_blend(inputs)

        # Apply residual if configured
        if self.config.use_residual:
            # Use first input as residual source
            residual = self.residual_proj(inputs[0])
            output = output + residual

        # Apply dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # Cache intermediates if exposing
        if self.expose_intermediates:
            self.cache_set('intermediates', intermediates)
            return {
                'output': output,
                'intermediates': intermediates,
                'diagnostics': self.controllers.get_diagnostics() if self.controllers else {}
            }

        return output

    def _forward_parallel(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, List[Tensor]]:
        """All inputs → fusion stages → combine."""
        intermediates = []

        if len(self.stages) == 1:
            # Single stage
            output = self.stages[0](*inputs)
            intermediates.append(output)
        else:
            # Multiple stages, use gamma weights to combine
            stage_outputs = []
            for i, stage in enumerate(self.stages):
                out = stage(*inputs)
                stage_outputs.append(out)
                intermediates.append(out)

            # Combine with gamma if available
            if self.controllers and self.controllers.gamma is not None:
                output = self.controllers.gamma.weighted_sum(stage_outputs)
            else:
                # Simple mean
                output = torch.stack(stage_outputs, dim=0).mean(dim=0)

        return output, intermediates

    def _forward_sequential(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, List[Tensor]]:
        """Progressive pairwise fusion."""
        intermediates = []

        # Start with first input
        current = inputs[0]

        for i, stage in enumerate(self.stages):
            # Get alpha for bleed-over if available
            alpha = self.controllers.get_alpha(i) if self.controllers else None

            # Fuse current with next input
            next_input = inputs[i + 1]

            # If stage supports alpha, pass it
            if alpha is not None and hasattr(stage, 'alpha'):
                stage.alpha = alpha

            current = stage(current, next_input)
            intermediates.append(current)

        return current, intermediates

    def _forward_hierarchical(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, List[Tensor]]:
        """Tree-structured fusion."""
        intermediates = []

        # Pad inputs to power of 2
        input_list = list(inputs)
        while len(input_list) & (len(input_list) - 1):  # Not power of 2
            input_list.append(input_list[-1])  # Duplicate last

        current_level = input_list
        stage_idx = 0

        while len(current_level) > 1:
            next_level = []

            for i in range(0, len(current_level), 2):
                a = current_level[i]
                b = current_level[i + 1] if i + 1 < len(current_level) else a

                stage = self.stages[min(stage_idx, len(self.stages) - 1)]
                fused = stage(a, b)
                next_level.append(fused)
                intermediates.append(fused)

            current_level = next_level
            stage_idx += 1

        return current_level[0], intermediates

    def _forward_blend(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, List[Tensor]]:
        """Two pathways with β mixing."""
        intermediates = []

        # Pathway A (typically learned)
        pathway_a = self.stages[0](*inputs)
        intermediates.append(pathway_a)

        # Pathway B (typically geometric)
        pathway_b = self.stages[1](*inputs)
        intermediates.append(pathway_b)

        # Blend with beta
        if self.controllers and self.controllers.beta is not None:
            output = self.controllers.beta.blend(pathway_a, pathway_b)
        else:
            output = (pathway_a + pathway_b) / 2

        return output, intermediates

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get fusion diagnostics."""
        diag = {
            'topology': self.topology.value,
            'num_stages': len(self.stages),
            'num_inputs': self.num_inputs,
        }

        if self.controllers:
            diag['controllers'] = self.controllers.get_diagnostics()

        return diag

    def __repr__(self) -> str:
        return (
            f"ConfigurableFusion(name='{self.name}', "
            f"topology={self.topology.value}, "
            f"inputs={self.num_inputs}, "
            f"stages={len(self.stages)}, "
            f"features={self.in_features}->{self.out_features})"
        )


# =============================================================================
# FUSION COLLECTIVE
# =============================================================================

class FusionCollective(WideRouter):
    """
    Collective that coordinates multiple fusion pipelines.

    Like WideRouter but specialized for fusion:
    - Each "tower" is a ConfigurableFusion
    - Final output fuses all pipeline outputs

    Use when you need:
    - Multiple fusion strategies in parallel
    - Different fusions for different input subsets
    - Ensemble of fusion approaches
    """

    def __init__(
        self,
        name: str,
        num_inputs: int,
        in_features: int,
        out_features: Optional[int] = None,
        final_fusion: Union[str, FusionStrategy] = FusionStrategy.ADAPTIVE,
    ):
        super().__init__(name, auto_discover=True)

        self.num_inputs = num_inputs
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.final_fusion_strategy = final_fusion

        # Will hold ConfigurableFusion instances
        self.fusion_pipelines: Dict[str, ConfigurableFusion] = {}

        # Final fusion created after pipelines are added
        self._final_fusion = None

    def add_pipeline(
        self,
        pipeline_name: str,
        config: FusionConfig,
    ) -> ConfigurableFusion:
        """Add a fusion pipeline."""
        pipeline = ConfigurableFusion(f"{self.name}_{pipeline_name}", config)
        self.attach(pipeline_name, pipeline)
        self.fusion_pipelines[pipeline_name] = pipeline
        return pipeline

    def add_preset(
        self,
        pipeline_name: str,
        preset_name: str,
        **overrides
    ) -> ConfigurableFusion:
        """Add pipeline from preset."""
        pipeline = FusionBuilder.preset(
            f"{self.name}_{pipeline_name}",
            preset_name,
            self.num_inputs,
            self.in_features,
            self.out_features,
            **overrides
        )
        self.attach(pipeline_name, pipeline)
        self.fusion_pipelines[pipeline_name] = pipeline
        return pipeline

    def finalize(self):
        """Finalize collective after adding all pipelines."""
        self.discover_towers()

        num_pipelines = len(self.fusion_pipelines)
        if num_pipelines > 1:
            self._final_fusion = FusionBuilder.simple(
                f"{self.name}_final",
                num_pipelines,
                self.out_features,
                self.final_fusion_strategy
            )
            self.attach('final_fusion', self._final_fusion)

    def forward(self, *inputs: Tensor) -> Tensor:
        """
        Forward through all pipelines then fuse.

        Args:
            *inputs: Input tensors

        Returns:
            Final fused output
        """
        # Run through all pipelines
        pipeline_outputs = []
        for name, pipeline in self.fusion_pipelines.items():
            out = pipeline(*inputs)
            if isinstance(out, dict):
                out = out['output']
            pipeline_outputs.append(out)

        # Final fusion
        if self._final_fusion is not None and len(pipeline_outputs) > 1:
            return self._final_fusion(*pipeline_outputs)
        elif len(pipeline_outputs) == 1:
            return pipeline_outputs[0]
        else:
            return torch.stack(pipeline_outputs, dim=0).mean(dim=0)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics from all pipelines."""
        return {
            name: pipeline.get_diagnostics()
            for name, pipeline in self.fusion_pipelines.items()
        }


# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':

    def test_section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test inputs
    B, D = 4, 256
    inputs = [torch.randn(B, D) for _ in range(5)]

    # -------------------------------------------------------------------------
    test_section("CONTROLLERS")
    # -------------------------------------------------------------------------

    alpha = AlphaController('test_alpha', num_stages=3, config=ControllerConfig(
        use_alpha=True, alpha_init=0.1, alpha_per_stage=True
    ))
    print(f"Alpha controller: {alpha.get_all()}")

    beta = BetaController('test_beta', num_stages=2, config=ControllerConfig(
        use_beta=True, beta_init=0.5
    ))
    print(f"Beta controller: {beta.get_all()}")

    gamma = GammaController('test_gamma', num_weights=5, config=ControllerConfig(
        use_gamma=True
    ))
    print(f"Gamma weights: {gamma()}")

    # -------------------------------------------------------------------------
    test_section("FUSION BUILDER - SIMPLE")
    # -------------------------------------------------------------------------

    fusion = FusionBuilder.simple('simple_adaptive', 5, 256, FusionStrategy.ADAPTIVE)
    print(f"Built: {fusion}")
    output = fusion(*inputs)
    print(f"Output: {output.shape}")

    # -------------------------------------------------------------------------
    test_section("FUSION BUILDER - PRESET: SIMPLE")
    # -------------------------------------------------------------------------

    fusion = FusionBuilder.preset('preset_simple', 'simple', 5, 256)
    print(f"Built: {fusion}")
    output = fusion(*inputs)
    print(f"Output: {output.shape}")

    # -------------------------------------------------------------------------
    test_section("FUSION BUILDER - PRESET: GEOMETRIC")
    # -------------------------------------------------------------------------

    fusion = FusionBuilder.preset('preset_geometric', 'geometric', 5, 256)
    print(f"Built: {fusion}")
    output = fusion(*inputs)
    print(f"Output: {output.shape}")
    print(f"Diagnostics: {fusion.get_diagnostics()}")

    # -------------------------------------------------------------------------
    test_section("FUSION BUILDER - PRESET: DAVID")
    # -------------------------------------------------------------------------

    fusion = FusionBuilder.preset('preset_david', 'david', 5, 256)
    print(f"Built: {fusion}")
    result = fusion(*inputs)
    if isinstance(result, dict):
        print(f"Output: {result['output'].shape}")
        print(f"Diagnostics: {result['diagnostics']}")

    # -------------------------------------------------------------------------
    test_section("CONFIGURABLE FUSION - SEQUENTIAL")
    # -------------------------------------------------------------------------

    config = FusionConfig(
        num_inputs=5,
        in_features=256,
        topology=FusionTopology.SEQUENTIAL,
        strategy=FusionStrategy.ADAPTIVE,
        controllers=ControllerConfig(use_alpha=True)
    )
    fusion = ConfigurableFusion('sequential', config)
    print(f"Built: {fusion}")
    output = fusion(*inputs)
    print(f"Output: {output.shape}")

    # -------------------------------------------------------------------------
    test_section("CONFIGURABLE FUSION - HIERARCHICAL")
    # -------------------------------------------------------------------------

    config = FusionConfig(
        num_inputs=5,
        in_features=256,
        topology=FusionTopology.HIERARCHICAL,
        strategy=FusionStrategy.GATED,
    )
    fusion = ConfigurableFusion('hierarchical', config)
    print(f"Built: {fusion}")
    output = fusion(*inputs)
    print(f"Output: {output.shape}")

    # -------------------------------------------------------------------------
    test_section("CONFIGURABLE FUSION - BLEND")
    # -------------------------------------------------------------------------

    config = FusionConfig(
        num_inputs=5,
        in_features=256,
        topology=FusionTopology.BLEND,
        stages=[
            FusionStageSpec(strategy=FusionStrategy.ADAPTIVE),
            FusionStageSpec(strategy=FusionStrategy.GEOMETRIC),
        ],
        controllers=ControllerConfig(use_beta=True, beta_init=0.7),
        expose_intermediates=True,
    )
    fusion = ConfigurableFusion('blend', config)
    print(f"Built: {fusion}")
    result = fusion(*inputs)
    print(f"Output: {result['output'].shape}")
    print(f"Intermediates: {len(result['intermediates'])}")
    print(f"Diagnostics: {result['diagnostics']}")

    # -------------------------------------------------------------------------
    test_section("FUSION COLLECTIVE")
    # -------------------------------------------------------------------------

    collective = FusionCollective('my_collective', num_inputs=5, in_features=256)
    collective.add_preset('adaptive', 'simple')
    collective.add_preset('geometric', 'geometric')
    collective.add_preset('cantor', 'cantor')
    collective.finalize()

    print(f"Collective pipelines: {list(collective.fusion_pipelines.keys())}")
    output = collective(*inputs)
    print(f"Output: {output.shape}")
    print(f"Diagnostics: {collective.get_diagnostics()}")

    # -------------------------------------------------------------------------
    test_section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nFusion Builder System Ready!")
    print("\nAvailable presets:")
    for name, preset in FusionBuilder.PRESETS.items():
        print(f"  - {name}: {preset.get('strategy', 'multi').value if hasattr(preset.get('strategy'), 'value') else preset.get('strategy', 'multi')}")

    print("\nTopologies:")
    for topo in FusionTopology:
        print(f"  - {topo.value}")