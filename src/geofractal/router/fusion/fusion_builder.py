"""
geofractal.router.fusion.builder
================================
Builder pattern and factory for fusion layers.

Provides easy composition of fusion strategies with
sensible defaults and customization options.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
from typing import Type, Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum, auto

from .fusion_protocols import BaseFusion, FusionInfo
from .fusion_methods import (
    FusionConfig,
    ConcatFusion,
    WeightedFusion,
    GatedFusion,
    AttentionFusion,
    FingerprintGuidedFusion,
    ResidualFusion,
    MoEFusion,
    HierarchicalTreeFusion,
)


# =============================================================================
# FUSION STRATEGY ENUM
# =============================================================================

class FusionStrategy(Enum):
    """Available fusion strategies."""
    CONCAT = auto()  # Simple concatenation (baseline)
    WEIGHTED = auto()  # Learnable static weights
    GATED = auto()  # Input-adaptive gating
    ATTENTION = auto()  # Cross-stream attention
    FINGERPRINT = auto()  # Fingerprint-guided
    RESIDUAL = auto()  # Additive with residuals
    MOE = auto()  # Mixture of experts
    HIERARCHICAL = auto()  # Tree-structured


STRATEGY_TO_CLASS: Dict[FusionStrategy, Type[BaseFusion]] = {
    FusionStrategy.CONCAT: ConcatFusion,
    FusionStrategy.WEIGHTED: WeightedFusion,
    FusionStrategy.GATED: GatedFusion,
    FusionStrategy.ATTENTION: AttentionFusion,
    FusionStrategy.FINGERPRINT: FingerprintGuidedFusion,
    FusionStrategy.RESIDUAL: ResidualFusion,
    FusionStrategy.MOE: MoEFusion,
    FusionStrategy.HIERARCHICAL: HierarchicalTreeFusion,
}


# =============================================================================
# FUSION PRESETS
# =============================================================================

@dataclass
class FusionPreset:
    """Preset configuration for fusion strategies."""
    strategy: FusionStrategy
    config: FusionConfig = field(default_factory=FusionConfig)
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


# Default preset (proven on ImageNet)
STANDARD_FUSION = FusionPreset(
    strategy=FusionStrategy.CONCAT,
    config=FusionConfig(output_dim=512, dropout=0.1, expansion=2),
    description="Simple concatenation fusion - proven on ImageNet (84.68%)",
)

# Adaptive preset (adjusts per-sample)
ADAPTIVE_FUSION = FusionPreset(
    strategy=FusionStrategy.GATED,
    config=FusionConfig(output_dim=512, dropout=0.1, temperature=1.0),
    description="Input-adaptive gated fusion",
)

# Attention preset (cross-stream communication)
ATTENTION_FUSION = FusionPreset(
    strategy=FusionStrategy.ATTENTION,
    config=FusionConfig(output_dim=512, num_heads=8, dropout=0.1),
    description="Multi-head attention across streams",
)

# Fingerprint preset (uses identity for fusion)
FINGERPRINT_FUSION = FusionPreset(
    strategy=FusionStrategy.FINGERPRINT,
    config=FusionConfig(output_dim=512),
    extra_kwargs={'fingerprint_dim': 64},
    description="Fingerprint-guided fusion using stream identities",
)

# MoE preset (sparse expert selection)
MOE_FUSION = FusionPreset(
    strategy=FusionStrategy.MOE,
    config=FusionConfig(output_dim=512, dropout=0.1),
    extra_kwargs={'num_experts': 4, 'top_k': 2},
    description="Mixture of experts with sparse activation",
)

# Hierarchical preset (tree-structured)
HIERARCHICAL_FUSION = FusionPreset(
    strategy=FusionStrategy.HIERARCHICAL,
    config=FusionConfig(output_dim=512, dropout=0.1),
    description="Tree-structured progressive fusion",
)


# =============================================================================
# FUSION BUILDER
# =============================================================================

class FusionBuilder:
    """
    Builder for constructing fusion layers.

    Supports:
    - Strategy selection by enum or preset
    - Custom configuration
    - Static injection of custom fusion
    - Validation before build

    Example:
        fusion = (FusionBuilder()
            .with_streams({'clip_b32': 512, 'clip_l14': 768})
            .with_output_dim(512)
            .with_strategy(FusionStrategy.ATTENTION)
            .build())
    """

    def __init__(self):
        self._stream_dims: Optional[Dict[str, int]] = None
        self._output_dim: int = 512
        self._strategy: FusionStrategy = FusionStrategy.CONCAT
        self._config: Optional[FusionConfig] = None
        self._extra_kwargs: Dict[str, Any] = {}
        self._custom_fusion: Optional[BaseFusion] = None

    # -------------------------------------------------------------------------
    # Fluent setters
    # -------------------------------------------------------------------------

    def with_streams(self, stream_dims: Dict[str, int]) -> 'FusionBuilder':
        """Set stream dimensions."""
        self._stream_dims = stream_dims
        return self

    def with_output_dim(self, dim: int) -> 'FusionBuilder':
        """Set output dimension."""
        self._output_dim = dim
        return self

    def with_strategy(self, strategy: FusionStrategy) -> 'FusionBuilder':
        """Set fusion strategy."""
        self._strategy = strategy
        return self

    def with_config(self, config: FusionConfig) -> 'FusionBuilder':
        """Set fusion configuration."""
        self._config = config
        return self

    def with_extra_kwargs(self, **kwargs) -> 'FusionBuilder':
        """Set extra constructor arguments."""
        self._extra_kwargs.update(kwargs)
        return self

    def from_preset(self, preset: FusionPreset) -> 'FusionBuilder':
        """Load from preset."""
        self._strategy = preset.strategy
        self._config = preset.config
        self._extra_kwargs = preset.extra_kwargs.copy()
        return self

    def inject_custom(self, fusion: BaseFusion) -> 'FusionBuilder':
        """Inject a custom fusion instance."""
        self._custom_fusion = fusion
        return self

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> bool:
        """Validate builder state."""
        if self._custom_fusion is not None:
            return True

        if self._stream_dims is None:
            raise ValueError("Stream dimensions not set. Call with_streams() first.")

        if len(self._stream_dims) < 2:
            raise ValueError("Need at least 2 streams for fusion.")

        return True

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def build(self) -> BaseFusion:
        """Build the fusion layer."""
        # Return custom if injected
        if self._custom_fusion is not None:
            return self._custom_fusion

        self.validate()

        # Get fusion class
        fusion_cls = STRATEGY_TO_CLASS[self._strategy]

        # Build config
        config = self._config or FusionConfig(output_dim=self._output_dim)
        if config.output_dim != self._output_dim:
            config.output_dim = self._output_dim

        # Build kwargs
        kwargs = {
            'stream_dims': self._stream_dims,
            'output_dim': self._output_dim,
            'config': config,
        }
        kwargs.update(self._extra_kwargs)

        return fusion_cls(**kwargs)

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def standard(cls, stream_dims: Dict[str, int], output_dim: int = 512) -> BaseFusion:
        """Build standard concat fusion."""
        return (cls()
                .with_streams(stream_dims)
                .with_output_dim(output_dim)
                .from_preset(STANDARD_FUSION)
                .build())

    @classmethod
    def adaptive(cls, stream_dims: Dict[str, int], output_dim: int = 512) -> BaseFusion:
        """Build adaptive gated fusion."""
        return (cls()
                .with_streams(stream_dims)
                .with_output_dim(output_dim)
                .from_preset(ADAPTIVE_FUSION)
                .build())

    @classmethod
    def attention(cls, stream_dims: Dict[str, int], output_dim: int = 512) -> BaseFusion:
        """Build attention-based fusion."""
        return (cls()
                .with_streams(stream_dims)
                .with_output_dim(output_dim)
                .from_preset(ATTENTION_FUSION)
                .build())


# =============================================================================
# COMPOUND FUSION
# =============================================================================

class CompoundFusion(BaseFusion):
    """
    Combines multiple fusion strategies.

    Each sub-fusion produces a candidate, and a meta-fusion
    (or learned weights) combines them.

    Example:
        compound = CompoundFusion(
            stream_dims={'a': 512, 'b': 512},
            output_dim=512,
            strategies=[FusionStrategy.CONCAT, FusionStrategy.ATTENTION],
        )
    """

    def __init__(
            self,
            stream_dims: Dict[str, int],
            output_dim: int,
            strategies: List[FusionStrategy],
            config: Optional[FusionConfig] = None,
    ):
        super().__init__(stream_dims, output_dim)
        config = config or FusionConfig(output_dim=output_dim)

        # Build sub-fusions
        self.sub_fusions = nn.ModuleList([
            STRATEGY_TO_CLASS[s](stream_dims, output_dim, config)
            for s in strategies
        ])
        self.strategy_names = [s.name for s in strategies]

        # Meta-combiner
        num_fusions = len(strategies)
        self.meta_weights = nn.Parameter(torch.ones(num_fusions))

        # Optional attention-based meta-fusion
        self.meta_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            batch_first=True,
        )
        self.use_meta_attention = num_fusions > 2

        self.norm = nn.LayerNorm(output_dim)

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> tuple:
        # Get outputs from each fusion strategy
        sub_outputs = []
        sub_infos = []
        for fusion in self.sub_fusions:
            out, info = fusion(stream_outputs, stream_fingerprints, return_weights=True)
            sub_outputs.append(out)
            sub_infos.append(info)

        if self.use_meta_attention:
            # Stack and use attention
            stacked = torch.stack(sub_outputs, dim=1)  # [B, F, D]
            attended, attn_weights = self.meta_attention(stacked, stacked, stacked)
            fused = attended.mean(dim=1)  # [B, D]
        else:
            # Simple weighted combination
            weights = torch.softmax(self.meta_weights, dim=0)
            fused = sum(w * out for w, out in zip(weights, sub_outputs))

        fused = self.norm(fused)

        info = None
        if return_weights:
            info = FusionInfo(
                weights=self.meta_weights.detach(),
                intermediate={name: out for name, out in zip(self.strategy_names, sub_outputs)},
                method="compound",
            )

        return fused, info


# =============================================================================
# ADAPTIVE STRATEGY SELECTION
# =============================================================================

class AdaptiveStrategyFusion(BaseFusion):
    """
    Learns to select fusion strategy per sample.

    A router network predicts which fusion strategy to use
    for each input, enabling input-dependent fusion.
    """

    def __init__(
            self,
            stream_dims: Dict[str, int],
            output_dim: int,
            strategies: List[FusionStrategy] = None,
            config: Optional[FusionConfig] = None,
    ):
        super().__init__(stream_dims, output_dim)
        config = config or FusionConfig(output_dim=output_dim)

        if strategies is None:
            strategies = [FusionStrategy.CONCAT, FusionStrategy.GATED, FusionStrategy.ATTENTION]

        # Build sub-fusions
        self.sub_fusions = nn.ModuleList([
            STRATEGY_TO_CLASS[s](stream_dims, output_dim, config)
            for s in strategies
        ])
        self.strategy_names = [s.name for s in strategies]
        num_strategies = len(strategies)

        # Strategy router
        self.router = nn.Sequential(
            nn.Linear(self.total_input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, num_strategies),
        )

        self.temperature = config.temperature
        self.norm = nn.LayerNorm(output_dim)

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> tuple:
        B = next(iter(stream_outputs.values())).shape[0]

        # Route to strategies
        concat = self._concat_outputs(stream_outputs)
        logits = self.router(concat)  # [B, S]
        weights = torch.softmax(logits / self.temperature, dim=-1)  # [B, S]

        # Compute all strategy outputs
        strategy_outputs = []
        for fusion in self.sub_fusions:
            out, _ = fusion(stream_outputs, stream_fingerprints, return_weights=False)
            strategy_outputs.append(out)

        stacked = torch.stack(strategy_outputs, dim=1)  # [B, S, D]

        # Weighted combination
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        fused = self.norm(fused)

        info = None
        if return_weights:
            info = FusionInfo(
                weights=weights.detach(),
                method="adaptive_strategy",
            )

        return fused, info


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def build_fusion(
        stream_dims: Dict[str, int],
        output_dim: int = 512,
        strategy: FusionStrategy = FusionStrategy.CONCAT,
        **kwargs,
) -> BaseFusion:
    """Quick factory for building fusion layers."""
    return (FusionBuilder()
            .with_streams(stream_dims)
            .with_output_dim(output_dim)
            .with_strategy(strategy)
            .with_extra_kwargs(**kwargs)
            .build())


def build_compound_fusion(
        stream_dims: Dict[str, int],
        output_dim: int = 512,
        strategies: List[FusionStrategy] = None,
) -> CompoundFusion:
    """Build compound fusion with multiple strategies."""
    if strategies is None:
        strategies = [FusionStrategy.CONCAT, FusionStrategy.ATTENTION]
    return CompoundFusion(stream_dims, output_dim, strategies)


def build_adaptive_fusion(
        stream_dims: Dict[str, int],
        output_dim: int = 512,
        strategies: List[FusionStrategy] = None,
) -> AdaptiveStrategyFusion:
    """Build adaptive strategy-selecting fusion."""
    return AdaptiveStrategyFusion(stream_dims, output_dim, strategies)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'FusionStrategy',
    # Presets
    'FusionPreset',
    'STANDARD_FUSION',
    'ADAPTIVE_FUSION',
    'ATTENTION_FUSION',
    'FINGERPRINT_FUSION',
    'MOE_FUSION',
    'HIERARCHICAL_FUSION',
    # Builder
    'FusionBuilder',
    # Compound
    'CompoundFusion',
    'AdaptiveStrategyFusion',
    # Factory functions
    'build_fusion',
    'build_compound_fusion',
    'build_adaptive_fusion',
]