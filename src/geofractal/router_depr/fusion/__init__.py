"""
geofractal.router.fusion
========================
Fusion layer implementations for combining divergent streams.

Fusion is the critical junction where multiple divergent stream
outputs become collective intelligence. This package provides
multiple fusion strategies, each representing a different
philosophy for combination.

Strategies:
-----------
- CONCAT: Simple concatenation → projection (baseline, proven)
- WEIGHTED: Learnable static weights per stream
- GATED: Input-adaptive gating (different weights per sample)
- ATTENTION: Cross-stream multi-head attention
- FINGERPRINT: Fingerprint-guided fusion
- RESIDUAL: Additive with learned residual corrections
- MOE: Mixture-of-experts with sparse activation
- HIERARCHICAL: Tree-structured progressive fusion

Advanced:
---------
- CompoundFusion: Combine multiple strategies
- AdaptiveStrategyFusion: Learn to select strategy per sample

Usage:
------
    from geofractal.router.fusion import (
        FusionBuilder,
        FusionStrategy,
        build_fusion,
        STANDARD_FUSION,
    )

    # Quick build
    fusion = build_fusion(
        stream_dims={'clip_b32': 512, 'clip_l14': 768},
        output_dim=512,
        strategy=FusionStrategy.ATTENTION,
    )

    # Builder pattern
    fusion = (FusionBuilder()
        .with_streams({'a': 512, 'b': 768, 'c': 512})
        .with_output_dim(512)
        .from_preset(ADAPTIVE_FUSION)
        .build())

    # Forward
    fused, info = fusion(stream_outputs, return_weights=True)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from .fusion_protocols import (
    # Protocols
    StreamFusion,
    AdaptiveFusion,
    HierarchicalFusion,
    # Abstract bases
    BaseFusion,
    BaseAdaptiveFusion,
    # Info containers
    FusionInfo,
)

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

from .fusion_builder import (
    # Enums
    FusionStrategy,
    # Presets
    FusionPreset,
    STANDARD_FUSION,
    ADAPTIVE_FUSION,
    ATTENTION_FUSION,
    FINGERPRINT_FUSION,
    MOE_FUSION,
    HIERARCHICAL_FUSION,
    # Builder
    FusionBuilder,
    # Compound
    CompoundFusion,
    AdaptiveStrategyFusion,
    # Factory functions
    build_fusion,
    build_compound_fusion,
    build_adaptive_fusion,
)

# Default fusion (proven on ImageNet)
DEFAULT_FUSION = ConcatFusion

__all__ = [
    # Config
    'FusionConfig',
    # Protocols
    'StreamFusion',
    'AdaptiveFusion',
    'HierarchicalFusion',
    # Abstract bases
    'BaseFusion',
    'BaseAdaptiveFusion',
    # Info
    'FusionInfo',
    # Strategy enum
    'FusionStrategy',
    # Concrete methods
    'ConcatFusion',
    'WeightedFusion',
    'GatedFusion',
    'AttentionFusion',
    'FingerprintGuidedFusion',
    'ResidualFusion',
    'MoEFusion',
    'HierarchicalTreeFusion',
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
    # Factory
    'build_fusion',
    'build_compound_fusion',
    'build_adaptive_fusion',
    # Default
    'DEFAULT_FUSION',
]