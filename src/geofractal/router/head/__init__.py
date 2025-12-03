"""
geofractal.router.head
======================
Decomposed head components for GlobalFractalRouter.

This package provides:
- Abstract protocols for each component type
- Concrete implementations (swappable)
- Builder pattern for easy composition
- Static injection support

The head is the decision-making core of the router.
It's been decomposed into:
- Attention (geometric structure)
- Router (sparse selection)
- Anchors (behavioral modes)
- Gate (fingerprint coordination)
- Combiner (signal fusion)
- Refinement (output transformation)

Usage:
    from geofractal.router.head import (
        HeadBuilder,
        HeadConfig,
        build_standard_head,
    )

    # Quick build
    head = build_standard_head(HeadConfig(feature_dim=512))

    # Custom composition
    head = (HeadBuilder(config)
        .with_attention(CantorAttention)
        .with_router(TopKRouter)
        .with_anchors(AttentiveAnchorBank)  # Custom anchor type
        .build())

    # Static injection
    my_custom_gate = MyGate(...)
    head = (HeadBuilder(config)
        .from_preset(STANDARD_HEAD)
        .inject_gate(my_custom_gate)
        .build())

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from .protocols import (
    # Protocols
    Fingerprinted,
    AttentionHead,
    Router,
    AnchorProvider,
    GatingMechanism,
    Combiner,
    Refinement,
    # Abstract bases
    BaseAttention,
    BaseRouter,
    BaseAnchorBank,
    BaseGate,
    BaseCombiner,
    BaseRefinement,
    # Composition
    HeadComponents,
)

from .components import (
    HeadConfig,
    # Attention
    StandardAttention,
    CantorAttention,
    # Routing
    TopKRouter,
    SoftRouter,
    # Anchors
    ConstitutiveAnchorBank,
    AttentiveAnchorBank,
    # Gates
    FingerprintGate,
    ChannelGate,
    # Combiners
    LearnableWeightCombiner,
    GatedCombiner,
    # Refinement
    FFNRefinement,
    MixtureOfExpertsRefinement,
    # Utilities
    cantor_pair,
    build_cantor_bias,
)

from .builder import (
    HeadBuilder,
    HeadPreset,
    ComposedHead,
    STANDARD_HEAD,
    LIGHTWEIGHT_HEAD,
    HEAVY_HEAD,
    build_standard_head,
    build_lightweight_head,
    build_custom_head,
)

__all__ = [
    # Config
    'HeadConfig',
    # Protocols
    'Fingerprinted',
    'AttentionHead',
    'Router',
    'AnchorProvider',
    'GatingMechanism',
    'Combiner',
    'Refinement',
    # Abstract bases
    'BaseAttention',
    'BaseRouter',
    'BaseAnchorBank',
    'BaseGate',
    'BaseCombiner',
    'BaseRefinement',
    # Attention implementations
    'StandardAttention',
    'CantorAttention',
    # Router implementations
    'TopKRouter',
    'SoftRouter',
    # Anchor implementations
    'ConstitutiveAnchorBank',
    'AttentiveAnchorBank',
    # Gate implementations
    'FingerprintGate',
    'ChannelGate',
    # Combiner implementations
    'LearnableWeightCombiner',
    'GatedCombiner',
    # Refinement implementations
    'FFNRefinement',
    'MixtureOfExpertsRefinement',
    # Builder
    'HeadBuilder',
    'HeadPreset',
    'ComposedHead',
    # Presets
    'STANDARD_HEAD',
    'LIGHTWEIGHT_HEAD',
    'HEAVY_HEAD',
    # Factory functions
    'build_standard_head',
    'build_lightweight_head',
    'build_custom_head',
    # Utilities
    'cantor_pair',
    'build_cantor_bias',
    'HeadComponents',
]