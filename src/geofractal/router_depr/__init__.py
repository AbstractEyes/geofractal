"""
geofractal.router
=================
Collective intelligence through geometric routing.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
See LICENSE and NOTICE files for attribution requirements.

Core infrastructure for coordinating multiple models/experts
via fingerprint-based divergence and mailbox coordination.

Proven Results:
- 5 streams at 0.1% → 84.68% collective (ImageNet)
- 10% + 10% + 10% = 93.4% (FashionMNIST)
- 98.6% frozen, 92.6% accuracy (Dual CLIP)

Usage:
    from geofractal.router import (
        RouterCollective,
        CollectiveConfig,
        StreamSpec,
        HeadSpec,
        FusionSpec,
    )

    # Build collective from specs
    collective = RouterCollective.from_specs(
        stream_specs=[
            StreamSpec.feature_vector("clip_b32", input_dim=512),
            StreamSpec.feature_vector("clip_l14", input_dim=768),
        ],
        config=CollectiveConfig(feature_dim=512, num_classes=1000),
    )

    # Train routing only
    history = collective.fit(train_loader, val_loader, epochs=20)

Subpackages:
    geofractal.router.head    - Decomposed head components
    geofractal.router.fusion  - Fusion layer implementations
    geofractal.router.streams - Stream types (vector, sequence)
    geofractal.router.factory - Prototype assembly

Author: AbstractPhil
Date: December 2025
"""

# Config
from geofractal.router.config import (
    GlobalFractalRouterConfig,
    CollectiveConfig,
    StreamConfig,
    IMAGENET_COLLECTIVE_CONFIG,
    FASHIONMNIST_COLLECTIVE_CONFIG,
    CIFAR_COLLECTIVE_CONFIG,
)

# Registry
from geofractal.router.registry import (
    RouterRegistry,
    RouterMailbox,
    get_registry,
)

# Collective
from geofractal.router.collective import (
    RouterCollective,
)

# Streams
from geofractal.router.streams import (
    StreamProtocol,
    InputShape,
    BaseStream,
    VectorStream,
    FeatureVectorStream,
    TrainableVectorStream,
    SequenceStream,
    TransformerSequenceStream,
    ConvSequenceStream,
    StreamBuilder,
    # Legacy aliases
    FrozenStream,
    FeatureStream,
    TrainableStream,
)

# Head components
from geofractal.router.head import (
    HeadBuilder,
    HeadConfig,
    ComposedHead,
    build_standard_head,
    STANDARD_HEAD,
    # Components
    CantorAttention,
    TopKRouter,
    FingerprintGate,
    ConstitutiveAnchorBank,
    AttentiveAnchorBank,
    LearnableWeightCombiner,
    FFNRefinement,
    # Cantor utilities
    cantor_pair,
    cantor_unpair,
    build_cantor_bias,
)

# Fusion strategies
from geofractal.router.fusion import (
    FusionBuilder,
    FusionStrategy,
    FusionConfig,
    build_fusion,
    ConcatFusion,
    GatedFusion,
    AttentionFusion,
    STANDARD_FUSION,
)

# Factory
from geofractal.router.factory import (
    PrototypeBuilder,
    PrototypeConfig,
    AssembledPrototype,
    StreamSpec,
    HeadSpec,
    FusionSpec,
    get_prototype_registry,
    ComponentSwapper,
    build_prototype,
    build_clip_prototype,
    build_feature_prototype,
)

__all__ = [
    # Config
    "GlobalFractalRouterConfig",
    "CollectiveConfig",
    "StreamConfig",
    "IMAGENET_COLLECTIVE_CONFIG",
    "FASHIONMNIST_COLLECTIVE_CONFIG",
    "CIFAR_COLLECTIVE_CONFIG",
    # Registry
    "RouterRegistry",
    "RouterMailbox",
    "get_registry",
    # Collective
    "RouterCollective",
    # Streams (new)
    "StreamProtocol",
    "InputShape",
    "BaseStream",
    "VectorStream",
    "FeatureVectorStream",
    "TrainableVectorStream",
    "SequenceStream",
    "TransformerSequenceStream",
    "ConvSequenceStream",
    "StreamBuilder",
    # Streams (legacy)
    "FrozenStream",
    "FeatureStream",
    "TrainableStream",
    # Head
    "HeadBuilder",
    "HeadConfig",
    "ComposedHead",
    "build_standard_head",
    "STANDARD_HEAD",
    # Head components
    "CantorAttention",
    "TopKRouter",
    "FingerprintGate",
    "ConstitutiveAnchorBank",
    "AttentiveAnchorBank",
    "LearnableWeightCombiner",
    "FFNRefinement",
    # Cantor utilities
    "cantor_pair",
    "cantor_unpair",
    "build_cantor_bias",
    # Fusion
    "FusionBuilder",
    "FusionStrategy",
    "FusionConfig",
    "build_fusion",
    "ConcatFusion",
    "GatedFusion",
    "AttentionFusion",
    "STANDARD_FUSION",
    # Factory
    "PrototypeBuilder",
    "PrototypeConfig",
    "AssembledPrototype",
    "StreamSpec",
    "HeadSpec",
    "FusionSpec",
    "get_prototype_registry",
    "ComponentSwapper",
    "build_prototype",
    "build_clip_prototype",
    "build_feature_prototype",
]

__version__ = "0.2.0"