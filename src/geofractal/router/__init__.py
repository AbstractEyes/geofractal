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
        GlobalFractalRouter,
        RouterCollective,
        FrozenStream,
        FeatureStream,
    )

    # Quick collective
    collective = RouterCollective.from_streams([
        FrozenStream.from_pretrained("openai/clip-vit-base-patch32"),
        FrozenStream.from_pretrained("openai/clip-vit-large-patch14"),
    ])

    # Train routing only
    collective.fit(train_loader, epochs=20)

Subpackages:
    geofractal.router.head    - Decomposed head components
    geofractal.router.fusion  - Fusion layer implementations

Author: AbstractPhil
Architecture: GlobalFractalRouter
Date: December 2025
"""

from geofractal.router.core import (
    GlobalFractalRouter,
    GlobalFractalRouterConfig,
    CantorMultiHeadAttention,
    AnchorBank,
    FingerprintGate,
    TopKRouter,
    cantor_pair,
    cantor_unpair,
    build_cantor_bias,
)

from geofractal.router.registry import (
    RouterRegistry,
    RouterMailbox,
    get_registry,
)

from geofractal.router.collective import (
    RouterCollective,
)

from geofractal.router.config import (
    CollectiveConfig,
    StreamConfig,
    IMAGENET_COLLECTIVE_CONFIG,
    FASHIONMNIST_COLLECTIVE_CONFIG,
    CIFAR_COLLECTIVE_CONFIG,
)

from geofractal.router.streams import (
    BaseStream,
    FrozenStream,
    FeatureStream,
    TrainableStream,
)

# Head components (decomposed)
from geofractal.router.head import (
    HeadBuilder,
    HeadConfig,
    ComposedHead,
    build_standard_head,
    STANDARD_HEAD,
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

# Factory (prototype assembly)
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
    # Core Router
    "GlobalFractalRouter",
    "GlobalFractalRouterConfig",
    # Core Components
    "CantorMultiHeadAttention",
    "AnchorBank",
    "FingerprintGate",
    "TopKRouter",
    # Cantor Functions
    "cantor_pair",
    "cantor_unpair",
    "build_cantor_bias",
    # Registry
    "RouterRegistry",
    "RouterMailbox",
    "get_registry",
    # Collective
    "RouterCollective",
    "CollectiveConfig",
    "StreamConfig",
    # Presets
    "IMAGENET_COLLECTIVE_CONFIG",
    "FASHIONMNIST_COLLECTIVE_CONFIG",
    "CIFAR_COLLECTIVE_CONFIG",
    # Streams
    "BaseStream",
    "FrozenStream",
    "FeatureStream",
    "TrainableStream",
    # Head (decomposed)
    "HeadBuilder",
    "HeadConfig",
    "ComposedHead",
    "build_standard_head",
    "STANDARD_HEAD",
    # Fusion
    "FusionBuilder",
    "FusionStrategy",
    "FusionConfig",
    "build_fusion",
    "ConcatFusion",
    "GatedFusion",
    "AttentionFusion",
    "STANDARD_FUSION",
    # Factory (prototype assembly)
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

__version__ = "0.1.0"