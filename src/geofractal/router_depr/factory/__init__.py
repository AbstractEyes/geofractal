"""
geofractal.router.factory
=========================
Prototype factory for building complete router systems.

This package provides the machinery to assemble complete,
trainable router prototypes from modular components:
- Streams (divergent feature extractors)
- Heads (routing decision makers)
- Fusion (combination strategies)
- Classifiers (final prediction)

Quick Start:
------------
    from geofractal.router.factory import PrototypeBuilder

    # Build from preset
    prototype = PrototypeBuilder.imagenet().build()

    # Custom configuration
    prototype = (PrototypeBuilder()
        .with_name("my_router")
        .with_num_classes(1000)
        .add_frozen_clip("clip_b32", "openai/clip-vit-base-patch32")
        .add_frozen_clip("clip_l14", "openai/clip-vit-large-patch14")
        .with_attention_fusion()
        .build())

    # Train
    logits, info = prototype(images, return_info=True)

Presets:
--------
    - imagenet: ImageNet-optimized (proven 84.68%)
    - cifar: CIFAR-10/100 configuration
    - fashion: FashionMNIST (proven ρ=9.34)
    - multimodal: Multi-modal with attention fusion
    - research: Full-featured for experimentation

Registry:
---------
    from geofractal.router.factory import get_prototype_registry

    registry = get_prototype_registry()
    pid = registry.register(prototype, tags=['experiment_1'])
    registry.update_metrics(pid, {'accuracy': 0.85, 'emergence': 8.5})

Hot-Swapping:
-------------
    from geofractal.router.factory import ComponentSwapper

    swapper = ComponentSwapper(prototype)
    swapper.swap_fusion(new_fusion)
    swapper.swap_head("clip_b32", new_head)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from .factory_protocols import (
    # Protocols
    RouterPrototype,
    Configurable,
    # Abstract bases
    BasePrototype,
    # Info containers
    PrototypeInfo,
    # Specs
    StreamSpec,
    HeadSpec,
    FusionSpec,
)

from .factory_prototype import (
    PrototypeConfig,
    AssembledPrototype,
    LightweightPrototype,
)

from .factory_builder import (
    # Presets
    PrototypePreset,
    PRESETS,
    # Builder
    PrototypeBuilder,
    # Factory functions
    build_prototype,
    build_clip_prototype,
    build_feature_prototype,
)

from .factory_registry import (
    PrototypeRecord,
    PrototypeRegistry,
    get_prototype_registry,
    ComponentSwapper,
    ExperimentRun,
    ExperimentTracker,
    get_experiment_tracker,
)


__all__ = [
    # Protocols
    'RouterPrototype',
    'Configurable',
    'BasePrototype',
    'PrototypeInfo',
    # Specs
    'StreamSpec',
    'HeadSpec',
    'FusionSpec',
    # Config
    'PrototypeConfig',
    # Prototypes
    'AssembledPrototype',
    'LightweightPrototype',
    # Presets
    'PrototypePreset',
    'PRESETS',
    # Builder
    'PrototypeBuilder',
    # Factory functions
    'build_prototype',
    'build_clip_prototype',
    'build_feature_prototype',
    # Registry
    'PrototypeRecord',
    'PrototypeRegistry',
    'get_prototype_registry',
    # Swapper
    'ComponentSwapper',
    # Experiments
    'ExperimentRun',
    'ExperimentTracker',
    'get_experiment_tracker',
]