"""
geofractal.router.factory.builder
=================================
Fluent builder for constructing router prototypes.

Provides an intuitive API for assembling complete router
systems from streams, heads, and fusion strategies.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
from typing import Type, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from copy import deepcopy

from .factory_protocols import StreamSpec, HeadSpec, FusionSpec
from .factory_prototype import PrototypeConfig, AssembledPrototype, LightweightPrototype


# =============================================================================
# PROTOTYPE PRESETS
# =============================================================================

@dataclass
class PrototypePreset:
    """Preset configuration for common prototype types."""
    name: str
    description: str
    config: PrototypeConfig


def _imagenet_preset() -> PrototypePreset:
    """ImageNet-optimized preset (proven 84.68%)."""
    return PrototypePreset(
        name="imagenet",
        description="ImageNet-optimized with frozen CLIP streams",
        config=PrototypeConfig(
            num_classes=1000,
            prototype_name="imagenet_gfr",
            stream_specs=[
                StreamSpec.frozen_clip("clip_b32", "openai/clip-vit-base-patch32"),
                StreamSpec.frozen_clip("clip_l14", "openai/clip-vit-large-patch14"),
            ],
            head_spec=HeadSpec.standard(feature_dim=512),
            fusion_spec=FusionSpec.standard(output_dim=512),
            freeze_streams=True,
        ),
    )


def _cifar_preset() -> PrototypePreset:
    """CIFAR-10/100 preset."""
    return PrototypePreset(
        name="cifar",
        description="CIFAR-optimized lightweight configuration",
        config=PrototypeConfig(
            num_classes=100,
            prototype_name="cifar_gfr",
            stream_specs=[
                StreamSpec.frozen_clip("clip_b32", "openai/clip-vit-base-patch32"),
            ],
            head_spec=HeadSpec.standard(feature_dim=512),
            fusion_spec=FusionSpec.standard(output_dim=512),
            classifier_hidden=512,
            freeze_streams=True,
        ),
    )


def _fashion_preset() -> PrototypePreset:
    """FashionMNIST preset (proven ρ=9.34)."""
    return PrototypePreset(
        name="fashion",
        description="FashionMNIST with 3 feature streams",
        config=PrototypeConfig(
            num_classes=10,
            prototype_name="fashion_gfr",
            stream_specs=[
                StreamSpec.feature_stream("stream_a", input_dim=784, feature_dim=128),
                StreamSpec.feature_stream("stream_b", input_dim=784, feature_dim=128),
                StreamSpec.feature_stream("stream_c", input_dim=784, feature_dim=128),
            ],
            head_spec=HeadSpec(
                feature_dim=128,
                fingerprint_dim=32,
                num_anchors=8,
                num_routes=4,
            ),
            fusion_spec=FusionSpec.standard(output_dim=128),
            classifier_hidden=128,
            freeze_streams=False,
        ),
    )


def _multimodal_preset() -> PrototypePreset:
    """Multi-modal preset with attention fusion."""
    return PrototypePreset(
        name="multimodal",
        description="Multi-modal with attention-based fusion",
        config=PrototypeConfig(
            num_classes=1000,
            prototype_name="multimodal_gfr",
            stream_specs=[
                StreamSpec.frozen_clip("vision_b32", "openai/clip-vit-base-patch32"),
                StreamSpec.frozen_clip("vision_l14", "openai/clip-vit-large-patch14"),
            ],
            head_spec=HeadSpec.standard(feature_dim=512),
            fusion_spec=FusionSpec.attention(output_dim=512, num_heads=8),
            freeze_streams=True,
        ),
    )


def _research_preset() -> PrototypePreset:
    """Research preset with all features enabled."""
    return PrototypePreset(
        name="research",
        description="Full-featured for research and experimentation",
        config=PrototypeConfig(
            num_classes=1000,
            prototype_name="research_gfr",
            stream_specs=[
                StreamSpec.frozen_clip("clip_b32", "openai/clip-vit-base-patch32"),
                StreamSpec.frozen_clip("clip_l14", "openai/clip-vit-large-patch14"),
            ],
            head_spec=HeadSpec(
                feature_dim=512,
                fingerprint_dim=64,
                num_anchors=16,
                num_routes=8,
                use_cantor=True,
                attention_type='cantor',
                router_type='topk',
                anchor_type='attentive',  # More expressive
            ),
            fusion_spec=FusionSpec(
                strategy='attention',
                output_dim=512,
                num_heads=8,
            ),
            freeze_streams=True,
        ),
    )


# Preset registry
PRESETS: Dict[str, PrototypePreset] = {
    'imagenet': _imagenet_preset(),
    'cifar': _cifar_preset(),
    'fashion': _fashion_preset(),
    'multimodal': _multimodal_preset(),
    'research': _research_preset(),
}


# =============================================================================
# PROTOTYPE BUILDER
# =============================================================================

class PrototypeBuilder:
    """
    Fluent builder for constructing router prototypes.

    Example:
        prototype = (PrototypeBuilder()
            .with_name("my_prototype")
            .with_num_classes(1000)
            .add_stream(StreamSpec.frozen_clip("clip_b32"))
            .add_stream(StreamSpec.frozen_clip("clip_l14"))
            .with_head(HeadSpec.standard())
            .with_fusion(FusionSpec.attention())
            .build())
    """

    def __init__(self):
        self._name: str = "prototype"
        self._num_classes: int = 1000
        self._stream_specs: List[StreamSpec] = []
        self._head_spec: Optional[HeadSpec] = None
        self._fusion_spec: Optional[FusionSpec] = None
        self._classifier_hidden: int = 512
        self._classifier_dropout: float = 0.1
        self._pool_type: str = 'cls'
        self._freeze_streams: bool = True
        self._custom_prototype: Optional[nn.Module] = None

    # -------------------------------------------------------------------------
    # Core settings
    # -------------------------------------------------------------------------

    def with_name(self, name: str) -> 'PrototypeBuilder':
        """Set prototype name."""
        self._name = name
        return self

    def with_num_classes(self, num_classes: int) -> 'PrototypeBuilder':
        """Set number of output classes."""
        self._num_classes = num_classes
        return self

    # -------------------------------------------------------------------------
    # Streams
    # -------------------------------------------------------------------------

    def add_stream(self, spec: StreamSpec) -> 'PrototypeBuilder':
        """Add a stream specification."""
        self._stream_specs.append(spec)
        return self

    def add_streams(self, specs: List[StreamSpec]) -> 'PrototypeBuilder':
        """Add multiple stream specifications."""
        self._stream_specs.extend(specs)
        return self

    def add_frozen_clip(
            self,
            name: str,
            variant: str = "openai/clip-vit-base-patch32",
    ) -> 'PrototypeBuilder':
        """Add a frozen CLIP stream."""
        self._stream_specs.append(StreamSpec.frozen_clip(name, variant))
        return self

    def add_feature_stream(
            self,
            name: str,
            input_dim: int,
            feature_dim: int = 512,
    ) -> 'PrototypeBuilder':
        """Add a feature projection stream."""
        self._stream_specs.append(StreamSpec.feature_stream(name, input_dim, feature_dim))
        return self

    def clear_streams(self) -> 'PrototypeBuilder':
        """Remove all streams."""
        self._stream_specs = []
        return self

    # -------------------------------------------------------------------------
    # Head
    # -------------------------------------------------------------------------

    def with_head(self, spec: HeadSpec) -> 'PrototypeBuilder':
        """Set head specification."""
        self._head_spec = spec
        return self

    def with_standard_head(self, feature_dim: int = 512) -> 'PrototypeBuilder':
        """Use standard head configuration."""
        self._head_spec = HeadSpec.standard(feature_dim)
        return self

    def with_lightweight_head(self, feature_dim: int = 512) -> 'PrototypeBuilder':
        """Use lightweight head configuration."""
        self._head_spec = HeadSpec.lightweight(feature_dim)
        return self

    def with_custom_head(
            self,
            feature_dim: int = 512,
            fingerprint_dim: int = 64,
            num_anchors: int = 16,
            num_routes: int = 4,
            use_cantor: bool = True,
            attention_type: str = 'cantor',
            router_type: str = 'topk',
            anchor_type: str = 'constitutive',
    ) -> 'PrototypeBuilder':
        """Configure custom head."""
        self._head_spec = HeadSpec(
            feature_dim=feature_dim,
            fingerprint_dim=fingerprint_dim,
            num_anchors=num_anchors,
            num_routes=num_routes,
            use_cantor=use_cantor,
            attention_type=attention_type,
            router_type=router_type,
            anchor_type=anchor_type,
        )
        return self

    # -------------------------------------------------------------------------
    # Fusion
    # -------------------------------------------------------------------------

    def with_fusion(self, spec: FusionSpec) -> 'PrototypeBuilder':
        """Set fusion specification."""
        self._fusion_spec = spec
        return self

    def with_concat_fusion(self, output_dim: int = 512) -> 'PrototypeBuilder':
        """Use concatenation fusion."""
        self._fusion_spec = FusionSpec.standard(output_dim)
        return self

    def with_gated_fusion(self, output_dim: int = 512) -> 'PrototypeBuilder':
        """Use gated adaptive fusion."""
        self._fusion_spec = FusionSpec.adaptive(output_dim)
        return self

    def with_attention_fusion(
            self,
            output_dim: int = 512,
            num_heads: int = 8,
    ) -> 'PrototypeBuilder':
        """Use attention-based fusion."""
        self._fusion_spec = FusionSpec.attention(output_dim, num_heads)
        return self

    def with_custom_fusion(
            self,
            strategy: str,
            output_dim: int = 512,
            **kwargs,
    ) -> 'PrototypeBuilder':
        """Configure custom fusion."""
        self._fusion_spec = FusionSpec(
            strategy=strategy,
            output_dim=output_dim,
            **kwargs,
        )
        return self

    # -------------------------------------------------------------------------
    # Classifier
    # -------------------------------------------------------------------------

    def with_classifier(
            self,
            hidden_dim: int = 512,
            dropout: float = 0.1,
    ) -> 'PrototypeBuilder':
        """Configure classifier."""
        self._classifier_hidden = hidden_dim
        self._classifier_dropout = dropout
        return self

    # -------------------------------------------------------------------------
    # Pooling
    # -------------------------------------------------------------------------

    def with_pool_type(self, pool_type: str) -> 'PrototypeBuilder':
        """Set pooling type: 'cls', 'mean', or 'max'."""
        self._pool_type = pool_type
        return self

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def freeze_streams(self, freeze: bool = True) -> 'PrototypeBuilder':
        """Set whether to freeze streams."""
        self._freeze_streams = freeze
        return self

    # -------------------------------------------------------------------------
    # Presets
    # -------------------------------------------------------------------------

    def from_preset(self, preset_name: str) -> 'PrototypeBuilder':
        """Load configuration from preset."""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

        preset = PRESETS[preset_name]
        config = preset.config

        self._name = config.prototype_name
        self._num_classes = config.num_classes
        self._stream_specs = deepcopy(config.stream_specs)
        self._head_spec = deepcopy(config.head_spec)
        self._fusion_spec = deepcopy(config.fusion_spec)
        self._classifier_hidden = config.classifier_hidden
        self._classifier_dropout = config.classifier_dropout
        self._pool_type = config.pool_type
        self._freeze_streams = config.freeze_streams

        return self

    # -------------------------------------------------------------------------
    # Static injection
    # -------------------------------------------------------------------------

    def inject_prototype(self, prototype: nn.Module) -> 'PrototypeBuilder':
        """Inject a custom prototype instance."""
        self._custom_prototype = prototype
        return self

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def validate(self) -> bool:
        """Validate builder state."""
        if self._custom_prototype is not None:
            return True

        if len(self._stream_specs) == 0:
            raise ValueError("No streams configured. Add at least one stream.")

        return True

    def build_config(self) -> PrototypeConfig:
        """Build configuration (without instantiating prototype)."""
        # Apply defaults
        head_spec = self._head_spec or HeadSpec.standard()
        fusion_spec = self._fusion_spec or FusionSpec.standard()

        return PrototypeConfig(
            num_classes=self._num_classes,
            prototype_name=self._name,
            stream_specs=self._stream_specs,
            head_spec=head_spec,
            fusion_spec=fusion_spec,
            classifier_hidden=self._classifier_hidden,
            classifier_dropout=self._classifier_dropout,
            pool_type=self._pool_type,
            freeze_streams=self._freeze_streams,
        )

    def build(self) -> AssembledPrototype:
        """Build the prototype."""
        if self._custom_prototype is not None:
            return self._custom_prototype

        self.validate()
        config = self.build_config()
        return AssembledPrototype(config)

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def imagenet(cls) -> 'PrototypeBuilder':
        """Create builder with ImageNet preset."""
        return cls().from_preset('imagenet')

    @classmethod
    def cifar(cls, num_classes: int = 100) -> 'PrototypeBuilder':
        """Create builder with CIFAR preset."""
        return cls().from_preset('cifar').with_num_classes(num_classes)

    @classmethod
    def fashion(cls) -> 'PrototypeBuilder':
        """Create builder with FashionMNIST preset."""
        return cls().from_preset('fashion')

    @classmethod
    def research(cls) -> 'PrototypeBuilder':
        """Create builder with research preset."""
        return cls().from_preset('research')


# =============================================================================
# QUICK FACTORY FUNCTIONS
# =============================================================================

def build_prototype(
        num_classes: int,
        stream_specs: List[StreamSpec],
        head_spec: Optional[HeadSpec] = None,
        fusion_spec: Optional[FusionSpec] = None,
        **kwargs,
) -> AssembledPrototype:
    """Quick factory for building prototypes."""
    builder = PrototypeBuilder()
    builder.with_num_classes(num_classes)
    builder.add_streams(stream_specs)

    if head_spec:
        builder.with_head(head_spec)
    if fusion_spec:
        builder.with_fusion(fusion_spec)

    for key, value in kwargs.items():
        if hasattr(builder, f'with_{key}'):
            getattr(builder, f'with_{key}')(value)

    return builder.build()


def build_clip_prototype(
        num_classes: int,
        clip_variants: List[str] = None,
        fusion_strategy: str = 'concat',
) -> AssembledPrototype:
    """Build prototype with CLIP streams."""
    if clip_variants is None:
        clip_variants = [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
        ]

    builder = PrototypeBuilder()
    builder.with_num_classes(num_classes)

    for i, variant in enumerate(clip_variants):
        name = f"clip_{i}"
        builder.add_frozen_clip(name, variant)

    builder.with_standard_head()
    builder.with_custom_fusion(fusion_strategy)

    return builder.build()


def build_feature_prototype(
        num_classes: int,
        input_dim: int,
        num_streams: int = 3,
        feature_dim: int = 128,
        fusion_strategy: str = 'concat',
) -> AssembledPrototype:
    """Build prototype with feature projection streams."""
    builder = PrototypeBuilder()
    builder.with_num_classes(num_classes)

    for i in range(num_streams):
        builder.add_feature_stream(f"stream_{i}", input_dim, feature_dim)

    builder.with_custom_head(feature_dim=feature_dim, fingerprint_dim=32, num_anchors=8)
    builder.with_custom_fusion(fusion_strategy, output_dim=feature_dim)
    builder.freeze_streams(False)

    return builder.build()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Presets
    'PrototypePreset',
    'PRESETS',
    # Builder
    'PrototypeBuilder',
    # Factory functions
    'build_prototype',
    'build_clip_prototype',
    'build_feature_prototype',
]