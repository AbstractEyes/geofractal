"""
geofractal.router.streams
=========================
Stream types for router systems.

Streams transform inputs into features suitable for heads:
- VectorStream: [B, D] → project → [B, D] (head unsqueezes to [B, 1, D])
- SequenceStream: [B, S, D] → project → [B, S, D]
- FrozenEncoderStream: [B, C, H, W] → encode → [B, D]

NO SLOT EXPANSION. Streams are simple projections.
The head handles attention/routing, not the stream.

Types:
------
Vector (for [B, D] inputs):
- FeatureVectorStream: Pre-extracted features, just projection
- TrainableVectorStream: Trainable backbone + projection

Sequence (for [B, S, D] inputs):
- SequenceStream: Basic projection
- TransformerSequenceStream: Transformer encoder + projection
- ConvSequenceStream: Multi-scale conv + projection

Frozen (for images):
- FrozenEncoderStream: Frozen pretrained encoder (CLIP, DINO, etc.)

Factory:
--------
- StreamBuilder: Factory for building streams

Usage:
------
    from geofractal.router.streams import StreamBuilder, FeatureVectorStream

    # Direct construction
    stream = FeatureVectorStream(
        name='clip_b32',
        input_dim=512,
        feature_dim=256,
    )

    # Via builder
    builder = StreamBuilder(feature_dim=256)
    stream = builder.feature_vector('clip_b32', input_dim=512)

    # Forward
    features = torch.randn(B, 512)
    output, info = stream(features)  # [B, 1, 256] ready for head

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from .stream_protocols import StreamProtocol, InputShape
from .stream_base import BaseStream
from .stream_vector import (
    FeatureVectorStream,
    TrainableVectorStream,
    # Legacy aliases
    VectorStream,
    FeatureStream,
    TrainableStream,
)
from .stream_sequence import (
    SequenceStream,
    TransformerSequenceStream,
    ConvSequenceStream,
)
from .stream_frozen import (
    FrozenEncoderStream,
    FrozenStream,
)
from .stream_builder import StreamBuilder


__all__ = [
    # Protocols
    'StreamProtocol',
    'InputShape',
    # Base
    'BaseStream',
    # Vector streams
    'FeatureVectorStream',
    'TrainableVectorStream',
    # Sequence streams
    'SequenceStream',
    'TransformerSequenceStream',
    'ConvSequenceStream',
    # Frozen encoder
    'FrozenEncoderStream',
    # Factory
    'StreamBuilder',
    # Legacy aliases (deprecated)
    'VectorStream',
    'FeatureStream',
    'TrainableStream',
    'FrozenStream',
]