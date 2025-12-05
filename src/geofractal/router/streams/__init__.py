"""
geofractal.router.streams
=========================
Stream types for router collectives.

Streams are complete processing units that:
1. Encode input (backbone, projection, or identity)
2. Expand to slots (vectors) or passthrough (sequences)
3. Route through internal head with mailbox coordination
4. Return routed output [B, S, D]

The stream CONTAINS the head - they are not separate components.
Mailbox is passed through forward() for inter-stream coordination.

Interface matches original collective:
    routed, info = stream(input, mailbox, target_fingerprint)

Types:
------
Vector (for [B, D] inputs):
- FeatureVectorStream: Pre-extracted features → slot expansion
- TrainableVectorStream: Trainable backbone → slot expansion
- FrozenEncoderStream: Frozen pretrained encoder → slot expansion

Sequence (for [B, S, D] inputs):
- SequenceStream: Basic projection → passthrough
- TransformerSequenceStream: Transformer encoder → passthrough
- ConvSequenceStream: Multi-scale conv → passthrough

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from .stream_base import BaseStream, InputShape

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
    # Base
    'BaseStream',
    'InputShape',
    # Vector streams
    'FeatureVectorStream',
    'TrainableVectorStream',
    # Sequence streams
    'SequenceStream',
    'TransformerSequenceStream',
    'ConvSequenceStream',
    # Frozen encoder
    'FrozenEncoderStream',
    # Builder
    'StreamBuilder',
    # Legacy aliases (deprecated)
    'VectorStream',
    'FeatureStream',
    'TrainableStream',
    'FrozenStream',
]