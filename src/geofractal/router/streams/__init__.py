"""
geofractal.router.streams
=========================
Stream types for router collectives.

Streams are the "experts" that get coordinated by the router.
Each stream has:
- A backbone (optional - encodes input)
- A head (ComposedHead from head/builder.py)
- Access to fingerprint via head

Input Shapes:
- VectorStream: [B, D] - expands to slots
- SequenceStream: [B, S, D] - routes tokens directly

Types:
- BaseStream: Abstract base class
- VectorStream: For [B, D] vector inputs
  - FeatureVectorStream: Pre-extracted features
  - TrainableVectorStream: Trainable backbone
- SequenceStream: For [B, S, D] sequence inputs
  - TransformerSequenceStream: Transformer backbone
  - ConvSequenceStream: Multi-scale conv backbone

Factory:
- StreamBuilder: Factory for building streams with consistent config
"""

from geofractal.router.streams.protocols import StreamProtocol, InputShape
from geofractal.router.streams.base import BaseStream
from geofractal.router.streams.vector import (
    VectorStream,
    FeatureVectorStream,
    TrainableVectorStream,
)
from geofractal.router.streams.sequence import (
    SequenceStream,
    TransformerSequenceStream,
    ConvSequenceStream,
)
from geofractal.router.streams.builder import StreamBuilder

# Legacy aliases (deprecated - use new names)
FrozenStream = FeatureVectorStream
FeatureStream = FeatureVectorStream
TrainableStream = TrainableVectorStream

__all__ = [
    # Protocols
    "StreamProtocol",
    "InputShape",
    # Base
    "BaseStream",
    # Vector streams
    "VectorStream",
    "FeatureVectorStream",
    "TrainableVectorStream",
    # Sequence streams
    "SequenceStream",
    "TransformerSequenceStream",
    "ConvSequenceStream",
    # Factory
    "StreamBuilder",
    # Legacy (deprecated)
    "FrozenStream",
    "FeatureStream",
    "TrainableStream",
]