"""
geofractal.router.streams
=========================
Stream types for router collectives.

Streams are the "experts" that get coordinated by the router.
Each stream has:
- A unique fingerprint (for divergence)
- A translation head (to common feature space)
- A router (for coordination)

Types:
- BaseStream: Abstract base class
- FrozenStream: Wraps frozen pretrained models (CLIP, DINO, etc.)
- FeatureStream: For pre-extracted features (fastest)
- TrainableStream: Fully trainable backbone + router
"""

from geofractal.router.streams.base import BaseStream
from geofractal.router.streams.frozen import FrozenStream
from geofractal.router.streams.feature import FeatureStream
from geofractal.router.streams.trainable import TrainableStream

__all__ = [
    "BaseStream",
    "FrozenStream",
    "FeatureStream",
    "TrainableStream",
]