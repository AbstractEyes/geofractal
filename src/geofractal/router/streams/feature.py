"""
geofractal.router.streams.feature
=================================
Stream for pre-extracted features.

This is the fastest stream type - no model inference needed.
Just routes pre-computed feature vectors.

Used in the ImageNet experiment with pre-extracted CLIP features.
"""

import torch
from typing import Dict, Tuple, Optional, Any

from geofractal.router.streams.base import BaseStream
from geofractal.router.config import CollectiveConfig
from geofractal.router.registry import RouterMailbox


class FeatureStream(BaseStream):
    """
    Stream for pre-extracted features.

    No backbone model - just translation + routing.
    Fastest possible throughput.

    Usage:
        stream = FeatureStream(
            config=config,
            name="clip_vit_b32",
            input_dim=512,
        )

        # features: [B, 512] pre-extracted
        routed, info = stream(features, mailbox)

    Results:
        - ImageNet: 5 streams at 0.1% → 84.68% collective
    """

    def __init__(
            self,
            config: CollectiveConfig,
            name: str,
            input_dim: int,
            parent_id: Optional[str] = None,
            cooperation_group: str = "feature_collective",
    ):
        super().__init__(
            config=config,
            name=name,
            input_dim=input_dim,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Identity encoding - features are already extracted.

        Args:
            x: [B, input_dim] pre-extracted features

        Returns:
            x: [B, input_dim] same features (no transform)
        """
        return x

    @classmethod
    def from_config(
            cls,
            collective_config: CollectiveConfig,
            name: str,
            input_dim: int,
            parent_id: Optional[str] = None,
    ) -> "FeatureStream":
        """Create from collective config."""
        return cls(
            config=collective_config,
            name=name,
            input_dim=input_dim,
            parent_id=parent_id,
        )