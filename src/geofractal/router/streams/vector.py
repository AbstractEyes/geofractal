# streams/vector.py
"""Streams for [B, D] vector inputs."""

import torch
import torch.nn as nn
from typing import Optional

from geofractal.router.config import CollectiveConfig
from geofractal.router.head import HeadConfig
from geofractal.router.head.builder import ComposedHead
from geofractal.router.streams.base import BaseStream
from geofractal.router.streams.protocols import InputShape


class VectorStream(BaseStream):
    """
    Stream for [B, D] vector inputs.
    Expands to slots via learned translation.
    """

    def __init__(
        self,
        config: CollectiveConfig,
        name: str,
        input_dim: int,
        num_slots: int = 4,
        head: Optional[ComposedHead] = None,
        head_config: Optional[HeadConfig] = None,
    ):
        super().__init__(config, name, head, head_config)

        self.input_dim = input_dim
        self.num_slots = num_slots

        # [B, input_dim] → [B, num_slots * feature_dim]
        self.translation = nn.Sequential(
            nn.Linear(input_dim, config.feature_dim * 2),
            nn.LayerNorm(config.feature_dim * 2),
            nn.GELU(),
            nn.Linear(config.feature_dim * 2, config.feature_dim * num_slots),
        )

        self.slot_embed = nn.Parameter(
            torch.randn(1, num_slots, config.feature_dim) * 0.02
        )

    @property
    def input_shape(self) -> str:
        return InputShape.VECTOR

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x  # Override in subclass for backbone

    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        translated = self.translation(features)
        slots = translated.view(B, self.num_slots, self.config.feature_dim)
        return slots + self.slot_embed


class FeatureVectorStream(VectorStream):
    """Pre-extracted features, no backbone."""
    pass


class TrainableVectorStream(VectorStream):
    """Vector stream with trainable backbone."""

    def __init__(
        self,
        config: CollectiveConfig,
        name: str,
        backbone: nn.Module,
        input_dim: int,
        **kwargs,
    ):
        super().__init__(config, name, input_dim, **kwargs)
        self.backbone = backbone

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)