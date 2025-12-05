# streams/vector.py
"""
Vector stream implementations.

For [B, D] feature vector inputs. Simple projection, no artificial
structure manufacturing.

The old "slot expansion" is GONE. If you need S > 1, use a sequence
stream or let the head handle it.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
from typing import Optional

from .stream_base import BaseStream
from .stream_protocols import InputShape


class FeatureVectorStream(BaseStream):
    """
    Stream for pre-extracted feature vectors.

    Transforms [B, input_dim] → [B, feature_dim]

    This is the simplest stream - just projection + normalization.
    Used with pre-extracted CLIP/DINO features.

    Usage:
        stream = FeatureVectorStream(
            name='clip_b32',
            input_dim=512,      # CLIP ViT-B/32 features
            feature_dim=256,    # Internal dimension
        )

        features = torch.randn(B, 512)  # Pre-extracted
        output, _ = stream(features)     # [B, 1, 256] ready for head
    """

    def __init__(
        self,
        name: str,
        input_dim: int,
        feature_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__(name, input_dim, feature_dim)

        # Simple projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @property
    def input_shape(self) -> str:
        return InputShape.VECTOR

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project feature vector.

        Args:
            x: [B, input_dim] pre-extracted features

        Returns:
            [B, feature_dim] projected features
        """
        return self.projection(x)


class TrainableVectorStream(BaseStream):
    """
    Stream with trainable backbone for vector outputs.

    Wraps any module that produces [B, D] vectors.
    Adds projection to feature_dim.

    Usage:
        backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        stream = TrainableVectorStream(
            name='conv_expert',
            backbone=backbone,
            input_dim=32,       # Backbone output dim
            feature_dim=256,
        )
    """

    def __init__(
        self,
        name: str,
        backbone: nn.Module,
        input_dim: int,
        feature_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__(name, input_dim, feature_dim)

        self.backbone = backbone

        # Projection after backbone
        if input_dim != feature_dim:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.projection = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
            )

    @property
    def input_shape(self) -> str:
        return InputShape.VECTOR

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode through backbone then project.

        Args:
            x: Input tensor (shape depends on backbone)

        Returns:
            [B, feature_dim] encoded features
        """
        features = self.backbone(x)
        return self.projection(features)

    @classmethod
    def conv_stream(
        cls,
        name: str,
        feature_dim: int,
        in_channels: int = 1,
        channels: list = None,
        image_size: int = 28,
        dropout: float = 0.1,
    ) -> 'TrainableVectorStream':
        """
        Factory for conv backbone stream.

        Creates a simple conv net suitable for MNIST/FashionMNIST.
        """
        channels = channels or [32, 64]
        layers = []
        current_size = image_size
        prev_ch = in_channels

        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.MaxPool2d(2),
            ])
            prev_ch = ch
            current_size //= 2

        flat_dim = channels[-1] * current_size * current_size
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        ])

        backbone = nn.Sequential(*layers)

        return cls(
            name=name,
            backbone=backbone,
            input_dim=channels[-1],  # After adaptive pool
            feature_dim=feature_dim,
            dropout=dropout,
        )

    @classmethod
    def mlp_stream(
        cls,
        name: str,
        input_dim: int,
        feature_dim: int,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ) -> 'TrainableVectorStream':
        """Factory for MLP backbone stream."""
        hidden_dims = hidden_dims or [256, 256]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        backbone = nn.Sequential(*layers)

        return cls(
            name=name,
            backbone=backbone,
            input_dim=hidden_dims[-1],
            feature_dim=feature_dim,
            dropout=dropout,
        )


# =============================================================================
# LEGACY ALIASES (deprecated)
# =============================================================================

VectorStream = FeatureVectorStream
FeatureStream = FeatureVectorStream
TrainableStream = TrainableVectorStream


__all__ = [
    'FeatureVectorStream',
    'TrainableVectorStream',
    # Legacy
    'VectorStream',
    'FeatureStream',
    'TrainableStream',
]