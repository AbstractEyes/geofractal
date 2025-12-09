"""
geofractal.router.streams.trainable
===================================
Stream with trainable backbone.

Unlike FrozenStream, the backbone learns along with the router.
For lightweight custom models like conv nets.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any, Callable

from geofractal.router.streams.stream_base import BaseStream
from geofractal.router.config import CollectiveConfig
from geofractal.router.registry import RouterMailbox


class TrainableStream(BaseStream):
    """
    Stream with trainable backbone.

    The backbone is a learnable module (conv net, MLP, etc.)
    Both backbone and router parameters are optimized.

    Usage:
        # With custom backbone
        backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 256),
        )

        stream = TrainableStream(
            config=config,
            name="conv_expert",
            backbone=backbone,
            input_dim=256,
        )

        # Or use factory for common architectures
        stream = TrainableStream.conv_stream(
            config=config,
            name="conv_expert",
            in_channels=1,
            channels=[32, 64],
        )
    """

    def __init__(
            self,
            config: CollectiveConfig,
            name: str,
            backbone: nn.Module,
            input_dim: int,
            parent_id: Optional[str] = None,
            cooperation_group: str = "trainable_collective",
    ):
        super().__init__(
            config=config,
            name=name,
            input_dim=input_dim,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
        )

        self.backbone = backbone

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode through trainable backbone.

        Args:
            x: Input tensor (shape depends on backbone)

        Returns:
            features: [B, input_dim] encoded features
        """
        return self.backbone(x)

    @classmethod
    def conv_stream(
            cls,
            config: CollectiveConfig,
            name: str,
            in_channels: int = 1,
            channels: list = [32, 64],
            image_size: int = 28,
            parent_id: Optional[str] = None,
    ) -> "TrainableStream":
        """
        Factory for conv backbone stream.

        Creates a simple conv net suitable for MNIST/FashionMNIST.

        Args:
            config: Collective configuration
            name: Stream name
            in_channels: Input channels (1 for grayscale)
            channels: Conv channel progression
            image_size: Input image size
            parent_id: Parent stream ID
        """
        layers = []
        current_size = image_size
        prev_channels = in_channels

        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_channels, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.MaxPool2d(2),
            ])
            prev_channels = ch
            current_size //= 2

        # Flatten and project
        flat_dim = channels[-1] * current_size * current_size
        layers.extend([
            nn.Flatten(),
            nn.Linear(flat_dim, config.feature_dim),
        ])

        backbone = nn.Sequential(*layers)

        return cls(
            config=config,
            name=name,
            backbone=backbone,
            input_dim=config.feature_dim,  # After projection
            parent_id=parent_id,
        )

    @classmethod
    def mlp_stream(
            cls,
            config: CollectiveConfig,
            name: str,
            input_dim: int,
            hidden_dims: list = [256, 256],
            parent_id: Optional[str] = None,
    ) -> "TrainableStream":
        """
        Factory for MLP backbone stream.

        Args:
            config: Collective configuration
            name: Stream name
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            parent_id: Parent stream ID
        """
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(prev_dim, config.feature_dim))

        backbone = nn.Sequential(*layers)

        return cls(
            config=config,
            name=name,
            backbone=backbone,
            input_dim=config.feature_dim,
            parent_id=parent_id,
        )