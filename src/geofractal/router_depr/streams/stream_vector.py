"""
geofractal.router.streams.vector
================================
Vector stream implementations with slot expansion.

For [B, D] feature vector inputs. Vectors are expanded to [B, num_slots, D]
to provide positions for the router to operate on.

The slot expansion is intentional and necessary:
- Router needs S > 1 for meaningful Top-K selection
- Attention needs S > 1 for cross-position relationships
- Cantor bias operates on position pairs

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
from typing import Optional, Any

from .stream_base import BaseStream, InputShape
from geofractal.router.head import HeadConfig, ComposedHead, build_standard_head


class FeatureVectorStream(BaseStream):
    """
    Stream for pre-extracted feature vectors with slot expansion.

    Transforms [B, input_dim] → [B, num_slots, feature_dim]

    This is the core stream type used in proven experiments:
        - ImageNet: 5 streams at 0.1% → 84.68% collective (ρ = 847)
        - FashionMNIST: 10% + 10% + 10% = 93.4%

    The slot expansion creates S positions for attention/routing to operate on.
    Each slot learns a unique perspective on the input.

    Usage:
        stream = FeatureVectorStream(
            name='clip_b32',
            input_dim=512,      # CLIP ViT-B/32 features
            feature_dim=256,    # Internal routing dimension
            num_slots=16,       # Number of routing slots
        )

        features = torch.randn(B, 512)  # Pre-extracted
        routed, info = stream(features, mailbox, target_fp)  # [B, 16, 256]
    """

    def __init__(
            self,
            name: str,
            input_dim: int,
            feature_dim: int,
            num_slots: int = 16,
            dropout: float = 0.1,
            head: Optional[ComposedHead] = None,
            head_config: Optional[HeadConfig] = None,
            parent_id: Optional[str] = None,
            cooperation_group: str = "default",
    ):
        super().__init__(
            name=name,
            input_dim=input_dim,
            feature_dim=feature_dim,
            head=head,
            head_config=head_config,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
        )

        self.num_slots = num_slots

        # Translation: [B, input_dim] → [B, feature_dim * num_slots]
        self.translation = nn.Sequential(
            nn.Linear(input_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim * num_slots),
        )

        # Learnable slot embeddings (unique identity per slot)
        self.slot_embed = nn.Parameter(
            torch.randn(1, num_slots, feature_dim) * 0.02
        )

    @property
    def input_shape(self) -> str:
        return InputShape.VECTOR

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Identity encoding - features are already extracted.

        Args:
            x: [B, input_dim] pre-extracted features

        Returns:
            [B, input_dim] same features
        """
        return x

    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Expand vector to slots.

        Args:
            features: [B, input_dim]

        Returns:
            [B, num_slots, feature_dim] slot sequence
        """
        B = features.shape[0]

        # Translate to slot space
        translated = self.translation(features)  # [B, feature_dim * num_slots]

        # Reshape to slots
        slots = translated.view(B, self.num_slots, self.feature_dim)

        # Add slot embeddings for unique slot identities
        return slots + self.slot_embed


class TrainableVectorStream(BaseStream):
    """
    Stream with trainable backbone for vector outputs.

    Wraps any module that produces [B, D] vectors, then expands to slots.

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
            backbone_dim=32,    # Backbone output dim
            feature_dim=256,
            num_slots=16,
        )
    """

    def __init__(
            self,
            name: str,
            backbone: nn.Module,
            backbone_dim: int,
            feature_dim: int,
            num_slots: int = 16,
            dropout: float = 0.1,
            head: Optional[ComposedHead] = None,
            head_config: Optional[HeadConfig] = None,
            parent_id: Optional[str] = None,
            cooperation_group: str = "default",
    ):
        super().__init__(
            name=name,
            input_dim=backbone_dim,
            feature_dim=feature_dim,
            head=head,
            head_config=head_config,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
        )

        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.num_slots = num_slots

        # Deeper translation with pre-norm
        self.pre_norm = nn.LayerNorm(backbone_dim)
        self.translation = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim * num_slots),
        )

        # Learnable slot embeddings
        self.slot_embed = nn.Parameter(
            torch.randn(1, num_slots, feature_dim) * 0.02
        )

    @property
    def input_shape(self) -> str:
        return InputShape.VECTOR

    def encode(self, x: Any) -> torch.Tensor:
        """
        Encode through trainable backbone.

        Args:
            x: Input tensor (shape depends on backbone)

        Returns:
            [B, backbone_dim] encoded features
        """
        return self.backbone(x)

    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize and expand to slots.

        Args:
            features: [B, backbone_dim]

        Returns:
            [B, num_slots, feature_dim] slot sequence
        """
        B = features.shape[0]

        # Pre-normalize
        features = self.pre_norm(features)

        # Translate to slot space
        translated = self.translation(features)

        # Reshape to slots
        slots = translated.view(B, self.num_slots, self.feature_dim)

        return slots + self.slot_embed

    @classmethod
    def conv_stream(
            cls,
            name: str,
            feature_dim: int,
            num_slots: int = 16,
            in_channels: int = 1,
            channels: list = None,
            image_size: int = 28,
            **kwargs,
    ) -> 'TrainableVectorStream':
        """
        Factory for conv backbone stream.

        Creates a simple conv net suitable for MNIST/FashionMNIST.
        """
        channels = channels or [32, 64]

        layers = []
        prev_ch = in_channels

        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.MaxPool2d(2),
            ])
            prev_ch = ch

        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        ])

        backbone = nn.Sequential(*layers)

        return cls(
            name=name,
            backbone=backbone,
            backbone_dim=channels[-1],
            feature_dim=feature_dim,
            num_slots=num_slots,
            **kwargs,
        )

    @classmethod
    def mlp_stream(
            cls,
            name: str,
            input_dim: int,
            feature_dim: int,
            num_slots: int = 16,
            hidden_dims: list = None,
            dropout: float = 0.1,
            **kwargs,
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
            backbone_dim=hidden_dims[-1],
            feature_dim=feature_dim,
            num_slots=num_slots,
            dropout=dropout,
            **kwargs,
        )


# Legacy aliases
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