"""
geofractal.router.streams.vector
================================
Vector stream implementations with slot expansion.

These streams transform [B, D] feature vectors into [B, S, D] slot sequences
for proper attention and routing dynamics.

The slot expansion is critical - without it, S=1 causes softmax degeneracy
and zero gradients through attention/routing components.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
from typing import Optional
from enum import Enum


class InputShape(Enum):
    """Input tensor shape type."""
    VECTOR = "vector"  # [B, D]
    SEQUENCE = "sequence"  # [B, S, D]


class FeatureVectorStream(nn.Module):
    """
    Stream for pre-extracted feature vectors with slot expansion.

    Transforms [B, input_dim] -> [B, num_slots, feature_dim]

    This is the core stream type used in proven experiments:
        - ImageNet: 5 streams at 0.1% → 84.68% collective
        - FashionMNIST: 10% + 10% + 10% = 93.4%

    The slot expansion creates S positions for attention/routing to operate on,
    preventing the S=1 degeneracy where softmax gradients collapse.

    Usage:
        stream = FeatureVectorStream(
            input_dim=512,      # CLIP ViT-B/32 features
            feature_dim=256,    # Internal routing dimension
            num_slots=16,       # Number of routing slots
        )

        features = torch.randn(B, 512)  # Pre-extracted
        slots = stream(features)         # [B, 16, 256]
    """

    def __init__(
            self,
            input_dim: int,
            feature_dim: int,
            num_slots: int = 16,
            dropout: float = 0.1,
            name: Optional[str] = None,
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_slots = num_slots
        self.input_shape = InputShape.VECTOR

        # Translation head: input_dim -> feature_dim * num_slots
        self.translation = nn.Sequential(
            nn.Linear(input_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim * num_slots),
        )

        # Learnable slot embeddings (unique per stream instance)
        self.slot_embed = nn.Parameter(
            torch.randn(1, num_slots, feature_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform feature vector to slot sequence.

        Args:
            x: [B, input_dim] pre-extracted features

        Returns:
            slots: [B, num_slots, feature_dim]
        """
        B = x.shape[0]
        translated = self.translation(x)  # [B, feature_dim * num_slots]
        slots = translated.view(B, self.num_slots, self.feature_dim)
        return slots + self.slot_embed

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool slots back to vector: [B, S, D] -> [B, D]"""
        return x.mean(dim=1)


class TrainableVectorStream(nn.Module):
    """
    Trainable stream with deeper transformation and slot expansion.

    Transforms [B, input_dim] -> [B, num_slots, feature_dim]

    Deeper than FeatureVectorStream - use when input features need
    more transformation before routing.
    """

    def __init__(
            self,
            input_dim: int,
            feature_dim: int,
            num_slots: int = 16,
            dropout: float = 0.1,
            name: Optional[str] = None,
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_slots = num_slots
        self.input_shape = InputShape.VECTOR

        # Deeper translation with pre-norm
        self.pre_norm = nn.LayerNorm(input_dim)
        self.translation = nn.Sequential(
            nn.Linear(input_dim, feature_dim * 2),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform feature vector to slot sequence.

        Args:
            x: [B, input_dim] features

        Returns:
            slots: [B, num_slots, feature_dim]
        """
        B = x.shape[0]
        x = self.pre_norm(x)
        translated = self.translation(x)
        slots = translated.view(B, self.num_slots, self.feature_dim)
        return slots + self.slot_embed

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool slots back to vector: [B, S, D] -> [B, D]"""
        return x.mean(dim=1)


class SequenceStream(nn.Module):
    """
    Stream for sequence inputs (already [B, S, D]).

    Projects to feature_dim if needed, passes through otherwise.
    Use for transformer embeddings, time series, etc.
    """

    def __init__(
            self,
            input_dim: int,
            feature_dim: int,
            name: Optional[str] = None,
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.input_shape = InputShape.SEQUENCE

        if input_dim != feature_dim:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project sequence to feature dimension.

        Args:
            x: [B, S, input_dim] sequence

        Returns:
            [B, S, feature_dim]
        """
        return self.proj(x)

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence to vector: [B, S, D] -> [B, D]"""
        return x.mean(dim=1)


class TransformerSequenceStream(nn.Module):
    """
    Sequence stream with transformer encoder layers.

    For sequences that need self-attention processing before routing.
    """

    def __init__(
            self,
            input_dim: int,
            feature_dim: int,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.1,
            name: Optional[str] = None,
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.input_shape = InputShape.SEQUENCE

        if input_dim != feature_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )
        else:
            self.input_proj = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence through transformer.

        Args:
            x: [B, S, input_dim] sequence

        Returns:
            [B, S, feature_dim]
        """
        x = self.input_proj(x)
        return self.encoder(x)

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence to vector: [B, S, D] -> [B, D]"""
        return x.mean(dim=1)


class ConvSequenceStream(nn.Module):
    """
    Multi-scale 1D convolution stream for sequences.

    Applies parallel convolutions with different kernel sizes,
    capturing patterns at multiple temporal scales.
    """

    def __init__(
            self,
            input_dim: int,
            feature_dim: int,
            kernel_sizes: tuple = (3, 5, 7),
            dropout: float = 0.1,
            name: Optional[str] = None,
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.input_shape = InputShape.SEQUENCE

        n_kernels = len(kernel_sizes)
        per_kernel_dim = feature_dim // n_kernels

        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, per_kernel_dim, k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.norm = nn.LayerNorm(per_kernel_dim * n_kernels)
        self.dropout = nn.Dropout(dropout)

        # Project to exact feature_dim if needed
        total_dim = per_kernel_dim * n_kernels
        if total_dim != feature_dim:
            self.out_proj = nn.Linear(total_dim, feature_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale convolution over sequence.

        Args:
            x: [B, S, input_dim] sequence

        Returns:
            [B, S, feature_dim]
        """
        # x: [B, S, D] -> [B, D, S] for conv1d
        x_t = x.transpose(1, 2)

        # Apply each conv and concatenate
        conv_outs = [conv(x_t) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)  # [B, D_out, S]

        # Back to [B, S, D]
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.out_proj(out)

        return out

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence to vector: [B, S, D] -> [B, D]"""
        return x.mean(dim=1)


class StreamBuilder:
    """
    Factory for building streams from specifications.

    Usage:
        stream = StreamBuilder.build(
            stream_type='feature_vector',
            input_dim=512,
            feature_dim=256,
            num_slots=16,
            name='clip_b32',
        )
    """

    @staticmethod
    def build(
            stream_type: str,
            input_dim: int,
            feature_dim: int,
            num_slots: int = 16,
            name: Optional[str] = None,
            **kwargs,
    ) -> nn.Module:
        """
        Build stream module from type string.

        Args:
            stream_type: One of 'feature_vector', 'trainable_vector', 
                        'sequence', 'transformer_sequence', 'conv_sequence'
            input_dim: Input feature dimension
            feature_dim: Output feature dimension
            num_slots: Number of slots for vector streams
            name: Optional stream name
            **kwargs: Additional arguments for specific stream types

        Returns:
            Stream module
        """
        if stream_type in ('feature', 'feature_vector'):
            return FeatureVectorStream(
                input_dim=input_dim,
                feature_dim=feature_dim,
                num_slots=num_slots,
                dropout=kwargs.get('dropout', 0.1),
                name=name,
            )

        elif stream_type == 'trainable_vector':
            return TrainableVectorStream(
                input_dim=input_dim,
                feature_dim=feature_dim,
                num_slots=num_slots,
                dropout=kwargs.get('dropout', 0.1),
                name=name,
            )

        elif stream_type in ('sequence', 'frozen'):
            return SequenceStream(
                input_dim=input_dim,
                feature_dim=feature_dim,
                name=name,
            )

        elif stream_type == 'transformer_sequence':
            return TransformerSequenceStream(
                input_dim=input_dim,
                feature_dim=feature_dim,
                num_layers=kwargs.get('num_layers', 2),
                num_heads=kwargs.get('num_heads', 8),
                dropout=kwargs.get('dropout', 0.1),
                name=name,
            )

        elif stream_type == 'conv_sequence':
            return ConvSequenceStream(
                input_dim=input_dim,
                feature_dim=feature_dim,
                kernel_sizes=kwargs.get('kernel_sizes', (3, 5, 7)),
                dropout=kwargs.get('dropout', 0.1),
                name=name,
            )

        else:
            raise ValueError(f"Unknown stream type: {stream_type}")


# =============================================================================
# LEGACY ALIASES
# =============================================================================

# For backward compatibility with older code
FrozenStream = FeatureVectorStream
FeatureStream = FeatureVectorStream
TrainableStream = TrainableVectorStream
VectorStream = FeatureVectorStream

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'InputShape',
    # Vector streams
    'FeatureVectorStream',
    'TrainableVectorStream',
    # Sequence streams
    'SequenceStream',
    'TransformerSequenceStream',
    'ConvSequenceStream',
    # Builder
    'StreamBuilder',
    # Legacy aliases
    'FrozenStream',
    'FeatureStream',
    'TrainableStream',
    'VectorStream',
]