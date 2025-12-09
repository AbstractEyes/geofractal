"""
geofractal.router.streams.sequence
==================================
Sequence stream implementations.

For [B, S, D] sequence inputs. These pass through naturally
since the router expects [B, S, D].

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
from typing import Optional, List

from .stream_base import BaseStream, InputShape
from geofractal.router.head import HeadConfig, ComposedHead


class SequenceStream(BaseStream):
    """
    Stream for [B, S, D] sequence inputs.

    Projects to feature_dim if needed, passes through otherwise.
    No slot expansion needed - sequences already have S > 1.

    Use for transformer embeddings, token sequences, time series.
    """

    def __init__(
            self,
            name: str,
            input_dim: int,
            feature_dim: int,
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

        # Project if dimensions differ
        if input_dim != feature_dim:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )
        else:
            self.projection = nn.Identity()

    @property
    def input_shape(self) -> str:
        return InputShape.SEQUENCE

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project sequence to feature dimension.

        Args:
            x: [B, S, input_dim] sequence

        Returns:
            [B, S, feature_dim]
        """
        return self.projection(x)

    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        """Sequences are already [B, S, D] - passthrough."""
        return features


class TransformerSequenceStream(SequenceStream):
    """
    Sequence stream with transformer encoder layers.

    For sequences that need self-attention before routing.
    """

    def __init__(
            self,
            name: str,
            input_dim: int,
            feature_dim: int,
            num_layers: int = 2,
            num_heads: int = 8,
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project and process through transformer.

        Args:
            x: [B, S, input_dim] sequence

        Returns:
            [B, S, feature_dim]
        """
        x = self.projection(x)
        return self.encoder(x)


class ConvSequenceStream(SequenceStream):
    """
    Multi-scale 1D convolution stream for sequences.

    Applies parallel convolutions with different kernel sizes,
    capturing patterns at multiple temporal scales.
    """

    def __init__(
            self,
            name: str,
            input_dim: int,
            feature_dim: int,
            kernel_sizes: List[int] = None,
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

        kernel_sizes = kernel_sizes or [3, 5, 7]

        self.convs = nn.ModuleList([
            nn.Conv1d(feature_dim, feature_dim, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.gate = nn.Linear(feature_dim * len(kernel_sizes), feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale convolution over sequence.

        Args:
            x: [B, S, input_dim] sequence

        Returns:
            [B, S, feature_dim]
        """
        # Project first
        x = self.projection(x)  # [B, S, D]

        # Conv expects [B, D, S]
        x_t = x.transpose(1, 2)

        # Apply each conv and concatenate
        conv_outs = [torch.relu(conv(x_t)) for conv in self.convs]
        concat = torch.cat(conv_outs, dim=1).transpose(1, 2)  # [B, S, D*K]

        # Gate and residual
        gated = self.gate(concat)  # [B, S, D]
        gated = self.dropout(gated)

        return self.norm(gated + x)


__all__ = [
    'SequenceStream',
    'TransformerSequenceStream',
    'ConvSequenceStream',
]