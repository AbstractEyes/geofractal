# streams/sequence.py
"""Streams for [B, S, D] sequence inputs."""

import torch
import torch.nn as nn
from typing import Optional

from geofractal.router.config import CollectiveConfig
from geofractal.router.head import HeadConfig
from geofractal.router.head.builder import ComposedHead
from geofractal.router.streams.base import BaseStream
from geofractal.router.streams.protocols import InputShape


class SequenceStream(BaseStream):
    """
    Stream for [B, S, D] sequence inputs.
    Routes tokens directly - no slot expansion.
    """

    def __init__(
        self,
        config: CollectiveConfig,
        name: str,
        input_dim: int,
        head: Optional[ComposedHead] = None,
        head_config: Optional[HeadConfig] = None,
    ):
        super().__init__(config, name, head, head_config)

        self.input_dim = input_dim

        # Project if dimensions differ
        if input_dim != config.feature_dim:
            self.projection = nn.Linear(input_dim, config.feature_dim)
        else:
            self.projection = nn.Identity()

    @property
    def input_shape(self) -> str:
        return InputShape.SEQUENCE

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        return features  # Already [B, S, D]


class TransformerSequenceStream(SequenceStream):
    """Sequence stream with transformer layers."""

    def __init__(
        self,
        config: CollectiveConfig,
        name: str,
        input_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        **kwargs,
    ):
        super().__init__(config, name, input_dim, **kwargs)

        layer = nn.TransformerEncoderLayer(
            d_model=config.feature_dim,
            nhead=num_heads,
            dim_feedforward=config.feature_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return self.transformer(x)


class ConvSequenceStream(SequenceStream):
    """Sequence stream with multi-scale convolutions."""

    def __init__(
        self,
        config: CollectiveConfig,
        name: str,
        input_dim: int,
        kernel_sizes: list = [3, 5, 7],
        **kwargs,
    ):
        super().__init__(config, name, input_dim, **kwargs)

        D = config.feature_dim
        self.convs = nn.ModuleList([
            nn.Conv1d(D, D, k, padding=k//2) for k in kernel_sizes
        ])
        self.gate = nn.Linear(D * len(kernel_sizes), D)
        self.norm = nn.LayerNorm(D)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)          # [B, S, D]
        x_t = x.transpose(1, 2)         # [B, D, S]

        outs = [torch.relu(c(x_t)) for c in self.convs]
        concat = torch.cat(outs, dim=1).transpose(1, 2)  # [B, S, D*3]

        gated = self.gate(concat)       # [B, S, D]
        return self.norm(gated + x)