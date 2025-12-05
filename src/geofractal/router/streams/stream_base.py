# streams/base.py
"""
Abstract base class for streams.

Streams are simple transformers:
- Encode input to features
- Prepare features for head (shape conversion)
- Pass through head
- Optionally pool output

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any

from .stream_protocols import InputShape


class BaseStream(nn.Module, ABC):
    """
    Abstract base for router streams.

    Streams transform inputs into features suitable for heads.
    They do NOT manufacture artificial structure (no slot expansion).

    Shape handling:
    - Vector inputs [B, D] → unsqueeze → [B, 1, D] for head
    - Sequence inputs [B, S, D] → passthrough for head
    """

    def __init__(
            self,
            name: str,
            input_dim: int,
            feature_dim: int,
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.feature_dim = feature_dim

    @property
    @abstractmethod
    def input_shape(self) -> str:
        """Return 'vector', 'sequence', or 'image'."""
        pass

    @abstractmethod
    def encode(self, x: Any) -> torch.Tensor:
        """
        Encode input to features.

        Returns:
            Vector streams: [B, D]
            Sequence streams: [B, S, D]
        """
        pass

    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Prepare features for head consumption.

        Head expects [B, S, D]. Default behavior:
        - If [B, D]: unsqueeze to [B, 1, D]
        - If [B, S, D]: passthrough
        """
        if features.dim() == 2:
            return features.unsqueeze(1)  # [B, D] → [B, 1, D]
        return features  # [B, S, D] passthrough

    def forward(
            self,
            x: Any,
            return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Full forward pass: encode → prepare for head.

        Note: This does NOT pass through a head. The prototype/router
        orchestrates head application separately.

        Returns:
            features: [B, S, D] ready for head
            info: Optional metadata dict
        """
        features = self.encode(x)
        prepared = self.prepare_for_head(features)

        info = None
        if return_info:
            info = {
                'stream_name': self.name,
                'input_shape': self.input_shape,
                'encoded_shape': list(features.shape),
                'prepared_shape': list(prepared.shape),
            }

        return prepared, info

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool [B, S, D] → [B, D]."""
        if x.dim() == 3:
            return x.mean(dim=1)
        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"input_dim={self.input_dim}, "
            f"feature_dim={self.feature_dim}, "
            f"input_shape='{self.input_shape}')"
        )


__all__ = ['BaseStream']