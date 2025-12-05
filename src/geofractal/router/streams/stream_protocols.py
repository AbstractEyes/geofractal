# streams/protocols.py
"""
Stream protocol definitions.

Streams are simple: encode input → prepare for head.
No artificial structure manufacturing.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Protocol, Tuple, Dict, Any, Optional, runtime_checkable
import torch


class InputShape:
    """Stream input shape types."""
    VECTOR = "vector"  # [B, D]
    SEQUENCE = "sequence"  # [B, S, D]
    IMAGE = "image"  # [B, C, H, W]


@runtime_checkable
class StreamProtocol(Protocol):
    """What all streams must implement."""

    name: str
    input_shape: str  # One of InputShape values

    def encode(self, x: Any) -> torch.Tensor:
        """Encode input to features."""
        ...

    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Prepare features for head consumption.

        Head expects [B, S, D]. This method handles the conversion:
        - Vector [B, D] → [B, 1, D] via unsqueeze
        - Sequence [B, S, D] → [B, S, D] passthrough
        """
        ...

    def forward(
            self,
            x: Any,
            return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Full forward: encode → prepare → head."""
        ...

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool [B, S, D] → [B, D]."""
        ...


__all__ = [
    'InputShape',
    'StreamProtocol',
]