"""
geofractal.router.streams.base
==============================
Abstract base class for streams.

Streams are the primary processing units in a collective:
- Encode input to features
- Expand to slots (for vectors) or pass through (for sequences)
- Route through internal head with mailbox coordination
- Pool output for fusion

The stream CONTAINS the head - this is not a separate component.
The mailbox is passed through forward() for inter-stream coordination.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, Union

from geofractal.router.head import HeadConfig, HeadBuilder, ComposedHead, build_standard_head
from geofractal.router.registry import get_registry


class InputShape:
    """Stream input shape types."""
    VECTOR = "vector"  # [B, D]
    SEQUENCE = "sequence"  # [B, S, D]
    IMAGE = "image"  # [B, C, H, W]


class BaseStream(nn.Module, ABC):
    """
    Abstract base for router streams.

    A stream is a complete processing unit that:
    1. Encodes input (backbone or projection)
    2. Prepares for routing (slot expansion for vectors)
    3. Routes through internal head with mailbox
    4. Returns routed output

    The stream OWNS its head - they are not separate.

    Interface matches original collective:
        routed, info = stream(input, mailbox, target_fingerprint)
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
        super().__init__()

        self.name = name
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.parent_id = parent_id
        self.cooperation_group = cooperation_group

        # Build or use provided head
        if head is not None:
            self.head = head
        else:
            _config = head_config or HeadConfig(feature_dim=feature_dim)
            self.head = build_standard_head(_config, name=f"{name}_head")

        # Register with global registry
        self.registry = get_registry()
        self._module_id = self.registry.register(
            name=name,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
            fingerprint_dim=self.head.config.fingerprint_dim,
            feature_dim=feature_dim,
        )

        # Set head's module_id to match
        self.head.module_id = self._module_id

    @property
    def module_id(self) -> str:
        """Unique identifier for this stream."""
        return self._module_id

    @property
    def fingerprint(self) -> torch.Tensor:
        """Access head's fingerprint."""
        return self.head.fingerprint

    @property
    @abstractmethod
    def input_shape(self) -> str:
        """Return 'vector', 'sequence', or 'image'."""
        pass

    @abstractmethod
    def encode(self, x: Any) -> torch.Tensor:
        """
        Encode input to features.

        For vectors: [B, D_in] → [B, D]
        For sequences: [B, S, D_in] → [B, S, D]
        For images: [B, C, H, W] → [B, D]
        """
        pass

    @abstractmethod
    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Prepare features for head routing.

        Head expects [B, S, D]. This handles:
        - Vector [B, D] → slot expansion → [B, num_slots, D]
        - Sequence [B, S, D] → passthrough
        """
        pass

    def forward(
            self,
            x: Any,
            mailbox: Optional[Any] = None,
            target_fingerprint: Optional[torch.Tensor] = None,
            return_info: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Full forward pass: encode → prepare → route through head.

        Args:
            x: Input tensor (shape depends on stream type)
            mailbox: RouterMailbox for inter-stream coordination
            target_fingerprint: Adjacent stream's fingerprint for gating
            return_info: Whether to return routing info

        Returns:
            output: [B, S, D] routed features
            info: Optional dict with routing metadata
        """
        # Encode input
        features = self.encode(x)

        # Prepare for head (slot expansion or passthrough)
        prepared = self.prepare_for_head(features)

        # Route through head with mailbox
        if return_info:
            output, head_info = self.head(
                prepared,
                mailbox=mailbox,
                target_fingerprint=target_fingerprint,
                return_info=True,
            )

            info = head_info or {}
            info['stream_name'] = self.name
            info['input_shape'] = self.input_shape
            info['encoded_shape'] = list(features.shape)
            info['prepared_shape'] = list(prepared.shape)

            return output, info
        else:
            output = self.head(
                prepared,
                mailbox=mailbox,
                target_fingerprint=target_fingerprint,
                return_info=False,
            )
            return output, None

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool [B, S, D] → [B, D].

        Used after routing to get fixed-size representation for fusion.
        """
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


__all__ = ['BaseStream', 'InputShape']