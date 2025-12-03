"""
geofractal.router.factory.protocols
===================================
Abstract protocols for router prototypes.

A prototype is a complete, runnable router system composed of:
- Streams (divergent feature extractors)
- Heads (routing decision makers)
- Fusion (combination strategy)
- Classifier (final prediction)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, List, Optional, Tuple, Any, runtime_checkable
import torch
import torch.nn as nn


# =============================================================================
# CORE PROTOCOLS
# =============================================================================

@runtime_checkable
class RouterPrototype(Protocol):
    """
    Protocol for complete router prototypes.

    A prototype is a fully assembled, trainable system that
    can process inputs and produce predictions.
    """

    def forward(
            self,
            x: torch.Tensor,
            return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Full forward pass.

        Args:
            x: Input tensor (images, features, etc.)
            return_info: Whether to return routing info

        Returns:
            logits: [B, num_classes] predictions
            info: Optional routing metadata
        """
        ...

    def get_stream_outputs(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get outputs from each stream."""
        ...

    def get_emergence_ratio(
            self,
            dataloader: Any,
    ) -> float:
        """Compute emergence ratio on dataset."""
        ...


@runtime_checkable
class Configurable(Protocol):
    """Protocol for configurable components."""

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict."""
        ...

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Configurable':
        """Construct from configuration dict."""
        ...


# =============================================================================
# ABSTRACT BASES
# =============================================================================

class BasePrototype(nn.Module, ABC):
    """
    Abstract base for router prototypes.

    Provides common functionality for all prototype implementations.
    """

    def __init__(
            self,
            num_classes: int,
            prototype_name: str = "prototype",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.prototype_name = prototype_name
        self._is_compiled = False

    @abstractmethod
    def forward(
            self,
            x: torch.Tensor,
            return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        pass

    @abstractmethod
    def get_stream_outputs(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pass

    def compile(self) -> 'BasePrototype':
        """Mark prototype as ready for training."""
        self._is_compiled = True
        return self

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_frozen_parameters(self) -> int:
        """Total frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def parameter_summary(self) -> Dict[str, int]:
        """Detailed parameter breakdown."""
        summary = {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': self.num_parameters,
            'frozen': self.num_frozen_parameters,
        }
        return summary

    def freeze_streams(self):
        """Freeze all stream parameters."""
        if hasattr(self, 'streams'):
            for stream in self.streams.values():
                for param in stream.parameters():
                    param.requires_grad = False

    def unfreeze_streams(self):
        """Unfreeze all stream parameters."""
        if hasattr(self, 'streams'):
            for stream in self.streams.values():
                for param in stream.parameters():
                    param.requires_grad = True


# =============================================================================
# PROTOTYPE INFO
# =============================================================================

class PrototypeInfo:
    """Container for prototype metadata and routing info."""

    def __init__(
            self,
            stream_outputs: Optional[Dict[str, torch.Tensor]] = None,
            head_outputs: Optional[Dict[str, torch.Tensor]] = None,
            fusion_weights: Optional[torch.Tensor] = None,
            routing_info: Optional[Dict[str, Any]] = None,
            fingerprints: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.stream_outputs = stream_outputs or {}
        self.head_outputs = head_outputs or {}
        self.fusion_weights = fusion_weights
        self.routing_info = routing_info or {}
        self.fingerprints = fingerprints or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stream_outputs': self.stream_outputs,
            'head_outputs': self.head_outputs,
            'fusion_weights': self.fusion_weights,
            'routing_info': self.routing_info,
            'fingerprints': self.fingerprints,
        }


# =============================================================================
# STREAM SPEC
# =============================================================================

class StreamSpec:
    """
    Specification for a stream in the prototype.

    Captures all information needed to construct a stream.
    """

    def __init__(
            self,
            name: str,
            stream_type: str,  # 'frozen', 'trainable', 'feature'
            model_name: Optional[str] = None,
            feature_dim: int = 512,
            frozen: bool = True,
            pretrained: bool = True,
            **kwargs,
    ):
        self.name = name
        self.stream_type = stream_type
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.frozen = frozen
        self.pretrained = pretrained
        self.kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'stream_type': self.stream_type,
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'frozen': self.frozen,
            'pretrained': self.pretrained,
            **self.kwargs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StreamSpec':
        return cls(**d)

    @classmethod
    def frozen_clip(cls, name: str, variant: str = "openai/clip-vit-base-patch32") -> 'StreamSpec':
        """Create spec for frozen CLIP stream."""
        dim_map = {
            "openai/clip-vit-base-patch32": 512,
            "openai/clip-vit-base-patch16": 512,
            "openai/clip-vit-large-patch14": 768,
            "openai/clip-vit-large-patch14-336": 768,
        }
        return cls(
            name=name,
            stream_type='frozen',
            model_name=variant,
            feature_dim=dim_map.get(variant, 512),
            frozen=True,
        )

    @classmethod
    def feature_stream(cls, name: str, input_dim: int, feature_dim: int = 512) -> 'StreamSpec':
        """Create spec for feature projection stream."""
        return cls(
            name=name,
            stream_type='feature',
            feature_dim=feature_dim,
            input_dim=input_dim,
            frozen=False,
        )


# =============================================================================
# HEAD SPEC
# =============================================================================

class HeadSpec:
    """
    Specification for a head in the prototype.

    Captures configuration for head construction.
    """

    def __init__(
            self,
            feature_dim: int = 512,
            fingerprint_dim: int = 64,
            num_heads: int = 8,
            num_anchors: int = 16,
            num_routes: int = 4,
            use_cantor: bool = True,
            attention_type: str = 'cantor',  # 'cantor', 'standard'
            router_type: str = 'topk',  # 'topk', 'soft'
            anchor_type: str = 'constitutive',  # 'constitutive', 'attentive'
            **kwargs,
    ):
        self.feature_dim = feature_dim
        self.fingerprint_dim = fingerprint_dim
        self.num_heads = num_heads
        self.num_anchors = num_anchors
        self.num_routes = num_routes
        self.use_cantor = use_cantor
        self.attention_type = attention_type
        self.router_type = router_type
        self.anchor_type = anchor_type
        self.kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_dim': self.feature_dim,
            'fingerprint_dim': self.fingerprint_dim,
            'num_heads': self.num_heads,
            'num_anchors': self.num_anchors,
            'num_routes': self.num_routes,
            'use_cantor': self.use_cantor,
            'attention_type': self.attention_type,
            'router_type': self.router_type,
            'anchor_type': self.anchor_type,
            **self.kwargs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HeadSpec':
        return cls(**d)

    @classmethod
    def standard(cls, feature_dim: int = 512) -> 'HeadSpec':
        """Standard head spec (proven on ImageNet)."""
        return cls(
            feature_dim=feature_dim,
            fingerprint_dim=64,
            num_anchors=16,
            num_routes=4,
            use_cantor=True,
            attention_type='cantor',
            router_type='topk',
            anchor_type='constitutive',
        )

    @classmethod
    def lightweight(cls, feature_dim: int = 512) -> 'HeadSpec':
        """Lightweight head spec."""
        return cls(
            feature_dim=feature_dim,
            fingerprint_dim=32,
            num_anchors=8,
            num_routes=2,
            use_cantor=False,
            attention_type='standard',
            router_type='soft',
            anchor_type='constitutive',
        )


# =============================================================================
# FUSION SPEC
# =============================================================================

class FusionSpec:
    """
    Specification for fusion layer.
    """

    def __init__(
            self,
            strategy: str = 'concat',  # 'concat', 'weighted', 'gated', 'attention', etc.
            output_dim: int = 512,
            dropout: float = 0.1,
            num_heads: int = 8,
            temperature: float = 1.0,
            **kwargs,
    ):
        self.strategy = strategy
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.temperature = temperature
        self.kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'num_heads': self.num_heads,
            'temperature': self.temperature,
            **self.kwargs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FusionSpec':
        return cls(**d)

    @classmethod
    def standard(cls, output_dim: int = 512) -> 'FusionSpec':
        """Standard concat fusion."""
        return cls(strategy='concat', output_dim=output_dim)

    @classmethod
    def adaptive(cls, output_dim: int = 512) -> 'FusionSpec':
        """Adaptive gated fusion."""
        return cls(strategy='gated', output_dim=output_dim)

    @classmethod
    def attention(cls, output_dim: int = 512, num_heads: int = 8) -> 'FusionSpec':
        """Attention fusion."""
        return cls(strategy='attention', output_dim=output_dim, num_heads=num_heads)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Protocols
    'RouterPrototype',
    'Configurable',
    # Abstract bases
    'BasePrototype',
    # Info containers
    'PrototypeInfo',
    # Specs
    'StreamSpec',
    'HeadSpec',
    'FusionSpec',
]