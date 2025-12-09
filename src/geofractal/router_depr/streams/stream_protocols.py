"""
geofractal.router.factory.protocols
===================================
Abstract protocols for router prototypes.

Updated to support new stream architecture with input shapes.

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
class StreamProtocol(Protocol):
    """
    Protocol that all streams must implement.

    Streams are the primary processing units:
    - Encode input to features
    - Expand to slots (vectors) or passthrough (sequences)
    - Route through internal head with mailbox coordination
    - Return routed output [B, S, D]
    """

    name: str
    input_dim: int
    feature_dim: int
    fingerprint: nn.Parameter

    @property
    def input_shape(self) -> str:
        """Return 'vector', 'sequence', or 'image'."""
        ...

    @property
    def module_id(self) -> str:
        """Unique identifier for mailbox communication."""
        ...

    def encode(self, x: Any) -> torch.Tensor:
        """Encode input to features."""
        ...

    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        """Prepare features for head (slot expansion or passthrough)."""
        ...

    def forward(
            self,
            x: Any,
            mailbox: Optional[Any] = None,
            target_fingerprint: Optional[torch.Tensor] = None,
            return_info: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Full forward: encode → prepare → route through head."""
        ...

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool [B, S, D] → [B, D] for fusion."""
        ...


@runtime_checkable
class HeadProtocol(Protocol):
    """
    Protocol that all heads must implement.

    Heads perform routing with mailbox coordination.
    """

    fingerprint: nn.Parameter
    module_id: str

    def forward(
            self,
            x: torch.Tensor,
            mailbox: Optional[Any] = None,
            target_fingerprint: Optional[torch.Tensor] = None,
            skip_first: bool = False,
            return_info: bool = False,
    ) -> torch.Tensor:
        """Route input with mailbox coordination."""
        ...


@runtime_checkable
class FusionProtocol(Protocol):
    """Protocol for fusion methods."""

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fuse stream outputs: Dict[name, [B, D]] → [B, D]."""
        ...


@runtime_checkable
class RouterPrototype(Protocol):
    """Protocol for complete router prototypes."""

    def forward(
            self,
            x: torch.Tensor,
            return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        ...

    def get_stream_outputs(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ...

    def get_emergence_ratio(
            self,
            dataloader: Any,
    ) -> float:
        ...


@runtime_checkable
class Configurable(Protocol):
    """Protocol for configurable components."""

    def get_config(self) -> Dict[str, Any]:
        ...

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Configurable':
        ...


# =============================================================================
# INPUT SHAPE ENUM
# =============================================================================

class InputShape:
    """Stream input shape types."""
    VECTOR = "vector"  # [B, D] - pooled features
    SEQUENCE = "sequence"  # [B, S, D] - token sequences
    IMAGE = "image"  # [B, C, H, W] - raw images


# =============================================================================
# ABSTRACT BASES
# =============================================================================

class BasePrototype(nn.Module, ABC):
    """Abstract base for router prototypes."""

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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def parameter_summary(self) -> Dict[str, int]:
        return {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': self.num_parameters,
            'frozen': self.num_frozen_parameters,
        }

    def freeze_streams(self):
        if hasattr(self, 'streams'):
            for stream in self.streams.values():
                for param in stream.parameters():
                    param.requires_grad = False

    def unfreeze_streams(self):
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

    Supports both vector [B, D] and sequence [B, S, D] inputs.
    Streams are simple projections - no artificial structure manufacturing.
    """

    def __init__(
            self,
            name: str,
            stream_type: str,
            # 'feature_vector', 'trainable_vector', 'sequence', 'transformer_sequence', 'conv_sequence', 'frozen'
            input_shape: str = InputShape.VECTOR,  # 'vector', 'sequence', 'image'
            input_dim: int = 512,
            feature_dim: int = 512,
            model_name: Optional[str] = None,
            frozen: bool = False,
            num_layers: int = 2,  # For transformer streams
            num_heads: int = 8,  # For transformer streams
            kernel_sizes: List[int] = None,  # For conv streams
            backbone: Optional[nn.Module] = None,  # For trainable streams
            **kwargs,
    ):
        self.name = name
        self.stream_type = stream_type
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.model_name = model_name
        self.frozen = frozen
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes or [3, 5, 7]
        self.backbone = backbone
        self.kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'stream_type': self.stream_type,
            'input_shape': self.input_shape,
            'input_dim': self.input_dim,
            'feature_dim': self.feature_dim,
            'model_name': self.model_name,
            'frozen': self.frozen,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'kernel_sizes': self.kernel_sizes,
            # backbone is not serializable
            **self.kwargs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StreamSpec':
        return cls(**d)

    # === VECTOR STREAM FACTORIES ===

    @classmethod
    def feature_vector(
            cls,
            name: str,
            input_dim: int,
            feature_dim: int = 512,
    ) -> 'StreamSpec':
        """Pre-extracted features, no backbone. Simple projection."""
        return cls(
            name=name,
            stream_type='feature_vector',
            input_shape=InputShape.VECTOR,
            input_dim=input_dim,
            feature_dim=feature_dim,
        )

    @classmethod
    def trainable_vector(
            cls,
            name: str,
            input_dim: int,
            feature_dim: int = 512,
            backbone: Optional[nn.Module] = None,
    ) -> 'StreamSpec':
        """Trainable backbone for vector features."""
        return cls(
            name=name,
            stream_type='trainable_vector',
            input_shape=InputShape.VECTOR,
            input_dim=input_dim,
            feature_dim=feature_dim,
            backbone=backbone,
        )

    @classmethod
    def frozen_encoder(
            cls,
            name: str,
            model_name: str,
            feature_dim: int = 512,
    ) -> 'StreamSpec':
        """Frozen pretrained encoder (CLIP, etc)."""
        return cls(
            name=name,
            stream_type='frozen',
            input_shape=InputShape.IMAGE,
            input_dim=feature_dim,  # Output dim of encoder
            feature_dim=feature_dim,
            model_name=model_name,
            frozen=True,
        )

    # === SEQUENCE STREAM FACTORIES ===

    @classmethod
    def sequence(
            cls,
            name: str,
            input_dim: int,
            feature_dim: int = 512,
    ) -> 'StreamSpec':
        """Basic sequence stream with optional projection."""
        return cls(
            name=name,
            stream_type='sequence',
            input_shape=InputShape.SEQUENCE,
            input_dim=input_dim,
            feature_dim=feature_dim,
        )

    @classmethod
    def transformer_sequence(
            cls,
            name: str,
            input_dim: int,
            feature_dim: int = 512,
            num_layers: int = 2,
            num_heads: int = 8,
    ) -> 'StreamSpec':
        """Sequence stream with transformer backbone."""
        return cls(
            name=name,
            stream_type='transformer_sequence',
            input_shape=InputShape.SEQUENCE,
            input_dim=input_dim,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

    @classmethod
    def conv_sequence(
            cls,
            name: str,
            input_dim: int,
            feature_dim: int = 512,
            kernel_sizes: List[int] = None,
    ) -> 'StreamSpec':
        """Sequence stream with multi-scale conv backbone."""
        return cls(
            name=name,
            stream_type='conv_sequence',
            input_shape=InputShape.SEQUENCE,
            input_dim=input_dim,
            feature_dim=feature_dim,
            kernel_sizes=kernel_sizes or [3, 5, 7],
        )

    # === LEGACY FACTORIES (for backwards compatibility) ===

    @classmethod
    def feature_stream(
            cls,
            name: str,
            input_dim: int,
            feature_dim: int = 512,
    ) -> 'StreamSpec':
        """DEPRECATED: Use feature_vector instead."""
        return cls.feature_vector(name, input_dim, feature_dim)


# =============================================================================
# HEAD SPEC
# =============================================================================

class HeadSpec:
    """Specification for a head in the prototype."""

    def __init__(
            self,
            feature_dim: int = 512,
            fingerprint_dim: int = 64,
            num_heads: int = 8,
            num_anchors: int = 16,
            num_routes: int = 4,
            use_cantor: bool = True,
            attention_type: str = 'cantor',
            router_type: str = 'topk',
            anchor_type: str = 'constitutive',
            gate_type: str = 'fingerprint',
            combiner_type: str = 'learnable_weight',
            refinement_type: str = 'ffn',
            preset: str = 'standard',  # 'standard', 'lightweight', 'heavy'
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
        self.gate_type = gate_type
        self.combiner_type = combiner_type
        self.refinement_type = refinement_type
        self.preset = preset
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
            'gate_type': self.gate_type,
            'combiner_type': self.combiner_type,
            'refinement_type': self.refinement_type,
            'preset': self.preset,
            **self.kwargs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HeadSpec':
        return cls(**d)

    @classmethod
    def standard(cls, feature_dim: int = 512) -> 'HeadSpec':
        """Standard head (proven on ImageNet)."""
        return cls(
            feature_dim=feature_dim,
            preset='standard',
            attention_type='cantor',
            router_type='topk',
            anchor_type='constitutive',
        )

    @classmethod
    def lightweight(cls, feature_dim: int = 512) -> 'HeadSpec':
        """Lightweight head (faster)."""
        return cls(
            feature_dim=feature_dim,
            preset='lightweight',
            fingerprint_dim=32,
            num_anchors=8,
            num_routes=2,
            attention_type='standard',
            router_type='soft',
        )

    @classmethod
    def heavy(cls, feature_dim: int = 512) -> 'HeadSpec':
        """Heavy head (maximum capacity)."""
        return cls(
            feature_dim=feature_dim,
            preset='heavy',
            num_anchors=32,
            num_routes=8,
            anchor_type='attentive',
            gate_type='channel',
            combiner_type='gated',
            refinement_type='moe',
        )


# =============================================================================
# FUSION SPEC
# =============================================================================

class FusionSpec:
    """Specification for fusion layer."""

    def __init__(
            self,
            strategy: str = 'concat',
            output_dim: int = 512,
            dropout: float = 0.1,
            num_heads: int = 8,
            temperature: float = 1.0,
            num_experts: int = 4,
            top_k: int = 2,
            fingerprint_dim: int = 64,
            **kwargs,
    ):
        self.strategy = strategy
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.temperature = temperature
        self.num_experts = num_experts
        self.top_k = top_k
        self.fingerprint_dim = fingerprint_dim
        self.kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'num_heads': self.num_heads,
            'temperature': self.temperature,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'fingerprint_dim': self.fingerprint_dim,
            **self.kwargs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FusionSpec':
        return cls(**d)

    @classmethod
    def concat(cls, output_dim: int = 512) -> 'FusionSpec':
        """Simple concatenation (baseline)."""
        return cls(strategy='concat', output_dim=output_dim)

    @classmethod
    def weighted(cls, output_dim: int = 512) -> 'FusionSpec':
        """Learnable static weights."""
        return cls(strategy='weighted', output_dim=output_dim)

    @classmethod
    def gated(cls, output_dim: int = 512) -> 'FusionSpec':
        """Input-adaptive gating."""
        return cls(strategy='gated', output_dim=output_dim)

    @classmethod
    def attention(cls, output_dim: int = 512, num_heads: int = 8) -> 'FusionSpec':
        """Cross-stream attention."""
        return cls(strategy='attention', output_dim=output_dim, num_heads=num_heads)

    @classmethod
    def fingerprint(cls, output_dim: int = 512, fingerprint_dim: int = 64) -> 'FusionSpec':
        """Fingerprint-guided fusion."""
        return cls(strategy='fingerprint', output_dim=output_dim, fingerprint_dim=fingerprint_dim)

    @classmethod
    def moe(cls, output_dim: int = 512, num_experts: int = 4, top_k: int = 2) -> 'FusionSpec':
        """Mixture of experts."""
        return cls(strategy='moe', output_dim=output_dim, num_experts=num_experts, top_k=top_k)

    # Legacy
    @classmethod
    def standard(cls, output_dim: int = 512) -> 'FusionSpec':
        return cls.concat(output_dim)

    @classmethod
    def adaptive(cls, output_dim: int = 512) -> 'FusionSpec':
        return cls.gated(output_dim)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Protocols
    'StreamProtocol',
    'HeadProtocol',
    'FusionProtocol',
    'RouterPrototype',
    'Configurable',
    # Abstract bases
    'BasePrototype',
    # Input shapes
    'InputShape',
    # Info containers
    'PrototypeInfo',
    # Specs
    'StreamSpec',
    'HeadSpec',
    'FusionSpec',
]