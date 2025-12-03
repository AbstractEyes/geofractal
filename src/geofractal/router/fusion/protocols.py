"""
geofractal.router.fusion.protocols
==================================
Abstract protocols defining fusion interfaces.

Fusion is the critical junction where divergent streams become
collective intelligence. These protocols ensure any fusion
strategy can be cleanly swapped.

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
class StreamFusion(Protocol):
    """
    Protocol for fusing multiple stream outputs.

    The fusion layer receives pooled outputs from each stream
    and must combine them into a unified representation.
    """

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Fuse stream outputs.

        Args:
            stream_outputs: {stream_name: [B, D]} pooled features
            stream_fingerprints: Optional {stream_name: [F]} fingerprints
            return_weights: Whether to return fusion weights/info

        Returns:
            fused: [B, D_out] fused representation
            info: Optional fusion metadata (weights, attention, etc.)
        """
        ...


@runtime_checkable
class AdaptiveFusion(Protocol):
    """
    Protocol for fusion that adapts based on input.

    Unlike static fusion, adaptive fusion computes different
    fusion weights for each input sample.
    """

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptively fuse based on input.

        Args:
            stream_outputs: {stream_name: [B, D]} pooled features
            context: Optional [B, D_ctx] context for adaptation

        Returns:
            fused: [B, D_out] fused representation
            weights: [B, N] per-sample fusion weights
        """
        ...


@runtime_checkable
class HierarchicalFusion(Protocol):
    """
    Protocol for tree-structured fusion.

    Enables hierarchical combination where related streams
    are fused first, then combined at higher levels.
    """

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            hierarchy: Dict[str, List[str]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hierarchically fuse according to tree structure.

        Args:
            stream_outputs: {stream_name: [B, D]} pooled features
            hierarchy: {parent: [children]} tree structure

        Returns:
            fused: [B, D_out] final fused representation
            intermediate: {node: [B, D]} intermediate fusion results
        """
        ...


# =============================================================================
# ABSTRACT BASES
# =============================================================================

class BaseFusion(nn.Module, ABC):
    """Abstract base for all fusion implementations."""

    def __init__(self, stream_dims: Dict[str, int], output_dim: int):
        """
        Args:
            stream_dims: {stream_name: feature_dim} for each stream
            output_dim: Dimension of fused output
        """
        super().__init__()
        self.stream_dims = stream_dims
        self.stream_names = list(stream_dims.keys())
        self.num_streams = len(stream_dims)
        self.output_dim = output_dim
        self.total_input_dim = sum(stream_dims.values())

    @abstractmethod
    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        pass

    def _stack_outputs(
            self,
            stream_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Stack outputs in consistent order."""
        return torch.stack(
            [stream_outputs[name] for name in self.stream_names],
            dim=1
        )  # [B, N, D]

    def _concat_outputs(
            self,
            stream_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Concatenate outputs in consistent order."""
        return torch.cat(
            [stream_outputs[name] for name in self.stream_names],
            dim=-1
        )  # [B, sum(D)]


class BaseAdaptiveFusion(BaseFusion, ABC):
    """Abstract base for adaptive fusion."""

    @abstractmethod
    def compute_weights(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-sample fusion weights."""
        pass


# =============================================================================
# FUSION INFO CONTAINERS
# =============================================================================

class FusionInfo:
    """Container for fusion metadata."""

    def __init__(
            self,
            weights: Optional[torch.Tensor] = None,
            attention: Optional[torch.Tensor] = None,
            intermediate: Optional[Dict[str, torch.Tensor]] = None,
            method: str = "unknown",
    ):
        self.weights = weights
        self.attention = attention
        self.intermediate = intermediate
        self.method = method

    def to_dict(self) -> Dict[str, Any]:
        return {
            'weights': self.weights,
            'attention': self.attention,
            'intermediate': self.intermediate,
            'method': self.method,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Protocols
    'StreamFusion',
    'AdaptiveFusion',
    'HierarchicalFusion',
    # Abstract bases
    'BaseFusion',
    'BaseAdaptiveFusion',
    # Info containers
    'FusionInfo',
]