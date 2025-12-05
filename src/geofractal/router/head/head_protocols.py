"""
geofractal.router.head.protocols
================================
Abstract protocols defining the head component interfaces.

These protocols enable:
- Static injection of custom components
- Easy swapping of implementations
- Type-safe composition
- Clear contracts for each component

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from abc import ABC, abstractmethod
from typing import Protocol, Tuple, Optional, Dict, Any, runtime_checkable
import torch
import torch.nn as nn


# =============================================================================
# CORE PROTOCOLS
# =============================================================================

@runtime_checkable
class Fingerprinted(Protocol):
    """Protocol for components that have a fingerprint."""
    fingerprint: nn.Parameter


@runtime_checkable
class AttentionHead(Protocol):
    """Protocol for attention mechanisms."""

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply attention.

        Args:
            x: [B, S, D] input
            mask: Optional attention mask
            return_weights: Whether to return attention weights

        Returns:
            output: [B, S, D] attended output
            weights: Optional [B, H, S, S] attention weights
        """
        ...


@runtime_checkable
class Router(Protocol):
    """Protocol for routing mechanisms."""

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route queries to values.

        Args:
            q: [B, S, D] queries
            k: [B, S, D] keys
            v: [B, S, D] values
            fingerprint: [F] identity vector

        Returns:
            routes: [B, S, K] selected indices
            weights: [B, S, K] route weights
            output: [B, S, D] routed output
        """
        ...


@runtime_checkable
class AnchorProvider(Protocol):
    """Protocol for anchor-based behavioral modes."""

    def forward(
            self,
            x: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anchor contribution.

        Args:
            x: [B, S, D] input
            fingerprint: [F] identity vector

        Returns:
            output: [B, S, D] anchor contribution
            affinities: [A] anchor activation weights
        """
        ...


@runtime_checkable
class GatingMechanism(Protocol):
    """Protocol for gating mechanisms."""

    def gate_values(
            self,
            v: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        """Gate values based on fingerprint."""
        ...

    def compute_similarity(
            self,
            fp_self: torch.Tensor,
            fp_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity between fingerprints."""
        ...


@runtime_checkable
class Combiner(Protocol):
    """Protocol for combining multiple signal sources."""

    def forward(
            self,
            signals: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Combine multiple signals.

        Args:
            signals: Named tensor dict (e.g., {'attention': x, 'routing': y})

        Returns:
            combined: [B, S, D] fused output
        """
        ...


@runtime_checkable
class Refinement(Protocol):
    """Protocol for output refinement (FFN-like)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine the combined output."""
        ...


# =============================================================================
# ABSTRACT BASE CLASSES
# =============================================================================

class BaseAttention(nn.Module, ABC):
    """Abstract base for attention implementations."""

    @abstractmethod
    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass


class BaseRouter(nn.Module, ABC):
    """Abstract base for routing implementations."""

    @abstractmethod
    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class BaseAnchorBank(nn.Module, ABC):
    """Abstract base for anchor implementations."""

    @abstractmethod
    def forward(
            self,
            x: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class BaseGate(nn.Module, ABC):
    """Abstract base for gating implementations."""

    @abstractmethod
    def gate_values(
            self,
            v: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_similarity(
            self,
            fp_self: torch.Tensor,
            fp_target: torch.Tensor,
    ) -> torch.Tensor:
        pass


class BaseCombiner(nn.Module, ABC):
    """Abstract base for signal combination."""

    @abstractmethod
    def forward(
            self,
            signals: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pass


class BaseRefinement(nn.Module, ABC):
    """Abstract base for output refinement."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


# =============================================================================
# HEAD COMPOSITION
# =============================================================================

class HeadComponents:
    """
    Container for head components.

    This is the decomposed head - each component can be
    individually replaced or extended.
    """

    def __init__(
            self,
            attention: AttentionHead,
            router: Router,
            anchors: AnchorProvider,
            gate: GatingMechanism,
            combiner: Combiner,
            refinement: Refinement,
            fingerprint_dim: int,
    ):
        self.attention = attention
        self.router = router
        self.anchors = anchors
        self.gate = gate
        self.combiner = combiner
        self.refinement = refinement
        self.fingerprint_dim = fingerprint_dim

    def validate(self) -> bool:
        """Validate all components implement required protocols."""
        checks = [
            isinstance(self.attention, AttentionHead),
            isinstance(self.router, Router),
            isinstance(self.anchors, AnchorProvider),
            isinstance(self.gate, GatingMechanism),
            isinstance(self.combiner, Combiner),
            isinstance(self.refinement, Refinement),
        ]
        return all(checks)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Protocols
    'Fingerprinted',
    'AttentionHead',
    'Router',
    'AnchorProvider',
    'GatingMechanism',
    'Combiner',
    'Refinement',
    # Abstract bases
    'BaseAttention',
    'BaseRouter',
    'BaseAnchorBank',
    'BaseGate',
    'BaseCombiner',
    'BaseRefinement',
    # Composition
    'HeadComponents',
]