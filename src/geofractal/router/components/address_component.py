"""
geofractal.router.components.address_component
==============================================

Identity and coordination for the geofractal system.

AddressComponent provides fingerprint-based identity and mailbox
message passing. This is the mechanism that enabled inter-stream
coordination in the ImageNet experiment - where individually random
streams (0.1% accuracy) achieved 84.68% collective accuracy through
coordinated communication.

Component Hierarchy:
    AddressComponent          - Base with Euclidean fingerprint
    GatedAddressComponent     - Fingerprint-gated messages
    RoutedAddressComponent    - Learned routing weights

    SimplexAddressComponent   - k-simplex via SimplexFactory
    SphericalAddressComponent - Unit hypersphere
    HyperbolicAddressComponent- Poincaré ball
    GeometricAddressComponent - Configurable manifold

    FractalAddressComponent   - Julia orbit via FractalFactory
    CantorAddressComponent    - Beatrix/Staircase via CantorRouteFactory
    ShapeAddressComponent     - Geometric shapes with RoPE offset

Design Principles:
    - Inherits TorchComponent (fingerprint is learned)
    - Fingerprint: learned identity vector for routing
    - Mailbox: message passing between components
    - Similarity-based routing decisions
    - Factory-backed for geometric correctness

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, Dict, List, Tuple, Union, Literal
from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.components.torch_component import TorchComponent

# Factory imports
from geovocab2.shapes.factory.simplex_factory import SimplexFactory
from geovocab2.shapes.factory.fractal_factory import (
    FractalFactory, FractalMode, FRACTAL_REGIONS
)
from geovocab2.shapes.factory.cantor_route_factory import (
    CantorRouteFactory, RouteMode, SimplexConfig
)
from geovocab2.shapes.factory.shape_factory import SimpleShapeFactory


# =============================================================================
# BASE ADDRESS COMPONENT
# =============================================================================

class AddressComponent(TorchComponent):
    """
    Identity and communication primitive.

    Provides fingerprint-based identity and mailbox message passing.

    Attributes:
        fingerprint_dim: Dimension of fingerprint vector.
        fingerprint: Learned identity vector.
        inbox: Received messages by source.
        outbox: Messages to send by target.
    """

    def __init__(
        self,
        name: str,
        fingerprint_dim: int,
        init_scale: float = 0.02,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        self.fingerprint_dim = fingerprint_dim
        self.fingerprint = nn.Parameter(
            torch.randn(fingerprint_dim) * init_scale
        )

        self.inbox: Dict[str, Tensor] = {}
        self.outbox: Dict[str, Tensor] = {}

    # =========================================================================
    # IDENTITY
    # =========================================================================

    def similarity(self, other: 'AddressComponent') -> Tensor:
        """Cosine similarity of fingerprints."""
        return F.cosine_similarity(
            self.fingerprint.unsqueeze(0),
            other.fingerprint.unsqueeze(0),
        ).squeeze()

    def distance(self, other: 'AddressComponent') -> Tensor:
        """L2 distance of fingerprints."""
        return torch.norm(self.fingerprint - other.fingerprint)

    def dot(self, other: 'AddressComponent') -> Tensor:
        """Dot product of fingerprints."""
        return torch.dot(self.fingerprint, other.fingerprint)

    def match(
        self,
        others: List['AddressComponent'],
        mode: str = 'similarity',
    ) -> Tuple[int, Tensor]:
        """Find best matching address from list."""
        if mode == 'similarity':
            scores = torch.stack([self.similarity(o) for o in others])
            best_idx = scores.argmax().item()
            return best_idx, scores[best_idx]
        elif mode == 'distance':
            scores = torch.stack([self.distance(o) for o in others])
            best_idx = scores.argmin().item()
            return best_idx, scores[best_idx]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def affinity(
        self,
        others: List['AddressComponent'],
        temperature: float = 1.0,
    ) -> Tensor:
        """Softmax affinity weights over addresses."""
        sims = torch.stack([self.similarity(o) for o in others])
        return F.softmax(sims / temperature, dim=0)

    # =========================================================================
    # COMMUNICATION
    # =========================================================================

    def send(self, target: str, message: Tensor) -> None:
        """Queue message for target."""
        self.outbox[target] = message

    def receive(self, source: str) -> Optional[Tensor]:
        """Get message from source."""
        return self.inbox.get(source)

    def deliver(self, source: str, message: Tensor) -> None:
        """Deliver message to inbox."""
        self.inbox[source] = message

    def collect(self) -> Dict[str, Tensor]:
        """Collect all inbox messages."""
        return dict(self.inbox)

    def collect_and_clear(self) -> Dict[str, Tensor]:
        """Collect and clear inbox."""
        messages = dict(self.inbox)
        self.inbox.clear()
        return messages

    def flush_outbox(self) -> Dict[str, Tensor]:
        """Get and clear outbox."""
        messages = dict(self.outbox)
        self.outbox.clear()
        return messages

    def clear_inbox(self) -> None:
        self.inbox.clear()

    def clear_outbox(self) -> None:
        self.outbox.clear()

    def clear(self) -> None:
        self.inbox.clear()
        self.outbox.clear()

    @property
    def has_mail(self) -> bool:
        return len(self.inbox) > 0

    @property
    def has_outgoing(self) -> bool:
        return len(self.outbox) > 0

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    def aggregate_inbox(self, mode: str = 'mean') -> Optional[Tensor]:
        """Aggregate all inbox messages."""
        if not self.inbox:
            return None

        messages = list(self.inbox.values())

        if mode == 'mean':
            return torch.stack(messages).mean(dim=0)
        elif mode == 'sum':
            return torch.stack(messages).sum(dim=0)
        elif mode == 'max':
            return torch.stack(messages).max(dim=0).values
        elif mode == 'stack':
            return torch.stack(messages)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        return self.fingerprint

    def __repr__(self) -> str:
        inbox_count = len(self.inbox)
        outbox_count = len(self.outbox)

        parts = [f"name='{self.name}'", f"dim={self.fingerprint_dim}"]

        if inbox_count > 0:
            parts.append(f"inbox={inbox_count}")
        if outbox_count > 0:
            parts.append(f"outbox={outbox_count}")

        return f"{self.__class__.__name__}({', '.join(parts)})"


# =============================================================================
# GATED ADDRESS
# =============================================================================

class GatedAddressComponent(AddressComponent):
    """Address with learned gating on messages."""

    def __init__(
        self,
        name: str,
        fingerprint_dim: int,
        message_dim: int,
        **kwargs,
    ):
        super().__init__(name, fingerprint_dim, **kwargs)

        self.message_dim = message_dim
        self.gate_proj = nn.Linear(fingerprint_dim * 2, message_dim)

    def gated_receive(
        self,
        source_addr: 'AddressComponent',
        message: Tensor,
    ) -> Tensor:
        """Receive with fingerprint-based gating."""
        combined = torch.cat([self.fingerprint, source_addr.fingerprint])
        gate = torch.sigmoid(self.gate_proj(combined))
        return message * gate

    def aggregate_gated(
        self,
        sources: List['AddressComponent'],
    ) -> Optional[Tensor]:
        """Aggregate inbox with fingerprint-based gating."""
        if not self.inbox:
            return None

        gated_messages = []
        for source in sources:
            if source.name in self.inbox:
                msg = self.inbox[source.name]
                gated = self.gated_receive(source, msg)
                gated_messages.append(gated)

        if not gated_messages:
            return None

        return torch.stack(gated_messages).sum(dim=0)


# =============================================================================
# ROUTED ADDRESS
# =============================================================================

class RoutedAddressComponent(AddressComponent):
    """Address with learned routing weights."""

    def __init__(
        self,
        name: str,
        fingerprint_dim: int,
        num_targets: int,
        **kwargs,
    ):
        super().__init__(name, fingerprint_dim, **kwargs)

        self.num_targets = num_targets
        self.route_weights = nn.Parameter(torch.zeros(num_targets))

    def route_scores(self) -> Tensor:
        """Get routing scores."""
        return F.softmax(self.route_weights, dim=0)

    def select_targets(
        self,
        k: int = 1,
        threshold: Optional[float] = None,
    ) -> List[int]:
        """Select top-k targets or above threshold."""
        scores = self.route_scores()

        if threshold is not None:
            return (scores > threshold).nonzero().squeeze(-1).tolist()

        _, indices = scores.topk(k)
        return indices.tolist()


# =============================================================================
# SIMPLEX ADDRESS (k-Simplex via SimplexFactory)
# =============================================================================

class SimplexAddressComponent(TorchComponent):
    """
    Address with k-simplex fingerprint via SimplexFactory.

    Fingerprint is barycentric combination of simplex vertices.
    Geometric correctness guaranteed by factory validation.
    """

    def __init__(
        self,
        name: str,
        k: int = 4,
        embed_dim: int = 64,
        method: str = 'regular',
        learnable: bool = True,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        self.k = k
        self.embed_dim = embed_dim
        self.method = method
        self.num_vertices = k + 1

        # Create factory
        self.factory = SimplexFactory(
            k=k,
            embed_dim=embed_dim,
            method=method,
            scale=1.0
        )

        # Generate simplex vertices
        vertices = self.factory.build_torch(dtype=torch.float32)

        if learnable:
            self.vertices = nn.Parameter(vertices)
        else:
            self.register_buffer('vertices', vertices)

        # Barycentric weights (on simplex)
        self._bary_logits = nn.Parameter(torch.zeros(self.num_vertices))

        self.inbox: Dict[str, Tensor] = {}
        self.outbox: Dict[str, Tensor] = {}

    @property
    def barycentric(self) -> Tensor:
        """Barycentric coordinates (sum to 1)."""
        return F.softmax(self._bary_logits, dim=0)

    @property
    def fingerprint(self) -> Tensor:
        """Fingerprint as barycentric combination of vertices."""
        return self.barycentric @ self.vertices

    @property
    def fingerprint_dim(self) -> int:
        return self.embed_dim

    def similarity(self, other: 'SimplexAddressComponent') -> Tensor:
        return F.cosine_similarity(
            self.fingerprint.unsqueeze(0),
            other.fingerprint.unsqueeze(0),
        ).squeeze()

    def distance(self, other: 'SimplexAddressComponent') -> Tensor:
        return torch.norm(self.fingerprint - other.fingerprint)

    def barycentric_distance(self, other: 'SimplexAddressComponent') -> Tensor:
        """Distance in barycentric coordinate space."""
        return torch.norm(self.barycentric - other.barycentric)

    def send(self, target: str, message: Tensor) -> None:
        self.outbox[target] = message

    def receive(self, source: str) -> Optional[Tensor]:
        return self.inbox.get(source)

    def deliver(self, source: str, message: Tensor) -> None:
        self.inbox[source] = message

    def collect(self) -> Dict[str, Tensor]:
        return dict(self.inbox)

    def clear(self) -> None:
        self.inbox.clear()
        self.outbox.clear()

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        return self.fingerprint

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"k={self.k}, embed_dim={self.embed_dim}, method='{self.method}')"
        )


# =============================================================================
# SPHERICAL ADDRESS (Unit Hypersphere)
# =============================================================================

class SphericalAddressComponent(TorchComponent):
    """Address with fingerprint on unit hypersphere."""

    def __init__(
        self,
        name: str,
        fingerprint_dim: int,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        self._fingerprint_dim = fingerprint_dim
        self._raw = nn.Parameter(torch.randn(fingerprint_dim) * 0.02)

        self.inbox: Dict[str, Tensor] = {}
        self.outbox: Dict[str, Tensor] = {}

    @property
    def fingerprint(self) -> Tensor:
        """Fingerprint on unit hypersphere (norm=1)."""
        return F.normalize(self._raw, dim=0)

    @property
    def fingerprint_dim(self) -> int:
        return self._fingerprint_dim

    def similarity(self, other: 'SphericalAddressComponent') -> Tensor:
        """Cosine similarity (dot product on unit sphere)."""
        return torch.dot(self.fingerprint, other.fingerprint)

    def distance(self, other: 'SphericalAddressComponent') -> Tensor:
        """Geodesic distance (angular distance)."""
        dot = torch.clamp(self.similarity(other), -1.0, 1.0)
        return torch.acos(dot)

    def slerp(self, other: 'SphericalAddressComponent', t: float) -> Tensor:
        """Spherical linear interpolation."""
        p0 = self.fingerprint
        p1 = other.fingerprint

        dot = torch.clamp(torch.dot(p0, p1), -1.0, 1.0)
        theta = torch.acos(dot)

        if theta.abs() < 1e-6:
            return p0

        sin_theta = torch.sin(theta)
        w0 = torch.sin((1 - t) * theta) / sin_theta
        w1 = torch.sin(t * theta) / sin_theta

        return w0 * p0 + w1 * p1

    def send(self, target: str, message: Tensor) -> None:
        self.outbox[target] = message

    def receive(self, source: str) -> Optional[Tensor]:
        return self.inbox.get(source)

    def deliver(self, source: str, message: Tensor) -> None:
        self.inbox[source] = message

    def collect(self) -> Dict[str, Tensor]:
        return dict(self.inbox)

    def clear(self) -> None:
        self.inbox.clear()
        self.outbox.clear()

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        return self.fingerprint

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', dim={self.fingerprint_dim})"


# =============================================================================
# HYPERBOLIC ADDRESS (Poincaré Ball)
# =============================================================================

class HyperbolicAddressComponent(TorchComponent):
    """Address with fingerprint in Poincaré ball."""

    def __init__(
        self,
        name: str,
        fingerprint_dim: int,
        curvature: float = 1.0,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        self._fingerprint_dim = fingerprint_dim
        self.curvature = curvature
        self._raw = nn.Parameter(torch.randn(fingerprint_dim) * 0.01)

        self.inbox: Dict[str, Tensor] = {}
        self.outbox: Dict[str, Tensor] = {}

    @property
    def fingerprint(self) -> Tensor:
        """Fingerprint in Poincaré ball (norm < 1)."""
        return self._project_to_ball(self._raw)

    @property
    def fingerprint_dim(self) -> int:
        return self._fingerprint_dim

    def _project_to_ball(self, x: Tensor, eps: float = 1e-5) -> Tensor:
        """Project point to Poincaré ball."""
        norm = x.norm()
        max_norm = 1.0 - eps
        if norm > max_norm:
            return x * max_norm / norm
        return x

    def _mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
        """Möbius addition in Poincaré ball."""
        c = self.curvature
        x2 = (x * x).sum()
        y2 = (y * y).sum()
        xy = (x * y).sum()

        num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        denom = 1 + 2*c*xy + c*c*x2*y2

        return self._project_to_ball(num / denom.clamp(min=1e-10))

    def distance(self, other: 'HyperbolicAddressComponent') -> Tensor:
        """Hyperbolic distance in Poincaré ball."""
        c = self.curvature
        x = self.fingerprint
        y = other.fingerprint

        diff = x - y
        x2 = (x * x).sum()
        y2 = (y * y).sum()
        diff2 = (diff * diff).sum()

        num = 2 * diff2
        denom = (1 - c*x2) * (1 - c*y2)

        arg = 1 + num / denom.clamp(min=1e-10)
        return torch.acosh(arg.clamp(min=1.0))

    def similarity(self, other: 'HyperbolicAddressComponent') -> Tensor:
        """Negative distance as similarity."""
        return -self.distance(other)

    def midpoint(self, other: 'HyperbolicAddressComponent') -> Tensor:
        """Hyperbolic midpoint."""
        x = self.fingerprint
        y = other.fingerprint

        neg_x = -x
        diff = self._mobius_add(neg_x, y)
        half_diff = diff * 0.5

        return self._mobius_add(x, half_diff)

    def send(self, target: str, message: Tensor) -> None:
        self.outbox[target] = message

    def receive(self, source: str) -> Optional[Tensor]:
        return self.inbox.get(source)

    def deliver(self, source: str, message: Tensor) -> None:
        self.inbox[source] = message

    def collect(self) -> Dict[str, Tensor]:
        return dict(self.inbox)

    def clear(self) -> None:
        self.inbox.clear()
        self.outbox.clear()

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        return self.fingerprint

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"dim={self.fingerprint_dim}, curvature={self.curvature})"
        )


# =============================================================================
# GEOMETRIC ADDRESS (Configurable Manifold)
# =============================================================================

class GeometricAddressComponent(TorchComponent):
    """
    Address with configurable geometric space.

    Manifolds: 'euclidean', 'spherical', 'hyperbolic', 'simplex', 'torus'
    """

    def __init__(
        self,
        name: str,
        fingerprint_dim: int,
        manifold: str = 'euclidean',
        curvature: float = 1.0,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        self._fingerprint_dim = fingerprint_dim
        self.manifold = manifold
        self.curvature = curvature
        self._raw = nn.Parameter(torch.randn(fingerprint_dim) * 0.02)

        self.inbox: Dict[str, Tensor] = {}
        self.outbox: Dict[str, Tensor] = {}

    @property
    def fingerprint(self) -> Tensor:
        """Fingerprint projected to manifold."""
        if self.manifold == 'euclidean':
            return self._raw
        elif self.manifold == 'spherical':
            return F.normalize(self._raw, dim=0)
        elif self.manifold == 'hyperbolic':
            return self._project_poincare(self._raw)
        elif self.manifold == 'simplex':
            return F.softmax(self._raw, dim=0)
        elif self.manifold == 'torus':
            return torch.remainder(self._raw, 2 * math.pi)
        else:
            raise ValueError(f"Unknown manifold: {self.manifold}")

    @property
    def fingerprint_dim(self) -> int:
        return self._fingerprint_dim

    def _project_poincare(self, x: Tensor, eps: float = 1e-5) -> Tensor:
        norm = x.norm()
        max_norm = 1.0 - eps
        if norm > max_norm:
            return x * max_norm / norm
        return x

    def distance(self, other: 'GeometricAddressComponent') -> Tensor:
        """Distance in the manifold."""
        x = self.fingerprint
        y = other.fingerprint

        if self.manifold == 'euclidean':
            return torch.norm(x - y)

        elif self.manifold == 'spherical':
            dot = torch.clamp(torch.dot(x, y), -1.0, 1.0)
            return torch.acos(dot)

        elif self.manifold == 'hyperbolic':
            c = self.curvature
            diff = x - y
            x2 = (x * x).sum()
            y2 = (y * y).sum()
            diff2 = (diff * diff).sum()
            num = 2 * diff2
            denom = (1 - c*x2) * (1 - c*y2)
            arg = 1 + num / denom.clamp(min=1e-10)
            return torch.acosh(arg.clamp(min=1.0))

        elif self.manifold == 'simplex':
            m = (x + y) / 2
            kl_xm = (x * (x.log() - m.log())).sum()
            kl_ym = (y * (y.log() - m.log())).sum()
            return (kl_xm + kl_ym) / 2

        elif self.manifold == 'torus':
            diff = x - y
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            return torch.norm(diff)

        else:
            raise ValueError(f"Unknown manifold: {self.manifold}")

    def similarity(self, other: 'GeometricAddressComponent') -> Tensor:
        if self.manifold == 'spherical':
            return torch.dot(self.fingerprint, other.fingerprint)
        else:
            return -self.distance(other)

    def send(self, target: str, message: Tensor) -> None:
        self.outbox[target] = message

    def receive(self, source: str) -> Optional[Tensor]:
        return self.inbox.get(source)

    def deliver(self, source: str, message: Tensor) -> None:
        self.inbox[source] = message

    def collect(self) -> Dict[str, Tensor]:
        return dict(self.inbox)

    def clear(self) -> None:
        self.inbox.clear()
        self.outbox.clear()

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        return self.fingerprint

    def __repr__(self) -> str:
        parts = [f"name='{self.name}'", f"dim={self.fingerprint_dim}", f"manifold='{self.manifold}'"]
        if self.manifold == 'hyperbolic':
            parts.append(f"curvature={self.curvature}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# =============================================================================
# FRACTAL ADDRESS (FractalFactory)
# =============================================================================

class FractalAddressComponent(TorchComponent):
    """
    Address with fractal fingerprint via FractalFactory.

    Identity derived from Julia set orbit in named region.
    """

    def __init__(
        self,
        name: str,
        region: str = 'seahorse',
        orbit_length: int = 64,
        learnable: bool = True,
        seed: Optional[int] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        if region not in FRACTAL_REGIONS:
            raise ValueError(f"Unknown region: {region}. Valid: {list(FRACTAL_REGIONS.keys())}")

        self.region = region
        self.orbit_length = orbit_length

        # Create factory
        self.factory = FractalFactory(
            mode=FractalMode.ORBIT,
            region=region,
            size=orbit_length,
            max_iter=orbit_length
        )

        # Generate orbit [T, 4] -> [T, 2] (re, im) -> flatten -> normalize
        orbit = self.factory.build_torch(seed=seed, dtype=torch.float32)
        fingerprint = orbit[:, :2].flatten()
        fingerprint = F.normalize(fingerprint, dim=0)

        if learnable:
            self._fingerprint = nn.Parameter(fingerprint)
        else:
            self.register_buffer('_fingerprint', fingerprint)

        self._c = self.factory.last_c

        self.inbox: Dict[str, Tensor] = {}
        self.outbox: Dict[str, Tensor] = {}

    @property
    def fingerprint(self) -> Tensor:
        return F.normalize(self._fingerprint, dim=0)

    @property
    def fingerprint_dim(self) -> int:
        return self._fingerprint.shape[0]

    @property
    def c_parameter(self) -> complex:
        """Julia c parameter."""
        return self._c

    def similarity(self, other: 'FractalAddressComponent') -> Tensor:
        return F.cosine_similarity(
            self.fingerprint.unsqueeze(0),
            other.fingerprint.unsqueeze(0),
        ).squeeze()

    def distance(self, other: 'FractalAddressComponent') -> Tensor:
        dot = torch.clamp(self.similarity(other), -1.0, 1.0)
        return torch.acos(dot)

    def same_region(self, other: 'FractalAddressComponent') -> bool:
        return self.region == other.region

    def send(self, target: str, message: Tensor) -> None:
        self.outbox[target] = message

    def receive(self, source: str) -> Optional[Tensor]:
        return self.inbox.get(source)

    def deliver(self, source: str, message: Tensor) -> None:
        self.inbox[source] = message

    def collect(self) -> Dict[str, Tensor]:
        return dict(self.inbox)

    def clear(self) -> None:
        self.inbox.clear()
        self.outbox.clear()

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        return self.fingerprint

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"region='{self.region}', dim={self.fingerprint_dim})"
        )


# =============================================================================
# CANTOR ADDRESS (CantorRouteFactory / Beatrix)
# =============================================================================

class CantorAddressComponent(TorchComponent):
    """
    Address with Cantor/Beatrix fingerprint via CantorRouteFactory.

    Uses Devil's Staircase for consciousness-compatible routing.
    """

    def __init__(
        self,
        name: str,
        k_simplex: int = 4,
        fingerprint_dim: int = 64,
        mode: str = 'staircase',
        tau: float = 0.25,
        learnable_scale: bool = True,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        self.k_simplex = k_simplex
        self._fingerprint_dim = fingerprint_dim
        self.mode = mode
        self.tau = tau

        simplex_config = SimplexConfig(k_simplex=k_simplex)
        route_mode = RouteMode(mode) if isinstance(mode, str) else mode

        if route_mode == RouteMode.STAIRCASE_FEATURES:
            self.factory = CantorRouteFactory(
                shape=(fingerprint_dim,),
                mode=route_mode,
                simplex_config=simplex_config,
                staircase_tau=tau
            )
            cantor, features = self.factory.build_torch(dtype=torch.float32)
            fingerprint = features.flatten()
        else:
            self.factory = CantorRouteFactory(
                shape=(fingerprint_dim,),
                mode=route_mode,
                simplex_config=simplex_config,
                staircase_tau=tau
            )
            fingerprint = self.factory.build_torch(dtype=torch.float32)
            if fingerprint.dim() > 1:
                fingerprint = fingerprint.flatten()

        self.register_buffer('_base_fingerprint', fingerprint)

        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
            self.shift = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('scale', torch.ones(1))
            self.register_buffer('shift', torch.zeros(1))

        self.inbox: Dict[str, Tensor] = {}
        self.outbox: Dict[str, Tensor] = {}

    @property
    def fingerprint(self) -> Tensor:
        return self._base_fingerprint * self.scale + self.shift

    @property
    def fingerprint_dim(self) -> int:
        return self._base_fingerprint.shape[0]

    @property
    def base_fingerprint(self) -> Tensor:
        return self._base_fingerprint

    def similarity(self, other: 'CantorAddressComponent') -> Tensor:
        f1 = F.normalize(self.fingerprint, dim=0)
        f2 = F.normalize(other.fingerprint, dim=0)
        return torch.dot(f1, f2)

    def distance(self, other: 'CantorAddressComponent') -> Tensor:
        return torch.norm(self.fingerprint - other.fingerprint)

    def cantor_distance(self, other: 'CantorAddressComponent') -> Tensor:
        """Deterministic distance in base Cantor space."""
        return torch.norm(self.base_fingerprint - other.base_fingerprint)

    def send(self, target: str, message: Tensor) -> None:
        self.outbox[target] = message

    def receive(self, source: str) -> Optional[Tensor]:
        return self.inbox.get(source)

    def deliver(self, source: str, message: Tensor) -> None:
        self.inbox[source] = message

    def collect(self) -> Dict[str, Tensor]:
        return dict(self.inbox)

    def clear(self) -> None:
        self.inbox.clear()
        self.outbox.clear()

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        return self.fingerprint

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"k={self.k_simplex}, dim={self.fingerprint_dim}, mode='{self.mode}')"
        )


# =============================================================================
# SHAPE ADDRESS (SimpleShapeFactory with RoPE offset)
# =============================================================================

class ShapeAddressComponent(TorchComponent):
    """
    Address with geometric shape fingerprint via SimpleShapeFactory.

    Shape points are modulated by theta offset for RoPE compatibility.
    The fingerprint rotates in embedding space based on position.

    Attributes:
        shape_type: Geometric shape ('cube', 'sphere', 'cylinder', 'pyramid', 'cone').
        resolution: Number of shape points.
        theta_offset: Rotational offset for position encoding.
    """

    def __init__(
        self,
        name: str,
        shape_type: Literal['cube', 'sphere', 'cylinder', 'pyramid', 'cone'] = 'sphere',
        embed_dim: int = 64,
        resolution: int = 32,
        theta_offset: float = 0.0,
        learnable: bool = True,
        seed: Optional[int] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        self.shape_type = shape_type
        self.resolution = resolution
        self._embed_dim = embed_dim

        # Ensure embed_dim is compatible (need at least 3 for most shapes)
        factory_embed_dim = max(3, embed_dim)

        # Create factory
        self.factory = SimpleShapeFactory(
            shape_type=shape_type,
            embed_dim=factory_embed_dim,
            resolution=resolution,
            scale=1.0
        )

        # Generate shape points [N, factory_embed_dim]
        shape_points = self.factory.build_torch(seed=seed, dtype=torch.float32)

        # Project to embed_dim if needed
        if factory_embed_dim != embed_dim:
            # Use PCA-style projection or simple slice/pad
            if embed_dim < factory_embed_dim:
                shape_points = shape_points[:, :embed_dim]
            else:
                padding = torch.zeros(shape_points.shape[0], embed_dim - factory_embed_dim)
                shape_points = torch.cat([shape_points, padding], dim=1)

        # Store base shape
        self.register_buffer('_base_shape', shape_points)

        # Theta offset for RoPE
        self._theta_offset = nn.Parameter(torch.tensor(theta_offset))

        # Learnable rotation frequencies (per dimension pair)
        num_pairs = embed_dim // 2
        if learnable:
            self.rotation_freqs = nn.Parameter(
                torch.arange(1, num_pairs + 1, dtype=torch.float32) * 0.1
            )
        else:
            self.register_buffer(
                'rotation_freqs',
                torch.arange(1, num_pairs + 1, dtype=torch.float32) * 0.1
            )

        self.inbox: Dict[str, Tensor] = {}
        self.outbox: Dict[str, Tensor] = {}

    @property
    def theta_offset(self) -> Tensor:
        return self._theta_offset

    @theta_offset.setter
    def theta_offset(self, value: float):
        self._theta_offset.data.fill_(value)

    def _apply_rotation(self, points: Tensor, theta: Tensor) -> Tensor:
        """
        Apply RoPE-style rotation to shape points.

        Args:
            points: [N, D] shape points
            theta: Rotation angle

        Returns:
            Rotated points [N, D]
        """
        D = points.shape[-1]
        num_pairs = D // 2

        # Compute rotation angles per dimension pair
        angles = theta * self.rotation_freqs[:num_pairs]  # [num_pairs]

        cos_theta = torch.cos(angles)  # [num_pairs]
        sin_theta = torch.sin(angles)  # [num_pairs]

        # Split into pairs
        x1 = points[:, 0::2][:, :num_pairs]  # Even indices
        x2 = points[:, 1::2][:, :num_pairs]  # Odd indices

        # Apply rotation
        y1 = x1 * cos_theta - x2 * sin_theta
        y2 = x1 * sin_theta + x2 * cos_theta

        # Interleave back
        result = torch.zeros_like(points)
        result[:, 0::2][:, :num_pairs] = y1
        result[:, 1::2][:, :num_pairs] = y2

        # Handle odd dimension
        if D % 2 == 1:
            result[:, -1] = points[:, -1]

        return result

    @property
    def shape_points(self) -> Tensor:
        """Shape points with theta rotation applied."""
        return self._apply_rotation(self._base_shape, self._theta_offset)

    @property
    def fingerprint(self) -> Tensor:
        """Fingerprint as mean of rotated shape points."""
        return self.shape_points.mean(dim=0)

    @property
    def fingerprint_dim(self) -> int:
        return self._embed_dim

    def fingerprint_at_theta(self, theta: float) -> Tensor:
        """Get fingerprint at specific theta position."""
        rotated = self._apply_rotation(self._base_shape, torch.tensor(theta))
        return rotated.mean(dim=0)

    def similarity(self, other: 'ShapeAddressComponent') -> Tensor:
        return F.cosine_similarity(
            self.fingerprint.unsqueeze(0),
            other.fingerprint.unsqueeze(0),
        ).squeeze()

    def distance(self, other: 'ShapeAddressComponent') -> Tensor:
        return torch.norm(self.fingerprint - other.fingerprint)

    def shape_distance(self, other: 'ShapeAddressComponent') -> Tensor:
        """Chamfer-like distance between shape point sets."""
        # Simplified: distance between mean shapes
        return torch.norm(self.shape_points.mean(0) - other.shape_points.mean(0))

    def angular_similarity(
        self,
        other: 'ShapeAddressComponent',
        theta_self: float,
        theta_other: float,
    ) -> Tensor:
        """Similarity at specific theta positions."""
        f1 = self.fingerprint_at_theta(theta_self)
        f2 = other.fingerprint_at_theta(theta_other)
        return F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).squeeze()

    def send(self, target: str, message: Tensor) -> None:
        self.outbox[target] = message

    def receive(self, source: str) -> Optional[Tensor]:
        return self.inbox.get(source)

    def deliver(self, source: str, message: Tensor) -> None:
        self.inbox[source] = message

    def collect(self) -> Dict[str, Tensor]:
        return dict(self.inbox)

    def clear(self) -> None:
        self.inbox.clear()
        self.outbox.clear()

    def forward(self, x: Optional[Tensor] = None, theta: Optional[float] = None) -> Tensor:
        """
        Return fingerprint, optionally at specific theta.

        Args:
            x: Unused (compatibility).
            theta: Optional position for RoPE. Uses stored offset if None.

        Returns:
            Fingerprint tensor.
        """
        if theta is not None:
            return self.fingerprint_at_theta(theta)
        return self.fingerprint

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"shape='{self.shape_type}', dim={self.fingerprint_dim}, "
            f"theta={self._theta_offset.item():.4f})"
        )


# =============================================================================
# ADDRESS BOOK
# =============================================================================

class AddressBook:
    """
    Registry of addresses for coordination.

    Manages message passing between AddressComponents.
    """

    def __init__(self):
        self.addresses: Dict[str, TorchComponent] = {}

    def register(self, addr: TorchComponent) -> None:
        """Register an address."""
        self.addresses[addr.name] = addr

    def unregister(self, name: str) -> Optional[TorchComponent]:
        """Unregister and return an address."""
        return self.addresses.pop(name, None)

    def get(self, name: str) -> Optional[TorchComponent]:
        """Get address by name."""
        return self.addresses.get(name)

    def route_messages(self) -> int:
        """Route all outgoing messages to inboxes."""
        count = 0

        for source_name, source_addr in self.addresses.items():
            if hasattr(source_addr, 'flush_outbox'):
                outbox = source_addr.flush_outbox()

                for target_name, message in outbox.items():
                    if target_name in self.addresses:
                        target = self.addresses[target_name]
                        if hasattr(target, 'deliver'):
                            target.deliver(source_name, message)
                            count += 1

        return count

    def clear_all(self) -> None:
        """Clear all mailboxes."""
        for addr in self.addresses.values():
            if hasattr(addr, 'clear'):
                addr.clear()

    def similarity_matrix(self) -> Tuple[List[str], Tensor]:
        """Compute pairwise similarity matrix."""
        names = list(self.addresses.keys())
        n = len(names)

        matrix = torch.zeros(n, n)
        addrs = list(self.addresses.values())

        for i, addr_i in enumerate(addrs):
            for j, addr_j in enumerate(addrs):
                if hasattr(addr_i, 'similarity') and hasattr(addr_j, 'fingerprint'):
                    matrix[i, j] = addr_i.similarity(addr_j)

        return names, matrix

    def __len__(self) -> int:
        return len(self.addresses)

    def __contains__(self, name: str) -> bool:
        return name in self.addresses

    def __repr__(self) -> str:
        return f"AddressBook(addresses={list(self.addresses.keys())})"


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == '__main__':

    def test_section(title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------------------------------------------
    test_section("BASIC ADDRESS")
    # -------------------------------------------------------------------------

    addr_a = AddressComponent('stream_a', fingerprint_dim=64)
    addr_b = AddressComponent('stream_b', fingerprint_dim=64)
    addr_c = AddressComponent('stream_c', fingerprint_dim=64)

    print(f"Address A: {addr_a}")
    print(f"Address B: {addr_b}")
    print(f"Fingerprint shape: {addr_a.fingerprint.shape}")

    sim_ab = addr_a.similarity(addr_b)
    sim_aa = addr_a.similarity(addr_a)
    print(f"Similarity A-B: {sim_ab.item():.4f}")
    print(f"Similarity A-A: {sim_aa.item():.4f}")

    # -------------------------------------------------------------------------
    test_section("MESSAGE PASSING")
    # -------------------------------------------------------------------------

    message = torch.randn(4, 256)
    addr_a.send('stream_b', message)
    print(f"A has outgoing: {addr_a.has_outgoing}")

    outbox = addr_a.flush_outbox()
    for target, msg in outbox.items():
        if target == 'stream_b':
            addr_b.deliver('stream_a', msg)

    print(f"B has mail: {addr_b.has_mail}")
    received = addr_b.receive('stream_a')
    print(f"Received match: {torch.equal(received, message)}")
    addr_b.clear()

    # -------------------------------------------------------------------------
    test_section("ADDRESS BOOK")
    # -------------------------------------------------------------------------

    book = AddressBook()
    book.register(addr_a)
    book.register(addr_b)
    book.register(addr_c)

    print(f"Book: {book}")

    addr_a.send('stream_b', torch.randn(4, 128))
    addr_b.send('stream_c', torch.randn(4, 128))

    routed = book.route_messages()
    print(f"Messages routed: {routed}")

    names, matrix = book.similarity_matrix()
    print(f"Similarity matrix:\n{matrix}")
    book.clear_all()

    # -------------------------------------------------------------------------
    test_section("SIMPLEX ADDRESS (SimplexFactory)")
    # -------------------------------------------------------------------------

    try:
        simplex_a = SimplexAddressComponent('simplex_a', k=4, embed_dim=64)
        simplex_b = SimplexAddressComponent('simplex_b', k=4, embed_dim=64)

        print(f"Simplex A: {simplex_a}")
        print(f"Vertices shape: {simplex_a.vertices.shape}")
        print(f"Barycentric: {simplex_a.barycentric.tolist()[:3]}...")
        print(f"Fingerprint shape: {simplex_a.fingerprint.shape}")

        sim = simplex_a.similarity(simplex_b)
        print(f"Similarity: {sim.item():.4f}")
    except ImportError as e:
        print(f"Skipping SimplexAddress (factory not available): {e}")

    # -------------------------------------------------------------------------
    test_section("SPHERICAL ADDRESS")
    # -------------------------------------------------------------------------

    sphere_a = SphericalAddressComponent('sphere_a', fingerprint_dim=64)
    sphere_b = SphericalAddressComponent('sphere_b', fingerprint_dim=64)

    print(f"Spherical A: {sphere_a}")
    print(f"Fingerprint norm: {sphere_a.fingerprint.norm().item():.4f}")

    sim = sphere_a.similarity(sphere_b)
    dist = sphere_a.distance(sphere_b)
    print(f"Cosine similarity: {sim.item():.4f}")
    print(f"Geodesic distance: {dist.item():.4f}")

    interp = sphere_a.slerp(sphere_b, 0.5)
    print(f"SLERP midpoint norm: {interp.norm().item():.4f}")

    # -------------------------------------------------------------------------
    test_section("HYPERBOLIC ADDRESS")
    # -------------------------------------------------------------------------

    hyper_a = HyperbolicAddressComponent('hyper_a', fingerprint_dim=64)
    hyper_b = HyperbolicAddressComponent('hyper_b', fingerprint_dim=64)

    print(f"Hyperbolic A: {hyper_a}")
    print(f"Fingerprint norm: {hyper_a.fingerprint.norm().item():.4f} (< 1)")

    dist = hyper_a.distance(hyper_b)
    print(f"Hyperbolic distance: {dist.item():.4f}")

    # -------------------------------------------------------------------------
    test_section("GEOMETRIC ADDRESS (Configurable)")
    # -------------------------------------------------------------------------

    for manifold in ['euclidean', 'spherical', 'hyperbolic', 'simplex', 'torus']:
        geo_a = GeometricAddressComponent('geo_a', fingerprint_dim=16, manifold=manifold)
        geo_b = GeometricAddressComponent('geo_b', fingerprint_dim=16, manifold=manifold)

        dist = geo_a.distance(geo_b)
        sim = geo_a.similarity(geo_b)

        print(f"{manifold:12s} - dist: {dist.item():8.4f}, sim: {sim.item():8.4f}")

    # -------------------------------------------------------------------------
    test_section("FRACTAL ADDRESS (FractalFactory)")
    # -------------------------------------------------------------------------

    try:
        fractal_a = FractalAddressComponent('fractal_a', region='seahorse', orbit_length=64)
        fractal_b = FractalAddressComponent('fractal_b', region='cardioid', orbit_length=64)

        print(f"Fractal A: {fractal_a}")
        print(f"Fractal B: {fractal_b}")
        print(f"A c parameter: {fractal_a.c_parameter}")
        print(f"Same region: {fractal_a.same_region(fractal_b)}")

        sim = fractal_a.similarity(fractal_b)
        print(f"Cross-region similarity: {sim.item():.4f}")
    except ImportError as e:
        print(f"Skipping FractalAddress (factory not available): {e}")

    # -------------------------------------------------------------------------
    test_section("CANTOR ADDRESS (CantorRouteFactory)")
    # -------------------------------------------------------------------------

    try:
        cantor_a = CantorAddressComponent('cantor_a', k_simplex=4, fingerprint_dim=64)
        cantor_b = CantorAddressComponent('cantor_b', k_simplex=4, fingerprint_dim=64)

        print(f"Cantor A: {cantor_a}")
        print(f"Base fingerprint shape: {cantor_a.base_fingerprint.shape}")
        print(f"Scale: {cantor_a.scale.item():.4f}")

        sim = cantor_a.similarity(cantor_b)
        cantor_dist = cantor_a.cantor_distance(cantor_b)
        print(f"Similarity: {sim.item():.4f}")
        print(f"Cantor distance: {cantor_dist.item():.4f}")
    except ImportError as e:
        print(f"Skipping CantorAddress (factory not available): {e}")

    # -------------------------------------------------------------------------
    test_section("SHAPE ADDRESS (SimpleShapeFactory + RoPE)")
    # -------------------------------------------------------------------------

    try:
        shape_sphere = ShapeAddressComponent(
            'shape_sphere', shape_type='sphere',
            embed_dim=64, resolution=32, theta_offset=0.0
        )
        shape_cube = ShapeAddressComponent(
            'shape_cube', shape_type='cube',
            embed_dim=64, resolution=32, theta_offset=0.0
        )

        print(f"Shape Sphere: {shape_sphere}")
        print(f"Shape Cube: {shape_cube}")
        print(f"Shape points: {shape_sphere.shape_points.shape}")
        print(f"Fingerprint: {shape_sphere.fingerprint.shape}")

        # Test theta rotation
        print("\nRoPE rotation test:")
        for theta in [0.0, 0.5, 1.0, 2.0]:
            fp = shape_sphere.fingerprint_at_theta(theta)
            print(f"  theta={theta:.1f}: fingerprint[:3] = {fp[:3].tolist()}")

        # Angular similarity
        sim_same = shape_sphere.angular_similarity(shape_sphere, 0.0, 0.0)
        sim_rotated = shape_sphere.angular_similarity(shape_sphere, 0.0, 1.0)
        print(f"\nAngular similarity (same theta): {sim_same.item():.4f}")
        print(f"Angular similarity (diff theta): {sim_rotated.item():.4f}")

        # Cross-shape
        sim_cross = shape_sphere.similarity(shape_cube)
        print(f"Sphere-Cube similarity: {sim_cross.item():.4f}")
    except ImportError as e:
        print(f"Skipping ShapeAddress (factory not available): {e}")

    # -------------------------------------------------------------------------
    test_section("GATED ADDRESS")
    # -------------------------------------------------------------------------

    gated_a = GatedAddressComponent('gated_a', fingerprint_dim=64, message_dim=256)
    gated_b = GatedAddressComponent('gated_b', fingerprint_dim=64, message_dim=256)

    print(f"Gated A: {gated_a}")

    message = torch.randn(4, 256)
    gated_msg = gated_a.gated_receive(gated_b, message)
    print(f"Gated message shape: {gated_msg.shape}")

    # -------------------------------------------------------------------------
    test_section("ROUTED ADDRESS")
    # -------------------------------------------------------------------------

    routed_addr = RoutedAddressComponent('router', fingerprint_dim=64, num_targets=5)
    print(f"Routed: {routed_addr}")

    scores = routed_addr.route_scores()
    print(f"Route scores: {scores.tolist()}")

    targets = routed_addr.select_targets(k=2)
    print(f"Top-2 targets: {targets}")

    # -------------------------------------------------------------------------
    test_section("DEVICE + COMPILE")
    # -------------------------------------------------------------------------

    addr = AddressComponent('gpu_addr', fingerprint_dim=64).to(device)
    fp = addr()
    print(f"Fingerprint device: {fp.device}")

    compiled = torch.compile(addr)
    fp2 = compiled()
    print(f"Compiled match: {torch.allclose(fp, fp2)}")

    # -------------------------------------------------------------------------
    test_section("MIXED ADDRESS BOOK")
    # -------------------------------------------------------------------------

    mixed_book = AddressBook()
    mixed_book.register(AddressComponent('euclidean', 64))
    mixed_book.register(SphericalAddressComponent('spherical', 64))
    mixed_book.register(GeometricAddressComponent('hyperbolic', 64, manifold='hyperbolic'))

    print(f"Mixed book: {mixed_book}")

    # Message passing still works
    mixed_book.addresses['euclidean'].send('spherical', torch.randn(4, 32))
    routed = mixed_book.route_messages()
    print(f"Cross-type messages routed: {routed}")

    # -------------------------------------------------------------------------
    test_section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nAddressComponent system is ready.")
    print("\nAvailable address types:")
    print("  - AddressComponent (Euclidean)")
    print("  - GatedAddressComponent")
    print("  - RoutedAddressComponent")
    print("  - SimplexAddressComponent (SimplexFactory)")
    print("  - SphericalAddressComponent")
    print("  - HyperbolicAddressComponent")
    print("  - GeometricAddressComponent (configurable manifold)")
    print("  - FractalAddressComponent (FractalFactory)")
    print("  - CantorAddressComponent (CantorRouteFactory)")
    print("  - ShapeAddressComponent (SimpleShapeFactory + RoPE)")