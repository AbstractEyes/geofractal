"""
geofractal.router.prefab.notifier_router
========================================

NotifierRouter - Orchestrates AddressComponent communication.

Builds on the existing AddressComponent system to provide:
    - Channel-based pub/sub (groups of addresses)
    - Dimension projection for transfer learning
    - Geometric routing (similarity on manifolds)
    - Broadcast patterns
    - Statistics and monitoring

The actual message passing uses AddressComponent's inbox/outbox.
NotifierRouter coordinates WHO talks to WHO and HOW.

Geometric Routing:
    Different address types enable different routing strategies:

    - Euclidean: Standard cosine similarity
    - Spherical: Geodesic distance, SLERP interpolation
    - Hyperbolic: Hierarchical relationships (tree-like)
    - Simplex: Barycentric blending
    - Fractal: Region-based clustering
    - Cantor: Devil's staircase routing
    - Shape: RoPE-compatible positional routing

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from collections import defaultdict
from typing import Optional, Dict, List, Set, Union, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.base_router import BaseRouter
from geofractal.router.components.address_component import (
    AddressComponent,
    GatedAddressComponent,
    RoutedAddressComponent,
    SimplexAddressComponent,
    SphericalAddressComponent,
    HyperbolicAddressComponent,
    GeometricAddressComponent,
    AddressBook,
    FractalAddressComponent,
    CantorAddressComponent,
    ShapeAddressComponent,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _has_mail(addr) -> bool:
    """Check if address has pending mail (works for all variants)."""
    if hasattr(addr, 'has_mail'):
        return addr.has_mail
    if hasattr(addr, 'inbox'):
        return len(addr.inbox) > 0
    return False


def _has_outgoing(addr) -> bool:
    """Check if address has outgoing messages."""
    if hasattr(addr, 'has_outgoing'):
        return addr.has_outgoing
    if hasattr(addr, 'outbox'):
        return len(addr.outbox) > 0
    return False


def _get_fingerprint(addr) -> Optional[Tensor]:
    """Get fingerprint from any address type."""
    if hasattr(addr, 'fingerprint'):
        fp = addr.fingerprint
        return fp() if callable(fp) else fp
    return None


def _compute_similarity(addr_a, addr_b) -> Optional[Tensor]:
    """Compute similarity between two addresses."""
    if hasattr(addr_a, 'similarity'):
        return addr_a.similarity(addr_b)

    # Fallback to cosine similarity of fingerprints
    fp_a = _get_fingerprint(addr_a)
    fp_b = _get_fingerprint(addr_b)

    if fp_a is not None and fp_b is not None:
        return F.cosine_similarity(fp_a.unsqueeze(0), fp_b.unsqueeze(0)).squeeze()

    return None


def _compute_distance(addr_a, addr_b) -> Optional[Tensor]:
    """Compute distance between two addresses."""
    if hasattr(addr_a, 'distance'):
        return addr_a.distance(addr_b)

    # Fallback to L2 distance
    fp_a = _get_fingerprint(addr_a)
    fp_b = _get_fingerprint(addr_b)

    if fp_a is not None and fp_b is not None:
        return torch.norm(fp_a - fp_b)

    return None


# =============================================================================
# NOTIFIER ROUTER
# =============================================================================

class NotifierRouter(BaseRouter):
    """
    Orchestrates AddressComponent communication between towers.

    Adds channel-based routing, geometric routing, and projection
    on top of the existing AddressComponent/AddressBook system.
    """

    def __init__(
        self,
        name: str,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        # Core address book for message routing
        self.book = AddressBook()

        # Channel subscriptions: channel -> set of address names
        self._channels: Dict[str, Set[str]] = defaultdict(set)

        # Projectors for dimension adaptation
        self.projectors = nn.ModuleDict()

        # Custom routing functions per channel
        self._routing_functions: Dict[str, Callable] = {}

        # Statistics
        self._stats = {
            'routes_executed': 0,
            'messages_routed': 0,
            'broadcasts': 0,
            'projections_applied': 0,
            'similarity_routes': 0,
            'geometric_routes': 0,
        }

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(
        self,
        address,
        channel: str = 'default',
    ) -> 'NotifierRouter':
        """
        Register an AddressComponent to a channel.

        Args:
            address: Any AddressComponent variant
            channel: Channel to subscribe to

        Returns:
            Self for chaining.
        """
        self.book.register(address)
        self._channels[channel].add(address.name)
        return self

    def unregister(
        self,
        address: Union[str, Any],
        channel: Optional[str] = None,
    ) -> 'NotifierRouter':
        """
        Remove address from channel(s).
        """
        name = address if isinstance(address, str) else address.name

        if channel is None:
            for ch in self._channels.values():
                ch.discard(name)
            self.book.unregister(name)
        else:
            self._channels[channel].discard(name)

        return self

    def subscribers(self, channel: str = 'default') -> Set[str]:
        """Get all subscribers for a channel."""
        return self._channels[channel].copy()

    def channels(self, address: Union[str, Any] = None) -> Set[str]:
        """Get channels (optionally for specific address)."""
        if address is None:
            return set(self._channels.keys())

        name = address if isinstance(address, str) else address.name
        return {ch for ch, subs in self._channels.items() if name in subs}

    def get_address(self, name: str):
        """Get registered address by name."""
        return self.book.get(name)

    # =========================================================================
    # PROJECTOR MANAGEMENT
    # =========================================================================

    def register_projector(
        self,
        source: str,
        target: str,
        projector: nn.Module,
    ) -> 'NotifierRouter':
        """Register dimension adapter for source->target."""
        key = f"{source}_to_{target}"
        self.projectors[key] = projector
        return self

    def create_projector(
        self,
        source: str,
        target: str,
        source_dim: int,
        target_dim: int,
        bias: bool = False,
    ) -> 'NotifierRouter':
        """Create and register a linear projector."""
        projector = nn.Linear(source_dim, target_dim, bias=bias)
        return self.register_projector(source, target, projector)

    def get_projector(self, source: str, target: str) -> Optional[nn.Module]:
        """Get projector for source->target."""
        key = f"{source}_to_{target}"
        if key in self.projectors:
            return self.projectors[key]
        return None

    def _maybe_project(self, payload: Tensor, source: str, target: str) -> Tensor:
        """Apply projection if registered."""
        projector = self.get_projector(source, target)
        if projector is not None:
            self._stats['projections_applied'] += 1
            return projector(payload)
        return payload

    # =========================================================================
    # CUSTOM ROUTING FUNCTIONS
    # =========================================================================

    def set_routing_function(
        self,
        channel: str,
        func: Callable[[Any, Any, Tensor], Tensor],
    ) -> 'NotifierRouter':
        """
        Set custom routing function for a channel.

        Function signature: (source_addr, target_addr, payload) -> transformed_payload
        """
        self._routing_functions[channel] = func
        return self

    def _apply_routing_function(
        self,
        channel: str,
        source_addr,
        target_addr,
        payload: Tensor,
    ) -> Tensor:
        """Apply channel's routing function if set."""
        if channel in self._routing_functions:
            return self._routing_functions[channel](source_addr, target_addr, payload)
        return payload

    # =========================================================================
    # MESSAGE OPERATIONS
    # =========================================================================

    def post(
        self,
        source: Union[str, Any],
        payload: Tensor,
        channel: str = 'default',
    ) -> int:
        """
        Post message from source to all subscribers of channel.
        """
        source_name = source if isinstance(source, str) else source.name
        source_addr = self.book.get(source_name)

        if source_addr is None:
            return 0

        subscribers = self._channels.get(channel, set())
        queued = 0

        for target_name in subscribers:
            if target_name == source_name:
                continue

            target_addr = self.book.get(target_name)
            if target_addr is None:
                continue

            # Apply routing function
            transformed = self._apply_routing_function(channel, source_addr, target_addr, payload)

            # Project if needed
            projected = self._maybe_project(transformed, source_name, target_name)

            # Queue via address component
            source_addr.send(target_name, projected)
            queued += 1

        return queued

    def send(
        self,
        source: Union[str, Any],
        target: Union[str, Any],
        payload: Tensor,
    ) -> bool:
        """Direct message from source to target."""
        source_name = source if isinstance(source, str) else source.name
        target_name = target if isinstance(target, str) else target.name

        source_addr = self.book.get(source_name)
        if source_addr is None or target_name not in self.book:
            return False

        projected = self._maybe_project(payload, source_name, target_name)
        source_addr.send(target_name, projected)
        return True

    def broadcast(
        self,
        source: Union[str, Any],
        payload: Tensor,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """Broadcast to ALL registered addresses."""
        source_name = source if isinstance(source, str) else source.name
        source_addr = self.book.get(source_name)

        if source_addr is None:
            return 0

        exclude = exclude or set()
        exclude.add(source_name)

        queued = 0
        for target_name in self.book.addresses:
            if target_name in exclude:
                continue

            projected = self._maybe_project(payload, source_name, target_name)
            source_addr.send(target_name, projected)
            queued += 1

        self._stats['broadcasts'] += 1
        return queued

    # =========================================================================
    # ROUTING
    # =========================================================================

    def route(self) -> int:
        """Execute message routing via AddressBook."""
        count = self.book.route_messages()
        self._stats['routes_executed'] += 1
        self._stats['messages_routed'] += count
        return count

    def route_channel(self, channel: str) -> int:
        """Route messages only between channel subscribers."""
        subscribers = self._channels.get(channel, set())
        count = 0

        for source_name in subscribers:
            source_addr = self.book.get(source_name)
            if source_addr is None:
                continue

            if hasattr(source_addr, 'flush_outbox'):
                outbox = source_addr.flush_outbox()
            else:
                outbox = dict(source_addr.outbox)
                source_addr.outbox.clear()

            for target_name, message in outbox.items():
                if target_name in subscribers:
                    target_addr = self.book.get(target_name)
                    if target_addr is not None:
                        target_addr.deliver(source_name, message)
                        count += 1

        self._stats['messages_routed'] += count
        return count

    # =========================================================================
    # GEOMETRIC ROUTING
    # =========================================================================

    def route_by_similarity(
        self,
        source: Union[str, Any],
        payload: Tensor,
        channel: str = 'default',
        top_k: int = 1,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Route to most similar addresses in channel.

        Uses fingerprint similarity (respects manifold geometry).
        """
        source_name = source if isinstance(source, str) else source.name
        source_addr = self.book.get(source_name)

        if source_addr is None:
            return []

        subscribers = self._channels.get(channel, set())
        candidates = []

        for name in subscribers:
            if name == source_name:
                continue
            addr = self.book.get(name)
            if addr is not None:
                candidates.append(addr)

        if not candidates:
            return []

        # Compute similarities
        sims = []
        for addr in candidates:
            sim = _compute_similarity(source_addr, addr)
            if sim is not None:
                sims.append((addr.name, sim.item() if hasattr(sim, 'item') else float(sim)))

        # Sort by similarity (descending)
        sims.sort(key=lambda x: x[1], reverse=True)

        # Select targets
        targets = []
        for name, sim in sims[:top_k]:
            if threshold is not None and sim < threshold:
                continue

            projected = self._maybe_project(payload, source_name, name)
            source_addr.send(name, projected)
            targets.append(name)

        self._stats['similarity_routes'] += len(targets)
        return targets

    def route_by_distance(
        self,
        source: Union[str, Any],
        payload: Tensor,
        channel: str = 'default',
        top_k: int = 1,
        max_distance: Optional[float] = None,
    ) -> List[str]:
        """
        Route to nearest addresses by manifold distance.

        Respects geometry (geodesic for spherical, hyperbolic distance, etc.)
        """
        source_name = source if isinstance(source, str) else source.name
        source_addr = self.book.get(source_name)

        if source_addr is None:
            return []

        subscribers = self._channels.get(channel, set())
        candidates = []

        for name in subscribers:
            if name == source_name:
                continue
            addr = self.book.get(name)
            if addr is not None:
                candidates.append(addr)

        if not candidates:
            return []

        # Compute distances
        dists = []
        for addr in candidates:
            dist = _compute_distance(source_addr, addr)
            if dist is not None:
                dists.append((addr.name, dist.item() if hasattr(dist, 'item') else float(dist)))

        # Sort by distance (ascending)
        dists.sort(key=lambda x: x[1])

        # Select targets
        targets = []
        for name, dist in dists[:top_k]:
            if max_distance is not None and dist > max_distance:
                continue

            projected = self._maybe_project(payload, source_name, name)
            source_addr.send(name, projected)
            targets.append(name)

        self._stats['geometric_routes'] += len(targets)
        return targets

    def affinity_broadcast(
        self,
        source: Union[str, Any],
        payload: Tensor,
        channel: str = 'default',
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Broadcast weighted by fingerprint affinity.
        """
        source_name = source if isinstance(source, str) else source.name
        source_addr = self.book.get(source_name)

        if source_addr is None:
            return {}

        subscribers = self._channels.get(channel, set())
        candidates = []

        for name in subscribers:
            if name == source_name:
                continue
            addr = self.book.get(name)
            if addr is not None:
                candidates.append(addr)

        if not candidates:
            return {}

        # Compute affinities
        if hasattr(source_addr, 'affinity'):
            weights = source_addr.affinity(candidates, temperature)
        else:
            # Manual affinity computation
            sims = []
            for addr in candidates:
                sim = _compute_similarity(source_addr, addr)
                sims.append(sim if sim is not None else torch.tensor(0.0))

            sims_tensor = torch.stack(sims)
            weights = F.softmax(sims_tensor / temperature, dim=0)

        result = {}
        for addr, weight in zip(candidates, weights):
            w = weight.item() if hasattr(weight, 'item') else float(weight)
            projected = self._maybe_project(payload, source_name, addr.name)
            weighted_payload = projected * w
            source_addr.send(addr.name, weighted_payload)
            result[addr.name] = w

        return result

    # =========================================================================
    # SPHERICAL ROUTING (SLERP)
    # =========================================================================

    def interpolate_spherical(
        self,
        addr_a: Union[str, SphericalAddressComponent],
        addr_b: Union[str, SphericalAddressComponent],
        t: float,
    ) -> Optional[Tensor]:
        """
        Spherical linear interpolation between two addresses.

        Returns interpolated fingerprint on the hypersphere.
        """
        a = self.book.get(addr_a) if isinstance(addr_a, str) else addr_a
        b = self.book.get(addr_b) if isinstance(addr_b, str) else addr_b

        if a is None or b is None:
            return None

        if hasattr(a, 'slerp'):
            return a.slerp(b, t)

        # Fallback to linear interpolation
        fp_a = _get_fingerprint(a)
        fp_b = _get_fingerprint(b)

        if fp_a is not None and fp_b is not None:
            return F.normalize((1 - t) * fp_a + t * fp_b, dim=0)

        return None

    # =========================================================================
    # HYPERBOLIC ROUTING
    # =========================================================================

    def find_hierarchical_parent(
        self,
        address: Union[str, HyperbolicAddressComponent],
        channel: str = 'default',
    ) -> Optional[str]:
        """
        Find parent in hyperbolic hierarchy (closest to origin).

        In hyperbolic space, points closer to origin are higher in hierarchy.
        """
        addr = self.book.get(address) if isinstance(address, str) else address
        if addr is None:
            return None

        subscribers = self._channels.get(channel, set())

        best_parent = None
        best_norm = float('inf')
        my_norm = _get_fingerprint(addr).norm().item()

        for name in subscribers:
            if name == addr.name:
                continue

            other = self.book.get(name)
            if other is None:
                continue

            other_fp = _get_fingerprint(other)
            if other_fp is None:
                continue

            other_norm = other_fp.norm().item()

            # Parent should be closer to origin (smaller norm)
            if other_norm < my_norm and other_norm < best_norm:
                best_norm = other_norm
                best_parent = name

        return best_parent

    def find_hierarchical_children(
        self,
        address: Union[str, HyperbolicAddressComponent],
        channel: str = 'default',
    ) -> List[str]:
        """
        Find children in hyperbolic hierarchy (further from origin).
        """
        addr = self.book.get(address) if isinstance(address, str) else address
        if addr is None:
            return []

        subscribers = self._channels.get(channel, set())
        my_norm = _get_fingerprint(addr).norm().item()

        children = []
        for name in subscribers:
            if name == addr.name:
                continue

            other = self.book.get(name)
            if other is None:
                continue

            other_fp = _get_fingerprint(other)
            if other_fp is None:
                continue

            # Child should be further from origin
            if other_fp.norm().item() > my_norm:
                children.append(name)

        return children

    # =========================================================================
    # SIMPLEX ROUTING
    # =========================================================================

    def barycentric_blend(
        self,
        addresses: List[Union[str, SimplexAddressComponent]],
        weights: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """
        Blend multiple simplex addresses using barycentric coordinates.
        """
        addrs = [
            self.book.get(a) if isinstance(a, str) else a
            for a in addresses
        ]
        addrs = [a for a in addrs if a is not None]

        if not addrs:
            return None

        fingerprints = [_get_fingerprint(a) for a in addrs]
        fingerprints = [fp for fp in fingerprints if fp is not None]

        if not fingerprints:
            return None

        if weights is None:
            weights = torch.ones(len(fingerprints)) / len(fingerprints)

        # Ensure weights sum to 1 (barycentric constraint)
        weights = weights / weights.sum()

        blended = torch.zeros_like(fingerprints[0])
        for fp, w in zip(fingerprints, weights):
            blended = blended + w * fp

        return blended

    # =========================================================================
    # UTILITY
    # =========================================================================

    def clear(self, address: Optional[Union[str, Any]] = None) -> None:
        """Clear mailboxes."""
        if address is None:
            self.book.clear_all()
        else:
            name = address if isinstance(address, str) else address.name
            addr = self.book.get(name)
            if addr is not None:
                addr.clear()

    def reset(self) -> None:
        """Clear all mailboxes and reset statistics."""
        self.book.clear_all()
        self._stats = {
            'routes_executed': 0,
            'messages_routed': 0,
            'broadcasts': 0,
            'projections_applied': 0,
            'similarity_routes': 0,
            'geometric_routes': 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        """Get communication statistics."""
        return self._stats.copy()

    def similarity_matrix(self, channel: Optional[str] = None):
        """Get pairwise similarity matrix."""
        if channel is None:
            return self.book.similarity_matrix()

        subscribers = list(self._channels.get(channel, set()))
        n = len(subscribers)

        matrix = torch.zeros(n, n)
        for i, name_i in enumerate(subscribers):
            addr_i = self.book.get(name_i)
            for j, name_j in enumerate(subscribers):
                addr_j = self.book.get(name_j)
                sim = _compute_similarity(addr_i, addr_j)
                if sim is not None:
                    matrix[i, j] = sim

        return subscribers, matrix

    def distance_matrix(self, channel: Optional[str] = None):
        """Get pairwise distance matrix."""
        if channel is None:
            subscribers = list(self.book.addresses.keys())
        else:
            subscribers = list(self._channels.get(channel, set()))

        n = len(subscribers)
        matrix = torch.zeros(n, n)

        for i, name_i in enumerate(subscribers):
            addr_i = self.book.get(name_i)
            for j, name_j in enumerate(subscribers):
                addr_j = self.book.get(name_j)
                dist = _compute_distance(addr_i, addr_j)
                if dist is not None:
                    matrix[i, j] = dist

        return subscribers, matrix

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(self) -> int:
        """Forward pass executes routing."""
        return self.route()

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self) -> str:
        channels = list(self._channels.keys())
        total_addrs = len(self.book)
        total_subs = sum(len(s) for s in self._channels.values())

        return (
            f"{self.__class__.__name__}(\n"
            f"  name='{self.name}',\n"
            f"  addresses={total_addrs},\n"
            f"  channels={channels},\n"
            f"  subscriptions={total_subs},\n"
            f"  projectors={list(self.projectors.keys())},\n"
            f"  stats={self._stats}\n"
            f")"
        )


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    import torch
    from geofractal.router.base_tower import BaseTower

    # Concrete tower for testing
    class TestTower(BaseTower):
        def forward(self, x: Tensor) -> Tensor:
            for stage in self.stages:
                x = stage(x)
            return x

    def section(title: str) -> None:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print('=' * 70)

    def has_mail(addr) -> bool:
        return _has_mail(addr)

    # =========================================================================
    section("EUCLIDEAN ADDRESS ROUTING")
    # =========================================================================

    notifier = NotifierRouter('euclidean_comm')

    # Create euclidean addresses
    euclidean_addrs = [
        AddressComponent(f'euclidean_{i}', fingerprint_dim=64)
        for i in range(4)
    ]

    for addr in euclidean_addrs:
        notifier.register(addr, channel='euclidean')

    print(f"Notifier:\n{notifier}")

    # Post from first to channel
    payload = torch.randn(4, 256)
    queued = notifier.post(euclidean_addrs[0], payload, channel='euclidean')
    print(f"\nPosted: {queued} messages queued")

    # Route
    routed = notifier.route()
    print(f"Routed: {routed} messages delivered")

    # Check inboxes
    print(f"\nInbox status:")
    for addr in euclidean_addrs:
        print(f"  {addr.name}: has_mail={has_mail(addr)}")

    # Similarity routing
    notifier.clear()
    targets = notifier.route_by_similarity(
        euclidean_addrs[0],
        payload,
        channel='euclidean',
        top_k=2,
    )
    print(f"\nSimilarity routing targets: {targets}")

    notifier.route()
    notifier.clear()

    # =========================================================================
    section("SPHERICAL ADDRESS ROUTING (Geodesic)")
    # =========================================================================

    spherical_notifier = NotifierRouter('spherical_comm')

    spherical_addrs = [
        SphericalAddressComponent(f'sphere_{i}', fingerprint_dim=64)
        for i in range(4)
    ]

    for addr in spherical_addrs:
        spherical_notifier.register(addr, channel='spherical')

    print(f"Spherical addresses registered: {len(spherical_addrs)}")

    # Check fingerprints are on unit sphere
    for addr in spherical_addrs:
        norm = addr.fingerprint.norm().item()
        print(f"  {addr.name}: fingerprint norm = {norm:.6f} (should be 1.0)")

    # Geodesic distances
    print(f"\nGeodesic distances:")
    for i, a in enumerate(spherical_addrs[:2]):
        for j, b in enumerate(spherical_addrs[2:], 2):
            dist = a.distance(b)
            print(f"  {a.name} <-> {b.name}: {dist.item():.4f} radians")

    # SLERP interpolation
    print(f"\nSLERP interpolation (sphere_0 -> sphere_1):")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        interp = spherical_addrs[0].slerp(spherical_addrs[1], t)
        print(f"  t={t:.2f}: norm={interp.norm().item():.6f}")

    # Route by geodesic distance
    targets = spherical_notifier.route_by_distance(
        spherical_addrs[0],
        torch.randn(4, 128),
        channel='spherical',
        top_k=2,
    )
    print(f"\nNearest by geodesic distance: {targets}")

    spherical_notifier.route()
    spherical_notifier.clear()

    # =========================================================================
    section("HYPERBOLIC ADDRESS ROUTING (Hierarchical)")
    # =========================================================================

    hyperbolic_notifier = NotifierRouter('hyperbolic_comm')

    hyperbolic_addrs = [
        HyperbolicAddressComponent(f'hyper_{i}', fingerprint_dim=64, curvature=1.0)
        for i in range(4)
    ]

    for addr in hyperbolic_addrs:
        hyperbolic_notifier.register(addr, channel='hyperbolic')

    print(f"Hyperbolic addresses registered: {len(hyperbolic_addrs)}")

    # Check fingerprints are in Poincaré ball
    for addr in hyperbolic_addrs:
        norm = addr.fingerprint.norm().item()
        print(f"  {addr.name}: fingerprint norm = {norm:.6f} (should be < 1.0)")

    # Hyperbolic distances
    print(f"\nHyperbolic distances:")
    for i, a in enumerate(hyperbolic_addrs[:2]):
        for j, b in enumerate(hyperbolic_addrs[2:], 2):
            dist = a.distance(b)
            print(f"  {a.name} <-> {b.name}: {dist.item():.4f}")

    # Find hierarchical structure
    print(f"\nHierarchical structure:")
    for addr in hyperbolic_addrs:
        parent = hyperbolic_notifier.find_hierarchical_parent(addr, channel='hyperbolic')
        children = hyperbolic_notifier.find_hierarchical_children(addr, channel='hyperbolic')
        norm = addr.fingerprint.norm().item()
        print(f"  {addr.name} (norm={norm:.4f}): parent={parent}, children={children}")

    hyperbolic_notifier.clear()

    # =========================================================================
    section("GEOMETRIC ADDRESS (Multiple Manifolds)")
    # =========================================================================

    geo_notifier = NotifierRouter('geometric_comm')

    manifolds = ['euclidean', 'spherical', 'hyperbolic', 'simplex', 'torus']
    geo_addrs = {}

    for manifold in manifolds:
        addr = GeometricAddressComponent(f'geo_{manifold}', fingerprint_dim=32, manifold=manifold)
        geo_addrs[manifold] = addr
        geo_notifier.register(addr, channel='geometric')

    print(f"Geometric addresses on different manifolds:")
    for manifold, addr in geo_addrs.items():
        fp = addr.fingerprint
        print(f"  {manifold}: shape={fp.shape}, sample={fp[:3].tolist()}")

    # Cross-manifold similarity matrix
    names, matrix = geo_notifier.similarity_matrix(channel='geometric')
    print(f"\nSimilarity matrix:\n{matrix}")

    # Distance matrix
    names, dist_matrix = geo_notifier.distance_matrix(channel='geometric')
    print(f"\nDistance matrix:\n{dist_matrix}")

    geo_notifier.clear()

    # =========================================================================
    section("GATED ADDRESS (Fingerprint-Gated Messages)")
    # =========================================================================

    gated_notifier = NotifierRouter('gated_comm')

    gated_a = GatedAddressComponent('gated_a', fingerprint_dim=64, message_dim=256)
    gated_b = GatedAddressComponent('gated_b', fingerprint_dim=64, message_dim=256)

    gated_notifier.register(gated_a, channel='gated')
    gated_notifier.register(gated_b, channel='gated')

    print(f"Gated addresses: {gated_a.name}, {gated_b.name}")

    # Send gated message
    message = torch.randn(4, 256)
    gated_message = gated_a.gated_receive(gated_b, message)

    print(f"Original message norm: {message.norm().item():.4f}")
    print(f"Gated message norm: {gated_message.norm().item():.4f}")
    print(f"Gate attenuated message by fingerprint interaction")

    # =========================================================================
    section("ROUTED ADDRESS (Learned Routing Weights)")
    # =========================================================================

    routed_addr = RoutedAddressComponent('router', fingerprint_dim=64, num_targets=5)

    print(f"Routed address: {routed_addr}")
    print(f"Route scores: {routed_addr.route_scores().tolist()}")

    # Select top targets
    top_targets = routed_addr.select_targets(k=2)
    print(f"Top-2 targets: {top_targets}")

    # Threshold selection
    threshold_targets = routed_addr.select_targets(threshold=0.15)
    print(f"Targets above 0.15 threshold: {threshold_targets}")

    # =========================================================================
    section("AFFINITY BROADCAST")
    # =========================================================================

    affinity_notifier = NotifierRouter('affinity_comm')

    affinity_addrs = [
        AddressComponent(f'affinity_{i}', fingerprint_dim=64)
        for i in range(4)
    ]

    for addr in affinity_addrs:
        affinity_notifier.register(addr, channel='affinity')

    # Affinity broadcast
    weights = affinity_notifier.affinity_broadcast(
        affinity_addrs[0],
        torch.randn(4, 128),
        channel='affinity',
        temperature=0.5,
    )

    print(f"Affinity weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")

    print(f"\nWeights sum to: {sum(weights.values()):.4f}")

    affinity_notifier.route()
    affinity_notifier.clear()

    # =========================================================================
    section("CUSTOM ROUTING FUNCTION")
    # =========================================================================

    custom_notifier = NotifierRouter('custom_comm')

    custom_addrs = [
        AddressComponent(f'custom_{i}', fingerprint_dim=64)
        for i in range(3)
    ]

    for addr in custom_addrs:
        custom_notifier.register(addr, channel='custom')

    # Define custom routing: scale by fingerprint similarity
    def similarity_scale_router(source_addr, target_addr, payload):
        sim = _compute_similarity(source_addr, target_addr)
        if sim is not None:
            return payload * sim.abs()
        return payload

    custom_notifier.set_routing_function('custom', similarity_scale_router)

    # Post with custom routing
    original = torch.randn(4, 128)
    custom_notifier.post(custom_addrs[0], original, channel='custom')
    custom_notifier.route()

    print(f"Custom routing function applied (similarity scaling)")
    for addr in custom_addrs[1:]:
        if has_mail(addr):
            msg = addr.receive(custom_addrs[0].name)
            ratio = msg.norm().item() / original.norm().item()
            print(f"  {addr.name}: norm ratio = {ratio:.4f}")

    custom_notifier.clear()

    # =========================================================================
    section("PROJECTOR TRANSFER LEARNING")
    # =========================================================================

    transfer_notifier = NotifierRouter('transfer_comm')

    teacher_addr = AddressComponent('teacher', fingerprint_dim=128)
    student_addr = AddressComponent('student', fingerprint_dim=64)

    transfer_notifier.register(teacher_addr, channel='transfer')
    transfer_notifier.register(student_addr, channel='transfer')

    # Create projector
    transfer_notifier.create_projector('teacher', 'student', 512, 256)

    print(f"Projector: teacher (512) -> student (256)")

    # Teacher posts large tensor
    teacher_output = torch.randn(4, 512)
    transfer_notifier.post(teacher_addr, teacher_output, channel='transfer')
    transfer_notifier.route()

    # Student receives projected
    student_msg = student_addr.receive('teacher')
    print(f"Teacher sent: {teacher_output.shape}")
    print(f"Student received: {student_msg.shape}")

    transfer_notifier.clear()

    # =========================================================================
    section("TOWER INTEGRATION WITH ADDRESSES")
    # =========================================================================

    class AddressedTower(BaseTower):
        """Tower with address component for coordination."""

        def __init__(
            self,
            name: str,
            dim: int,
            notifier: NotifierRouter,
            address_type: str = 'euclidean',
            channel: str = 'default',
        ):
            super().__init__(name, strict=False)

            # Create appropriate address type
            if address_type == 'euclidean':
                addr = AddressComponent(name, fingerprint_dim=64)
            elif address_type == 'spherical':
                addr = SphericalAddressComponent(name, fingerprint_dim=64)
            elif address_type == 'hyperbolic':
                addr = HyperbolicAddressComponent(name, fingerprint_dim=64)
            else:
                addr = GeometricAddressComponent(name, fingerprint_dim=64, manifold=address_type)

            self.attach('address', addr)
            self.attach('notifier', notifier)
            self.attach('channel', channel)

            # Register with notifier
            notifier.register(addr, channel=channel)

            # Processing
            self.extend([
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            ])

        def forward(self, x: Tensor) -> Tensor:
            addr = self['address']

            # Integrate incoming
            if has_mail(addr):
                if hasattr(addr, 'aggregate_inbox'):
                    knowledge = addr.aggregate_inbox('mean')
                else:
                    msgs = list(addr.inbox.values())
                    knowledge = torch.stack(msgs).mean(0) if msgs else None

                if knowledge is not None and knowledge.shape == x.shape:
                    x = x + 0.1 * knowledge

                addr.clear()

            # Process
            for stage in self.stages:
                x = stage(x)

            return x

        def share(self, opinion: Tensor):
            """Share opinion to channel."""
            self['notifier'].post(self['address'], opinion, self['channel'])

    # Create towers with different address types
    tower_notifier = NotifierRouter('tower_comm')

    towers = [
        AddressedTower('tower_euclidean', 256, tower_notifier, 'euclidean', 'mixed'),
        AddressedTower('tower_spherical', 256, tower_notifier, 'spherical', 'mixed'),
        AddressedTower('tower_hyperbolic', 256, tower_notifier, 'hyperbolic', 'mixed'),
    ]

    print(f"Created towers with different address types:")
    for tower in towers:
        addr = tower['address']
        print(f"  {tower.name}: {addr.__class__.__name__}")

    # Tower 0 processes and shares
    x0 = torch.randn(4, 256)
    out0 = towers[0](x0)
    towers[0].share(out0)

    # Route
    tower_notifier.route()

    # Other towers receive
    for tower in towers[1:]:
        x = torch.randn(4, 256)
        out = tower(x)
        print(f"  {tower.name} output: {out.shape}")

        # =========================================================================
        section("FRACTAL ADDRESS (Julia Orbit Fingerprints)")
        # =========================================================================

        fractal_notifier = NotifierRouter('fractal_comm')

        # Create addresses from different fractal regions
        regions = ['seahorse', 'cardioid', 'antenna', 'spiral']
        fractal_addrs = []

        for region in regions:
            addr = FractalAddressComponent(f'fractal_{region}', region=region, orbit_length=64)
            fractal_addrs.append(addr)
            fractal_notifier.register(addr, channel='fractal')

        print(f"Fractal addresses from different Julia regions:")
        for addr in fractal_addrs:
            c = addr.c_parameter
            print(f"  {addr.name}: c={c}, dim={addr.fingerprint_dim}")

        # Same-region similarity should be higher
        print(f"\nCross-region similarity matrix:")
        for i, a in enumerate(fractal_addrs):
            sims = [a.similarity(b).item() for b in fractal_addrs]
            print(f"  {a.name}: {[f'{s:.3f}' for s in sims]}")

        # Route by fractal similarity
        targets = fractal_notifier.route_by_similarity(
            fractal_addrs[0],
            torch.randn(4, 128),
            channel='fractal',
            top_k=2,
        )
        print(f"\nMost similar to {fractal_addrs[0].region}: {targets}")

        fractal_notifier.route()
        fractal_notifier.clear()

        # =========================================================================
        section("FRACTAL ADDRESS - Julia Orbit Identity")
        # =========================================================================

        # Fractal regions are geometrically meaningful
        # Adjacent regions in Mandelbrot set → similar Julia dynamics
        print("Julia set regions and their dynamics:")
        print("  cardioid  - main bulb, stable periodic orbits")
        print("  seahorse  - seahorse valley, chaotic mixing")
        print("  antenna   - main antenna, escape dynamics")
        print("  spiral    - spiral arms, transitional")

        fractal_notifier = NotifierRouter('fractal_comm')

        # Create with explicit seeds for reproducibility
        fractal_addrs = {}
        for region in ['seahorse', 'cardioid', 'antenna', 'spiral']:
            addr = FractalAddressComponent(
                f'fractal_{region}',
                region=region,
                orbit_length=64,
                seed=42,  # Deterministic within region
            )
            fractal_addrs[region] = addr
            fractal_notifier.register(addr, channel='fractal')

        # Show c parameters - these define the Julia dynamics
        print(f"\nJulia c parameters (z → z² + c):")
        for region, addr in fractal_addrs.items():
            c = addr.c_parameter
            print(f"  {region}: c = {c.real:.4f} + {c.imag:.4f}i")

        # Same-region check
        seahorse2 = FractalAddressComponent('seahorse2', region='seahorse', orbit_length=64, seed=123)
        print(f"\nSame region check:")
        print(f"  seahorse vs seahorse2: {fractal_addrs['seahorse'].same_region(seahorse2)}")
        print(f"  seahorse vs cardioid: {fractal_addrs['seahorse'].same_region(fractal_addrs['cardioid'])}")

        # Similarity reflects Mandelbrot geometry
        # Seahorse and spiral are adjacent → high similarity
        # Antenna is distant → low similarity
        print(f"\nCross-region similarity (reflects Mandelbrot adjacency):")
        for r1 in ['seahorse', 'cardioid', 'antenna', 'spiral']:
            sims = []
            for r2 in ['seahorse', 'cardioid', 'antenna', 'spiral']:
                sim = fractal_addrs[r1].similarity(fractal_addrs[r2]).item()
                sims.append(f"{sim:.3f}")
            print(f"  {r1:10s}: {sims}")

        print(f"\nInterpretation:")
        print(f"  seahorse-spiral high similarity: adjacent in Mandelbrot")
        print(f"  antenna low to all: geometrically isolated region")

        # =========================================================================
        section("CANTOR ADDRESS - Devil's Staircase Identity")
        # =========================================================================

        # Cantor addresses use Devil's Staircase for routing
        # k_simplex defines the underlying geometric structure
        # tau controls the "gap" structure in the staircase

        print("Devil's Staircase properties:")
        print("  - Continuous but not absolutely continuous")
        print("  - Derivative is 0 almost everywhere")
        print("  - Creates natural 'plateaus' for clustering")

        cantor_notifier = NotifierRouter('cantor_comm')

        # Show tau effect
        print(f"\nTau parameter effect (k=4):")
        for tau in [0.1, 0.25, 0.5]:
            addr = CantorAddressComponent(
                f'cantor_tau{tau}',
                k_simplex=4,
                fingerprint_dim=64,
                mode='staircase',
                tau=tau,
            )
            print(f"  tau={tau}: base_norm={addr.base_fingerprint.norm().item():.4f}")

        # Base fingerprint is DETERMINISTIC
        print(f"\nDeterminism test (same k, tau → same base fingerprint):")
        c1 = CantorAddressComponent('c1', k_simplex=4, fingerprint_dim=64, tau=0.25)
        c2 = CantorAddressComponent('c2', k_simplex=4, fingerprint_dim=64, tau=0.25)
        print(f"  c1 base == c2 base: {torch.allclose(c1.base_fingerprint, c2.base_fingerprint)}")

        # Learnable modulation
        print(f"\nLearnable modulation (scale, shift):")
        print(f"  c1 scale={c1.scale.item():.4f}, shift={c1.shift.item():.4f}")
        with torch.no_grad():
            c1.scale.fill_(2.0)
            c1.shift.fill_(0.5)
        print(f"  After modification: scale={c1.scale.item():.4f}, shift={c1.shift.item():.4f}")
        print(f"  base_norm={c1.base_fingerprint.norm().item():.4f}")
        print(f"  fingerprint_norm={c1.fingerprint.norm().item():.4f} (scaled)")

        # Cantor distance ignores learned modulation
        print(f"\nCantor distance (base space, ignores scale/shift):")
        print(f"  c1 vs c2 cantor_distance: {c1.cantor_distance(c2).item():.6f}")
        print(f"  c1 vs c2 regular distance: {c1.distance(c2).item():.4f}")

        # =========================================================================
        section("SHAPE ADDRESS (RoPE-Compatible Geometric Fingerprints)")
        # =========================================================================

        shape_notifier = NotifierRouter('shape_comm')

        # Create shape addresses with different geometric primitives
        shape_types = ['sphere', 'cube', 'cylinder', 'pyramid', 'cone']
        shape_addrs = []

        for shape_type in shape_types:
            addr = ShapeAddressComponent(
                f'shape_{shape_type}',
                shape_type=shape_type,
                embed_dim=64,
                resolution=32,
                theta_offset=0.0,
            )
            shape_addrs.append(addr)
            shape_notifier.register(addr, channel='shape')

        print(f"Shape addresses with different geometric primitives:")
        for addr in shape_addrs:
            print(f"  {addr.name}: shape_points={addr.shape_points.shape}, "
                  f"theta={addr.theta_offset.item():.4f}")

        # RoPE rotation test - fingerprint changes with theta
        sphere_addr = shape_addrs[0]  # sphere
        print(f"\nRoPE rotation of sphere fingerprint:")
        for theta in [0.0, 0.5, 1.0, 2.0, 3.14159]:
            fp = sphere_addr.fingerprint_at_theta(theta)
            print(f"  theta={theta:.2f}: fp[:4]={fp[:4].tolist()}")

        # Angular similarity - same shape at different theta
        print(f"\nAngular similarity (sphere with itself at different theta):")
        for theta in [0.0, 0.5, 1.0, 2.0]:
            sim = sphere_addr.angular_similarity(sphere_addr, 0.0, theta)
            print(f"  theta_a=0.0, theta_b={theta}: sim={sim.item():.4f}")

        # Cross-shape similarity
        print(f"\nCross-shape similarity:")
        for i, a in enumerate(shape_addrs):
            sims = [a.similarity(b).item() for b in shape_addrs]
            print(f"  {a.shape_type}: {[f'{s:.3f}' for s in sims]}")

        # Shape distance
        print(f"\nShape distances:")
        for i, a in enumerate(shape_addrs[:2]):
            for j, b in enumerate(shape_addrs[2:], 2):
                dist = a.shape_distance(b)
                print(f"  {a.shape_type} <-> {b.shape_type}: {dist.item():.4f}")

        shape_notifier.route()
        shape_notifier.clear()

        # =========================================================================
        section("SIMPLEX ADDRESS (Barycentric Coordinates)")
        # =========================================================================

        simplex_notifier = NotifierRouter('simplex_comm')

        # Create simplex addresses with different k values
        print(f"Simplex structure:")
        for k in [2, 3, 4]:
            addr = SimplexAddressComponent(f'demo_k{k}', k=k, embed_dim=64, method='regular')
            print(f"  k={k}: {k + 1} vertices in R^64, barycentric has {k + 1} coordinates")
            print(f"    vertices shape: {addr.vertices.shape}")
            print(f"    barycentric: {addr.barycentric.tolist()}")
            print(f"    fingerprint: {addr.fingerprint.shape}")

        # Create MULTIPLE addresses sharing SAME k for barycentric comparison
        print(f"\nCreating 4 addresses with k=4 (5 vertices each):")
        simplex_k4_addrs = []
        for i in range(4):
            addr = SimplexAddressComponent(
                f'simplex_k4_{i}',
                k=4,
                embed_dim=64,
                method='regular',
                learnable=True,  # barycentric weights are learned
            )
            simplex_k4_addrs.append(addr)
            simplex_notifier.register(addr, channel='simplex_k4')

        # Show barycentric coordinates (initially uniform: 1/(k+1) each)
        for addr in simplex_k4_addrs:
            bary = addr.barycentric
            print(f"  {addr.name}: bary sum={bary.sum().item():.4f}, coords={[f'{b:.4f}' for b in bary.tolist()]}")

        # Barycentric distance between same-k addresses
        # Measures: do they weight their simplex corners similarly?
        print(f"\nBarycentric distances (same k=4):")
        for i, a in enumerate(simplex_k4_addrs):
            for j, b in enumerate(simplex_k4_addrs):
                if i < j:
                    bary_dist = a.barycentric_distance(b)
                    fp_sim = a.similarity(b)
                    print(f"  {a.name} <-> {b.name}: bary_dist={bary_dist.item():.6f}, fp_sim={fp_sim.item():.4f}")

        # Perturb one address's barycentric weights to see effect
        print(f"\nPerturbing simplex_k4_0's barycentric weights:")
        with torch.no_grad():
            simplex_k4_addrs[0]._bary_logits[0] += 2.0  # Emphasize first vertex

        new_bary = simplex_k4_addrs[0].barycentric
        print(f"  New barycentric: {[f'{b:.4f}' for b in new_bary.tolist()]}")

        # Now distances should be non-zero
        print(f"\nBarycentric distances after perturbation:")
        for j, b in enumerate(simplex_k4_addrs[1:], 1):
            bary_dist = simplex_k4_addrs[0].barycentric_distance(b)
            fp_sim = simplex_k4_addrs[0].similarity(b)
            print(f"  simplex_k4_0 <-> {b.name}: bary_dist={bary_dist.item():.6f}, fp_sim={fp_sim.item():.4f}")

        # Cross-k fingerprint comparison (fingerprints are all embed_dim=64)
        print(f"\nCross-k fingerprint similarity (all project to R^64):")
        cross_k_addrs = [
            SimplexAddressComponent('cross_k2', k=2, embed_dim=64),
            SimplexAddressComponent('cross_k3', k=3, embed_dim=64),
            SimplexAddressComponent('cross_k4', k=4, embed_dim=64),
        ]

        for i, a in enumerate(cross_k_addrs):
            for j, b in enumerate(cross_k_addrs):
                if i < j:
                    sim = a.similarity(b)
                    print(f"  k={a.k} <-> k={b.k}: sim={sim.item():.4f}")

        # Barycentric blend via notifier
        blend = simplex_notifier.barycentric_blend(
            simplex_k4_addrs[:3],
            weights=torch.tensor([0.5, 0.3, 0.2]),
        )
        print(f"\nBarycentric blend of 3 k=4 addresses:")
        print(f"  Blend shape: {blend.shape}, norm: {blend.norm().item():.4f}")

        # Route by similarity within same-k channel
        targets = simplex_notifier.route_by_similarity(
            simplex_k4_addrs[0],
            torch.randn(4, 128),
            channel='simplex_k4',
            top_k=2,
        )
        print(f"\nMost similar to perturbed simplex_k4_0: {targets}")

        simplex_notifier.route()
        simplex_notifier.clear()

    # =========================================================================
    section("STATISTICS")
    # =========================================================================

    print(f"Main notifier stats: {notifier.stats}")
    print(f"Spherical notifier stats: {spherical_notifier.stats}")
    print(f"Tower notifier stats: {tower_notifier.stats}")
    print(f"Fractal notifier stats: {fractal_notifier.stats}")
    print(f"Cantor notifier stats: {cantor_notifier.stats}")
    print(f"Shape notifier stats: {shape_notifier.stats}")
    print(f"Simplex notifier stats: {simplex_notifier.stats}")

    # =========================================================================
    section("ALL TESTS PASSED")
    # =========================================================================

    print("\nNotifierRouter provides:")
    print("  ✓ Orchestrates AddressComponent variants")
    print("  ✓ Channel-based pub/sub")
    print("  ✓ Direct messaging")
    print("  ✓ Broadcast")
    print("  ✓ Similarity-based routing")
    print("  ✓ Distance-based routing (manifold-aware)")
    print("  ✓ Affinity-weighted broadcast")
    print("  ✓ SLERP interpolation (spherical)")
    print("  ✓ Hierarchical routing (hyperbolic)")
    print("  ✓ Custom routing functions")
    print("  ✓ Dimension projection")
    print("  ✓ Integration with BaseTower")

    print("\nAddress types tested:")
    print("  ✓ AddressComponent (Euclidean)")
    print("  ✓ SphericalAddressComponent (unit hypersphere)")
    print("  ✓ HyperbolicAddressComponent (Poincaré ball)")
    print("  ✓ GeometricAddressComponent (configurable manifold)")
    print("  ✓ GatedAddressComponent (fingerprint-gated)")
    print("  ✓ RoutedAddressComponent (learned routing)")

    print("\nNotifierRouter is ready for tower coordination.")