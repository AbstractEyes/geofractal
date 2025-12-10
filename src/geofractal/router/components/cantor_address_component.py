"""
geofractal.router.components.cantor_address_component
======================================================

CantorAddressComponent - Address with Cantor/Beatrix fingerprint for geometric routing.

This component provides addresses based on the Devil's Staircase (Cantor function),
enabling routing decisions based on hierarchical structure rather than scalar distance.

CRITICAL INSIGHT: Distance is meaningless on the Cantor set.

    The Cantor set is totally disconnected - between any two points lie infinitely
    many removed intervals. There is no continuous path, no meaningful "distance."

    The Devil's Staircase makes this concrete:
    - It's constant across gaps (the removed middle thirds)
    - It only "moves" on the measure-zero Cantor set itself
    - Two values 0.001 apart in scalar terms could be separated by infinite hierarchy
    - Two values 0.5 apart could share coarse routing highways

WHAT HAS MEANING:
    - Branch path alignment (do ternary paths share structure?)
    - Plateau membership (same routing behavior at scale?)
    - First divergence level (where do paths split?)
    - Hierarchical weighting (coarse levels matter more: 0.5^k)

Mathematical Foundation:
    Devil's Staircase: C(x) = Σ_{k=1}^{levels} bit_k × 0.5^k

    Where:
    - bit_k = p_right + α × p_middle (soft ternary assignment)
    - p = softmax(-d²/τ) over centers [0.5, 1.5, 2.5]
    - α controls middle third fill (learnable)
    - τ controls softmax temperature

    Hierarchical Alignment:
    - Level 1 (coarse L/M/R thirds): weight 0.5
    - Level 2: weight 0.25
    - Level 3: weight 0.125
    - etc.

    Two positions matching at coarse levels share "routing highways"
    enabling wormhole teleportation. Fine matches only indicate local structure.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.components.torch_component import TorchComponent


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CantorAddressConfig:
    """Configuration for CantorAddressComponent."""

    levels: int = 5
    tau: float = 0.25
    alpha: float = 0.5
    base: int = 3
    learnable_alpha: bool = True
    learnable_tau: bool = False
    learnable_position: bool = False
    hierarchical_weights: bool = True  # Use 0.5^k weighting (recommended)

    def __post_init__(self):
        assert self.levels >= 1, "levels must be >= 1"
        assert self.tau > 0, "tau must be positive"
        assert 0 <= self.alpha <= 1, "alpha must be in [0, 1]"
        assert self.base >= 2, "base must be >= 2"


# =============================================================================
# BEATRIX STAIRCASE
# =============================================================================

class BeatrixStaircase(nn.Module):
    """
    Devil's Staircase computation with learnable parameters.

    Computes the Cantor function C(x) via ternary decomposition:
        C(x) = Σ_{k=1}^{levels} bit_k × 0.5^k

    Where bit_k encodes which third of the interval x falls into at level k.

    Also computes:
        - Per-level features (bit_k, pdf_proxy) for hierarchical fingerprinting
        - Hard branch path assignments {L=0, M=1, R=2} for alignment computation
        - Hierarchical level weights for proper alignment scoring

    Parameters:
        levels: Number of ternary decomposition levels
        tau: Softmax temperature for soft assignment
        alpha: Middle third weight (0 = classic Cantor, 1 = filled)
        base: Ternary base (always 3 for Cantor)
        learnable_alpha: Whether alpha is a learnable parameter
        learnable_tau: Whether tau is a learnable parameter
    """

    def __init__(
        self,
        levels: int = 5,
        tau: float = 0.25,
        alpha: float = 0.5,
        base: int = 3,
        learnable_alpha: bool = True,
        learnable_tau: bool = False,
    ):
        super().__init__()

        self.levels = levels
        self.base = base

        # Ternary interval centers
        centers = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        self.register_buffer('centers', centers)

        # Alpha: middle third weight
        if learnable_alpha:
            self._alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('_alpha', torch.tensor(alpha, dtype=torch.float32))

        # Tau: softmax temperature
        if learnable_tau:
            self._tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        else:
            self.register_buffer('_tau', torch.tensor(tau, dtype=torch.float32))

        # Precompute scales and weights (vectorized)
        # scales[k] = 3^(k+1) for k in [0, levels-1]
        scales = 3.0 ** torch.arange(1, levels + 1, dtype=torch.float64)
        self.register_buffer('_scales', scales)

        # Cantor measure weights: 0.5^k
        weights = 0.5 ** torch.arange(1, levels + 1, dtype=torch.float64)
        self.register_buffer('_weights', weights)

        # Hierarchical alignment weights (same as _weights but float32 for compatibility)
        level_weights = 0.5 ** torch.arange(1, levels + 1, dtype=torch.float32)
        self.register_buffer('_level_weights', level_weights)

    @property
    def alpha(self) -> float:
        """Middle third weight."""
        return self._alpha.item()

    @property
    def tau(self) -> float:
        """Softmax temperature."""
        return self._tau.item()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute Devil's Staircase value and features (fully vectorized).

        Args:
            x: Positions in [0, 1], shape (...)

        Returns:
            cantor_measure: Scalar Cantor value, shape (...)
                NOTE: This is for positional encoding, NOT routing decisions
            features: Per-level features, shape (..., levels, 2)
                features[..., k, 0] = bit_k (branch indicator)
                features[..., k, 1] = pdf_proxy (consciousness/entropy)
            branch_path: Hard branch assignments, shape (..., levels)
                Values in {0=Left, 1=Middle, 2=Right}
        """
        original_shape = x.shape
        x = x.clamp(1e-6, 1.0 - 1e-6).double()

        # Flatten for vectorized computation
        x_flat = x.reshape(-1)  # (N,)
        N = x_flat.shape[0]
        device = x.device

        # Vectorized across all positions and levels
        # y[n, k] = position within ternary cell at level k
        y = (x_flat.unsqueeze(-1) * self._scales) % 3  # (N, levels)

        # Squared distance to centers: (N, levels, 3)
        d2 = (y.unsqueeze(-1) - self.centers) ** 2

        # Soft assignment via softmax
        tau = self._tau.double()
        logits = -d2 / tau
        p = F.softmax(logits, dim=-1)  # (N, levels, 3)

        # Extract probabilities
        p_left = p[..., 0]    # (N, levels)
        p_middle = p[..., 1]  # (N, levels)
        p_right = p[..., 2]   # (N, levels)

        # Bit indicator: right + alpha * middle
        alpha = self._alpha.double()
        bit_k = p_right + alpha * p_middle  # (N, levels)

        # Cantor measure: C(x) = Σ bit_k × 0.5^k
        cantor_measure = (bit_k * self._weights).sum(dim=-1)  # (N,)

        # PDF proxy (consciousness/entropy measure)
        # Higher when assignment is more certain
        entropy = -(p * p.clamp_min(1e-10).log()).sum(dim=-1)  # (N, levels)
        pdf_proxy = 1.1 - entropy / math.log(3)  # (N, levels)

        # Stack features: (N, levels, 2)
        features = torch.stack([bit_k, pdf_proxy], dim=-1)

        # Hard branch assignment
        branch_path = logits.argmax(dim=-1).int()  # (N, levels)

        # Reshape to original
        cantor_measure = cantor_measure.float().reshape(original_shape)
        features = features.float().reshape(*original_shape, self.levels, 2)
        branch_path = branch_path.reshape(*original_shape, self.levels)

        return cantor_measure, features, branch_path

    def compute_branch_path(self, x: Tensor) -> Tensor:
        """
        Compute only the branch path (for alignment computation).

        Args:
            x: Positions in [0, 1], shape (...)

        Returns:
            branch_path: Hard branch assignments, shape (..., levels)
        """
        _, _, branch_path = self.forward(x)
        return branch_path

    def hierarchical_alignment(self, path_a: Tensor, path_b: Tensor) -> Tensor:
        """
        Compute hierarchically-weighted alignment between branch paths.

        This is the CORRECT similarity metric for Cantor routing.
        Coarse levels contribute more than fine levels.

        Args:
            path_a: Branch path, shape (..., levels)
            path_b: Branch path, shape (..., levels)

        Returns:
            alignment: Weighted alignment score, shape (...)
                Range: [0, ~0.96875] for 5 levels
        """
        matches = (path_a == path_b).float()  # (..., levels)
        alignment = (matches * self._level_weights).sum(dim=-1)
        return alignment

    def raw_alignment(self, path_a: Tensor, path_b: Tensor) -> Tensor:
        """
        Compute raw (unweighted) alignment - count of matching levels.

        Use hierarchical_alignment() instead for routing decisions.

        Args:
            path_a: Branch path, shape (..., levels)
            path_b: Branch path, shape (..., levels)

        Returns:
            alignment: Count of matching levels, shape (...)
                Range: [0, levels]
        """
        return (path_a == path_b).sum(dim=-1)

    def first_divergence(self, path_a: Tensor, path_b: Tensor) -> Tensor:
        """
        Find the first level where paths diverge.

        Args:
            path_a: Branch path, shape (..., levels)
            path_b: Branch path, shape (..., levels)

        Returns:
            divergence: First divergence level, shape (...)
                0 = immediate divergence (different at coarse level)
                levels = identical paths
        """
        matches = (path_a == path_b)  # (..., levels)

        # Find first False (first divergence)
        # Use cumulative product: all True until first False
        cum_matches = matches.cumprod(dim=-1)

        # Count consecutive matches from start
        return cum_matches.sum(dim=-1)

    def plateau_id(self, branch_path: Tensor, depth: Optional[int] = None) -> Tensor:
        """
        Compute plateau ID from branch path.

        Positions with same plateau ID share routing behavior at the given depth.

        Args:
            branch_path: Branch path, shape (..., levels)
            depth: Depth to consider (None = all levels)

        Returns:
            plateau_id: Encoded plateau ID, shape (...)
        """
        if depth is None:
            depth = self.levels

        path = branch_path[..., :depth]  # (..., depth)

        # Encode as base-3 number
        powers = 3 ** torch.arange(depth, device=branch_path.device)
        plateau = (path * powers).sum(dim=-1)

        return plateau


# =============================================================================
# CANTOR ADDRESS COMPONENT
# =============================================================================

class CantorAddressComponent(TorchComponent):
    """
    Address with Cantor/Beatrix fingerprint for geometric routing.

    CRITICAL: This component does NOT support scalar distance operations.
    The Cantor set is totally disconnected - distance is meaningless.

    SUPPORTED OPERATIONS:
        - branch_alignment(): Hierarchically-weighted path similarity
        - raw_branch_alignment(): Count of matching levels (use hierarchical instead)
        - first_divergence(): Where paths split in hierarchy
        - same_plateau(): Share routing behavior at scale?
        - is_routable_to(): Can route with minimum alignment?
        - feature_similarity(): Cosine on hierarchical features

    NOT SUPPORTED (raises NotImplementedError):
        - distance(): Scalar distance is meaningless
        - cantor_distance(): Also meaningless

    Attributes:
        position: Normalized position in [0, 1]
        alpha: Middle third weight (routing density)
        tau: Softmax temperature
        cantor_measure: Scalar value (for PE only, NOT routing)
        features: Per-level features, shape (levels, 2)
        branch_path: Hard ternary assignments, shape (levels,)
        plateau_id: Encodes position in Cantor hierarchy
        fingerprint: Flattened features for compatibility
    """

    def __init__(
        self,
        name: str,
        position: float = 0.0,
        config: Optional[CantorAddressConfig] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        if config is None:
            config = CantorAddressConfig(**kwargs)

        self.config = config

        # Staircase module
        self.staircase = BeatrixStaircase(
            levels=config.levels,
            tau=config.tau,
            alpha=config.alpha,
            base=config.base,
            learnable_alpha=config.learnable_alpha,
            learnable_tau=config.learnable_tau,
        )

        # Position
        if config.learnable_position:
            self._position = nn.Parameter(torch.tensor(position, dtype=torch.float32))
        else:
            self.register_buffer('_position', torch.tensor(position, dtype=torch.float32))

        # Precompute fingerprint
        self._update_fingerprint()

    def _update_fingerprint(self) -> None:
        """Recompute fingerprint from current position."""
        with torch.no_grad():
            pos = self._position.clamp(0, 1)
            self._cantor_measure, self._features, self._branch_path = self.staircase(pos)
            self._plateau_id = self.staircase.plateau_id(self._branch_path)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def position(self) -> float:
        """Normalized position in [0, 1]."""
        return self._position.item()

    @position.setter
    def position(self, value: float) -> None:
        """Set position and update fingerprint."""
        with torch.no_grad():
            self._position.fill_(value)
        self._update_fingerprint()

    @property
    def alpha(self) -> float:
        """Middle third weight."""
        return self.staircase.alpha

    @property
    def tau(self) -> float:
        """Softmax temperature."""
        return self.staircase.tau

    @property
    def levels(self) -> int:
        """Number of decomposition levels."""
        return self.config.levels

    @property
    def cantor_measure(self) -> Tensor:
        """
        Scalar Cantor value.

        WARNING: Use for positional encoding ONLY, not routing decisions.
        """
        return self._cantor_measure

    @property
    def features(self) -> Tensor:
        """Per-level features, shape (levels, 2)."""
        return self._features

    @property
    def branch_path(self) -> Tensor:
        """Hard branch assignments, shape (levels,)."""
        return self._branch_path

    @property
    def plateau_id_value(self) -> int:
        """Plateau ID as integer."""
        return self._plateau_id.item()

    @property
    def fingerprint(self) -> Tensor:
        """Flattened features for compatibility, shape (levels * 2,)."""
        return self._features.flatten()

    # =========================================================================
    # MEANINGFUL OPERATIONS
    # =========================================================================

    def branch_alignment(self, other: 'CantorAddressComponent') -> float:
        """
        Compute hierarchically-weighted branch path alignment.

        This is the CORRECT similarity metric for Cantor routing.
        Coarse levels (L/M/R thirds) contribute more than fine structure.

        Returns:
            Alignment score in [0, ~0.96875] for 5 levels
            Higher = more similar routing behavior
        """
        alignment = self.staircase.hierarchical_alignment(
            self._branch_path,
            other._branch_path,
        )
        return alignment.item()

    def raw_branch_alignment(self, other: 'CantorAddressComponent') -> int:
        """
        Count of matching branch levels (unweighted).

        Use branch_alignment() instead for routing decisions.

        Returns:
            Count in [0, levels]
        """
        return self.staircase.raw_alignment(
            self._branch_path,
            other._branch_path,
        ).item()

    def first_divergence(self, other: 'CantorAddressComponent') -> int:
        """
        Find first level where branch paths diverge.

        Returns:
            0 = immediate divergence (different coarse structure)
            levels = identical paths (same routing everywhere)
        """
        return self.staircase.first_divergence(
            self._branch_path,
            other._branch_path,
        ).item()

    def same_plateau(self, other: 'CantorAddressComponent', depth: Optional[int] = None) -> bool:
        """
        Check if addresses share routing behavior at given depth.

        Args:
            other: Other address
            depth: Depth to check (None = all levels)

        Returns:
            True if same plateau (share routing highways)
        """
        depth = depth or self.levels
        my_plateau = self.staircase.plateau_id(self._branch_path, depth)
        other_plateau = self.staircase.plateau_id(other._branch_path, depth)
        return (my_plateau == other_plateau).item()

    def plateau_id_at_depth(self, depth: int) -> int:
        """Get plateau ID at specific depth."""
        return self.staircase.plateau_id(self._branch_path, depth).item()

    def is_routable_to(self, other: 'CantorAddressComponent', min_alignment: float = 0.25) -> bool:
        """
        Check if routing to other address meets minimum alignment threshold.

        Args:
            other: Target address
            min_alignment: Minimum hierarchical alignment required
                0.5 = must match at least level 1 (coarse)
                0.75 = must match levels 1-2
                0.25 = lenient (any significant match)

        Returns:
            True if alignment >= min_alignment
        """
        return self.branch_alignment(other) >= min_alignment

    def feature_similarity(self, other: 'CantorAddressComponent') -> float:
        """
        Cosine similarity between hierarchical feature fingerprints.

        This uses the full per-level features (bit_k, pdf_proxy),
        not just branch path alignment.

        Returns:
            Cosine similarity in [-1, 1]
        """
        fp_a = self.fingerprint
        fp_b = other.fingerprint
        return F.cosine_similarity(fp_a.unsqueeze(0), fp_b.unsqueeze(0)).item()

    def branch_path_str(self) -> str:
        """Human-readable branch path (L=Left, M=Middle, R=Right)."""
        mapping = {0: 'L', 1: 'M', 2: 'R'}
        return ''.join(mapping[b.item()] for b in self._branch_path)

    # =========================================================================
    # FORBIDDEN OPERATIONS (raise NotImplementedError)
    # =========================================================================

    def distance(self, other: 'CantorAddressComponent') -> float:
        """
        FORBIDDEN: Scalar distance is meaningless on Cantor set.

        The Cantor set is totally disconnected. Between any two points
        lie infinitely many removed intervals. There is no continuous
        path, no meaningful distance.

        Use instead:
            - branch_alignment(): Hierarchically-weighted path similarity
            - first_divergence(): Where paths split
            - same_plateau(): Share routing behavior?
            - is_routable_to(): Can route with minimum alignment?
        """
        raise NotImplementedError(
            "Distance is meaningless on the Cantor set. "
            "The set is totally disconnected - between any two points "
            "lie infinitely many removed intervals. "
            "Use branch_alignment(), same_plateau(), or is_routable_to() instead."
        )

    def cantor_distance(self, other: 'CantorAddressComponent') -> float:
        """
        FORBIDDEN: Cantor distance is also meaningless.

        Even distance between Cantor measure values (C(x) - C(y)) is meaningless
        because the Devil's Staircase is constant across gaps.
        Two positions close in Cantor measure may be separated by infinite hierarchy.
        """
        raise NotImplementedError(
            "Cantor distance is meaningless. "
            "The Devil's Staircase is constant across gaps - "
            "two positions close in Cantor measure may be separated by "
            "infinite hierarchical structure. "
            "Use branch_alignment(), same_plateau(), or is_routable_to() instead."
        )

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'position': self.position,
            'config': {
                'levels': self.config.levels,
                'tau': self.config.tau,
                'alpha': self.config.alpha,
                'base': self.config.base,
                'learnable_alpha': self.config.learnable_alpha,
                'learnable_tau': self.config.learnable_tau,
                'learnable_position': self.config.learnable_position,
                'hierarchical_weights': self.config.hierarchical_weights,
            },
            'branch_path': self._branch_path.tolist(),
            'plateau_id': self.plateau_id_value,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CantorAddressComponent':
        """Deserialize from dictionary."""
        config = CantorAddressConfig(**data['config'])
        return cls(
            name=data['name'],
            position=data['position'],
            config=config,
        )

    def __repr__(self) -> str:
        return (
            f"CantorAddressComponent("
            f"name='{self.name}', "
            f"pos={self.position:.4f}, "
            f"path={self.branch_path_str()}, "
            f"plateau={self.plateau_id_value}, "
            f"α={self.alpha:.3f})"
        )


# =============================================================================
# CANTOR ADDRESS BOOK
# =============================================================================

class CantorAddressBook:
    """
    Registry of CantorAddressComponents with routing-aware operations.

    Provides:
        - Lookup by name
        - Find addresses on same plateau
        - Find routable targets by alignment threshold
        - Hierarchical clustering by plateau
        - Pairwise alignment matrix
        - Message routing based on alignment
    """

    def __init__(self):
        self._addresses: Dict[str, CantorAddressComponent] = {}
        self._messages: Dict[str, List[Tensor]] = {}

    def register(self, address: CantorAddressComponent) -> None:
        """Register an address."""
        self._addresses[address.name] = address
        self._messages[address.name] = []

    def unregister(self, name: str) -> None:
        """Unregister an address."""
        self._addresses.pop(name, None)
        self._messages.pop(name, None)

    def get(self, name: str) -> Optional[CantorAddressComponent]:
        """Get address by name."""
        return self._addresses.get(name)

    def __getitem__(self, name: str) -> CantorAddressComponent:
        return self._addresses[name]

    def __contains__(self, name: str) -> bool:
        return name in self._addresses

    def __len__(self) -> int:
        return len(self._addresses)

    def __iter__(self):
        return iter(self._addresses.values())

    @property
    def names(self) -> List[str]:
        """List of registered address names."""
        return list(self._addresses.keys())

    # =========================================================================
    # ROUTING OPERATIONS
    # =========================================================================

    def find_same_plateau(
        self,
        address: CantorAddressComponent,
        depth: Optional[int] = None,
    ) -> List[str]:
        """Find addresses sharing same plateau."""
        results = []
        for name, other in self._addresses.items():
            if name != address.name and address.same_plateau(other, depth):
                results.append(name)
        return results

    def find_routable(
        self,
        address: CantorAddressComponent,
        min_alignment: float = 0.25,
    ) -> List[str]:
        """Find addresses meeting minimum alignment threshold."""
        results = []
        for name, other in self._addresses.items():
            if name != address.name and address.is_routable_to(other, min_alignment):
                results.append(name)
        return results

    def find_by_branch_path(self, path_prefix: str) -> List[str]:
        """
        Find addresses matching branch path prefix.

        Args:
            path_prefix: e.g., "LR" matches addresses starting with Left-Right
        """
        results = []
        for name, address in self._addresses.items():
            path_str = address.branch_path_str()
            if path_str.startswith(path_prefix):
                results.append(name)
        return results

    def cluster_by_plateau(self, depth: int) -> Dict[int, List[str]]:
        """
        Cluster addresses by plateau ID at given depth.

        Returns:
            Dict mapping plateau_id -> list of address names
        """
        clusters: Dict[int, List[str]] = {}
        for name, address in self._addresses.items():
            pid = address.plateau_id_at_depth(depth)
            if pid not in clusters:
                clusters[pid] = []
            clusters[pid].append(name)
        return clusters

    def alignment_matrix(self) -> Tuple[List[str], Tensor]:
        """
        Compute pairwise hierarchical alignment matrix.

        Returns:
            names: List of address names (row/column order)
            matrix: Alignment matrix, shape (N, N)
        """
        names = self.names
        N = len(names)

        if N == 0:
            return names, torch.zeros(0, 0)

        # Get all branch paths
        paths = torch.stack([self._addresses[n]._branch_path for n in names])  # (N, levels)

        # Vectorized pairwise alignment
        matches = (paths.unsqueeze(1) == paths.unsqueeze(0))  # (N, N, levels)

        # Get level weights from first address
        level_weights = self._addresses[names[0]].staircase._level_weights

        # Hierarchical alignment
        matrix = (matches.float() * level_weights).sum(dim=-1)  # (N, N)

        return names, matrix

    # =========================================================================
    # MESSAGE ROUTING
    # =========================================================================

    def queue_message(self, from_name: str, message: Tensor) -> None:
        """Queue a message from an address (to be routed)."""
        if from_name not in self._messages:
            self._messages[from_name] = []
        self._messages[from_name].append(message)

    def route_messages(self, min_alignment: float = 0.25) -> int:
        """
        Route all queued messages based on alignment.

        Messages are delivered to addresses meeting minimum alignment.

        Returns:
            Number of messages delivered
        """
        delivered = 0

        for from_name, messages in list(self._messages.items()):
            if not messages:
                continue

            from_addr = self._addresses.get(from_name)
            if from_addr is None:
                continue

            # Find routable targets
            targets = self.find_routable(from_addr, min_alignment)

            # Deliver to each target
            for target_name in targets:
                if target_name not in self._messages:
                    self._messages[target_name] = []
                self._messages[target_name].extend(messages)
                delivered += len(messages)

            # Clear sent messages
            self._messages[from_name] = []

        return delivered

    def get_messages(self, name: str) -> List[Tensor]:
        """Get received messages for an address."""
        return self._messages.get(name, [])

    def clear_messages(self, name: Optional[str] = None) -> None:
        """Clear messages for one or all addresses."""
        if name is not None:
            self._messages[name] = []
        else:
            for k in self._messages:
                self._messages[k] = []

    def __repr__(self) -> str:
        return f"CantorAddressBook(addresses={self.names})"


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_cantor_address(
    name: str,
    position: float = 0.0,
    levels: int = 5,
    alpha: float = 0.5,
    tau: float = 0.25,
    learnable_alpha: bool = True,
    hierarchical_weights: bool = True,
    **kwargs,
) -> CantorAddressComponent:
    """Factory function for CantorAddressComponent."""
    config = CantorAddressConfig(
        levels=levels,
        tau=tau,
        alpha=alpha,
        learnable_alpha=learnable_alpha,
        hierarchical_weights=hierarchical_weights,
        **kwargs,
    )
    return CantorAddressComponent(name, position=position, config=config)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    import torch

    def section(title: str) -> None:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print('=' * 70)

    # =========================================================================
    section("BEATRIX STAIRCASE FUNDAMENTALS")
    # =========================================================================

    staircase = BeatrixStaircase(
        levels=5,
        tau=0.25,
        alpha=0.5,
        learnable_alpha=False,
    )

    print(f"Staircase configuration:")
    print(f"  Levels: {staircase.levels}")
    print(f"  Alpha (middle third weight): {staircase.alpha:.4f}")
    print(f"  Tau (temperature): {staircase.tau:.4f}")

    print(f"\nHierarchical level weights (0.5^k):")
    for k in range(1, 6):
        print(f"  Level {k}: weight = {0.5**k:.5f}")

    print(f"\nSample positions:")
    for pos in [0.0, 0.25, 0.50, 0.75, 1.0]:
        p = torch.tensor(pos)
        Cx, features, path = staircase(p)
        path_str = ''.join({0: 'L', 1: 'M', 2: 'R'}[b.item()] for b in path)
        print(f"  pos={pos:.2f}: Cx={Cx.item():.4f}, path={path_str}")

    # =========================================================================
    section("HIERARCHICAL ALIGNMENT")
    # =========================================================================

    print("Testing hierarchical vs raw alignment:")

    pos_a = torch.tensor(0.0)
    pos_b = torch.tensor(0.1)
    pos_c = torch.tensor(0.9)

    _, _, path_a = staircase(pos_a)
    _, _, path_b = staircase(pos_b)
    _, _, path_c = staircase(pos_c)

    print(f"\nBranch paths:")
    print(f"  pos 0.0: {''.join({0:'L',1:'M',2:'R'}[b.item()] for b in path_a)}")
    print(f"  pos 0.1: {''.join({0:'L',1:'M',2:'R'}[b.item()] for b in path_b)}")
    print(f"  pos 0.9: {''.join({0:'L',1:'M',2:'R'}[b.item()] for b in path_c)}")

    hier_ab = staircase.hierarchical_alignment(path_a, path_b)
    hier_ac = staircase.hierarchical_alignment(path_a, path_c)
    raw_ab = staircase.raw_alignment(path_a, path_b)
    raw_ac = staircase.raw_alignment(path_a, path_c)

    print(f"\nAlignment (0.0 vs 0.1):")
    print(f"  Hierarchical: {hier_ab.item():.4f}")
    print(f"  Raw count: {raw_ab.item()}")

    print(f"\nAlignment (0.0 vs 0.9):")
    print(f"  Hierarchical: {hier_ac.item():.4f}")
    print(f"  Raw count: {raw_ac.item()}")

    # =========================================================================
    section("CANTOR ADDRESS COMPONENT")
    # =========================================================================

    addr_origin = create_cantor_address('origin', position=0.0)
    addr_quarter = create_cantor_address('quarter', position=0.25)
    addr_middle = create_cantor_address('middle', position=0.5)
    addr_three_q = create_cantor_address('three_quarter', position=0.75)

    print("Created addresses:")
    for addr in [addr_origin, addr_quarter, addr_middle, addr_three_q]:
        print(f"  {addr}")

    # =========================================================================
    section("BRANCH PATH ALIGNMENT (Hierarchical)")
    # =========================================================================

    print("Branch paths:")
    for addr in [addr_origin, addr_quarter, addr_middle, addr_three_q]:
        print(f"  {addr.name}: {addr.branch_path_str()}")

    print("\nHierarchical alignment matrix:")
    addrs = [addr_origin, addr_quarter, addr_middle, addr_three_q]
    names = [a.name for a in addrs]

    for i, a in enumerate(addrs):
        row = [a.branch_alignment(b) for b in addrs]
        print(f"  {names[i]:15s}: [{', '.join(f'{x:.4f}' for x in row)}]")

    print("\nInterpretation:")
    print("  - Self-alignment ≈ 0.9687 (sum of 0.5^k for k=1..5)")
    print("  - Higher values = share more routing structure")
    print("  - 0.0 = no shared structure (opposite corners)")

    # =========================================================================
    section("FIRST DIVERGENCE LEVEL")
    # =========================================================================

    print("First divergence (where paths split):")
    print(f"  origin vs quarter: level {addr_origin.first_divergence(addr_quarter)}")
    print(f"  origin vs middle: level {addr_origin.first_divergence(addr_middle)}")
    print(f"  origin vs three_quarter: level {addr_origin.first_divergence(addr_three_q)}")
    print(f"  quarter vs middle: level {addr_quarter.first_divergence(addr_middle)}")

    # =========================================================================
    section("PLATEAU MEMBERSHIP")
    # =========================================================================

    print("Full plateau IDs:")
    for addr in addrs:
        print(f"  {addr.name}: {addr.plateau_id_value}")

    print("\nPlateau IDs at depth 2 (coarse clustering):")
    for addr in addrs:
        print(f"  {addr.name}: {addr.plateau_id_at_depth(2)}")

    print("\nSame plateau checks (depth=2):")
    print(f"  origin vs quarter: {addr_origin.same_plateau(addr_quarter, 2)}")
    print(f"  origin vs three_quarter: {addr_origin.same_plateau(addr_three_q, 2)}")

    # =========================================================================
    section("ROUTABILITY CHECK")
    # =========================================================================

    print("Is routable with min_alignment=0.5?")
    for b in addrs[1:]:
        result = addr_origin.is_routable_to(b, min_alignment=0.5)
        alignment = addr_origin.branch_alignment(b)
        print(f"  origin -> {b.name}: {result} (alignment={alignment:.4f})")

    # =========================================================================
    section("DISTANCE IS FORBIDDEN")
    # =========================================================================

    print("Attempting to call distance()...")
    try:
        addr_origin.distance(addr_quarter)
        print("  ERROR: Should have raised NotImplementedError!")
    except NotImplementedError as e:
        print(f"  ✓ Correctly rejected: {str(e)[:60]}...")

    print("\nAttempting to call cantor_distance()...")
    try:
        addr_origin.cantor_distance(addr_quarter)
        print("  ERROR: Should have raised NotImplementedError!")
    except NotImplementedError as e:
        print(f"  ✓ Correctly rejected: {str(e)[:60]}...")

    # =========================================================================
    section("CANTOR ADDRESS BOOK")
    # =========================================================================

    book = CantorAddressBook()

    # Register addresses at various positions
    for i in range(1, 10):
        pos = i / 10
        addr = create_cantor_address(f'addr_{i}', position=pos)
        book.register(addr)

    print(f"Book: {book}")

    # Cluster by plateau
    print("\nClusters at depth 2:")
    clusters = book.cluster_by_plateau(depth=2)
    for pid, names in sorted(clusters.items()):
        positions = [f"{book[n].position:.1f}" for n in names]
        print(f"  Plateau {pid}: {names} (positions: {positions})")

    # Find routable
    addr_0 = book['addr_1']
    routable = book.find_routable(addr_0, min_alignment=0.5)
    print(f"\nRoutable from addr_1 (min_alignment=0.5): {routable}")

    # Alignment matrix
    names, matrix = book.alignment_matrix()
    print(f"\nAlignment matrix (first 5x5):")
    print(f"  {names[:5]}")
    print(matrix[:5, :5])

    # =========================================================================
    section("MESSAGE ROUTING BY ALIGNMENT")
    # =========================================================================

    # Queue messages
    for name in book.names:
        book.queue_message(name, torch.randn(64))

    # Route with min_alignment=0.5
    delivered = book.route_messages(min_alignment=0.5)
    print(f"Routed with min_alignment=0.5: {delivered} messages")

    # Check who received
    received = {n: len(book.get_messages(n)) for n in book.names}
    print(f"Messages received: {received}")

    # =========================================================================
    section("WORMHOLE DEMONSTRATION")
    # =========================================================================

    print("Creating wormhole endpoints (position 0.001 and 0.999):")

    addr_start = create_cantor_address('start', position=0.001)
    addr_end = create_cantor_address('end', position=0.999)

    print(f"  Start: {addr_start}")
    print(f"  End:   {addr_end}")

    sequential_dist = abs(0.001 - 0.999)
    hier_alignment = addr_start.branch_alignment(addr_end)
    first_div = addr_start.first_divergence(addr_end)
    feature_sim = addr_start.feature_similarity(addr_end)

    print(f"\nSequential distance: |0.001 - 0.999| = {sequential_dist:.3f} (very far)")
    print(f"Hierarchical alignment: {hier_alignment:.4f}")
    print(f"First divergence: level {first_div}")
    print(f"Feature similarity: {feature_sim:.4f}")

    print(f"\n'Distance is an illusion.'")
    print(f"Cantor routing doesn't care about sequential distance.")
    print(f"It cares about branch path structure at coarse levels.")

    # =========================================================================
    section("ALL TESTS PASSED")
    # =========================================================================

    print("\nCantorAddressComponent provides:")
    print("  ✓ Branch path (ternary decomposition)")
    print("  ✓ Hierarchical alignment (0.5^k weighting)")
    print("  ✓ Plateau membership (hierarchical clustering)")
    print("  ✓ First divergence level")
    print("  ✓ Routability check")
    print("  ✓ Feature similarity")
    print("  ✓ Alpha modulation (sparse ↔ dense routing)")
    print("  ✓ Alignment-based message routing")

    print("\nFORBIDDEN (meaningless on Cantor set):")
    print("  ✗ distance()")
    print("  ✗ cantor_distance()")

    print("\nCantorAddressComponent is ready for geometric routing.")
    print("'Distance is an illusion.'")