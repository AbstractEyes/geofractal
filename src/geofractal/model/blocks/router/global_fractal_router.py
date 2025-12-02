"""
global_fractal_router.py - Global Fingerprint Routing with Provenance Tracking

Core Concepts:
1. Every module gets a unique fingerprint from a global registry
2. Information carries provenance (source fingerprint chain)
3. Routers communicate via shared anchor space
4. Adjacent gating controls information flow between potential fields

Author: AbstractPhil
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from weakref import WeakValueDictionary
import uuid


# =============================================================================
# CONFIGURATION
# =============================================================================

class CooperationMode(Enum):
    """How fingerprinted modules can cooperate."""
    ISOLATED = "isolated"  # No shared fingerprint space
    SIBLINGS = "siblings"  # Same parent, can share
    HIERARCHICAL = "hierarchical"  # Parent-child sharing
    BROADCAST = "broadcast"  # Visible to all


@dataclass
class GlobalFractalRouterConfig:
    """Configuration for the global fractal routing system."""

    # Fingerprint dimensions
    fingerprint_dim: int = 64
    max_fingerprint_depth: int = 16  # Max transformation chain length

    # Routing
    feature_dim: int = 512
    num_anchors: int = 32  # Behavioral anchor count
    num_routes: int = 8
    temperature: float = 0.1

    # Multi-router coordination
    num_router_slots: int = 16  # Max concurrent routers
    router_comm_dim: int = 128  # Inter-router message dim

    # Adjacent gating
    use_adjacent_gating: bool = True
    gate_hidden_dim: int = 256
    num_potential_fields: int = 4  # Parallel potential landscapes

    # Cantor geometric prior
    use_cantor_prior: bool = True
    cantor_weight: float = 0.2
    grid_size: Optional[Tuple[int, int]] = None

    # Cooperation
    default_cooperation: CooperationMode = CooperationMode.HIERARCHICAL


# =============================================================================
# FINGERPRINT REGISTRY (Global Singleton)
# =============================================================================

class FingerprintRegistry:
    """
    Global registry ensuring unique fingerprints across all modules.

    Fingerprints are hierarchical: parent.child.grandchild
    Modules can opt into cooperation groups for intentional sharing.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if FingerprintRegistry._initialized:
            return
        FingerprintRegistry._initialized = True

        self.fingerprint_dim = 64
        self._registry: Dict[str, torch.Tensor] = {}
        self._hierarchy: Dict[str, str] = {}  # child -> parent
        self._cooperation_groups: Dict[str, Set[str]] = {}
        self._modules: WeakValueDictionary = WeakValueDictionary()
        self._counter = 0

        # Basis vectors for fingerprint construction
        self._basis = self._build_orthogonal_basis(64, 1024)

    def _build_orthogonal_basis(self, dim: int, count: int) -> torch.Tensor:
        """Build quasi-orthogonal basis using prime-indexed frequencies."""
        primes = self._generate_primes(count)
        basis = torch.zeros(count, dim)

        for i, p in enumerate(primes):
            t = torch.linspace(0, 2 * math.pi * p, dim)
            basis[i] = torch.sin(t) * math.cos(i * 0.1)

        # Normalize
        basis = F.normalize(basis, dim=-1)
        return basis

    def _generate_primes(self, n: int) -> List[int]:
        """Generate first n primes."""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = all(candidate % p != 0 for p in primes if p * p <= candidate)
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    def register(
            self,
            module: nn.Module,
            name: Optional[str] = None,
            parent_id: Optional[str] = None,
            cooperation_group: Optional[str] = None,
    ) -> str:
        """
        Register a module and assign a unique fingerprint.

        Returns:
            module_id: Unique identifier for this module
        """
        module_id = name or f"module_{self._counter}"
        self._counter += 1

        # Generate unique fingerprint
        idx = self._counter % self._basis.shape[0]
        fingerprint = self._basis[idx].clone()

        # Add hierarchical component if parent exists
        if parent_id and parent_id in self._registry:
            parent_fp = self._registry[parent_id]
            # Child fingerprint = parent + orthogonal offset
            offset = self._basis[(idx + 17) % self._basis.shape[0]]  # Prime offset
            fingerprint = F.normalize(parent_fp * 0.7 + offset * 0.3, dim=-1)
            self._hierarchy[module_id] = parent_id

        self._registry[module_id] = fingerprint
        self._modules[module_id] = module

        # Cooperation group
        if cooperation_group:
            if cooperation_group not in self._cooperation_groups:
                self._cooperation_groups[cooperation_group] = set()
            self._cooperation_groups[cooperation_group].add(module_id)

        return module_id

    def get_fingerprint(self, module_id: str) -> torch.Tensor:
        """Get fingerprint for a registered module."""
        if module_id not in self._registry:
            raise KeyError(f"Module {module_id} not registered")
        return self._registry[module_id]

    def get_lineage(self, module_id: str) -> List[str]:
        """Get full parent chain for a module."""
        lineage = [module_id]
        current = module_id
        while current in self._hierarchy:
            current = self._hierarchy[current]
            lineage.append(current)
        return lineage[::-1]  # Root first

    def compute_affinity(self, id_a: str, id_b: str) -> float:
        """Compute cooperation affinity between two modules."""
        fp_a = self._registry.get(id_a)
        fp_b = self._registry.get(id_b)

        if fp_a is None or fp_b is None:
            return 0.0

        # Check cooperation groups
        for group, members in self._cooperation_groups.items():
            if id_a in members and id_b in members:
                return 1.0  # Same group = full cooperation

        # Hierarchical affinity
        lineage_a = set(self.get_lineage(id_a))
        lineage_b = set(self.get_lineage(id_b))
        shared = lineage_a & lineage_b

        if shared:
            # Closer in hierarchy = higher affinity
            depth_bonus = len(shared) / max(len(lineage_a), len(lineage_b))
        else:
            depth_bonus = 0.0

        # Cosine similarity of fingerprints
        cosine = F.cosine_similarity(fp_a.unsqueeze(0), fp_b.unsqueeze(0)).item()

        return 0.5 * cosine + 0.5 * depth_bonus

    def get_cooperation_matrix(self, module_ids: List[str]) -> torch.Tensor:
        """Get pairwise cooperation affinity matrix."""
        n = len(module_ids)
        matrix = torch.zeros(n, n)

        for i, id_a in enumerate(module_ids):
            for j, id_b in enumerate(module_ids):
                matrix[i, j] = self.compute_affinity(id_a, id_b)

        return matrix

    def reset(self):
        """Reset registry (for testing)."""
        self._registry.clear()
        self._hierarchy.clear()
        self._cooperation_groups.clear()
        self._modules.clear()
        self._counter = 0


# Global accessor
def get_registry() -> FingerprintRegistry:
    return FingerprintRegistry()


# =============================================================================
# PROVENANCE TENSOR
# =============================================================================

class ProvenanceTensor:
    """
    Wrapper that carries fingerprint provenance with tensor data.

    Tracks the chain of transformations applied to data.
    """

    def __init__(
            self,
            data: torch.Tensor,
            source_fingerprint: torch.Tensor,
            transformation_chain: Optional[List[torch.Tensor]] = None,
    ):
        self.data = data
        self.source_fingerprint = source_fingerprint
        self.transformation_chain = transformation_chain or []

    @property
    def current_fingerprint(self) -> torch.Tensor:
        """Compute current fingerprint from source + transformations."""
        fp = self.source_fingerprint
        for transform_fp in self.transformation_chain:
            # Combine via rotation in fingerprint space
            fp = F.normalize(fp + 0.1 * transform_fp, dim=-1)
        return fp

    @property
    def depth(self) -> int:
        return len(self.transformation_chain)

    def transform(self, transformer_fingerprint: torch.Tensor) -> 'ProvenanceTensor':
        """Create new ProvenanceTensor with added transformation."""
        return ProvenanceTensor(
            data=self.data,  # Data updated externally
            source_fingerprint=self.source_fingerprint,
            transformation_chain=self.transformation_chain + [transformer_fingerprint],
        )

    def with_data(self, new_data: torch.Tensor) -> 'ProvenanceTensor':
        """Create copy with new data, same provenance."""
        return ProvenanceTensor(
            data=new_data,
            source_fingerprint=self.source_fingerprint,
            transformation_chain=self.transformation_chain.copy(),
        )


# =============================================================================
# ANCHOR BANK
# =============================================================================

class AnchorBank(nn.Module):
    """
    Learned behavioral anchors for routing decisions.

    Anchors represent stable attractor states that guide information flow.
    Each anchor has a fingerprint and learned embedding.
    """

    def __init__(
            self,
            num_anchors: int,
            anchor_dim: int,
            fingerprint_dim: int,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.anchor_dim = anchor_dim
        self.fingerprint_dim = fingerprint_dim

        # Learned anchor embeddings
        self.anchor_embeddings = nn.Parameter(
            torch.randn(num_anchors, anchor_dim) * 0.02
        )

        # Anchor fingerprints (fixed, based on golden ratio spiral)
        fingerprints = self._build_anchor_fingerprints()
        self.register_buffer('anchor_fingerprints', fingerprints)

        # Projection for comparing features to anchors
        self.feature_proj = nn.Linear(anchor_dim, anchor_dim)
        self.fingerprint_proj = nn.Linear(fingerprint_dim, anchor_dim)

    def _build_anchor_fingerprints(self) -> torch.Tensor:
        """Build anchor fingerprints using golden ratio spacing."""
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio

        fingerprints = torch.zeros(self.num_anchors, self.fingerprint_dim)
        for i in range(self.num_anchors):
            theta = 2 * math.pi * i / phi
            for d in range(self.fingerprint_dim):
                fingerprints[i, d] = math.sin(theta * (d + 1) / self.fingerprint_dim)

        return F.normalize(fingerprints, dim=-1)

    def forward(
            self,
            features: torch.Tensor,
            query_fingerprint: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anchor affinities for input features.

        Args:
            features: [B, ..., D] input features
            query_fingerprint: [fingerprint_dim] optional fingerprint for biasing

        Returns:
            affinities: [B, ..., num_anchors] anchor weights
            anchor_features: [B, ..., D] anchor-weighted representation
        """
        # Project features
        feat_proj = F.normalize(self.feature_proj(features), dim=-1)
        anchors_norm = F.normalize(self.anchor_embeddings, dim=-1)

        # Content affinity
        affinities = torch.matmul(feat_proj, anchors_norm.T)  # [B, ..., A]

        # Fingerprint biasing
        if query_fingerprint is not None:
            fp_proj = self.fingerprint_proj(query_fingerprint)  # [D]
            fp_affinity = torch.matmul(anchors_norm, fp_proj)  # [A]
            affinities = affinities + 0.3 * fp_affinity

        affinities = F.softmax(affinities, dim=-1)

        # Anchor-weighted features
        anchor_features = torch.matmul(affinities, self.anchor_embeddings)

        return affinities, anchor_features


# =============================================================================
# ADJACENT GATE
# =============================================================================

class AdjacentGate(nn.Module):
    """
    Gates information flow based on adjacent potential fields.

    Models information as flowing through a landscape where:
    - Potentials attract/repel based on fingerprint compatibility
    - Gates open when potential gradient favors flow
    """

    def __init__(
            self,
            feature_dim: int,
            fingerprint_dim: int,
            num_fields: int = 4,
            hidden_dim: int = 256,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.fingerprint_dim = fingerprint_dim
        self.num_fields = num_fields

        # Potential field generators
        self.field_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim + fingerprint_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_fields)
        ])

        # Gate computer
        self.gate_net = nn.Sequential(
            nn.Linear(num_fields * 2 + fingerprint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

        # Fingerprint compatibility
        self.compat_proj = nn.Linear(fingerprint_dim * 2, num_fields)

    def compute_potential(
            self,
            features: torch.Tensor,
            fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        """Compute potential values across all fields."""
        B = features.shape[0]

        # Expand fingerprint
        if fingerprint.dim() == 1:
            fingerprint = fingerprint.unsqueeze(0).expand(B, -1)

        combined = torch.cat([features, fingerprint], dim=-1)

        potentials = []
        for gen in self.field_generators:
            potentials.append(gen(combined))

        return torch.cat(potentials, dim=-1)  # [B, num_fields]

    def forward(
            self,
            source_features: torch.Tensor,
            source_fingerprint: torch.Tensor,
            target_fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gated features for flow from source to target.

        Args:
            source_features: [B, D] features to gate
            source_fingerprint: [fp_dim] source module fingerprint
            target_fingerprint: [fp_dim] target module fingerprint

        Returns:
            gated_features: [B, D] gated output
            gate_values: [B, D] gate activations (for analysis)
        """
        B = source_features.shape[0]

        # Compute potentials at source and target
        source_potential = self.compute_potential(source_features, source_fingerprint)

        # For target, use projected features as proxy
        target_proxy = source_features  # Could be more sophisticated
        target_potential = self.compute_potential(target_proxy, target_fingerprint)

        # Potential gradient (target - source = flow direction)
        gradient = target_potential - source_potential  # [B, num_fields]

        # Fingerprint compatibility
        fp_combined = torch.cat([
            source_fingerprint.unsqueeze(0).expand(B, -1),
            target_fingerprint.unsqueeze(0).expand(B, -1),
        ], dim=-1)
        compatibility = torch.tanh(self.compat_proj(fp_combined))  # [B, num_fields]

        # Gate input
        gate_input = torch.cat([
            gradient,
            compatibility,
            source_fingerprint.unsqueeze(0).expand(B, -1),
        ], dim=-1)

        gate_values = self.gate_net(gate_input)  # [B, D]
        gated_features = source_features * gate_values

        return gated_features, gate_values


# =============================================================================
# ROUTER COMMUNICATION PROTOCOL
# =============================================================================

class RouterMessage:
    """Message passed between routers for coordination."""

    def __init__(
            self,
            sender_id: str,
            sender_fingerprint: torch.Tensor,
            routing_state: torch.Tensor,
            anchor_affinities: Optional[torch.Tensor] = None,
            metadata: Optional[Dict] = None,
    ):
        self.sender_id = sender_id
        self.sender_fingerprint = sender_fingerprint
        self.routing_state = routing_state
        self.anchor_affinities = anchor_affinities
        self.metadata = metadata or {}


class RouterMailbox:
    """
    Communication hub for multi-router coordination.

    Routers post messages and read from neighbors based on fingerprint affinity.
    """

    def __init__(self, config: GlobalFractalRouterConfig):
        self.config = config
        self.messages: Dict[str, RouterMessage] = {}
        self.registry = get_registry()

    def post(self, message: RouterMessage):
        """Post a message from a router."""
        self.messages[message.sender_id] = message

    def read(
            self,
            reader_id: str,
            reader_fingerprint: torch.Tensor,
            top_k: int = 4,
    ) -> List[RouterMessage]:
        """Read most relevant messages for a router."""
        if not self.messages:
            return []

        # Score messages by fingerprint affinity
        scored = []
        for sender_id, msg in self.messages.items():
            if sender_id == reader_id:
                continue

            affinity = self.registry.compute_affinity(reader_id, sender_id)
            fp_sim = F.cosine_similarity(
                reader_fingerprint.unsqueeze(0),
                msg.sender_fingerprint.unsqueeze(0)
            ).item()

            score = 0.6 * affinity + 0.4 * fp_sim
            scored.append((score, msg))

        # Return top-k
        scored.sort(key=lambda x: -x[0])
        return [msg for _, msg in scored[:top_k]]

    def clear(self):
        """Clear all messages (call between forward passes)."""
        self.messages.clear()


# =============================================================================
# GLOBAL FRACTAL ROUTER
# =============================================================================

class GlobalFractalRouter(nn.Module):
    """
    Fractal router with global fingerprint coordination.

    Features:
    - Unique fingerprint from global registry
    - Provenance tracking through transformations
    - Multi-router communication via mailbox
    - Adjacent gating based on potential fields
    - Behavioral routing via anchor bank
    """

    def __init__(
            self,
            config: GlobalFractalRouterConfig,
            parent_id: Optional[str] = None,
            cooperation_group: Optional[str] = None,
            name: Optional[str] = None,
    ):
        super().__init__()
        self.config = config

        # Register with global fingerprint registry
        self.registry = get_registry()
        self.module_id = self.registry.register(
            module=self,
            name=name,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
        )

        # Store own fingerprint
        fingerprint = self.registry.get_fingerprint(self.module_id)
        self.register_buffer('fingerprint', fingerprint)

        # Core routing components
        self.query_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.key_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.value_proj = nn.Linear(config.feature_dim, config.feature_dim)

        # Fingerprint integration
        self.fingerprint_to_bias = nn.Linear(
            config.fingerprint_dim,
            config.feature_dim
        )

        # Anchor bank for behavioral routing
        self.anchor_bank = AnchorBank(
            num_anchors=config.num_anchors,
            anchor_dim=config.feature_dim,
            fingerprint_dim=config.fingerprint_dim,
        )

        # Adjacent gating
        if config.use_adjacent_gating:
            self.adjacent_gate = AdjacentGate(
                feature_dim=config.feature_dim,
                fingerprint_dim=config.fingerprint_dim,
                num_fields=config.num_potential_fields,
                hidden_dim=config.gate_hidden_dim,
            )
        else:
            self.adjacent_gate = None

        # Router communication
        self.comm_encoder = nn.Linear(config.feature_dim, config.router_comm_dim)
        self.comm_decoder = nn.Linear(config.router_comm_dim, config.feature_dim)

        # Cantor prior (optional)
        if config.use_cantor_prior:
            self.cantor_weight = config.cantor_weight
            # Built lazily based on input size
            self._cantor_bias = None
            self._cantor_size = None
        else:
            self.cantor_weight = 0.0

    def _ensure_cantor_bias(self, num_positions: int, device: torch.device):
        """Lazily build Cantor bias for given size."""
        if self._cantor_bias is not None and self._cantor_size == num_positions:
            return

        grid_size = self.config.grid_size
        if grid_size is None:
            G = int(math.sqrt(num_positions))
        else:
            G = grid_size[0]  # Use height

        P = num_positions
        x = torch.arange(P, device=device) % G
        y = torch.arange(P, device=device) // G
        z = ((x + y) * (x + y + 1)) // 2 + y
        z = z.float() / z.max().clamp(min=1)

        dist = (z.unsqueeze(0) - z.unsqueeze(1)).abs()
        affinity = 1.0 - dist
        affinity.fill_diagonal_(-1e9)

        self._cantor_bias = affinity
        self._cantor_size = num_positions

    def _compute_routing_scores(
            self,
            x: torch.Tensor,
            external_fingerprint: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute routing scores with fingerprint modulation."""
        B, P, D = x.shape

        # Content-based scores
        q = F.normalize(self.query_proj(x), dim=-1)
        k = F.normalize(self.key_proj(x), dim=-1)
        scores = torch.bmm(q, k.transpose(1, 2))  # [B, P, P]

        # Fingerprint bias
        fp = external_fingerprint if external_fingerprint is not None else self.fingerprint
        fp_bias = self.fingerprint_to_bias(fp)  # [D]

        # Apply as query-key modulation
        q_mod = q + 0.1 * fp_bias.unsqueeze(0).unsqueeze(0)
        scores = scores + 0.2 * torch.bmm(q_mod, k.transpose(1, 2))

        # Cantor prior
        if self.cantor_weight > 0:
            self._ensure_cantor_bias(P, x.device)
            scores = scores + self.cantor_weight * self._cantor_bias[:P, :P].unsqueeze(0)

        return scores

    def forward(
            self,
            x: Union[torch.Tensor, ProvenanceTensor],
            mailbox: Optional[RouterMailbox] = None,
            target_fingerprint: Optional[torch.Tensor] = None,
            skip_first: bool = True,
            return_provenance: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[ProvenanceTensor, torch.Tensor, torch.Tensor, Dict],
    ]:
        """
        Forward pass with full fingerprint tracking.

        Args:
            x: Input features [B, S, D] or ProvenanceTensor
            mailbox: Optional router mailbox for multi-router communication
            target_fingerprint: Fingerprint of next module (for gating)
            skip_first: Skip first token (CLS)
            return_provenance: Return full provenance tracking info

        Returns:
            If return_provenance=False:
                routes: [B, P, K] destination indices
                weights: [B, P, K] routing weights
                routed_features: [B, P, D] gathered and gated features
            If return_provenance=True:
                output: ProvenanceTensor with updated chain
                routes, weights, routed_features
                metadata: Dict with routing analysis
        """
        # Handle ProvenanceTensor input
        if isinstance(x, ProvenanceTensor):
            data = x.data
            source_fp = x.current_fingerprint
            provenance = x
        else:
            data = x
            source_fp = self.fingerprint
            provenance = ProvenanceTensor(data, self.fingerprint)

        if skip_first:
            data_route = data[:, 1:, :]
        else:
            data_route = data

        B, P, D = data_route.shape
        K = self.config.num_routes

        # Read messages from other routers
        router_context = None
        if mailbox is not None:
            messages = mailbox.read(self.module_id, self.fingerprint)
            if messages:
                # Aggregate routing states from neighbors
                states = torch.stack([m.routing_state for m in messages])
                router_context = self.comm_decoder(states.mean(dim=0))

        # Compute anchor affinities
        anchor_affinities, anchor_features = self.anchor_bank(
            data_route.mean(dim=1),  # [B, D]
            query_fingerprint=source_fp,
        )

        # Compute routing scores
        scores = self._compute_routing_scores(data_route, source_fp)

        # Integrate router context
        if router_context is not None:
            context_bias = torch.einsum('d,bpd->bp', router_context, data_route)
            scores = scores + 0.1 * context_bias.unsqueeze(-1)

        # Mask self-connections
        mask = torch.eye(P, device=data.device, dtype=torch.bool)
        scores = scores.masked_fill(mask.unsqueeze(0), -1e9)

        # Top-K selection
        topk_scores, routes = torch.topk(
            scores / self.config.temperature,
            K,
            dim=-1
        )
        weights = F.softmax(topk_scores, dim=-1)  # [B, P, K]

        # Gather values
        v = self.value_proj(data_route)
        v_gathered = self._gather(v, routes)  # [B, P, K, D]

        # Weighted combination
        routed_features = torch.einsum('bpk,bpkd->bpd', weights, v_gathered)

        # Adjacent gating (if target specified)
        gate_values = None
        if self.adjacent_gate is not None and target_fingerprint is not None:
            routed_flat = routed_features.reshape(B * P, D)
            gated, gate_values = self.adjacent_gate(
                routed_flat,
                self.fingerprint,
                target_fingerprint,
            )
            routed_features = gated.reshape(B, P, D)

        # Post message to mailbox
        if mailbox is not None:
            routing_state = self.comm_encoder(routed_features.mean(dim=(0, 1)))
            mailbox.post(RouterMessage(
                sender_id=self.module_id,
                sender_fingerprint=self.fingerprint,
                routing_state=routing_state,
                anchor_affinities=anchor_affinities.mean(dim=0),
            ))

        # Reconstruct full sequence if needed
        if skip_first:
            routed_features = torch.cat([data[:, :1, :], routed_features], dim=1)

        if return_provenance:
            output_provenance = provenance.transform(self.fingerprint).with_data(routed_features)
            metadata = {
                'scores': scores,
                'anchor_affinities': anchor_affinities,
                'anchor_features': anchor_features,
                'gate_values': gate_values,
                'source_fingerprint': source_fp,
                'router_fingerprint': self.fingerprint,
            }
            return output_provenance, routes, weights, metadata

        return routes, weights, routed_features

    @staticmethod
    def _gather(x: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        """[B, P, D] + [B, P, K] → [B, P, K, D]"""
        B, P, D = x.shape
        K = routes.shape[-1]
        routes_flat = routes.reshape(B, P * K).unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(x, 1, routes_flat).view(B, P, K, D)

    def get_lineage(self) -> List[str]:
        """Get this router's lineage in the global hierarchy."""
        return self.registry.get_lineage(self.module_id)


# =============================================================================
# ROUTER NETWORK (Multiple coordinated routers)
# =============================================================================

class FractalRouterNetwork(nn.Module):
    """
    Network of GlobalFractalRouters that coordinate via mailbox.

    Use this to create hierarchical or parallel routing topologies.
    """

    def __init__(
            self,
            config: GlobalFractalRouterConfig,
            num_routers: int,
            topology: str = "chain",  # "chain", "parallel", "tree"
            cooperation_group: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.num_routers = num_routers
        self.topology = topology

        # Create routers with appropriate hierarchy
        self.routers = nn.ModuleList()

        group = cooperation_group or f"network_{id(self)}"

        if topology == "chain":
            parent = None
            for i in range(num_routers):
                router = GlobalFractalRouter(
                    config=config,
                    parent_id=parent,
                    cooperation_group=group,
                    name=f"router_{i}",
                )
                self.routers.append(router)
                parent = router.module_id

        elif topology == "parallel":
            for i in range(num_routers):
                router = GlobalFractalRouter(
                    config=config,
                    parent_id=None,
                    cooperation_group=group,
                    name=f"router_{i}",
                )
                self.routers.append(router)

        elif topology == "tree":
            # Binary tree structure
            parents = [None]
            router_idx = 0
            while router_idx < num_routers:
                parent = parents[router_idx // 2] if router_idx > 0 else None
                router = GlobalFractalRouter(
                    config=config,
                    parent_id=parent,
                    cooperation_group=group,
                    name=f"router_{router_idx}",
                )
                self.routers.append(router)
                parents.append(router.module_id)
                router_idx += 1

        self.mailbox = RouterMailbox(config)

    def forward(
            self,
            x: torch.Tensor,
            return_all: bool = False,
    ) -> Union[torch.Tensor, List[Tuple]]:
        """
        Forward through router network.

        Args:
            x: [B, S, D] input features
            return_all: Return outputs from all routers

        Returns:
            If return_all=False: Final routed features
            If return_all=True: List of (routes, weights, features) per router
        """
        self.mailbox.clear()

        outputs = []
        current = x

        for i, router in enumerate(self.routers):
            # Determine target fingerprint
            if i < len(self.routers) - 1:
                target_fp = self.routers[i + 1].fingerprint
            else:
                target_fp = None

            routes, weights, features = router(
                current,
                mailbox=self.mailbox,
                target_fingerprint=target_fp,
            )

            outputs.append((routes, weights, features))

            if self.topology == "chain":
                current = features

        if return_all:
            return outputs

        # Return based on topology
        if self.topology == "chain":
            return outputs[-1][2]  # Last router's features
        elif self.topology == "parallel":
            # Average all router outputs
            return torch.stack([o[2] for o in outputs]).mean(dim=0)
        elif self.topology == "tree":
            # Return root output (could be more sophisticated)
            return outputs[-1][2]


# =============================================================================
# TESTS
# =============================================================================

def test_global_fractal_router():
    print("=" * 70)
    print("Global Fractal Router Test Suite")
    print("=" * 70)

    # Reset registry
    get_registry().reset()

    config = GlobalFractalRouterConfig(
        fingerprint_dim=64,
        feature_dim=256,
        num_anchors=16,
        num_routes=8,
    )

    # Test 1: Single router
    print("\n[1] Single Router")
    router = GlobalFractalRouter(config, name="test_router")
    x = torch.randn(2, 65, 256)

    routes, weights, features = router(x)
    print(f"    Routes: {routes.shape}, Weights: {weights.shape}, Features: {features.shape}")
    print(f"    Module ID: {router.module_id}")
    print(f"    Fingerprint norm: {router.fingerprint.norm().item():.4f}")

    # Test 2: Provenance tracking
    print("\n[2] Provenance Tracking")
    prov_tensor = ProvenanceTensor(x, router.fingerprint)
    output, routes, weights, meta = router(prov_tensor, return_provenance=True)
    print(f"    Input depth: {prov_tensor.depth}")
    print(f"    Output depth: {output.depth}")
    print(f"    Lineage: {router.get_lineage()}")

    # Test 3: Router network (chain)
    print("\n[3] Router Network (Chain Topology)")
    get_registry().reset()
    network = FractalRouterNetwork(config, num_routers=3, topology="chain")

    outputs = network(x, return_all=True)
    print(f"    Routers: {len(network.routers)}")
    for i, (r, w, f) in enumerate(outputs):
        print(f"    Router {i}: routes={r.shape}, features={f.shape}")

    # Test 4: Cooperation matrix
    print("\n[4] Cooperation Matrix")
    registry = get_registry()
    router_ids = [r.module_id for r in network.routers]
    coop_matrix = registry.get_cooperation_matrix(router_ids)
    print(f"    Matrix shape: {coop_matrix.shape}")
    print(f"    Diagonal (self): {coop_matrix.diag().tolist()}")
    print(f"    Adjacent affinity: {coop_matrix[0, 1].item():.4f}")

    # Test 5: Multi-router mailbox communication
    print("\n[5] Mailbox Communication")
    mailbox = RouterMailbox(config)

    # Create parallel routers
    get_registry().reset()
    routers = [
        GlobalFractalRouter(config, name=f"parallel_{i}", cooperation_group="parallel_group")
        for i in range(4)
    ]

    # Each router processes and posts
    for router in routers:
        router(x, mailbox=mailbox)

    print(f"    Messages posted: {len(mailbox.messages)}")

    # Read messages
    messages = mailbox.read(routers[0].module_id, routers[0].fingerprint, top_k=2)
    print(f"    Messages read by router_0: {len(messages)}")
    for msg in messages:
        print(f"      From: {msg.sender_id}, state_norm: {msg.routing_state.norm().item():.4f}")

    # Test 6: Adjacent gating
    print("\n[6] Adjacent Gating")
    get_registry().reset()
    router_a = GlobalFractalRouter(config, name="gated_a")
    router_b = GlobalFractalRouter(config, name="gated_b", parent_id="gated_a")

    # Route with target fingerprint
    routes, weights, gated_features = router_a(
        x,
        target_fingerprint=router_b.fingerprint,
    )
    print(f"    Gated features: {gated_features.shape}")
    print(f"    Parent-child affinity: {registry.compute_affinity('gated_a', 'gated_b'):.4f}")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_global_fractal_router()