"""
global_fractal_router_v2.py - Global Fingerprint Routing with Provenance Tracking

Vectorized implementation with all hotspots addressed:
- local_mask: meshgrid vectorization
- mailbox_read: batched cosine similarity + caching
- prime_generation: precomputed lookup table
- potential_fields: single batched forward
- basis_construction: vectorized sin/cos

Author: AbstractPhil
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum
from weakref import WeakValueDictionary


# =============================================================================
# PRECOMPUTED PRIMES (eliminates runtime generation)
# =============================================================================

PRECOMPUTED_PRIMES: Tuple[int, ...] = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
    331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
    613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
    709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811,
    821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911,
    919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
    1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
    1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181,
    1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277,
    1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361,
    1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451,
    1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531,
    1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609,
    1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699,
    1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789,
    1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889,
    1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997,
    1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083,
    2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161,
    2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273,
    2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357,
    2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441,
    2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551,
    2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663,
    2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729,
    2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819,
    2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917,
    2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023,
    3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137,
    3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251,
    3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331,
    3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449,
    3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533,
    3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617,
    3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709,
    3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821,
    3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917,
    3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013,
    4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111,
    4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219,
    4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297,
    4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423,
    4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519,
    4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639,
    4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729,
    4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831,
    4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951,
    4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023,
    5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147,
    5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261,
    5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387,
    5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471,
    5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563,
    5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659,
    5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779,
    5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857,
    5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981,
    5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089,
    6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199,
    6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287,
    6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367,
    6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491,
    6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607,
    6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709,
    6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827,
    6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917,
    6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013,
    7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129,
    7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243,
    7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369,
    7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499,
    7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577,
    7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681,
    7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789,
    7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901,
    7907, 7919, 7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011, 8017,
    8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, 8117, 8123,
    8147, 8161,
)


def get_primes(n: int) -> List[int]:
    """Get first n primes from precomputed table or generate via sieve."""
    if n <= len(PRECOMPUTED_PRIMES):
        return list(PRECOMPUTED_PRIMES[:n])

    # Fallback: Sieve of Eratosthenes
    limit = max(n * 15, 1000)
    sieve = torch.ones(limit, dtype=torch.bool)
    sieve[0] = sieve[1] = False

    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False

    primes = torch.where(sieve)[0].tolist()
    return primes[:n]


# =============================================================================
# VECTORIZED UTILITIES
# =============================================================================

def build_local_mask(
    num_positions: int,
    grid_size: int,
    window_size: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Vectorized local window mask using meshgrid.
    Returns True where positions are OUTSIDE the window (to be masked).
    """
    device = device or torch.device('cpu')

    pos = torch.arange(num_positions, device=device)
    x = pos % grid_size
    y = pos // grid_size

    xi, xj = torch.meshgrid(x, x, indexing='ij')
    yi, yj = torch.meshgrid(y, y, indexing='ij')

    x_dist = (xi - xj).abs()
    y_dist = (yi - yj).abs()

    mask = (x_dist > window_size) | (y_dist > window_size)

    return mask


# =============================================================================
# CONFIGURATION
# =============================================================================

class CooperationMode(Enum):
    ISOLATED = "isolated"
    SIBLINGS = "siblings"
    HIERARCHICAL = "hierarchical"
    BROADCAST = "broadcast"


@dataclass
class GlobalFractalRouterConfig:
    """Configuration for the global fractal routing system."""

    fingerprint_dim: int = 64
    max_fingerprint_depth: int = 16

    feature_dim: int = 512
    num_anchors: int = 32
    num_routes: int = 8
    temperature: float = 0.1

    num_router_slots: int = 16
    router_comm_dim: int = 128

    use_adjacent_gating: bool = True
    gate_hidden_dim: int = 256
    num_potential_fields: int = 4

    use_cantor_prior: bool = True
    cantor_weight: float = 0.2
    grid_size: Optional[Tuple[int, int]] = None

    default_cooperation: CooperationMode = CooperationMode.HIERARCHICAL


# =============================================================================
# FINGERPRINT REGISTRY
# =============================================================================

class FingerprintRegistry:
    """Global registry ensuring unique fingerprints with vectorized operations."""

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
        self._hierarchy: Dict[str, str] = {}
        self._cooperation_groups: Dict[str, Set[str]] = {}
        self._modules: WeakValueDictionary = WeakValueDictionary()
        self._counter = 0

        self._basis = self._build_orthogonal_basis(64, 1024)

    def _build_orthogonal_basis(self, dim: int, count: int) -> torch.Tensor:
        """Vectorized basis construction - no Python loops."""
        primes = torch.tensor(get_primes(count), dtype=torch.float32)

        t = torch.linspace(0, 2 * math.pi, dim).unsqueeze(0)  # [1, dim]
        p = primes.unsqueeze(1)  # [count, 1]
        i = torch.arange(count, dtype=torch.float32).unsqueeze(1)  # [count, 1]

        basis = torch.sin(t * p) * torch.cos(i * 0.1)

        return F.normalize(basis, dim=-1)

    def register(
        self,
        module: nn.Module,
        name: Optional[str] = None,
        parent_id: Optional[str] = None,
        cooperation_group: Optional[str] = None,
    ) -> str:
        """Register a module and assign a unique fingerprint."""
        module_id = name or f"module_{self._counter}"
        self._counter += 1

        idx = self._counter % self._basis.shape[0]
        fingerprint = self._basis[idx].clone()

        if parent_id and parent_id in self._registry:
            parent_fp = self._registry[parent_id]
            offset = self._basis[(idx + 17) % self._basis.shape[0]]
            fingerprint = F.normalize(parent_fp * 0.7 + offset * 0.3, dim=-1)
            self._hierarchy[module_id] = parent_id

        self._registry[module_id] = fingerprint
        self._modules[module_id] = module

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
        return lineage[::-1]

    def compute_affinity_batched(self, query_id: str, candidate_ids: List[str]) -> torch.Tensor:
        """Batched affinity computation - single vectorized operation."""
        if not candidate_ids:
            return torch.tensor([])

        query_fp = self._registry.get(query_id)
        if query_fp is None:
            return torch.zeros(len(candidate_ids))

        candidate_fps = []
        valid_mask = []
        for cid in candidate_ids:
            fp = self._registry.get(cid)
            if fp is not None:
                candidate_fps.append(fp)
                valid_mask.append(True)
            else:
                candidate_fps.append(torch.zeros_like(query_fp))
                valid_mask.append(False)

        candidates = torch.stack(candidate_fps)
        valid_mask = torch.tensor(valid_mask)

        cosine_sims = F.cosine_similarity(
            query_fp.unsqueeze(0),
            candidates,
            dim=-1
        )

        cosine_sims = cosine_sims * valid_mask.float()

        return cosine_sims

    def compute_affinity(self, id_a: str, id_b: str) -> float:
        """Single affinity computation (for compatibility)."""
        return self.compute_affinity_batched(id_a, [id_b])[0].item()

    def get_cooperation_matrix(self, module_ids: List[str]) -> torch.Tensor:
        """Vectorized cooperation matrix."""
        n = len(module_ids)
        matrix = torch.zeros(n, n)

        for i, id_a in enumerate(module_ids):
            affinities = self.compute_affinity_batched(id_a, module_ids)
            matrix[i] = affinities

        return matrix

    def reset(self):
        """Reset registry."""
        self._registry.clear()
        self._hierarchy.clear()
        self._cooperation_groups.clear()
        self._modules.clear()
        self._counter = 0


def get_registry() -> FingerprintRegistry:
    return FingerprintRegistry()


# =============================================================================
# PROVENANCE TENSOR
# =============================================================================

class ProvenanceTensor:
    """Wrapper that carries fingerprint provenance with tensor data."""

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
            fp = F.normalize(fp + 0.1 * transform_fp, dim=-1)
        return fp

    @property
    def depth(self) -> int:
        return len(self.transformation_chain)

    def transform(self, transformer_fingerprint: torch.Tensor) -> 'ProvenanceTensor':
        """Create new ProvenanceTensor with added transformation."""
        return ProvenanceTensor(
            data=self.data,
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
    """Learned behavioral anchors for routing decisions."""

    def __init__(self, num_anchors: int, anchor_dim: int, fingerprint_dim: int):
        super().__init__()
        self.num_anchors = num_anchors
        self.anchor_dim = anchor_dim
        self.fingerprint_dim = fingerprint_dim

        self.anchor_embeddings = nn.Parameter(torch.randn(num_anchors, anchor_dim) * 0.02)

        fingerprints = self._build_anchor_fingerprints()
        self.register_buffer('anchor_fingerprints', fingerprints)

        self.feature_proj = nn.Linear(anchor_dim, anchor_dim)
        self.fingerprint_proj = nn.Linear(fingerprint_dim, anchor_dim)

    def _build_anchor_fingerprints(self) -> torch.Tensor:
        """Vectorized anchor fingerprint construction."""
        phi = (1 + math.sqrt(5)) / 2
        i = torch.arange(self.num_anchors, dtype=torch.float32)
        d = torch.arange(self.fingerprint_dim, dtype=torch.float32)

        theta = 2 * math.pi * i / phi
        fingerprints = torch.sin(theta.unsqueeze(1) * (d.unsqueeze(0) + 1) / self.fingerprint_dim)

        return F.normalize(fingerprints, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        query_fingerprint: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute anchor affinities for input features."""
        feat_proj = F.normalize(self.feature_proj(features), dim=-1)
        anchors_norm = F.normalize(self.anchor_embeddings, dim=-1)

        affinities = torch.matmul(feat_proj, anchors_norm.T)

        if query_fingerprint is not None:
            fp_proj = self.fingerprint_proj(query_fingerprint)
            fp_affinity = torch.matmul(anchors_norm, fp_proj)
            affinities = affinities + 0.3 * fp_affinity

        affinities = F.softmax(affinities, dim=-1)
        anchor_features = torch.matmul(affinities, self.anchor_embeddings)

        return affinities, anchor_features


# =============================================================================
# ADJACENT GATE (Batched)
# =============================================================================

class AdjacentGate(nn.Module):
    """
    Gates information flow based on adjacent potential fields.
    Uses single batched forward for all fields.
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

        # Single network outputs all fields at once
        self.field_net = nn.Sequential(
            nn.Linear(feature_dim + fingerprint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_fields),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(num_fields * 2 + fingerprint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

        self.compat_proj = nn.Linear(fingerprint_dim * 2, num_fields)

    def compute_potential(
        self,
        features: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        """Single forward for all potential fields."""
        B = features.shape[0]

        if fingerprint.dim() == 1:
            fingerprint = fingerprint.unsqueeze(0).expand(B, -1)

        combined = torch.cat([features, fingerprint], dim=-1)
        potentials = self.field_net(combined)

        return potentials

    def forward(
        self,
        source_features: torch.Tensor,
        source_fingerprint: torch.Tensor,
        target_fingerprint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gated features for flow from source to target."""
        B = source_features.shape[0]

        source_potential = self.compute_potential(source_features, source_fingerprint)
        target_potential = self.compute_potential(source_features, target_fingerprint)

        gradient = target_potential - source_potential

        fp_combined = torch.cat([
            source_fingerprint.unsqueeze(0).expand(B, -1),
            target_fingerprint.unsqueeze(0).expand(B, -1),
        ], dim=-1)
        compatibility = torch.tanh(self.compat_proj(fp_combined))

        gate_input = torch.cat([
            gradient,
            compatibility,
            source_fingerprint.unsqueeze(0).expand(B, -1),
        ], dim=-1)

        gate_values = self.gate_net(gate_input)
        gated_features = source_features * gate_values

        return gated_features, gate_values


# =============================================================================
# ROUTER COMMUNICATION
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
    """Communication hub with batched similarity computation."""

    def __init__(self, config: GlobalFractalRouterConfig):
        self.config = config
        self.messages: Dict[str, RouterMessage] = {}
        self.registry = get_registry()

        self._fingerprint_cache: Optional[torch.Tensor] = None
        self._id_cache: Optional[List[str]] = None
        self._cache_valid = False

    def post(self, message: RouterMessage):
        """Post a message from a router."""
        self.messages[message.sender_id] = message
        self._cache_valid = False

    def _rebuild_cache(self):
        """Build cached tensors for batched operations."""
        if not self.messages:
            self._fingerprint_cache = None
            self._id_cache = None
            return

        self._id_cache = list(self.messages.keys())
        fingerprints = [self.messages[sid].sender_fingerprint for sid in self._id_cache]
        self._fingerprint_cache = torch.stack(fingerprints)
        self._cache_valid = True

    def read(
        self,
        reader_id: str,
        reader_fingerprint: torch.Tensor,
        top_k: int = 4,
    ) -> List[RouterMessage]:
        """Batched message retrieval."""
        if not self.messages:
            return []

        if not self._cache_valid:
            self._rebuild_cache()

        if self._fingerprint_cache is None:
            return []

        # Batched cosine similarity
        fp_sims = F.cosine_similarity(
            reader_fingerprint.unsqueeze(0),
            self._fingerprint_cache,
            dim=-1
        )

        reg_affinities = self.registry.compute_affinity_batched(reader_id, self._id_cache)

        scores = 0.6 * reg_affinities + 0.4 * fp_sims

        # Mask self
        for i, sid in enumerate(self._id_cache):
            if sid == reader_id:
                scores[i] = -float('inf')

        k = min(top_k, len(self._id_cache))
        _, top_indices = torch.topk(scores, k)

        return [self.messages[self._id_cache[i]] for i in top_indices.tolist()]

    def clear(self):
        """Clear all messages."""
        self.messages.clear()
        self._cache_valid = False


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

        self.registry = get_registry()
        self.module_id = self.registry.register(
            module=self,
            name=name,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
        )

        fingerprint = self.registry.get_fingerprint(self.module_id)
        self.register_buffer('fingerprint', fingerprint)

        self.query_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.key_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.value_proj = nn.Linear(config.feature_dim, config.feature_dim)

        self.fingerprint_to_bias = nn.Linear(config.fingerprint_dim, config.feature_dim)

        self.anchor_bank = AnchorBank(
            num_anchors=config.num_anchors,
            anchor_dim=config.feature_dim,
            fingerprint_dim=config.fingerprint_dim,
        )

        if config.use_adjacent_gating:
            self.adjacent_gate = AdjacentGate(
                feature_dim=config.feature_dim,
                fingerprint_dim=config.fingerprint_dim,
                num_fields=config.num_potential_fields,
                hidden_dim=config.gate_hidden_dim,
            )
        else:
            self.adjacent_gate = None

        self.comm_encoder = nn.Linear(config.feature_dim, config.router_comm_dim)
        self.comm_decoder = nn.Linear(config.router_comm_dim, config.feature_dim)

        if config.use_cantor_prior:
            self.cantor_weight = config.cantor_weight
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
            G = grid_size[0]

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

        q = F.normalize(self.query_proj(x), dim=-1)
        k = F.normalize(self.key_proj(x), dim=-1)
        scores = torch.bmm(q, k.transpose(1, 2))

        fp = external_fingerprint if external_fingerprint is not None else self.fingerprint
        fp_bias = self.fingerprint_to_bias(fp)

        q_mod = q + 0.1 * fp_bias.unsqueeze(0).unsqueeze(0)
        scores = scores + 0.2 * torch.bmm(q_mod, k.transpose(1, 2))

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
                routed_features: [B, S, D] output features
            If return_provenance=True:
                output: ProvenanceTensor with updated chain
                routes, weights, metadata dict
        """
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
                states = torch.stack([m.routing_state for m in messages])
                router_context = self.comm_decoder(states.mean(dim=0))

        # Compute anchor affinities
        anchor_affinities, anchor_features = self.anchor_bank(
            data_route.mean(dim=1),
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
        topk_scores, routes = torch.topk(scores / self.config.temperature, K, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        # Gather values
        v = self.value_proj(data_route)
        v_gathered = self._gather(v, routes)

        # Weighted combination
        routed_features = torch.einsum('bpk,bpkd->bpd', weights, v_gathered)

        # Adjacent gating
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

        # Reconstruct full sequence
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
# ROUTER NETWORK
# =============================================================================

class FractalRouterNetwork(nn.Module):
    """Network of GlobalFractalRouters that coordinate via mailbox."""

    def __init__(
        self,
        config: GlobalFractalRouterConfig,
        num_routers: int,
        topology: str = "chain",
        cooperation_group: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.num_routers = num_routers
        self.topology = topology

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
        """Forward through router network."""
        self.mailbox.clear()

        outputs = []
        current = x

        for i, router in enumerate(self.routers):
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

        if self.topology == "chain":
            return outputs[-1][2]
        elif self.topology == "parallel":
            return torch.stack([o[2] for o in outputs]).mean(dim=0)
        elif self.topology == "tree":
            return outputs[-1][2]


# =============================================================================
# TEST
# =============================================================================

def test_global_fractal_router():
    """Quick functionality test."""
    print("=" * 60)
    print("Global Fractal Router V2 Test")
    print("=" * 60)

    get_registry().reset()

    config = GlobalFractalRouterConfig(
        fingerprint_dim=64,
        feature_dim=256,
        num_anchors=16,
        num_routes=8,
    )

    router = GlobalFractalRouter(config, name="test_router")
    x = torch.randn(2, 65, 256)

    routes, weights, features = router(x)
    print(f"Routes: {routes.shape}")
    print(f"Weights: {weights.shape}")
    print(f"Features: {features.shape}")
    print(f"Module ID: {router.module_id}")

    # Test network
    get_registry().reset()
    network = FractalRouterNetwork(config, num_routers=3, topology="chain")
    out = network(x)
    print(f"Network output: {out.shape}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_global_fractal_router()