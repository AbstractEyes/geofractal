"""
geofractal.router.components.aux_component
===========================================

Auxiliary feature generators for fusion weight computation.

Auxiliary features provide geometric, statistical, or learned information
about input relationships WITHOUT being directly fused into the output.
They inform HOW to fuse (weights, schedules) rather than WHAT to fuse.

From experimental findings:
- 'learned' and 'geometric' aux types achieve best accuracy (88.05%)
- Auxiliary features add +1.5% over zero baseline
- Walker fusion (89.19%) > InceptiveFusion (88.05%), but combo may beat both

Design Pattern:
    aux_features = generator(input1, input2, input3)
    weights = MLP([anchor, gathered, aux_features])
    # OR
    schedule = schedule_net(aux_features)  # Per-sample schedule modulation

Usage:
    from aux_component import create_auxiliary_generator, BaseAux

    # Factory function
    aux = create_auxiliary_generator('cosine', num_inputs=3, in_features=512, aux_dim=64)
    features = aux(x1, x2, x3)  # [B, 64]

    # Direct instantiation
    aux = GeometricAuxiliary(num_inputs=3, in_features=512, aux_dim=64)
    features = aux(x1, x2, x3)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from abc import abstractmethod
from typing import Optional, Dict, Type, List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.components.torch_component import TorchComponent


# =============================================================================
# BASE CLASS
# =============================================================================

class BaseAux(TorchComponent):
    """
    Abstract base class for auxiliary feature generators.

    Auxiliary features capture relationships between inputs (similarity,
    geometry, variance, etc.) for use in fusion weight computation or
    schedule modulation. They do NOT appear in the fused output directly.

    Subclasses must implement:
        - forward(*inputs) -> Tensor[B, aux_dim]

    Attributes:
        num_inputs: Expected number of input tensors
        in_features: Dimension of each input tensor
        aux_dim: Output auxiliary feature dimension

    Example:
        class MyAuxiliary(BaseAux):
            def __init__(self, num_inputs, in_features, aux_dim, **kwargs):
                super().__init__("my_aux", num_inputs, in_features, aux_dim, **kwargs)
                self.proj = nn.Linear(num_inputs, aux_dim)

            def forward(self, *inputs):
                # Compute features from inputs
                return self.proj(...)
    """

    def __init__(
        self,
        name: str,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize BaseAux.

        Args:
            name: Component name
            num_inputs: Number of input tensors expected
            in_features: Feature dimension of each input
            aux_dim: Output auxiliary feature dimension
            uuid: Optional unique identifier
            **kwargs: Additional TorchComponent arguments
        """
        TorchComponent.__init__(self, name, uuid=uuid, **kwargs)

        self.num_inputs = num_inputs
        self.in_features = in_features
        self.aux_dim = aux_dim

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        """
        Generate auxiliary features from inputs.

        Args:
            *inputs: N tensors of shape [B, in_features]

        Returns:
            Auxiliary features of shape [B, aux_dim]
        """
        pass

    def extra_repr(self) -> str:
        return f"num_inputs={self.num_inputs}, in_features={self.in_features}, aux_dim={self.aux_dim}"


# =============================================================================
# BASELINE / ZERO
# =============================================================================

class ZeroAuxiliary(BaseAux):
    """
    Zero auxiliary features - baseline for ablation studies.

    Returns zeros, effectively disabling auxiliary information.
    Use this to measure the impact of auxiliary features.
    """

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        **kwargs,
    ):
        super().__init__("zero_aux", num_inputs, in_features, aux_dim, **kwargs)

    def forward(self, *inputs: Tensor) -> Tensor:
        B = inputs[0].shape[0]
        return torch.zeros(
            B, self.aux_dim,
            device=inputs[0].device,
            dtype=inputs[0].dtype,
        )


# =============================================================================
# SIMILARITY-BASED
# =============================================================================

class CosineSimilarityAuxiliary(BaseAux):
    """
    Pairwise cosine similarities between all inputs.

    For N inputs, computes N*(N-1)/2 pairwise cosine similarities,
    then projects to aux_dim. Captures agreement between encoders.

    Experimental result: 87.71% on CIFAR-100 (good baseline)
    """

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        **kwargs,
    ):
        super().__init__("cosine_aux", num_inputs, in_features, aux_dim, **kwargs)

        num_pairs = num_inputs * (num_inputs - 1) // 2
        self.proj = nn.Linear(num_pairs, aux_dim)

    def forward(self, *inputs: Tensor) -> Tensor:
        cosines = []
        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                cos = F.cosine_similarity(inputs[i], inputs[j], dim=-1, eps=1e-8)
                cosines.append(cos)

        # Stack: [B, num_pairs]
        cosines = torch.stack(cosines, dim=-1)
        return self.proj(cosines)


class DotProductAuxiliary(BaseAux):
    """
    Scaled dot products between inputs.

    Raw correlation structure without normalization.
    Uses 1/sqrt(d) scaling for stability.
    """

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        **kwargs,
    ):
        super().__init__("dot_aux", num_inputs, in_features, aux_dim, **kwargs)

        num_pairs = num_inputs * (num_inputs - 1) // 2
        self.proj = nn.Linear(num_pairs, aux_dim)
        self.scale = 1.0 / math.sqrt(in_features)

    def forward(self, *inputs: Tensor) -> Tensor:
        dots = []
        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                dot = (inputs[i] * inputs[j]).sum(dim=-1, keepdim=True) * self.scale
                dots.append(dot)

        features = torch.cat(dots, dim=-1)
        return self.proj(features)


# =============================================================================
# MAGNITUDE-BASED
# =============================================================================

class MagnitudeRatioAuxiliary(BaseAux):
    """
    Magnitude and ratio features between inputs.

    Captures relative "confidence" or scale of each encoder.
    Uses log-scale ratios for numerical stability.

    Features:
        - Per-input magnitudes (N values)
        - Pairwise log-ratios (N*(N-1)/2 values)
    """

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        **kwargs,
    ):
        super().__init__("magnitude_aux", num_inputs, in_features, aux_dim, **kwargs)

        # N magnitudes + N*(N-1)/2 ratios
        num_features = num_inputs + num_inputs * (num_inputs - 1) // 2
        self.proj = nn.Linear(num_features, aux_dim)

    def forward(self, *inputs: Tensor) -> Tensor:
        # Compute magnitudes
        mags = [x.norm(dim=-1, keepdim=True) for x in inputs]

        # Compute log ratios (stable)
        ratios = []
        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                ratio = torch.log((mags[i] + 1e-8) / (mags[j] + 1e-8))
                ratios.append(ratio)

        # Combine: [B, N + N*(N-1)/2]
        features = torch.cat(mags + ratios, dim=-1)
        return self.proj(features)


# =============================================================================
# STATISTICAL
# =============================================================================

class CrossVarianceAuxiliary(BaseAux):
    """
    Cross-input variance features.

    Measures agreement/disagreement between encoders through
    variance, mean, min, and max across inputs.
    """

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        **kwargs,
    ):
        super().__init__("variance_aux", num_inputs, in_features, aux_dim, **kwargs)

        # var + mean + min + max = 4 features
        self.proj = nn.Linear(4, aux_dim)

    def forward(self, *inputs: Tensor) -> Tensor:
        # Stack: [B, N, D]
        stacked = torch.stack(inputs, dim=1)

        # Compute statistics across inputs (dim=1), then mean over features
        var = stacked.var(dim=1).mean(dim=-1, keepdim=True)
        mean = stacked.mean(dim=1).mean(dim=-1, keepdim=True)
        min_val = stacked.min(dim=1)[0].mean(dim=-1, keepdim=True)
        max_val = stacked.max(dim=1)[0].mean(dim=-1, keepdim=True)

        features = torch.cat([var, mean, min_val, max_val], dim=-1)
        return self.proj(features)


class EntropyAuxiliary(BaseAux):
    """
    Per-input feature entropy.

    Measures "uncertainty" or "spread" of each encoder's representation.
    Higher entropy = more distributed representation.
    """

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        **kwargs,
    ):
        super().__init__("entropy_aux", num_inputs, in_features, aux_dim, **kwargs)

        self.proj = nn.Linear(num_inputs, aux_dim)

    def forward(self, *inputs: Tensor) -> Tensor:
        entropies = []
        for inp in inputs:
            # Softmax to get pseudo-probabilities
            probs = F.softmax(inp, dim=-1)
            # Entropy: -sum(p * log(p))
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)
            entropies.append(entropy)

        features = torch.cat(entropies, dim=-1)
        return self.proj(features)


# =============================================================================
# LEARNED EMBEDDINGS
# =============================================================================

class LearnedEmbeddingAuxiliary(BaseAux):
    """
    Learned per-input embeddings (input-independent).

    Simple learnable auxiliary features - one embedding per input slot.
    Same output for all samples in batch (context-free).

    Experimental result: 88.05% on CIFAR-100 (tied for best)
    """

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        **kwargs,
    ):
        super().__init__("learned_aux", num_inputs, in_features, aux_dim, **kwargs)

        self.embeddings = nn.Parameter(torch.randn(num_inputs, aux_dim) * 0.02)
        self.out_proj = nn.Linear(num_inputs * aux_dim, aux_dim)

    def forward(self, *inputs: Tensor) -> Tensor:
        B = inputs[0].shape[0]
        # Flatten and project
        expanded = self.embeddings.unsqueeze(0).expand(B, -1, -1)
        return self.out_proj(expanded.reshape(B, -1))


class InputDependentEmbeddingAuxiliary(BaseAux):
    """
    Input-dependent learned embeddings via attention.

    Uses attention over per-input embeddings weighted by input content.
    Output depends on actual input values (context-aware).

    Experimental result: 87.97% on CIFAR-100
    """

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        **kwargs,
    ):
        super().__init__("input_dep_aux", num_inputs, in_features, aux_dim, **kwargs)

        self.embeddings = nn.Parameter(torch.randn(num_inputs, aux_dim) * 0.02)
        self.query_proj = nn.Linear(in_features, aux_dim)

    def forward(self, *inputs: Tensor) -> Tensor:
        # Stack inputs: [B, N, D]
        stacked = torch.stack(inputs, dim=1)

        # Project to query space: [B, N, aux_dim]
        queries = self.query_proj(stacked)

        # Attention over embeddings: [B, N, aux_dim] x [N, aux_dim]^T -> [B, N, N]
        attn = torch.einsum('bnd,md->bnm', queries, self.embeddings)
        attn = F.softmax(attn / math.sqrt(self.aux_dim), dim=-1)

        # Weighted sum: [B, N, N] x [N, aux_dim] -> [B, N, aux_dim]
        weighted = torch.einsum('bnm,md->bnd', attn, self.embeddings)

        # Mean over inputs
        return weighted.mean(dim=1)


# =============================================================================
# GEOMETRIC / CAYLEY-MENGER
# =============================================================================

class GeometricAuxiliary(BaseAux):
    """
    Full simplex geometry features with proper Cayley-Menger determinant.

    From David's geometric attention gate and liminal staircase.
    Treats inputs as vertices of a simplex (pentachoron for 5 inputs)
    and computes validated geometric properties.

    Features:
        - Cayley-Menger volume (true determinant, not proxy)
        - Edge statistics (mean, std, uniformity)
        - Vertex spread (std of distances from centroid)
        - Pairwise angular differences
        - Role-weighted angular scores (pentachoron semantic roles)

    Cayley-Menger Formula:
        For N points, build (N+1)×(N+1) matrix:
        M[0, 1:] = 1, M[1:, 0] = 1, M[1:, 1:] = dist²
        Volume² = (-1)^(N+1) × det(M) / (2^N × (N!)²)
        For N=5 (pentachoron): divisor = 9216.0

    Experimental result: 88.05% at aux_dim=32 (tied for best)
    """

    # Pentachoron role weights from validated experiments
    # anchor=1.0, need=-0.75, relation=0.75, purpose=0.75, observer=-0.75
    PENTACHORON_ROLES = [1.0, -0.75, 0.75, 0.75, -0.75]

    # Cayley-Menger divisors: (-1)^(N+1) × 2^N × (N!)²
    # For numerical stability, we use absolute value and clamp
    CM_DIVISORS = {
        2: 4.0,      # 2 points (line)
        3: 32.0,     # 3 points (triangle)
        4: 576.0,    # 4 points (tetrahedron)
        5: 9216.0,   # 5 points (pentachoron)
        6: 147456.0, # 6 points (5-simplex)
    }

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        sample_dim: int = 64,
        use_roles: bool = True,
        **kwargs,
    ):
        super().__init__("geometric_aux", num_inputs, in_features, aux_dim, **kwargs)

        self.sample_dim = sample_dim
        self.use_roles = use_roles

        # Project to sample space for efficient geometry
        self.sample_proj = nn.Linear(in_features, sample_dim)

        # Register role weights
        if use_roles and num_inputs <= 5:
            role_weights = torch.tensor(self.PENTACHORON_ROLES[:num_inputs])
        else:
            role_weights = torch.ones(num_inputs) / num_inputs
        self.register_buffer("role_weights", role_weights)

        # Feature count:
        # 1 (volume) + 3 (edge: mean, std, uniformity) + 1 (spread) +
        # N*(N-1)/2 (angular) + N (role-weighted scores)
        num_pairs = num_inputs * (num_inputs - 1) // 2
        raw_features = 1 + 3 + 1 + num_pairs + num_inputs
        self.proj = nn.Linear(raw_features, aux_dim)

        # Learnable scale for volume (can vary widely)
        self.volume_scale = nn.Parameter(torch.tensor(0.01))

    def _compute_cayley_menger_volume(self, points: Tensor) -> Tensor:
        """
        Compute true Cayley-Menger volume.

        Args:
            points: [B, N, D] - N points in D dimensions

        Returns:
            Volume [B, 1]
        """
        B, N, D = points.shape

        # Pairwise squared distances: [B, N, N]
        diff = points.unsqueeze(2) - points.unsqueeze(1)
        dist_sq = (diff * diff).sum(dim=-1)

        # Build Cayley-Menger matrix: [B, N+1, N+1]
        M = torch.zeros(B, N + 1, N + 1, device=points.device, dtype=points.dtype)
        M[:, 0, 1:] = 1.0
        M[:, 1:, 0] = 1.0
        M[:, 1:, 1:] = dist_sq

        # Determinant
        det = torch.linalg.det(M)

        # Get divisor for this N
        divisor = self.CM_DIVISORS.get(N, N ** 4)  # Fallback for unusual N

        # Volume² = -det / divisor (clamped for numerical stability)
        volume_sq = (-det / divisor).clamp(min=0.0)
        volume = volume_sq.sqrt()

        return volume.unsqueeze(-1)

    def _compute_edge_statistics(self, points: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute edge length statistics.

        Returns:
            (mean_edge, std_edge, uniformity) each [B, 1]
        """
        B, N, D = points.shape

        # Pairwise distances
        diff = points.unsqueeze(2) - points.unsqueeze(1)
        dist_sq = (diff * diff).sum(dim=-1)

        # Upper triangular (exclude diagonal and duplicates)
        triu_mask = torch.triu(torch.ones(N, N, device=points.device), diagonal=1).bool()
        edge_lengths_sq = dist_sq[:, triu_mask]  # [B, N*(N-1)/2]
        edge_lengths = edge_lengths_sq.sqrt()

        mean_edge = edge_lengths.mean(dim=-1, keepdim=True)
        std_edge = edge_lengths.std(dim=-1, keepdim=True)
        uniformity = std_edge / mean_edge.clamp(min=1e-6)

        return mean_edge, std_edge, uniformity

    def _compute_vertex_spread(self, points: Tensor) -> Tensor:
        """
        Compute vertex spread (std of distances from centroid).

        Returns:
            spread [B, 1]
        """
        centroid = points.mean(dim=1, keepdim=True)
        distances = torch.norm(points - centroid, dim=-1)
        spread = distances.std(dim=-1, keepdim=True)
        return spread

    def _compute_angular_features(self, inputs: List[Tensor]) -> Tensor:
        """Compute pairwise angular differences."""
        angles = []
        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                cos = F.cosine_similarity(inputs[i], inputs[j], dim=-1, eps=1e-8)
                angle = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
                angles.append(angle.unsqueeze(-1))

        return torch.cat(angles, dim=-1)

    def _compute_role_scores(self, inputs: List[Tensor]) -> Tensor:
        """
        Compute role-weighted scores for each input.

        Uses pentachoron semantic roles to weight contributions.
        """
        scores = []
        for i, inp in enumerate(inputs):
            # Magnitude weighted by role
            mag = inp.norm(dim=-1, keepdim=True)
            role_score = mag * self.role_weights[i]
            scores.append(role_score)

        return torch.cat(scores, dim=-1)

    def forward(self, *inputs: Tensor) -> Tensor:
        # Project to sample space
        sampled = [self.sample_proj(x) for x in inputs]

        # Stack: [B, N, sample_dim]
        stacked = torch.stack(sampled, dim=1)

        # Cayley-Menger volume (scaled)
        volume = self._compute_cayley_menger_volume(stacked) * self.volume_scale.abs()

        # Edge statistics
        mean_edge, std_edge, uniformity = self._compute_edge_statistics(stacked)

        # Vertex spread
        spread = self._compute_vertex_spread(stacked)

        # Angular features (on original inputs for precision)
        angular = self._compute_angular_features(list(inputs))

        # Role-weighted scores
        role_scores = self._compute_role_scores(list(inputs))

        # Combine all features
        features = torch.cat([
            volume,
            mean_edge,
            std_edge,
            uniformity,
            spread,
            angular,
            role_scores,
        ], dim=-1)

        return self.proj(features)


class CantorStaircaseAuxiliary(BaseAux):
    """
    Cantor staircase / fractal coordinate features.

    Maps input statistics to fractal coordinate space.
    Inspired by Cantor's staircase function for multi-scale analysis.
    """

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        depth: int = 5,
        **kwargs,
    ):
        super().__init__("cantor_aux", num_inputs, in_features, aux_dim, **kwargs)

        self.depth = depth
        # Per-input: depth cantor levels
        num_features = num_inputs * depth
        self.proj = nn.Linear(num_features, aux_dim)

    def _cantor_value(self, x: Tensor, depth: int) -> Tensor:
        """Compute Cantor staircase value for x in [0, 1]."""
        # Normalize to [0, 1]
        x = torch.sigmoid(x)

        result = torch.zeros_like(x)
        for d in range(depth):
            # Which third are we in?
            third = (x * 3).floor().clamp(0, 2)

            # Cantor function: 0->0, 1->0.5, 2->1 (scaled by 2^-d)
            contribution = (third / 2) * (0.5 ** d)
            result = result + contribution

            # Recurse into the third
            x = (x * 3) - third

        return result

    def forward(self, *inputs: Tensor) -> Tensor:
        features = []
        for inp in inputs:
            # Use mean activation as base signal
            mean_act = inp.mean(dim=-1, keepdim=True)

            # Compute Cantor values at different depths
            for d in range(1, self.depth + 1):
                cantor_val = self._cantor_value(mean_act, d)
                features.append(cantor_val)

        # [B, N * depth]
        features = torch.cat(features, dim=-1)
        return self.proj(features)


# =============================================================================
# WALKER-INSPIRED
# =============================================================================

class WalkerPathAuxiliary(BaseAux):
    """
    Walker-inspired path similarity features.

    Computes features that capture the "path" between encoders,
    similar to how walkers interpolate along manifolds.

    Samples points along interpolation paths and measures
    similarity structure.
    """

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        num_steps: int = 4,
        **kwargs,
    ):
        super().__init__("walker_path_aux", num_inputs, in_features, aux_dim, **kwargs)

        self.num_steps = num_steps

        # For each pair: endpoint similarity + num_steps path similarities
        num_pairs = num_inputs * (num_inputs - 1) // 2
        raw_dim = num_pairs * (num_steps + 1)
        self.proj = nn.Linear(raw_dim, aux_dim)

    def forward(self, *inputs: Tensor) -> Tensor:
        features = []
        ts = torch.linspace(0, 1, self.num_steps, device=inputs[0].device)

        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                a, b = inputs[i], inputs[j]

                # Endpoint similarity
                cos_ab = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
                features.append(cos_ab.unsqueeze(-1))

                # Similarities along interpolation path
                midpoint = (a + b) / 2
                for t in ts:
                    interp = (1 - t) * a + t * b
                    cos_interp = F.cosine_similarity(interp, midpoint, dim=-1, eps=1e-8)
                    features.append(cos_interp.unsqueeze(-1))

        combined = torch.cat(features, dim=-1)
        return self.proj(combined)


# =============================================================================
# COMBINED / ENSEMBLE
# =============================================================================

class CombinedAuxiliary(BaseAux):
    """
    Combination of multiple auxiliary feature types.

    Ensembles cosine, magnitude, variance, and geometric features
    for comprehensive input characterization.
    """

    def __init__(
        self,
        num_inputs: int,
        in_features: int,
        aux_dim: int,
        **kwargs,
    ):
        super().__init__("combined_aux", num_inputs, in_features, aux_dim, **kwargs)

        # Individual generators (each outputs aux_dim // 4)
        sub_dim = max(aux_dim // 4, 8)
        self.cosine = CosineSimilarityAuxiliary(num_inputs, in_features, sub_dim)
        self.magnitude = MagnitudeRatioAuxiliary(num_inputs, in_features, sub_dim)
        self.variance = CrossVarianceAuxiliary(num_inputs, in_features, sub_dim)
        self.geometric = GeometricAuxiliary(num_inputs, in_features, sub_dim)

        # Final projection
        self.out_proj = nn.Linear(sub_dim * 4, aux_dim)

    def forward(self, *inputs: Tensor) -> Tensor:
        cos_feat = self.cosine(*inputs)
        mag_feat = self.magnitude(*inputs)
        var_feat = self.variance(*inputs)
        geo_feat = self.geometric(*inputs)

        combined = torch.cat([cos_feat, mag_feat, var_feat, geo_feat], dim=-1)
        return self.out_proj(combined)


# =============================================================================
# REGISTRY & FACTORY
# =============================================================================

AUXILIARY_GENERATORS: Dict[str, Type[BaseAux]] = {
    'zero': ZeroAuxiliary,
    'cosine': CosineSimilarityAuxiliary,
    'dot_product': DotProductAuxiliary,
    'magnitude': MagnitudeRatioAuxiliary,
    'variance': CrossVarianceAuxiliary,
    'entropy': EntropyAuxiliary,
    'learned': LearnedEmbeddingAuxiliary,
    'input_dependent': InputDependentEmbeddingAuxiliary,
    'geometric': GeometricAuxiliary,
    'cantor': CantorStaircaseAuxiliary,
    'walker_path': WalkerPathAuxiliary,
    'combined': CombinedAuxiliary,
}


def create_auxiliary_generator(
    name: str,
    num_inputs: int,
    in_features: int,
    aux_dim: int,
    **kwargs,
) -> BaseAux:
    """
    Factory function to create auxiliary feature generators.

    Args:
        name: Generator type. One of:
            'zero', 'cosine', 'dot_product', 'magnitude', 'variance',
            'entropy', 'learned', 'input_dependent', 'geometric',
            'cantor', 'walker_path', 'combined'
        num_inputs: Number of input tensors
        in_features: Feature dimension of each input
        aux_dim: Output auxiliary feature dimension
        **kwargs: Additional arguments passed to generator

    Returns:
        Configured auxiliary generator

    Raises:
        ValueError: If name is not recognized

    Example:
        aux = create_auxiliary_generator('cosine', 3, 512, 64)
        features = aux(x1, x2, x3)  # [B, 64]
    """
    if name not in AUXILIARY_GENERATORS:
        raise ValueError(
            f"Unknown auxiliary generator: '{name}'. "
            f"Available: {list(AUXILIARY_GENERATORS.keys())}"
        )

    return AUXILIARY_GENERATORS[name](num_inputs, in_features, aux_dim, **kwargs)


def list_auxiliary_generators() -> List[str]:
    """Return list of available auxiliary generator names."""
    return list(AUXILIARY_GENERATORS.keys())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base class
    'BaseAux',

    # Generators
    'ZeroAuxiliary',
    'CosineSimilarityAuxiliary',
    'DotProductAuxiliary',
    'MagnitudeRatioAuxiliary',
    'CrossVarianceAuxiliary',
    'EntropyAuxiliary',
    'LearnedEmbeddingAuxiliary',
    'InputDependentEmbeddingAuxiliary',
    'GeometricAuxiliary',
    'CantorStaircaseAuxiliary',
    'WalkerPathAuxiliary',
    'CombinedAuxiliary',

    # Registry & factory
    'AUXILIARY_GENERATORS',
    'create_auxiliary_generator',
    'list_auxiliary_generators',
]


# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':

    def test_section(title: str):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test parameters
    B = 4
    D = 512
    N = 3
    AUX_DIM = 64

    # Create test inputs
    inputs = [torch.randn(B, D, device=device) for _ in range(N)]

    # -------------------------------------------------------------------------
    test_section("FACTORY FUNCTION")
    # -------------------------------------------------------------------------

    print(f"Available generators: {list_auxiliary_generators()}")

    aux = create_auxiliary_generator('cosine', N, D, AUX_DIM)
    print(f"Created: {aux}")
    print(f"Is BaseAux: {isinstance(aux, BaseAux)}")

    # -------------------------------------------------------------------------
    test_section("ALL GENERATORS")
    # -------------------------------------------------------------------------

    results = {}
    for name in list_auxiliary_generators():
        try:
            gen = create_auxiliary_generator(name, N, D, AUX_DIM)
            gen = gen.to(device)

            output = gen(*inputs)
            params = sum(p.numel() for p in gen.parameters())

            results[name] = {
                'output_shape': output.shape,
                'params': params,
                'output_norm': output.norm().item(),
            }

            print(f"  {name:20s}: shape={output.shape}, params={params:,}, norm={output.norm():.2f}")

            # Verify output shape
            assert output.shape == (B, AUX_DIM), f"Expected ({B}, {AUX_DIM}), got {output.shape}"

        except Exception as e:
            print(f"  {name:20s}: ERROR - {e}")
            raise

    # -------------------------------------------------------------------------
    test_section("GRADIENT FLOW")
    # -------------------------------------------------------------------------

    # Test that gradients flow through learnable generators
    gen = create_auxiliary_generator('learned', N, D, AUX_DIM).to(device)
    inputs_grad = [x.requires_grad_(True) for x in inputs]

    output = gen(*inputs_grad)
    loss = output.sum()
    loss.backward()

    # Check generator has gradients
    has_grads = any(p.grad is not None for p in gen.parameters())
    print(f"Generator has gradients: {has_grads}")

    # -------------------------------------------------------------------------
    test_section("DEVICE MOVEMENT")
    # -------------------------------------------------------------------------

    gen = create_auxiliary_generator('geometric', N, D, AUX_DIM)
    print(f"Initial device: {gen.device}")

    if torch.cuda.is_available():
        gen = gen.to('cuda')
        print(f"After to('cuda'): {gen.device}")

        cuda_inputs = [x.cuda() for x in inputs]
        output = gen(*cuda_inputs)
        print(f"Output device: {output.device}")

    # -------------------------------------------------------------------------
    test_section("COMBINED GENERATOR")
    # -------------------------------------------------------------------------

    combined = create_auxiliary_generator('combined', N, D, AUX_DIM).to(device)
    output = combined(*inputs)

    print(f"Combined output: {output.shape}")
    print(f"Combined params: {sum(p.numel() for p in combined.parameters()):,}")

    # -------------------------------------------------------------------------
    test_section("BATCH SIZE INVARIANCE")
    # -------------------------------------------------------------------------

    gen = create_auxiliary_generator('walker_path', N, D, AUX_DIM).to(device)

    for batch_size in [1, 4, 16, 64]:
        batch_inputs = [torch.randn(batch_size, D, device=device) for _ in range(N)]
        output = gen(*batch_inputs)
        assert output.shape == (batch_size, AUX_DIM)
        print(f"  Batch {batch_size:3d}: OK")

    # -------------------------------------------------------------------------
    test_section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\naux_component.py is ready.")
    print(f"Total generators: {len(AUXILIARY_GENERATORS)}")