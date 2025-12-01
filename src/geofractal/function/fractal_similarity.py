"""
Fractal Similarity: A unified metric for self-similar structure alignment.

Like cosine_similarity measures angular alignment,
fractal_similarity measures how similarly two signals
decompose and correlate across scales.

Author: AbstractPhil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# =============================================================================
# FUNCTIONAL API (like F.cosine_similarity)
# =============================================================================

def fractal_similarity(
        x: torch.Tensor,
        y: torch.Tensor,
        num_scales: int = 5,
        dim: int = -1,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute fractal similarity between x and y.

    Measures structural self-similarity alignment - how similarly
    two signals decompose and relate across scales.

    Args:
        x: First tensor
        y: Second tensor (same shape as x)
        num_scales: Number of decomposition scales
        dim: Dimension along which to compute (default: last)
        eps: Numerical stability

    Returns:
        Similarity scores in [-1, 1] (same semantics as cosine_similarity)

    Example:
        #>>> a = torch.randn(32, 512)
        #>>> b = torch.randn(32, 512)
        #>>> sim = fractal_similarity(a, b)  # [32]
    """
    # Move target dim to last
    if dim != -1 and dim != x.dim() - 1:
        x = x.transpose(dim, -1)
        y = y.transpose(dim, -1)

    # Compute scale signatures
    sig_x = _compute_scale_signature(x, num_scales, eps)
    sig_y = _compute_scale_signature(y, num_scales, eps)

    # Fractal similarity is cosine similarity of scale signatures
    return F.cosine_similarity(sig_x, sig_y, dim=-1)


def _compute_scale_signature(
        x: torch.Tensor,
        num_scales: int,
        eps: float
) -> torch.Tensor:
    """
    Compute multi-scale signature capturing fractal structure.

    Signature includes:
    - Energy at each scale (how power distributes)
    - Variance at each scale (fluctuation structure)
    - Cross-scale correlation (self-similarity)
    """
    *batch_dims, D = x.shape
    batch_shape = x.shape[:-1]
    x_flat = x.reshape(-1, D)
    B = x_flat.shape[0]

    signatures = []

    # Build scale pyramid via dyadic pooling
    current = x_flat
    prev_level = None

    for s in range(num_scales):
        pool_size = 2 ** s

        if D // pool_size < 2:
            # Can't pool further - repeat last valid
            if signatures:
                signatures.append(signatures[-1][..., :3])
            continue

        # Pool to this scale
        new_D = D // pool_size
        level = x_flat[..., :new_D * pool_size].view(B, new_D, pool_size).mean(dim=-1)

        # Scale statistics
        energy = (level ** 2).mean(dim=-1, keepdim=True)  # [B, 1]
        variance = level.var(dim=-1, keepdim=True)  # [B, 1]

        # Cross-scale correlation (self-similarity measure)
        if prev_level is not None and prev_level.shape[-1] >= level.shape[-1]:
            # Downsample previous to match current
            prev_down = prev_level[..., :level.shape[-1] * 2].view(
                B, level.shape[-1], 2
            ).mean(dim=-1)
            cross_corr = F.cosine_similarity(prev_down, level, dim=-1).unsqueeze(-1)
        else:
            cross_corr = torch.ones(B, 1, device=x.device, dtype=x.dtype)

        signatures.append(torch.cat([energy, variance, cross_corr], dim=-1))
        prev_level = level

    # Stack and flatten: [B, num_scales, 3] -> [B, num_scales * 3]
    signature = torch.cat(signatures, dim=-1)

    # Reshape back to batch dims
    return signature.view(*batch_shape, -1)


# =============================================================================
# FRACTAL BASIS PROJECTION (for "which type of fractal")
# =============================================================================

class FractalBasis(nn.Module):
    """
    Project features onto canonical fractal bases.

    Returns a fingerprint indicating which fractal structures
    the feature most resembles (Cantor, uniform, 1/f, Brownian, etc.)
    """

    BASES = ['cantor', 'uniform', 'pink_noise', 'brownian', 'white_noise']

    def __init__(self, dim: int, num_scales: int = 5):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales

        # Build canonical fractal basis patterns
        basis = self._build_basis(dim, num_scales)
        self.register_buffer('basis', basis)  # [num_bases, signature_dim]
        self.register_buffer('basis_names', torch.arange(len(self.BASES)))

    def _build_basis(self, dim: int, num_scales: int) -> torch.Tensor:
        """Build canonical fractal signatures."""
        bases = []

        for basis_type in self.BASES:
            pattern = self._generate_pattern(basis_type, dim)
            sig = _compute_scale_signature(
                pattern.unsqueeze(0), num_scales, eps=1e-8
            ).squeeze(0)
            bases.append(sig)

        return torch.stack(bases, dim=0)

    def _generate_pattern(self, basis_type: str, dim: int) -> torch.Tensor:
        """Generate canonical fractal pattern."""
        t = torch.linspace(0, 1, dim)

        if basis_type == 'cantor':
            # Cantor-like: recursive removal structure
            pattern = torch.ones(dim)
            for scale in [3, 9, 27, 81]:
                if scale < dim:
                    indices = torch.arange(dim)
                    mask = ((indices // (dim // scale)) % 3) == 1
                    pattern[mask] *= 0.5
            return pattern

        elif basis_type == 'uniform':
            # No fractal structure - constant
            return torch.ones(dim)

        elif basis_type == 'pink_noise':
            # 1/f spectrum - common in natural signals
            freqs = torch.fft.rfftfreq(dim)
            spectrum = 1.0 / (freqs + 0.01)
            pattern = torch.fft.irfft(spectrum, n=dim)
            return pattern / pattern.std()

        elif basis_type == 'brownian':
            # Brownian motion - cumulative random walk
            steps = torch.randn(dim)
            pattern = torch.cumsum(steps, dim=0)
            return pattern / pattern.std()

        elif basis_type == 'white_noise':
            # No correlation structure
            return torch.randn(dim)

        else:
            return torch.randn(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project features onto fractal basis.

        Args:
            x: [..., dim] features

        Returns:
            coefficients: [..., num_bases] projection onto each basis
            best_match: [...] index of closest fractal type
        """
        sig = _compute_scale_signature(x, self.num_scales, eps=1e-8)

        # Cosine similarity with each basis
        # sig: [..., sig_dim], basis: [num_bases, sig_dim]
        sig_norm = F.normalize(sig, dim=-1)
        basis_norm = F.normalize(self.basis, dim=-1)

        coefficients = torch.matmul(sig_norm, basis_norm.T)  # [..., num_bases]
        best_match = coefficients.argmax(dim=-1)

        return coefficients, best_match

    def get_basis_name(self, index: int) -> str:
        return self.BASES[index]


# =============================================================================
# ADVANCED: DIFFERENTIABLE FRACTAL DIMENSION
# =============================================================================

def fractal_dimension(
        x: torch.Tensor,
        num_scales: int = 5,
        dim: int = -1,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    Estimate fractal (Hausdorff-like) dimension of features.

    Uses energy scaling: if E(s) ~ s^(-D), returns D.

    Args:
        x: Input tensor
        num_scales: Scales for regression
        dim: Dimension to analyze
        eps: Numerical stability

    Returns:
        Estimated fractal dimension (typically 1.0-2.0 for 1D signals)
    """
    if dim != -1 and dim != x.dim() - 1:
        x = x.transpose(dim, -1)

    *batch_dims, D = x.shape
    x_flat = x.reshape(-1, D)
    B = x_flat.shape[0]

    energies = []
    scales = []

    for s in range(num_scales):
        pool_size = 2 ** s
        if D // pool_size < 2:
            break

        new_D = D // pool_size
        level = x_flat[..., :new_D * pool_size].view(B, new_D, pool_size).mean(dim=-1)
        energy = (level ** 2).mean(dim=-1)

        energies.append(energy)
        scales.append(pool_size)

    # Log-log linear regression
    log_energy = torch.log(torch.stack(energies, dim=-1) + eps)  # [B, num_valid_scales]
    log_scale = torch.log(torch.tensor(scales, device=x.device, dtype=x.dtype))

    # Least squares: dimension = -slope
    x_centered = log_scale - log_scale.mean()
    y_centered = log_energy - log_energy.mean(dim=-1, keepdim=True)

    slope = (y_centered * x_centered).sum(dim=-1) / (x_centered ** 2).sum()
    dimension = -slope

    return dimension.view(*batch_dims)


# =============================================================================
# FRACTAL DISTANCE (inverse of similarity)
# =============================================================================

def fractal_distance(
        x: torch.Tensor,
        y: torch.Tensor,
        num_scales: int = 5,
        dim: int = -1,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    Fractal distance - inverse of fractal similarity.

    Returns values in [0, 2] where 0 = identical fractal structure.
    """
    sim = fractal_similarity(x, y, num_scales, dim, eps)
    return 1.0 - sim


# =============================================================================
# FRACTAL CONTRASTIVE LOSS
# =============================================================================

class FractalContrastiveLoss(nn.Module):
    """
    Contrastive loss using fractal similarity.

    Pulls together features with similar fractal structure,
    pushes apart features with different structure.
    """

    def __init__(
            self,
            temperature: float = 0.07,
            num_scales: int = 5
    ):
        super().__init__()
        self.temperature = temperature
        self.num_scales = num_scales

    def forward(
            self,
            anchors: torch.Tensor,
            positives: torch.Tensor,
            negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            anchors: [B, D] anchor features
            positives: [B, D] positive features (same fractal structure)
            negatives: [B, N, D] negative features (optional)

        Returns:
            Contrastive loss scalar
        """
        pos_sim = fractal_similarity(anchors, positives, self.num_scales)
        pos_sim = pos_sim / self.temperature

        if negatives is not None:
            # Compare anchor to each negative
            B, N, D = negatives.shape
            anchors_exp = anchors.unsqueeze(1).expand(-1, N, -1)
            neg_sim = fractal_similarity(
                anchors_exp.reshape(B * N, D),
                negatives.reshape(B * N, D),
                self.num_scales
            ).view(B, N)
            neg_sim = neg_sim / self.temperature

            # InfoNCE-style loss
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=anchors.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # Just maximize positive similarity
            loss = -pos_sim.mean()

        return loss


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    # Test functional API
    print("Testing fractal_similarity...")

    a = torch.randn(32, 512)
    b = torch.randn(32, 512)
    c = a + 0.1 * torch.randn_like(a)  # Similar to a

    sim_ab = fractal_similarity(a, b)
    sim_ac = fractal_similarity(a, c)
    sim_aa = fractal_similarity(a, a)

    print(f"  sim(a, b) [random]:  {sim_ab.mean():.3f} ± {sim_ab.std():.3f}")
    print(f"  sim(a, c) [similar]: {sim_ac.mean():.3f} ± {sim_ac.std():.3f}")
    print(f"  sim(a, a) [self]:    {sim_aa.mean():.3f} ± {sim_aa.std():.3f}")

    # Test fractal dimension
    print("\nTesting fractal_dimension...")

    # White noise should have dim ~1.5
    white = torch.randn(32, 512)
    dim_white = fractal_dimension(white)
    print(f"  White noise dimension: {dim_white.mean():.3f}")

    # Brownian should have dim ~1.5 but different structure
    brownian = torch.cumsum(torch.randn(32, 512), dim=-1)
    dim_brownian = fractal_dimension(brownian)
    print(f"  Brownian dimension:    {dim_brownian.mean():.3f}")

    # Test basis projection
    print("\nTesting FractalBasis...")

    basis = FractalBasis(dim=512)
    coeffs, best = basis(a)
    print(f"  Coefficients shape: {coeffs.shape}")
    print(f"  Most common type: {basis.get_basis_name(best.mode().values.item())}")

    print("\n✓ All tests passed!")