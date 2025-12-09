"""
geofractal.router.components.embedding_component
================================================

Multi-embedding system for wide models.

BaseEmbedding -> DualEmbedding -> TriEmbedding -> QuadEmbedding

Each level has:
    - dimension: core dimensional structure
    - modulate(): formula for interaction between components
    - augment(): basis for extension/modification
    - embed(): final output

Design Philosophy:
    Wide models require multiple coordinated embeddings per tower.
    Each embedding level has geometric meaning:
    - Dual: Binary opposition (real/imaginary, content/position)
    - Tri: Barycentric (3-simplex, triangular interpolation)
    - Quad: Quaternionic (4D rotation, crystalline structure)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, Tuple, Union, Literal
from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.components.torch_component import TorchComponent


# =============================================================================
# BASE EMBEDDING
# =============================================================================

class BaseEmbedding(TorchComponent):
    """
    Abstract embedding primitive.

    All embeddings define:
        - dimension: structural shape
        - modulate(): interaction formula
        - augment(): extension basis
        - embed(): final computation
    """

    def __init__(
        self,
        name: str,
        embed_dim: int,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Core dimensional structure (1 for base, 2 for dual, etc.)."""
        ...

    @abstractmethod
    def modulate(self, x: Tensor, modulator: Tensor) -> Tensor:
        """Formula for modulation. How external signals affect embedding."""
        ...

    @abstractmethod
    def augment(self, x: Tensor, augmentation: Tensor) -> Tensor:
        """Basis for augmentation. How to extend/modify embedding."""
        ...

    @abstractmethod
    def embed(self, x: Union[Tensor, int]) -> Tensor:
        """Core embedding computation."""
        ...

    def forward(self, x: Union[Tensor, int]) -> Tensor:
        return self.embed(x)


# =============================================================================
# DUAL EMBEDDING
# =============================================================================

class DualEmbedding(BaseEmbedding):
    """
    Two-component embedding with binary interaction.

    Structure: (primary, secondary)
    Geometry: Complex plane, opposition, complementarity

    Modulation: Multiplicative gating (primary * gate(secondary))
    Augmentation: Additive residual (primary + scale * secondary)

    Use cases:
        - Content + Position
        - Real + Imaginary
        - Signal + Noise
        - Base + Delta
    """

    def __init__(
        self,
        name: str,
        embed_dim: int,
        modulation: Literal['multiply', 'gate', 'complex'] = 'gate',
        augmentation: Literal['add', 'lerp', 'residual'] = 'add',
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, embed_dim, uuid, **kwargs)

        self._modulation_mode = modulation
        self._augmentation_mode = augmentation

        # Learnable interaction parameters
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.scale = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.tensor(0.5))

    @property
    def dimension(self) -> int:
        return 2

    def modulate(self, primary: Tensor, secondary: Tensor) -> Tensor:
        """
        Binary modulation formula.

        multiply: primary * secondary
        gate: primary * sigmoid(gate(secondary))
        complex: complex multiplication (real/imag interpretation)
        """
        if self._modulation_mode == 'multiply':
            return primary * secondary

        elif self._modulation_mode == 'gate':
            g = torch.sigmoid(self.gate(secondary))
            return primary * g

        elif self._modulation_mode == 'complex':
            # Interpret as complex: primary=real, secondary=imag
            # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            half = self.embed_dim // 2
            p1, p2 = primary[..., :half], primary[..., half:]
            s1, s2 = secondary[..., :half], secondary[..., half:]

            out1 = p1 * s1 - p2 * s2
            out2 = p1 * s2 + p2 * s1
            return torch.cat([out1, out2], dim=-1)

        else:
            raise ValueError(f"Unknown modulation: {self._modulation_mode}")

    def augment(self, primary: Tensor, secondary: Tensor) -> Tensor:
        """
        Binary augmentation basis.

        add: primary + scale * secondary
        lerp: lerp(primary, secondary, alpha)
        residual: primary + secondary (unit scale)
        """
        if self._augmentation_mode == 'add':
            return primary + self.scale * secondary

        elif self._augmentation_mode == 'lerp':
            alpha = torch.sigmoid(self.alpha)
            return (1 - alpha) * primary + alpha * secondary

        elif self._augmentation_mode == 'residual':
            return primary + secondary

        else:
            raise ValueError(f"Unknown augmentation: {self._augmentation_mode}")

    @abstractmethod
    def embed_primary(self, x: Union[Tensor, int]) -> Tensor:
        """Override: compute primary embedding."""
        ...

    @abstractmethod
    def embed_secondary(self, x: Union[Tensor, int]) -> Tensor:
        """Override: compute secondary embedding."""
        ...

    def embed(self, x: Union[Tensor, int]) -> Tuple[Tensor, Tensor]:
        """Return both embeddings."""
        return self.embed_primary(x), self.embed_secondary(x)

    def embed_combined(
        self,
        x: Union[Tensor, int],
        mode: Literal['modulate', 'augment', 'both'] = 'augment',
    ) -> Tensor:
        """Combined embedding using modulation or augmentation."""
        primary, secondary = self.embed(x)

        if mode == 'modulate':
            return self.modulate(primary, secondary)
        elif mode == 'augment':
            return self.augment(primary, secondary)
        elif mode == 'both':
            modulated = self.modulate(primary, secondary)
            return self.augment(modulated, secondary)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# TRI EMBEDDING
# =============================================================================

class TriEmbedding(BaseEmbedding):
    """
    Three-component embedding with barycentric interaction.

    Structure: (alpha, beta, gamma)
    Geometry: 2-simplex (triangle), barycentric coordinates

    Modulation: Weighted geometric mean
    Augmentation: Barycentric interpolation

    Use cases:
        - RGB channels
        - Past/Present/Future
        - Low/Mid/High frequency
        - Anchor/Positive/Negative (triplet)
    """

    def __init__(
        self,
        name: str,
        embed_dim: int,
        modulation: Literal['geometric', 'hadamard', 'bilinear'] = 'geometric',
        augmentation: Literal['barycentric', 'sum', 'attention'] = 'barycentric',
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, embed_dim, uuid, **kwargs)

        self._modulation_mode = modulation
        self._augmentation_mode = augmentation

        # Barycentric weights (learnable, normalized via softmax)
        self._bary_logits = nn.Parameter(torch.zeros(3))

        # Bilinear interaction
        self.bilinear_ab = nn.Bilinear(embed_dim, embed_dim, embed_dim, bias=False)
        self.bilinear_bc = nn.Bilinear(embed_dim, embed_dim, embed_dim, bias=False)

        # Attention for attention mode
        self.attn_proj = nn.Linear(embed_dim, 1)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def barycentric_weights(self) -> Tensor:
        """Normalized barycentric coordinates (sum to 1)."""
        return F.softmax(self._bary_logits, dim=0)

    def modulate(self, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
        """
        Ternary modulation formula.

        geometric: (alpha * beta * gamma)^(1/3) - geometric mean
        hadamard: alpha * beta * gamma - element-wise triple product
        bilinear: bilinear(bilinear(alpha, beta), gamma)
        """
        if self._modulation_mode == 'geometric':
            # Geometric mean (stabilized)
            product = alpha * beta * gamma
            sign = torch.sign(product)
            return sign * (torch.abs(product) + 1e-8).pow(1/3)

        elif self._modulation_mode == 'hadamard':
            return alpha * beta * gamma

        elif self._modulation_mode == 'bilinear':
            ab = self.bilinear_ab(alpha, beta)
            return self.bilinear_bc(ab, gamma)

        else:
            raise ValueError(f"Unknown modulation: {self._modulation_mode}")

    def augment(self, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
        """
        Ternary augmentation basis.

        barycentric: w_a * alpha + w_b * beta + w_c * gamma (weights sum to 1)
        sum: alpha + beta + gamma
        attention: softmax attention over three components
        """
        if self._augmentation_mode == 'barycentric':
            w = self.barycentric_weights
            return w[0] * alpha + w[1] * beta + w[2] * gamma

        elif self._augmentation_mode == 'sum':
            return alpha + beta + gamma

        elif self._augmentation_mode == 'attention':
            # Stack and attend
            stacked = torch.stack([alpha, beta, gamma], dim=-2)  # [..., 3, D]
            scores = self.attn_proj(stacked).squeeze(-1)  # [..., 3]
            weights = F.softmax(scores, dim=-1)  # [..., 3]
            return (stacked * weights.unsqueeze(-1)).sum(dim=-2)

        else:
            raise ValueError(f"Unknown augmentation: {self._augmentation_mode}")

    @abstractmethod
    def embed_alpha(self, x: Union[Tensor, int]) -> Tensor:
        """Override: first component."""
        ...

    @abstractmethod
    def embed_beta(self, x: Union[Tensor, int]) -> Tensor:
        """Override: second component."""
        ...

    @abstractmethod
    def embed_gamma(self, x: Union[Tensor, int]) -> Tensor:
        """Override: third component."""
        ...

    def embed(self, x: Union[Tensor, int]) -> Tuple[Tensor, Tensor, Tensor]:
        """Return all three embeddings."""
        return self.embed_alpha(x), self.embed_beta(x), self.embed_gamma(x)

    def embed_combined(
        self,
        x: Union[Tensor, int],
        mode: Literal['modulate', 'augment', 'both'] = 'augment',
    ) -> Tensor:
        """Combined embedding."""
        alpha, beta, gamma = self.embed(x)

        if mode == 'modulate':
            return self.modulate(alpha, beta, gamma)
        elif mode == 'augment':
            return self.augment(alpha, beta, gamma)
        elif mode == 'both':
            modulated = self.modulate(alpha, beta, gamma)
            return self.augment(modulated, modulated, modulated)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# QUAD EMBEDDING
# =============================================================================

class QuadEmbedding(BaseEmbedding):
    """
    Four-component embedding with quaternionic interaction.

    Structure: (w, x, y, z) or (1, i, j, k)
    Geometry: 3-simplex (tetrahedron), quaternion algebra

    Modulation: Quaternion multiplication
    Augmentation: 4-way interpolation or crystalline structure

    Use cases:
        - 3D rotation encoding
        - Four-way attention (Q, K, V, O)
        - Crystalline token structure
        - Quad-tree hierarchical
    """

    def __init__(
        self,
        name: str,
        embed_dim: int,
        modulation: Literal['quaternion', 'hadamard', 'pairwise'] = 'quaternion',
        augmentation: Literal['simplex', 'sum', 'hierarchical'] = 'simplex',
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, embed_dim, uuid, **kwargs)

        self._modulation_mode = modulation
        self._augmentation_mode = augmentation

        # Simplex weights (4 vertices)
        self._simplex_logits = nn.Parameter(torch.zeros(4))

        # Hierarchical combination weights
        self.hier_ab = nn.Parameter(torch.tensor(0.5))
        self.hier_cd = nn.Parameter(torch.tensor(0.5))
        self.hier_final = nn.Parameter(torch.tensor(0.5))

    @property
    def dimension(self) -> int:
        return 4

    @property
    def simplex_weights(self) -> Tensor:
        """Normalized 3-simplex coordinates."""
        return F.softmax(self._simplex_logits, dim=0)

    def modulate(self, w: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        """
        Quaternary modulation formula.

        quaternion: Hamilton product (w + xi + yj + zk)
        hadamard: w * x * y * z
        pairwise: (w*x) * (y*z) - pairwise then combine
        """
        if self._modulation_mode == 'quaternion':
            # Quaternion multiplication: q1 * q2
            # Split embed_dim into 4 parts for quaternion components
            d = self.embed_dim // 4

            w1, x1, y1, z1 = w[..., :d], x[..., :d], y[..., :d], z[..., :d]
            w2, x2, y2, z2 = w[..., d:2*d], x[..., d:2*d], y[..., d:2*d], z[..., d:2*d]

            # Hamilton product
            out_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            out_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            out_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            out_z = w1*z2 + x1*y2 - y1*x2 + z1*w2

            return torch.cat([out_w, out_x, out_y, out_z], dim=-1)

        elif self._modulation_mode == 'hadamard':
            return w * x * y * z

        elif self._modulation_mode == 'pairwise':
            wx = w * x
            yz = y * z
            return wx * yz

        else:
            raise ValueError(f"Unknown modulation: {self._modulation_mode}")

    def augment(self, w: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        """
        Quaternary augmentation basis.

        simplex: weighted sum on 3-simplex (tetrahedron)
        sum: w + x + y + z
        hierarchical: lerp(lerp(w,x), lerp(y,z)) - binary tree
        """
        if self._augmentation_mode == 'simplex':
            s = self.simplex_weights
            return s[0] * w + s[1] * x + s[2] * y + s[3] * z

        elif self._augmentation_mode == 'sum':
            return w + x + y + z

        elif self._augmentation_mode == 'hierarchical':
            # Binary tree combination
            a_ab = torch.sigmoid(self.hier_ab)
            a_cd = torch.sigmoid(self.hier_cd)
            a_final = torch.sigmoid(self.hier_final)

            ab = (1 - a_ab) * w + a_ab * x
            cd = (1 - a_cd) * y + a_cd * z
            return (1 - a_final) * ab + a_final * cd

        else:
            raise ValueError(f"Unknown augmentation: {self._augmentation_mode}")

    @abstractmethod
    def embed_w(self, x: Union[Tensor, int]) -> Tensor:
        """Override: w component (real/scalar)."""
        ...

    @abstractmethod
    def embed_x(self, x: Union[Tensor, int]) -> Tensor:
        """Override: x component (i)."""
        ...

    @abstractmethod
    def embed_y(self, x: Union[Tensor, int]) -> Tensor:
        """Override: y component (j)."""
        ...

    @abstractmethod
    def embed_z(self, x: Union[Tensor, int]) -> Tensor:
        """Override: z component (k)."""
        ...

    def embed(self, inp: Union[Tensor, int]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return all four embeddings."""
        return self.embed_w(inp), self.embed_x(inp), self.embed_y(inp), self.embed_z(inp)

    def embed_combined(
        self,
        inp: Union[Tensor, int],
        mode: Literal['modulate', 'augment', 'both'] = 'augment',
    ) -> Tensor:
        """Combined embedding."""
        w, x, y, z = self.embed(inp)

        if mode == 'modulate':
            return self.modulate(w, x, y, z)
        elif mode == 'augment':
            return self.augment(w, x, y, z)
        elif mode == 'both':
            modulated = self.modulate(w, x, y, z)
            return self.augment(modulated, modulated, modulated, modulated)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# CONCRETE IMPLEMENTATIONS
# =============================================================================

class LookupEmbedding(BaseEmbedding):
    """Simple lookup table."""

    def __init__(self, name: str, num_embeddings: int, embed_dim: int):
        super().__init__(name, embed_dim)
        self.table = nn.Embedding(num_embeddings, embed_dim)

    @property
    def dimension(self) -> int:
        return 1

    def modulate(self, x: Tensor, modulator: Tensor) -> Tensor:
        return x * torch.sigmoid(modulator)

    def augment(self, x: Tensor, augmentation: Tensor) -> Tensor:
        return x + augmentation

    def embed(self, x: Tensor) -> Tensor:
        return self.table(x)


class SinusoidalEmbedding(BaseEmbedding):
    """Sinusoidal positional/timestep embedding."""

    def __init__(self, name: str, embed_dim: int, max_period: float = 10000.0):
        super().__init__(name, embed_dim)
        half = embed_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer('freqs', freqs)

    @property
    def dimension(self) -> int:
        return 1

    def modulate(self, x: Tensor, modulator: Tensor) -> Tensor:
        return x * modulator

    def augment(self, x: Tensor, augmentation: Tensor) -> Tensor:
        return x + augmentation

    def embed(self, x: Tensor) -> Tensor:
        angles = x.float().unsqueeze(-1) * self.freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class RotaryEmbedding(BaseEmbedding):
    """Rotary position embedding (RoPE)."""

    def __init__(
        self,
        name: str,
        embed_dim: int,
        theta: float = 10000.0,
        theta_scale: float = 1.0,
    ):
        super().__init__(name, embed_dim)
        inv_freq = 1.0 / (theta ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inv_freq = inv_freq * theta_scale
        self.register_buffer('inv_freq', inv_freq)
        self._cos_cache = None
        self._sin_cache = None
        self._cache_len = 0

    @property
    def dimension(self) -> int:
        return 1

    def modulate(self, x: Tensor, modulator: Tensor) -> Tensor:
        return x * modulator

    def augment(self, x: Tensor, augmentation: Tensor) -> Tensor:
        return x + augmentation

    def embed(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Returns (cos, sin) each of shape [L, D]."""
        if self._cos_cache is not None and seq_len <= self._cache_len:
            return self._cos_cache[:seq_len], self._sin_cache[:seq_len]

        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [L, D//2]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [L, D]

        self._cos_cache = freqs.cos()
        self._sin_cache = freqs.sin()
        self._cache_len = seq_len

        return self._cos_cache, self._sin_cache

    def apply_rotary(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply rotary embeddings to Q and K."""
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _rotate(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return (x * cos) + (rotated * sin)


class DualLookupEmbedding(DualEmbedding):
    """Dual embedding with two lookup tables."""

    def __init__(
        self,
        name: str,
        num_embeddings: int,
        embed_dim: int,
        modulation: str = 'gate',
        augmentation: str = 'add',
    ):
        super().__init__(name, embed_dim, modulation, augmentation)
        self.primary_table = nn.Embedding(num_embeddings, embed_dim)
        self.secondary_table = nn.Embedding(num_embeddings, embed_dim)

    def embed_primary(self, x: Tensor) -> Tensor:
        return self.primary_table(x)

    def embed_secondary(self, x: Tensor) -> Tensor:
        return self.secondary_table(x)


class DualContentPositionEmbedding(DualEmbedding):
    """Dual embedding: content (lookup) + position (sinusoidal)."""

    def __init__(
        self,
        name: str,
        num_embeddings: int,
        embed_dim: int,
        max_period: float = 10000.0,
        modulation: str = 'gate',
        augmentation: str = 'add',
    ):
        super().__init__(name, embed_dim, modulation, augmentation)
        self.content = nn.Embedding(num_embeddings, embed_dim)

        half = embed_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer('freqs', freqs)

    def embed_primary(self, x: Tensor) -> Tensor:
        return self.content(x)

    def embed_secondary(self, x: Tensor) -> Tensor:
        # Generate positions from sequence length
        positions = torch.arange(x.shape[-1], device=x.device)
        angles = positions.float().unsqueeze(-1) * self.freqs
        pos_emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        # Broadcast to batch
        return pos_emb.unsqueeze(0).expand(x.shape[0], -1, -1)


class TriFrequencyEmbedding(TriEmbedding):
    """Tri embedding with low/mid/high frequency bands."""

    def __init__(
        self,
        name: str,
        embed_dim: int,
        max_periods: Tuple[float, float, float] = (100.0, 1000.0, 10000.0),
        modulation: str = 'geometric',
        augmentation: str = 'barycentric',
    ):
        super().__init__(name, embed_dim, modulation, augmentation)

        half = embed_dim // 2
        for period, label in zip(max_periods, ['low', 'mid', 'high']):
            freqs = torch.exp(-math.log(period) * torch.arange(half) / half)
            self.register_buffer(f'freqs_{label}', freqs)

    def _sinusoidal(self, x: Tensor, freqs: Tensor) -> Tensor:
        angles = x.float().unsqueeze(-1) * freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def embed_alpha(self, x: Tensor) -> Tensor:
        return self._sinusoidal(x, self.freqs_low)

    def embed_beta(self, x: Tensor) -> Tensor:
        return self._sinusoidal(x, self.freqs_mid)

    def embed_gamma(self, x: Tensor) -> Tensor:
        return self._sinusoidal(x, self.freqs_high)


class TriLookupEmbedding(TriEmbedding):
    """Tri embedding with three lookup tables."""

    def __init__(
        self,
        name: str,
        num_embeddings: int,
        embed_dim: int,
        modulation: str = 'geometric',
        augmentation: str = 'barycentric',
    ):
        super().__init__(name, embed_dim, modulation, augmentation)
        self.table_alpha = nn.Embedding(num_embeddings, embed_dim)
        self.table_beta = nn.Embedding(num_embeddings, embed_dim)
        self.table_gamma = nn.Embedding(num_embeddings, embed_dim)

    def embed_alpha(self, x: Tensor) -> Tensor:
        return self.table_alpha(x)

    def embed_beta(self, x: Tensor) -> Tensor:
        return self.table_beta(x)

    def embed_gamma(self, x: Tensor) -> Tensor:
        return self.table_gamma(x)


class QuadRotaryEmbedding(QuadEmbedding):
    """Quad embedding with four rotary configurations (different theta)."""

    def __init__(
        self,
        name: str,
        embed_dim: int,
        thetas: Tuple[float, float, float, float] = (10000.0, 5000.0, 2500.0, 1250.0),
        modulation: str = 'quaternion',
        augmentation: str = 'simplex',
    ):
        super().__init__(name, embed_dim, modulation, augmentation)

        for theta, label in zip(thetas, ['w', 'x', 'y', 'z']):
            inv_freq = 1.0 / (theta ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
            self.register_buffer(f'inv_freq_{label}', inv_freq)

    def _rotary(self, seq_len: int, inv_freq: Tensor, use_cos: bool = True) -> Tensor:
        """Generate rotary embedding of shape [L, D]."""
        t = torch.arange(seq_len, device=inv_freq.device)
        freqs = torch.outer(t, inv_freq)  # [L, D//2]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [L, D]
        return freqs.cos() if use_cos else freqs.sin()

    def embed_w(self, seq_len: int) -> Tensor:
        return self._rotary(seq_len, self.inv_freq_w, use_cos=True)

    def embed_x(self, seq_len: int) -> Tensor:
        return self._rotary(seq_len, self.inv_freq_x, use_cos=False)

    def embed_y(self, seq_len: int) -> Tensor:
        return self._rotary(seq_len, self.inv_freq_y, use_cos=True)

    def embed_z(self, seq_len: int) -> Tensor:
        return self._rotary(seq_len, self.inv_freq_z, use_cos=False)


class QuadLookupEmbedding(QuadEmbedding):
    """Quad embedding with four lookup tables."""

    def __init__(
        self,
        name: str,
        num_embeddings: int,
        embed_dim: int,
        modulation: str = 'quaternion',
        augmentation: str = 'simplex',
    ):
        super().__init__(name, embed_dim, modulation, augmentation)
        self.table_w = nn.Embedding(num_embeddings, embed_dim)
        self.table_x = nn.Embedding(num_embeddings, embed_dim)
        self.table_y = nn.Embedding(num_embeddings, embed_dim)
        self.table_z = nn.Embedding(num_embeddings, embed_dim)

    def embed_w(self, x: Tensor) -> Tensor:
        return self.table_w(x)

    def embed_x(self, x: Tensor) -> Tensor:
        return self.table_x(x)

    def embed_y(self, x: Tensor) -> Tensor:
        return self.table_y(x)

    def embed_z(self, x: Tensor) -> Tensor:
        return self.table_z(x)


# =============================================================================
# TEST
# =============================================================================

# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':

    def section(title):
        print(f"\n{'='*60}\n  {title}\n{'='*60}")

    # -------------------------------------------------------------------------
    section("BASE EMBEDDING (Lookup)")
    # -------------------------------------------------------------------------

    lookup = LookupEmbedding('lookup', num_embeddings=1000, embed_dim=64)
    print(f"Dimension: {lookup.dimension}")

    x = torch.randint(0, 1000, (4, 16))
    emb = lookup(x)
    print(f"Input: {x.shape} -> Output: {emb.shape}")

    # Modulate/augment
    mod = lookup.modulate(emb, torch.randn_like(emb))
    aug = lookup.augment(emb, torch.randn_like(emb))
    print(f"Modulated: {mod.shape}, Augmented: {aug.shape}")

    # -------------------------------------------------------------------------
    section("SINUSOIDAL EMBEDDING")
    # -------------------------------------------------------------------------

    sin_emb = SinusoidalEmbedding('sin', embed_dim=64)
    print(f"Dimension: {sin_emb.dimension}")

    positions = torch.arange(128)
    emb = sin_emb(positions)
    print(f"Input: {positions.shape} -> Output: {emb.shape}")

    # -------------------------------------------------------------------------
    section("ROTARY EMBEDDING")
    # -------------------------------------------------------------------------

    rope = RotaryEmbedding('rope', embed_dim=64, theta=10000.0)
    print(f"Dimension: {rope.dimension}")

    cos, sin = rope(128)
    print(f"cos: {cos.shape}, sin: {sin.shape}")

    q = torch.randn(4, 8, 128, 64)
    k = torch.randn(4, 8, 128, 64)
    q_rot, k_rot = rope.apply_rotary(q, k, cos, sin)
    print(f"Q rotated: {q_rot.shape}")

    # -------------------------------------------------------------------------
    section("DUAL EMBEDDING (Lookup)")
    # -------------------------------------------------------------------------

    dual = DualLookupEmbedding('dual', num_embeddings=1000, embed_dim=64)
    print(f"Dimension: {dual.dimension}")

    primary, secondary = dual(x)
    print(f"Primary: {primary.shape}, Secondary: {secondary.shape}")

    for mode in ['modulate', 'augment', 'both']:
        combined = dual.embed_combined(x, mode=mode)
        print(f"Combined ({mode}): {combined.shape}")

    # -------------------------------------------------------------------------
    section("DUAL EMBEDDING (Content + Position)")
    # -------------------------------------------------------------------------

    dual_cp = DualContentPositionEmbedding('dual_cp', num_embeddings=1000, embed_dim=64)
    print(f"Dimension: {dual_cp.dimension}")

    content, position = dual_cp(x)
    print(f"Content: {content.shape}, Position: {position.shape}")

    combined = dual_cp.embed_combined(x, mode='augment')
    print(f"Combined: {combined.shape}")

    # -------------------------------------------------------------------------
    section("TRI EMBEDDING (Frequency Bands)")
    # -------------------------------------------------------------------------

    tri = TriFrequencyEmbedding('tri', embed_dim=64)
    print(f"Dimension: {tri.dimension}")
    print(f"Barycentric weights: {tri.barycentric_weights.tolist()}")

    alpha, beta, gamma = tri(positions)
    print(f"Alpha: {alpha.shape}, Beta: {beta.shape}, Gamma: {gamma.shape}")

    for mode in ['modulate', 'augment', 'both']:
        combined = tri.embed_combined(positions, mode=mode)
        print(f"Combined ({mode}): {combined.shape}")

    # -------------------------------------------------------------------------
    section("TRI EMBEDDING (Lookup)")
    # -------------------------------------------------------------------------

    tri_lookup = TriLookupEmbedding('tri_lookup', num_embeddings=1000, embed_dim=64)
    print(f"Dimension: {tri_lookup.dimension}")

    a, b, c = tri_lookup(x)
    print(f"Alpha: {a.shape}, Beta: {b.shape}, Gamma: {c.shape}")

    for mode in ['modulate', 'augment', 'both']:
        combined = tri_lookup.embed_combined(x, mode=mode)
        print(f"Combined ({mode}): {combined.shape}")

    # -------------------------------------------------------------------------
    section("QUAD EMBEDDING (Multi-theta Rotary)")
    # -------------------------------------------------------------------------

    quad = QuadRotaryEmbedding('quad', embed_dim=64)
    print(f"Dimension: {quad.dimension}")
    print(f"Simplex weights: {quad.simplex_weights.tolist()}")

    seq_len = 128
    w, qx, y, z = quad(seq_len)
    print(f"W: {w.shape}, X: {qx.shape}, Y: {y.shape}, Z: {z.shape}")

    for mode in ['modulate', 'augment', 'both']:
        combined = quad.embed_combined(seq_len, mode=mode)
        print(f"Combined ({mode}): {combined.shape}")

    # -------------------------------------------------------------------------
    section("QUAD EMBEDDING (Lookup)")
    # -------------------------------------------------------------------------

    quad_lookup = QuadLookupEmbedding('quad_lookup', num_embeddings=1000, embed_dim=64)
    print(f"Dimension: {quad_lookup.dimension}")

    qw, qx, qy, qz = quad_lookup(x)
    print(f"W: {qw.shape}, X: {qx.shape}, Y: {qy.shape}, Z: {qz.shape}")

    for mode in ['modulate', 'augment', 'both']:
        combined = quad_lookup.embed_combined(x, mode=mode)
        print(f"Combined ({mode}): {combined.shape}")

    # -------------------------------------------------------------------------
    section("MODULATION MODES")
    # -------------------------------------------------------------------------

    x_small = torch.randint(0, 100, (4, 8))

    print("\nDual modulation modes:")
    for mod in ['multiply', 'gate', 'complex']:
        d = DualLookupEmbedding('test', 100, 64, modulation=mod)
        out = d.embed_combined(x_small, mode='modulate')
        print(f"  {mod}: {out.shape}")

    print("\nTri modulation modes:")
    for mod in ['geometric', 'hadamard', 'bilinear']:
        t = TriFrequencyEmbedding('test', 64, modulation=mod)
        out = t.embed_combined(positions[:32], mode='modulate')
        print(f"  {mod}: {out.shape}")

    print("\nQuad modulation modes:")
    for mod in ['quaternion', 'hadamard', 'pairwise']:
        q = QuadRotaryEmbedding('test', 64, modulation=mod)
        out = q.embed_combined(32, mode='modulate')
        print(f"  {mod}: {out.shape}")

    # -------------------------------------------------------------------------
    section("AUGMENTATION MODES")
    # -------------------------------------------------------------------------

    print("\nDual augmentation modes:")
    for aug in ['add', 'lerp', 'residual']:
        d = DualLookupEmbedding('test', 100, 64, augmentation=aug)
        out = d.embed_combined(x_small, mode='augment')
        print(f"  {aug}: {out.shape}")

    print("\nTri augmentation modes:")
    for aug in ['barycentric', 'sum', 'attention']:
        t = TriFrequencyEmbedding('test', 64, augmentation=aug)
        out = t.embed_combined(positions[:32], mode='augment')
        print(f"  {aug}: {out.shape}")

    print("\nQuad augmentation modes:")
    for aug in ['simplex', 'sum', 'hierarchical']:
        q = QuadRotaryEmbedding('test', 64, augmentation=aug)
        out = q.embed_combined(32, mode='augment')
        print(f"  {aug}: {out.shape}")

    # -------------------------------------------------------------------------
    section("DEVICE + COMPILE")
    # -------------------------------------------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dual = DualLookupEmbedding('gpu', 1000, 64).to(device)
    x_gpu = torch.randint(0, 1000, (4, 16), device=device)

    out = dual.embed_combined(x_gpu)
    print(f"Device: {out.device}")

    compiled = torch.compile(dual)
    out2 = compiled.embed_combined(x_gpu)
    print(f"Compiled match: {torch.allclose(out, out2)}")

    # -------------------------------------------------------------------------
    section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nEmbedding hierarchy ready:")
    print("  BaseEmbedding  - dimension=1, simple modulate/augment")
    print("  DualEmbedding  - dimension=2, binary opposition")
    print("  TriEmbedding   - dimension=3, barycentric/triangular")
    print("  QuadEmbedding  - dimension=4, quaternionic/crystalline")
    print("\nConcrete implementations:")
    print("  LookupEmbedding, SinusoidalEmbedding, RotaryEmbedding")
    print("  DualLookupEmbedding, DualContentPositionEmbedding")
    print("  TriFrequencyEmbedding, TriLookupEmbedding")
    print("  QuadRotaryEmbedding, QuadLookupEmbedding")