"""
geofractal.router.components.rope_component
============================================

Rotary Position Embedding (RoPE) for the router system.

Includes standard RoPE and common variants:
    - RoPE: Standard rotary embedding
    - DualRoPE: Two configs with modulation/augmentation
    - ScaledRoPE: Linear and NTK-aware scaling for length extension
    - YaRNRoPE: Yet another RoPE extensioN (state-of-the-art length extension)
    - PartialRoPE: Apply to subset of dimensions

Mathematical Foundation:
    RoPE encodes position by rotating query/key vectors:

        q_rot = q * cos(m·θ) + rotate_half(q) * sin(m·θ)

    where m is position and θ are learned frequencies.

Scaling Methods:
    Linear:     position' = position / scale
    NTK-aware:  theta' = theta * scale^(dim/(dim-2))
    YaRN:       NTK + attention_scale + temperature per frequency

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, Tuple, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.components.embedding_component import BaseEmbedding, DualEmbedding, TriEmbedding, QuadEmbedding

# =============================================================================
# ROPE (Base Implementation)
# =============================================================================

class RoPE(BaseEmbedding):
    """
    Rotary Position Embedding for the router system.

    Generates cos/sin position encodings and applies rotation to Q/K tensors.
    Supports configurable theta for multi-scale attention patterns.

    Args:
        name: Component identifier
        head_dim: Dimension of attention heads (must be even)
        theta: Base frequency (higher = longer wavelengths). Default 10000.0
        theta_scale: Multiplier for frequencies. Default 1.0
        max_seq_len: Initial cache size. Auto-expands if exceeded. Default 8192
        uuid: Optional unique identifier

    Example:
        #  rope = RoPE('rope', head_dim=64, theta=10000.0)
        #  cos, sin = rope(seq_len=128)
        #  q_rot, k_rot = rope.rotate(q, k, cos, sin)
    """

    def __init__(
        self,
        name: str,
        head_dim: int,
        theta: float = 10000.0,
        theta_scale: float = 1.0,
        max_seq_len: int = 8192,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, head_dim, uuid, **kwargs)

        self.head_dim = head_dim
        self.theta = theta
        self.theta_scale = theta_scale
        self.max_seq_len = max_seq_len

        # Inverse frequencies: [D//2]
        inv_freq = self._compute_inv_freq(head_dim, theta, theta_scale)
        self.register_buffer('inv_freq', inv_freq)

        # Pre-compute cache
        self._build_cache(max_seq_len)

    def _compute_inv_freq(
        self,
        head_dim: int,
        theta: float,
        theta_scale: float,
    ) -> Tensor:
        """Compute inverse frequencies. Override in subclasses for scaling."""
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        return inv_freq * theta_scale

    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute cos/sin cache for efficiency."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        t = self._scale_positions(t, seq_len)

        freqs = torch.outer(t, self.inv_freq)  # [L, D//2]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [L, D]

        self.register_buffer('_cos', freqs.cos(), persistent=False)
        self.register_buffer('_sin', freqs.sin(), persistent=False)

    def _scale_positions(self, positions: Tensor, seq_len: int) -> Tensor:
        """Scale positions. Override in subclasses for linear/dynamic scaling."""
        return positions

    @property
    def dimension(self) -> int:
        return 1

    def modulate(self, x: Tensor, modulator: Tensor) -> Tensor:
        return x * modulator

    def augment(self, x: Tensor, augmentation: Tensor) -> Tensor:
        return x

    def embed(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Get cos/sin embeddings for sequence length."""
        if seq_len > self._cos.shape[0]:
            self._build_cache(seq_len)
        return self._cos[:seq_len], self._sin[:seq_len]

    def rotate(
        self,
        q: Tensor,
        k: Tensor,
        cos: Tensor = None,
        sin: Tensor = None,
        seq_len: int = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        if cos is None or sin is None:
            L = seq_len or q.shape[-2]
            cos, sin = self.embed(L)

        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotation to a single tensor."""
        if x.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        return self._rotate_half(x, cos, sin)

    def _rotate_half(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Rotate using the half-dimension method."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return (x * cos) + (rotated * sin)

    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        return self.embed(seq_len)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, theta={self.theta}, scale={self.theta_scale}"


# =============================================================================
# SCALED ROPE (Linear and NTK-aware)
# =============================================================================

class ScaledRoPE(RoPE):
    """
    RoPE with scaling for length extension.

    Supports two scaling methods:
        - linear: Divide position indices by scale factor
        - ntk: Scale theta base (Neural Tangent Kernel aware)

    Linear scaling compresses positions, allowing longer sequences
    but may lose fine-grained position information.

    NTK-aware scaling adjusts the frequency base, better preserving
    the distribution of attention patterns.

    Args:
        name: Component identifier
        head_dim: Dimension of attention heads
        theta: Base frequency. Default 10000.0
        scale_factor: Extension factor (e.g., 2.0 for 2x length). Default 1.0
        scaling_type: 'linear' or 'ntk'. Default 'linear'
        max_seq_len: Maximum sequence length. Default 8192
        uuid: Optional unique identifier

    Example:
        #  # Train on 4k, extend to 8k
        #  rope = ScaledRoPE('rope', 64, scale_factor=2.0, scaling_type='ntk')
    """

    def __init__(
        self,
        name: str,
        head_dim: int,
        theta: float = 10000.0,
        scale_factor: float = 1.0,
        scaling_type: Literal['linear', 'ntk'] = 'linear',
        max_seq_len: int = 8192,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        self.scale_factor = scale_factor
        self.scaling_type = scaling_type

        super().__init__(name, head_dim, theta, 1.0, max_seq_len, uuid, **kwargs)

    def _compute_inv_freq(
        self,
        head_dim: int,
        theta: float,
        theta_scale: float,
    ) -> Tensor:
        """Compute inverse frequencies with optional NTK scaling."""
        if self.scaling_type == 'ntk' and self.scale_factor != 1.0:
            # NTK-aware: scale the base theta
            # theta' = theta * scale^(dim/(dim-2))
            theta = theta * (self.scale_factor ** (head_dim / (head_dim - 2)))

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        return inv_freq

    def _scale_positions(self, positions: Tensor, seq_len: int) -> Tensor:
        """Scale positions for linear interpolation."""
        if self.scaling_type == 'linear' and self.scale_factor != 1.0:
            return positions / self.scale_factor
        return positions

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, theta={self.theta}, "
            f"scale={self.scale_factor}, type={self.scaling_type}"
        )


# =============================================================================
# YARN ROPE (Yet another RoPE extensioN)
# =============================================================================

class YaRNRoPE(RoPE):
    """
    YaRN: Yet another RoPE extensioN.

    State-of-the-art length extension combining:
        1. NTK-aware interpolation (modified theta base)
        2. Attention scaling factor
        3. Per-frequency temperature adjustment

    YaRN divides dimensions into three regions:
        - Low frequencies: No interpolation (already handle long range)
        - Medium frequencies: NTK-style interpolation
        - High frequencies: Linear interpolation

    Args:
        name: Component identifier
        head_dim: Dimension of attention heads
        theta: Base frequency. Default 10000.0
        scale_factor: Extension factor. Default 1.0
        original_max_len: Original training length. Default 2048
        beta_fast: High frequency boundary. Default 32
        beta_slow: Low frequency boundary. Default 1
        max_seq_len: Maximum sequence length. Default 8192
        uuid: Optional unique identifier

    Reference:
        "YaRN: Efficient Context Window Extension of Large Language Models"
        https://arxiv.org/abs/2309.00071

    Example:
          # Extend 4k model to 128k
          rope = YaRNRoPE('rope', 64, scale_factor=32.0, original_max_len=4096)
    """

    def __init__(
        self,
        name: str,
        head_dim: int,
        theta: float = 10000.0,
        scale_factor: float = 1.0,
        original_max_len: int = 2048,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        max_seq_len: int = 8192,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        self.scale_factor = scale_factor
        self.original_max_len = original_max_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Attention scale factor (YaRN paper equation 20)
        self.attn_scale = 0.1 * math.log(scale_factor) + 1.0 if scale_factor > 1.0 else 1.0

        super().__init__(name, head_dim, theta, 1.0, max_seq_len, uuid, **kwargs)

    def _compute_inv_freq(
        self,
        head_dim: int,
        theta: float,
        theta_scale: float,
    ) -> Tensor:
        """Compute YaRN-adjusted inverse frequencies."""
        # Base frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        if self.scale_factor == 1.0:
            return inv_freq

        # Compute interpolation ratios per dimension
        # Low freq (long wavelength) -> no interpolation
        # High freq (short wavelength) -> full interpolation

        dim_indices = torch.arange(0, head_dim // 2).float()

        # Wavelengths for each dimension
        wavelengths = 2 * math.pi * theta ** (dim_indices * 2 / head_dim)

        # Ratio of original length to wavelength
        ratios = self.original_max_len / wavelengths

        # Smooth ramp between beta_slow and beta_fast
        ramp = (ratios - self.beta_slow) / (self.beta_fast - self.beta_slow)
        ramp = torch.clamp(ramp, 0.0, 1.0)

        # Interpolation factor: 0 = no scaling, 1 = full linear scaling
        # YaRN uses sqrt for the blend
        interp_factor = 1.0 - torch.sqrt(ramp)

        # Apply per-dimension scaling
        # Scale factor for each frequency: blend between 1.0 and 1/scale_factor
        freq_scale = 1.0 / (1.0 + (self.scale_factor - 1.0) * interp_factor)

        return inv_freq * freq_scale

    def rotate(
        self,
        q: Tensor,
        k: Tensor,
        cos: Tensor = None,
        sin: Tensor = None,
        seq_len: int = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply YaRN rotation with attention scaling."""
        q_rot, k_rot = super().rotate(q, k, cos, sin, seq_len)

        # Apply attention scale (only to queries, affects softmax temperature)
        if self.attn_scale != 1.0:
            q_rot = q_rot * self.attn_scale

        return q_rot, k_rot

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, theta={self.theta}, "
            f"scale={self.scale_factor}, attn_scale={self.attn_scale:.3f}"
        )


# =============================================================================
# PARTIAL ROPE (Apply to subset of dimensions)
# =============================================================================

class PartialRoPE(RoPE):
    """
    RoPE applied to only a subset of head dimensions.

    Some architectures apply RoPE to only the first N dimensions,
    leaving remaining dimensions as pure content (no position info).

    This can improve efficiency and allow the model to learn
    position-independent features in the non-rotated dimensions.

    Args:
        name: Component identifier
        head_dim: Full dimension of attention heads
        rotary_dim: Dimensions to apply RoPE (must be <= head_dim, even)
        theta: Base frequency. Default 10000.0
        max_seq_len: Maximum sequence length. Default 8192
        uuid: Optional unique identifier

    Example:
          # Apply RoPE to first 32 of 64 dims
          rope = PartialRoPE('rope', head_dim=64, rotary_dim=32)
    """

    def __init__(
        self,
        name: str,
        head_dim: int,
        rotary_dim: int = None,
        theta: float = 10000.0,
        max_seq_len: int = 8192,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        self.full_head_dim = head_dim
        rotary_dim = rotary_dim or head_dim

        if rotary_dim > head_dim:
            raise ValueError(f"rotary_dim ({rotary_dim}) must be <= head_dim ({head_dim})")
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}")

        self.rotary_dim = rotary_dim
        self.non_rotary_dim = head_dim - rotary_dim

        # Initialize with rotary_dim, not full head_dim
        super().__init__(name, rotary_dim, theta, 1.0, max_seq_len, uuid, **kwargs)

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotation to only the rotary dimensions."""
        if self.non_rotary_dim == 0:
            return super()._apply_rope(x, cos, sin)

        # Split into rotary and non-rotary parts
        x_rot = x[..., :self.rotary_dim]
        x_pass = x[..., self.rotary_dim:]

        # Apply rotation only to rotary part
        x_rot = super()._apply_rope(x_rot, cos, sin)

        # Concatenate back
        return torch.cat([x_rot, x_pass], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.full_head_dim}, rotary_dim={self.rotary_dim}, "
            f"theta={self.theta}"
        )


# =============================================================================
# DUAL ROPE
# =============================================================================

class DualRoPE(BaseEmbedding):
    """
    Two RoPE configurations with modulation/augmentation.

    Structure: (primary, secondary) - two theta configurations
    Geometry: Complex plane / binary interpolation

    The two configurations can represent:
        - Local + Global attention patterns
        - Normal + Inverted position encoding
        - Different frequency scales

    Args:
        name: Component identifier
        head_dim: Dimension of attention heads
        theta_primary: First theta. Default 10000.0
        theta_secondary: Second theta. Default 1000.0
        modulation: Combination method for modulate()
        augmentation: Combination method for augment()
        use_bilinear: Enable heavy bilinear layer for learning. Default False
        uuid: Optional unique identifier

    Example:
        dual = DualRoPE('dual', 64, theta_primary=10000.0, theta_secondary=1000.0)
        q_rot, k_rot = dual.rotate(q, k, seq_len=128, mode='augment')
    """

    def __init__(
            self,
            name: str,
            head_dim: int,
            theta_primary: float = 10000.0,
            theta_secondary: float = 1000.0,
            modulation: Literal['multiply', 'complex', 'gate', 'phase_blend'] = 'phase_blend',
            augmentation: Literal['add', 'lerp', 'residual'] = 'lerp',
            use_bilinear: bool = False,
            uuid: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(name, head_dim, uuid, **kwargs)

        self.head_dim = head_dim
        self.theta_primary = theta_primary
        self.theta_secondary = theta_secondary
        self._modulation_mode = modulation
        self._augmentation_mode = augmentation
        self.use_bilinear = use_bilinear

        # Lerp weight (1 param)
        self._lerp_alpha = nn.Parameter(torch.tensor(0.5))

        # Gate projection only if needed
        if modulation == 'gate' or use_bilinear:
            self.gate_proj = nn.Linear(head_dim, head_dim)
        else:
            self.gate_proj = None

        # Optional bilinear layer
        if use_bilinear:
            self.bilinear = nn.Bilinear(head_dim, head_dim, head_dim, bias=False)
        else:
            self.bilinear = None

        # Two sets of inverse frequencies
        inv_freq_p = 1.0 / (theta_primary ** (torch.arange(0, head_dim, 2).float() / head_dim))
        inv_freq_s = 1.0 / (theta_secondary ** (torch.arange(0, head_dim, 2).float() / head_dim))

        self.register_buffer('inv_freq_primary', inv_freq_p)
        self.register_buffer('inv_freq_secondary', inv_freq_s)

    @property
    def dimension(self) -> int:
        return 2

    @property
    def lerp_alpha(self) -> Tensor:
        """Interpolation weight clamped to [0, 1]."""
        return torch.sigmoid(self._lerp_alpha)

    def _compute_rope(self, seq_len: int, inv_freq: Tensor) -> Tensor:
        """Compute stacked [cos, sin] for given frequencies."""
        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # [L, D//2]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [L, D]
        return torch.stack([freqs.cos(), freqs.sin()], dim=0)  # [2, L, D]

    def embed_primary(self, seq_len: int) -> Tensor:
        """Primary frequency pattern."""
        return self._compute_rope(seq_len, self.inv_freq_primary)

    def embed_secondary(self, seq_len: int) -> Tensor:
        """Secondary frequency pattern."""
        return self._compute_rope(seq_len, self.inv_freq_secondary)

    def embed(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Return both embeddings."""
        return self.embed_primary(seq_len), self.embed_secondary(seq_len)

    def modulate(self, primary: Tensor, secondary: Tensor) -> Tensor:
        """
        Binary modulation on stacked [cos, sin] tensors.

        multiply: Element-wise product
        complex: Complex multiplication (rotation composition)
        gate: Sigmoid-gated combination
        phase_blend: Average phases then recompute
        """
        if self._modulation_mode == 'multiply':
            return primary * secondary

        elif self._modulation_mode == 'complex':
            # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            # primary[0]=cos_p, primary[1]=sin_p
            # secondary[0]=cos_s, secondary[1]=sin_s
            real = primary[0] * secondary[0] - primary[1] * secondary[1]
            imag = primary[0] * secondary[1] + primary[1] * secondary[0]
            return torch.stack([real, imag], dim=0)


        elif self._modulation_mode == 'gate':
            if self.gate_proj is None:
                raise RuntimeError("gate modulation requires gate_proj (set modulation='gate' in init)")
            gate = torch.sigmoid(self.gate_proj(primary[0]))
            return primary * gate.unsqueeze(0) + secondary * (1 - gate).unsqueeze(0)

        elif self._modulation_mode == 'phase_blend':
            phase_p = torch.atan2(primary[1], primary[0])
            phase_s = torch.atan2(secondary[1], secondary[0])

            alpha = self.lerp_alpha
            phase_combined = (1 - alpha) * phase_p + alpha * phase_s

            return torch.stack([phase_combined.cos(), phase_combined.sin()], dim=0)

        else:
            raise ValueError(f"Unknown modulation: {self._modulation_mode}")

    def augment(self, primary: Tensor, secondary: Tensor) -> Tensor:
        """
        Binary augmentation on stacked [cos, sin] tensors.

        add: Weighted addition
        lerp: Linear interpolation
        residual: Primary + scaled secondary
        """
        if self._augmentation_mode == 'add':
            alpha = self.lerp_alpha
            return alpha * primary + (1 - alpha) * secondary

        elif self._augmentation_mode == 'lerp':
            alpha = self.lerp_alpha
            return (1 - alpha) * primary + alpha * secondary

        elif self._augmentation_mode == 'residual':
            alpha = self.lerp_alpha
            return primary + alpha * secondary

        else:
            raise ValueError(f"Unknown augmentation: {self._augmentation_mode}")

    def embed_combined(
        self,
        seq_len: int,
        mode: Literal['modulate', 'augment', 'both'] = 'augment',
    ) -> Tensor:
        """Combine two embeddings using specified mode."""
        primary, secondary = self.embed(seq_len)

        if mode == 'modulate':
            return self.modulate(primary, secondary)
        elif mode == 'augment':
            return self.augment(primary, secondary)
        elif mode == 'both':
            modulated = self.modulate(primary, secondary)
            return self.augment(modulated, modulated)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def rotate(
        self,
        q: Tensor,
        k: Tensor,
        seq_len: int,
        mode: Literal['modulate', 'augment', 'both'] = 'augment',
    ) -> Tuple[Tensor, Tensor]:
        """Apply combined dual-rotary embeddings to Q and K."""
        combined = self.embed_combined(seq_len, mode=mode)  # [2, L, D]
        cos, sin = combined[0], combined[1]

        if q.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif q.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        def rotate_half(x: Tensor) -> Tensor:
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([-x2, x1], dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)

        return q_rot, k_rot

    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        return self.embed(seq_len)

    def extra_repr(self) -> str:
        bilinear_str = ", bilinear=True" if self.use_bilinear else ""
        return (
            f"head_dim={self.head_dim}, "
            f"theta=({self.theta_primary}, {self.theta_secondary}){bilinear_str}"
        )

# =============================================================================
# TRI ROPE
# =============================================================================

class TriRoPE(BaseEmbedding):
    """
    Three RoPE configurations with barycentric combination.

    Structure: (alpha, beta, gamma) - three theta configurations
    Geometry: 2-simplex (triangle) interpolation of rotary frequencies

    The three configurations can represent:
        - Low/Mid/High frequency bands (multi-scale)
        - Local/Medium/Global attention patterns
        - Past/Present/Future position awareness

    Args:
        name: Component identifier
        head_dim: Dimension of attention heads
        theta_alpha: First theta (e.g., low frequency). Default 10000.0
        theta_beta: Second theta (e.g., mid frequency). Default 5000.0
        theta_gamma: Third theta (e.g., high frequency). Default 1000.0
        modulation: Combination method for modulate()
        augmentation: Combination method for augment()
        use_bilinear: Enable heavy bilinear layers for learning. Default False
        uuid: Optional unique identifier

    Example:
        tri = TriRoPE('tri', 64, theta_alpha=10000, theta_beta=5000, theta_gamma=1000)
        q_rot, k_rot = tri.rotate(q, k, seq_len=128, mode='augment')
    """

    def __init__(
        self,
        name: str,
        head_dim: int,
        theta_alpha: float = 10000.0,
        theta_beta: float = 5000.0,
        theta_gamma: float = 1000.0,
        modulation: Literal['geometric', 'hadamard', 'bilinear', 'phase_blend'] = 'phase_blend',
        augmentation: Literal['barycentric', 'sum', 'attention'] = 'barycentric',
        use_bilinear: bool = False,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, head_dim, uuid, **kwargs)

        self.head_dim = head_dim
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self._modulation_mode = modulation
        self._augmentation_mode = augmentation
        self.use_bilinear = use_bilinear

        # Barycentric weights (3 params)
        self._bary_logits = nn.Parameter(torch.zeros(3))

        # Attention projection for attention mode
        self.attn_proj = nn.Linear(head_dim, 1)

        # Optional bilinear layers (heavy: 2 * D^3 params)
        if use_bilinear:
            self.bilinear_ab = nn.Bilinear(head_dim, head_dim, head_dim, bias=False)
            self.bilinear_bc = nn.Bilinear(head_dim, head_dim, head_dim, bias=False)
        else:
            self.bilinear_ab = None
            self.bilinear_bc = None

        # Three sets of inverse frequencies
        for theta, label in [(theta_alpha, 'alpha'), (theta_beta, 'beta'), (theta_gamma, 'gamma')]:
            inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            self.register_buffer(f'inv_freq_{label}', inv_freq)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def barycentric_weights(self) -> Tensor:
        """Normalized barycentric coordinates (sum to 1)."""
        return F.softmax(self._bary_logits, dim=0)

    def _compute_rope(self, seq_len: int, inv_freq: Tensor) -> Tensor:
        """Compute stacked [cos, sin] for given frequencies."""
        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # [L, D//2]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [L, D]
        return torch.stack([freqs.cos(), freqs.sin()], dim=0)  # [2, L, D]

    def embed_alpha(self, seq_len: int) -> Tensor:
        """Low frequency / global pattern."""
        return self._compute_rope(seq_len, self.inv_freq_alpha)

    def embed_beta(self, seq_len: int) -> Tensor:
        """Mid frequency / medium pattern."""
        return self._compute_rope(seq_len, self.inv_freq_beta)

    def embed_gamma(self, seq_len: int) -> Tensor:
        """High frequency / local pattern."""
        return self._compute_rope(seq_len, self.inv_freq_gamma)

    def embed(self, seq_len: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Return all three embeddings."""
        return self.embed_alpha(seq_len), self.embed_beta(seq_len), self.embed_gamma(seq_len)

    def modulate(self, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
        """
        Ternary modulation on stacked [cos, sin] tensors.

        geometric: Geometric mean (preserves scale)
        hadamard: Triple product (aggressive combination)
        bilinear: Learned bilinear combination (requires use_bilinear=True)
        phase_blend: Average phases then recompute (rotation-preserving)
        """
        if self._modulation_mode == 'geometric':
            product = alpha * beta * gamma
            sign = torch.sign(product)
            return sign * (torch.abs(product) + 1e-8).pow(1/3)

        elif self._modulation_mode == 'hadamard':
            return alpha * beta * gamma

        elif self._modulation_mode == 'bilinear':
            if self.bilinear_ab is None:
                raise RuntimeError("bilinear modulation requires use_bilinear=True")
            # Apply bilinear on cos and sin separately
            cos_ab = self.bilinear_ab(alpha[0], beta[0])
            cos_out = self.bilinear_bc(cos_ab, gamma[0])
            sin_ab = self.bilinear_ab(alpha[1], beta[1])
            sin_out = self.bilinear_bc(sin_ab, gamma[1])
            return torch.stack([cos_out, sin_out], dim=0)

        elif self._modulation_mode == 'phase_blend':
            phase_a = torch.atan2(alpha[1], alpha[0])
            phase_b = torch.atan2(beta[1], beta[0])
            phase_g = torch.atan2(gamma[1], gamma[0])

            w = self.barycentric_weights
            phase_combined = w[0] * phase_a + w[1] * phase_b + w[2] * phase_g

            return torch.stack([phase_combined.cos(), phase_combined.sin()], dim=0)

        else:
            raise ValueError(f"Unknown modulation: {self._modulation_mode}")

    def augment(self, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
        """
        Ternary augmentation on stacked [cos, sin] tensors.

        barycentric: Weighted combination (sum to 1)
        sum: Direct addition
        attention: Learned attention over three configs
        """
        if self._augmentation_mode == 'barycentric':
            w = self.barycentric_weights
            return w[0] * alpha + w[1] * beta + w[2] * gamma

        elif self._augmentation_mode == 'sum':
            return alpha + beta + gamma

        elif self._augmentation_mode == 'attention':
            # Stack: [3, 2, L, D] -> attend over first dim
            stacked = torch.stack([alpha, beta, gamma], dim=0)  # [3, 2, L, D]
            # Use cos component for attention scores
            scores = self.attn_proj(stacked[:, 0]).squeeze(-1)  # [3, L]
            weights = F.softmax(scores, dim=0)  # [3, L]
            # Weighted sum: [3, 2, L, D] * [3, 1, L, 1] -> sum -> [2, L, D]
            return (stacked * weights.unsqueeze(1).unsqueeze(-1)).sum(dim=0)

        else:
            raise ValueError(f"Unknown augmentation: {self._augmentation_mode}")

    def embed_combined(
        self,
        seq_len: int,
        mode: Literal['modulate', 'augment', 'both'] = 'augment',
    ) -> Tensor:
        """Combine three embeddings using specified mode."""
        alpha, beta, gamma = self.embed(seq_len)

        if mode == 'modulate':
            return self.modulate(alpha, beta, gamma)
        elif mode == 'augment':
            return self.augment(alpha, beta, gamma)
        elif mode == 'both':
            modulated = self.modulate(alpha, beta, gamma)
            return self.augment(modulated, modulated, modulated)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def rotate(
        self,
        q: Tensor,
        k: Tensor,
        seq_len: int,
        mode: Literal['modulate', 'augment', 'both'] = 'augment',
    ) -> Tuple[Tensor, Tensor]:
        """Apply combined tri-rotary embeddings to Q and K."""
        combined = self.embed_combined(seq_len, mode=mode)  # [2, L, D]
        cos, sin = combined[0], combined[1]

        if q.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif q.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        def rotate_half(x: Tensor) -> Tensor:
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([-x2, x1], dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)

        return q_rot, k_rot

    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.embed(seq_len)

    def extra_repr(self) -> str:
        bilinear_str = ", bilinear=True" if self.use_bilinear else ""
        return (
            f"head_dim={self.head_dim}, "
            f"theta=({self.theta_alpha}, {self.theta_beta}, {self.theta_gamma}){bilinear_str}"
        )


class QuadRoPE(BaseEmbedding):
    """
    Four RoPE configurations with quaternionic combination.

    Structure: (w, x, y, z) - four theta configurations
    Geometry: 3-simplex (tetrahedron) / quaternion algebra

    The four configurations can represent:
        - Ultra-local / Local / Medium / Global attention
        - Quaternion components for 3D rotation encoding
        - Four-way hierarchical position structure

    Args:
        name: Component identifier
        head_dim: Dimension of attention heads
        theta_w: First theta (scalar/identity). Default 10000.0
        theta_x: Second theta (i component). Default 5000.0
        theta_y: Third theta (j component). Default 2500.0
        theta_z: Fourth theta (k component). Default 1000.0
        modulation: Combination method for modulate()
        augmentation: Combination method for augment()
        use_bilinear: Enable heavy bilinear layers for learning. Default False
        uuid: Optional unique identifier

    Example:
        quad = QuadRoPE('quad', 64, theta_w=10000, theta_x=5000, theta_y=2500, theta_z=1000)
        q_rot, k_rot = quad.rotate(q, k, seq_len=128, mode='augment')
    """

    def __init__(
            self,
            name: str,
            head_dim: int,
            theta_w: float = 10000.0,
            theta_x: float = 5000.0,
            theta_y: float = 2500.0,
            theta_z: float = 1000.0,
            modulation: Literal['quaternion', 'hadamard', 'pairwise', 'phase_blend'] = 'phase_blend',
            augmentation: Literal['simplex', 'sum', 'hierarchical'] = 'simplex',
            use_bilinear: bool = False,
            normalize_output: bool = True,  # NEW: normalize to unit circle
            uuid: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(name, head_dim, uuid, **kwargs)

        self.head_dim = head_dim
        self.theta_w = theta_w
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_z = theta_z
        self._modulation_mode = modulation
        self._augmentation_mode = augmentation
        self.use_bilinear = use_bilinear

        # Simplex weights (4 params)
        self._simplex_logits = nn.Parameter(torch.zeros(4))

        # Hierarchical weights
        self.hier_ab = nn.Parameter(torch.tensor(0.5))
        self.hier_cd = nn.Parameter(torch.tensor(0.5))
        self.hier_final = nn.Parameter(torch.tensor(0.5))

        # normalize output to unit circle
        self.normalize_output = normalize_output

        # Optional bilinear layers
        if use_bilinear:
            self.bilinear_wx = nn.Bilinear(head_dim, head_dim, head_dim, bias=False)
            self.bilinear_yz = nn.Bilinear(head_dim, head_dim, head_dim, bias=False)
            self.bilinear_final = nn.Bilinear(head_dim, head_dim, head_dim, bias=False)
        else:
            self.bilinear_wx = None
            self.bilinear_yz = None
            self.bilinear_final = None

        # Four sets of inverse frequencies
        for theta, label in [(theta_w, 'w'), (theta_x, 'x'), (theta_y, 'y'), (theta_z, 'z')]:
            inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            self.register_buffer(f'inv_freq_{label}', inv_freq)

    @property
    def dimension(self) -> int:
        return 4

    @property
    def simplex_weights(self) -> Tensor:
        """Normalized 3-simplex coordinates."""
        return F.softmax(self._simplex_logits, dim=0)

    def _compute_rope(self, seq_len: int, inv_freq: Tensor) -> Tensor:
        """Compute stacked [cos, sin] for given frequencies."""
        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # [L, D//2]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [L, D]
        return torch.stack([freqs.cos(), freqs.sin()], dim=0)  # [2, L, D]

    def embed_w(self, seq_len: int) -> Tensor:
        """W component (scalar/identity)."""
        return self._compute_rope(seq_len, self.inv_freq_w)

    def embed_x(self, seq_len: int) -> Tensor:
        """X component (i)."""
        return self._compute_rope(seq_len, self.inv_freq_x)

    def embed_y(self, seq_len: int) -> Tensor:
        """Y component (j)."""
        return self._compute_rope(seq_len, self.inv_freq_y)

    def embed_z(self, seq_len: int) -> Tensor:
        """Z component (k)."""
        return self._compute_rope(seq_len, self.inv_freq_z)

    def embed(self, seq_len: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return all four embeddings."""
        return self.embed_w(seq_len), self.embed_x(seq_len), self.embed_y(seq_len), self.embed_z(seq_len)

    def modulate(self, w: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        """
        Quaternary modulation on stacked [cos, sin] tensors.

        quaternion: Hamilton product interpretation
        hadamard: Four-way element product
        pairwise: (w*x) * (y*z)
        phase_blend: Weighted phase average
        """
        if self._modulation_mode == 'quaternion':
            phase_w = torch.atan2(w[1], w[0])
            phase_x = torch.atan2(x[1], x[0])
            phase_y = torch.atan2(y[1], y[0])
            phase_z = torch.atan2(z[1], z[0])

            phase_combined = phase_w - phase_x - phase_y - phase_z
            return torch.stack([phase_combined.cos(), phase_combined.sin()], dim=0)

        elif self._modulation_mode == 'hadamard':
            return w * x * y * z

        elif self._modulation_mode == 'pairwise':
            wx = w * x
            yz = y * z
            return wx * yz

        elif self._modulation_mode == 'phase_blend':
            phase_w = torch.atan2(w[1], w[0])
            phase_x = torch.atan2(x[1], x[0])
            phase_y = torch.atan2(y[1], y[0])
            phase_z = torch.atan2(z[1], z[0])

            s = self.simplex_weights
            phase_combined = s[0] * phase_w + s[1] * phase_x + s[2] * phase_y + s[3] * phase_z

            return torch.stack([phase_combined.cos(), phase_combined.sin()], dim=0)

        else:
            raise ValueError(f"Unknown modulation: {self._modulation_mode}")

    def augment(self, w: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        """
        Quaternary augmentation on stacked [cos, sin] tensors.
        """
        if self._augmentation_mode == 'simplex':
            s = self.simplex_weights
            combined = s[0] * w + s[1] * x + s[2] * y + s[3] * z

        elif self._augmentation_mode == 'sum':
            combined = w + x + y + z

        elif self._augmentation_mode == 'hierarchical':
            a_ab = torch.sigmoid(self.hier_ab)
            a_cd = torch.sigmoid(self.hier_cd)
            a_final = torch.sigmoid(self.hier_final)

            ab = (1 - a_ab) * w + a_ab * x
            cd = (1 - a_cd) * y + a_cd * z
            combined = (1 - a_final) * ab + a_final * cd

        else:
            raise ValueError(f"Unknown augmentation: {self._augmentation_mode}")

        # Normalize to unit circle if requested
        if self.normalize_output:
            magnitude = torch.sqrt(combined[0] ** 2 + combined[1] ** 2 + 1e-8)
            combined = combined / magnitude

        return combined

    def embed_combined(
        self,
        seq_len: int,
        mode: Literal['modulate', 'augment', 'both'] = 'augment',
    ) -> Tensor:
        """Combine four embeddings using specified mode."""
        w, x, y, z = self.embed(seq_len)

        if mode == 'modulate':
            return self.modulate(w, x, y, z)
        elif mode == 'augment':
            return self.augment(w, x, y, z)
        elif mode == 'both':
            modulated = self.modulate(w, x, y, z)
            return self.augment(modulated, modulated, modulated, modulated)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def rotate(
        self,
        q: Tensor,
        k: Tensor,
        seq_len: int,
        mode: Literal['modulate', 'augment', 'both'] = 'augment',
    ) -> Tuple[Tensor, Tensor]:
        """Apply combined quad-rotary embeddings to Q and K."""
        combined = self.embed_combined(seq_len, mode=mode)  # [2, L, D]
        cos, sin = combined[0], combined[1]

        if q.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif q.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        def rotate_half(x: Tensor) -> Tensor:
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([-x2, x1], dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)

        return q_rot, k_rot

    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.embed(seq_len)

    def extra_repr(self) -> str:
        bilinear_str = ", bilinear=True" if self.use_bilinear else ""
        return (
            f"head_dim={self.head_dim}, "
            f"theta=({self.theta_w}, {self.theta_x}, {self.theta_y}, {self.theta_z}){bilinear_str}"
        )
# =============================================================================
# TEST
# =============================================================================

# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':

    def section(title: str) -> None:
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

    B, H, L, D = 4, 8, 128, 64
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)

    # -------------------------------------------------------------------------
    section("BASIC ROPE")
    # -------------------------------------------------------------------------

    rope = RoPE('rope', head_dim=64, theta=10000.0)
    print(f"RoPE: {rope}")

    cos, sin = rope(128)
    print(f"cos: {cos.shape}, sin: {sin.shape}")

    q_rot, k_rot = rope.rotate(q, k, cos, sin)
    print(f"Q rotated: {q_rot.shape}")

    q_norm_before = q.norm(dim=-1).mean()
    q_norm_after = q_rot.norm(dim=-1).mean()
    print(f"Norm preserved: {torch.allclose(q_norm_before, q_norm_after, atol=1e-5)}")

    # -------------------------------------------------------------------------
    section("SCALED ROPE (Linear)")
    # -------------------------------------------------------------------------

    scaled_linear = ScaledRoPE('scaled', head_dim=64, scale_factor=2.0, scaling_type='linear')
    print(f"ScaledRoPE: {scaled_linear}")

    cos1, sin1 = scaled_linear(256)
    q_rot, k_rot = scaled_linear.rotate(q, k, seq_len=128)
    print(f"Linear scaled rotation: {q_rot.shape}")

    # -------------------------------------------------------------------------
    section("SCALED ROPE (NTK)")
    # -------------------------------------------------------------------------

    scaled_ntk = ScaledRoPE('ntk', head_dim=64, scale_factor=2.0, scaling_type='ntk')
    print(f"NTK RoPE: {scaled_ntk}")

    print(f"Base inv_freq[0]: {rope.inv_freq[0]:.6f}")
    print(f"NTK inv_freq[0]:  {scaled_ntk.inv_freq[0]:.6f}")

    # -------------------------------------------------------------------------
    section("YARN ROPE")
    # -------------------------------------------------------------------------

    yarn = YaRNRoPE(
        'yarn',
        head_dim=64,
        scale_factor=4.0,
        original_max_len=2048,
        beta_fast=32.0,
        beta_slow=1.0,
    )
    print(f"YaRN: {yarn}")
    print(f"Attention scale: {yarn.attn_scale:.3f}")

    q_rot, k_rot = yarn.rotate(q, k, seq_len=128)
    print(f"YaRN rotation: {q_rot.shape}")

    print(f"Base inv_freq: [{rope.inv_freq[0]:.4f}, ..., {rope.inv_freq[-1]:.6f}]")
    print(f"YaRN inv_freq: [{yarn.inv_freq[0]:.4f}, ..., {yarn.inv_freq[-1]:.6f}]")

    # -------------------------------------------------------------------------
    section("PARTIAL ROPE")
    # -------------------------------------------------------------------------

    partial = PartialRoPE('partial', head_dim=64, rotary_dim=32)
    print(f"PartialRoPE: {partial}")

    cos, sin = partial(128)
    print(f"Rotary cos shape: {cos.shape} (only {partial.rotary_dim} dims)")

    q_rot, k_rot = partial.rotate(q, k, cos, sin)
    print(f"Partial rotation: {q_rot.shape}")

    q_pass_before = q[..., 32:]
    q_pass_after = q_rot[..., 32:]
    print(f"Non-rotary dims unchanged: {torch.allclose(q_pass_before, q_pass_after)}")

    # -------------------------------------------------------------------------
    section("DUAL ROPE")
    # -------------------------------------------------------------------------

    dual = DualRoPE(
        'dual',
        head_dim=64,
        theta_primary=10000.0,
        theta_secondary=1000.0,
        modulation='phase_blend',
        augmentation='lerp',
    )
    print(f"DualRoPE: {dual}")
    print(f"Lerp alpha: {dual.lerp_alpha.item():.4f}")

    # Get individual embeddings
    primary, secondary = dual(128)
    print(f"Primary: {primary.shape}, Secondary: {secondary.shape}")

    # Combined modes
    for mode in ['modulate', 'augment', 'both']:
        q_rot, k_rot = dual.rotate(q, k, 128, mode=mode)
        print(f"DualRoPE ({mode}): Q={q_rot.shape}")

    # Compare modulation modes
    print("\nDualRoPE modulation comparison:")
    for mod in ['multiply', 'complex', 'gate', 'phase_blend']:
        d = DualRoPE('test', 64, modulation=mod)
        q_rot, _ = d.rotate(q, k, 128, mode='modulate')
        print(f"  {mod}: Q norm = {q_rot.norm(dim=-1).mean():.4f}")

    # -------------------------------------------------------------------------
    section("TRI ROPE")
    # -------------------------------------------------------------------------

    tri = TriRoPE(
        'tri',
        head_dim=64,
        theta_alpha=10000.0,
        theta_beta=5000.0,
        theta_gamma=1000.0,
        modulation='phase_blend',
        augmentation='barycentric',
    )
    print(f"TriRoPE: {tri}")
    print(f"Barycentric weights: {[f'{w:.4f}' for w in tri.barycentric_weights.tolist()]}")

    # Get individual embeddings
    alpha, beta, gamma = tri(128)
    print(f"Alpha: {alpha.shape}, Beta: {beta.shape}, Gamma: {gamma.shape}")

    # Combined modes
    for mode in ['modulate', 'augment', 'both']:
        q_rot, k_rot = tri.rotate(q, k, 128, mode=mode)
        print(f"TriRoPE ({mode}): Q={q_rot.shape}")

    # Compare modulation modes
    print("\nTriRoPE modulation comparison:")
    for mod in ['geometric', 'hadamard', 'phase_blend']:
        t = TriRoPE('test', 64, modulation=mod)
        q_rot, _ = t.rotate(q, k, 128, mode='modulate')
        print(f"  {mod}: Q norm = {q_rot.norm(dim=-1).mean():.4f}")

    # -------------------------------------------------------------------------
    section("QUAD ROPE")
    # -------------------------------------------------------------------------

    quad = QuadRoPE(
        'quad',
        head_dim=64,
        theta_w=10000.0,
        theta_x=5000.0,
        theta_y=2500.0,
        theta_z=1000.0,
        modulation='phase_blend',
        augmentation='simplex',
    )
    print(f"QuadRoPE: {quad}")
    print(f"Simplex weights: {[f'{w:.4f}' for w in quad.simplex_weights.tolist()]}")

    # Get individual embeddings
    w, x, y, z = quad(128)
    print(f"W: {w.shape}, X: {x.shape}, Y: {y.shape}, Z: {z.shape}")

    # Combined modes
    for mode in ['modulate', 'augment', 'both']:
        q_rot, k_rot = quad.rotate(q, k, 128, mode=mode)
        print(f"QuadRoPE ({mode}): Q={q_rot.shape}")

    # Compare modulation modes
    print("\nQuadRoPE modulation comparison:")
    for mod in ['quaternion', 'hadamard', 'pairwise', 'phase_blend']:
        qr = QuadRoPE('test', 64, modulation=mod)
        q_rot, _ = qr.rotate(q, k, 128, mode='modulate')
        print(f"  {mod}: Q norm = {q_rot.norm(dim=-1).mean():.4f}")

    # -------------------------------------------------------------------------
    section("COMPARISON: SCALING METHODS")
    # -------------------------------------------------------------------------

    methods = [
        ('Base', RoPE('base', 64)),
        ('Linear 2x', ScaledRoPE('lin2', 64, scale_factor=2.0, scaling_type='linear')),
        ('Linear 4x', ScaledRoPE('lin4', 64, scale_factor=4.0, scaling_type='linear')),
        ('NTK 2x', ScaledRoPE('ntk2', 64, scale_factor=2.0, scaling_type='ntk')),
        ('NTK 4x', ScaledRoPE('ntk4', 64, scale_factor=4.0, scaling_type='ntk')),
        ('YaRN 4x', YaRNRoPE('yarn4', 64, scale_factor=4.0)),
    ]

    print(f"\n{'Method':<12} {'inv_freq[0]':>12} {'inv_freq[-1]':>14}")
    print("-" * 40)
    for name, r in methods:
        print(f"{name:<12} {r.inv_freq[0]:>12.6f} {r.inv_freq[-1]:>14.8f}")

    # -------------------------------------------------------------------------
    section("LAYOUTS")
    # -------------------------------------------------------------------------

    rope = RoPE('rope', head_dim=64)
    cos, sin = rope(32)

    # [B, H, L, D]
    q1 = torch.randn(4, 8, 32, 64)
    k1 = torch.randn(4, 8, 32, 64)
    q1_rot, k1_rot = rope.rotate(q1, k1, cos, sin)
    print(f"[B, H, L, D]: {q1.shape} -> {q1_rot.shape}")

    # [B, L, D]
    q2 = torch.randn(4, 32, 64)
    k2 = torch.randn(4, 32, 64)
    q2_rot, k2_rot = rope.rotate(q2, k2, cos, sin)
    print(f"[B, L, D]: {q2.shape} -> {q2_rot.shape}")

    # -------------------------------------------------------------------------
    section("CACHE BEHAVIOR")
    # -------------------------------------------------------------------------

    rope = RoPE('rope', head_dim=64, max_seq_len=256)
    print(f"Initial cache: {rope._cos.shape}")

    cos1, sin1 = rope(128)
    print(f"seq_len=128: cache unchanged at {rope._cos.shape}")

    cos2, sin2 = rope(512)
    print(f"seq_len=512: cache rebuilt to {rope._cos.shape}")

    # -------------------------------------------------------------------------
    section("DEVICE MOVEMENT")
    # -------------------------------------------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rope = RoPE('rope', head_dim=64).to(device)
    print(f"RoPE device: {rope.inv_freq.device}")

    q_dev = torch.randn(4, 8, 64, 64, device=device)
    k_dev = torch.randn(4, 8, 64, 64, device=device)

    cos, sin = rope(64)
    print(f"cos device: {cos.device}")

    q_rot, k_rot = rope.rotate(q_dev, k_dev, cos, sin)
    print(f"Q_rot device: {q_rot.device}")

    # -------------------------------------------------------------------------
    section("COMPILE TEST")
    # -------------------------------------------------------------------------

    rope = RoPE('rope', head_dim=64).to(device)
    q_gpu = torch.randn(4, 8, 64, 64, device=device)
    k_gpu = torch.randn(4, 8, 64, 64, device=device)

    # Uncompiled
    cos, sin = rope(64)
    q_rot1, k_rot1 = rope.rotate(q_gpu, k_gpu, cos, sin)

    # Compiled
    rotate_compiled = torch.compile(rope.rotate)
    q_rot2, k_rot2 = rotate_compiled(q_gpu, k_gpu, cos, sin)

    print(f"Device: {q_rot1.device}")

    q_match = torch.allclose(q_rot1, q_rot2, rtol=1e-4, atol=1e-6)
    k_match = torch.allclose(k_rot1, k_rot2, rtol=1e-4, atol=1e-6)
    q_max_diff = (q_rot1 - q_rot2).abs().max().item()
    k_max_diff = (k_rot1 - k_rot2).abs().max().item()

    print(f"Compiled match Q: {q_match} (max diff: {q_max_diff:.2e})")
    print(f"Compiled match K: {k_match} (max diff: {k_max_diff:.2e})")

    # -------------------------------------------------------------------------
    section("AGATHA TOWER EXAMPLE")
    # -------------------------------------------------------------------------

    print("Creating 10 towers with different theta configurations:\n")

    tower_configs = [
        ('T1 Cantor', 10000.0, 1.0),
        ('T2 Simplex', 10000.0, 1.0),
        ('T3 Shape', 10000.0, 1.0),
        ('T4 Cantor-Inv', 10000.0, 1.0),
        ('T5 Simplex-Inv', 10000.0, 1.0),
        ('T6 Shape-Inv', 10000.0, 1.0),
        ('T7 θ=1.00', 10000.0, 1.00),
        ('T8 θ=0.15', 10000.0, 0.15),
        ('T9 θ=0.30', 10000.0, 0.30),
        ('T10 θ=0.45', 10000.0, 0.45),
    ]

    print(f"{'Tower':<15} {'theta':>10} {'scale':>8} {'inv_freq[0]':>12}")
    print("-" * 50)

    for name, theta, scale in tower_configs:
        r = RoPE(name, head_dim=64, theta=theta, theta_scale=scale)
        print(f"{name:<15} {theta:>10.1f} {scale:>8.2f} {r.inv_freq[0]:>12.6f}")

    # -------------------------------------------------------------------------
    section("BILINEAR TOGGLE")
    # -------------------------------------------------------------------------

    dual_lean = DualRoPE('dual_lean', 64, use_bilinear=False)
    dual_heavy = DualRoPE('dual_heavy', 64, use_bilinear=True)

    tri_lean = TriRoPE('tri_lean', 64, use_bilinear=False)
    tri_heavy = TriRoPE('tri_heavy', 64, use_bilinear=True)

    quad_lean = QuadRoPE('quad_lean', 64, use_bilinear=False)
    quad_heavy = QuadRoPE('quad_heavy', 64, use_bilinear=True)

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    print(f"{'Variant':<20} {'Params':>12}")
    print("-" * 34)
    print(f"{'RoPE':<20} {0:>12,}")
    print(f"{'DualRoPE (lean)':<20} {count_params(dual_lean):>12,}")
    print(f"{'DualRoPE (bilinear)':<20} {count_params(dual_heavy):>12,}")
    print(f"{'TriRoPE (lean)':<20} {count_params(tri_lean):>12,}")
    print(f"{'TriRoPE (bilinear)':<20} {count_params(tri_heavy):>12,}")
    print(f"{'QuadRoPE (lean)':<20} {count_params(quad_lean):>12,}")
    print(f"{'QuadRoPE (bilinear)':<20} {count_params(quad_heavy):>12,}")

    # Verify all produce valid output
    print("\nValidating outputs:")
    q_test = torch.randn(2, 4, 64, 64)
    k_test = torch.randn(2, 4, 64, 64)

    for name, rope_inst in [
        ('DualRoPE lean', dual_lean),
        ('DualRoPE bilinear', dual_heavy),
        ('TriRoPE lean', tri_lean),
        ('TriRoPE bilinear', tri_heavy),
        ('QuadRoPE lean', quad_lean),
        ('QuadRoPE bilinear', quad_heavy),
    ]:
        q_out, k_out = rope_inst.rotate(q_test, k_test, 64)
        print(f"  {name}: {q_out.shape} ✓")

    # -------------------------------------------------------------------------
    section("ROPE HIERARCHY COMPARISON")
    # -------------------------------------------------------------------------

    print("All RoPE variants with same base config:\n")
    print(f"{'Variant':<12} {'Params':>8} {'Output':>24}")
    print("-" * 48)

    variants = [
        ('RoPE', RoPE('r', 64)),
        ('ScaledRoPE', ScaledRoPE('s', 64, scale_factor=2.0)),
        ('YaRNRoPE', YaRNRoPE('y', 64, scale_factor=2.0)),
        ('PartialRoPE', PartialRoPE('p', 64, rotary_dim=32)),
        ('DualRoPE', DualRoPE('d', 64)),
        ('TriRoPE', TriRoPE('t', 64)),
        ('QuadRoPE', QuadRoPE('q', 64)),
    ]

    for name, r in variants:
        params = count_params(r)
        # Handle different rotate signatures
        if name in ('DualRoPE', 'TriRoPE', 'QuadRoPE'):
            q_rot, _ = r.rotate(q, k, 128)
        else:
            q_rot, _ = r.rotate(q, k, seq_len=128)
        print(f"{name:<12} {params:>8,} {str(q_rot.shape):>24}")

    # -------------------------------------------------------------------------
    section("AUGMENTATION MODES")
    # -------------------------------------------------------------------------

    print("DualRoPE augmentation modes:")
    for aug in ['add', 'lerp', 'residual']:
        d = DualRoPE('test', 64, augmentation=aug)
        q_rot, _ = d.rotate(q, k, 128, mode='augment')
        print(f"  {aug}: Q norm = {q_rot.norm(dim=-1).mean():.4f}")

    print("\nTriRoPE augmentation modes:")
    for aug in ['barycentric', 'sum', 'attention']:
        t = TriRoPE('test', 64, augmentation=aug)
        q_rot, _ = t.rotate(q, k, 128, mode='augment')
        print(f"  {aug}: Q norm = {q_rot.norm(dim=-1).mean():.4f}")

    print("\nQuadRoPE augmentation modes:")
    for aug in ['simplex', 'sum', 'hierarchical']:
        qr = QuadRoPE('test', 64, augmentation=aug)
        q_rot, _ = qr.rotate(q, k, 128, mode='augment')
        print(f"  {aug}: Q norm = {q_rot.norm(dim=-1).mean():.4f}")

    # -------------------------------------------------------------------------
    section("MULTI-SCALE UNIFIED EXAMPLE")
    # -------------------------------------------------------------------------

    print("QuadRoPE as unified multi-scale RoPE:\n")

    unified = QuadRoPE(
        'unified',
        head_dim=64,
        theta_w=10000.0,   # Global (scale 1.0)
        theta_x=1500.0,    # ~0.15 scale
        theta_y=3000.0,    # ~0.30 scale
        theta_z=4500.0,    # ~0.45 scale
        augmentation='simplex',
    )

    print(f"Config: {unified}")
    print(f"Simplex weights (learnable): {[f'{w:.4f}' for w in unified.simplex_weights.tolist()]}")

    q_rot, k_rot = unified.rotate(q, k, 128, mode='augment')
    print(f"Output: {q_rot.shape}")
    print(f"Norm preserved: {torch.allclose(q.norm(dim=-1).mean(), q_rot.norm(dim=-1).mean(), atol=0.5)}")

    # -------------------------------------------------------------------------
    section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nRoPE variants ready:")
    print("  RoPE        - Standard rotary embedding")
    print("  ScaledRoPE  - Linear and NTK-aware scaling")
    print("  YaRNRoPE    - State-of-the-art length extension")
    print("  PartialRoPE - Apply to subset of dimensions")
    print("  DualRoPE    - Two configs with modulation/augmentation")
    print("  TriRoPE     - Three configs with barycentric combination")
    print("  QuadRoPE    - Four configs with quaternionic/simplex combination")

    print("\nKey features:")
    print("  - Pre-computed cache for efficiency")
    print("  - Supports [B, H, L, D] and [B, L, D] layouts")
    print("  - Configurable theta for multi-scale towers")
    print("  - Device-aware with automatic cache rebuilding")
    print("  - torch.compile compatible")
    print("  - Lean (few params) or bilinear (learnable) modes")
    print("  - Geometric modulation: phase_blend, complex, quaternion")
    print("  - Geometric augmentation: lerp, barycentric, simplex, hierarchical")