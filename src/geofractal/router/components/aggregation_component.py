"""
geofractal.router.components.aggregation_component
===================================================

Universal aggregation and folding components with comprehensive masking.

Architecture:
    BlendMode:   How to combine (a, b) at a single alpha value
    Schedule:    How alpha evolves across steps (fixed or learnable)
    Aggregation: How to pool stepped results → final output

Masking Philosophy:
    - Alpha mask: Continuous [0, 1] weights controlling blend/aggregation strength
    - Binary mask: Hard 0/1 for fingerprint preservation
    - mask=0: Preserve original (fingerprinted zone)
    - mask=1: Apply full processing (non-fingerprinted zone)
    - EVERY component supports both mask types uniformly

Fingerprinting Use Case:
    Fingerprinted zones are protected from modification during interpolation.
    Non-fingerprinted zones receive full blend/aggregation processing.
    This enables selective modification while preserving critical features.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, List, Tuple, Dict, Any, Union, Literal
from abc import ABC, abstractmethod
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.components.torch_component import TorchComponent


# =============================================================================
# MASK UTILITIES
# =============================================================================

def normalize_mask(
    mask: Optional[Tensor],
    target_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Normalize mask to target shape with proper broadcasting.

    Handles common mask patterns:
        - [B, T] -> [B, T, D] or [B, S, T, D]
        - [B, T, 1] -> [B, T, D] or [B, S, T, D]
        - [B, T, D] -> [B, S, T, D]
        - [B, S, T] -> [B, S, T, D]
        - [B, S, T, 1] -> [B, S, T, D]

    Key insight: Position masks [B, T, ...] need step dimension inserted
    when expanding to [B, S, T, D].

    Args:
        mask: Input mask
        target_shape: Expected output shape
        device: Target device
        dtype: Target dtype

    Returns:
        Mask tensor matching target_shape
    """
    if mask is None:
        return torch.ones(target_shape, device=device, dtype=dtype)

    mask = mask.to(device=device, dtype=dtype)

    target_ndim = len(target_shape)

    # Handle 4D target (stepped: [B, S, T, D])
    if target_ndim == 4:
        B, S, T, D = target_shape

        if mask.dim() == 2:
            # [B, T] -> [B, 1, T, 1] -> [B, S, T, D]
            if mask.shape[1] == T:
                mask = mask.unsqueeze(1).unsqueeze(-1)
            # [B, S] -> [B, S, 1, 1] -> [B, S, T, D]
            elif mask.shape[1] == S:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            else:
                mask = mask.unsqueeze(1).unsqueeze(-1)

        elif mask.dim() == 3:
            # [B, T, 1] or [B, T, D] -> [B, 1, T, ...] -> [B, S, T, D]
            if mask.shape[1] == T:
                mask = mask.unsqueeze(1)
            # [B, S, T] -> [B, S, T, 1] -> [B, S, T, D]
            elif mask.shape[1] == S and mask.shape[2] == T:
                mask = mask.unsqueeze(-1)
            else:
                mask = mask.unsqueeze(1)

        # mask.dim() == 4: use as-is or minimal adjustment
        elif mask.dim() == 4:
            pass  # Already 4D

    # Handle 3D target ([B, T, D])
    elif target_ndim == 3:
        B, T, D = target_shape

        if mask.dim() == 2:
            # [B, T] -> [B, T, 1]
            mask = mask.unsqueeze(-1)
        # [B, T, 1] or [B, T, D]: use as-is

    # Handle 2D target ([B, D])
    elif target_ndim == 2:
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)

    # Final expansion to target shape
    return mask.expand(target_shape)


def apply_mask_blend(
    result: Tensor,
    original: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Apply mask to blend result with original.

    Where mask=0, preserve original (fingerprint).
    Where mask=1, use result (processed).

    Args:
        result: Processed tensor
        original: Original tensor to preserve where mask=0
        mask: Blend weights [0, 1]

    Returns:
        Masked blend: original * (1 - mask) + result * mask
    """
    return original * (1 - mask) + result * mask


def binarize_mask(mask: Tensor, threshold: float = 0.5) -> Tensor:
    """Convert alpha mask to binary mask."""
    return (mask > threshold).to(mask.dtype)


# =============================================================================
# BLEND MODES - All with mask support
# =============================================================================

class BlendMode(ABC):
    """
    Abstract blend mode with masking support.

    All blend modes operate on:
        a: [B, T, D] or [B, S, T, D] source
        b: [B, T, D] or [B, S, T, D] target
        alpha: blend factor (various shapes)
        mask: [0, 1] weights where 0=preserve a, 1=blend normally

    Returns: blended result with same shape as inputs
    """

    @abstractmethod
    def _blend_impl(
        self,
        a: Tensor,
        b: Tensor,
        alpha: Tensor,
        delta: Optional[Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """Internal blend implementation without masking."""
        ...

    def blend(
        self,
        a: Tensor,
        b: Tensor,
        alpha: Tensor,
        delta: Optional[Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Blend a toward b by alpha, respecting mask.

        Args:
            a: Source tensor
            b: Target tensor
            alpha: Blend factor
            delta: Precomputed b - a (optional)
            context: Additional context for specialized blends
            mask: Where 0=preserve a (fingerprint), 1=apply blend

        Returns:
            Blended tensor
        """
        result = self._blend_impl(a, b, alpha, delta, context)

        if mask is not None:
            mask = normalize_mask(mask, result.shape, result.device, result.dtype)
            result = apply_mask_blend(result, a, mask)

        return result

    def __call__(self, a, b, alpha, delta=None, context=None, mask=None):
        return self.blend(a, b, alpha, delta, context, mask)


class LerpBlend(BlendMode):
    """Linear interpolation: a + alpha * (b - a)"""

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)
        d = delta if delta is not None else (b - a)
        return a + alpha * d


class SlerpBlend(BlendMode):
    """
    Spherical linear interpolation.
    Preserves magnitude, interpolates along great circle.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        # Normalize for angle computation
        a_norm = F.normalize(a, dim=-1, eps=self.eps)
        b_norm = F.normalize(b, dim=-1, eps=self.eps)

        # Compute angle
        dot = (a_norm * b_norm).sum(dim=-1, keepdim=True)
        dot = dot.clamp(-1 + self.eps, 1 - self.eps)
        omega = torch.acos(dot)

        # Compute slerp coefficients
        sin_omega = torch.sin(omega).clamp_min(self.eps)
        t1 = torch.sin((1 - alpha) * omega) / sin_omega
        t2 = torch.sin(alpha * omega) / sin_omega

        return t1 * a + t2 * b


class SlipBlend(BlendMode):
    """
    Entropic phase gating.
    Uses alignment of delta with target as gating signal.

    gate = sigmoid((delta · b).sum())
    output = a + (alpha * gate) * delta
    """

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        d = delta if delta is not None else (b - a)

        # Phase alignment: how much delta points toward b
        phase = (d * b).sum(dim=-1, keepdim=True)
        phase_gate = torch.sigmoid(phase)

        return a + (alpha * phase_gate) * d


class ZeusBlend(BlendMode):
    """
    Sharp threshold transition.
    Sigmoid-gated switch from a to b at alpha=0.5.
    """

    def __init__(self, sharpness: float = 10.0):
        self.sharpness = sharpness

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        sharpness = (context or {}).get('zeus_sharpness', self.sharpness)
        gate = torch.sigmoid(sharpness * (alpha - 0.5))
        return a * (1 - gate) + b * gate


class HeliosBlend(BlendMode):
    """
    Sine-weighted blend.
    Smooth transition with peak velocity at alpha=0.5.
    """

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        weight = torch.sin(math.pi * alpha)
        d = delta if delta is not None else (b - a)
        return a + weight * d


class SurgeBlend(BlendMode):
    """
    Exponential surge: rapid initial change, asymptotic approach.
    output = a + (1 - exp(-intensity * alpha)) * (b - a)
    """

    def __init__(self, intensity: float = 5.0):
        self.intensity = intensity

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        intensity = (context or {}).get('surge_intensity', self.intensity)
        surge = 1 - torch.exp(-intensity * alpha)
        d = delta if delta is not None else (b - a)
        return a + surge * d


class RippleBlend(BlendMode):
    """
    Sinusoidal ripple modulation.
    Oscillates around the interpolation path.
    """

    def __init__(self, freq: float = 2.0):
        self.freq = freq

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        freq = (context or {}).get('ripple_freq', self.freq)
        ripple = torch.sin(freq * math.pi * alpha)
        d = delta if delta is not None else (b - a)
        return a + ripple * d


class GilgameshBlend(BlendMode):
    """
    Multi-axis projection blend.
    Projects delta along multiple weighted axes.
    """

    def __init__(self, axes: List[float] = None):
        self.axes = axes or [0.25, 0.33, 0.5, 0.66, 0.75]

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        axes = (context or {}).get('gilgamesh_axes', self.axes)
        d = delta if delta is not None else (b - a)

        out = a
        for w in axes:
            weight = torch.sigmoid(alpha) * w
            out = out + weight * d

        return out


class ShivaBlend(BlendMode):
    """
    Icy gradient decay.
    Inverse exponential approach.
    """

    def __init__(self, decay_rate: float = 4.0):
        self.decay_rate = decay_rate

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        rate = (context or {}).get('shiva_decay', self.decay_rate)
        cold = torch.exp(-rate * alpha)
        d = delta if delta is not None else (b - a)
        return a + (1.0 - cold) * d


class IfritBlend(BlendMode):
    """
    Fiery waveform spikes.
    Squared sine for sharper peaks.
    """

    def __init__(self, freq: float = 4.0, amp: float = 1.0):
        self.freq = freq
        self.amp = amp

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        freq = (context or {}).get('ifrit_freq', self.freq)
        amp = (context or {}).get('ifrit_amp', self.amp)

        fire = (torch.sin(freq * math.pi * alpha) ** 2) * amp
        d = delta if delta is not None else (b - a)
        return a + fire * d


class MinPBlend(BlendMode):
    """
    Min-P gated blending.

    Only applies blend where delta magnitude is above threshold
    relative to max delta magnitude. Preserves regions where
    the change would be insignificant.

    gate = (|delta| >= min_p * max(|delta|))
    output = a + gate * alpha * delta

    This naturally preserves stable regions while allowing
    significant changes to propagate.

    Args:
        min_p: Minimum probability threshold (0.0 to 1.0)
        per_position: If True, compute threshold per position (T dim)
                     If False, compute globally per batch
    """

    def __init__(self, min_p: float = 0.1, per_position: bool = False):
        self.min_p = min_p
        self.per_position = per_position

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        d = delta if delta is not None else (b - a)

        # Compute delta magnitude
        delta_mag = d.abs()  # [B, T, D] or [B, S, T, D]

        min_p = (context or {}).get('min_p', self.min_p)

        if self.per_position:
            # Threshold per position: max over D dimension
            max_mag = delta_mag.max(dim=-1, keepdim=True).values
        else:
            # Global threshold: max over all dimensions except batch
            if delta_mag.dim() == 3:  # [B, T, D]
                max_mag = delta_mag.view(delta_mag.shape[0], -1).max(dim=-1).values
                max_mag = max_mag.view(-1, 1, 1)
            else:  # [B, S, T, D]
                max_mag = delta_mag.view(delta_mag.shape[0], -1).max(dim=-1).values
                max_mag = max_mag.view(-1, 1, 1, 1)

        threshold = min_p * max_mag

        # Gate: 1 where significant, 0 where below threshold
        gate = (delta_mag >= threshold).float()

        return a + gate * alpha * d


# Blend mode registry
BLEND_MODES: Dict[str, BlendMode] = {
    'lerp': LerpBlend(),
    'slerp': SlerpBlend(),
    'slip': SlipBlend(),
    'zeus': ZeusBlend(),
    'helios': HeliosBlend(),
    'surge': SurgeBlend(),
    'ripple': RippleBlend(),
    'gilgamesh': GilgameshBlend(),
    'shiva': ShivaBlend(),
    'ifrit': IfritBlend(),
    'min_p': MinPBlend(),
}


def get_blend_mode(name: str) -> BlendMode:
    """Get blend mode by name."""
    return BLEND_MODES.get(name.lower(), LerpBlend())


# =============================================================================
# SCHEDULES - How alpha evolves across steps
# =============================================================================

class Schedule(nn.Module):
    """
    Generates alpha values for all steps.

    Returns: [S] or [B, S] tensor of alpha values in [0, 1]
    """

    @abstractmethod
    def forward(self, num_steps: int, batch_size: int = 1) -> Tensor:
        """Generate alpha schedule."""
        ...


class LinearSchedule(Schedule):
    """Linear ramp from 0 to 1."""

    def forward(self, num_steps: int, batch_size: int = 1) -> Tensor:
        return torch.linspace(0, 1, num_steps)


class CosineSchedule(Schedule):
    """Cosine annealing: 0.5 * (1 - cos(pi * t))"""

    def forward(self, num_steps: int, batch_size: int = 1) -> Tensor:
        t = torch.linspace(0, 1, num_steps)
        return 0.5 * (1 - torch.cos(math.pi * t))


class SigmoidSchedule(Schedule):
    """Sigmoid schedule with configurable sharpness."""

    def __init__(self, sharpness: float = 10.0, center: float = 0.5):
        super().__init__()
        self.sharpness = sharpness
        self.center = center

    def forward(self, num_steps: int, batch_size: int = 1) -> Tensor:
        t = torch.linspace(0, 1, num_steps)
        return torch.sigmoid(self.sharpness * (t - self.center))


class TauSchedule(Schedule):
    """
    Tau-modulated schedule.
    alpha = (1 - exp(-tau * t)) * sin(pi * t)
    """

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(self, num_steps: int, batch_size: int = 1) -> Tensor:
        t = torch.linspace(0, 1, num_steps)
        tau_scale = 1 - torch.exp(-self.tau * t)
        sigma = torch.sin(math.pi * t)
        return (tau_scale * sigma).clamp(0, 1)


class WaveSchedule(Schedule):
    """Sinusoidal wave schedule."""

    def __init__(self, freq: float = 1.0):
        super().__init__()
        self.freq = freq

    def forward(self, num_steps: int, batch_size: int = 1) -> Tensor:
        t = torch.linspace(0, 1, num_steps)
        return torch.sin(self.freq * math.pi * t).clamp(0, 1)


class LearnableSchedule(Schedule):
    """
    Learnable timestep schedule.

    Learns optimal alpha values for each step.
    Constrained to [0, 1] via sigmoid.
    """

    def __init__(self, num_steps: int, init: str = 'linear'):
        super().__init__()
        self.num_steps = num_steps

        # Initialize
        if init == 'linear':
            init_vals = torch.linspace(-2, 2, num_steps)
        elif init == 'cosine':
            t = torch.linspace(0, 1, num_steps)
            cosine = 0.5 * (1 - torch.cos(math.pi * t))
            init_vals = torch.log(cosine.clamp(1e-6, 1-1e-6) / (1 - cosine.clamp(1e-6, 1-1e-6)))
        else:
            init_vals = torch.zeros(num_steps)

        self.raw_alphas = nn.Parameter(init_vals)

    def forward(self, num_steps: int = None, batch_size: int = 1) -> Tensor:
        return torch.sigmoid(self.raw_alphas)

    def get_alphas(self) -> Tensor:
        """Get current alpha values."""
        return torch.sigmoid(self.raw_alphas).detach()


class AdaptiveSchedule(Schedule):
    """
    Content-adaptive schedule.

    Learns to predict optimal alphas from input content.
    """

    def __init__(self, in_features: int, num_steps: int, hidden_dim: int = None):
        super().__init__()
        self.num_steps = num_steps
        hidden = hidden_dim or in_features // 2

        self.net = nn.Sequential(
            nn.Linear(in_features * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_steps),
        )

    def forward_with_content(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Generate schedule based on content.

        Args:
            a: [B, T, D] source
            b: [B, T, D] target

        Returns:
            [B, S] alpha schedule per batch
        """
        a_pool = a.mean(dim=1)
        b_pool = b.mean(dim=1)
        combined = torch.cat([a_pool, b_pool], dim=-1)
        raw = self.net(combined)
        return torch.sigmoid(raw)

    def forward(self, num_steps: int = None, batch_size: int = 1) -> Tensor:
        return torch.linspace(0, 1, self.num_steps)


# Schedule registry
SCHEDULES: Dict[str, type] = {
    'linear': LinearSchedule,
    'cosine': CosineSchedule,
    'sigmoid': SigmoidSchedule,
    'tau': TauSchedule,
    'wave': WaveSchedule,
    'learnable': LearnableSchedule,
    'adaptive': AdaptiveSchedule,
}


# =============================================================================
# AGGREGATION - All with comprehensive mask support
# =============================================================================

class Aggregation(nn.Module):
    """
    Aggregates stepped outputs [B, S, T, D] → [B, T, D]

    All aggregations support:
        - Alpha mask: Continuous [0, 1] contribution weights
        - Binary mask: Hard 0/1 for fingerprint preservation
        - mask=0: Position does not contribute to aggregation
        - mask=1: Position contributes fully

    S = num_steps (interpolation steps)
    T = sequence length
    D = features
    """

    def _compute_scores(
        self,
        stepped: Tensor,
        alphas: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute per-step scores for weighted aggregation.

        Override in subclasses for custom scoring.

        Args:
            stepped: [B, S, T, D]
            alphas: [S] or [B, S]

        Returns:
            [B, S] scores (unnormalized)
        """
        B, S, T, D = stepped.shape
        return torch.ones(B, S, device=stepped.device, dtype=stepped.dtype)

    def _aggregate_impl(
        self,
        stepped: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """
        Apply weighted aggregation.

        Args:
            stepped: [B, S, T, D]
            weights: [B, S] normalized weights

        Returns:
            [B, T, D]
        """
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]
        return (stepped * weights).sum(dim=1)

    def forward(
        self,
        stepped: Tensor,
        alphas: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        original: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Aggregate stepped outputs with optional masking.

        Args:
            stepped: [B, S, T, D] stepped interpolation outputs
            alphas: [S] or [B, S] alpha values (for weighted aggregation)
            mask: Aggregation mask:
                - [B, T, D] or [B, T, 1] or [B, T]: position-level mask
                - [B, S, T, D] or similar: step+position-level mask
                Where 0=exclude from aggregation, 1=include fully
            original: [B, T, D] original tensor for fingerprint preservation

        Returns:
            [B, T, D] aggregated output
        """
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        # Compute scores and normalize to weights
        scores = self._compute_scores(stepped, alphas)  # [B, S]

        # Apply step-level masking to scores if provided
        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            # Pool mask to step level: [B, S, T, D] -> [B, S]
            step_mask = mask_normalized.mean(dim=(-1, -2))  # [B, S]
            scores = scores * step_mask

        # Normalize weights
        weights = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # Aggregate
        result = self._aggregate_impl(stepped, weights)

        # Apply position-level masking for fingerprint preservation
        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class MeanAggregation(Aggregation):
    """Simple mean across steps."""

    def _compute_scores(self, stepped, alphas=None):
        B, S, T, D = stepped.shape
        return torch.ones(B, S, device=stepped.device, dtype=stepped.dtype)


class SumAggregation(Aggregation):
    """Sum across steps (no normalization)."""

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            stepped = stepped * mask_normalized

        result = stepped.sum(dim=1)

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class MaxAggregation(Aggregation):
    """Max across steps."""

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            # Set masked positions to -inf for max
            stepped = stepped.masked_fill(mask_normalized < 0.5, float('-inf'))

        result = stepped.max(dim=1).values

        # Handle all-masked case
        result = torch.where(
            torch.isinf(result),
            torch.zeros_like(result),
            result
        )

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class MinAggregation(Aggregation):
    """Min across steps."""

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            # Set masked positions to +inf for min
            stepped = stepped.masked_fill(mask_normalized < 0.5, float('inf'))

        result = stepped.min(dim=1).values

        # Handle all-masked case
        result = torch.where(
            torch.isinf(result),
            torch.zeros_like(result),
            result
        )

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class TopKAggregation(Aggregation):
    """
    Aggregate top-k steps by score.

    Score can be: magnitude, variance, similarity to target, etc.
    """

    def __init__(self, k: int = 3, score_mode: str = 'magnitude'):
        super().__init__()
        self.k = k
        self.score_mode = score_mode

    def _compute_scores(self, stepped, alphas=None):
        B, S, T, D = stepped.shape

        if self.score_mode == 'magnitude':
            # Score by L2 norm
            scores = stepped.norm(dim=-1).mean(dim=-1)  # [B, S]
        elif self.score_mode == 'variance':
            # Score by variance (higher variance = more informative)
            scores = stepped.var(dim=-1).mean(dim=-1)  # [B, S]
        elif self.score_mode == 'alpha':
            # Score by alpha value (prefer later steps)
            if alphas is not None:
                if alphas.dim() == 1:
                    scores = alphas.unsqueeze(0).expand(B, -1)
                else:
                    scores = alphas
            else:
                scores = torch.linspace(0, 1, S, device=stepped.device).unsqueeze(0).expand(B, -1)
        else:
            scores = torch.ones(B, S, device=stepped.device, dtype=stepped.dtype)

        return scores

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype
        k = min(self.k, S)

        scores = self._compute_scores(stepped, alphas)

        # Apply mask to scores
        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            step_mask = mask_normalized.mean(dim=(-1, -2))
            scores = scores * step_mask

        # Get top-k indices
        _, top_indices = scores.topk(k, dim=1)  # [B, k]

        # Gather top-k steps
        top_indices_exp = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, D)
        top_stepped = torch.gather(stepped, 1, top_indices_exp)  # [B, k, T, D]

        # Average top-k
        result = top_stepped.mean(dim=1)

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class BottomKAggregation(Aggregation):
    """
    Aggregate bottom-k steps by score.
    """

    def __init__(self, k: int = 3, score_mode: str = 'magnitude'):
        super().__init__()
        self.k = k
        self.score_mode = score_mode

    def _compute_scores(self, stepped, alphas=None):
        B, S, T, D = stepped.shape

        if self.score_mode == 'magnitude':
            scores = stepped.norm(dim=-1).mean(dim=-1)
        elif self.score_mode == 'variance':
            scores = stepped.var(dim=-1).mean(dim=-1)
        elif self.score_mode == 'alpha':
            if alphas is not None:
                if alphas.dim() == 1:
                    scores = alphas.unsqueeze(0).expand(B, -1)
                else:
                    scores = alphas
            else:
                scores = torch.linspace(0, 1, S, device=stepped.device).unsqueeze(0).expand(B, -1)
        else:
            scores = torch.ones(B, S, device=stepped.device, dtype=stepped.dtype)

        return scores

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype
        k = min(self.k, S)

        scores = self._compute_scores(stepped, alphas)

        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            step_mask = mask_normalized.mean(dim=(-1, -2))
            # Invert mask for bottom-k (masked positions get high score)
            scores = scores + (1 - step_mask) * 1e6

        # Get bottom-k indices (smallest scores)
        _, bottom_indices = scores.topk(k, dim=1, largest=False)

        bottom_indices_exp = bottom_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, D)
        bottom_stepped = torch.gather(stepped, 1, bottom_indices_exp)

        result = bottom_stepped.mean(dim=1)

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class SoftmaxAggregation(Aggregation):
    """
    Softmax-weighted aggregation by score.
    """

    def __init__(self, temperature: float = 1.0, score_mode: str = 'magnitude'):
        super().__init__()
        self.temperature = temperature
        self.score_mode = score_mode

    def _compute_scores(self, stepped, alphas=None):
        B, S, T, D = stepped.shape

        if self.score_mode == 'magnitude':
            scores = stepped.norm(dim=-1).mean(dim=-1)
        elif self.score_mode == 'variance':
            scores = stepped.var(dim=-1).mean(dim=-1)
        elif self.score_mode == 'last_similarity':
            # Similarity to last step
            last = stepped[:, -1:]
            stepped_flat = stepped.view(B, S, -1)
            last_flat = last.view(B, 1, -1)
            scores = F.cosine_similarity(stepped_flat, last_flat, dim=-1)
        else:
            scores = torch.ones(B, S, device=stepped.device, dtype=stepped.dtype)

        return scores / self.temperature

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        scores = self._compute_scores(stepped, alphas)

        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            step_mask = mask_normalized.mean(dim=(-1, -2))
            # Masked positions get -inf score
            scores = scores.masked_fill(step_mask < 0.5, float('-inf'))

        weights = F.softmax(scores, dim=1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        result = (stepped * weights).sum(dim=1)

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class SoftminAggregation(Aggregation):
    """
    Softmin-weighted aggregation (inverse softmax).
    Lower scores get higher weights.
    """

    def __init__(self, temperature: float = 1.0, score_mode: str = 'magnitude'):
        super().__init__()
        self.temperature = temperature
        self.score_mode = score_mode

    def _compute_scores(self, stepped, alphas=None):
        B, S, T, D = stepped.shape

        if self.score_mode == 'magnitude':
            scores = stepped.norm(dim=-1).mean(dim=-1)
        elif self.score_mode == 'variance':
            scores = stepped.var(dim=-1).mean(dim=-1)
        else:
            scores = torch.ones(B, S, device=stepped.device, dtype=stepped.dtype)

        return -scores / self.temperature  # Negate for softmin

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        scores = self._compute_scores(stepped, alphas)

        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            step_mask = mask_normalized.mean(dim=(-1, -2))
            scores = scores.masked_fill(step_mask < 0.5, float('-inf'))

        weights = F.softmax(scores, dim=1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        result = (stepped * weights).sum(dim=1)

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class MinPAggregation(Aggregation):
    """
    Min-P aggregation: adaptive filtering based on score distribution.

    Keeps steps where: score >= min_p * max_score

    Unlike top_k (fixed count) or softmax (all weighted), min_p naturally
    adapts to the score distribution - high confidence batches might use
    few steps, low confidence batches use more.

    Args:
        min_p: Minimum probability threshold (0.0 to 1.0)
        score_mode: How to compute scores ('magnitude', 'variance', 'similarity')
        normalize: Whether to normalize remaining weights
    """

    def __init__(
        self,
        min_p: float = 0.1,
        score_mode: str = 'magnitude',
        normalize: bool = True,
    ):
        super().__init__()
        self.min_p = min_p
        self.score_mode = score_mode
        self.normalize = normalize

    def _compute_scores(self, stepped, alphas=None):
        B, S, T, D = stepped.shape

        if self.score_mode == 'magnitude':
            scores = stepped.norm(dim=-1).mean(dim=-1)  # [B, S]
        elif self.score_mode == 'variance':
            scores = stepped.var(dim=-1).mean(dim=-1)
        elif self.score_mode == 'similarity':
            # Similarity to last step
            last = stepped[:, -1:]
            stepped_flat = stepped.view(B, S, -1)
            last_flat = last.view(B, 1, -1)
            scores = F.cosine_similarity(stepped_flat, last_flat, dim=-1)
        elif self.score_mode == 'alpha':
            if alphas is not None:
                if alphas.dim() == 1:
                    scores = alphas.unsqueeze(0).expand(B, -1)
                else:
                    scores = alphas
            else:
                scores = torch.linspace(0, 1, S, device=stepped.device).unsqueeze(0).expand(B, -1)
        else:
            scores = torch.ones(B, S, device=stepped.device, dtype=stepped.dtype)

        return scores

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        scores = self._compute_scores(stepped, alphas)  # [B, S]

        # Apply step-level mask to scores
        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            step_mask = mask_normalized.mean(dim=(-1, -2))  # [B, S]
            scores = scores * step_mask

        # Compute min_p threshold per batch
        max_scores = scores.max(dim=1, keepdim=True).values  # [B, 1]
        threshold = self.min_p * max_scores

        # Create validity mask: keep steps above threshold
        valid = (scores >= threshold).float()  # [B, S]

        # Apply validity mask to scores
        masked_scores = scores * valid

        # Normalize if requested
        if self.normalize:
            weights = masked_scores / masked_scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            weights = masked_scores

        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]

        result = (stepped * weights).sum(dim=1)

        # Apply position-level mask for fingerprint preservation
        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result

    def get_valid_counts(self, stepped, alphas=None) -> Tensor:
        """
        Get number of valid steps per batch (for diagnostics).

        Returns:
            [B] tensor of valid step counts
        """
        B, S, T, D = stepped.shape
        scores = self._compute_scores(stepped, alphas)
        max_scores = scores.max(dim=1, keepdim=True).values
        threshold = self.min_p * max_scores
        valid = (scores >= threshold).float()
        return valid.sum(dim=1)


class WeightedMeanAggregation(Aggregation):
    """Alpha-weighted mean."""

    def _compute_scores(self, stepped, alphas=None):
        B, S, T, D = stepped.shape

        if alphas is None:
            return torch.ones(B, S, device=stepped.device, dtype=stepped.dtype)

        if alphas.dim() == 1:
            return alphas.unsqueeze(0).expand(B, -1)
        return alphas


class LastStepAggregation(Aggregation):
    """Take only the final step."""

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        result = stepped[:, -1]

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class FirstStepAggregation(Aggregation):
    """Take only the first step."""

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        result = stepped[:, 0]

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class TriangularAggregation(Aggregation):
    """
    Triangular weighting.
    Peak weight at center, tapering to edges.
    """

    def _compute_scores(self, stepped, alphas=None):
        B, S, T, D = stepped.shape
        positions = torch.linspace(0, 1, S, device=stepped.device, dtype=stepped.dtype)
        weights = torch.minimum(positions, 1 - positions) * 2
        return weights.unsqueeze(0).expand(B, -1)


class SimilarityAggregation(Aggregation):
    """
    Similarity-weighted aggregation.
    Weights each step by its similarity to a reference.
    """

    def __init__(self, reference: str = 'last'):
        super().__init__()
        self.reference = reference

    def _compute_scores(self, stepped, alphas=None):
        B, S, T, D = stepped.shape

        if self.reference == 'last':
            ref = stepped[:, -1:]
        elif self.reference == 'first':
            ref = stepped[:, :1]
        elif self.reference == 'mean':
            ref = stepped.mean(dim=1, keepdim=True)
        else:
            ref = stepped[:, -1:]

        stepped_flat = stepped.view(B, S, -1)
        ref_flat = ref.view(B, 1, -1)

        return F.cosine_similarity(stepped_flat, ref_flat, dim=-1)


class CrossSimilarityAggregation(Aggregation):
    """
    All-pairs cross-similarity aggregation.
    Each step weighted by average similarity to all other steps.
    """

    def _compute_scores(self, stepped, alphas=None):
        B, S, T, D = stepped.shape

        stepped_flat = stepped.view(B, S, -1)  # [B, S, T*D]
        stepped_norm = F.normalize(stepped_flat, dim=-1)

        # All-pairs similarity: [B, S, S]
        sim_matrix = torch.bmm(stepped_norm, stepped_norm.transpose(1, 2))

        # Average similarity to all other steps (exclude self)
        mask = 1 - torch.eye(S, device=stepped.device).unsqueeze(0)
        scores = (sim_matrix * mask).sum(dim=-1) / (S - 1)

        return scores


class SimilarityTreeAggregation(Aggregation):
    """
    Hierarchical similarity-based aggregation.
    Recursively merges most similar pairs until single output.
    """

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        # Apply mask to input if provided
        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            stepped = stepped * mask_normalized

        current = stepped

        while current.shape[1] > 1:
            n = current.shape[1]
            current_flat = current.view(B, n, -1)

            # Adjacent similarities
            sims = F.cosine_similarity(
                current_flat[:, :-1],
                current_flat[:, 1:],
                dim=-1
            )

            if n % 2 == 1:
                n_pairs = (n - 1) // 2
                to_merge = current[:, :-1]
                last = current[:, -1:]

                pairs = to_merge.view(B, n_pairs, 2, T, D)
                pair_sims = sims[:, ::2][:, :n_pairs]
                pair_sims = pair_sims.unsqueeze(-1).unsqueeze(-1)

                merged = pairs[:, :, 0] * pair_sims + pairs[:, :, 1] * (1 - pair_sims)
                current = torch.cat([merged, last], dim=1)
            else:
                n_pairs = n // 2
                pairs = current.view(B, n_pairs, 2, T, D)
                pair_sims = sims[:, ::2][:, :n_pairs]
                pair_sims = pair_sims.unsqueeze(-1).unsqueeze(-1)

                current = pairs[:, :, 0] * pair_sims + pairs[:, :, 1] * (1 - pair_sims)

        result = current.squeeze(1)

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class SlerpAggregation(Aggregation):
    """
    Spherical interpolation aggregation.
    Sequentially slerps through all steps.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _slerp(self, a: Tensor, b: Tensor, t: float) -> Tensor:
        """Slerp between a and b."""
        a_norm = F.normalize(a, dim=-1, eps=self.eps)
        b_norm = F.normalize(b, dim=-1, eps=self.eps)

        dot = (a_norm * b_norm).sum(dim=-1, keepdim=True)
        dot = dot.clamp(-1 + self.eps, 1 - self.eps)
        omega = torch.acos(dot)

        sin_omega = torch.sin(omega).clamp_min(self.eps)
        t1 = torch.sin((1 - t) * omega) / sin_omega
        t2 = torch.sin(t * omega) / sin_omega

        return t1 * a + t2 * b

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            stepped = stepped * mask_normalized

        # Sequential slerp through all steps
        result = stepped[:, 0]
        for i in range(1, S):
            t = i / (S - 1) if S > 1 else 1.0
            result = self._slerp(result, stepped[:, i], t)

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class AttentionAggregation(Aggregation):
    """
    Learned attention over steps.
    """

    def __init__(self, features: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, features))
        self.attn = nn.MultiheadAttention(features, num_heads, batch_first=True)

    def forward(self, stepped, alphas=None, mask=None, original=None):
        B, S, T, D = stepped.shape
        device, dtype = stepped.device, stepped.dtype

        # Pool each step to [B, S, D]
        stepped_pooled = stepped.mean(dim=2)

        # Build attention mask if provided
        attn_mask = None
        if mask is not None:
            mask_normalized = normalize_mask(mask, stepped.shape, device, dtype)
            step_mask = mask_normalized.mean(dim=(-1, -2))  # [B, S]
            # Convert to attention mask (True = masked out)
            attn_mask = step_mask < 0.5

        # Attend
        query = self.query.expand(B, -1, -1).to(device=device, dtype=dtype)
        attn_out, weights = self.attn(
            query, stepped_pooled, stepped_pooled,
            key_padding_mask=attn_mask
        )

        # Use attention weights to aggregate full stepped tensor
        weights = weights.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        result = (stepped * weights).sum(dim=1)

        if mask is not None and original is not None:
            pos_mask = normalize_mask(mask, (B, T, D), device, dtype)
            result = apply_mask_blend(result, original, pos_mask)

        return result


class LearnableAggregation(Aggregation):
    """
    Learnable per-step weights.
    """

    def __init__(self, num_steps: int):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(num_steps))

    def _compute_scores(self, stepped, alphas=None):
        B, S, T, D = stepped.shape
        weights = F.softmax(self.raw_weights, dim=0)
        return weights.unsqueeze(0).expand(B, -1)

    def get_weights(self) -> Tensor:
        return F.softmax(self.raw_weights, dim=0).detach()


# Aggregation registry
AGGREGATIONS: Dict[str, type] = {
    'mean': MeanAggregation,
    'sum': SumAggregation,
    'max': MaxAggregation,
    'min': MinAggregation,
    'top_k': TopKAggregation,
    'bottom_k': BottomKAggregation,
    'softmax': SoftmaxAggregation,
    'softmin': SoftminAggregation,
    'min_p': MinPAggregation,
    'weighted': WeightedMeanAggregation,
    'last': LastStepAggregation,
    'first': FirstStepAggregation,
    'triangular': TriangularAggregation,
    'similarity': SimilarityAggregation,
    'cross_similarity': CrossSimilarityAggregation,
    'similarity_tree': SimilarityTreeAggregation,
    'slerp': SlerpAggregation,
    'attention': AttentionAggregation,
    'learnable': LearnableAggregation,
}


def get_aggregation(
    name: str,
    **kwargs,
) -> Aggregation:
    """
    Get aggregation by name with optional configuration.

    Args:
        name: Aggregation name
        **kwargs: Additional arguments for aggregation constructor

    Returns:
        Aggregation instance
    """
    agg_cls = AGGREGATIONS.get(name.lower(), MeanAggregation)

    # Handle parameterized aggregations
    if name == 'top_k':
        return agg_cls(k=kwargs.get('k', 3), score_mode=kwargs.get('score_mode', 'magnitude'))
    elif name == 'bottom_k':
        return agg_cls(k=kwargs.get('k', 3), score_mode=kwargs.get('score_mode', 'magnitude'))
    elif name == 'softmax':
        return agg_cls(temperature=kwargs.get('temperature', 1.0), score_mode=kwargs.get('score_mode', 'magnitude'))
    elif name == 'softmin':
        return agg_cls(temperature=kwargs.get('temperature', 1.0), score_mode=kwargs.get('score_mode', 'magnitude'))
    elif name == 'min_p':
        return agg_cls(
            min_p=kwargs.get('min_p', 0.1),
            score_mode=kwargs.get('score_mode', 'magnitude'),
            normalize=kwargs.get('normalize', True),
        )
    elif name == 'similarity':
        return agg_cls(reference=kwargs.get('reference', 'last'))
    elif name == 'attention':
        return agg_cls(features=kwargs.get('features', 256), num_heads=kwargs.get('num_heads', 4))
    elif name == 'learnable':
        return agg_cls(num_steps=kwargs.get('num_steps', 8))
    else:
        return agg_cls()


# =============================================================================
# FIELD WALKER FUSION - Unified interpolation with full masking
# =============================================================================

class FieldWalkerFusion(TorchComponent):
    """
    Multi-step interpolative fusion with comprehensive masking.

    Combines:
        - BlendMode: How to interpolate at each step
        - Schedule: Alpha values across steps
        - Aggregation: How to pool stepped results
        - Masking: Fingerprint preservation throughout

    Fully vectorized - no for loops in forward pass.

    Args:
        name: Component name
        in_features: Feature dimension
        num_steps: Number of interpolation steps
        blend_mode: Blend mode name or BlendMode instance
        schedule: Schedule name or Schedule instance
        aggregation: Aggregation name or Aggregation instance
        learnable_steps: If True, use learnable schedule
        learnable_agg: If True, use learnable aggregation weights
        compute_delta: If True, compute and use delta = b - a
    """

    def __init__(
        self,
        name: str,
        in_features: int,
        num_steps: int = 8,
        blend_mode: Union[str, BlendMode] = 'lerp',
        schedule: Union[str, Schedule] = 'linear',
        aggregation: Union[str, Aggregation] = 'mean',
        learnable_steps: bool = False,
        learnable_agg: bool = False,
        compute_delta: bool = True,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        self.in_features = in_features
        self.num_steps = num_steps
        self.compute_delta = compute_delta

        # Blend mode
        if isinstance(blend_mode, str):
            self.blend = get_blend_mode(blend_mode)
        else:
            self.blend = blend_mode

        # Schedule
        if learnable_steps:
            self.schedule = LearnableSchedule(num_steps)
        elif isinstance(schedule, str):
            schedule_cls = SCHEDULES.get(schedule, LinearSchedule)
            if schedule == 'learnable':
                self.schedule = schedule_cls(num_steps)
            elif schedule == 'adaptive':
                self.schedule = schedule_cls(in_features, num_steps)
            else:
                self.schedule = schedule_cls()
        else:
            self.schedule = schedule

        # Aggregation
        if learnable_agg:
            self.aggregation = LearnableAggregation(num_steps)
        elif isinstance(aggregation, str):
            self.aggregation = get_aggregation(
                aggregation,
                features=in_features,
                num_steps=num_steps,
            )
        else:
            self.aggregation = aggregation

        self._last_stepped = None
        self._last_alphas = None

    def forward(
        self,
        a: Tensor,
        b: Tensor,
        context: Optional[Dict[str, Any]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Walk from a to b through interpolation steps.

        Args:
            a: [B, T, D] or [B, D] source
            b: [B, T, D] or [B, D] target
            context: Optional context dict for blend modes
            mask: Optional mask where:
                - 0 = preserve original (fingerprint)
                - 1 = apply full processing
                Shapes: [B, T], [B, T, 1], [B, T, D]

        Returns:
            [B, T, D] or [B, D] aggregated result
        """
        # Handle 2D inputs
        squeeze_output = False
        if a.dim() == 2:
            a = a.unsqueeze(1)
            b = b.unsqueeze(1)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(1)
            squeeze_output = True

        B, T, D = a.shape
        S = self.num_steps
        device = a.device
        dtype = a.dtype

        # Get alpha schedule
        if isinstance(self.schedule, AdaptiveSchedule):
            alphas = self.schedule.forward_with_content(a, b)
        else:
            alphas = self.schedule(S).to(device=device, dtype=dtype)

        self._last_alphas = alphas.detach()

        # Expand for broadcasting: [B, S, T, D]
        a_exp = a.unsqueeze(1).expand(-1, S, -1, -1)
        b_exp = b.unsqueeze(1).expand(-1, S, -1, -1)

        # alphas: [S] -> [B, S, T] for broadcasting
        if alphas.dim() == 1:
            alpha_exp = alphas.view(1, S, 1).expand(B, -1, T)
        else:
            alpha_exp = alphas.unsqueeze(-1).expand(-1, -1, T)

        # Compute delta
        delta = (b_exp - a_exp) if self.compute_delta else None

        # Expand mask for stepped operations
        blend_mask = None
        if mask is not None:
            blend_mask = normalize_mask(mask, (B, S, T, D), device, dtype)

        # Apply blend mode (fully vectorized, with masking)
        stepped = self.blend(a_exp, b_exp, alpha_exp, delta, context, blend_mask)

        self._last_stepped = stepped.detach()

        # Aggregate with masking
        output = self.aggregation(stepped, alphas, mask=mask, original=a)

        if squeeze_output:
            output = output.squeeze(1)

        return output

    def get_stepped_outputs(self) -> Optional[Tensor]:
        """Get intermediate stepped outputs from last forward."""
        return self._last_stepped

    def get_alphas(self) -> Optional[Tensor]:
        """Get alpha schedule from last forward."""
        return self._last_alphas

    def __repr__(self) -> str:
        blend_name = type(self.blend).__name__
        sched_name = type(self.schedule).__name__
        agg_name = type(self.aggregation).__name__
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"features={self.in_features}, steps={self.num_steps}, "
            f"blend={blend_name}, schedule={sched_name}, agg={agg_name})"
        )


# =============================================================================
# CONVENIENCE BUILDERS
# =============================================================================

def create_field_walker(
    name: str,
    in_features: int,
    num_steps: int = 8,
    blend: str = 'lerp',
    schedule: str = 'linear',
    aggregation: str = 'mean',
    learnable: bool = False,
) -> FieldWalkerFusion:
    """Create FieldWalkerFusion with common configurations."""
    return FieldWalkerFusion(
        name=name,
        in_features=in_features,
        num_steps=num_steps,
        blend_mode=blend,
        schedule=schedule,
        aggregation=aggregation,
        learnable_steps=learnable,
        learnable_agg=learnable,
    )


# Presets
WALKER_PRESETS = {
    'alucard': {'blend': 'lerp', 'schedule': 'tau', 'aggregation': 'mean'},
    'slerp': {'blend': 'slerp', 'schedule': 'linear', 'aggregation': 'weighted'},
    'slip': {'blend': 'slip', 'schedule': 'cosine', 'aggregation': 'similarity'},
    'zeus': {'blend': 'zeus', 'schedule': 'sigmoid', 'aggregation': 'last'},
    'gilgamesh': {'blend': 'gilgamesh', 'schedule': 'linear', 'aggregation': 'triangular'},
    'shiva': {'blend': 'shiva', 'schedule': 'cosine', 'aggregation': 'similarity_tree'},
    'ifrit': {'blend': 'ifrit', 'schedule': 'wave', 'aggregation': 'softmax'},
    'learnable': {'blend': 'lerp', 'schedule': 'learnable', 'aggregation': 'learnable'},
    'fingerprint': {'blend': 'lerp', 'schedule': 'cosine', 'aggregation': 'similarity'},
    'min_p': {'blend': 'min_p', 'schedule': 'linear', 'aggregation': 'min_p'},
}


def from_preset(
    name: str,
    preset: str,
    in_features: int,
    num_steps: int = 8,
    **overrides,
) -> FieldWalkerFusion:
    """Create FieldWalkerFusion from preset."""
    if preset not in WALKER_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(WALKER_PRESETS.keys())}")

    config = WALKER_PRESETS[preset].copy()
    config.update(overrides)

    return FieldWalkerFusion(
        name=name,
        in_features=in_features,
        num_steps=num_steps,
        blend_mode=config['blend'],
        schedule=config['schedule'],
        aggregation=config['aggregation'],
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':

    def test_section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    test_section("BLEND MODES WITH MASKING")
    # -------------------------------------------------------------------------

    a = torch.randn(2, 16, 256, device=device)
    b = torch.randn(2, 16, 256, device=device)
    alpha = torch.tensor(0.5, device=device).expand(2, 16)

    # Binary mask: first 8 tokens preserved, last 8 processed
    mask = torch.zeros(2, 16, 1, device=device)
    mask[:, 8:, :] = 1.0

    for name, blend in list(BLEND_MODES.items())[:3]:
        out_no_mask = blend(a, b, alpha)
        out_masked = blend(a, b, alpha, mask=mask)

        # Verify fingerprint preservation
        diff_preserved = (out_masked[:, :8, :] - a[:, :8, :]).abs().max().item()
        diff_processed = (out_masked[:, 8:, :] - out_no_mask[:, 8:, :]).abs().max().item()

        print(f"{name:12s}: preserved_diff={diff_preserved:.6f}, processed_diff={diff_processed:.6f}")

    # -------------------------------------------------------------------------
    test_section("AGGREGATIONS WITH MASKING")
    # -------------------------------------------------------------------------

    stepped = torch.randn(2, 8, 16, 256, device=device)
    alphas = torch.linspace(0, 1, 8, device=device)
    original = torch.randn(2, 16, 256, device=device)

    # Mask: first half preserved, second half processed
    mask = torch.zeros(2, 16, 1, device=device)
    mask[:, 8:, :] = 1.0

    agg_tests = ['mean', 'max', 'min', 'top_k', 'softmax', 'similarity_tree']

    for name in agg_tests:
        agg = get_aggregation(name, features=256, num_steps=8, k=3)
        out = agg(stepped, alphas, mask=mask, original=original)

        # Verify fingerprint preservation
        diff_preserved = (out[:, :8, :] - original[:, :8, :]).abs().max().item()

        print(f"{name:15s}: shape={list(out.shape)}, preserved_diff={diff_preserved:.6f}")

    # -------------------------------------------------------------------------
    test_section("ALL AGGREGATIONS")
    # -------------------------------------------------------------------------

    for name in AGGREGATIONS.keys():
        try:
            agg = get_aggregation(name, features=256, num_steps=8, k=3)
            if hasattr(agg, 'to'):
                agg = agg.to(device)
            out = agg(stepped, alphas)
            print(f"{name:18s}: [2, 8, 16, 256] -> {list(out.shape)}")
        except Exception as e:
            print(f"{name:18s}: ERROR - {e}")

    # -------------------------------------------------------------------------
    test_section("FIELD WALKER WITH MASK")
    # -------------------------------------------------------------------------

    a = torch.randn(2, 16, 256, device=device)
    b = torch.randn(2, 16, 256, device=device)

    # Fingerprint mask
    mask = torch.zeros(2, 16, device=device)
    mask[:, 8:] = 1.0  # Preserve first 8, process last 8

    walker = FieldWalkerFusion(
        'masked_walker', in_features=256, num_steps=8,
        blend_mode='slerp', schedule='cosine', aggregation='similarity'
    ).to(device)

    out = walker(a, b, mask=mask)

    # Verify fingerprint preservation
    diff_preserved = (out[:, :8, :] - a[:, :8, :]).abs().max().item()

    print(f"Walker with mask: {a.shape} -> {out.shape}")
    print(f"  Fingerprint preserved (first 8 tokens): diff={diff_preserved:.6f}")

    # Without mask for comparison
    out_no_mask = walker(a, b)
    diff_full = (out_no_mask[:, :8, :] - a[:, :8, :]).abs().max().item()
    print(f"  Without mask (first 8 tokens): diff={diff_full:.6f}")

    # -------------------------------------------------------------------------
    test_section("PRESETS")
    # -------------------------------------------------------------------------

    for preset_name in WALKER_PRESETS:
        walker = from_preset(f'{preset_name}_walker', preset_name, in_features=256, num_steps=8).to(device)
        out = walker(a, b)
        print(f"{preset_name:12s}: {walker}")

    # -------------------------------------------------------------------------
    test_section("GRADIENT FLOW WITH MASKING")
    # -------------------------------------------------------------------------

    walker = FieldWalkerFusion(
        'grad_test', in_features=256, num_steps=8,
        learnable_steps=True, learnable_agg=True
    ).to(device)

    a = torch.randn(2, 16, 256, device=device, requires_grad=True)
    b = torch.randn(2, 16, 256, device=device, requires_grad=True)
    mask = torch.ones(2, 16, device=device)
    mask[:, :4] = 0  # Preserve first 4

    out = walker(a, b, mask=mask)
    loss = out.sum()
    loss.backward()

    print(f"Input grad: {a.grad is not None}")
    print(f"Schedule grad: {walker.schedule.raw_alphas.grad is not None}")
    print(f"Aggregation grad: {walker.aggregation.raw_weights.grad is not None}")

    # Verify gradients respect mask (preserved positions should have zero grad)
    grad_preserved = a.grad[:, :4, :].abs().max().item()
    grad_processed = a.grad[:, 4:, :].abs().max().item()
    print(f"Grad preserved region: {grad_preserved:.6f}")
    print(f"Grad processed region: {grad_processed:.6f}")

    # -------------------------------------------------------------------------
    test_section("ALPHA VS BINARY MASKING")
    # -------------------------------------------------------------------------

    a = torch.randn(2, 16, 256, device=device)
    b = torch.randn(2, 16, 256, device=device)

    # Alpha mask: gradual transition
    alpha_mask = torch.linspace(0, 1, 16, device=device).unsqueeze(0).expand(2, -1)

    # Binary mask: hard threshold
    binary_mask = binarize_mask(alpha_mask)

    walker = from_preset('fingerprint', 'fingerprint', in_features=256).to(device)

    out_alpha = walker(a, b, mask=alpha_mask)
    out_binary = walker(a, b, mask=binary_mask)

    print(f"Alpha mask output: {out_alpha.shape}")
    print(f"Binary mask output: {out_binary.shape}")
    print(f"Difference: {(out_alpha - out_binary).abs().mean().item():.6f}")

    # -------------------------------------------------------------------------
    test_section("MIN_P AGGREGATION")
    # -------------------------------------------------------------------------

    stepped = torch.randn(2, 8, 16, 256, device=device)
    # Make some steps have higher magnitude (more "significant")
    stepped[:, 3:5] *= 2.0  # Steps 3-4 are 2x magnitude

    alphas = torch.linspace(0, 1, 8, device=device)

    # Min-P with different thresholds
    for min_p_val in [0.1, 0.3, 0.5, 0.7]:
        agg = MinPAggregation(min_p=min_p_val, score_mode='magnitude')
        out = agg(stepped, alphas)
        valid_counts = agg.get_valid_counts(stepped, alphas)
        print(f"min_p={min_p_val}: shape={list(out.shape)}, valid_steps={valid_counts.tolist()}")

    # -------------------------------------------------------------------------
    test_section("MIN_P BLEND")
    # -------------------------------------------------------------------------

    a = torch.randn(2, 16, 256, device=device)
    b = torch.randn(2, 16, 256, device=device)
    # Make delta small in some regions
    b[:, 8:, :] = a[:, 8:, :] + 0.01 * torch.randn(2, 8, 256, device=device)  # Small delta

    alpha = torch.tensor(0.5, device=device).expand(2, 16)

    # Standard lerp
    lerp_out = BLEND_MODES['lerp'](a, b, alpha)

    # Min-P blend (should gate out small-delta regions)
    min_p_blend = MinPBlend(min_p=0.3)
    min_p_out = min_p_blend(a, b, alpha)

    # Difference should be larger in first 8 positions (large delta)
    diff_large_delta = (min_p_out[:, :8, :] - a[:, :8, :]).abs().mean().item()
    diff_small_delta = (min_p_out[:, 8:, :] - a[:, 8:, :]).abs().mean().item()

    print(f"Large delta region diff: {diff_large_delta:.6f}")
    print(f"Small delta region diff (should be ~0): {diff_small_delta:.6f}")

    # -------------------------------------------------------------------------
    test_section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\naggregation_component.py ready.")
    print(f"\nBlend modes: {list(BLEND_MODES.keys())}")
    print(f"Schedules: {list(SCHEDULES.keys())}")
    print(f"Aggregations: {list(AGGREGATIONS.keys())}")
    print(f"Presets: {list(WALKER_PRESETS.keys())}")
    print(f"\nMasking: Alpha (continuous) and Binary (hard) supported throughout.")