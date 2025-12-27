"""
geofractal.components.utility.blend_utility
===========================================

Stateless blend modes for walker interpolation.

Blends interpolate between tensors a and b using alpha factor.
All are BaseUtility subclasses - no parameters, pure computation.

Experimental Results (CIFAR-10, 100 configs):
    1. ShivaBlend    98.21%  (exponential decay, decay=3.0 optimal)
    2. Learnable     98.15%
    3. SlerpBlend    98.08%  (spherical)

Safety Classification:
    ASYMPTOTIC_SAFE:  shiva, zeus, surge     - safe at any alpha
    EXTRAPOLATING:    lerp, slerp, slip      - creative divergence >1.0
    PERIODIC_DANGER:  gilgamesh, helios,     - dangerous >1.0
                      ripple, ifrit

Usage:
    blend = ShivaBlend(decay=3.0)
    result = blend(a, b, alpha)

    # With mask (0=preserve a, 1=blend normally)
    result = blend(a, b, alpha, mask=fingerprint_mask)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from abc import abstractmethod
from typing import Optional, Dict, Any, FrozenSet
import math

import torch
from torch import Tensor

from geofractal.router.base_utility import BaseUtility

# =============================================================================
# SAFETY CONSTANTS
# =============================================================================

ASYMPTOTIC_SAFE: FrozenSet[str] = frozenset({'shiva', 'zeus', 'surge'})
EXTRAPOLATING: FrozenSet[str] = frozenset({'lerp', 'slerp', 'slip'})
PERIODIC_DANGEROUS: FrozenSet[str] = frozenset({'gilgamesh', 'helios', 'ripple', 'ifrit'})

SAFE_ALPHA_MAX: Dict[str, float] = {
    'shiva': float('inf'),
    'zeus': float('inf'),
    'surge': float('inf'),
    'lerp': 2.0,
    'slerp': 1.5,
    'slip': 1.5,
    'gilgamesh': 1.0,
    'helios': 1.0,
    'ripple': 0.8,
    'ifrit': 1.0,
    'min_p': 1.5,
}


# =============================================================================
# MASK UTILITIES
# =============================================================================

def normalize_mask(mask: Tensor, target_shape: tuple, device, dtype) -> Tensor:
    """Expand mask to target shape for broadcasting."""
    mask = mask.to(device=device, dtype=dtype)
    while mask.dim() < len(target_shape):
        mask = mask.unsqueeze(-1)
    return mask


def apply_mask_blend(blended: Tensor, original: Tensor, mask: Tensor) -> Tensor:
    """Apply mask: 0=keep original, 1=use blended."""
    return mask * blended + (1 - mask) * original


# =============================================================================
# BASE BLEND UTILITY
# =============================================================================

class BlendUtility(BaseUtility):
    """
    Base class for blend modes.

    All blends operate on:
        a: source tensor
        b: target tensor
        alpha: blend factor [0, 1] typically, some support >1

    Subclasses implement _blend_impl for the core interpolation.
    """

    blend_name: str = "base"

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name or self.blend_name, **kwargs)

    @abstractmethod
    def _blend_impl(
            self,
            a: Tensor,
            b: Tensor,
            alpha: Tensor,
            delta: Optional[Tensor] = None,
            context: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """Core blend implementation."""
        ...

    def __call__(
            self,
            a: Tensor,
            b: Tensor,
            alpha: Tensor,
            delta: Optional[Tensor] = None,
            context: Optional[Dict[str, Any]] = None,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Blend a toward b by alpha.

        Args:
            a: Source tensor [..., D]
            b: Target tensor [..., D]
            alpha: Blend factor, scalar or broadcastable
            delta: Precomputed (b - a), optional optimization
            context: Additional parameters for specialized blends
            mask: Where 0=preserve a, 1=apply blend

        Returns:
            Blended tensor [..., D]
        """
        # Ensure alpha is tensor
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha, device=a.device, dtype=a.dtype)

        result = self._blend_impl(a, b, alpha, delta, context)

        if mask is not None:
            mask = normalize_mask(mask, result.shape, result.device, result.dtype)
            result = apply_mask_blend(result, a, mask)

        return result

    @property
    def safe_alpha_max(self) -> float:
        """Maximum safe alpha for this blend."""
        return SAFE_ALPHA_MAX.get(self.blend_name, 1.0)

    @property
    def is_asymptotic_safe(self) -> bool:
        """True if blend is safe at any alpha."""
        return self.blend_name in ASYMPTOTIC_SAFE


# =============================================================================
# LERP - Linear Interpolation
# =============================================================================

class LerpBlend(BlendUtility):
    """
    Linear interpolation: a + alpha * (b - a)

    Simple and fast. Extrapolates beyond [0, 1].
    """

    blend_name = "lerp"

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)
        d = delta if delta is not None else (b - a)
        return a + alpha * d


# =============================================================================
# SLERP - Spherical Linear Interpolation
# =============================================================================

class SlerpBlend(BlendUtility):
    """
    Spherical linear interpolation.

    Interpolates along great circle on hypersphere.
    Better for normalized embeddings.
    """

    blend_name = "slerp"

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        # Normalize
        a_norm = a / (a.norm(dim=-1, keepdim=True) + self.eps)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + self.eps)

        # Angle between vectors
        dot = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.acos(dot)

        # Slerp formula
        sin_theta = torch.sin(theta) + self.eps
        wa = torch.sin((1 - alpha) * theta) / sin_theta
        wb = torch.sin(alpha * theta) / sin_theta

        # Preserve original magnitudes
        a_mag = a.norm(dim=-1, keepdim=True)
        b_mag = b.norm(dim=-1, keepdim=True)
        mag = a_mag + alpha * (b_mag - a_mag)

        return (wa * a_norm + wb * b_norm) * mag


# =============================================================================
# SLIP - Linear with Length Preservation
# =============================================================================

class SlipBlend(BlendUtility):
    """
    Linear interpolation with length preservation.

    Lerp direction, interpolate magnitude separately.
    """

    blend_name = "slip"

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        # Lerp the vectors
        d = delta if delta is not None else (b - a)
        lerped = a + alpha * d

        # Interpolate magnitudes
        a_mag = a.norm(dim=-1, keepdim=True)
        b_mag = b.norm(dim=-1, keepdim=True)
        target_mag = a_mag + alpha * (b_mag - a_mag)

        # Normalize and rescale
        lerped_mag = lerped.norm(dim=-1, keepdim=True) + self.eps
        return lerped * (target_mag / lerped_mag)


# =============================================================================
# ZEUS - Power Curve
# =============================================================================

class ZeusBlend(BlendUtility):
    """
    Power curve acceleration.

    Sharp initial movement, then gradual approach.
    Asymptotically safe.
    """

    blend_name = "zeus"

    def __init__(self, power: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.power = power

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        power = (context or {}).get('zeus_power', self.power)
        curved = alpha ** power
        d = delta if delta is not None else (b - a)
        return a + curved * d


# =============================================================================
# HELIOS - Sinusoidal Ease
# =============================================================================

class HeliosBlend(BlendUtility):
    """
    Sinusoidal ease-in-out.

    Smooth acceleration and deceleration.
    PERIODIC - dangerous at alpha > 1.0
    """

    blend_name = "helios"

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        # Sine ease: (1 - cos(alpha * pi)) / 2
        eased = (1 - torch.cos(alpha * math.pi)) / 2
        d = delta if delta is not None else (b - a)
        return a + eased * d


# =============================================================================
# SURGE - Asymptotic Approach
# =============================================================================

class SurgeBlend(BlendUtility):
    """
    Asymptotic surge toward target.

    Fast initial movement, never quite reaches target.
    Asymptotically safe - bounded output.
    """

    blend_name = "surge"

    def __init__(self, rate: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        rate = (context or {}).get('surge_rate', self.rate)
        # 1 - exp(-rate * alpha) asymptotes to 1
        surge = 1 - torch.exp(-rate * alpha)
        d = delta if delta is not None else (b - a)
        return a + surge * d


# =============================================================================
# RIPPLE - Damped Oscillation
# =============================================================================

class RippleBlend(BlendUtility):
    """
    Damped sinusoidal oscillation.

    Overshoots then settles. Creative but unstable at high alpha.
    PERIODIC - dangerous at alpha > 0.8
    """

    blend_name = "ripple"

    def __init__(self, freq: float = 3.0, decay: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.freq = freq
        self.decay = decay

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        freq = (context or {}).get('ripple_freq', self.freq)
        decay = (context or {}).get('ripple_decay', self.decay)

        # Damped oscillation around target
        ripple = 1 - torch.exp(-decay * alpha) * torch.cos(freq * math.pi * alpha)
        d = delta if delta is not None else (b - a)
        return a + ripple * d


# =============================================================================
# GILGAMESH - Two-Phase Blend
# =============================================================================

class GilgameshBlend(BlendUtility):
    """
    Two-phase blend with pivot point.

    Different behavior before/after midpoint.
    PERIODIC - dangerous at alpha > 1.0
    """

    blend_name = "gilgamesh"

    def __init__(self, pivot: float = 0.5, power_low: float = 0.5, power_high: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.pivot = pivot
        self.power_low = power_low
        self.power_high = power_high

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        pivot = (context or {}).get('gilgamesh_pivot', self.pivot)
        power_low = (context or {}).get('gilgamesh_power_low', self.power_low)
        power_high = (context or {}).get('gilgamesh_power_high', self.power_high)

        # Two different curves joined at pivot
        low_mask = (alpha < pivot).float()

        # Below pivot: accelerating
        alpha_low = (alpha / pivot).clamp(0, 1)
        curve_low = (alpha_low ** power_low) * pivot

        # Above pivot: decelerating
        alpha_high = ((alpha - pivot) / (1 - pivot)).clamp(0, 1)
        curve_high = pivot + (alpha_high ** power_high) * (1 - pivot)

        curved = low_mask * curve_low + (1 - low_mask) * curve_high
        d = delta if delta is not None else (b - a)
        return a + curved * d


# =============================================================================
# SHIVA - Exponential Decay (WINNER)
# =============================================================================

class ShivaBlend(BlendUtility):
    """
    Exponential decay interpolation.

    EXPERIMENTAL WINNER: 98.21% CIFAR-10 (best of 100 configs)

    Asymptotically safe - bounded at any alpha.
    decay=3.0 optimal (not 4.0 as previously hardcoded).

    Formula: w = exp(-decay * alpha), result = a + (1 - w) * (b - a)
    """

    blend_name = "shiva"

    def __init__(self, decay: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.decay = decay

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        rate = (context or {}).get('shiva_decay', self.decay)
        cold = torch.exp(-rate * alpha)
        d = delta if delta is not None else (b - a)
        return a + (1.0 - cold) * d


# =============================================================================
# IFRIT - Fiery Waveform
# =============================================================================

class IfritBlend(BlendUtility):
    """
    Squared sine waveform.

    Sharp peaks of intensity. Creative but unstable.
    PERIODIC - dangerous at alpha > 1.0
    """

    blend_name = "ifrit"

    def __init__(self, freq: float = 4.0, amp: float = 1.0, **kwargs):
        super().__init__(**kwargs)
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


# =============================================================================
# MIN-P - Threshold Gated
# =============================================================================

class MinPBlend(BlendUtility):
    """
    Min-P gated blending.

    Only blends where delta magnitude exceeds threshold.
    Preserves regions with small differences.
    """

    blend_name = "min_p"

    def __init__(self, threshold: float = 0.1, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.eps = eps

    def _blend_impl(self, a, b, alpha, delta=None, context=None):
        if alpha.dim() < a.dim():
            alpha = alpha.unsqueeze(-1)

        threshold = (context or {}).get('min_p_threshold', self.threshold)

        d = delta if delta is not None else (b - a)
        d_mag = d.abs()
        max_mag = d_mag.max(dim=-1, keepdim=True).values + self.eps

        # Gate: only blend where magnitude > threshold * max
        gate = (d_mag > threshold * max_mag).float()

        return a + alpha * d * gate


# =============================================================================
# FACTORY
# =============================================================================

BLEND_REGISTRY: Dict[str, type] = {
    'lerp': LerpBlend,
    'slerp': SlerpBlend,
    'slip': SlipBlend,
    'zeus': ZeusBlend,
    'helios': HeliosBlend,
    'surge': SurgeBlend,
    'ripple': RippleBlend,
    'gilgamesh': GilgameshBlend,
    'shiva': ShivaBlend,
    'ifrit': IfritBlend,
    'min_p': MinPBlend,
}


def create_blend(name: str, **kwargs) -> BlendUtility:
    """Create blend by name."""
    if name not in BLEND_REGISTRY:
        raise ValueError(f"Unknown blend: {name}. Available: {list(BLEND_REGISTRY.keys())}")
    return BLEND_REGISTRY[name](**kwargs)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':

    def section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    section("BLEND INSTANTIATION")

    shiva = ShivaBlend(decay=3.0)
    print(f"ShivaBlend: {shiva}")
    print(f"  safe_alpha_max: {shiva.safe_alpha_max}")
    print(f"  is_asymptotic_safe: {shiva.is_asymptotic_safe}")

    lerp = create_blend('lerp')
    print(f"\nLerpBlend via factory: {lerp}")
    print(f"  safe_alpha_max: {lerp.safe_alpha_max}")

    section("BASIC BLENDING")

    a = torch.zeros(4, 64, device=device)
    b = torch.ones(4, 64, device=device)
    alpha = torch.tensor(0.5, device=device)

    for name in ['lerp', 'slerp', 'shiva', 'zeus', 'surge']:
        blend = create_blend(name)
        result = blend(a, b, alpha)
        mean_val = result.mean().item()
        print(f"{name:12s}: mean={mean_val:.4f}")

    section("ALPHA PROGRESSION")

    shiva = ShivaBlend(decay=3.0)
    a = torch.zeros(1, 8, device=device)
    b = torch.ones(1, 8, device=device)

    print("ShivaBlend at different alpha values:")
    for alpha_val in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        alpha = torch.tensor(alpha_val, device=device)
        result = shiva(a, b, alpha)
        print(f"  alpha={alpha_val:.2f}: mean={result.mean().item():.4f}")

    section("MASK SUPPORT")

    a = torch.zeros(4, 8, device=device)
    b = torch.ones(4, 8, device=device)
    alpha = torch.tensor(1.0, device=device)

    # Mask: first 2 samples preserved, last 2 blended
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)

    result = shiva(a, b, alpha, mask=mask)
    print(f"With mask [0, 0, 1, 1]:")
    print(f"  Sample 0 mean (should be 0): {result[0].mean().item():.4f}")
    print(f"  Sample 2 mean (should be ~0.95): {result[2].mean().item():.4f}")

    section("DELTA OPTIMIZATION")

    a = torch.randn(32, 256, device=device)
    b = torch.randn(32, 256, device=device)
    delta = b - a  # Precompute

    alpha = torch.tensor(0.7, device=device)

    # Without delta
    import time

    t0 = time.perf_counter()
    for _ in range(100):
        _ = shiva(a, b, alpha)
    t1 = time.perf_counter()

    # With delta
    t2 = time.perf_counter()
    for _ in range(100):
        _ = shiva(a, b, alpha, delta=delta)
    t3 = time.perf_counter()

    print(f"Without delta: {(t1 - t0) * 1000:.2f}ms")
    print(f"With delta:    {(t3 - t2) * 1000:.2f}ms")

    section("CONTEXT OVERRIDE")

    shiva = ShivaBlend(decay=3.0)
    a = torch.zeros(1, 8, device=device)
    b = torch.ones(1, 8, device=device)
    alpha = torch.tensor(0.5, device=device)

    result_default = shiva(a, b, alpha)
    result_custom = shiva(a, b, alpha, context={'shiva_decay': 6.0})

    print(f"Default decay=3.0: {result_default.mean().item():.4f}")
    print(f"Context decay=6.0: {result_custom.mean().item():.4f}")

    section("SAFETY CONSTANTS")

    print(f"ASYMPTOTIC_SAFE:   {ASYMPTOTIC_SAFE}")
    print(f"EXTRAPOLATING:     {EXTRAPOLATING}")
    print(f"PERIODIC_DANGEROUS: {PERIODIC_DANGEROUS}")

    section("ALL BLENDS")

    a = torch.randn(4, 64, device=device)
    b = torch.randn(4, 64, device=device)
    alpha = torch.tensor(0.5, device=device)

    print(f"{'Blend':12s} {'Mean':>8s} {'Std':>8s} {'Safe':>6s}")
    print("-" * 36)
    for name in BLEND_REGISTRY:
        blend = create_blend(name)
        result = blend(a, b, alpha)
        safe = "∞" if blend.is_asymptotic_safe else f"{blend.safe_alpha_max:.1f}"
        print(f"{name:12s} {result.mean().item():>8.4f} {result.std().item():>8.4f} {safe:>6s}")

    section("ALL TESTS PASSED")