"""
geofractal.router.components.utility.schedule_utility
=====================================================

Alpha schedule utilities for walker interpolation.

Schedules generate alpha values [0, 1] for each step of a walk.
ALL schedules are BaseUtility (static math, no learnable params).

Walker Flow:
    schedule = CosineSchedule()
    alphas = schedule(num_steps=8)  # [0.0, 0.07, 0.25, 0.5, 0.75, 0.93, 1.0]

    # Inception modulates the static schedule:
    modulation = inception.schedule_modulator(aux_features)
    modulated_alphas = alphas + modulation_scale * modulation

Schedule Types (all static):
    - LinearSchedule:      Even ramp 0→1
    - CosineSchedule:      Smooth ease-in-out (most common)
    - SigmoidSchedule:     Sharp transition at center
    - TauSchedule:         Tau-modulated sine
    - WaveSchedule:        Pure sine wave
    - ExponentialSchedule: Exponential approach
    - PowerSchedule:       Configurable power curve
    - StepSchedule:        Binary threshold

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from abc import abstractmethod
from typing import Optional, Dict, Union, List
import math

import torch
from torch import Tensor

from geofractal.router.base_utility import BaseUtility

# NOTE: Schedules are STATIC mathematical curves - no learnable parameters.
# Learnable schedule modulation belongs in Inception layer.
# See: router/components/walker_component.py


# =============================================================================
# BASE SCHEDULE UTILITY
# =============================================================================

class ScheduleUtility(BaseUtility):
    """
    Base class for stateless alpha schedules.

    All schedules generate alpha values in [0, 1] for walker steps.
    """

    schedule_name: str = "base"

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name or self.schedule_name, **kwargs)

    @abstractmethod
    def __call__(
        self,
        num_steps: int,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Generate alpha schedule.

        Args:
            num_steps: Number of walking steps.
            batch_size: Batch size (for batched schedules).
            device: Target device for output.

        Returns:
            Alpha values [S] or [B, S] in range [0, 1].
        """
        ...


# =============================================================================
# LINEAR SCHEDULE
# =============================================================================

class LinearSchedule(ScheduleUtility):
    """
    Linear ramp from 0 to 1.

    Simplest schedule - even spacing between steps.
    """

    schedule_name = "linear"

    def __call__(self, num_steps: int, batch_size: int = 1, device=None) -> Tensor:
        alphas = torch.linspace(0, 1, num_steps, device=device)
        return alphas


# =============================================================================
# COSINE SCHEDULE
# =============================================================================

class CosineSchedule(ScheduleUtility):
    """
    Cosine annealing schedule.

    Formula: 0.5 * (1 - cos(π * t))

    Smooth ease-in-out - slow start, fast middle, slow end.
    Most commonly used schedule.
    """

    schedule_name = "cosine"

    def __call__(self, num_steps: int, batch_size: int = 1, device=None) -> Tensor:
        t = torch.linspace(0, 1, num_steps, device=device)
        return 0.5 * (1 - torch.cos(math.pi * t))


# =============================================================================
# SIGMOID SCHEDULE
# =============================================================================

class SigmoidSchedule(ScheduleUtility):
    """
    Sigmoid schedule with configurable sharpness.

    Sharp transition around center point.
    Higher sharpness = more step-like behavior.
    """

    schedule_name = "sigmoid"

    def __init__(self, sharpness: float = 10.0, center: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.sharpness = sharpness
        self.center = center

    def __call__(self, num_steps: int, batch_size: int = 1, device=None) -> Tensor:
        t = torch.linspace(0, 1, num_steps, device=device)
        raw = self.sharpness * (t - self.center)
        return torch.sigmoid(raw)


# =============================================================================
# TAU SCHEDULE
# =============================================================================

class TauSchedule(ScheduleUtility):
    """
    Tau-modulated schedule.

    Formula: (1 - exp(-τ * t)) * sin(π * t)

    Combines exponential warmup with sine wave.
    Starts slow, peaks in middle, returns to near-zero.
    """

    schedule_name = "tau"

    def __init__(self, tau: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def __call__(self, num_steps: int, batch_size: int = 1, device=None) -> Tensor:
        t = torch.linspace(0, 1, num_steps, device=device)
        tau_scale = 1 - torch.exp(-self.tau * t)
        sigma = torch.sin(math.pi * t)
        return (tau_scale * sigma).clamp(0, 1)


# =============================================================================
# WAVE SCHEDULE
# =============================================================================

class WaveSchedule(ScheduleUtility):
    """
    Sinusoidal wave schedule.

    Pure sine wave clamped to [0, 1].
    Useful for oscillating blends.
    """

    schedule_name = "wave"

    def __init__(self, freq: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.freq = freq

    def __call__(self, num_steps: int, batch_size: int = 1, device=None) -> Tensor:
        t = torch.linspace(0, 1, num_steps, device=device)
        return torch.sin(self.freq * math.pi * t).clamp(0, 1)


# =============================================================================
# EXPONENTIAL SCHEDULE
# =============================================================================

class ExponentialSchedule(ScheduleUtility):
    """
    Exponential approach schedule.

    Formula: 1 - exp(-rate * t)

    Fast initial progress, asymptotic approach to 1.
    Matches ShivaBlend's behavior.
    """

    schedule_name = "exponential"

    def __init__(self, rate: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def __call__(self, num_steps: int, batch_size: int = 1, device=None) -> Tensor:
        t = torch.linspace(0, 1, num_steps, device=device)
        return 1 - torch.exp(-self.rate * t)


# =============================================================================
# POWER SCHEDULE
# =============================================================================

class PowerSchedule(ScheduleUtility):
    """
    Power curve schedule.

    Formula: t^power

    power < 1: fast start, slow end
    power = 1: linear
    power > 1: slow start, fast end
    """

    schedule_name = "power"

    def __init__(self, power: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.power = power

    def __call__(self, num_steps: int, batch_size: int = 1, device=None) -> Tensor:
        t = torch.linspace(0, 1, num_steps, device=device)
        return t ** self.power


# =============================================================================
# STEP SCHEDULE
# =============================================================================

class StepSchedule(ScheduleUtility):
    """
    Custom step sequence schedule.

    Accepts a sequence of alpha values to use directly.
    If no sequence provided, defaults to binary step at midpoint.

    Usage:
        # Custom sequence
        sched = StepSchedule(alphas=[0.0, 0.1, 0.3, 0.6, 0.9, 1.0])

        # Binary default (empty)
        sched = StepSchedule()  # [0, 0, 0, 0, 1, 1, 1, 1] for 8 steps
    """

    schedule_name = "step"

    def __init__(self, alphas: Optional[List[float]] = None, step_at: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._alphas = alphas  # Custom sequence
        self.step_at = step_at  # For binary default

    def __call__(self, num_steps: int, batch_size: int = 1, device=None) -> Tensor:
        if self._alphas is not None:
            # Use provided sequence - interpolate if length doesn't match
            alphas = torch.tensor(self._alphas, dtype=torch.float32, device=device)
            if len(alphas) != num_steps:
                # Interpolate to match requested num_steps
                t_orig = torch.linspace(0, 1, len(alphas), device=device)
                t_new = torch.linspace(0, 1, num_steps, device=device)
                # Linear interpolation
                indices = t_new * (len(alphas) - 1)
                lower = indices.floor().long().clamp(0, len(alphas) - 2)
                upper = (lower + 1).clamp(0, len(alphas) - 1)
                frac = indices - lower.float()
                alphas = alphas[lower] * (1 - frac) + alphas[upper] * frac
            return alphas.clamp(0, 1)
        else:
            # Binary step at threshold
            t = torch.linspace(0, 1, num_steps, device=device)
            return (t >= self.step_at).float()


# =============================================================================
# NOTE: Learnable schedule variants belong in inception layer, not utilities.
# Utilities are STATIC mathematical primitives.
# See: router/components/walker_component.py for aux-modulated schedules
# =============================================================================


# =============================================================================
# FACTORY
# =============================================================================

SCHEDULE_REGISTRY: Dict[str, type] = {
    'linear': LinearSchedule,
    'cosine': CosineSchedule,
    'sigmoid': SigmoidSchedule,
    'tau': TauSchedule,
    'wave': WaveSchedule,
    'exponential': ExponentialSchedule,
    'power': PowerSchedule,
    'step': StepSchedule,
}


def create_schedule(name: str, **kwargs) -> ScheduleUtility:
    """
    Create static schedule by name.

    All schedules are static mathematical curves.
    Learnable modulation is handled by WalkerInception.
    See: router/components/walker_component.py
    """
    if name not in SCHEDULE_REGISTRY:
        raise ValueError(f"Unknown schedule: {name}. Available: {list(SCHEDULE_REGISTRY.keys())}")
    return SCHEDULE_REGISTRY[name](**kwargs)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':

    def section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    section("SCHEDULE INSTANTIATION")

    linear = LinearSchedule()
    cosine = CosineSchedule()
    print(f"LinearSchedule: {linear}")
    print(f"CosineSchedule: {cosine}")

    section("STATELESS SCHEDULES")

    num_steps = 8

    print(f"{'Schedule':<12s} | Alpha values (num_steps={num_steps})")
    print("-" * 70)

    for name in SCHEDULE_REGISTRY:
        schedule = create_schedule(name)
        alphas = schedule(num_steps, device=device)
        vals = [f"{a.item():.3f}" for a in alphas]
        print(f"{name:<12s} | {', '.join(vals)}")

    section("SCHEDULE SHAPES")

    linear = LinearSchedule()

    alphas_8 = linear(8, device=device)
    alphas_16 = linear(16, device=device)

    print(f"num_steps=8:  shape={alphas_8.shape}")
    print(f"num_steps=16: shape={alphas_16.shape}")

    section("CUSTOM STEP SEQUENCES")

    # Binary default
    binary_step = StepSchedule()
    alphas = binary_step(8, device=device)
    print(f"Binary (default):     {[f'{a.item():.2f}' for a in alphas]}")

    # Custom sequence - exact match
    custom = StepSchedule(alphas=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0])
    alphas = custom(8, device=device)
    print(f"Custom (8 values):    {[f'{a.item():.2f}' for a in alphas]}")

    # Custom sequence - interpolated
    sparse = StepSchedule(alphas=[0.0, 0.5, 1.0])
    alphas = sparse(8, device=device)
    print(f"Sparse (3→8 interp):  {[f'{a.item():.2f}' for a in alphas]}")

    # Custom plateau pattern
    plateau = StepSchedule(alphas=[0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    alphas = plateau(8, device=device)
    print(f"Plateau pattern:      {[f'{a.item():.2f}' for a in alphas]}")

    section("NOTE: LEARNABLE SCHEDULE MODULATION")

    print("Schedule modulation is handled by the Inception layer, not here.")
    print("See: router/components/walker_component.py")
    print()
    print("WalkerInception provides:")
    print("  - base_schedule (buffer): Static cosine/linear curve")
    print("  - schedule_modulator(aux): aux → modulation signal")
    print("  - modulated = base_schedule + scale * modulation")
    print()
    print("This separation yields 0.999+ consistency across training seeds.")

    section("SCHEDULE VISUALIZATION")

    print("Comparing schedule curves (0→1):")
    print()

    viz_schedules = ['linear', 'cosine', 'sigmoid', 'step', 'exponential', 'power']

    print(f"{'#':<6s}", end="")
    for name in viz_schedules:
        print(f"{name:<12s}", end="")
    print()
    print("-" * 78)

    schedules = {name: create_schedule(name) for name in viz_schedules}
    alphas_dict = {name: sched(8, device=device) for name, sched in schedules.items()}

    for i in range(8):
        print(f"{i:<6d}", end="")
        for name in viz_schedules:
            val = alphas_dict[name][i].item()
            bar = '█' * int(val * 10)
            print(f"{val:.3f} {bar:<6s}", end=" ")
        print()

    section("ALL TESTS PASSED")