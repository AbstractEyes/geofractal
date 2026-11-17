"""
geofractal.router.prefab.agatha_v2.oscillator
=============================================

A damped second-order integrator over latent tokens.

Per token, with gains that are constant over the internal steps:

    a  = -2 beta v + omega^2 (x_ref - x) + kappa u
    v' = v + h a
    x' = x + h v'                       (semi-implicit Euler, h = 1 / n)

started at x_0 (the input tokens) with v_0 = 0. The spring pulls toward x_ref
for every omega > 0.

Because the gains do not depend on x, the n-step map is affine. With
Omega = omega^2 and c = Omega x_ref + kappa u it is

    x_n = A_n x_0 + G_n c

where (A_n, G_n) are scalars per sample obtained from an n-step scalar
recurrence. Since x_ref is a fixed point when u = 0, A_n + G_n Omega = 1, and
the same map can be written

    x_n = x_ref + A_n (x_0 - x_ref) + G_n kappa u,

which is the form used here (it returns x_0 exactly when x_ref = x_0 and u = 0).
The stepped loop is kept as the reference and for diagnostics.

Stability. With p = (h omega)^2 and q = 2 h beta the step matrix in the state
(x, h v) is [[1 - p, 1 - q], [-p, 1 - q]], characteristic polynomial
lambda^2 - (2 - p - q) lambda + (1 - q). Both roots lie inside the unit circle
iff 0 < q < 2, p > 0 and p + 2 q < 4. The gain maps below keep every admissible
value inside that region.

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# GAIN MAPS (raw parameter -> bounded gain); used by the model and by the tests
# =============================================================================

def damping_from_raw(raw: Tensor, beta_min: float, beta_max: float) -> Tensor:
    """beta = beta_min + (beta_max - beta_min) * sigmoid(raw)."""
    return beta_min + (beta_max - beta_min) * torch.sigmoid(raw)


def frequency_from_raw(raw: Tensor, omega_max: float) -> Tensor:
    """omega = omega_max * sigmoid(raw)."""
    return omega_max * torch.sigmoid(raw)


def control_gain_from_raw(raw: Tensor) -> Tensor:
    """kappa = softplus(raw) >= 0."""
    return F.softplus(raw)


def step_matrix(beta: Tensor, omega: Tensor, n_steps: int) -> Tensor:
    """The one-step map of (x, h v) for a = -2 beta v - omega^2 x. Shape [..., 2, 2]."""
    h = 1.0 / n_steps
    p = (h * omega) ** 2
    q = 2.0 * h * beta
    row0 = torch.stack([1 - p, 1 - q], dim=-1)
    row1 = torch.stack([-p, 1 - q], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def spectral_radius(beta: Tensor, omega: Tensor, n_steps: int) -> Tensor:
    """Largest eigenvalue modulus of the step matrix, elementwise over the gains."""
    m = step_matrix(beta, omega, n_steps)
    return torch.linalg.eigvals(m).abs().amax(dim=-1)


# =============================================================================
# EXACT FORM
# =============================================================================

def affine_coefficients(beta: Tensor, omega: Tensor, n_steps: int) -> Tuple[Tensor, Tensor]:
    """Scalars (A_n, G_n) with x_n = A_n x_0 + G_n c for x_0 given, v_0 = 0.

    Runs the n-step recurrence on scalars (same dtype as the gains).
    """
    h = 1.0 / n_steps
    big_omega = omega * omega
    damp = 1.0 - 2.0 * beta * h
    ax = torch.ones_like(beta)
    av = torch.zeros_like(beta)
    gx = torch.zeros_like(beta)
    gv = torch.zeros_like(beta)
    for _ in range(n_steps):
        av = damp * av - h * big_omega * ax
        ax = ax + h * av
        gv = damp * gv - h * big_omega * gx + h
        gx = gx + h * gv
    return ax, gx


def _expand(g: Tensor, like: Tensor) -> Tensor:
    while g.dim() < like.dim():
        g = g.unsqueeze(-1)
    return g


def integrate_exact(
    x0: Tensor,
    x_ref: Tensor,
    force: Optional[Tensor],
    beta: Tensor,
    omega: Tensor,
    kappa: Tensor,
    n_steps: int,
) -> Tensor:
    """x_n = x_ref + A_n (x_0 - x_ref) + G_n kappa u (one combination)."""
    a_n, g_n = affine_coefficients(beta, omega, n_steps)
    a_n = _expand(a_n, x0).to(x0.dtype)
    out = x_ref + a_n * (x0 - x_ref)
    if force is not None:
        gk = _expand(g_n * kappa, x0).to(x0.dtype)
        out = out + gk * force
    return out


# =============================================================================
# STEPPED REFERENCE
# =============================================================================

def integrate_stepped(
    x0: Tensor,
    x_ref: Tensor,
    force: Optional[Tensor],
    beta: Tensor,
    omega: Tensor,
    kappa: Tensor,
    n_steps: int,
    spring_sign: float = 1.0,
    return_trajectory: bool = False,
):
    """The step-by-step loop. spring_sign = -1 flips the spring (a test control)."""
    h = 1.0 / n_steps
    beta_e = _expand(beta, x0).to(x0.dtype)
    big_omega = _expand(omega * omega, x0).to(x0.dtype)
    x = x0
    v = torch.zeros_like(x0)
    kf = None
    if force is not None:
        kf = _expand(kappa, x0).to(x0.dtype) * force
    traj: List[Tuple[Tensor, Tensor]] = [(x, v)] if return_trajectory else []
    for _ in range(n_steps):
        a = -2.0 * beta_e * v + spring_sign * big_omega * (x_ref - x)
        if kf is not None:
            a = a + kf
        v = v + h * a
        x = x + h * v
        if return_trajectory:
            traj.append((x, v))
    if return_trajectory:
        return x, traj
    return x


def energy(x: Tensor, v: Tensor, x_ref: Tensor, omega: Tensor) -> Tensor:
    """Per-sample oscillator energy 0.5 |v|^2 + 0.5 omega^2 |x - x_ref|^2 (diagnostics only).

    omega has shape [B]; x, v, x_ref have shape [B, ...].
    """
    dims = tuple(range(1, x.dim()))
    kinetic = 0.5 * (v * v).sum(dim=dims)
    potential = 0.5 * (omega * omega) * ((x - x_ref) ** 2).sum(dim=dims)
    return kinetic + potential


# =============================================================================
# MODULE
# =============================================================================

class DampedOscillator(nn.Module):
    """Holds the step count and the form ('exact' or 'stepped'); no parameters.

    The gains are passed in per call (they come from the banded parameters).
    Coefficients are computed in float32 at least.
    """

    def __init__(self, num_steps: int, form: str = "exact"):
        super().__init__()
        if form not in ("exact", "stepped"):
            raise ValueError("form must be 'exact' or 'stepped'")
        self.num_steps = int(num_steps)
        self.form = form

    def forward(self, x0: Tensor, x_ref: Tensor, force: Optional[Tensor],
                beta: Tensor, omega: Tensor, kappa: Tensor) -> Tensor:
        dtype = torch.promote_types(x0.dtype, torch.float32)
        args = (x0.to(dtype), x_ref.to(dtype), None if force is None else force.to(dtype),
                beta.to(dtype), omega.to(dtype), kappa.to(dtype), self.num_steps)
        if self.form == "exact":
            return integrate_exact(*args)
        return integrate_stepped(*args)

    def trajectory_energy(self, x0: Tensor, x_ref: Tensor, force: Optional[Tensor],
                          beta: Tensor, omega: Tensor, kappa: Tensor) -> Tensor:
        """Energy after each stepped update, [n_steps + 1, B] (diagnostics only)."""
        _, traj = integrate_stepped(x0, x_ref, force, beta, omega, kappa, self.num_steps,
                                    return_trajectory=True)
        return torch.stack([energy(x, v, x_ref, omega) for x, v in traj])

    def extra_repr(self) -> str:
        return f"num_steps={self.num_steps}, form={self.form!r}"
