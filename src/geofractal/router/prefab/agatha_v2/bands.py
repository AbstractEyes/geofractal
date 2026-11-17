"""
geofractal.router.prefab.agatha_v2.bands
========================================

Noise bands on the flow time, and the per-band parameters they mix.

Convention: z_t = (1 - t) z_0 + t z_1, t = 1 is data, so sigma = 1 - t.

Band coordinate (the shift-warped sigma):
    s = shift * sigma / (1 + (shift - 1) * sigma)

Three windows on s with edges e_1 < e_2 and a cosine crossfade of half-width w
around each edge:
    ramp(x) = 0.5 - 0.5 cos(pi * (clamp(x / w, -1, 1) + 1) / 2)
    b_0 = 1 - ramp(s - e_1)
    b_1 = ramp(s - e_1) * (1 - ramp(s - e_2))
    b_2 = ramp(s - e_1) * ramp(s - e_2)
Each b_j >= 0 and b_0 + b_1 + b_2 = 1. The windows are fixed functions of s;
nothing is compared against anything.

Every scheduled quantity q is mixed as q(t) = sum_j b_j(s) q_j from learned
per-band values q_j.

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.prefab.agatha_v2.oscillator import (
    control_gain_from_raw,
    damping_from_raw,
    frequency_from_raw,
)

NUM_BANDS = 3


def band_coordinate(t: Tensor, shift: float = 1.0) -> Tensor:
    """Shift-warped sigma of the flow time t (sigma = 1 - t)."""
    sigma = 1.0 - t
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _ramp(x: Tensor, half_width: float) -> Tensor:
    u = ((x / half_width).clamp(-1.0, 1.0) + 1.0) / 2.0
    return 0.5 - 0.5 * torch.cos(u * math.pi)


def band_weights(s: Tensor, edges: Tuple[float, float] = (0.35, 0.75), half_width: float = 0.06) -> Tensor:
    """Window weights [..., 3] on the band coordinate s; they sum to 1."""
    e1, e2 = edges
    up1 = _ramp(s - e1, half_width)
    up2 = _ramp(s - e2, half_width)
    low = 1.0 - up1
    mid = up1 * (1.0 - up2)
    high = up1 * up2
    return torch.stack([low, mid, high], dim=-1)


class BandParameters(nn.Module):
    """Per-band raw gains and (optionally) per-band signed tower weights.

    Gains per band j:
        beta_j  = beta_min + (beta_max - beta_min) sigmoid(rho_j)   (initialized at rho = 0)
        omega_j = omega_max sigmoid(phi_j)                          (initialized at phi = 0)
        kappa_j = softplus(psi_j)                                   (initialized at psi = kappa_init)
    Tower weights w_{j,k} are plain real numbers, initialized to 1 / K, with no sign
    constraint and no normalization.

    forward(bw) mixes them with band weights bw [B, 3] and returns
    beta [B], omega [B], kappa [B] and w [B, K] (or None when num_weights == 0).
    Gains are mixed after their bounded maps, so a convex mix stays in bounds.
    """

    def __init__(self, num_weights: int, beta_min: float, beta_max: float, omega_max: float,
                 kappa_init: float = 0.0):
        super().__init__()
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.omega_max = float(omega_max)
        self.rho = nn.Parameter(torch.zeros(NUM_BANDS))
        self.phi = nn.Parameter(torch.zeros(NUM_BANDS))
        self.psi = nn.Parameter(torch.full((NUM_BANDS,), float(kappa_init)))
        self.num_weights = int(num_weights)
        if self.num_weights > 0:
            self.weights = nn.Parameter(torch.full((NUM_BANDS, self.num_weights), 1.0 / self.num_weights))
        else:
            self.register_parameter("weights", None)

    def per_band(self) -> Tuple[Tensor, Tensor, Tensor]:
        beta = damping_from_raw(self.rho, self.beta_min, self.beta_max)
        omega = frequency_from_raw(self.phi, self.omega_max)
        kappa = control_gain_from_raw(self.psi)
        return beta, omega, kappa

    def forward(self, bw: Tensor):
        bw = bw.float()
        beta_j, omega_j, kappa_j = self.per_band()
        beta = bw @ beta_j
        omega = bw @ omega_j
        kappa = bw @ kappa_j
        w = bw @ self.weights if self.weights is not None else None
        return beta, omega, kappa, w

    def extra_repr(self) -> str:
        return (f"bands={NUM_BANDS}, weights={self.num_weights}, beta in [{self.beta_min:.3g}, "
                f"{self.beta_max:.3g}], omega in (0, {self.omega_max:.3g}]")


class SignedDispatch(nn.Module):
    """EXPERIMENTAL, off by default. Per-token signed shares over the towers.

        u_{n,k} = cos(W_q h_n, a_k) / tau
        s_{n,k} = sinh(u_{n,k}) / sum_j cosh(u_{n,j})

    The shares are signed and are not a probability distribution; they replace
    the banded tower weights when enabled. Returns s [B, N, K].
    """

    def __init__(self, dim: int, num_towers: int, key_dim: int = 64, tau: float = 1.0):
        super().__init__()
        self.query = nn.Linear(dim, key_dim)
        self.anchors = nn.Parameter(torch.randn(num_towers, key_dim) / math.sqrt(key_dim))
        self.tau = float(tau)

    def forward(self, h: Tensor) -> Tensor:
        q = F.normalize(self.query(h), dim=-1)                  # [B, N, Dk]
        a = F.normalize(self.anchors, dim=-1)                   # [K, Dk]
        u = torch.einsum("bnd,kd->bnk", q, a) / self.tau
        return torch.sinh(u) / torch.cosh(u).sum(dim=-1, keepdim=True)
