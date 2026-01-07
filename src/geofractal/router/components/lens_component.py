import torch
from torch import nn, Tensor
import math
from typing import List

from geofractal.router.components.torch_component import TorchComponent


# ============================================================================
# CANTOR ZIPPER LENS
# ============================================================================

class CantorZipperLens(TorchComponent):
    """
    Continuous Cantor lens with counter-propagating wave interference.

    Gate modes:
    - 'interference': L * R (peaks where both align)
    - 'zipper': |L - R| (peaks where waves differ)
    - 'soft_xor': L + R - 2*L*R (differentiable XOR)
    """

    def __init__(
        self,
        scales: List[float],
        omega: float = math.pi,
        alpha_base: float = 1.5,
        gate_mode: str = 'soft_xor',
        invert: bool = False,
        learnable: bool = True,
        init_drift: float = 1.0,
        wave_mix: List[float] = [0.5, 0.5],
    ):
        super().__init__()
        scales = torch.tensor(scales)
        self.num_scales = len(scales)
        self.gate_mode = gate_mode
        self.invert = invert

        self.register_buffer('scales', scales)
        self.register_buffer('omega', torch.tensor(omega))

        alpha = alpha_base / torch.sqrt(scales)

        if learnable:
            self.alpha = nn.Parameter(alpha)
            self.phase = nn.Parameter(torch.zeros(len(scales)))
            self.drift = nn.Parameter(torch.full((len(scales),), init_drift))
            self.wave_mix = nn.Parameter(torch.tensor(wave_mix))
        else:
            self.register_buffer('alpha', alpha)
            self.register_buffer('phase', torch.zeros(len(scales)))
            self.register_buffer('drift', torch.full((len(scales),), init_drift))
            self.register_buffer('wave_mix', torch.tensor(wave_mix))

    def _compute_wave(self, x: Tensor, direction: float = 1.0) -> Tensor:
        x_norm = torch.tanh(x)
        x_exp = x_norm.unsqueeze(-2)
        t = x_norm.abs().mean(dim=-1, keepdim=True).unsqueeze(-2)

        s = self.scales.view(-1, 1)
        a = self.alpha.abs().view(-1, 1) + 0.1
        p = self.phase.view(-1, 1)
        d = self.drift.view(-1, 1)

        position = s * self.omega * (x_exp + direction * d * t) + p
        wave = torch.sin(position)
        gate = torch.exp(-a * wave.pow(2))

        return gate

    def forward(self, x: Tensor) -> Tensor:
        gate_left = self._compute_wave(x, direction=1.0)
        gate_right = self._compute_wave(x, direction=-1.0)

        w = torch.softmax(self.wave_mix, dim=0)

        if self.gate_mode == 'interference':
            combined = gate_left * gate_right
        elif self.gate_mode == 'zipper':
            combined = (gate_left - gate_right).abs()
        elif self.gate_mode == 'soft_xor':
            combined = w[0] * gate_left + w[1] * gate_right - 2 * gate_left * gate_right
            combined = combined.abs()
        else:
            combined = w[0] * gate_left + w[1] * gate_right

        final_gate = combined.prod(dim=-2)

        if self.invert:
            final_gate = 1.0 - final_gate

        final_gate = final_gate / (final_gate.mean() + 1e-6) * 0.5
        final_gate = final_gate.clamp(0, 1)

        return x * final_gate


# ============================================================================
# TRI-WAVE LENS
# ============================================================================

class TriWaveLens(nn.Module):
    """Three independent Cantor waves with a learnable combination."""

    def __init__(
        self,
        scales: List[float],
        omega: float = math.pi,
        alpha_base: float = 1.5,
        invert: bool = False,
        xor_weight: float = 0.7,
        accum_weights: List[float] = [0.4, 0.2, 0.4],
    ):
        super().__init__()
        scales = torch.tensor(scales)
        self.num_scales = len(scales)
        self.invert = invert

        self.register_buffer('scales', scales)
        self.register_buffer('omega', torch.tensor(omega))

        n_scales = len(scales)

        # Left wave
        self.alpha_l = nn.Parameter(alpha_base / torch.sqrt(scales))
        self.phase_l = nn.Parameter(torch.zeros(n_scales))
        self.drift_l = nn.Parameter(torch.ones(n_scales))

        # Middle wave (static)
        self.alpha_m = nn.Parameter(alpha_base / torch.sqrt(scales))
        self.phase_m = nn.Parameter(torch.zeros(n_scales))
        self.drift_m = nn.Parameter(torch.zeros(n_scales))

        # Right wave
        self.alpha_r = nn.Parameter(alpha_base / torch.sqrt(scales))
        self.phase_r = nn.Parameter(torch.zeros(n_scales))
        self.drift_r = nn.Parameter(-torch.ones(n_scales))

        self.accum_weights = nn.Parameter(torch.tensor(accum_weights))
        self.xor_weight = nn.Parameter(torch.tensor(xor_weight))

    def _wave(self, x: Tensor, alpha: Tensor, phase: Tensor, drift: Tensor) -> Tensor:
        x_norm = torch.tanh(x)
        t = x_norm.abs().mean(dim=-1, keepdim=True).unsqueeze(-2)

        x_exp = x_norm.unsqueeze(-2)
        s = self.scales.view(-1, 1)
        a = alpha.abs().view(-1, 1) + 0.1
        p = phase.view(-1, 1)
        d = drift.view(-1, 1)

        position = s * self.omega * (x_exp + d * t) + p
        wave = torch.sin(position)
        gate = torch.exp(-a * wave.pow(2))

        return gate.prod(dim=-2)

    def forward(self, x: Tensor) -> Tensor:
        L = self._wave(x, self.alpha_l, self.phase_l, self.drift_l)
        M = self._wave(x, self.alpha_m, self.phase_m, self.drift_m)
        R = self._wave(x, self.alpha_r, self.phase_r, self.drift_r)

        w = torch.softmax(self.accum_weights, dim=0)
        xor_w = torch.sigmoid(self.xor_weight)

        xor_component = L + R - 2 * L * R
        and_component = L * R
        lr_combined = xor_w * xor_component.abs() + (1 - xor_w) * and_component

        final_gate = w[0] * L + w[1] * M + w[2] * R
        final_gate = final_gate * (0.5 + 0.5 * lr_combined)

        if self.invert:
            final_gate = 1.0 - final_gate

        final_gate = final_gate / (final_gate.mean() + 1e-6) * 0.5
        final_gate = final_gate.clamp(0, 1)

        return x * final_gate