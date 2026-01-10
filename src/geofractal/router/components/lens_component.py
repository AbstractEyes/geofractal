"""
Geometric Lens Gates
====================

Learnable activation gates based on continuous Cantor set wave interference.
Drop-in replacements for GELU/ReLU/etc with geometric inductive bias.

Use Cases:
    - Replace standard activations in transformers, CNNs, MLPs
    - Add learnable gating with continuous topology across depth
    - Introduce scale-aware nonlinearity without destroying geometric structure
    - Works with both channel-last (NLP) and channel-first (vision) layouts

Description:
    These gates use counter-propagating waves at learned scales to create
    interference patterns that gate activations. Unlike static activations,
    they learn WHERE to gate based on wave alignment.

    Scale regulation: Each layer gets a 2-scale window [scale_low, scale_high].
    The windows form a CONTINUUM across depth - layer N's scale_high equals
    layer N+1's scale_low. This ensures no gaps in scale coverage.

    scale_low = scale_range[0] + t * (scale_range[1] - scale_range[0])
    scale_high = scale_low + step
    where t = layer_idx / (total_layers - 1)
    and step = (scale_range[1] - scale_range[0]) / (total_layers - 1)

    MobiusLens: Twist-in → wave gate → twist-out (preserves orientation)
    TriWaveLens: Three independent waves (L/M/R) with XOR combination
    CantorZipperLens: Counter-propagating waves with mode selection

Components:
    - MobiusLens: Full geometric lens with orthogonal twist projections
    - TriWaveLens: Three-wave interference with learnable accumulation
    - CantorZipperLens: Dual-wave with interference/zipper/soft_xor modes

License: Apache 2.0
Date: 2025-01-10
Author: AbstractPhil
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Dict

from geofractal.router.components.torch_component import TorchComponent


# ============================================================================
# MÖBIUS LENS
# ============================================================================

class MobiusLens(TorchComponent):
    """
    Geometric activation gate with orthogonal twist projections.

    Applies twist-in rotation, wave-based gating, then twist-out rotation.
    The twists preserve geometric orientation while the center lens gates
    based on wave interference across a 2-scale window.

    Args:
        name: Component identifier
        dim: Feature dimension
        layer_idx: Position in network (0-indexed)
        total_layers: Total depth of network
        scale_range: (min_scale, max_scale) - window slides through this range

    Example:
        #>>> lens = MobiusLens('lens_0', dim=256, layer_idx=0, total_layers=12)
        #>>> x = torch.randn(B, N, 256)
        #>>> out = lens(x)  # Same shape, gated activation
    """

    def __init__(
        self,
        name: str,
        dim: int,
        layer_idx: int,
        total_layers: int,
        scale_range: Tuple[float, float] = (1.0, 9.0),
    ):
        super().__init__(name)

        self.dim = dim
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.t = layer_idx / max(total_layers - 1, 1)

        # Two-scale window per layer forming continuum
        # Layer N's scale_high == Layer N+1's scale_low
        scale_span = scale_range[1] - scale_range[0]
        step = scale_span / max(total_layers - 1, 1)
        scale_low = scale_range[0] + self.t * scale_span
        scale_high = scale_low + step

        self.register_buffer('scales', torch.tensor([scale_low, scale_high]))

        # Twist-in projection (orthogonal init)
        self.twist_in_angle = nn.Parameter(torch.tensor(self.t * math.pi))
        self.twist_in_proj = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.twist_in_proj.weight)

        # Wave parameters
        self.omega = nn.Parameter(torch.tensor(math.pi))
        self.alpha = nn.Parameter(torch.tensor(1.5))

        # Phase/drift for L, M, R waves (size 2 for both scales)
        self.phase_l = nn.Parameter(torch.zeros(2))
        self.drift_l = nn.Parameter(torch.ones(2))
        self.phase_m = nn.Parameter(torch.zeros(2))
        self.drift_m = nn.Parameter(torch.zeros(2))
        self.phase_r = nn.Parameter(torch.zeros(2))
        self.drift_r = nn.Parameter(-torch.ones(2))

        # Accumulation weights
        self.accum_weights = nn.Parameter(torch.tensor([0.4, 0.2, 0.4]))
        self.xor_weight = nn.Parameter(torch.tensor(0.7))

        # Gate normalization
        self.gate_norm = nn.LayerNorm(dim)

        # Twist-out projection (orthogonal init)
        self.twist_out_angle = nn.Parameter(torch.tensor(-self.t * math.pi))
        self.twist_out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.twist_out_proj.weight)

    def _twist_in(self, x: Tensor) -> Tensor:
        cos_t = torch.cos(self.twist_in_angle)
        sin_t = torch.sin(self.twist_in_angle)
        return x * cos_t + self.twist_in_proj(x) * sin_t

    def _center_lens(self, x: Tensor) -> Tensor:
        x_norm = torch.tanh(x)
        t = x_norm.abs().mean(dim=-1, keepdim=True).unsqueeze(-2)

        x_exp = x_norm.unsqueeze(-2)
        s = self.scales.view(-1, 1)

        def wave(phase, drift):
            a = self.alpha.abs() + 0.1
            pos = s * self.omega * (x_exp + drift.view(-1, 1) * t) + phase.view(-1, 1)
            return torch.exp(-a * torch.sin(pos).pow(2)).prod(dim=-2)

        L = wave(self.phase_l, self.drift_l)
        M = wave(self.phase_m, self.drift_m)
        R = wave(self.phase_r, self.drift_r)

        w = torch.softmax(self.accum_weights, dim=0)
        xor_w = torch.sigmoid(self.xor_weight)

        xor_comp = (L + R - 2 * L * R).abs()
        and_comp = L * R
        lr = xor_w * xor_comp + (1 - xor_w) * and_comp

        gate = w[0] * L + w[1] * M + w[2] * R
        gate = gate * (0.5 + 0.5 * lr)
        gate = torch.sigmoid(self.gate_norm(gate))

        return x * gate

    def _twist_out(self, x: Tensor) -> Tensor:
        cos_t = torch.cos(self.twist_out_angle)
        sin_t = torch.sin(self.twist_out_angle)
        return x * cos_t + self.twist_out_proj(x) * sin_t

    def forward(self, x: Tensor) -> Tensor:
        return self._twist_out(self._center_lens(self._twist_in(x)))

    def get_lens_stats(self) -> Dict[str, float]:
        """Return lens parameters for logging."""
        w = torch.softmax(self.accum_weights, dim=0)
        return {
            't': self.t,
            'scale_low': self.scales[0].item(),
            'scale_high': self.scales[1].item(),
            'omega': self.omega.item(),
            'alpha': self.alpha.item(),
            'twist_in_angle': self.twist_in_angle.item(),
            'twist_out_angle': self.twist_out_angle.item(),
            'xor_weight': torch.sigmoid(self.xor_weight).item(),
            'accum_l': w[0].item(),
            'accum_m': w[1].item(),
            'accum_r': w[2].item(),
        }


# ============================================================================
# TRI-WAVE LENS
# ============================================================================

class TriWaveLens(TorchComponent):
    """
    Three independent Cantor waves with learnable combination.

    Lighter than MobiusLens (no twist projections). Uses left/middle/right
    waves with soft XOR combination for interference patterns across a
    2-scale window.

    Args:
        name: Component identifier
        dim: Feature dimension
        layer_idx: Position in network (0-indexed)
        total_layers: Total depth of network
        scale_range: (min_scale, max_scale) - window slides through this range
        alpha_base: Base sharpness for wave peaks
        invert: If True, invert final gate (gate zeros instead of ones)

    Example:
        #>>> lens = TriWaveLens('tri_0', dim=256, layer_idx=0, total_layers=12)
        #>>> x = torch.randn(B, N, 256)
        #>>> out = lens(x)  # Same shape, gated activation
    """

    def __init__(
        self,
        name: str,
        dim: int,
        layer_idx: int,
        total_layers: int,
        scale_range: Tuple[float, float] = (1.0, 9.0),
        alpha_base: float = 1.5,
        invert: bool = False,
    ):
        super().__init__(name)

        self.dim = dim
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.invert = invert

        # Two-scale window per layer forming continuum
        # Layer N's scale_high == Layer N+1's scale_low
        self.t = layer_idx / max(total_layers - 1, 1)
        scale_span = scale_range[1] - scale_range[0]
        step = scale_span / max(total_layers - 1, 1)
        scale_low = scale_range[0] + self.t * scale_span
        scale_high = scale_low + step

        scales = torch.tensor([scale_low, scale_high])
        self.register_buffer('scales', scales)
        self.register_buffer('omega', torch.tensor(math.pi))

        # Left wave (forward propagating)
        self.alpha_l = nn.Parameter(alpha_base / torch.sqrt(scales))
        self.phase_l = nn.Parameter(torch.zeros(2))
        self.drift_l = nn.Parameter(torch.ones(2))

        # Middle wave (stationary reference)
        self.alpha_m = nn.Parameter(alpha_base / torch.sqrt(scales))
        self.phase_m = nn.Parameter(torch.zeros(2))
        self.drift_m = nn.Parameter(torch.zeros(2))

        # Right wave (backward propagating)
        self.alpha_r = nn.Parameter(alpha_base / torch.sqrt(scales))
        self.phase_r = nn.Parameter(torch.zeros(2))
        self.drift_r = nn.Parameter(-torch.ones(2))

        # Combination weights
        self.accum_weights = nn.Parameter(torch.tensor([0.4, 0.2, 0.4]))
        self.xor_weight = nn.Parameter(torch.tensor(0.7))

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

    def get_lens_stats(self) -> Dict[str, float]:
        """Return lens parameters for logging."""
        w = torch.softmax(self.accum_weights, dim=0)
        return {
            't': self.t,
            'scale_low': self.scales[0].item(),
            'scale_high': self.scales[1].item(),
            'xor_weight': torch.sigmoid(self.xor_weight).item(),
            'accum_l': w[0].item(),
            'accum_m': w[1].item(),
            'accum_r': w[2].item(),
            'drift_l_mean': self.drift_l.mean().item(),
            'drift_r_mean': self.drift_r.mean().item(),
        }


# ============================================================================
# CANTOR ZIPPER LENS
# ============================================================================

class CantorZipperLens(TorchComponent):
    """
    Continuous Cantor lens with counter-propagating wave interference.

    Simplest of the three - just two waves (left/right) with selectable
    combination mode. Uses a 2-scale window for multi-scale interference.

    Gate modes:
        - 'interference': L * R (peaks where both waves align)
        - 'zipper': |L - R| (peaks where waves maximally differ)
        - 'soft_xor': L + R - 2*L*R (differentiable XOR approximation)

    Args:
        name: Component identifier
        dim: Feature dimension
        layer_idx: Position in network (0-indexed)
        total_layers: Total depth of network
        scale_range: (min_scale, max_scale) - window slides through this range
        alpha_base: Base sharpness for wave peaks
        gate_mode: One of 'interference', 'zipper', 'soft_xor'
        invert: If True, invert final gate
        learnable: If False, freeze wave parameters
        init_drift: Initial drift magnitude

    Example:
        #>>> lens = CantorZipperLens('zip_0', dim=256, layer_idx=0, total_layers=12)
        #>>> x = torch.randn(B, N, 256)
        #>>> out = lens(x)  # Same shape, gated activation
    """

    def __init__(
        self,
        name: str,
        dim: int,
        layer_idx: int,
        total_layers: int,
        scale_range: Tuple[float, float] = (1.0, 9.0),
        alpha_base: float = 1.5,
        gate_mode: str = 'soft_xor',
        invert: bool = False,
        learnable: bool = True,
        init_drift: float = 1.0,
    ):
        super().__init__(name)

        self.dim = dim
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.gate_mode = gate_mode
        self.invert = invert

        # Two-scale window per layer forming continuum
        # Layer N's scale_high == Layer N+1's scale_low
        self.t = layer_idx / max(total_layers - 1, 1)
        scale_span = scale_range[1] - scale_range[0]
        step = scale_span / max(total_layers - 1, 1)
        scale_low = scale_range[0] + self.t * scale_span
        scale_high = scale_low + step

        scales = torch.tensor([scale_low, scale_high])
        self.register_buffer('scales', scales)
        self.register_buffer('omega', torch.tensor(math.pi))

        alpha = alpha_base / torch.sqrt(scales)

        if learnable:
            self.alpha = nn.Parameter(alpha)
            self.phase = nn.Parameter(torch.zeros(2))
            self.drift = nn.Parameter(torch.full((2,), init_drift))
            self.wave_mix = nn.Parameter(torch.tensor([0.5, 0.5]))
        else:
            self.register_buffer('alpha', alpha)
            self.register_buffer('phase', torch.zeros(2))
            self.register_buffer('drift', torch.full((2,), init_drift))
            self.register_buffer('wave_mix', torch.tensor([0.5, 0.5]))

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

    def get_lens_stats(self) -> Dict[str, float]:
        """Return lens parameters for logging."""
        w = torch.softmax(self.wave_mix, dim=0)
        return {
            't': self.t,
            'scale_low': self.scales[0].item(),
            'scale_high': self.scales[1].item(),
            'gate_mode': self.gate_mode,
            'wave_mix_l': w[0].item(),
            'wave_mix_r': w[1].item(),
            'drift_mean': self.drift.mean().item(),
            'alpha_mean': self.alpha.abs().mean().item(),
        }


# ============================================================================
# MAIN - TEST HARNESS
# ============================================================================

if __name__ == '__main__':

    def test_lens(lens_class, name: str, **kwargs):
        """Test a lens class with various input shapes."""
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)

        dim = kwargs.get('dim', 256)
        total_layers = kwargs.get('total_layers', 12)

        # Test shapes: (batch, seq, dim) for NLP, will permute for vision
        test_cases = [
            ('NLP small',    (2, 16, dim)),
            ('NLP medium',   (4, 128, dim)),
            ('NLP large',    (2, 512, dim)),
            ('Vision 2D',    (2, 64, 64, dim)),   # B, H, W, D (channel-last)
            ('Single token', (1, 1, dim)),
            ('Batch=1',      (1, 32, dim)),
        ]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Test at different layer positions
        layer_positions = [0, total_layers // 2, total_layers - 1]

        for layer_idx in layer_positions:
            lens = lens_class(
                name=f'{name}_layer{layer_idx}',
                dim=dim,
                layer_idx=layer_idx,
                total_layers=total_layers,
                **{k: v for k, v in kwargs.items() if k not in ['dim', 'total_layers']}
            ).to(device)

            params = sum(p.numel() for p in lens.parameters())
            print(f"\nLayer {layer_idx}/{total_layers-1} (t={lens.t:.3f}) | Params: {params:,}")

            for case_name, shape in test_cases:
                x = torch.randn(*shape, device=device)

                try:
                    with torch.no_grad():
                        out = lens(x)

                    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

                    # Check output statistics
                    gate_ratio = (out.abs() > 1e-6).float().mean().item()
                    scale = (out.abs().mean() / (x.abs().mean() + 1e-6)).item()

                    print(f"  {case_name:15} {str(shape):25} ✓  active={gate_ratio:.1%}  scale={scale:.3f}")

                except Exception as e:
                    print(f"  {case_name:15} {str(shape):25} ✗  {e}")

            # Print lens stats for this layer
            stats = lens.get_lens_stats()
            stat_str = ', '.join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in list(stats.items())[:5])
            print(f"  Stats: {stat_str}")

    def test_gradient_flow():
        """Test that gradients flow through all lenses."""
        print(f"\n{'='*60}")
        print("Testing: Gradient Flow")
        print('='*60)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dim = 128
        total_layers = 8

        # Test MobiusLens at layer 4 (not 0) because twist_in/out_angle=0 at layer 0
        # means sin(0)=0, so twist projections have no effect (expected behavior)
        lenses = [
            ('MobiusLens (layer=4)', MobiusLens('mobius', dim, 4, total_layers)),
            ('MobiusLens (layer=0)', MobiusLens('mobius_0', dim, 0, total_layers)),
            ('TriWaveLens', TriWaveLens('tri', dim, 0, total_layers)),
            ('CantorZipperLens', CantorZipperLens('zipper', dim, 0, total_layers)),
        ]

        for name, lens in lenses:
            lens = lens.to(device)
            x = torch.randn(2, 16, dim, device=device, requires_grad=True)

            out = lens(x)
            loss = out.sum()
            loss.backward()

            grad_ok = x.grad is not None and x.grad.abs().sum() > 0
            param_grads = sum(1 for p in lens.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
            total_params = sum(1 for p in lens.parameters())

            status = "✓" if grad_ok and param_grads == total_params else "✗"
            print(f"  {name:25} input_grad={grad_ok}  param_grads={param_grads}/{total_params}  {status}")

            # Debug: show which params are missing gradients
            if param_grads < total_params:
                missing = [pname for pname, p in lens.named_parameters()
                          if p.grad is None or p.grad.abs().sum() == 0]
                if 'layer=0' in name and all('twist' in m for m in missing):
                    print(f"    (Expected: twist projs inactive when sin(angle)=0 at layer 0)")

    def test_zipper_modes():
        """Test CantorZipperLens with different gate modes."""
        print(f"\n{'='*60}")
        print("Testing: CantorZipperLens Gate Modes")
        print('='*60)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dim = 128
        x = torch.randn(2, 32, dim, device=device)

        modes = ['interference', 'zipper', 'soft_xor']

        for mode in modes:
            lens = CantorZipperLens(
                name=f'zipper_{mode}',
                dim=dim,
                layer_idx=4,
                total_layers=8,
                gate_mode=mode,
            ).to(device)

            with torch.no_grad():
                out = lens(x)

            gate_ratio = (out.abs() > 1e-6).float().mean().item()
            scale = (out.abs().mean() / (x.abs().mean() + 1e-6)).item()
            print(f"  {mode:15} active={gate_ratio:.1%}  scale={scale:.3f}")

    def test_scale_continuity():
        """Test that scale windows form a true continuum across depth."""
        print(f"\n{'='*60}")
        print("Testing: Scale Window Continuum Across Depth")
        print('='*60)

        dim = 128
        total_layers = 12
        scale_range = (0.5, 2.5)

        scale_span = scale_range[1] - scale_range[0]
        step = scale_span / (total_layers - 1)

        print(f"  Scale range: {scale_range}, Layers: {total_layers}")
        print(f"  Window step: {step:.4f}")
        print(f"  Expected: Layer N's high == Layer N+1's low (true continuum)")
        print()
        print(f"  {'Layer':<8} {'t':<8} {'scale_low':<12} {'scale_high':<12} {'continuity'}")
        print(f"  {'-'*60}")

        prev_high = None
        all_continuous = True
        for layer_idx in range(total_layers):
            lens = TriWaveLens(
                name=f'test_{layer_idx}',
                dim=dim,
                layer_idx=layer_idx,
                total_layers=total_layers,
                scale_range=scale_range,
            )

            low = lens.scales[0].item()
            high = lens.scales[1].item()

            continuity = ""
            if prev_high is not None:
                diff = abs(low - prev_high)
                if diff < 1e-6:
                    continuity = "✓ continuous"
                else:
                    continuity = f"✗ gap={diff:.6f}"
                    all_continuous = False

            print(f"  {layer_idx:<8} {lens.t:<8.3f} {low:<12.4f} {high:<12.4f} {continuity}")
            prev_high = high

        print()
        if all_continuous:
            print("  ✓ Scale windows form a TRUE CONTINUUM")
        else:
            print("  ✗ Gaps detected in scale coverage")

    def test_memory_and_speed():
        """Benchmark memory and speed."""
        print(f"\n{'='*60}")
        print("Testing: Memory & Speed Benchmark")
        print('='*60)

        if not torch.cuda.is_available():
            print("  Skipping (no CUDA)")
            return

        device = torch.device('cuda')
        dim = 512
        batch, seq = 8, 256
        total_layers = 12
        n_iters = 100

        lenses = [
            ('MobiusLens', MobiusLens('mobius', dim, 6, total_layers)),
            ('TriWaveLens', TriWaveLens('tri', dim, 6, total_layers)),
            ('CantorZipperLens', CantorZipperLens('zipper', dim, 6, total_layers)),
            ('GELU (baseline)', nn.GELU()),
        ]

        x = torch.randn(batch, seq, dim, device=device)

        for name, lens in lenses:
            lens = lens.to(device)

            # Warmup
            for _ in range(10):
                _ = lens(x)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            import time
            start = time.perf_counter()
            for _ in range(n_iters):
                _ = lens(x)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            ms_per_iter = (elapsed / n_iters) * 1000

            params = sum(p.numel() for p in lens.parameters()) if hasattr(lens, 'parameters') else 0
            print(f"  {name:20} {ms_per_iter:>6.2f} ms/iter  {mem_mb:>6.1f} MB peak  {params:>8,} params")

    # Run all tests
    print("\n" + "="*60)
    print("GEOMETRIC LENS GATES - TEST SUITE")
    print("="*60)

    # Test each lens type
    test_lens(MobiusLens, 'MobiusLens', dim=256, total_layers=12)
    test_lens(TriWaveLens, 'TriWaveLens', dim=256, total_layers=12)
    test_lens(CantorZipperLens, 'CantorZipperLens', dim=256, total_layers=12, gate_mode='soft_xor')

    # Additional tests
    test_gradient_flow()
    test_zipper_modes()
    test_scale_continuity()
    test_memory_and_speed()

    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETE")
    print("="*60)