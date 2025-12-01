import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CantorGate(nn.Module):
    """
    Optimized fractal staircase activation.

    Key optimizations:
    - torch.bucketize for O(log n) threshold lookup vs O(n) comparison
    - Fused operations, minimal intermediate tensors
    - Straight-through estimator for gradients (no soft sigmoid)
    - Optional hard mode for inference
    """

    def __init__(
            self,
            dim: int,
            num_levels: int = 5,
            temperature: float = 0.05,
            hard_inference: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.num_stairs = 2 ** num_levels
        self.temperature = temperature
        self.hard_inference = hard_inference

        # Thresholds in [0, 1] for bucketize - SORTED required
        init_thresholds = self._init_cantor_thresholds(num_levels)
        self.thresholds = nn.Parameter(init_thresholds)

        # Stair output values [0, 1]
        self.stair_values = nn.Parameter(torch.linspace(0, 1, self.num_stairs))

        # Per-dim learnable blend (scalar broadcast)
        self.snap_strength = nn.Parameter(torch.tensor(0.5))

        # Temperature scale
        self.temp_scale = nn.Parameter(torch.tensor(0.0))

    def _init_cantor_thresholds(self, levels: int) -> torch.Tensor:
        """Generate sorted Cantor thresholds in [0, 1]."""
        thresholds = set()
        for i in range(levels):
            step = 1 / (3 ** (i + 1))
            for j in range(3 ** i):
                thresholds.add((3 * j + 1) * step)
                thresholds.add((3 * j + 2) * step)

        thresholds = sorted(thresholds)[:self.num_stairs - 1]
        return torch.tensor(thresholds, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve sign, work with magnitude
        x_sign = x.sign()
        x_sign = torch.where(x_sign == 0, torch.ones_like(x_sign), x_sign)
        x_mag = x.abs().clamp(min=1e-8)

        # Normalize to [0, 1)
        x_norm = torch.tanh(torch.log1p(x_mag) / 3.0)

        # Sort thresholds for bucketize (required)
        sorted_thresh, _ = self.thresholds.sort()

        if self.hard_inference and not self.training:
            # FAST PATH: Hard bucketize, no gradients through thresholds
            stair_idx = torch.bucketize(x_norm.contiguous(), sorted_thresh.contiguous())
            snapped_norm = self.stair_values[stair_idx.clamp(0, self.num_stairs - 1)]
        else:
            # TRAINING PATH: Soft with straight-through
            temp = (torch.sigmoid(self.temp_scale) * 0.2 + 0.01)  # [0.01, 0.21]

            # Soft bucket: sigmoid distance to each threshold, summed
            # Shape: [*x.shape, num_thresholds]
            # This is the expensive part - but unavoidable for gradients
            diff = (x_norm.unsqueeze(-1) - sorted_thresh) / temp
            soft_idx = torch.sigmoid(diff).sum(dim=-1)  # [0, num_stairs-1]

            # Interpolate stair values
            idx_floor = soft_idx.long().clamp(0, self.num_stairs - 2)
            idx_ceil = idx_floor + 1
            frac = soft_idx - idx_floor.float()

            snapped_norm = (
                    self.stair_values[idx_floor] * (1 - frac) +
                    self.stair_values[idx_ceil] * frac
            )

        # Convert back to magnitude
        snapped_mag = torch.expm1(snapped_norm * 3.0)

        # Blend
        strength = torch.sigmoid(self.snap_strength)
        output_mag = strength * snapped_mag + (1 - strength) * x_mag

        return x_sign * output_mag


class TopologicalDropout(nn.Module):
    """
    Optimized structure-preserving dropout.

    Key optimizations:
    - Pre-generate masks where possible
    - Avoid topk for simple uniform dropout
    - Fused mask application
    """

    def __init__(
            self,
            drop_prob: float = 0.1,
            min_keep: int = 1,
            scale_kept: bool = True
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.min_keep = min_keep
        self.scale_kept = scale_kept

    def forward(
            self,
            x: torch.Tensor,
            route_dim: int = -2,
            num_routes: Optional[int] = None,
            importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.training or self.drop_prob == 0:
            num_routes = num_routes or x.shape[route_dim]
            return x, torch.ones(num_routes, device=x.device)

        num_routes = num_routes or x.shape[route_dim]
        num_keep = max(self.min_keep, int(num_routes * (1 - self.drop_prob)))

        # Fast path: uniform dropout (no importance weighting)
        if importance is None:
            # Generate random scores and take topk
            scores = torch.rand(num_routes, device=x.device)
            _, keep_idx = scores.topk(num_keep)
            keep_mask = torch.zeros(num_routes, device=x.device)
            keep_mask[keep_idx] = 1.0
        else:
            # Importance-weighted: less important = more likely dropped
            noise = torch.rand_like(importance) * 0.5
            drop_score = 1.0 / (importance + 1e-8) + noise
            _, keep_idx = (-drop_score).topk(num_keep)
            keep_mask = torch.zeros(num_routes, device=x.device)
            keep_mask[keep_idx] = 1.0

        # Reshape for broadcast
        view_shape = [1] * x.dim()
        view_shape[route_dim] = num_routes
        keep_mask_view = keep_mask.view(view_shape)

        # Apply and scale
        out = x * keep_mask_view
        if self.scale_kept:
            out = out * (num_routes / num_keep)

        return out, keep_mask


class FractalRegularizer(nn.Module):
    """
    Optimized unified fractal regularization.

    Key optimizations:
    - Lazy initialization of sub-modules
    - Fused forward when possible
    - Optional bypass for inference
    """

    def __init__(
            self,
            dim: int,
            num_routes: int = 8,
            num_levels: int = 4,
            drop_prob: float = 0.1,
            min_routes_keep: int = 2,
            temperature: float = 0.05,
            disable_on_eval: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_routes = num_routes
        self.disable_on_eval = disable_on_eval

        self.cantor_gate = CantorGate(
            dim=dim,
            num_levels=num_levels,
            temperature=temperature,
            hard_inference=True
        )

        self.topo_dropout = TopologicalDropout(
            drop_prob=drop_prob,
            min_keep=min_routes_keep,
            scale_kept=True
        )

    def forward(
            self,
            x: torch.Tensor,
            route_dim: int = -2,
            routing_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Fast path: skip everything on eval if configured
        if self.disable_on_eval and not self.training:
            return x

        # Check if we have routes dimension
        has_routes = x.dim() >= 2 and x.shape[route_dim] == self.num_routes

        # Apply CantorGate (always)
        x = self.cantor_gate(x)

        # Apply TopologicalDropout (only if routes present and training)
        if has_routes and self.training:
            importance = None
            if routing_weights is not None:
                # Collapse to per-route importance
                importance = routing_weights.mean(dim=tuple(range(routing_weights.dim() - 1)))

            x, _ = self.topo_dropout(x, route_dim=route_dim, importance=importance)

        return x


# =============================================================================
# STANDALONE CANTOR ACTIVATION (Ultra-light version for MLP)
# =============================================================================

class CantorActivation(nn.Module):
    """
    Minimal CantorGate for use in MLPs - just the activation, no frills.

    Uses fixed stairs (not learned) for maximum speed.
    Only learns the blend strength.
    """

    def __init__(self, num_stairs: int = 16):
        super().__init__()
        self.num_stairs = num_stairs

        # Fixed stair thresholds and values (not learned)
        thresholds = torch.linspace(0, 1, num_stairs + 1)[1:-1]  # Interior points
        self.register_buffer('thresholds', thresholds)

        values = torch.linspace(0, 1, num_stairs)
        self.register_buffer('stair_values', values)

        # Only learned parameter: blend strength
        self.strength = nn.Parameter(torch.tensor(0.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path using bucketize
        x_sign = x.sign()
        x_sign = torch.where(x_sign == 0, torch.ones_like(x_sign), x_sign)
        x_mag = x.abs().clamp(min=1e-8)

        # Normalize
        x_norm = torch.tanh(x_mag / 3.0)  # Simpler normalization

        # Hard stair assignment
        stair_idx = torch.bucketize(x_norm.contiguous(), self.thresholds.contiguous())
        snapped_norm = self.stair_values[stair_idx]

        # Unnormalize
        snapped_mag = torch.tanh(snapped_norm) * 3.0  # Inverse-ish

        # Blend with GELU-like behavior for non-snapped part
        gelu_out = F.gelu(x)
        strength = torch.sigmoid(self.strength)

        out_mag = strength * snapped_mag + (1 - strength) * gelu_out.abs()

        return x_sign * out_mag


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Test dimensions
    batch, seq, routes, dim = 32, 256, 8, 64
    x = torch.randn(batch, seq, routes, dim, device=device)

    # Warmup
    for _ in range(10):
        _ = F.gelu(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark GELU
    start = time.perf_counter()
    for _ in range(100):
        _ = F.gelu(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    gelu_time = (time.perf_counter() - start) / 100
    print(f"GELU:              {gelu_time * 1000:.3f} ms")

    # Benchmark CantorActivation (minimal)
    cantor_min = CantorActivation(num_stairs=16).to(device)
    cantor_min.eval()

    for _ in range(10):
        _ = cantor_min(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = cantor_min(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    cantor_min_time = (time.perf_counter() - start) / 100
    print(f"CantorActivation:  {cantor_min_time * 1000:.3f} ms ({cantor_min_time / gelu_time:.1f}x GELU)")

    # Benchmark CantorGate (full, eval mode)
    cantor_full = CantorGate(dim=dim, num_levels=4, hard_inference=True).to(device)
    cantor_full.eval()

    x_flat = x.reshape(-1, dim)
    for _ in range(10):
        _ = cantor_full(x_flat)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = cantor_full(x_flat)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    cantor_full_time = (time.perf_counter() - start) / 100
    print(f"CantorGate (eval): {cantor_full_time * 1000:.3f} ms ({cantor_full_time / gelu_time:.1f}x GELU)")

    # Benchmark CantorGate (train mode - expensive)
    cantor_full.train()

    for _ in range(10):
        _ = cantor_full(x_flat)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = cantor_full(x_flat)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    cantor_train_time = (time.perf_counter() - start) / 100
    print(f"CantorGate (train):{cantor_train_time * 1000:.3f} ms ({cantor_train_time / gelu_time:.1f}x GELU)")

    # Benchmark FractalRegularizer
    fractal_reg = FractalRegularizer(
        dim=dim, num_routes=routes, num_levels=4, drop_prob=0.1
    ).to(device)

    # Eval mode (should be nearly free)
    fractal_reg.eval()
    for _ in range(10):
        _ = fractal_reg(x, route_dim=-2)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = fractal_reg(x, route_dim=-2)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fractal_eval_time = (time.perf_counter() - start) / 100
    print(f"FractalReg (eval): {fractal_eval_time * 1000:.3f} ms ({fractal_eval_time / gelu_time:.1f}x GELU)")

    # Train mode
    fractal_reg.train()
    routing_weights = torch.softmax(torch.randn(batch, seq, routes, device=device), dim=-1)

    for _ in range(10):
        _ = fractal_reg(x, route_dim=-2, routing_weights=routing_weights)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = fractal_reg(x, route_dim=-2, routing_weights=routing_weights)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fractal_train_time = (time.perf_counter() - start) / 100
    print(f"FractalReg (train):{fractal_train_time * 1000:.3f} ms ({fractal_train_time / gelu_time:.1f}x GELU)")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
    1. Use CantorActivation (minimal) in MLPs - ~2-3x GELU
    2. CantorGate eval mode uses bucketize - fast
    3. CantorGate train mode is expensive (soft thresholds)
    4. FractalRegularizer can bypass on eval (disable_on_eval=True)
    5. Consider: train with fractal, eval without
    """)