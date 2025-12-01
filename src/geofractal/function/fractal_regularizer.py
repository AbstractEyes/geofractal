import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal

import triton
import triton.language as tl


FractalMode = Literal["speed", "balanced", "quality"]


# ==========================
# Triton kernels
# ==========================

@triton.jit
def cantor_hard_bucketize_kernel(
    x_ptr, thresholds_ptr, out_idx_ptr,
    N,
    NUM_THRESHOLDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    idx = tl.zeros((), dtype=tl.int32)
    for i in range(NUM_THRESHOLDS):
        thr = tl.load(thresholds_ptr + i)
        idx += (x > thr)

    tl.store(out_idx_ptr + offs, idx, mask=mask)


@triton.jit
def cantor_soft_index_kernel(
    x_ptr, thresholds_ptr, out_soft_idx_ptr,
    N, TEMP,
    NUM_THRESHOLDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    temp = TEMP

    soft_idx = tl.zeros_like(x)
    for i in range(NUM_THRESHOLDS):
        thr = tl.load(thresholds_ptr + i)
        diff = (x - thr) / temp
        soft_idx += tl.sigmoid(diff)

    tl.store(out_soft_idx_ptr + offs, soft_idx, mask=mask)


def _cast_for_compute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.dtype]:
    """
    Upcast fp8/unsupported dtypes to fp32, keep fp16/bf16/fp32 as-is.
    """
    orig = x.dtype
    if orig in (torch.float16, torch.bfloat16, torch.float32):
        return x, orig
    return x.to(torch.float32), orig


# ==========================
# CantorGate
# ==========================

class CantorGate(nn.Module):
    """
    Triton-accelerated fractal staircase activation.

    Modes:
      - "speed":    hard bucketize, no STE; gradients mainly from identity path.
      - "balanced": hard bucketize + STE; forward snapped, grad ~ identity.
      - "quality":  soft index (Triton kernel), full fractal shaping.

    Dtype:
      - fp32 / bf16 / fp16 natively
      - fp8 via upcast to fp32
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 5,
        mode: FractalMode = "balanced",
        hard_inference: bool = True,
        block_size: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.num_stairs = 2 ** num_levels
        self.mode: FractalMode = mode
        self.hard_inference = hard_inference
        self.block_size = block_size

        init_thresholds = self._init_cantor_thresholds(num_levels)
        # Keep as buffer; "quality" mode will add small learned offsets if desired
        self.register_buffer("thresholds", init_thresholds)  # [S-1]
        self.register_buffer("stair_values", torch.linspace(0, 1, self.num_stairs))

        # Optional offsets for "quality" mode (kept small)
        self.threshold_offsets = nn.Parameter(torch.zeros_like(init_thresholds))
        self.stair_offsets = nn.Parameter(torch.zeros_like(self.stair_values))

        # Blend strength
        self.snap_strength = nn.Parameter(torch.tensor(0.5))

        # Temperature scale for "quality"
        self.temp_scale = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _init_cantor_thresholds(levels: int) -> torch.Tensor:
        thresholds = set()
        num_stairs = 2 ** levels
        for i in range(levels):
            step = 1.0 / (3 ** (i + 1))
            for j in range(3 ** i):
                thresholds.add((3 * j + 1) * step)
                thresholds.add((3 * j + 2) * step)
        thresholds = sorted(thresholds)[: num_stairs - 1]
        return torch.tensor(thresholds, dtype=torch.float32)

    def _get_thresholds_and_stairs(self, x: torch.Tensor, training: bool):
        device = x.device
        dtype = x.dtype
        base_t = self.thresholds.to(device=device, dtype=dtype)
        base_s = self.stair_values.to(device=device, dtype=dtype)

        if training and self.mode == "quality":
            t = base_t + self.threshold_offsets.to(device=device, dtype=dtype)
            t, _ = t.sort()
            s = base_s + self.stair_offsets.to(device=device, dtype=dtype)
        else:
            t, s = base_t, base_s

        return t, s

    def _hard_bucketize_triton(self, x_norm: torch.Tensor, thresholds: torch.Tensor):
        """
        x_norm: (...,), thresholds: (S-1,)
        returns: stair_idx: same shape as x_norm, int32
        """
        x_flat = x_norm.contiguous().view(-1)
        N = x_flat.numel()
        out_idx = torch.empty(N, device=x_flat.device, dtype=torch.int32)

        BLOCK = self.block_size
        NUM_THRESHOLDS = thresholds.numel()

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)

        cantor_hard_bucketize_kernel[grid](
            x_flat,
            thresholds,
            out_idx,
            N,
            NUM_THRESHOLDS=NUM_THRESHOLDS,
            BLOCK=BLOCK,
        )
        return out_idx.view_as(x_norm)

    def _soft_index_triton(self, x_norm: torch.Tensor, thresholds: torch.Tensor, temp: float):
        x_flat = x_norm.contiguous().view(-1)
        N = x_flat.numel()
        out_soft = torch.empty_like(x_flat)

        BLOCK = self.block_size
        NUM_THRESHOLDS = thresholds.numel()
        TEMP = temp

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)

        cantor_soft_index_kernel[grid](
            x_flat,
            thresholds,
            out_soft,
            N,
            TEMP,
            NUM_THRESHOLDS=NUM_THRESHOLDS,
            BLOCK=BLOCK,
        )
        return out_soft.view_as(x_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x

        x, orig_dtype = _cast_for_compute(x)
        device = x.device
        dtype = x.dtype

        # sign + magnitude
        x_sign = x.sign()
        x_sign = torch.where(x_sign == 0, torch.ones_like(x_sign), x_sign)
        x_mag = x.abs().clamp(min=1e-8)

        # normalize to [0, 1)
        x_norm = torch.tanh(torch.log1p(x_mag) / 3.0)

        thresholds, stair_values = self._get_thresholds_and_stairs(x_norm, self.training)

        if (not self.training) and self.hard_inference:
            # Inference: pure hard bucketization (fast)
            stair_idx = self._hard_bucketize_triton(x_norm, thresholds)
            stair_idx = stair_idx.clamp(0, self.num_stairs - 1)
            snapped_norm = stair_values[stair_idx]
        else:
            if self.mode == "quality":
                # Full soft index via Triton
                temp = (torch.sigmoid(self.temp_scale) * 0.2 + 0.01).item()
                soft_idx = self._soft_index_triton(x_norm, thresholds, temp)
                idx_floor = soft_idx.long().clamp(0, self.num_stairs - 2)
                idx_ceil = idx_floor + 1
                frac = soft_idx - idx_floor.to(dtype)

                snapped_norm = (
                    stair_values[idx_floor] * (1.0 - frac)
                    + stair_values[idx_ceil] * frac
                )
            else:
                # "speed" / "balanced": hard bucketization via Triton, with optional STE
                stair_idx = self._hard_bucketize_triton(x_norm, thresholds)
                stair_idx = stair_idx.clamp(0, self.num_stairs - 1)
                snapped_hard = stair_values[stair_idx]

                if self.mode == "balanced":
                    # Straight-through: forward snapped, grad ~ identity
                    snapped_norm = snapped_hard.detach() + x_norm - x_norm.detach()
                else:  # "speed"
                    snapped_norm = snapped_hard.detach()

        snapped_mag = torch.expm1(snapped_norm * 3.0)

        strength = torch.sigmoid(self.snap_strength).to(dtype=dtype, device=device)
        out_mag = strength * snapped_mag + (1.0 - strength) * x_mag
        out = x_sign * out_mag
        return out.to(orig_dtype)


# ==========================
# TopologicalDropout
# ==========================

class TopologicalDropout(nn.Module):
    """
    Structure-preserving dropout over a route dimension.

    Keeps at least min_keep routes, optionally importance-weighted.
    All ops stay on GPU; topk over num_routes is cheap.
    """

    def __init__(self, drop_prob: float = 0.1, min_keep: int = 1, scale_kept: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.min_keep = int(min_keep)
        self.scale_kept = bool(scale_kept)

    def forward(
        self,
        x: torch.Tensor,
        route_dim: int = -2,
        num_routes: Optional[int] = None,
        importance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not self.training) or self.drop_prob <= 0.0:
            num_routes = num_routes or x.shape[route_dim]
            return x, torch.ones(num_routes, device=x.device, dtype=x.dtype)

        num_routes = num_routes or x.shape[route_dim]
        num_keep = max(self.min_keep, int(num_routes * (1.0 - self.drop_prob)))

        if importance is None:
            scores = torch.rand(num_routes, device=x.device)
            _, keep_idx = scores.topk(num_keep, dim=0)
        else:
            imp = importance.to(x.device)
            noise = torch.rand_like(imp) * 0.5
            drop_score = 1.0 / (imp + 1e-8) + noise
            _, keep_idx = (-drop_score).topk(num_keep, dim=0)

        keep_mask = x.new_zeros(num_routes)
        keep_mask[keep_idx] = 1.0

        view_shape = [1] * x.dim()
        view_shape[route_dim] = num_routes
        keep_mask_view = keep_mask.view(view_shape)

        out = x * keep_mask_view
        if self.scale_kept:
            out = out * (num_routes / float(num_keep))

        return out, keep_mask


# ==========================
# FractalRegularizer
# ==========================

class FractalRegularizer(nn.Module):
    """
    Unified fractal regularizer:

      - CantorGate with Triton bucketization
      - TopologicalDropout on route dimension (if present)
    """

    def __init__(
        self,
        dim: int,
        num_routes: int = 8,
        num_levels: int = 4,
        drop_prob: float = 0.1,
        min_routes_keep: int = 2,
        mode: FractalMode = "balanced",
        disable_on_eval: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_routes = num_routes
        self.disable_on_eval = disable_on_eval

        self.cantor_gate = CantorGate(
            dim=dim,
            num_levels=num_levels,
            mode=mode,
            hard_inference=True,
        )
        self.topo_dropout = TopologicalDropout(
            drop_prob=drop_prob,
            min_keep=min_routes_keep,
            scale_kept=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        route_dim: int = -2,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.disable_on_eval and (not self.training):
            return x

        x, orig_dtype = _cast_for_compute(x)
        has_routes = x.dim() >= 2 and x.shape[route_dim] == self.num_routes

        x = self.cantor_gate(x)

        if has_routes and self.training and self.topo_dropout.drop_prob > 0.0:
            importance = None
            if routing_weights is not None:
                # routing_weights: [..., num_routes]
                rt = routing_weights.to(x.device)
                reduce_dims = tuple(range(rt.dim() - 1))
                importance = rt.mean(dim=reduce_dims)

            x, _ = self.topo_dropout(x, route_dim=route_dim, importance=importance)

        return x.to(orig_dtype)


# ==========================
# CantorActivation (MLP)
# ==========================

class CantorActivation(nn.Module):
    """
    Lightweight fractal activation for MLPs using Triton bucketization.

    - Fixed thresholds/stairs (buffers)
    - Uses CantorGate's Triton hard bucketize in all modes
    - Only learns blend strength
    """

    def __init__(self, num_stairs: int = 16, block_size: int = 256):
        super().__init__()
        self.num_stairs = int(num_stairs)
        self.block_size = block_size

        thresholds = torch.linspace(0.0, 1.0, num_stairs + 1)[1:-1]
        self.register_buffer("thresholds", thresholds)
        stair_values = torch.linspace(0.0, 1.0, num_stairs)
        self.register_buffer("stair_values", stair_values)

        self.strength = nn.Parameter(torch.tensor(0.3))

    def _hard_bucketize_triton(self, x_norm: torch.Tensor):
        x_flat = x_norm.contiguous().view(-1)
        N = x_flat.numel()
        out_idx = torch.empty(N, device=x_flat.device, dtype=torch.int32)

        BLOCK = self.block_size
        NUM_THRESHOLDS = self.thresholds.numel()
        thresholds = self.thresholds.to(device=x_flat.device, dtype=x_flat.dtype)

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)

        cantor_hard_bucketize_kernel[grid](
            x_flat,
            thresholds,
            out_idx,
            N,
            NUM_THRESHOLDS=NUM_THRESHOLDS,
            BLOCK=BLOCK,
        )
        return out_idx.view_as(x_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x

        x, orig_dtype = _cast_for_compute(x)
        device = x.device
        dtype = x.dtype

        x_sign = x.sign()
        x_sign = torch.where(x_sign == 0, torch.ones_like(x_sign), x_sign)
        x_mag = x.abs().clamp(min=1e-8)

        x_norm = torch.tanh(x_mag / 3.0)

        stair_idx = self._hard_bucketize_triton(x_norm)
        stair_idx = stair_idx.clamp(0, self.num_stairs - 1)
        stair_values = self.stair_values.to(device=device, dtype=dtype)
        snapped_norm = stair_values[stair_idx]

        snapped_mag = torch.tanh(snapped_norm) * 3.0

        gelu_out = F.gelu(x)
        strength = torch.sigmoid(self.strength).to(dtype=dtype, device=device)

        out_mag = strength * snapped_mag + (1.0 - strength) * gelu_out.abs()
        out = x_sign * out_mag

        return out.to(orig_dtype)
