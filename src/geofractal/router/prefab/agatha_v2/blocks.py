"""
geofractal.router.prefab.agatha_v2.blocks
=========================================

Building blocks.

TransformerBlock (pre-norm, three residual branches, no gates):
    x = x + SelfAttn( mod_1(LN(x)) )          rotary applied to q, k
    x = x + CrossAttn( mod_2(LN(x)), C )      C = text sequence, padding masked
    x = x + MLP( mod_3(LN(x)) )
    mod_i(y) = y * (1 + ds_i(e)) + dh_i(e)

The modulation's last layer (e -> 6 d) is zero at initialization, so every norm starts
out passing its input through. No residual branch is multiplied by a learned gate.

ConvResBlock (on the token grid):
    y = conv3x3( SiLU( FiLM( GN( conv3x3( SiLU( GN(x_in) ) ) ), e ) ) )
    x = x + y,   FiLM(y, e) = y * (1 + ds(e)) + dh(e), last layer zero at initialization
    x_in = x (kind 'wide_resnet') or x - avgpool3x3(x) (kind 'frequency', a fixed
    high-pass of the branch input).

Rotary helpers: a rotary is a callable (q, k) -> (q, k) acting on
[B, heads, N, head_dim]. AxialRotary2D rotates the first half of each head with
row positions and the second half with column positions, using tables from
geofractal's RoPE.

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.components.rope_component import RoPE

Rotary = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]


def identity_rotary(q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
    return q, k


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x [B, H, N, D]; cos, sin [N, D]."""
    return x * cos + rotate_half(x) * sin


def table_rotary(cos: Tensor, sin: Tensor) -> Rotary:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def _rot(q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        c, s = cos.to(q.dtype), sin.to(q.dtype)
        return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s

    return _rot


class AxialRotary2D(nn.Module):
    """Plain 2-D rotary position code for a raster-ordered grid of tokens."""

    def __init__(self, head_dim: int, grid_h: int, grid_w: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by 4")
        half = head_dim // 2
        rope = RoPE("axial_rope", head_dim=half, theta=theta, max_seq_len=max(grid_h, grid_w))
        cos, sin = rope.embed(max(grid_h, grid_w))                  # [L, half]
        rows = torch.arange(grid_h).repeat_interleave(grid_w)       # raster order
        cols = torch.arange(grid_w).repeat(grid_h)
        self.register_buffer("cos_r", cos[rows].clone(), persistent=False)
        self.register_buffer("sin_r", sin[rows].clone(), persistent=False)
        self.register_buffer("cos_c", cos[cols].clone(), persistent=False)
        self.register_buffer("sin_c", sin[cols].clone(), persistent=False)
        self.half = half
        self.num_tokens = grid_h * grid_w

    def rotary(self) -> Rotary:
        half = self.half
        cr, sr, cc, sc = self.cos_r, self.sin_r, self.cos_c, self.sin_c

        def _one(x: Tensor) -> Tensor:
            a, b = x[..., :half], x[..., half:]
            return torch.cat([apply_rotary(a, cr.to(x.dtype), sr.to(x.dtype)),
                              apply_rotary(b, cc.to(x.dtype), sc.to(x.dtype))], dim=-1)

        def _rot(q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
            if q.shape[-2] != self.num_tokens:
                raise ValueError(f"expected {self.num_tokens} tokens, got {q.shape[-2]}")
            return _one(q), _one(k)

        return _rot


# =============================================================================
# ATTENTION
# =============================================================================

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rotary: Rotary) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = rotary(q, k)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, ctx: Tensor, ctx_mask: Optional[Tensor]) -> Tensor:
        b, n, d = x.shape
        l = ctx.shape[1]
        q = self.q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).view(b, l, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn_mask = None
        if ctx_mask is not None:
            attn_mask = ctx_mask[:, None, None, :]                      # [B, 1, 1, L], True = attend
        y = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=attn_mask)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

def zero_linear(in_dim: int, out_dim: int) -> nn.Linear:
    lin = nn.Linear(in_dim, out_dim)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross = CrossAttention(dim, num_heads)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.modulation = nn.Sequential(nn.SiLU(), zero_linear(dim, 6 * dim))

    def forward(self, x: Tensor, ctx: Tensor, ctx_mask: Optional[Tensor], e: Tensor, rotary: Rotary) -> Tensor:
        ds1, dh1, ds2, dh2, ds3, dh3 = self.modulation(e).unsqueeze(1).chunk(6, dim=-1)
        x = x + self.attn(self.norm1(x) * (1 + ds1) + dh1, rotary)
        x = x + self.cross(self.norm2(x) * (1 + ds2) + dh2, ctx, ctx_mask)
        x = x + self.mlp(self.norm3(x) * (1 + ds3) + dh3)
        return x


# =============================================================================
# CONV RESIDUAL BLOCK
# =============================================================================

def _groups(channels: int) -> int:
    return math.gcd(32, channels)


class ConvResBlock(nn.Module):
    def __init__(self, dim: int, kind: str = "wide_resnet"):
        super().__init__()
        if kind not in ("wide_resnet", "frequency"):
            raise ValueError(f"unknown conv block kind {kind!r}")
        self.kind = kind
        self.norm1 = nn.GroupNorm(_groups(dim), dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(dim), dim)
        self.film = nn.Sequential(nn.SiLU(), zero_linear(dim, 2 * dim))
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        inp = x
        if self.kind == "frequency":
            inp = x - F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        y = self.conv1(F.silu(self.norm1(inp)))
        ds, dh = self.film(e)[:, :, None, None].chunk(2, dim=1)
        y = self.norm2(y) * (1 + ds) + dh
        y = self.conv2(F.silu(y))
        return x + y
