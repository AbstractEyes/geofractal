"""
geofractal.router.prefab.agatha_v2.towers
=========================================

Force towers. Every tower reads the trunk's token states h [B, N, d] and emits
a per-token force Y [B, N, P] through a head whose last layer is zero at initialization.

GeometryTower: n blocks of [self-attention with the tower's own position code
over the N tokens in raster order + cross-attention to the text + MLP], with
time modulation, then the zero head.

ConvTower: n residual 3x3 conv blocks on the (H/p) x (W/p) grid of h with FiLM
from the time embedding, then the zero head.

Position codes reuse geofractal's rotary classes at the true head width:
    cantor  -> CantorRoPE (mode 'hybrid', levels 5, tau 0.25, blend 0.5)
    beatrix -> BeatrixRoPE (levels 5, alpha 0.5, tau 0.25)
    simplex -> QuadRoPE (thetas 10000 / 5000 / 2500 / 1000, 'simplex' mix)
    helix   -> TriRoPE (thetas 10000 / 5000 / 2500, 'barycentric' mix)
    plain   -> RoPE
Each is constructed with max_seq_len equal to the token count. CantorRoPE and
BeatrixRoPE normalize positions over their cached length and slice it, so a
cache longer than the sequence would leave only the first part of their warp
in use. The tower therefore requires the exact token count it was built for.

Parameters of a position code that its table computation never reads (in the
chosen mode) are found by an autograd probe at construction and frozen
(requires_grad False); their names are kept in `frozen_position_parameters`.

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.components.rope_component import (
    BeatrixRoPE,
    CantorRoPE,
    QuadRoPE,
    RoPE,
    TriRoPE,
)
from geofractal.router.prefab.agatha_v2.blocks import (
    ConvResBlock,
    Rotary,
    TransformerBlock,
    identity_rotary,
    table_rotary,
    zero_linear,
)


def build_position_code(kind: str, head_dim: int, num_tokens: int, theta: float = 10000.0) -> nn.Module:
    if kind == "cantor":
        return CantorRoPE("cantor_rope", head_dim=head_dim, theta=theta, levels=5, tau=0.25,
                          mode="hybrid", blend_alpha=0.5, max_seq_len=num_tokens)
    if kind == "beatrix":
        return BeatrixRoPE("beatrix_rope", head_dim=head_dim, theta=theta, levels=5, alpha=0.5,
                           tau=0.25, max_seq_len=num_tokens)
    if kind == "simplex":
        return QuadRoPE("simplex_rope", head_dim=head_dim, theta_w=theta, theta_x=theta / 2,
                        theta_y=theta / 4, theta_z=theta / 10, augmentation="simplex")
    if kind == "helix":
        return TriRoPE("helix_rope", head_dim=head_dim, theta_alpha=theta, theta_beta=theta / 2,
                       theta_gamma=theta / 4, augmentation="barycentric")
    if kind == "plain":
        return RoPE("plain_rope", head_dim=head_dim, theta=theta, max_seq_len=num_tokens)
    raise ValueError(f"unknown position code {kind!r}")


class PositionCode(nn.Module):
    """Adapter giving every reused rotary class one interface: tables(N) -> (cos, sin) [N, D]."""

    def __init__(self, kind: str, head_dim: int, num_tokens: int, theta: float = 10000.0):
        super().__init__()
        self.kind = kind
        self.num_tokens = int(num_tokens)
        self.code = build_position_code(kind, head_dim, num_tokens, theta)
        self.frozen_position_parameters: List[str] = self._freeze_unread()

    def tables(self, n: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        n = self.num_tokens if n is None else n
        if n != self.num_tokens:
            raise ValueError(f"position code built for {self.num_tokens} tokens, called with {n}")
        if isinstance(self.code, (QuadRoPE, TriRoPE)):
            comb = self.code.embed_combined(n, mode="augment")
            return comb[0], comb[1]
        cos, sin = self.code.embed(n)
        return cos, sin

    def rotary(self) -> Rotary:
        cos, sin = self.tables()
        return table_rotary(cos.float(), sin.float())

    def _freeze_unread(self) -> List[str]:
        named = [(n, p) for n, p in self.code.named_parameters() if p.requires_grad]
        if not named:
            return []
        with torch.enable_grad():
            cos, sin = self.tables()
            out = cos.sum() + sin.sum()
            if not out.requires_grad:
                grads = [None] * len(named)
            else:
                grads = torch.autograd.grad(out, [p for _, p in named], allow_unused=True)
        frozen = []
        for (n, p), g in zip(named, grads):
            if g is None:
                p.requires_grad_(False)
                frozen.append(n)
        return frozen


class ZeroHead(nn.Module):
    """LayerNorm (no affine) then Linear(d, P) with zero weight and bias at initialization."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = zero_linear(dim, out_dim)

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(self.norm(h))


class GeometryTower(nn.Module):
    def __init__(self, kind: str, dim: int, num_heads: int, depth: int, num_tokens: int,
                 out_dim: int, mlp_ratio: float = 4.0, theta: float = 10000.0):
        super().__init__()
        self.kind = kind
        self.position_code = PositionCode(kind, dim // num_heads, num_tokens, theta)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.head = ZeroHead(dim, out_dim)
        self.use_position_code = True

    def features(self, h: Tensor, ctx: Tensor, ctx_mask: Optional[Tensor], e: Tensor) -> Tensor:
        rotary = self.position_code.rotary() if self.use_position_code else identity_rotary
        for blk in self.blocks:
            h = blk(h, ctx, ctx_mask, e, rotary)
        return h

    def forward(self, h: Tensor, ctx: Tensor, ctx_mask: Optional[Tensor], e: Tensor) -> Tensor:
        return self.head(self.features(h, ctx, ctx_mask, e))


class ConvTower(nn.Module):
    def __init__(self, kind: str, dim: int, depth: int, grid_h: int, grid_w: int, out_dim: int):
        super().__init__()
        self.kind = kind
        self.grid = (grid_h, grid_w)
        self.blocks = nn.ModuleList([ConvResBlock(dim, kind) for _ in range(depth)])
        self.head = ZeroHead(dim, out_dim)

    def features(self, h: Tensor, ctx: Tensor, ctx_mask: Optional[Tensor], e: Tensor) -> Tensor:
        b, n, d = h.shape
        gh, gw = self.grid
        x = h.transpose(1, 2).reshape(b, d, gh, gw)
        for blk in self.blocks:
            x = blk(x, e)
        return x.reshape(b, d, n).transpose(1, 2)

    def forward(self, h: Tensor, ctx: Tensor, ctx_mask: Optional[Tensor], e: Tensor) -> Tensor:
        return self.head(self.features(h, ctx, ctx_mask, e))
