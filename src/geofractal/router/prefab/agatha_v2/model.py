"""
geofractal.router.prefab.agatha_v2.model
========================================

AgathaFlow: a flow-matching velocity model over VAE latent tokens.

Inputs are exactly (z_t, t, text): the noisy latent, the flow time and a frozen
text encoder's token sequence (with its padding mask). The clean latent never
enters the model except through z_t.

    X  = patchify(z_t)                                     [B, N, P]
    C  = W_c E + b_c                                       [B, L, d]
    e  = MLP([sin(f t'), cos(f t')]),  t' = time_scale t   [B, d]
    h  = trunk(W_in X + b_in; C, e)                        [B, N, d]
    x_ref = X + D_ref(h)                                   (zero last layer)
    Y_k = tower_k(h; C, e)                                 [K, B, N, P] (zero last layers)
    bw = band_weights(band_coordinate(t))                  [B, 3]
    beta, omega, kappa, w = banded parameters(bw)
    u  = sum_k w_k Y_k                                     (or signed dispatch, optional)
    x_N = oscillator(x_0 = X, v_0 = 0; x_ref, u)           N_osc steps
    v_hat = unpatchify( W_v (x_N - X) + b_v )              W_v initialized to the identity, b_v initialized to 0

loss = mean (v_hat - (z_1 - z_0))^2 with z_t = (1 - t) z_0 + t z_1.
sample() and sample_midpoint() integrate dz/dt = v_hat(z, t, text) with the same
velocity() call, hence the same oscillator step count, as loss().

PlainFlow is the control of equal parts: the same front (X, C, e, trunk) and
    v_hat = unpatchify( D(h) ),   D = Linear(LN(h)), zero at initialization
with no towers and no oscillator. It inherits the objective, the condition
handling and the samplers, so both models are trained and sampled by the same
code. build_model(config) returns the model named by config.arch.

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.prefab.agatha_v2.bands import (
    BandParameters,
    SignedDispatch,
    band_coordinate,
    band_weights,
)
from geofractal.router.prefab.agatha_v2.blocks import AxialRotary2D, TransformerBlock
from geofractal.router.prefab.agatha_v2.config import AgathaConfig, plain_counterpart
from geofractal.router.prefab.agatha_v2.oscillator import DampedOscillator, affine_coefficients
from geofractal.router.prefab.agatha_v2.towers import ConvTower, GeometryTower, ZeroHead


def _float_section(device: torch.device):
    """Context in which autocast is off, so a short section runs in full precision."""
    kind = device.type if device.type in ("cuda", "cpu") else "cpu"
    return torch.autocast(device_type=kind, enabled=False)


class TimeEmbedding(nn.Module):
    def __init__(self, freq_dim: int, dim: int, scale: float = 1000.0):
        super().__init__()
        half = freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.scale = float(scale)
        self.mlp = nn.Sequential(nn.Linear(2 * half, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        arg = (t.float() * self.scale)[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1))


class AgathaFlow(nn.Module):
    def __init__(self, config: AgathaConfig):
        super().__init__()
        self.config = config
        self._build_front(config)
        self._build_readout(config)

    def _build_front(self, cfg: AgathaConfig) -> None:
        """Patch embedding, time embedding, text projection, null caption, trunk."""
        d, p_dim = cfg.dim, cfg.token_dim
        self.patch_embed = nn.Linear(p_dim, d)
        self.time_embed = TimeEmbedding(cfg.time_freq_dim, d, cfg.time_scale)
        self.text_proj = nn.Linear(cfg.text_dim, d)
        # States of the empty caption, set from the cache with set_null_text().
        self.register_buffer("null_text", torch.zeros(cfg.null_text_len, cfg.text_dim))

        self.trunk_rotary = AxialRotary2D(cfg.head_dim, cfg.grid_height, cfg.grid_width, cfg.rope_theta)
        self.trunk = nn.ModuleList([TransformerBlock(d, cfg.num_heads, cfg.mlp_ratio)
                                    for _ in range(cfg.trunk_depth)])

    def _build_readout(self, cfg: AgathaConfig) -> None:
        """Anchor head, towers, banded parameters, oscillator, velocity head."""
        d, p_dim, n_tok = cfg.dim, cfg.token_dim, cfg.num_tokens
        self.anchor_head = ZeroHead(d, p_dim)

        towers = []
        for kind in cfg.geometry_towers:
            towers.append(GeometryTower(kind, d, cfg.num_heads, cfg.tower_depth, n_tok, p_dim,
                                        cfg.mlp_ratio, cfg.rope_theta))
        for kind in cfg.conv_towers:
            towers.append(ConvTower(kind, d, cfg.conv_depth, cfg.grid_height, cfg.grid_width, p_dim))
        self.towers = nn.ModuleList(towers)

        k = len(self.towers)
        self.bands = BandParameters(0 if cfg.signed_dispatch else k, cfg.beta_min, cfg.beta_max,
                                    cfg.omega_max, cfg.kappa_init)
        self.dispatch = (SignedDispatch(d, k, cfg.dispatch_dim, cfg.dispatch_tau)
                         if cfg.signed_dispatch else None)
        self.oscillator = DampedOscillator(cfg.num_osc_steps, form="exact")

        self.velocity_head = nn.Linear(p_dim, p_dim)
        with torch.no_grad():
            self.velocity_head.weight.copy_(torch.eye(p_dim))
            self.velocity_head.bias.zero_()

    # ------------------------------------------------------------------ shapes
    def patchify(self, z: Tensor) -> Tensor:
        cfg = self.config
        b, c, hh, ww = z.shape
        if (c, hh, ww) != cfg.latent_shape:
            raise ValueError(f"latent shape {tuple(z.shape[1:])} != configured {cfg.latent_shape}")
        p = cfg.patch_size
        x = z.reshape(b, c, hh // p, p, ww // p, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, (hh // p) * (ww // p), c * p * p)

    def unpatchify(self, x: Tensor) -> Tensor:
        cfg = self.config
        b = x.shape[0]
        p, c = cfg.patch_size, cfg.latent_channels
        gh, gw = cfg.grid_height, cfg.grid_width
        z = x.reshape(b, gh, gw, c, p, p).permute(0, 3, 1, 4, 2, 5)
        return z.reshape(b, c, gh * p, gw * p)

    # ------------------------------------------------------------------ inputs
    def _time(self, t: Union[float, Tensor], b: int, like: Tensor) -> Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t))
        t = t.to(device=like.device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.expand(b)
        if t.shape != (b,):
            raise ValueError(f"t must be a scalar or shape [{b}], got {tuple(t.shape)}")
        return t

    def check_text(self, text: Tensor, text_mask: Optional[Tensor], b: int) -> Tuple[Tensor, Tensor]:
        cfg = self.config
        if text.dim() != 3 or text.shape[0] != b or text.shape[2] != cfg.text_dim:
            raise ValueError(f"text must be [B={b}, L, {cfg.text_dim}], got {tuple(text.shape)}")
        if text_mask is None:
            text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
        if text_mask.shape != text.shape[:2] or text_mask.dtype != torch.bool:
            raise ValueError(f"text_mask must be bool [B, L] = {tuple(text.shape[:2])}, "
                             f"got {tuple(text_mask.shape)} {text_mask.dtype}")
        if not bool(text_mask.any(dim=1).all()):
            raise ValueError("every sample needs at least one valid text token")
        return text, text_mask

    @torch.no_grad()
    def set_null_text(self, states: Tensor) -> None:
        """Store the encoder's states of the empty caption, [null_text_len, text_dim]."""
        if states.shape != self.null_text.shape:
            raise ValueError(f"null text must be {tuple(self.null_text.shape)}, got {tuple(states.shape)}")
        self.null_text.copy_(states.to(self.null_text.dtype))

    def null_condition(self, b: int, length: int) -> Tuple[Tensor, Tensor]:
        """The null sequence padded to `length` tokens, with its mask."""
        n0 = self.null_text.shape[0]
        length = max(length, n0)
        e = self.null_text.new_zeros(b, length, self.config.text_dim)
        e[:, :n0] = self.null_text
        m = torch.zeros(b, length, dtype=torch.bool, device=e.device)
        m[:, :n0] = True
        return e, m

    def drop_condition(self, text: Tensor, text_mask: Tensor, drop: Tensor) -> Tuple[Tensor, Tensor]:
        """Replace the samples where drop is True by the null sequence (per sample, no mixing)."""
        b, length, _ = text.shape
        e0, m0 = self.null_condition(b, length)
        if e0.shape[1] > length:
            pad = e0.shape[1] - length
            text = torch.cat([text, text.new_zeros(b, pad, text.shape[2])], dim=1)
            text_mask = torch.cat([text_mask, text_mask.new_zeros(b, pad)], dim=1)
        dm = drop.view(b, 1)
        return torch.where(dm[..., None], e0.to(text.dtype), text), torch.where(dm, m0, text_mask)

    # ------------------------------------------------------------------ parts
    def encode(self, z_t: Tensor, t: Tensor, text: Tensor, text_mask: Tensor):
        """Shared front: tokens X, trunk states h, projected text C, time embedding e."""
        x = self.patchify(z_t)
        c = self.text_proj(text)
        e = self.time_embed(t)
        h = self.patch_embed(x)
        rot = self.trunk_rotary.rotary()
        for blk in self.trunk:
            h = blk(h, c, text_mask, e, rot)
        return x, h, c, e

    def tower_forces(self, h: Tensor, c: Tensor, text_mask: Tensor, e: Tensor) -> Tensor:
        """All towers, by index: [K, B, N, P]."""
        return torch.stack([tower(h, c, text_mask, e) for tower in self.towers], dim=0)

    def schedule(self, t: Tensor):
        """Band weights [B, 3] and the mixed gains / tower weights for flow time t [B]."""
        bw = band_weights(band_coordinate(t.float(), self.config.shift),
                          self.config.band_edges, self.config.band_half_width)
        beta, omega, kappa, w = self.bands(bw)
        return bw, beta, omega, kappa, w

    def combine_forces(self, forces: Tensor, w: Optional[Tensor], h: Tensor) -> Tensor:
        """u = sum_k w_k Y_k (or the signed dispatch), in at least float32."""
        dtype = torch.promote_types(forces.dtype, torch.float32)
        forces = forces.to(dtype)
        if self.dispatch is not None:
            s = self.dispatch(h.to(dtype))                          # [B, N, K]
            return torch.einsum("bnk,kbnp->bnp", s, forces)
        return torch.einsum("bk,kbnp->bnp", w.to(dtype), forces)

    # ------------------------------------------------------------------ the one step function
    def velocity(self, z_t: Tensor, t: Union[float, Tensor], text: Tensor,
                 text_mask: Optional[Tensor] = None) -> Tensor:
        """Predicted velocity [B, C, H, W] for the noisy latent z_t at flow time t."""
        b = z_t.shape[0]
        t = self._time(t, b, z_t)
        text, text_mask = self.check_text(text, text_mask, b)
        x, h, c, e = self.encode(z_t, t, text, text_mask)
        d_ref = self.anchor_head(h)
        forces = self.tower_forces(h, c, text_mask, e)
        _, beta, omega, kappa, w = self.schedule(t)
        # The banded combination, the oscillator and the velocity head are a few scalars and
        # one P x P matrix: they run in at least float32 even when the caller uses autocast.
        with _float_section(z_t.device):
            dtype = torch.promote_types(x.dtype, torch.float32)
            x = x.to(dtype)
            x_ref = x + d_ref.to(dtype)
            u = self.combine_forces(forces, w, h)
            x_n = self.oscillator(x, x_ref, u, beta, omega, kappa)
            v = self.velocity_head(x_n - x)
        return self.unpatchify(v)

    # ------------------------------------------------------------------ training objective
    def sample_time(self, b: int, device, generator: Optional[torch.Generator] = None) -> Tensor:
        cfg = self.config
        if cfg.t_sampling == "uniform":
            return torch.rand(b, device=device, generator=generator)
        n = torch.randn(b, device=device, generator=generator)
        return torch.sigmoid(cfg.logit_normal_mean + cfg.logit_normal_std * n)

    def loss(self, z_1: Tensor, text: Tensor, text_mask: Optional[Tensor] = None,
             z_0: Optional[Tensor] = None, t: Optional[Tensor] = None,
             drop: Optional[Tensor] = None, generator: Optional[torch.Generator] = None,
             return_parts: bool = False):
        """Flow-matching squared error. Condition dropout applies in training mode only,
        unless `drop` [B] bool is given explicitly."""
        b = z_1.shape[0]
        text, text_mask = self.check_text(text, text_mask, b)
        if z_0 is None:
            z_0 = torch.randn(z_1.shape, device=z_1.device, dtype=z_1.dtype, generator=generator)
        if t is None:
            t = self.sample_time(b, z_1.device, generator)
        t = self._time(t, b, z_1)
        if drop is None and self.training and self.config.cond_dropout > 0:
            drop = torch.rand(b, device=z_1.device, generator=generator) < self.config.cond_dropout
        if drop is not None and bool(drop.any()):
            text, text_mask = self.drop_condition(text, text_mask, drop.to(torch.bool))
        tt = t.view(b, 1, 1, 1).to(z_1.dtype)
        z_t = (1 - tt) * z_0 + tt * z_1
        v_target = z_1 - z_0
        v_hat = self.velocity(z_t, t, text, text_mask)
        loss = ((v_hat.float() - v_target.float()) ** 2).mean()
        if return_parts:
            return loss, v_hat, v_target
        return loss

    # ------------------------------------------------------------------ samplers
    def _guided(self, z: Tensor, t: float, text: Tensor, text_mask: Tensor,
                guidance_scale: Optional[float]) -> Tensor:
        v_c = self.velocity(z, t, text, text_mask)
        if guidance_scale is None:
            return v_c
        e0, m0 = self.null_condition(z.shape[0], 1)
        v_0 = self.velocity(z, t, e0.to(text.dtype), m0)
        return v_0 + guidance_scale * (v_c - v_0)

    def _start(self, text: Tensor, noise: Optional[Tensor], generator: Optional[torch.Generator]) -> Tensor:
        if noise is not None:
            return noise
        b = text.shape[0]
        return torch.randn((b,) + self.config.latent_shape, device=text.device, generator=generator)

    @torch.no_grad()
    def sample(self, text: Tensor, text_mask: Optional[Tensor] = None, noise: Optional[Tensor] = None,
               num_steps: Optional[int] = None, guidance_scale: Optional[float] = None,
               generator: Optional[torch.Generator] = None) -> Tensor:
        """Euler integration of dz/dt = v_hat from t = 0 (noise) to t = 1 (data)."""
        n = num_steps or self.config.sample_steps
        z = self._start(text, noise, generator)
        text, text_mask = self.check_text(text, text_mask, z.shape[0])
        dt = 1.0 / n
        for i in range(n):
            z = z + dt * self._guided(z, i / n, text, text_mask, guidance_scale)
        return z

    @torch.no_grad()
    def sample_midpoint(self, text: Tensor, text_mask: Optional[Tensor] = None, noise: Optional[Tensor] = None,
                        num_steps: Optional[int] = None, guidance_scale: Optional[float] = None,
                        generator: Optional[torch.Generator] = None) -> Tensor:
        """Midpoint (second-order) integration with the same velocity() call."""
        n = num_steps or self.config.sample_steps
        z = self._start(text, noise, generator)
        text, text_mask = self.check_text(text, text_mask, z.shape[0])
        dt = 1.0 / n
        for i in range(n):
            t0 = i / n
            v1 = self._guided(z, t0, text, text_mask, guidance_scale)
            v2 = self._guided(z + 0.5 * dt * v1, t0 + 0.5 * dt, text, text_mask, guidance_scale)
            z = z + dt * v2
        return z

    # ------------------------------------------------------------------ diagnostics (never in forward)
    @torch.no_grad()
    def diagnostics(self, z_t: Tensor, t: Union[float, Tensor], text: Tensor,
                    text_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        b = z_t.shape[0]
        t = self._time(t, b, z_t)
        text, text_mask = self.check_text(text, text_mask, b)
        x, h, c, e = self.encode(z_t, t, text, text_mask)
        x = x.float()
        x_ref = x + self.anchor_head(h).float()
        forces = self.tower_forces(h, c, text_mask, e)
        bw, beta, omega, kappa, w = self.schedule(t)
        u = self.combine_forces(forces, w, h)
        a_n, g_n = affine_coefficients(beta.double(), omega.double(), self.oscillator.num_steps)
        out = {
            "band_weights": bw, "beta": beta, "omega": omega, "kappa": kappa,
            "A_n": a_n.float(), "G_n": g_n.float(),
            "anchor_gain": (g_n * omega.double() ** 2).float(),
            "force_gain": (g_n * kappa.double()).float(),
            "force_norm_per_tower": forces.flatten(2).norm(dim=-1),          # [K, B]
            "energy": self.oscillator.trajectory_energy(x, x_ref, u, beta, omega, kappa),
        }
        if w is not None:
            out["tower_weights"] = w
        return out

    def num_trained_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PlainFlow(AgathaFlow):
    """The control: AgathaFlow's front and trunk, then v_hat = unpatchify(Linear(LN(h))).

    No towers, no banded parameters, no oscillator. The head's weight and bias are zero
    at initialization (there is no gate), so the first velocity is exactly zero, as for
    AgathaFlow. The head runs in at least float32 under autocast, like AgathaFlow's
    velocity head. `towers` is an empty list, so tower-wise utilities see no towers.
    """

    def _build_readout(self, cfg: AgathaConfig) -> None:
        self.head = ZeroHead(cfg.dim, cfg.token_dim)
        self.towers = nn.ModuleList()

    def velocity(self, z_t: Tensor, t: Union[float, Tensor], text: Tensor,
                 text_mask: Optional[Tensor] = None) -> Tensor:
        """Predicted velocity [B, C, H, W] for the noisy latent z_t at flow time t."""
        b = z_t.shape[0]
        t = self._time(t, b, z_t)
        text, text_mask = self.check_text(text, text_mask, b)
        _, h, _, _ = self.encode(z_t, t, text, text_mask)
        with _float_section(z_t.device):
            v = self.head(h.to(torch.promote_types(h.dtype, torch.float32)))
        return self.unpatchify(v)

    def tower_forces(self, h, c, text_mask, e):
        raise NotImplementedError("the plain control has no towers")

    def schedule(self, t):
        raise NotImplementedError("the plain control has no banded parameters")

    def combine_forces(self, forces, w, h):
        raise NotImplementedError("the plain control has no towers")

    @torch.no_grad()
    def diagnostics(self, z_t: Tensor, t: Union[float, Tensor], text: Tensor,
                    text_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Norms of the trunk states and of the velocity (the plain model has no gains)."""
        b = z_t.shape[0]
        t = self._time(t, b, z_t)
        text, text_mask = self.check_text(text, text_mask, b)
        _, h, _, _ = self.encode(z_t, t, text, text_mask)
        v = self.velocity(z_t, t, text, text_mask)
        return {"trunk_norm": h.float().flatten(1).norm(dim=1), "velocity_norm": v.flatten(1).norm(dim=1)}


def build_model(config: AgathaConfig) -> AgathaFlow:
    """The model named by config.arch: AgathaFlow ("agatha") or PlainFlow ("plain")."""
    if config.arch == "plain":
        return PlainFlow(config)
    return AgathaFlow(config)


def count_trained_parameters(config: AgathaConfig) -> int:
    """Trained parameters of the model a configuration builds (the model is built on the CPU)."""
    return build_model(config).num_trained_parameters()


def match_plain_trunk_depth(config: AgathaConfig, max_depth: int = 256) -> Tuple[int, int, int]:
    """The plain control's trunk depth whose trained-parameter count is closest to `config`'s.

    Returns (depth, plain_parameters, reference_parameters). Trunk blocks are identical, so the
    plain count is affine in the depth: count(D) = count(0) + D * (count(1) - count(0)); the two
    end points are built and the depth is the nearest whole number in [1, max_depth].
    """
    reference = count_trained_parameters(config)
    base = count_trained_parameters(plain_counterpart(config, 0))
    per_block = count_trained_parameters(plain_counterpart(config, 1)) - base
    best = min(range(1, max_depth + 1), key=lambda depth: abs(base + depth * per_block - reference))
    return best, base + best * per_block, reference
