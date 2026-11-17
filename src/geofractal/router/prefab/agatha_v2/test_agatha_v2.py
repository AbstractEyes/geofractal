"""
Tests for geofractal.router.prefab.agatha_v2 (CPU, smoke size, no training).

Each test carries a control that must be able to fail. Where the control is the
original implementation, the original folder geofractal.router.prefab.agatha is
imported read-only at the smallest size at which it constructs; if a package it
needs is missing in the running environment, that control is skipped with the
missing package named.

Run with pytest:
    python -B -m pytest -q -p no:cacheprovider .../agatha_v2/test_agatha_v2.py
or without pytest:
    python -B .../agatha_v2/test_agatha_v2.py
"""

from __future__ import annotations

import inspect
import io
import os
import sys
import tokenize
import unittest

import torch
import torch.nn as nn

from geofractal.router.prefab.agatha_v2 import bands as bands_mod
from geofractal.router.prefab.agatha_v2 import oscillator as osc_mod
from geofractal.router.prefab.agatha_v2.config import smoke_config
from geofractal.router.prefab.agatha_v2.model import AgathaFlow
from geofractal.router.prefab.agatha_v2.blocks import AxialRotary2D, identity_rotary, table_rotary
from geofractal.router.prefab.agatha_v2.towers import GeometryTower, PositionCode, build_position_code
from geofractal.router.prefab.agatha_v2.trainer import (
    build_optimizer,
    equalize_tower_gradients,
    load_checkpoint,
    save_checkpoint,
)
from geofractal.router.components.rope_component import BeatrixRoPE, CantorRoPE, RoPE

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DIR = os.path.join(os.path.dirname(HERE), "agatha")
REPORT = {}


def _record(key, value):
    REPORT[key] = value
    print(f"  [{key}] {value}")


# =============================================================================
# helpers
# =============================================================================

def _skip(reason):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        raise unittest.SkipTest(reason)


def _original(module):
    """Import a module of the original folder (read-only); skip if a dependency is missing."""
    import importlib
    try:
        return importlib.import_module(f"geofractal.router.prefab.agatha.{module}")
    except ImportError as e:  # noqa: PERF203
        _skip(f"original control needs a package missing here: {e.name or e}")


def _model(seed=0, **over):
    torch.manual_seed(seed)
    return AgathaFlow(smoke_config(**over))


def _inputs(cfg, b=4, seed=1, length=6):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn((b,) + cfg.latent_shape, generator=g)
    t = torch.tensor([0.1, 0.45, 0.9, 0.6, 0.3, 0.8][:b])  # sigma 0.9 / 0.55 / 0.1 / 0.4: all three bands
    text = torch.randn(b, length, cfg.text_dim, generator=g)
    mask = torch.ones(b, length, dtype=torch.bool)
    mask[1, -2:] = False
    return z, t, text, mask


def _head_layers(model):
    return [model.anchor_head.proj] + [tw.head.proj for tw in model.towers]


@torch.no_grad()
def _perturb(model, seed=7, std=0.05, heads_only=False):
    """Set zero-initialized tensors to small random values (the state after a first step)."""
    g = torch.Generator().manual_seed(seed)
    if heads_only:
        targets = [p for lin in _head_layers(model) for p in lin.parameters()]
    else:
        targets = [p for p in model.parameters() if p.requires_grad and bool((p == 0).all())]
    for p in targets:
        p.copy_(torch.randn(p.shape, generator=g) * std)
    return model


def _no_grad_names(model):
    out = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or float(p.grad.abs().sum()) == 0.0:
            out.append(n)
    return out


def _backward(model, seed=1, **kw):
    z, t, text, mask = _inputs(model.config, seed=seed)
    model.zero_grad(set_to_none=True)
    g = torch.Generator().manual_seed(3)
    loss = model.loss(z, text, mask, t=t, drop=torch.zeros(z.shape[0], dtype=torch.bool), generator=g, **kw)
    loss.backward()
    return loss


class _Ident(nn.Module):
    def forward(self, x):
        return x


class _MockDino(nn.Module):
    def __init__(self, h=24):
        super().__init__()
        self.hidden_size = h
        self.lin = nn.Linear(3 * 8 * 8, h)

    def forward(self, x):
        return self.lin(x.flatten(1))


_ORIGINAL_CACHE = {}


def _original_beatrix():
    """The original model at the smallest size it constructs (latent 2x4x4, width 256, fingerprint 64)."""
    if "beatrix" not in _ORIGINAL_CACHE:
        bx = _original("beatrix")
        torch.manual_seed(0)
        cfg = bx.BeatrixConfig(latent_channels=2, latent_height=4, latent_width=4, head_embed_dim=256,
                               fingerprint_dim=64, num_training_steps=10, num_steps=3)
        m = bx.Beatrix("beatrix", cfg)
        m.attach_flux_ae(_Ident())
        m.attach_dino(_MockDino(24))
        m.eval()
        _ORIGINAL_CACHE["beatrix"] = m
    return _ORIGINAL_CACHE["beatrix"]


# =============================================================================
# T1  the spring converges; the gain bounds are stable
# =============================================================================

def _pure_band_times():
    return torch.tensor([1 - 0.15, 1 - 0.55, 1 - 0.9])  # band coordinate 0.15 / 0.55 / 0.9 (shift 1)


def test_t1a_spring_converges_at_initialization_and_original_diverges():
    worst = 0.0
    for n in (1, 4, 16):
        m = _model(num_osc_steps=n)
        t = _pure_band_times()
        bw, beta, omega, kappa, _ = m.schedule(t)
        assert torch.allclose(bw, torch.eye(3), atol=1e-6), "times are not inside single bands"
        g = torch.Generator().manual_seed(n)
        x0 = torch.randn(3, 16, 16, generator=g)
        xr = torch.randn(3, 16, 16, generator=g)
        for form in (osc_mod.integrate_exact, osc_mod.integrate_stepped):
            xn = form(x0, xr, None, beta.detach(), omega.detach(), kappa.detach(), n)
            ratio = (xn - xr).flatten(1).norm(dim=1) / (x0 - xr).flatten(1).norm(dim=1)
            assert bool((ratio < 1).all()), f"N={n} {form.__name__} ratios {ratio.tolist()}"
            worst = max(worst, float(ratio.max()))
    _record("T1a new: largest distance ratio at initialization (N 1/4/16, 3 bands)", round(worst, 4))

    # control: the original oscillator, towers and guidance off, v_0 = 0, same inputs (flattened)
    bo = _original("beatrix_oscillator")
    ratios = {}
    for n in (1, 4, 16):
        g = torch.Generator().manual_seed(n)
        x0 = torch.randn(3, 16, 16, generator=g).flatten(1)
        xr = torch.randn(3, 16, 16, generator=g).flatten(1)
        torch.manual_seed(1)
        osc = bo.BeatrixOscillator(manifold_dim=x0.shape[1], tower_dim=16, num_tower_pairs=4, num_theta_probes=4)
        with torch.no_grad():
            osc.initial_velocity_scale.fill_(0.0)
            xn, _ = osc.integrate(x0, xr, tower_outputs_fn=lambda s: [], num_steps=n)
        ratios[n] = float(((xn - xr).norm(dim=1) / (x0 - xr).norm(dim=1)).min())
        assert ratios[n] > 1.0, f"original did not diverge at N={n}: {ratios[n]}"
    _record("T1a control: original smallest distance ratio by N", {k: round(v, 4) for k, v in ratios.items()})


def _grid_check(beta_fn, omega_fn, steps_list, n_grid=25, lo=-12.0, hi=12.0):
    raw = torch.linspace(lo, hi, n_grid, dtype=torch.float64)
    rho, phi = torch.meshgrid(raw, raw, indexing="ij")
    worst_rad, worst_a, bad = 0.0, 0.0, 0
    for n in steps_list:
        beta = beta_fn(rho, n)
        omega = omega_fn(phi, n)
        rad = osc_mod.spectral_radius(beta, omega, n)
        a_n, _ = osc_mod.affine_coefficients(beta, omega, n)
        bad += int(((rad >= 1) | (a_n.abs() >= 1)).sum())
        worst_rad = max(worst_rad, float(rad.max()))
        worst_a = max(worst_a, float(a_n.abs().max()))
    return bad, worst_rad, worst_a


def test_t1b_gain_bounds_stable_over_whole_raw_range():
    steps = (1, 2, 4, 8, 16, 32, 64)

    def beta_new(rho, n):
        c = smoke_config(num_osc_steps=n)
        return osc_mod.damping_from_raw(rho, c.beta_min, c.beta_max)

    def omega_new(phi, n):
        c = smoke_config(num_osc_steps=n)
        return osc_mod.frequency_from_raw(phi, c.omega_max)

    bad, rad, a = _grid_check(beta_new, omega_new, steps)
    _record("T1b new: unstable grid points / 1 - max spectral radius / 1 - max |A_N|",
            (bad, f"{1 - rad:.3e}", f"{1 - a:.3e}"))
    bad_i, rad_i, a_i = _grid_check(beta_new, omega_new, steps, lo=-4.0, hi=4.0)
    _record("T1b new, raw range [-4, 4]: unstable / max radius / max |A_N|", (bad_i, round(rad_i, 4), round(a_i, 4)))
    assert bad == 0 and rad < 1 and a < 1

    # control: looser bounds (omega_max = 1.9 N, damping = softplus(rho) with no upper limit)
    # leave the stable region, so the same grid check must find unstable gains
    bad_d, rad_d, a_d = _grid_check(lambda r, n: torch.nn.functional.softplus(r),
                                    lambda p, n: 1.9 * n * torch.sigmoid(p), steps)
    _record("T1b control (looser bounds): unstable points / max radius / max |A_N|",
            (bad_d, round(rad_d, 3), round(a_d, 3)))
    assert bad_d > 0


# =============================================================================
# T2  no path from the target to the prediction
# =============================================================================

def test_t2_no_target_path():
    sig = inspect.signature(AgathaFlow.velocity)
    assert list(sig.parameters) == ["self", "z_t", "t", "text", "text_mask"]

    m = _perturb(_model()).eval()
    cfg = m.config
    z1, t, text, mask = _inputs(cfg)
    g = torch.Generator().manual_seed(5)
    z0 = torch.randn(z1.shape, generator=g)
    tt = t.view(-1, 1, 1, 1)
    z_t = (1 - tt) * z0 + tt * z1
    z1b = z1 + torch.randn(z1.shape, generator=g)
    z0b = (z_t - tt * z1b) / (1 - tt)
    nodrop = torch.zeros(4, dtype=torch.bool)
    with torch.no_grad():
        _, va, _ = m.loss(z1, text, mask, z_0=z0, t=t, drop=nodrop, return_parts=True)
        _, vb, _ = m.loss(z1b, text, mask, z_0=z0b, t=t, drop=nodrop, return_parts=True)
        # the check can see a change: move z_t itself
        vc = m.velocity(z_t + 0.1, t, text, mask)
    change = float((va - vb).abs().max())
    seen = float((va - vc).abs().max())
    _record("T2 new: max change of v_hat when only z_1 changes (z_t fixed) / when z_t moves by 0.1",
            (f"{change:.2e}", f"{seen:.3g}"))
    assert change < 1e-5 and seen > 1e-3


def test_t2_control_original_moves():
    """Control for T2: the original model moves when only the clean latent changes."""
    om = _original_beatrix()
    g = torch.Generator().manual_seed(6)
    z1 = torch.randn(4, 2, 4, 4, generator=g)
    img = torch.rand(4, 3, 8, 8, generator=g)
    t = torch.full((4,), 0.3)
    z0 = torch.randn(z1.shape, generator=g)
    z_t = 0.7 * z0 + 0.3 * z1
    z1b = z1 + torch.randn(z1.shape, generator=g)
    z0b = (z_t - 0.3 * z1b) / 0.7
    with torch.no_grad():
        torch.manual_seed(11)
        va, _, _ = om({"flux_ae": z1, "dino": img}, t=t, z_0=z0)
        torch.manual_seed(11)
        vb, _, _ = om({"flux_ae": z1b, "dino": img}, t=t, z_0=z0b)
    ochange = float((va - vb).abs().max())
    _record("T2 control: original max change of v_pred when only z_1 changes", round(ochange, 4))
    assert ochange > 1e-3


# =============================================================================
# T3  every trained parameter gets a gradient
# =============================================================================

def test_t3_gradients():
    # (a) at initialization: the head last layers and the velocity head's bias
    m = _model()
    _backward(m)
    dead = set(_no_grad_names(m))
    head_names = [f"anchor_head.proj.weight"] + [f"towers.{i}.head.proj.weight" for i in range(len(m.towers))]
    for n in head_names + ["velocity_head.bias"]:
        assert n not in dead, f"(a) no gradient at initialization: {n}"
    _record("T3a: tensors with a gradient at initialization / trained tensors",
            (sum(1 for _ in m.parameters() if _.requires_grad) - len(dead),
             sum(1 for _ in m.parameters() if _.requires_grad)))

    # (a2) only the head last layers moved: every modulation layer and every trunk / tower weight
    m2 = _perturb(_model(), heads_only=True)
    _backward(m2)
    dead2 = _no_grad_names(m2)
    upstream = [n for n in dead2 if n.startswith(("trunk.", "towers."))]
    modul = [n for n, _ in m2.named_parameters() if (".modulation." in n or ".film." in n)]
    assert modul, "no modulation layers found"
    assert not upstream, f"(a2) upstream tensors without gradient: {upstream}"

    # (b) every zero layer perturbed: every trained tensor (default and with the signed dispatch option)
    for over in ({}, {"signed_dispatch": True}):
        m3 = _perturb(_model(**over))
        _backward(m3)
        dead3 = _no_grad_names(m3)
        assert not dead3, f"(b) {over}: tensors without gradient: {dead3}"
    frozen = {f"towers.{i}.{tw.kind}": tw.position_code.frozen_position_parameters
              for i, tw in enumerate(m3.towers) if isinstance(tw, GeometryTower)}
    _record("T3b: position-code parameters never read by the chosen mode (frozen, not trained)", frozen)

    # control: a deliberately detached tower must be reported as gradient-free
    m4 = _perturb(_model())
    k = 2
    tw = m4.towers[k]
    orig_forward = tw.forward
    tw.forward = lambda *a, **kw: orig_forward(*a, **kw).detach()
    _backward(m4)
    dead4 = set(_no_grad_names(m4))
    tower_params = {f"towers.{k}.{n}" for n, p in tw.named_parameters() if p.requires_grad}
    assert tower_params and tower_params <= dead4 and not (dead4 - tower_params), \
        f"detached tower not seen exactly: missing {tower_params - dead4}, extra {dead4 - tower_params}"
    _record("T3 control: detached tower tensors reported gradient-free", len(tower_params))


# =============================================================================
# T4  the weighting and the forces depend on the input
# =============================================================================

def test_t4_weights_and_forces_depend_on_input():
    m = _perturb(_model()).eval()
    t = _pure_band_times()
    with torch.no_grad():
        _, _, _, _, w_init = m.schedule(t)
        assert torch.equal(w_init[0], w_init[2])          # initialized to 1/K in every band
        m.bands.weights.add_(0.1 * torch.randn_like(m.bands.weights))   # as after training
        _, _, _, _, w = m.schedule(t)
        _, _, _, _, w_again = m.schedule(t)
    assert torch.equal(w, w_again)
    band_spread = float((w[0] - w[2]).abs().max())
    assert band_spread > 1e-4

    cfg = m.config
    z, tt, text, mask = _inputs(cfg)
    z2 = torch.randn_like(z)
    text2 = torch.randn_like(text)
    with torch.no_grad():
        def forces(zz, txt):
            x, h, c, e = m.encode(zz, tt, txt, mask)
            return m.tower_forces(h, c, mask, e)
        f0 = forces(z, text)
        dz = (f0 - forces(z2, text)).flatten(2).abs().amax(dim=2).amin(dim=1)
        dt_ = (f0 - forces(z, text2)).flatten(2).abs().amax(dim=2).amin(dim=1)
    assert bool((dz > 1e-5).all()), f"some tower ignores z_t: {dz.tolist()}"
    assert bool((dt_ > 1e-6).all()), f"some tower ignores the caption: {dt_.tolist()}"
    _record("T4 new: band weight spread / min force change per tower for new z_t / new caption",
            (round(band_spread, 4), [f"{v:.2e}" for v in dz.tolist()], [f"{v:.2e}" for v in dt_.tolist()]))


def test_t4_control_original_routing_constant():
    """Control for T4: the original's routing weights are the same for different inputs."""
    om = _original_beatrix()
    fg = om.denoiser["oscillator"].force_generator
    seen = []
    orig = fg.compute_routing_weights

    def capture(fp):
        w_ = orig(fp)
        seen.append(w_.detach().clone())
        return w_

    fg.compute_routing_weights = capture
    try:
        g = torch.Generator().manual_seed(8)
        with torch.no_grad():
            for _ in range(2):
                z1 = torch.randn(4, 2, 4, 4, generator=g)
                img = torch.rand(4, 3, 8, 8, generator=g)
                torch.manual_seed(1)
                om({"flux_ae": z1, "dino": img}, t=torch.rand(4, generator=g), num_steps=2)
    finally:
        fg.compute_routing_weights = orig
    first, second = seen[0], seen[len(seen) // 2]
    same = torch.equal(first, second) and torch.allclose(first[0], first[1])
    _record("T4 control: original routing weights identical across inputs and rows", same)
    assert same


# =============================================================================
# T5  the position code changes the output; reused codes depend on position
# =============================================================================

@torch.no_grad()
def _code_checks(code, kind, n):
    """(rotates, matches its own definition at length n)."""
    g = torch.Generator().manual_seed(n)
    q = torch.randn(1, 1, 1, code.head_dim, generator=g).expand(1, 1, n, code.head_dim).contiguous()
    q2, _ = code.rotate(q, q, seq_len=n)
    rel = float((q2 - q).norm() / q.norm())
    distinct = float((q2[0, 0, 1:] - q2[0, 0, :1]).norm(dim=-1).min()) if n > 1 else 0.0
    own = True
    if kind == "beatrix":
        cos, sin = code.embed(n)
        pos = code.get_cantor_positions(n)
        ref = torch.outer(pos, code.inv_freq)
        ref = torch.cat([ref, ref], dim=-1)
        own = bool(torch.allclose(cos, ref.cos(), atol=1e-4) and torch.allclose(sin, ref.sin(), atol=1e-4))
    elif kind == "cantor":
        cos, sin = code.embed(n)
        tp = code.get_path_positions(n)
        fp = torch.cat([torch.outer(tp, code.inv_freq)] * 2, dim=-1)
        fs = torch.cat([torch.outer(torch.arange(n, dtype=torch.float32), code.inv_freq)] * 2, dim=-1)
        a = torch.sigmoid(code._blend_alpha)
        c = (1 - a) * fs.cos() + a * fp.cos()
        s = (1 - a) * fs.sin() + a * fp.sin()
        mag = torch.sqrt(c ** 2 + s ** 2 + 1e-8)
        own = bool(torch.allclose(cos, c / mag, atol=1e-4) and torch.allclose(sin, s / mag, atol=1e-4))
    return rel, distinct, own


def test_t5_position_codes():
    # (1) each geometry tower's output changes when its code is replaced by the identity (N = 16)
    m = _perturb(_model()).eval()
    z, t, text, mask = _inputs(m.config)
    changes = {}
    with torch.no_grad():
        x, h, c, e = m.encode(z, t, text, mask)
        for tw in m.towers:
            if not isinstance(tw, GeometryTower):
                continue
            a = tw(h, c, mask, e)
            tw.use_position_code = False
            b = tw(h, c, mask, e)
            tw.use_position_code = True
            changes[tw.kind] = float((a - b).abs().max())
    _record("T5 new N=16: max tower output change, code -> identity", {k: f"{v:.2e}" for k, v in changes.items()})
    assert all(v > 1e-5 for v in changes.values())

    # control: at one token the same replacement changes nothing
    m1 = _perturb(_model(latent_height=2, latent_width=2)).eval()
    z, t, text, mask = _inputs(m1.config)
    with torch.no_grad():
        x, h, c, e = m1.encode(z, t, text, mask)
        for tw in m1.towers:
            if isinstance(tw, GeometryTower):
                a = tw(h, c, mask, e)
                tw.use_position_code = False
                b = tw(h, c, mask, e)
                tw.use_position_code = True
                assert float((a - b).abs().max()) == 0.0, tw.kind

    # (2) every reused code depends on position at the token counts used (16 smoke, 256 default),
    #     and the length-normalized codes match their own definition at that length
    rows = {}
    for n in (16, 256):
        for kind in ("cantor", "beatrix", "simplex", "helix"):
            code = build_position_code(kind, 64, n)
            rel, distinct, own = _code_checks(code, kind, n)
            rows[(kind, n)] = (round(rel, 3), round(distinct, 3), own)
            assert rel > 0.05 and distinct > 1e-3 and own, (kind, n, rel, distinct, own)
    _record("T5 codes (rel rotation, min distinctness, matches own definition)", rows)

    # control: the same checks on the library default construction (cache length 8192) must fail
    bad_b = _code_checks(BeatrixRoPE("b", head_dim=64), "beatrix", 16)
    bad_c = _code_checks(CantorRoPE("c", head_dim=64, levels=5, tau=0.25, mode="hybrid"), "cantor", 16)
    _record("T5 control: default-cache BeatrixRoPE / CantorRoPE at 16 tokens", (bad_b, bad_c))
    assert bad_b[0] < 0.05 and not bad_b[2]
    assert not bad_c[2]


# =============================================================================
# T6  one step function for training and sampling
# =============================================================================

def _trace(model, fn):
    calls = {"velocity": 0, "osc_steps": []}
    orig_v = model.velocity
    orig_o = model.oscillator.forward

    def v(*a, **k):
        calls["velocity"] += 1
        return orig_v(*a, **k)

    def o(*a, **k):
        calls["osc_steps"].append(model.oscillator.num_steps)
        return orig_o(*a, **k)

    model.velocity = v
    model.oscillator.forward = o
    try:
        fn()
    finally:
        del model.velocity
        del model.oscillator.forward
    return calls


def test_t6_same_step_function():
    m = _perturb(_model()).eval()
    z, t, text, mask = _inputs(m.config)
    tr_loss = _trace(m, lambda: m.loss(z, text, mask, t=t))
    tr_samp = _trace(m, lambda: m.sample(text, mask, noise=z, num_steps=3))
    tr_mid = _trace(m, lambda: m.sample_midpoint(text, mask, noise=z, num_steps=2))
    assert tr_loss["velocity"] == 1 and tr_samp["velocity"] == 3 and tr_mid["velocity"] == 4
    steps = set(tr_loss["osc_steps"]) | set(tr_samp["osc_steps"]) | set(tr_mid["osc_steps"])
    assert steps == {m.config.num_osc_steps}
    with torch.no_grad():
        one = m.sample(text, mask, noise=z, num_steps=1)
        ref = z + m.velocity(z, 0.0, text, mask)
    assert torch.equal(one, ref)

    # control: a sampler patched to one oscillator step is detected
    def patched():
        m.oscillator.num_steps = 1
        try:
            m.sample(text, mask, noise=z, num_steps=3)
        finally:
            m.oscillator.num_steps = m.config.num_osc_steps

    tr_bad = _trace(m, patched)
    assert set(tr_bad["osc_steps"]) != set(tr_loss["osc_steps"])
    _record("T6: oscillator steps loss / sample / midpoint / patched sampler",
            (tr_loss["osc_steps"], tr_samp["osc_steps"], tr_mid["osc_steps"], tr_bad["osc_steps"]))


# =============================================================================
# T7  save and reload predict the same; the optimizer is pure Adam over every trained tensor
# =============================================================================

def test_t7_checkpoint_roundtrip():
    m = _perturb(_model(seed=0)).eval()
    with torch.no_grad():
        m.set_null_text(torch.randn(1, m.config.text_dim))
    opt = build_optimizer(m)
    assert type(opt) is torch.optim.Adam
    assert all(gr["weight_decay"] == 0 for gr in opt.param_groups)
    opt_ids = {id(p) for gr in opt.param_groups for p in gr["params"]}
    assert opt_ids == {id(p) for p in m.parameters() if p.requires_grad}

    buf = io.BytesIO()
    save_checkpoint(buf, m, opt, step=0)
    buf.seek(0)
    m2, opt2, step = load_checkpoint(buf)
    m2.eval()
    z, t, text, mask = _inputs(m.config)
    with torch.no_grad():
        a = m.velocity(z, t, text, mask)
        b = m2.velocity(z, t, text, mask)
    assert not torch.isnan(a).any() and torch.equal(a, b)
    assert isinstance(opt2, torch.optim.Adam)

    # control: another seed, no load
    m3 = _perturb(_model(seed=1), seed=8).eval()
    with torch.no_grad():
        c = m3.velocity(z, t, text, mask)
    diff = float((a - c).abs().max())
    _record("T7: reload max diff / other-seed max diff", (float((a - b).abs().max()), round(diff, 4)))
    assert diff > 1e-3


# =============================================================================
# T8  the exact recurrence equals the stepped oscillator
# =============================================================================

def test_t8_exact_equals_stepped():
    worst64, worst32, flip = 0.0, 0.0, float("inf")
    for n in (1, 4, 16):
        cfg = smoke_config(num_osc_steps=n)
        g = torch.Generator().manual_seed(n)
        beta = osc_mod.damping_from_raw(torch.randn(5, generator=g, dtype=torch.float64) * 3, cfg.beta_min, cfg.beta_max)
        omega = osc_mod.frequency_from_raw(torch.randn(5, generator=g, dtype=torch.float64) * 3, cfg.omega_max)
        kappa = osc_mod.control_gain_from_raw(torch.randn(5, generator=g, dtype=torch.float64))
        x0, xr, u = (torch.randn(5, 16, 16, generator=g, dtype=torch.float64) for _ in range(3))
        ex = osc_mod.integrate_exact(x0, xr, u, beta, omega, kappa, n)
        st = osc_mod.integrate_stepped(x0, xr, u, beta, omega, kappa, n)
        scale = float(st.abs().max())
        worst64 = max(worst64, float((ex - st).abs().max()) / scale)
        ex32 = osc_mod.integrate_exact(*(v.float() for v in (x0, xr, u, beta, omega, kappa)), n)
        st32 = osc_mod.integrate_stepped(*(v.float() for v in (x0, xr, u, beta, omega, kappa)), n)
        worst32 = max(worst32, float((ex32 - st32).abs().max()) / scale)
        bad = osc_mod.integrate_stepped(x0, xr, u, beta, omega, kappa, n, spring_sign=-1.0)
        flip = min(flip, float((ex - bad).abs().max()) / scale)
    _record("T8: relative max diff fp64 / fp32 / sign-flipped control", (f"{worst64:.2e}", f"{worst32:.2e}", f"{flip:.3g}"))
    assert worst64 <= 1e-12 and worst32 <= 1e-5 and flip > 1e-3


# =============================================================================
# T9  band coordinate and windows
# =============================================================================

def _band_failures(weights_of_t, shift):
    fails = []
    t = torch.linspace(0, 1, 2001, dtype=torch.float64)
    w = weights_of_t(t, shift)
    if not torch.allclose(w.sum(-1), torch.ones_like(t), atol=1e-12):
        fails.append("sum")
    if bool((w < 0).any()):
        fails.append("negative")
    s_ref = shift * (1 - t) / (1 + (shift - 1) * (1 - t))
    # pure windows away from the edges, half/half at the edges
    for s_val, j in ((0.2, 0), (0.55, 1), (0.9, 2)):
        tt = 1 - torch.tensor([s_val / (shift - (shift - 1) * s_val)], dtype=torch.float64)
        if abs(float(weights_of_t(tt, shift)[0, j]) - 1.0) > 1e-9:
            fails.append(f"pure band {j}")
    for e in (0.35, 0.75):
        for off, expect_up in ((-0.0601, 0.0), (0.0601, 1.0), (0.0, 0.5)):
            sv = e + off
            tt = 1 - torch.tensor([sv / (shift - (shift - 1) * sv)], dtype=torch.float64)
            ww = weights_of_t(tt, shift)[0]
            up = float(ww[1] + ww[2]) if e == 0.35 else float(ww[2])
            if abs(up - expect_up) > 1e-6:
                fails.append(f"edge {e}{off:+}")
    if not torch.allclose(bands_mod.band_coordinate(t, shift), s_ref, atol=1e-12):
        fails.append("coordinate")
    return fails


def test_t9_band_coordinate():
    def real(t, shift):
        return bands_mod.band_weights(bands_mod.band_coordinate(t, shift))

    def on_t(t, shift):  # sabotage: windows on t instead of the warped sigma
        return bands_mod.band_weights(t)

    for shift in (1.0, 3.0):
        assert _band_failures(real, shift) == [], shift
    bad = _band_failures(on_t, 3.0)
    _record("T9 control: failures when the windows read t instead of sigma", bad)
    assert bad


# =============================================================================
# T10  shapes, smoke forward and backward; wrong inputs raise
# =============================================================================

def test_t10_shapes_and_errors():
    m = _perturb(_model())
    cfg = m.config
    z, t, text, mask = _inputs(cfg)
    v = m.velocity(z, t, text, mask)
    assert v.shape == z.shape
    loss = _backward(m)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

    norms = equalize_tower_gradients(m)
    after = [float(torch.stack([p.grad.pow(2).sum() for p in tw.parameters() if p.grad is not None]).sum().sqrt())
             for tw in m.towers]
    med = sorted(norms)[len(norms) // 2]
    assert all(abs(a - med) <= 1e-4 * med for a in after)

    bad_inputs = [
        (z, t, text[:3], mask[:3]),                                   # batch mismatch
        (z, t, torch.randn(4, 6, cfg.text_dim + 1), None),            # width mismatch
        (z, t, text[0], None),                                        # missing batch axis
        (z, t, text, mask[:, :5]),                                    # mask shape
        (z, t, text, mask.float()),                                   # mask dtype
        (z[:, :, :4], t, text, mask),                                 # latent shape
        (z, t[:3], text, mask),                                       # time shape
    ]
    raised = 0
    for args in bad_inputs:
        try:
            m.velocity(*args)
        except (ValueError, RuntimeError):
            raised += 1
    _record("T10: loss / wrong inputs raising", (round(float(loss.detach()), 4), f"{raised}/{len(bad_inputs)}"))
    assert raised == len(bad_inputs)


# =============================================================================
# T11  no comparative pick in the model
# =============================================================================

PICK_NAMES = {"softmax", "log_softmax", "gumbel_softmax", "argmax", "argmin", "topk", "argsort", "kthvalue"}


def _pick_hits(folder):
    hits = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".py") or fn.startswith("test_"):
            continue
        with open(os.path.join(folder, fn), "rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type == tokenize.NAME and tok.string in PICK_NAMES:
                    hits.append(f"{fn}:{tok.start[0]}:{tok.string}")
    return hits


def test_t11_no_comparative_pick():
    # names only (comments and strings ignored); attention is scaled_dot_product_attention, not listed
    assert _pick_hits(HERE) == []
    orig = _pick_hits(ORIGINAL_DIR)
    _record("T11 control: hits in the original folder", len(orig))
    assert orig


# =============================================================================
# T12  samples do not mix across the batch
# =============================================================================

def test_t12_batch_independence():
    for over in ({}, {"signed_dispatch": True}):
        m = _perturb(_model(**over)).eval()
        z, t, text, mask = _inputs(m.config)
        z2, t2, text2, mask2 = z.clone(), t.clone(), text.clone(), mask.clone()
        z2[1:] = torch.randn_like(z2[1:])
        t2[1:] = 0.77
        text2[1:] = torch.randn_like(text2[1:])
        mask2[2, -3:] = False
        with torch.no_grad():
            a = m.velocity(z, t, text, mask)
            b = m.velocity(z2, t2, text2, mask2)
        d0 = float((a[0] - b[0]).abs().max())
        d1 = float((a[1] - b[1]).abs().max())
        _record(f"T12 new {over}: sample-0 change / sample-1 change", (f"{d0:.2e}", f"{d1:.3g}"))
        assert d0 <= 1e-6 and d1 > 1e-3

    # control: the original head router's attention fusion mixes samples
    hr = _original("head_router")
    torch.manual_seed(0)
    head = hr.AgathaHeadRouter("h", embed_dim=32, fingerprint_dim=8, fusion_type="attention")
    head.attach_encoder("qwen", hr.MockEncoder(16, 24), embed_dim=24)
    head.attach_encoder("dino", hr.MockEncoder(16, 40), embed_dim=40)
    x = {k: torch.randn(4, 16) for k in ("qwen", "dino")}
    with torch.no_grad():
        f1 = head(x).fused
        x2 = {k: v.clone() for k, v in x.items()}
        x2["qwen"][1:] += 5.0
        f2 = head(x2).fused
    leak = float((f1[0] - f2[0]).abs().max())
    _record("T12 control: original attention fusion, sample-0 change", round(leak, 4))
    assert leak > 1e-3


# =============================================================================
# T13  eval is deterministic
# =============================================================================

def test_t13_deterministic_eval():
    m = _perturb(_model()).eval()
    z, t, text, mask = _inputs(m.config)
    with torch.no_grad():
        torch.manual_seed(1)
        a = m.velocity(z, t, text, mask)
        torch.manual_seed(2)
        b = m.velocity(z, t, text, mask)
    assert torch.equal(a, b)

    bo = _original("beatrix_oscillator")
    torch.manual_seed(0)
    osc = bo.BeatrixOscillator(manifold_dim=64, tower_dim=16, num_tower_pairs=4, num_theta_probes=4).eval()
    g = torch.Generator().manual_seed(3)
    x0, xr = torch.randn(3, 64, generator=g), torch.randn(3, 64, generator=g)
    with torch.no_grad():
        torch.manual_seed(1)
        oa = osc(x0, xr, [], num_steps=10)
        torch.manual_seed(2)
        ob = osc(x0, xr, [], num_steps=10)
    od = float((oa - ob).abs().max())
    _record("T13 control: original oscillator, two seeds, eval mode, max diff", round(od, 4))
    assert od > 1e-3


# =============================================================================
# extra: zero at initialization, and the closed expression of the velocity
# =============================================================================

def test_initial_output_is_exactly_zero():
    for cfg in (smoke_config(), smoke_config(signed_dispatch=True)):
        torch.manual_seed(0)
        m = AgathaFlow(cfg).eval()
        z, t, text, mask = _inputs(cfg)
        with torch.no_grad():
            v = m.velocity(z, t, text, mask)
        assert bool((v == 0).all())


def test_velocity_closed_expression():
    """v_hat = W_v G_N [omega^2 D_ref(h) + kappa sum_k w_k Y_k] + b_v, and the stepped path agrees."""
    m = _perturb(_model()).eval()
    with torch.no_grad():
        m.velocity_head.weight.add_(0.05 * torch.randn_like(m.velocity_head.weight))
    z, t, text, mask = _inputs(m.config)
    with torch.no_grad():
        v = m.velocity(z, t, text, mask)
        x, h, c, e = m.encode(z, t, text, mask)
        d_ref = m.anchor_head(h)
        forces = m.tower_forces(h, c, mask, e)
        _, beta, omega, kappa, w = m.schedule(t)
        _, g_n = osc_mod.affine_coefficients(beta, omega, m.config.num_osc_steps)
        u = torch.einsum("bk,kbnp->bnp", w, forces)
        inner = g_n[:, None, None] * (omega[:, None, None] ** 2 * d_ref + kappa[:, None, None] * u)
        closed = m.unpatchify(m.velocity_head(inner))
        m.oscillator.form = "stepped"
        v_st = m.velocity(z, t, text, mask)
        m.oscillator.form = "exact"
    scale = float(v.abs().max())
    d1 = float((v - closed).abs().max()) / scale
    d2 = float((v - v_st).abs().max()) / scale
    _record("closed expression: relative diff exact-vs-closed / exact-vs-stepped", (f"{d1:.2e}", f"{d2:.2e}"))
    assert d1 < 1e-5 and d2 < 1e-5


# =============================================================================
# extra: the position-code adapter, condition dropout and guidance, autocast
# =============================================================================

def test_position_code_adapter_matches_component():
    """The tower applies each reused code exactly as the component's own rotate() does."""
    g = torch.Generator().manual_seed(4)
    worst, moved = 0.0, float("inf")
    for n in (16, 256):
        q = torch.randn(2, 1, n, 64, generator=g)
        k = torch.randn(2, 1, n, 64, generator=g)
        for kind in ("cantor", "beatrix", "simplex", "helix", "plain"):
            pc = PositionCode(kind, 64, n)
            with torch.no_grad():
                qa, ka = pc.rotary()(q, k)
                if kind in ("simplex", "helix"):
                    qo, ko = pc.code.rotate(q, k, seq_len=n, mode="augment")
                else:
                    qo, ko = pc.code.rotate(q, k, seq_len=n)
            worst = max(worst, float((qa - qo).abs().max()), float((ka - ko).abs().max()))
            # control: the identity rotary is not the component's rotation
            qi, _ = identity_rotary(q, k)
            moved = min(moved, float((qi - qo).abs().max()))
    _record("adapter vs component rotate(): max diff / identity control min diff", (f"{worst:.2e}", round(moved, 3)))
    assert worst <= 1e-5 and moved > 1e-2


def test_trunk_rotary_is_two_dimensional():
    """The trunk code is a pure rotation whose first half reads the row and second half the column."""
    gh, gw, hd = 4, 4, 64
    rot = AxialRotary2D(hd, gh, gw).rotary()
    g = torch.Generator().manual_seed(5)
    one = torch.randn(1, 1, 1, hd, generator=g)
    q = one.expand(1, 1, gh * gw, hd).contiguous()      # the same vector at every grid position
    with torch.no_grad():
        r, _ = rot(q, q)
    r = r[0, 0]                                          # [N, hd]
    half = hd // 2
    assert torch.allclose(r.norm(dim=-1), q[0, 0].norm(dim=-1), atol=1e-5)          # norm kept
    same_row = float((r[0, :half] - r[1, :half]).abs().max())        # tokens (0,0) and (0,1)
    diff_col = float((r[0, half:] - r[1, half:]).abs().max())
    same_col = float((r[0, half:] - r[gw, half:]).abs().max())       # tokens (0,0) and (1,0)
    diff_row = float((r[0, :half] - r[gw, :half]).abs().max())
    _record("trunk 2-D code: same row / other column / same column / other row",
            (same_row, round(diff_col, 3), same_col, round(diff_row, 3)))
    assert same_row == 0.0 and same_col == 0.0 and diff_col > 1e-2 and diff_row > 1e-2
    # control: a raster 1-D code does not keep the row half equal along a row
    cos, sin = RoPE("r", head_dim=hd, max_seq_len=gh * gw).embed(gh * gw)
    with torch.no_grad():
        r1, _ = table_rotary(cos, sin)(q, q)
    assert float((r1[0, 0, 0, :half] - r1[0, 0, 1, :half]).abs().max()) > 1e-2


def test_condition_dropout_and_guidance():
    m = _perturb(_model()).eval()
    cfg = m.config
    with torch.no_grad():
        m.set_null_text(torch.randn(cfg.null_text_len, cfg.text_dim, generator=torch.Generator().manual_seed(9)))
    z, t, text, mask = _inputs(cfg)
    drop = torch.tensor([True, False, True, False])
    with torch.no_grad():
        text_d, mask_d = m.drop_condition(text, mask, drop)
        e0, m0 = m.null_condition(4, text.shape[1])
        assert torch.equal(text_d[1], text[1]) and torch.equal(mask_d[3], mask[3])
        assert torch.equal(text_d[0], e0[0]) and torch.equal(mask_d[2], m0[2])
        v_text = m.velocity(z, t, text, mask)
        v_drop = m.velocity(z, t, text_d, mask_d)
        v_null = m.velocity(z, t, *m.null_condition(4, 1))       # the null sequence without padding
        kept = float((v_drop[[1, 3]] - v_text[[1, 3]]).abs().max())
        dropped = float((v_drop[[0, 2]] - v_null[[0, 2]]).abs().max())
        # control: the caption matters, so a drop that did nothing would be seen
        matters = float((v_text[[0, 2]] - v_null[[0, 2]]).abs().max())
        # the loss applies the same replacement
        g1, g2 = torch.Generator().manual_seed(2), torch.Generator().manual_seed(2)
        l_drop = m.loss(z, text, mask, t=t, drop=torch.ones(4, dtype=torch.bool), generator=g1)
        l_null = m.loss(z, e0, m0, t=t, drop=torch.zeros(4, dtype=torch.bool), generator=g2)
        # guidance: scale 1 is the conditional velocity, scale 0 the null velocity
        tt = 0.3
        v_c = m.velocity(z, tt, text, mask)
        v_0 = m.velocity(z, tt, *m.null_condition(4, 1))
        g_one = float((m._guided(z, tt, text, mask, 1.0) - v_c).abs().max())
        g_zero = float((m._guided(z, tt, text, mask, 0.0) - v_0).abs().max())
        g_two = float((m._guided(z, tt, text, mask, 2.0) - v_c).abs().max())
    _record("dropout: kept rows / dropped rows vs null / caption effect; loss diff; guidance 1 / 0 / 2",
            (f"{kept:.1e}", f"{dropped:.1e}", round(matters, 4), f"{abs(float(l_drop) - float(l_null)):.1e}",
             f"{g_one:.1e}", f"{g_zero:.1e}", round(g_two, 4)))
    assert kept <= 1e-6 and dropped <= 1e-5 and matters > 1e-3
    assert abs(float(l_drop) - float(l_null)) <= 1e-6
    assert g_one <= 1e-5 and g_zero <= 1e-5 and g_two > 1e-3


def test_float_section_under_autocast():
    """Under bfloat16 autocast the final combination, the oscillator and the head stay float32."""
    m = _perturb(_model()).eval()
    z, t, text, mask = _inputs(m.config)
    with torch.no_grad():
        ref = m.velocity(z, t, text, mask)
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                v = m.velocity(z, t, text, mask)
                x, h, c, e = m.encode(z, t, text, mask)
                low = m.towers[0](h, c, mask, e).dtype
        except RuntimeError as err:
            _skip(f"bfloat16 autocast cannot run on this CPU build: {str(err)[:80]}")
    rel = float((v.float() - ref).abs().max() / ref.abs().max())
    _record("autocast: output dtype / tower dtype inside autocast / relative diff to float32",
            (str(v.dtype), str(low), f"{rel:.2e}"))
    assert v.dtype == torch.float32 and torch.isfinite(v).all()
    assert low == torch.bfloat16, "control: autocast was not active, the check proves nothing"
    assert rel < 0.1


# =============================================================================
# runner without pytest
# =============================================================================

def _main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed, failed, skipped = 0, [], []
    for name, fn in tests:
        print(name)
        try:
            fn()
            passed += 1
            print("  PASSED")
        except unittest.SkipTest as e:
            skipped.append((name, str(e)))
            print(f"  SKIPPED {e}")
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "Skipped":
                skipped.append((name, str(e)))
                print(f"  SKIPPED {e}")
                continue
            import traceback
            traceback.print_exc()
            failed.append(name)
            print("  FAILED")
    print(f"\n{passed} passed, {len(failed)} failed, {len(skipped)} skipped")
    for n, r in skipped:
        print(f"  skipped {n}: {r}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
