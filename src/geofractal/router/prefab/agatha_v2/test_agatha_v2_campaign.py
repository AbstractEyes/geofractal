"""
Tests for the training-campaign pieces of geofractal.router.prefab.agatha_v2 (CPU, smoke
size, no training for accuracy): the plain control, EMA and exact resume, the
memory-mapped cache and the step-order sampler, the evaluation and push hooks, the sphere
check, and the metric plumbing with fake feature extractors and embedders.

Each test carries a control that must be able to fail.

Run with pytest:
    python -B -m pytest -q -p no:cacheprovider .../agatha_v2/test_agatha_v2_campaign.py
or without pytest:
    python -B .../agatha_v2/test_agatha_v2_campaign.py
"""

from __future__ import annotations

import gc
import io
import json
import math
import os
import pickle
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from geofractal.router.prefab.agatha_v2 import sphere_check
from geofractal.router.prefab.agatha_v2.config import AgathaConfig, plain_counterpart, smoke_config
from geofractal.router.prefab.agatha_v2.data import (
    MemmapLatentTextDataset,
    StepOrderSampler,
    build_memmap_cache,
    collate_latent_text,
    fixed_caption_pairs,
    gather_pairs,
    mix_seed,
)
from geofractal.router.prefab.agatha_v2.hooks import (
    CheckpointPushHook,
    ClipHook,
    FidKidHook,
    HeldOutLossHook,
    Hook,
    HookRunner,
    MilestonePushHook,
    SampleSheetHook,
    SphereCheckHook,
    derangement,
)
from geofractal.router.prefab.agatha_v2.metrics import ClipJudge, FidKid, frechet_distance, gaussian_stats
from geofractal.router.prefab.agatha_v2.model import (
    AgathaFlow,
    PlainFlow,
    build_model,
    count_trained_parameters,
    match_plain_trunk_depth,
)
from geofractal.router.prefab.agatha_v2.towers import ZeroHead
from geofractal.router.prefab.agatha_v2.trainer import (
    EMA,
    HookContext,
    TrainConfig,
    build_optimizer,
    capture_rng_state,
    fit,
    load_checkpoint,
    load_training_state,
    restore_rng_state,
    save_checkpoint,
    save_training_state,
    step_generator,
    train_step,
)

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

REPORT = {}


def _record(key, value):
    REPORT[key] = value
    print(f"  [{key}] {value}")


def _skip(reason):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        raise unittest.SkipTest(reason)


def _quiet(_line):
    return None


# =============================================================================
# helpers
# =============================================================================

TEXT_DIM = 32           # smoke_config().text_dim
WORDS = ("a", "red", "cat", "on", "the", "table", "two", "dogs", "in", "snow", "blue", "car", "near", "tree")


@torch.no_grad()
def fake_encode_images(images):
    """[n, 3, 16, 16] pictures -> [n, 4, 8, 8] latents with different channel means and scales."""
    x = torch.stack([torch.as_tensor(im, dtype=torch.float32) for im in images])
    lat = F.avg_pool2d(x, 2)
    lat = torch.cat([lat, lat.mean(dim=1, keepdim=True)], dim=1)
    scale = torch.tensor([1.0, 2.0, 3.0, 0.5]).view(1, 4, 1, 1)
    shift = torch.tensor([0.5, -1.0, 2.0, 0.0]).view(1, 4, 1, 1)
    return lat * scale + shift


def _word_vector(word):
    g = torch.Generator().manual_seed(mix_seed(*[ord(ch) for ch in word]))
    return torch.randn(TEXT_DIM, generator=g)


@torch.no_grad()
def fake_encode_texts(captions):
    """Words then an end token, like a T5 encoder; '' gives the end token alone."""
    toks = [[_word_vector(w) for w in c.split()] + [_word_vector("</s>")] for c in captions]
    length = max(len(t) for t in toks)
    states = torch.zeros(len(captions), length, TEXT_DIM)
    mask = torch.zeros(len(captions), length, dtype=torch.bool)
    for i, t in enumerate(toks):
        states[i, :len(t)] = torch.stack(t)
        mask[i, :len(t)] = True
    return states, mask


def toy_records(n=10, seed=0):
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        caps = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 7))) for _ in range(rng.randint(1, 4))]
        out.append({"id": 1000 + i, "file_name": f"img_{i:04d}.jpg", "license": i % 3,
                    "image": torch.rand(3, 16, 16, generator=g), "captions": caps,
                    "caption_ids": [50000 + 10 * i + k for k in range(len(caps))]})
    return out


class _CountingImages:
    def __init__(self):
        self.sizes = []

    def __call__(self, images):
        self.sizes.append(len(images))
        return fake_encode_images(images)


def build_toy_cache(folder, n=10, seed=0, batch_size=3):
    enc = _CountingImages()
    recs = toy_records(n, seed)
    meta = build_memmap_cache(iter(recs), enc, fake_encode_texts, folder, batch_size=batch_size,
                              text_encoder="fake words", text_encoder_revision="r1", vae="fake pool",
                              vae_revision="r2", preprocessing="none (synthetic)")
    return meta, recs, enc


def _close(ds):
    ds.close()
    gc.collect()


def _tmpdir():
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


@torch.no_grad()
def _perturb(model, seed=7, std=0.05):
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad and bool((p == 0).all()):
            p.copy_(torch.randn(p.shape, generator=g) * std)
    return model


def _inputs(cfg, b=4, seed=1, length=6):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn((b,) + cfg.latent_shape, generator=g)
    t = torch.tensor([0.1, 0.45, 0.9, 0.6, 0.3, 0.8][:b])
    text = torch.randn(b, length, cfg.text_dim, generator=g)
    mask = torch.ones(b, length, dtype=torch.bool)
    mask[1, -2:] = False
    return z, t, text, mask


def _no_grad_names(model):
    return [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]


def fake_decoder(z):
    """[n, 4, 8, 8] latents -> [n, 3, 16, 16] pictures in (0, 1)."""
    return torch.sigmoid(F.interpolate(z[:, :3].float(), scale_factor=2, mode="nearest"))


def _ctx(model, ema=None, step=1, out_dir=".", run="toy", optimizer=None, data_position=None, cfg=None):
    return HookContext(step=step, end_step=step, model=model, ema=ema, optimizer=optimizer, device="cpu",
                       use_amp=False, run_name=run, out_dir=out_dir, log=_quiet,
                       data_position=data_position or {"next_step": step}, train_config=cfg)


# =============================================================================
# P  the plain control
# =============================================================================

def test_p1_plain_structure_same_front_zero_start():
    cfg_a = smoke_config()
    cfg_p = smoke_config(arch="plain", trunk_depth=3)
    torch.manual_seed(0)
    a = build_model(cfg_a)
    torch.manual_seed(0)
    p = build_model(smoke_config(arch="plain"))
    assert type(a) is AgathaFlow and type(p) is PlainFlow
    assert len(p.towers) == 0 and not hasattr(p, "bands") and not hasattr(p, "oscillator")
    assert not hasattr(p, "velocity_head") and not hasattr(p, "anchor_head")
    assert isinstance(p.head, ZeroHead) and bool((p.head.proj.weight == 0).all()) and bool((p.head.proj.bias == 0).all())
    # the same front, built by the same code in the same order: equal tensors at the same seed
    front = [k for k in a.state_dict() if k.split(".")[0] in ("patch_embed", "time_embed", "text_proj", "trunk", "null_text")]
    same = all(torch.equal(a.state_dict()[k], p.state_dict()[k]) for k in front)
    assert front and same, "the plain control's front differs from Agatha's"
    # control: Agatha has towers and an oscillator
    assert len(a.towers) == 6 and hasattr(a, "oscillator")
    # velocity is one tensor, zero at initialization; the deeper plain trunk too
    p3 = build_model(cfg_p)
    z, t, text, mask = _inputs(cfg_p)
    with torch.no_grad():
        v = p3.velocity(z, t, text, mask)
    assert torch.is_tensor(v) and v.shape == z.shape and bool((v == 0).all())
    assert len(p3.trunk) == 3
    _record("P1: front tensors equal to Agatha's / plain trunk depth / initial |v| max",
            (len(front), len(p3.trunk), float(v.abs().max())))


def test_p2_plain_every_tensor_gets_a_gradient():
    torch.manual_seed(0)
    m = _perturb(build_model(smoke_config(arch="plain", trunk_depth=2)))
    z, t, text, mask = _inputs(m.config)
    m.zero_grad(set_to_none=True)
    m.loss(z, text, mask, t=t, drop=torch.zeros(4, dtype=torch.bool), generator=torch.Generator().manual_seed(3)).backward()
    dead = _no_grad_names(m)
    n_trained = sum(1 for p in m.parameters() if p.requires_grad)
    assert not dead, f"tensors without gradient: {dead}"
    # at initialization (zero head) only the head gets a gradient: the check sees missing gradients
    torch.manual_seed(0)
    m0 = build_model(smoke_config(arch="plain", trunk_depth=2))
    m0.zero_grad(set_to_none=True)
    m0.loss(z, text, mask, t=t, drop=torch.zeros(4, dtype=torch.bool), generator=torch.Generator().manual_seed(3)).backward()
    dead0 = _no_grad_names(m0)
    assert dead0 and not any(n.startswith("head.") for n in dead0)
    # control: a detached trunk is reported gradient-free
    m2 = _perturb(build_model(smoke_config(arch="plain", trunk_depth=2)))
    blk = m2.trunk[1]
    orig = blk.forward
    blk.forward = lambda *a_, **k_: orig(*a_, **k_).detach()
    m2.zero_grad(set_to_none=True)
    m2.loss(z, text, mask, t=t, drop=torch.zeros(4, dtype=torch.bool), generator=torch.Generator().manual_seed(3)).backward()
    dead2 = set(_no_grad_names(m2))
    assert any(n.startswith("trunk.0.") for n in dead2) and any(n.startswith("patch_embed") for n in dead2)
    _record("P2: trained tensors / gradient-free after a perturbed step / at initialization / detached-trunk control",
            (n_trained, len(dead), len(dead0), len(dead2)))


def _toy_batch(cfg, n=8, seed=11):
    g = torch.Generator().manual_seed(seed)
    z1 = torch.randn((n,) + cfg.latent_shape, generator=g) * 0.5 + 0.3
    text = torch.randn(n, 5, cfg.text_dim, generator=g)
    mask = torch.ones(n, 5, dtype=torch.bool)
    return z1, text, mask


@torch.no_grad()
def _fixed_loss(model, z1, text, mask):
    was = model.training
    model.eval()
    g = torch.Generator().manual_seed(99)
    t = torch.rand(z1.shape[0], generator=g)
    z0 = torch.randn(z1.shape, generator=g)
    out = float(model.loss(z1, text, mask, z_0=z0, t=t, drop=torch.zeros(z1.shape[0], dtype=torch.bool)))
    model.train(was)
    return out


def _descent(cfg, lr, steps=60):
    torch.manual_seed(0)
    m = build_model(cfg)
    tc = TrainConfig(lr=lr, seed=0)
    opt = build_optimizer(m, tc)
    z1, text, mask = _toy_batch(cfg)
    before = _fixed_loss(m, z1, text, mask)
    m.train()
    for s in range(steps):
        train_step(m, opt, None, z1, text, mask, s, tc)
    return before, _fixed_loss(m, z1, text, mask)


def test_p3_plain_descent_on_toy():
    cfg = smoke_config(arch="plain", trunk_depth=2)
    b, a = _descent(cfg, 3e-3)
    b0, a0 = _descent(cfg, 0.0)
    _record("P3: fixed-draw loss before / after 60 steps (lr 3e-3) / lr 0 control", (round(b, 4), round(a, 4), round(a0, 4)))
    assert a < 0.8 * b
    assert a0 == b0


def test_p4_plain_reload_equal_and_eval_deterministic():
    torch.manual_seed(0)
    m = _perturb(build_model(smoke_config(arch="plain", trunk_depth=2))).eval()
    with torch.no_grad():
        m.set_null_text(torch.randn(1, TEXT_DIM))
    buf = io.BytesIO()
    save_checkpoint(buf, m, build_optimizer(m), step=5)
    buf.seek(0)
    m2, opt2, step = load_checkpoint(buf)
    m2.eval()
    z, t, text, mask = _inputs(m.config)
    with torch.no_grad():
        torch.manual_seed(1)
        va = m.velocity(z, t, text, mask)
        torch.manual_seed(2)
        va2 = m.velocity(z, t, text, mask)
        vb = m2.velocity(z, t, text, mask)
        vg = m._guided(z, 0.3, text, mask, 3.0)
    assert type(m2) is PlainFlow and step == 5 and isinstance(opt2, torch.optim.Adam)
    assert torch.equal(va, vb) and torch.equal(va, va2) and torch.isfinite(vg).all()
    torch.manual_seed(1)
    m3 = _perturb(build_model(smoke_config(arch="plain", trunk_depth=2)), seed=8).eval()
    with torch.no_grad():
        vc = m3.velocity(z, t, text, mask)
    diff = float((va - vc).abs().max())
    _record("P4: reload max diff / two torch seeds in eval / other-weights control", (float((va - vb).abs().max()),
            float((va - va2).abs().max()), round(diff, 4)))
    assert diff > 1e-3


def test_p5_matched_trunk_depth():
    smoke = smoke_config()
    d, n_plain, n_ref = match_plain_trunk_depth(smoke)
    built = count_trained_parameters(plain_counterpart(smoke, d))
    near = {dd: abs(count_trained_parameters(plain_counterpart(smoke, dd)) - n_ref) for dd in (d - 1, d + 1) if dd >= 1}
    assert built == n_plain and all(abs(n_plain - n_ref) <= v for v in near.values())
    cfg = AgathaConfig()
    dd, pp, aa = match_plain_trunk_depth(cfg)
    rel = pp / aa - 1.0
    _record("P5: default config: Agatha parameters / matched plain depth / plain parameters / relative difference",
            (aa, dd, pp, round(rel, 5)))
    assert abs(rel) <= 0.03, "no plain depth within 3% of the default Agatha"
    pc = plain_counterpart(cfg, dd)
    assert pc.arch == "plain" and pc.geometry_towers == () and pc.text_dim == cfg.text_dim and pc.dim == cfg.dim
    # control: a depth one higher misses the 3% band
    assert abs(count_trained_parameters(plain_counterpart(cfg, dd + 1)) / aa - 1.0) > 0.03


# =============================================================================
# E  EMA, random states, exact resume, the loop and the hooks
# =============================================================================

def test_e1_ema_update_rule():
    torch.manual_seed(0)
    m = _perturb(build_model(smoke_config()))
    ema = EMA(m, decay=0.9)
    assert all(not p.requires_grad and p.dtype == torch.float32 for p in ema.model.parameters())
    assert all(torch.equal(e, p) for e, p in zip(ema.model.parameters(), m.parameters()))
    # buffers keep their dtype (the position codes' float64 tables are not cast down)
    assert any(b.dtype == torch.float64 for b in m.buffers())
    assert all(eb.dtype == b.dtype for eb, b in zip(ema.model.buffers(), m.buffers()))
    old = [e.detach().clone() for e in ema.model.parameters()]
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
        m.set_null_text(torch.randn(1, TEXT_DIM))
    ema.update(m)
    exact = all(torch.equal(e, o.lerp(p.detach(), 0.1)) for e, o, p in zip(ema.model.parameters(), old, m.parameters()))
    close = all(torch.allclose(e, 0.9 * o + 0.1 * p.detach(), atol=1e-6) for e, o, p in zip(ema.model.parameters(), old, m.parameters()))
    wrong = all(torch.allclose(e, 0.1 * o + 0.9 * p.detach(), atol=1e-6) for e, o, p in zip(ema.model.parameters(), old, m.parameters()))
    assert exact and close and not wrong and ema.num_updates == 1
    assert torch.equal(ema.model.null_text, m.null_text)
    st = ema.state_dict()
    ema2 = EMA(build_model(smoke_config()), 0.5)
    ema2.load_state_dict(st)
    assert ema2.decay == 0.9 and ema2.num_updates == 1
    assert all(torch.equal(a, b) for a, b in zip(ema.model.state_dict().values(), ema2.model.state_dict().values()))
    _record("EMA: exact lerp / 0.9 old + 0.1 new / swapped-weights control", (exact, close, wrong))


def test_e2_rng_states_and_step_generator():
    st = capture_rng_state()
    a = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    restore_rng_state(st)
    b = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    c = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert a == b and a != c
    buf = io.BytesIO()
    torch.save({"rng": st}, buf)
    buf.seek(0)
    st2 = torch.load(buf, weights_only=True)["rng"]
    restore_rng_state(st2)
    assert (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist()) == a
    x1 = torch.randn(5, generator=step_generator(3, 17))
    x2 = torch.randn(5, generator=step_generator(3, 17))
    x3 = torch.randn(5, generator=step_generator(3, 18))
    x4 = torch.randn(5, generator=step_generator(4, 17))
    assert torch.equal(x1, x2) and not torch.equal(x1, x3) and not torch.equal(x1, x4)


def _loader(ds, sampler):
    return torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate_latent_text, num_workers=0)


def _resume_case(folder, arch, n_steps=6, k=3, batch=4, seed=5, sabotage=None):
    """Unbroken run of n_steps against k steps + save + rebuild from the file + n_steps - k steps."""
    ds = MemmapLatentTextDataset(folder)
    cfg = smoke_config(arch=arch, trunk_depth=1 if arch == "agatha" else 2)
    tc = TrainConfig(lr=1e-3, batch_size=batch, seed=seed, log_every=2, ema_decay=0.9)
    null = ds.null_text()

    def fresh():
        torch.manual_seed(seed)
        m = build_model(cfg)
        m.set_null_text(null)
        return m, build_optimizer(m, tc)

    # unbroken
    m_a, opt_a = fresh()
    ema_a = EMA(m_a, tc.ema_decay)
    smp = StepOrderSampler(ds.image_captions, batch, seed, 0, n_steps)
    fit(m_a, opt_a, _loader(ds, smp), tc, start_step=0, end_step=n_steps, ema=ema_a, device="cpu", log=_quiet)
    # first part, saved
    m_b, opt_b = fresh()
    ema_b = EMA(m_b, tc.ema_decay)
    smp_b = StepOrderSampler(ds.image_captions, batch, seed, 0, k)
    fit(m_b, opt_b, _loader(ds, smp_b), tc, start_step=0, end_step=k, ema=ema_b, device="cpu", log=_quiet)
    buf = io.BytesIO()
    save_training_state(buf, m_b, opt_b, ema_b, k, smp_b.position(k), tc)
    del m_b, opt_b, ema_b
    # second part, from the file alone
    buf.seek(0)
    st = load_training_state(buf, device="cpu")
    pos = st.data_position["next_step"]
    tc2 = st.train_config
    opt2 = st.optimizer
    if sabotage == "data_from_zero":
        pos = 0
    if sabotage == "fresh_optimizer":
        opt2 = build_optimizer(st.model, tc2)
    if sabotage == "other_noise_seed":
        tc2 = TrainConfig.from_dict(dict(tc2.to_dict(), seed=seed + 1))
    smp2 = StepOrderSampler(ds.image_captions, batch, st.data_position["seed"], pos, pos + (n_steps - st.step))
    fit(st.model, opt2, _loader(ds, smp2), tc2, start_step=st.step, end_step=n_steps, ema=st.ema, device="cpu",
        log=_quiet)
    same_p = all(torch.equal(x, y) for x, y in zip(m_a.parameters(), st.model.parameters()))
    same_p = same_p and all(x.dtype == y.dtype for x, y in zip(m_a.buffers(), st.model.buffers()))
    same_e = all(torch.equal(x, y) for x, y in zip(ema_a.model.state_dict().values(), st.ema.model.state_dict().values()))
    worst = max(float((x.detach() - y.detach()).abs().max()) for x, y in zip(m_a.parameters(), st.model.parameters()))
    _close(ds)
    return same_p, same_e, worst, st.ema.num_updates


def test_e3_exact_resume():
    with _tmpdir() as d:
        build_toy_cache(d, n=10)
        rows = {}
        for arch in ("agatha", "plain"):
            sp, se, worst, nu = _resume_case(d, arch)
            rows[arch] = (sp, se, worst, nu)
            assert sp and se and worst == 0.0 and nu == 6, (arch, sp, se, worst, nu)
        controls = {s: _resume_case(d, "agatha", sabotage=s)[2] for s in ("data_from_zero", "fresh_optimizer",
                                                                         "other_noise_seed")}
    _record("E3: (params equal, EMA equal, max diff, EMA updates) per arm", rows)
    _record("E3 controls: max parameter diff when the resume skips a piece", {k: f"{v:.2e}" for k, v in controls.items()})
    assert all(v > 0 for v in controls.values())


class _Boom(Hook):
    name = "boom"

    def __call__(self, ctx):
        if ctx.step == 2:
            raise RuntimeError("deliberate failure")
        return {"ok": ctx.step}


class _Count(Hook):
    name = "count"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.steps = []

    def __call__(self, ctx):
        self.steps.append((ctx.step, ctx.final))
        return {"step": ctx.step}


def test_e4_fit_progress_lines_and_hooks_never_raise():
    with _tmpdir() as d:
        build_toy_cache(os.path.join(d, "cache"), n=10)
        ds = MemmapLatentTextDataset(os.path.join(d, "cache"))
        tc = TrainConfig(lr=1e-3, batch_size=4, seed=0, log_every=2)
        torch.manual_seed(0)
        m = build_model(smoke_config())
        lines = []
        count = _Count(every=1, at_end=True)
        runner = HookRunner([_Boom(every=1), count], log=lines.append, results_path=os.path.join(d, "evals.jsonl"))
        out = fit(m, build_optimizer(m, tc), _loader(ds, StepOrderSampler(ds.image_captions, 4, 0, 0, 5)), tc,
                  start_step=0, end_step=5, ema=EMA(m, 0.99), device="cpu", hooks=runner, log=lines.append,
                  run_name="toy")
        with open(os.path.join(d, "evals.jsonl"), encoding="utf-8") as fh:
            recs = [json.loads(x) for x in fh]
        _close(ds)
    progress = [x for x in lines if " spent " in x and " left " in x]
    assert out["step"] == 5 and not out["stopped"]
    assert len(runner.errors) == 1 and runner.errors[0]["hook"] == "boom" and runner.errors[0]["step"] == 2
    assert [s for s, f in count.steps] == [1, 2, 3, 4, 5]          # at_end does not repeat step 5
    assert len(progress) == 3 and progress[-1].startswith("[toy] step 5/5")
    assert len(recs) == 10 and sum(1 for r in recs if not r["ok"]) == 1
    try:                                                           # control: the hook alone raises
        _Boom(every=1)(_ctx(m, step=2))
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    _record("E4: progress lines / hook errors logged / count hook steps", (len(progress), len(runner.errors), count.steps))


def test_e4b_end_of_run_hooks_fire_off_the_interval_grid():
    """A run that ends between intervals still gets its end-of-run hooks; a hook without at_end does not."""
    with _tmpdir() as d:
        build_toy_cache(os.path.join(d, "cache"), n=10)
        ds = MemmapLatentTextDataset(os.path.join(d, "cache"))
        tc = TrainConfig(lr=1e-3, batch_size=4, seed=0, log_every=10)
        torch.manual_seed(0)
        m = build_model(smoke_config())
        with_end = _Count(every=2, at_end=True)
        without_end = _Count(every=2, at_end=False, name="count_no_end")    # control: must not fire at step 5
        runner = HookRunner([with_end, without_end], log=lambda s: None)
        fit(m, build_optimizer(m, tc), _loader(ds, StepOrderSampler(ds.image_captions, 4, 0, 0, 5)), tc,
            start_step=0, end_step=5, ema=EMA(m, 0.99), device="cpu", hooks=runner, log=lambda s: None, run_name="toy")
        _close(ds)
    assert with_end.steps == [(2, False), (4, False), (5, True)], with_end.steps
    assert without_end.steps == [(2, False), (4, False)], without_end.steps
    _record("E4b: end-of-run hook steps / control steps", (with_end.steps, without_end.steps))


def _eval_tensors(cfg, n=8, seed=21):
    g = torch.Generator().manual_seed(seed)
    lat = torch.randn((n,) + cfg.latent_shape, generator=g)
    text = torch.randn(n, 6, cfg.text_dim, generator=g)
    mask = torch.ones(n, 6, dtype=torch.bool)
    mask[::2, -2:] = False
    return lat, text, mask


def test_e5_heldout_loss_and_caption_gap():
    torch.manual_seed(0)
    m = _perturb(build_model(smoke_config()))
    with torch.no_grad():
        m.set_null_text(torch.randn(1, TEXT_DIM, generator=torch.Generator().manual_seed(4)))
    ema = EMA(m, 0.9)
    lat, text, mask = _eval_tensors(m.config)
    hook = HeldOutLossHook(lat, text, mask, every=1)
    r1 = hook(_ctx(m, ema))
    r2 = hook(_ctx(m, ema))
    assert r1 == r2, "the held-out read is not deterministic"
    raw = r1["raw"]
    assert set(r1) == {"items", "raw", "ema"} and len(raw["bands"]) == 3 and math.isfinite(raw["loss"])
    assert abs(raw["gap_shuffled"]) > 1e-6 and abs(raw["gap_null"]) > 1e-6
    d = derangement(8, 3)
    assert bool((d != torch.arange(8)).all()) and sorted(d.tolist()) == list(range(8))
    # control: an identity "derangement" gives a gap of exactly zero
    hook.order = torch.arange(8)
    r3 = hook(_ctx(m, ema))
    assert r3["raw"]["gap_shuffled"] == 0.0
    # the plain control runs through the same hook (bands from its config)
    torch.manual_seed(0)
    p = _perturb(build_model(smoke_config(arch="plain", trunk_depth=2)))
    rp = HeldOutLossHook(lat, text, mask, every=1, weights=("raw",))(_ctx(p))
    assert math.isfinite(rp["raw"]["loss"]) and "ema" not in rp
    _record("E5: raw loss / bands / gap shuffled / gap null / identity-order control",
            (round(raw["loss"], 4), [round(x, 4) for x in raw["bands"]], f"{raw['gap_shuffled']:.3e}",
             f"{raw['gap_null']:.3e}", r3["raw"]["gap_shuffled"]))


def _fake_extractor(images_u8):
    x = images_u8.float() / 255.0
    small = F.adaptive_avg_pool2d(x, 4).flatten(1)               # [n, 48]
    w = torch.randn(48, 16, generator=torch.Generator().manual_seed(123))
    return torch.tanh(small @ w)


def _fake_image_embedder(images):
    small = F.adaptive_avg_pool2d(images.float(), 4).flatten(1)
    w = torch.randn(48, 24, generator=torch.Generator().manual_seed(321))
    return small @ w - 0.5 * w.sum(0)


def _fake_text_embedder(captions):
    out = []
    for c in captions:
        v = torch.zeros(24)
        for wd in c.split():
            g = torch.Generator().manual_seed(mix_seed(*[ord(ch) for ch in wd]))
            v += torch.randn(24, generator=g)
        out.append(v)
    return torch.stack(out)


def test_e6_sheet_fid_clip_sphere_hooks_with_fakes():
    torch.manual_seed(0)
    m = _perturb(build_model(smoke_config()))
    ema = EMA(m, 0.9)
    lat, text, mask = _eval_tensors(m.config, n=6)
    with _tmpdir() as d:
        sheet = SampleSheetHook(text[:3], mask[:3], fake_decoder, every=1, seeds=(0, 1), guidances=(1.0, 2.0), steps=2)
        rs = sheet(_ctx(m, ema, out_dir=d))
        rs2 = sheet(_ctx(m, ema, out_dir=d, step=2))
        files = rs["files"]
        from PIL import Image
        sizes = [Image.open(f).size for f in files]
        same_png = open(rs["files"][0], "rb").read() == open(rs2["files"][0], "rb").read()
        n_files = len(files)
    assert n_files == 4 and all(s == (3 * 16, 2 * 16) for s in sizes) and same_png

    metric = FidKid(extractor=_fake_extractor, kid_subsets=10, kid_subset_size=5)
    g = torch.Generator().manual_seed(9)
    metric.set_reference_images([torch.rand(12, 3, 16, 16, generator=g)], note="toy")
    rf = FidKidHook(metric, text, mask, fake_decoder, every=1, guidances=(1.0,), steps=2, weights=("raw", "ema"))(_ctx(m, ema))
    assert set(rf) == {"prompts", "raw_g1", "ema_g1"} and rf["raw_g1"]["fid"] >= -1e-6 and rf["raw_g1"]["n_fake"] == 6

    judge = ClipJudge(image_embedder=_fake_image_embedder, text_embedder=_fake_text_embedder)
    caps = ["a red cat", "two dogs", "blue car near tree", "snow", "the table", "a cat on the table"]
    photos = torch.rand(6, 3, 16, 16, generator=g)
    rc = ClipHook(judge, text, mask, caps, photos, fake_decoder, every=1, guidances=(1.0, 3.0), steps=2,
                  weights=("raw",))(_ctx(m))
    gr = rc["raw_g3"]["grounding"]
    assert math.isfinite(gr["mean"]) and gr["n"] == 6 and 0.0 <= rc["raw_g3"]["clip_score"]["mean"] <= 100.0
    # control: a model at initialization outputs zero velocity, so own and shuffled samples are the
    # same noise image: the grounding read is exactly zero
    torch.manual_seed(0)
    m0 = build_model(smoke_config())
    r0 = ClipHook(judge, text, mask, caps, photos, fake_decoder, every=1, guidances=(1.0,), steps=2,
                  weights=("raw",))(_ctx(m0))
    assert r0["raw_g1"]["grounding"]["mean"] == 0.0 and r0["raw_g1"]["grounding"]["se"] == 0.0
    assert gr["mean"] != 0.0

    sph = SphereCheckHook(text, mask, fake_decoder, every=1, thresholds={"a": 0.5, "b": 0.5}, guidances=(1.0,), steps=2)
    rsp = sph(_ctx(m, ema))
    assert {"a", "b", "c", "sphere_like"} <= set(rsp["raw_g1"]) and isinstance(rsp["ema_g1"]["sphere_like"], bool)
    _record("E6: sheet files / FID raw / grounding g3 (mean, se) / zero-velocity control / sphere a,b",
            (n_files, round(rf["raw_g1"]["fid"], 4), (round(gr["mean"], 4), round(gr["se"], 4)),
             r0["raw_g1"]["grounding"]["mean"], (round(rsp["raw_g1"]["a"], 3), round(rsp["raw_g1"]["b"], 3))))


def test_e7_checkpoint_and_milestone_push_hooks():
    torch.manual_seed(0)
    m = _perturb(build_model(smoke_config()))
    tc = TrainConfig(seed=3)
    opt = build_optimizer(m, tc)
    ema = EMA(m, 0.9)
    with _tmpdir() as d:
        calls = {"n": 0, "paths": []}

        def flaky(path, step):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise ConnectionError("transient")
            calls["paths"].append((os.path.basename(path), step))

        hook = CheckpointPushHook(flaky, os.path.join(d, "ckpt"), every=1, attempts=4, wait_seconds=0.0)
        r1 = hook(_ctx(m, ema, step=7, optimizer=opt, cfg=tc, data_position={"seed": 1, "next_step": 7}))
        st = load_training_state(os.path.join(d, "ckpt", "toy_state_step000007.pt"), restore_rng=False)
        assert r1["pushed"] is True and calls["n"] == 3 and st.step == 7 and st.data_position["seed"] == 1
        assert all(torch.equal(a, b) for a, b in zip(m.parameters(), st.model.parameters()))
        hook(_ctx(m, ema, step=8, optimizer=opt, cfg=tc))
        left = sorted(os.listdir(os.path.join(d, "ckpt")))
        assert left == ["toy_state_step000008.pt"]

        def broken(path, step):
            raise ConnectionError("down")

        runner = HookRunner([CheckpointPushHook(broken, os.path.join(d, "ckpt2"), every=1, attempts=2, wait_seconds=0.0)],
                            log=_quiet)
        runner.run(_ctx(m, ema, step=9, optimizer=opt, cfg=tc))
        assert len(runner.errors) == 1 and os.path.exists(os.path.join(d, "ckpt2", "toy_state_step000009.pt"))
        # a non-finite parameter is refused
        bad = build_model(smoke_config())
        with torch.no_grad():
            bad.patch_embed.weight[0, 0] = float("nan")
        runner2 = HookRunner([CheckpointPushHook(None, os.path.join(d, "ckpt3"), every=1)], log=_quiet)
        runner2.run(_ctx(bad, None, step=1, optimizer=build_optimizer(bad), cfg=tc))
        assert len(runner2.errors) == 1 and not os.path.exists(os.path.join(d, "ckpt3", "toy_state_step000001.pt"))

        pushed = []
        ms = MilestonePushHook(lambda paths, step: pushed.append(([os.path.basename(p) for p in paths], step)),
                               os.path.join(d, "ms"), milestones=[2, 4])
        assert ms.due(2) and not ms.due(3) and ms.due(4)
        rm = ms(_ctx(m, ema, step=2))
        m_raw, _, s_raw = load_checkpoint(os.path.join(d, "ms", "toy_step000002_raw.pt"))
        m_ema, _, _ = load_checkpoint(os.path.join(d, "ms", "toy_step000002_ema.pt"))
        assert pushed == [(["toy_step000002_raw.pt", "toy_step000002_ema.pt"], 2)] and rm["pushed"] is True
        assert all(torch.equal(a, b) for a, b in zip(m_raw.parameters(), m.parameters())) and s_raw == 2
        assert all(torch.equal(a, b) for a, b in zip(m_ema.parameters(), ema.model.parameters()))
    _record("E7: flaky push attempts / files kept locally / broken-push errors / non-finite refusals",
            (calls["n"], left, len(runner.errors), len(runner2.errors)))


# =============================================================================
# D  the memory-mapped cache, the sampler, the collate
# =============================================================================

def test_d1_memmap_cache_contents():
    with _tmpdir() as d:
        meta, recs, enc = build_toy_cache(d, n=10, batch_size=3)
        caps = [c for r in recs for c in r["captions"]]
        names = sorted(os.listdir(d))
        assert names == sorted(["caption_image.npy", "captions.jsonl", "image_captions.npy", "images.jsonl",
                                "latents.bin", "meta.json", "null_text.npy", "text_offsets.npy", "text_tokens.bin"])
        assert enc.sizes == [3, 3, 3, 1], "the builder must encode one batch at a time"
        true_len = [len(c.split()) + 1 for c in caps]
        assert meta["num_images"] == 10 and meta["num_captions"] == len(caps)
        assert meta["num_text_tokens"] == sum(true_len) and meta["latent_shape"] == [4, 8, 8]
        assert meta["null_text"]["length"] == 1 and meta["text_encoder"] == {"name": "fake words", "revision": "r1"}
        assert os.path.getsize(os.path.join(d, "latents.bin")) == 10 * 4 * 8 * 8 * 2
        assert os.path.getsize(os.path.join(d, "text_tokens.bin")) == sum(true_len) * TEXT_DIM * 2
        ds = MemmapLatentTextDataset(d)
        assert np.diff(ds.text_offsets).tolist() == true_len
        assert ds.caption_image.tolist() == [i for i, r in enumerate(recs) for _ in r["captions"]]
        with open(os.path.join(d, "captions.jsonl"), encoding="utf-8") as fh:
            lines = [json.loads(x) for x in fh]
        with open(os.path.join(d, "images.jsonl"), encoding="utf-8") as fh:
            ims = [json.loads(x) for x in fh]
        assert [x["text"] for x in lines] == caps and lines[3]["image_index"] == ds.caption_image[3]
        assert [x["file_name"] for x in ims] == [r["file_name"] for r in recs] and ims[4]["license"] == recs[4]["license"]
        # statistics over the stored float16 latents
        raw = fake_encode_images([r["image"] for r in recs]).numpy().astype(np.float16).astype(np.float64)
        mean, std = raw.mean(axis=(0, 2, 3)), raw.std(axis=(0, 2, 3))
        assert np.allclose(meta["latent_mean"], mean, rtol=1e-10, atol=1e-12)
        assert np.allclose(meta["latent_std"], std, rtol=1e-10, atol=1e-12)
        # an item: normalized latent, its caption's states at true length
        i = next(k for k in range(5, 10) if len(recs[k]["captions"]) >= 2)
        j = ds.captions_of(i)[-1]
        z, e = ds[(i, j)]
        ref_z = (torch.from_numpy(raw[i]).float() - torch.tensor(mean).float().view(4, 1, 1)) / torch.tensor(std).float().view(4, 1, 1)
        st, mk = fake_encode_texts([caps[j]])
        ref_e = st[0][mk[0]].half().float()
        assert z.dtype == torch.float32 and torch.allclose(z, ref_z, atol=1e-5)
        assert torch.equal(e, ref_e) and e.shape[0] == true_len[j]
        # control: the image's other caption reads other states, and a caption of another image is refused
        j0 = ds.captions_of(i)[0]
        other = ds[(i, j0)][1]
        assert j0 != j and caps[j0] != caps[j]
        assert other.shape != e.shape or not torch.equal(other, e)
        try:
            ds[(i, ds.captions_of(i - 1)[0])]
            refused = False
        except IndexError:
            refused = True
        assert refused
        # the null caption, and statistics passed in (no normalization effect)
        assert ds.null_text().shape == (1, TEXT_DIM)
        ds_raw = MemmapLatentTextDataset(d, stats=([0.0] * 4, [1.0] * 4))
        assert torch.allclose(ds_raw[(i, j)][0], torch.from_numpy(raw[i]).float())
        assert torch.allclose(ds.denormalize(z), torch.from_numpy(raw[i]).float(), atol=1e-4)
        pairs = fixed_caption_pairs(ds.image_captions, [0, 3, 5], seed=1)
        lat, text, mask = gather_pairs(ds, pairs)
        assert pairs == fixed_caption_pairs(ds.image_captions, [0, 3, 5], seed=1)
        assert lat.shape == (3, 4, 8, 8) and mask.sum(1).tolist() == [true_len[p[1]] for p in pairs]
        _close(ds)
        _close(ds_raw)
    _record("D1: images / captions / tokens / encoder batch sizes / latent mean",
            (meta["num_images"], meta["num_captions"], meta["num_text_tokens"], enc.sizes,
             [round(x, 4) for x in meta["latent_mean"]]))


def test_d2_dataset_pickles_without_maps_and_workers_agree():
    with _tmpdir() as d:
        build_toy_cache(d, n=10)
        ds = MemmapLatentTextDataset(d)
        first = ds[(2, ds.captions_of(2)[0])]
        blob = pickle.dumps(ds)
        ds2 = pickle.loads(blob)
        assert ds2._latents is None and ds2._tokens is None and len(blob) < 50_000
        again = ds2[(2, ds.captions_of(2)[0])]
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
        smp0 = StepOrderSampler(ds.image_captions, 4, 1, 0, 5)
        smp2 = StepOrderSampler(ds.image_captions, 4, 1, 0, 5)
        b0 = list(torch.utils.data.DataLoader(ds, batch_sampler=smp0, collate_fn=collate_latent_text, num_workers=0))
        b2 = list(torch.utils.data.DataLoader(ds, batch_sampler=smp2, collate_fn=collate_latent_text, num_workers=2))
        same = all(all(torch.equal(x, y) for x, y in zip(p, q)) for p, q in zip(b0, b2)) and len(b0) == len(b2) == 5
        _close(ds)
        _close(ds2)
    assert same
    _record("D2: pickled dataset bytes / batches equal with 0 and 2 workers", (len(blob), same))


def test_d3_step_order_sampler():
    offsets = np.cumsum([0] + [1, 3, 2, 4, 1, 1, 5, 2, 3, 1]).tolist()          # 10 images
    s = StepOrderSampler(offsets, batch_size=5, seed=7)
    pairs = [p for k in range(6) for p in s.batch(k)]                          # 30 samples = 3 epochs
    for e in range(3):
        imgs = sorted(p[0] for p in pairs[10 * e:10 * (e + 1)])
        assert imgs == list(range(10)), f"epoch {e} is not a permutation"
    assert all(offsets[i] <= c < offsets[i + 1] for i, c in pairs)
    assert [p[0] for p in pairs[:10]] != [p[0] for p in pairs[10:20]]          # control: epochs differ
    again = [p for k in range(6) for p in StepOrderSampler(offsets, 5, 7).batch(k)]
    other = [p for k in range(6) for p in StepOrderSampler(offsets, 5, 8).batch(k)]
    assert pairs == again and pairs != other
    resumed = list(StepOrderSampler(offsets, 5, 7, start_step=4, end_step=6))
    assert resumed == [s.batch(4), s.batch(5)] and len(StepOrderSampler(offsets, 5, 7, 4, 6)) == 2
    # batches across an epoch edge: batch 4 of 10 images, step 2 = positions 8..11
    s4 = StepOrderSampler(offsets, 4, 7)
    perm0, _ = s4.epoch_order(0)
    perm1, _ = s4.epoch_order(1)
    assert [p[0] for p in s4.batch(2)] == [int(perm0[8]), int(perm0[9]), int(perm1[0]), int(perm1[1])]
    counts = np.zeros(10)
    for k in range(400):
        for i, c in s.batch(k):
            if i == 6:
                counts[c - offsets[6]] += 1
    assert (counts[:5] > 0).all() and counts[5:].sum() == 0
    _record("D3: caption draws over 5 captions of image 6 in 400 steps", counts[:5].tolist())


def test_d4_collate_pads_to_longest_and_masks():
    items = [(torch.randn(4, 8, 8), torch.randn(n, TEXT_DIM)) for n in (2, 5, 3)]
    z, text, mask = collate_latent_text(items)
    assert z.shape == (3, 4, 8, 8) and text.shape == (3, 5, TEXT_DIM) and mask.sum(1).tolist() == [2, 5, 3]
    assert torch.equal(text[0, :2], items[0][1]) and bool((text[0, 2:] == 0).all()) and not bool(mask[2, 3:].any())


# =============================================================================
# S  the sphere check
# =============================================================================

def test_s1_sphere_check_known_answers():
    sph = sphere_check.render_spheres(4, size=64, seed=0, jitter=0.0, light=(0.0, 0.0, 1.0))
    ms = sphere_check.measure(sph)
    noise = torch.rand(8, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    mn = sphere_check.measure(noise)
    ramp = torch.linspace(0, 1, 64).view(1, 1, 1, 64).expand(2, 3, 64, 64).contiguous()
    mg = sphere_check.measure(ramp)
    assert ms["a"] >= 0.95 and ms["b"] >= 0.95, ms
    assert abs(mn["a"]) < 0.05 and mn["b"] < 0.05, mn
    assert mg["a180"] <= -0.99 and abs(mg["a90"]) < 1e-6, mg
    same = sphere_check.sameness_across_prompts(sphere_check.as_images(sph[:1].repeat(4, 1, 1, 1)))
    one_prompt = sphere_check.sameness_across_prompts(sphere_check.as_images(noise), prompt_ids=[0] * 8)
    assert abs(same["c"] - 1.0) < 1e-9 and abs(mn["c"]) < 0.1 and one_prompt["pairs"] == 0
    u8 = (noise * 255).round().to(torch.uint8)
    assert abs(sphere_check.measure(u8)["a"] - sphere_check.measure(u8.double() / 255)["a"]) < 1e-12
    _record("S1: centred sphere (a, b) / noise (a, b, c) / left-right ramp a180",
            ((round(ms["a"], 4), round(ms["b"], 4)), (round(mn["a"], 4), round(mn["b"], 4), round(mn["c"], 4)),
             round(mg["a180"], 6)))


def _scene_like(n, size, seed):
    """Off-centre rectangles on vertical gradients: a stand-in for photos in CPU tests."""
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(n, 3, size, size)
    yy = torch.linspace(0, 1, size).view(size, 1)
    for k in range(n):
        top, bot = torch.rand(3, generator=g), torch.rand(3, generator=g)
        img = top.view(3, 1, 1) * (1 - yy) + bot.view(3, 1, 1) * yy
        img = img.expand(3, size, size).clone()
        for _ in range(4):
            x0, y0 = (torch.randint(0, size - 8, (2,), generator=g)).tolist()
            w, h = (torch.randint(4, size // 2, (2,), generator=g)).tolist()
            img[:, y0:y0 + h, x0:x0 + w] = torch.rand(3, 1, 1, generator=g)
        out[k] = img
    return out


def test_s2_sphere_calibration_controls():
    real = _scene_like(32, 64, 3)
    cal = sphere_check.calibrate(real, n_controls=32, seed=0)
    th = cal["thresholds"]
    assert cal["valid"] and cal["flags"]["sphere"] and not cal["flags"]["real"]
    assert cal["real"]["a"] < th["a"] < cal["sphere"]["a"] and cal["real"]["b"] < th["b"] < cal["sphere"]["b"]
    assert cal["blob"]["b"] > cal["real"]["b"]
    # control: when the "real" set is spheres, the controls cannot separate and calibration says so
    bad = sphere_check.calibrate(sphere_check.render_spheres(32, 64, seed=5), n_controls=32, seed=0)
    assert not bad["valid"] or abs(bad["sphere"]["a"] - bad["real"]["a"]) < 0.05
    _record("S2: (a, b) means real / sphere / blob; thresholds",
            ((round(cal["real"]["a"], 3), round(cal["real"]["b"], 3)), (round(cal["sphere"]["a"], 3), round(cal["sphere"]["b"], 3)),
             (round(cal["blob"]["a"], 3), round(cal["blob"]["b"], 3)), {k: round(v, 3) for k, v in th.items()}))


# =============================================================================
# M  metric plumbing (fake extractor and embedders; the real ones run on the card)
# =============================================================================

def _sqrtm_psd(m):
    w, v = torch.linalg.eigh(m)
    return (v * w.clamp_min(0).sqrt()) @ v.T


def test_m1_fid_kid_plumbing():
    g = torch.Generator().manual_seed(0)
    real = torch.rand(40, 3, 16, 16, generator=g)
    fake = (real * 0.7 + 0.2 * torch.rand(40, 3, 16, 16, generator=g)).clamp(0, 1)
    a = FidKid(extractor=_fake_extractor, kid_subsets=20, kid_subset_size=15, batch_size=7)
    a.set_reference_images([real[:20], real[20:]])
    a.reset()
    a.update(real)
    self_fid = a.compute()
    a.reset()
    a.update(fake[:13])
    a.update(fake[13:])
    r = a.compute()
    # an independent closed form of the Frechet distance (symmetric square roots)
    f_r, f_f = _fake_extractor((real * 255).round().to(torch.uint8)), _fake_extractor((fake * 255).round().to(torch.uint8))
    mu1, s1 = gaussian_stats(f_r)
    mu2, s2 = gaussian_stats(f_f)
    r1 = _sqrtm_psd(s1)
    closed = float((mu1 - mu2).square().sum() + torch.trace(s1 + s2 - 2 * _sqrtm_psd(r1 @ s2 @ r1)))
    assert abs(self_fid["fid"]) < 1e-6 and abs(r["fid"] - closed) <= 1e-6 * max(1.0, closed)
    assert abs(frechet_distance(mu1, s1, mu2, s2) - closed) <= 1e-6 * max(1.0, closed)
    # batch size does not matter; the saved reference gives the same numbers
    b = FidKid(extractor=_fake_extractor, kid_subsets=20, kid_subset_size=15, batch_size=40)
    with _tmpdir() as d:
        a.save_reference(os.path.join(d, "ref.pt"))
        b.load_reference(os.path.join(d, "ref.pt"))
    b.update(fake)
    rb = b.compute()
    assert abs(rb["fid"] - r["fid"]) <= 1e-9 * max(1.0, r["fid"]) and abs(rb["kid_mean"] - r["kid_mean"]) <= 1e-9
    # KID on features: the same set reads near zero; a set shifted by one unit per feature reads far above
    gk = torch.Generator().manual_seed(4)
    f0 = torch.randn(200, 16, generator=gk)
    k = FidKid(extractor=_fake_extractor, kid_subsets=50, kid_subset_size=100, seed=1)
    k.set_reference_features(f0)
    k.update_features(f0.clone())
    k_self = k.compute()
    k.reset()
    k.update_features(f0 + 1.0)
    k_shift = k.compute()
    assert abs(k_self["kid_mean"]) < 0.05 * k_shift["kid_mean"] and k_shift["kid_mean"] > 0
    assert k_self["subset_size_used"] == 100 and abs(k_self["fid"]) < 1e-9 and abs(k_shift["fid"] - 16.0) < 1e-6
    # lazy imports: no extractor package was imported, and building the standard one needs it
    assert "torchmetrics" not in sys.modules
    try:
        import torchmetrics  # noqa: F401
        have_tm = True
    except ImportError:
        have_tm = False
    if not have_tm:
        try:
            FidKid().features(real[:2])
            raised = ""
        except ImportError as e:
            raised = str(e)
        assert "torchmetrics" in raised
    _record("M1: FID self / FID shifted / closed form / KID self / KID shifted by one unit",
            (f"{self_fid['fid']:.2e}", round(r["fid"], 6), round(closed, 6), f"{k_self['kid_mean']:.2e}",
             f"{k_shift['kid_mean']:.3e}"))


def test_m2_clip_plumbing_and_grounding_control():
    before = "transformers" in sys.modules
    judge = ClipJudge(image_embedder=_fake_image_embedder, text_embedder=_fake_text_embedder, batch_size=3)
    lazy = ClipJudge()                                                          # the default judge loads nothing yet
    assert ("transformers" in sys.modules) == before and lazy._model is None
    g = torch.Generator().manual_seed(1)
    photos = torch.rand(7, 3, 16, 16, generator=g)
    samples = torch.rand(7, 3, 16, 16, generator=g)
    zero = judge.paired_grounding(samples, samples.clone(), photos)
    assert zero["mean"] == 0.0 and zero["se"] == 0.0 and zero["n"] == 7
    own_is_photo = judge.paired_grounding(photos, samples, photos)
    e_p, e_s = judge.embed_images(photos), judge.embed_images(samples)
    manual = 1.0 - (e_s * e_p).sum(-1)
    assert abs(own_is_photo["mean"] - float(manual.mean())) < 1e-12 and own_is_photo["mean"] > 0
    assert abs(own_is_photo["se"] - float(manual.std(unbiased=True) / math.sqrt(7))) < 1e-12
    caps = ["a red cat", "two dogs", "snow", "blue car", "the table", "near tree", "in snow"]
    cs = judge.clip_score(samples, caps)
    e_t = judge.embed_texts(caps)
    ref = (100.0 * (e_s * e_t).sum(-1).clamp_min(0)).mean()
    assert abs(cs["mean"] - float(ref)) < 1e-10
    _record("M2: identical own/shuffled grounding / own = photo grounding (mean, se) / CLIP score",
            (zero["mean"], (round(own_is_photo["mean"], 4), round(own_is_photo["se"], 4)), round(cs["mean"], 3)))


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
