"""
geofractal.router.prefab.agatha_v2.hooks
========================================

Evaluation and shipping callbacks for trainer.fit().

A hook runs after the optimizer step and the EMA update: at every `every`-th step,
at the listed steps `at`, and, when `at_end` is set, once after the last step.
HookRunner calls the hooks that are due, catches every exception, logs it and goes
on: a failing hook never stops a run. Each result is printed in one line and
appended as one JSON line to the runner's results file.

Built in:
    HeldOutLossHook     held-out flow loss per noise band on fixed noise and a fixed
                        grid of flow times, for the raw and the EMA weights; the caption
                        gap: the same loss with a fixed derangement of the evaluation
                        captions and with the null caption, minus the own-caption loss
    SampleSheetHook     fixed prompts x fixed seeds, sampled and decoded by a decoder
                        callable, saved as PNG grids (rows = seeds, columns = prompts)
    FidKidHook          FID / KID of decoded samples through a metric object passed in
    ClipHook            the text-image CLIP score of samples and the paired grounding
                        read (own caption against a deranged caption, same noise,
                        both compared with the real photo) through a judge passed in
    SphereCheckHook     the measures of sphere_check.py on decoded samples, and the
                        sphere-like flag when thresholds are given
    CheckpointPushHook  the full training state (exact resume), saved and handed to a
                        push callable, with retries
    MilestonePushHook   raw and EMA weights (and listed files) at milestone steps,
                        handed to a push callable, with retries

Every sampled read uses the model's midpoint sampler. Guidance 1 is the conditional
velocity alone (no unconditional pass); other values use v_null + g (v_c - v_null).
A decoder maps normalized latents [n, C, H, W] to images [n, 3, h, w] in [0, 1].

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from geofractal.router.prefab.agatha_v2 import sphere_check
from geofractal.router.prefab.agatha_v2.bands import band_coordinate, band_weights
from geofractal.router.prefab.agatha_v2.data import mix_seed
from geofractal.router.prefab.agatha_v2.trainer import (
    HookContext,
    autocast_for,
    save_checkpoint,
    save_training_state,
)

Decoder = Callable[[Tensor], Tensor]

NOISE_SALT = 0x686F6F6B


# =============================================================================
# FIXED DRAWS
# =============================================================================

def fixed_noise(shape: Sequence[int], seed: int) -> Tensor:
    """Standard normal draws of the given shape from a seed (CPU, float32)."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(tuple(shape), generator=g)


def per_item_noise(n: int, latent_shape: Sequence[int], seed: int) -> Tensor:
    """[n, C, H, W]: item i's noise depends only on (seed, i), so a longer list keeps its prefix."""
    return torch.stack([fixed_noise(latent_shape, mix_seed(seed, i, NOISE_SALT)) for i in range(n)])


def fixed_t_grid(n: int, seed: int, lo: float = 0.02, hi: float = 0.98) -> Tensor:
    """n flow times evenly spread over [lo, hi] (cell centres), assigned to items by a fixed permutation."""
    g = torch.Generator().manual_seed(int(seed))
    return lo + (hi - lo) * (torch.randperm(n, generator=g).float() + 0.5) / n


def derangement(n: int, seed: int) -> Tensor:
    """A fixed permutation with no fixed point (one n-cycle through a random order); n >= 2."""
    if n < 2:
        raise ValueError("a derangement needs at least two items")
    g = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(n, generator=g)
    d = torch.empty(n, dtype=torch.long)
    d[order] = order.roll(-1)
    return d


# =============================================================================
# SAMPLING AND DECODING
# =============================================================================

class _EvalMode:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.was = model.training

    def __enter__(self):
        self.model.eval()
        return self.model

    def __exit__(self, *exc):
        self.model.train(self.was)
        return False


def models_of(ctx: HookContext, weights: Sequence[str]) -> List[Tuple[str, torch.nn.Module]]:
    """[(tag, model)] for the requested weights: "raw" (the trained model), "ema" (when kept)."""
    out = []
    for w in weights:
        if w == "raw":
            out.append(("raw", ctx.model))
        elif w == "ema" and ctx.ema is not None:
            out.append(("ema", ctx.ema.model))
    return out


@torch.no_grad()
def sample_latents(model, text: Tensor, mask: Tensor, noise: Tensor, steps: int, guidance: Optional[float],
                   device: str, use_amp: bool, batch_size: int = 64) -> Tensor:
    """Midpoint sampling from the given noise; returns normalized latents on the CPU (float32)."""
    g = None if guidance is None or float(guidance) == 1.0 else float(guidance)
    out = []
    with _EvalMode(model):
        for i in range(0, noise.shape[0], batch_size):
            tb = text[i:i + batch_size].to(device).float()
            mb = mask[i:i + batch_size].to(device)
            nb = noise[i:i + batch_size].to(device).float()
            with autocast_for(device, use_amp):
                z = model.sample_midpoint(tb, mb, noise=nb, num_steps=steps, guidance_scale=g)
            out.append(z.float().cpu())
    return torch.cat(out)


@torch.no_grad()
def decode_images(decoder: Decoder, latents: Tensor, batch_size: int = 32) -> Tensor:
    """Images [n, 3, h, w] in [0, 1] (CPU, float32) from normalized latents, in batches."""
    out = []
    for i in range(0, latents.shape[0], batch_size):
        out.append(decoder(latents[i:i + batch_size]).float().clamp(0, 1).cpu())
    return torch.cat(out)


def save_png_grid(images: Tensor, path: str, ncols: int) -> Tuple[int, int]:
    """Save images [n, 3, h, w] in [0, 1] as one PNG grid with `ncols` columns; returns (rows, cols)."""
    from PIL import Image
    n, _, h, w = images.shape
    rows = int(math.ceil(n / ncols))
    grid = torch.zeros(3, rows * h, ncols * w)
    for k in range(n):
        r, c = divmod(k, ncols)
        grid[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = images[k]
    arr = (grid.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")
    return rows, ncols


def with_retries(fn: Callable[[], Any], attempts: int = 4, wait_seconds: float = 10.0,
                 log: Callable[[str], None] = print, what: str = "push") -> bool:
    """Call fn until it returns without an exception, at most `attempts` times (waits grow linearly)."""
    for a in range(1, attempts + 1):
        try:
            fn()
            return True
        except Exception as e:  # noqa: BLE001
            log(f"  {what}: attempt {a} of {attempts} failed: {type(e).__name__}: {str(e)[:300]}")
            if a < attempts and wait_seconds > 0:
                time.sleep(wait_seconds * a)
    return False


# =============================================================================
# HOOK BASE AND RUNNER
# =============================================================================

class Hook:
    """Base class: schedule (every / at / at_end) and a name; __call__(ctx) returns a dict."""

    name = "hook"

    def __init__(self, every: int = 0, at: Iterable[int] = (), at_end: bool = False, name: Optional[str] = None):
        self.every = int(every)
        self.at = {int(s) for s in at}
        self.at_end = bool(at_end)
        if name:
            self.name = name
        self.last_step: Optional[int] = None

    def due(self, step: int, final: bool = False) -> bool:
        if self.last_step == step:
            return False
        if final:
            return self.at_end
        return (self.every > 0 and step % self.every == 0) or step in self.at

    def __call__(self, ctx: HookContext) -> Dict[str, Any]:
        raise NotImplementedError


def _jsonable(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def _summary(result: Dict[str, Any], limit: int = 600) -> str:
    try:
        text = json.dumps(result, default=_jsonable)
    except Exception:  # noqa: BLE001
        text = str(result)
    return text if len(text) <= limit else text[:limit] + " ..."


class HookRunner:
    """Runs due hooks after a step; never raises. Results go to the log and to a JSON-lines file."""

    def __init__(self, hooks: Sequence[Hook], log: Callable[[str], None] = print, results_path: Optional[str] = None):
        self.hooks = list(hooks)
        self.log = log
        self.results_path = results_path
        self.errors: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []

    def run(self, ctx: HookContext, final: bool = False, force: bool = False) -> List[Dict[str, Any]]:
        """Call the hooks due at ctx.step (all of them when force is set)."""
        done = []
        for hook in self.hooks:
            try:
                due = force or hook.due(ctx.step, final)
            except Exception:  # noqa: BLE001
                due = False
            if not due:
                continue
            hook.last_step = ctx.step
            t0 = time.time()
            try:
                result = hook(ctx)
                ok = True
            except Exception as e:  # noqa: BLE001
                ok = False
                result = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
                self.errors.append({"step": ctx.step, "hook": hook.name, "error": result["error"],
                                    "traceback": traceback.format_exc()[-4000:]})
            record = {"run": ctx.run_name, "step": ctx.step, "hook": hook.name, "ok": ok,
                      "seconds": round(time.time() - t0, 2), "result": result}
            self.records.append(record)
            try:
                self.log(f"[hook {hook.name}] step {ctx.step} {'ok' if ok else 'FAILED'} in {record['seconds']} s: "
                         f"{_summary(result)}")
            except Exception:  # noqa: BLE001
                pass
            if self.results_path:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(self.results_path)), exist_ok=True)
                    with open(self.results_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, default=_jsonable) + "\n")
                except Exception as e:  # noqa: BLE001
                    self.errors.append({"step": ctx.step, "hook": "results file", "error": repr(e)[:300]})
            done.append(record)
        return done


# =============================================================================
# HELD-OUT FLOW LOSS AND THE CAPTION GAP
# =============================================================================

@torch.no_grad()
def per_sample_flow_loss(model, latents: Tensor, text: Tensor, mask: Tensor, noise: Tensor, t: Tensor,
                         device: str, use_amp: bool, batch_size: int = 250,
                         caption_order: Optional[Tensor] = None, null_caption: bool = False) -> Tensor:
    """Per-sample squared error of the velocity on fixed draws, [M] (float64, CPU).

    caption_order: item i reads the caption of item caption_order[i] (a derangement for the gap);
    null_caption: every item reads the null caption.
    """
    out = []
    with _EvalMode(model):
        for i in range(0, latents.shape[0], batch_size):
            z1 = latents[i:i + batch_size].to(device).float()
            z0 = noise[i:i + batch_size].to(device).float()
            tt = t[i:i + batch_size].to(device).float()
            b = z1.shape[0]
            if null_caption:
                tx, mk = model.null_condition(b, 1)
            else:
                idx = torch.arange(i, i + b) if caption_order is None else caption_order[i:i + b]
                tx, mk = text[idx].to(device).float(), mask[idx].to(device)
            tv = tt.view(b, 1, 1, 1)
            z_t = (1 - tv) * z0 + tv * z1
            with autocast_for(device, use_amp):
                v = model.velocity(z_t, tt, tx, mk)
            out.append(((v.float() - (z1 - z0)) ** 2).flatten(1).mean(1).double().cpu())
    return torch.cat(out)


def band_means(per_sample: Tensor, t: Tensor, config) -> List[float]:
    """Means of a per-sample quantity within the three noise bands (window-weighted)."""
    bw = band_weights(band_coordinate(t.double(), config.shift), config.band_edges, config.band_half_width)
    return ((bw * per_sample[:, None]).sum(0) / bw.sum(0).clamp_min(1e-12)).tolist()


class HeldOutLossHook(Hook):
    """Held-out flow loss on fixed draws, per noise band, for raw and EMA weights, with the caption gap.

    latents [M, C, H, W] normalized, text [M, L, D], mask [M, L]: one caption per item.
    """

    name = "heldout"

    def __init__(self, latents: Tensor, text: Tensor, mask: Tensor, *, every: int = 0, at: Iterable[int] = (),
                 at_end: bool = False, seed: int = 20261117, caption_gap: bool = True,
                 weights: Sequence[str] = ("raw", "ema"), batch_size: int = 250, name: Optional[str] = None):
        super().__init__(every, at, at_end, name)
        m = latents.shape[0]
        self.latents, self.text, self.mask = latents, text, mask
        self.noise = fixed_noise(latents.shape, mix_seed(seed, 1))
        self.t = fixed_t_grid(m, mix_seed(seed, 2))
        self.order = derangement(m, mix_seed(seed, 3)) if caption_gap else None
        self.weights = tuple(weights)
        self.batch_size = int(batch_size)

    def __call__(self, ctx: HookContext) -> Dict[str, Any]:
        out: Dict[str, Any] = {"items": int(self.latents.shape[0])}
        for tag, model in models_of(ctx, self.weights):
            args = (model, self.latents, self.text, self.mask, self.noise, self.t, ctx.device, ctx.use_amp,
                    self.batch_size)
            own = per_sample_flow_loss(*args)
            row = {"loss": float(own.mean()), "bands": band_means(own, self.t, model.config)}
            if self.order is not None:
                shuf = per_sample_flow_loss(*args, caption_order=self.order)
                null = per_sample_flow_loss(*args, null_caption=True)
                d_shuf, d_null = shuf - own, null - own
                n = own.numel()
                row.update(loss_shuffled=float(shuf.mean()), loss_null=float(null.mean()),
                           gap_shuffled=float(d_shuf.mean()),
                           gap_shuffled_se=float(d_shuf.std(unbiased=True) / math.sqrt(n)),
                           gap_null=float(d_null.mean()),
                           gap_null_se=float(d_null.std(unbiased=True) / math.sqrt(n)),
                           gap_shuffled_bands=band_means(d_shuf, self.t, model.config))
            out[tag] = row
        return out


# =============================================================================
# SAMPLE SHEETS
# =============================================================================

class SampleSheetHook(Hook):
    """PNG grids of fixed prompts (columns) x fixed seeds (rows), per weights and guidance."""

    name = "sheet"

    def __init__(self, text: Tensor, mask: Tensor, decoder: Decoder, *, every: int = 0, at: Iterable[int] = (),
                 at_end: bool = False, seeds: Sequence[int] = (0, 1, 2, 3), guidances: Sequence[float] = (1.0, 3.0),
                 steps: int = 25, weights: Sequence[str] = ("raw", "ema"), batch_size: int = 64,
                 decode_batch: int = 32, out_subdir: str = "sheets", name: Optional[str] = None):
        super().__init__(every, at, at_end, name)
        self.text, self.mask, self.decoder = text, mask, decoder
        self.seeds = tuple(int(s) for s in seeds)
        self.guidances = tuple(float(g) for g in guidances)
        self.steps, self.weights = int(steps), tuple(weights)
        self.batch_size, self.decode_batch, self.out_subdir = int(batch_size), int(decode_batch), out_subdir
        self._noise: Optional[Tensor] = None
        self.last_files: List[str] = []

    def noise_for(self, latent_shape: Sequence[int]) -> Tensor:
        if self._noise is None:
            p = self.text.shape[0]
            rows = [torch.stack([fixed_noise(latent_shape, mix_seed(s, j, NOISE_SALT)) for j in range(p)])
                    for s in self.seeds]
            self._noise = torch.cat(rows)                      # [S * P], row-major: seed, then prompt
        return self._noise

    def __call__(self, ctx: HookContext) -> Dict[str, Any]:
        p = self.text.shape[0]
        reps = len(self.seeds)
        text = self.text.repeat(reps, 1, 1)
        mask = self.mask.repeat(reps, 1)
        out: Dict[str, Any] = {"prompts": p, "seeds": list(self.seeds), "files": []}
        for tag, model in models_of(ctx, self.weights):
            noise = self.noise_for(model.config.latent_shape)
            for g in self.guidances:
                z = sample_latents(model, text, mask, noise, self.steps, g, ctx.device, ctx.use_amp, self.batch_size)
                images = decode_images(self.decoder, z, self.decode_batch)
                fn = f"{ctx.run_name}_step{ctx.step:06d}_{tag}_g{g:g}.png"
                path = os.path.join(ctx.out_dir, self.out_subdir, fn)
                save_png_grid(images, path, ncols=p)
                out["files"].append(path)
                out[f"{tag}_g{g:g}"] = {"finite": bool(torch.isfinite(z).all()), "latent_mean": float(z.mean()),
                                         "latent_std": float(z.std()), "image_mean": float(images.mean()),
                                         "image_std": float(images.std())}
        self.last_files = list(out["files"])
        return out


# =============================================================================
# FID / KID
# =============================================================================

class FidKidHook(Hook):
    """FID / KID of decoded samples for fixed prompts and fixed noise, through `metric`.

    metric: an object with reset(), update(images [n, 3, h, w] in [0, 1]) and compute() -> dict,
    whose reference statistics are already set (metrics.FidKid).
    """

    name = "fid"

    def __init__(self, metric: Any, text: Tensor, mask: Tensor, decoder: Decoder, *, every: int = 0,
                 at: Iterable[int] = (), at_end: bool = False, seed: int = 20261118,
                 guidances: Sequence[float] = (1.0, 3.0), steps: int = 25, weights: Sequence[str] = ("ema",),
                 batch_size: int = 256, decode_batch: int = 64, name: Optional[str] = None):
        super().__init__(every, at, at_end, name)
        self.metric, self.text, self.mask, self.decoder = metric, text, mask, decoder
        self.seed = int(seed)
        self.guidances = tuple(float(g) for g in guidances)
        self.steps, self.weights = int(steps), tuple(weights)
        self.batch_size, self.decode_batch = int(batch_size), int(decode_batch)

    def __call__(self, ctx: HookContext) -> Dict[str, Any]:
        n = self.text.shape[0]
        out: Dict[str, Any] = {"prompts": n}
        for tag, model in models_of(ctx, self.weights):
            shape = model.config.latent_shape
            for g in self.guidances:
                self.metric.reset()
                for i in range(0, n, self.batch_size):
                    j = min(i + self.batch_size, n)
                    noise = torch.stack([fixed_noise(shape, mix_seed(self.seed, k, NOISE_SALT)) for k in range(i, j)])
                    z = sample_latents(model, self.text[i:j], self.mask[i:j], noise, self.steps, g, ctx.device,
                                       ctx.use_amp, self.batch_size)
                    self.metric.update(decode_images(self.decoder, z, self.decode_batch))
                out[f"{tag}_g{g:g}"] = self.metric.compute()
        return out


# =============================================================================
# CLIP SCORE AND THE PAIRED GROUNDING READ
# =============================================================================

class ClipHook(Hook):
    """Text-image CLIP score of samples, and the paired grounding read, through a judge.

    For each item i of a fixed evaluation subset (its caption's token states text[i], its
    caption string captions[i], its real photo photos[i]): one sample from its own caption
    and one from caption order[i] (a fixed derangement), with the SAME noise. The judge
    (metrics.ClipJudge) embeds both samples and the photo with its image tower:
        grounding = mean_i [cos(own_i, photo_i) - cos(shuffled_i, photo_i)]  (with its standard error)
    and scores the own samples against their caption strings (CLIP score).
    """

    name = "clip"

    def __init__(self, judge: Any, text: Tensor, mask: Tensor, captions: Sequence[str], photos: Tensor,
                 decoder: Decoder, *, every: int = 0, at: Iterable[int] = (), at_end: bool = False,
                 seed: int = 20261119, guidances: Sequence[float] = (1.0, 3.0), steps: int = 25,
                 weights: Sequence[str] = ("ema",), batch_size: int = 128, decode_batch: int = 64,
                 name: Optional[str] = None):
        super().__init__(every, at, at_end, name)
        if not (text.shape[0] == mask.shape[0] == len(captions) == photos.shape[0]):
            raise ValueError("text, mask, captions and photos must have the same length")
        self.judge, self.text, self.mask, self.decoder = judge, text, mask, decoder
        self.captions = list(captions)
        self.photos = photos
        self.order = derangement(text.shape[0], mix_seed(seed, 3))
        self.seed = int(seed)
        self.guidances = tuple(float(g) for g in guidances)
        self.steps, self.weights = int(steps), tuple(weights)
        self.batch_size, self.decode_batch = int(batch_size), int(decode_batch)
        self._photo_emb: Optional[Tensor] = None
        self._text_emb: Optional[Tensor] = None

    def __call__(self, ctx: HookContext) -> Dict[str, Any]:
        n = self.text.shape[0]
        if self._photo_emb is None:
            self._photo_emb = self.judge.embed_images(self.photos)
            self._text_emb = self.judge.embed_texts(self.captions)
        out: Dict[str, Any] = {"items": n}
        for tag, model in models_of(ctx, self.weights):
            shape = model.config.latent_shape
            for g in self.guidances:
                own_e, shuf_e = [], []
                for i in range(0, n, self.batch_size):
                    j = min(i + self.batch_size, n)
                    noise = torch.stack([fixed_noise(shape, mix_seed(self.seed, k, NOISE_SALT)) for k in range(i, j)])
                    oi = self.order[i:j]
                    z_own = sample_latents(model, self.text[i:j], self.mask[i:j], noise, self.steps, g, ctx.device,
                                           ctx.use_amp, self.batch_size)
                    z_shuf = sample_latents(model, self.text[oi], self.mask[oi], noise, self.steps, g, ctx.device,
                                            ctx.use_amp, self.batch_size)
                    own_e.append(self.judge.embed_images(decode_images(self.decoder, z_own, self.decode_batch)))
                    shuf_e.append(self.judge.embed_images(decode_images(self.decoder, z_shuf, self.decode_batch)))
                e_own, e_shuf = torch.cat(own_e), torch.cat(shuf_e)
                out[f"{tag}_g{g:g}"] = {
                    "clip_score": self.judge.clip_score_from_embeddings(e_own, self._text_emb),
                    "grounding": self.judge.grounding_from_embeddings(e_own, e_shuf, self._photo_emb),
                }
        return out


# =============================================================================
# SPHERE CHECK
# =============================================================================

class SphereCheckHook(Hook):
    """The sphere check of decoded samples: one sample per prompt (distinct prompts)."""

    name = "sphere"

    def __init__(self, text: Tensor, mask: Tensor, decoder: Decoder, *, every: int = 0, at: Iterable[int] = (),
                 at_end: bool = False, thresholds: Optional[Dict[str, float]] = None, seed: int = 20261120,
                 guidances: Sequence[float] = (1.0, 3.0), steps: int = 25, weights: Sequence[str] = ("raw", "ema"),
                 batch_size: int = 128, decode_batch: int = 64, name: Optional[str] = None):
        super().__init__(every, at, at_end, name)
        self.text, self.mask, self.decoder = text, mask, decoder
        self.thresholds = dict(thresholds) if thresholds else None
        self.seed = int(seed)
        self.guidances = tuple(float(g) for g in guidances)
        self.steps, self.weights = int(steps), tuple(weights)
        self.batch_size, self.decode_batch = int(batch_size), int(decode_batch)

    def __call__(self, ctx: HookContext) -> Dict[str, Any]:
        n = self.text.shape[0]
        out: Dict[str, Any] = {"samples": n, "thresholds": self.thresholds}
        for tag, model in models_of(ctx, self.weights):
            noise = per_item_noise(n, model.config.latent_shape, self.seed)
            for g in self.guidances:
                z = sample_latents(model, self.text, self.mask, noise, self.steps, g, ctx.device, ctx.use_amp,
                                   self.batch_size)
                m = sphere_check.measure(decode_images(self.decoder, z, self.decode_batch))
                if self.thresholds:
                    m["sphere_like"] = sphere_check.is_sphere_like(m, self.thresholds)
                out[f"{tag}_g{g:g}"] = m
        return out


# =============================================================================
# CHECKPOINT AND MILESTONE PUSHES
# =============================================================================

def _all_finite(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


class CheckpointPushHook(Hook):
    """Save the full training state (exact resume) and hand it to push(local_path, step).

    The push is retried; a failure is logged and recorded, never raised. A state with a
    non-finite parameter is not saved. Only the newest `keep_local` files stay on disk.
    """

    name = "checkpoint"

    def __init__(self, push: Optional[Callable[[str, int], Any]], local_dir: str, *, every: int = 0,
                 at: Iterable[int] = (), at_end: bool = True, keep_local: int = 1, attempts: int = 4,
                 wait_seconds: float = 10.0, extra: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(every, at, at_end, name)
        self.push, self.local_dir = push, local_dir
        self.keep_local, self.attempts, self.wait_seconds = int(keep_local), int(attempts), float(wait_seconds)
        self.extra = dict(extra or {})
        self.saved: List[str] = []

    def __call__(self, ctx: HookContext) -> Dict[str, Any]:
        if not _all_finite(ctx.model):
            raise RuntimeError("a parameter is not finite: the state is not saved")
        os.makedirs(self.local_dir, exist_ok=True)
        path = os.path.join(self.local_dir, f"{ctx.run_name}_state_step{ctx.step:06d}.pt")
        save_training_state(path, ctx.model, ctx.optimizer, ctx.ema, ctx.step, ctx.data_position,
                            ctx.train_config, extra=dict(self.extra, run=ctx.run_name))
        size = os.path.getsize(path)
        pushed = None
        if self.push is not None:
            pushed = with_retries(lambda: self.push(path, ctx.step), self.attempts, self.wait_seconds, ctx.log,
                                  what=f"checkpoint push (step {ctx.step})")
            if not pushed:
                raise RuntimeError(f"checkpoint push failed after {self.attempts} attempts (kept locally: {path})")
        self.saved.append(path)
        while len(self.saved) > max(self.keep_local, 1):
            old = self.saved.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass
        return {"file": os.path.basename(path), "bytes": size, "pushed": pushed}


class MilestonePushHook(Hook):
    """At milestone steps: raw and EMA weights (model checkpoints with their config) plus the
    files returned by files_fn(ctx), handed to push_files(paths, step); retried, never raised."""

    name = "milestone"

    def __init__(self, push_files: Optional[Callable[[List[str], int], Any]], local_dir: str,
                 milestones: Iterable[int], *, files_fn: Optional[Callable[[HookContext], List[str]]] = None,
                 attempts: int = 4, wait_seconds: float = 10.0, extra: Optional[Dict[str, Any]] = None,
                 name: Optional[str] = None):
        super().__init__(0, milestones, False, name)
        self.push_files, self.local_dir, self.files_fn = push_files, local_dir, files_fn
        self.attempts, self.wait_seconds = int(attempts), float(wait_seconds)
        self.extra = dict(extra or {})

    def __call__(self, ctx: HookContext) -> Dict[str, Any]:
        os.makedirs(self.local_dir, exist_ok=True)
        stem = os.path.join(self.local_dir, f"{ctx.run_name}_step{ctx.step:06d}")
        files = []
        save_checkpoint(stem + "_raw.pt", ctx.model, None, ctx.step,
                        extra=dict(self.extra, run=ctx.run_name, weights="raw"))
        files.append(stem + "_raw.pt")
        if ctx.ema is not None:
            save_checkpoint(stem + "_ema.pt", ctx.ema.model, None, ctx.step,
                            extra=dict(self.extra, run=ctx.run_name, weights="ema", decay=ctx.ema.decay))
            files.append(stem + "_ema.pt")
        if self.files_fn is not None:
            files += [f for f in self.files_fn(ctx) if os.path.exists(f)]
        pushed = None
        if self.push_files is not None:
            pushed = with_retries(lambda: self.push_files(files, ctx.step), self.attempts, self.wait_seconds,
                                  ctx.log, what=f"milestone push (step {ctx.step})")
            if not pushed:
                raise RuntimeError(f"milestone push failed after {self.attempts} attempts (kept locally)")
        return {"files": [os.path.basename(f) for f in files], "pushed": pushed}
