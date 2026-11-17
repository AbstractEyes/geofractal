"""
geofractal.router.prefab.agatha_v2.trainer
==========================================

Training utilities for AgathaFlow.

- build_optimizer: Adam over every trained parameter, weight decay 0.
- equalize_tower_gradients: after backward, scale each tower's gradients to the
  median tower gradient norm; returns the per-tower norms before scaling.
- save_checkpoint / load_checkpoint: one file with the config, the whole model
  state, and optionally the optimizer state and the step. Accepts a path or a
  file-like object. The model is rebuilt by config.arch (AgathaFlow or PlainFlow).
- train: a plain loop with progress, elapsed and remaining-time lines. Parameters
  stay float32; bfloat16 autocast is optional on CUDA.

For long runs:
- EMA: a float32 copy of the model, p_ema <- d p_ema + (1 - d) p after every
  optimizer step, from the first step (decay 0.9999 by default, no warm-up).
- step_generator: the noise, flow times and condition dropout of a training step
  come from a generator seeded by (seed, step) alone, so they do not depend on
  where a run was stopped and resumed.
- save_training_state / load_training_state: model, EMA, optimizer, step, RNG
  states (python, numpy, torch, CUDA when present) and the data position. A run
  resumed from such a file continues exactly as the unbroken run would (on the
  CPU the parameters are bit-identical; the tests check it).
- fit: the loop for long runs. It reads batches for steps start_step, start_step + 1,
  ... (data.StepOrderSampler gives the same batches to an unbroken and a resumed
  run), prints one progress line per log interval with time spent and time left,
  and runs hooks after the optimizer step and the EMA update (hooks.HookRunner).

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field, fields
from typing import IO, Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from geofractal.router.prefab.agatha_v2.config import AgathaConfig
from geofractal.router.prefab.agatha_v2.data import mix_seed
from geofractal.router.prefab.agatha_v2.model import AgathaFlow, build_model

CHECKPOINT_FORMAT = "agatha_v2"
NOISE_SALT = 0x6E6F697365      # separates the noise seed stream from the data-order streams


@dataclass
class TrainConfig:
    lr: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    batch_size: int = 64
    steps: int = 20000
    equalize_tower_gradients: bool = True
    bf16_autocast: bool = True           # CUDA only
    windows_memory_fraction: float = 0.73
    log_every: int = 50
    checkpoint_every: int = 2000
    out_dir: str = "agatha_v2_run"
    seed: int = 0
    num_workers: int = 0
    ema_decay: float = 0.9999            # fit() only; train() keeps no EMA

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if "betas" in kw:
            kw["betas"] = tuple(kw["betas"])
        return cls(**kw)


def build_optimizer(model: AgathaFlow, cfg: Optional[TrainConfig] = None) -> torch.optim.Adam:
    cfg = cfg or TrainConfig()
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=0.0)


@torch.no_grad()
def equalize_tower_gradients(model: AgathaFlow) -> List[float]:
    """Scale each tower's gradients to the median tower gradient norm.

    Returns the per-tower gradient norms before scaling (the gauge to log).
    Towers with no gradient are left alone. A model without towers (the plain
    control) is left alone and gives an empty list.
    """
    if len(model.towers) == 0:
        return []
    norms = []
    for tower in model.towers:
        sq = [p.grad.pow(2).sum() for p in tower.parameters() if p.grad is not None]
        norms.append(torch.stack(sq).sum().sqrt() if sq else torch.zeros(()))
    norms_t = torch.stack([n.float().cpu() for n in norms])
    median = norms_t.sort().values[len(norms) // 2]
    if float(median) < 1e-20:
        return norms_t.tolist()
    for tower, n in zip(model.towers, norms_t):
        if float(n) < 1e-20:
            continue
        scale = float(median / n)
        for p in tower.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
    return norms_t.tolist()


def save_checkpoint(f: Union[str, IO[bytes]], model: AgathaFlow,
                    optimizer: Optional[torch.optim.Optimizer] = None, step: int = 0,
                    extra: Optional[Dict] = None) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "extra": extra or {},
    }
    torch.save(payload, f)


def load_checkpoint(f: Union[str, IO[bytes]], map_location: str = "cpu",
                    optimizer_cfg: Optional[TrainConfig] = None):
    """Rebuild the model from the stored config and load every tensor.

    Returns (model, optimizer or None, step).
    """
    payload = torch.load(f, map_location=map_location, weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not an agatha_v2 checkpoint")
    model = build_model(AgathaConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    opt = None
    if payload.get("optimizer") is not None:
        opt = build_optimizer(model, optimizer_cfg)
        opt.load_state_dict(payload["optimizer"])
    return model, opt, payload.get("step", 0)


def _fmt(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def train(model: AgathaFlow, dataset: Dataset, cfg: TrainConfig,
          device: Optional[str] = None, null_text: Optional[torch.Tensor] = None) -> AgathaFlow:
    """A plain training loop (flow-matching loss, Adam wd 0, gradient equalization)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and os.name == "nt":
        torch.cuda.set_per_process_memory_fraction(cfg.windows_memory_fraction)
    torch.manual_seed(cfg.seed)
    model = model.to(device).float()
    if null_text is not None:
        model.set_null_text(null_text.to(device))
    model.train()
    opt = build_optimizer(model, cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers)
    os.makedirs(cfg.out_dir, exist_ok=True)
    use_amp = cfg.bf16_autocast and device.startswith("cuda")
    print(f"[agatha_v2] {model.num_trained_parameters():,} trained parameters; {cfg.steps} steps of "
          f"batch {cfg.batch_size}; {len(dataset):,} items")
    step, t_start = 0, time.time()
    while step < cfg.steps:
        for z1, text, mask in loader:
            z1, text, mask = z1.to(device), text.to(device), mask.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model.loss(z1, text, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gauge = equalize_tower_gradients(model) if cfg.equalize_tower_gradients else []
            opt.step()
            step += 1
            if step % cfg.log_every == 0 or step == cfg.steps:
                spent = time.time() - t_start
                left = spent / step * (cfg.steps - step)
                g = " ".join(f"{x:.3g}" for x in gauge)
                print(f"[agatha_v2] step {step}/{cfg.steps} loss {loss.detach().item():.5f} | tower grad norms {g} | "
                      f"spent {_fmt(spent)} left {_fmt(left)}")
            if step % cfg.checkpoint_every == 0 or step == cfg.steps:
                save_checkpoint(os.path.join(cfg.out_dir, f"agatha_v2_seed{cfg.seed}_step{step}.pt"),
                                model, opt, step)
            if step >= cfg.steps:
                break
    return model


# =============================================================================
# EMA
# =============================================================================

class EMA:
    """A float32 copy of a model whose parameters follow the model after every optimizer step:

        p_ema <- p_ema + (1 - decay) (p - p_ema)     (= decay p_ema + (1 - decay) p)

    The copy starts equal to the model and the decay applies from the first update (no
    warm-up). Buffers that are part of the state (for example the stored null caption) are
    copied from the model at every update. `model` is an ordinary module in eval mode, so
    every sampler and evaluation runs on it unchanged. Only parameters are cast to float32:
    buffers keep their dtype (the position codes keep float64 tables), so the copy computes
    exactly as the model does.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            if p.dtype != torch.float32:
                p.data = p.data.float()
            p.requires_grad_(False)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        w = 1.0 - self.decay
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach().to(e.dtype), w)
        for eb, b in zip(self.model.buffers(), model.buffers()):
            eb.copy_(b)
        self.num_updates += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "num_updates": self.num_updates, "model": self.model.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self.model.load_state_dict(state["model"], strict=True)


# =============================================================================
# RANDOM STATES AND THE PER-STEP GENERATOR
# =============================================================================

def capture_rng_state() -> Dict[str, Any]:
    """Global random states: python, numpy, torch (CPU) and every CUDA device when present."""
    np_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {"name": str(np_state[0]), "keys": torch.from_numpy(np.asarray(np_state[1], dtype=np.int64)),
                  "pos": int(np_state[2]), "has_gauss": int(np_state[3]), "cached_gaussian": float(np_state[4])},
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Set the global random states captured by capture_rng_state()."""
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    n = state["numpy"]
    np.random.set_state((n["name"], n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"],
                         n["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [s.cpu() for s in state["cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)


def step_generator(seed: int, step: int, device: Union[str, torch.device] = "cpu") -> torch.Generator:
    """The generator of training step `step` (0-based) of a run with this seed, on `device`."""
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(mix_seed(seed, step, NOISE_SALT))
    return g


# =============================================================================
# FULL TRAINING STATE (exact resume)
# =============================================================================

@dataclass
class TrainingState:
    model: AgathaFlow
    optimizer: torch.optim.Optimizer
    ema: Optional[EMA]
    step: int
    data_position: Dict[str, Any]
    train_config: TrainConfig
    extra: Dict[str, Any] = field(default_factory=dict)


def save_training_state(f: Union[str, IO[bytes]], model: AgathaFlow, optimizer: torch.optim.Optimizer,
                        ema: Optional[EMA], step: int, data_position: Optional[Dict[str, Any]] = None,
                        train_cfg: Optional[TrainConfig] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """Everything an exact resume needs, in one file (also readable by load_checkpoint()).

    data_position describes where the data order continues; by default {"next_step": step}.
    A path is written to a temporary name first and renamed, so a stopped write never
    leaves a damaged file under the final name.
    """
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": "training_state",
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "extra": extra or {},
        "ema": ema.state_dict() if ema is not None else None,
        "rng": capture_rng_state(),
        "data_position": dict(data_position) if data_position else {"next_step": int(step)},
        "train_config": (train_cfg or TrainConfig()).to_dict(),
    }
    if isinstance(f, (str, os.PathLike)):
        tmp = str(f) + ".partial"
        torch.save(payload, tmp)
        os.replace(tmp, f)
    else:
        torch.save(payload, f)


def load_training_state(f: Union[str, IO[bytes]], device: Union[str, torch.device] = "cpu",
                        restore_rng: bool = True) -> TrainingState:
    """Rebuild the model from its config and restore model, optimizer, EMA, step, data position
    and (by default) the global random states."""
    payload = torch.load(f, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("kind") != "training_state":
        raise ValueError("not an agatha_v2 training state")
    model = build_model(AgathaConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    # no .float() here: the parameters are float32 already, and .float() would also cast the
    # float64 position-code buffers, which changes the towers' arithmetic after a resume
    model = model.to(device)
    tc = TrainConfig.from_dict(payload.get("train_config") or {})
    opt = build_optimizer(model, tc)
    opt.load_state_dict(payload["optimizer"])
    ema = None
    if payload.get("ema") is not None:
        ema = EMA(model, payload["ema"]["decay"])
        ema.load_state_dict(payload["ema"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return TrainingState(model=model, optimizer=opt, ema=ema, step=int(payload["step"]),
                         data_position=dict(payload.get("data_position") or {}), train_config=tc,
                         extra=dict(payload.get("extra") or {}))


# =============================================================================
# THE LOOP FOR LONG RUNS
# =============================================================================

def autocast_for(device: Union[str, torch.device], enabled: bool):
    """bfloat16 autocast on a CUDA device when enabled; a disabled context otherwise."""
    kind = "cuda" if torch.device(device).type == "cuda" else "cpu"
    return torch.autocast(kind, dtype=torch.bfloat16, enabled=bool(enabled and kind == "cuda"))


@dataclass
class HookContext:
    """What a hook sees after a step: the step count is the number of optimizer steps done."""
    step: int
    end_step: int
    model: AgathaFlow
    ema: Optional[EMA]
    optimizer: Optional[torch.optim.Optimizer]
    device: str
    use_amp: bool
    run_name: str
    out_dir: str
    log: Callable[[str], None]
    data_position: Dict[str, Any]
    train_config: Optional[TrainConfig] = None
    final: bool = False


def train_step(model: AgathaFlow, optimizer: torch.optim.Optimizer, ema: Optional[EMA], z1: torch.Tensor,
               text: torch.Tensor, mask: torch.Tensor, step: int, cfg: TrainConfig, use_amp: bool = False):
    """One optimizer step (step = its 0-based index): loss under optional bfloat16 autocast with the
    step's own generator, backward, tower gradient equalization, Adam, EMA update.
    Returns (loss, per-tower gradient norms)."""
    gen = step_generator(cfg.seed, step, z1.device)
    with autocast_for(z1.device, use_amp):
        loss = model.loss(z1, text, mask, generator=gen)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gauge = equalize_tower_gradients(model) if cfg.equalize_tower_gradients else []
    optimizer.step()
    if ema is not None:
        ema.update(model)
    return loss.detach(), gauge


def fit(model: AgathaFlow, optimizer: torch.optim.Optimizer, batches: Iterable, cfg: TrainConfig, *,
        start_step: int = 0, end_step: Optional[int] = None, ema: Optional[EMA] = None,
        device: Optional[str] = None, hooks: Any = None, run_name: str = "agatha_v2",
        log: Callable[[str], None] = print, data_position: Optional[Dict[str, Any]] = None,
        stop_on_nonfinite: bool = True) -> Dict[str, Any]:
    """Train from start_step to end_step.

    `batches` yields (z1, text, mask) for steps start_step, start_step + 1, ... in order.
    After every optimizer step and EMA update, `hooks.run(ctx)` is called when hooks are
    given (a HookRunner); after the last step it is called once more with ctx.final True.
    One progress line per cfg.log_every steps: the mean loss of the interval, the tower
    gradient norms of the last step, seconds a step, time spent and time left.
    data_position is the template stored in checkpoints; "next_step" is set at each step.
    Returns a summary (steps done, last mean loss, seconds, whether it stopped early).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    end_step = int(cfg.steps if end_step is None else end_step)
    use_amp = bool(cfg.bf16_autocast and str(device).startswith("cuda"))
    model.train()
    position = dict(data_position or {})
    step = int(start_step)
    losses: List[torch.Tensor] = []
    last_mean, stopped = None, ""
    t_start = t_mark = time.time()
    mark_step = step
    gauge: List[float] = []

    def context(final: bool = False) -> HookContext:
        position["next_step"] = step
        return HookContext(step=step, end_step=end_step, model=model, ema=ema, optimizer=optimizer,
                           device=str(device), use_amp=use_amp, run_name=run_name, out_dir=cfg.out_dir,
                           log=log, data_position=dict(position), train_config=cfg, final=final)

    log(f"[{run_name}] {model.num_trained_parameters():,} trained parameters ({model.config.arch}); steps "
        f"{start_step} -> {end_step} at batch {cfg.batch_size}; lr {cfg.lr:g}; EMA "
        f"{'off' if ema is None else ema.decay}; autocast {'bf16' if use_amp else 'off'}")
    iterator = iter(batches)
    while step < end_step:
        try:
            z1, text, mask = next(iterator)
        except StopIteration:
            stopped = "the data ended before end_step"
            log(f"[{run_name}] {stopped} (step {step})")
            break
        z1 = z1.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        loss, gauge = train_step(model, optimizer, ema, z1, text, mask, step, cfg, use_amp)
        step += 1
        losses.append(loss)
        if step % cfg.log_every == 0 or step == end_step:
            last_mean = float(torch.stack(losses).float().mean())
            losses.clear()
            now = time.time()
            per_step = (now - t_mark) / max(step - mark_step, 1)
            t_mark, mark_step = now, step
            spent = now - t_start
            left = spent / max(step - start_step, 1) * (end_step - step)
            g = " ".join(f"{x:.3g}" for x in gauge) if gauge else "-"
            log(f"[{run_name}] step {step}/{end_step} loss {last_mean:.5f} | tower grad norms {g} | "
                f"{per_step:.3f} s/step | spent {_fmt(spent)} left {_fmt(left)}")
            if stop_on_nonfinite and not math.isfinite(last_mean):
                stopped = "non-finite loss"
                log(f"[{run_name}] {stopped} at step {step}: stopping; hooks are not run on this state")
        if stopped:
            break
        if hooks is not None:
            hooks.run(context(final=False))
    if hooks is not None and stopped != "non-finite loss":
        hooks.run(context(final=True))
    return {"run": run_name, "start_step": int(start_step), "step": step, "end_step": end_step,
            "last_mean_loss": last_mean, "seconds": round(time.time() - t_start, 1), "stopped": stopped}
