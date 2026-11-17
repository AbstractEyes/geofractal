"""
geofractal.router.prefab.agatha_v2.trainer
==========================================

Training utilities for AgathaFlow.

- build_optimizer: Adam over every trained parameter, weight decay 0.
- equalize_tower_gradients: after backward, scale each tower's gradients to the
  median tower gradient norm; returns the per-tower norms before scaling.
- save_checkpoint / load_checkpoint: one file with the config, the whole model
  state, and optionally the optimizer state and the step. Accepts a path or a
  file-like object.
- train: a plain loop with progress, elapsed and remaining-time lines. Parameters
  stay float32; bfloat16 autocast is optional on CUDA.

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import IO, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from geofractal.router.prefab.agatha_v2.config import AgathaConfig
from geofractal.router.prefab.agatha_v2.model import AgathaFlow

CHECKPOINT_FORMAT = "agatha_v2"


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


def build_optimizer(model: AgathaFlow, cfg: Optional[TrainConfig] = None) -> torch.optim.Adam:
    cfg = cfg or TrainConfig()
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=0.0)


@torch.no_grad()
def equalize_tower_gradients(model: AgathaFlow) -> List[float]:
    """Scale each tower's gradients to the median tower gradient norm.

    Returns the per-tower gradient norms before scaling (the gauge to log).
    Towers with no gradient are left alone.
    """
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
    model = AgathaFlow(AgathaConfig.from_dict(payload["config"]))
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
