"""
fusion_barrage_harness
======================

Full multi-tower barrage harness.

Runs a matrix of experiments across:
- Tower configuration variants (ablation sets)
- Fusion types
- Training modes
- Validation-driven reporting
- Deep diagnostics (tower agreement, fingerprint variance)

Single progress bar only:
- One tqdm over TOTAL training steps across ALL runs.

Uses real router structure:
- build_tower_collective() from geometric_tower_builder
- ConfigurableCollective / WideRouter execution
- Fingerprints preserved and used

NOTE:
- Data generator is the current structured synthetic tokens generator.
  Swap in your 10-shape renderer later by replacing sample_tokens().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import itertools
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from geofractal.router.prefab.geometric_tower_builder import (
    TowerConfig,
    build_tower_collective,
)

# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class BaseConfig:
    device: str = "cuda"
    seed: int = 42

    # collective / tower
    dim: int = 256
    depth: int = 2
    num_heads: int = 4
    fingerprint_dim: int = 64

    # data
    batch_size: int = 32
    seq_len: int = 256
    num_classes: int = 10

    # training
    train_steps: int = 1000
    val_every: int = 200
    val_batches: int = 50

    lr: float = 3e-4
    weight_decay: float = 1e-2

    # loss weights
    aux_weight: float = 0.5

    # barrage space
    tower_variants: Tuple[str, ...] = (
        "full_10",
        "no_inversive",
        "inversive_only",
        "pos_only",
        "orthogonal_only",
        "posneg_pairs_only",
    )
    fusion_types: Tuple[str, ...] = ("adaptive", "gated", "sum", "concat")
    train_modes: Tuple[str, ...] = ("joint", "freeze_towers", "freeze_fusion", "heads_only")


CFG = BaseConfig()


# =============================================================================
# TOWER SETS
# =============================================================================

def tower_configs_full_10() -> List[TowerConfig]:
    cfgs: List[TowerConfig] = []

    # 2 inversive pairs
    for geom in ["cantor", "beatrix"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
        cfgs.append(TowerConfig(f"{geom}_neg", rope=geom, address=geom, inverted=True))

    # 2 positive-only
    for geom in ["helix", "simplex"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))

    # 4 orthogonal
    for geom in ["golden", "fibonacci", "sinusoidal", "fractal"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))

    assert len(cfgs) == 10
    return cfgs


def tower_configs_no_inversive() -> List[TowerConfig]:
    # remove cantor/beatrix posneg pairs; keep pos-only + orthogonal
    cfgs: List[TowerConfig] = []
    for geom in ["helix", "simplex", "golden", "fibonacci", "sinusoidal", "fractal"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
    return cfgs


def tower_configs_inversive_only() -> List[TowerConfig]:
    cfgs: List[TowerConfig] = []
    for geom in ["cantor", "beatrix", "helix", "simplex", "fractal"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
        cfgs.append(TowerConfig(f"{geom}_neg", rope=geom, address=geom, inverted=True))
    # 5 pairs = 10 towers
    assert len(cfgs) == 10
    return cfgs


def tower_configs_pos_only() -> List[TowerConfig]:
    # drop all neg towers from full_10
    cfgs: List[TowerConfig] = []
    for geom in ["cantor", "beatrix", "helix", "simplex", "golden", "fibonacci", "sinusoidal", "fractal"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
    return cfgs


def tower_configs_orthogonal_only() -> List[TowerConfig]:
    cfgs: List[TowerConfig] = []
    for geom in ["golden", "fibonacci", "sinusoidal", "fractal"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
        cfgs.append(TowerConfig(f"{geom}_neg", rope=geom, address=geom, inverted=True))
    # 4 pairs = 8 towers
    return cfgs


def tower_configs_posneg_pairs_only() -> List[TowerConfig]:
    # two pos/neg pairs only (as requested category)
    cfgs: List[TowerConfig] = []
    for geom in ["cantor", "beatrix"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
        cfgs.append(TowerConfig(f"{geom}_neg", rope=geom, address=geom, inverted=True))
    return cfgs


def build_tower_variant(name: str) -> List[TowerConfig]:
    if name == "full_10":
        return tower_configs_full_10()
    if name == "no_inversive":
        return tower_configs_no_inversive()
    if name == "inversive_only":
        return tower_configs_inversive_only()
    if name == "pos_only":
        return tower_configs_pos_only()
    if name == "orthogonal_only":
        return tower_configs_orthogonal_only()
    if name == "posneg_pairs_only":
        return tower_configs_posneg_pairs_only()
    raise ValueError(f"Unknown tower variant: {name}")


# =============================================================================
# DATA
# =============================================================================

def sample_tokens(batch_size: int, seq_len: int, dim: int, num_classes: int, device: torch.device):
    """
    Structured synthetic tokens.
    Replace this with your 10-shape renderer later.
    """
    y_main = torch.randint(0, num_classes, (batch_size,), device=device)
    y_aux = (y_main >= num_classes // 2).long()

    class_means = torch.randn(num_classes, dim, device=device)
    x = class_means[y_main][:, None, :].expand(-1, seq_len, -1)
    x = x + 0.05 * torch.randn_like(x)
    return x, y_main, y_aux


def sample_tokens_val(batch_size: int, seq_len: int, dim: int, num_classes: int, device: torch.device):
    # stable validation stream
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(123456)
        return sample_tokens(batch_size, seq_len, dim, num_classes, device)


# =============================================================================
# HEADS
# =============================================================================

class MultiDecisionHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.main = nn.Linear(dim, num_classes)
        self.aux = nn.Linear(dim, 2)

    def forward(self, fused: torch.Tensor):
        return self.main(fused), self.aux(fused)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)


@torch.no_grad()
def probe_collective(collective, x: torch.Tensor) -> Dict[str, float]:
    """
    Diagnostics:
      - tower_fused_cos: mean cosine(opinion_i, fused)
      - tower_pairwise_cos: mean cosine(opinion_i, opinion_j) for i<j
      - fp_var: mean var of collective_fp
      - tower_norm_mean/std: mean/std of tower opinion norms
    """
    out = collective(x)
    fused = out.fused
    opinions = out.opinions

    names = list(opinions.keys())
    op_vecs = torch.stack([opinions[n].opinion for n in names], dim=1)  # [B,T,D]

    # tower -> fused cosine
    tf_cos = cosine_sim(op_vecs, fused[:, None, :]).mean().item()

    # tower-tower cosine
    T = op_vecs.shape[1]
    sims = []
    for i in range(T):
        for j in range(i + 1, T):
            sims.append(cosine_sim(op_vecs[:, i], op_vecs[:, j]).mean())
    tt_cos = torch.stack(sims).mean().item() if sims else float("nan")

    # opinion norms
    norms = op_vecs.norm(dim=-1)  # [B,T]
    tower_norm_mean = norms.mean().item()
    tower_norm_std = norms.std(unbiased=False).item()

    # fingerprint var
    if out.collective_fingerprint is None:
        fp_var = float("nan")
    else:
        fp_var = out.collective_fingerprint.var(dim=0, unbiased=False).mean().item()

    return {
        "tower_fused_cos": tf_cos,
        "tower_pairwise_cos": tt_cos,
        "tower_norm_mean": tower_norm_mean,
        "tower_norm_std": tower_norm_std,
        "fp_var": fp_var,
        "num_towers": float(T),
    }


@torch.no_grad()
def run_validation(collective, head, device: torch.device) -> Dict[str, float]:
    collective.eval()
    head.eval()

    cm = ca = total = 0
    loss_main_accum = 0.0
    loss_aux_accum = 0.0

    for _ in range(CFG.val_batches):
        x, y_main, y_aux = sample_tokens_val(
            CFG.batch_size, CFG.seq_len, CFG.dim, CFG.num_classes, device
        )
        out = collective(x)
        fused = out.fused
        lm, la = head(fused)

        loss_main_accum += float(F.cross_entropy(lm, y_main).item())
        loss_aux_accum += float(F.cross_entropy(la, y_aux).item())

        cm += (lm.argmax(-1) == y_main).sum().item()
        ca += (la.argmax(-1) == y_aux).sum().item()
        total += y_main.numel()

    return {
        "val_main": cm / total,
        "val_aux": ca / total,
        "val_loss_main": loss_main_accum / CFG.val_batches,
        "val_loss_aux": loss_aux_accum / CFG.val_batches,
    }


# =============================================================================
# TRAIN MODES
# =============================================================================

def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = enabled


def apply_train_mode(collective, head: nn.Module, mode: str) -> List[nn.Parameter]:
    """
    Returns the parameter list to optimize under the given mode.
    """
    # default: everything trainable
    set_requires_grad(collective, True)
    set_requires_grad(head, True)

    if mode == "joint":
        return list(collective.parameters()) + list(head.parameters())

    if mode == "heads_only":
        set_requires_grad(collective, False)
        return list(head.parameters())

    if mode == "freeze_towers":
        # freeze each tower module; keep fusion + projections + fp_proj trainable
        for name in collective.tower_names:
            set_requires_grad(collective[name], False)
        # keep fusion/input_proj/fp_proj trainable + head trainable
        return list(filter(lambda p: p.requires_grad, collective.parameters())) + list(head.parameters())

    if mode == "freeze_fusion":
        # freeze fusion module only; allow towers to adapt
        set_requires_grad(collective["fusion"], False)
        return list(filter(lambda p: p.requires_grad, collective.parameters())) + list(head.parameters())

    raise ValueError(f"Unknown train mode: {mode}")


# =============================================================================
# SINGLE RUN
# =============================================================================

@dataclass
class RunResult:
    key: str
    tower_variant: str
    fusion_type: str
    train_mode: str

    best_val_main: float
    best_val_aux: float
    final_val_main: float
    final_val_aux: float

    best_step: int
    final_diag: Dict[str, float]
    final_val: Dict[str, float]


def run_one(
    tower_variant: str,
    fusion_type: str,
    train_mode: str,
    device: torch.device,
    global_pbar: tqdm,
) -> RunResult:
    torch.manual_seed(CFG.seed)

    configs = build_tower_variant(tower_variant)
    collective = build_tower_collective(
        configs=configs,
        dim=CFG.dim,
        default_depth=CFG.depth,
        num_heads=CFG.num_heads,
        fingerprint_dim=CFG.fingerprint_dim,
        fusion_type=fusion_type,
        name=f"collective_{tower_variant}_{fusion_type}_{train_mode}",
    )
    collective.network_to(device=device)

    head = MultiDecisionHead(CFG.dim, CFG.num_classes).to(device)

    params = apply_train_mode(collective, head, train_mode)

    opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=CFG.weight_decay)

    best_val_main = -1.0
    best_val_aux = -1.0
    best_step = 0

    final_val = {}
    final_diag = {}

    for step in range(1, CFG.train_steps + 1):
        collective.train()
        head.train()

        x, y_main, y_aux = sample_tokens(
            CFG.batch_size, CFG.seq_len, CFG.dim, CFG.num_classes, device
        )

        out = collective(x)
        fused = out.fused
        lm, la = head(fused)

        loss = F.cross_entropy(lm, y_main) + CFG.aux_weight * F.cross_entropy(la, y_aux)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # advance ONE global bar tick per train step
        global_pbar.update(1)

        if step % CFG.val_every == 0 or step == CFG.train_steps:
            val = run_validation(collective, head, device)
            diag = probe_collective(collective, x)

            final_val = val
            final_diag = diag

            if val["val_main"] > best_val_main:
                best_val_main = val["val_main"]
                best_val_aux = val["val_aux"]
                best_step = step

            # update global bar postfix sparingly (single bar)
            global_pbar.set_postfix(
                run=f"{tower_variant}|{fusion_type}|{train_mode}",
                vmain=f"{val['val_main']:.3f}",
                vaux=f"{val['val_aux']:.3f}",
            )

    key = f"{tower_variant}::{fusion_type}::{train_mode}"
    return RunResult(
        key=key,
        tower_variant=tower_variant,
        fusion_type=fusion_type,
        train_mode=train_mode,
        best_val_main=best_val_main,
        best_val_aux=best_val_aux,
        final_val_main=final_val.get("val_main", float("nan")),
        final_val_aux=final_val.get("val_aux", float("nan")),
        best_step=best_step,
        final_diag=final_diag,
        final_val=final_val,
    )


# =============================================================================
# BARRAGE HARNESS
# =============================================================================

def main():
    device = torch.device(CFG.device if torch.cuda.is_available() else "cpu")

    runs = list(itertools.product(CFG.tower_variants, CFG.fusion_types, CFG.train_modes))
    total_train_ticks = len(runs) * CFG.train_steps

    print("\n" + "=" * 110)
    print("MULTI-TOWER BARRAGE HARNESS")
    print(f"device={device} | runs={len(runs)} | train_steps/run={CFG.train_steps} | total_train_steps={total_train_ticks}")
    print(f"tower_variants={CFG.tower_variants}")
    print(f"fusion_types={CFG.fusion_types}")
    print(f"train_modes={CFG.train_modes}")
    print("=" * 110)

    results: List[RunResult] = []
    t0 = time.time()

    with tqdm(total=total_train_ticks, desc="Barrage", smoothing=0.02) as pbar:
        for (tv, ft, tm) in runs:
            r = run_one(tv, ft, tm, device, pbar)
            results.append(r)

    dt = time.time() - t0

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("BARRAGE COMPLETE")
    print(f"elapsed={dt:.1f}s | runs={len(results)}")
    print("=" * 110)

    # Sort by best main accuracy (descending)
    results_sorted = sorted(results, key=lambda r: r.best_val_main, reverse=True)

    # Print top 15
    print("\nTOP RESULTS (sorted by best_val_main)")
    print("-" * 110)
    print("best_main  best_aux  best_step  final_main final_aux  towers           fusion    mode")
    print("-" * 110)
    for r in results_sorted[:15]:
        print(
            f"{r.best_val_main:8.3f}  {r.best_val_aux:8.3f}  {r.best_step:9d}  "
            f"{r.final_val_main:9.3f} {r.final_val_aux:8.3f}  "
            f"{r.tower_variant:15s} {r.fusion_type:8s} {r.train_mode}"
        )

    # Detailed per-run diagnostics (final) for top 5
    print("\nDETAILED DIAGNOSTICS (top 5 runs)")
    print("-" * 110)
    for r in results_sorted[:5]:
        d = r.final_diag
        v = r.final_val
        print(f"\n[{r.key}]")
        print(f"  best_step={r.best_step} best_val_main={r.best_val_main:.3f} best_val_aux={r.best_val_aux:.3f}")
        print(f"  final_val_main={r.final_val_main:.3f} final_val_aux={r.final_val_aux:.3f}")
        print(f"  val_loss_main={v.get('val_loss_main', float('nan')):.3f} val_loss_aux={v.get('val_loss_aux', float('nan')):.3f}")
        print(f"  tower_fused_cos={d.get('tower_fused_cos', float('nan')):.3f}")
        print(f"  tower_pairwise_cos={d.get('tower_pairwise_cos', float('nan')):.3f}")
        print(f"  tower_norm_mean={d.get('tower_norm_mean', float('nan')):.3f} tower_norm_std={d.get('tower_norm_std', float('nan')):.3f}")
        print(f"  fp_var={d.get('fp_var', float('nan')):.6f}")
        print(f"  num_towers={int(d.get('num_towers', 0))}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
