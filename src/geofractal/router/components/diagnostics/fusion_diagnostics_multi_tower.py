"""
fusion_barrage_harness
======================

Full multi-tower barrage harness with MNIST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import itertools
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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

    # Model
    dim: int = 256
    depth: int = 2
    num_heads: int = 4
    fingerprint_dim: int = 64

    # MNIST patch config
    patch_size: int = 4  # 28x28 -> 7x7 = 49 patches
    num_classes: int = 10

    # Training
    batch_size: int = 128
    train_steps: int = 1000
    val_every: int = 200
    val_batches: int = 50  # ~6400 val samples

    lr: float = 3e-4
    weight_decay: float = 1e-2
    aux_weight: float = 0.5  # aux task: odd/even digit

    # Diagnostics
    print_every_n_runs: int = 4
    verbose_validation: bool = True

    # Barrage space
    tower_variants: Tuple[str, ...] = (
        "full_10", "no_inversive", "inversive_only",
        "pos_only", "orthogonal_only", "posneg_pairs_only",
    )
    fusion_types: Tuple[str, ...] = ("adaptive", "gated", "sum", "concat")
    train_modes: Tuple[str, ...] = ("joint", "freeze_towers", "freeze_fusion", "heads_only")


CFG = BaseConfig()


# =============================================================================
# MNIST DATA
# =============================================================================

class MNISTPatches:
    """MNIST with patch extraction for sequence input."""

    def __init__(self, patch_size: int = 4, data_dir: str = "./data"):
        self.patch_size = patch_size
        self.seq_len = (28 // patch_size) ** 2  # 49 for patch_size=4

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        self.val_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._train_iter = None
        self._val_iter = None

    def _get_train_loader(self, batch_size: int) -> DataLoader:
        if self._train_loader is None or self._train_loader.batch_size != batch_size:
            self._train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True,
            )
            self._train_iter = None
        return self._train_loader

    def _get_val_loader(self, batch_size: int) -> DataLoader:
        if self._val_loader is None or self._val_loader.batch_size != batch_size:
            self._val_loader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=True,
            )
            self._val_iter = None
        return self._val_loader

    def _to_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Convert [B, 1, 28, 28] -> [B, seq_len, patch_size^2]"""
        B = images.shape[0]
        p = self.patch_size
        # unfold: [B, 1, 7, 7, 4, 4] for patch_size=4
        patches = images.unfold(2, p, p).unfold(3, p, p)
        # reshape to [B, 49, 16]
        patches = patches.contiguous().view(B, -1, p * p)
        return patches

    def sample_train(self, batch_size: int, device: torch.device):
        loader = self._get_train_loader(batch_size)
        if self._train_iter is None:
            self._train_iter = iter(loader)

        try:
            images, labels = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(loader)
            images, labels = next(self._train_iter)

        patches = self._to_patches(images).to(device)
        y_main = labels.to(device)
        y_aux = (labels % 2).to(device)  # odd/even

        return patches, y_main, y_aux

    def sample_val(self, batch_size: int, device: torch.device):
        loader = self._get_val_loader(batch_size)
        if self._val_iter is None:
            self._val_iter = iter(loader)

        try:
            images, labels = next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(loader)
            images, labels = next(self._val_iter)

        patches = self._to_patches(images).to(device)
        y_main = labels.to(device)
        y_aux = (labels % 2).to(device)

        return patches, y_main, y_aux

    def reset_val_iter(self):
        self._val_iter = None


# Global data handler
DATA: Optional[MNISTPatches] = None


def get_data() -> MNISTPatches:
    global DATA
    if DATA is None:
        DATA = MNISTPatches(patch_size=CFG.patch_size)
    return DATA


# =============================================================================
# TOWER SETS
# =============================================================================

def tower_configs_full_10() -> List[TowerConfig]:
    cfgs: List[TowerConfig] = []
    for geom in ["cantor", "beatrix"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
        cfgs.append(TowerConfig(f"{geom}_neg", rope=geom, address=geom, inverted=True))
    for geom in ["helix", "simplex"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
    for geom in ["golden", "fibonacci", "sinusoidal", "fractal"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
    return cfgs


def tower_configs_no_inversive() -> List[TowerConfig]:
    cfgs: List[TowerConfig] = []
    for geom in ["helix", "simplex", "golden", "fibonacci", "sinusoidal", "fractal"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
    return cfgs


def tower_configs_inversive_only() -> List[TowerConfig]:
    cfgs: List[TowerConfig] = []
    for geom in ["cantor", "beatrix", "helix", "simplex", "fractal"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
        cfgs.append(TowerConfig(f"{geom}_neg", rope=geom, address=geom, inverted=True))
    return cfgs


def tower_configs_pos_only() -> List[TowerConfig]:
    cfgs: List[TowerConfig] = []
    for geom in ["cantor", "beatrix", "helix", "simplex", "golden", "fibonacci", "sinusoidal", "fractal"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
    return cfgs


def tower_configs_orthogonal_only() -> List[TowerConfig]:
    cfgs: List[TowerConfig] = []
    for geom in ["golden", "fibonacci", "sinusoidal", "fractal"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
        cfgs.append(TowerConfig(f"{geom}_neg", rope=geom, address=geom, inverted=True))
    return cfgs


def tower_configs_posneg_pairs_only() -> List[TowerConfig]:
    cfgs: List[TowerConfig] = []
    for geom in ["cantor", "beatrix"]:
        cfgs.append(TowerConfig(f"{geom}_pos", rope=geom, address=geom, inverted=False))
        cfgs.append(TowerConfig(f"{geom}_neg", rope=geom, address=geom, inverted=True))
    return cfgs


def build_tower_variant(name: str) -> List[TowerConfig]:
    builders = {
        "full_10": tower_configs_full_10,
        "no_inversive": tower_configs_no_inversive,
        "inversive_only": tower_configs_inversive_only,
        "pos_only": tower_configs_pos_only,
        "orthogonal_only": tower_configs_orthogonal_only,
        "posneg_pairs_only": tower_configs_posneg_pairs_only,
    }
    if name not in builders:
        raise ValueError(f"Unknown tower variant: {name}")
    return builders[name]()


# =============================================================================
# PATCH EMBEDDING + HEADS
# =============================================================================

class PatchEmbedding(nn.Module):
    """Project flattened patches to model dimension."""

    def __init__(self, patch_dim: int, model_dim: int, seq_len: int):
        super().__init__()
        self.proj = nn.Linear(patch_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, model_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: [B, seq_len, patch_dim]
        B = patches.shape[0]
        x = self.proj(patches) + self.pos_embed

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x


class MultiDecisionHead(nn.Module):
    """Classification heads operating on CLS token."""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.main = nn.Linear(dim, num_classes)
        self.aux = nn.Linear(dim, 2)

    def forward(self, fused: torch.Tensor):
        # fused: [B, D] (CLS representation)
        x = self.norm(fused)
        return self.main(x), self.aux(x)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)


@torch.no_grad()
def probe_collective(collective, x: torch.Tensor) -> Dict[str, float]:
    out = collective(x)
    fused = out.fused
    opinions = out.opinions

    names = list(opinions.keys())
    op_vecs = torch.stack([opinions[n].opinion for n in names], dim=1)

    tf_cos = cosine_sim(op_vecs, fused[:, None, :]).mean().item()

    T = op_vecs.shape[1]
    sims = []
    for i in range(T):
        for j in range(i + 1, T):
            sims.append(cosine_sim(op_vecs[:, i], op_vecs[:, j]).mean())
    tt_cos = torch.stack(sims).mean().item() if sims else float("nan")

    norms = op_vecs.norm(dim=-1)
    tower_norm_mean = norms.mean().item()
    tower_norm_std = norms.std(unbiased=False).item()

    fp_var = float("nan")
    if out.collective_fingerprint is not None:
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
def run_validation(
    patch_embed: PatchEmbedding,
    collective,
    head: MultiDecisionHead,
    device: torch.device,
) -> Dict[str, float]:
    patch_embed.eval()
    collective.eval()
    head.eval()

    data = get_data()
    data.reset_val_iter()

    cm = ca = total = 0
    loss_main_accum = loss_aux_accum = 0.0

    for _ in range(CFG.val_batches):
        patches, y_main, y_aux = data.sample_val(CFG.batch_size, device)
        x = patch_embed(patches)
        out = collective(x)

        # Use CLS token (first position)
        cls_repr = out.fused[:, 0] if out.fused.dim() == 3 else out.fused
        lm, la = head(cls_repr)

        loss_main_accum += F.cross_entropy(lm, y_main).item()
        loss_aux_accum += F.cross_entropy(la, y_aux).item()

        cm += (lm.argmax(-1) == y_main).sum().item()
        ca += (la.argmax(-1) == y_aux).sum().item()
        total += y_main.numel()

    return {
        "val_main": cm / total,
        "val_aux": ca / total,
        "val_loss_main": loss_main_accum / CFG.val_batches,
        "val_loss_aux": loss_aux_accum / CFG.val_batches,
    }


def print_leaderboard(results: List["RunResult"], top_n: int = 5):
    if not results:
        return
    sorted_r = sorted(results, key=lambda r: r.best_val_main, reverse=True)[:top_n]
    print(f"\n{'─'*90}")
    print(f"  LEADERBOARD (top {min(top_n, len(sorted_r))} of {len(results)} runs)")
    print(f"{'─'*90}")
    print(f"  {'#':<3} {'digit':>6} {'odd/e':>6} {'step':>5} {'towers':<16} {'fusion':<9} {'mode':<14}")
    print(f"{'─'*90}")
    for i, r in enumerate(sorted_r, 1):
        print(f"  {i:<3} {r.best_val_main:>6.1%} {r.best_val_aux:>6.1%} {r.best_step:>5} "
              f"{r.tower_variant:<16} {r.fusion_type:<9} {r.train_mode:<14}")
    print(f"{'─'*90}\n")


# =============================================================================
# TRAIN MODES
# =============================================================================

def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = enabled


def apply_train_mode(
    patch_embed: PatchEmbedding,
    collective,
    head: nn.Module,
    mode: str,
) -> List[nn.Parameter]:
    set_requires_grad(patch_embed, True)
    set_requires_grad(collective, True)
    set_requires_grad(head, True)

    if mode == "joint":
        return (list(patch_embed.parameters()) +
                list(collective.parameters()) +
                list(head.parameters()))

    if mode == "heads_only":
        set_requires_grad(patch_embed, False)
        set_requires_grad(collective, False)
        return list(head.parameters())

    if mode == "freeze_towers":
        for name in collective.tower_names:
            set_requires_grad(collective[name], False)
        return (list(patch_embed.parameters()) +
                list(filter(lambda p: p.requires_grad, collective.parameters())) +
                list(head.parameters()))

    if mode == "freeze_fusion":
        set_requires_grad(collective["fusion"], False)
        return (list(patch_embed.parameters()) +
                list(filter(lambda p: p.requires_grad, collective.parameters())) +
                list(head.parameters()))

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
    run_idx: int,
    total_runs: int,
) -> RunResult:
    torch.manual_seed(CFG.seed)

    data = get_data()

    # Build models
    patch_dim = CFG.patch_size ** 2
    seq_len = (28 // CFG.patch_size) ** 2  # 49 patches + 1 CLS = 50

    patch_embed = PatchEmbedding(patch_dim, CFG.dim, seq_len).to(device)

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

    params = apply_train_mode(patch_embed, collective, head, train_mode)
    opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=CFG.weight_decay)

    best_val_main = best_val_aux = -1.0
    best_step = 0
    final_val = {}
    final_diag = {}

    run_tag = f"{tower_variant}|{fusion_type}|{train_mode}"

    if CFG.verbose_validation:
        tqdm.write(f"\n┌─ RUN {run_idx}/{total_runs}: {run_tag} (towers={len(configs)})")

    for step in range(1, CFG.train_steps + 1):
        patch_embed.train()
        collective.train()
        head.train()

        patches, y_main, y_aux = data.sample_train(CFG.batch_size, device)
        x = patch_embed(patches)
        out = collective(x)

        # CLS token
        cls_repr = out.fused[:, 0] if out.fused.dim() == 3 else out.fused
        lm, la = head(cls_repr)

        loss = F.cross_entropy(lm, y_main) + CFG.aux_weight * F.cross_entropy(la, y_aux)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        global_pbar.update(1)

        if step % CFG.val_every == 0 or step == CFG.train_steps:
            val = run_validation(patch_embed, collective, head, device)

            # Probe on a training batch for diagnostics
            with torch.no_grad():
                patches_diag, _, _ = data.sample_train(CFG.batch_size, device)
                x_diag = patch_embed(patches_diag)
                diag = probe_collective(collective, x_diag)

            final_val = val
            final_diag = diag

            improved = ""
            if val["val_main"] > best_val_main:
                best_val_main = val["val_main"]
                best_val_aux = val["val_aux"]
                best_step = step
                improved = " ★"

            if CFG.verbose_validation:
                tqdm.write(
                    f"│  step {step:4d} │ digit={val['val_main']:.1%} odd/e={val['val_aux']:.1%} │ "
                    f"loss={val['val_loss_main']:.3f}/{val['val_loss_aux']:.3f} │ "
                    f"cos(t→f)={diag['tower_fused_cos']:.3f} cos(t↔t)={diag['tower_pairwise_cos']:.3f}{improved}"
                )

            global_pbar.set_postfix(
                run=f"{run_idx}/{total_runs}",
                best=f"{best_val_main:.1%}",
                curr=f"{val['val_main']:.1%}",
            )

    if CFG.verbose_validation:
        tqdm.write(f"└─ BEST: step={best_step} digit={best_val_main:.1%} odd/e={best_val_aux:.1%}")

    return RunResult(
        key=f"{tower_variant}::{fusion_type}::{train_mode}",
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

    # Pre-load data
    print("Loading MNIST...")
    data = get_data()
    print(f"  Train: {len(data.train_dataset)} | Val: {len(data.val_dataset)}")
    print(f"  Patches: {data.seq_len} x {CFG.patch_size}x{CFG.patch_size} = {data.seq_len * CFG.patch_size**2} pixels")

    runs = list(itertools.product(CFG.tower_variants, CFG.fusion_types, CFG.train_modes))
    total_train_ticks = len(runs) * CFG.train_steps

    print("\n" + "=" * 100)
    print("MULTI-TOWER BARRAGE HARNESS (MNIST)")
    print("=" * 100)
    print(f"  device         : {device}")
    print(f"  runs           : {len(runs)}")
    print(f"  steps/run      : {CFG.train_steps}")
    print(f"  total steps    : {total_train_ticks}")
    print(f"  batch_size     : {CFG.batch_size}")
    print(f"  val_every      : {CFG.val_every}")
    print(f"  model_dim      : {CFG.dim}")
    print(f"  tower_variants : {CFG.tower_variants}")
    print(f"  fusion_types   : {CFG.fusion_types}")
    print(f"  train_modes    : {CFG.train_modes}")
    print("=" * 100)

    results: List[RunResult] = []
    t0 = time.time()

    with tqdm(total=total_train_ticks, desc="Barrage", smoothing=0.02, position=0) as pbar:
        for idx, (tv, ft, tm) in enumerate(runs, 1):
            r = run_one(tv, ft, tm, device, pbar, idx, len(runs))
            results.append(r)

            if idx % CFG.print_every_n_runs == 0:
                print_leaderboard(results, top_n=5)

    dt = time.time() - t0

    # Final summary
    print("\n" + "=" * 100)
    print(f"BARRAGE COMPLETE │ elapsed={dt:.1f}s ({dt/60:.1f}m) │ runs={len(results)}")
    print("=" * 100)

    results_sorted = sorted(results, key=lambda r: r.best_val_main, reverse=True)

    print("\nFINAL LEADERBOARD (top 15)")
    print("-" * 100)
    print(f"{'#':<3} {'digit':>7} {'odd/e':>7} {'step':>5} {'towers':<16} {'fusion':<9} {'mode':<14}")
    print("-" * 100)
    for i, r in enumerate(results_sorted[:15], 1):
        print(f"{i:<3} {r.best_val_main:>7.2%} {r.best_val_aux:>7.2%} {r.best_step:>5} "
              f"{r.tower_variant:<16} {r.fusion_type:<9} {r.train_mode:<14}")

    print("\nDETAILED DIAGNOSTICS (top 5)")
    print("-" * 100)
    for r in results_sorted[:5]:
        d, v = r.final_diag, r.final_val
        print(f"\n[{r.key}]")
        print(f"  best: step={r.best_step} digit={r.best_val_main:.2%} odd/e={r.best_val_aux:.2%}")
        print(f"  final: digit={r.final_val_main:.2%} odd/e={r.final_val_aux:.2%} "
              f"loss={v.get('val_loss_main', 0):.3f}/{v.get('val_loss_aux', 0):.3f}")
        print(f"  tower_cos: fused={d.get('tower_fused_cos', 0):.3f} "
              f"pairwise={d.get('tower_pairwise_cos', 0):.3f}")
        print(f"  norms: mean={d.get('tower_norm_mean', 0):.3f} std={d.get('tower_norm_std', 0):.3f}")
        print(f"  fp_var={d.get('fp_var', 0):.6f} num_towers={int(d.get('num_towers', 0))}")

    # Aggregated insights
    print("\n" + "=" * 100)
    print("AGGREGATED INSIGHTS")
    print("=" * 100)

    # Best by tower variant
    print("\nBest by tower variant:")
    for tv in CFG.tower_variants:
        subset = [r for r in results if r.tower_variant == tv]
        if subset:
            best = max(subset, key=lambda r: r.best_val_main)
            print(f"  {tv:<18} {best.best_val_main:.2%} ({best.fusion_type}/{best.train_mode})")

    # Best by fusion type
    print("\nBest by fusion type:")
    for ft in CFG.fusion_types:
        subset = [r for r in results if r.fusion_type == ft]
        if subset:
            best = max(subset, key=lambda r: r.best_val_main)
            print(f"  {ft:<10} {best.best_val_main:.2%} ({best.tower_variant}/{best.train_mode})")

    # Best by train mode
    print("\nBest by train mode:")
    for tm in CFG.train_modes:
        subset = [r for r in results if r.train_mode == tm]
        if subset:
            best = max(subset, key=lambda r: r.best_val_main)
            avg = sum(r.best_val_main for r in subset) / len(subset)
            print(f"  {tm:<14} best={best.best_val_main:.2%} avg={avg:.2%}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()