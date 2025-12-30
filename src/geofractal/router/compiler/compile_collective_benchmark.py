"""
Geometric Tower Profiler
========================

Deep profiling for identifying bottlenecks in tower architectures.

Profiles:
- Forward pass breakdown (patch_embed, collective, classifier)
- Backward pass breakdown
- Inside collective (tower execution, fusion, attention)
- Memory allocation
- CUDA kernel timing
- Per-layer timing within towers

Usage:
    python profile_towers.py
"""

import time
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
from collections import defaultdict
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader, TensorDataset

from geofractal.router.prefab.collective_builder import (
    HealthyCollective,
    CollectiveConfig,
    quick_experiment,
    quick_isolated_experiment,
)
from geofractal.router.prefab.geometric_tower_builder import (
    quick_collective,
    quick_pair,
    ConfigurableCollective,
    FusionType,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# TIMING UTILITIES
# =============================================================================

class CUDATimer:
    """High-precision CUDA timing using events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self) -> float:
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)  # ms


class TimingAccumulator:
    """Accumulate timing statistics."""

    def __init__(self):
        self.times: Dict[str, List[float]] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)

    def add(self, name: str, time_ms: float):
        self.times[name].append(time_ms)
        self.counts[name] += 1

    def stats(self, name: str) -> Dict[str, float]:
        if name not in self.times or len(self.times[name]) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'total': 0, 'count': 0}

        t = self.times[name]
        return {
            'mean': statistics.mean(t),
            'std': statistics.stdev(t) if len(t) > 1 else 0,
            'min': min(t),
            'max': max(t),
            'total': sum(t),
            'count': len(t),
        }

    def summary(self) -> str:
        lines = []
        total_time = sum(sum(t) for t in self.times.values())

        # Sort by total time descending
        sorted_names = sorted(self.times.keys(), key=lambda n: sum(self.times[n]), reverse=True)

        lines.append("=" * 80)
        lines.append(f"{'Component':<40} {'Mean':>10} {'Std':>10} {'Total':>10} {'%':>8}")
        lines.append("=" * 80)

        for name in sorted_names:
            s = self.stats(name)
            pct = 100 * s['total'] / total_time if total_time > 0 else 0
            lines.append(
                f"{name:<40} {s['mean']:>9.2f}ms {s['std']:>9.2f}ms {s['total']:>9.1f}ms {pct:>7.1f}%"
            )

        lines.append("=" * 80)
        lines.append(f"{'TOTAL':<40} {'':<10} {'':<10} {total_time:>9.1f}ms {100:>7.1f}%")

        return '\n'.join(lines)


@contextmanager
def cuda_timer(accumulator: TimingAccumulator, name: str):
    """Context manager for CUDA timing."""
    timer = CUDATimer()
    timer.start()
    yield
    elapsed = timer.stop()
    accumulator.add(name, elapsed)


# =============================================================================
# HOOKED MODEL FOR DEEP PROFILING
# =============================================================================

class ProfiledModule(nn.Module):
    """Wrapper that profiles a module's forward pass."""

    def __init__(self, module: nn.Module, name: str, accumulator: TimingAccumulator):
        super().__init__()
        self.module = module
        self.name = name
        self.accumulator = accumulator

    def forward(self, *args, **kwargs):
        with cuda_timer(self.accumulator, self.name):
            return self.module(*args, **kwargs)


class TowerProfiler:
    """
    Deep profiler for geometric tower architectures.

    Hooks into model internals to time:
    - Overall forward/backward
    - Each major component
    - Each tower individually
    - Fusion operations
    - Attention operations
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.accumulator = TimingAccumulator()
        self.hooks = []
        self.memory_snapshots = []

    def _get_base_model(self):
        """Unwrap compiled model if needed."""
        return self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

    def _get_collective(self):
        """Get the collective from model."""
        base = self._get_base_model()
        if hasattr(base, 'collective'):
            return base.collective
        return None

    def profile_forward(self, x: Tensor) -> Tensor:
        """Profile a single forward pass."""
        base = self._get_base_model()

        # Overall forward
        with cuda_timer(self.accumulator, "TOTAL_FORWARD"):

            # Patch embedding
            if hasattr(base, 'patch_embed'):
                with cuda_timer(self.accumulator, "patch_embed"):
                    x_embed = base.patch_embed(x)
            else:
                x_embed = x

            # Collective - just time the whole thing
            if hasattr(base, 'collective'):
                with cuda_timer(self.accumulator, "collective"):
                    result = base.collective(x_embed)

                # Extract fused output
                if hasattr(result, 'fused'):
                    fused = result.fused
                else:
                    fused = result
            else:
                fused = x_embed

            # Classifier
            if hasattr(base, 'classifier'):
                cls_out = fused[:, 0] if fused.dim() == 3 else fused
                with cuda_timer(self.accumulator, "classifier"):
                    logits = base.classifier(cls_out)
            else:
                logits = fused

        return logits

    def profile_backward(self, loss: Tensor):
        """Profile backward pass."""
        with cuda_timer(self.accumulator, "TOTAL_BACKWARD"):
            loss.backward()

    def profile_step(
        self,
        x: Tensor,
        targets: Tensor,
        optimizer: torch.optim.Optimizer = None,
    ) -> Dict[str, Any]:
        """Profile a complete training step."""

        # Forward
        logits = self.profile_forward(x)

        # Loss
        with cuda_timer(self.accumulator, "loss_computation"):
            loss = F.cross_entropy(logits, targets)

        # Backward
        self.profile_backward(loss)

        # Optimizer step
        if optimizer is not None:
            with cuda_timer(self.accumulator, "optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()

        return {
            'loss': loss.item(),
            'logits_shape': logits.shape,
        }

    def memory_snapshot(self, label: str = ""):
        """Capture memory state."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            self.memory_snapshots.append({
                'label': label,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
            })

    def reset(self):
        """Reset timing data."""
        self.accumulator = TimingAccumulator()
        self.memory_snapshots = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def report(self) -> str:
        """Generate timing report."""
        lines = ["\n" + "=" * 80]
        lines.append("  TOWER PROFILING REPORT")
        lines.append("=" * 80)

        lines.append("\n" + self.accumulator.summary())

        # Memory report
        if self.memory_snapshots:
            lines.append("\n" + "-" * 80)
            lines.append("  MEMORY USAGE")
            lines.append("-" * 80)
            for snap in self.memory_snapshots:
                lines.append(
                    f"  {snap['label']:<30} "
                    f"Allocated: {snap['allocated_gb']:.2f} GB  "
                    f"Reserved: {snap['reserved_gb']:.2f} GB  "
                    f"Peak: {snap['max_allocated_gb']:.2f} GB"
                )

        return '\n'.join(lines)


# =============================================================================
# PYTORCH PROFILER INTEGRATION
# =============================================================================

def run_torch_profiler(
    model: nn.Module,
    x: Tensor,
    targets: Tensor,
    num_steps: int = 5,
    export_chrome: bool = False,
) -> str:
    """Run PyTorch profiler for detailed kernel analysis."""

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_steps):
            with record_function("forward"):
                logits = model(x)

            with record_function("loss"):
                loss = F.cross_entropy(logits, targets)

            with record_function("backward"):
                loss.backward()

    # Get summary
    summary = prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=30
    )

    if export_chrome:
        prof.export_chrome_trace("trace.json")
        print("Exported Chrome trace to trace.json")

    return summary


# =============================================================================
# BENCHMARK CONFIG
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    # Model
    dim: int = 512
    depth: int = 8
    num_heads: int = 32
    geometries: tuple = (('fibonacci', 1),)

    # Input
    batch_size: int = 32
    seq_len: int = 65  # 64 patches + CLS
    image_size: int = 32
    patch_size: int = 4
    num_classes: int = 100

    # Profiling
    warmup_steps: int = 3
    profile_steps: int = 10

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch_size * self.patch_size

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


# =============================================================================
# MODEL CREATION
# =============================================================================

class PatchEmbedding(nn.Module):
    """Patch embedding for profiling."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Linear(config.patch_dim, config.dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, config.num_patches + 1, config.dim) * 0.02)
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        p = self.patch_size

        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * p * p)

        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        return self.norm(x)


class BenchmarkModel(nn.Module):
    """Model for benchmarking."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)

        self.collective = quick_collective(
            geometries=list(config.geometries),
            dim=config.dim,
            depth=config.depth,
            num_heads=config.num_heads,
            fingerprint_dim=64,
            fusion_type='walker_inception',
            name='benchmark_collective',
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        result = self.collective(x)
        fused = result.fused if hasattr(result, 'fused') else result
        cls_out = fused[:, 0] if fused.dim() == 3 else fused
        return self.classifier(cls_out)


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(config: BenchmarkConfig = None):
    """Run comprehensive benchmark."""

    if config is None:
        config = BenchmarkConfig()

    print("\n" + "=" * 80)
    print("  GEOMETRIC TOWER PROFILER")
    print("=" * 80)
    print(f"\nConfig:")
    print(f"  dim={config.dim}, depth={config.depth}, heads={config.num_heads}")
    print(f"  geometries={config.geometries}")
    print(f"  batch_size={config.batch_size}, seq_len={config.seq_len}")
    print(f"  warmup={config.warmup_steps}, profile={config.profile_steps}")

    # Create model
    print("\nCreating model...")
    model = BenchmarkModel(config).to(device)
    model.collective.prepare_for_compile()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Create profiler
    profiler = TowerProfiler(model)

    # Create dummy data
    print("\nCreating data...")
    x = torch.randn(config.batch_size, 3, config.image_size, config.image_size, device=device)
    targets = torch.randint(0, config.num_classes, (config.batch_size,), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Memory baseline
    profiler.memory_snapshot("before_warmup")

    # Warmup
    print(f"\nWarmup ({config.warmup_steps} steps)...")
    model.train()
    for i in range(config.warmup_steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        print(f"  Step {i+1}: loss={loss.item():.4f}")

    torch.cuda.synchronize()
    profiler.memory_snapshot("after_warmup")

    # Profile forward/backward breakdown
    print(f"\nProfiling ({config.profile_steps} steps)...")
    profiler.reset()
    profiler.memory_snapshot("profile_start")

    model.train()
    for i in range(config.profile_steps):
        optimizer.zero_grad()
        result = profiler.profile_step(x, targets, optimizer=None)
        optimizer.step()

        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}/{config.profile_steps}")

    profiler.memory_snapshot("profile_end")

    # Print report
    print(profiler.report())

    # PyTorch profiler for kernel-level detail
    print("\n" + "=" * 80)
    print("  PYTORCH PROFILER (Kernel-level)")
    print("=" * 80)

    model.train()
    summary = run_torch_profiler(model, x, targets, num_steps=3)
    print(summary)

    # Individual tower timing
    print("\n" + "=" * 80)
    print("  PER-TOWER BREAKDOWN")
    print("=" * 80)

    collective = profiler._get_collective()
    if collective is not None:
        print(f"\n  Towers: {collective.tower_names}")

        # Get embedded input
        x_embed = model.patch_embed(x)

        # Input projection if exists
        if 'input_proj' in collective.components:
            x_embed = collective['input_proj'](x_embed)

        # Time each tower individually
        tower_times = {}
        for name in collective.tower_names:
            tower = collective[name]

            # Warmup
            for _ in range(3):
                _ = tower(x_embed)

            torch.cuda.synchronize()

            # Time
            timer = CUDATimer()
            times = []
            for _ in range(10):
                timer.start()
                _ = tower(x_embed)
                times.append(timer.stop())

            tower_times[name] = {
                'mean': statistics.mean(times),
                'std': statistics.stdev(times),
            }

        # Print tower times
        print(f"\n  {'Tower':<30} {'Mean':>12} {'Std':>12}")
        print("  " + "-" * 56)

        total = sum(t['mean'] for t in tower_times.values())
        for name, times in sorted(tower_times.items(), key=lambda x: -x[1]['mean']):
            pct = 100 * times['mean'] / total if total > 0 else 0
            print(f"  {name:<30} {times['mean']:>10.2f}ms {times['std']:>10.2f}ms ({pct:.1f}%)")

        print("  " + "-" * 56)
        print(f"  {'TOTAL':<30} {total:>10.2f}ms")

    return profiler


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Default benchmark
    config = BenchmarkConfig(
        dim=512,
        depth=8,
        num_heads=32,
        geometries=(('fibonacci', 1),),
        batch_size=32,
        warmup_steps=3,
        profile_steps=10,
    )

    profiler = run_benchmark(config)

    # Additional quick comparisons
    print("\n" + "=" * 80)
    print("  QUICK COMPARISONS")
    print("=" * 80)

    # Compare different depths
    for depth in [2, 4, 8]:
        cfg = BenchmarkConfig(dim=256, depth=depth, num_heads=16, batch_size=64,
                              warmup_steps=2, profile_steps=5)
        print(f"\n--- depth={depth} ---")
        run_benchmark(cfg)