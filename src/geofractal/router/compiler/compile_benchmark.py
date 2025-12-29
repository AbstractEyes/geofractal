"""
geofractal.router.compiler.compile_benchmark
============================================

Benchmark comparing CompileRouter introspection/staging vs raw torch.compile
on a deliberately complex, entangled deep+wide architecture.

The test model includes:
- Multiple parallel towers (wide)
- Deep nested blocks within each tower
- Cross-tower attention bridges
- Mixed attachment styles (Sequential, ModuleList, raw attributes)
- Shared projection layers
- Skip connections across depths
- Heterogeneous module types (Linear, Conv1d, Attention, Norms, Gating)

This represents a "worst case" for compilation - maximum entanglement
to stress-test both introspection accuracy and compile performance.

Usage:
    python compiler_benchmark.py

    # Or with custom settings
    python compiler_benchmark.py --towers 8 --depth 4 --dim 512 --warmup 10 --iters 100

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import argparse
import time
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.compiler.compile_router import CompileRouter, compile_module


# =============================================================================
# BUILDING BLOCKS - Deliberately varied attachment styles
# =============================================================================

class GatedMLP(nn.Module):
    """MLP with SwiGLU-style gating. Uses Sequential."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = dim * mult
        # Sloppy: raw Sequential
        self.gate_proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
        )
        self.up_proj = nn.Linear(dim, hidden)
        self.down_proj = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class ConvMixer(nn.Module):
    """1D conv mixing. Uses ModuleList."""

    def __init__(self, dim: int, kernel: int = 7):
        super().__init__()
        # Sloppy: ModuleList with manual iteration
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim),
            nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim),
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
        ])
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, D]
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = x.transpose(1, 2)  # [B, D, L]
            x = conv(x)
            x = x.transpose(1, 2)  # [B, L, D]
            x = norm(x) + residual
        return self.proj(x)


class CrossAttention(nn.Module):
    """Cross-attention between two streams."""

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        # Sloppy: individual Linear layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        B, L, D = q.shape

        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        q = self.q_proj(q).view(B, L, self.heads, -1).transpose(1, 2)
        k = self.k_proj(kv).view(B, -1, self.heads, D // self.heads).transpose(1, 2)
        v = self.v_proj(kv).view(B, -1, self.heads, D // self.heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class SelfAttentionBlock(nn.Module):
    """Self-attention with pre-norm and residual."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x, attn_mask=mask)
        return residual + self.drop(x)


# =============================================================================
# TOWER - Deep processing unit
# =============================================================================

class EntangledBlock(nn.Module):
    """
    Single block with multiple parallel paths that merge.

    Paths:
    1. Self-attention path
    2. Gated MLP path
    3. Conv mixer path

    All paths merge via learned gating.
    """

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()

        # Three parallel paths
        self.attn_path = SelfAttentionBlock(dim, heads)
        self.mlp_path = GatedMLP(dim)
        self.conv_path = ConvMixer(dim)

        # Pre-norms for MLP and Conv paths
        self.norm_mlp = nn.LayerNorm(dim)
        self.norm_conv = nn.LayerNorm(dim)

        # Learned path gating
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid(),
        )

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        # Parallel paths
        attn_out = self.attn_path(x)
        mlp_out = x + self.mlp_path(self.norm_mlp(x))
        conv_out = x + self.conv_path(self.norm_conv(x))

        # Gated combination
        combined = torch.cat([attn_out, mlp_out, conv_out], dim=-1)
        gate = self.gate(combined)

        # Weighted merge
        merged = gate * attn_out + (1 - gate) * (mlp_out + conv_out) / 2

        return self.out_proj(merged)


class DeepTower(nn.Module):
    """
    Deep tower with skip connections across layers.

    Features:
    - N EntangledBlocks in sequence
    - Skip connections from layer i to layer i+2
    - Final pooling projection
    """

    def __init__(self, dim: int, depth: int = 4, heads: int = 8):
        super().__init__()
        self.depth = depth

        # Sloppy: mix of ModuleList and direct attributes
        self.blocks = nn.ModuleList([
            EntangledBlock(dim, heads) for _ in range(depth)
        ])

        # Skip projections for non-adjacent connections
        self.skip_projs = nn.ModuleDict({
            f'skip_{i}': nn.Linear(dim, dim)
            for i in range(depth - 2)
        })

        self.final_norm = nn.LayerNorm(dim)
        self.pool_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        skip_cache = {}

        for i, block in enumerate(self.blocks):
            # Apply skip from 2 layers ago if available
            if i >= 2 and f'skip_{i-2}' in self.skip_projs:
                skip = self.skip_projs[f'skip_{i-2}'](skip_cache[i-2])
                x = x + skip * 0.1  # Small skip weight

            # Cache for future skip
            skip_cache[i] = x

            # Process block
            x = block(x)

        x = self.final_norm(x)
        return self.pool_proj(x.mean(dim=1))  # Pool to [B, D]


# =============================================================================
# WIDE MODEL - Multiple towers with cross-connections
# =============================================================================

class EntangledWideModel(nn.Module):
    """
    Super entangled deep+wide model.

    Architecture:
    - N parallel DeepTowers
    - Cross-tower attention bridges at each depth level
    - Shared input/output projections
    - Final fusion with learned weights

    This is deliberately complex to stress-test compilation.
    """

    def __init__(
        self,
        dim: int = 256,
        num_towers: int = 4,
        tower_depth: int = 3,
        heads: int = 8,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.dim = dim
        self.num_towers = num_towers
        self.tower_depth = tower_depth

        # Shared input projection
        self.input_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Multiple deep towers
        self.towers = nn.ModuleList([
            DeepTower(dim, tower_depth, heads)
            for _ in range(num_towers)
        ])

        # Cross-tower bridges (tower i attends to tower i+1)
        self.bridges = nn.ModuleList([
            CrossAttention(dim, heads)
            for _ in range(num_towers - 1)
        ])

        # Pre-tower norms (one per tower)
        self.tower_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_towers)
        ])

        # Fusion weights (learned)
        self.fusion_weights = nn.Parameter(torch.ones(num_towers) / num_towers)

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape

        # Shared input projection
        x = self.input_proj(x)

        # Create tower-specific views with cross-attention
        tower_inputs = [self.tower_norms[i](x) for i in range(self.num_towers)]

        # Apply cross-tower bridges
        for i, bridge in enumerate(self.bridges):
            # Tower i gets information from tower i+1
            cross_info = bridge(tower_inputs[i], tower_inputs[i + 1])
            tower_inputs[i] = tower_inputs[i] + cross_info * 0.1

        # Process through deep towers
        tower_outputs = []
        for i, tower in enumerate(self.towers):
            out = tower(tower_inputs[i])  # [B, D]
            tower_outputs.append(out)

        # Fuse tower outputs with learned weights
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, tower_outputs))

        # Classify
        return self.classifier(fused)


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    warmup_time: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float  # samples/sec

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Mean: {self.mean_time_ms:.3f}ms ± {self.std_time_ms:.3f}ms\n"
            f"  Range: [{self.min_time_ms:.3f}, {self.max_time_ms:.3f}]ms\n"
            f"  Throughput: {self.throughput:.1f} samples/sec"
        )


def benchmark_forward(
    model: nn.Module,
    x: Tensor,
    name: str,
    warmup: int = 10,
    iterations: int = 100,
    sync_cuda: bool = True,
) -> BenchmarkResult:
    """Benchmark forward pass timing."""

    device = x.device

    # Warmup
    warmup_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if sync_cuda and device.type == 'cuda':
                torch.cuda.synchronize()
    warmup_time = time.perf_counter() - warmup_start

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if sync_cuda and device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(x)

            if sync_cuda and device.type == 'cuda':
                torch.cuda.synchronize()

            times.append((time.perf_counter() - t0) * 1000)  # ms

    times_tensor = torch.tensor(times)
    batch_size = x.shape[0]
    mean_time = times_tensor.mean().item()

    return BenchmarkResult(
        name=name,
        warmup_time=warmup_time,
        mean_time_ms=mean_time,
        std_time_ms=times_tensor.std().item(),
        min_time_ms=times_tensor.min().item(),
        max_time_ms=times_tensor.max().item(),
        throughput=batch_size / (mean_time / 1000),
    )


def count_parameters(model: nn.Module) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def get_memory_mb() -> float:
    """Get current CUDA memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# =============================================================================
# STANDALONE BENCHMARK FUNCTION
# =============================================================================

def is_compiled(model: nn.Module) -> bool:
    """Check if a model has already been torch.compiled."""
    # Check for dynamo compiled marker
    if hasattr(model, '_compiled_call_impl'):
        return True
    if hasattr(model, '__dict__') and '_orig_mod' in model.__dict__:
        return True
    # Check module name for OptimizedModule wrapper
    if 'OptimizedModule' in type(model).__name__:
        return True
    return False


def benchmark_model(
    model: nn.Module,
    input_tensor: Tensor,
    name: str = "model",
    warmup: int = 10,
    iterations: int = 100,
    include_fullgraph: bool = True,
    print_analysis: bool = True,
    print_tree: bool = False,
    print_stages: bool = False,
) -> Dict[str, Any]:
    """
    Benchmark a model comparing raw vs CompileRouter-organized compilation.

    This is the standalone entry point for benchmarking any PyTorch model.

    Args:
        model: Any nn.Module to benchmark
        input_tensor: Example input tensor (on correct device)
        name: Name for the model in output
        warmup: Warmup iterations before timing
        iterations: Number of timed iterations
        include_fullgraph: Include fullgraph compilation tests
        print_analysis: Print CompileRouter analysis
        print_tree: Print module tree (verbose)
        print_stages: Print execution stages (verbose)

    Returns:
        Dict with 'results', 'compile_times', 'stats', and 'comparison'

    Example:
        #>>> model = MyModel().cuda()
        #>>> x = torch.randn(32, 256, device='cuda')
        #>>> results = benchmark_model(model, x)
        #>>> print(results['comparison'])
    """

    device = input_tensor.device
    sync_cuda = device.type == 'cuda'

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    if is_compiled(model):
        import warnings
        warnings.warn(
            f"Model '{name}' appears to already be torch.compiled. "
            "Results may be inaccurate. Pass the uncompiled model for proper benchmarking.",
            UserWarning
        )

    # Ensure eval mode
    model.eval()

    # Verify forward works
    with torch.no_grad():
        try:
            output = model(input_tensor)
        except Exception as e:
            raise RuntimeError(f"Model forward failed with input shape {input_tensor.shape}: {e}")

    print("=" * 70)
    print(f"BENCHMARKING: {name}")
    print("=" * 70)

    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    params = count_parameters(model)
    print(f"Parameters: {params:,}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # ==========================================================================
    # COMPILE ROUTER ANALYSIS
    # ==========================================================================

    print(f"\n--- CompileRouter Analysis ---")

    # Create a fresh copy for organization (don't modify original)
    # We wrap the original model reference, not a copy
    compiler = CompileRouter.from_module(model, f"{name}_organized")

    t0 = time.perf_counter()
    compiler.introspect()
    introspect_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    structure = compiler.compile_towers()
    staging_time = (time.perf_counter() - t0) * 1000

    stats = compiler.get_compilation_stats()

    if print_analysis:
        print(f"Introspection: {introspect_time:.2f}ms")
        print(f"Staging: {staging_time:.2f}ms")
        print(f"Modules: {stats['total_modules']} total, {stats['leaf_modules']} leaf")
        print(f"Stages: {stats['total_stages']} ({stats['batchable_stages']} batchable)")
        print(f"Signature groups: {stats['signature_groups']}")

    if print_tree:
        print(f"\n--- Module Tree ---")
        compiler.print_tree()

    if print_stages:
        compiler.print_stages()

    # ==========================================================================
    # BENCHMARK RUNS
    # ==========================================================================

    results = {}
    compile_times = {}

    # Clear memory
    gc.collect()
    if sync_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # -------------------------------------------------------------------------
    # 1. Eager baseline
    # -------------------------------------------------------------------------
    print(f"\n--- Eager Baseline ---")
    results['eager'] = benchmark_forward(
        model, input_tensor, "Eager",
        warmup=warmup, iterations=iterations, sync_cuda=sync_cuda,
    )
    print(f"  {results['eager'].mean_time_ms:.3f}ms ± {results['eager'].std_time_ms:.3f}ms")

    # -------------------------------------------------------------------------
    # 2. torch.compile on raw model
    # -------------------------------------------------------------------------
    print(f"\n--- torch.compile (raw) ---")
    gc.collect()
    if sync_cuda:
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    compiled_raw = torch.compile(model, mode='reduce-overhead')
    with torch.no_grad():
        _ = compiled_raw(input_tensor)
        if sync_cuda:
            torch.cuda.synchronize()
    compile_times['raw'] = time.perf_counter() - t0
    print(f"  Compile: {compile_times['raw']:.2f}s")

    results['compiled_raw'] = benchmark_forward(
        compiled_raw, input_tensor, "Compiled (raw)",
        warmup=warmup, iterations=iterations, sync_cuda=sync_cuda,
    )
    print(f"  {results['compiled_raw'].mean_time_ms:.3f}ms ± {results['compiled_raw'].std_time_ms:.3f}ms")

    # -------------------------------------------------------------------------
    # 3. torch.compile on organized model
    # -------------------------------------------------------------------------
    print(f"\n--- torch.compile (organized) ---")
    gc.collect()
    if sync_cuda:
        torch.cuda.empty_cache()

    # Wrap compiler for forward
    class _StagedWrapper(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c
        def forward(self, x):
            return self.c(x)

    staged_wrapper = _StagedWrapper(compiler)
    staged_wrapper.eval()

    t0 = time.perf_counter()
    compiled_organized = torch.compile(staged_wrapper, mode='reduce-overhead')
    with torch.no_grad():
        _ = compiled_organized(input_tensor)
        if sync_cuda:
            torch.cuda.synchronize()
    compile_times['organized'] = time.perf_counter() - t0
    print(f"  Compile: {compile_times['organized']:.2f}s")

    results['compiled_organized'] = benchmark_forward(
        compiled_organized, input_tensor, "Compiled (organized)",
        warmup=warmup, iterations=iterations, sync_cuda=sync_cuda,
    )
    print(f"  {results['compiled_organized'].mean_time_ms:.3f}ms ± {results['compiled_organized'].std_time_ms:.3f}ms")

    # -------------------------------------------------------------------------
    # 4-5. Fullgraph tests (optional)
    # -------------------------------------------------------------------------
    if include_fullgraph:
        print(f"\n--- torch.compile fullgraph (raw) ---")
        gc.collect()
        if sync_cuda:
            torch.cuda.empty_cache()

        try:
            t0 = time.perf_counter()
            compiled_raw_fg = torch.compile(model, mode='reduce-overhead', fullgraph=True)
            with torch.no_grad():
                _ = compiled_raw_fg(input_tensor)
                if sync_cuda:
                    torch.cuda.synchronize()
            compile_times['raw_fullgraph'] = time.perf_counter() - t0
            print(f"  Compile: {compile_times['raw_fullgraph']:.2f}s")

            results['compiled_raw_fullgraph'] = benchmark_forward(
                compiled_raw_fg, input_tensor, "Compiled fullgraph (raw)",
                warmup=warmup, iterations=iterations, sync_cuda=sync_cuda,
            )
            print(f"  {results['compiled_raw_fullgraph'].mean_time_ms:.3f}ms")
        except Exception as e:
            print(f"  Failed: {e}")
            results['compiled_raw_fullgraph'] = None

        print(f"\n--- torch.compile fullgraph (organized) ---")
        gc.collect()
        if sync_cuda:
            torch.cuda.empty_cache()

        try:
            t0 = time.perf_counter()
            compiled_org_fg = torch.compile(staged_wrapper, mode='reduce-overhead', fullgraph=True)
            with torch.no_grad():
                _ = compiled_org_fg(input_tensor)
                if sync_cuda:
                    torch.cuda.synchronize()
            compile_times['organized_fullgraph'] = time.perf_counter() - t0
            print(f"  Compile: {compile_times['organized_fullgraph']:.2f}s")

            results['compiled_organized_fullgraph'] = benchmark_forward(
                compiled_org_fg, input_tensor, "Compiled fullgraph (organized)",
                warmup=warmup, iterations=iterations, sync_cuda=sync_cuda,
            )
            print(f"  {results['compiled_organized_fullgraph'].mean_time_ms:.3f}ms")
        except Exception as e:
            print(f"  Failed: {e}")
            results['compiled_organized_fullgraph'] = None

    # ==========================================================================
    # COMPARISON
    # ==========================================================================

    comparison = {}
    baseline_ms = results['eager'].mean_time_ms
    raw_ms = results['compiled_raw'].mean_time_ms
    org_ms = results['compiled_organized'].mean_time_ms

    comparison['eager_ms'] = baseline_ms
    comparison['raw_compiled_ms'] = raw_ms
    comparison['organized_compiled_ms'] = org_ms
    comparison['raw_speedup'] = baseline_ms / raw_ms
    comparison['organized_speedup'] = baseline_ms / org_ms
    comparison['organization_benefit_ms'] = raw_ms - org_ms
    comparison['organization_benefit_pct'] = ((raw_ms - org_ms) / raw_ms) * 100
    comparison['analysis_overhead_ms'] = introspect_time + staging_time
    comparison['raw_compile_time_s'] = compile_times['raw']
    comparison['organized_compile_time_s'] = compile_times['organized']

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Method':<35} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 57)
    print(f"{'Eager':<35} {baseline_ms:<12.3f} {'1.00x':<10}")
    print(f"{'torch.compile (raw)':<35} {raw_ms:<12.3f} {comparison['raw_speedup']:<10.2f}x")
    print(f"{'torch.compile (organized)':<35} {org_ms:<12.3f} {comparison['organized_speedup']:<10.2f}x")

    print(f"\n--- Organization Impact ---")
    if comparison['organization_benefit_ms'] > 0:
        print(f"✓ Organized is {comparison['organization_benefit_ms']:.3f}ms FASTER "
              f"({comparison['organization_benefit_pct']:.1f}% improvement)")
    else:
        print(f"✗ Raw is {-comparison['organization_benefit_ms']:.3f}ms FASTER "
              f"({-comparison['organization_benefit_pct']:.1f}% overhead from organization)")

    print(f"\nAnalysis overhead: {comparison['analysis_overhead_ms']:.2f}ms (one-time)")
    print(f"Compile time - raw: {comparison['raw_compile_time_s']:.2f}s, "
          f"organized: {comparison['organized_compile_time_s']:.2f}s")

    if sync_cuda:
        print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f} MB")

    return {
        'results': results,
        'compile_times': compile_times,
        'stats': stats,
        'comparison': comparison,
    }


# =============================================================================
# MAIN BENCHMARK (built-in test model)
# =============================================================================

def run_benchmark(args):
    """Run benchmark using the built-in EntangledWideModel."""

    print("=" * 70)
    print("COMPILE ROUTER BENCHMARK")
    print("Using built-in EntangledWideModel test case")
    print("=" * 70)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    # Model configuration
    print(f"\n--- Model Configuration ---")
    print(f"Dimension: {args.dim}")
    print(f"Towers: {args.towers}")
    print(f"Tower Depth: {args.depth}")
    print(f"Heads: {args.heads}")
    print(f"Classes: {args.classes}")
    print(f"Batch: {args.batch}")
    print(f"Seq Length: {args.seq_len}")

    # Create model
    model = EntangledWideModel(
        dim=args.dim,
        num_towers=args.towers,
        tower_depth=args.depth,
        heads=args.heads,
        num_classes=args.classes,
    )
    model = model.to(device)

    # Create input
    x = torch.randn(args.batch, args.seq_len, args.dim, device=device)

    # Run benchmark
    return benchmark_model(
        model=model,
        input_tensor=x,
        name="EntangledWideModel",
        warmup=args.warmup,
        iterations=args.iterations,
        include_fullgraph=args.fullgraph,
        print_analysis=True,
        print_tree=args.print_tree,
        print_stages=args.print_stages,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark CompileRouter organization vs raw torch.compile',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model config
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--towers', type=int, default=4, help='Number of parallel towers')
    parser.add_argument('--depth', type=int, default=3, help='Depth of each tower')
    parser.add_argument('--heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--classes', type=int, default=1000, help='Output classes')

    # Input config
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')

    # Benchmark config
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--fullgraph', action='store_true', help='Include fullgraph compilation tests')

    # Output config
    parser.add_argument('--print-tree', action='store_true', help='Print module tree')
    parser.add_argument('--print-stages', action='store_true', help='Print execution stages')

    args = parser.parse_args()

    run_benchmark(args)


if __name__ == '__main__':
    main()