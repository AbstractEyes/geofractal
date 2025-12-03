"""
global_fractal_router_benchmark_v2.py - Performance analysis for GlobalFractalRouter

Benchmarks:
1. Component-level timing (identify bottlenecks)
2. Scaling behavior (batch, sequence, routes)
3. Memory profiling
4. Comparison vs baseline wormhole router
5. For-loop hotspot analysis (verifies optimizations)

Author: AbstractPhil
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import math

from geofractal.model.blocks.router.global_fractal_router import (
    GlobalFractalRouter,
    GlobalFractalRouterConfig,
    get_registry,
    FractalRouterNetwork,
    get_primes,
)
IMPORT_SUCCESS = True


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@contextmanager
def cuda_timer(name: str, results: Dict[str, List[float]], sync: bool = True):
    """Context manager for CUDA-aware timing."""
    if torch.cuda.is_available() and sync:
        torch.cuda.synchronize()

    start = time.perf_counter()
    yield

    if torch.cuda.is_available() and sync:
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) * 1000  # ms

    if name not in results:
        results[name] = []
    results[name].append(elapsed)


def get_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# BASELINE ROUTER (for comparison)
# =============================================================================

class BaselineWormholeRouter(nn.Module):
    """Minimal wormhole router without fingerprinting overhead."""

    def __init__(self, dim: int, num_positions: int, num_routes: int = 8, temperature: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_positions = num_positions
        self.num_routes = min(num_routes, num_positions - 1)
        self.temperature = temperature

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, skip_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if skip_first:
            x = x[:, 1:, :]

        B, P, D = x.shape

        q = F.normalize(self.query_proj(x), dim=-1)
        k = F.normalize(self.key_proj(x), dim=-1)
        v = self.value_proj(x)

        scores = torch.bmm(q, k.transpose(1, 2))

        mask = torch.eye(P, device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(mask.unsqueeze(0), -1e9)

        topk_scores, routes = torch.topk(scores / self.temperature, self.num_routes, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        K = self.num_routes
        routes_flat = routes.reshape(B, P * K).unsqueeze(-1).expand(-1, -1, D)
        v_gathered = torch.gather(v, 1, routes_flat).view(B, P, K, D)

        features = torch.einsum('bpk,bpkd->bpd', weights, v_gathered)

        return routes, weights, features


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    seq_lengths: List[int] = field(default_factory=lambda: [65, 129, 257, 513])
    feature_dims: List[int] = field(default_factory=lambda: [256, 512])
    num_routes: List[int] = field(default_factory=lambda: [4, 8, 16])

    warmup_iterations: int = 3
    benchmark_iterations: int = 10

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    run_component_breakdown: bool = True
    run_scaling_analysis: bool = True
    run_memory_profile: bool = True
    run_baseline_comparison: bool = True
    run_loop_analysis: bool = True
    run_network_benchmark: bool = True
    run_optimization_verification: bool = True


# =============================================================================
# LOOP ANALYSIS (with optimization verification)
# =============================================================================

class LoopAnalysisBenchmark:
    """Analyzes loop hotspots and verifies optimizations."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, Dict] = {}

    def run(self) -> Dict[str, Dict]:
        """Analyze all known loop locations."""
        device = self.config.device

        print("\n  Analyzing loop hotspots...")

        self._benchmark_prime_generation()
        self._benchmark_basis_construction()
        self._benchmark_potential_fields(device)
        self._benchmark_mailbox_read(device)
        self._benchmark_local_mask()

        if IMPORT_SUCCESS:
            self._verify_optimizations(device)

        return self.results

    def _benchmark_prime_generation(self):
        """Benchmark prime number generation."""
        print("\n    Prime generation:")

        # Original slow version
        def generate_primes_slow(n: int) -> List[int]:
            primes = []
            candidate = 2
            while len(primes) < n:
                is_prime = all(candidate % p != 0 for p in primes if p * p <= candidate)
                if is_prime:
                    primes.append(candidate)
                candidate += 1
            return primes

        for n in [100, 500, 1000, 2000]:
            # Time slow version
            times_slow = []
            for _ in range(3):
                start = time.perf_counter()
                _ = generate_primes_slow(n)
                times_slow.append((time.perf_counter() - start) * 1000)

            mean_slow = sum(times_slow) / len(times_slow)

            # Time optimized version (if available)
            if IMPORT_SUCCESS:
                times_fast = []
                for _ in range(3):
                    start = time.perf_counter()
                    _ = get_primes(n)
                    times_fast.append((time.perf_counter() - start) * 1000)
                mean_fast = sum(times_fast) / len(times_fast)
                speedup = mean_slow / max(mean_fast, 0.001)
                print(f"      n={n:4d}: slow={mean_slow:.2f}ms, fast={mean_fast:.4f}ms, speedup={speedup:.0f}x")
            else:
                print(f"      n={n:4d}: {mean_slow:.3f} ms")

            self.results[f'primes_{n}'] = {'mean_ms': mean_slow, 'complexity': 'O(n²) worst case'}

    def _benchmark_basis_construction(self):
        """Benchmark orthogonal basis construction."""
        print("\n    Basis construction:")

        # Original slow version
        def build_basis_slow(dim: int, count: int) -> torch.Tensor:
            basis = torch.zeros(count, dim)
            for i in range(count):
                t = torch.linspace(0, 2 * math.pi * (i + 2), dim)
                basis[i] = torch.sin(t) * math.cos(i * 0.1)
            return F.normalize(basis, dim=-1)

        # Optimized version
        def build_basis_fast(dim: int, count: int) -> torch.Tensor:
            primes = torch.arange(2, count + 2, dtype=torch.float32)
            t = torch.linspace(0, 2 * math.pi, dim).unsqueeze(0)
            p = primes.unsqueeze(1)
            i = torch.arange(count, dtype=torch.float32).unsqueeze(1)
            basis = torch.sin(t * p) * torch.cos(i * 0.1)
            return F.normalize(basis, dim=-1)

        for count in [256, 512, 1024]:
            times_slow = []
            for _ in range(3):
                start = time.perf_counter()
                _ = build_basis_slow(64, count)
                times_slow.append((time.perf_counter() - start) * 1000)

            times_fast = []
            for _ in range(3):
                start = time.perf_counter()
                _ = build_basis_fast(64, count)
                times_fast.append((time.perf_counter() - start) * 1000)

            mean_slow = sum(times_slow) / len(times_slow)
            mean_fast = sum(times_fast) / len(times_fast)
            speedup = mean_slow / max(mean_fast, 0.001)

            print(f"      count={count:4d}: slow={mean_slow:.2f}ms, fast={mean_fast:.3f}ms, speedup={speedup:.1f}x")
            self.results[f'basis_{count}'] = {'mean_ms': mean_fast, 'complexity': 'O(count × dim)'}

    def _benchmark_potential_fields(self, device: str):
        """Benchmark potential field computation."""
        print("\n    Potential field computation:")

        feature_dim = 256
        fingerprint_dim = 64
        hidden_dim = 256
        batch_size = 64

        for num_fields in [2, 4, 8, 16]:
            # Slow: separate MLPs
            generators = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim + fingerprint_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_fields)
            ]).to(device)

            # Fast: single MLP with multi-output
            single_net = nn.Sequential(
                nn.Linear(feature_dim + fingerprint_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_fields),
            ).to(device)

            features = torch.randn(batch_size, feature_dim, device=device)
            fingerprint = torch.randn(fingerprint_dim, device=device)
            combined = torch.cat([features, fingerprint.expand(batch_size, -1)], dim=-1)

            # Warmup
            for _ in range(3):
                potentials = []
                for gen in generators:
                    potentials.append(gen(combined))
                _ = torch.cat(potentials, dim=-1)
                _ = single_net(combined)

            # Benchmark slow
            times_slow = []
            for _ in range(10):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                potentials = []
                for gen in generators:
                    potentials.append(gen(combined))
                _ = torch.cat(potentials, dim=-1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times_slow.append((time.perf_counter() - start) * 1000)

            # Benchmark fast
            times_fast = []
            for _ in range(10):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = single_net(combined)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times_fast.append((time.perf_counter() - start) * 1000)

            mean_slow = sum(times_slow) / len(times_slow)
            mean_fast = sum(times_fast) / len(times_fast)
            speedup = mean_slow / max(mean_fast, 0.001)

            print(f"      num_fields={num_fields:2d}: slow={mean_slow:.3f}ms, fast={mean_fast:.3f}ms, speedup={speedup:.1f}x")
            self.results[f'potential_fields_{num_fields}'] = {
                'mean_ms': mean_fast,
                'complexity': 'O(1) batched',
            }

            del generators, single_net
            clear_memory()

    def _benchmark_mailbox_read(self, device: str):
        """Benchmark mailbox message reading."""
        print("\n    Mailbox read operation:")

        fingerprint_dim = 64

        for num_messages in [4, 16, 64, 256]:
            fingerprints = torch.randn(num_messages, fingerprint_dim, device=device)
            reader_fp = torch.randn(fingerprint_dim, device=device)

            # Slow: loop
            times_slow = []
            for _ in range(10):
                start = time.perf_counter()
                scored = []
                for i in range(num_messages):
                    fp_sim = F.cosine_similarity(
                        reader_fp.unsqueeze(0),
                        fingerprints[i].unsqueeze(0)
                    ).item()
                    scored.append((fp_sim, i))
                scored.sort(key=lambda x: -x[0])
                _ = scored[:4]
                times_slow.append((time.perf_counter() - start) * 1000)

            # Fast: batched
            times_fast = []
            for _ in range(10):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                sims = F.cosine_similarity(reader_fp.unsqueeze(0), fingerprints, dim=-1)
                _, top_indices = torch.topk(sims, min(4, num_messages))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times_fast.append((time.perf_counter() - start) * 1000)

            mean_slow = sum(times_slow) / len(times_slow)
            mean_fast = sum(times_fast) / len(times_fast)
            speedup = mean_slow / max(mean_fast, 0.001)

            print(f"      num_messages={num_messages:3d}: slow={mean_slow:.2f}ms, fast={mean_fast:.3f}ms, speedup={speedup:.0f}x")
            self.results[f'mailbox_read_{num_messages}'] = {
                'mean_ms': mean_fast,
                'complexity': 'O(1) batched',
            }

    def _benchmark_local_mask(self):
        """Benchmark local window mask construction."""
        print("\n    Local window mask construction:")

        # Slow: nested loops
        def build_local_mask_slow(num_positions: int, grid_size: int, window: int) -> torch.Tensor:
            P, G, W = num_positions, grid_size, window
            mask = torch.ones(P, P, dtype=torch.bool)
            for i in range(P):
                xi, yi = i % G, i // G
                for j in range(P):
                    xj, yj = j % G, j // G
                    if abs(xi - xj) <= W and abs(yi - yj) <= W:
                        mask[i, j] = False
            return mask

        # Fast: meshgrid
        def build_local_mask_fast(num_positions: int, grid_size: int, window: int) -> torch.Tensor:
            pos = torch.arange(num_positions)
            x = pos % grid_size
            y = pos // grid_size
            xi, xj = torch.meshgrid(x, x, indexing='ij')
            yi, yj = torch.meshgrid(y, y, indexing='ij')
            return ((xi - xj).abs() > window) | ((yi - yj).abs() > window)

        for num_positions in [64, 256, 1024]:
            grid_size = int(math.sqrt(num_positions))

            times_slow = []
            for _ in range(3):
                start = time.perf_counter()
                mask_slow = build_local_mask_slow(num_positions, grid_size, 3)
                times_slow.append((time.perf_counter() - start) * 1000)

            times_fast = []
            for _ in range(3):
                start = time.perf_counter()
                mask_fast = build_local_mask_fast(num_positions, grid_size, 3)
                times_fast.append((time.perf_counter() - start) * 1000)

            mean_slow = sum(times_slow) / len(times_slow)
            mean_fast = sum(times_fast) / len(times_fast)
            speedup = mean_slow / max(mean_fast, 0.001)

            # Verify correctness
            match = torch.equal(mask_slow, mask_fast)

            print(f"      positions={num_positions:4d}: slow={mean_slow:.1f}ms, fast={mean_fast:.3f}ms, speedup={speedup:.0f}x, match={match}")
            self.results[f'local_mask_{num_positions}'] = {
                'mean_ms': mean_fast,
                'complexity': 'O(P²) vectorized',
                'speedup': speedup,
            }

    def _verify_optimizations(self, device: str):
        """Verify that optimized router uses all optimizations."""
        print("\n    Verifying optimizations in GlobalFractalRouter:")

        get_registry().reset()
        config = GlobalFractalRouterConfig(
            feature_dim=256,
            fingerprint_dim=64,
            num_anchors=16,
            num_routes=8,
        )

        router = GlobalFractalRouter(config, name="verify_test").to(device)

        # Check AdjacentGate uses single MLP
        if hasattr(router.adjacent_gate, 'field_net'):
            print("      ✓ AdjacentGate uses batched field_net")
        else:
            print("      ✗ AdjacentGate still uses separate generators")

        # Check AnchorBank uses vectorized fingerprints
        if router.anchor_bank.anchor_fingerprints.shape[0] == config.num_anchors:
            print("      ✓ AnchorBank fingerprints vectorized")

        del router
        clear_memory()


# =============================================================================
# SCALING ANALYSIS
# =============================================================================

class ScalingAnalysisBenchmark:
    """Analyzes how performance scales with input dimensions."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self, router_factory: Callable) -> Dict[str, Dict]:
        results = {
            'batch_scaling': {},
            'sequence_scaling': {},
            'dimension_scaling': {},
        }

        device = self.config.device

        print("\n  Batch scaling...")
        dim, seq = 256, 65
        router = router_factory(dim, seq - 1).to(device)
        router.eval()

        for batch in self.config.batch_sizes:
            x = torch.randn(batch, seq, dim, device=device, dtype=self.config.dtype)
            time_ms = self._time_forward(router, x)
            results['batch_scaling'][batch] = time_ms
            print(f"    B={batch:3d}: {time_ms:.3f} ms")

        del router
        clear_memory()

        print("\n  Sequence scaling...")
        batch, dim = 4, 256

        for seq in self.config.seq_lengths:
            router = router_factory(dim, seq - 1).to(device)
            router.eval()
            x = torch.randn(batch, seq, dim, device=device, dtype=self.config.dtype)
            time_ms = self._time_forward(router, x)
            results['sequence_scaling'][seq] = time_ms
            print(f"    S={seq:4d}: {time_ms:.3f} ms")
            del router
            clear_memory()

        print("\n  Dimension scaling...")
        batch, seq = 4, 65

        for dim in self.config.feature_dims:
            router = router_factory(dim, seq - 1).to(device)
            router.eval()
            x = torch.randn(batch, seq, dim, device=device, dtype=self.config.dtype)
            time_ms = self._time_forward(router, x)
            results['dimension_scaling'][dim] = time_ms
            print(f"    D={dim:4d}: {time_ms:.3f} ms")
            del router
            clear_memory()

        return results

    def _time_forward(self, router: nn.Module, x: torch.Tensor) -> float:
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = router(x)

        times = []
        for _ in range(self.config.benchmark_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = router(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        return sum(times) / len(times)


# =============================================================================
# MEMORY PROFILE
# =============================================================================

class MemoryProfileBenchmark:
    """Profiles memory usage."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self, router_factory: Callable) -> Dict[str, Dict]:
        if not torch.cuda.is_available():
            print("  Memory profiling requires CUDA")
            return {}

        results = {}
        device = self.config.device

        test_configs = [
            (4, 65, 256, "small"),
            (8, 129, 512, "medium"),
            (16, 257, 512, "large"),
        ]

        for batch, seq, dim, name in test_configs:
            clear_memory()
            torch.cuda.reset_peak_memory_stats()

            base_mem = get_memory_mb()
            router = router_factory(dim, seq - 1).to(device)
            router_mem = get_memory_mb()

            x = torch.randn(batch, seq, dim, device=device, dtype=self.config.dtype)
            input_mem = get_memory_mb()

            with torch.no_grad():
                routes, weights, features = router(x)
            forward_mem = get_memory_mb()

            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

            results[name] = {
                'config': f"B={batch}, S={seq}, D={dim}",
                'router_mb': router_mem - base_mem,
                'input_mb': input_mem - router_mem,
                'forward_mb': forward_mem - input_mem,
                'peak_mb': peak_mem,
                'params': sum(p.numel() for p in router.parameters()),
            }

            print(f"  {name}: router={results[name]['router_mb']:.1f}MB, "
                  f"forward={results[name]['forward_mb']:.1f}MB, peak={peak_mem:.1f}MB")

            del router, x, routes, weights, features
            clear_memory()

        return results


# =============================================================================
# BASELINE COMPARISON
# =============================================================================

class BaselineComparisonBenchmark:
    """Compares GlobalFractalRouter against baseline."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(
        self,
        global_router_factory: Callable,
        baseline_router_factory: Callable,
    ) -> Dict[str, Dict]:
        results = {}
        device = self.config.device

        test_configs = [
            (4, 65, 256, "4x64x256"),
            (8, 129, 256, "8x128x256"),
            (4, 65, 512, "4x64x512"),
        ]

        for batch, seq, dim, name in test_configs:
            print(f"\n  Config: {name}")

            x = torch.randn(batch, seq, dim, device=device, dtype=self.config.dtype)

            baseline = baseline_router_factory(dim, seq - 1).to(device)
            baseline.eval()
            baseline_time = self._time_forward(baseline, x)

            global_router = global_router_factory(dim, seq - 1).to(device)
            global_router.eval()
            global_time = self._time_forward(global_router, x)

            overhead = (global_time - baseline_time) / baseline_time * 100

            results[name] = {
                'baseline_ms': baseline_time,
                'global_ms': global_time,
                'overhead_pct': overhead,
                'baseline_params': sum(p.numel() for p in baseline.parameters()),
                'global_params': sum(p.numel() for p in global_router.parameters()),
            }

            print(f"    Baseline: {baseline_time:.3f} ms ({results[name]['baseline_params']:,} params)")
            print(f"    Global:   {global_time:.3f} ms ({results[name]['global_params']:,} params)")
            print(f"    Overhead: {overhead:+.1f}%")

            del baseline, global_router
            clear_memory()

        return results

    def _time_forward(self, router: nn.Module, x: torch.Tensor) -> float:
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = router(x)

        times = []
        for _ in range(self.config.benchmark_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = router(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        return sum(times) / len(times)


# =============================================================================
# NETWORK BENCHMARK
# =============================================================================

class NetworkBenchmark:
    """Benchmarks FractalRouterNetwork topologies."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self, network_factory: Callable) -> Dict[str, Dict]:
        results = {}
        device = self.config.device

        batch, seq, dim = 4, 65, 256
        x = torch.randn(batch, seq, dim, device=device, dtype=self.config.dtype)

        for topology in ["chain", "parallel", "tree"]:
            for num_routers in [2, 4, 8]:
                name = f"{topology}_{num_routers}"
                print(f"\n    {name}:")

                try:
                    network = network_factory(dim, seq - 1, num_routers, topology).to(device)
                    network.eval()

                    for _ in range(self.config.warmup_iterations):
                        with torch.no_grad():
                            _ = network(x)

                    times = []
                    for _ in range(self.config.benchmark_iterations):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        with torch.no_grad():
                            _ = network(x)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) * 1000)

                    mean_time = sum(times) / len(times)
                    per_router = mean_time / num_routers

                    results[name] = {
                        'total_ms': mean_time,
                        'per_router_ms': per_router,
                        'num_routers': num_routers,
                        'topology': topology,
                    }

                    print(f"      Total: {mean_time:.3f} ms, Per-router: {per_router:.3f} ms")

                    del network
                    clear_memory()

                except Exception as e:
                    print(f"      Error: {e}")
                    results[name] = {'error': str(e)}

        return results


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

class GlobalFractalRouterBenchmark:
    """Main benchmark orchestrator."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: Dict[str, Dict] = {}

    def run_all(self) -> Dict[str, Dict]:
        print("=" * 70)
        print("Global Fractal Router V2 Benchmark Suite")
        print("=" * 70)
        print(f"Device: {self.config.device}")
        print(f"Dtype: {self.config.dtype}")
        print(f"Warmup: {self.config.warmup_iterations}, Iterations: {self.config.benchmark_iterations}")

        if IMPORT_SUCCESS:
            get_registry().reset()

        def global_router_factory(dim: int, num_positions: int):
            if IMPORT_SUCCESS:
                config = GlobalFractalRouterConfig(
                    feature_dim=dim,
                    fingerprint_dim=64,
                    num_anchors=16,
                    num_routes=8,
                )
                get_registry().reset()
                return GlobalFractalRouter(config)
            else:
                raise RuntimeError("GlobalFractalRouter not available")

        def baseline_factory(dim: int, num_positions: int):
            return BaselineWormholeRouter(dim, num_positions, num_routes=8)

        def network_factory(dim: int, num_positions: int, num_routers: int, topology: str):
            if IMPORT_SUCCESS:
                config = GlobalFractalRouterConfig(feature_dim=dim, num_routes=8)
                get_registry().reset()
                return FractalRouterNetwork(config, num_routers=num_routers, topology=topology)
            else:
                raise RuntimeError("FractalRouterNetwork not available")

        if self.config.run_loop_analysis:
            print("\n" + "-" * 70)
            print("LOOP ANALYSIS & OPTIMIZATION VERIFICATION")
            print("-" * 70)
            bench = LoopAnalysisBenchmark(self.config)
            self.results['loop_analysis'] = bench.run()

        if self.config.run_baseline_comparison and IMPORT_SUCCESS:
            print("\n" + "-" * 70)
            print("BASELINE COMPARISON")
            print("-" * 70)
            bench = BaselineComparisonBenchmark(self.config)
            self.results['baseline_comparison'] = bench.run(global_router_factory, baseline_factory)

        if self.config.run_scaling_analysis:
            print("\n" + "-" * 70)
            print("SCALING ANALYSIS (baseline router)")
            print("-" * 70)
            bench = ScalingAnalysisBenchmark(self.config)
            self.results['baseline_scaling'] = bench.run(baseline_factory)

            if IMPORT_SUCCESS:
                print("\n" + "-" * 70)
                print("SCALING ANALYSIS (global fractal router)")
                print("-" * 70)
                self.results['global_scaling'] = bench.run(global_router_factory)

        if self.config.run_memory_profile and torch.cuda.is_available() and IMPORT_SUCCESS:
            print("\n" + "-" * 70)
            print("MEMORY PROFILE")
            print("-" * 70)
            bench = MemoryProfileBenchmark(self.config)
            self.results['memory_profile'] = bench.run(global_router_factory)

        if self.config.run_network_benchmark and IMPORT_SUCCESS:
            print("\n" + "-" * 70)
            print("NETWORK BENCHMARK")
            print("-" * 70)
            bench = NetworkBenchmark(self.config)
            self.results['network'] = bench.run(network_factory)

        self._print_summary()

        return self.results

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 70)

        if 'loop_analysis' in self.results:
            print("\n[Optimization Status]")
            for key, data in self.results['loop_analysis'].items():
                if isinstance(data, dict) and 'speedup' in data:
                    speedup = data['speedup']
                    status = "✓" if speedup > 10 else "⚠" if speedup > 2 else "✗"
                    print(f"  {status} {key}: {speedup:.0f}x speedup")

        if 'baseline_comparison' in self.results:
            print("\n[Overhead Analysis]")
            for config, data in self.results['baseline_comparison'].items():
                if 'overhead_pct' in data:
                    overhead = data['overhead_pct']
                    status = "✓" if overhead < 50 else "⚠" if overhead < 100 else "✗"
                    print(f"  {status} {config}: {overhead:+.1f}% overhead")

        if 'memory_profile' in self.results:
            print("\n[Memory Efficiency]")
            for config, data in self.results['memory_profile'].items():
                if 'peak_mb' in data:
                    print(f"  {config}: {data['peak_mb']:.1f} MB peak")

        print("\n" + "=" * 70)


# =============================================================================
# COLAB / NOTEBOOK RUNNER
# =============================================================================

def run_benchmark(
    device: str = None,
    iterations: int = 10,
    warmup: int = 3,
    quick: bool = False,
    loops_only: bool = False,
    scaling_only: bool = False,
    memory_only: bool = False,
    comparison_only: bool = False,
    network_only: bool = False,
) -> Dict[str, Dict]:
    """
    Run GlobalFractalRouter benchmarks.

    Args:
        device: "cuda" or "cpu" (auto-detected if None)
        iterations: Benchmark iterations per test
        warmup: Warmup iterations
        quick: Reduced test suite
        loops_only: Only run loop analysis
        scaling_only: Only run scaling analysis
        memory_only: Only run memory profiling
        comparison_only: Only run baseline comparison
        network_only: Only run network benchmark

    Returns:
        Dict of benchmark results

    Example:
        results = run_benchmark(quick=True)
        results = run_benchmark(device="cuda", iterations=20)
        results = run_benchmark(loops_only=True)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = BenchmarkConfig(
        device=device,
        benchmark_iterations=3 if quick else iterations,
        warmup_iterations=1 if quick else warmup,
        batch_sizes=[1, 4, 8] if quick else [1, 2, 4, 8, 16],
        seq_lengths=[65, 129] if quick else [65, 129, 257, 513],
        feature_dims=[256] if quick else [256, 512],
        num_routes=[4, 8] if quick else [4, 8, 16],
    )

    if loops_only:
        config.run_component_breakdown = False
        config.run_scaling_analysis = False
        config.run_memory_profile = False
        config.run_baseline_comparison = False
        config.run_network_benchmark = False
        config.run_loop_analysis = True
    elif scaling_only:
        config.run_component_breakdown = False
        config.run_loop_analysis = False
        config.run_memory_profile = False
        config.run_baseline_comparison = False
        config.run_network_benchmark = False
        config.run_scaling_analysis = True
    elif memory_only:
        config.run_component_breakdown = False
        config.run_loop_analysis = False
        config.run_scaling_analysis = False
        config.run_baseline_comparison = False
        config.run_network_benchmark = False
        config.run_memory_profile = True
    elif comparison_only:
        config.run_component_breakdown = False
        config.run_loop_analysis = False
        config.run_scaling_analysis = False
        config.run_memory_profile = False
        config.run_network_benchmark = False
        config.run_baseline_comparison = True
    elif network_only:
        config.run_component_breakdown = False
        config.run_loop_analysis = False
        config.run_scaling_analysis = False
        config.run_memory_profile = False
        config.run_baseline_comparison = False
        config.run_network_benchmark = True

    benchmark = GlobalFractalRouterBenchmark(config)
    return benchmark.run_all()


if __name__ == "__main__":
    # =========================================================================
    # INLINE CONFIGURATION
    # =========================================================================

    DEVICE = None          # None = auto-detect
    ITERATIONS = 10
    WARMUP = 3
    QUICK = True           # Fast iteration mode

    # Selective benchmarks (all False = run everything)
    LOOPS_ONLY = False
    SCALING_ONLY = False
    MEMORY_ONLY = False
    COMPARISON_ONLY = False
    NETWORK_ONLY = False

    # =========================================================================

    results = run_benchmark(
        device=DEVICE,
        iterations=ITERATIONS,
        warmup=WARMUP,
        quick=QUICK,
        loops_only=LOOPS_ONLY,
        scaling_only=SCALING_ONLY,
        memory_only=MEMORY_ONLY,
        comparison_only=COMPARISON_ONLY,
        network_only=NETWORK_ONLY,
    )

    print("\nResults keys:", list(results.keys()))