"""
global_fractal_router_benchmark.py - Performance analysis for GlobalFractalRouter

Benchmarks:
1. Component-level timing (identify bottlenecks)
2. Scaling behavior (batch, sequence, routes)
3. Memory profiling
4. Comparison vs baseline wormhole router
5. For-loop hotspot analysis

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

# Attempt imports
try:
    from geofractal.model.blocks.router.global_fractal_router import (
        GlobalFractalRouter,
        GlobalFractalRouterConfig,
        FingerprintRegistry,
        get_registry,
        AnchorBank,
        AdjacentGate,
        RouterMailbox,
        FractalRouterNetwork,
        ProvenanceTensor,
    )

    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import GlobalFractalRouter: {e}")
    print("Running in standalone mode with inline definitions")
    IMPORT_SUCCESS = False


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


class Timer:
    """Simple timer for inline measurements."""

    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self.start_time = None

    def start(self):
        if torch.cuda.is_available() and self.sync_cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if torch.cuda.is_available() and self.sync_cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - self.start_time) * 1000


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

        # Gather
        K = self.num_routes
        routes_flat = routes.reshape(B, P * K).unsqueeze(-1).expand(-1, -1, D)
        v_gathered = torch.gather(v, 1, routes_flat).view(B, P, K, D)

        features = torch.einsum('bpk,bpkd->bpd', weights, v_gathered)

        return routes, weights, features


# =============================================================================
# BENCHMARK CONFIGURATIONS
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Scaling parameters
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    seq_lengths: List[int] = field(default_factory=lambda: [65, 129, 257, 513])  # 1 + patches
    feature_dims: List[int] = field(default_factory=lambda: [256, 512])
    num_routes: List[int] = field(default_factory=lambda: [4, 8, 16])

    # Timing
    warmup_iterations: int = 3
    benchmark_iterations: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # What to benchmark
    run_component_breakdown: bool = True
    run_scaling_analysis: bool = True
    run_memory_profile: bool = True
    run_baseline_comparison: bool = True
    run_loop_analysis: bool = True
    run_network_benchmark: bool = True


# =============================================================================
# COMPONENT BREAKDOWN BENCHMARK
# =============================================================================

class ComponentBreakdownBenchmark:
    """Profiles individual components of GlobalFractalRouter."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, List[float]] = {}

    def run(self, router: nn.Module, x: torch.Tensor) -> Dict[str, float]:
        """Run component-level profiling."""
        self.results.clear()

        router.eval()
        B, S, D = x.shape
        P = S - 1  # Assuming skip_first=True

        # Warmup
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = router(x)

        # Profile each component
        for _ in range(self.config.benchmark_iterations):
            with torch.no_grad():
                self._profile_forward(router, x)

        # Compute statistics
        stats = {}
        for name, times in self.results.items():
            stats[name] = {
                'mean_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'std_ms': (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
            }

        return stats

    def _profile_forward(self, router: nn.Module, x: torch.Tensor):
        """Profile a single forward pass."""

        # Note: This requires access to router internals
        # We'll time the full forward and key sub-operations

        with cuda_timer("full_forward", self.results):
            routes, weights, features = router(x)

        # Profile sub-components if accessible
        if hasattr(router, 'query_proj'):
            x_skip = x[:, 1:, :]

            with cuda_timer("query_proj", self.results):
                q = router.query_proj(x_skip)

            with cuda_timer("key_proj", self.results):
                k = router.key_proj(x_skip)

            with cuda_timer("qk_norm", self.results):
                q = F.normalize(q, dim=-1)
                k = F.normalize(k, dim=-1)

            with cuda_timer("score_bmm", self.results):
                scores = torch.bmm(q, k.transpose(1, 2))

            with cuda_timer("topk", self.results):
                _, routes = torch.topk(scores, router.config.num_routes, dim=-1)

        if hasattr(router, 'anchor_bank'):
            with cuda_timer("anchor_bank", self.results):
                _ = router.anchor_bank(x[:, 1:, :].mean(dim=1))

        if hasattr(router, 'adjacent_gate') and router.adjacent_gate is not None:
            with cuda_timer("adjacent_gate", self.results):
                flat = features[:, 1:, :].reshape(-1, features.shape[-1])
                _ = router.adjacent_gate.compute_potential(
                    flat[:16],  # Subset for timing
                    router.fingerprint
                )


# =============================================================================
# SCALING ANALYSIS BENCHMARK
# =============================================================================

class ScalingAnalysisBenchmark:
    """Analyzes how performance scales with input dimensions."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self, router_factory: Callable) -> Dict[str, Dict]:
        """
        Run scaling analysis.

        Args:
            router_factory: Callable(dim, num_positions) -> router
        """
        results = {
            'batch_scaling': {},
            'sequence_scaling': {},
            'dimension_scaling': {},
            'route_scaling': {},
        }

        device = self.config.device

        # Batch scaling (fixed seq=65, dim=256)
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

        # Sequence scaling (fixed batch=4, dim=256)
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

        # Dimension scaling (fixed batch=4, seq=65)
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
        """Time forward pass with warmup."""
        # Warmup
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = router(x)

        # Benchmark
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
# MEMORY PROFILE BENCHMARK
# =============================================================================

class MemoryProfileBenchmark:
    """Profiles memory usage of router components."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self, router_factory: Callable) -> Dict[str, Dict]:
        """Run memory profiling."""
        if not torch.cuda.is_available():
            print("  Memory profiling requires CUDA")
            return {}

        results = {}
        device = self.config.device

        # Test different configurations
        test_configs = [
            (4, 65, 256, "small"),
            (8, 129, 512, "medium"),
            (16, 257, 512, "large"),
        ]

        for batch, seq, dim, name in test_configs:
            clear_memory()

            # Baseline memory
            base_mem = get_memory_mb()

            # Create router
            router = router_factory(dim, seq - 1).to(device)
            router_mem = get_memory_mb()

            # Create input
            x = torch.randn(batch, seq, dim, device=device, dtype=self.config.dtype)
            input_mem = get_memory_mb()

            # Forward pass
            with torch.no_grad():
                routes, weights, features = router(x)
            forward_mem = get_memory_mb()

            # Peak memory
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
            torch.cuda.reset_peak_memory_stats()

        return results


# =============================================================================
# BASELINE COMPARISON BENCHMARK
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
        """Compare global router vs baseline."""
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

            # Baseline router
            baseline = baseline_router_factory(dim, seq - 1).to(device)
            baseline.eval()
            baseline_time = self._time_forward(baseline, x)

            # Global fractal router
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
# LOOP ANALYSIS BENCHMARK
# =============================================================================

class LoopAnalysisBenchmark:
    """
    Specifically analyzes for-loop hotspots.

    Known loop locations in GlobalFractalRouter:
    1. FingerprintRegistry._generate_primes() - O(n) primes
    2. FingerprintRegistry._build_orthogonal_basis() - O(count) basis vectors
    3. AdjacentGate.compute_potential() - O(num_fields) field generators
    4. AnchorBank (implicit in matmul, not a loop)
    5. RouterMailbox.read() - O(n) messages
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, Dict] = {}

    def run(self) -> Dict[str, Dict]:
        """Analyze all known loop locations."""
        device = self.config.device

        print("\n  Analyzing loop hotspots...")

        # 1. Prime generation (one-time cost)
        self._benchmark_prime_generation()

        # 2. Orthogonal basis construction (one-time cost)
        self._benchmark_basis_construction()

        # 3. Adjacent gate potential computation (per-forward cost)
        self._benchmark_potential_fields(device)

        # 4. Mailbox read operation
        self._benchmark_mailbox_read(device)

        # 5. Local window mask construction
        self._benchmark_local_mask()

        return self.results

    def _benchmark_prime_generation(self):
        """Benchmark prime number generation."""
        print("\n    Prime generation:")

        def generate_primes(n: int) -> List[int]:
            primes = []
            candidate = 2
            while len(primes) < n:
                is_prime = all(candidate % p != 0 for p in primes if p * p <= candidate)
                if is_prime:
                    primes.append(candidate)
                candidate += 1
            return primes

        for n in [100, 500, 1000, 2000]:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = generate_primes(n)
                times.append((time.perf_counter() - start) * 1000)

            mean_time = sum(times) / len(times)
            print(f"      n={n:4d}: {mean_time:.3f} ms")
            self.results[f'primes_{n}'] = {'mean_ms': mean_time, 'complexity': 'O(n²) worst case'}

    def _benchmark_basis_construction(self):
        """Benchmark orthogonal basis construction."""
        print("\n    Basis construction:")

        def build_basis(dim: int, count: int) -> torch.Tensor:
            # Simplified - just the loop part
            basis = torch.zeros(count, dim)
            for i in range(count):
                t = torch.linspace(0, 2 * math.pi * (i + 2), dim)
                basis[i] = torch.sin(t) * math.cos(i * 0.1)
            return F.normalize(basis, dim=-1)

        for count in [256, 512, 1024]:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = build_basis(64, count)
                times.append((time.perf_counter() - start) * 1000)

            mean_time = sum(times) / len(times)
            print(f"      count={count:4d}: {mean_time:.3f} ms")
            self.results[f'basis_{count}'] = {'mean_ms': mean_time, 'complexity': 'O(count × dim)'}

    def _benchmark_potential_fields(self, device: str):
        """Benchmark potential field computation loop."""
        print("\n    Potential field computation:")

        feature_dim = 256
        fingerprint_dim = 64
        hidden_dim = 256
        batch_size = 64

        for num_fields in [2, 4, 8, 16]:
            # Create field generators
            generators = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim + fingerprint_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_fields)
            ]).to(device)

            features = torch.randn(batch_size, feature_dim, device=device)
            fingerprint = torch.randn(fingerprint_dim, device=device)
            combined = torch.cat([features, fingerprint.expand(batch_size, -1)], dim=-1)

            # Warmup
            for _ in range(3):
                potentials = []
                for gen in generators:
                    potentials.append(gen(combined))
                _ = torch.cat(potentials, dim=-1)

            # Benchmark
            times = []
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
                times.append((time.perf_counter() - start) * 1000)

            mean_time = sum(times) / len(times)
            print(f"      num_fields={num_fields:2d}: {mean_time:.3f} ms")
            self.results[f'potential_fields_{num_fields}'] = {
                'mean_ms': mean_time,
                'complexity': 'O(num_fields)',
                'note': 'HOTSPOT - consider batched forward',
            }

            del generators
            clear_memory()

    def _benchmark_mailbox_read(self, device: str):
        """Benchmark mailbox message reading."""
        print("\n    Mailbox read operation:")

        fingerprint_dim = 64
        comm_dim = 128

        for num_messages in [4, 16, 64, 256]:
            # Simulate messages
            messages = {}
            for i in range(num_messages):
                messages[f'router_{i}'] = {
                    'fingerprint': torch.randn(fingerprint_dim, device=device),
                    'state': torch.randn(comm_dim, device=device),
                }

            reader_fp = torch.randn(fingerprint_dim, device=device)

            # Benchmark read (fingerprint comparison loop)
            times = []
            for _ in range(10):
                start = time.perf_counter()

                scored = []
                for sender_id, msg in messages.items():
                    fp_sim = F.cosine_similarity(
                        reader_fp.unsqueeze(0),
                        msg['fingerprint'].unsqueeze(0)
                    ).item()
                    scored.append((fp_sim, sender_id))

                scored.sort(key=lambda x: -x[0])
                top_k = scored[:4]

                times.append((time.perf_counter() - start) * 1000)

            mean_time = sum(times) / len(times)
            print(f"      num_messages={num_messages:3d}: {mean_time:.3f} ms")
            self.results[f'mailbox_read_{num_messages}'] = {
                'mean_ms': mean_time,
                'complexity': 'O(n)',
                'note': 'HOTSPOT - consider batched cosine sim',
            }

    def _benchmark_local_mask(self):
        """Benchmark local window mask construction."""
        print("\n    Local window mask construction:")

        def build_local_mask(num_positions: int, grid_size: int, window: int) -> torch.Tensor:
            P, G, W = num_positions, grid_size, window
            mask = torch.ones(P, P, dtype=torch.bool)

            for i in range(P):
                xi, yi = i % G, i // G
                for j in range(P):
                    xj, yj = j % G, j // G
                    if abs(xi - xj) <= W and abs(yi - yj) <= W:
                        mask[i, j] = False

            return mask

        for num_positions in [64, 256, 1024]:
            grid_size = int(math.sqrt(num_positions))

            times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = build_local_mask(num_positions, grid_size, 3)
                times.append((time.perf_counter() - start) * 1000)

            mean_time = sum(times) / len(times)
            print(f"      positions={num_positions:4d}: {mean_time:.3f} ms")
            self.results[f'local_mask_{num_positions}'] = {
                'mean_ms': mean_time,
                'complexity': 'O(P²)',
                'note': 'HOTSPOT - should vectorize with meshgrid',
            }


# =============================================================================
# NETWORK BENCHMARK
# =============================================================================

class NetworkBenchmark:
    """Benchmarks FractalRouterNetwork with different topologies."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self, network_factory: Callable) -> Dict[str, Dict]:
        """Benchmark router network configurations."""
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

                    # Warmup
                    for _ in range(self.config.warmup_iterations):
                        with torch.no_grad():
                            _ = network(x)

                    # Benchmark
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
        """Run all benchmarks."""
        print("=" * 70)
        print("Global Fractal Router Benchmark Suite")
        print("=" * 70)
        print(f"Device: {self.config.device}")
        print(f"Dtype: {self.config.dtype}")
        print(f"Warmup: {self.config.warmup_iterations}, Iterations: {self.config.benchmark_iterations}")

        # Reset registry before benchmarks
        if IMPORT_SUCCESS:
            get_registry().reset()

        # Factory functions
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

        # Run benchmarks
        if self.config.run_loop_analysis:
            print("\n" + "-" * 70)
            print("LOOP ANALYSIS (identify Python loop hotspots)")
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

        # Summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print benchmark summary with actionable insights."""
        print("\n" + "=" * 70)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 70)

        # Loop hotspots
        if 'loop_analysis' in self.results:
            print("\n[Loop Hotspots]")
            hotspots = []
            for key, data in self.results['loop_analysis'].items():
                if isinstance(data, dict) and 'mean_ms' in data:
                    if data.get('note', '').startswith('HOTSPOT'):
                        hotspots.append((key, data['mean_ms'], data.get('note', '')))

            if hotspots:
                hotspots.sort(key=lambda x: -x[1])
                for name, time_ms, note in hotspots:
                    print(f"  ⚠ {name}: {time_ms:.3f} ms - {note}")
            else:
                print("  ✓ No major loop hotspots detected")

        # Overhead analysis
        if 'baseline_comparison' in self.results:
            print("\n[Overhead Analysis]")
            for config, data in self.results['baseline_comparison'].items():
                if 'overhead_pct' in data:
                    overhead = data['overhead_pct']
                    status = "✓" if overhead < 50 else "⚠" if overhead < 100 else "✗"
                    print(f"  {status} {config}: {overhead:+.1f}% overhead")

        # Memory efficiency
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
    Run GlobalFractalRouter benchmarks with inline configuration.

    Args:
        device: "cuda" or "cpu" (auto-detected if None)
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
        quick: Use reduced iterations and smaller configs
        loops_only: Only run loop hotspot analysis
        scaling_only: Only run scaling analysis
        memory_only: Only run memory profiling
        comparison_only: Only run baseline comparison
        network_only: Only run network topology benchmark

    Returns:
        Dict of benchmark results

    Example (Colab):
        ```python
        from global_fractal_router_benchmark import run_benchmark

        # Quick test
        results = run_benchmark(quick=True)

        # Full benchmark on GPU
        results = run_benchmark(device="cuda", iterations=20)

        # Just check loop hotspots
        results = run_benchmark(loops_only=True)
        ```
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build config
    config = BenchmarkConfig(
        device=device,
        benchmark_iterations=3 if quick else iterations,
        warmup_iterations=1 if quick else warmup,
        batch_sizes=[1, 4, 8] if quick else [1, 2, 4, 8, 16],
        seq_lengths=[65, 129] if quick else [65, 129, 257, 513],
        feature_dims=[256] if quick else [256, 512],
        num_routes=[4, 8] if quick else [4, 8, 16],
    )

    # Handle selective benchmarks
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
    results = benchmark.run_all()

    return results


if __name__ == "__main__":
    # =========================================================================
    # INLINE CONFIGURATION - EDIT THESE FOR YOUR RUN
    # =========================================================================

    DEVICE = None  # None = auto-detect, or "cuda" / "cpu"
    ITERATIONS = 10  # Benchmark iterations per test
    WARMUP = 3  # Warmup iterations
    QUICK = True  # Reduced test suite for fast iteration

    # Select which benchmarks to run (set one to True, others False for focused testing)
    LOOPS_ONLY = False  # Just analyze Python for-loop hotspots
    SCALING_ONLY = False  # Just analyze batch/seq/dim scaling
    MEMORY_ONLY = False  # Just analyze GPU memory usage
    COMPARISON_ONLY = False  # Just compare vs baseline router
    NETWORK_ONLY = False  # Just benchmark router network topologies

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

    # Results are now in `results` dict for further analysis
    print("\nResults keys:", list(results.keys()))