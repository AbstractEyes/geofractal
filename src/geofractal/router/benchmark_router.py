import torch
import torch.nn as nn
from torch import Tensor
import time
import gc

from geofractal.router.base_router import BaseRouter


# =============================================================================
# MEMORY ESTIMATION
# =============================================================================

def estimate_memory(dim, num_layers, expansion, batch_size, seq_len):
    """Estimate GPU memory including activations."""

    params_per_layer = (dim * dim * expansion + dim * expansion +
                        dim * expansion * dim + dim +
                        dim + dim)
    total_params = params_per_layer * num_layers

    param_bytes = total_params * 4
    grad_bytes = total_params * 4
    optimizer_bytes = total_params * 8

    activation_elements = batch_size * seq_len * dim * (3 + 2 * expansion) * num_layers
    activation_bytes = activation_elements * 4

    total_bytes = param_bytes + grad_bytes + optimizer_bytes + activation_bytes

    return {
        'params_mb': param_bytes / 1024 ** 2,
        'grads_mb': grad_bytes / 1024 ** 2,
        'optimizer_mb': optimizer_bytes / 1024 ** 2,
        'activations_mb': activation_bytes / 1024 ** 2,
        'total_gb': total_bytes / 1024 ** 3,
        'total_params': total_params,
    }


# =============================================================================
# MODELS
# =============================================================================

class TraditionalSequential(nn.Module):
    """Traditional with nn.Sequential (extra wrapper overhead)."""

    def __init__(self, dim: int, num_layers: int, expansion: int = 4):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim * expansion),
                nn.GELU(),
                nn.Linear(dim * expansion, dim),
                nn.LayerNorm(dim),
            ))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class TraditionalDirect(nn.Module):
    """Traditional with direct module calls (no Sequential)."""

    def __init__(self, dim: int, num_layers: int, expansion: int = 4):
        super().__init__()

        self.num_layers = num_layers

        self.fc1_layers = nn.ModuleList([
            nn.Linear(dim, dim * expansion) for _ in range(num_layers)
        ])
        self.act_layers = nn.ModuleList([
            nn.GELU() for _ in range(num_layers)
        ])
        self.fc2_layers = nn.ModuleList([
            nn.Linear(dim * expansion, dim) for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.num_layers):
            residual = x
            x = self.fc1_layers[i](x)
            x = self.act_layers[i](x)
            x = self.fc2_layers[i](x)
            x = self.norm_layers[i](x)
            x = residual + x
        return x


class RouterLayerBlock(BaseRouter):
    """Single residual block as router."""

    def __init__(self, name: str, dim: int, expansion: int = 4):
        super().__init__(name=name, strict=False)

        self.attach('fc1', nn.Linear(dim, dim * expansion))
        self.attach('act', nn.GELU())
        self.attach('fc2', nn.Linear(dim * expansion, dim))
        self.attach('norm', nn.LayerNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self['fc1'](x)
        x = self['act'](x)
        x = self['fc2'](x)
        x = self['norm'](x)
        return residual + x


class RouterLarge(BaseRouter):
    """Large router-based model."""

    def __init__(self, dim: int, num_layers: int, expansion: int = 4):
        super().__init__(name='router_large', strict=False)

        self.num_layers = num_layers
        for i in range(num_layers):
            self.attach(f'block_{i}', RouterLayerBlock(f'block_{i}', dim, expansion))

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.num_layers):
            x = self[f'block_{i}'](x)
        return x


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark(model, x, target, iterations: int, warmup: int = 100, name: str = "Model"):
    """Benchmark forward + backward."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(warmup):
        optimizer.zero_grad()
        y = model(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    start = time.perf_counter()

    for i in range(iterations):
        optimizer.zero_grad()
        y = model(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        optimizer.step()

        if iterations >= 1000 and (i + 1) % 1000 == 0:
            print(f"      {i + 1}/{iterations}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    return elapsed, elapsed / iterations * 1000, peak_gb


def run_comparison(dim, num_layers, expansion, batch_size, seq_len, iterations):
    """Run full comparison."""

    device = torch.device('cuda')
    est = estimate_memory(dim, num_layers, expansion, batch_size, seq_len)

    print("=" * 70)
    print(f"CONFIG: dim={dim}, layers={num_layers}, expansion={expansion}")
    print(f"        batch={batch_size}, seq_len={seq_len}, iterations={iterations}")
    print("=" * 70)
    print()
    print("MEMORY ESTIMATE:")
    print(f"  Parameters:  {est['params_mb']:>8,.0f} MB  ({est['total_params']:,} params)")
    print(f"  Gradients:   {est['grads_mb']:>8,.0f} MB")
    print(f"  Optimizer:   {est['optimizer_mb']:>8,.0f} MB")
    print(f"  Activations: {est['activations_mb']:>8,.0f} MB")
    print(f"  ─────────────────────────────")
    print(f"  Estimated:   {est['total_gb']:>8.2f} GB")
    print()

    x = torch.randn(batch_size, seq_len, dim, device=device)
    target = torch.randn(batch_size, seq_len, dim, device=device)

    results = {}

    # -------------------------------------------------------------------------
    print("[1/3] TRADITIONAL + nn.Sequential")
    gc.collect()
    torch.cuda.empty_cache()

    model = TraditionalSequential(dim, num_layers, expansion).to(device)
    elapsed, ms_per_iter, peak = benchmark(model, x, target, iterations, name="Sequential")
    results['sequential'] = {'ms': ms_per_iter, 'peak_gb': peak}

    print(f"    Time: {elapsed:.2f}s | {ms_per_iter:.4f}ms/iter | Peak: {peak:.2f} GB")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    print("[2/3] TRADITIONAL + Direct Calls")
    gc.collect()
    torch.cuda.empty_cache()

    model = TraditionalDirect(dim, num_layers, expansion).to(device)
    elapsed, ms_per_iter, peak = benchmark(model, x, target, iterations, name="Direct")
    results['direct'] = {'ms': ms_per_iter, 'peak_gb': peak}

    print(f"    Time: {elapsed:.2f}s | {ms_per_iter:.4f}ms/iter | Peak: {peak:.2f} GB")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    print("[3/3] ROUTER")
    gc.collect()
    torch.cuda.empty_cache()

    model = RouterLarge(dim, num_layers, expansion).to(device)
    elapsed, ms_per_iter, peak = benchmark(model, x, target, iterations, name="Router")
    results['router'] = {'ms': ms_per_iter, 'peak_gb': peak}

    print(f"    Time: {elapsed:.2f}s | {ms_per_iter:.4f}ms/iter | Peak: {peak:.2f} GB")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    seq_ms = results['sequential']['ms']
    direct_ms = results['direct']['ms']
    router_ms = results['router']['ms']

    print(f"  Sequential:  {seq_ms:.4f} ms/iter (baseline)")
    print(f"  Direct:      {direct_ms:.4f} ms/iter ({(direct_ms / seq_ms - 1) * 100:+.2f}%)")
    print(f"  Router:      {router_ms:.4f} ms/iter ({(router_ms / seq_ms - 1) * 100:+.2f}%)")
    print()
    print(f"  Router vs Sequential: {router_ms / seq_ms:.4f}x")
    print(f"  Router vs Direct:     {router_ms / direct_ms:.4f}x")
    print()

    # Memory
    print(f"  Peak VRAM:")
    print(f"    Sequential: {results['sequential']['peak_gb']:.2f} GB")
    print(f"    Direct:     {results['direct']['peak_gb']:.2f} GB")
    print(f"    Router:     {results['router']['peak_gb']:.2f} GB")
    print(f"    Estimated:  {est['total_gb']:.2f} GB")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print()

    # Test 1: Small, many iterations
    print("TEST 1: Small model, many iterations")
    run_comparison(
        dim=256,
        num_layers=32,
        expansion=4,
        batch_size=16,
        seq_len=512,
        iterations=100,
    )

    print("\n" * 2)

    # Test 2: Medium
    print("TEST 2: Medium model")
    run_comparison(
        dim=512,
        num_layers=32,
        expansion=4,
        batch_size=16,
        seq_len=512,
        iterations=100,
    )

    print("\n" * 2)

    # Test 3: Large
    print("TEST 3: Large model")
    run_comparison(
        dim=1024,
        num_layers=24,
        expansion=4,
        batch_size=8,
        seq_len=512,
        iterations=100,
    )