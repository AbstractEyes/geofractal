# GeoFractal Router

**Collective Intelligence through Geometric Routing**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.2.0-green.svg)]()

---

## What Is This?

GeoFractal Router is a coordination architecture for building **collectives of autonomous AI units**. Instead of one monolithic model, you build multiple *towers* that produce opinions, coordinate through *geometric routing*, and fuse their perspectives into emergent collective intelligence.

**The key insight:** Individual units don't need to be accurate. They need to *see differently*. The collective triangulates truth from divergent viewpoints.

```
Traditional Ensemble:    Smart Model + Smart Model + Smart Model → Average
GeoFractal Collective:   Different View + Different View + Different View → Triangulate
```

**Diagnostics & Proofs:**

See the diagnostic implementations and transfer learning experiments:

- [`src/geofractal/router/components/diagnostics/`](src/geofractal/components/diagnostics/) - Fusion diagnostics, frozen encoder tests, multi-tower stress tests
- [`studies/Router_Transfer_Learning-12_19_25.ipynb`](studies/Router_Transfer_Learning-12_19_25.ipynb) - Transfer learning experiments
- [`studies/InceptionFusionTowerResearch-12_23_25.ipynb`](studies/InceptionFusionTowerResearch-12_23_25.ipynb) - Inception tower architecture research
- [`studies/BaselineVisionConsistencyMeasure-12_24_25.ipynb`](studies/BaselineVisionConsistencyMeasure-12_24_25.ipynb) - A baseline test for vision model consistency measurement
---

## Direct Utilizable AI Helpers
To help curve the learning curve, we provide direct crash-course modules for popular AI models.

With this first document, Claude can help guide you to the current established practices for utilizing these models effectively and efficiently within the GeoFractal framework.

| Module                                       | Description                                                                            |
|----------------------------------------------|----------------------------------------------------------------------------------------|
| **ai_helpers.v101_claude_helpers.txt**       | Direct crash-course for claude to prep Claude for utilization.                         |
| **src.geofractal.router.GETTING_STARTED.md** | More human direction, AI can benefit from it if the AI drifts.                         |
| **src.geofractal.router.QUICK_BUILD.md**   | Quick reference for building with GeoFractal Router, AI can benefit from this as well. |

To get started with AI development, you can drag and drop v101_claude_helpers.txt into Claude, GPT, DeepSeek, or any other AI platform that supports text file input. 
This will help the AI understand how to effectively use the GeoFractal framework and its components.

Drag and drop the other two documents into the AI platform as well, to provide more context and guidance for building with GeoFractal Router if needed.
Drop the v101_claude_helpers.txt again if needed to reinforce the concepts. AI tends to drift and become forgetful.



## Core Concepts

| Concept | What It Is | Key Insight |
|---------|------------|-------------|
| **Router** | Coordination architecture | Collective intelligence through geometric routing |
| **Tower** | Self-encapsulated processing unit | Produces an *opinion*, not just an output |
| **Port** | Encoder wrapper with lifecycle | Standardized interface for any encoder |
| **WideRouter** | Compile-optimized router for wide models | Near-linear scaling with tower count |
| **ExecutionStrategy** | Batching strategy selector | VMAP, WIDE_COMPILER, AUTO, SEQUENTIAL |
| **CompileRouter** | Universal model compiler | Introspects any nn.Module for optimization |
| **VMapTowerGroup** | Vectorized tower executor | True batching via torch.func.vmap |
| **WidePrimitiveTowerGroup** | Wide primitive executor | Fused einsum/grouped-conv via wide_compiler registry |
| **SubEnsembleGroup** | Multi-vantage gradient pooling | Learnable interpolation across execution paths |
| **NotifierRouter** | Communication backbone | Routes messages based on geometry |
| **Collective** | Multi-tower ensemble | Triangulates truth from diverse perspectives |
| **Component** | Attachable unit with identity and lifecycle | Building block for routers and towers |
| **Address** | Geometric identity on a manifold | Fingerprints enable similarity/distance routing |
| **Fusion** | Opinion aggregation | Where emergence happens |
| **Walker** | Geometric interpolation system | Blend tensors along learned/static paths |
| **Cache** | Ephemeral tensor storage | Optional - only for towers exposing intermediates |

More routers, towers, components, and collective patterns are planned for immediate and future releases.

---

## Architecture

### Storage Model

Every router has three distinct storage mechanisms:

| Storage | Type | Device-Managed | In state_dict | Use For |
|---------|------|----------------|---------------|---------|
| `components` | `nn.ModuleDict` | ✅ Yes | ✅ Yes | nn.Module children |
| `objects` | `dict` | ❌ No | ❌ No | Config, metadata |
| `_cache` | `dict` | ❌ No | ❌ No | Ephemeral tensors (optional) |

```python
# components[] - Learnable modules (moved by .to(), saved in state_dict)
# Named components can be raw nn.Module OR TorchComponent
self.attach('encoder', nn.Linear(256, 512))  # OK as named component

# objects[] - Config and metadata (persistent, NOT tensors)
self.attach('config', {'dropout': 0.1, 'scale': 1.0})

# _cache - Ephemeral tensors for external retrieval (use only if needed)
self.cache_set('features', intermediate_tensor)
```

**⚠️ CRITICAL:** Never store tensors in `objects[]` - this causes memory leaks. If external code needs tensors after `forward()`, use `cache_set()`. For data used only within `forward()`, use local variables.

### The Component Hierarchy

GeoFractal has five base types: **BaseComponent**, **BaseRouter**, **BaseTower**, **WideRouter**, and **BasePort**:

```
BaseRouter (ABC - nn.Module)
│   - name, uuid
│   - components: nn.ModuleDict (learnable children)
│   - objects: dict (config, metadata)
│   - _cache: dict (ephemeral tensors)
│   - Lifecycle: attach(), detach(), reset()
│
├── BaseTower (BaseRouter + stages)
│       - stages: nn.ModuleList (ordered pipeline)
│       - Dual indexing: tower[0] (stage), tower['name'] (component)
│       - Produces opinions
│
├── WideRouter (BaseRouter + wide execution)
│       - Tower registration and discovery
│       - wide_forward() for vectorized execution
│       - ExecutionStrategy: VMAP, WIDE_COMPILER, AUTO, SEQUENTIAL
│       - VMapTowerGroup for vmap-based batching
│       - WidePrimitiveTowerGroup for wide_compiler primitive fusion
│       - SubEnsembleGroup for multi-vantage gradient interpolation
│       - CacheDeviceController for device-aware cache lifecycle
│       - GradientDebugger for multi-level gradient monitoring
│       - torch.compile integration
│
├── CompileRouter (BaseRouter + introspection)
│       - Module tree introspection
│       - Signature-based grouping
│       - VMap group building
│       - build_wide_router() for WideRouter generation
│
└── NotifierRouter (BaseRouter + messaging)
        - Geometric message routing
        - Channel-based communication

BaseComponent (ABC - pure Python)
│   - name, uuid, parent
│   - Lifecycle: on_attach(), on_detach()
│
└── TorchComponent (BaseComponent + nn.Module)
        - Learnable parameters
        - Device affinity (home_device, allowed_devices)
        │
        ├── AddressComponent      # Geometric identity, fingerprints
        ├── FusionComponent       # Combine opinions
        ├── WalkerFusion          # Geometric interpolation
        └── ProjectionComponent   # Transform shapes

VMapTowerGroup (nn.Module)
        - Vectorized tower execution
        - Uses torch.func.vmap + stack_module_state
        - Lazy parameter stacking with cache invalidation

WidePrimitiveTowerGroup (nn.Module)
        - Operation-level fusion via wide_compiler primitives
        - Stage classification: fully_fused / partially_fused / opaque
        - Einsum-based linear, grouped-conv, manual LayerNorm kernels
        - Three-tier fallback: Wide kernel → vmap → sequential
        - Towers own parameters (no copying, no sync)
        - Training: re-stacks per-forward; eval: cached stacked params

SubEnsembleGroup (nn.Module)
        - Multiple execution paths over same tower group
        - Learnable interpolation weights (softmax-normalized)
        - Gradients flow to ALL execution paths
        - Gradient diversity from different computational vantage points
```

### Port Hierarchy

Ports wrap encoders with standardized lifecycle and data flow:

```
BasePort (ABC - pure protocol, no torch)
│   - preprocess(raw) → prepared
│   - encode(prepared) → encoded
│   - postprocess(encoded) → output
│   - load() / unload()
│
└── TorchPort (BasePort + device/dtype management)
        - Device movement: to(), cuda(), cpu()
        - Dtype control: half(), float(), bfloat16()
        - Gradient control: freeze(), unfreeze()
        │
        ├── QwenPort      # Qwen2, Qwen2.5, Instruct
        ├── DINOPort      # DINOv1, DINOv2
        ├── CLIPPort      # CLIP text/vision
        └── VAEPort       # Latent encoders
```

### WideRouter: Compile-Optimized Wide Models

**WideRouter** is designed for collectives with many towers processing the same input. It supports multiple execution strategies for batched tower execution, with automatic fallback chains and optional multi-vantage gradient interpolation:

| Towers | Time | Per-Tower |
|--------|------|-----------|
| 4 | 1.06ms | 265µs |
| 8 | 1.89ms | 237µs |
| 16 | 3.96ms | 248µs |
| 32 | 7.27ms | 227µs |

```python
from geofractal.router.wide_router import WideRouter, ExecutionStrategy


class MyCollective(WideRouter):
    def __init__(self, name: str, num_towers: int, dim: int):
        super().__init__(name, auto_discover=True,
                         execution_strategy=ExecutionStrategy.AUTO)

        for i in range(num_towers):
            self.attach(f'tower_{i}', ExpertTower(f'tower_{i}', dim))

        self.discover_towers()  # Register for wide execution
        self.attach('fusion', AdaptiveFusion('fusion', num_towers, dim))

    def forward(self, x: Tensor) -> Tensor:
        opinions = self.wide_forward(x)  # Batched tower execution
        return self['fusion'](*opinions.values())


# Usage
collective = MyCollective('wide', num_towers=16, dim=256)
compiled = collective.prepare_and_compile()  # Analyze + compile
output = compiled(x)  # 1.4x faster than eager
```

**Execution Strategies:**

| Strategy | Behavior | Best For |
|----------|----------|----------|
| `VMAP` | `torch.func.vmap` vectorization (default) | General use, compiled training |
| `WIDE_COMPILER` | Wide primitive kernels (einsum, grouped conv) | Compiled inference |
| `AUTO` | Training: VMAP, Eval: WIDE_COMPILER → VMAP → sequential | Adaptive |
| `SEQUENTIAL` | Direct per-tower execution | Debugging, compatibility |

**Key features:**
- **Auto-discovery**: Finds all `BaseTower` instances automatically
- **Structure analysis**: Identifies aligned operations for fusion
- **Dual execution engines**: VMap batching + Wide primitive kernels with automatic fallback
- **Operation-level fusion**: Linear, LayerNorm, Conv layers fused via `wide_compiler` registry
- **Fusion coverage tracking**: Reports what % of ops actually fused per tower group
- **Sub-ensemble pooling**: `build_sub_ensembles()` creates multi-vantage gradient interpolation
- **Compile-safe**: All preparation is `@torch.compiler.disable`; forward uses only compile-safe ops
- **Near-linear scaling**: Per-tower cost *decreases* with more towers
- **Gradient debugging**: Multi-level monitoring (per-tower norms, anomaly detection)
- **Cache device controller**: Policies for cache lifecycle across device moves (`clear`, `migrate`, `reconstruct`)

### CompileRouter: Universal Model Compilation

**CompileRouter** introspects *any* `nn.Module` and optimizes it for `torch.compile`. It doesn't require you to restructure your code into the GeoFractal hierarchy.

```python
from geofractal.router.compiler import CompileRouter, compile_module

# Any messy PyTorch model
class SloppyModel(nn.Module):
    def __init__(self):
        self.stuff = nn.ModuleList([nn.Linear(256, 256) for _ in range(8)])
        self.thing = nn.Sequential(nn.Linear(256, 512), nn.GELU())
        # ... arbitrarily nested
    
    def forward(self, x):
        for s in self.stuff:
            x = x + s(x)
        return self.thing(x)

# One-liner: analyze + stage + compile
compiler = compile_module(SloppyModel(), "sloppy")
compiled = compiler.compile(mode='reduce-overhead')

# Or with analysis visibility
compiler = CompileRouter.from_module(model)
compiler.introspect()
compiler.compile_towers()
compiler.print_stages()  # See what's batchable
print(compiler.get_compilation_stats())
```

**What it does:**
1. **Introspects** the module tree recursively
2. **Categorizes** modules (attention, linear, conv, norm, etc.)
3. **Groups by signature** - modules with identical structure
4. **Identifies batchable stages** - groups that can benefit from vmap
5. **Builds VMapTowerGroups** for true vectorized execution

**Integration with WideRouter:**
```python
# Build a WideRouter from any model's structure
compiler = CompileRouter.from_module(complex_model)
wide = compiler.build_wide_router()  # Returns CompiledWideRouter
compiled = torch.compile(wide, mode='reduce-overhead')
```

### The Collective Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                        Collective                               │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Tower A   │  │   Tower B   │  │   Tower C   │             │
│  │ + Address   │  │ + Address   │  │ + Address   │             │
│  │ (+ _cache)  │  │ (+ _cache)  │  │ (+ _cache)  │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ↓                                      │
│              wide_forward() / NotifierRouter                    │
│                          ↓                                      │
│              (cache_clear() if towers use cache)                │
│                          ↓                                      │
│              FusionComponent (aggregate opinions)               │
│                          ↓                                      │
│                    Collective Output                            │
└─────────────────────────────────────────────────────────────────┘
```

*Note: `_cache` is optional - only towers exposing intermediates use it.*

---

## Quick Start

### Installation

```bash
git clone https://github.com/AbstractPhil/geofractal.git
cd geofractal
pip install -e .
```

### Build a Wide Collective

```python
import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.wide_router import WideRouter
from geofractal.router.base_tower import BaseTower
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.components.fusion_component import AdaptiveFusion


class FFNBlock(TorchComponent):
    """Feed-forward block as a component."""
    
    def __init__(self, name: str, dim: int, expansion: int = 2):
        super().__init__(name)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SimpleTower(BaseTower):
    def __init__(self, name: str, dim: int):
        super().__init__(name, strict=False)
        for i in range(2):
            self.append(FFNBlock(f'{name}_ffn_{i}', dim))
        self.attach('norm', nn.LayerNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = x + stage(x)
        return self['norm'](x)


class WideCollective(WideRouter):
    def __init__(self, name: str, dim: int, num_towers: int = 8):
        super().__init__(name, auto_discover=True)

        for i in range(num_towers):
            self.attach(f'tower_{i}', SimpleTower(f'tower_{i}', dim))

        self.discover_towers()
        self.attach('fusion', AdaptiveFusion('fusion', num_towers, dim))

    def forward(self, x: Tensor) -> Tensor:
        opinions = self.wide_forward(x)
        # SimpleTower doesn't use cache, so no clearing needed
        return self['fusion'](*opinions.values())


# Create, move to GPU, compile
torch.set_float32_matmul_precision('high')
collective = WideCollective('wide', dim=256, num_towers=16)
collective.network_to(device='cuda')
compiled = collective.prepare_and_compile()

x = torch.randn(32, 64, 256, device='cuda')
output = compiled(x)  # ~1.4x faster than eager
```

### Using Encoder Ports

```python
from geofractal.router.ports import QwenPort

# Create and load
port = QwenPort('qwen', 'Qwen/Qwen2.5-1.5B-Instruct', pool='last')
port.load()

# Single input → [D]
embedding = port('a cat sitting on a mat')

# Batch input → [B, D]
embeddings = port(['hello', 'world', 'test'])

# Device management
port.to('cpu')
port.half()

# Cleanup
port.unload()
```

---

## Cache System

### When Cache Is Used

**Most towers don't need cache.** Cache is only for towers that expose intermediates to external code:

| Tower Type | Uses Cache? | Why |
|------------|-------------|-----|
| Simple feedforward | ❌ No | No external access needed |
| Residual tower | ❌ No | Residual is local variable |
| `ConfigurableTower` | ✅ Yes | Exposes features to collective |
| `ConfigurableConvTower` | ✅ Yes | Exposes features to collective |
| Custom tower exposing intermediates | ✅ Yes | External code retrieves features |

### Why Cache Exists

The cache system prevents VRAM memory leaks in towers that *do* expose intermediates:

```python
# ❌ OLD (LEAKED ~33MB per tower per forward)
self.objects['_cached_features'] = features  # Never cleared!

# ✅ NEW (Managed lifecycle)
self.cache_set('features', features)  # Collective clears after retrieval
```

### Cache API

| Method | Description |
|--------|-------------|
| `cache_set(key, value)` | Store tensor in ephemeral cache |
| `cache_get(key, default=None)` | Retrieve from cache |
| `cache_clear()` | Clear this router's cache only |
| `cache_clear_recursive()` | Clear entire router tree |
| `cache_keys()` | List current cache keys |
| `cache_size_bytes()` | Estimate VRAM held in cache |
| `cache_to(device, dtype)` | Explicitly move cache tensors |
| `cache_debug(prefix='')` | Debug cache state across tree |
| `reset()` | Clear cache recursively (call before device moves) |

### When to Use Cache vs Local Variables

| Situation | Use | Example |
|-----------|-----|---------|
| Residual within same `forward()` | Local variable | `residual = x` |
| Gate computed and used in same `forward()` | Local variable | `gate = self['gate'](x)` |
| Features needed by Collective after `forward()` | Cache | `self.cache_set('features', x)` |
| Intermediates retrieved by WideRouter | Cache | ConfigurableTower pattern |

**Rule of thumb:** If the data never leaves `forward()`, use a local variable. If external code needs it after `forward()` returns, use cache.

```python
# Simple tower - NO cache needed
class ResidualTower(BaseTower):
    def forward(self, x: Tensor) -> Tensor:
        residual = x  # Local variable - used only here
        for stage in self.stages:
            x = stage(x)
        return x + residual  # No cache involved

# Tower exposing features - uses cache
class FeatureExposingTower(BaseTower):
    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        
        # Cache because collective retrieves this after forward()
        self.cache_set('features', x)
        return self['output_proj'](x)
    
    @property
    def cached_features(self):
        return self.cache_get('features')
```

### Debugging Memory Issues

```python
# Check cache state across entire model
debug_info = model.cache_debug()
for path, cache in debug_info.items():
    print(f"{path}: {list(cache.keys())}")

# If towers use cache, it should be empty between batches
# (after collective clears it)
# If towers don't use cache, this is already empty

# Force clear everything (safe, no-op if already empty)
model.reset()
```

---

## Device Movement

### network_to() vs .to()

| Method | Cache Behavior | Use When |
|--------|----------------|----------|
| `.to(device)` | ❌ Not moved | Quick testing |
| `network_to(device)` | 🗑️ Cleared by default | Production |

```python
# Standard PyTorch - cache NOT moved (unsafe)
model.to('cuda:1')

# Router-aware - cache cleared by default (safe)
model.network_to(device='cuda:1')

# Explicit cache control
model.network_to(device='cuda:1', clear_cache=False)
model.cache_to_recursive(device='cuda:1')  # Manual move
```

### Accelerate Compatibility

```python
# ✅ Recommended pattern
model.reset()  # Clear all caches first
model = accelerate.prepare(model)

# ❌ Risky - cache on wrong device
model = accelerate.prepare(model)
model(x)  # Cache created
model.network_to('cpu')  # Cache stays on GPU!
```

---

## Router Types

| Router | Purpose | Best For |
|--------|---------|----------|
| `BaseRouter` | Abstract base | Custom routing logic |
| `BaseTower` | Ordered stage processing | Individual expert units |
| `NotifierRouter` | Geometric message routing | Tower coordination |
| `WideRouter` | Compile-optimized execution | Many towers (4+) |

### When to Use WideRouter

**Use WideRouter when:**
- You have 4+ towers with identical structure
- All towers process the same input
- You want maximum throughput via `torch.compile`
- Scaling efficiency matters

**Use BaseRouter when:**
- Towers have different structures
- Towers process different inputs
- You need fine-grained control over execution order

---

## Critical Dos and Don'ts

### ✅ DO

```python
# Use cache ONLY for tensors needed by external code after forward()
self.cache_set('features', features)  # Collective will retrieve this

# Clear cache in collective IF towers use cache
if towers_use_cache:
    self.clear_tower_caches()

# Call reset() before device changes (safe even if cache is empty)
model.reset()
model.network_to(device='cuda:1')

# Use network_to() for production
model.network_to(device='cuda', dtype=torch.float16)

# Use local variables for forward()-scoped data
residual = x  # Only used within this forward()

# Put config in objects[]
self.attach('config', {'scale': 1.0})

# Call discover_towers() after attaching towers
self.discover_towers()

# Use prepare_and_compile() for WideRouter
compiled = collective.prepare_and_compile()
```

### ❌ DON'T

```python
# Store tensors in objects[] - MEMORY LEAK!
self.objects['features'] = features

# Use cache for data only needed within forward()
def forward(self, x):
    self.cache_set('residual', x)  # Wrong! Use local variable
    ...
    return x + self.cache_get('residual')

# Forget to clear cache IF you use it
def forward(self, x):
    self.cache_set('features', tensor)  # For external access
    return output  # Collective must clear this!

# Assume .to() moves cache
model.to('cuda:1')  # Cache stays on old device!

# Use raw torch.compile() on WideRouter
compiled = torch.compile(collective)  # May fail
```

---

## Key Principles

1. **Three Storage Types** - `components` (modules), `objects` (config), `_cache` (ephemeral tensors)
2. **Never Tensor in objects[]** - Use `cache_set()` if external access needed, local variable otherwise
3. **Cache Is Optional** - Only towers exposing intermediates need it
4. **Local Variables First** - Use cache only when data must persist after `forward()`
5. **Stages Are Components** - Not raw primitives
6. **Towers Produce Opinions** - Local conclusions, not final answers
7. **Use network_to()** - Safe device movement with cache clearing
8. **Divergence Over Accuracy** - See differently, triangulate truth
9. **Compile First for Wide Models** - Use `prepare_and_compile()` or `CompileRouter`
10. **VMap Over For-Loops** - Use `VMapTowerGroup` for true vectorized batching
11. **ExecutionStrategy for Control** - `AUTO` adapts between training (VMAP) and inference (WIDE_COMPILER)
12. **Wide Primitives for Inference** - `WidePrimitiveTowerGroup` fuses Linear/LayerNorm/Conv via einsum

---

## Documentation

| Document | Description |
|----------|-------------|
| [QUICK_BUILD.md](src/geofractal/router/QUICK_BUILD.md) | Cheat sheet for rapid development |
| [GETTING_STARTED.md](src/geofractal/router/GETTING_STARTED.md) | Complete tutorial with cache system |

---

## Changelog

### v1.2.0 (2026-03-31)

**Wide Compiler Integration** - Operation-Level Fusion via `pytorch-parallel-compiler`

- **`ExecutionStrategy` enum**: `VMAP`, `WIDE_COMPILER`, `SEQUENTIAL`, `AUTO` — controls how aligned towers are batched
- **`WidePrimitiveTowerGroup`**: Walks tower stages and fuses compatible layers (Linear, LayerNorm, Conv) using `wide_compiler` registry primitives (einsum, grouped conv). Three-tier fallback: Wide kernel → vmap → sequential. Towers own parameters; group stacks views at prepare time via `torch.stack`. Gradients flow back through the stack to originals.
- **`SubEnsembleGroup`**: Runs both VMap and Wide execution paths on the same tower group, with learnable interpolation weights (softmax-normalized). Gradients flow to ALL paths for multi-vantage gradient diversity.
- **`CacheDeviceController`**: Manages cache lifecycle across device moves with three policies: `clear` (default, current behavior), `migrate` (move caches to new device), `reconstruct` (clear and mark for lazy rebuild). Validates cache device consistency.
- **`GradientDebugger`**: Multi-level gradient monitoring. Level 0: off. Level 1: per-tower gradient norms after backward. Level 2+: stage norms, anomaly detection (NaN, Inf, dead, exploding gradients).
- **`FusionCoverage` tracking**: Reports what percentage of ops fused per tower group (e.g., "4/6 ops fused (67%), 2 vmap, 0 sequential")
- **`StagePlan` / `OpPlan`**: Stage classification system — fully_fused, partially_fused, opaque. Opaque stages (custom forward with residuals) use `vmap(functional_call)` instead of decomposition.

**New WideRouter constructor parameters** (all backwards-compatible defaults):
- `execution_strategy=ExecutionStrategy.VMAP`
- `gradient_debug_level=0`
- `cache_preservation_policy='clear'`

**New WideRouter methods:**
- `prepare_for_compile(execution_strategy=..., build_sub_ensembles=...)` — extended signature
- `build_sub_ensembles()` — creates dual-path sub-ensembles
- `wide_forward_ensemble(x)` — sub-ensemble-aware forward
- `gradient_report()` — human-readable gradient health report
- `check_fusion_health()` — Wide primitive gradient diagnostics
- `get_fusion_coverage()` — per-group fusion stats
- `validate_cache_devices()` — cache device mismatch checker

**Behavioral changes:**
- `_batched_tower_forward()` now dispatches based on `ExecutionStrategy` instead of always using VMap. Default `VMAP` preserves existing behavior.
- `network_to()` now brackets with `CacheDeviceController` pre/post hooks. Default `clear` policy preserves existing behavior.
- `reset()` also clears `_wide_primitive_groups` and sub-ensemble gradient tracking.
- `get_wide_stats()` returns additional fields: `execution_strategy`, `wide_primitive_groups`, `sub_ensembles`, `fusion_coverage`.
- `__repr__` includes strategy name and wide group count.
- `compile()` and `prepare_and_compile()` accept `build_sub_ensembles` parameter.

**Staleness note:** `torch.stack` creates new tensors, not views. During training, `WidePrimitiveTowerGroup` re-stacks every forward to pick up optimizer updates (causes graph break under `torch.compile`). For compiled training, prefer `ExecutionStrategy.VMAP`. For compiled inference, use `WIDE_COMPILER`. `AUTO` handles this automatically.

**Deprecations:** None. All changes are additive with backwards-compatible defaults.

### v1.1.0 (2025-12-29)

**CompileRouter** - Introspective Compilation System

- **New `CompileRouter`**: Auto-discovers, wraps, and stages arbitrary `nn.Module` structures for optimized execution
- **Module introspection**: Categorizes modules (attention, linear, conv, norm, gating, pooling, embedding)
- **Execution staging**: Groups modules by signature for batching opportunities
- **`compile_module()`**: One-liner standalone wrapper for any model
- **`build_vmap_groups()`**: Creates VMapTowerGroups from batchable stages
- **`build_wide_router()`**: Generates CompiledWideRouter with proper tower registration

```python
from geofractal.router.compiler import CompileRouter, compile_module

# One-liner compilation
compiler = compile_module(any_model, "my_model")
compiled = compiler.compile(mode='reduce-overhead')

# Or step-by-step with analysis
compiler = CompileRouter.from_module(model)
compiler.introspect()
compiler.compile_towers()
compiler.print_stages()  # See batching opportunities
```

**VMapTowerGroup** - True Vectorized Batching

- **Replaces fake for-loop batching** with real vectorized execution via `torch.func.vmap`
- **Uses `stack_module_state()`** to batch parameters across identical-signature towers
- **Uses `functional_call()`** for efficient parameter-batched forward passes
- **Lazy caching**: Stacked params/buffers cached until invalidated by `train()` or `to()`

```python
from torch.func import vmap, functional_call, stack_module_state

# Old: Sequential (just a for loop)
for tower in towers:
    results[name] = tower(x)

# New: True vectorized execution
params, buffers = stack_module_state(towers)
vmapped_forward = vmap(single_forward, in_dims=(0, 0, None))
outputs = vmapped_forward(params, buffers, x)  # ONE operation
```

**WideRouter Enhancements**

- **New `_batched_tower_forward()`**: Uses VMapTowerGroup for genuine parallel execution
- **VMap group caching**: Groups cached by `(signature, frozenset(tower_names))`
- **Cache invalidation**: Cleared on `register_tower()`, `unregister_tower()`, `reset()`
- **Integration with CompileRouter**: `build_wide_router()` produces properly configured WideRouter

**Walker Fusion System** - Geometric Interpolation

- **`ConfigurableWalker`**: Static composition of blend/schedule/aggregation functions (NOT nn.Module)
- **`WalkerInception`**: Optional learned modulation (~20k params, TorchComponent)
- **`WalkerFusion`**: Interface wrapper housing walker + optional inception
- **Preset walkers**: `shiva`, `slerp`, `lerp`, `slip`, `zeus`, `gilgamesh`
- **Aux types**: `cosine`, `geometric`, `learned`, `walker_path`

```python
from geofractal.router.components.walker_component import (
    WalkerFusion, WalkerInception, create_walker_fusion
)

# Static walker (no learning)
fusion = WalkerFusion("walk", in_features=512, preset='shiva')

# With learned modulation
inception = WalkerInception("inc", in_features=512, num_steps=8, aux_type='cosine')
fusion = WalkerFusion("walk", in_features=512, preset='shiva', inception=inception)

# Factory functions
static = create_walker_fusion("s", 512, preset='shiva')
learned = create_walker_fusion("l", 512, preset='shiva', with_inception=True)
```

**Fusion System Updates**

- **`AdaptiveBindingFusion`** (Lyra): Full binding system with mask + visibility + boost
- **`CantorScaleFusion`**: Fractal geometry routing with Cantor set mathematics
- **`HierarchicalTreeGating`**: Tree-structured gating for deep fusion
- **`FusionBuilder`**: Mirrors ConfigurableTower pattern for fusion construction
- **`FusionCollective`**: Multi-fusion ensemble with strategy selection

**Benchmark Tools**

- **`compile_benchmark.py`**: Comprehensive benchmark comparing:
  - Eager execution (baseline)
  - `torch.compile` (standard)
  - `torch.compile` with `fullgraph=True`
  - VMap WideRouter (vectorized batching)
  - VMap WideRouter + `torch.compile` (best of both)
- **`benchmark_model()`**: Standalone function for benchmarking any model

```bash
python compile_benchmark.py --towers 8 --depth 4 --dim 512
```

### v1.0.1 (2025-12-23)

**Cache System** - Managed ephemeral tensor storage

- **New `_cache` dict** on all routers for intermediate tensors
- **Cache API**: `cache_set()`, `cache_get()`, `cache_clear()`, `cache_clear_recursive()`
- **Debug tools**: `cache_debug()`, `cache_size_bytes()`, `cache_keys()`
- **Device safety**: `cache_to()`, `cache_to_recursive()`
- **Updated `reset()`**: Now clears cache recursively
- **Updated `network_to()`**: New `clear_cache=True` parameter (default)

**Memory Leak Fix** - Eliminated ~268MB/forward VRAM leak

- Fixed `objects['_cached_features']` → `cache_set('features', ...)`
- Auto-clearing in `ConfigurableCollective.forward()` and `ConvTowerCollective.forward()`
- New `WideRouter.clear_tower_caches()` method

**Multi-Channel VAE Support** - Direct latent processing

- **FlexibleInputComponent**: Handles `[B,C,H,W]` (spatial) or `[B,L,D]` (sequence) inputs
- **MultiScaleConvBlock**: Local/regional/global feature extraction with SE attention
- **ChannelMixerBlock**: Cross-channel attention for multi-channel latents
- **New presets**: `preset_flux_vae_towers()` (16-ch), `preset_sd_vae_towers()` (4-ch)
- **ConvTowerConfig options**: `in_channels`, `input_mode`, `pool_mode`, `use_channel_mixer`

**Documentation** - Comprehensive updates

- New GETTING_STARTED.md sections: Storage Types, Cache Control, Device Movement, Dos/Don'ts

### v1.0.0-beta (2025-12-23)

**Port System** - Standardized encoder integration

- **BasePort**: Pure protocol for data-in → data-out with lifecycle
- **TorchPort**: Torch-specific base with device/dtype management
- **QwenPort**: Full Qwen family support with proper pooling

**WideRouter** - Compile-optimized wide models

- Auto-discovery of aligned towers
- `prepare_and_compile()` for safe compilation
- Near-linear scaling benchmarks

### v0.2.1

- WideRouter compile optimizations
- BaseTower stage management
- TorchComponent device affinity

### v0.1.0

- Initial release
- BaseRouter, BaseTower, NotifierRouter
- Component hierarchy
- Geometric addressing

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

*"Individual towers don't need to be accurate. They need to see differently. The routing fabric triangulates truth from divergent viewpoints."*