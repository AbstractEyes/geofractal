# GeoFractal Router

**Collective Intelligence through Geometric Routing**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0.1-green.svg)]()

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

- [`src/geofractal/router/diagnostics/`](src/geofractal/router/diagnostics/) - Fusion diagnostics, frozen encoder tests, multi-tower stress tests
- [`src/geofractal/router/Router_Transfer_Learning-12_19_25.ipynb`](src/geofractal/router/Router_Transfer_Learning-12_19_25.ipynb) - Transfer learning experiments

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
| **NotifierRouter** | Communication backbone | Routes messages based on geometry |
| **Collective** | Multi-tower ensemble | Triangulates truth from diverse perspectives |
| **Component** | Attachable unit with identity and lifecycle | Building block for routers and towers |
| **Address** | Geometric identity on a manifold | Fingerprints enable similarity/distance routing |
| **Fusion** | Opinion aggregation | Where emergence happens |
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
│       - wide_forward() for batched execution
│       - torch.compile integration
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
        └── ProjectionComponent   # Transform shapes
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

**WideRouter** is designed for collectives with many towers processing the same input. It leverages `torch.compile` for kernel fusion, achieving near-linear scaling:

| Towers | Time | Per-Tower |
|--------|------|-----------|
| 4 | 1.06ms | 265µs |
| 8 | 1.89ms | 237µs |
| 16 | 3.96ms | 248µs |
| 32 | 7.27ms | 227µs |

```python
from geofractal.router.wide_router import WideRouter


class MyCollective(WideRouter):
    def __init__(self, name: str, num_towers: int, dim: int):
        super().__init__(name, auto_discover=True)

        for i in range(num_towers):
            self.attach(f'tower_{i}', ExpertTower(f'tower_{i}', dim))

        self.discover_towers()  # Register for wide execution
        self.attach('fusion', AdaptiveFusion('fusion', num_towers, dim))

    def forward(self, x: Tensor) -> Tensor:
        opinions = self.wide_forward(x)  # Batched tower execution
        
        # If towers cache intermediates for retrieval, clear after use:
        # self.clear_tower_caches()
            
        return self['fusion'](*opinions.values())


# Usage
collective = MyCollective('wide', num_towers=16, dim=256)
compiled = collective.prepare_and_compile()  # Analyze + compile
output = compiled(x)  # 1.4x faster than eager
```

**Key features:**
- **Auto-discovery**: Finds all `BaseTower` instances automatically
- **Structure analysis**: Identifies aligned operations for fusion
- **Compile-safe**: Separates Python bookkeeping from tensor hot path
- **Near-linear scaling**: Per-tower cost *decreases* with more towers
- **Cache management**: `reset()` and `clear_tower_caches()` available if towers use cache

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
9. **Compile First for Wide Models** - Let `torch.compile` handle fusion

---

## Documentation

| Document | Description |
|----------|-------------|
| [QUICK_BUILD.md](src/geofractal/router/QUICK_BUILD.md) | Cheat sheet for rapid development |
| [GETTING_STARTED.md](src/geofractal/router/GETTING_STARTED.md) | Complete tutorial with cache system |

---

## Changelog

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