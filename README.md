# GeoFractal Router

**Collective Intelligence through Geometric Routing**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()

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

## Core Concepts

| Concept | What It Is | Key Insight |
|---------|------------|-------------|
| **Compiles** | Device affinity management capable of torch compilation | Multi-GPU deployment made easy |
| **Router** | Coordination architecture | Collective intelligence through geometric routing |
| **Tower** | Self-encapsulated processing unit | Produces an *opinion*, not just an output |
| **Port** | Encoder wrapper with lifecycle | Standardized interface for any encoder |
| **WideRouter** | Compile-optimized router for wide models | Near-linear scaling with tower count |
| **NotifierRouter** | Communication backbone | Routes messages based on geometry |
| **Collective** | Multi-tower ensemble | Triangulates truth from diverse perspectives |
| **Component** | Attachable unit with identity and lifecycle | Building block for routers and towers |
| **Address** | Geometric identity on a manifold | Fingerprints enable similarity/distance routing |
| **Fusion** | Opinion aggregation | Where emergence happens |

More routers, towers, components, and collective patterns are planned for immediate and future releases.

---

## Architecture

### The Component Hierarchy

GeoFractal has five base types: **BaseComponent**, **BaseRouter**, **BaseTower**, **WideRouter**, and **BasePort**:

```
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

**Port Data Flow:**

```
INPUT (unknown device)
    │
    ▼
preprocess() ──────── CPU (tokenization, normalization)
    │
    ▼
[cache check] ─────── External (PyArrow, datasets)
    │
    ▼
encode() ──────────── GPU (encoder's device)
    │
    ▼
postprocess() ─────── GPU (pooling, projection)
    │
    ▼
[cache store] ─────── External (PyArrow, datasets)
    │
    ▼
OUTPUT (target device)
```

### Router Hierarchy

```
BaseRouter (ABC - nn.Module)
│   - Component container with strict hardware control
│   - network_to() for recursive device movement
│
├── BaseTower (BaseRouter)
│       - Ordered stages (nn.ModuleList)
│       - Produces opinions
│
├── NotifierRouter (BaseRouter)
│       - Geometric message routing
│       - Channel-based communication
│
└── WideRouter (BaseRouter)
        - Compile-optimized for wide models
        - Auto-discovers aligned towers
        - Near-linear scaling with torch.compile
```

### WideRouter: Compile-Optimized Wide Models

**WideRouter** is designed for collectives with many towers processing the same input. It leverages `torch.compile` for kernel fusion, achieving near-linear scaling:

| Towers | Time | Per-Tower |
|--------|------|-----------|
| 4 | 1.03ms | 258µs |
| 8 | 1.90ms | 238µs |
| 16 | 3.63ms | 227µs |
| 32 | 7.21ms | 225µs |

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

### The Collective Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                        Collective                               │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Tower A   │  │   Tower B   │  │   Tower C   │             │
│  │ + Address   │  │ + Address   │  │ + Address   │             │
│  │ (Spherical) │  │ (Spherical) │  │ (Hyperbolic)│             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ↓                                      │
│              NotifierRouter (geometric routing)                 │
│                          ↓                                      │
│              FusionComponent (aggregate opinions)               │
│                          ↓                                      │
│                    Collective Output                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/AbstractPhil/geofractal.git
cd geofractal
pip install -e .
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

### Build a Wide Collective (Recommended)

```python
from geofractal.router.wide_router import WideRouter
from geofractal.router.base_tower import BaseTower
from geofractal.router.components.fusion_component import AdaptiveFusion


class SimpleTower(BaseTower):
    def __init__(self, name: str, dim: int):
        super().__init__(name, strict=False)
        for i in range(2):
            self.append(nn.Sequential(
                nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
            ))
        self.attach('norm', nn.LayerNorm(dim))

    def forward(self, x):
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

    def forward(self, x):
        opinions = self.wide_forward(x)
        return self['fusion'](*opinions.values())


# Create, move to GPU, compile
collective = WideCollective('wide', dim=256, num_towers=16)
collective.network_to(device='cuda')
compiled = collective.prepare_and_compile()

x = torch.randn(32, 64, 256, device='cuda')
output = compiled(x)  # ~1.5x faster than eager
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

## Key Principles

1. **Five Base Types** - BaseComponent, BaseRouter, BaseTower, WideRouter, BasePort
2. **Stages Are Components** - Not raw primitives
3. **Towers Produce Opinions** - Local conclusions, not final answers
4. **Ports Wrap Encoders** - Standardized lifecycle for any encoder
5. **Geometric Routing** - Manifold-based coordination
6. **Divergence Over Accuracy** - See differently, triangulate truth
7. **Compile First for Wide Models** - Let `torch.compile` handle fusion

---

## Documentation

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](src/geofractal/router/GETTING_STARTED.md) | Complete tutorial |

---

## Changelog

### v1.0.0 (2025-12-23) Beta Release

**Port System** - Standardized encoder integration

- **BasePort**: Pure protocol for data-in → data-out with lifecycle
  - `preprocess(raw) → prepared` (CPU, no device movement)
  - `encode(prepared) → encoded` (GPU, returns context for postprocess)
  - `postprocess(encoded) → output` (proper pooling with attention mask)
  - `load()` / `unload()` lifecycle management

- **TorchPort**: Torch-specific base with device/dtype management
  - `to(device, dtype)` - move encoder
  - `cuda()`, `cpu()`, `half()`, `float()`, `bfloat16()` - convenience methods
  - `freeze()` / `unfreeze()` - gradient control
  - Automatic VRAM cleanup on unload

- **QwenPort**: Full Qwen family support
  - Qwen2, Qwen2.5, Instruct variants
  - Proper pooling with attention mask (last, first, mean, max)
  - Chat template support for Instruct models
  - Single input → `[D]`, batch input → `[B, D]`

**Architecture designed for external caching** - Ports handle encode pipeline, caching handled by composition (CachedPort, DatasetPort, PyArrow/datasets integration).

### v0.2.1

- WideRouter compile optimizations
- BaseTower stage management
- TorchComponent device affinity
- Large system implementations
- Bug fixes and refactors
- Documentation updates pre-release

### v0.1.0

- Initial release
- BaseRouter, BaseTower, NotifierRouter
- Component hierarchy
- Geometric addressing

---

## License

Apache License 2.0 with Attribution

---

*"Individual towers don't need to be accurate. They need to see differently. The routing fabric triangulates truth from divergent viewpoints."*