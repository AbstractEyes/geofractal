# GeoFractal Router

**Collective Intelligence through Geometric Routing**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## What Is This?

GeoFractal Router is a coordination architecture for building **collectives of autonomous AI units**. Instead of one monolithic model, you build multiple *towers* that produce opinions, coordinate through *geometric routing*, and fuse their perspectives into emergent collective intelligence.

**The key insight:** Individual units don't need to be accurate. They need to *see differently*. The collective triangulates truth from divergent viewpoints.

```
Traditional Ensemble:    Smart Model + Smart Model + Smart Model → Average
GeoFractal Collective:   Different View + Different View + Different View → Triangulate
```

**Proven emergence:**

| Experiment | Individual Accuracy | Collective Accuracy | Multiplier |
|------------|---------------------|---------------------|------------|
| ImageNet (5 streams) | 0.1% each | 84.68% | **847×** |
| FashionMNIST (3 streams) | 10% each | 93.4% | **9.34×** |
| Dual CLIP (frozen) | 7-18% | 92.6% | **5-13×** |

The collective achieves what no individual can.

---

## Core Concepts

| Concept | What It Is | Key Insight |
|---------|------------|-------------|
| **Compiles** | Device affinity management capable of torch compilation | Multi-GPU deployment made easy |
| **Router** | Coordination architecture | Collective intelligence through geometric routing |
| **Tower** | Self-encapsulated processing unit | Produces an *opinion*, not just an output |
| **WideRouter** | Compile-optimized router for wide models | Near-linear scaling with tower count |
| **NotifierRouter** | Communication backbone | Routes messages based on geometry |
| **Collective** | Multi-tower ensemble | Triangulates truth from diverse perspectives |
| **Component** | Attachable unit with identity and lifecycle | The building block - everything is a component |
| **Address** | Geometric identity on a manifold | Fingerprints enable similarity/distance routing |
| **Fusion** | Opinion aggregation | Where emergence happens |

More routers, towers, components, and collective patterns are planned for immediate and future releases.

---

## Architecture

### The Component Hierarchy

Everything in GeoFractal is a **component** - an attachable unit with identity and lifecycle:

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

1. **Components, Not Modules** - Everything has identity and lifecycle
2. **Stages Are Components** - Not raw primitives
3. **Towers Produce Opinions** - Local conclusions, not final answers
4. **Geometric Routing** - Manifold-based coordination
5. **Divergence Over Accuracy** - See differently, triangulate truth
6. **Compile First for Wide Models** - Let `torch.compile` handle fusion

---

## Documentation

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](src/geofractal/router/GETTING_STARTED.md) | Complete tutorial |

---

## License

Apache License 2.0 with Attribution

---

*"Individual towers don't need to be accurate. They need to see differently. The routing fabric triangulates truth from divergent viewpoints."*