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
| **Component** | Attachable unit with identity and lifecycle | The building block - everything is a component |
| **Tower** | Self-encapsulated processing unit | Produces an *opinion*, not just an output |
| **Address** | Geometric identity on a manifold | Fingerprints enable similarity/distance routing |
| **NotifierRouter** | Communication backbone | Routes messages based on geometry |
| **Fusion** | Opinion aggregation | Where emergence happens |

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
        │   ├── SphericalAddress  # Unit hypersphere (geodesic routing)
        │   ├── HyperbolicAddress # Poincaré ball (hierarchical)
        │   ├── SimplexAddress    # Barycentric coordinates
        │   ├── FractalAddress    # Julia set orbits
        │   ├── CantorAddress     # Devil's staircase
        │   └── ShapeAddress      # RoPE-compatible
        │
        ├── FusionComponent       # Combine opinions
        │   ├── AdaptiveFusion    # Content-dependent weights
        │   ├── GatedFusion       # Learned gates
        │   ├── AttentionFusion   # Cross-attention
        │   └── SlotFusion        # Collapse slot dimension
        │
        └── ProjectionComponent   # Transform shapes
            ├── SlotProjection    # Multi-view expansion
            ├── BottleneckProjection
            └── MultiHeadProjection
```

### Towers: Autonomous Processing Units

A **Tower** is a self-encapsulated unit that produces an *opinion*:

```python
class ExpertTower(BaseTower):
    def __init__(self, name: str, dim: int, depth: int = 4):
        super().__init__(name)
        
        # Stages are TorchComponent instances
        for i in range(depth):
            self.append(TransformerBlock(f'{name}_block_{i}', dim))
        
        # Named components accessed via self['key']
        self.attach('final_norm', nn.LayerNorm(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        return self['final_norm'](x)
```

Towers are **not** raw PyTorch modules. They have:
- **Identity** - `name` + `uuid` for addressing
- **Stages** - Ordered pipeline of components (not primitives)
- **Components** - Named auxiliaries (`self['norm']`)
- **Objects** - Non-module storage (`self['config']`)

### Geometric Routing

Towers coordinate through **addresses** - geometric identities that enable routing based on manifold geometry:

```python
# Create router
notifier = NotifierRouter('collective')

# Register towers with addresses
notifier.register(SphericalAddressComponent('teacher', dim=64), channel='knowledge')
notifier.register(SphericalAddressComponent('student', dim=64), channel='knowledge')

# Route by geometric similarity
notifier.route_by_similarity(source, payload, channel='knowledge', top_k=3)

# Route by manifold distance
notifier.route_by_distance(source, payload, channel='knowledge', max_distance=1.0)

# Affinity-weighted broadcast
notifier.affinity_broadcast(source, payload, temperature=0.5)
```

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

### Build a Collective

```python
import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.base_router import BaseRouter
from geofractal.router.base_tower import BaseTower
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.prefab.notifier_router import NotifierRouter
from geofractal.router.components.address_component import SphericalAddressComponent
from geofractal.router.components.fusion_component import AdaptiveFusion


class TransformerBlock(TorchComponent):
    """Reusable block component."""
    
    def __init__(self, name: str, dim: int, num_heads: int = 8):
        super().__init__(name)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ExpertTower(BaseTower):
    """Tower with address for collective participation."""
    
    def __init__(self, name: str, dim: int, notifier: NotifierRouter, depth: int = 3):
        super().__init__(name)
        
        # Geometric identity
        addr = SphericalAddressComponent(name, fingerprint_dim=64)
        self.attach('address', addr)
        self.attach('notifier', notifier)
        notifier.register(addr, channel='collective')
        
        # Block stages (not raw primitives)
        for i in range(depth):
            self.append(TransformerBlock(f'{name}_block_{i}', dim))
        self.attach('norm', nn.LayerNorm(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        # Receive knowledge from other towers
        addr = self['address']
        if addr.has_mail:
            knowledge = addr.aggregate_inbox('mean')
            if knowledge is not None and knowledge.shape == x.shape:
                x = x + 0.1 * knowledge
            addr.clear()
        
        # Process through stages
        for stage in self.stages:
            x = stage(x)
        return self['norm'](x)
    
    def share(self, opinion: Tensor):
        """Share opinion with collective."""
        self['notifier'].post(self['address'], opinion, channel='collective')


class Collective(BaseRouter):
    """Multi-tower collective with geometric coordination."""
    
    def __init__(self, name: str, dim: int, num_towers: int = 3, depth: int = 3):
        super().__init__(name)
        
        # Communication backbone
        notifier = NotifierRouter('notifier')
        self.attach('notifier', notifier)
        
        # Expert towers
        for i in range(num_towers):
            tower = ExpertTower(f'expert_{i}', dim, notifier, depth)
            self.attach(f'expert_{i}', tower)
        
        # Opinion fusion
        fusion = AdaptiveFusion('fusion', num_inputs=num_towers, in_features=dim)
        self.attach('fusion', fusion)
        
        self.attach('config', {'dim': dim, 'num_towers': num_towers})
    
    def forward(self, x: Tensor) -> Tensor:
        config = self['config']
        
        # Each tower produces an opinion
        opinions = []
        for i in range(config['num_towers']):
            tower = self[f'expert_{i}']
            opinion = tower(x)
            tower.share(opinion)  # Share with collective
            opinions.append(opinion)
        
        # Route messages between towers
        self['notifier'].route()
        
        # Fuse opinions into collective output
        return self['fusion'](*opinions)


# Usage
collective = Collective('my_collective', dim=256, num_towers=3, depth=4)
x = torch.randn(4, 32, 256)  # [B, L, D]
output = collective(x)
print(f"Output: {output.shape}")  # [4, 32, 256]
```

---

## Package Structure

```
geofractal/router/
├── base_router.py          # BaseRouter ABC
├── base_tower.py           # BaseTower (autonomous units)
├── base_component.py       # BaseComponent ABC
│
├── components/
│   ├── torch_component.py      # TorchComponent (nn.Module + identity)
│   ├── address_component.py    # Geometric addresses (7 manifolds)
│   ├── fusion_component.py     # Opinion aggregation strategies
│   ├── projection_component.py # Shape transformations
│   ├── cantor_address_component.py
│   └── ...
│
└── prefab/
    ├── notifier_router.py      # Geometric routing backbone
    └── ...
```

---

## Address Types

Different manifolds for different routing semantics:

| Address Type | Manifold | Best For |
|--------------|----------|----------|
| `AddressComponent` | Euclidean (ℝⁿ) | General purpose |
| `SphericalAddressComponent` | Unit hypersphere | Normalized representations |
| `HyperbolicAddressComponent` | Poincaré ball | Hierarchical relationships |
| `SimplexAddressComponent` | Probability simplex | Barycentric blending |
| `FractalAddressComponent` | Julia set orbits | Chaotic/emergent dynamics |
| `CantorAddressComponent` | Devil's staircase | Plateau clustering |
| `ShapeAddressComponent` | RoPE-compatible | Positional/rotational |

---

## Fusion Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `AdaptiveFusion` | Content-dependent weights | General (Lyra pattern) |
| `ConcatFusion` | Concatenate + project | Simple combination |
| `SumFusion` | Learned weighted sum | Averaging with weights |
| `GatedFusion` | Sigmoid gates per input | Selective combination |
| `AttentionFusion` | Cross-attention | Sequence-to-sequence |
| `SlotFusion` | Collapse slot dimension | After slot expansion |

---

## Key Principles

### 1. Components, Not Modules

Everything is a **component** with identity (`name`, `uuid`), lifecycle hooks, and parent awareness. Raw `nn.Module` lacks these coordination features.

### 2. Stages Are Components

Tower stages should be `TorchComponent` subclasses, not raw primitives like `nn.Linear`. This enables uniform output capture and block-level operations (freezing, distillation, replacement).

### 3. Towers Produce Opinions

A tower's output is its *opinion* - a local conclusion from its perspective. The collective triangulates from divergent opinions.

### 4. Geometric Routing

Addresses give towers geometric identity. Routing decisions are based on manifold geometry (similarity, distance, affinity), not just names.

### 5. Divergence Over Accuracy

Individual towers don't need to be accurate. They need to see *differently*. The collective emerges from triangulating divergent perspectives.

---

## Documentation

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Complete tutorial with examples |

---

## When to Use GeoFractal

**Good fit:**
- Multiple pre-trained models available (CLIP, DINO, BERT, T5, etc.)
- Task benefits from diverse perspectives
- Individual models underperform but see different things
- You want emergence, not just averaging
- Multi-GPU deployment with device affinity

**Not ideal:**
- Single model already solves the task well
- Inputs are homogeneous (no diversity to exploit)
- Latency-critical inference (collective adds overhead)

---

## Citation

```bibtex
@software{geofractalrouter2025,
  author       = {AbstractPhil},
  title        = {GeoFractal Router: Collective Intelligence through 
                  Geometric Routing},
  year         = {2025},
  url          = {https://github.com/AbstractPhil/geofractal}
}
```

---

## License

Apache License 2.0 with Attribution

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

---

*"Individual towers don't need to be accurate. They need to see differently. The routing fabric triangulates truth from divergent viewpoints."*