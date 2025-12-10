# Getting Started with Geofractal Router

A system for building collectives of autonomous AI units that coordinate through geometric routing.

## Core Concepts

| Concept | What It Is | Key Insight |
|---------|------------|-------------|
| **Component** | Attachable unit with identity and lifecycle | The building block - everything is a component |
| **Tower** | Self-encapsulated processing unit | Produces an *opinion*, not just an output |
| **Address** | Geometric identity on a manifold | Fingerprints enable similarity/distance routing |
| **NotifierRouter** | Communication backbone | Routes messages based on geometry |
| **Fusion** | Opinion aggregation | Where emergence happens (0.1% → 84.68%) |
| **Projection** | Shape transformation | SlotProjection enables multi-view processing |

## The Component Hierarchy

Everything attachable to a router is a component. The hierarchy:

```
BaseComponent (ABC - pure Python)
│   - name, uuid, parent
│   - Lifecycle: on_attach(), on_detach()
│   - No PyTorch dependency
│
└── TorchComponent (BaseComponent + nn.Module)
        - Learnable parameters
        - Device management (home_device, allowed_devices)
        - forward() method
        │
        ├── AddressComponent      # Geometric identity, fingerprints
        ├── FusionComponent       # Combine multiple signals
        ├── ProjectionComponent   # Transform shapes
        ├── DataComponent         # Data flow
        └── ... specialized components
```

### Why Components Matter

Components are **not raw PyTorch modules**. They have:

1. **Identity** - `name` (human-readable) + `uuid` (machine addressing)
2. **Lifecycle** - `on_attach()` / `on_detach()` hooks when attached to routers
3. **Parent awareness** - Components know which router owns them
4. **Device affinity** - `home_device`, `allowed_devices` for multi-GPU control

```python
from geofractal.router.base_component import BaseComponent
from geofractal.router.components.torch_component import TorchComponent

# Pure Python component (config, state, cache)
class ConfigComponent(BaseComponent):
    def __init__(self, name: str, **kwargs):
        super().__init__(name)
        self.config = kwargs
    
    def on_attach(self, parent):
        print(f"Attached to {parent.name}")

# PyTorch component (learnable parameters)
class AttentionComponent(TorchComponent):
    def __init__(self, name: str, dim: int, num_heads: int = 8):
        super().__init__(name)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return x
```

### Device Control

TorchComponent provides hardware control that raw nn.Module lacks:

```python
# Component with device constraints
encoder = AttentionComponent(
    'encoder',
    dim=512,
    home_device='cuda:0',
    allowed_devices={'cuda:0', 'cuda:1', 'cpu'},
)

encoder.to('cuda:1')  # OK - in allowed set
encoder.to('cuda:2')  # Raises ValueError - not allowed
encoder.to_home()     # Returns to cuda:0
encoder.is_home       # True
```

## Quick Start

```python
import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.base_tower import BaseTower
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.prefab.notifier_router import NotifierRouter
from geofractal.router.components.address_component import (
    AddressComponent,
    SphericalAddressComponent,
)
from geofractal.router.components.fusion_component import AdaptiveFusion
```

## 1. Building Components

Before building towers, understand that towers are composed of components.

### TorchComponent as a Block

```python
class TransformerBlock(TorchComponent):
    """A complete transformer block as a reusable component."""
    
    def __init__(self, name: str, dim: int, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__(name)
        
        # Sub-modules (standard PyTorch)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

This is a **component**, not a raw module. It has:
- `name` and `uuid` for addressing
- `parent` reference when attached
- Device affinity controls
- Lifecycle hooks

## 2. Building Towers

Towers are autonomous units with ordered stages and named components.

> **⚠️ IMPORTANT: Stages should be complete components, not raw PyTorch primitives.**
>
> Using `nn.Linear`, `nn.GELU`, etc. directly as stages defeats the purpose of the tower system.
> Stages should be `TorchComponent` subclasses or other towers. This enables uniform output 
> capture and proper coordination between towers.

### ❌ Wrong: Raw primitives as stages

```python
# DON'T DO THIS - loses the benefit of tower coordination
tower.extend([
    nn.Linear(dim, dim * 4),
    nn.GELU(),
    nn.Linear(dim * 4, dim),
])
```

### ✓ Correct: Components as stages

```python
class ExpertTower(BaseTower):
    """Domain expert built from block components."""
    
    def __init__(self, name: str, dim: int, depth: int = 4):
        super().__init__(name)
        
        # Stages are TorchComponent instances
        for i in range(depth):
            self.append(TransformerBlock(f'{name}_block_{i}', dim))
        
        # Named component (accessed via self['final_norm'])
        self.attach('final_norm', nn.LayerNorm(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)  # Each stage is a complete component
        return self['final_norm'](x)

# Usage
tower = ExpertTower('vision_expert', dim=256, depth=4)
opinion = tower(torch.randn(4, 32, 256))  # [B, L, D]
```

### Why This Matters

When stages are components:
- Each stage produces a **capturable opinion** at its level
- Towers can nest (tower of towers)
- Coordination (routing, fusion) operates on meaningful units
- Block-level freezing, distillation, replacement become trivial
- **Components know their parent** - they can communicate upward

### Tower Access Patterns

```python
tower[0]          # First stage (nn.Module)
tower['norm']     # Named component
len(tower)        # Number of stages
for stage in tower:  # Iterate stages
    ...
```

### Tower Construction (Fluent API)

```python
tower = (
    ExpertTower('encoder', dim=256, depth=2)
    .append(TransformerBlock('encoder_extra', dim=256))  # Add another block
)
```

## 3. Adding Addresses for Coordination

Addresses give towers geometric identity for routing.

```python
class AddressedTower(BaseTower):
    """Tower with geometric identity for routing."""
    
    def __init__(
        self,
        name: str,
        dim: int,
        notifier: NotifierRouter,
        depth: int = 2,
        address_type: str = 'euclidean',
        channel: str = 'default',
    ):
        super().__init__(name)
        
        # Create address component based on geometry
        if address_type == 'euclidean':
            addr = AddressComponent(name, fingerprint_dim=64)
        elif address_type == 'spherical':
            addr = SphericalAddressComponent(name, fingerprint_dim=64)
        # ... other types: hyperbolic, simplex, fractal, cantor, shape
        
        # Attach address as component
        self.attach('address', addr)
        self.attach('notifier', notifier)
        self.attach('channel', channel)
        
        # Register with router
        notifier.register(addr, channel=channel)
        
        # Stages are TorchComponent instances (TransformerBlock from section 1)
        for i in range(depth):
            self.append(TransformerBlock(f'{name}_block_{i}', dim))
        
        self.attach('final_norm', nn.LayerNorm(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        # Check for incoming messages
        addr = self['address']
        if addr.has_mail:
            knowledge = addr.aggregate_inbox('mean')
            if knowledge is not None and knowledge.shape == x.shape:
                x = x + 0.1 * knowledge
            addr.clear()
        
        # Process through component stages
        for stage in self.stages:
            x = stage(x)
        return self['final_norm'](x)
    
    def share(self, opinion: Tensor):
        """Broadcast opinion to channel subscribers."""
        self['notifier'].post(self['address'], opinion, self['channel'])
```

## 4. Setting Up Communication

NotifierRouter orchestrates message passing between towers.

```python
# Create router
notifier = NotifierRouter('collective_comm')

# Create towers with addresses (depth parameter controls block count)
teacher = AddressedTower('teacher', dim=512, notifier=notifier, depth=4, address_type='spherical', channel='knowledge')
student_a = AddressedTower('student_a', dim=256, notifier=notifier, depth=2, address_type='spherical', channel='knowledge')
student_b = AddressedTower('student_b', dim=256, notifier=notifier, depth=2, address_type='spherical', channel='knowledge')

# Add projector for dimension mismatch (teacher 512 → student 256)
notifier.create_projector('teacher', 'student_a', 512, 256)
notifier.create_projector('teacher', 'student_b', 512, 256)
```

### Message Patterns

```python
# Post to channel (one-to-many)
notifier.post(source_addr, payload, channel='knowledge')

# Direct send (one-to-one)
notifier.send(source_addr, target_addr, payload)

# Broadcast to all (one-to-all)
notifier.broadcast(source_addr, payload)

# Execute routing (delivers queued messages)
notifier.route()
```

## 5. Geometric Routing

Route based on fingerprint geometry, not just names.

```python
# Route to k most similar addresses
targets = notifier.route_by_similarity(
    source_addr,
    payload,
    channel='experts',
    top_k=3,
    threshold=0.5,  # Optional minimum similarity
)

# Route to nearest by manifold distance
targets = notifier.route_by_distance(
    source_addr,
    payload,
    channel='experts',
    top_k=3,
    max_distance=1.0,  # Optional maximum distance
)

# Affinity broadcast (weighted by similarity)
weights = notifier.affinity_broadcast(
    source_addr,
    payload,
    channel='experts',
    temperature=0.5,  # Lower = sharper weights
)
```

### Address Types and Their Geometry

| Type | Manifold | Best For |
|------|----------|----------|
| `AddressComponent` | Euclidean (R^n) | General purpose |
| `SphericalAddressComponent` | Unit hypersphere | Normalized representations |
| `HyperbolicAddressComponent` | Poincaré ball | Hierarchical relationships |
| `SimplexAddressComponent` | Probability simplex | Barycentric blending |
| `FractalAddressComponent` | Julia set orbits | Chaotic/emergent dynamics |
| `CantorAddressComponent` | Devil's staircase | Plateau clustering |
| `ShapeAddressComponent` | RoPE-compatible | Positional/rotational |

## 6. Fusing Opinions

Combine multiple tower outputs into collective decision.

```python
from geofractal.router.components.fusion_component import (
    AdaptiveFusion,  # Content-dependent weights (Lyra pattern)
    ConcatFusion,    # Concatenate then project
    SumFusion,       # Learned weighted sum
    GatedFusion,     # Sigmoid gates per input
    AttentionFusion, # Cross-attention
    SlotFusion,      # Collapse slot dimension
)

# Adaptive fusion - each input influences its own weight
fusion = AdaptiveFusion('collective', num_inputs=3, in_features=256)

# Get opinions from towers
opinion_a = tower_a(x)
opinion_b = tower_b(x)
opinion_c = tower_c(x)

# Fuse into collective decision
collective_opinion = fusion(opinion_a, opinion_b, opinion_c)
```

## 7. Slot Expansion (The Emergence Mechanism)

SlotProjection transforms single vectors into multi-view representations.

```python
from geofractal.router.components.projection_component import SlotProjection

# Expand to multiple learned viewpoints
slot_proj = SlotProjection('views', features=256, num_slots=16)

x = torch.randn(4, 256)        # [B, D]
expanded = slot_proj(x)         # [B, 16, D] - 16 different views
collapsed = slot_proj.collapse(expanded, mode='mean')  # [B, D]
```

This is the mechanism that enabled 0.1% individual accuracy → 84.68% collective accuracy.

## 8. Complete Example: Teacher-Student Collective

A complete, copy-paste ready example combining all concepts:

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
from geofractal.router.components.projection_component import SlotProjection


class TransformerBlock(TorchComponent):
    """Reusable transformer block component.
    
    Note: This is a TorchComponent, not a BaseTower.
    It has identity (name, uuid), lifecycle hooks, and device affinity.
    """
    
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


class TeacherTower(BaseTower):
    def __init__(self, name: str, dim: int, notifier: NotifierRouter, depth: int = 6):
        super().__init__(name)
        
        addr = SphericalAddressComponent(name, fingerprint_dim=64)
        self.attach('address', addr)
        self.attach('notifier', notifier)
        notifier.register(addr, channel='distillation')
        
        # Deep tower with block stages
        for i in range(depth):
            self.append(TransformerBlock(f'{name}_block_{i}', dim))
        self.attach('final_norm', nn.LayerNorm(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        x = self['final_norm'](x)
        
        # Broadcast knowledge
        self['notifier'].post(self['address'], x, channel='distillation')
        return x


class StudentTower(BaseTower):
    def __init__(self, name: str, dim: int, notifier: NotifierRouter, depth: int = 3, distill_weight: float = 0.3):
        super().__init__(name)
        
        addr = SphericalAddressComponent(name, fingerprint_dim=64)
        self.attach('address', addr)
        self.attach('notifier', notifier)
        self.attach('distill_weight', distill_weight)
        notifier.register(addr, channel='distillation')
        
        # Shallower tower
        for i in range(depth):
            self.append(TransformerBlock(f'{name}_block_{i}', dim))
        self.attach('final_norm', nn.LayerNorm(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        x = self['final_norm'](x)
        
        # Receive teacher knowledge
        addr = self['address']
        if addr.has_mail:
            knowledge = addr.aggregate_inbox('mean')
            if knowledge is not None and knowledge.shape == x.shape:
                w = self['distill_weight']
                x = (1 - w) * x + w * knowledge
            addr.clear()
        
        return x


class Collective(BaseRouter):
    def __init__(
        self,
        name: str,
        teacher_dim: int,
        student_dim: int,
        num_students: int,
        teacher_depth: int = 6,
        student_depth: int = 3,
    ):
        super().__init__(name)
        
        # Communication backbone (component)
        notifier = NotifierRouter('notifier')
        self.attach('notifier', notifier)
        
        # Teacher tower - deeper, more capacity
        teacher = TeacherTower('teacher', teacher_dim, notifier, depth=teacher_depth)
        self.attach('teacher', teacher)
        
        # Student towers - shallower, learn from teacher
        for i in range(num_students):
            student = StudentTower(f'student_{i}', student_dim, notifier, depth=student_depth)
            self.attach(f'student_{i}', student)
            
            # Projector for dimension mismatch
            if teacher_dim != student_dim:
                notifier.create_projector('teacher', student.name, teacher_dim, student_dim)
        
        # Fusion (component)
        fusion = AdaptiveFusion('fusion', num_inputs=num_students, in_features=student_dim)
        self.attach('fusion', fusion)
        
        # Config (object)
        self.attach('config', {
            'teacher_dim': teacher_dim,
            'student_dim': student_dim,
            'num_students': num_students,
            'teacher_depth': teacher_depth,
            'student_depth': student_depth,
        })
    
    def forward(self, teacher_input: Tensor, student_inputs: list[Tensor]) -> Tensor:
        config = self['config']
        
        # Teacher processes and broadcasts
        _ = self['teacher'](teacher_input)
        
        # Route messages
        self['notifier'].route()
        
        # Students process with teacher knowledge
        opinions = [
            self[f'student_{i}'](student_inputs[i])
            for i in range(config['num_students'])
        ]
        
        # Fuse student opinions
        return self['fusion'](*opinions)


# Usage
collective = Collective('my_collective', teacher_dim=512, student_dim=256, num_students=3)

# Sequence inputs: [B, L, D]
teacher_x = torch.randn(4, 32, 512)
student_xs = [torch.randn(4, 32, 256) for _ in range(3)]

output = collective(teacher_x, student_xs)
print(f"Collective output: {output.shape}")  # [4, 32, 256]
print(f"Stats: {collective['notifier'].stats}")
```

## 9. Hierarchical Routing (Hyperbolic)

For tree-like relationships where some towers are "parents" of others.

```python
from geofractal.router.components.address_component import HyperbolicAddressComponent

notifier = NotifierRouter('hierarchy')

# Create hyperbolic addresses
root = HyperbolicAddressComponent('root', fingerprint_dim=64, curvature=1.0)
child_a = HyperbolicAddressComponent('child_a', fingerprint_dim=64, curvature=1.0)
child_b = HyperbolicAddressComponent('child_b', fingerprint_dim=64, curvature=1.0)

for addr in [root, child_a, child_b]:
    notifier.register(addr, channel='tree')

# Find parent (closer to origin in Poincaré ball)
parent = notifier.find_hierarchical_parent(child_a, channel='tree')

# Find children (further from origin)
children = notifier.find_hierarchical_children(root, channel='tree')
```

## 10. Custom Routing Functions

Define your own routing logic per channel.

```python
def similarity_decay_router(source_addr, target_addr, payload):
    """Scale payload by similarity (distant = attenuated)."""
    sim = source_addr.similarity(target_addr)
    return payload * sim.abs()

notifier.set_routing_function('experts', similarity_decay_router)
```

## 11. Debugging and Monitoring

```python
# Statistics
print(notifier.stats)
# {'routes_executed': 5, 'messages_routed': 12, 'broadcasts': 1, ...}

# Similarity matrix for a channel
names, matrix = notifier.similarity_matrix(channel='experts')

# Distance matrix
names, dist_matrix = notifier.distance_matrix(channel='experts')

# Check subscriptions
print(notifier.subscribers('knowledge'))  # Set of address names
print(notifier.channels('teacher'))       # Set of channel names

# Clear state
notifier.reset()  # Clear mailboxes and stats
```

## Summary

### Design Philosophy: Components → Towers → Collectives

The power of the geofractal system comes from **hierarchical composition**:

```
Collective (BaseRouter)
├── NotifierRouter (communication)
├── TeacherTower (BaseTower)
│   ├── TransformerBlock (TorchComponent) ← stage 0
│   ├── TransformerBlock (TorchComponent) ← stage 1
│   └── ...
├── StudentTower (BaseTower)
│   ├── TransformerBlock (TorchComponent) ← stage 0
│   └── ...
└── FusionComponent (TorchComponent)
```

Each level produces a **capturable opinion**. You can:
- Freeze any tower or component independently
- Replace a teacher's block with a different architecture
- Distill knowledge at any granularity
- Route between components, towers, or collectives

**Raw primitives (`nn.Linear`, `nn.GELU`) belong inside TorchComponents, not as tower stages.**

### The Component Difference

| Raw nn.Module | TorchComponent |
|---------------|----------------|
| No identity | name + uuid |
| No lifecycle | on_attach / on_detach |
| No parent awareness | knows its router |
| No device affinity | home_device, allowed_devices |
| Just computes | Participates in coordination |

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

**The key insight:** Towers don't need to see the whole picture. They produce local opinions, geometric routing determines who talks to whom, and fusion creates emergence where the whole exceeds the sum of parts.