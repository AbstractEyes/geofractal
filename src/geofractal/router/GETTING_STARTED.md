# Getting Started with Geofractal Router

A system for building collectives of autonomous AI units that coordinate through geometric routing.

## Core Concepts

| Concept | What It Is | Key Insight |
|---------|------------|-------------|
| **Component** | Attachable unit with identity and lifecycle | The building block - everything is a component |
| **Tower** | Self-encapsulated processing unit | Produces an *opinion*, not just an output |
| **Address** | Geometric identity on a manifold | Fingerprints enable similarity/distance routing |
| **NotifierRouter** | Communication backbone | Routes messages based on geometry |
| **WideRouter** | Compile-optimized wide model router | Near-linear scaling via torch.compile |
| **Fusion** | Opinion aggregation | Where emergence happens (0.1% → 84.68%) |
| **Projection** | Shape transformation | SlotProjection enables multi-view processing |
| **Cache** | Ephemeral tensor storage | Managed lifecycle prevents memory leaks |

## The Component Hierarchy

Everything attachable to a router is a component. The hierarchy:

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
│       │
│       └── ConfigurableTower, ConfigurableConvTower, etc.
│
├── WideRouter (BaseRouter + wide execution)
│       - tower registration and discovery
│       - wide_forward() for batched execution
│       - torch.compile integration
│
└── NotifierRouter (BaseRouter + messaging)
        - geometric routing between towers
        - message posting and aggregation

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

---

## Storage Types

Routers and towers have three distinct storage mechanisms. Understanding when to use each is critical for avoiding memory leaks and ensuring proper behavior.

| Storage | Type | Moved by `.to()` | In `state_dict` | Use For |
|---------|------|------------------|-----------------|---------|
| `components` | `nn.ModuleDict` | ✅ Yes | ✅ Yes | nn.Module children (layers, sub-routers) |
| `objects` | `dict` | ❌ No | ❌ No | Config, metadata, constants |
| `_cache` | `dict` | ❌ No | ❌ No | Ephemeral tensors during forward |

### components[] - Learnable Modules

```python
# Stored in nn.ModuleDict - parameters tracked, device-managed
# Named components can be raw nn.Module OR TorchComponent
self.attach('encoder', nn.Linear(256, 512))  # OK as named component
self.attach('sub_tower', MyTower('sub', dim=256))

# Access
encoder = self['encoder']
self.components.keys()  # List all module components
```

### objects[] - Persistent Non-Tensor Data

```python
# Stored in plain dict - NOT moved by .to(), NOT in state_dict
self.attach('config', {'dropout': 0.1, 'scale': 1.0})
self.attach('tower_names', ['expert_0', 'expert_1'])

# Access
config = self['config']
self.objects.keys()  # List all object keys
```

**⚠️ WARNING: Never store tensors in objects[]**

```python
# ❌ WRONG - causes memory leaks, device mismatches
self.objects['cached_features'] = features  # LEAK!

# ✅ CORRECT - use cache for ephemeral tensors
self.cache_set('features', features)
```

### _cache - Ephemeral Tensor Storage

The cache is for intermediate tensors that:
- Need to persist after `forward()` returns
- Will be accessed by external code (e.g., WideRouter, Collectives)
- Should be cleared after use

```python
# Store intermediate for external retrieval
self.cache_set('features', features)
self.cache_set('opinion', opinion)

# Retrieve
features = self.cache_get('features')
features = self.cache_get('missing_key', default=None)

# Clear after use
self.cache_clear()  # This router only
self.cache_clear_recursive()  # Entire tree
```

### When to Use Cache vs Local Variables

| Situation | Use |
|-----------|-----|
| Residual within same forward() | Local variable |
| Gate computed and used in same forward() | Local variable |
| Features needed by Collective after forward() | Cache |
| Intermediates for WideRouter integration | Cache |
| Data shared between separate method calls | Cache |

```python
class MyTower(BaseTower):
    def forward(self, x: Tensor) -> Tensor:
        # ✅ Local variable - only used within this forward()
        residual = x
        
        for stage in self.stages:
            x = stage(x)
        
        # ✅ Cache - needed by Collective after forward() returns
        self.cache_set('features', x)
        
        return x + residual
```

---

## Cache Control and Memory Management

### The Memory Leak Pattern (Fixed)

Previous versions stored tensors in `objects[]`, causing VRAM leaks:

```python
# ❌ OLD CODE - leaked ~33MB per tower per forward
self.objects['_cached_features'] = features  # Never cleared!

# ✅ NEW CODE - uses managed cache
self.cache_set('features', features)  # Cleared by collective
```

### Cache API Reference

| Method | Description |
|--------|-------------|
| `cache_set(key, value)` | Store tensor in cache |
| `cache_get(key, default=None)` | Retrieve from cache |
| `cache_clear()` | Clear this router's cache |
| `cache_clear_recursive()` | Clear entire router tree |
| `cache_keys()` | List current cache keys |
| `cache_size_bytes()` | Estimate VRAM held in cache |
| `cache_to(device, dtype)` | Move cache tensors (explicit) |
| `cache_to_recursive(...)` | Move cache across tree |
| `cache_debug(prefix='')` | Debug cache state across tree |
| `reset()` | Clear cache + call reset() on children |

### Collective Cache Management

Collectives (ConfigurableCollective, ConvTowerCollective) automatically clear tower caches:

```python
class MyCollective(WideRouter):
    def forward(self, x: Tensor) -> Tensor:
        opinions = self.wide_forward(x)
        
        # ... use cached features from towers ...
        
        # Clear tower caches to prevent leaks
        for name in self.tower_names:
            self[name].cache_clear()
        
        return self['fusion'](*opinions.values())
```

### Debugging Memory Issues

```python
# Check cache state across entire model
debug_info = model.cache_debug()
for path, cache in debug_info.items():
    print(f"{path}:")
    for key, info in cache.items():
        print(f"  {key}: {info['shape']} = {info['bytes']} bytes")

# Estimate total cache VRAM
total = sum(
    router.cache_size_bytes() 
    for router in model.modules() 
    if hasattr(router, 'cache_size_bytes')
)
print(f"Total cache: {total / 1024 / 1024:.2f} MB")

# Force clear everything
model.reset()  # Clears cache recursively
```

---

## Device Movement

### network_to() vs .to()

| Method | Cache Behavior | Strict Validation | Use When |
|--------|----------------|-------------------|----------|
| `.to(device)` | ❌ Not moved | ⚠️ Warning only | Quick moves, testing |
| `network_to(device)` | 🗑️ Cleared by default | ✅ Updates constraints | Production, multi-GPU |

```python
# Standard PyTorch .to() - cache NOT moved
model.to('cuda:0')  # Warning if strict=True

# Router-aware movement - cache cleared by default
model.network_to(device='cuda:0')  # Safe for production

# Preserve cache during movement (advanced)
model.network_to(device='cuda:0', clear_cache=False)
model.cache_to_recursive(device='cuda:0')  # Manual cache move
```

### Accelerate Compatibility

The cache is intentionally NOT registered with PyTorch's parameter system:

```python
# ✅ Recommended pattern with accelerate
model.reset()  # Clear all caches first
model = accelerate.prepare(model)

# ❌ WRONG - cache tensors on wrong device
model = accelerate.prepare(model)
model(x)  # Cache created on accelerate device
model.network_to('cpu')  # Parameters move, cache doesn't!
```

### Multi-GPU Patterns

```python
# Pattern 1: Clear before any device change
model.reset()
model.network_to(device='cuda:1')

# Pattern 2: Explicit cache movement
model.network_to(device='cuda:1', clear_cache=False)
model.cache_to_recursive(device='cuda:1')

# Pattern 3: Device-specific inference
with torch.cuda.device(1):
    model.network_to(device='cuda:1')
    output = model(x.to('cuda:1'))
    model.reset()  # Clear cache before next device
```

---

## Critical Dos and Don'ts

### ✅ DO

```python
# DO: Use cache for tensors needed after forward()
self.cache_set('features', features)

# DO: Clear cache in collective forward()
for tower_name in self.tower_names:
    self[tower_name].cache_clear()

# DO: Call reset() before device changes
model.reset()
model.network_to(device='cuda:1')

# DO: Use network_to() for production code
model.network_to(device='cuda', dtype=torch.float16)

# DO: Use local variables for forward()-scoped data
residual = x  # Only used within this forward()

# DO: Put config in objects[]
self.attach('config', {'scale': 1.0})

# DO: Call discover_towers() after attaching towers
for i in range(n):
    self.attach(f'tower_{i}', MyTower(...))
self.discover_towers()  # Register for wide_forward

# DO: Use prepare_and_compile() for WideRouter
compiled = collective.prepare_and_compile()
```

### ❌ DON'T

```python
# DON'T: Store tensors in objects[] - causes memory leaks!
self.objects['features'] = features  # LEAK!

# DON'T: Forget to clear cache - accumulates VRAM
def forward(self, x):
    self.cache_set('temp', expensive_tensor)
    return output  # Cache never cleared!

# DON'T: Assume .to() moves cache
model.to('cuda:1')  # Cache stays on old device!

# DON'T: Use raw nn.Linear as tower stages
tower.extend([nn.Linear(d, d), nn.GELU()])  # Loses coordination

# DON'T: Call torch.compile() directly on WideRouter
compiled = torch.compile(collective)  # May fail on analysis code

# DON'T: Access cache after clear
self.cache_clear()
features = self.cache_get('features')  # Returns None!

# DON'T: Forget clear_cache=False when moving cache explicitly
model.network_to('cuda:1')  # Clears cache!
model.cache_to_recursive('cuda:1')  # Nothing to move!
```

### Memory Leak Checklist

If you're seeing VRAM grow during training:

1. **Check for `objects[]` tensor storage:**
   ```python
   grep -r "self\.objects\[.*\] =" *.py | grep -v config
   ```

2. **Verify cache clearing in collectives:**
   ```python
   # At end of Collective.forward()
   for name in self.tower_names:
       self[name].cache_clear()
   ```

3. **Use cache_debug() to find leaks:**
   ```python
   print(model.cache_debug())  # Should be {} between batches
   ```

4. **Call reset() in training loop (optional but safe):**
   ```python
   for batch in dataloader:
       loss = model(batch)
       loss.backward()
       optimizer.step()
       model.reset()  # Paranoid but safe
   ```

## Quick Start

```python
import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.base_tower import BaseTower
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.prefab.notifier_router import NotifierRouter
from geofractal.router.wide_router import WideRouter
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

---

## 3. WideRouter for Wide Models

When you have multiple towers processing the same input, **WideRouter** provides compile-optimized execution with near-linear scaling.

### Why WideRouter?

| Method | 8 Towers | Per-Tower |
|--------|----------|-----------|
| Sequential (eager) | 2.92ms | 365µs |
| WideRouter (eager) | 2.81ms | 351µs |
| Sequential (compiled) | 1.88ms | 235µs |
| **WideRouter (compiled)** | **1.89ms** | **236µs** |

With more towers, per-tower cost *decreases*:

| Towers | Total Time | Per-Tower |
|--------|------------|-----------|
| 4 | 1.03ms | 258µs |
| 8 | 1.90ms | 238µs |
| 16 | 3.63ms | 227µs |
| 32 | 7.21ms | 225µs |

### Basic WideRouter Usage

```python
from geofractal.router.wide_router import WideRouter
from geofractal.router.components.torch_component import TorchComponent


class FFNBlock(TorchComponent):
    """Feed-forward block as a reusable component."""
    
    def __init__(self, name: str, dim: int, expansion: int = 2):
        super().__init__(name)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SimpleTower(BaseTower):
    """Simple tower for wide collective."""

    def __init__(self, name: str, dim: int, depth: int = 2):
        super().__init__(name, strict=False)

        for i in range(depth):
            self.append(FFNBlock(f'{name}_ffn_{i}', dim))
        self.attach('norm', nn.LayerNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = x + stage(x)
        return self['norm'](x)


class WideCollective(WideRouter):
    """Compile-optimized collective."""

    def __init__(self, name: str, num_towers: int, dim: int):
        super().__init__(name, auto_discover=True)

        # Attach towers
        for i in range(num_towers):
            self.attach(f'tower_{i}', SimpleTower(f'tower_{i}', dim))

        # IMPORTANT: Call discover_towers() after attaching
        self.discover_towers()

        # Fusion
        self.attach('fusion', AdaptiveFusion('fusion', num_towers, dim))

    def forward(self, x: Tensor) -> Tensor:
        # wide_forward returns Dict[tower_name, output]
        opinions = self.wide_forward(x)
        return self['fusion'](*opinions.values())


# Create collective
collective = WideCollective('wide', num_towers=8, dim=256)
collective.network_to(device='cuda')

# Compile for best performance
compiled = collective.prepare_and_compile()

# Use compiled model
x = torch.randn(32, 64, 256, device='cuda')
output = compiled(x)
```

### WideRouter Methods

| Method | Purpose |
|--------|---------|
| `discover_towers()` | Find and register all BaseTower components |
| `analyze_structure()` | Analyze tower structure for alignment |
| `wide_forward(x)` | Execute all towers, return Dict[name, output] |
| `compile(**kwargs)` | Compile with torch.compile |
| `prepare_and_compile(**kwargs)` | Analyze + compile (recommended) |
| `get_wide_stats()` | Get execution statistics |
| `reset()` | Clear cache + pool, call reset() on towers |
| `clear_tower_caches()` | Clear cache on all registered towers |

### WideRouter Best Practices

1. **Call `discover_towers()` in `__init__`** after attaching all towers
2. **Use `prepare_and_compile()`** instead of raw `torch.compile()`
3. **Pre-analyze before compile** - structure analysis uses Python constructs that dynamo can't trace
4. **Set `torch.set_float32_matmul_precision('high')`** for better performance

```python
# Recommended pattern
torch.set_float32_matmul_precision('high')

collective = WideCollective('wide', num_towers=16, dim=256)
collective.network_to(device='cuda')

# prepare_and_compile = analyze_structure() + compile()
compiled = collective.prepare_and_compile()

# For fine control:
# collective.analyze_structure()  # Separate analysis
# compiled = collective.compile(mode='reduce-overhead')
```

### When to Use WideRouter vs BaseRouter

| Use WideRouter | Use BaseRouter |
|----------------|----------------|
| 4+ towers with same structure | Heterogeneous tower structures |
| All towers process same input | Different inputs per tower |
| Throughput-critical | Fine-grained execution control |
| Want `torch.compile` benefits | Need geometric routing between towers |

---

## 4. Adding Addresses for Coordination

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
        
        # Create address based on type
        if address_type == 'spherical':
            addr = SphericalAddressComponent(name, fingerprint_dim=64)
        else:
            addr = AddressComponent(name, fingerprint_dim=64)
        
        self.attach('address', addr)
        self.attach('notifier', notifier)
        notifier.register(addr, channel=channel)
        
        # Stages
        for i in range(depth):
            self.append(TransformerBlock(f'{name}_block_{i}', dim))
        self.attach('final_norm', nn.LayerNorm(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        # Check for incoming messages
        addr = self['address']
        if addr.has_mail:
            mail = addr.aggregate_inbox('mean')
            if mail is not None and mail.shape == x.shape:
                x = x + 0.1 * mail
            addr.clear()
        
        # Process
        for stage in self.stages:
            x = stage(x)
        return self['final_norm'](x)
    
    def broadcast(self, opinion: Tensor):
        """Share opinion with collective."""
        self['notifier'].post(self['address'], opinion, channel='default')
```

---

## 5. Fusion Strategies

Different fusion strategies for different needs:

```python
from geofractal.router.components.fusion_component import (
    AdaptiveFusion,    # Content-dependent weights (recommended)
    GatedFusion,       # Sigmoid gates per input
    AttentionFusion,   # Cross-attention between inputs
    ConcatFusion,      # Concatenate + project
    SumFusion,         # Learned weighted sum
)

# AdaptiveFusion - learns to weight based on content
fusion = AdaptiveFusion('fusion', num_inputs=3, in_features=256)
fused = fusion(opinion_a, opinion_b, opinion_c)

# GatedFusion - binary-ish selection
fusion = GatedFusion('fusion', num_inputs=3, in_features=256)
fused = fusion(opinion_a, opinion_b, opinion_c)
```

---

## 6. Complete Example: Wide Collective with Routing

```python
import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.wide_router import WideRouter
from geofractal.router.prefab.notifier_router import NotifierRouter
from geofractal.router.base_tower import BaseTower
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.components.address_component import SphericalAddressComponent
from geofractal.router.components.fusion_component import AdaptiveFusion


class TransformerBlock(TorchComponent):
    def __init__(self, name: str, dim: int, num_heads: int = 8):
        super().__init__(name)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.ffn(self.norm2(x))


class ExpertTower(BaseTower):
    def __init__(self, name: str, dim: int, depth: int = 3):
        super().__init__(name, strict=False)

        for i in range(depth):
            self.append(TransformerBlock(f'{name}_block_{i}', dim))
        self.attach('norm', nn.LayerNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        return self['norm'](x)


class RoutedWideCollective(WideRouter):
    """Wide collective with geometric routing."""

    def __init__(self, name: str, num_towers: int, dim: int):
        super().__init__(name, auto_discover=True)

        # Communication backbone
        notifier = NotifierRouter('notifier')
        self.attach('notifier', notifier)

        # Expert towers with addresses
        for i in range(num_towers):
            tower = ExpertTower(f'expert_{i}', dim)
            addr = SphericalAddressComponent(f'expert_{i}', fingerprint_dim=64)
            tower.attach('address', addr)
            notifier.register(addr, channel='experts')
            self.attach(f'expert_{i}', tower)

        self.discover_towers()
        self.attach('fusion', AdaptiveFusion('fusion', num_towers, dim))
        self.objects['num_towers'] = num_towers

    def forward(self, x: Tensor) -> Tensor:
        # Get opinions from all towers
        opinions = self.wide_forward(x)

        # Each tower shares its opinion
        notifier = self['notifier']
        for name, opinion in opinions.items():
            tower = self[name]
            notifier.post(tower['address'], opinion, channel='experts')

        # Route messages
        notifier.route()

        # Fuse
        return self['fusion'](*opinions.values())


# Usage
torch.set_float32_matmul_precision('high')

collective = RoutedWideCollective('routed', num_towers=8, dim=256)
collective.network_to(device='cuda')
compiled = collective.prepare_and_compile()

x = torch.randn(32, 64, 256, device='cuda')
output = compiled(x)
print(f"Output: {output.shape}")  # [32, 64, 256]
```

---

## Summary

### Design Philosophy: Components → Towers → Collectives

The power of the geofractal system comes from **hierarchical composition**:

```
Collective (WideRouter or BaseRouter)
├── NotifierRouter (communication)
├── ExpertTower (BaseTower)
│   ├── TransformerBlock (TorchComponent) ← stage 0
│   ├── TransformerBlock (TorchComponent) ← stage 1
│   ├── _cache: {'features': Tensor}      ← ephemeral storage
│   └── ...
├── ExpertTower (BaseTower)
│   └── ...
└── FusionComponent (TorchComponent)
```

### Storage Model

| Storage | Contents | Lifetime | Device-Managed |
|---------|----------|----------|----------------|
| `components` | nn.Module children | Persistent | ✅ Yes |
| `objects` | Config, metadata | Persistent | ❌ No |
| `_cache` | Intermediate tensors | Ephemeral | ❌ Manual |

### Router Selection Guide

| Scenario | Router | Why |
|----------|--------|-----|
| 8+ identical towers, same input | `WideRouter` | Near-linear scaling with compile |
| 4-8 towers, throughput matters | `WideRouter` | Compile benefits |
| Heterogeneous towers | `BaseRouter` | Flexibility |
| Complex inter-tower routing | `BaseRouter` + `NotifierRouter` | Fine control |

### The WideRouter Advantage

```
Eager (sequential):     O(n) kernel launches, O(n) Python overhead
Compiled WideRouter:    Fused kernels, minimal Python in hot path

Result: Per-tower cost DECREASES as tower count INCREASES
        4 towers: 258µs/tower
       32 towers: 225µs/tower
```

### Key Takeaways

1. **Use WideRouter for wide models** - let `torch.compile` handle fusion
2. **Call `discover_towers()` after attaching** - ensures registration
3. **Use `prepare_and_compile()`** - handles analysis before compile
4. **Stages are components** - not raw nn.Linear
5. **Divergence over accuracy** - towers see differently, collective triangulates
6. **Cache for external access** - local variables for forward()-scoped data
7. **Clear cache in collectives** - prevents VRAM leaks
8. **Use network_to() for device moves** - handles cache safely

### Memory Management Checklist

```python
# Before training
model.reset()  # Clear any stale cache

# In Collective.forward()
opinions = self.wide_forward(x)
# ... process opinions ...
for name in self.tower_names:
    self[name].cache_clear()  # Prevent accumulation

# Before device changes
model.reset()
model.network_to(device='cuda:1')

# Debug memory issues
print(model.cache_debug())  # Should be empty between batches
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    WideRouter Collective                        │
│                                                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Tower 0 │ │ Tower 1 │ │ Tower 2 │ │ Tower 3 │ │  ...    │   │
│  │ _cache  │ │ _cache  │ │ _cache  │ │ _cache  │ │ _cache  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │          │          │          │          │            │
│       └──────────┴──────────┴──────────┴──────────┘            │
│                             ↓                                   │
│                    wide_forward(x)                              │
│              (torch.compile fuses kernels)                      │
│                             ↓                                   │
│               cache_clear() on each tower                       │
│                             ↓                                   │
│                   FusionComponent                               │
│                             ↓                                   │
│                    Collective Output                            │
└─────────────────────────────────────────────────────────────────┘
```

**The key insight:** Towers don't need to see the whole picture. They produce local opinions, and the collective triangulates truth from divergent viewpoints. WideRouter makes this efficient at scale, and the cache system ensures memory doesn't accumulate.

---

## Appendix: Architecture Changes (v1.0.1)

### Cache System Addition

**Files modified:**
- `base_router.py` - Added `_cache` dict and management methods
- `base_tower.py` - Updated docstrings, added `CachingTower` example
- `wide_router.py` - Added `reset()` override, `clear_tower_caches()`
- `geometric_tower_builder.py` - Fixed `objects[]` leak → `cache_set()`
- `geometric_conv_tower_builder.py` - Same fix + multi-channel support

**New BaseRouter methods:**
```python
cache_set(key, value)           # Store tensor
cache_get(key, default=None)    # Retrieve tensor
cache_clear()                   # Clear this router
cache_clear_recursive()         # Clear entire tree
cache_keys()                    # List keys
cache_size_bytes()              # Estimate VRAM
cache_to(device, dtype)         # Move cache (explicit)
cache_to_recursive(...)         # Move cache tree
cache_debug(prefix='')          # Debug cache state
```

**Updated network_to():**
```python
# New parameter: clear_cache (default True)
model.network_to(device='cuda', clear_cache=True)  # Safe default
model.network_to(device='cuda', clear_cache=False) # Manual control
```

### Multi-Channel Conv Tower Support

**New components in geometric_conv_tower_builder.py:**
- `FlexibleInputComponent` - Handles `[B,C,H,W]` or `[B,L,D]` inputs
- `MultiScaleConvBlock` - Local/regional/global with SE attention
- `ChannelMixerBlock` - Cross-channel attention for VAE latents
- `SpatialToOpinionComponent` - Configurable pooling modes

**New ConvTowerConfig options:**
```python
ConvTowerConfig(
    in_channels=16,           # Flux VAE: 16, SD VAE: 4
    input_mode='spatial',     # 'spatial', 'sequence', 'auto'
    pool_mode='attention',    # 'adaptive', 'attention', 'multiscale'
    use_channel_mixer=True,   # Cross-channel attention
    use_multiscale=True,      # MultiScaleConvBlock injection
)
```

**New presets:**
```python
preset_flux_vae_towers()    # 16-channel, attention pooling
preset_sd_vae_towers()      # 4-channel, adaptive pooling
preset_sequence_towers()    # Sequence input mode
```

### Memory Leak Fix

**Before (leaked ~33MB per tower per forward):**
```python
self.objects['_cached_features'] = features  # Never cleared
```

**After (managed lifecycle):**
```python
self.cache_set('features', features)  # Cleared by collective
```

**Automatic clearing in collectives:**
```python
# ConfigurableCollective.forward() and ConvTowerCollective.forward()
# now call cache_clear() on each tower after use
```