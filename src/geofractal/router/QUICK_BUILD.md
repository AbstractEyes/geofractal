# Quick Build Guide

Fast reference for building with GeoFractal Router.

---

## Storage Quick Reference

```python
# Modules (learnable, device-managed, in state_dict)
self.attach('encoder', MyComponent('enc', dim=256))

# Config (persistent, non-tensor)
self.attach('config', {'scale': 1.0})

# Cache (ephemeral tensors - only if external code needs them)
self.cache_set('features', tensor)  # Optional!
```

| Storage | Use For | Device-Managed |
|---------|---------|----------------|
| `components` | nn.Module / TorchComponent | ✅ |
| `objects` | Config, metadata | ❌ |
| `_cache` | Tensors for external access | ❌ |

**Rule:** Local variables for `forward()` scope. Cache only if external code retrieves after `forward()`.

---

## Components (Building Blocks)

Components are the building blocks. Use `TorchComponent` instead of raw `nn.Module`:

```python
from geofractal.router.components.torch_component import TorchComponent

class FFNBlock(TorchComponent):
    """Feed-forward block as a reusable component."""
    
    def __init__(self, name: str, dim: int, expansion: int = 4):
        super().__init__(name)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.act = nn.GELU()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class AttentionBlock(TorchComponent):
    """Self-attention as a component."""
    
    def __init__(self, name: str, dim: int, num_heads: int = 8):
        super().__init__(name)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    
    def forward(self, x: Tensor) -> Tensor:
        normed = self.norm(x)
        out, _ = self.attn(normed, normed, normed)
        return out


class TransformerBlock(TorchComponent):
    """Complete transformer block."""
    
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
```

**Why components over raw nn.Module?**
- Identity: `name` and `uuid` for addressing
- Lifecycle: `on_attach()` / `on_detach()` hooks
- Device affinity: `home_device`, `allowed_devices`
- Parent awareness: Components know their router

---

## Simple Tower (Components as Stages)

```python
class SimpleTower(BaseTower):
    def __init__(self, name: str, dim: int, depth: int = 2):
        super().__init__(name, strict=False)
        
        # Stages are components, not raw nn.Linear
        for i in range(depth):
            self.append(TransformerBlock(f'{name}_block_{i}', dim))
        
        self.attach('norm', nn.LayerNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        return self['norm'](x)
```

---

## Tower with Residual (No Cache)

```python
class ResidualTower(BaseTower):
    def __init__(self, name: str, dim: int, depth: int = 3):
        super().__init__(name, strict=False)
        
        for i in range(depth):
            self.append(FFNBlock(f'{name}_ffn_{i}', dim))
        
        self.attach('norm', nn.LayerNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        residual = x  # Local variable - fine
        for stage in self.stages:
            x = x + stage(x)  # Residual per stage
        return self['norm'](x) + residual  # Global residual
```

---

## Tower Exposing Features (Uses Cache)

```python
class FeatureTower(BaseTower):
    def __init__(self, name: str, dim: int, depth: int = 2):
        super().__init__(name, strict=False)
        
        for i in range(depth):
            self.append(TransformerBlock(f'{name}_block_{i}', dim))
        
        self.attach('proj', nn.Linear(dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        
        # Cache for collective to retrieve
        self.cache_set('features', x)
        return self['proj'](x)
    
    @property
    def cached_features(self):
        return self.cache_get('features')
```

---

## Simple Collective (No Cache)

```python
class SimpleCollective(WideRouter):
    def __init__(self, name: str, dim: int, num_towers: int = 8):
        super().__init__(name, auto_discover=True)
        
        for i in range(num_towers):
            self.attach(f't_{i}', SimpleTower(f't_{i}', dim))
        
        self.discover_towers()
        self.attach('fusion', AdaptiveFusion('fusion', num_towers, dim))

    def forward(self, x: Tensor) -> Tensor:
        opinions = self.wide_forward(x)
        return self['fusion'](*opinions.values())
```

---

## Collective Using Tower Features (Clears Cache)

```python
class FeatureCollective(WideRouter):
    def __init__(self, name: str, dim: int, num_towers: int = 8):
        super().__init__(name, auto_discover=True)
        
        for i in range(num_towers):
            self.attach(f't_{i}', FeatureTower(f't_{i}', dim))
        
        self.discover_towers()
        self.attach('fusion', AdaptiveFusion('fusion', num_towers, dim))
        self.attach('feature_proj', nn.Linear(dim * num_towers, dim))

    def forward(self, x: Tensor) -> Tensor:
        opinions = self.wide_forward(x)
        
        # Retrieve cached features
        features = [self[n].cached_features for n in self.tower_names]
        
        # Clear cache after retrieval
        self.clear_tower_caches()
        
        # Use both opinions and features
        fused_opinions = self['fusion'](*opinions.values())
        fused_features = self['feature_proj'](torch.cat(features, dim=-1))
        return fused_opinions + fused_features
```

---

## Device Movement

```python
# Safe (clears cache by default)
model.network_to(device='cuda')
model.network_to(device='cuda', dtype=torch.float16)

# Before device changes
model.reset()  # Safe even if no cache

# Compile
compiled = collective.prepare_and_compile()
```

---

## Imports

```python
import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.base_tower import BaseTower
from geofractal.router.wide_router import WideRouter
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.components.fusion_component import AdaptiveFusion
```

---

## Checklist

- [ ] Components inherit `TorchComponent`
- [ ] Towers inherit `BaseTower`
- [ ] Stages are components (not raw `nn.Linear`)
- [ ] Collectives inherit `WideRouter` (4+ towers) or `BaseRouter`
- [ ] Call `discover_towers()` after attaching towers
- [ ] Use `prepare_and_compile()` not raw `torch.compile()`
- [ ] Use `network_to()` not `.to()` for production
- [ ] Local variables for data used only in `forward()`
- [ ] Cache only if external code needs tensors after `forward()`
- [ ] Clear cache in collective if towers use it

---

## Common Patterns

### Residual (local variable)
```python
def forward(self, x):
    residual = x
    for stage in self.stages:
        x = stage(x)
    return x + residual
```

### Gated (local variable)
```python
def forward(self, x):
    gate = self['gate'](x)
    for stage in self.stages:
        x = stage(x)
    return x * gate
```

### Pre-Norm
```python
def forward(self, x):
    x = self['norm'](x)
    for stage in self.stages:
        x = stage(x)
    return x
```

### Per-Stage Residual
```python
def forward(self, x):
    for stage in self.stages:
        x = x + stage(x)
    return x
```

---

## Anti-Patterns

```python
# ❌ Raw nn.Linear as stage - loses coordination benefits
tower.append(nn.Linear(dim, dim))

# ❌ Tensor in objects - LEAKS MEMORY
self.objects['features'] = tensor

# ❌ Cache for local-only data - UNNECESSARY
self.cache_set('residual', x)  # Just use: residual = x

# ❌ Forget to clear cache - ACCUMULATES
self.cache_set('features', x)
return output  # Collective must clear!

# ❌ Raw compile on WideRouter
torch.compile(collective)  # Use prepare_and_compile()

# ❌ .to() in production
model.to('cuda')  # Use network_to()
```

---

## Debug

```python
# Check cache state
print(model.cache_debug())

# Check VRAM in cache
print(f"{model.cache_size_bytes() / 1024 / 1024:.2f} MB")

# Force clear
model.reset()

# List tower names
print(collective.tower_names)

# Get stats
print(collective.get_wide_stats())

# Parameter count
print(sum(p.numel() for p in model.parameters()))
```