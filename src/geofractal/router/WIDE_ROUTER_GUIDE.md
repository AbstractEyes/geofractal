# WideRouter Guide

WideRouter coordinates parallel execution of multiple towers, enabling batched forward passes and optimized compilation.

## Basic Pattern

```python
from geofractal.router.wide_router import WideRouter
from geofractal.router.base_tower import BaseTower

class MyCollective(WideRouter):
    def __init__(self, num_towers: int, dim: int):
        super().__init__('my_collective', auto_discover=True)
        
        # Attach towers
        for i in range(num_towers):
            self.attach(f'tower_{i}', MyTower(f'tower_{i}', dim))
        
        # Attach fusion (not a tower)
        self.attach('fusion', MyFusion(dim))
        
        # Discover towers before compile
        self.discover_towers()
    
    def forward(self, x: Tensor) -> Tensor:
        # wide_forward executes all registered towers
        opinions = self.wide_forward(x)  # Dict[str, Tensor]
        
        # Fuse opinions your way
        return self['fusion'](opinions)
```

## Key Methods

| Method | Purpose |
|--------|---------|
| `attach(name, module)` | Add a component (tower or other) |
| `register_tower(name)` | Manually register for wide execution |
| `discover_towers()` | Auto-register all BaseTower instances |
| `wide_forward(x)` | Execute all towers, return dict of outputs |
| `prepare_for_compile()` | Pre-build vmap groups for torch.compile |
| `compile(**kwargs)` | Prepare + torch.compile in one call |

## Compilation

For optimal `torch.compile` performance, prepare the router before compiling:

```python
# Option 1: Explicit preparation
model.collective.prepare_for_compile()
model = torch.compile(model, mode='reduce-overhead')

# Option 2: Use WideRouter's compile method
compiled_collective = collective.compile(mode='reduce-overhead')

# Option 3: With CompileRouter (call prepare first)
model.collective.prepare_for_compile()  # ← Don't forget this!
compiler = CompileRouter.from_module(model)
model = compiler.compile(mode='reduce-overhead')
```

### What `prepare_for_compile()` Does

1. Calls `analyze_structure()` to find aligned towers
2. Groups towers by structural signature
3. Pre-builds `VMapTowerGroup` for each group
4. Pre-stacks parameters (avoids graph breaks from `stack_module_state`)

Without preparation, `wide_forward` falls back to sequential execution.

## Tower Registration

Towers are registered for wide execution via:

```python
# Auto-discovery (recommended)
class MyCollective(WideRouter):
    def __init__(self):
        super().__init__('collective', auto_discover=True)
        self.attach('tower_a', MyTower(...))  # Auto-registered
        self.attach('fusion', Fusion(...))    # Not a tower, ignored
        self.discover_towers()

# Manual registration
class MyCollective(WideRouter):
    def __init__(self):
        super().__init__('collective', auto_discover=False)
        self.attach('tower_a', MyTower(...))
        self.register_tower('tower_a')  # Explicit
```

## Execution Flow

```
forward(x)
    │
    ▼
wide_forward(x)
    │
    ├─► Groups towers by structural signature
    │
    ├─► Single tower? → Direct execution
    │
    └─► Multiple aligned? → _batched_tower_forward()
            │
            ├─► VMapTowerGroup exists? → Vectorized vmap execution
            │
            └─► No group? → Sequential fallback
```

## Common Patterns

### Inverse Pairs

```python
class InversePairCollective(WideRouter):
    def __init__(self, dim: int):
        super().__init__('pairs', auto_discover=True)
        
        # Positive and negative towers
        self.attach('fib_pos', FibonacciTower('fib_pos', dim, sign=+1))
        self.attach('fib_neg', FibonacciTower('fib_neg', dim, sign=-1))
        
        self.discover_towers()
    
    def forward(self, x):
        opinions = self.wide_forward(x)
        # Combine inverse perspectives
        return opinions['fib_pos'] - opinions['fib_neg']
```

### Selective Execution

```python
def forward(self, x, use_towers=None):
    # Execute subset of towers
    opinions = self.wide_forward(x, tower_names=use_towers)
    return self.fuse(opinions)
```

### With Attention Mask

```python
def forward(self, x, mask=None):
    opinions = self.wide_forward(x, mask=mask)
    return self.fuse(opinions)
```

## Avoiding Graph Breaks

| Issue | Cause | Solution |
|-------|-------|----------|
| `analyze_structure()` in forward | Not prepared | Call `prepare_for_compile()` before compile |
| `VMapTowerGroup` creation | Dynamic module creation | Pre-build via `prepare_for_compile()` |
| `stack_module_state` | Uses `requires_grad_()` | Handled by `prepare_for_compile()` |
| Device mismatch | Moving modules in forward | Let `prepare_for_compile(device)` handle it |

## Integration with CompileRouter

`CompileRouter` and `WideRouter` are complementary systems:

- **WideRouter**: Coordinates tower execution, handles vmap batching
- **CompileRouter**: Analyzes module structure, builds optimized execution plans

When using both:

```python
# 1. Prepare WideRouter first
model.collective.prepare_for_compile()

# 2. Then use CompileRouter
compiler = CompileRouter.from_module(model)
compiler.compile_towers()
model = compiler.compile(mode='reduce-overhead')
```

## Debugging

Enable graph break logging:

```python
torch._logging.set_logs(graph_breaks=True, recompiles=True)
```

Check router state:

```python
print(collective)
# WideRouter(name='my_collective', towers=8, analyzed=True, prepared=True)

print(collective.tower_names)
# ['tower_0', 'tower_1', ...]

print(collective.objects['_vmap_groups'].keys())
# Shows pre-built vmap groups
```

## Performance Tips

1. **Call `prepare_for_compile()` once** before training, not per-batch
2. **Use `auto_discover=True`** unless you need fine control
3. **Group similar towers** - same architecture = better batching
4. **Depth over breadth** - fewer towers with more layers often beats many shallow towers
5. **Watch tower balance** - one dominant tower can drown out others in fusion