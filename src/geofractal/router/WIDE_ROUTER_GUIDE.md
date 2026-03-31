# WideRouter Guide

WideRouter coordinates parallel execution of multiple towers, enabling batched forward passes and optimized compilation. As of v1.2.0, WideRouter supports multiple execution strategies via `ExecutionStrategy`, including operation-level fusion through `wide_compiler` primitives, multi-vantage sub-ensemble gradient interpolation, and device-aware cache lifecycle management.

## Basic Pattern

```python
from geofractal.router.wide_router import WideRouter, ExecutionStrategy
from geofractal.router.base_tower import BaseTower

class MyCollective(WideRouter):
    def __init__(self, num_towers: int, dim: int):
        super().__init__('my_collective', auto_discover=True,
                         execution_strategy=ExecutionStrategy.AUTO)

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
| `wide_forward_ensemble(x)` | Execute via sub-ensemble pooling (if built) |
| `prepare_for_compile()` | Pre-build execution groups for torch.compile |
| `compile(**kwargs)` | Prepare + torch.compile in one call |
| `build_sub_ensembles()` | Create dual-path sub-ensembles for gradient interpolation |
| `gradient_report()` | Human-readable gradient health report |
| `check_fusion_health()` | Wide primitive fusion gradient diagnostics |
| `get_fusion_coverage()` | Per-group fusion stats (% of ops fused) |
| `validate_cache_devices()` | Check all caches for device mismatches |

## Execution Strategies

WideRouter supports four execution strategies via `ExecutionStrategy`:

| Strategy | Behavior | Best For |
|----------|----------|----------|
| `VMAP` (default) | `torch.func.vmap` vectorization | General use, compiled training |
| `WIDE_COMPILER` | Wide primitive kernels (einsum, grouped conv) | Compiled inference |
| `AUTO` | Training: VMAP, Eval: WIDE_COMPILER → VMAP → sequential | Adaptive |
| `SEQUENTIAL` | Direct per-tower loop | Debugging, compatibility |

```python
from geofractal.router.wide_router import WideRouter, ExecutionStrategy

# Set strategy at construction
collective = MyCollective('wide', execution_strategy=ExecutionStrategy.AUTO)

# Or override at prepare time
collective.prepare_for_compile(execution_strategy=ExecutionStrategy.WIDE_COMPILER)
```

### How Wide Primitives Work

`WIDE_COMPILER` uses the `wide_compiler` package's registry of Wide primitives. For each stage in aligned towers, WideRouter:

1. **Classifies** the stage: fully_fused, partially_fused, or opaque
2. **Fused ops** (Linear, LayerNorm, Conv): executed through Wide kernels (einsum, grouped conv)
3. **Unfused ops** (GELU, custom modules): fall back to `vmap(functional_call)`
4. **Opaque stages** (custom forward with residuals): `vmap(functional_call)` on entire stage

Towers always own their parameters. The group stacks views at prepare time via `torch.stack`. Gradients flow back through the stack to original tower parameters.

**Staleness during training:** `torch.stack` creates new tensors, not views. During training, `WidePrimitiveTowerGroup` re-stacks every forward to pick up optimizer updates. This causes a graph break under `torch.compile`. For compiled training, use `VMAP` or `AUTO` (which selects VMAP during training automatically).

## Compilation

For optimal `torch.compile` performance, prepare the router before compiling:

```python
# Option 1: Explicit preparation
model.collective.prepare_for_compile()
model = torch.compile(model, mode='reduce-overhead')

# Option 2: Use WideRouter's compile method
compiled_collective = collective.compile(mode='reduce-overhead')

# Option 3: With sub-ensembles for gradient interpolation
compiled = collective.compile(build_sub_ensembles=True, mode='reduce-overhead')

# Option 4: With CompileRouter (call prepare first)
model.collective.prepare_for_compile()
compiler = CompileRouter.from_module(model)
model = compiler.compile(mode='reduce-overhead')
```

### What `prepare_for_compile()` Does

1. Calls `analyze_structure()` to find aligned towers
2. Groups towers by structural signature
3. Based on `ExecutionStrategy`:
   - **VMAP**: Pre-builds `VMapTowerGroup` for each group
   - **WIDE_COMPILER**: Classifies stages, builds `WidePrimitiveTowerGroup` with fusion coverage
   - **AUTO**: Builds both VMap and Wide groups for fallback chain
   - **SEQUENTIAL**: No groups built
4. Pre-stacks parameters (avoids graph breaks from `stack_module_state`)
5. Optionally builds `SubEnsembleGroup` instances if `build_sub_ensembles=True`
6. Attaches `GradientDebugger` hooks if `gradient_debug_level > 0`

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
            ├─► Strategy = WIDE_COMPILER or AUTO(eval)?
            │       │
            │       ├─► WidePrimitiveTowerGroup exists? → Wide kernel execution
            │       │       (einsum for Linear, grouped conv for Conv, etc.)
            │       │
            │       └─► Not fusable? → Fall through to vmap (AUTO) or sequential
            │
            ├─► Strategy = VMAP or AUTO(train)?
            │       │
            │       ├─► VMapTowerGroup exists? → Vectorized vmap execution
            │       │
            │       └─► Not vmappable? → Sequential fallback
            │
            └─► Strategy = SEQUENTIAL → Direct per-tower loop
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
| `_stack_params()` in training | Re-stacking each forward | Use `VMAP` or `AUTO` for compiled training |
| Enum comparison in forward | Strategy resolution | Pre-resolved to int at prepare time |
| `set.add()` in error handler | Mutating non-wide-fusable set | Wrapped with `@torch.compiler.disable` |

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
# WideRouter(name='my_collective', towers=8, strategy=AUTO, analyzed=True, prepared=True, wide_groups=1)

print(collective.tower_names)
# ['tower_0', 'tower_1', ...]

print(collective.get_wide_stats())
# Includes execution_strategy, vmap_groups, wide_primitive_groups, fusion_coverage

print(collective.objects['_vmap_groups'].keys())
# Shows pre-built vmap groups

print(collective.objects['_wide_primitive_groups'].keys())
# Shows pre-built wide primitive groups
```

### Gradient Debugging

```python
# Enable at construction
collective = MyCollective('wide', gradient_debug_level=1)

# After backward:
loss.backward()
print(collective.gradient_report())
# Shows per-tower gradient norms and anomalies (NaN, dead, exploding)

# Check fusion health
diag = collective.check_fusion_health()
print(diag.starving_groups)  # Groups getting < 1% gradient flow

# Check fusion coverage
for key, coverage in collective.get_fusion_coverage().items():
    print(f"{key}: {coverage.summary}")
    # e.g. "4/6 ops fused (67%), 2 vmap, 0 sequential"
```

### Cache Device Validation

```python
# Check for device mismatches after moves
issues = collective.validate_cache_devices()
if issues:
    for issue in issues:
        print(f"  MISMATCH: {issue}")
```

## Sub-Ensemble Gradient Interpolation

For multi-vantage gradient diversity, build sub-ensembles that run both VMap and Wide execution paths:

```python
collective.prepare_for_compile(build_sub_ensembles=True)

# Use sub-ensemble-aware forward
opinions = collective.wide_forward_ensemble(x)

# Check interpolation weight distribution
for name, se in collective.objects['_sub_ensembles'].items():
    print(se.get_gradient_diagnostics())
```

Each sub-ensemble has learnable interpolation weights (softmax-normalized). Gradients flow to ALL execution paths through the weighted sum, enabling the optimizer to discover which computational path produces the most useful gradient signal.

## Performance Tips

1. **Call `prepare_for_compile()` once** before training, not per-batch
2. **Use `auto_discover=True`** unless you need fine control
3. **Use `ExecutionStrategy.AUTO`** - adapts between training (VMAP) and inference (WIDE_COMPILER)
4. **Check fusion coverage** - `get_fusion_coverage()` shows if Wide integration is worth it
5. **Group similar towers** - same architecture = better batching
6. **Depth over breadth** - fewer towers with more layers often beats many shallow towers
7. **Watch tower balance** - one dominant tower can drown out others in fusion
8. **Use gradient_debug_level=1** during development to catch dead/exploding gradients early