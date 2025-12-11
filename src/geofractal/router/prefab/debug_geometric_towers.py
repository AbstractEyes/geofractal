"""
Debug script for tower collective recursion issue.
"""

import sys
import torch
import torch.nn as nn

# Increase recursion limit temporarily for debugging
sys.setrecursionlimit(200)

print("=" * 60)
print("GEOFRACTAL TOWER COLLECTIVE DEBUG")
print("=" * 60)

# Step 1: Test imports
print("\n[1] Testing imports...")
try:
    from geofractal.router.base_router import BaseRouter
    from geofractal.router.base_tower import BaseTower

    print("    BaseRouter:", BaseRouter)
    print("    BaseTower:", BaseTower)
    print("    BaseTower MRO:", [c.__name__ for c in BaseTower.__mro__])
except Exception as e:
    print(f"    IMPORT ERROR: {e}")
    sys.exit(1)

# Step 2: Test minimal tower
print("\n[2] Testing minimal BaseTower...")
try:
    class MinimalTower(BaseTower):
        def __init__(self, name):
            super().__init__(name, strict=False)
            self.linear = nn.Linear(64, 64)

        def forward(self, x):
            return self.linear(x)


    tower = MinimalTower('test_tower')
    print(f"    Created: {tower.name}")
    print(f"    Parameters: {sum(p.numel() for p in tower.parameters())}")
    print(f"    Children: {list(tower.named_children())}")

    # Test .to()
    tower = tower.to('cpu')
    print("    .to('cpu') OK")
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback

    traceback.print_exc()

# Step 3: Test tower with attach
print("\n[3] Testing BaseTower with attach()...")
try:
    class TowerWithAttach(BaseTower):
        def __init__(self, name):
            super().__init__(name, strict=False)
            self.attach('norm', nn.LayerNorm(64))
            self.attach('proj', nn.Linear(64, 64))

        def forward(self, x):
            return self['proj'](self['norm'](x))


    tower = TowerWithAttach('attach_tower')
    print(f"    Created: {tower.name}")
    print(f"    Components: {list(tower.components.keys())}")
    print(f"    Children: {[n for n, _ in tower.named_children()]}")

    # Check for parent attribute
    norm = tower['norm']
    print(f"    norm.parent exists: {hasattr(norm, 'parent')}")
    if hasattr(norm, 'parent'):
        print(f"    norm.parent: {norm.parent}")
        print(f"    norm.parent is tower: {norm.parent is tower}")

    # Test .to()
    tower = tower.to('cpu')
    print("    .to('cpu') OK")
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback

    traceback.print_exc()

# Step 4: Test nested router/tower
print("\n[4] Testing nested Router containing Tower...")
try:
    class InnerTower(BaseTower):
        def __init__(self, name):
            super().__init__(name, strict=False)
            self.attach('linear', nn.Linear(64, 64))

        def forward(self, x):
            return self['linear'](x)


    class OuterRouter(BaseRouter):
        def __init__(self, name):
            super().__init__(name, strict=False)
            self.attach('tower', InnerTower('inner'))
            self.attach('output', nn.Linear(64, 64))

        def forward(self, x):
            return self['output'](self['tower'](x))


    router = OuterRouter('outer')
    print(f"    Created: {router.name}")
    print(f"    Components: {list(router.components.keys())}")

    inner = router['tower']
    print(f"    inner.parent exists: {hasattr(inner, 'parent')}")
    if hasattr(inner, 'parent'):
        print(f"    inner.parent: {inner.parent}")
        print(f"    inner.parent is router: {inner.parent is router}")

    # Check module hierarchy
    print("\n    Module hierarchy:")
    for name, module in router.named_modules():
        print(f"      {name or 'ROOT'}: {type(module).__name__}")

    # Test .to() - this is where recursion might happen
    print("\n    Testing .to('cpu')...")
    router = router.to('cpu')
    print("    .to('cpu') OK")

except RecursionError as e:
    print(f"    RECURSION ERROR!")
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback

    traceback.print_exc()

# Step 5: Check if parent creates cycle
print("\n[5] Checking parent attribute behavior...")
try:
    class TestComponent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64))
            self.parent = None  # This might register as submodule!


    comp = TestComponent()
    print(f"    TestComponent children: {list(comp.named_children())}")


    # Now set parent to another module
    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.data = nn.Parameter(torch.randn(64))


    parent = Parent()
    comp.parent = parent
    print(f"    After setting parent:")
    print(f"    comp.children: {list(comp.named_children())}")
    print(f"    'parent' in comp._modules: {'parent' in comp._modules}")

except Exception as e:
    print(f"    ERROR: {e}")
    import traceback

    traceback.print_exc()

# Step 6: Test with object.__setattr__
print("\n[6] Testing parent with object.__setattr__...")
try:
    class SafeComponent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64))


    comp = SafeComponent()
    parent = Parent()

    # Use object.__setattr__ to bypass nn.Module tracking
    object.__setattr__(comp, 'parent', parent)

    print(f"    comp.parent: {comp.parent}")
    print(f"    comp.children: {list(comp.named_children())}")
    print(f"    'parent' in comp._modules: {'parent' in comp._modules}")

    comp = comp.to('cpu')
    print("    .to('cpu') OK")

except Exception as e:
    print(f"    ERROR: {e}")
    import traceback

    traceback.print_exc()

# Step 7: Check BaseRouter's components ModuleDict
print("\n[7] Checking BaseRouter.components type...")
try:
    router = BaseRouter('test', strict=False)
    print(f"    router.components type: {type(router.components)}")
    print(f"    router.components: {router.components}")
    print(f"    Is ModuleDict: {isinstance(router.components, nn.ModuleDict)}")
except Exception as e:
    print(f"    ERROR: {e}")

# Step 8: Check BaseTower's stages type
print("\n[8] Checking BaseTower.stages type...")
try:
    tower = BaseTower('test', strict=False)
    print(f"    tower.stages type: {type(tower.stages)}")
    print(f"    tower.components type: {type(tower.components)}")
except Exception as e:
    print(f"    ERROR: {e}")

# Step 9: Full module tree with cycle detection
print("\n[9] Testing full collective with cycle detection...")
try:
    from geofractal.router.prefab.geometric_towers import create_tower_collective

    print("    Creating collective...")
    collective = create_tower_collective(
        dim=256,  # Smaller for debug
        tower_depth=1,
        num_heads=4,
        head_dim=64,
        fingerprint_dim=32,
    )

    print(f"    Created collective: {collective.name}")
    print(f"    Top-level children:")
    for name, child in collective.named_children():
        print(f"      {name}: {type(child).__name__}")

    # Check for cycles
    print("\n    Checking for module cycles...")
    seen = set()


    def check_cycles(module, path=""):
        mid = id(module)
        if mid in seen:
            print(f"    CYCLE DETECTED at {path}!")
            return True
        seen.add(mid)
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            if check_cycles(child, child_path):
                return True
        return False


    has_cycle = check_cycles(collective)
    print(f"    Cycles found: {has_cycle}")

    if not has_cycle:
        print("\n    Testing .to('cpu')...")
        collective = collective.to('cpu')
        print("    .to('cpu') OK")

except RecursionError as e:
    print(f"    RECURSION ERROR during creation or .to()!")
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)