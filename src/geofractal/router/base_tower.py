"""
geofractal.router.base_tower
==============================

BaseTower - Self-encapsulated processing unit producing an opinion.

A tower is a complete AI subsystem that:
    - Processes through ordered stages (like Sequential)
    - Does not see global attention (only local context)
    - Produces an opinion (output tensor)
    - Coordinates with pools and other towers

Structural Model:
    stages:     Ordered pipeline (nn.ModuleList) - int indexed
    components: Named auxiliaries (nn.ModuleDict) - str indexed
    objects:    Non-module storage (dict) - str indexed

Coordination Model:
    Towers communicate through pools, not direct connection.
    Multiple towers can feed the same pool.
    A tower can draw from multiple pools.

    Tower A ──┐
              ├──► Pool X ──► Tower D
    Tower B ──┤
              └──► Pool Y ──► Tower E
    Tower C ──────► Pool Y

Interface:
    Mirrors torch.Sequential for stage management while
    maintaining router flexibility for auxiliary components.

    tower.append(module)     # Add to pipeline
    tower.extend([m1, m2])   # Add multiple
    tower[0]                 # Stage by index
    tower['config']          # Component by name
    len(tower)               # Stage count
    for stage in tower:      # Iterate stages

Opinion:
    The output of forward() is the tower's opinion - its local
    conclusion based on input and internal processing. This
    opinion flows to pools for aggregation with other towers.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from abc import abstractmethod
from typing import Optional, Iterator, Union, Any

import torch.nn as nn
from torch import Tensor

from geofractal.router.base_router import BaseRouter

class BaseTower(BaseRouter):
    """
    Self-encapsulated processing unit producing an opinion.

    Extends BaseRouter with ordered stage processing.
    """

    def __init__(self, name: str, uuid: Optional[str] = None, **kwargs):
        super().__init__(name, uuid, **kwargs)
        self.stages = nn.ModuleList()

    # =========================================================================
    # PIPELINE CONSTRUCTION
    # =========================================================================

    def append(self, module: nn.Module) -> 'BaseTower':
        """Append module to processing pipeline."""
        self.stages.append(module)
        return self

    def extend(self, modules) -> 'BaseTower':
        """Append multiple modules to pipeline."""
        for module in modules:
            self.stages.append(module)
        return self

    def insert(self, index: int, module: nn.Module) -> 'BaseTower':
        """Insert module at pipeline index."""
        self.stages.insert(index, module)
        return self

    def pop(self, index: int = -1) -> nn.Module:
        """Remove and return stage at index."""
        module = self.stages[index]
        del self.stages[index]
        return module

    # =========================================================================
    # FORWARD
    # =========================================================================

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Produce opinion from input.

        Subclass defines signature and topology.
        Use self.stages for ordered processing.
        """
        ...

    # =========================================================================
    # ACCESS PATTERNS
    # =========================================================================

    def __len__(self) -> int:
        """Number of stages in pipeline."""
        return len(self.stages)

    def __getitem__(self, key: Union[int, str]) -> Any:
        """
        Dual access pattern.

        Int key:  tower[0]      → stages[0]
        Str key:  tower['norm'] → components['norm'] or objects['norm']
        """
        if isinstance(key, int):
            return self.stages[key]
        return super().__getitem__(key)

    def __iter__(self) -> Iterator[nn.Module]:
        """Iterate over stages in order."""
        return iter(self.stages)

    def __contains__(self, key: Union[int, str, nn.Module]) -> bool:
        """
        Check membership.

        Int:      index in range
        Str:      name in components or objects
        Module:   module in stages
        """
        if isinstance(key, int):
            return 0 <= key < len(self.stages)
        if isinstance(key, nn.Module):
            return key in self.stages
        return super().__contains__(key)

    def __repr__(self) -> str:
        stage_reprs = [f"    ({i}): {repr(s)}" for i, s in enumerate(self.stages)]
        stages_str = "\n".join(stage_reprs) if stage_reprs else "    (empty)"

        components = list(self.components.keys())
        objects = list(self.objects.keys())

        return (
            f"{self.__class__.__name__}(\n"
            f"  name='{self.name}',\n"
            f"  stages=[\n{stages_str}\n  ],\n"
            f"  components={components},\n"
            f"  objects={objects}\n"
            f")"
        )


# =============================================================================
# TEST TOWERS
# =============================================================================

class SequentialTower(BaseTower):
    """Simple sequential processing - stages in order."""

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class ResidualTower(BaseTower):
    """Sequential with residual connection around all stages."""

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        for stage in self.stages:
            x = stage(x)
        return x + residual


class PreNormTower(BaseTower):
    """Uses attached component for pre-normalization."""

    def __init__(self, name: str, dim: int, **kwargs):
        super().__init__(name, **kwargs)
        self.attach('norm', nn.LayerNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = self['norm'](x)
        for stage in self.stages:
            x = stage(x)
        return x


class GatedTower(BaseTower):
    """Gated output - stages produce value, gate controls flow."""

    def __init__(self, name: str, dim: int, **kwargs):
        super().__init__(name, **kwargs)
        self.attach('gate', nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        ))

    def forward(self, x: Tensor) -> Tensor:
        gate = self['gate'](x)
        for stage in self.stages:
            x = stage(x)
        return x * gate


class DualPathTower(BaseTower):
    """
    Two paths through stages - demonstrates non-linear topology.

    Even stages process path A, odd stages process path B.
    Outputs are summed.
    """

    def forward(self, x: Tensor) -> Tensor:
        path_a = x
        path_b = x

        for i, stage in enumerate(self.stages):
            if i % 2 == 0:
                path_a = stage(path_a)
            else:
                path_b = stage(path_b)

        return path_a + path_b


class AttentionTower(BaseTower):
    """
    Tower with self-attention as a component.

    Demonstrates mixing stages with named components for
    more complex architectures.
    """

    def __init__(self, name: str, dim: int, num_heads: int = 8, **kwargs):
        super().__init__(name, **kwargs)
        self.attach('attn', nn.MultiheadAttention(dim, num_heads, batch_first=True))
        self.attach('norm1', nn.LayerNorm(dim))
        self.attach('norm2', nn.LayerNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Self-attention block
        residual = x
        x = self['norm1'](x)
        x, _ = self['attn'](x, x, x)
        x = x + residual

        # FFN through stages
        residual = x
        x = self['norm2'](x)
        for stage in self.stages:
            x = stage(x)
        x = x + residual

        return x


class ConditionalTower(BaseTower):
    """
    Conditional processing based on config object.

    Demonstrates using non-module objects to control behavior.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.attach('config', {
            'use_residual': True,
            'scale': 1.0,
        })

    def forward(self, x: Tensor) -> Tensor:
        config = self['config']
        residual = x if config['use_residual'] else None

        for stage in self.stages:
            x = stage(x)

        x = x * config['scale']

        if residual is not None:
            x = x + residual

        return x


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == '__main__':
    import torch
    import time

    def section(title: str) -> None:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print('=' * 70)

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =========================================================================
    section("SEQUENTIAL TOWER - Basic Pipeline")
    # =========================================================================

    seq_tower = SequentialTower('sequential')
    seq_tower.append(nn.Linear(256, 512))
    seq_tower.append(nn.ReLU())
    seq_tower.append(nn.Linear(512, 512))
    seq_tower.append(nn.ReLU())
    seq_tower.append(nn.Linear(512, 256))

    print(f"Tower:\n{seq_tower}")
    print(f"\nStage count: {len(seq_tower)}")
    print(f"Parameters: {count_params(seq_tower):,}")

    x = torch.randn(4, 256)
    y = seq_tower(x)
    print(f"\nForward: {x.shape} -> {y.shape}")
    print(f"Output sample: {y[0, :5]}")

    # =========================================================================
    section("CHAINED CONSTRUCTION - Fluent API")
    # =========================================================================

    chained = (
        SequentialTower('chained')
        .append(nn.Linear(128, 256))
        .append(nn.GELU())
        .append(nn.Dropout(0.1))
        .append(nn.Linear(256, 256))
        .append(nn.GELU())
        .append(nn.Linear(256, 128))
    )

    print(f"Tower:\n{chained}")
    print(f"\nBuilt with fluent chaining - clean and readable")

    x = torch.randn(4, 128)
    y = chained(x)
    print(f"Forward: {x.shape} -> {y.shape}")

    # =========================================================================
    section("EXTEND CONSTRUCTION - Bulk Addition")
    # =========================================================================

    extended = SequentialTower('extended')
    extended.extend([
        nn.Linear(64, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Linear(128, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Linear(128, 64),
    ])

    print(f"Tower:\n{extended}")
    print(f"\nStage count: {len(extended)}")
    print(f"Parameters: {count_params(extended):,}")

    x = torch.randn(4, 64)
    y = extended(x)
    print(f"Forward: {x.shape} -> {y.shape}")

    # =========================================================================
    section("RESIDUAL TOWER - Skip Connection")
    # =========================================================================

    residual = ResidualTower('residual')
    residual.extend([
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 256),
    ])

    print(f"Tower:\n{residual}")

    x = torch.randn(4, 256)
    y = residual(x)
    print(f"\nForward: {x.shape} -> {y.shape}")

    # Verify residual connection works
    diff = (y - x).abs().mean()
    print(f"Mean diff from input: {diff:.4f} (would be 0 if identity)")

    # =========================================================================
    section("PRENORM TOWER - Using Components")
    # =========================================================================

    prenorm = PreNormTower('prenorm', dim=512)
    prenorm.extend([
        nn.Linear(512, 2048),
        nn.GELU(),
        nn.Linear(2048, 512),
    ])

    print(f"Tower:\n{prenorm}")
    print(f"\nAccess patterns:")
    print(f"  tower[0] (stage): {prenorm[0]}")
    print(f"  tower['norm'] (component): {prenorm['norm']}")
    print(f"\nParameters: {count_params(prenorm):,}")

    x = torch.randn(4, 16, 512)
    y = prenorm(x)
    print(f"Forward: {x.shape} -> {y.shape}")

    # =========================================================================
    section("GATED TOWER - Learned Gating")
    # =========================================================================

    gated = GatedTower('gated', dim=256)
    gated.extend([
        nn.Linear(256, 512),
        nn.GELU(),
        nn.Linear(512, 256),
    ])

    print(f"Tower:\n{gated}")
    print(f"\nGate component: {gated['gate']}")

    x = torch.randn(4, 256)
    y = gated(x)
    print(f"Forward: {x.shape} -> {y.shape}")

    # =========================================================================
    section("DUAL PATH TOWER - Non-Linear Topology")
    # =========================================================================

    dual = DualPathTower('dual_path')
    dual.extend([
        nn.Linear(128, 128),  # Path A - stage 0
        nn.Linear(128, 128),  # Path B - stage 1
        nn.Linear(128, 128),  # Path A - stage 2
        nn.Linear(128, 128),  # Path B - stage 3
    ])

    print(f"Tower:\n{dual}")
    print(f"\nEven stages: Path A")
    print(f"Odd stages: Path B")
    print(f"Output: A + B")

    x = torch.randn(4, 128)
    y = dual(x)
    print(f"\nForward: {x.shape} -> {y.shape}")

    # =========================================================================
    section("ATTENTION TOWER - Mixed Architecture")
    # =========================================================================

    attn_tower = AttentionTower('attention', dim=256, num_heads=8)
    attn_tower.extend([
        nn.Linear(256, 1024),
        nn.GELU(),
        nn.Linear(1024, 256),
    ])

    print(f"Tower:\n{attn_tower}")
    print(f"\nComponents:")
    print(f"  attn: {attn_tower['attn']}")
    print(f"  norm1: {attn_tower['norm1']}")
    print(f"  norm2: {attn_tower['norm2']}")
    print(f"\nStages (FFN): {len(attn_tower)}")
    print(f"Parameters: {count_params(attn_tower):,}")

    x = torch.randn(4, 16, 256)
    y = attn_tower(x)
    print(f"\nForward: {x.shape} -> {y.shape}")

    # =========================================================================
    section("CONDITIONAL TOWER - Config Objects")
    # =========================================================================

    conditional = ConditionalTower('conditional')
    conditional.extend([
        nn.Linear(64, 128),
        nn.GELU(),
        nn.Linear(128, 64),
    ])

    print(f"Tower:\n{conditional}")
    print(f"\nConfig object: {conditional['config']}")

    x = torch.randn(4, 64)

    # With residual
    conditional['config']['use_residual'] = True
    y1 = conditional(x)

    # Without residual
    conditional['config']['use_residual'] = False
    y2 = conditional(x)

    print(f"\nWith residual: {y1[0, :3]}")
    print(f"Without residual: {y2[0, :3]}")
    print(f"Different outputs: {not torch.allclose(y1, y2)}")

    # =========================================================================
    section("INSERT / POP - Dynamic Modification")
    # =========================================================================

    modifiable = SequentialTower('modifiable')
    modifiable.extend([
        nn.Linear(64, 64),
        nn.Linear(64, 64),
    ])

    print(f"Initial: {len(modifiable)} stages")
    print(f"  Stage 0: {modifiable[0]}")
    print(f"  Stage 1: {modifiable[1]}")

    # Insert activation in the middle
    modifiable.insert(1, nn.ReLU())
    print(f"\nAfter insert(1, ReLU): {len(modifiable)} stages")
    print(f"  Stage 0: {modifiable[0]}")
    print(f"  Stage 1: {modifiable[1]}")
    print(f"  Stage 2: {modifiable[2]}")

    # Pop the activation
    popped = modifiable.pop(1)
    print(f"\nPopped stage 1: {popped}")
    print(f"After pop: {len(modifiable)} stages")

    # =========================================================================
    section("ITERATION - Deterministic Order")
    # =========================================================================

    iter_tower = SequentialTower('iteration')
    iter_tower.extend([
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    ])

    print("Iterating over stages:")
    for i, stage in enumerate(iter_tower):
        print(f"  {i}: {stage}")

    print(f"\nMembership checks:")
    print(f"  0 in tower: {0 in iter_tower}")
    print(f"  10 in tower: {10 in iter_tower}")
    print(f"  iter_tower[0] in tower: {iter_tower[0] in iter_tower}")

    # =========================================================================
    section("NESTED TOWERS - Hierarchical Composition")
    # =========================================================================

    inner_a = SequentialTower('inner_a')
    inner_a.extend([nn.Linear(256, 256), nn.GELU()])

    inner_b = SequentialTower('inner_b')
    inner_b.extend([nn.Linear(256, 256), nn.GELU()])

    inner_c = SequentialTower('inner_c')
    inner_c.extend([nn.Linear(256, 256), nn.GELU()])

    outer = SequentialTower('outer')
    outer.append(inner_a)
    outer.append(inner_b)
    outer.append(inner_c)

    print(f"Outer tower:\n{outer}")
    print(f"\nNested structure:")
    print(f"  outer[0]: {outer[0].name}")
    print(f"  outer[1]: {outer[1].name}")
    print(f"  outer[2]: {outer[2].name}")
    print(f"\nTotal parameters: {count_params(outer):,}")

    x = torch.randn(4, 256)
    y = outer(x)
    print(f"Forward through nested: {x.shape} -> {y.shape}")

    # =========================================================================
    section("DEVICE MOVEMENT - Hardware Control")
    # =========================================================================

    print(f"Target device: {device}")

    gpu_tower = SequentialTower('gpu', strict=False)
    gpu_tower.extend([
        nn.Linear(128, 256),
        nn.GELU(),
        nn.Linear(256, 128),
    ])

    print(f"Before move - device: {gpu_tower.device}")

    gpu_tower.network_to(device=device)
    print(f"After network_to - device: {gpu_tower.device}")

    x = torch.randn(4, 128, device=device)
    y = gpu_tower(x)
    print(f"Forward on {device}: {x.shape} -> {y.shape}")
    print(f"Output device: {y.device}")

    # =========================================================================
    section("COMPILATION - torch.compile")
    # =========================================================================

    compile_tower = SequentialTower('compile', strict=False)
    compile_tower.extend([
        nn.Linear(256, 512),
        nn.GELU(),
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Linear(512, 256),
    ])
    compile_tower.network_to(device=device)

    x = torch.randn(32, 256, device=device)

    print("Compiling tower...")
    t0 = time.perf_counter()
    compiled_tower = torch.compile(compile_tower)

    # Warmup
    with torch.no_grad():
        _ = compiled_tower(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    compile_time = time.perf_counter() - t0
    print(f"Compile time: {compile_time:.2f}s")

    # Benchmark
    iterations = 100

    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = compile_tower(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        eager_time = (time.perf_counter() - t0) / iterations

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = compiled_tower(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        compiled_time = (time.perf_counter() - t0) / iterations

    print(f"Eager: {eager_time*1000:.3f}ms")
    print(f"Compiled: {compiled_time*1000:.3f}ms")
    print(f"Speedup: {eager_time/compiled_time:.2f}x")

    # =========================================================================
    section("UUID UNIQUENESS")
    # =========================================================================

    t1 = SequentialTower('encoder')
    t2 = SequentialTower('encoder')
    t3 = SequentialTower('encoder')

    print(f"Three towers with same name:")
    print(f"  Tower 1: name='{t1.name}', uuid='{t1.uuid[:12]}...'")
    print(f"  Tower 2: name='{t2.name}', uuid='{t2.uuid[:12]}...'")
    print(f"  Tower 3: name='{t3.name}', uuid='{t3.uuid[:12]}...'")
    print(f"\nSame name: {t1.name == t2.name == t3.name}")
    print(f"Unique UUIDs: {len({t1.uuid, t2.uuid, t3.uuid}) == 3}")

    # =========================================================================
    section("COMPLEX REAL-WORLD EXAMPLE")
    # =========================================================================

    class TransformerBlockTower(BaseTower):
        """
        A complete transformer block as a tower.

        This demonstrates how BaseTower can represent
        any self-contained AI mechanism.
        """

        def __init__(self, name: str, dim: int, num_heads: int, mlp_ratio: int = 4):
            super().__init__(name, strict=False)

            # Attention components
            self.attach('norm1', nn.LayerNorm(dim))
            self.attach('attn', nn.MultiheadAttention(dim, num_heads, batch_first=True))

            # FFN components
            self.attach('norm2', nn.LayerNorm(dim))

            # FFN as stages
            self.extend([
                nn.Linear(dim, dim * mlp_ratio),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * mlp_ratio, dim),
                nn.Dropout(0.1),
            ])

            # Config
            self.attach('config', {
                'dim': dim,
                'num_heads': num_heads,
                'mlp_ratio': mlp_ratio,
            })

        def forward(self, x: Tensor) -> Tensor:
            # Self-attention with residual
            residual = x
            x = self['norm1'](x)
            x, _ = self['attn'](x, x, x)
            x = x + residual

            # FFN with residual
            residual = x
            x = self['norm2'](x)
            for stage in self.stages:
                x = stage(x)
            x = x + residual

            return x

    transformer_block = TransformerBlockTower('transformer', dim=512, num_heads=8)
    transformer_block.network_to(device=device)

    print(f"Transformer Block Tower:\n{transformer_block}")
    print(f"\nConfig: {transformer_block['config']}")
    print(f"Parameters: {count_params(transformer_block):,}")

    x = torch.randn(4, 32, 512, device=device)
    y = transformer_block(x)
    print(f"\nForward: {x.shape} -> {y.shape}")

    # Stack multiple blocks
    class TransformerTower(BaseTower):
        """Stack of transformer blocks."""

        def __init__(self, name: str, dim: int, num_heads: int, depth: int):
            super().__init__(name, strict=False)

            for i in range(depth):
                self.append(TransformerBlockTower(f'{name}_block_{i}', dim, num_heads))

            self.attach('final_norm', nn.LayerNorm(dim))

        def forward(self, x: Tensor) -> Tensor:
            for stage in self.stages:
                x = stage(x)
            return self['final_norm'](x)

    full_transformer = TransformerTower('full_transformer', dim=512, num_heads=8, depth=6)
    full_transformer.network_to(device=device)

    print(f"\n\nFull Transformer Tower:")
    print(f"  Depth: {len(full_transformer)} blocks")
    print(f"  Parameters: {count_params(full_transformer):,}")

    x = torch.randn(4, 32, 512, device=device)
    y = full_transformer(x)
    print(f"  Forward: {x.shape} -> {y.shape}")

    # =========================================================================
    section("PARAMETER SUMMARY")
    # =========================================================================

    all_towers = [
        ('SequentialTower (5 layers)', seq_tower),
        ('ResidualTower', residual),
        ('PreNormTower', prenorm),
        ('GatedTower', gated),
        ('DualPathTower', dual),
        ('AttentionTower', attn_tower),
        ('ConditionalTower', conditional),
        ('NestedTower (3 inner)', outer),
        ('TransformerBlock', transformer_block),
        ('FullTransformer (6 blocks)', full_transformer),
    ]

    print(f"{'Tower Type':<35} {'Stages':>8} {'Params':>15}")
    print("-" * 60)
    for name, tower in all_towers:
        print(f"{name:<35} {len(tower):>8} {count_params(tower):>15,}")

    # =========================================================================
    section("ALL TESTS PASSED")
    # =========================================================================

    print("\nBaseTower provides:")
    print("  ✓ Ordered stage pipeline (nn.ModuleList)")
    print("  ✓ Named component storage (nn.ModuleDict)")
    print("  ✓ Non-module object storage (dict)")
    print("  ✓ Sequential-like interface (append/extend/insert/pop)")
    print("  ✓ Dual indexing (int for stages, str for components)")
    print("  ✓ Deterministic iteration over stages")
    print("  ✓ Chainable fluent construction")
    print("  ✓ Nested tower composition")
    print("  ✓ Device movement with network_to()")
    print("  ✓ torch.compile compatibility")
    print("  ✓ UUID uniqueness for addressing")

    print("\nTower patterns demonstrated:")
    print("  ✓ Sequential processing")
    print("  ✓ Residual connections")
    print("  ✓ Pre-normalization")
    print("  ✓ Learned gating")
    print("  ✓ Multi-path topologies")
    print("  ✓ Mixed stage/component architectures")
    print("  ✓ Config-driven behavior")
    print("  ✓ Hierarchical nesting")
    print("  ✓ Full transformer blocks")

    print("\nBaseTower is ready for pool coordination.")