"""
geofractal.router.prefab.wide_collective_builder
=================================================

Experimental builder for creating collectives with:
- Gradient health monitoring
- Tower collapse detection and resurrection
- Automatic gradient equalization
- Multi-router coordination

This is the experimental playground for collective-level optimizations.

Usage:
    from geofractal.router.prefab.wide_collective_builder import (
        quick_experiment,
        HealthyCollective,
        CollectiveConfig,
    )

    # Quick experiment setup
    collective = quick_experiment(
        geometries=['simplex', ['fibonacci', 3], ['helix', 3]],
        dim=64,
        depth=4,
        equalize_gradients=True,
        resurrect_dead=True,
    )

    # Or full control
    config = CollectiveConfig(
        dim=64,
        depth=4,
        gradient_equalization=True,
        collapse_threshold=1e-10,
        dominance_ratio=100,
    )
    collective = HealthyCollective(config)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Optional, Dict, List, Tuple, Any, Union,
    Callable, Literal
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.base_collective import BaseCollective, DevicePlacement, RouterHealth
from geofractal.router.wide_router import WideRouter
from geofractal.router.prefab.geometric_tower_builder import (
    quick_collective,
    quick_pair,
    FusionType,
    ConfigurableCollective,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CollectiveConfig:
    """
    Configuration for experimental collective.

    FusionType options:
        - mean, sum, concat (simple)
        - walker_static, walker_inception, walker_shiva, walker_slerp (walker)
    """

    # Architecture
    dim: int = 64
    depth: int = 4
    num_heads: int = 4
    fingerprint_dim: int = 64

    # Geometries
    geometries: List[Union[str, List]] = field(default_factory=lambda: [
        'simplex',
        ['fibonacci', 3],
        ['golden', 3],
        ['helix', 3],
    ])

    # Fusion
    fusion_type: str = 'walker_inception'

    # Health management
    gradient_equalization: bool = True
    equalization_frequency: int = 1  # Every N batches

    collapse_detection: bool = True
    collapse_threshold: float = 1e-10  # Gradient below this = dead
    dominance_ratio: float = 100.0  # Gradient > median * this = dominant

    resurrection_enabled: bool = False  # Risky - enable manually
    resurrection_scale: float = 0.02

    # Normalization
    normalize_opinions: bool = False  # L2 normalize before fusion
    opinion_temperature: float = 1.0  # Scale opinions

    # Per-tower learning rate scaling
    adaptive_lr: bool = False
    lr_scale_min: float = 0.1
    lr_scale_max: float = 10.0


# =============================================================================
# HEALTHY COLLECTIVE
# =============================================================================

class HealthyCollective(BaseCollective):
    """
    Collective with built-in health management.

    Monitors gradient flow, detects collapse, and can intervene
    to keep all towers learning.

    Features:
        - Gradient equalization (normalize across towers)
        - Collapse detection (identify dead/dominant towers)
        - Opinion normalization (prevent one tower dominating fusion)
        - Adaptive resurrection (reinitialize dead towers)
    """

    def __init__(
        self,
        config: CollectiveConfig,
        name: str = 'healthy_collective',
    ):
        super().__init__(name, auto_discover=True)

        self.config = config

        # Build the inner collective using geometric_tower_builder
        self.inner = quick_collective(
            geometries=config.geometries,
            dim=config.dim,
            depth=config.depth,
            num_heads=config.num_heads,
            fingerprint_dim=config.fingerprint_dim,
            fusion_type=config.fusion_type,
            name=f'{name}_inner',
        )

        # Attach as a router
        self.attach_router('inner', self.inner)

        # Health settings
        self.objects['_dead_threshold'] = config.collapse_threshold
        self.objects['_dominance_ratio'] = config.dominance_ratio
        self.objects['_gradient_equalization'] = config.gradient_equalization

        # Tracking
        self._batch_count = 0
        self._last_health: Optional[Dict[str, RouterHealth]] = None
        self._intervention_history: List[Dict] = []

    @property
    def tower_names(self) -> List[str]:
        """Expose inner tower names."""
        return self.inner.tower_names

    def forward(self, x: Tensor, mask: Tensor = None) -> Any:
        """Forward through inner collective."""
        return self.inner(x, mask=mask)

    # =========================================================================
    # TRAINING HOOKS
    # =========================================================================

    def pre_backward(self) -> None:
        """Call before loss.backward()."""
        pass  # Placeholder for future hooks

    def post_backward(self) -> Dict[str, Any]:
        """
        Call after loss.backward(), before optimizer.step().

        Returns intervention report.
        """
        self._batch_count += 1
        report = {'batch': self._batch_count, 'interventions': []}

        # Gradient equalization
        if self.config.gradient_equalization:
            if self._batch_count % self.config.equalization_frequency == 0:
                self.equalize_gradients()
                report['interventions'].append('gradient_equalization')

        # Collapse detection
        if self.config.collapse_detection:
            health = self.check_health()
            self._last_health = health

            for router_name, router_health in health.items():
                if router_health.is_collapsed:
                    report['interventions'].append(f'collapse_detected:{router_name}')

                    if self.config.resurrection_enabled:
                        resurrected = self.resurrect_dead_towers(
                            reinit_scale=self.config.resurrection_scale
                        )
                        report['interventions'].append(f'resurrected:{resurrected}')

        if report['interventions']:
            self._intervention_history.append(report)

        return report

    def pre_optimizer_step(self) -> None:
        """Call before optimizer.step()."""
        # Apply any per-tower LR scaling if enabled
        if self.config.adaptive_lr:
            self._apply_adaptive_lr()

    def _apply_adaptive_lr(self) -> None:
        """Scale gradients based on tower health (pseudo-LR scaling)."""
        if self._last_health is None:
            return

        for router_name, router_health in self._last_health.items():
            router = self[router_name]

            # Boost starving towers, dampen dominant ones
            for tower_name in router_health.starving_towers:
                tower = router[tower_name]
                for p in tower.parameters():
                    if p.grad is not None:
                        p.grad.mul_(self.config.lr_scale_max)

            for tower_name in router_health.dominant_towers:
                tower = router[tower_name]
                for p in tower.parameters():
                    if p.grad is not None:
                        p.grad.mul_(self.config.lr_scale_min)

    # =========================================================================
    # OPINION PROCESSING
    # =========================================================================

    def normalize_opinions(
        self,
        opinions: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Normalize opinion magnitudes to prevent dominance.

        Args:
            opinions: Dict of tower outputs

        Returns:
            Normalized opinions
        """
        if not self.config.normalize_opinions:
            return opinions

        normalized = {}
        for name, op in opinions.items():
            # L2 normalize
            normalized[name] = F.normalize(op, p=2, dim=-1) * self.config.opinion_temperature

        return normalized

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_health_summary(self) -> str:
        """Get human-readable health summary."""
        if self._last_health is None:
            return "No health data (call post_backward first)"

        lines = ["Health Summary:"]

        for router_name, health in self._last_health.items():
            status = "🔴 COLLAPSED" if health.is_collapsed else "🟢 OK"
            lines.append(f"  {router_name}: {status}")
            lines.append(f"    Alive: {health.alive_towers}, Dead: {health.dead_towers}")
            lines.append(f"    Gradient spread: {health.gradient_spread:.1f} orders of magnitude")

            if health.dominant_towers:
                lines.append(f"    Dominant: {', '.join(health.dominant_towers)}")
            if health.starving_towers:
                lines.append(f"    Starving: {', '.join(health.starving_towers)}")

        return '\n'.join(lines)

    def get_intervention_history(self) -> List[Dict]:
        """Get history of interventions."""
        return self._intervention_history.copy()


# =============================================================================
# TRAINING WRAPPER
# =============================================================================

class CollectiveTrainer:
    """
    Training wrapper that integrates health management into training loop.

    Usage:
        collective = HealthyCollective(config)
        trainer = CollectiveTrainer(collective, optimizer)

        for batch in loader:
            loss = trainer.train_step(batch)
    """

    def __init__(
        self,
        collective: HealthyCollective,
        optimizer: torch.optim.Optimizer,
        gradient_clip: float = 1.0,
    ):
        self.collective = collective
        self.optimizer = optimizer
        self.gradient_clip = gradient_clip
        self.step_count = 0

    def train_step(
        self,
        x: Tensor,
        targets: Tensor,
        loss_fn: Callable = F.cross_entropy,
    ) -> Dict[str, float]:
        """
        Single training step with health management.

        Returns dict with loss and any intervention info.
        """
        self.collective.train()
        self.optimizer.zero_grad()

        # Forward
        output = self.collective(x)

        # Handle structured output
        if hasattr(output, 'fused'):
            logits = output.fused
            if logits.dim() == 3:
                logits = logits[:, 0]  # CLS token
        else:
            logits = output

        # Loss
        loss = loss_fn(logits, targets)

        # Backward with hooks
        self.collective.pre_backward()
        loss.backward()
        report = self.collective.post_backward()

        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.collective.parameters(),
                self.gradient_clip
            )

        # Pre-optimizer hook
        self.collective.pre_optimizer_step()

        # Optimizer step
        self.optimizer.step()
        self.step_count += 1

        return {
            'loss': loss.item(),
            'step': self.step_count,
            'interventions': report.get('interventions', []),
        }


# =============================================================================
# QUICK BUILDERS
# =============================================================================

def quick_experiment(
    geometries: List[Union[str, List]] = None,
    dim: int = 64,
    depth: int = 4,
    num_heads: int = 4,
    equalize_gradients: bool = True,
    resurrect_dead: bool = False,
    normalize_opinions: bool = False,
    fusion_type: str = 'walker_inception',
    name: str = 'experiment',
    **kwargs,
) -> HealthyCollective:
    """
    Quick setup for gradient-healthy experiments.

    Args:
        geometries: Tower geometries (default: simplex + fib/golden/helix pairs)
        dim: Hidden dimension
        depth: Tower depth
        num_heads: Attention heads
        equalize_gradients: Enable gradient equalization
        resurrect_dead: Enable tower resurrection (risky!)
        normalize_opinions: L2 normalize opinions before fusion
        fusion_type: Fusion strategy
        name: Collective name
        **kwargs: Additional CollectiveConfig fields

    Returns:
        HealthyCollective ready for training
    """
    if geometries is None:
        geometries = [
            'simplex',
            ['fibonacci', 3],
            ['golden', 3],
            ['helix', 3],
        ]

    config = CollectiveConfig(
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        geometries=geometries,
        fusion_type=fusion_type,
        gradient_equalization=equalize_gradients,
        resurrection_enabled=resurrect_dead,
        normalize_opinions=normalize_opinions,
        **kwargs,
    )

    return HealthyCollective(config, name=name)


def quick_isolated_experiment(
    geometries: List[Union[str, List]] = None,
    dim: int = 64,
    depth: int = 4,
    num_heads: int = 4,
    fingerprint_dim: int = 64,
    fusion_type: str = 'walker_inception',
    name: str = 'isolated',
) -> 'IsolatedCollective':
    """
    Create collective where each geometry family is isolated.

    Each geometry type gets its own WideRouter, preventing
    cross-geometry gradient competition.
    """
    if geometries is None:
        geometries = [
            'simplex',
            ['fibonacci', 3],
            ['golden', 3],
            ['helix', 3],
        ]

    return IsolatedCollective(
        geometries=geometries,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        fingerprint_dim=fingerprint_dim,
        fusion_type=fusion_type,
        name=name,
    )


# =============================================================================
# ISOLATED COLLECTIVE
# =============================================================================

class IsolatedCollective(BaseCollective):
    """
    Collective where each geometry family has its own router.

    This prevents gradient competition between geometries.
    Each family evolves independently, then outputs are fused.

    Architecture:
        Input
          ├─► SimplexRouter (simplex towers only)
          ├─► FibonacciRouter (fibonacci towers only)
          ├─► GoldenRouter (golden towers only)
          └─► HelixRouter (helix towers only)
                 ↓
              Fusion (combines family outputs)
                 ↓
              Output
    """

    def __init__(
        self,
        geometries: List[Union[str, List]],
        dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        fingerprint_dim: int = 64,
        fusion_type: str = 'walker_inception',
        name: str = 'isolated',
    ):
        super().__init__(name, auto_discover=True)

        self.dim = dim
        self._family_names = []

        # Build separate router for each geometry family
        for geom in geometries:
            if isinstance(geom, str):
                family_name = geom
                family_geoms = [geom]
            else:
                family_name = geom[0]
                count = geom[1] if len(geom) > 1 else 1
                family_geoms = [[family_name, count]]

            # Create isolated router for this family
            router = quick_collective(
                geometries=family_geoms,
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                fingerprint_dim=fingerprint_dim,
                fusion_type=fusion_type,
                name=f'{name}_{family_name}',
            )

            self.attach_router(family_name, router)
            self._family_names.append(family_name)

        # Cross-family fusion
        num_families = len(self._family_names)
        self.fusion = nn.Sequential(
            nn.Linear(dim * num_families, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.attach('fusion', self.fusion)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Forward through isolated families then fuse."""
        family_outputs = []

        for family_name in self._family_names:
            router = self.get_compiled(family_name)
            result = router(x, mask=mask) if mask else router(x)

            # Extract fused output
            if hasattr(result, 'fused'):
                out = result.fused
            else:
                out = result

            # Pool to [B, D]
            if out.dim() == 3:
                out = out[:, 0]  # CLS token

            family_outputs.append(out)

        # Concatenate and fuse
        combined = torch.cat(family_outputs, dim=-1)
        return self.fusion(combined)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Wide Collective Builder Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test HealthyCollective
    print("\n--- HealthyCollective ---")
    config = CollectiveConfig(
        dim=64,
        depth=2,
        geometries=['simplex', ['fibonacci', 2], ['helix', 2]],
        gradient_equalization=True,
    )

    collective = HealthyCollective(config)
    collective.to(device)  # Move to device FIRST
    collective.prepare_routers()  # Then prepare (will use current device)

    print(f"Collective: {collective}")
    print(f"Tower names: {collective.tower_names}")

    # Test forward
    x = torch.randn(4, 65, 64, device=device)  # [B, T, D]
    output = collective(x)
    print(f"Output: {output.fused.shape}")

    # Test health check (after backward)
    target = torch.randn_like(output.fused)
    loss = F.mse_loss(output.fused, target)
    loss.backward()

    report = collective.post_backward()
    print(f"Post-backward report: {report}")
    print(collective.get_health_summary())

    # Test IsolatedCollective
    print("\n--- IsolatedCollective ---")
    isolated = quick_isolated_experiment(
        geometries=['simplex', ['fibonacci', 2], ['helix', 2]],
        dim=64,
        depth=2,
    )
    isolated.to(device)
    isolated.prepare_routers()

    print(f"Isolated: {isolated}")
    output = isolated(x)
    print(f"Output: {output.shape}")

    print("\n✓ Wide Collective Builder ready")