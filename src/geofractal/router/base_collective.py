"""
geofractal.router.base_collective
=================================

BaseCollective - Coordination layer for multiple WideRouters.

Architecture hierarchy:
    BaseTower      - Single processing unit (stages + components)
    WideRouter     - Coordinates towers on ONE device
    BaseCollective - Coordinates routers across MULTIPLE devices

BaseCollective doesn't compile itself - it compiles internal components
based on device placement. This enables:
    - Multi-GPU distribution (one WideRouter per GPU)
    - Pipeline parallelism (routers as pipeline stages)
    - Hybrid strategies (some routers local, some remote)

Usage:
    class MyCollective(BaseCollective):
        def __init__(self, num_routers: int, dim: int):
            super().__init__('my_collective')

            for i in range(num_routers):
                router = MyWideRouter(f'router_{i}', dim)
                self.attach_router(f'router_{i}', router)

        def forward(self, x: Tensor) -> Tensor:
            # Execute routers (potentially distributed)
            results = self.collective_forward(x)
            return self.fuse(results)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    Optional, Dict, List, Tuple, Any, Union,
    Callable, Set, Sequence
)
from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.base_router import BaseRouter
from geofractal.router.wide_router import WideRouter


# =============================================================================
# DEVICE PLACEMENT
# =============================================================================

@dataclass
class DevicePlacement:
    """Describes where a router should execute."""
    device: torch.device
    rank: int = 0  # For distributed
    pipeline_stage: int = 0  # For pipeline parallelism

    @classmethod
    def local(cls, device: Union[str, torch.device] = 'cuda') -> 'DevicePlacement':
        """Single-device local placement."""
        if isinstance(device, str):
            device = torch.device(device)
        return cls(device=device, rank=0, pipeline_stage=0)

    @classmethod
    def distributed(cls, rank: int, device: Union[str, torch.device] = None) -> 'DevicePlacement':
        """Distributed placement on specific rank."""
        if device is None:
            device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        return cls(device=device, rank=rank, pipeline_stage=0)


@dataclass
class RouterHealth:
    """Health metrics for a router's towers."""
    name: str
    alive_towers: int
    dead_towers: int  # gradient < threshold
    dominant_towers: List[str]  # gradient >> median
    starving_towers: List[str]  # gradient << median
    gradient_spread: float  # max/min ratio (log scale)

    @property
    def is_collapsed(self) -> bool:
        """True if most towers are dead or one dominates completely."""
        total = self.alive_towers + self.dead_towers
        if total == 0:
            return True
        return self.dead_towers / total > 0.5 or self.gradient_spread > 10


# =============================================================================
# BASE COLLECTIVE
# =============================================================================

class BaseCollective(BaseRouter):
    """
    Coordination layer for multiple WideRouters.

    BaseCollective manages a collection of WideRouters that may execute
    on different devices. It provides:

    - Router registration and discovery
    - Device placement strategies
    - Health monitoring (gradient flow, tower collapse)
    - Selective compilation (per-router, not whole-collective)
    - Distributed-ready forward patterns

    Unlike WideRouter.compile(), BaseCollective.compile_routers() compiles
    each router independently, allowing different devices and strategies.

    Attributes:
        routers: Dict of registered WideRouters
        placements: Device placement per router
        health: Latest health metrics per router
    """

    def __init__(
        self,
        name: str,
        strict: bool = False,
        auto_discover: bool = True,
        **kwargs,
    ):
        super().__init__(name, strict=strict, **kwargs)

        # Router tracking
        self.objects['_router_names'] = []
        self.objects['_placements'] = {}  # name -> DevicePlacement
        self.objects['_health'] = {}  # name -> RouterHealth
        self.objects['_compiled'] = {}  # name -> compiled module
        self.objects['_auto_discover'] = auto_discover

        # Collective state
        self.objects['_collective_prepared'] = False
        self.objects['_gradient_equalization'] = False
        self.objects['_dead_threshold'] = 1e-10  # Below this = dead
        self.objects['_dominance_ratio'] = 100  # Above median*this = dominant

    # =========================================================================
    # ROUTER REGISTRATION
    # =========================================================================

    @property
    def router_names(self) -> List[str]:
        """Names of registered routers."""
        return self.objects['_router_names']

    @property
    def routers(self) -> Dict[str, WideRouter]:
        """Dict of registered routers."""
        return {name: self[name] for name in self.router_names}

    def attach_router(
        self,
        name: str,
        router: WideRouter,
        placement: DevicePlacement = None,
    ) -> None:
        """
        Attach and register a WideRouter.

        Args:
            name: Router identifier
            router: WideRouter instance
            placement: Optional device placement (defaults to current device)
        """
        if not isinstance(router, WideRouter):
            raise TypeError(f"Expected WideRouter, got {type(router).__name__}")

        self.attach(name, router)

        if name not in self.objects['_router_names']:
            self.objects['_router_names'].append(name)

        # Default placement
        if placement is None:
            # Infer from router's current device
            device = next(router.parameters()).device if list(router.parameters()) else torch.device('cpu')
            placement = DevicePlacement.local(device)

        self.objects['_placements'][name] = placement
        self.objects['_collective_prepared'] = False

    def detach_router(self, name: str) -> Optional[WideRouter]:
        """Remove a router from the collective."""
        if name in self.objects['_router_names']:
            self.objects['_router_names'].remove(name)
            self.objects['_placements'].pop(name, None)
            self.objects['_health'].pop(name, None)
            self.objects['_compiled'].pop(name, None)
            self.objects['_collective_prepared'] = False
        return self.detach(name)

    def discover_routers(self) -> List[str]:
        """Auto-discover WideRouters from attached components."""
        for name, component in self.components.items():
            if isinstance(component, WideRouter):
                if name not in self.objects['_router_names']:
                    self.attach_router(name, component)
        return self.router_names

    # =========================================================================
    # DEVICE PLACEMENT
    # =========================================================================

    def get_placement(self, name: str) -> Optional[DevicePlacement]:
        """Get device placement for a router."""
        return self.objects['_placements'].get(name)

    def set_placement(self, name: str, placement: DevicePlacement) -> None:
        """Set device placement for a router."""
        if name not in self.router_names:
            raise KeyError(f"Router '{name}' not registered")
        self.objects['_placements'][name] = placement
        self.objects['_collective_prepared'] = False

    def apply_placements(self) -> None:
        """Move routers to their assigned devices."""
        for name in self.router_names:
            placement = self.objects['_placements'].get(name)
            if placement:
                router = self[name]
                router.to(placement.device)

    # =========================================================================
    # COMPILATION
    # =========================================================================

    def prepare_routers(self, device: torch.device = None) -> 'BaseCollective':
        """
        Prepare all routers for compilation.

        Calls prepare_for_compile() on each router and applies device placements.

        Args:
            device: Target device. If None, inferred from current parameters.
        """
        if self.objects.get('_auto_discover', True):
            self.discover_routers()

        # Infer device if not specified
        if device is None:
            for p in self.parameters():
                device = p.device
                break

        # Update placements to use the inferred/specified device
        if device is not None:
            for name in self.router_names:
                self.objects['_placements'][name] = DevicePlacement.local(device)

        self.apply_placements()

        for name in self.router_names:
            router = self[name]
            placement = self.objects['_placements'].get(name)
            router_device = placement.device if placement else device
            router.prepare_for_compile(router_device)

        self.objects['_collective_prepared'] = True
        return self

    def compile_routers(
        self,
        mode: str = 'reduce-overhead',
        selective: List[str] = None,
        **kwargs,
    ) -> 'BaseCollective':
        """
        Compile routers independently.

        Unlike torch.compile(collective), this compiles each router
        separately, respecting device placements.

        Args:
            mode: Compilation mode for torch.compile
            selective: Only compile these routers (None = all)
            **kwargs: Passed to torch.compile

        Returns:
            self (routers compiled in-place, accessible via get_compiled)
        """
        if not self.objects.get('_collective_prepared', False):
            self.prepare_routers()

        names = selective or self.router_names

        for name in names:
            if name not in self.router_names:
                warnings.warn(f"Router '{name}' not found, skipping")
                continue

            router = self[name]
            compiled = torch.compile(router, mode=mode, **kwargs)
            self.objects['_compiled'][name] = compiled

        return self

    def get_compiled(self, name: str) -> nn.Module:
        """
        Get compiled version of router (or original if not compiled).
        """
        return self.objects['_compiled'].get(name, self[name])

    # =========================================================================
    # HEALTH MONITORING
    # =========================================================================

    @torch.no_grad()
    def check_health(self) -> Dict[str, RouterHealth]:
        """
        Analyze gradient health across all routers.

        Returns dict of RouterHealth per router.
        Call after backward() to get meaningful results.
        """
        dead_threshold = self.objects['_dead_threshold']
        dominance_ratio = self.objects['_dominance_ratio']

        health = {}

        for router_name in self.router_names:
            router = self[router_name]

            # Collect gradient norms per tower
            tower_grads = {}
            for tower_name in router.tower_names:
                tower = router[tower_name]
                grad_norm = 0.0
                param_count = 0
                for p in tower.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                        param_count += 1
                tower_grads[tower_name] = (grad_norm ** 0.5) / max(param_count, 1)

            if not tower_grads:
                health[router_name] = RouterHealth(
                    name=router_name,
                    alive_towers=0,
                    dead_towers=0,
                    dominant_towers=[],
                    starving_towers=[],
                    gradient_spread=0.0,
                )
                continue

            # Analyze
            values = list(tower_grads.values())
            median = sorted(values)[len(values) // 2] if values else 0
            max_grad = max(values) if values else 0
            min_grad = min(v for v in values if v > 0) if any(v > 0 for v in values) else 1e-30

            alive = [k for k, v in tower_grads.items() if v > dead_threshold]
            dead = [k for k, v in tower_grads.items() if v <= dead_threshold]
            dominant = [k for k, v in tower_grads.items() if v > median * dominance_ratio]
            starving = [k for k, v in tower_grads.items() if dead_threshold < v < median / dominance_ratio]

            import math
            spread = math.log10(max_grad / min_grad) if min_grad > 0 else float('inf')

            health[router_name] = RouterHealth(
                name=router_name,
                alive_towers=len(alive),
                dead_towers=len(dead),
                dominant_towers=dominant,
                starving_towers=starving,
                gradient_spread=spread,
            )

        self.objects['_health'] = health
        return health

    def is_collapsed(self) -> bool:
        """Check if any router has collapsed."""
        health = self.objects.get('_health', {})
        return any(h.is_collapsed for h in health.values())

    # =========================================================================
    # GRADIENT EQUALIZATION
    # =========================================================================

    def enable_gradient_equalization(self, enable: bool = True) -> None:
        """Enable/disable gradient equalization across towers."""
        self.objects['_gradient_equalization'] = enable

    def equalize_gradients(self) -> None:
        """
        Normalize gradient magnitudes across towers within each router.

        Call after backward(), before optimizer.step().
        Helps prevent tower collapse by ensuring all towers receive
        comparable learning signal.
        """
        for router_name in self.router_names:
            router = self[router_name]
            self._equalize_router_gradients(router)

    def _equalize_router_gradients(self, router: WideRouter) -> None:
        """Equalize gradients within a single router."""
        # Collect gradient norms
        tower_grad_norms = {}
        for name in router.tower_names:
            tower = router[name]
            norm_sq = 0.0
            for p in tower.parameters():
                if p.grad is not None:
                    norm_sq += p.grad.norm().item() ** 2
            tower_grad_norms[name] = norm_sq ** 0.5

        if not tower_grad_norms:
            return

        # Compute median (robust to outliers)
        values = sorted(tower_grad_norms.values())
        median = values[len(values) // 2]

        if median < 1e-20:
            return  # All dead, nothing to equalize

        # Scale each tower's gradients to median
        for name, norm in tower_grad_norms.items():
            if norm < 1e-20:
                continue  # Can't scale from zero

            scale = median / norm
            tower = router[name]
            for p in tower.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)

    # =========================================================================
    # TOWER RESURRECTION
    # =========================================================================

    def resurrect_dead_towers(self, reinit_scale: float = 0.02) -> List[str]:
        """
        Reinitialize towers that have collapsed (near-zero gradients).

        Args:
            reinit_scale: Std for reinitialization

        Returns:
            List of resurrected tower names
        """
        health = self.check_health()
        resurrected = []

        for router_name, router_health in health.items():
            router = self[router_name]

            for tower_name in router_health.starving_towers + [
                t for t in router.tower_names
                if t not in router_health.dominant_towers
            ]:
                tower = router[tower_name]

                # Check if truly dead
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in tower.parameters()
                    if p.grad is not None
                ) ** 0.5

                if grad_norm < self.objects['_dead_threshold']:
                    self._reinit_tower(tower, scale=reinit_scale)
                    resurrected.append(f"{router_name}/{tower_name}")

        return resurrected

    def _reinit_tower(self, tower: nn.Module, scale: float = 0.02) -> None:
        """Reinitialize tower parameters."""
        for name, param in tower.named_parameters():
            if 'weight' in name:
                nn.init.trunc_normal_(param, std=scale)
            elif 'bias' in name:
                nn.init.zeros_(param)

    # =========================================================================
    # COLLECTIVE FORWARD
    # =========================================================================

    def collective_forward(
        self,
        x: Tensor,
        router_names: List[str] = None,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Execute multiple routers, collecting their outputs.

        Args:
            x: Input tensor
            router_names: Specific routers to execute (None = all)
            mask: Optional attention mask

        Returns:
            Dict[router_name, Dict[tower_name, output]]
        """
        names = router_names or self.router_names
        results = {}

        for name in names:
            router = self.get_compiled(name)

            # Move input to router's device if needed
            placement = self.objects['_placements'].get(name)
            if placement and x.device != placement.device:
                x_local = x.to(placement.device)
            else:
                x_local = x

            # Execute via wide_forward
            if isinstance(router, WideRouter):
                results[name] = router.wide_forward(x_local, mask=mask)
            else:
                # Compiled version - call forward which should use wide_forward
                out = router(x_local, mask=mask) if mask is not None else router(x_local)
                # If it returns a structured result, extract opinions
                if hasattr(out, 'opinions'):
                    results[name] = {k: v.opinion for k, v in out.opinions.items()}
                else:
                    results[name] = {'output': out}

        return results

    # =========================================================================
    # FORWARD (abstract)
    # =========================================================================

    @abstractmethod
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Subclass must implement forward.

        Typical pattern:
            def forward(self, x):
                results = self.collective_forward(x)
                return self.fuse(results)
        """
        raise NotImplementedError(
            "BaseCollective subclass must implement forward(). "
            "Use self.collective_forward(x) to execute registered routers."
        )

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def reset(self) -> None:
        """Reset all state."""
        super().reset()
        self.objects['_health'] = {}
        self.objects['_compiled'] = {}
        self.objects['_collective_prepared'] = False

        for name in self.router_names:
            router = self[name]
            if hasattr(router, 'reset'):
                router.reset()

    def __repr__(self) -> str:
        router_count = len(self.router_names)
        prepared = self.objects.get('_collective_prepared', False)
        compiled = len(self.objects.get('_compiled', {}))
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"routers={router_count}, "
            f"prepared={prepared}, "
            f"compiled={compiled}"
            f")"
        )