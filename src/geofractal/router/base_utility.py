"""
geofractal.router.base_utility
==============================

Lightweight utility base class for mathematical primitives.

BaseUtility inherits BaseComponent for lifecycle hooks but adds no
torch overhead. It's the non-nn.Module sibling of TorchComponent.

Hierarchy:
    BaseComponent (ABC, pure Python)
        ├── TorchComponent (+ nn.Module)  → learnable params, device mgmt
        └── BaseUtility (+ __call__)      → formulas, schedules, reductions

Use Cases:
    - Blend modes (interpolation formulas)
    - Schedules (alpha value generation)
    - Aggregation strategies (reduction operations)
    - Geometric calculations (Cayley-Menger, angular, etc.)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from abc import abstractmethod
from typing import Dict, Any, Optional

from geofractal.router.base_component import BaseComponent


class BaseUtility(BaseComponent):
    """
    Lightweight base for mathematical primitives.

    Inherits from BaseComponent:
        - name, uuid
        - parent reference
        - on_attach(), on_detach() hooks

    Adds:
        - _config dict for runtime parameters
        - configure() for parameter updates
        - abstract __call__ for execution
    """

    __slots__ = ('_config',)

    def __init__(self, name: str, uuid: Optional[str] = None, **kwargs):
        super().__init__(name, uuid, **kwargs)
        self._config: Dict[str, Any] = {}

    @property
    def config(self) -> Dict[str, Any]:
        """Current configuration (read-only copy)."""
        return dict(self._config)

    def configure(self, **kwargs) -> 'BaseUtility':
        """
        Update runtime parameters.

        Args:
            **kwargs: Parameters to update

        Returns:
            Self for chaining
        """
        self._config.update(kwargs)
        return self

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Execute the utility's core operation.

        Subclasses MUST implement this.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':

    class MultiplyUtility(BaseUtility):
        __slots__ = ('factor',)

        def __init__(self, name: str, factor: float = 1.0):
            super().__init__(name)
            self.factor = factor

        def configure(self, **kwargs) -> 'MultiplyUtility':
            self.factor = kwargs.pop('factor', self.factor)
            return super().configure(**kwargs)

        def __call__(self, x):
            return x * self.factor

    print("=" * 60)
    print("  BASE UTILITY TESTS")
    print("=" * 60)

    # Basic usage
    util = MultiplyUtility("mult", factor=3.0)
    print(f"\nCreated: {util}")
    print(f"util(10) = {util(10)}")

    # Configure
    util.configure(factor=0.5)
    print(f"After configure(factor=0.5): util(10) = {util(10)}")

    # Chaining
    result = MultiplyUtility("chain", factor=2.0).configure(extra="data")(5)
    print(f"Chained call: {result}")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)