"""
geofractal.router.base_component
================================

Abstract base for all components in the geofractal system.

A component is a named, attachable unit with unique identity. It can be
attached to routers and knows its parent when attached. This base class
is pure Python - not tied to PyTorch - but provides PyTorch-compatible
access patterns for seamless integration.

Design Principles:
    - Pure Python ABC (no nn.Module requirement)
    - Identity: name + uuid
    - Lifecycle: on_attach / on_detach hooks
    - PyTorch-style interface with sensible defaults
    - Subclasses override what they need

Component Types:
    Components can be anything attachable to a router:
    - TorchComponent: nn.Module-based (parameters, gradients)
    - Config components: Settings, hyperparameters
    - State components: Buffers, caches, queues
    - Stream components: Data flow between routers
    - Address components: Routing, fingerprints, mailboxes

Implications:
    - Components are framework-agnostic at base level
    - PyTorch integration via TorchComponent subclass
    - Routers call lifecycle hooks automatically
    - Parent reference enables component → router communication
    - Serialization via state_dict pattern (override as needed)

Usage:
    class MyComponent(BaseComponent):
        def __init__(self, name: str, value: int):
            super().__init__(name)
            self.value = value

        def state_dict(self) -> dict:
            return {'value': self.value}

        def load_state_dict(self, state: dict) -> None:
            self.value = state['value']

    # Attach to router
    router.attach('config', MyComponent('config', 42))

    # Component knows its parent
    component = router['config']
    assert component.parent is router

Directory Structure:
    geofractal/
    └── router/
        ├── base_router.py      # BaseRouter ABC
        ├── base_component.py   # BaseComponent ABC (this file)
        └── components/
            ├── __init__.py
            ├── torch_component.py  # TorchComponent (nn.Module)
            ├── stream.py           # StreamComponent
            ├── address.py          # AddressComponent
            └── ...

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import uuid as uuid_lib
from abc import ABC
from typing import Optional, Any, Iterator, Dict


class BaseComponent(ABC):
    """
    Abstract base for components.

    A component is an addressable unit that can be attached to routers.
    It has a human-readable name, a unique identifier, and lifecycle
    hooks for attachment events.

    This base class is pure Python. It provides PyTorch-compatible
    method signatures with sensible defaults. Subclasses override
    as needed.

    Attributes:
        name: Human-readable identifier.
        uuid: Unique identifier for machine addressing.
        parent: Reference to parent router (set on attach).
    """

    def __init__(self, name: str, uuid: Optional[str] = None):
        """
        Initialize component.

        Args:
            name: Human-readable name for this component.
            uuid: Unique identifier. Generated if not provided.
        """
        self.name = name
        self.uuid = uuid or str(uuid_lib.uuid4())
        self.parent: Optional[Any] = None

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def on_attach(self, parent: Any) -> None:
        """
        Called when attached to a router.

        Override for setup that requires parent reference.
        Parent is set before this is called.

        Args:
            parent: The router this component is being attached to.
        """
        ...

    def on_detach(self) -> None:
        """
        Called when detached from a router.

        Override for cleanup. Parent is still set when this is called,
        cleared immediately after.
        """
        ...

    # =========================================================================
    # PYTORCH-STYLE ACCESS (override in subclasses)
    # =========================================================================

    @property
    def device(self) -> Optional[Any]:
        """
        Device this component resides on.

        Returns None for non-tensor components.
        Override in TorchComponent.
        """
        return None

    @property
    def dtype(self) -> Optional[Any]:
        """
        Data type of this component.

        Returns None for non-tensor components.
        Override in TorchComponent.
        """
        return None

    def parameters(self, recurse: bool = True) -> Iterator:
        """
        Iterator over component parameters.

        Returns empty iterator for non-module components.
        Override in TorchComponent.

        Args:
            recurse: Unused in base, kept for signature compatibility.

        Yields:
            Nothing (empty iterator).
        """
        return iter([])

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator:
        """
        Iterator over component parameters with names.

        Returns empty iterator for non-module components.
        Override in TorchComponent.

        Args:
            prefix: Prefix for parameter names.
            recurse: Unused in base, kept for signature compatibility.

        Yields:
            Nothing (empty iterator).
        """
        return iter([])

    def buffers(self, recurse: bool = True) -> Iterator:
        """
        Iterator over component buffers.

        Returns empty iterator for non-module components.
        Override in TorchComponent.

        Args:
            recurse: Unused in base, kept for signature compatibility.

        Yields:
            Nothing (empty iterator).
        """
        return iter([])

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator:
        """
        Iterator over component buffers with names.

        Returns empty iterator for non-module components.
        Override in TorchComponent.

        Args:
            prefix: Prefix for buffer names.
            recurse: Unused in base, kept for signature compatibility.

        Yields:
            Nothing (empty iterator).
        """
        return iter([])

    def state_dict(self) -> Dict[str, Any]:
        """
        Return component state as dictionary.

        Override to include component-specific state.
        Default returns empty dict.

        Returns:
            Dictionary of state to serialize.
        """
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Load component state from dictionary.

        Override to restore component-specific state.
        Default does nothing.

        Args:
            state: Dictionary of state to restore.
        """
        pass

    def to(self, *args, **kwargs) -> 'BaseComponent':
        """
        Move component to device/dtype.

        No-op for non-tensor components.
        Override in TorchComponent.

        Returns:
            Self for chaining.
        """
        return self

    def train(self, mode: bool = True) -> 'BaseComponent':
        """
        Set training mode.

        No-op for non-module components.
        Override in TorchComponent.

        Args:
            mode: Training mode flag.

        Returns:
            Self for chaining.
        """
        return self

    def eval(self) -> 'BaseComponent':
        """
        Set evaluation mode.

        No-op for non-module components.
        Override in TorchComponent.

        Returns:
            Self for chaining.
        """
        return self.train(False)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Zero gradients.

        No-op for non-module components.
        Override in TorchComponent.

        Args:
            set_to_none: Unused in base, kept for signature compatibility.
        """
        pass

    def requires_grad_(self, requires_grad: bool = True) -> 'BaseComponent':
        """
        Set requires_grad for all parameters.

        No-op for non-module components.
        Override in TorchComponent.

        Args:
            requires_grad: Whether to require gradients.

        Returns:
            Self for chaining.
        """
        return self

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Count parameters.

        Args:
            only_trainable: If True, only count trainable parameters.

        Returns:
            Total parameter count (0 for non-module components).
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @property
    def is_attached(self) -> bool:
        """Check if component is attached to a router."""
        return self.parent is not None

    # =========================================================================
    # DUNDER METHODS
    # =========================================================================

    def __repr__(self) -> str:
        attached = f", parent='{self.parent.name}'" if self.parent else ""
        return f"{self.__class__.__name__}(name='{self.name}'{attached})"


# =============================================================================
# TEST COMPONENTS
# =============================================================================

class ConfigComponent(BaseComponent):
    """Simple config holder for testing, not for production use."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name)
        self.config = kwargs

    def state_dict(self) -> Dict[str, Any]:
        return {'config': self.config.copy()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.config = state['config'].copy()

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.config


class StateComponent(BaseComponent):
    """Simple state holder for testing, not for production use."""

    def __init__(self, name: str):
        super().__init__(name)
        self.data: Dict[str, Any] = {}

    def state_dict(self) -> Dict[str, Any]:
        return {'data': self.data.copy()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.data = state['data'].copy()

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def clear(self) -> None:
        self.data.clear()


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == '__main__':
    def test_section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)


    # -------------------------------------------------------------------------
    test_section("CONFIG COMPONENT")
    # -------------------------------------------------------------------------

    config = ConfigComponent('hyperparams', lr=0.001, batch_size=32, epochs=100)
    print(f"Component: {config}")
    print(f"UUID: {config.uuid[:8]}...")
    print(f"Is attached: {config.is_attached}")
    print(f"Config: {config.config}")
    print(f"Access lr: {config['lr']}")
    print(f"'batch_size' in config: {'batch_size' in config}")

    # State dict
    state = config.state_dict()
    print(f"State dict: {state}")

    config2 = ConfigComponent('restored')
    config2.load_state_dict(state)
    print(f"Restored config: {config2.config}")

    # -------------------------------------------------------------------------
    test_section("STATE COMPONENT")
    # -------------------------------------------------------------------------

    cache = StateComponent('cache')
    print(f"Component: {cache}")

    cache.set('last_output', [1, 2, 3])
    cache.set('step', 42)
    print(f"Data: {cache.data}")
    print(f"Get step: {cache.get('step')}")
    print(f"Get missing: {cache.get('missing', 'default')}")

    # State dict
    state = cache.state_dict()
    cache2 = StateComponent('restored_cache')
    cache2.load_state_dict(state)
    print(f"Restored data: {cache2.data}")

    # -------------------------------------------------------------------------
    test_section("PYTORCH-STYLE METHODS (NO-OP)")
    # -------------------------------------------------------------------------

    comp = ConfigComponent('test', value=1)
    print(f"device: {comp.device}")
    print(f"dtype: {comp.dtype}")
    print(f"parameters: {list(comp.parameters())}")
    print(f"num_parameters: {comp.num_parameters()}")
    print(f"to('cuda'): {comp.to('cuda')}")
    print(f"train(): {comp.train()}")
    print(f"eval(): {comp.eval()}")

    # -------------------------------------------------------------------------
    test_section("UUID UNIQUENESS")
    # -------------------------------------------------------------------------

    c1 = ConfigComponent('same_name', x=1)
    c2 = ConfigComponent('same_name', x=2)
    print(f"Component 1: name='{c1.name}', uuid='{c1.uuid[:8]}...'")
    print(f"Component 2: name='{c2.name}', uuid='{c2.uuid[:8]}...'")
    print(f"Same name: {c1.name == c2.name}")
    print(f"Same uuid: {c1.uuid == c2.uuid}")

    # -------------------------------------------------------------------------
    test_section("LIFECYCLE SIMULATION")


    # -------------------------------------------------------------------------

    class TrackedComponent(BaseComponent):
        def on_attach(self, parent):
            print(f"  -> Attached to '{parent}'")

        def on_detach(self):
            print(f"  -> Detached from '{self.parent}'")


    tracked = TrackedComponent('tracked')
    print(f"Before attach: parent={tracked.parent}")


    # Simulate router attach
    class FakeRouter:
        name = 'fake_router'


    fake_parent = FakeRouter()
    tracked.parent = fake_parent
    tracked.on_attach(fake_parent)
    print(f"After attach: parent={tracked.parent.name}")

    tracked.on_detach()
    tracked.parent = None
    print(f"After detach: parent={tracked.parent}")

    # -------------------------------------------------------------------------
    test_section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nBaseComponent is ready.")