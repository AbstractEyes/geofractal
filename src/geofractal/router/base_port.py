"""
geofractal.router.base_port
===========================

Abstract base for encoder ports.

BasePort is the pure protocol. No torch, no caching, no device management.
Just the interface for: data-in → data-out with lifecycle.

Protocol:
    preprocess(raw) -> prepared      # Transform input for encoder
    encode(prepared) -> encoded      # Core operation (abstract)
    postprocess(encoded) -> output   # Transform encoder output

    load() / unload()                # Lifecycle management

    __call__ chains: preprocess → encode → postprocess

Subclasses:
    TorchPort - adds nn.Module, device tracking, .to()

Composition:
    CachedPort - wraps any port with cache backend
    DatasetPort - batch-oriented, yields from pre-built cache

Directory Structure:
    geofractal/router/
    ├── base_port.py        # This file (protocol)
    └── ports/
        ├── torch_port.py   # TorchPort base
        ├── cached_port.py  # Cache wrapper
        ├── dataset_port.py # Training loader
        ├── qwen.py         # QwenPort(TorchPort)
        ├── dino.py         # DINOPort(TorchPort)
        └── ...

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from geofractal.router.base_component import BaseComponent


class BasePort(BaseComponent, ABC):
    """
    Abstract base for encoder ports.

    Pure protocol. Subclass adds torch, caching, device management.

    Flow:
        raw → preprocess → encode → postprocess → output

    Lifecycle:
        load() - bring resources into memory
        unload() - release resources
        is_loaded - check readiness

    Attributes:
        name: Human-readable identifier (from BaseComponent).
        uuid: Unique identifier (from BaseComponent).
        parent: Reference to parent router (from BaseComponent).
    """

    def __init__(
        self,
        name: str,
        uuid: Optional[str] = None,
    ):
        """
        Initialize BasePort.

        Args:
            name: Human-readable name for this port.
            uuid: Unique identifier. Generated if not provided.
        """
        super().__init__(name, uuid)
        self._loaded = False

    # =========================================================================
    # ABSTRACT: Must implement
    # =========================================================================

    @abstractmethod
    def encode(self, prepared: Any) -> Any:
        """
        Core encoding operation.

        Receives preprocessed input, returns encoded output.
        This is the only method subclasses MUST implement.

        Args:
            prepared: Preprocessed input (from preprocess()).

        Returns:
            Encoded output (passed to postprocess()).
        """
        ...

    # =========================================================================
    # OPTIONAL: Override for custom behavior
    # =========================================================================

    def preprocess(self, raw: Any) -> Any:
        """
        Transform raw input for encoder.

        Override for tokenization, normalization, resize, etc.
        Default: identity (return raw unchanged).

        Args:
            raw: Raw input data.

        Returns:
            Preprocessed data ready for encode().
        """
        return raw

    def postprocess(self, encoded: Any) -> Any:
        """
        Transform encoder output.

        Override for pooling, projection, type conversion, etc.
        Default: identity (return encoded unchanged).

        Args:
            encoded: Output from encode().

        Returns:
            Final output.
        """
        return encoded

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def load(self) -> 'BasePort':
        """
        Load resources into memory.

        Override to load model, tokenizer, processor, etc.

        Returns:
            Self for chaining.
        """
        self._loaded = True
        return self

    def unload(self) -> 'BasePort':
        """
        Release resources from memory.

        Override to free memory.

        Returns:
            Self for chaining.
        """
        self._loaded = False
        return self

    @property
    def is_loaded(self) -> bool:
        """Check if port is ready."""
        return self._loaded

    # =========================================================================
    # CALL
    # =========================================================================

    def __call__(self, raw: Any) -> Any:
        """
        Full pipeline: preprocess → encode → postprocess.

        Args:
            raw: Raw input data.

        Returns:
            Processed output.
        """
        prepared = self.preprocess(raw)
        encoded = self.encode(prepared)
        return self.postprocess(encoded)

    # =========================================================================
    # STATE
    # =========================================================================

    def state_dict(self) -> Dict[str, Any]:
        """Return port state."""
        return {
            'name': self.name,
            'uuid': self.uuid,
            'is_loaded': self._loaded,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load port state."""
        pass  # Subclass handles specifics

    # =========================================================================
    # DUNDER
    # =========================================================================

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"{self.__class__.__name__}('{self.name}', {status})"


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':

    def section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)

    # -------------------------------------------------------------------------
    section("ABSTRACT ENFORCEMENT")
    # -------------------------------------------------------------------------

    try:
        port = BasePort('test')
        print("ERROR: Should have raised TypeError")
    except TypeError as e:
        print(f"✓ Cannot instantiate abstract: {type(e).__name__}")

    # -------------------------------------------------------------------------
    section("MINIMAL IMPLEMENTATION")
    # -------------------------------------------------------------------------

    class EchoPort(BasePort):
        def encode(self, prepared: Any) -> Any:
            return f"encoded:{prepared}"

    port = EchoPort('echo')
    print(f"Port: {port}")
    print(f"Result: {port('hello')}")

    # -------------------------------------------------------------------------
    section("WITH PREPROCESS/POSTPROCESS")
    # -------------------------------------------------------------------------

    class UpperPort(BasePort):
        def preprocess(self, raw: Any) -> Any:
            return raw.strip().lower()

        def encode(self, prepared: Any) -> Any:
            return prepared.upper()

        def postprocess(self, encoded: Any) -> Any:
            return f"[{encoded}]"

    port = UpperPort('upper')
    result = port('  Hello World  ')
    print(f"'  Hello World  ' -> '{result}'")

    # -------------------------------------------------------------------------
    section("LIFECYCLE")
    # -------------------------------------------------------------------------

    class LazyPort(BasePort):
        def encode(self, prepared: Any) -> Any:
            if not self.is_loaded:
                raise RuntimeError("Not loaded")
            return f"encoded:{prepared}"

    port = LazyPort('lazy')
    print(f"Before load: {port}")

    try:
        port('test')
    except RuntimeError as e:
        print(f"✓ Correctly failed: {e}")

    port.load()
    print(f"After load: {port}")
    print(f"Result: {port('test')}")

    port.unload()
    print(f"After unload: {port}")

    # -------------------------------------------------------------------------
    section("STATE DICT")
    # -------------------------------------------------------------------------

    port = EchoPort('state_test')
    port.load()
    print(f"State: {port.state_dict()}")

    # -------------------------------------------------------------------------
    section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nBasePort ready.")
    print("Protocol: preprocess → encode → postprocess")
    print("Lifecycle: load / unload / is_loaded")