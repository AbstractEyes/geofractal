"""
geofractal.components.utility.chain_component
=============================================

Device-dispatching pipeline for chained operations.

This will be utilized with device distribution and is not currently in use.

ChainComponent chains operations that may run on different devices,
handling tensor movement between operations automatically.

Use Case:
    Expensive operations (Cayley-Menger projections, Fusion) run better
    on specific devices. cuda:1 busy with latent extraction, cuda:3 free.
    Chain dispatches each operation to its optimal device.

    chain = ChainComponent("cayley_pipeline", home_device='cuda:0')
    chain.append(ProjectionOp("proj"), device='cuda:3')
    chain.append(FusionOp("fuse"), device='cuda:3')
    chain.append(AggregationOp("agg"), device='cuda:1')

    # Execution: cuda:0 → cuda:3 → proj → fuse → cuda:1 → agg → cuda:0
    result = chain(input_tensor)

Device Strategy:
    - Each operation can target a specific device
    - Tensor moves to target device before operation
    - Result returns to home_device after chain completes
    - TorchComponents must have parameters on their target device
    - BaseUtilities are stateless, just need tensor on right device

Future: Accelerate integration for distributed training.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, Union, List, Iterator, Dict, Any, Tuple, overload

import torch
import torch.nn as nn

from geofractal.router.base_component import BaseComponent
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.base_utility import BaseUtility


class ChainComponent(TorchComponent):
    """
    Device-dispatching pipeline for chained operations.

    Chains TorchComponents and BaseUtilities with per-operation
    device targeting. Handles tensor movement automatically.

    Attributes:
        _chain_modules: nn.ModuleList of TorchComponent children
        _chain_utilities: BaseUtility children
        _chain_order: List of (kind, index, device) tuples
        home_device: Device for final result (and parameter home)
    """

    def __init__(
        self,
        name: str,
        home_device: Optional[Union[str, torch.device]] = None,
        components: Optional[List[Tuple[BaseComponent, Optional[str]]]] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize ChainComponent.

        Args:
            name: Human-readable name.
            home_device: Device for final result. None = keep on last op's device.
            components: List of (component, device) tuples.
            uuid: Unique identifier.
            **kwargs: Passed to TorchComponent.
        """
        super().__init__(name, uuid, home_device=home_device, **kwargs)

        self._chain_modules = nn.ModuleList()
        self._chain_utilities: List[BaseUtility] = []
        self._chain_order: List[Tuple[str, int, Optional[torch.device]]] = []

        if components:
            for item in components:
                if isinstance(item, tuple):
                    comp, dev = item
                    self.append(comp, device=dev)
                else:
                    self.append(item)

    # =========================================================================
    # MODIFICATION
    # =========================================================================

    def append(
        self,
        component: BaseComponent,
        device: Optional[Union[str, torch.device]] = None,
    ) -> 'ChainComponent':
        """
        Add component to end of chain.

        Args:
            component: TorchComponent or BaseUtility to add.
            device: Target device for this operation. None = use current device.

        Returns:
            Self for chaining.
        """
        target = torch.device(device) if device else None

        if isinstance(component, TorchComponent):
            idx = len(self._chain_modules)
            self._chain_modules.append(component)
            self._chain_order.append(('module', idx, target))

            # Move component to target device if specified
            if target is not None:
                component.to(target)

        elif isinstance(component, BaseUtility):
            idx = len(self._chain_utilities)
            self._chain_utilities.append(component)
            self._chain_order.append(('utility', idx, target))
        else:
            raise TypeError(
                f"Expected TorchComponent or BaseUtility, got {type(component).__name__}"
            )

        component.on_attach(self)
        return self

    def extend(
        self,
        components: List[Union[BaseComponent, Tuple[BaseComponent, str]]],
    ) -> 'ChainComponent':
        """Add multiple components. Each can be component or (component, device) tuple."""
        for item in components:
            if isinstance(item, tuple):
                comp, dev = item
                self.append(comp, device=dev)
            else:
                self.append(item)
        return self

    # =========================================================================
    # ACCESS
    # =========================================================================

    def _get_component(self, kind: str, idx: int) -> BaseComponent:
        """Get component by kind and index."""
        if kind == 'module':
            return self._chain_modules[idx]
        return self._chain_utilities[idx]

    def _get_by_order(self, idx: int) -> BaseComponent:
        """Get component by order index."""
        kind, sub_idx, _ = self._chain_order[idx]
        return self._get_component(kind, sub_idx)

    def get_device(self, idx: int) -> Optional[torch.device]:
        """Get target device for operation at index."""
        _, _, device = self._chain_order[idx]
        return device

    @overload
    def __getitem__(self, idx: int) -> BaseComponent: ...
    @overload
    def __getitem__(self, idx: slice) -> List[BaseComponent]: ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[BaseComponent, List[BaseComponent]]:
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self._get_by_order(i) for i in indices]
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for chain of length {len(self)}")
        return self._get_by_order(idx)

    def __len__(self) -> int:
        return len(self._chain_order)

    def __iter__(self) -> Iterator[BaseComponent]:
        for i in range(len(self)):
            yield self._get_by_order(i)

    def __contains__(self, component: BaseComponent) -> bool:
        if isinstance(component, TorchComponent):
            return component in self._chain_modules
        if isinstance(component, BaseUtility):
            return component in self._chain_utilities
        return False

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Execute chain with device transfers.

        Args:
            x: Input tensor.
            **kwargs: Passed to each operation.

        Returns:
            Result tensor, on home_device if set.
        """
        for kind, idx, target_device in self._chain_order:
            # Move to target device if specified and needed
            if target_device is not None and x.device != target_device:
                x = x.to(target_device)

            # Execute operation
            comp = self._get_component(kind, idx)
            x = comp(x, **kwargs)

        # Return to home device if set
        if self.home_device is not None and x.device != self.home_device:
            x = x.to(self.home_device)

        return x

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def modules_list(self) -> List[TorchComponent]:
        """All TorchComponent children."""
        return list(self._chain_modules)

    @property
    def utilities_list(self) -> List[BaseUtility]:
        """All BaseUtility children."""
        return list(self._chain_utilities)

    @property
    def device_map(self) -> Dict[str, Optional[torch.device]]:
        """Map of component names to target devices."""
        result = {}
        for kind, idx, device in self._chain_order:
            comp = self._get_component(kind, idx)
            result[comp.name] = device
        return result

    # =========================================================================
    # STATE DICT
    # =========================================================================

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Include utility configs and device mapping in state dict."""
        state = super().state_dict(*args, **kwargs)

        utility_state = {}
        for i, util in enumerate(self._chain_utilities):
            utility_state[f"utility_{i}"] = {
                'name': util.name,
                'config': util.state_dict() if hasattr(util, 'state_dict') else {},
            }
        state['_chain_utilities_state'] = utility_state

        # Save order with device info as strings (serializable)
        order_serializable = [
            (kind, idx, str(dev) if dev else None)
            for kind, idx, dev in self._chain_order
        ]
        state['_chain_order'] = order_serializable

        return state

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        """Load utility configs from state dict."""
        utility_state = state.pop('_chain_utilities_state', {})
        order = state.pop('_chain_order', None)

        super().load_state_dict(state, strict=strict)

        for i, util in enumerate(self._chain_utilities):
            key = f"utility_{i}"
            if key in utility_state and hasattr(util, 'load_state_dict'):
                util.load_state_dict(utility_state[key].get('config', {}))

    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}(name='{self.name}', home={self.home_device}, ["]
        for i, (kind, idx, device) in enumerate(self._chain_order):
            comp = self._get_component(kind, idx)
            kind_char = 'T' if kind == 'module' else 'U'
            dev_str = f" @{device}" if device else ""
            lines.append(f"  ({i})[{kind_char}]: {comp.name}{dev_str}")
        lines.append(f"], params={self.num_parameters():,})")
        return '\n'.join(lines)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':
    from torch import Tensor

    # -------------------------------------------------------------------------
    # Test Prefabs
    # -------------------------------------------------------------------------

    class LinearComponent(TorchComponent):
        """Simple linear layer."""
        def __init__(self, name: str, in_dim: int, out_dim: int):
            super().__init__(name)
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

    class MLPComponent(TorchComponent):
        """Two-layer MLP."""
        def __init__(self, name: str, dim: int, hidden_mult: int = 4):
            super().__init__(name)
            self.net = nn.Sequential(
                nn.Linear(dim, dim * hidden_mult),
                nn.GELU(),
                nn.Linear(dim * hidden_mult, dim),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)

    class ScaleUtility(BaseUtility):
        """Stateless scaling."""
        def __init__(self, name: str, scale: float = 1.0):
            super().__init__(name)
            self.scale = scale

        def __call__(self, x: Tensor) -> Tensor:
            return x * self.scale

    # -------------------------------------------------------------------------
    # Tests
    # -------------------------------------------------------------------------

    def section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    section("BASIC CHAIN WITH DEVICE TARGETING")

    chain = ChainComponent("test_chain", home_device=device)
    chain.append(LinearComponent("proj_in", 256, 512), device=device)
    chain.append(ScaleUtility("scale", 0.5))  # No device = use current
    chain.append(MLPComponent("mlp", 512), device=device)
    chain.append(LinearComponent("proj_out", 512, 256), device=device)

    print(chain)
    print(f"\nDevice map: {chain.device_map}")

    section("FORWARD EXECUTION")

    x = torch.randn(4, 256, device=device)
    print(f"Input device: {x.device}")

    y = chain(x)
    print(f"Output device: {y.device}")
    print(f"Output shape: {y.shape}")

    section("GRADIENT FLOW")

    x = torch.randn(4, 256, device=device, requires_grad=True)
    y = chain(x)
    loss = y.sum()
    loss.backward()

    has_grad = all(
        p.grad is not None
        for comp in chain.modules_list
        for p in comp.parameters()
    )
    print(f"All parameters have gradients: {has_grad}")
    print(f"Input has gradient: {x.grad is not None}")

    section("FLUENT CONSTRUCTION")

    fluent_chain = (
        ChainComponent("fluent", home_device=device)
        .append(LinearComponent("l1", 128, 256), device=device)
        .append(ScaleUtility("scale", 2.0))
        .append(LinearComponent("l2", 256, 128), device=device)
    )
    print(fluent_chain)

    section("INIT WITH COMPONENT-DEVICE TUPLES")

    components = [
        (LinearComponent("a", 64, 128), device),
        (ScaleUtility("s", 0.5), None),
        (LinearComponent("b", 128, 64), device),
    ]
    tuple_chain = ChainComponent("from_tuples", home_device=device, components=components)
    print(tuple_chain)

    x = torch.randn(4, 64, device=device)
    y = tuple_chain(x)
    print(f"Forward OK: {y.shape}")

    if torch.cuda.is_available():
        section("CROSS-DEVICE DISPATCH (CPU <-> CUDA)")

        cpu = torch.device('cpu')
        gpu = torch.device('cuda')

        cross_chain = ChainComponent("cross_device", home_device=gpu)
        cross_chain.append(LinearComponent("on_gpu", 256, 256), device=gpu)
        cross_chain.append(LinearComponent("on_cpu", 256, 256), device=cpu)
        cross_chain.append(LinearComponent("back_gpu", 256, 256), device=gpu)

        print(cross_chain)

        x = torch.randn(4, 256, device=gpu)
        y = cross_chain(x)

        print(f"Input device: {x.device}")
        print(f"Output device: {y.device}")
        print(f"Returned to home (cuda): {y.device.type == 'cuda'}")

        # Verify intermediate devices
        print(f"\nComponent devices:")
        for comp in cross_chain.modules_list:
            comp_dev = next(comp.parameters()).device
            print(f"  {comp.name}: {comp_dev}")
    else:
        section("CROSS-DEVICE DISPATCH (skipped - no CUDA)")
        print("Would test dispatching between CPU and CUDA")

    section("INDEXING AND ITERATION")

    print(f"chain[0]: {chain[0].name}")
    print(f"chain[-1]: {chain[-1].name}")
    print(f"chain.get_device(0): {chain.get_device(0)}")

    print("\nIteration:")
    for i, comp in enumerate(chain):
        dev = chain.get_device(i)
        print(f"  {i}: {comp.name} @{dev}")

    section("ALL TESTS PASSED")