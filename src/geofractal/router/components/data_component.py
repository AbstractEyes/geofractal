"""
geofractal.router.components.data_component
===========================================

Data mover for the geofractal system.

DataComponent is a thin wrapper for moving data between devices,
generating synthetic data, and applying non-learned transforms.
It acts as a connection line within the router network.

Design Principles:
    - Mover, not manager
    - Thin wrapper - Accelerate handles heavy lifting
    - Device-to-device transfer
    - Factory-based synthetic data generation
    - Non-learned transforms only

Responsibilities:
    - Move tensors between devices
    - Generate synthetic/dummy data via factory
    - Non-learned transforms (reshape, cast, normalize)
    - Receive from external systems
    - Feed to registered consumers

Usage:
    # Basic mover
    mover = DataComponent('gpu_mover', target_device='cuda:0')
    batch = mover.move(cpu_batch)

    # Noise generator
    noise = DataComponent('noise_gen')
    noise.bind_factory(lambda: torch.randn(32, 512))
    z = noise.generate()

    # With transforms
    mover = DataComponent('preprocessor', target_device='cuda:0')
    mover.move_and_cast(batch, torch.float16)

    # Passthrough iteration (Accelerate handles DataLoader)
    data = DataComponent('train')
    data.bind(accelerator.prepare(dataloader))
    for batch in data:
        ...

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, Any, Dict, Callable, Union, Tuple, List

import torch
from torch import Tensor

from geofractal.router.base_component import BaseComponent
from geofractal.router.components.torch_component import TorchComponent


class DataComponent(TorchComponent):
    """
    Data mover.

    Connects, transforms, transfers. Thin wrapper that lets
    Accelerate and DataLoader do the heavy lifting.

    Attributes:
        name: Human-readable identifier.
        uuid: Unique identifier.
        source: Bound data source.
        factory: Callable for synthetic data generation.
        target_device: Default device to move data to.
        target_dtype: Default dtype to cast data to.
    """

    def __init__(
        self,
        name: str,
        uuid: Optional[str] = None,
        target_device: Optional[Union[str, torch.device]] = None,
        target_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize DataComponent.

        Args:
            name: Human-readable name.
            uuid: Unique identifier. Generated if not provided.
            target_device: Default device for move operations.
            target_dtype: Default dtype for cast operations.
        """
        super().__init__(name, uuid)

        self.source: Optional[Any] = None
        self.factory: Optional[Callable] = None
        self.target_device = torch.device(target_device) if target_device else None
        self.target_dtype = target_dtype

    # =========================================================================
    # BINDING
    # =========================================================================

    def bind(self, source: Any) -> 'DataComponent':
        """
        Bind a data source.

        Args:
            source: Any iterable (DataLoader, list, generator, etc.)

        Returns:
            Self for chaining.
        """
        self.source = source
        return self

    def bind_factory(self, fn: Callable) -> 'DataComponent':
        """
        Bind a factory function for synthetic data.

        Args:
            fn: Callable that returns data when called.

        Returns:
            Self for chaining.
        """
        self.factory = fn
        return self

    def unbind(self) -> 'DataComponent':
        """
        Clear source and factory.

        Returns:
            Self for chaining.
        """
        self.source = None
        self.factory = None
        return self

    @property
    def is_bound(self) -> bool:
        """Check if source is bound."""
        return self.source is not None

    @property
    def has_factory(self) -> bool:
        """Check if factory is bound."""
        return self.factory is not None

    # =========================================================================
    # MOVING
    # =========================================================================

    def move(
        self,
        data: Any,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = True,
    ) -> Any:
        """
        Move data to device.

        Recursively moves all tensors. Uses target_device if
        device not specified.

        Args:
            data: Data to move (tensor, dict, list, tuple).
            device: Target device. Uses target_device if None.
            non_blocking: Use non-blocking transfer.

        Returns:
            Data with tensors on target device.
        """
        device = device or self.target_device
        if device is None:
            return data

        device = torch.device(device) if isinstance(device, str) else device
        return self._recursive_apply(
            data,
            lambda t: t.to(device, non_blocking=non_blocking)
        )

    def move_to_target(self, data: Any, non_blocking: bool = True) -> Any:
        """
        Move data to target_device.

        Args:
            data: Data to move.
            non_blocking: Use non-blocking transfer.

        Returns:
            Data on target device.

        Raises:
            ValueError: If target_device not set.
        """
        if self.target_device is None:
            raise ValueError(f"DataComponent '{self.name}' has no target_device set.")
        return self.move(data, self.target_device, non_blocking)

    # =========================================================================
    # GENERATING
    # =========================================================================

    def generate(self, *args, **kwargs) -> Any:
        """
        Generate data using factory.

        Args:
            *args: Passed to factory.
            **kwargs: Passed to factory.

        Returns:
            Generated data.

        Raises:
            RuntimeError: If no factory bound.
        """
        if self.factory is None:
            raise RuntimeError(f"DataComponent '{self.name}' has no factory bound.")
        return self.factory(*args, **kwargs)

    def noise(
        self,
        shape: Tuple[int, ...],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Generate random noise.

        Args:
            shape: Tensor shape.
            device: Target device. Uses target_device if None.
            dtype: Data type. Uses target_dtype if None.

        Returns:
            Random tensor from standard normal.
        """
        device = device or self.target_device
        dtype = dtype or self.target_dtype
        return torch.randn(shape, device=device, dtype=dtype)

    def zeros(
        self,
        shape: Tuple[int, ...],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Generate zeros tensor.

        Args:
            shape: Tensor shape.
            device: Target device. Uses target_device if None.
            dtype: Data type. Uses target_dtype if None.

        Returns:
            Zeros tensor.
        """
        device = device or self.target_device
        dtype = dtype or self.target_dtype
        return torch.zeros(shape, device=device, dtype=dtype)

    def ones(
        self,
        shape: Tuple[int, ...],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Generate ones tensor.

        Args:
            shape: Tensor shape.
            device: Target device. Uses target_device if None.
            dtype: Data type. Uses target_dtype if None.

        Returns:
            Ones tensor.
        """
        device = device or self.target_device
        dtype = dtype or self.target_dtype
        return torch.ones(shape, device=device, dtype=dtype)

    def randn_like(
        self,
        tensor: Tensor,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Generate noise with same shape as input.

        Args:
            tensor: Reference tensor for shape.
            device: Target device. Uses tensor's device if None.
            dtype: Data type. Uses tensor's dtype if None.

        Returns:
            Random tensor.
        """
        device = device or tensor.device
        dtype = dtype or tensor.dtype
        return torch.randn_like(tensor, device=device, dtype=dtype)

    def zeros_like(self, tensor: Tensor) -> Tensor:
        """Generate zeros with same shape/device/dtype as input."""
        return torch.zeros_like(tensor)

    def ones_like(self, tensor: Tensor) -> Tensor:
        """Generate ones with same shape/device/dtype as input."""
        return torch.ones_like(tensor)

    # =========================================================================
    # TRANSFORMS (non-learned)
    # =========================================================================

    def cast(
        self,
        data: Any,
        dtype: Optional[torch.dtype] = None,
    ) -> Any:
        """
        Cast data to dtype.

        Args:
            data: Data to cast.
            dtype: Target dtype. Uses target_dtype if None.

        Returns:
            Data with tensors cast to dtype.
        """
        dtype = dtype or self.target_dtype
        if dtype is None:
            return data
        return self._recursive_apply(data, lambda t: t.to(dtype))

    def reshape(self, data: Tensor, shape: Tuple[int, ...]) -> Tensor:
        """
        Reshape tensor.

        Args:
            data: Tensor to reshape.
            shape: New shape.

        Returns:
            Reshaped tensor.
        """
        return data.reshape(shape)

    def flatten(self, data: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        """
        Flatten tensor dimensions.

        Args:
            data: Tensor to flatten.
            start_dim: First dim to flatten.
            end_dim: Last dim to flatten.

        Returns:
            Flattened tensor.
        """
        return data.flatten(start_dim, end_dim)

    def permute(self, data: Tensor, dims: Tuple[int, ...]) -> Tensor:
        """
        Permute tensor dimensions.

        Args:
            data: Tensor to permute.
            dims: New dimension order.

        Returns:
            Permuted tensor.
        """
        return data.permute(dims)

    def squeeze(self, data: Tensor, dim: Optional[int] = None) -> Tensor:
        """Remove size-1 dimensions."""
        return data.squeeze(dim) if dim is not None else data.squeeze()

    def unsqueeze(self, data: Tensor, dim: int) -> Tensor:
        """Add size-1 dimension."""
        return data.unsqueeze(dim)

    def normalize(
        self,
        data: Tensor,
        dim: int = -1,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        L2 normalize along dimension.

        Args:
            data: Tensor to normalize.
            dim: Dimension to normalize along.
            eps: Epsilon for numerical stability.

        Returns:
            Normalized tensor.
        """
        return data / (data.norm(dim=dim, keepdim=True) + eps)

    def clamp(
        self,
        data: Tensor,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Tensor:
        """
        Clamp values to range.

        Args:
            data: Tensor to clamp.
            min_val: Minimum value.
            max_val: Maximum value.

        Returns:
            Clamped tensor.
        """
        return data.clamp(min=min_val, max=max_val)

    def detach(self, data: Any) -> Any:
        """
        Detach tensors from computation graph.

        Args:
            data: Data to detach.

        Returns:
            Detached data.
        """
        return self._recursive_apply(data, lambda t: t.detach())

    def clone(self, data: Any) -> Any:
        """
        Clone tensors.

        Args:
            data: Data to clone.

        Returns:
            Cloned data.
        """
        return self._recursive_apply(data, lambda t: t.clone())

    def contiguous(self, data: Any) -> Any:
        """
        Make tensors contiguous.

        Args:
            data: Data to make contiguous.

        Returns:
            Contiguous data.
        """
        return self._recursive_apply(data, lambda t: t.contiguous())

    # =========================================================================
    # COMPOUND OPERATIONS
    # =========================================================================

    def move_and_cast(
        self,
        data: Any,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = True,
    ) -> Any:
        """
        Move and cast in one operation.

        Args:
            data: Data to process.
            dtype: Target dtype.
            device: Target device.
            non_blocking: Use non-blocking transfer.

        Returns:
            Moved and cast data.
        """
        device = device or self.target_device
        dtype = dtype or self.target_dtype

        def transform(t: Tensor) -> Tensor:
            if device is not None and dtype is not None:
                return t.to(device=device, dtype=dtype, non_blocking=non_blocking)
            elif device is not None:
                return t.to(device=device, non_blocking=non_blocking)
            elif dtype is not None:
                return t.to(dtype=dtype)
            return t

        return self._recursive_apply(data, transform)

    # =========================================================================
    # ITERATION (passthrough)
    # =========================================================================

    def __iter__(self):
        """Iterate over source."""
        if self.source is None:
            raise RuntimeError(f"DataComponent '{self.name}' has no source bound.")
        return iter(self.source)

    def __len__(self) -> int:
        """Length of source."""
        if self.source is None:
            return 0
        return len(self.source)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _recursive_apply(
        self,
        data: Any,
        fn: Callable[[Tensor], Tensor],
    ) -> Any:
        """
        Recursively apply function to all tensors.

        Args:
            data: Data structure.
            fn: Function to apply to tensors.

        Returns:
            Transformed data.
        """
        if isinstance(data, Tensor):
            return fn(data)
        elif isinstance(data, dict):
            return {k: self._recursive_apply(v, fn) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            result = [self._recursive_apply(v, fn) for v in data]
            return type(data)(result)
        else:
            return data

    # =========================================================================
    # STATE
    # =========================================================================

    def state_dict(self) -> Dict[str, Any]:
        """Return component state."""
        return {
            'target_device': str(self.target_device) if self.target_device else None,
            'target_dtype': str(self.target_dtype) if self.target_dtype else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load component state."""
        if state.get('target_device'):
            self.target_device = torch.device(state['target_device'])
        if state.get('target_dtype'):
            # Convert string back to dtype
            dtype_str = state['target_dtype']
            self.target_dtype = getattr(torch, dtype_str.replace('torch.', ''))

    # =========================================================================
    # DUNDER
    # =========================================================================

    def __repr__(self) -> str:
        parts = [f"name='{self.name}'"]

        if self.source is not None:
            parts.append(f"source={type(self.source).__name__}")

        if self.factory is not None:
            parts.append("factory=bound")

        if self.target_device:
            parts.append(f"device={self.target_device}")

        if self.target_dtype:
            parts.append(f"dtype={self.target_dtype}")

        if self.parent:
            parts.append(f"parent='{self.parent.name}'")

        return f"{self.__class__.__name__}({', '.join(parts)})"


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == '__main__':
    from torch.utils.data import DataLoader, TensorDataset

    def test_section(title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    has_cuda = torch.cuda.is_available()

    # -------------------------------------------------------------------------
    test_section("BASIC MOVER")
    # -------------------------------------------------------------------------

    mover = DataComponent('mover', target_device=device)
    print(f"Component: {mover}")

    # Move tensor
    x = torch.randn(4, 64)
    y = mover.move(x)
    print(f"Moved: {x.device} -> {y.device}")

    # Move dict
    batch = {'x': torch.randn(4, 64), 'y': torch.tensor([0, 1, 2, 3])}
    moved = mover.move(batch)
    print(f"Dict moved: x={moved['x'].device}, y={moved['y'].device}")

    # -------------------------------------------------------------------------
    test_section("FACTORY / GENERATION")
    # -------------------------------------------------------------------------

    gen = DataComponent('noise_gen', target_device=device, target_dtype=torch.float32)
    gen.bind_factory(lambda shape: torch.randn(shape))
    print(f"Component: {gen}")

    # Generate via factory
    z = gen.generate((4, 128))
    print(f"Factory output: {z.shape}, {z.device}")

    # Built-in generators
    noise = gen.noise((4, 64))
    print(f"Noise: {noise.shape}, {noise.device}, {noise.dtype}")

    zeros = gen.zeros((2, 32))
    print(f"Zeros: {zeros.shape}, {zeros.device}")

    ones = gen.ones((2, 32))
    print(f"Ones: {ones.shape}, {ones.device}")

    # -------------------------------------------------------------------------
    test_section("TRANSFORMS")
    # -------------------------------------------------------------------------

    transformer = DataComponent('transformer')

    x = torch.randn(4, 8, 16)

    # Reshape
    y = transformer.reshape(x, (4, 128))
    print(f"Reshape: {x.shape} -> {y.shape}")

    # Flatten
    y = transformer.flatten(x, 1)
    print(f"Flatten: {x.shape} -> {y.shape}")

    # Permute
    y = transformer.permute(x, (0, 2, 1))
    print(f"Permute: {x.shape} -> {y.shape}")

    # Normalize
    y = transformer.normalize(x)
    print(f"Normalize: norm={y.norm(dim=-1)[0, 0].item():.4f}")

    # Cast
    if has_cuda:
        caster = DataComponent('caster', target_dtype=torch.float16)
        y = caster.cast(x)
        print(f"Cast: {x.dtype} -> {y.dtype}")

    # -------------------------------------------------------------------------
    test_section("COMPOUND OPERATIONS")
    # -------------------------------------------------------------------------

    if has_cuda:
        proc = DataComponent('proc', target_device='cuda', target_dtype=torch.float16)

        x = torch.randn(4, 64)
        y = proc.move_and_cast(x)
        print(f"Move+Cast: {x.device}/{x.dtype} -> {y.device}/{y.dtype}")
    else:
        print("Skipping (no CUDA)")

    # -------------------------------------------------------------------------
    test_section("PASSTHROUGH ITERATION")
    # -------------------------------------------------------------------------

    dataset = TensorDataset(torch.randn(32, 64), torch.randint(0, 10, (32,)))
    loader = DataLoader(dataset, batch_size=8)

    data = DataComponent('train', target_device=device)
    data.bind(loader)
    print(f"Component: {data}")
    print(f"Length: {len(data)}")

    batch_count = 0
    for x, y in data:
        x = data.move(x)
        batch_count += 1
    print(f"Batches: {batch_count}")

    # -------------------------------------------------------------------------
    test_section("CLONE / DETACH / CONTIGUOUS")
    # -------------------------------------------------------------------------

    util = DataComponent('util')

    x = torch.randn(4, 64, requires_grad=True)

    cloned = util.clone(x)
    print(f"Clone: same data={x.data_ptr() == cloned.data_ptr()}")

    detached = util.detach(x)
    print(f"Detach: requires_grad={detached.requires_grad}")

    # Non-contiguous -> contiguous
    x = torch.randn(4, 8, 16).permute(2, 0, 1)
    print(f"Before contiguous: {x.is_contiguous()}")
    y = util.contiguous(x)
    print(f"After contiguous: {y.is_contiguous()}")

    # -------------------------------------------------------------------------
    test_section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nDataComponent is ready.")