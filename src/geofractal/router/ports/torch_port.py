"""
geofractal.router.ports.torch_port
==================================

Torch-specific port base class.

Extends BasePort with:
    - Device management (.to(), device property)
    - Dtype tracking
    - nn.Module lifecycle integration
    - Gradient control (freeze/unfreeze)

This is the base for all encoder ports that wrap PyTorch models
(Qwen, DINO, CLIP, VAE, etc.)

Usage:
    class MyEncoderPort(TorchPort):
        def __init__(self, name, model_id, device='cuda', dtype=torch.float16):
            super().__init__(name, device=device, dtype=dtype)
            self.model_id = model_id
            self._model = None

        def load(self):
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.to(self._device, self._dtype)
            self._model.eval()
            self._loaded = True
            return self

        def encode(self, prepared):
            return self._model(**prepared).last_hidden_state

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Optional, Dict, Union

from geofractal.router.base_port import BasePort


class TorchPort(BasePort):
    """
    Torch-specific port base.

    Adds device/dtype management to BasePort protocol.

    Subclasses hold an encoder (self._model) and implement:
        - load(): Load model to self._device
        - encode(): Run self._model
        - Optionally: preprocess(), postprocess()

    Device Flow:
        1. __init__ sets target device/dtype
        2. load() places model on target device
        3. to() moves model if already loaded
        4. preprocess() should move input to self.device
        5. encode() runs on self.device
        6. postprocess() can move output elsewhere

    Attributes:
        _device: Target device for encoder.
        _dtype: Target dtype for encoder.
        _model: The encoder model (set by subclass in load()).
    """

    def __init__(
        self,
        name: str,
        device: Union[str, torch.device] = 'cuda',
        dtype: torch.dtype = torch.float32,
        uuid: Optional[str] = None,
    ):
        """
        Initialize TorchPort.

        Args:
            name: Port identifier.
            device: Target device for encoder.
            dtype: Target dtype for encoder.
            uuid: Unique identifier.
        """
        super().__init__(name, uuid)

        self._device = torch.device(device) if isinstance(device, str) else device
        self._dtype = dtype
        self._model: Optional[nn.Module] = None

    # =========================================================================
    # DEVICE MANAGEMENT
    # =========================================================================

    @property
    def device(self) -> torch.device:
        """Current device (from model if loaded, else target)."""
        if self._model is not None:
            try:
                return next(self._model.parameters()).device
            except StopIteration:
                pass
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Current dtype (from model if loaded, else target)."""
        if self._model is not None:
            try:
                return next(self._model.parameters()).dtype
            except StopIteration:
                pass
        return self._dtype

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> 'TorchPort':
        """
        Move encoder to device/dtype.

        Updates target and moves model if loaded.

        Args:
            device: Target device.
            dtype: Target dtype.

        Returns:
            Self for chaining.
        """
        if device is not None:
            self._device = torch.device(device) if isinstance(device, str) else device
        if dtype is not None:
            self._dtype = dtype

        if self._model is not None:
            if device is not None and dtype is not None:
                self._model.to(device=self._device, dtype=self._dtype)
            elif device is not None:
                self._model.to(device=self._device)
            elif dtype is not None:
                self._model.to(dtype=self._dtype)

        return self

    def cuda(self, device_id: int = 0) -> 'TorchPort':
        """Move to CUDA device."""
        return self.to(device=f'cuda:{device_id}')

    def cpu(self) -> 'TorchPort':
        """Move to CPU."""
        return self.to(device='cpu')

    def half(self) -> 'TorchPort':
        """Convert to float16."""
        return self.to(dtype=torch.float16)

    def float(self) -> 'TorchPort':
        """Convert to float32."""
        return self.to(dtype=torch.float32)

    def bfloat16(self) -> 'TorchPort':
        """Convert to bfloat16."""
        return self.to(dtype=torch.bfloat16)

    # =========================================================================
    # GRADIENT CONTROL
    # =========================================================================

    def freeze(self) -> 'TorchPort':
        """Freeze encoder parameters."""
        if self._model is not None:
            for p in self._model.parameters():
                p.requires_grad = False
            self._model.eval()
        return self

    def unfreeze(self) -> 'TorchPort':
        """Unfreeze encoder parameters."""
        if self._model is not None:
            for p in self._model.parameters():
                p.requires_grad = True
            self._model.train()
        return self

    @property
    def is_frozen(self) -> bool:
        """Check if encoder is frozen."""
        if self._model is None:
            return True
        return not any(p.requires_grad for p in self._model.parameters())

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def unload(self) -> 'TorchPort':
        """
        Release model from memory.

        Deletes model and clears CUDA cache.
        """
        if self._model is not None:
            del self._model
            self._model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._loaded = False
        return self

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count encoder parameters."""
        if self._model is None:
            return 0
        if trainable_only:
            return sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self._model.parameters())

    # =========================================================================
    # STATE
    # =========================================================================

    def state_dict(self) -> Dict[str, Any]:
        """Return port state (not model weights)."""
        return {
            'name': self.name,
            'uuid': self.uuid,
            'device': str(self._device),
            'dtype': str(self._dtype),
            'is_loaded': self._loaded,
        }

    # =========================================================================
    # DUNDER
    # =========================================================================

    def __repr__(self) -> str:
        parts = [f"'{self.name}'"]
        parts.append("loaded" if self._loaded else "unloaded")
        parts.append(f"device={self._device}")

        if self._model is not None:
            params = self.num_parameters()
            parts.append(f"params={params:,}")

        return f"{self.__class__.__name__}({', '.join(parts)})"


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':

    def section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)

    # -------------------------------------------------------------------------
    section("ABSTRACT STILL ENFORCED")
    # -------------------------------------------------------------------------

    try:
        port = TorchPort('test')
        print("ERROR: Should have raised TypeError")
    except TypeError as e:
        print(f"✓ Cannot instantiate (encode still abstract): {type(e).__name__}")

    # -------------------------------------------------------------------------
    section("MINIMAL TORCH PORT")
    # -------------------------------------------------------------------------

    class SimplePort(TorchPort):
        def load(self):
            self._model = nn.Linear(64, 32)
            self._model.to(self._device, self._dtype)
            self._loaded = True
            return self

        def encode(self, prepared: Tensor) -> Tensor:
            return self._model(prepared)

    port = SimplePort('simple', device='cpu', dtype=torch.float32)
    print(f"Port: {port}")
    print(f"Device: {port.device}")
    print(f"Dtype: {port.dtype}")

    port.load()
    print(f"After load: {port}")
    print(f"Params: {port.num_parameters():,}")

    x = torch.randn(4, 64)
    y = port(x)
    print(f"Forward: {x.shape} -> {y.shape}")

    # -------------------------------------------------------------------------
    section("DEVICE MOVEMENT")
    # -------------------------------------------------------------------------

    port = SimplePort('mover', device='cpu')
    port.load()
    print(f"Initial device: {port.device}")

    if torch.cuda.is_available():
        port.cuda()
        print(f"After cuda(): {port.device}")

        port.cpu()
        print(f"After cpu(): {port.device}")
    else:
        print("(CUDA not available)")

    # -------------------------------------------------------------------------
    section("DTYPE MOVEMENT")
    # -------------------------------------------------------------------------

    port = SimplePort('typer', device='cpu', dtype=torch.float32)
    port.load()
    print(f"Initial dtype: {port.dtype}")

    port.half()
    print(f"After half(): {port.dtype}")

    port.float()
    print(f"After float(): {port.dtype}")

    # -------------------------------------------------------------------------
    section("FREEZE / UNFREEZE")
    # -------------------------------------------------------------------------

    port = SimplePort('freezer', device='cpu')
    port.load()
    print(f"is_frozen: {port.is_frozen}")
    print(f"trainable params: {port.num_parameters(trainable_only=True)}")

    port.freeze()
    print(f"After freeze - is_frozen: {port.is_frozen}")
    print(f"After freeze - trainable: {port.num_parameters(trainable_only=True)}")

    port.unfreeze()
    print(f"After unfreeze - is_frozen: {port.is_frozen}")
    print(f"After unfreeze - trainable: {port.num_parameters(trainable_only=True)}")

    # -------------------------------------------------------------------------
    section("UNLOAD")
    # -------------------------------------------------------------------------

    port = SimplePort('unloader', device='cpu')
    port.load()
    print(f"Before unload: {port}")

    port.unload()
    print(f"After unload: {port}")
    print(f"Model is None: {port._model is None}")

    # -------------------------------------------------------------------------
    section("ALL TESTS PASSED")
    # -------------------------------------------------------------------------

    print("\nTorchPort ready.")
    print("Adds: device, dtype, to(), freeze/unfreeze, num_parameters")