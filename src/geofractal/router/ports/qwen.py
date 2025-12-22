"""
geofractal.router.ports.qwen
============================

Qwen encoder family port.

Supports Qwen2, Qwen2.5, and Instruct variants.
Inherits TorchPort for device/dtype management.

Usage:
    port = QwenPort('qwen', 'Qwen/Qwen2.5-1.5B-Instruct')
    port.load()

    embedding = port('a cat sitting on a mat')
    embeddings = port(['batch', 'of', 'texts'])

    port.to('cpu')  # Move encoder
    port.unload()   # Free VRAM

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Optional, List, Union, Literal

from geofractal.router.ports.torch_port import TorchPort

PoolStrategy = Literal['last', 'first', 'mean', 'max']


class QwenPort(TorchPort):
    """
    Port for Qwen model family.

    Extracts hidden states with configurable pooling and layer selection.

    Args:
        name: Port identifier.
        model_id: HuggingFace model ID or local path.
        pool: Pooling strategy ('last', 'first', 'mean', 'max').
        layer: Which hidden layer (-1 = last, -2 = second to last).
        use_chat_template: Apply chat template for Instruct models.
        system_prompt: Optional system prompt for chat template.
        device: Target device.
        dtype: Target dtype.
    """

    def __init__(
            self,
            name: str,
            model_id: str,
            pool: PoolStrategy = 'last',
            layer: int = -1,
            use_chat_template: bool = True,
            system_prompt: Optional[str] = None,
            device: str = 'cuda',
            dtype: torch.dtype = torch.float16,
    ):
        super().__init__(name, device=device, dtype=dtype)

        self.model_id = model_id
        self.pool = pool
        self.layer = layer
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt

        self._tokenizer = None
        self._hidden_size: Optional[int] = None

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def load(self) -> 'QwenPort':
        """Load model and tokenizer."""
        if self._model is not None:
            return self

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            padding_side='left',
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype,
            device_map=str(self._device),
            trust_remote_code=True,
        )
        self._model.eval()

        self._hidden_size = self._model.config.hidden_size
        self._loaded = True

        return self

    def unload(self) -> 'QwenPort':
        """Release model and tokenizer."""
        super().unload()
        self._tokenizer = None
        return self

    @property
    def hidden_size(self) -> Optional[int]:
        """Model hidden dimension."""
        return self._hidden_size

    # =========================================================================
    # PREPROCESS
    # =========================================================================

    def _apply_chat_template(self, text: str) -> str:
        """Apply chat template for Instruct models."""
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": text})

        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def preprocess(self, raw: Union[str, List[str]]) -> dict:
        """Tokenize input text(s)."""
        # Normalize to list
        if isinstance(raw, str):
            texts = [raw]
        else:
            texts = list(raw)

        # Apply chat template if enabled
        if self.use_chat_template and hasattr(self._tokenizer, 'apply_chat_template'):
            texts = [self._apply_chat_template(t) for t in texts]

        # Tokenize
        tokens = self._tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048,
        )

        return {k: v.to(self.device) for k, v in tokens.items()}

    # =========================================================================
    # ENCODE
    # =========================================================================

    @torch.no_grad()
    def encode(self, prepared: dict) -> Tensor:
        """Run encoder and extract hidden states."""
        outputs = self._model(
            input_ids=prepared['input_ids'],
            attention_mask=prepared['attention_mask'],
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract specified layer
        hidden_states = outputs.hidden_states[self.layer]  # [B, L, D]

        return hidden_states

    # =========================================================================
    # POSTPROCESS
    # =========================================================================

    def postprocess(self, encoded: Tensor) -> Tensor:
        """Pool hidden states to embeddings."""
        # Get attention mask from last preprocess call
        # This is a simplification - in production you'd pass it through

        if self.pool == 'last':
            # Last token (simplified - assumes no padding on right)
            return encoded[:, -1]

        elif self.pool == 'first':
            return encoded[:, 0]

        elif self.pool == 'mean':
            return encoded.mean(dim=1)

        elif self.pool == 'max':
            return encoded.max(dim=1).values

        else:
            raise ValueError(f"Unknown pool strategy: {self.pool}")

    # =========================================================================
    # STATE
    # =========================================================================

    def state_dict(self) -> dict:
        base = super().state_dict()
        base.update({
            'model_id': self.model_id,
            'pool': self.pool,
            'layer': self.layer,
            'use_chat_template': self.use_chat_template,
            'hidden_size': self._hidden_size,
        })
        return base

    def __repr__(self) -> str:
        parts = [f"'{self.name}'"]
        parts.append(f"'{self.model_id}'")
        parts.append("loaded" if self._loaded else "unloaded")
        parts.append(f"device={self._device}")
        if self._hidden_size:
            parts.append(f"dim={self._hidden_size}")
        return f"QwenPort({', '.join(parts)})"


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  QWEN PORT (Structure Only)")
    print("=" * 60)

    port = QwenPort(
        'qwen_test',
        'Qwen/Qwen2.5-1.5B-Instruct',
        pool='last',
        layer=-1,
        device='cuda',
    )

    print(f"\nPort: {port}")
    print(f"is_loaded: {port.is_loaded}")
    print(f"device: {port.device}")
    print(f"state_dict: {port.state_dict()}")

    print("\n" + "=" * 60)
    print("  TO TEST WITH ACTUAL MODEL:")
    print("=" * 60)
    print("""
    port = QwenPort('qwen', 'Qwen/Qwen2.5-1.5B-Instruct')
    port.load()

    # Single
    emb = port('hello world')
    print(emb.shape)  # [1536] or similar

    # Batch
    embs = port(['hello', 'world', 'test'])
    print(embs.shape)  # [3, 1536]

    # Move
    port.cpu()
    port.half()

    port.unload()
    """)