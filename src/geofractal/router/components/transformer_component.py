"""
geofractal.router.components.transformer
=========================================

Standard Transformer components with Cantor integration.

Variants:
    - PreNorm (GPT-2 style): LayerNorm before attention/FFN
    - PostNorm (Original): LayerNorm after attention/FFN
    - Parallel (GPT-J/PaLM): Attention and FFN in parallel
    - Sandwich: LN-Attn-LN-FFN-LN

Activations (replaceable):
    - GELU (default, most common)
    - ReLU (classic)
    - SiLU/Swish (LLaMA)
    - GeGLU (gated linear unit with GELU)
    - SwiGLU (gated linear unit with SiLU, LLaMA style)

Cantor Mode:
    When cantor=True:
    - Replaces standard attention with CantorEuclideanAttention
    - Enables hierarchical wormhole routing
    - Each block gets a CantorAddressComponent for fingerprinting
    - Addresses enable cross-block routing based on branch alignment

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import math
from typing import Optional, Dict, Tuple, Literal, Union, Type
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.components.cantor_address_component import (
    CantorAddressComponent,
    CantorAddressConfig,
    create_cantor_address,
)
from geofractal.router.components.cantor_euclidean_attention_component import (
    CantorEuclideanAttention,
    CantorEuclideanConfig,
    create_cantor_euclidean_attention,
)

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

class ActivationType(str, Enum):
    """Supported activation functions."""
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"      # Also known as Swish
    GEGLU = "geglu"    # Gated GELU
    SWIGLU = "swiglu"  # Gated SiLU (LLaMA style)
    TANH = "tanh"
    MISH = "mish"


class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation."""

    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class SwiGLU(nn.Module):
    """Gated Linear Unit with SiLU activation (LLaMA style)."""

    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))"""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(F.softplus(x))


def get_activation(activation: Union[str, ActivationType]) -> nn.Module:
    """Get activation module by name."""
    if isinstance(activation, str):
        activation = ActivationType(activation.lower())

    activations = {
        ActivationType.GELU: nn.GELU(),
        ActivationType.RELU: nn.ReLU(),
        ActivationType.SILU: nn.SiLU(),
        ActivationType.GEGLU: GEGLU(),
        ActivationType.SWIGLU: SwiGLU(),
        ActivationType.TANH: nn.Tanh(),
        ActivationType.MISH: Mish(),
    }

    return activations[activation]


def is_gated_activation(activation: Union[str, ActivationType]) -> bool:
    """Check if activation is gated (requires 2x hidden dim)."""
    if isinstance(activation, str):
        activation = ActivationType(activation.lower())
    return activation in {ActivationType.GEGLU, ActivationType.SWIGLU}


# =============================================================================
# CONFIGURATION
# =============================================================================

class TransformerVariant(str, Enum):
    """Transformer architectural variants."""
    PRENORM = "prenorm"      # GPT-2 style: LN → Attn → + → LN → FFN → +
    POSTNORM = "postnorm"    # Original: Attn → + → LN → FFN → + → LN
    PARALLEL = "parallel"    # GPT-J/PaLM: (Attn + FFN) in parallel
    SANDWICH = "sandwich"    # LN → Attn → LN → FFN → LN


@dataclass
class TransformerConfig:
    """Configuration for transformer components."""

    # Dimensions
    dim: int = 512
    num_heads: int = 8
    ffn_mult: float = 4.0

    # Architecture
    variant: TransformerVariant = TransformerVariant.PRENORM
    activation: ActivationType = ActivationType.GELU

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.0
    ffn_dropout: float = 0.0

    # Options
    bias: bool = True
    qkv_bias: bool = False

    # Cantor integration
    cantor: bool = False
    cantor_levels: int = 5
    cantor_tau: float = 1.0
    cantor_gate_threshold: float = 0.3
    cantor_wormhole_skip: bool = True

    # Depth (for stacked blocks)
    depth: int = 6

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def ffn_dim(self) -> int:
        base = int(self.dim * self.ffn_mult)
        # Gated activations need 2x for the gate
        if is_gated_activation(self.activation):
            return base * 2
        return base

    @property
    def ffn_output_dim(self) -> int:
        """Actual FFN intermediate dim (before gating)."""
        return int(self.dim * self.ffn_mult)


# =============================================================================
# STANDARD ATTENTION
# =============================================================================

class StandardAttention(TorchComponent):
    """
    Standard multi-head self-attention.

    This is the Euclidean baseline - pure Q·K dot product attention.
    """

    def __init__(
        self,
        name: str,
        config: TransformerConfig,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(config.dim, config.dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Args:
            x: (B, S, D) input
            mask: Optional attention mask
            return_info: Whether to return attention weights
        """
        B, S, D = x.shape
        H, head_dim = self.num_heads, self.head_dim

        # Project to Q, K, V
        q = self.q_proj(x).view(B, S, H, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask.bool(), float('-inf'))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)

        if return_info:
            return out, {'attention_weights': attn}
        return out, None


# =============================================================================
# FEED-FORWARD NETWORK
# =============================================================================

class FeedForward(TorchComponent):
    """
    Feed-forward network with configurable activation.

    Supports gated variants (GeGLU, SwiGLU) automatically.
    """

    def __init__(
        self,
        name: str,
        config: TransformerConfig,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.config = config

        # For gated activations, we need different dims
        self.is_gated = is_gated_activation(config.activation)

        if self.is_gated:
            # Gated: project to 2x hidden (for value and gate)
            self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=config.bias)
        else:
            # Standard: project to hidden
            self.up_proj = nn.Linear(config.dim, config.ffn_output_dim, bias=config.bias)

        self.down_proj = nn.Linear(config.ffn_output_dim, config.dim, bias=config.bias)

        self.activation = get_activation(config.activation)
        self.dropout = nn.Dropout(config.ffn_dropout or config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


# =============================================================================
# TRANSFORMER BLOCKS (Variants)
# =============================================================================

class PreNormBlock(TorchComponent):
    """
    Pre-normalization transformer block (GPT-2 style).

    Structure: LN → Attn → + → LN → FFN → +

    Most stable for training deep networks.
    """

    def __init__(
        self,
        name: str,
        config: TransformerConfig,
        block_idx: int = 0,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.config = config
        self.block_idx = block_idx

        # Normalization
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)

        # Attention (Cantor or Standard)
        if config.cantor:
            cantor_config = CantorEuclideanConfig(
                feature_dim=config.dim,
                num_heads=config.num_heads,
                levels=config.cantor_levels,
                branch_tau=config.cantor_tau,
                gate_threshold=config.cantor_gate_threshold,
                use_wormhole_skip=config.cantor_wormhole_skip,
                dropout=config.attention_dropout or config.dropout,
            )
            self.attn = CantorEuclideanAttention(
                name=f'{name}_cantor_attn',
                config=cantor_config,
            )

            # Address for this block
            position = block_idx / max(1, config.depth - 1)
            self.address = create_cantor_address(
                name=f'{name}_address',
                position=position,
                levels=config.cantor_levels,
            )
        else:
            self.attn = StandardAttention(f'{name}_attn', config)
            self.address = None

        # FFN
        self.ffn = FeedForward(f'{name}_ffn', config)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:

        # Attention with residual
        attn_out, attn_info = self.attn(self.norm1(x), mask=mask, return_info=return_info)
        x = x + self.dropout(attn_out)

        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        if return_info:
            info = attn_info or {}
            if self.address is not None:
                info['block_address'] = self.address.branch_path_str()
                info['block_position'] = self.address.position
            return x, info

        return x, None


class PostNormBlock(TorchComponent):
    """
    Post-normalization transformer block (Original Transformer).

    Structure: Attn → + → LN → FFN → + → LN

    Original design, but can be unstable for deep networks.
    """

    def __init__(
        self,
        name: str,
        config: TransformerConfig,
        block_idx: int = 0,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.config = config
        self.block_idx = block_idx

        # Normalization
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)

        # Attention
        if config.cantor:
            cantor_config = CantorEuclideanConfig(
                feature_dim=config.dim,
                num_heads=config.num_heads,
                levels=config.cantor_levels,
                branch_tau=config.cantor_tau,
                gate_threshold=config.cantor_gate_threshold,
                use_wormhole_skip=config.cantor_wormhole_skip,
                dropout=config.attention_dropout or config.dropout,
            )
            self.attn = CantorEuclideanAttention(f'{name}_cantor_attn', cantor_config)

            position = block_idx / max(1, config.depth - 1)
            self.address = create_cantor_address(f'{name}_address', position=position)
        else:
            self.attn = StandardAttention(f'{name}_attn', config)
            self.address = None

        # FFN
        self.ffn = FeedForward(f'{name}_ffn', config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:

        # Attention with residual, then norm
        attn_out, attn_info = self.attn(x, mask=mask, return_info=return_info)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual, then norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        if return_info:
            info = attn_info or {}
            if self.address is not None:
                info['block_address'] = self.address.branch_path_str()
            return x, info

        return x, None


class ParallelBlock(TorchComponent):
    """
    Parallel transformer block (GPT-J/PaLM style).

    Structure: x + Attn(LN(x)) + FFN(LN(x))

    Attention and FFN computed in parallel, more efficient.
    """

    def __init__(
        self,
        name: str,
        config: TransformerConfig,
        block_idx: int = 0,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.config = config
        self.block_idx = block_idx

        # Single normalization for both branches
        self.norm = nn.LayerNorm(config.dim)

        # Attention
        if config.cantor:
            cantor_config = CantorEuclideanConfig(
                feature_dim=config.dim,
                num_heads=config.num_heads,
                levels=config.cantor_levels,
                branch_tau=config.cantor_tau,
                gate_threshold=config.cantor_gate_threshold,
                use_wormhole_skip=config.cantor_wormhole_skip,
                dropout=config.attention_dropout or config.dropout,
            )
            self.attn = CantorEuclideanAttention(f'{name}_cantor_attn', cantor_config)

            position = block_idx / max(1, config.depth - 1)
            self.address = create_cantor_address(f'{name}_address', position=position)
        else:
            self.attn = StandardAttention(f'{name}_attn', config)
            self.address = None

        # FFN
        self.ffn = FeedForward(f'{name}_ffn', config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:

        # Normalize once
        x_norm = self.norm(x)

        # Parallel attention and FFN
        attn_out, attn_info = self.attn(x_norm, mask=mask, return_info=return_info)
        ffn_out = self.ffn(x_norm)

        # Combined residual
        x = x + self.dropout(attn_out) + self.dropout(ffn_out)

        if return_info:
            info = attn_info or {}
            if self.address is not None:
                info['block_address'] = self.address.branch_path_str()
            return x, info

        return x, None


class SandwichBlock(TorchComponent):
    """
    Sandwich transformer block.

    Structure: LN → Attn → LN → FFN → LN

    Extra normalization for stability.
    """

    def __init__(
        self,
        name: str,
        config: TransformerConfig,
        block_idx: int = 0,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.config = config
        self.block_idx = block_idx

        # Three layer norms
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)
        self.norm3 = nn.LayerNorm(config.dim)

        # Attention
        if config.cantor:
            cantor_config = CantorEuclideanConfig(
                feature_dim=config.dim,
                num_heads=config.num_heads,
                levels=config.cantor_levels,
                branch_tau=config.cantor_tau,
                gate_threshold=config.cantor_gate_threshold,
                use_wormhole_skip=config.cantor_wormhole_skip,
                dropout=config.attention_dropout or config.dropout,
            )
            self.attn = CantorEuclideanAttention(f'{name}_cantor_attn', cantor_config)

            position = block_idx / max(1, config.depth - 1)
            self.address = create_cantor_address(f'{name}_address', position=position)
        else:
            self.attn = StandardAttention(f'{name}_attn', config)
            self.address = None

        # FFN
        self.ffn = FeedForward(f'{name}_ffn', config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:

        # Attention
        attn_out, attn_info = self.attn(self.norm1(x), mask=mask, return_info=return_info)
        x = x + self.dropout(attn_out)

        # FFN with extra norm
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)

        if return_info:
            info = attn_info or {}
            if self.address is not None:
                info['block_address'] = self.address.branch_path_str()
            return x, info

        return x, None


# =============================================================================
# BLOCK FACTORY
# =============================================================================

def get_block_class(variant: Union[str, TransformerVariant]) -> Type[TorchComponent]:
    """Get block class for variant."""
    if isinstance(variant, str):
        variant = TransformerVariant(variant.lower())

    blocks = {
        TransformerVariant.PRENORM: PreNormBlock,
        TransformerVariant.POSTNORM: PostNormBlock,
        TransformerVariant.PARALLEL: ParallelBlock,
        TransformerVariant.SANDWICH: SandwichBlock,
    }

    return blocks[variant]


# =============================================================================
# FULL TRANSFORMER
# =============================================================================

class Transformer(TorchComponent):
    """
    Full transformer encoder with configurable architecture.

    Supports:
        - Multiple variants (prenorm, postnorm, parallel, sandwich)
        - Multiple activations (gelu, relu, silu, geglu, swiglu)
        - Cantor mode for hierarchical wormhole attention
        - Per-block addressing for cross-block routing

    When cantor=True:
        - Each block uses CantorEuclideanAttention
        - Each block has a CantorAddressComponent
        - Addresses are spaced across [0, 1] based on depth position
        - Enables hierarchical routing between blocks
    """

    def __init__(
        self,
        name: str,
        config: TransformerConfig,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.config = config
        self.dim = config.dim
        self.depth = config.depth

        # Get block class for variant
        BlockClass = get_block_class(config.variant)

        # Create blocks
        self.blocks = nn.ModuleList([
            BlockClass(
                name=f'{name}_block_{i}',
                config=config,
                block_idx=i,
            )
            for i in range(config.depth)
        ])

        # Final normalization (for prenorm and parallel)
        if config.variant in {TransformerVariant.PRENORM, TransformerVariant.PARALLEL}:
            self.final_norm = nn.LayerNorm(config.dim)
        else:
            self.final_norm = nn.Identity()

        # Collect addresses if Cantor mode
        if config.cantor:
            self._addresses = [block.address for block in self.blocks if hasattr(block, 'address')]
        else:
            self._addresses = []

    @property
    def addresses(self):
        """List of block addresses (Cantor mode only)."""
        return self._addresses

    def get_block_alignments(self) -> Optional[Tensor]:
        """
        Get pairwise hierarchical alignment between all blocks.

        Returns:
            (depth, depth) alignment matrix, or None if not Cantor mode
        """
        if not self._addresses:
            return None

        paths = torch.stack([addr._branch_path for addr in self._addresses])
        staircase = self._addresses[0].staircase

        # Pairwise alignment
        matches = (paths.unsqueeze(1) == paths.unsqueeze(0)).float()
        alignment = (matches * staircase._level_weights).sum(dim=-1)

        return alignment

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Args:
            x: (B, S, D) input embeddings
            mask: Optional attention mask
            return_info: Whether to return per-block info
        """
        all_info = [] if return_info else None

        for block in self.blocks:
            x, info = block(x, mask=mask, return_info=return_info)
            if return_info:
                all_info.append(info)

        x = self.final_norm(x)

        if return_info:
            return x, {
                'block_info': all_info,
                'block_alignments': self.get_block_alignments(),
            }

        return x, None

    def __repr__(self) -> str:
        return (
            f"Transformer("
            f"name='{self.name}', "
            f"dim={self.dim}, "
            f"depth={self.depth}, "
            f"variant={self.config.variant.value}, "
            f"activation={self.config.activation.value}, "
            f"cantor={self.config.cantor}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
            f")"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_transformer(
    name: str,
    dim: int = 512,
    depth: int = 6,
    num_heads: int = 8,
    variant: str = "prenorm",
    activation: str = "gelu",
    cantor: bool = False,
    **kwargs,
) -> Transformer:
    """Factory function for Transformer."""
    config = TransformerConfig(
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        variant=TransformerVariant(variant.lower()),
        activation=ActivationType(activation.lower()),
        cantor=cantor,
        **kwargs,
    )
    return Transformer(name, config)


def create_cantor_transformer(
    name: str,
    dim: int = 512,
    depth: int = 6,
    num_heads: int = 8,
    variant: str = "prenorm",
    activation: str = "gelu",
    cantor_levels: int = 5,
    **kwargs,
) -> Transformer:
    """Factory function for Cantor-enabled Transformer."""
    return create_transformer(
        name=name,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        variant=variant,
        activation=activation,
        cantor=True,
        cantor_levels=cantor_levels,
        **kwargs,
    )


# =============================================================================
# TESTS
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  Transformer Components Test")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    B, S, D = 2, 64, 256
    x = torch.randn(B, S, D, device=device)

    # =========================================================================
    print("\n" + "=" * 70)
    print("  ACTIVATION FUNCTIONS")
    print("=" * 70)

    for act_type in ActivationType:
        act = get_activation(act_type)
        is_gated = is_gated_activation(act_type)

        if is_gated:
            test_x = torch.randn(4, 64)  # Gated needs even dim
        else:
            test_x = torch.randn(4, 32)

        out = act(test_x)
        print(f"  {act_type.value:8s}: gated={is_gated}, {test_x.shape} -> {out.shape}")

    # =========================================================================
    print("\n" + "=" * 70)
    print("  STANDARD TRANSFORMER VARIANTS")
    print("=" * 70)

    for variant in TransformerVariant:
        config = TransformerConfig(
            dim=D,
            num_heads=8,
            depth=4,
            variant=variant,
            activation=ActivationType.GELU,
            cantor=False,
        )

        transformer = create_transformer(
            name=f'test_{variant.value}',
            dim=D,
            depth=4,
            variant=variant.value,
        ).to(device)

        out, _ = transformer(x)
        params = sum(p.numel() for p in transformer.parameters())

        print(f"  {variant.value:10s}: {x.shape} -> {out.shape}, params={params:,}")

    # =========================================================================
    print("\n" + "=" * 70)
    print("  ACTIVATION VARIANTS")
    print("=" * 70)

    for activation in [ActivationType.GELU, ActivationType.SILU, ActivationType.SWIGLU]:
        transformer = create_transformer(
            name=f'test_{activation.value}',
            dim=D,
            depth=4,
            activation=activation.value,
        ).to(device)

        out, _ = transformer(x)
        params = sum(p.numel() for p in transformer.parameters())

        print(f"  {activation.value:8s}: {x.shape} -> {out.shape}, params={params:,}")

    # =========================================================================
    print("\n" + "=" * 70)
    print("  CANTOR TRANSFORMER")
    print("=" * 70)

    try:
        cantor_transformer = create_cantor_transformer(
            name='cantor_test',
            dim=D,
            depth=6,
            num_heads=8,
            variant='prenorm',
            cantor_levels=5,
        ).to(device)

        print(f"Created: {cantor_transformer}")

        # Forward pass
        out, info = cantor_transformer(x, return_info=True)
        print(f"\nForward: {x.shape} -> {out.shape}")

        # Check addresses
        if cantor_transformer.addresses:
            print(f"\nBlock addresses:")
            for i, addr in enumerate(cantor_transformer.addresses):
                print(f"  Block {i}: pos={addr.position:.3f}, path={addr.branch_path_str()}")

            # Alignment matrix
            alignments = cantor_transformer.get_block_alignments()
            if alignments is not None:
                print(f"\nBlock alignment matrix:")
                print(alignments)

        # Check info
        if info and 'block_info' in info:
            print(f"\nBlock info available: {len(info['block_info'])} blocks")
            if info['block_info'][0]:
                print(f"  First block keys: {list(info['block_info'][0].keys())}")

    except Exception as e:
        # print the stack trace
        import traceback
        traceback.print_exc()
        print(f"  ❌ Cantor transformer test failed: {e}")

    # =========================================================================
    print("\n" + "=" * 70)
    print("  GRADIENT CHECK")
    print("=" * 70)

    transformer = create_transformer(
        name='grad_test',
        dim=D,
        depth=4,
    ).to(device)

    x_grad = torch.randn(B, S, D, device=device, requires_grad=True)
    out, _ = transformer(x_grad)
    loss = out.sum()
    loss.backward()

    print(f"  grad norm: {x_grad.grad.norm():.4f}")
    print(f"  grad finite: {torch.isfinite(x_grad.grad).all()}")

    # =========================================================================
    print("\n" + "=" * 70)
    print("  ✓ All tests passed")
    print("=" * 70)