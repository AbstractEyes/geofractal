"""
geofractal.router.config
========================
Configuration classes for collectives and streams.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0

Note: GlobalFractalRouterConfig is defined in core.py
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch

# Import router config from core
from geofractal.router.deprecated.core import GlobalFractalRouterConfig

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GlobalFractalRouterConfig:
    """
    Configuration for GlobalFractalRouter.

    Core Dimensions:
        feature_dim: Internal routing dimension (default: 512)
        fingerprint_dim: Identity/divergence dimension (default: 64)

    Routing:
        num_anchors: Shared behavioral modes (default: 16)
        num_routes: Top-K routes per position (default: 4)
        num_heads: Attention heads (default: 8)
        temperature: Softmax temperature (default: 1.0)

    Coordination:
        use_adjacent_gating: Enable parent→child fingerprint gating
        use_cantor_prior: Enable Cantor diagonal structure
        use_mailbox: Enable inter-router communication
        grid_size: Spatial structure for Cantor pairing (H, W)

    Proven Settings:
        - ImageNet: feature_dim=512, num_anchors=16, num_routes=8
        - FashionMNIST: feature_dim=128, num_anchors=8, num_routes=4
    """

    # Core dimensions
    feature_dim: int = 512
    fingerprint_dim: int = 64

    # Routing
    num_anchors: int = 16
    num_routes: int = 4
    num_heads: int = 8
    head_dim: Optional[int] = None  # Computed if None
    temperature: float = 1.0

    # Sequence structure
    num_slots: int = 16
    grid_size: Tuple[int, int] = (16, 1)

    # Coordination features
    use_adjacent_gating: bool = True
    use_cantor_prior: bool = True
    use_mailbox: bool = True

    # Regularization
    dropout: float = 0.1

    # Anchor contribution weight
    anchor_weight: float = 0.1

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.feature_dim // self.num_heads

        # Ensure grid matches slots
        if self.grid_size[0] * self.grid_size[1] != self.num_slots:
            self.grid_size = (self.num_slots, 1)


@dataclass
class CollectiveConfig:
    """
    Configuration for RouterCollective.

    Architecture:
        feature_dim: Shared internal dimension (default: 512)
        fingerprint_dim: Per-stream identity dimension (default: 64)
        num_classes: Output classes for classification

    Training:
        batch_size: Training batch size (default: 256)
        epochs: Number of epochs (default: 20)
        lr: Learning rate (default: 3e-4)
        warmup_epochs: LR warmup period (default: 2)

    DataLoader (A100 optimized):
        num_workers: Total workers (default: 8)
        pin_memory: Pin to GPU memory (default: True)
        persistent_workers: Keep workers alive (default: True)
        prefetch_factor: Prefetch batches (default: 4)

    AMP:
        use_amp: Mixed precision training (default: True)
    """

    # Architecture
    feature_dim: int = 512
    fingerprint_dim: int = 64
    num_classes: int = 1000

    # Router settings
    num_anchors: int = 16
    num_routes: int = 8
    num_slots: int = 16

    # Training
    batch_size: int = 256
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    grad_clip: float = 1.0

    # DataLoader - A100 optimized
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    # AMP
    use_amp: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_router_config(self) -> GlobalFractalRouterConfig:
        """Convert to router-level config."""
        return GlobalFractalRouterConfig(
            feature_dim=self.feature_dim,
            fingerprint_dim=self.fingerprint_dim,
            num_anchors=self.num_anchors,
            num_routes=self.num_routes,
            num_slots=self.num_slots,
            grid_size=(self.num_slots, 1),
        )


@dataclass
class StreamConfig:
    """
    Configuration for a single stream in a collective.

    Attributes:
        name: Unique identifier for this stream
        input_dim: Dimension of input features
        frozen: Whether the backbone is frozen
        parent_name: Name of parent stream (for hierarchy)
    """

    name: str
    input_dim: int
    frozen: bool = True
    parent_name: Optional[str] = None

    # Optional: pretrained model path
    pretrained: Optional[str] = None


# Preset configurations for common setups

IMAGENET_COLLECTIVE_CONFIG = CollectiveConfig(
    feature_dim=512,
    fingerprint_dim=64,
    num_classes=1000,
    num_anchors=16,
    num_routes=8,
    num_slots=16,
    batch_size=256,
    epochs=20,
)

FASHIONMNIST_COLLECTIVE_CONFIG = CollectiveConfig(
    feature_dim=128,
    fingerprint_dim=64,
    num_classes=10,
    num_anchors=8,
    num_routes=4,
    num_slots=8,
    batch_size=128,
    epochs=30,
)

CIFAR_COLLECTIVE_CONFIG = CollectiveConfig(
    feature_dim=256,
    fingerprint_dim=64,
    num_classes=100,
    num_anchors=12,
    num_routes=6,
    num_slots=12,
    batch_size=128,
    epochs=50,
)