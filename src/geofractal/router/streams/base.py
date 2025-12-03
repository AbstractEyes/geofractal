"""
geofractal.router.streams.base
==============================
Abstract base class for all stream types.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any

from geofractal.router.config import GlobalFractalRouterConfig, CollectiveConfig
from geofractal.router.registry import RouterMailbox


class BaseStream(nn.Module, ABC):
    """
    Abstract base class for router streams.

    A stream is an "expert" in the collective. It:
    1. Takes input (image, features, etc.)
    2. Produces features in a common space
    3. Routes those features via its router
    4. Coordinates with other streams via mailbox

    Subclasses must implement:
    - encode(x) -> features before routing
    - input_dim property

    The base class handles:
    - Translation to common feature space
    - Router with unique fingerprint
    - Slot embeddings
    - Forward pass coordination
    """

    def __init__(
            self,
            config: CollectiveConfig,
            name: str,
            input_dim: int,
            parent_id: Optional[str] = None,
            cooperation_group: str = "default_collective",
    ):
        super().__init__()

        self.config = config
        self.name = name
        self._input_dim = input_dim
        self.parent_id = parent_id
        self.cooperation_group = cooperation_group

        # Translation head: input_dim -> feature_dim * num_slots
        self.translation = nn.Sequential(
            nn.Linear(input_dim, config.feature_dim * 2),
            nn.LayerNorm(config.feature_dim * 2),
            nn.GELU(),
            nn.Dropout(config.to_router_config().dropout),
            nn.Linear(config.feature_dim * 2, config.feature_dim * config.num_slots),
        )

        # Learnable slot embeddings (unique per stream)
        self.slot_embed = nn.Parameter(
            torch.randn(1, config.num_slots, config.feature_dim) * 0.02
        )

        # Router with unique fingerprint
        # Import here to avoid circular dependency
        from geofractal.router.core import GlobalFractalRouter

        router_config = config.to_router_config()
        self.router = GlobalFractalRouter(
            config=router_config,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
            name=name,
        )

    @property
    def input_dim(self) -> int:
        """Dimension of features before translation."""
        return self._input_dim

    @property
    def fingerprint(self) -> torch.Tensor:
        """This stream's unique fingerprint."""
        return self.router.fingerprint

    @property
    def module_id(self) -> str:
        """This stream's unique ID in the registry."""
        return self.router.module_id

    @abstractmethod
    def encode(self, x: Any) -> torch.Tensor:
        """
        Encode input to feature vector.

        Args:
            x: Input (image tensor, feature dict, etc.)

        Returns:
            features: [B, input_dim] encoded features
        """
        pass

    def forward(
            self,
            x: Any,
            mailbox: RouterMailbox,
            target_fingerprint: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Full forward pass: encode -> translate -> route.

        Args:
            x: Input to encode
            mailbox: Shared mailbox for coordination
            target_fingerprint: Next stream's fingerprint (for gating)

        Returns:
            routed: [B, num_slots, feature_dim] routed features
            info: Dict with routing metrics
        """
        # Encode input
        features = self.encode(x)  # [B, input_dim]
        B = features.shape[0]

        # Translate to routing space
        translated = self.translation(features)  # [B, feature_dim * num_slots]
        slots = translated.view(B, self.config.num_slots, self.config.feature_dim)

        # Add slot embeddings
        slots = slots + self.slot_embed

        # Route
        routes, weights, routed = self.router(
            slots,
            mailbox=mailbox,
            target_fingerprint=target_fingerprint,
            skip_first=False,
        )

        # Compute metrics
        info = {
            'route_entropy': -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item(),
            'stream_name': self.name,
        }

        return routed, info

    def pool(self, routed: torch.Tensor) -> torch.Tensor:
        """Pool routed features across slots."""
        return routed.mean(dim=1)  # [B, feature_dim]