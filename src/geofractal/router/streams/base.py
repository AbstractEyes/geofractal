# streams/base.py
"""Abstract base class - decoupled from core.py."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any

from geofractal.router.config import CollectiveConfig
from geofractal.router.head import build_standard_head, HeadConfig
from geofractal.router.head.builder import ComposedHead


class BaseStream(nn.Module, ABC):
    """
    Abstract base for router streams.
    
    Decoupled from core.py - uses head/builder.py.
    Fingerprint lives in the head, not the stream.
    """

    def __init__(
        self,
        config: CollectiveConfig,
        name: str,
        head: Optional[ComposedHead] = None,
        head_config: Optional[HeadConfig] = None,
    ):
        super().__init__()
        
        self.config = config
        self.name = name
        
        # Head - injected or built from config
        if head is not None:
            self.head = head
        else:
            _head_config = head_config or HeadConfig(
                feature_dim=config.feature_dim,
                fingerprint_dim=config.fingerprint_dim,
                num_anchors=config.num_anchors,
                num_routes=config.num_routes,
            )
            self.head = build_standard_head(_head_config)
    
    @property
    def fingerprint(self) -> torch.Tensor:
        """Access head's fingerprint."""
        return self.head.fingerprint
    
    @property
    @abstractmethod
    def input_shape(self) -> str:
        """Return 'vector', 'sequence', or 'image'."""
        pass
    
    @abstractmethod
    def encode(self, x: Any) -> torch.Tensor:
        """Encode input to features."""
        pass
    
    @abstractmethod
    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        """Convert to [B, S, D] for head."""
        pass
    
    def forward(
        self,
        x: Any,
        target_fingerprint: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """encode → prepare → head"""
        
        features = self.encode(x)
        prepared = self.prepare_for_head(features)
        
        if return_info:
            output, info = self.head(prepared, target_fingerprint, return_info=True)
            info['stream_name'] = self.name
            info['input_shape'] = self.input_shape
            return output, info
        else:
            output = self.head(prepared, target_fingerprint, return_info=False)
            return output, None
    
    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool [B, S, D] → [B, D]."""
        if x.dim() == 3:
            return x.mean(dim=1)
        return x