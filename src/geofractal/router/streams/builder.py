# streams/builder.py
"""Stream factory."""

from typing import Optional
import torch.nn as nn

from geofractal.router.config import CollectiveConfig
from geofractal.router.head import HeadConfig, build_standard_head, build_lightweight_head
from geofractal.router.head.builder import ComposedHead, HeadBuilder
from geofractal.router.streams.vector import (
    VectorStream, FeatureVectorStream, TrainableVectorStream
)
from geofractal.router.streams.sequence import (
    SequenceStream, TransformerSequenceStream, ConvSequenceStream
)


class StreamBuilder:
    """Factory for streams with consistent config."""
    
    def __init__(
        self,
        config: CollectiveConfig,
        head_config: Optional[HeadConfig] = None,
        head_preset: str = "standard",  # "standard", "lightweight", "heavy"
    ):
        self.config = config
        self.head_config = head_config or HeadConfig(
            feature_dim=config.feature_dim,
            fingerprint_dim=config.fingerprint_dim,
            num_anchors=config.num_anchors,
            num_routes=config.num_routes,
        )
        self.head_preset = head_preset
    
    def _make_head(self) -> ComposedHead:
        if self.head_preset == "lightweight":
            return build_lightweight_head(self.head_config)
        elif self.head_preset == "heavy":
            from geofractal.router.head.builder import HEAVY_HEAD
            return HeadBuilder(self.head_config).from_preset(HEAVY_HEAD).build()
        else:
            return build_standard_head(self.head_config)
    
    # === VECTOR ===
    
    def feature_vector(self, name: str, input_dim: int, num_slots: int = 4) -> FeatureVectorStream:
        return FeatureVectorStream(
            self.config, name, input_dim, num_slots, head=self._make_head()
        )
    
    def trainable_vector(
        self, name: str, backbone: nn.Module, input_dim: int, num_slots: int = 4
    ) -> TrainableVectorStream:
        return TrainableVectorStream(
            self.config, name, backbone, input_dim, num_slots=num_slots, head=self._make_head()
        )
    
    # === SEQUENCE ===
    
    def sequence(self, name: str, input_dim: int) -> SequenceStream:
        return SequenceStream(self.config, name, input_dim, head=self._make_head())
    
    def transformer_sequence(
        self, name: str, input_dim: int, num_layers: int = 2, num_heads: int = 8
    ) -> TransformerSequenceStream:
        return TransformerSequenceStream(
            self.config, name, input_dim, num_layers, num_heads, head=self._make_head()
        )
    
    def conv_sequence(
        self, name: str, input_dim: int, kernel_sizes: list = [3, 5, 7]
    ) -> ConvSequenceStream:
        return ConvSequenceStream(
            self.config, name, input_dim, kernel_sizes, head=self._make_head()
        )
