"""
geofractal.router.streams.builder
=================================
Stream factory/builder.

Creates streams with consistent configuration.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, List
import torch.nn as nn

from geofractal.router.head import HeadConfig, HeadBuilder, ComposedHead, build_standard_head

from .stream_vector import FeatureVectorStream, TrainableVectorStream
from .stream_sequence import SequenceStream, TransformerSequenceStream, ConvSequenceStream
from .stream_frozen import FrozenEncoderStream


class StreamBuilder:
    """
    Factory for building streams with consistent configuration.

    Usage:
        builder = StreamBuilder(
            feature_dim=256,
            num_slots=16,
            fingerprint_dim=64,
        )

        # Vector streams
        clip_stream = builder.feature_vector('clip', input_dim=512)
        conv_stream = builder.trainable_conv('conv', in_channels=1)

        # Sequence streams
        seq_stream = builder.sequence('tokens', input_dim=768)

        # Frozen encoder
        frozen = builder.frozen_encoder('openai/clip-vit-base-patch32')
    """

    def __init__(
            self,
            feature_dim: int = 256,
            num_slots: int = 16,
            fingerprint_dim: int = 64,
            num_anchors: int = 16,
            num_routes: int = 4,
            dropout: float = 0.1,
            head_preset: str = "standard",  # "standard", "lightweight", "heavy"
            cooperation_group: str = "default",
    ):
        self.feature_dim = feature_dim
        self.num_slots = num_slots
        self.dropout = dropout
        self.head_preset = head_preset
        self.cooperation_group = cooperation_group

        # Head config template
        self.head_config = HeadConfig(
            feature_dim=feature_dim,
            fingerprint_dim=fingerprint_dim,
            num_anchors=num_anchors,
            num_routes=num_routes,
            dropout=dropout,
        )

        # Track parent for chaining
        self._last_parent_id: Optional[str] = None

    def _make_head(self, name: str) -> ComposedHead:
        """Build a head with current config."""
        if self.head_preset == "lightweight":
            return HeadBuilder.lightweight(self.head_config).with_name(f"{name}_head").build()
        elif self.head_preset == "heavy":
            return HeadBuilder.heavy(self.head_config).with_name(f"{name}_head").build()
        else:
            return HeadBuilder.standard(self.head_config).with_name(f"{name}_head").build()

    def _chain_parent(self, stream) -> str:
        """Update parent chain and return previous parent."""
        parent = self._last_parent_id
        self._last_parent_id = stream.module_id
        return parent

    # === VECTOR STREAMS ===

    def feature_vector(
            self,
            name: str,
            input_dim: int,
            num_slots: Optional[int] = None,
            chain: bool = True,
    ) -> FeatureVectorStream:
        """Build stream for pre-extracted features."""
        parent_id = self._last_parent_id if chain else None

        stream = FeatureVectorStream(
            name=name,
            input_dim=input_dim,
            feature_dim=self.feature_dim,
            num_slots=num_slots or self.num_slots,
            dropout=self.dropout,
            head=self._make_head(name),
            parent_id=parent_id,
            cooperation_group=self.cooperation_group,
        )

        if chain:
            self._chain_parent(stream)

        return stream

    def trainable_vector(
            self,
            name: str,
            backbone: nn.Module,
            backbone_dim: int,
            num_slots: Optional[int] = None,
            chain: bool = True,
    ) -> TrainableVectorStream:
        """Build stream with custom trainable backbone."""
        parent_id = self._last_parent_id if chain else None

        stream = TrainableVectorStream(
            name=name,
            backbone=backbone,
            backbone_dim=backbone_dim,
            feature_dim=self.feature_dim,
            num_slots=num_slots or self.num_slots,
            dropout=self.dropout,
            head=self._make_head(name),
            parent_id=parent_id,
            cooperation_group=self.cooperation_group,
        )

        if chain:
            self._chain_parent(stream)

        return stream

    def trainable_conv(
            self,
            name: str,
            in_channels: int = 1,
            channels: List[int] = None,
            image_size: int = 28,
            num_slots: Optional[int] = None,
            chain: bool = True,
    ) -> TrainableVectorStream:
        """Build stream with conv backbone."""
        parent_id = self._last_parent_id if chain else None

        stream = TrainableVectorStream.conv_stream(
            name=name,
            feature_dim=self.feature_dim,
            num_slots=num_slots or self.num_slots,
            in_channels=in_channels,
            channels=channels,
            image_size=image_size,
            dropout=self.dropout,
            head=self._make_head(name),
            parent_id=parent_id,
            cooperation_group=self.cooperation_group,
        )

        if chain:
            self._chain_parent(stream)

        return stream

    def trainable_mlp(
            self,
            name: str,
            input_dim: int,
            hidden_dims: List[int] = None,
            num_slots: Optional[int] = None,
            chain: bool = True,
    ) -> TrainableVectorStream:
        """Build stream with MLP backbone."""
        parent_id = self._last_parent_id if chain else None

        stream = TrainableVectorStream.mlp_stream(
            name=name,
            input_dim=input_dim,
            feature_dim=self.feature_dim,
            num_slots=num_slots or self.num_slots,
            hidden_dims=hidden_dims,
            dropout=self.dropout,
            head=self._make_head(name),
            parent_id=parent_id,
            cooperation_group=self.cooperation_group,
        )

        if chain:
            self._chain_parent(stream)

        return stream

    # === SEQUENCE STREAMS ===

    def sequence(
            self,
            name: str,
            input_dim: int,
            chain: bool = True,
    ) -> SequenceStream:
        """Build basic sequence projection stream."""
        parent_id = self._last_parent_id if chain else None

        stream = SequenceStream(
            name=name,
            input_dim=input_dim,
            feature_dim=self.feature_dim,
            head=self._make_head(name),
            parent_id=parent_id,
            cooperation_group=self.cooperation_group,
        )

        if chain:
            self._chain_parent(stream)

        return stream

    def transformer(
            self,
            name: str,
            input_dim: int,
            num_layers: int = 2,
            num_heads: int = 8,
            chain: bool = True,
    ) -> TransformerSequenceStream:
        """Build sequence stream with transformer layers."""
        parent_id = self._last_parent_id if chain else None

        stream = TransformerSequenceStream(
            name=name,
            input_dim=input_dim,
            feature_dim=self.feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=self.dropout,
            head=self._make_head(name),
            parent_id=parent_id,
            cooperation_group=self.cooperation_group,
        )

        if chain:
            self._chain_parent(stream)

        return stream

    def conv_sequence(
            self,
            name: str,
            input_dim: int,
            kernel_sizes: List[int] = None,
            chain: bool = True,
    ) -> ConvSequenceStream:
        """Build sequence stream with multi-scale conv."""
        parent_id = self._last_parent_id if chain else None

        stream = ConvSequenceStream(
            name=name,
            input_dim=input_dim,
            feature_dim=self.feature_dim,
            kernel_sizes=kernel_sizes,
            dropout=self.dropout,
            head=self._make_head(name),
            parent_id=parent_id,
            cooperation_group=self.cooperation_group,
        )

        if chain:
            self._chain_parent(stream)

        return stream

    # === FROZEN ENCODER ===

    def frozen_encoder(
            self,
            model_name: str,
            name: Optional[str] = None,
            num_slots: Optional[int] = None,
            use_fp16: bool = True,
            chain: bool = True,
    ) -> FrozenEncoderStream:
        """Build stream with frozen pretrained encoder."""
        parent_id = self._last_parent_id if chain else None

        stream = FrozenEncoderStream.from_pretrained(
            model_name=model_name,
            name=name,
            feature_dim=self.feature_dim,
            num_slots=num_slots or self.num_slots,
            use_fp16=use_fp16,
            dropout=self.dropout,
            head=self._make_head(name or model_name.split('/')[-1]),
            parent_id=parent_id,
            cooperation_group=self.cooperation_group,
        )

        if chain:
            self._chain_parent(stream)

        return stream

    def reset_chain(self):
        """Reset parent chain for new independent streams."""
        self._last_parent_id = None


__all__ = ['StreamBuilder']