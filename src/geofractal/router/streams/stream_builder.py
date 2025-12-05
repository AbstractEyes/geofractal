# streams/builder.py
"""
Stream factory/builder.

Simple factory for creating streams with consistent configuration.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, List
import torch.nn as nn

from .stream_vector import FeatureVectorStream, TrainableVectorStream
from .stream_sequence import SequenceStream, TransformerSequenceStream, ConvSequenceStream
from .stream_frozen import FrozenEncoderStream


class StreamBuilder:
    """
    Factory for building streams.

    Usage:
        builder = StreamBuilder(feature_dim=256)

        # Vector streams
        clip_stream = builder.feature_vector('clip', input_dim=512)
        conv_stream = builder.trainable_conv('conv', in_channels=1)

        # Sequence streams
        seq_stream = builder.sequence('tokens', input_dim=768)
        transformer_stream = builder.transformer('encoder', input_dim=768)

        # Frozen encoder
        frozen = builder.frozen_encoder('openai/clip-vit-base-patch32')
    """

    def __init__(
            self,
            feature_dim: int = 256,
            dropout: float = 0.1,
    ):
        self.feature_dim = feature_dim
        self.dropout = dropout

    # === VECTOR STREAMS ===

    def feature_vector(
            self,
            name: str,
            input_dim: int,
            feature_dim: Optional[int] = None,
    ) -> FeatureVectorStream:
        """Build stream for pre-extracted features."""
        return FeatureVectorStream(
            name=name,
            input_dim=input_dim,
            feature_dim=feature_dim or self.feature_dim,
            dropout=self.dropout,
        )

    def trainable_vector(
            self,
            name: str,
            backbone: nn.Module,
            input_dim: int,
            feature_dim: Optional[int] = None,
    ) -> TrainableVectorStream:
        """Build stream with custom trainable backbone."""
        return TrainableVectorStream(
            name=name,
            backbone=backbone,
            input_dim=input_dim,
            feature_dim=feature_dim or self.feature_dim,
            dropout=self.dropout,
        )

    def trainable_conv(
            self,
            name: str,
            in_channels: int = 1,
            channels: List[int] = None,
            image_size: int = 28,
            feature_dim: Optional[int] = None,
    ) -> TrainableVectorStream:
        """Build stream with conv backbone."""
        return TrainableVectorStream.conv_stream(
            name=name,
            feature_dim=feature_dim or self.feature_dim,
            in_channels=in_channels,
            channels=channels,
            image_size=image_size,
            dropout=self.dropout,
        )

    def trainable_mlp(
            self,
            name: str,
            input_dim: int,
            hidden_dims: List[int] = None,
            feature_dim: Optional[int] = None,
    ) -> TrainableVectorStream:
        """Build stream with MLP backbone."""
        return TrainableVectorStream.mlp_stream(
            name=name,
            input_dim=input_dim,
            feature_dim=feature_dim or self.feature_dim,
            hidden_dims=hidden_dims,
            dropout=self.dropout,
        )

    # === SEQUENCE STREAMS ===

    def sequence(
            self,
            name: str,
            input_dim: int,
            feature_dim: Optional[int] = None,
    ) -> SequenceStream:
        """Build basic sequence projection stream."""
        return SequenceStream(
            name=name,
            input_dim=input_dim,
            feature_dim=feature_dim or self.feature_dim,
        )

    def transformer(
            self,
            name: str,
            input_dim: int,
            num_layers: int = 2,
            num_heads: int = 8,
            feature_dim: Optional[int] = None,
    ) -> TransformerSequenceStream:
        """Build sequence stream with transformer layers."""
        return TransformerSequenceStream(
            name=name,
            input_dim=input_dim,
            feature_dim=feature_dim or self.feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=self.dropout,
        )

    def conv_sequence(
            self,
            name: str,
            input_dim: int,
            kernel_sizes: List[int] = None,
            feature_dim: Optional[int] = None,
    ) -> ConvSequenceStream:
        """Build sequence stream with multi-scale conv."""
        return ConvSequenceStream(
            name=name,
            input_dim=input_dim,
            feature_dim=feature_dim or self.feature_dim,
            kernel_sizes=kernel_sizes,
            dropout=self.dropout,
        )

    # === FROZEN ENCODER ===

    def frozen_encoder(
            self,
            model_name: str,
            name: Optional[str] = None,
            feature_dim: Optional[int] = None,
            device: Optional[str] = None,
            use_fp16: bool = True,
    ) -> FrozenEncoderStream:
        """Build stream with frozen pretrained encoder."""
        return FrozenEncoderStream.from_pretrained(
            model_name=model_name,
            name=name,
            feature_dim=feature_dim or self.feature_dim,
            device=device,
            use_fp16=use_fp16,
            dropout=self.dropout,
        )

    # === STATIC BUILD ===

    @staticmethod
    def build(
            stream_type: str,
            name: str,
            input_dim: int,
            feature_dim: int,
            dropout: float = 0.1,
            **kwargs,
    ) -> nn.Module:
        """
        Build stream from type string.

        Args:
            stream_type: One of 'feature_vector', 'trainable_vector',
                        'sequence', 'transformer', 'conv_sequence'
            name: Stream name
            input_dim: Input dimension
            feature_dim: Output feature dimension
            **kwargs: Additional args for specific stream types
        """
        if stream_type in ('feature', 'feature_vector'):
            return FeatureVectorStream(
                name=name,
                input_dim=input_dim,
                feature_dim=feature_dim,
                dropout=dropout,
            )

        elif stream_type in ('sequence', 'frozen'):
            return SequenceStream(
                name=name,
                input_dim=input_dim,
                feature_dim=feature_dim,
            )

        elif stream_type == 'transformer':
            return TransformerSequenceStream(
                name=name,
                input_dim=input_dim,
                feature_dim=feature_dim,
                num_layers=kwargs.get('num_layers', 2),
                num_heads=kwargs.get('num_heads', 8),
                dropout=dropout,
            )

        elif stream_type == 'conv_sequence':
            return ConvSequenceStream(
                name=name,
                input_dim=input_dim,
                feature_dim=feature_dim,
                kernel_sizes=kwargs.get('kernel_sizes'),
                dropout=dropout,
            )

        else:
            raise ValueError(f"Unknown stream type: {stream_type}")


__all__ = ['StreamBuilder']