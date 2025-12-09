"""
geofractal.router.streams.frozen
================================
Frozen encoder stream for pretrained models.

Wraps pretrained vision models (CLIP, DINO, etc.) with frozen weights.
Only the slot expansion and head learn.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .stream_base import BaseStream, InputShape
from geofractal.router.head import HeadConfig, ComposedHead


class FrozenEncoderStream(BaseStream):
    """
    Stream wrapping a frozen pretrained encoder.

    The backbone is completely frozen - only slot expansion and head learn.

    Usage:
        stream = FrozenEncoderStream.from_pretrained(
            'openai/clip-vit-base-patch32',
            name='clip_b32',
            feature_dim=256,
            num_slots=16,
        )

        images = torch.randn(B, 3, 224, 224)
        routed, info = stream(images, mailbox, target_fp)
    """

    # Known model dimensions
    MODEL_DIMS = {
        "openai/clip-vit-base-patch32": 512,
        "openai/clip-vit-base-patch16": 512,
        "openai/clip-vit-large-patch14": 768,
        "openai/clip-vit-large-patch14-336": 768,
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": 512,
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": 1024,
        "facebook/dinov2-base": 768,
        "facebook/dinov2-large": 1024,
        "facebook/dinov2-giant": 1536,
    }

    def __init__(
            self,
            name: str,
            backbone: nn.Module,
            backbone_dim: int,
            feature_dim: int,
            num_slots: int = 16,
            use_fp16: bool = True,
            dropout: float = 0.1,
            head: Optional[ComposedHead] = None,
            head_config: Optional[HeadConfig] = None,
            parent_id: Optional[str] = None,
            cooperation_group: str = "default",
    ):
        super().__init__(
            name=name,
            input_dim=backbone_dim,
            feature_dim=feature_dim,
            head=head,
            head_config=head_config,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
        )

        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.num_slots = num_slots
        self.use_fp16 = use_fp16

        # Freeze backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Slot expansion (trainable)
        self.translation = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim * num_slots),
        )

        self.slot_embed = nn.Parameter(
            torch.randn(1, num_slots, feature_dim) * 0.02
        )

        # CLIP normalization constants
        self.register_buffer(
            'norm_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'norm_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    @property
    def input_shape(self) -> str:
        return InputShape.IMAGE

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for CLIP-style models.

        Args:
            x: [B, C, H, W] images in [0, 1] or normalized

        Returns:
            [B, 3, 224, 224] preprocessed
        """
        # Ensure 3 channels
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        # Resize if needed
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize for CLIP
        x = (x - self.norm_mean.to(x.device)) / self.norm_std.to(x.device)

        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images through frozen backbone.

        Args:
            x: [B, C, H, W] input images

        Returns:
            [B, backbone_dim] features
        """
        # Preprocess
        x = self.preprocess(x)

        # Encode through frozen backbone
        with torch.no_grad():
            if self.use_fp16 and x.device.type == 'cuda':
                outputs = self.backbone(pixel_values=x.half())
                features = outputs.pooler_output.float()
            else:
                outputs = self.backbone(pixel_values=x)
                features = outputs.pooler_output

        return features

    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Expand to slots.

        Args:
            features: [B, backbone_dim]

        Returns:
            [B, num_slots, feature_dim]
        """
        B = features.shape[0]
        translated = self.translation(features)
        slots = translated.view(B, self.num_slots, self.feature_dim)
        return slots + self.slot_embed

    @classmethod
    def from_pretrained(
            cls,
            model_name: str,
            name: Optional[str] = None,
            feature_dim: int = 256,
            num_slots: int = 16,
            device: Optional[str] = None,
            use_fp16: bool = True,
            dropout: float = 0.1,
            head: Optional[ComposedHead] = None,
            head_config: Optional[HeadConfig] = None,
            parent_id: Optional[str] = None,
            cooperation_group: str = "default",
    ) -> 'FrozenEncoderStream':
        """
        Create from HuggingFace model name.

        Args:
            model_name: HuggingFace model identifier
            name: Stream name (defaults to sanitized model_name)
            feature_dim: Output feature dimension
            num_slots: Number of routing slots
            device: Device to load model on
            use_fp16: Use FP16 for inference
        """
        from transformers import AutoModel

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Loading {model_name}...")
        dtype = torch.float16 if use_fp16 else torch.float32
        backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)

        # Get dimension
        if model_name in cls.MODEL_DIMS:
            backbone_dim = cls.MODEL_DIMS[model_name]
        elif hasattr(backbone.config, 'hidden_size'):
            backbone_dim = backbone.config.hidden_size
        elif hasattr(backbone.config, 'projection_dim'):
            backbone_dim = backbone.config.projection_dim
        else:
            raise ValueError(
                f"Unknown dimension for {model_name}. "
                f"Add to MODEL_DIMS or specify backbone_dim manually."
            )

        if name is None:
            name = model_name.split('/')[-1].replace('-', '_')

        return cls(
            name=name,
            backbone=backbone,
            backbone_dim=backbone_dim,
            feature_dim=feature_dim,
            num_slots=num_slots,
            use_fp16=use_fp16,
            dropout=dropout,
            head=head,
            head_config=head_config,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
        )


# Legacy alias
FrozenStream = FrozenEncoderStream

__all__ = [
    'FrozenEncoderStream',
    'FrozenStream',
]