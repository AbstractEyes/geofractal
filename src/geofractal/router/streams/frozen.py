"""
geofractal.router.streams.frozen
================================
Stream wrapping frozen pretrained models.

The backbone is completely frozen - no gradients.
Only translation head and router learn.

Proven with CLIP, designed for any HuggingFace vision model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, Union

from geofractal.router.streams.base import BaseStream
from geofractal.router.config import CollectiveConfig
from geofractal.router.registry import RouterMailbox


class FrozenStream(BaseStream):
    """
    Stream with frozen pretrained backbone.

    Wraps any HuggingFace vision model (CLIP, DINO, etc.)
    Backbone is frozen - only translation + router learn.

    Usage:
        stream = FrozenStream.from_pretrained(
            "openai/clip-vit-base-patch32",
            config=config,
            name="clip_base",
        )

        # images: [B, 3, 224, 224]
        routed, info = stream(images, mailbox)

    Results:
        - Dual CLIP: 98.6% frozen, 92.6% accuracy
        - CLIP + Conv: 93.4% with 10% individual streams
    """

    # Known model dimensions
    MODEL_DIMS = {
        "openai/clip-vit-base-patch32": 768,
        "openai/clip-vit-base-patch16": 768,
        "openai/clip-vit-large-patch14": 1024,
        "openai/clip-vit-large-patch14-336": 1024,
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": 512,
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": 1280,
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": 1024,
        "facebook/dinov2-base": 768,
        "facebook/dinov2-large": 1024,
        "facebook/dinov2-giant": 1536,
    }

    def __init__(
            self,
            config: CollectiveConfig,
            name: str,
            backbone: nn.Module,
            input_dim: int,
            parent_id: Optional[str] = None,
            cooperation_group: str = "frozen_collective",
            use_fp16: bool = True,
    ):
        super().__init__(
            config=config,
            name=name,
            input_dim=input_dim,
            parent_id=parent_id,
            cooperation_group=cooperation_group,
        )

        self.backbone = backbone
        self.use_fp16 = use_fp16

        # Freeze backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Register CLIP normalization constants
        self.register_buffer('norm_mean',
                             torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('norm_std',
                             torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images through frozen backbone.

        Args:
            x: [B, C, H, W] input images

        Returns:
            features: [B, input_dim] extracted features
        """
        with torch.no_grad():
            if self.use_fp16 and x.device.type == 'cuda':
                outputs = self.backbone(pixel_values=x.half())
                features = outputs.pooler_output.float()
            else:
                outputs = self.backbone(pixel_values=x)
                features = outputs.pooler_output

        return features

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for CLIP-style models.

        Args:
            x: [B, C, H, W] images in [0, 1] range or normalized

        Returns:
            preprocessed: [B, 3, 224, 224] ready for backbone
        """
        # Ensure 3 channels
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        # Resize if needed
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize for CLIP
        x = (x - self.norm_mean) / self.norm_std

        return x

    def forward(
            self,
            x: torch.Tensor,
            mailbox: RouterMailbox,
            target_fingerprint: Optional[torch.Tensor] = None,
            preprocess: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward with optional preprocessing.

        Args:
            x: Input images
            mailbox: Shared mailbox
            target_fingerprint: Next stream's fingerprint
            preprocess: Whether to preprocess images
        """
        if preprocess:
            x = self.preprocess(x)

        return super().forward(x, mailbox, target_fingerprint)

    @classmethod
    def from_pretrained(
            cls,
            model_name: str,
            config: CollectiveConfig,
            name: Optional[str] = None,
            parent_id: Optional[str] = None,
            device: Optional[str] = None,
            use_fp16: bool = True,
    ) -> "FrozenStream":
        """
        Create from HuggingFace model name.

        Args:
            model_name: HuggingFace model identifier
            config: Collective configuration
            name: Stream name (defaults to model_name)
            parent_id: Parent stream ID for hierarchy
            device: Device to load model on
            use_fp16: Use FP16 for inference
        """
        from transformers import AutoModel

        # Determine device
        if device is None:
            device = config.device

        # Load model
        print(f"  Loading {model_name}...")

        dtype = torch.float16 if use_fp16 else torch.float32
        backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)

        # Get dimension
        if model_name in cls.MODEL_DIMS:
            input_dim = cls.MODEL_DIMS[model_name]
        else:
            # Try to infer from model config
            if hasattr(backbone.config, 'hidden_size'):
                input_dim = backbone.config.hidden_size
            else:
                raise ValueError(f"Unknown model dimension for {model_name}. "
                                 f"Please specify input_dim manually.")

        # Create name
        if name is None:
            name = model_name.split('/')[-1].replace('-', '_')

        return cls(
            config=config,
            name=name,
            backbone=backbone,
            input_dim=input_dim,
            parent_id=parent_id,
            use_fp16=use_fp16,
        )