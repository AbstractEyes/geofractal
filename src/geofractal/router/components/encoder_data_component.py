"""
geofractal.router.components.encoder_data_component
====================================================

Encoder-based DataComponent with intelligent caching, multi-model support,
and flexible loading from local paths or HuggingFace.

Classes:
    EncoderDataComponent: Base class for encoder-based data processing
    MultiTextEncode: Multiple text encoders with optional caching
    MultiVisionEncode: Multiple vision encoders with optional caching
    MultiEncode: Efficient combination of text and vision encoders

Features:
    - Local model loading or HuggingFace AutoModel
    - Full epoch caching for rapid recall
    - Device management and mixed precision
    - Batched encoding for efficiency
    - Cache persistence (save/load to disk)

Usage:
    # Text encoding with caching
    text_enc = MultiTextEncode(
        encoders=['clip_l', 'clip_g'],
        device='cuda',
        cache_enabled=True,
    )
    embeddings = text_enc.encode_batch(texts)
    text_enc.save_cache('text_cache.pt')

    # Vision encoding with DINO
    vision_enc = MultiVisionEncode(
        encoders=['dinov2_base'],
        device='cuda',
    )
    features = vision_enc.encode_batch(images)

    # Combined multi-modal
    multi_enc = MultiEncode(
        text_encoders=['clip_l', 'clip_g'],
        vision_encoders=['dinov2_base'],
        device='cuda',
    )
    text_emb, vision_emb = multi_enc.encode(texts, images)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Any, Dict, Callable, Union, Tuple, List, Literal
from dataclasses import dataclass, field
from pathlib import Path
from abc import abstractmethod
from tqdm.auto import tqdm
import hashlib
import json
import os

# HuggingFace imports
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModel,
    CLIPProcessor,
)
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from huggingface_hub import hf_hub_download, HfApi

# Local imports
from geofractal.router.components.data_component import DataComponent


# =============================================================================
# CONSTANTS
# =============================================================================

# Supported encoder types
TEXT_ENCODER_TYPES = Literal[
    # CLIP text
    'clip_l', 'clip_g', 'clip_h', 'clip_b16', 'clip_b32', 'clip_l14_336',
    'clip_l_illustrious', 'clip_g_illustrious',
    'openclip_h', 'openclip_g',
    # SigLIP text
    'siglip_base', 'siglip_large', 'siglip_so400m',
    # T5
    't5_small', 't5_base', 't5_large', 't5_xl', 't5_xxl',
    # Flan-T5
    'flan_t5_small', 'flan_t5_base', 'flan_t5_large', 'flan_t5_xl', 'flan_t5_xxl',
    # Qwen2
    'qwen2_0.5b', 'qwen2_1.5b', 'qwen2_7b',
    'qwen2_0.5b_instruct', 'qwen2_1.5b_instruct', 'qwen2_7b_instruct',
    # Qwen2.5
    'qwen2.5_0.5b', 'qwen2.5_1.5b', 'qwen2.5_3b', 'qwen2.5_7b', 'qwen2.5_14b', 'qwen2.5_32b',
    'qwen2.5_0.5b_instruct', 'qwen2.5_1.5b_instruct', 'qwen2.5_3b_instruct',
    'qwen2.5_7b_instruct', 'qwen2.5_14b_instruct', 'qwen2.5_32b_instruct',
    'qwen2.5_coder_1.5b', 'qwen2.5_coder_7b',
    # BERT
    'bert_base', 'bert_large',
]

VISION_ENCODER_TYPES = Literal[
    # DINO v1
    'dino_vit_small', 'dino_vit_small_8', 'dino_vit_base', 'dino_vit_base_8',
    # DINO v2
    'dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant',
    'dinov2_small_reg', 'dinov2_base_reg', 'dinov2_large_reg', 'dinov2_giant_reg',
    # CLIP vision
    'clip_vit_b16', 'clip_vit_b32', 'clip_vit_large', 'clip_vit_large_336',
    'clip_vit_h', 'clip_vit_g',
    # SigLIP vision
    'siglip_vit_base', 'siglip_vit_large', 'siglip_vit_so400m',
    # EVA
    'eva_vit_g', 'eva02_base', 'eva02_large',
    # ConvNeXt
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
]

# Model registry with HuggingFace paths
MODEL_REGISTRY = {
    # =========================================================================
    # CLIP TEXT ENCODERS
    # =========================================================================
    'clip_l': {
        'type': 'text',
        'hf_path': 'openai/clip-vit-large-patch14',
        'model_class': CLIPTextModel,
        'tokenizer_class': CLIPTokenizer,
        'dim': 768,
        'max_length': 77,
    },
    'clip_g': {
        'type': 'text',
        'hf_path': 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
        'model_class': CLIPTextModelWithProjection,
        'tokenizer_class': CLIPTokenizer,
        'dim': 1280,
        'max_length': 77,
    },
    'clip_h': {
        'type': 'text',
        'hf_path': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        'model_class': CLIPTextModelWithProjection,
        'tokenizer_class': CLIPTokenizer,
        'dim': 1024,
        'max_length': 77,
    },
    'clip_b16': {
        'type': 'text',
        'hf_path': 'openai/clip-vit-base-patch16',
        'model_class': CLIPTextModel,
        'tokenizer_class': CLIPTokenizer,
        'dim': 512,
        'max_length': 77,
    },
    'clip_b32': {
        'type': 'text',
        'hf_path': 'openai/clip-vit-base-patch32',
        'model_class': CLIPTextModel,
        'tokenizer_class': CLIPTokenizer,
        'dim': 512,
        'max_length': 77,
    },
    'clip_l14_336': {
        'type': 'text',
        'hf_path': 'openai/clip-vit-large-patch14-336',
        'model_class': CLIPTextModel,
        'tokenizer_class': CLIPTokenizer,
        'dim': 768,
        'max_length': 77,
    },

    # Illustrious CLIP (custom weights)
    'clip_l_illustrious': {
        'type': 'text',
        'hf_path': 'openai/clip-vit-large-patch14',
        'weights_repo': 'AbstractPhil/clips',
        'weights_file': 'IllustriousXL20_v20_clip_l.safetensors',
        'model_class': CLIPTextModel,
        'tokenizer_class': CLIPTokenizer,
        'dim': 768,
        'max_length': 77,
    },
    'clip_g_illustrious': {
        'type': 'text',
        'hf_path': 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
        'weights_repo': 'AbstractPhil/clips',
        'weights_file': 'IllustriousXL20_v20_clip_g.safetensors',
        'model_class': CLIPTextModelWithProjection,
        'tokenizer_class': CLIPTokenizer,
        'dim': 1280,
        'max_length': 77,
    },

    # OpenCLIP variants
    'openclip_h': {
        'type': 'text',
        'hf_path': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        'model_class': CLIPTextModelWithProjection,
        'tokenizer_class': CLIPTokenizer,
        'dim': 1024,
        'max_length': 77,
    },
    'openclip_g': {
        'type': 'text',
        'hf_path': 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
        'model_class': CLIPTextModelWithProjection,
        'tokenizer_class': CLIPTokenizer,
        'dim': 1280,
        'max_length': 77,
    },

    # SigLIP text
    'siglip_base': {
        'type': 'text',
        'hf_path': 'google/siglip-base-patch16-224',
        'dim': 768,
        'max_length': 64,
    },
    'siglip_large': {
        'type': 'text',
        'hf_path': 'google/siglip-large-patch16-256',
        'dim': 1024,
        'max_length': 64,
    },
    'siglip_so400m': {
        'type': 'text',
        'hf_path': 'google/siglip-so400m-patch14-384',
        'dim': 1152,
        'max_length': 64,
    },

    # =========================================================================
    # T5 ENCODERS
    # =========================================================================
    't5_small': {
        'type': 'text',
        'hf_path': 'google-t5/t5-small',
        'dim': 512,
        'max_length': 512,
    },
    't5_base': {
        'type': 'text',
        'hf_path': 'google-t5/t5-base',
        'dim': 768,
        'max_length': 512,
    },
    't5_large': {
        'type': 'text',
        'hf_path': 'google-t5/t5-large',
        'dim': 1024,
        'max_length': 512,
    },
    't5_xl': {
        'type': 'text',
        'hf_path': 'google-t5/t5-3b',
        'dim': 2048,
        'max_length': 512,
    },
    't5_xxl': {
        'type': 'text',
        'hf_path': 'google-t5/t5-11b',
        'dim': 4096,
        'max_length': 512,
    },

    # Flan-T5 (instruction-tuned)
    'flan_t5_small': {
        'type': 'text',
        'hf_path': 'google/flan-t5-small',
        'dim': 512,
        'max_length': 512,
    },
    'flan_t5_base': {
        'type': 'text',
        'hf_path': 'google/flan-t5-base',
        'dim': 768,
        'max_length': 512,
    },
    'flan_t5_large': {
        'type': 'text',
        'hf_path': 'google/flan-t5-large',
        'dim': 1024,
        'max_length': 512,
    },
    'flan_t5_xl': {
        'type': 'text',
        'hf_path': 'google/flan-t5-xl',
        'dim': 2048,
        'max_length': 512,
    },
    'flan_t5_xxl': {
        'type': 'text',
        'hf_path': 'google/flan-t5-xxl',
        'dim': 4096,
        'max_length': 512,
    },

    # =========================================================================
    # QWEN MODELS
    # =========================================================================
    # Qwen2 Base
    'qwen2_0.5b': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2-0.5B',
        'dim': 896,
        'max_length': 2048,
        'is_decoder': True,
    },
    'qwen2_1.5b': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2-1.5B',
        'dim': 1536,
        'max_length': 2048,
        'is_decoder': True,
    },
    'qwen2_7b': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2-7B',
        'dim': 3584,
        'max_length': 4096,
        'is_decoder': True,
    },

    # Qwen2 Instruct
    'qwen2_0.5b_instruct': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2-0.5B-Instruct',
        'dim': 896,
        'max_length': 2048,
        'is_decoder': True,
    },
    'qwen2_1.5b_instruct': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2-1.5B-Instruct',
        'dim': 1536,
        'max_length': 2048,
        'is_decoder': True,
    },
    'qwen2_7b_instruct': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2-7B-Instruct',
        'dim': 3584,
        'max_length': 4096,
        'is_decoder': True,
    },

    # Qwen2.5 Base
    'qwen2.5_0.5b': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-0.5B',
        'dim': 896,
        'max_length': 4096,
        'is_decoder': True,
    },
    'qwen2.5_1.5b': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-1.5B',
        'dim': 1536,
        'max_length': 4096,
        'is_decoder': True,
    },
    'qwen2.5_3b': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-3B',
        'dim': 2048,
        'max_length': 4096,
        'is_decoder': True,
    },
    'qwen2.5_7b': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-7B',
        'dim': 3584,
        'max_length': 4096,
        'is_decoder': True,
    },
    'qwen2.5_14b': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-14B',
        'dim': 5120,
        'max_length': 4096,
        'is_decoder': True,
    },
    'qwen2.5_32b': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-32B',
        'dim': 5120,
        'max_length': 4096,
        'is_decoder': True,
    },

    # Qwen2.5 Instruct
    'qwen2.5_0.5b_instruct': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-0.5B-Instruct',
        'dim': 896,
        'max_length': 4096,
        'is_decoder': True,
    },
    'qwen2.5_1.5b_instruct': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-1.5B-Instruct',
        'dim': 1536,
        'max_length': 4096,
        'is_decoder': True,
    },
    'qwen2.5_3b_instruct': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-3B-Instruct',
        'dim': 2048,
        'max_length': 4096,
        'is_decoder': True,
    },
    'qwen2.5_7b_instruct': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-7B-Instruct',
        'dim': 3584,
        'max_length': 4096,
        'is_decoder': True,
    },
    'qwen2.5_14b_instruct': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-14B-Instruct',
        'dim': 5120,
        'max_length': 4096,
        'is_decoder': True,
    },
    'qwen2.5_32b_instruct': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-32B-Instruct',
        'dim': 5120,
        'max_length': 4096,
        'is_decoder': True,
    },

    # Qwen2.5 Coder
    'qwen2.5_coder_1.5b': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-Coder-1.5B',
        'dim': 1536,
        'max_length': 4096,
        'is_decoder': True,
    },
    'qwen2.5_coder_7b': {
        'type': 'text',
        'hf_path': 'Qwen/Qwen2.5-Coder-7B',
        'dim': 3584,
        'max_length': 4096,
        'is_decoder': True,
    },

    # =========================================================================
    # BERT ENCODERS
    # =========================================================================
    'bert_base': {
        'type': 'text',
        'hf_path': 'google-bert/bert-base-uncased',
        'dim': 768,
        'max_length': 512,
    },
    'bert_large': {
        'type': 'text',
        'hf_path': 'google-bert/bert-large-uncased',
        'dim': 1024,
        'max_length': 512,
    },

    # =========================================================================
    # DINO V1 VISION ENCODERS
    # =========================================================================
    'dino_vit_small': {
        'type': 'vision',
        'hf_path': 'facebook/dino-vits16',
        'dim': 384,
        'patch_size': 16,
    },
    'dino_vit_small_8': {
        'type': 'vision',
        'hf_path': 'facebook/dino-vits8',
        'dim': 384,
        'patch_size': 8,
    },
    'dino_vit_base': {
        'type': 'vision',
        'hf_path': 'facebook/dino-vitb16',
        'dim': 768,
        'patch_size': 16,
    },
    'dino_vit_base_8': {
        'type': 'vision',
        'hf_path': 'facebook/dino-vitb8',
        'dim': 768,
        'patch_size': 8,
    },

    # =========================================================================
    # DINO V2 VISION ENCODERS
    # =========================================================================
    'dinov2_small': {
        'type': 'vision',
        'hf_path': 'facebook/dinov2-small',
        'dim': 384,
        'patch_size': 14,
    },
    'dinov2_base': {
        'type': 'vision',
        'hf_path': 'facebook/dinov2-base',
        'dim': 768,
        'patch_size': 14,
    },
    'dinov2_large': {
        'type': 'vision',
        'hf_path': 'facebook/dinov2-large',
        'dim': 1024,
        'patch_size': 14,
    },
    'dinov2_giant': {
        'type': 'vision',
        'hf_path': 'facebook/dinov2-giant',
        'dim': 1536,
        'patch_size': 14,
    },

    # DINOv2 with registers (improved)
    'dinov2_small_reg': {
        'type': 'vision',
        'hf_path': 'facebook/dinov2-small-imagenet1k-1-layer',
        'dim': 384,
        'patch_size': 14,
    },
    'dinov2_base_reg': {
        'type': 'vision',
        'hf_path': 'facebook/dinov2-base-imagenet1k-1-layer',
        'dim': 768,
        'patch_size': 14,
    },
    'dinov2_large_reg': {
        'type': 'vision',
        'hf_path': 'facebook/dinov2-large-imagenet1k-1-layer',
        'dim': 1024,
        'patch_size': 14,
    },
    'dinov2_giant_reg': {
        'type': 'vision',
        'hf_path': 'facebook/dinov2-giant-imagenet1k-1-layer',
        'dim': 1536,
        'patch_size': 14,
    },

    # =========================================================================
    # CLIP VISION ENCODERS
    # =========================================================================
    'clip_vit_b16': {
        'type': 'vision',
        'hf_path': 'openai/clip-vit-base-patch16',
        'dim': 768,
        'patch_size': 16,
    },
    'clip_vit_b32': {
        'type': 'vision',
        'hf_path': 'openai/clip-vit-base-patch32',
        'dim': 768,
        'patch_size': 32,
    },
    'clip_vit_large': {
        'type': 'vision',
        'hf_path': 'openai/clip-vit-large-patch14',
        'dim': 1024,
        'patch_size': 14,
    },
    'clip_vit_large_336': {
        'type': 'vision',
        'hf_path': 'openai/clip-vit-large-patch14-336',
        'dim': 1024,
        'patch_size': 14,
    },
    'clip_vit_h': {
        'type': 'vision',
        'hf_path': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        'dim': 1280,
        'patch_size': 14,
    },
    'clip_vit_g': {
        'type': 'vision',
        'hf_path': 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
        'dim': 1664,
        'patch_size': 14,
    },

    # =========================================================================
    # SIGLIP VISION ENCODERS
    # =========================================================================
    'siglip_vit_base': {
        'type': 'vision',
        'hf_path': 'google/siglip-base-patch16-224',
        'dim': 768,
        'patch_size': 16,
    },
    'siglip_vit_large': {
        'type': 'vision',
        'hf_path': 'google/siglip-large-patch16-256',
        'dim': 1024,
        'patch_size': 16,
    },
    'siglip_vit_so400m': {
        'type': 'vision',
        'hf_path': 'google/siglip-so400m-patch14-384',
        'dim': 1152,
        'patch_size': 14,
    },

    # =========================================================================
    # EVA VISION ENCODERS
    # =========================================================================
    'eva_vit_g': {
        'type': 'vision',
        'hf_path': 'BAAI/EVA-CLIP-8B',
        'dim': 1024,
        'patch_size': 14,
    },
    'eva02_base': {
        'type': 'vision',
        'hf_path': 'Salesforce/eva02-base-patch14-224',
        'dim': 768,
        'patch_size': 14,
    },
    'eva02_large': {
        'type': 'vision',
        'hf_path': 'Salesforce/eva02-large-patch14-224',
        'dim': 1024,
        'patch_size': 14,
    },

    # =========================================================================
    # CONVNEXT VISION ENCODERS (for DINOv2 style)
    # =========================================================================
    'convnext_tiny': {
        'type': 'vision',
        'hf_path': 'facebook/convnext-tiny-224',
        'dim': 768,
    },
    'convnext_small': {
        'type': 'vision',
        'hf_path': 'facebook/convnext-small-224',
        'dim': 768,
    },
    'convnext_base': {
        'type': 'vision',
        'hf_path': 'facebook/convnext-base-224',
        'dim': 1024,
    },
    'convnext_large': {
        'type': 'vision',
        'hf_path': 'facebook/convnext-large-224',
        'dim': 1536,
    },
}


# =============================================================================
# STAGED CACHE BUILDER
# =============================================================================

class StagedCacheBuilder:
    """
    Staged caching utility for sequential model encoding with VRAM management.

    Workflow:
        1. Load model A → encode all data → cache to disk → unload A
        2. Load model B → encode all data → cache to disk → unload B
        3. Training: CachedDataset yields from disk, no models needed

    Usage:
        builder = StagedCacheBuilder(
            dataset_name='danbooru_100k',
            cache_dir='./cache',
        )

        # Stage 1: Small fast model
        builder.add_stage('clip_l_illustrious', texts=all_texts)
        builder.add_stage('clip_g_illustrious', texts=all_texts)

        # Stage 2: Large slow model
        builder.add_stage('dinov2_large', images=all_images)

        # Execute all stages sequentially (models unloaded between stages)
        builder.build()

        # Get dataset that yields only cached tensors (no models needed)
        dataset = builder.get_cached_dataset()
    """

    def __init__(
        self,
        dataset_name: str,
        cache_dir: str = './encoder_cache',
        device: str = 'cuda',
    ):
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir) / dataset_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Stages to execute
        self.stages: List[Dict] = []

        # Metadata about completed caches
        self.cache_manifest: Dict[str, Dict] = {}
        self._load_manifest()

    def _manifest_path(self) -> Path:
        return self.cache_dir / 'manifest.json'

    def _load_manifest(self) -> None:
        """Load existing cache manifest."""
        if self._manifest_path().exists():
            with open(self._manifest_path()) as f:
                self.cache_manifest = json.load(f)

    def _save_manifest(self) -> None:
        """Save cache manifest."""
        with open(self._manifest_path(), 'w') as f:
            json.dump(self.cache_manifest, f, indent=2)

    def add_stage(
        self,
        encoder_name: str,
        texts: Optional[List[str]] = None,
        images: Optional[Tensor] = None,
        batch_size: int = 32,
        clip_skip: int = 2,
        force_rebuild: bool = False,
    ) -> 'StagedCacheBuilder':
        """
        Add an encoding stage.

        Args:
            encoder_name: Model from registry
            texts: Text inputs (for text encoders)
            images: Image inputs (for vision encoders)
            batch_size: Encoding batch size
            clip_skip: CLIP skip layers
            force_rebuild: Rebuild even if cache exists
        """
        self.stages.append({
            'encoder_name': encoder_name,
            'texts': texts,
            'images': images,
            'batch_size': batch_size,
            'clip_skip': clip_skip,
            'force_rebuild': force_rebuild,
        })
        return self

    def _get_cache_path(self, encoder_name: str) -> Path:
        """Get cache file path for encoder."""
        return self.cache_dir / f'{encoder_name}.safetensors'

    def _is_cached(self, encoder_name: str, num_samples: int) -> bool:
        """Check if encoder output is already cached with correct count."""
        if encoder_name not in self.cache_manifest:
            return False

        meta = self.cache_manifest[encoder_name]
        cache_path = self._get_cache_path(encoder_name)

        return (
            cache_path.exists() and
            meta.get('num_samples') == num_samples and
            meta.get('complete', False)
        )

    def build(self, show_progress: bool = True) -> 'StagedCacheBuilder':
        """
        Execute all stages sequentially.

        Each stage:
            1. Checks if cache exists (skips if so)
            2. Loads model
            3. Encodes all data
            4. Saves to disk
            5. Unloads model (frees VRAM)
        """
        print(f"\n{'='*60}")
        print(f"StagedCacheBuilder: {self.dataset_name}")
        print(f"{'='*60}")
        print(f"Stages: {len(self.stages)}")
        print(f"Cache dir: {self.cache_dir}")

        for i, stage in enumerate(self.stages):
            encoder_name = stage['encoder_name']
            texts = stage['texts']
            images = stage['images']
            batch_size = stage['batch_size']
            force_rebuild = stage['force_rebuild']

            num_samples = len(texts) if texts is not None else len(images)

            print(f"\n--- Stage {i+1}/{len(self.stages)}: {encoder_name} ---")

            # Check cache
            if not force_rebuild and self._is_cached(encoder_name, num_samples):
                print(f"  ✓ Already cached ({num_samples} samples)")
                continue

            # Get encoder config
            if encoder_name not in MODEL_REGISTRY:
                raise ValueError(f"Unknown encoder: {encoder_name}")

            config = MODEL_REGISTRY[encoder_name]
            encoder_type = config['type']

            # Create appropriate encoder
            if encoder_type == 'text':
                encoder = MultiTextEncode(
                    encoders=[encoder_name],
                    dataset_name=self.dataset_name,
                    device=self.device,
                    cache_enabled=False,  # We handle caching ourselves
                    concatenate=True,
                    clip_skip=stage['clip_skip'],
                )
            else:
                encoder = MultiVisionEncode(
                    encoders=[encoder_name],
                    dataset_name=self.dataset_name,
                    device=self.device,
                    cache_enabled=False,
                    concatenate=True,
                )

            # Encode in batches
            print(f"  Encoding {num_samples} samples...")
            all_embeddings = []

            if texts is not None:
                iterator = range(0, len(texts), batch_size)
                if show_progress:
                    iterator = tqdm(iterator, desc=f"  {encoder_name}")

                for j in iterator:
                    batch = texts[j:j + batch_size]
                    emb = encoder.encode(batch)
                    all_embeddings.append(emb.cpu())

            elif images is not None:
                iterator = range(0, len(images), batch_size)
                if show_progress:
                    iterator = tqdm(iterator, desc=f"  {encoder_name}")

                for j in iterator:
                    batch = images[j:j + batch_size]
                    emb = encoder.encode(batch)
                    all_embeddings.append(emb.cpu())

            # Concatenate and save
            embeddings = torch.cat(all_embeddings, dim=0)
            cache_path = self._get_cache_path(encoder_name)

            save_safetensors({'embeddings': embeddings}, str(cache_path))

            # Update manifest
            self.cache_manifest[encoder_name] = {
                'num_samples': num_samples,
                'shape': list(embeddings.shape),
                'dtype': str(embeddings.dtype),
                'dim': config.get('dim'),
                'complete': True,
            }
            self._save_manifest()

            print(f"  ✓ Saved: {cache_path.name} {list(embeddings.shape)}")

            # CRITICAL: Unload model to free VRAM
            encoder.unload_all()
            del encoder
            torch.cuda.empty_cache()
            print(f"  ✓ Unloaded {encoder_name} from VRAM")

        print(f"\n{'='*60}")
        print(f"✓ All stages complete")
        print(f"{'='*60}\n")

        return self

    def get_cached_dataset(
        self,
        encoder_names: Optional[List[str]] = None,
    ) -> 'CachedEmbeddingDataset':
        """
        Get a dataset that yields cached embeddings only.

        No models loaded - pure tensor retrieval from disk.

        Args:
            encoder_names: Which encoders to include (None = all cached)
        """
        if encoder_names is None:
            encoder_names = list(self.cache_manifest.keys())

        return CachedEmbeddingDataset(
            cache_dir=self.cache_dir,
            encoder_names=encoder_names,
            manifest=self.cache_manifest,
        )

    def get_encoder_dim(self, encoder_name: str) -> int:
        """Get dimension of cached encoder."""
        if encoder_name in self.cache_manifest:
            return self.cache_manifest[encoder_name].get('dim', 0)
        if encoder_name in MODEL_REGISTRY:
            return MODEL_REGISTRY[encoder_name].get('dim', 0)
        return 0

    def get_combined_dim(self, encoder_names: Optional[List[str]] = None) -> int:
        """Get combined dimension of multiple encoders."""
        if encoder_names is None:
            encoder_names = list(self.cache_manifest.keys())
        return sum(self.get_encoder_dim(n) for n in encoder_names)

    @property
    def is_complete(self) -> bool:
        """Check if all stages are cached."""
        for stage in self.stages:
            encoder_name = stage['encoder_name']
            if encoder_name not in self.cache_manifest:
                return False
            if not self.cache_manifest[encoder_name].get('complete', False):
                return False
        return True

    def status(self) -> Dict:
        """Get build status."""
        return {
            'dataset_name': self.dataset_name,
            'cache_dir': str(self.cache_dir),
            'stages': len(self.stages),
            'cached': list(self.cache_manifest.keys()),
            'complete': self.is_complete,
        }


class CachedEmbeddingDataset(Dataset):
    """
    Dataset that yields pre-cached embeddings.

    No models required - loads from safetensors on disk.
    Memory-mapped for efficiency with large caches.
    """

    def __init__(
        self,
        cache_dir: Path,
        encoder_names: List[str],
        manifest: Dict[str, Dict],
        concatenate: bool = True,
        mmap: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.encoder_names = encoder_names
        self.manifest = manifest
        self.concatenate = concatenate

        # Load all cached tensors
        self.embeddings: Dict[str, Tensor] = {}

        for name in encoder_names:
            cache_path = self.cache_dir / f'{name}.safetensors'
            if not cache_path.exists():
                raise FileNotFoundError(f"Cache not found: {cache_path}")

            # Load with memory mapping for large files
            loaded = load_safetensors(str(cache_path))
            self.embeddings[name] = loaded['embeddings']

        # Verify all same length
        lengths = [len(e) for e in self.embeddings.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"Mismatched cache lengths: {dict(zip(encoder_names, lengths))}")

        self.num_samples = lengths[0] if lengths else 0

        # Pre-compute combined dimension
        self.dims = {name: self.embeddings[name].shape[-1] for name in encoder_names}
        self.combined_dim = sum(self.dims.values())

        print(f"CachedEmbeddingDataset: {self.num_samples} samples")
        print(f"  Encoders: {self.dims}")
        print(f"  Combined dim: {self.combined_dim}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Union[Tensor, Dict[str, Tensor]]:
        if self.concatenate:
            # Return concatenated tensor
            parts = [self.embeddings[name][idx] for name in self.encoder_names]
            return torch.cat(parts, dim=-1)
        else:
            # Return dict of separate tensors
            return {name: self.embeddings[name][idx] for name in self.encoder_names}

    def get_encoder(self, encoder_name: str) -> Tensor:
        """Get all embeddings for a specific encoder."""
        return self.embeddings[encoder_name]

    def get_batch(self, indices: List[int]) -> Union[Tensor, Dict[str, Tensor]]:
        """Get batch by indices."""
        if self.concatenate:
            parts = [self.embeddings[name][indices] for name in self.encoder_names]
            return torch.cat(parts, dim=-1)
        else:
            return {name: self.embeddings[name][indices] for name in self.encoder_names}


# =============================================================================
# CACHE MANAGER (for runtime use)
# =============================================================================

class CacheManager:
    """
    Runtime cache manager for yielding pre-built caches.

    No models loaded - pure cache retrieval.

    Usage:
        # After building with StagedCacheBuilder
        manager = CacheManager('./encoder_cache')

        # Get dataset for training
        dataset = manager.get_dataset('danbooru_100k')
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for batch in loader:
            # batch is pure tensors, no encoding needed
            ...
    """

    def __init__(self, cache_root: str = './encoder_cache'):
        self.cache_root = Path(cache_root)
        self.datasets: Dict[str, Dict] = {}
        self._scan_caches()

    def _scan_caches(self) -> None:
        """Scan for available cached datasets."""
        if not self.cache_root.exists():
            return

        for dataset_dir in self.cache_root.iterdir():
            if dataset_dir.is_dir():
                manifest_path = dataset_dir / 'manifest.json'
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    self.datasets[dataset_dir.name] = {
                        'path': dataset_dir,
                        'manifest': manifest,
                        'encoders': list(manifest.keys()),
                    }

    def list_datasets(self) -> List[str]:
        """List available cached datasets."""
        return list(self.datasets.keys())

    def get_info(self, dataset_name: str) -> Dict:
        """Get info about a cached dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        return self.datasets[dataset_name]

    def get_dataset(
        self,
        dataset_name: str,
        encoder_names: Optional[List[str]] = None,
        concatenate: bool = True,
    ) -> CachedEmbeddingDataset:
        """
        Get a cached dataset ready for DataLoader.

        No models loaded - pure tensor retrieval.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}. Available: {self.list_datasets()}")

        info = self.datasets[dataset_name]

        if encoder_names is None:
            encoder_names = info['encoders']

        return CachedEmbeddingDataset(
            cache_dir=info['path'],
            encoder_names=encoder_names,
            manifest=info['manifest'],
            concatenate=concatenate,
        )

    def get_dim(self, dataset_name: str, encoder_names: Optional[List[str]] = None) -> int:
        """Get combined dimension for dataset."""
        info = self.datasets[dataset_name]
        manifest = info['manifest']

        if encoder_names is None:
            encoder_names = info['encoders']

        return sum(manifest[n].get('dim', manifest[n]['shape'][-1]) for n in encoder_names)

class EmbeddingCache:
    """
    Manages cached embeddings for fast epoch recall.

    Supports:
        - Dataset namespacing (prevents cross-contamination)
        - In-memory caching
        - Disk persistence (safetensors)
        - Hash-based key generation
        - Lazy loading
    """

    def __init__(
        self,
        dataset_name: str = "default",
        cache_dir: Optional[str] = None,
        max_memory_mb: int = 4096,
    ):
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir) / dataset_name if cache_dir else None
        self.max_memory_bytes = max_memory_mb * 1024 * 1024

        # In-memory cache: hash -> tensor
        self._memory: Dict[str, Tensor] = {}
        self._memory_bytes = 0

        # Metadata
        self._hits = 0
        self._misses = 0

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def set_dataset(self, dataset_name: str) -> 'EmbeddingCache':
        """
        Switch to a different dataset namespace.

        Clears in-memory cache and updates disk path.
        """
        if dataset_name != self.dataset_name:
            self.clear()
            self.dataset_name = dataset_name
            if self.cache_dir:
                # Update path to new dataset
                self.cache_dir = self.cache_dir.parent / dataset_name
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self

    def _namespaced_key(self, key: str) -> str:
        """Add dataset namespace to key."""
        return f"{self.dataset_name}_{key}"

    @staticmethod
    def hash_key(text: str) -> str:
        """Generate hash key for text."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    @staticmethod
    def hash_tensor(tensor: Tensor) -> str:
        """Generate hash key for tensor (by shape and sample values)."""
        shape_str = str(tensor.shape)
        sample = tensor.flatten()[:8].tolist() if tensor.numel() >= 8 else tensor.flatten().tolist()
        return hashlib.md5(f"{shape_str}_{sample}".encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Tensor]:
        """Get cached embedding by key."""
        ns_key = self._namespaced_key(key)

        if ns_key in self._memory:
            self._hits += 1
            return self._memory[ns_key]

        # Try disk cache
        if self.cache_dir:
            disk_path = self.cache_dir / f"{key}.safetensors"
            if disk_path.exists():
                self._hits += 1
                tensor = load_safetensors(str(disk_path))['embedding']
                self._memory[ns_key] = tensor
                return tensor

        self._misses += 1
        return None

    def put(self, key: str, tensor: Tensor, persist: bool = False) -> None:
        """Store embedding in cache."""
        ns_key = self._namespaced_key(key)
        tensor = tensor.cpu()
        tensor_bytes = tensor.numel() * tensor.element_size()

        # Evict if needed
        while self._memory_bytes + tensor_bytes > self.max_memory_bytes and self._memory:
            evict_key = next(iter(self._memory))
            evicted = self._memory.pop(evict_key)
            self._memory_bytes -= evicted.numel() * evicted.element_size()

        self._memory[ns_key] = tensor
        self._memory_bytes += tensor_bytes

        # Persist to disk
        if persist and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            disk_path = self.cache_dir / f"{key}.safetensors"
            save_safetensors({'embedding': tensor}, str(disk_path))

    def get_batch(self, keys: List[str]) -> Tuple[Dict[str, Tensor], List[str]]:
        """
        Get batch of cached embeddings.

        Returns:
            Tuple of (found_dict, missing_keys)
        """
        found = {}
        missing = []

        for key in keys:
            cached = self.get(key)
            if cached is not None:
                found[key] = cached
            else:
                missing.append(key)

        return found, missing

    def put_batch(self, embeddings: Dict[str, Tensor], persist: bool = False) -> None:
        """Store batch of embeddings."""
        for key, tensor in embeddings.items():
            self.put(key, tensor, persist)

    def save(self, path: str) -> None:
        """Save entire cache to single file."""
        if not self._memory:
            return

        # Include dataset name in filename
        path = Path(path)
        if not str(path).endswith('.safetensors'):
            path = path.with_suffix('.safetensors')

        # Prepend dataset name if not already in path
        if self.dataset_name not in str(path):
            path = path.parent / f"{self.dataset_name}_{path.name}"

        path.parent.mkdir(parents=True, exist_ok=True)
        save_safetensors(self._memory, str(path))
        print(f"✓ Saved cache [{self.dataset_name}]: {len(self._memory)} embeddings to {path}")

    def load(self, path: str) -> None:
        """Load cache from file."""
        path = Path(path)

        # Try exact path first
        if not path.exists():
            # Try with dataset name prepended
            alt_path = path.parent / f"{self.dataset_name}_{path.name}"
            if alt_path.exists():
                path = alt_path
            else:
                print(f"⚠️  Cache not found: {path}")
                return

        loaded = load_safetensors(str(path))

        # Verify dataset match (keys should be namespaced)
        sample_key = next(iter(loaded.keys()), "")
        if sample_key and not sample_key.startswith(self.dataset_name):
            print(f"⚠️  Cache dataset mismatch: expected '{self.dataset_name}', got '{sample_key.split('_')[0]}'")
            print(f"   Loading anyway - use set_dataset() to switch namespaces")

        self._memory.update(loaded)
        self._memory_bytes = sum(t.numel() * t.element_size() for t in self._memory.values())
        print(f"✓ Loaded cache [{self.dataset_name}]: {len(loaded)} embeddings from {path}")

    def clear(self) -> None:
        """Clear in-memory cache."""
        self._memory.clear()
        self._memory_bytes = 0

    @property
    def stats(self) -> Dict:
        """Cache statistics."""
        total = self._hits + self._misses
        return {
            'dataset': self.dataset_name,
            'entries': len(self._memory),
            'memory_mb': self._memory_bytes / (1024 * 1024),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / max(1, total),
        }


# =============================================================================
# BASE ENCODER DATA COMPONENT
# =============================================================================

class EncoderDataComponent(DataComponent):
    """
    Base class for encoder-based data processing with caching.

    Extends DataComponent with:
        - Model loading (local, HuggingFace, custom weights)
        - Intelligent caching with dataset namespacing
        - Batched encoding
        - Mixed precision support
    """

    def __init__(
        self,
        name: str,
        dataset_name: str = "default",
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
        cache_max_memory_mb: int = 4096,
    ):
        super().__init__(name, target_device=device, target_dtype=dtype)

        self.device = torch.device(device)
        self.dtype = dtype
        self.cache_enabled = cache_enabled
        self.dataset_name = dataset_name

        # Initialize cache with dataset namespace
        self.cache = EmbeddingCache(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            max_memory_mb=cache_max_memory_mb,
        ) if cache_enabled else None

        # Model storage
        self.models: Dict[str, nn.Module] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {}

    def set_dataset(self, dataset_name: str) -> 'EncoderDataComponent':
        """
        Switch to a different dataset namespace.

        Prevents cache cross-contamination between datasets.
        """
        self.dataset_name = dataset_name
        if self.cache:
            self.cache.set_dataset(dataset_name)
        return self

    def load_model(
        self,
        encoder_name: str,
        local_path: Optional[str] = None,
        hf_repo: Optional[str] = None,
        weights_path: Optional[str] = None,
        config_override: Optional[Dict] = None,
    ) -> nn.Module:
        """
        Load an encoder model.

        Priority:
            1. local_path (direct file)
            2. hf_repo (HuggingFace model)
            3. Registry default

        Args:
            encoder_name: Name from registry or custom name
            local_path: Path to local model/weights
            hf_repo: HuggingFace repo ID
            weights_path: Path to custom weights (safetensors)
            config_override: Override registry config

        Returns:
            Loaded model
        """
        # Get config from registry or use override
        if encoder_name in MODEL_REGISTRY:
            config = MODEL_REGISTRY[encoder_name].copy()
            if config_override:
                config.update(config_override)
        else:
            config = config_override or {}

        self.model_configs[encoder_name] = config

        hf_path = hf_repo or config.get('hf_path')

        print(f"  Loading {encoder_name}...")

        # Load model
        if local_path and Path(local_path).exists():
            # Load from local path
            if local_path.endswith('.safetensors'):
                # Load weights into base model
                model_class = config.get('model_class', AutoModel)
                model = model_class.from_pretrained(hf_path)
                state_dict = load_safetensors(local_path)
                model.load_state_dict(state_dict, strict=False)
            else:
                model = AutoModel.from_pretrained(local_path)
        elif hf_path:
            # Load from HuggingFace
            model_class = config.get('model_class', AutoModel)
            model = model_class.from_pretrained(hf_path)

            # Apply custom weights if specified
            if weights_path or config.get('weights_repo'):
                weights_file = weights_path
                if not weights_file and config.get('weights_repo'):
                    weights_file = hf_hub_download(
                        repo_id=config['weights_repo'],
                        filename=config.get('weights_file', f'{encoder_name}.safetensors'),
                        repo_type='model',
                    )

                if weights_file:
                    state_dict = load_safetensors(weights_file)
                    # Try direct load, then with remapping
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    if unexpected and not missing:
                        # Try key remapping
                        remapped = {}
                        for k, v in state_dict.items():
                            if k.startswith('text_model.') or k.startswith('vision_model.'):
                                remapped[k] = v
                            else:
                                remapped[f'text_model.{k}'] = v
                        model.load_state_dict(remapped, strict=False)
        else:
            raise ValueError(f"No path specified for {encoder_name}")

        # Load tokenizer/processor
        tokenizer_class = config.get('tokenizer_class')
        if tokenizer_class:
            self.tokenizers[encoder_name] = tokenizer_class.from_pretrained(hf_path)
        else:
            try:
                self.tokenizers[encoder_name] = AutoTokenizer.from_pretrained(hf_path)
            except:
                try:
                    self.processors[encoder_name] = AutoProcessor.from_pretrained(hf_path)
                except:
                    pass

        model.to(self.device).eval()
        self.models[encoder_name] = model

        params = sum(p.numel() for p in model.parameters())
        dim = config.get('dim', 'unknown')
        print(f"  ✓ {encoder_name}: {params:,} params, dim={dim}")

        return model

    def unload_model(self, encoder_name: str) -> None:
        """Unload a model to free memory."""
        if encoder_name in self.models:
            del self.models[encoder_name]
            torch.cuda.empty_cache()

    def unload_all(self) -> None:
        """Unload all models."""
        self.models.clear()
        self.tokenizers.clear()
        self.processors.clear()
        torch.cuda.empty_cache()

    @abstractmethod
    def encode(self, inputs: Any) -> Tensor:
        """Encode inputs. Override in subclasses."""
        pass

    def forward(self, inputs: Any) -> Tensor:
        """Forward pass - alias for encode."""
        return self.encode(inputs)

    def __call__(self, inputs: Any) -> Tensor:
        """Callable interface."""
        return self.encode(inputs)

    def save_cache(self, path: str) -> None:
        """Save cache to disk."""
        if self.cache:
            self.cache.save(path)

    def load_cache(self, path: str) -> None:
        """Load cache from disk."""
        if self.cache:
            self.cache.load(path)

    def clear_cache(self) -> None:
        """Clear cache."""
        if self.cache:
            self.cache.clear()

    @property
    def cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.stats if self.cache else {}


# =============================================================================
# MULTI TEXT ENCODER
# =============================================================================

class MultiTextEncode(EncoderDataComponent):
    """
    Multiple text encoders with optional caching.

    Encodes text with multiple models and optionally concatenates outputs.
    Supports CLIP, T5, Qwen, LLaMA, and custom models.

    Usage:
        enc = MultiTextEncode(
            encoders=['clip_l_illustrious', 'clip_g_illustrious'],
            dataset_name='danbooru_tags',
            device='cuda',
            concatenate=True,  # -> [B, seq, 768+1280]
        )
        embeddings = enc.encode_batch(texts)
    """

    def __init__(
        self,
        encoders: List[str],
        dataset_name: str = "default",
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
        concatenate: bool = True,
        clip_skip: int = 1,
        max_length: Optional[int] = None,
    ):
        super().__init__(
            name='multi_text_encode',
            dataset_name=dataset_name,
            device=device,
            dtype=dtype,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )

        self.encoder_names = encoders
        self.concatenate = concatenate
        self.clip_skip = clip_skip
        self.max_length = max_length

        # Load all encoders
        print(f"\n{'='*60}")
        print(f"Initializing MultiTextEncode")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Encoders: {encoders}")

        for enc_name in encoders:
            self.load_model(enc_name)

        # Calculate combined dimension
        self.encoder_dims = {
            name: self.model_configs[name].get('dim', 768)
            for name in encoders
        }
        self.combined_dim = sum(self.encoder_dims.values()) if concatenate else None

        print(f"\nDimensions: {self.encoder_dims}")
        if concatenate:
            print(f"Combined: {self.combined_dim}")
        print(f"{'='*60}\n")

    def _get_hidden_state(
        self,
        model_output,
        clip_skip: int = 1,
    ) -> Tensor:
        """Extract hidden state with clip_skip support."""
        if clip_skip == 1:
            return model_output.last_hidden_state

        if hasattr(model_output, 'hidden_states') and model_output.hidden_states:
            return model_output.hidden_states[-clip_skip]

        return model_output.last_hidden_state

    @torch.no_grad()
    def encode_single(
        self,
        encoder_name: str,
        text: Union[str, List[str]],
    ) -> Tensor:
        """
        Encode text with a single encoder.

        Args:
            encoder_name: Which encoder to use
            text: Single string or list of strings

        Returns:
            Tensor [B, seq_len, dim]
        """
        if isinstance(text, str):
            text = [text]

        model = self.models[encoder_name]
        config = self.model_configs[encoder_name]
        max_len = self.max_length or config.get('max_length', 77)

        # Get tokenizer
        tokenizer = self.tokenizers.get(encoder_name)
        if tokenizer is None:
            raise ValueError(f"No tokenizer for {encoder_name}")

        # Tokenize
        tokens = tokenizer(
            text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).to(self.device)

        # Encode
        output = model(
            **tokens,
            output_hidden_states=(self.clip_skip > 1),
        )

        return self._get_hidden_state(output, self.clip_skip)

    @torch.no_grad()
    def encode(self, text: Union[str, List[str]]) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Encode text with all encoders.

        Args:
            text: Single string or list of strings

        Returns:
            If concatenate=True: Tensor [B, seq_len, combined_dim]
            If concatenate=False: Dict[encoder_name, Tensor]
        """
        embeddings = {}

        for enc_name in self.encoder_names:
            embeddings[enc_name] = self.encode_single(enc_name, text)

        if self.concatenate:
            return torch.cat(list(embeddings.values()), dim=-1)

        return embeddings

    @torch.no_grad()
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Encode batch of texts with caching.

        Args:
            texts: List of strings
            batch_size: Batch size for encoding
            use_cache: Whether to use caching
            show_progress: Show progress bar

        Returns:
            Tensor [N, seq_len, combined_dim]
        """
        all_embeddings = []

        # Check cache
        if use_cache and self.cache:
            cache_keys = [self.cache.hash_key(t) for t in texts]
            cached, missing_keys = self.cache.get_batch(cache_keys)

            if len(cached) == len(texts):
                # All cached
                return torch.stack([cached[k] for k in cache_keys])

            # Find texts that need encoding
            missing_indices = [i for i, k in enumerate(cache_keys) if k not in cached]
            texts_to_encode = [texts[i] for i in missing_indices]
        else:
            missing_indices = list(range(len(texts)))
            texts_to_encode = texts
            cached = {}
            cache_keys = [self.cache.hash_key(t) for t in texts] if self.cache else []

        # Encode missing texts
        if texts_to_encode:
            iterator = range(0, len(texts_to_encode), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Encoding")

            new_embeddings = []
            for i in iterator:
                batch = texts_to_encode[i:i + batch_size]
                emb = self.encode(batch)
                new_embeddings.append(emb.cpu())

            new_embeddings = torch.cat(new_embeddings, dim=0)

            # Cache new embeddings
            if use_cache and self.cache:
                for i, idx in enumerate(missing_indices):
                    self.cache.put(cache_keys[idx], new_embeddings[i])

        # Combine cached and new
        if use_cache and self.cache and cached:
            result = []
            new_idx = 0
            for i, key in enumerate(cache_keys):
                if key in cached:
                    result.append(cached[key])
                else:
                    result.append(new_embeddings[new_idx])
                    new_idx += 1
            return torch.stack(result)
        else:
            return new_embeddings if texts_to_encode else torch.stack(list(cached.values()))

    @property
    def output_dim(self) -> int:
        """Combined output dimension."""
        return self.combined_dim or list(self.encoder_dims.values())[0]


# =============================================================================
# MULTI VISION ENCODER
# =============================================================================

class MultiVisionEncode(EncoderDataComponent):
    """
    Multiple vision encoders with optional caching.

    Encodes images with multiple models (DINO, CLIP vision, etc.)
    and optionally concatenates outputs.

    Usage:
        enc = MultiVisionEncode(
            encoders=['dinov2_base', 'dinov3_small'],
            dataset_name='cifar100',
            device='cuda',
            concatenate=True,
        )
        features = enc.encode_batch(images)
    """

    def __init__(
        self,
        encoders: List[str],
        dataset_name: str = "default",
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
        concatenate: bool = True,
        image_size: int = 224,
        pool_output: bool = True,  # Pool to [B, D] or keep [B, N, D]
    ):
        super().__init__(
            name='multi_vision_encode',
            dataset_name=dataset_name,
            device=device,
            dtype=dtype,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )

        self.encoder_names = encoders
        self.concatenate = concatenate
        self.image_size = image_size
        self.pool_output = pool_output

        # Load all encoders
        print(f"\n{'='*60}")
        print(f"Initializing MultiVisionEncode")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Encoders: {encoders}")

        for enc_name in encoders:
            self.load_model(enc_name)

        # Calculate combined dimension
        self.encoder_dims = {
            name: self.model_configs[name].get('dim', 768)
            for name in encoders
        }
        self.combined_dim = sum(self.encoder_dims.values()) if concatenate else None

        print(f"\nDimensions: {self.encoder_dims}")
        if concatenate:
            print(f"Combined: {self.combined_dim}")
        print(f"{'='*60}\n")

    def _preprocess_images(self, images: Tensor) -> Tensor:
        """Preprocess images for encoding."""
        # Ensure [B, 3, H, W]
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Resize if needed
        if images.shape[-1] != self.image_size:
            images = F.interpolate(
                images,
                size=self.image_size,
                mode='bilinear',
                align_corners=False,
            )

        return images.to(self.device, self.dtype)

    @torch.no_grad()
    def encode_single(
        self,
        encoder_name: str,
        images: Tensor,
    ) -> Tensor:
        """
        Encode images with a single encoder.

        Args:
            encoder_name: Which encoder to use
            images: Tensor [B, 3, H, W]

        Returns:
            Tensor [B, dim] if pool_output else [B, N, dim]
        """
        model = self.models[encoder_name]
        images = self._preprocess_images(images)

        # Get processor if available
        processor = self.processors.get(encoder_name)
        if processor:
            # Some models need special preprocessing
            inputs = processor(images=images, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output = model(**inputs)
        else:
            output = model(pixel_values=images)

        # Extract features
        if hasattr(output, 'pooler_output') and self.pool_output:
            return output.pooler_output
        elif hasattr(output, 'last_hidden_state'):
            features = output.last_hidden_state
            if self.pool_output:
                return features.mean(dim=1)
            return features
        else:
            return output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def encode(self, images: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Encode images with all encoders.

        Args:
            images: Tensor [B, 3, H, W]

        Returns:
            If concatenate=True: Tensor [B, combined_dim]
            If concatenate=False: Dict[encoder_name, Tensor]
        """
        embeddings = {}

        for enc_name in self.encoder_names:
            embeddings[enc_name] = self.encode_single(enc_name, images)

        if self.concatenate:
            return torch.cat(list(embeddings.values()), dim=-1)

        return embeddings

    @torch.no_grad()
    def encode_batch(
        self,
        images: Tensor,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Encode batch of images.

        Args:
            images: Tensor [N, 3, H, W]
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Tensor [N, combined_dim]
        """
        all_embeddings = []

        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding images")

        for i in iterator:
            batch = images[i:i + batch_size]
            emb = self.encode(batch)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        cache_path: Optional[str] = None,
    ) -> Tensor:
        """
        Encode entire dataset with caching.

        Args:
            dataset: Dataset returning (image, ...) tuples
            batch_size: Batch size
            cache_path: Path to save/load cache (dataset name auto-prepended)

        Returns:
            Tensor [N, combined_dim]
        """
        # Build cache path with dataset name
        if cache_path:
            cache_path = Path(cache_path)
            if self.dataset_name not in str(cache_path):
                cache_path = cache_path.parent / f"{self.dataset_name}_{cache_path.name}"
            cache_path = str(cache_path)

        # Try to load from cache
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached features [{self.dataset_name}] from {cache_path}")
            return torch.load(cache_path, weights_only=True)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        all_features = []
        for batch in tqdm(loader, desc=f"Caching [{self.dataset_name}] vision features"):
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            features = self.encode(images.to(self.device))
            all_features.append(features.cpu())

        features = torch.cat(all_features, dim=0)

        # Save cache
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(features, cache_path)
            print(f"✓ Cached {len(features)} features [{self.dataset_name}] to {cache_path}")

        return features

    @property
    def output_dim(self) -> int:
        """Combined output dimension."""
        return self.combined_dim or list(self.encoder_dims.values())[0]


# =============================================================================
# MULTI ENCODE (COMBINED TEXT + VISION)
# =============================================================================

class MultiEncode(EncoderDataComponent):
    """
    Efficient combination of text and vision encoders.

    Manages both MultiTextEncode and MultiVisionEncode with
    shared caching and device management.

    Usage:
        enc = MultiEncode(
            text_encoders=['clip_l_illustrious', 'clip_g_illustrious'],
            vision_encoders=['dinov2_base'],
            dataset_name='my_dataset',
            device='cuda',
        )

        # Encode both modalities
        text_emb, vision_emb = enc.encode(texts, images)

        # Or encode separately
        text_emb = enc.encode_text(texts)
        vision_emb = enc.encode_vision(images)
    """

    def __init__(
        self,
        text_encoders: Optional[List[str]] = None,
        vision_encoders: Optional[List[str]] = None,
        dataset_name: str = "default",
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
        text_concatenate: bool = True,
        vision_concatenate: bool = True,
        clip_skip: int = 2,
        image_size: int = 224,
        pool_vision: bool = True,
    ):
        super().__init__(
            name='multi_encode',
            dataset_name=dataset_name,
            device=device,
            dtype=dtype,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )

        self.text_encoder: Optional[MultiTextEncode] = None
        self.vision_encoder: Optional[MultiVisionEncode] = None

        # Initialize text encoders
        if text_encoders:
            self.text_encoder = MultiTextEncode(
                encoders=text_encoders,
                dataset_name=dataset_name,
                device=device,
                dtype=dtype,
                cache_enabled=cache_enabled,
                cache_dir=f"{cache_dir}/text" if cache_dir else None,
                concatenate=text_concatenate,
                clip_skip=clip_skip,
            )

        # Initialize vision encoders
        if vision_encoders:
            self.vision_encoder = MultiVisionEncode(
                encoders=vision_encoders,
                dataset_name=dataset_name,
                device=device,
                dtype=dtype,
                cache_enabled=cache_enabled,
                cache_dir=f"{cache_dir}/vision" if cache_dir else None,
                concatenate=vision_concatenate,
                image_size=image_size,
                pool_output=pool_vision,
            )

        print(f"\n{'='*60}")
        print(f"MultiEncode Ready")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name}")
        if self.text_encoder:
            print(f"Text: {text_encoders} -> dim={self.text_encoder.output_dim}")
        if self.vision_encoder:
            print(f"Vision: {vision_encoders} -> dim={self.vision_encoder.output_dim}")
        print(f"{'='*60}\n")

    def set_dataset(self, dataset_name: str) -> 'MultiEncode':
        """Switch dataset namespace for all encoders."""
        super().set_dataset(dataset_name)
        if self.text_encoder:
            self.text_encoder.set_dataset(dataset_name)
        if self.vision_encoder:
            self.vision_encoder.set_dataset(dataset_name)
        return self

    def encode_text(
        self,
        texts: Union[str, List[str]],
        use_cache: bool = True,
    ) -> Tensor:
        """Encode text inputs."""
        if self.text_encoder is None:
            raise ValueError("No text encoders configured")

        if isinstance(texts, list) and len(texts) > 1:
            return self.text_encoder.encode_batch(texts, use_cache=use_cache)
        return self.text_encoder.encode(texts)

    def encode_vision(
        self,
        images: Tensor,
        use_cache: bool = True,
    ) -> Tensor:
        """Encode vision inputs."""
        if self.vision_encoder is None:
            raise ValueError("No vision encoders configured")

        if images.shape[0] > 1:
            return self.vision_encoder.encode_batch(images)
        return self.vision_encoder.encode(images)

    def encode(
        self,
        texts: Optional[Union[str, List[str]]] = None,
        images: Optional[Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Encode both text and vision inputs.

        Args:
            texts: Text inputs (optional)
            images: Image inputs (optional)
            use_cache: Use caching

        Returns:
            Tuple of (text_embeddings, vision_embeddings)
        """
        text_emb = None
        vision_emb = None

        if texts is not None and self.text_encoder:
            text_emb = self.encode_text(texts, use_cache)

        if images is not None and self.vision_encoder:
            vision_emb = self.encode_vision(images, use_cache)

        return text_emb, vision_emb

    def forward(
        self,
        texts: Optional[Union[str, List[str]]] = None,
        images: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Forward pass."""
        return self.encode(texts, images)

    def save_cache(self, path: str) -> None:
        """Save all caches."""
        base = Path(path)
        if self.text_encoder:
            self.text_encoder.save_cache(str(base / 'text_cache.safetensors'))
        if self.vision_encoder:
            self.vision_encoder.save_cache(str(base / 'vision_cache.safetensors'))

    def load_cache(self, path: str) -> None:
        """Load all caches."""
        base = Path(path)
        if self.text_encoder and (base / 'text_cache.safetensors').exists():
            self.text_encoder.load_cache(str(base / 'text_cache.safetensors'))
        if self.vision_encoder and (base / 'vision_cache.safetensors').exists():
            self.vision_encoder.load_cache(str(base / 'vision_cache.safetensors'))

    @property
    def text_dim(self) -> Optional[int]:
        """Text encoder output dimension."""
        return self.text_encoder.output_dim if self.text_encoder else None

    @property
    def vision_dim(self) -> Optional[int]:
        """Vision encoder output dimension."""
        return self.vision_encoder.output_dim if self.vision_encoder else None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_illustrious_text_encoder(
    dataset_name: str = "default",
    device: str = 'cuda',
    clip_skip: int = 2,
    cache_enabled: bool = True,
    cache_dir: Optional[str] = None,
) -> MultiTextEncode:
    """Create Illustrious CLIP-L + CLIP-G text encoder."""
    return MultiTextEncode(
        encoders=['clip_l_illustrious', 'clip_g_illustrious'],
        dataset_name=dataset_name,
        device=device,
        concatenate=True,
        clip_skip=clip_skip,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
    )


def create_dino_vision_encoder(
    variant: str = 'dinov2_base',
    dataset_name: str = "default",
    device: str = 'cuda',
    cache_enabled: bool = True,
    cache_dir: Optional[str] = None,
) -> MultiVisionEncode:
    """Create DINO vision encoder."""
    return MultiVisionEncode(
        encoders=[variant],
        dataset_name=dataset_name,
        device=device,
        concatenate=True,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
    )


def create_multimodal_encoder(
    text_encoders: List[str] = ['clip_l_illustrious', 'clip_g_illustrious'],
    vision_encoders: List[str] = ['dinov2_base'],
    dataset_name: str = "default",
    device: str = 'cuda',
    cache_enabled: bool = True,
    cache_dir: str = './encoder_cache',
) -> MultiEncode:
    """Create full multimodal encoder."""
    return MultiEncode(
        text_encoders=text_encoders,
        vision_encoders=vision_encoders,
        dataset_name=dataset_name,
        device=device,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
    )


def build_staged_cache(
    dataset_name: str,
    texts: Optional[List[str]] = None,
    images: Optional[Tensor] = None,
    text_encoders: List[str] = ['clip_l_illustrious', 'clip_g_illustrious'],
    vision_encoders: List[str] = ['dinov2_base'],
    cache_dir: str = './encoder_cache',
    device: str = 'cuda',
    batch_size: int = 32,
    clip_skip: int = 2,
) -> CachedEmbeddingDataset:
    """
    Build staged cache and return dataset ready for training.

    Sequential workflow:
        1. For each text encoder: load → encode → cache → unload
        2. For each vision encoder: load → encode → cache → unload
        3. Return dataset yielding cached tensors only

    Example:
        # Prep 10k text embeddings with CLIP-L + CLIP-G (sequential, VRAM efficient)
        dataset = build_staged_cache(
            dataset_name='my_prompts',
            texts=all_prompts,
            text_encoders=['clip_l_illustrious', 'clip_g_illustrious'],
            vision_encoders=[],  # No vision
        )

        # Now train with no encoder models in memory
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        for embeddings in loader:
            # embeddings: [B, 77, 2048] - pure cached tensors
            model(embeddings)
    """
    builder = StagedCacheBuilder(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        device=device,
    )

    # Add text stages
    if texts is not None:
        for enc_name in text_encoders:
            builder.add_stage(
                encoder_name=enc_name,
                texts=texts,
                batch_size=batch_size,
                clip_skip=clip_skip,
            )

    # Add vision stages
    if images is not None:
        for enc_name in vision_encoders:
            builder.add_stage(
                encoder_name=enc_name,
                images=images,
                batch_size=batch_size,
            )

    # Build all stages (models loaded/unloaded sequentially)
    builder.build()

    # Return dataset (no models in memory)
    return builder.get_cached_dataset()


def load_cached_dataset(
    dataset_name: str,
    cache_dir: str = './encoder_cache',
    encoder_names: Optional[List[str]] = None,
    concatenate: bool = True,
) -> CachedEmbeddingDataset:
    """
    Load a pre-built cached dataset.

    No models loaded - immediate tensor access.

    Example:
        # Load previously cached embeddings
        dataset = load_cached_dataset('danbooru_100k')
        print(f"Samples: {len(dataset)}, Dim: {dataset.combined_dim}")

        loader = DataLoader(dataset, batch_size=64)
        for batch in loader:
            # Pure tensors, no encoding
            ...
    """
    manager = CacheManager(cache_dir)
    return manager.get_dataset(dataset_name, encoder_names, concatenate)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing Encoder Data Components\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test EmbeddingCache
    print("=" * 60)
    print("Testing EmbeddingCache with Dataset Namespacing")
    print("=" * 60)

    cache = EmbeddingCache(dataset_name='test_dataset', max_memory_mb=100)

    # Put and get
    key1 = cache.hash_key("test text 1")
    tensor1 = torch.randn(77, 768)
    cache.put(key1, tensor1)

    retrieved = cache.get(key1)
    print(f"Cache put/get: {retrieved is not None}")
    print(f"Stats: {cache.stats}")

    # Test dataset switching
    cache.set_dataset('other_dataset')
    retrieved2 = cache.get(key1)  # Should miss - different namespace
    print(f"After dataset switch, same key found: {retrieved2 is not None}")

    # Test StagedCacheBuilder structure
    print("\n" + "=" * 60)
    print("StagedCacheBuilder Structure")
    print("=" * 60)

    print("""
STAGED CACHING WORKFLOW:
========================

# 1. Build cache (models loaded/unloaded sequentially)
builder = StagedCacheBuilder(
    dataset_name='danbooru_100k',
    cache_dir='./cache',
)

# Add stages - each model loaded, encodes, saves, unloads
builder.add_stage('clip_l_illustrious', texts=all_texts)
builder.add_stage('clip_g_illustrious', texts=all_texts)
builder.add_stage('dinov2_large', images=all_images)

# Execute (VRAM efficient - one model at a time)
builder.build()

# 2. Training (NO MODELS IN MEMORY)
dataset = builder.get_cached_dataset()
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for batch in loader:
    # batch: [B, combined_dim] - pure cached tensors
    model(batch)  # Your model, not encoders


# OR use convenience function:
dataset = build_staged_cache(
    dataset_name='my_prompts',
    texts=prompts_8500,
    text_encoders=['clip_l_illustrious', 'clip_g_illustrious'],
)

# Later, load existing cache (no rebuild):
dataset = load_cached_dataset('my_prompts')
""")

    # Test CacheManager structure
    print("=" * 60)
    print("CacheManager (Runtime Cache Retrieval)")
    print("=" * 60)

    print("""
RUNTIME USAGE:
==============

manager = CacheManager('./encoder_cache')

# List available datasets
print(manager.list_datasets())
# ['danbooru_100k', 'gelbooru_50k', 'cifar100']

# Get dataset info
info = manager.get_info('danbooru_100k')
# {'encoders': ['clip_l_illustrious', 'clip_g_illustrious'], ...}

# Get dataset for training
dataset = manager.get_dataset('danbooru_100k')
# CachedEmbeddingDataset: 100000 samples, dim=2048

# No models loaded - pure tensor retrieval from safetensors
loader = DataLoader(dataset, batch_size=128, num_workers=4)
""")

    # Test MODEL_REGISTRY
    print("=" * 60)
    print(f"MODEL_REGISTRY: {len(MODEL_REGISTRY)} models")
    print("=" * 60)

    text_models = [k for k, v in MODEL_REGISTRY.items() if v['type'] == 'text']
    vision_models = [k for k, v in MODEL_REGISTRY.items() if v['type'] == 'vision']

    print(f"\nText encoders ({len(text_models)}):")
    for name in sorted(text_models)[:10]:
        dim = MODEL_REGISTRY[name].get('dim', '?')
        print(f"  {name}: {dim}d")
    print(f"  ... and {len(text_models) - 10} more")

    print(f"\nVision encoders ({len(vision_models)}):")
    for name in sorted(vision_models)[:10]:
        dim = MODEL_REGISTRY[name].get('dim', '?')
        print(f"  {name}: {dim}d")
    print(f"  ... and {len(vision_models) - 10} more")

    print("\n" + "=" * 60)
    print("✓ All structure tests passed!")
    print("=" * 60)

    print("""
TO TEST WITH ACTUAL MODELS:
===========================

# Quick test
texts = ['1girl, blue hair', '1boy, armor', 'landscape, mountains']

# Build staged cache
dataset = build_staged_cache(
    dataset_name='test_3_samples',
    texts=texts,
    text_encoders=['clip_l', 'clip_g'],  # Or illustrious variants
)

# Check output
print(f"Samples: {len(dataset)}")
print(f"Combined dim: {dataset.combined_dim}")
print(f"Sample shape: {dataset[0].shape}")
""")