# geofractal/trainers/vae_lyra/lyra_xl_multimodal.py

"""
Trainer for VAE Lyra - Multi-Modal Variational Autoencoder
Using Custom Illustrious CLIP-L + CLIP-G + T5-XL for SDXL Compatibility

Supports:
- BooruSynthesizer for anime/illustration style prompts
- SynthesisSystem for symbolic prompts
- LAION flavors fallback
- clip_skip for penultimate layer extraction
- Qwen summarizer for T5 natural language input
- Prompt caching (save/load generated prompts + summaries)
- Checkpoint pushing to HF Hub with safetensors export
- Proper resume with optimizer state control

Downloads custom CLIP weights from AbstractPhil/clips:
- IllustriousXL20_v20_clip_l.safetensors
- IllustriousXL20_v20_clip_g.safetensors

Install via:
    !pip install git+https://github.com/AbstractEyes/geofractal.git
    !pip install safetensors

Usage:
    from lyra_xl_multimodal import VAELyraTrainer, VAELyraTrainerConfig

    # Train with caching (auto-loads if cache exists)
    trainer = create_lyra_trainer(num_samples=10000)
    dataloader = trainer.prepare_data()
    trainer.train(dataloader)

    # Pre-generate prompts only (for separate machines/sessions)
    from lyra_xl_multimodal import generate_prompt_cache
    cache_path = generate_prompt_cache(num_samples=50000)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPTextConfig,
    T5EncoderModel,
    T5Tokenizer
)
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from huggingface_hub import HfApi, hf_hub_download, create_repo
from tqdm.auto import tqdm
import wandb
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import requests
import random
from collections import Counter

from geofractal.model.vae.vae_lyra_v2 import (
    MultiModalVAE,
    MultiModalVAEConfig,
    MultiModalVAELoss,
    FusionStrategy
)
from geovocab2.data.prompt.symbolic_tree import SynthesisSystem
from geovocab2.data.prompt.booru_synthesizer import (
    BooruSynthesizer,
    BooruConfig,
    ConduitConfig,
    sample_coherent_gender,
    generate_people_count_prefix
)
from geovocab2.shapes.factory.summary_factory import CaptionFactory, CaptionFactoryConfig

# ============================================================================
# CONSTANTS
# ============================================================================

# Pilcrow separator for T5 mode-switching
# This rare token signals the transition from tags to natural language summary
# CLIP encoders won't see this - only T5 receives the summary portion
SUMMARY_SEPARATOR = "¶"


# ============================================================================
# PROMPT SOURCE ENUM
# ============================================================================

class PromptSource(Enum):
    """Available prompt generation sources."""
    BOORU = "booru"
    SYNTHETIC = "synthetic"
    LAION = "laion"
    MIXED = "mixed"


# ============================================================================
# CUSTOM CLIP LOADING WITH CLIP_SKIP SUPPORT
# ============================================================================

def get_clip_hidden_state(
        model_output,
        clip_skip: int = 1,
        output_hidden_states: bool = True
) -> torch.Tensor:
    """
    Extract hidden state with clip_skip support.

    Args:
        model_output: Output from CLIP model
        clip_skip: Number of layers to skip from the end (1 = last layer, 2 = penultimate)
        output_hidden_states: Whether hidden_states are available

    Returns:
        Hidden state tensor [batch, seq_len, hidden_dim]
    """
    if clip_skip == 1 or not output_hidden_states:
        return model_output.last_hidden_state

    # hidden_states is tuple of (embedding, layer1, layer2, ..., layerN)
    # For clip_skip=2, we want hidden_states[-2]
    if hasattr(model_output, 'hidden_states') and model_output.hidden_states is not None:
        return model_output.hidden_states[-clip_skip]

    # Fallback to last_hidden_state
    return model_output.last_hidden_state


def load_illustrious_clip_l(
        safetensors_path: str,
        device: str = 'cuda',
        base_model: str = "openai/clip-vit-large-patch14"
) -> Tuple[CLIPTextModel, CLIPTokenizer]:
    """
    Load CLIP-L text encoder with custom Illustrious weights.

    Args:
        safetensors_path: Path to IllustriousXL20_v20_clip_l.safetensors
        device: Target device
        base_model: Base model for architecture and tokenizer

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"    Loading CLIP-L architecture from {base_model}...")

    tokenizer = CLIPTokenizer.from_pretrained(base_model)
    model = CLIPTextModel.from_pretrained(base_model)

    print(f"    Loading Illustrious weights from {Path(safetensors_path).name}...")
    state_dict = load_safetensors(safetensors_path)

    sample_keys = list(state_dict.keys())[:5]
    print(f"    Sample keys: {sample_keys}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"    ⚠️  Missing keys: {len(missing)} (may be expected)")
    if unexpected:
        print(f"    ⚠️  Unexpected keys: {len(unexpected)}")

        if unexpected and not missing:
            print("    Attempting key remapping...")
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith('text_model.'):
                    remapped[k] = v
                else:
                    remapped[f'text_model.{k}'] = v

            missing2, unexpected2 = model.load_state_dict(remapped, strict=False)
            if len(missing2) < len(missing) or len(unexpected2) < len(unexpected):
                print(f"    ✓ Remapping improved: {len(missing2)} missing, {len(unexpected2)} unexpected")

    model.to(device).eval()
    print(f"    ✓ CLIP-L loaded: {sum(p.numel() for p in model.parameters()):,} params")

    return model, tokenizer


def load_illustrious_clip_g(
        safetensors_path: str,
        device: str = 'cuda',
        base_model: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
) -> Tuple[CLIPTextModelWithProjection, CLIPTokenizer]:
    """
    Load CLIP-G text encoder with custom Illustrious weights.

    Args:
        safetensors_path: Path to IllustriousXL20_v20_clip_g.safetensors
        device: Target device
        base_model: Base model for architecture and tokenizer

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"    Loading CLIP-G architecture from {base_model}...")

    tokenizer = CLIPTokenizer.from_pretrained(base_model)
    model = CLIPTextModelWithProjection.from_pretrained(base_model)

    print(f"    Loading Illustrious weights from {Path(safetensors_path).name}...")
    state_dict = load_safetensors(safetensors_path)

    sample_keys = list(state_dict.keys())[:5]
    print(f"    Sample keys: {sample_keys}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"    ⚠️  Missing keys: {len(missing)} (may be expected)")
    if unexpected:
        print(f"    ⚠️  Unexpected keys: {len(unexpected)}")

        if unexpected and not missing:
            print("    Attempting key remapping...")
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith('text_model.'):
                    remapped[k] = v
                else:
                    remapped[f'text_model.{k}'] = v

            missing2, unexpected2 = model.load_state_dict(remapped, strict=False)
            if len(missing2) < len(missing) or len(unexpected2) < len(unexpected):
                print(f"    ✓ Remapping improved: {len(missing2)} missing, {len(unexpected2)} unexpected")

    model.to(device).eval()
    print(f"    ✓ CLIP-G loaded: {sum(p.numel() for p in model.parameters()):,} params")

    return model, tokenizer


def list_clip_files(repo_id: str = "AbstractPhil/clips") -> Dict[str, List[str]]:
    """
    List available CLIP files in a HuggingFace repo.

    Returns:
        Dict with 'clip_l', 'clip_g', 't5' keys mapping to available filenames
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")

    result = {'clip_l': [], 'clip_g': [], 't5': [], 'other': []}

    for f in files:
        f_lower = f.lower()
        if f.endswith('.safetensors') or f.endswith('.bin'):
            if 'clip_l' in f_lower or 'clipl' in f_lower or 'clip-l' in f_lower:
                result['clip_l'].append(f)
            elif 'clip_g' in f_lower or 'clipg' in f_lower or 'clip-g' in f_lower:
                result['clip_g'].append(f)
            elif 't5' in f_lower:
                result['t5'].append(f)
            else:
                result['other'].append(f)

    return result


def download_illustrious_clips(
        repo_id: str = "AbstractPhil/clips",
        clip_l_filename: Optional[str] = None,
        clip_g_filename: Optional[str] = None,
        auto_discover: bool = True
) -> Tuple[str, str]:
    """
    Download Illustrious CLIP weights from HuggingFace.

    Args:
        repo_id: HuggingFace repo ID
        clip_l_filename: Specific filename for CLIP-L (or None to auto-discover)
        clip_g_filename: Specific filename for CLIP-G (or None to auto-discover)
        auto_discover: If True and filenames not provided, discover from repo

    Returns:
        Tuple of (clip_l_path, clip_g_path)
    """
    print(f"\n📥 Downloading CLIP weights from {repo_id}...")

    api = HfApi()

    # Auto-discover files if not specified
    if auto_discover and (clip_l_filename is None or clip_g_filename is None):
        print("  🔍 Discovering available files...")
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")

        # Print available safetensors files
        safetensor_files = [f for f in files if f.endswith('.safetensors')]
        print(f"  📁 Found {len(safetensor_files)} safetensors files:")
        for f in safetensor_files[:10]:
            print(f"      - {f}")
        if len(safetensor_files) > 10:
            print(f"      ... and {len(safetensor_files) - 10} more")

        # Try to find CLIP-L and CLIP-G
        if clip_l_filename is None:
            for f in files:
                f_lower = f.lower()
                if ('clip_l' in f_lower or 'clipl' in f_lower or 'clip-l' in f_lower) and f.endswith('.safetensors'):
                    clip_l_filename = f
                    break
            if clip_l_filename is None:
                for f in files:
                    if 'illustrious' in f.lower() and 'clip_l' in f.lower():
                        clip_l_filename = f
                        break

        if clip_g_filename is None:
            for f in files:
                f_lower = f.lower()
                if ('clip_g' in f_lower or 'clipg' in f_lower or 'clip-g' in f_lower) and f.endswith('.safetensors'):
                    clip_g_filename = f
                    break
            if clip_g_filename is None:
                for f in files:
                    if 'illustrious' in f.lower() and 'clip_g' in f.lower():
                        clip_g_filename = f
                        break

    # Validate we have filenames
    if clip_l_filename is None:
        raise ValueError(
            f"Could not find CLIP-L file in {repo_id}. "
            f"Please specify clip_l_filename explicitly."
        )

    if clip_g_filename is None:
        raise ValueError(
            f"Could not find CLIP-G file in {repo_id}. "
            f"Please specify clip_g_filename explicitly."
        )

    print(f"  📥 Downloading CLIP-L: {clip_l_filename}")
    clip_l_path = hf_hub_download(
        repo_id=repo_id,
        filename=clip_l_filename,
        repo_type="model"
    )
    print(f"  ✓ CLIP-L: {clip_l_path}")

    print(f"  📥 Downloading CLIP-G: {clip_g_filename}")
    clip_g_path = hf_hub_download(
        repo_id=repo_id,
        filename=clip_g_filename,
        repo_type="model"
    )
    print(f"  ✓ CLIP-G: {clip_g_path}")

    return clip_l_path, clip_g_path


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class VAELyraTrainerConfig:
    """Training configuration for VAE Lyra with SDXL support and decoupled T5."""

    # Custom CLIP configuration
    use_illustrious_clips: bool = True
    illustrious_repo: str = "AbstractPhil/clips"
    clip_l_filename: Optional[str] = None
    clip_g_filename: Optional[str] = None
    auto_discover_clips: bool = True

    # CLIP skip (1 = last layer, 2 = penultimate layer)
    clip_skip: int = 2

    # Model architecture - SDXL with decoupled T5
    modality_dims: Dict[str, int] = None
    modality_seq_lens: Dict[str, int] = None
    binding_config: Dict[str, Dict[str, float]] = None

    latent_dim: int = 2048
    seq_len: int = 77
    encoder_layers: int = 3
    decoder_layers: int = 3
    hidden_dim: int = 1024
    dropout: float = 0.1

    # Fusion
    fusion_strategy: str = "adaptive_cantor"
    fusion_heads: int = 8
    fusion_dropout: float = 0.1
    cantor_depth: int = 8
    cantor_local_window: int = 3

    # Adaptive fusion parameters
    alpha_init: float = 1.0
    beta_init: float = 0.3
    alpha_lr_scale: float = 0.1
    beta_lr_scale: float = 1.0

    # Loss weights
    beta_kl: float = 0.1
    beta_reconstruction: float = 1.0
    beta_cross_modal: float = 0.05
    beta_alpha_regularization: float = 0.01
    recon_type: str = 'mse'

    # Per-modality reconstruction weights
    modality_recon_weights: Dict[str, float] = None

    # KL annealing
    use_kl_annealing: bool = True
    kl_anneal_epochs: int = 10
    kl_start_beta: float = 0.0

    # Training hyperparameters
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'

    # Data generation
    num_samples: int = 10000

    # Prompt source configuration
    prompt_source: str = "booru"

    # Ratios for mixed mode (must sum to 1.0)
    booru_ratio: float = 0.7
    synthetic_ratio: float = 0.15
    laion_ratio: float = 0.15

    # BooruSynthesizer CSV paths
    danbooru_csv: Optional[str] = None
    gelbooru_csv: Optional[str] = None
    e621_csv: Optional[str] = None
    rule34x_csv: Optional[str] = None

    # Coherent gender sampling (prevents 2girls + 3girls conflicts)
    use_coherent_gender: bool = True

    # T5 people count prefix ("3people", "there are three people", etc.)
    generate_t5_prefix: bool = True

    # Conduit system - injects random top-N tags for generalization
    use_conduit: bool = True
    conduit_top_n: int = 1000
    conduit_sample_k: int = 10
    conduit_sample_k_min: int = 5
    conduit_sample_k_max: int = 15
    conduit_position: str = "prepend"
    conduit_exclude_categories: List[str] = field(default_factory=lambda: [
        "artist", "copyright", "character", "metadata"
    ])

    # Summarization configuration
    use_summarizer: bool = True
    summarizer_model: str = "qwen2.5-1.5b"
    summarizer_batch_size: int = 16
    summary_separator: str = "¶"
    shuffle_tags_before_summary: bool = True
    summarizer_max_new_tokens: int = 64
    summarizer_temperature: float = 0.7
    use_summarizer_int8: bool = False

    # Prompt caching
    prompt_cache_dir: str = "./prompt_cache"
    prompt_cache_name: Optional[str] = None
    use_prompt_cache: bool = True
    save_prompt_cache: bool = True

    # Checkpointing
    checkpoint_dir: str = './checkpoints_lyra_illustrious'
    save_every: int = 1000
    keep_last_n: int = 3
    model_name: str = "lyra"  # Used in checkpoint filenames

    # Resume behavior
    resume_optimizer: bool = True  # False = fresh optimizer (fine-tuning mode)

    # HuggingFace Hub
    hf_repo: str = "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious"
    hub_checkpoint_file: Optional[str] = None  # Which checkpoint to load from hub (None = model.pt)
    push_to_hub: bool = True
    push_every: int = 2000
    push_checkpoints: bool = True  # Push step checkpoints to hub
    auto_load_from_hub: bool = True

    # Safetensors export
    export_safetensors: bool = True
    weights_subdir: str = "weights"

    # Logging
    use_wandb: bool = False
    wandb_project: str = 'vae-lyra-illustrious'
    wandb_entity: Optional[str] = None
    log_every: int = 50

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True

    # Misc
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self):
        if self.modality_dims is None:
            self.modality_dims = {
                "clip_l": 768,
                "clip_g": 1280,
                "t5_xl_l": 2048,
                "t5_xl_g": 2048
            }

        if self.modality_seq_lens is None:
            self.modality_seq_lens = {
                "clip_l": 77,
                "clip_g": 77,
                "t5_xl_l": 512,
                "t5_xl_g": 512
            }

        if self.binding_config is None:
            self.binding_config = {
                "clip_l": {"t5_xl_l": 0.3},
                "clip_g": {"t5_xl_g": 0.3},
                "t5_xl_l": {},
                "t5_xl_g": {}
            }

        if self.modality_recon_weights is None:
            self.modality_recon_weights = {
                "clip_l": 1.0,
                "clip_g": 1.0,
                "t5_xl_l": 0.3,
                "t5_xl_g": 0.3
            }

        # Validate ratios in mixed mode
        if self.prompt_source == "mixed":
            total = self.booru_ratio + self.synthetic_ratio + self.laion_ratio
            if abs(total - 1.0) > 0.01:
                print(f"⚠️  Prompt ratios sum to {total}, normalizing...")
                self.booru_ratio /= total
                self.synthetic_ratio /= total
                self.laion_ratio /= total


# ============================================================================
# DATASET WITH CLIP_SKIP SUPPORT AND SEPARATE T5 INPUT
# ============================================================================

class TextEmbeddingDataset(Dataset):
    """
    Dataset that generates CLIP-L, CLIP-G, and T5-XL embeddings on-the-fly.

    Supports separate text inputs for CLIP vs T5:
    - CLIP sees: raw booru tags
    - T5 sees: shuffled tags + separator + natural language summary
    """

    def __init__(
            self,
            clip_texts: List[str],
            t5_texts: List[str],
            clip_l_tokenizer: CLIPTokenizer,
            clip_l_model: CLIPTextModel,
            clip_g_tokenizer: CLIPTokenizer,
            clip_g_model: CLIPTextModelWithProjection,
            t5_tokenizer: T5Tokenizer,
            t5_model: T5EncoderModel,
            device: str = 'cuda',
            clip_max_length: int = 77,
            t5_max_length: int = 512,
            clip_skip: int = 1
    ):
        assert len(clip_texts) == len(t5_texts), "CLIP and T5 text lists must have same length"

        self.clip_texts = clip_texts
        self.t5_texts = t5_texts
        self.clip_l_tokenizer = clip_l_tokenizer
        self.clip_l_model = clip_l_model
        self.clip_g_tokenizer = clip_g_tokenizer
        self.clip_g_model = clip_g_model
        self.t5_tokenizer = t5_tokenizer
        self.t5_model = t5_model
        self.device = device
        self.clip_max_length = clip_max_length
        self.t5_max_length = t5_max_length
        self.clip_skip = clip_skip

        self.clip_l_model.to(device).eval()
        self.clip_g_model.to(device).eval()
        self.t5_model.to(device).eval()

        if clip_skip > 1:
            print(f"    Using clip_skip={clip_skip} (penultimate layer extraction)")

    def __len__(self):
        return len(self.clip_texts)

    @torch.no_grad()
    def __getitem__(self, idx):
        clip_text = self.clip_texts[idx]
        t5_text = self.t5_texts[idx]

        # CLIP-L embedding with clip_skip support
        clip_l_tokens = self.clip_l_tokenizer(
            clip_text,
            max_length=self.clip_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        clip_l_output = self.clip_l_model(
            **clip_l_tokens,
            output_hidden_states=(self.clip_skip > 1)
        )
        clip_l_embed = get_clip_hidden_state(
            clip_l_output,
            clip_skip=self.clip_skip,
            output_hidden_states=(self.clip_skip > 1)
        ).squeeze(0)

        # CLIP-G embedding with clip_skip support
        clip_g_tokens = self.clip_g_tokenizer(
            clip_text,
            max_length=self.clip_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        clip_g_output = self.clip_g_model(
            **clip_g_tokens,
            output_hidden_states=(self.clip_skip > 1)
        )
        clip_g_embed = get_clip_hidden_state(
            clip_g_output,
            clip_skip=self.clip_skip,
            output_hidden_states=(self.clip_skip > 1)
        ).squeeze(0)

        # T5-XL embeddings
        t5_tokens = self.t5_tokenizer(
            t5_text,
            max_length=self.t5_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        t5_output = self.t5_model(**t5_tokens)
        t5_embed = t5_output.last_hidden_state.squeeze(0)

        return {
            'clip_l': clip_l_embed.cpu(),
            'clip_g': clip_g_embed.cpu(),
            't5_xl_l': t5_embed.cpu(),
            't5_xl_g': t5_embed.cpu(),
            'clip_text': clip_text,
            't5_text': t5_text
        }


# ============================================================================
# TRAINER
# ============================================================================

class VAELyraTrainer:
    """Trainer for VAE Lyra with Illustrious CLIP weights and multi-source prompts."""

    def __init__(self, config: VAELyraTrainerConfig):
        self.config = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.hf_api = HfApi()
        self._init_hf_repo()

        print("🎵 Initializing VAE Lyra (Illustrious Edition)...")
        print(f"   CLIP weights: {config.illustrious_repo}")
        print(f"   CLIP skip: {config.clip_skip}")
        print(f"   Fusion: {config.fusion_strategy}")
        print(f"   Prompt source: {config.prompt_source}")
        print(f"   Resume optimizer: {config.resume_optimizer}")

        # Track last loss for checkpoint messages
        self._last_loss = float('inf')
        self.start_epoch = 0

        if config.auto_load_from_hub:
            loaded = self._try_load_from_hub()
            if loaded:
                print("✓ Loaded model from HuggingFace Hub")

        if not hasattr(self, 'model'):
            self.model = self._build_model()
            self.optimizer = None
            self.scheduler = None
            self.global_step = 0
            self.epoch = 0
            self.best_loss = float('inf')

        self.loss_fn = self._build_loss_fn()

        if self.optimizer is None:
            self.optimizer = self._build_optimizer()

        if self.scheduler is None and config.use_scheduler:
            self.scheduler = self._build_scheduler()

        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize prompt generators
        self._init_prompt_generators()

        # Initialize summarizer if enabled
        self.summarizer = None
        if config.use_summarizer:
            self._init_summarizer()

        self.used_prompts = []
        self.prompt_sources = []
        self.summaries = []

        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=asdict(config),
                name=f"lyra_illustrious_{config.fusion_strategy}_{config.prompt_source}",
                resume="allow"
            )

    def _init_prompt_generators(self):
        """Initialize all prompt generation systems."""
        print("\nInitializing prompt generators...")

        self.prompt_gen = SynthesisSystem(seed=self.config.seed)
        print("  ✓ SynthesisSystem ready")

        if self.config.prompt_source in ["booru", "mixed"]:
            # Build conduit config if enabled
            conduit_config = None
            if self.config.use_conduit:
                conduit_config = ConduitConfig(
                    enabled=True,
                    top_n=self.config.conduit_top_n,
                    sample_k=self.config.conduit_sample_k,
                    sample_k_min=self.config.conduit_sample_k_min,
                    sample_k_max=self.config.conduit_sample_k_max,
                    position=self.config.conduit_position,
                    exclude_categories=self.config.conduit_exclude_categories,
                )

            booru_config = BooruConfig(
                danbooru_csv=self.config.danbooru_csv,
                gelbooru_csv=self.config.gelbooru_csv,
                e621_csv=self.config.e621_csv,
                rule34x_csv=self.config.rule34x_csv,
                use_coherent_gender=self.config.use_coherent_gender,
                generate_t5_prefix=self.config.generate_t5_prefix,
                conduit=conduit_config,
                seed=self.config.seed
            )
            self.booru_gen = BooruSynthesizer(booru_config)
            stats = self.booru_gen.stats()
            print(f"  ✓ BooruSynthesizer ready: {stats['total_tags']:,} tags, {stats['templates']} templates")
            print(f"    Coherent gender: {self.config.use_coherent_gender}")
            print(f"    T5 prefix: {self.config.generate_t5_prefix}")
            if self.config.use_conduit:
                conduit_stats = stats.get('conduit', {})
                print(f"    Conduit: {conduit_stats.get('size', 0)} tags from top {self.config.conduit_top_n}")
        else:
            self.booru_gen = None

        if self.config.prompt_source in ["laion", "mixed"]:
            self.flavors = self._load_flavors()
        else:
            self.flavors = []

    def _init_summarizer(self):
        """Initialize the CaptionFactory summarizer for T5 inputs."""
        print(f"\n🔤 Initializing summarizer ({self.config.summarizer_model})...")

        summarizer_config = CaptionFactoryConfig(
            model_name=self.config.summarizer_model,
            use_int8=self.config.use_summarizer_int8,
            max_new_tokens=self.config.summarizer_max_new_tokens,
            temperature=self.config.summarizer_temperature,
            device=self.config.device,
            batch_size=self.config.summarizer_batch_size,
            keep_model_loaded=True,
        )

        self.summarizer = CaptionFactory(summarizer_config)
        print(f"  ✓ Summarizer ready: {self.config.summarizer_model}")
        print(f"    Batch size: {self.config.summarizer_batch_size}")
        print(f"    Separator token: '{self.config.summary_separator}' (pilcrow)")

    def _shuffle_tags(self, tags_str: str) -> str:
        """Shuffle comma-separated tags while preserving quality tags at front."""
        tags = [t.strip() for t in tags_str.split(',') if t.strip()]

        if not tags:
            return tags_str

        quality_tags = []
        content_tags = []

        quality_keywords = {'masterpiece', 'best quality', 'high quality', 'absurdres',
                            'highres', 'very aesthetic', 'aesthetic', 'newest', 'recent'}

        for tag in tags:
            tag_lower = tag.lower().strip()
            if tag_lower in quality_keywords:
                quality_tags.append(tag)
            else:
                content_tags.append(tag)

        random.shuffle(content_tags)
        shuffled = quality_tags + content_tags
        return ', '.join(shuffled)

    def _generate_summaries(self, tags_list: List[str]) -> List[str]:
        """Generate natural language summaries for a batch of tag strings."""
        if self.summarizer is None:
            return [""] * len(tags_list)

        print(f"\n📝 Generating summaries for {len(tags_list):,} prompts...")

        try:
            chunk_size = self.config.summarizer_batch_size * 10
            all_summaries = []

            pbar = tqdm(range(0, len(tags_list), chunk_size), desc="Summarizing")
            for start_idx in pbar:
                end_idx = min(start_idx + chunk_size, len(tags_list))
                chunk = tags_list[start_idx:end_idx]

                chunk_summaries = self.summarizer.summarize_batch(chunk)
                all_summaries.extend(chunk_summaries)

                pbar.set_postfix({"done": len(all_summaries)})

            cleaned = []
            for s in all_summaries:
                s = s.replace(self.config.summary_separator, ' ')
                s = ' '.join(s.split())
                cleaned.append(s)

            print(f"  ✓ Generated {len(cleaned):,} summaries")

            print(f"\n  Sample summaries:")
            for i in range(min(3, len(cleaned))):
                sample = cleaned[i][:80] + "..." if len(cleaned[i]) > 80 else cleaned[i]
                print(f"    [{i}] {sample}")

            return cleaned

        finally:
            if hasattr(self.summarizer, 'unload_model'):
                self.summarizer.unload_model()
                print(f"  ✓ Unloaded summarizer to free VRAM")

    def _build_t5_text(self, tags: str, summary: str, t5_prefix: Optional[str] = None) -> str:
        """
        Build the T5 input text with optional people count prefix, tags, separator, and summary.

        Format: "{t5_prefix}, {tags} {separator} {summary}"
        Example: "three people, 1girl, 2boys, uniform ¶ A group of students smiling"
        """
        if self.config.shuffle_tags_before_summary:
            tags = self._shuffle_tags(tags)

        # Build the full T5 input
        parts = []

        # Add people count prefix if provided
        if t5_prefix:
            parts.append(t5_prefix)

        # Add tags
        parts.append(tags)

        # Join prefix and tags
        t5_text = ", ".join(parts) if t5_prefix else tags

        # Add summary after separator
        if summary:
            t5_text = f"{t5_text} {self.config.summary_separator} {summary}"

        return t5_text

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Prompt Cache
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_cache_path(self) -> Path:
        """Get the path for the prompt cache file."""
        cache_dir = Path(self.config.prompt_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self.config.prompt_cache_name:
            name = self.config.prompt_cache_name
        else:
            name = f"prompts_{self.config.prompt_source}_{self.config.num_samples}"
            if self.config.use_summarizer:
                name += f"_{self.config.summarizer_model.replace('/', '_')}"

        return cache_dir / f"{name}.json"

    def _save_prompt_cache(
            self,
            tags_list: List[str],
            summaries: List[str],
            sources: List[str],
            t5_prefixes: List[Optional[str]],
            people_counts: List[int]
    ) -> None:
        """Save generated prompts and summaries to cache."""
        if not self.config.save_prompt_cache:
            return

        cache_path = self._get_cache_path()

        cache_data = {
            "version": "1.1",  # Bumped version for new fields
            "config": {
                "prompt_source": self.config.prompt_source,
                "num_samples": len(tags_list),
                "summarizer_model": self.config.summarizer_model if self.config.use_summarizer else None,
                "summary_separator": self.config.summary_separator,
                "use_coherent_gender": self.config.use_coherent_gender,
                "generate_t5_prefix": self.config.generate_t5_prefix,
                "use_conduit": self.config.use_conduit,
            },
            "data": [
                {
                    "tags": tags,
                    "summary": summary,
                    "source": source,
                    "t5_prefix": t5_prefix,
                    "people_count": count
                }
                for tags, summary, source, t5_prefix, count in zip(
                    tags_list, summaries, sources, t5_prefixes, people_counts
                )
            ]
        }

        print(f"\n💾 Saving prompt cache to {cache_path}...")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        jsonl_path = cache_path.with_suffix('.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in cache_data["data"]:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved {len(tags_list):,} prompts ({size_mb:.1f} MB)")

    def _load_prompt_cache(self) -> Optional[Tuple[List[str], List[str], List[str], List[Optional[str]], List[int]]]:
        """
        Load prompts and summaries from cache if available.

        Returns:
            Tuple of (tags_list, summaries, sources, t5_prefixes, people_counts) or None
        """
        if not self.config.use_prompt_cache:
            return None

        cache_path = self._get_cache_path()

        if not cache_path.exists():
            print(f"  📂 No cache found at {cache_path}")
            return None

        print(f"\n📂 Loading prompt cache from {cache_path}...")

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            version = cache_data.get("version", "1.0")

            # Accept both 1.0 and 1.1 versions
            if version not in ["1.0", "1.1"]:
                print(f"  ⚠️  Cache version {version} not supported, regenerating...")
                return None

            cached_config = cache_data.get("config", {})

            if cached_config.get("num_samples") != self.config.num_samples:
                print(f"  ⚠️  Sample count mismatch, regenerating...")
                return None

            if cached_config.get("prompt_source") != self.config.prompt_source:
                print(f"  ⚠️  Prompt source mismatch, regenerating...")
                return None

            data = cache_data["data"]
            tags_list = [item["tags"] for item in data]
            summaries = [item.get("summary", "") for item in data]
            sources = [item.get("source", "unknown") for item in data]

            # Handle both old (1.0) and new (1.1) cache formats
            t5_prefixes = [item.get("t5_prefix", None) for item in data]
            people_counts = [item.get("people_count", 0) for item in data]

            print(f"  ✓ Loaded {len(tags_list):,} cached prompts (version {version})")

            source_counts = Counter(sources)
            print(f"  Distribution: {dict(source_counts)}")

            # Show people count distribution if available
            if any(c > 0 for c in people_counts):
                count_dist = Counter(people_counts)
                print(f"  People counts: {dict(sorted(count_dist.items()))}")

            return tags_list, summaries, sources, t5_prefixes, people_counts

        except Exception as e:
            print(f"  ⚠️  Failed to load cache: {e}")
            return None

    def _load_or_generate_prompts(self, num_samples: int) -> Tuple[
        List[str], List[str], List[str], List[Optional[str]], List[int]]:
        """
        Load prompts from cache or generate new ones.

        Returns:
            Tuple of (tags_list, summaries, sources, t5_prefixes, people_counts)
        """
        cached = self._load_prompt_cache()
        if cached is not None:
            return cached

        print(f"\n🎼 Generating {num_samples:,} prompts (source: {self.config.prompt_source})...")

        tags_list, sources, t5_prefixes, people_counts = [], [], [], []
        for _ in tqdm(range(num_samples), desc="Generating tags"):
            tags, source, t5_prefix, count = self._generate_prompt()
            tags_list.append(tags)
            sources.append(source)
            t5_prefixes.append(t5_prefix)
            people_counts.append(count)

        source_counts = Counter(sources)
        print(f"\nDistribution: {dict(source_counts)}")

        # Show people count distribution
        count_dist = Counter(people_counts)
        print(f"People counts: {dict(sorted(count_dist.items()))}")

        if self.config.use_summarizer:
            summaries = self._generate_summaries(tags_list)
        else:
            summaries = [""] * len(tags_list)

        self._save_prompt_cache(tags_list, summaries, sources, t5_prefixes, people_counts)

        return tags_list, summaries, sources, t5_prefixes, people_counts

    def _load_flavors(self) -> List[str]:
        """Load LAION flavors from remote source."""
        print("  Loading LAION flavors...")
        try:
            r = requests.get(
                "https://raw.githubusercontent.com/pharmapsychotic/clip-interrogator/main/clip_interrogator/data/flavors.txt",
                timeout=30
            )
            flavors = [line.strip() for line in r.text.split('\n') if line.strip()]
            print(f"  ✓ Loaded {len(flavors):,} flavors")
            return flavors
        except Exception as e:
            print(f"  ⚠️  Fallback flavors: {e}")
            return ["a beautiful landscape", "abstract art", "detailed portrait"]

    def _generate_prompt(self) -> Tuple[str, str, Optional[str], int]:
        """
        Generate a single prompt based on configured source.

        Returns:
            Tuple of (tags, source, t5_prefix, people_count)
        """
        source = self.config.prompt_source

        if source == "booru":
            result = self.booru_gen.generate_with_t5()
            return result['clip'], "booru", result['t5_prefix'], result['people_count']

        elif source == "synthetic":
            complexity = random.choice([2, 3, 4, 5])
            prompt = self.prompt_gen.synthesize(complexity=complexity)['text']
            # For synthetic, generate coherent gender separately
            if self.config.use_coherent_gender:
                gender_tags, count = sample_coherent_gender()
                t5_prefix = generate_people_count_prefix(count) if self.config.generate_t5_prefix else None
                # Prepend gender tags to prompt
                prompt = ", ".join(gender_tags) + ", " + prompt
                return prompt, "synthetic", t5_prefix, count
            return prompt, "synthetic", None, 0

        elif source == "laion":
            prompt = random.choice(self.flavors)
            # LAION flavors don't have structured people counts
            return prompt, "laion", None, 0

        elif source == "mixed":
            r = random.random()
            if r < self.config.booru_ratio:
                result = self.booru_gen.generate_with_t5()
                return result['clip'], "booru", result['t5_prefix'], result['people_count']
            elif r < self.config.booru_ratio + self.config.synthetic_ratio:
                complexity = random.choice([2, 3, 4, 5])
                prompt = self.prompt_gen.synthesize(complexity=complexity)['text']
                if self.config.use_coherent_gender:
                    gender_tags, count = sample_coherent_gender()
                    t5_prefix = generate_people_count_prefix(count) if self.config.generate_t5_prefix else None
                    prompt = ", ".join(gender_tags) + ", " + prompt
                    return prompt, "synthetic", t5_prefix, count
                return prompt, "synthetic", None, 0
            else:
                prompt = random.choice(self.flavors)
                return prompt, "laion", None, 0

        else:
            raise ValueError(f"Unknown prompt_source: {source}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HuggingFace Hub
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _init_hf_repo(self):
        """Initialize HuggingFace repo."""
        if not self.config.push_to_hub:
            return
        try:
            create_repo(
                self.config.hf_repo,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            print(f"✓ HF repo: https://huggingface.co/{self.config.hf_repo}")
        except Exception as e:
            print(f"⚠️  Could not create HF repo: {e}")

    def _try_load_from_hub(self) -> bool:
        """
        Try to load model from HuggingFace Hub.

        Uses config.hub_checkpoint_file to determine which file to load.
        If None, defaults to "model.pt".

        Respects config.resume_optimizer:
            True  = Load optimizer state (continue training)
            False = Skip optimizer (fine-tuning with fresh optimizer)
        """
        try:
            checkpoint_file = self.config.hub_checkpoint_file or "model.pt"
            print(f"🔍 Loading from hub: {self.config.hf_repo}/{checkpoint_file}")

            try:
                model_path = hf_hub_download(
                    repo_id=self.config.hf_repo,
                    filename=checkpoint_file,
                    repo_type="model"
                )

                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

                # Build model
                self.model = self._build_model()

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Assume the dict IS the state dict
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)

                # Build optimizer
                self.optimizer = self._build_optimizer()

                # Optionally load optimizer state
                if isinstance(checkpoint, dict):
                    if self.config.resume_optimizer and 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print(f"  ✓ Loaded optimizer state (resume_optimizer=True)")
                    else:
                        print(f"  ⚠️  Fresh optimizer (resume_optimizer=False)")

                    # Load training state
                    self.global_step = checkpoint.get('global_step', 0)
                    self.epoch = checkpoint.get('epoch', 0)
                    self.best_loss = checkpoint.get('best_loss', float('inf'))
                    self.start_epoch = self.epoch

                    # Optionally load scheduler
                    if self.config.use_scheduler:
                        self.scheduler = self._build_scheduler()
                        if self.config.resume_optimizer and 'scheduler_state_dict' in checkpoint:
                            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                else:
                    # Raw state dict - no training state
                    self.global_step = 0
                    self.epoch = 0
                    self.best_loss = float('inf')
                    self.start_epoch = 0
                    print(f"  ⚠️  Loaded raw state dict (no training state)")

                print(f"✓ Resumed from step {self.global_step}, epoch {self.epoch}, best_loss={self.best_loss:.4f}")
                return True

            except Exception as e:
                print(f"   Could not load {checkpoint_file}: {e}")
                return False

        except Exception as e:
            print(f"⚠️  Could not access HF Hub: {e}")
            return False

    def load_checkpoint(
            self,
            checkpoint_path: str,
            repo_id: Optional[str] = None,
            resume_optimizer: Optional[bool] = None,
            strict: bool = False
    ) -> Dict[str, Any]:
        """
        Load a checkpoint from local path or HF Hub.

        Args:
            checkpoint_path: Local path OR filename on HF Hub (e.g., "checkpoint_3000.pt")
            repo_id: HF repo to download from (defaults to config.hf_repo)
            resume_optimizer: Override config.resume_optimizer if provided
            strict: If True, require exact state dict match

        Returns:
            Dict with loading info: loaded_keys, skipped_keys, new_keys

        Examples:
            # Load from local file
            trainer.load_checkpoint("./checkpoints/checkpoint_3000.pt")

            # Load from default HF repo
            trainer.load_checkpoint("checkpoint_3000.pt")

            # Load from specific HF repo
            trainer.load_checkpoint("checkpoint_3000.pt", repo_id="user/other-repo")
        """
        resume_opt = resume_optimizer if resume_optimizer is not None else self.config.resume_optimizer
        repo_id = repo_id or self.config.hf_repo

        # Determine if this is a local file or needs to be downloaded
        local_path = checkpoint_path

        if not os.path.exists(checkpoint_path):
            # Not a local file - try to download from HF Hub
            print(f"📥 Downloading {checkpoint_path} from {repo_id}...")
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=checkpoint_path,
                    repo_type="model",
                    local_dir="./hf_cache"
                )
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not find checkpoint. Tried:\n"
                    f"  - Local: {checkpoint_path}\n"
                    f"  - HF Hub: {repo_id}/{checkpoint_path}\n"
                    f"Error: {e}"
                )

        print(f"🔄 Loading checkpoint: {local_path}")
        checkpoint = torch.load(local_path, map_location=self.device, weights_only=False)

        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                old_state = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                old_state = checkpoint['state_dict']
            else:
                old_state = checkpoint
        else:
            old_state = checkpoint

        # Load model weights
        current_state = self.model.state_dict()

        loaded_keys = []
        skipped_keys = []
        new_keys = []

        for key in old_state:
            if key in current_state:
                if old_state[key].shape == current_state[key].shape:
                    current_state[key] = old_state[key]
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape: {old_state[key].shape} vs {current_state[key].shape})")
            else:
                skipped_keys.append(f"{key} (not in model)")

        for key in current_state:
            if key not in old_state:
                new_keys.append(key)

        if strict and (skipped_keys or new_keys):
            raise ValueError(f"Strict loading failed: {len(skipped_keys)} skipped, {len(new_keys)} new")

        self.model.load_state_dict(current_state)

        # Load training state
        if isinstance(checkpoint, dict):
            self.global_step = checkpoint.get('global_step', 0)
            self.epoch = checkpoint.get('epoch', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.start_epoch = self.epoch

            # Optimizer
            if resume_opt and 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"  ✓ Loaded optimizer state")
                except Exception as e:
                    print(f"  ⚠️  Could not load optimizer: {e}")
            else:
                print(f"  ⚠️  Fresh optimizer (resume_optimizer={resume_opt})")

            # Scheduler
            if resume_opt and self.scheduler and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print(f"  ✓ Loaded scheduler state")
                except Exception as e:
                    print(f"  ⚠️  Could not load scheduler: {e}")

        print(f"\n✅ Checkpoint loaded:")
        print(f"   Loaded:  {len(loaded_keys)} parameters")
        print(f"   Skipped: {len(skipped_keys)} parameters")
        print(f"   New:     {len(new_keys)} parameters")
        print(f"   Step:    {self.global_step}, Epoch: {self.epoch}")

        return {
            'loaded_keys': loaded_keys,
            'skipped_keys': skipped_keys,
            'new_keys': new_keys,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Checkpointing
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save_checkpoint(self, best: bool = False) -> Path:
        """
        Save checkpoint with optional safetensors export.

        Filenames use config.model_name:
            - Best: model.pt, {model_name}_best.safetensors
            - Step: checkpoint_{model_name}_{step}.pt, {model_name}_step_{step}.safetensors

        Returns:
            Path to the saved checkpoint
        """
        model_name = self.config.model_name

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'best_loss': self.best_loss,
            'model_name': model_name,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Determine filename
        if best:
            pt_filename = 'model.pt'  # Keep model.pt for best (compatibility)
            st_filename = f'{model_name}_best.safetensors'
        else:
            pt_filename = f'checkpoint_{model_name}_{self.global_step}.pt'
            st_filename = f'{model_name}_step_{self.global_step}.safetensors'

        pt_path = Path(self.config.checkpoint_dir) / pt_filename
        torch.save(checkpoint, pt_path)
        print(f"  💾 Saved: {pt_path.name}")

        # Export safetensors (model weights only, no optimizer)
        if self.config.export_safetensors:
            weights_dir = Path(self.config.checkpoint_dir) / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            st_path = weights_dir / st_filename

            # Convert state dict for safetensors (must be contiguous, on CPU)
            state_dict = {k: v.contiguous().cpu() for k, v in self.model.state_dict().items()}
            save_safetensors(state_dict, st_path)
            print(f"  💾 Saved: weights/{st_filename}")

        if not best:
            self._cleanup_checkpoints()

        return pt_path

    def _push_checkpoint_to_hub(self, checkpoint_path: Path, is_best: bool = False):
        """
        Push a specific checkpoint to HF Hub with safetensors in weights/ subdirectory.

        Uses config.model_name for consistent naming.
        """
        if not self.config.push_to_hub:
            return

        model_name = self.config.model_name

        try:
            step = self.global_step
            loss_str = f"{self.best_loss:.4f}" if is_best else f"{self._last_loss:.4f}"

            # Determine filenames
            if is_best:
                pt_repo_path = "model.pt"
                st_repo_path = f"{self.config.weights_subdir}/{model_name}_best.safetensors"
                commit_msg = f"Best model @ step {step} (loss={loss_str})"
            else:
                pt_repo_path = f"checkpoint_{model_name}_{step}.pt"
                st_repo_path = f"{self.config.weights_subdir}/{model_name}_step_{step}.safetensors"
                commit_msg = f"Checkpoint @ step {step} (loss={loss_str})"

            print(f"\n📤 Pushing to HF Hub: {pt_repo_path}", end=" ", flush=True)

            # Upload .pt checkpoint
            self.hf_api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=pt_repo_path,
                repo_id=self.config.hf_repo,
                repo_type="model",
                commit_message=commit_msg
            )
            print("✓")

            # Upload safetensors to weights/ subdirectory
            if self.config.export_safetensors:
                weights_dir = Path(self.config.checkpoint_dir) / "weights"

                if is_best:
                    st_local = weights_dir / f"{model_name}_best.safetensors"
                else:
                    st_local = weights_dir / f"{model_name}_step_{step}.safetensors"

                if st_local.exists():
                    print(f"  📤 Pushing: {st_repo_path}", end=" ", flush=True)
                    self.hf_api.upload_file(
                        path_or_fileobj=str(st_local),
                        path_in_repo=st_repo_path,
                        repo_id=self.config.hf_repo,
                        repo_type="model",
                        commit_message=f"Weights {st_repo_path}"
                    )
                    print("✓")

            # Update config.json
            config_path = Path(self.config.checkpoint_dir) / "config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)

            self.hf_api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="config.json",
                repo_id=self.config.hf_repo,
                repo_type="model",
                commit_message=f"Config @ step {step}"
            )

            # Update model card on best
            if is_best:
                self._create_model_card()

        except Exception as e:
            print(f"✗ Push failed: {e}")

    def _push_to_hub(self, is_best: bool = False):
        """Push current state to hub (wrapper for backward compatibility)."""
        if not self.config.push_to_hub:
            return

        model_name = self.config.model_name

        # Save temp checkpoint for upload
        temp_path = Path(self.config.checkpoint_dir) / "temp_upload.pt"
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': asdict(self.config),
            'model_name': model_name,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, temp_path)

        # Also save safetensors
        if self.config.export_safetensors:
            weights_dir = Path(self.config.checkpoint_dir) / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)

            if is_best:
                st_path = weights_dir / f"{model_name}_best.safetensors"
            else:
                st_path = weights_dir / f"{model_name}_step_{self.global_step}.safetensors"

            state_dict = {k: v.contiguous().cpu() for k, v in self.model.state_dict().items()}
            save_safetensors(state_dict, st_path)

        self._push_checkpoint_to_hub(temp_path, is_best=is_best)
        temp_path.unlink(missing_ok=True)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent."""
        model_name = self.config.model_name
        checkpoint_dir = Path(self.config.checkpoint_dir)

        # Clean up .pt checkpoints (pattern: checkpoint_{model_name}_{step}.pt)
        checkpoints = sorted(
            [f for f in checkpoint_dir.glob(f'checkpoint_{model_name}_*.pt')],
            key=lambda x: int(x.stem.split('_')[-1])
        )
        if len(checkpoints) > self.config.keep_last_n:
            for ckpt in checkpoints[:-self.config.keep_last_n]:
                ckpt.unlink()

        # Also clean up safetensors (pattern: {model_name}_step_{step}.safetensors)
        if self.config.export_safetensors:
            weights_dir = checkpoint_dir / "weights"
            if weights_dir.exists():
                st_files = sorted(
                    [f for f in weights_dir.glob(f'{model_name}_step_*.safetensors')],
                    key=lambda x: int(x.stem.split('_')[-1])
                )
                if len(st_files) > self.config.keep_last_n:
                    for st in st_files[:-self.config.keep_last_n]:
                        st.unlink()

    def _create_model_card(self):
        """Create and upload model card."""
        fusion_params_str = ""
        if hasattr(self.model, 'get_fusion_params'):
            params = self.model.get_fusion_params()
            if params:
                fusion_params_str = "\n## Learned Parameters\n\n"
                if 'alphas' in params:
                    fusion_params_str += "**Alpha (Visibility):**\n"
                    for name, alpha in params['alphas'].items():
                        fusion_params_str += f"- {name}: {torch.sigmoid(alpha).item():.4f}\n"
                if 'betas' in params:
                    fusion_params_str += "\n**Beta (Capacity):**\n"
                    for name, beta in params['betas'].items():
                        fusion_params_str += f"- {name}: {torch.sigmoid(beta).item():.4f}\n"

        prompt_info = f"- **Prompt Source**: {self.config.prompt_source}\n"
        if self.config.prompt_source == "mixed":
            prompt_info += f"  - Booru: {self.config.booru_ratio * 100:.0f}%\n"
            prompt_info += f"  - Synthetic: {self.config.synthetic_ratio * 100:.0f}%\n"
            prompt_info += f"  - LAION: {self.config.laion_ratio * 100:.0f}%\n"

        model_card = f"""---
tags:
- vae
- multimodal
- clip
- t5
- sdxl
- illustrious
- adaptive-cantor
- booru
license: mit
---

# VAE Lyra 🎵 - Illustrious Edition

Multi-modal VAE trained with **custom CLIP weights**.

## CLIP Encoders

Uses CLIP weights from `{self.config.illustrious_repo}`:
- CLIP-L: `{self.config.clip_l_filename or 'auto-discovered'}`
- CLIP-G: `{self.config.clip_g_filename or 'auto-discovered'}`

**CLIP Skip**: {self.config.clip_skip} ({"penultimate layer" if self.config.clip_skip == 2 else "last layer"})

## Model Details

- **Fusion Strategy**: {self.config.fusion_strategy}
- **Latent Dimension**: {self.config.latent_dim}
- **Training Steps**: {self.global_step:,}
- **Best Loss**: {self.best_loss:.4f}
{prompt_info}

## Quick Load (Safetensors)

```python
from safetensors.torch import load_file

# Load just the weights (fast)
state_dict = load_file("weights/{self.config.model_name}_best.safetensors")

# Or specific step
state_dict = load_file("weights/{self.config.model_name}_step_5000.safetensors")
```

## T5 Input Format

T5 receives a different input than CLIP to enable richer semantic understanding:

```
CLIP sees:  "masterpiece, 1girl, blue hair, school uniform, smile"
T5 sees:    "masterpiece, 1girl, blue hair, school uniform, smile ¶ A cheerful schoolgirl with blue hair smiling warmly"
```

The pilcrow (`¶`) separator acts as a mode-switch token.

{fusion_params_str}

## Usage

```python
from lyra_xl_multimodal import load_lyra_from_hub

model = load_lyra_from_hub("{self.config.hf_repo}")
model.eval()

inputs = {{
    "clip_l": clip_l_embeddings,     # [batch, 77, 768]
    "clip_g": clip_g_embeddings,     # [batch, 77, 1280]
    "t5_xl_l": t5_xl_embeddings,     # [batch, 512, 2048]
    "t5_xl_g": t5_xl_embeddings      # [batch, 512, 2048]
}}

recons, mu, logvar, _ = model(inputs, target_modalities=["clip_l", "clip_g"])
```

## Files

- `model.pt` - Full checkpoint (model + optimizer + scheduler)
- `checkpoint_{self.config.model_name}_XXXX.pt` - Step checkpoints
- `config.json` - Training configuration
- `weights/{self.config.model_name}_best.safetensors` - Best model weights only
- `weights/{self.config.model_name}_step_XXXX.safetensors` - Step checkpoints (weights only)
"""
        try:
            card_path = Path(self.config.checkpoint_dir) / "README.md"
            with open(card_path, 'w') as f:
                f.write(model_card)
            self.hf_api.upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=self.config.hf_repo,
                repo_type="model",
                commit_message=f"Model card @ step {self.global_step}"
            )
        except Exception as e:
            print(f"⚠️  Model card update failed: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Model Building
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_model(self) -> nn.Module:
        """Build the VAE model."""
        vae_config = MultiModalVAEConfig(
            modality_dims=self.config.modality_dims,
            modality_seq_lens=self.config.modality_seq_lens,
            binding_config=self.config.binding_config,
            latent_dim=self.config.latent_dim,
            seq_len=self.config.seq_len,
            encoder_layers=self.config.encoder_layers,
            decoder_layers=self.config.decoder_layers,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            fusion_strategy=self.config.fusion_strategy,
            fusion_heads=self.config.fusion_heads,
            fusion_dropout=self.config.fusion_dropout,
            cantor_depth=self.config.cantor_depth,
            cantor_local_window=self.config.cantor_local_window,
            alpha_init=self.config.alpha_init,
            beta_init=self.config.beta_init,
            alpha_lr_scale=self.config.alpha_lr_scale,
            beta_lr_scale=self.config.beta_lr_scale,
            beta_alpha_regularization=self.config.beta_alpha_regularization,
            seed=self.config.seed
        )
        model = MultiModalVAE(vae_config)
        model.to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ VAE Lyra: {total_params:,} params")
        return model

    def _build_optimizer(self):
        """Build optimizer with separate param groups for alpha/beta."""
        param_groups = []
        regular_params = []
        alpha_params = []
        beta_params = []

        for name, param in self.model.named_parameters():
            if 'alphas' in name:
                alpha_params.append(param)
            elif 'betas' in name:
                beta_params.append(param)
            else:
                regular_params.append(param)

        if regular_params:
            param_groups.append({
                'params': regular_params,
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            })
        if alpha_params:
            param_groups.append({
                'params': alpha_params,
                'lr': self.config.learning_rate * self.config.alpha_lr_scale,
                'weight_decay': 0.0
            })
        if beta_params:
            param_groups.append({
                'params': beta_params,
                'lr': self.config.learning_rate * self.config.beta_lr_scale,
                'weight_decay': 0.0
            })

        return AdamW(param_groups)

    def _build_loss_fn(self):
        """Build the loss function."""
        return MultiModalVAELoss(
            beta_kl=self.config.beta_kl,
            beta_reconstruction=self.config.beta_reconstruction,
            beta_cross_modal=self.config.beta_cross_modal,
            beta_alpha_regularization=self.config.beta_alpha_regularization,
            recon_type=self.config.recon_type,
            modality_weights=self.config.modality_recon_weights
        )

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config.scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.1
            )
        return None

    def _get_current_kl_beta(self) -> float:
        """Get current KL beta with annealing."""
        if not self.config.use_kl_annealing:
            return self.config.beta_kl
        if self.epoch >= self.config.kl_anneal_epochs:
            return self.config.beta_kl
        progress = self.epoch / self.config.kl_anneal_epochs
        return self.config.kl_start_beta + (self.config.beta_kl - self.config.kl_start_beta) * progress

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Training
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def prepare_data(self, num_samples: Optional[int] = None) -> DataLoader:
        """Prepare dataset with Illustrious CLIP + T5-XL encoders."""
        num_samples = num_samples or self.config.num_samples

        tags_list, summaries, sources, t5_prefixes, people_counts = self._load_or_generate_prompts(num_samples)

        self.used_prompts = tags_list
        self.prompt_sources = sources
        self.summaries = summaries
        self.t5_prefixes = t5_prefixes
        self.people_counts = people_counts

        clip_texts = tags_list
        t5_texts = []

        for tags, summary, t5_prefix in zip(tags_list, summaries, t5_prefixes):
            t5_text = self._build_t5_text(tags, summary, t5_prefix)
            t5_texts.append(t5_text)

        print(f"\n📝 Text format examples:")
        for i in range(min(3, len(t5_texts))):
            clip_sample = clip_texts[i][:80] + "..." if len(clip_texts[i]) > 80 else clip_texts[i]
            t5_sample = t5_texts[i][:120] + "..." if len(t5_texts[i]) > 120 else t5_texts[i]
            print(f"\n  [{i}] CLIP: {clip_sample}")
            print(f"      T5:   {t5_sample}")
            if t5_prefixes[i]:
                print(f"      Count: {people_counts[i]} ({t5_prefixes[i]})")

        # Download custom CLIP weights
        if self.config.use_illustrious_clips:
            clip_l_path, clip_g_path = download_illustrious_clips(
                repo_id=self.config.illustrious_repo,
                clip_l_filename=self.config.clip_l_filename,
                clip_g_filename=self.config.clip_g_filename,
                auto_discover=self.config.auto_discover_clips
            )

            print("\n🔧 Loading Illustrious CLIP encoders...")
            print("  [1/3] CLIP-L (Illustrious)...")
            clip_l_model, clip_l_tokenizer = load_illustrious_clip_l(
                clip_l_path, self.device
            )

            print("  [2/3] CLIP-G (Illustrious)...")
            clip_g_model, clip_g_tokenizer = load_illustrious_clip_g(
                clip_g_path, self.device
            )
        else:
            print("\n🔧 Loading standard CLIP encoders...")
            print("  [1/3] CLIP-L (openai)...")
            clip_l_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            clip_l_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
            clip_l_model.to(self.device).eval()

            print("  [2/3] CLIP-G (laion)...")
            clip_g_tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
            clip_g_model = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
            clip_g_model.to(self.device).eval()

        print("  [3/3] T5-XL (google/flan-t5-xl)...")
        t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        t5_model = T5EncoderModel.from_pretrained("google/flan-t5-xl")

        print(f"✓ All encoders loaded (clip_skip={self.config.clip_skip})")

        dataset = TextEmbeddingDataset(
            clip_texts=clip_texts,
            t5_texts=t5_texts,
            clip_l_tokenizer=clip_l_tokenizer,
            clip_l_model=clip_l_model,
            clip_g_tokenizer=clip_g_tokenizer,
            clip_g_model=clip_g_model,
            t5_tokenizer=t5_tokenizer,
            t5_model=t5_model,
            device=self.device,
            clip_max_length=self.config.modality_seq_lens['clip_l'],
            t5_max_length=self.config.modality_seq_lens['t5_xl_l'],
            clip_skip=self.config.clip_skip
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        if self.config.scheduler_type == 'onecycle' and self.scheduler is None:
            total_steps = len(dataloader) * self.config.num_epochs
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.3
            )

        return dataloader

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        modality_inputs = {
            'clip_l': batch['clip_l'].to(self.device),
            'clip_g': batch['clip_g'].to(self.device),
            't5_xl_l': batch['t5_xl_l'].to(self.device),
            't5_xl_g': batch['t5_xl_g'].to(self.device)
        }

        current_kl_beta = self._get_current_kl_beta()
        self.loss_fn.beta_kl = current_kl_beta

        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                reconstructions, mu, logvar, per_mod_mus = self.model(modality_inputs)
                fusion_params = self.model.get_fusion_params()
                alphas = fusion_params.get('alphas', None)
                projected_recons = self.model.project_for_cross_modal(reconstructions)
                loss, components = self.loss_fn(
                    inputs=modality_inputs,
                    reconstructions=reconstructions,
                    mu=mu,
                    logvar=logvar,
                    per_modality_mus=per_mod_mus,
                    alphas=alphas,
                    projected_recons=projected_recons,
                    return_components=True
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            reconstructions, mu, logvar, per_mod_mus = self.model(modality_inputs)
            fusion_params = self.model.get_fusion_params()
            alphas = fusion_params.get('alphas', None)
            projected_recons = self.model.project_for_cross_modal(reconstructions)
            loss, components = self.loss_fn(
                inputs=modality_inputs,
                reconstructions=reconstructions,
                mu=mu,
                logvar=logvar,
                per_modality_mus=per_mod_mus,
                alphas=alphas,
                projected_recons=projected_recons,
                return_components=True
            )
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

        if self.scheduler is not None and self.config.scheduler_type == 'onecycle':
            self.scheduler.step()

        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in components.items()}
        metrics['kl_beta'] = current_kl_beta

        betas = fusion_params.get('betas', {})
        for name, beta in betas.items():
            metrics[f'beta_{name}'] = torch.sigmoid(beta).item()

        return metrics

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with checkpoint pushing."""
        self.model.train()
        epoch_metrics = {}

        pbar = tqdm(dataloader, desc=f"🎵 Epoch {self.epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)
            self.global_step += 1
            self._last_loss = metrics['total']

            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

            pbar.set_postfix({
                'loss': f"{metrics['total']:.4f}",
                'r_l': f"{metrics.get('recon_clip_l', 0):.4f}",
                'kl': f"{metrics['kl']:.4f}"
            })

            if self.config.use_wandb and self.global_step % self.config.log_every == 0:
                wandb.log({
                    **{f'train/{k}': v for k, v in metrics.items()},
                    'train/step': self.global_step
                })

            # Save and push checkpoints at intervals
            if self.global_step % self.config.save_every == 0:
                ckpt_path = self.save_checkpoint(best=False)

                if self.config.push_checkpoints:
                    self._push_checkpoint_to_hub(ckpt_path, is_best=False)

        return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

    def train(self, dataloader: DataLoader):
        """Main training loop."""
        print(f"\n{'=' * 60}")
        print(f"🎵 Training VAE Lyra (Illustrious) for {self.config.num_epochs} epochs")
        print(f"   Starting from epoch {self.start_epoch}, step {self.global_step}")
        print(f"   Prompt source: {self.config.prompt_source}")
        print(f"   Resume optimizer: {self.config.resume_optimizer}")
        print(f"   Push checkpoints: {self.config.push_checkpoints}")
        print(f"{'=' * 60}")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            self.epoch = epoch
            metrics = self.train_epoch(dataloader)

            if self.scheduler is not None and self.config.scheduler_type == 'cosine':
                self.scheduler.step()

            print(f"\nEpoch {epoch}: loss={metrics['total']:.4f}")

            if metrics['total'] < self.best_loss:
                self.best_loss = metrics['total']
                ckpt_path = self.save_checkpoint(best=True)
                self._push_checkpoint_to_hub(ckpt_path, is_best=True)
                print(f"  ✨ New best: {self.best_loss:.4f}")

        print(f"\n✨ Training complete!")
        print(f"📤 Model: https://huggingface.co/{self.config.hf_repo}")

        if self.config.use_wandb:
            wandb.finish()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_lyra_trainer(
        num_samples: int = 10000,
        batch_size: int = 8,
        num_epochs: int = 100,
        push_to_hub: bool = True,
        hf_repo: str = "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious",
        hub_checkpoint_file: Optional[str] = None,
        model_name: str = "lyra",
        prompt_source: str = "booru",
        clip_skip: int = 2,
        illustrious_repo: str = "AbstractPhil/clips",
        clip_l_filename: Optional[str] = None,
        clip_g_filename: Optional[str] = None,
        auto_discover_clips: bool = True,
        use_summarizer: bool = True,
        summarizer_model: str = "qwen2.5-1.5b",
        summarizer_batch_size: int = 16,
        summary_separator: str = "¶",
        shuffle_tags_before_summary: bool = True,
        prompt_cache_dir: str = "./prompt_cache",
        use_prompt_cache: bool = True,
        save_prompt_cache: bool = True,
        resume_optimizer: bool = True,
        push_checkpoints: bool = True,
        export_safetensors: bool = True,
        # Coherent gender and T5 prefix
        use_coherent_gender: bool = True,
        generate_t5_prefix: bool = True,
        # Conduit options
        use_conduit: bool = True,
        conduit_top_n: int = 1000,
        conduit_sample_k: int = 10,
        conduit_sample_k_min: int = 5,
        conduit_sample_k_max: int = 15,
        conduit_position: str = "prepend",
        conduit_exclude_categories: Optional[List[str]] = None,
        # CSV paths
        danbooru_csv: Optional[str] = None,
        gelbooru_csv: Optional[str] = None,
        e621_csv: Optional[str] = None,
        rule34x_csv: Optional[str] = None,
        **kwargs
) -> VAELyraTrainer:
    """Create VAE Lyra trainer with all options."""

    if conduit_exclude_categories is None:
        conduit_exclude_categories = ["artist", "copyright", "character", "metadata"]

    config = VAELyraTrainerConfig(
        num_samples=num_samples,
        batch_size=batch_size,
        num_epochs=num_epochs,
        push_to_hub=push_to_hub,
        hf_repo=hf_repo,
        hub_checkpoint_file=hub_checkpoint_file,
        model_name=model_name,
        prompt_source=prompt_source,
        clip_skip=clip_skip,
        illustrious_repo=illustrious_repo,
        clip_l_filename=clip_l_filename,
        clip_g_filename=clip_g_filename,
        auto_discover_clips=auto_discover_clips,
        use_summarizer=use_summarizer,
        summarizer_model=summarizer_model,
        summarizer_batch_size=summarizer_batch_size,
        summary_separator=summary_separator,
        shuffle_tags_before_summary=shuffle_tags_before_summary,
        prompt_cache_dir=prompt_cache_dir,
        use_prompt_cache=use_prompt_cache,
        save_prompt_cache=save_prompt_cache,
        resume_optimizer=resume_optimizer,
        push_checkpoints=push_checkpoints,
        export_safetensors=export_safetensors,
        use_coherent_gender=use_coherent_gender,
        generate_t5_prefix=generate_t5_prefix,
        use_conduit=use_conduit,
        conduit_top_n=conduit_top_n,
        conduit_sample_k=conduit_sample_k,
        conduit_sample_k_min=conduit_sample_k_min,
        conduit_sample_k_max=conduit_sample_k_max,
        conduit_position=conduit_position,
        conduit_exclude_categories=conduit_exclude_categories,
        danbooru_csv=danbooru_csv,
        gelbooru_csv=gelbooru_csv,
        e621_csv=e621_csv,
        rule34x_csv=rule34x_csv,
        **kwargs
    )
    return VAELyraTrainer(config)


def generate_prompt_cache(
        num_samples: int = 10000,
        prompt_source: str = "booru",
        summarizer_model: str = "qwen2.5-1.5b",
        summarizer_batch_size: int = 16,
        cache_dir: str = "./prompt_cache",
        cache_name: Optional[str] = None,
        # Coherent gender and T5 prefix
        use_coherent_gender: bool = True,
        generate_t5_prefix: bool = True,
        # Conduit options
        use_conduit: bool = True,
        conduit_top_n: int = 1000,
        conduit_sample_k: int = 10,
        # CSV paths
        danbooru_csv: Optional[str] = None,
        gelbooru_csv: Optional[str] = None,
        e621_csv: Optional[str] = None,
        rule34x_csv: Optional[str] = None,
        device: str = "cuda",
        **kwargs
) -> Path:
    """Generate and cache prompts + summaries without training."""
    config = VAELyraTrainerConfig(
        num_samples=num_samples,
        prompt_source=prompt_source,
        use_summarizer=True,
        summarizer_model=summarizer_model,
        summarizer_batch_size=summarizer_batch_size,
        prompt_cache_dir=cache_dir,
        prompt_cache_name=cache_name,
        use_prompt_cache=False,
        save_prompt_cache=True,
        use_coherent_gender=use_coherent_gender,
        generate_t5_prefix=generate_t5_prefix,
        use_conduit=use_conduit,
        conduit_top_n=conduit_top_n,
        conduit_sample_k=conduit_sample_k,
        danbooru_csv=danbooru_csv,
        gelbooru_csv=gelbooru_csv,
        e621_csv=e621_csv,
        rule34x_csv=rule34x_csv,
        device=device,
        push_to_hub=False,
        auto_load_from_hub=False,
        **kwargs
    )

    trainer = VAELyraTrainer(config)
    tags_list, summaries, sources, t5_prefixes, people_counts = trainer._load_or_generate_prompts(num_samples)

    cache_path = trainer._get_cache_path()
    print(f"\n✓ Cache saved to: {cache_path}")

    return cache_path


def load_lyra_from_hub(
        repo_id: str = "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious",
        device: str = "cuda",
        use_safetensors: bool = True,
        checkpoint_file: Optional[str] = None,
        model_name: Optional[str] = None
) -> MultiModalVAE:
    """
    Load VAE Lyra from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID
        device: Target device
        use_safetensors: Try to load safetensors first (faster)
        checkpoint_file: Specific checkpoint file to load (e.g., "checkpoint_lyra_5000.pt")
        model_name: Model name for safetensors lookup (reads from config.json if not provided)
    """
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", repo_type="model")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Get model_name from config if not provided
    if model_name is None:
        model_name = config_dict.get('model_name', 'lyra')

    vae_config = MultiModalVAEConfig(
        modality_dims=config_dict.get('modality_dims'),
        modality_seq_lens=config_dict.get('modality_seq_lens'),
        binding_config=config_dict.get('binding_config'),
        latent_dim=config_dict.get('latent_dim', 2048),
        fusion_strategy=config_dict.get('fusion_strategy', 'adaptive_cantor'),
        cantor_depth=config_dict.get('cantor_depth', 8),
        cantor_local_window=config_dict.get('cantor_local_window', 3)
    )

    model = MultiModalVAE(vae_config)

    # If specific checkpoint requested, load that
    if checkpoint_file:
        model_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_file, repo_type="model")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            step = checkpoint.get('global_step', '?')
        else:
            model.load_state_dict(checkpoint)
            step = '?'

        print(f"✓ Loaded {checkpoint_file} from {repo_id} @ step {step}")
        model.to(device)
        return model

    # Try safetensors first
    if use_safetensors:
        try:
            weights_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"weights/{model_name}_best.safetensors",
                repo_type="model"
            )
            state_dict = load_safetensors(weights_path)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded from {repo_id} (safetensors: {model_name}_best)")
            model.to(device)
            return model
        except Exception:
            pass  # Fall back to .pt

    # Fall back to model.pt
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pt", repo_type="model")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded from {repo_id} @ step {checkpoint.get('global_step', '?')}")

    model.to(device)
    return model


# ============================================================================
# MAIN
# ============================================================================

#if __name__ == "__main__":
#    config = VAELyraTrainerConfig(
#        # Model naming
#        model_name="lyra_illustrious",
#
#        # Prompt source
#        prompt_source="booru",
#        booru_ratio=0.7,
#        synthetic_ratio=0.15,
#        laion_ratio=0.15,
#        danbooru_csv=None,
#        gelbooru_csv=None,
#        e621_csv=None,
#        rule34x_csv=None,
#
#        # Summarization
#        use_summarizer=True,
#        summarizer_model="qwen2.5-1.5b",
#        summarizer_batch_size=32,
#        summary_separator="¶",
#        shuffle_tags_before_summary=True,
#        summarizer_max_new_tokens=64,
#
#        # Prompt caching
#        prompt_cache_dir="./prompt_cache",
#        use_prompt_cache=True,
#        save_prompt_cache=True,
#
#        # CLIP configuration
#        use_illustrious_clips=True,
#        illustrious_repo="AbstractPhil/clips",
#        clip_l_filename=None,
#        clip_g_filename=None,
#        auto_discover_clips=True,
#        clip_skip=2,
#
#        # Training
#        num_samples=10000,
#        batch_size=8,
#        num_epochs=100,
#        learning_rate=1e-4,
#
#        # Resume behavior
#        resume_optimizer=True,
#        hub_checkpoint_file=None,  # None = model.pt, or specify e.g. "checkpoint_lyra_illustrious_3000.pt"
#
#        # Hub settings
#        push_to_hub=True,
#        push_checkpoints=True,
#        export_safetensors=True,
#        hf_repo="AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious",
#
#        use_wandb=False
#    )
#
#    trainer = VAELyraTrainer(config)
#    dataloader = trainer.prepare_data()
#    trainer.train(dataloader)