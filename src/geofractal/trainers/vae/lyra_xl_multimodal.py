# geofractal/trainers/vae_lyra/lyra_xl_multimodal_illustrious.py

"""
Trainer for VAE Lyra - Multi-Modal Variational Autoencoder
Using Custom Illustrious CLIP-L + CLIP-G + T5-XL for SDXL Compatibility

Downloads custom CLIP weights from AbstractPhil/clips:
- IllustriousXL20_v20_clip_l.safetensors
- IllustriousXL20_v20_clip_g.safetensors

Install via:
    !pip install git+https://github.com/AbstractEyes/geofractal.git
    !pip install safetensors

Usage:
    from lyra_xl_multimodal_illustrious import (
        VAELyraTrainer, VAELyraTrainerConfig
    )
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
from huggingface_hub import HfApi, hf_hub_download, create_repo
from tqdm.auto import tqdm
import wandb
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import requests
import random
from collections import Counter

from geofractal.model.vae.vae_lyra import (
    MultiModalVAE,
    MultiModalVAEConfig,
    MultiModalVAELoss,
    FusionStrategy
)
from geovocab2.data.prompt.symbolic_tree import SynthesisSystem


# ============================================================================
# CUSTOM CLIP LOADING FROM SAFETENSORS
# ============================================================================

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

    # Get tokenizer from base model
    tokenizer = CLIPTokenizer.from_pretrained(base_model)

    # Initialize model architecture
    model = CLIPTextModel.from_pretrained(base_model)

    # Load custom weights
    print(f"    Loading Illustrious weights from {Path(safetensors_path).name}...")
    state_dict = load_safetensors(safetensors_path)

    # Debug: show some keys to understand structure
    sample_keys = list(state_dict.keys())[:5]
    print(f"    Sample keys: {sample_keys}")

    # Try to load - may need key remapping
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"    ⚠️  Missing keys: {len(missing)} (may be expected)")
    if unexpected:
        print(f"    ⚠️  Unexpected keys: {len(unexpected)}")

        # If keys have different prefix, try remapping
        if unexpected and not missing:
            print("    Attempting key remapping...")
            # Common pattern: 'text_model.' prefix
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

    # Get tokenizer from base model
    tokenizer = CLIPTokenizer.from_pretrained(base_model)

    # Initialize model architecture
    model = CLIPTextModelWithProjection.from_pretrained(base_model)

    # Load custom weights
    print(f"    Loading Illustrious weights from {Path(safetensors_path).name}...")
    state_dict = load_safetensors(safetensors_path)

    # Debug: show some keys
    sample_keys = list(state_dict.keys())[:5]
    print(f"    Sample keys: {sample_keys}")

    # Try to load
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"    ⚠️  Missing keys: {len(missing)} (may be expected)")
    if unexpected:
        print(f"    ⚠️  Unexpected keys: {len(unexpected)}")

        # Try remapping if needed
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


def download_illustrious_clips(
        repo_id: str = "AbstractPhil/clips",
        clip_l_filename: str = "IllustriousXL20_v20_clip_l.safetensors",
        clip_g_filename: str = "IllustriousXL20_v20_clip_g.safetensors"
) -> Tuple[str, str]:
    """
    Download Illustrious CLIP weights from HuggingFace.

    Returns:
        Tuple of (clip_l_path, clip_g_path)
    """
    print(f"\n📥 Downloading Illustrious CLIP weights from {repo_id}...")

    clip_l_path = hf_hub_download(
        repo_id=repo_id,
        filename=clip_l_filename,
        repo_type="model"
    )
    print(f"  ✓ CLIP-L: {clip_l_path}")

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
    clip_l_filename: str = "IllustriousXL20_v20_clip_l.safetensors"
    clip_g_filename: str = "IllustriousXL20_v20_clip_g.safetensors"

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

    # Data
    num_samples: int = 10000
    synthetic_ratio: float = 0.15

    # Checkpointing
    checkpoint_dir: str = './checkpoints_lyra_illustrious'
    save_every: int = 1000
    keep_last_n: int = 3

    # HuggingFace Hub
    hf_repo: str = "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious"
    push_to_hub: bool = True
    push_every: int = 2000
    auto_load_from_hub: bool = True

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


# ============================================================================
# DATASET
# ============================================================================

class TextEmbeddingDataset(Dataset):
    """Dataset that generates CLIP-L, CLIP-G, and T5-XL embeddings on-the-fly."""

    def __init__(
            self,
            texts: List[str],
            clip_l_tokenizer: CLIPTokenizer,
            clip_l_model: CLIPTextModel,
            clip_g_tokenizer: CLIPTokenizer,
            clip_g_model: CLIPTextModelWithProjection,
            t5_tokenizer: T5Tokenizer,
            t5_model: T5EncoderModel,
            device: str = 'cuda',
            clip_max_length: int = 77,
            t5_max_length: int = 512
    ):
        self.texts = texts
        self.clip_l_tokenizer = clip_l_tokenizer
        self.clip_l_model = clip_l_model
        self.clip_g_tokenizer = clip_g_tokenizer
        self.clip_g_model = clip_g_model
        self.t5_tokenizer = t5_tokenizer
        self.t5_model = t5_model
        self.device = device
        self.clip_max_length = clip_max_length
        self.t5_max_length = t5_max_length

        self.clip_l_model.to(device).eval()
        self.clip_g_model.to(device).eval()
        self.t5_model.to(device).eval()

    def __len__(self):
        return len(self.texts)

    @torch.no_grad()
    def __getitem__(self, idx):
        text = self.texts[idx]

        # CLIP-L embedding
        clip_l_tokens = self.clip_l_tokenizer(
            text,
            max_length=self.clip_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        clip_l_output = self.clip_l_model(**clip_l_tokens)
        clip_l_embed = clip_l_output.last_hidden_state.squeeze(0)

        # CLIP-G embedding
        clip_g_tokens = self.clip_g_tokenizer(
            text,
            max_length=self.clip_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        clip_g_output = self.clip_g_model(**clip_g_tokens)
        clip_g_embed = clip_g_output.last_hidden_state.squeeze(0)

        # T5-XL embeddings
        t5_tokens = self.t5_tokenizer(
            text,
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
            'text': text
        }


# ============================================================================
# TRAINER
# ============================================================================

class VAELyraTrainer:
    """Trainer for VAE Lyra with Illustrious CLIP weights."""

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
        print(f"   Fusion: {config.fusion_strategy}")

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

        print("Initializing prompt generation...")
        self.prompt_gen = SynthesisSystem(seed=config.seed)
        self.flavors = self._load_flavors()
        self.used_prompts = []
        self.prompt_sources = []

        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=asdict(config),
                name=f"lyra_illustrious_{config.fusion_strategy}",
                resume="allow"
            )

    def _init_hf_repo(self):
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
        try:
            print(f"🔍 Checking for existing model: {self.config.hf_repo}")
            try:
                model_path = hf_hub_download(
                    repo_id=self.config.hf_repo,
                    filename="model.pt",
                    repo_type="model"
                )
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model = self._build_model()
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer = self._build_optimizer()
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.global_step = checkpoint.get('global_step', 0)
                self.epoch = checkpoint.get('epoch', 0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))
                if 'scheduler_state_dict' in checkpoint and self.config.use_scheduler:
                    self.scheduler = self._build_scheduler()
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✓ Resumed from step {self.global_step}, epoch {self.epoch}")
                return True
            except Exception as e:
                print(f"   No existing model found: {e}")
                return False
        except Exception as e:
            print(f"⚠️  Could not access HF Hub: {e}")
            return False

    def _push_to_hub(self, is_best: bool = False):
        if not self.config.push_to_hub:
            return
        try:
            print(f"\n📤 Pushing to HF Hub...", end=" ", flush=True)
            temp_path = Path(self.config.checkpoint_dir) / "temp_upload.pt"
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'global_step': self.global_step,
                'epoch': self.epoch,
                'best_loss': self.best_loss,
                'config': asdict(self.config)
            }
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            torch.save(checkpoint, temp_path)
            self.hf_api.upload_file(
                path_or_fileobj=str(temp_path),
                path_in_repo="model.pt",
                repo_id=self.config.hf_repo,
                repo_type="model",
                commit_message=f"Step {self.global_step}: loss={self.best_loss:.4f}" if is_best else f"Step {self.global_step}"
            )
            config_path = Path(self.config.checkpoint_dir) / "config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            self.hf_api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="config.json",
                repo_id=self.config.hf_repo,
                repo_type="model",
                commit_message=f"Config @ step {self.global_step}"
            )
            if is_best:
                self._create_model_card()
            temp_path.unlink()
            print(f"✓")
        except Exception as e:
            print(f"✗ {e}")

    def _create_model_card(self):
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

        model_card = f"""---
tags:
- vae
- multimodal
- clip
- t5
- sdxl
- illustrious
- adaptive-cantor
license: mit
---

# VAE Lyra 🎵 - Illustrious Edition

Multi-modal VAE trained with **Illustrious XL 2.0 CLIP weights**.

## CLIP Encoders

Uses custom fine-tuned CLIP weights from `AbstractPhil/clips`:
- `IllustriousXL20_v20_clip_l.safetensors` (768d)
- `IllustriousXL20_v20_clip_g.safetensors` (1280d)

## Model Details

- **Fusion Strategy**: {self.config.fusion_strategy}
- **Latent Dimension**: {self.config.latent_dim}
- **Training Steps**: {self.global_step:,}
- **Best Loss**: {self.best_loss:.4f}
{fusion_params_str}

## Usage

```python
from lyra_xl_multimodal_illustrious import load_lyra_from_hub

model = load_lyra_from_hub("{self.config.hf_repo}")
model.eval()

# Inputs (use Illustrious CLIP encoders for best results)
inputs = {{
    "clip_l": clip_l_embeddings,     # [batch, 77, 768]
    "clip_g": clip_g_embeddings,     # [batch, 77, 1280]
    "t5_xl_l": t5_xl_embeddings,     # [batch, 512, 2048]
    "t5_xl_g": t5_xl_embeddings      # [batch, 512, 2048]
}}

recons, mu, logvar, _ = model(inputs, target_modalities=["clip_l", "clip_g"])
```
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

    def _build_model(self) -> nn.Module:
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
        return MultiModalVAELoss(
            beta_kl=self.config.beta_kl,
            beta_reconstruction=self.config.beta_reconstruction,
            beta_cross_modal=self.config.beta_cross_modal,
            beta_alpha_regularization=self.config.beta_alpha_regularization,
            recon_type=self.config.recon_type,
            modality_weights=self.config.modality_recon_weights
        )

    def _build_scheduler(self):
        if self.config.scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.1
            )
        return None

    def _load_flavors(self):
        print("Loading LAION flavors...")
        try:
            r = requests.get(
                "https://raw.githubusercontent.com/pharmapsychotic/clip-interrogator/main/clip_interrogator/data/flavors.txt",
                timeout=30
            )
            flavors = [line.strip() for line in r.text.split('\n') if line.strip()]
            print(f"✓ Loaded {len(flavors):,} flavors")
            return flavors
        except Exception as e:
            print(f"⚠️  Fallback flavors: {e}")
            return ["a beautiful landscape", "abstract art", "detailed portrait"]

    def _generate_prompt(self):
        if random.random() < self.config.synthetic_ratio:
            complexity = random.choice([2, 3, 4, 5])
            prompt = self.prompt_gen.synthesize(complexity=complexity)['text']
            return prompt, "synthetic"
        return random.choice(self.flavors), "laion"

    def _get_current_kl_beta(self) -> float:
        if not self.config.use_kl_annealing:
            return self.config.beta_kl
        if self.epoch >= self.config.kl_anneal_epochs:
            return self.config.beta_kl
        progress = self.epoch / self.config.kl_anneal_epochs
        return self.config.kl_start_beta + (self.config.beta_kl - self.config.kl_start_beta) * progress

    def prepare_data(self, num_samples: Optional[int] = None) -> DataLoader:
        """Prepare dataset with Illustrious CLIP + T5-XL encoders."""
        num_samples = num_samples or self.config.num_samples

        print(f"\n🎼 Generating {num_samples:,} prompts...")

        texts, sources = [], []
        for _ in tqdm(range(num_samples), desc="Generating"):
            prompt, source = self._generate_prompt()
            texts.append(prompt)
            sources.append(source)

        self.used_prompts = texts
        self.prompt_sources = sources

        source_counts = Counter(sources)
        print(f"\nDistribution: {dict(source_counts)}")
        print(f"Samples: {texts[:3]}")

        # Download custom CLIP weights
        if self.config.use_illustrious_clips:
            clip_l_path, clip_g_path = download_illustrious_clips(
                repo_id=self.config.illustrious_repo,
                clip_l_filename=self.config.clip_l_filename,
                clip_g_filename=self.config.clip_g_filename
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
            # Fallback to standard CLIP
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

        print(f"✓ All encoders loaded")

        dataset = TextEmbeddingDataset(
            texts=texts,
            clip_l_tokenizer=clip_l_tokenizer,
            clip_l_model=clip_l_model,
            clip_g_tokenizer=clip_g_tokenizer,
            clip_g_model=clip_g_model,
            t5_tokenizer=t5_tokenizer,
            t5_model=t5_model,
            device=self.device,
            clip_max_length=self.config.modality_seq_lens['clip_l'],
            t5_max_length=self.config.modality_seq_lens['t5_xl_l']
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
        self.model.train()
        epoch_metrics = {}

        pbar = tqdm(dataloader, desc=f"🎵 Epoch {self.epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)
            self.global_step += 1

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

            if self.global_step % self.config.push_every == 0:
                self._push_to_hub()

            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()

        return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

    def train(self, dataloader: DataLoader):
        print(f"\n{'=' * 60}")
        print(f"🎵 Training VAE Lyra (Illustrious) for {self.config.num_epochs} epochs")
        print(f"{'=' * 60}")

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            metrics = self.train_epoch(dataloader)

            if self.scheduler is not None and self.config.scheduler_type == 'cosine':
                self.scheduler.step()

            print(f"\nEpoch {epoch}: loss={metrics['total']:.4f}")

            if metrics['total'] < self.best_loss:
                self.best_loss = metrics['total']
                self.save_checkpoint(best=True)
                self._push_to_hub(is_best=True)
                print(f"  ✨ New best: {self.best_loss:.4f}")

        print(f"\n✨ Training complete!")
        print(f"📤 Model: https://huggingface.co/{self.config.hf_repo}")

        if self.config.use_wandb:
            wandb.finish()

    def save_checkpoint(self, best: bool = False):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'best_loss': self.best_loss,
            'used_prompts': self.used_prompts,
            'prompt_sources': self.prompt_sources
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if best:
            path = Path(self.config.checkpoint_dir) / 'best_model.pt'
        else:
            path = Path(self.config.checkpoint_dir) / f'checkpoint_{self.global_step}.pt'

        torch.save(checkpoint, path)
        if not best:
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            [f for f in checkpoint_dir.glob('checkpoint_*.pt')],
            key=lambda x: int(x.stem.split('_')[-1])
        )
        if len(checkpoints) > self.config.keep_last_n:
            for ckpt in checkpoints[:-self.config.keep_last_n]:
                ckpt.unlink()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_lyra_trainer(
        num_samples: int = 10000,
        batch_size: int = 8,
        num_epochs: int = 100,
        push_to_hub: bool = True,
        hf_repo: str = "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious",
        **kwargs
) -> VAELyraTrainer:
    """Create VAE Lyra trainer with Illustrious CLIP."""
    config = VAELyraTrainerConfig(
        num_samples=num_samples,
        batch_size=batch_size,
        num_epochs=num_epochs,
        push_to_hub=push_to_hub,
        hf_repo=hf_repo,
        **kwargs
    )
    return VAELyraTrainer(config)


def load_lyra_from_hub(
        repo_id: str = "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious",
        device: str = "cuda"
) -> MultiModalVAE:
    """Load VAE Lyra from HuggingFace Hub."""
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pt", repo_type="model")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", repo_type="model")

    with open(config_path) as f:
        config_dict = json.load(f)

    vae_config = MultiModalVAEConfig(
        modality_dims=config_dict.get('modality_dims'),
        modality_seq_lens=config_dict.get('modality_seq_lens'),
        binding_config=config_dict.get('binding_config'),
        latent_dim=config_dict.get('latent_dim', 2048),
        fusion_strategy=config_dict.get('fusion_strategy', 'adaptive_cantor'),
        cantor_depth=config_dict.get('cantor_depth', 8),
        cantor_local_window=config_dict.get('cantor_local_window', 3)
    )

    checkpoint = torch.load(model_path, map_location=device)
    model = MultiModalVAE(vae_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"✓ Loaded from {repo_id} @ step {checkpoint.get('global_step', '?')}")
    return model