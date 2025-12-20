"""
BEATRIX TRAINER
===============

Complete training system for Beatrix diffusion.

Contains:
    - Encoder wrappers (FluxVAEWrapper, DINOWrapper, FluxVAEDecoder)
    - FluxAEStream (specialized EncoderStream preserving raw latent)
    - BeatrixDenoiser (trainable component)
    - Beatrix (full model wrapping head router + denoiser)
    - BeatrixTrainer (training manager)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    AGATHA HEAD ROUTER                        │
    │                                                              │
    │  Image ──→ flux_ae stream ──→ EncoderMail                   │
    │            │                   ├─ content: projected [B,L,D] │
    │            │                   └─ metadata['raw']: [B,C,H,W] │
    │            │                                                 │
    │  Image ──→ dino stream ─────→ EncoderMail                   │
    │                                                              │
    │            All streams ──→ AdaptiveFusion ──→ mail.fused    │
    └─────────────────────────────────────────────────────────────┘
                                       ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  z_clean = mail['flux_ae'].metadata['raw']                  │
    │  z_noisy = z_clean + σ·ε                                    │
    │                                                              │
    │  mail.fused ──→ BeatrixCollective ──→ Tower Forces          │
    │                                             ↓                │
    │  (z_noisy, x_ref, forces) ──→ Oscillator ──→ z_pred         │
    │                                                              │
    │  Loss = ||z_pred - z_clean||²                               │
    └─────────────────────────────────────────────────────────────┘

Author: AbstractPhil + Claude
Date: December 2024
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from geofractal.router.prefab.agatha.head_router import (
    AgathaHeadRouter,
    create_agatha_head,
    HeadMail,
    EncoderMail,
    EncoderStream,
    StreamType,
)
from geofractal.router.prefab.agatha.beatrix_oscillator import (
    BeatrixOscillator,
    ScheduleType,
)
from geofractal.router.prefab.agatha.beatrix_collective import (
    BeatrixCollective,
    BeatrixCollectiveConfig,
)


# =============================================================================
# ENCODER WRAPPERS
# =============================================================================

class FluxVAEWrapper(nn.Module):
    """
    Wraps Flux VAE encoder for head router integration.
    Returns [B, C, H, W] latent.
    """

    def __init__(self, vae, scale: float = 1.0, shift: float = 0.0):
        super().__init__()
        self.vae = vae
        self.scale = scale
        self.shift = shift

    def forward(self, x: Tensor) -> Tensor:
        """Encode [B, 3, H, W] image in [0,1] to latent."""
        x = 2 * x - 1  # Scale to [-1, 1]
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
        return (latent - self.shift) * self.scale


class FluxVAEDecoder(nn.Module):
    """Wraps Flux VAE decoder for image generation."""

    def __init__(self, vae, scale: float = 1.0, shift: float = 0.0):
        super().__init__()
        self.vae = vae
        self.scale = scale
        self.shift = shift

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent to [B, 3, H, W] image in [0,1]."""
        z = z / self.scale + self.shift
        with torch.no_grad():
            image = self.vae.decode(z).sample
        return ((image + 1) / 2).clamp(0, 1)


class DINOWrapper(nn.Module):
    """Wraps DINO for head router integration."""

    def __init__(self, dino):
        super().__init__()
        self.dino = dino
        self.hidden_size = 768

    def forward(self, x: Tensor) -> Tensor:
        """Extract [B, 768] CLS token from [B, 3, H, W] image."""
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            return self.dino(x).last_hidden_state[:, 0]


# =============================================================================
# FLUX AE STREAM (preserves raw latent in metadata)
# =============================================================================

class FluxAEStream(EncoderStream):
    """
    Specialized encoder stream for Flux VAE.

    Stores raw latent [B, C, H, W] in EncoderMail.metadata['raw']
    while projecting content [B, L, D] for fusion.
    """

    def __init__(
        self,
        name: str = 'flux_ae',
        latent_channels: int = 32,
        latent_size: int = 64,
        embed_dim: int = 256,
        fingerprint_dim: int = 64,
        **kwargs,
    ):
        latent_dim = latent_channels * latent_size * latent_size

        super().__init__(
            name=name,
            stream_type=StreamType.IMAGE,
            embed_dim=latent_dim,
            fingerprint_dim=fingerprint_dim,
            project_dim=embed_dim,
            frozen=True,
            **kwargs,
        )

        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.latent_dim = latent_dim

    def forward(self, x: Any, **kwargs) -> EncoderMail:
        """Extract and create mail with raw latent in metadata."""
        raw_latent = self.extract(x, **kwargs)

        if raw_latent.dim() == 4:
            B, C, H, W = raw_latent.shape
            features = raw_latent.reshape(B, -1)
        else:
            features = raw_latent
            B = features.shape[0]
            raw_latent = features.reshape(B, self.latent_channels,
                                          self.latent_size, self.latent_size)

        features = features.unsqueeze(1)

        if self.projection is not None:
            features = self.projection(features.to(self.projection.weight.dtype))

        B = features.shape[0]
        fingerprint = self.address.fingerprint.unsqueeze(0).expand(B, -1)
        fingerprint = fingerprint + self.stream_identity

        return EncoderMail(
            content=features,
            fingerprint=fingerprint,
            stream_type=self.stream_type,
            source=self.name,
            metadata={
                'raw': raw_latent,
                'latent_shape': (self.latent_channels, self.latent_size, self.latent_size),
            },
        )


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BeatrixConfig:
    """Configuration for Beatrix."""

    # Latent space (Flux2 VAE)
    latent_channels: int = 32
    latent_height: int = 64
    latent_width: int = 64

    # Head router
    head_embed_dim: int = 256
    fingerprint_dim: int = 64
    fusion_type: str = 'adaptive'

    # Collective
    geometric_types: List[str] = field(default_factory=lambda: [
        'cantor', 'beatrix', 'simplex', 'helix'
    ])
    conv_types: List[str] = field(default_factory=lambda: [
        'wide_resnet', 'frequency'
    ])
    num_theta_probes: int = 4
    use_signed_pairs: bool = True

    # Oscillator
    num_steps: int = 50
    num_training_steps: int = 10
    beta_range: Tuple[float, float] = (0.1, 2.0)
    omega_range: Tuple[float, float] = (1.0, 0.1)
    kappa_range: Tuple[float, float] = (1.0, 0.5)
    gamma_range: Tuple[float, float] = (1.0, 0.0)
    schedule_type: str = "tesla_369"

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    sigma_min: float = 0.002
    sigma_max: float = 80.0

    @property
    def latent_dim(self) -> int:
        return self.latent_channels * self.latent_height * self.latent_width

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return (self.latent_channels, self.latent_height, self.latent_width)


# =============================================================================
# SIGMA EMBEDDING
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for sigma/timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# =============================================================================
# BEATRIX DENOISER
# =============================================================================

class BeatrixDenoiser(nn.Module):
    """
    Trainable denoiser component.

    Takes HeadMail from AgathaHeadRouter and denoises z_noisy.
    """

    def __init__(self, config: BeatrixConfig):
        super().__init__()
        self.config = config

        # Tower collective (WideRouter)
        collective_config = BeatrixCollectiveConfig(
            dim=config.head_embed_dim,
            fingerprint_dim=config.fingerprint_dim,
            geometric_types=config.geometric_types,
            conv_types=config.conv_types,
            num_theta_probes=config.num_theta_probes,
            use_signed_pairs=config.use_signed_pairs,
        )
        self.collective = BeatrixCollective(collective_config)

        # Oscillator
        num_tower_pairs = len(config.geometric_types) + len(config.conv_types)
        self.oscillator = BeatrixOscillator(
            manifold_dim=config.latent_dim,
            tower_dim=config.head_embed_dim,
            num_tower_pairs=num_tower_pairs,
            num_theta_probes=config.num_theta_probes,
            beta_start=config.beta_range[0],
            beta_end=config.beta_range[1],
            omega_start=config.omega_range[0],
            omega_end=config.omega_range[1],
            kappa_start=config.kappa_range[0],
            kappa_end=config.kappa_range[1],
            gamma_start=config.gamma_range[0],
            gamma_end=config.gamma_range[1],
            kappa_schedule=ScheduleType(config.schedule_type),
        )

        # Projections
        self.anchor_proj = nn.Sequential(
            nn.Linear(config.head_embed_dim, config.head_embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.head_embed_dim * 2, config.latent_dim),
        )

        self.guidance_proj = nn.Sequential(
            nn.Linear(config.head_embed_dim, config.head_embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.head_embed_dim * 2, config.latent_dim),
        )

        self.sigma_embed = nn.Sequential(
            SinusoidalPosEmb(config.head_embed_dim),
            nn.Linear(config.head_embed_dim, config.head_embed_dim),
            nn.GELU(),
            nn.Linear(config.head_embed_dim, config.head_embed_dim),
        )

    def forward(
        self,
        z_noisy: Tensor,
        mail: HeadMail,
        sigma: Tensor,
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """Denoise z_noisy using HeadMail conditioning."""
        B = z_noisy.shape[0]
        num_steps = num_steps or self.config.num_training_steps

        sigma_emb = self.sigma_embed(sigma)

        fused_cond = mail.fused + sigma_emb if mail.fused is not None else sigma_emb

        x_ref = self.anchor_proj(fused_cond)

        guidance_flat = None
        if 'dino' in mail.streams:
            dino_content = mail.streams['dino'].content
            if dino_content.dim() == 3:
                dino_content = dino_content.mean(dim=1)
            guidance_flat = self.guidance_proj(dino_content + sigma_emb)

        collective_result = self.collective(fused_cond, return_all=True)
        tower_outputs = collective_result['outputs']
        fingerprint = collective_result['fingerprint']

        z_flat = z_noisy.reshape(B, -1)

        z_pred_flat = self.oscillator(
            x_init=z_flat,
            x_ref=x_ref,
            tower_outputs=tower_outputs,
            guidance=guidance_flat,
            state_fingerprint=fingerprint,
            num_steps=num_steps,
        )

        return z_pred_flat.reshape(B, *self.config.latent_shape)

    def network_to(self, device: torch.device):
        self.to(device)
        self.collective.network_to(device)
        return self


# =============================================================================
# BEATRIX MODEL
# =============================================================================

class Beatrix(nn.Module):
    """
    Complete Beatrix system: AgathaHeadRouter + BeatrixDenoiser.
    """

    def __init__(
        self,
        config: BeatrixConfig,
        head: Optional[AgathaHeadRouter] = None,
    ):
        super().__init__()
        self.config = config

        self.head = head or AgathaHeadRouter(
            name='agatha_head',
            embed_dim=config.head_embed_dim,
            fingerprint_dim=config.fingerprint_dim,
            fusion_type=config.fusion_type,
        )

        self.denoiser = BeatrixDenoiser(config)
        self.vae_decoder: Optional[FluxVAEDecoder] = None

    def attach_flux_ae(
        self,
        encoder: nn.Module,
        extract_fn: Optional[Callable] = None,
    ) -> 'Beatrix':
        """Attach Flux VAE using specialized FluxAEStream."""
        stream = FluxAEStream(
            name='flux_ae',
            latent_channels=self.config.latent_channels,
            latent_size=self.config.latent_height,
            embed_dim=self.config.head_embed_dim,
            fingerprint_dim=self.config.fingerprint_dim,
        )
        stream.attach_encoder(encoder, extract_fn)

        self.head.streams['flux_ae'] = stream
        self.head.attach('flux_ae', stream)
        self.head._rebuild_fusion()

        return self

    def attach_encoder(
        self,
        name: str,
        encoder: nn.Module,
        embed_dim: int,
        stream_type: StreamType = StreamType.GUIDANCE,
        extract_fn: Optional[Callable] = None,
        frozen: bool = True,
    ) -> 'Beatrix':
        """Attach any encoder to head router."""
        self.head.attach_encoder(
            name=name,
            encoder=encoder,
            embed_dim=embed_dim,
            stream_type=stream_type,
            extract_fn=extract_fn,
            frozen=frozen,
        )
        return self

    def set_vae_decoder(self, decoder: FluxVAEDecoder) -> 'Beatrix':
        """Set VAE decoder for image generation."""
        self.vae_decoder = decoder
        return self

    def forward(
        self,
        inputs: Dict[str, Tensor],
        sigma: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Tuple[Tensor, HeadMail]:
        """Forward pass for training."""
        mail = self.head(inputs)

        if 'flux_ae' not in mail.streams:
            raise ValueError("flux_ae stream required")

        z_clean = mail.streams['flux_ae'].metadata.get('raw')
        if z_clean is None:
            raise ValueError("flux_ae missing raw latent - use attach_flux_ae()")

        B = z_clean.shape[0]
        device = z_clean.device

        if sigma is None:
            sigma = torch.rand(B, device=device) * (
                self.config.sigma_max - self.config.sigma_min
            ) + self.config.sigma_min

        noise = torch.randn_like(z_clean)
        z_noisy = z_clean + sigma.view(-1, 1, 1, 1) * noise

        z_pred = self.denoiser(z_noisy, mail, sigma, num_steps)

        return z_pred, mail

    def compute_loss(self, inputs: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """Compute training loss."""
        z_pred, mail = self.forward(inputs)
        z_clean = mail.streams['flux_ae'].metadata['raw']

        loss = F.mse_loss(z_pred, z_clean)

        with torch.no_grad():
            mse = loss.item()
            psnr = -10 * math.log10(mse) if mse > 0 else float('inf')

        return loss, {'loss': mse, 'psnr': psnr}

    @torch.no_grad()
    def sample(
        self,
        inputs: Dict[str, Tensor],
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """Generate from noise."""
        num_steps = num_steps or self.config.num_steps

        mail = self.head(inputs)
        B = mail.batch_size
        device = mail.fused.device if mail.fused is not None else next(self.parameters()).device

        z = torch.randn(B, *self.config.latent_shape, device=device) * self.config.sigma_max

        sigmas = torch.linspace(
            self.config.sigma_max, self.config.sigma_min,
            num_steps + 1, device=device
        )

        for i in range(num_steps):
            sigma = sigmas[i].expand(B)
            sigma_next = sigmas[i + 1].expand(B)

            z_denoised = self.denoiser(z, mail, sigma, num_steps=1)

            d = (z - z_denoised) / sigma.view(-1, 1, 1, 1)
            z = z_denoised + sigma_next.view(-1, 1, 1, 1) * d

        return z

    @torch.no_grad()
    def generate(
        self,
        inputs: Dict[str, Tensor],
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """Generate images (requires VAE decoder)."""
        z = self.sample(inputs, num_steps)

        if self.vae_decoder is not None:
            return self.vae_decoder(z)
        return z

    def network_to(self, device: torch.device):
        self.to(device)
        self.head.to(device)
        self.denoiser.network_to(device)
        if self.vae_decoder is not None:
            self.vae_decoder.to(device)
        return self

    def trainable_parameters(self):
        """Only denoiser parameters (encoders stay frozen)."""
        return self.denoiser.parameters()


# =============================================================================
# TRAINER
# =============================================================================

class BeatrixTrainer:
    """Training manager for Beatrix."""

    def __init__(
        self,
        model: Beatrix,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
    ):
        self.model = model
        self.device = torch.device(device)
        self.gradient_clip = gradient_clip

        model.network_to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        self.global_step = 0
        self.best_loss = float('inf')

    def train_step(self, inputs: Dict[str, Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        loss, metrics = self.model.compute_loss(inputs)
        loss.backward()

        if self.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(),
                self.gradient_clip,
            )
            metrics['grad_norm'] = grad_norm.item()

        self.optimizer.step()
        self.global_step += 1

        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {}

        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)

            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'psnr': f"{metrics['psnr']:.1f}dB",
                })

        avg = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

        if avg['loss'] < self.best_loss:
            self.best_loss = avg['loss']
            avg['is_best'] = True

        return avg

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 100,
        save_dir: str = "./checkpoints",
        save_every: int = 10,
    ):
        """Full training loop."""
        Path(save_dir).mkdir(exist_ok=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            metrics = self.train_epoch(dataloader)

            print(f"  Loss: {metrics['loss']:.6f}, PSNR: {metrics['psnr']:.1f}dB")
            if metrics.get('is_best'):
                print("  ★ New best!")
                self.save(f"{save_dir}/beatrix_best.pt")

            scheduler.step()

            if (epoch + 1) % save_every == 0:
                self.save(f"{save_dir}/beatrix_epoch_{epoch + 1}.pt")

        self.save(f"{save_dir}/beatrix_final.pt")

    def save(self, path: str):
        """Save checkpoint."""
        torch.save({
            'denoiser': self.model.denoiser.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
        }, path)
        print(f"Saved: {path}")

    def load(self, path: str):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.denoiser.load_state_dict(ckpt['denoiser'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.global_step = ckpt['global_step']
        self.best_loss = ckpt['best_loss']
        print(f"Loaded: {path} (step {self.global_step})")


# =============================================================================
# FACTORY: Load with real encoders
# =============================================================================

def load_beatrix_with_encoders(
    config: BeatrixConfig,
    vae_model: str = "black-forest-labs/FLUX.1-dev",
    dino_model: str = "facebook/dinov2-base",
    device: str = 'cuda',
) -> Beatrix:
    """
    Create Beatrix with real Flux VAE and DINO encoders.

    Returns model ready for training.
    """
    from diffusers import AutoencoderKL
    from transformers import Dinov2Model

    device = torch.device(device)
    model = Beatrix(config)

    print("Loading Flux VAE...")
    vae = AutoencoderKL.from_pretrained(
        vae_model, subfolder="vae", torch_dtype=torch.float32
    ).to(device)

    scale = getattr(vae.config, 'scaling_factor', 1.0)
    shift = getattr(vae.config, 'shift_factor', 0.0) or 0.0

    model.attach_flux_ae(FluxVAEWrapper(vae, scale, shift).to(device))
    model.set_vae_decoder(FluxVAEDecoder(vae, scale, shift).to(device))

    print("Loading DINO...")
    dino = Dinov2Model.from_pretrained(dino_model).to(device)
    model.attach_encoder('dino', DINOWrapper(dino).to(device), embed_dim=768)

    model.network_to(device)

    print(f"Streams: {list(model.head.streams.keys())}")
    print(f"Trainable: {sum(p.numel() for p in model.trainable_parameters()):,}")

    return model


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    from geofractal.router.prefab.agatha.head_router import MockEncoder

    print("=" * 60)
    print("  BEATRIX TRAINER TEST")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    config = BeatrixConfig(
        latent_channels=32,
        latent_height=16,
        latent_width=16,
        head_embed_dim=256,
        geometric_types=['cantor', 'beatrix'],
        conv_types=['wide_resnet'],
        num_theta_probes=2,
        num_training_steps=5,
    )

    print(f"\nConfig: {config.latent_shape}")

    # Build model with mock encoders
    model = Beatrix(config)

    class MockVAE(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape
        def forward(self, x):
            B = x.shape[0]
            return torch.randn(B, *self.shape, device=x.device)

    model.attach_flux_ae(MockVAE(config.latent_shape))
    model.attach_encoder('dino', MockEncoder(128, 768), embed_dim=768)
    model.network_to(torch.device(device))

    print(f"Streams: {list(model.head.streams.keys())}")
    print(f"Trainable: {sum(p.numel() for p in model.trainable_parameters()):,}")

    # Test forward
    print("\n--- Forward Test ---")
    B = 2
    inputs = {
        'flux_ae': torch.randn(B, 64, device=device),
        'dino': torch.randn(B, 128, device=device),
    }

    loss, metrics = model.compute_loss(inputs)
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"PSNR: {metrics['psnr']:.1f}dB")

    # Test trainer
    print("\n--- Trainer Test ---")
    trainer = BeatrixTrainer(model, device=device)
    metrics = trainer.train_step(inputs)
    print(f"Step {trainer.global_step}: loss={metrics['loss']:.6f}")

    # Test sampling
    print("\n--- Sampling ---")
    with torch.no_grad():
        z = model.sample(inputs, num_steps=5)
        print(f"Sample: {z.shape}")

    print("\n" + "=" * 60)
    print("  ✓ BEATRIX TRAINER READY")
    print("=" * 60)