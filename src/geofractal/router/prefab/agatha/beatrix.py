"""
BEATRIX
=======

Beatrix diffusion model - oscillator-based flow matching.

Contains:
    - BeatrixConfig (model configuration)
    - Encoder wrappers (FluxVAEWrapper, DINOWrapper, FluxVAEDecoder)
    - FluxAEStream (specialized EncoderStream preserving raw latent)
    - BeatrixDenoiser (trainable component)
    - Beatrix (full model: head router + collective + oscillator + denoiser)
    - Factory functions (load_beatrix_with_encoders, save_beatrix, load_beatrix)

Training is handled by beatrix_trainer.py.

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
    │  z_1 = mail['flux_ae'].metadata['raw']   (data)             │
    │  z_0 ~ N(0, I)                           (noise)            │
    │  z_t = (1-t)·z_0 + t·z_1                 (interpolate)      │
    │                                                              │
    │  mail.fused ──→ BeatrixCollective ──→ Tower Forces          │
    │                                             ↓                │
    │  (z_t, x_ref, forces) ──→ Oscillator ──→ v_pred             │
    │                                                              │
    │  Loss = ||v_pred - (z_1 - z_0)||²        (velocity)         │
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
    Trainable denoiser with flow matching velocity prediction.

    Predicts velocity field v that transports z_0 (noise) → z_1 (data).

    Flow Matching Formulation:
        z_t = (1-t)·z_0 + t·z_1           # Linear interpolation
        v_target = z_1 - z_0               # Target velocity (constant)
        v_pred = network(z_t, t, cond)     # Predicted velocity
        loss = ||v_pred - v_target||²      # Velocity matching

    The oscillator integrates tower forces to produce velocity,
    NOT position. This is the key difference from standard diffusion.
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

        # Timestep embedding (t ∈ [0,1], not sigma)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(config.head_embed_dim),
            nn.Linear(config.head_embed_dim, config.head_embed_dim),
            nn.GELU(),
            nn.Linear(config.head_embed_dim, config.head_embed_dim),
        )

        # Velocity output projection (oscillator output → velocity)
        self.velocity_proj = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )

    def forward(
        self,
        z_t: Tensor,                    # [B, C, H, W] interpolated latent
        mail: HeadMail,                 # Conditioning from head router
        t: Tensor,                      # [B] timestep in [0, 1]
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Predict velocity field at z_t.

        Args:
            z_t: Interpolated latent z_t = (1-t)·z_0 + t·z_1
            mail: HeadMail containing fused conditioning
            t: Timestep in [0, 1] (0=noise, 1=data)

        Returns:
            v_pred: Predicted velocity [B, C, H, W]
        """
        B = z_t.shape[0]
        num_steps = num_steps or self.config.num_training_steps

        # Embed timestep
        t_emb = self.time_embed(t)  # [B, embed_dim]

        # Condition fused representation with timestep
        fused_cond = mail.fused + t_emb if mail.fused is not None else t_emb

        # Anchor: where we want to go (data manifold direction)
        x_ref = self.anchor_proj(fused_cond)  # [B, latent_dim]

        # Guidance from DINO
        guidance_flat = None
        if 'dino' in mail.streams:
            dino_content = mail.streams['dino'].content
            if dino_content.dim() == 3:
                dino_content = dino_content.mean(dim=1)
            guidance_flat = self.guidance_proj(dino_content + t_emb)

        # Get tower forces from collective
        collective_result = self.collective(fused_cond, return_all=True)
        tower_outputs = collective_result['outputs']
        fingerprint = collective_result['fingerprint']

        # Flatten z_t for oscillator
        z_flat = z_t.reshape(B, -1)  # [B, latent_dim]

        # Oscillator computes the integrated force field
        # This represents the "push" from current position toward data
        osc_out = self.oscillator(
            x_init=z_flat,
            x_ref=x_ref,
            tower_outputs=tower_outputs,
            guidance=guidance_flat,
            state_fingerprint=fingerprint,
            num_steps=num_steps,
        )

        # The velocity is the difference: where oscillator wants to go - where we are
        # v = f(z_t, t, cond) that predicts the direction toward data
        v_flat = self.velocity_proj(osc_out - z_flat)

        return v_flat.reshape(B, *self.config.latent_shape)

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
        t: Optional[Tensor] = None,
        z_0: Optional[Tensor] = None,       # Noise (if provided)
        num_steps: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, HeadMail]:
        """
        Forward pass for flow matching training.

        Flow Matching:
            z_t = (1-t)·z_0 + t·z_1
            v_target = z_1 - z_0
            v_pred = denoiser(z_t, mail, t)

        Args:
            inputs: Dict for head router (must include 'flux_ae')
            t: Timestep in [0,1] (sampled if None)
            z_0: Noise tensor (sampled if None)

        Returns:
            v_pred: Predicted velocity [B, C, H, W]
            v_target: Target velocity [B, C, H, W]
            mail: HeadMail for inspection
        """
        mail = self.head(inputs)

        if 'flux_ae' not in mail.streams:
            raise ValueError("flux_ae stream required")

        z_1 = mail.streams['flux_ae'].metadata.get('raw')  # Data (clean)
        if z_1 is None:
            raise ValueError("flux_ae missing raw latent - use attach_flux_ae()")

        B = z_1.shape[0]
        device = z_1.device

        # Sample timestep t ~ Uniform(0, 1)
        if t is None:
            t = torch.rand(B, device=device)

        # Sample noise z_0 ~ N(0, I)
        if z_0 is None:
            z_0 = torch.randn_like(z_1)

        # Flow matching interpolation: z_t = (1-t)·z_0 + t·z_1
        t_expand = t.view(-1, 1, 1, 1)
        z_t = (1 - t_expand) * z_0 + t_expand * z_1

        # Target velocity: v = z_1 - z_0 (constant along path)
        v_target = z_1 - z_0

        # Predict velocity
        v_pred = self.denoiser(z_t, mail, t, num_steps)

        return v_pred, v_target, mail

    def compute_loss(self, inputs: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute flow matching loss.

        Loss = E_t,z_0 [ ||v_pred(z_t, t) - v_target||² ]
        """
        v_pred, v_target, mail = self.forward(inputs)

        # Velocity matching loss
        loss = F.mse_loss(v_pred, v_target)

        with torch.no_grad():
            mse = loss.item()
            # Cosine similarity between predicted and target velocity
            v_pred_flat = v_pred.reshape(v_pred.shape[0], -1)
            v_target_flat = v_target.reshape(v_target.shape[0], -1)
            cos_sim = F.cosine_similarity(v_pred_flat, v_target_flat, dim=-1).mean().item()

        return loss, {'loss': mse, 'cos_sim': cos_sim}

    @torch.no_grad()
    def sample(
        self,
        inputs: Dict[str, Tensor],
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Generate via flow matching ODE integration.

        Integrate: dz/dt = v_pred(z_t, t)
        From t=0 (noise) to t=1 (data)

        Uses Euler method: z_{t+dt} = z_t + dt · v_pred(z_t, t)
        """
        num_steps = num_steps or self.config.num_steps

        # Get conditioning mail
        mail = self.head(inputs)
        B = mail.batch_size
        device = mail.fused.device if mail.fused is not None else next(self.parameters()).device

        # Start from pure noise at t=0
        z = torch.randn(B, *self.config.latent_shape, device=device)

        # Time schedule: t goes from 0 (noise) to 1 (data)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device)

            # Predict velocity at current (z, t)
            v_pred = self.denoiser(z, mail, t, num_steps=1)

            # Euler step: z = z + dt * v
            z = z + dt * v_pred

        return z

    @torch.no_grad()
    def sample_midpoint(
        self,
        inputs: Dict[str, Tensor],
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Generate via midpoint ODE integration (2nd order, more accurate).

        Midpoint method:
            v_mid = v_pred(z + 0.5·dt·v_pred(z,t), t + 0.5·dt)
            z_{t+dt} = z_t + dt · v_mid
        """
        num_steps = num_steps or self.config.num_steps

        mail = self.head(inputs)
        B = mail.batch_size
        device = mail.fused.device if mail.fused is not None else next(self.parameters()).device

        z = torch.randn(B, *self.config.latent_shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device)
            t_mid = torch.full((B,), (i + 0.5) / num_steps, device=device)

            # Euler predictor
            v1 = self.denoiser(z, mail, t, num_steps=1)
            z_mid = z + 0.5 * dt * v1

            # Midpoint corrector
            v_mid = self.denoiser(z_mid, mail, t_mid, num_steps=1)
            z = z + dt * v_mid

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

    @torch.no_grad()
    def interpolate(
        self,
        image_a: Tensor,
        image_b: Tensor,
        num_steps: int = 10,
    ) -> List[Tensor]:
        """
        Interpolate between two images in latent space.

        Args:
            image_a: [1, 3, H, W] start image in [0, 1]
            image_b: [1, 3, H, W] end image in [0, 1]
            num_steps: Number of interpolation steps

        Returns:
            List of interpolated images [num_steps + 1] x [1, 3, H, W]
        """
        device = next(self.parameters()).device
        image_a = image_a.to(device)
        image_b = image_b.to(device)

        # Get latents for both images
        mail_a = self.head({'flux_ae': image_a, 'dino': image_a})
        mail_b = self.head({'flux_ae': image_b, 'dino': image_b})

        z_a = mail_a.streams['flux_ae'].metadata['raw']
        z_b = mail_b.streams['flux_ae'].metadata['raw']

        results = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            z_interp = (1 - alpha) * z_a + alpha * z_b

            if self.vae_decoder is not None:
                img = self.vae_decoder(z_interp)
            else:
                img = z_interp

            results.append(img)

        return results

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


def save_beatrix(model: Beatrix, path: str):
    """Save complete model state."""
    torch.save({
        'config': model.config,
        'denoiser': model.denoiser.state_dict(),
        'head': model.head.state_dict(),
    }, path)
    print(f"Model saved: {path}")


def load_beatrix(path: str, device: str = 'cuda') -> Beatrix:
    """Load complete model from checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model = Beatrix(ckpt['config'])
    model.denoiser.load_state_dict(ckpt['denoiser'])
    model.head.load_state_dict(ckpt['head'])
    model.network_to(torch.device(device))
    print(f"Model loaded: {path}")
    return model