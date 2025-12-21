"""
BEATRIX
=======

Beatrix diffusion model - oscillator-based flow matching.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    AGATHA HEAD ROUTER                            │
    │                                                                  │
    │  Image ──→ flux_ae stream ──→ [raw latent in metadata]          │
    │  Image ──→ dino stream ─────→ [structural CLS]                  │
    │  Text ───→ text stream ─────→ [Qwen two-shot + T5 fused]        │
    │                                                                  │
    │            All streams ──→ AdaptiveFusion ──→ mail.fused        │
    └─────────────────────────────────────────────────────────────────┘
                                       ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  z_1 = mail['flux_ae'].metadata['raw']   (data)                 │
    │  z_0 ~ N(0, I)                           (noise)                │
    │  z_t = (1-t)·z_0 + t·z_1                 (interpolate)          │
    │                                                                  │
    │  mail.fused ──→ BeatrixCollective ──→ Tower Forces              │
    │                                             ↓                    │
    │  (z_t, x_ref, forces) ──→ Oscillator ──→ v_pred                 │
    │                                                                  │
    │  Loss = ||v_pred - (z_1 - z_0)||²        (velocity)             │
    └─────────────────────────────────────────────────────────────────┘

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

from geofractal.router.base_router import BaseRouter
from geofractal.router.base_tower import BaseTower
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.components.fusion_component import AdaptiveFusion

from geofractal.router.prefab.agatha.head_router import (
    AgathaHeadRouter,
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

    # Text encoders
    qwen_model: str = 'Qwen/Qwen2.5-1.5B-Instruct'
    t5_model: str = 'google-t5/t5-base'
    qwen_dim: int = 1536
    t5_dim: int = 768

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

    @property
    def latent_dim(self) -> int:
        return self.latent_channels * self.latent_height * self.latent_width

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return (self.latent_channels, self.latent_height, self.latent_width)


# =============================================================================
# VISION ENCODER COMPONENTS
# =============================================================================

class FluxVAEEncoder(TorchComponent):
    """
    Wraps Flux VAE encoder as a TorchComponent.
    Returns [B, C, H, W] latent.
    """

    def __init__(
        self,
        name: str = 'flux_vae_encoder',
        vae: nn.Module = None,
        scale: float = 1.0,
        shift: float = 0.0,
    ):
        super().__init__(name)
        self.vae = vae
        self.scale = scale
        self.shift = shift

    def forward(self, x: Tensor) -> Tensor:
        """Encode [B, 3, H, W] image in [0,1] to latent."""
        x = 2 * x - 1  # Scale to [-1, 1]
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
        return (latent - self.shift) * self.scale


class FluxVAEDecoder(TorchComponent):
    """Wraps Flux VAE decoder as a TorchComponent."""

    def __init__(
        self,
        name: str = 'flux_vae_decoder',
        vae: nn.Module = None,
        scale: float = 1.0,
        shift: float = 0.0,
    ):
        super().__init__(name)
        self.vae = vae
        self.scale = scale
        self.shift = shift

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent to [B, 3, H, W] image in [0,1]."""
        z = z / self.scale + self.shift
        with torch.no_grad():
            image = self.vae.decode(z).sample
        return ((image + 1) / 2).clamp(0, 1)


class DINOEncoder(TorchComponent):
    """Wraps DINO as a TorchComponent."""

    def __init__(
        self,
        name: str = 'dino_encoder',
        dino: nn.Module = None,
    ):
        super().__init__(name)
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
# TEXT ENCODER COMPONENTS
# =============================================================================

class QwenDualShotEncoder(TorchComponent):
    """
    Qwen two-shot text encoder as a TorchComponent.

    Two-Shot Strategy:
        1. Build prompt with two examples (car, sunflower)
        2. Generate description for target
        3. Encode the GENERATED description (not the prompt)
    """

    SYSTEM_PROMPT = "You describe objects in exactly one sentence. Be specific about visual features."

    EXAMPLES = [
        ("car", "A four-wheeled motor vehicle with windows, doors, headlights, and a metal body used for transportation on roads."),
        ("sunflower", "A tall plant with a large circular flower head containing yellow petals surrounding a brown seed-filled center."),
    ]

    def __init__(
        self,
        name: str = 'qwen_encoder',
        model: nn.Module = None,
        tokenizer: Any = None,
        max_new_tokens: int = 50,
        cache_descriptions: bool = True,
    ):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.cache_descriptions = cache_descriptions
        self.hidden_size = model.config.hidden_size if model else 1536

        # Caches
        self._description_cache: Dict[str, str] = {}
        self._embedding_cache: Dict[str, Tensor] = {}

    def _build_twoshot_prompt(self, text: str) -> str:
        """Build two-shot prompt for description generation."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        for example_input, example_output in self.EXAMPLES:
            messages.append({"role": "user", "content": f"Describe: {example_input}"})
            messages.append({"role": "assistant", "content": example_output})
        messages.append({"role": "user", "content": f"Describe: {text}"})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate_description(self, text: str) -> str:
        """Generate description using two-shot prompting."""
        if self.cache_descriptions and text in self._description_cache:
            return self._description_cache[text]

        prompt = self._build_twoshot_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
        description = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if self.cache_descriptions:
            self._description_cache[text] = description

        return description

    @torch.no_grad()
    def encode_text(self, text: str) -> Tensor:
        """Encode text to hidden state (last token of last layer)."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return outputs.hidden_states[-1][0, -1, :]

    @torch.no_grad()
    def forward(self, texts: List[str]) -> Tensor:
        """
        Process batch of texts through two-shot generation + encoding.

        Args:
            texts: List of text prompts

        Returns:
            embeddings: [B, hidden_size] tensor
        """
        embeddings = []

        for text in texts:
            if self.cache_descriptions and text in self._embedding_cache:
                emb = self._embedding_cache[text].to(self.model.device)
            else:
                description = self.generate_description(text)
                emb = self.encode_text(description)

                if self.cache_descriptions:
                    self._embedding_cache[text] = emb.cpu()

            embeddings.append(emb)

        return torch.stack(embeddings, dim=0)

    def get_description(self, text: str) -> str:
        """Get cached description for a text."""
        return self._description_cache.get(text, text)

    def clear_cache(self):
        """Clear description and embedding caches."""
        self._description_cache.clear()
        self._embedding_cache.clear()


class T5Encoder(TorchComponent):
    """
    T5 encoder as a TorchComponent.

    Encodes the same text/description that Qwen generates.
    """

    def __init__(
        self,
        name: str = 't5_encoder',
        model: nn.Module = None,
        tokenizer: Any = None,
        max_length: int = 128,
    ):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hidden_size = model.config.d_model if model else 768

    @torch.no_grad()
    def forward(self, texts: List[str]) -> Tensor:
        """
        Encode batch of texts.

        Returns:
            embeddings: [B, hidden_size] mean-pooled tensor
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.model.device)

        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )

        hidden = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled


# =============================================================================
# PROJECTION COMPONENTS
# =============================================================================

class ProjectionBlock(TorchComponent):
    """
    MLP projection as a proper TorchComponent.
    Replaces raw nn.Sequential patterns.
    """

    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = 'gelu',
    ):
        super().__init__(name)
        hidden_dim = hidden_dim or in_dim * 2

        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SinusoidalTimeEmbed(TorchComponent):
    """Sinusoidal timestep embedding as a TorchComponent."""

    def __init__(self, name: str = 'time_embed', dim: int = 256):
        super().__init__(name)
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.proj(emb)


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
# TEXT STREAM (for fused Qwen + T5)
# =============================================================================

class TextStream(EncoderStream):
    """
    Encoder stream for fused text (Qwen two-shot + T5).

    Uses AdaptiveFusion to combine both text encoders.
    """

    def __init__(
        self,
        name: str = 'text',
        qwen_encoder: Optional[QwenDualShotEncoder] = None,
        t5_encoder: Optional[T5Encoder] = None,
        embed_dim: int = 256,
        fingerprint_dim: int = 64,
        **kwargs,
    ):
        super().__init__(
            name=name,
            stream_type=StreamType.TEXT,
            embed_dim=embed_dim,
            fingerprint_dim=fingerprint_dim,
            project_dim=embed_dim,
            frozen=True,
            **kwargs,
        )

        self.qwen_encoder = qwen_encoder
        self.t5_encoder = t5_encoder

        qwen_dim = qwen_encoder.hidden_size if qwen_encoder else 1536
        t5_dim = t5_encoder.hidden_size if t5_encoder else 768

        # Projection components (proper TorchComponent)
        self.qwen_proj = ProjectionBlock('qwen_proj', qwen_dim, embed_dim)
        self.t5_proj = ProjectionBlock('t5_proj', t5_dim, embed_dim)

        # Fusion using AdaptiveFusion
        self.text_fusion = AdaptiveFusion('text_fusion', num_inputs=2, in_features=embed_dim)

    def forward(self, texts: List[str], **kwargs) -> EncoderMail:
        """
        Process texts through Qwen two-shot + T5, fuse, create mail.
        """
        if self.qwen_encoder is None or self.t5_encoder is None:
            raise ValueError("Both qwen_encoder and t5_encoder must be set")

        # Get embeddings from both encoders
        qwen_emb = self.qwen_encoder(texts)  # [B, qwen_dim]

        # T5 encodes the generated descriptions (from Qwen cache)
        descriptions = [self.qwen_encoder.get_description(t) for t in texts]
        t5_emb = self.t5_encoder(descriptions)  # [B, t5_dim]

        # Project both to embed_dim
        qwen_proj = self.qwen_proj(qwen_emb)  # [B, embed_dim]
        t5_proj = self.t5_proj(t5_emb)  # [B, embed_dim]

        # Fuse with AdaptiveFusion
        fused = self.text_fusion(qwen_proj, t5_proj)  # [B, embed_dim]
        content = fused.unsqueeze(1)  # [B, 1, embed_dim]

        B = content.shape[0]
        fingerprint = self.address.fingerprint.unsqueeze(0).expand(B, -1)
        fingerprint = fingerprint + self.stream_identity

        return EncoderMail(
            content=content,
            fingerprint=fingerprint,
            stream_type=self.stream_type,
            source=self.name,
            metadata={
                'qwen_emb': qwen_emb,
                't5_emb': t5_emb,
                'descriptions': descriptions,
            },
        )


# =============================================================================
# BEATRIX DENOISER (BaseTower)
# =============================================================================

class BeatrixDenoiser(BaseTower):
    """
    Trainable denoiser as a BaseTower.

    Stages are TorchComponent instances.

    Flow Matching:
        z_t = (1-t)·z_0 + t·z_1
        v_target = z_1 - z_0
        v_pred = denoiser(z_t, mail, t)
    """

    def __init__(self, name: str, config: BeatrixConfig):
        super().__init__(name, strict=False)
        self.config = config

        # === STAGE 0: Time embedding ===
        self.append(SinusoidalTimeEmbed('time_embed', config.head_embed_dim))

        # === STAGE 1: Anchor projection ===
        self.append(ProjectionBlock(
            'anchor_proj',
            in_dim=config.head_embed_dim,
            out_dim=config.latent_dim,
            hidden_dim=config.head_embed_dim * 2,
        ))

        # === STAGE 2: Guidance projection ===
        self.append(ProjectionBlock(
            'guidance_proj',
            in_dim=config.head_embed_dim,
            out_dim=config.latent_dim,
            hidden_dim=config.head_embed_dim * 2,
        ))

        # === STAGE 3: Text projection ===
        self.append(ProjectionBlock(
            'text_proj',
            in_dim=config.head_embed_dim,
            out_dim=config.latent_dim,
            hidden_dim=config.head_embed_dim * 2,
        ))

        # === STAGE 4: Velocity projection ===
        self.append(ProjectionBlock(
            'velocity_proj',
            in_dim=config.latent_dim,
            out_dim=config.latent_dim,
            hidden_dim=config.latent_dim,
        ))

        # === Named components (not stages) ===

        # Collective (WideRouter)
        collective_config = BeatrixCollectiveConfig(
            dim=config.head_embed_dim,
            fingerprint_dim=config.fingerprint_dim,
            geometric_types=config.geometric_types,
            conv_types=config.conv_types,
            num_theta_probes=config.num_theta_probes,
            use_signed_pairs=config.use_signed_pairs,
        )
        self.attach('collective', BeatrixCollective(collective_config))

        # Oscillator
        num_tower_pairs = len(config.geometric_types) + len(config.conv_types)
        self.attach('oscillator', BeatrixOscillator(
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
        ))

    def forward(
        self,
        z_t: Tensor,
        mail: HeadMail,
        t: Tensor,
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Predict velocity field at z_t.

        Args:
            z_t: Interpolated latent [B, C, H, W]
            mail: HeadMail containing fused conditioning
            t: Timestep in [0, 1]

        Returns:
            v_pred: Predicted velocity [B, C, H, W]
        """
        B = z_t.shape[0]
        num_steps = num_steps or self.config.num_training_steps

        # Access stages by index
        time_embed = self.stages[0]      # SinusoidalTimeEmbed
        anchor_proj = self.stages[1]     # ProjectionBlock
        guidance_proj = self.stages[2]   # ProjectionBlock
        text_proj = self.stages[3]       # ProjectionBlock
        velocity_proj = self.stages[4]   # ProjectionBlock

        # Embed timestep
        t_emb = time_embed(t)  # [B, embed_dim]

        # Condition fused representation with timestep
        fused_cond = mail.fused + t_emb if mail.fused is not None else t_emb

        # Anchor: where we want to go
        x_ref = anchor_proj(fused_cond)  # [B, latent_dim]

        # Guidance from DINO (structural)
        guidance_flat = None
        if 'dino' in mail.streams:
            dino_content = mail.streams['dino'].content
            if dino_content.dim() == 3:
                dino_content = dino_content.mean(dim=1)
            guidance_flat = guidance_proj(dino_content + t_emb)

        # Text conditioning (semantic from Qwen + T5)
        if 'text' in mail.streams:
            text_content = mail.streams['text'].content
            if text_content.dim() == 3:
                text_content = text_content.mean(dim=1)
            text_cond = text_proj(text_content + t_emb)

            if guidance_flat is not None:
                guidance_flat = guidance_flat + text_cond
            else:
                guidance_flat = text_cond

        # Get tower forces from collective
        collective_result = self['collective'](fused_cond, return_all=True)
        tower_outputs = collective_result['outputs']
        fingerprint = collective_result['fingerprint']

        # Flatten z_t for oscillator
        z_flat = z_t.reshape(B, -1)  # [B, latent_dim]

        # Oscillator computes integrated force field
        osc_out = self['oscillator'](
            x_init=z_flat,
            x_ref=x_ref,
            tower_outputs=tower_outputs,
            guidance=guidance_flat,
            state_fingerprint=fingerprint,
            num_steps=num_steps,
        )

        # Velocity = direction toward data
        v_flat = velocity_proj(osc_out - z_flat)

        return v_flat.reshape(B, *self.config.latent_shape)


# =============================================================================
# BEATRIX MODEL (BaseRouter)
# =============================================================================

class Beatrix(BaseRouter):
    """
    Complete Beatrix system as a BaseRouter.

    Coordinates: AgathaHeadRouter + BeatrixDenoiser

    Streams:
        - flux_ae: VAE latent (required)
        - dino: Structural guidance (optional)
        - text: Fused Qwen + T5 (optional)
    """

    def __init__(self, name: str, config: BeatrixConfig):
        super().__init__(name, strict=False)
        self.config = config

        # Store config in objects dict
        self.objects['config'] = config

        # === Head Router ===
        self.attach('head', AgathaHeadRouter(
            name='agatha_head',
            embed_dim=config.head_embed_dim,
            fingerprint_dim=config.fingerprint_dim,
            fusion_type=config.fusion_type,
        ))

        # === Denoiser (BaseTower) ===
        self.attach('denoiser', BeatrixDenoiser('beatrix_denoiser', config))

        # === VAE Decoder (optional, set via attach) ===
        # Will be attached as 'vae_decoder'

    @property
    def head(self) -> AgathaHeadRouter:
        return self['head']

    @property
    def denoiser(self) -> BeatrixDenoiser:
        return self['denoiser']

    @property
    def vae_decoder(self) -> Optional[FluxVAEDecoder]:
        return self.components.get('vae_decoder')

    def attach_flux_ae(
        self,
        encoder: FluxVAEEncoder,
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

    def attach_dino(
        self,
        encoder: DINOEncoder,
        extract_fn: Optional[Callable] = None,
    ) -> 'Beatrix':
        """Attach DINO encoder."""
        self.head.attach_encoder(
            name='dino',
            encoder=encoder,
            embed_dim=encoder.hidden_size,
            stream_type=StreamType.GUIDANCE,
            extract_fn=extract_fn,
            frozen=True,
        )
        return self

    def attach_text_encoders(
        self,
        qwen_encoder: QwenDualShotEncoder,
        t5_encoder: T5Encoder,
    ) -> 'Beatrix':
        """
        Attach Qwen two-shot and T5 text encoders.
        """
        text_stream = TextStream(
            name='text',
            qwen_encoder=qwen_encoder,
            t5_encoder=t5_encoder,
            embed_dim=self.config.head_embed_dim,
            fingerprint_dim=self.config.fingerprint_dim,
        )

        self.head.streams['text'] = text_stream
        self.head.attach('text', text_stream)
        self.head._rebuild_fusion()

        return self

    def set_vae_decoder(self, decoder: FluxVAEDecoder) -> 'Beatrix':
        """Set VAE decoder for image generation."""
        self.attach('vae_decoder', decoder)
        return self

    def forward(
        self,
        inputs: Dict[str, Any],
        t: Optional[Tensor] = None,
        z_0: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, HeadMail]:
        """
        Forward pass for flow matching training.

        Args:
            inputs: Dict for head router
                - 'flux_ae': [B, 3, H, W] image (required)
                - 'dino': [B, 3, H, W] image (optional)
                - 'text': List[str] prompts (optional)
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

        z_1 = mail.streams['flux_ae'].metadata.get('raw')
        if z_1 is None:
            raise ValueError("flux_ae missing raw latent")

        B = z_1.shape[0]
        device = z_1.device

        if t is None:
            t = torch.rand(B, device=device)

        if z_0 is None:
            z_0 = torch.randn_like(z_1)

        t_expand = t.view(-1, 1, 1, 1)
        z_t = (1 - t_expand) * z_0 + t_expand * z_1

        v_target = z_1 - z_0

        v_pred = self.denoiser(z_t, mail, t, num_steps)

        return v_pred, v_target, mail

    def compute_loss(self, inputs: Dict[str, Any]) -> Tuple[Tensor, Dict[str, float]]:
        """Compute flow matching loss."""
        v_pred, v_target, mail = self.forward(inputs)

        loss = F.mse_loss(v_pred, v_target)

        with torch.no_grad():
            mse = loss.item()
            v_pred_flat = v_pred.reshape(v_pred.shape[0], -1)
            v_target_flat = v_target.reshape(v_target.shape[0], -1)
            cos_sim = F.cosine_similarity(v_pred_flat, v_target_flat, dim=-1).mean().item()

        return loss, {'loss': mse, 'cos_sim': cos_sim}

    @torch.no_grad()
    def sample(
        self,
        inputs: Dict[str, Any],
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """Generate via flow matching ODE (Euler)."""
        num_steps = num_steps or self.config.num_steps

        mail = self.head(inputs)
        B = mail.batch_size
        device = mail.fused.device if mail.fused is not None else next(self.parameters()).device

        z = torch.randn(B, *self.config.latent_shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device)
            v_pred = self.denoiser(z, mail, t, num_steps=1)
            z = z + dt * v_pred

        return z

    @torch.no_grad()
    def sample_midpoint(
        self,
        inputs: Dict[str, Any],
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """Generate via midpoint ODE (2nd order)."""
        num_steps = num_steps or self.config.num_steps

        mail = self.head(inputs)
        B = mail.batch_size
        device = mail.fused.device if mail.fused is not None else next(self.parameters()).device

        z = torch.randn(B, *self.config.latent_shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device)
            t_mid = torch.full((B,), (i + 0.5) / num_steps, device=device)

            v1 = self.denoiser(z, mail, t, num_steps=1)
            z_mid = z + 0.5 * dt * v1

            v_mid = self.denoiser(z_mid, mail, t_mid, num_steps=1)
            z = z + dt * v_mid

        return z

    @torch.no_grad()
    def generate(
        self,
        inputs: Dict[str, Any],
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """Generate images (requires VAE decoder)."""
        z = self.sample(inputs, num_steps)

        if self.vae_decoder is not None:
            return self.vae_decoder(z)
        return z

    def trainable_parameters(self):
        """Only denoiser parameters (encoders stay frozen)."""
        return self.denoiser.parameters()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def load_beatrix_with_encoders(
    config: BeatrixConfig,
    vae_model: str = "black-forest-labs/FLUX.1-dev",
    dino_model: str = "facebook/dinov2-base",
    load_text_encoders: bool = True,
    device: str = 'cuda',
) -> Beatrix:
    """
    Create Beatrix with real encoders.
    """
    from diffusers import AutoencoderKL
    from transformers import Dinov2Model

    device = torch.device(device)
    model = Beatrix('beatrix', config)

    # === Flux VAE ===
    print("Loading Flux VAE...")
    vae = AutoencoderKL.from_pretrained(
        vae_model, subfolder="vae", torch_dtype=torch.float32
    ).to(device)

    scale = getattr(vae.config, 'scaling_factor', 1.0)
    shift = getattr(vae.config, 'shift_factor', 0.0) or 0.0

    flux_encoder = FluxVAEEncoder('flux_vae_encoder', vae, scale, shift)
    flux_encoder.to(device)
    model.attach_flux_ae(flux_encoder)

    flux_decoder = FluxVAEDecoder('flux_vae_decoder', vae, scale, shift)
    flux_decoder.to(device)
    model.set_vae_decoder(flux_decoder)

    # === DINO ===
    print("Loading DINO...")
    dino = Dinov2Model.from_pretrained(dino_model).to(device)
    dino_encoder = DINOEncoder('dino_encoder', dino)
    dino_encoder.to(device)
    model.attach_dino(dino_encoder)

    # === Text Encoders ===
    if load_text_encoders:
        print(f"Loading Qwen ({config.qwen_model})...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        qwen_tokenizer = AutoTokenizer.from_pretrained(
            config.qwen_model, trust_remote_code=True
        )
        qwen_model = AutoModelForCausalLM.from_pretrained(
            config.qwen_model,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        qwen_model.eval()

        qwen_encoder = QwenDualShotEncoder(
            'qwen_encoder',
            model=qwen_model,
            tokenizer=qwen_tokenizer,
            cache_descriptions=True,
        )

        print(f"Loading T5 ({config.t5_model})...")
        from transformers import T5EncoderModel, T5Tokenizer

        t5_tokenizer = T5Tokenizer.from_pretrained(config.t5_model)
        t5_model = T5EncoderModel.from_pretrained(
            config.t5_model, torch_dtype=torch.float16
        ).to(device)
        t5_model.eval()

        t5_encoder = T5Encoder('t5_encoder', model=t5_model, tokenizer=t5_tokenizer)

        model.attach_text_encoders(qwen_encoder, t5_encoder)

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
    model = Beatrix('beatrix', ckpt['config'])
    model.denoiser.load_state_dict(ckpt['denoiser'])
    model.head.load_state_dict(ckpt['head'])
    model.network_to(torch.device(device))
    print(f"Model loaded: {path}")
    return model