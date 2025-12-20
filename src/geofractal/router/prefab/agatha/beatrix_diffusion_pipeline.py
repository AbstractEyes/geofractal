"""
BEATRIX DIFFUSION (Inference)
=============================

Inference-only pipeline for generating images with a trained Beatrix model.

For training, use beatrix_trainer.py.

Usage:
    from beatrix_diffusion import BeatrixInference

    # Load trained model
    pipeline = BeatrixInference.from_checkpoint(
        "checkpoints/beatrix_best.pt",
        device='cuda',
    )

    # Generate from conditioning image
    images = pipeline.generate(reference_images, num_steps=50)

    # Generate unconditionally
    images = pipeline.generate_unconditional(batch_size=4, num_steps=50)

Author: AbstractPhil + Claude
Date: December 2024
"""

from __future__ import annotations

from typing import Optional, Dict, List
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from geofractal.router.prefab.agatha.head_router import StreamType
from geofractal.router.prefab.agatha.beatrix_trainer import (
    Beatrix,
    BeatrixConfig,
    FluxVAEWrapper,
    FluxVAEDecoder,
    DINOWrapper,
)


class BeatrixInference:
    """
    Inference pipeline for trained Beatrix models.
    """

    def __init__(
        self,
        model: Beatrix,
        device: str = 'cuda',
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[BeatrixConfig] = None,
        vae_model: str = "black-forest-labs/FLUX.1-dev",
        dino_model: str = "facebook/dinov2-base",
        device: str = 'cuda',
    ) -> 'BeatrixInference':
        """
        Load trained model from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint
            config: BeatrixConfig (uses default if None)
            vae_model: HuggingFace VAE model ID
            dino_model: HuggingFace DINO model ID
            device: Device to load to
        """
        from diffusers import AutoencoderKL
        from transformers import Dinov2Model

        device = torch.device(device)

        # Default config
        if config is None:
            config = BeatrixConfig()

        # Create model
        model = Beatrix(config)

        # Load encoders
        print("Loading VAE...")
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

        # Load trained weights
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.denoiser.load_state_dict(ckpt['denoiser'])

        model.eval()

        return cls(model, device=str(device))

    @torch.no_grad()
    def generate(
        self,
        reference_images: Tensor,
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Generate images conditioned on reference images.

        Args:
            reference_images: [B, 3, H, W] in [0, 1]
            num_steps: Sampling steps (default: config.num_steps)

        Returns:
            Generated images [B, 3, H, W] in [0, 1]
        """
        reference_images = reference_images.to(self.device)

        inputs = {
            'flux_ae': reference_images,
            'dino': reference_images,
        }

        return self.model.generate(inputs, num_steps=num_steps)

    @torch.no_grad()
    def generate_unconditional(
        self,
        batch_size: int = 1,
        num_steps: Optional[int] = None,
        image_size: int = 512,
    ) -> Tensor:
        """
        Generate images without conditioning.

        Uses random noise as input to encoders.
        """
        noise = torch.randn(
            batch_size, 3, image_size, image_size,
            device=self.device
        )

        return self.generate(noise, num_steps=num_steps)

    @torch.no_grad()
    def interpolate(
        self,
        image_a: Tensor,
        image_b: Tensor,
        num_steps: int = 10,
        sampling_steps: Optional[int] = None,
    ) -> List[Tensor]:
        """
        Interpolate between two images in latent space.

        Args:
            image_a: [1, 3, H, W] start image
            image_b: [1, 3, H, W] end image
            num_steps: Number of interpolation steps
            sampling_steps: Diffusion sampling steps

        Returns:
            List of interpolated images
        """
        image_a = image_a.to(self.device)
        image_b = image_b.to(self.device)

        # Get conditioning for both
        mail_a = self.model.head({'flux_ae': image_a, 'dino': image_a})
        mail_b = self.model.head({'flux_ae': image_b, 'dino': image_b})

        z_a = mail_a.streams['flux_ae'].metadata['raw']
        z_b = mail_b.streams['flux_ae'].metadata['raw']

        results = []

        for i in range(num_steps + 1):
            alpha = i / num_steps

            # Interpolate latent
            z_interp = (1 - alpha) * z_a + alpha * z_b

            # Decode
            if self.model.vae_decoder is not None:
                img = self.model.vae_decoder(z_interp)
            else:
                img = z_interp

            results.append(img)

        return results


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  BEATRIX INFERENCE TEST")
    print("=" * 60)
    print("\nThis module is for inference only.")
    print("To test, first train a model with beatrix_trainer.py")
    print("\nUsage:")
    print("  pipeline = BeatrixInference.from_checkpoint('model.pt')")
    print("  images = pipeline.generate(reference_images)")
    print("=" * 60)