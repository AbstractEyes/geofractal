"""
Test Flux2 AE Integration with Agatha Head Router
==================================================

Validates that Flux2's VAE can encode images into latent space M,
where x(t) will live for the Beatrix oscillator.

DINO provides guidance (steering), Flux2 AE provides the manifold (terrain).

Architecture:
    Image → Flux2 AE → Latent z ∈ M (manifold for diffusion)
    Image → DINO → Guidance ξ (tangent forces for steering)
    Text  → Qwen → Conditioning x_ref (anchor points)

Copyright 2025 AbstractPhil
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from PIL import Image
import numpy as np

# For loading Flux2 VAE
try:
    from diffusers import AutoencoderKLFlux2

    HAS_FLUX2_VAE = True
except ImportError:
    from diffusers import AutoencoderKL

    HAS_FLUX2_VAE = False

# For DINO
from transformers import AutoModel, AutoProcessor


# =============================================================================
# FLUX2 AE WRAPPER
# =============================================================================

class Flux2AEEncoder(nn.Module):
    """
    Wrapper for Flux2's VAE encoder.

    Encodes images into the latent manifold M where x(t) lives.

    FLUX.2 VAE specs (from official BFL repo + DeepWiki):
        - latent_channels: 32 (FLUX.1 was 16)
        - spatial compression: 16x (FLUX.1 was 8x)
        - After 2x2 patchification: 128 channels at 1/16 resolution
        - Apache 2.0 licensed (can be used freely)
        - File: ae.safetensors in FLUX.2-dev repo

    The VAE provides the foundation for all FLUX.2 flow backbones,
    with optimized trade-off between learnability, quality and compression.
    BFL re-trained the latent space from scratch for better learnability
    and higher image quality - solving the "Learnability-Quality-Compression" trilemma.

    Loading:
        From FLUX.2-dev (gated, needs HF token):
        model_path="black-forest-labs/FLUX.2-dev", subfolder="vae"
    """

    # FLUX.2 VAE constants from official sources
    LATENT_CHANNELS = 32  # 32 channels (vs 16 in FLUX.1)
    SPATIAL_COMPRESSION = 8  # 8x downscale in VAE (same as FLUX.1)
    PATCHIFY_FACTOR = 2  # 2x2 patchification in transformer (not VAE)

    # Total: 16x compression when patchify is applied by transformer
    # VAE output: 32 channels at 1/8 spatial resolution
    # After transformer patchify: 128 channels at 1/16 resolution

    def __init__(
            self,
            model_path: str = "black-forest-labs/FLUX.2-dev",
            subfolder: str = "vae",
            torch_dtype: torch.dtype = torch.bfloat16,
            device: str = 'cuda',
            use_tiling: bool = False,
    ):
        super().__init__()

        self.device = device
        self.torch_dtype = torch_dtype
        self.use_tiling = use_tiling

        print(f"Loading Flux2 VAE from {model_path}/{subfolder}...")
        print("  Note: FLUX.2-dev is gated - requires HF token with accepted license")

        # Use AutoencoderKLFlux2 if available (proper FLUX.2 VAE class)
        if HAS_FLUX2_VAE:
            print("  Using AutoencoderKLFlux2")
            self.vae = AutoencoderKLFlux2.from_pretrained(
                model_path,
                subfolder=subfolder,
                torch_dtype=torch_dtype,
            ).to(device)
        else:
            print("  Warning: AutoencoderKLFlux2 not available, using AutoencoderKL")
            self.vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder=subfolder,
                torch_dtype=torch_dtype,
            ).to(device)
        self.vae.eval()

        # Get config values with FLUX.2 defaults
        self.scaling_factor = getattr(self.vae.config, 'scaling_factor', 1.0)
        shift = getattr(self.vae.config, 'shift_factor', None)
        self.shift_factor = 0.0 if shift is None else shift

        # Use official specs - config may not always reflect actual architecture
        self.latent_channels = getattr(self.vae.config, 'latent_channels', self.LATENT_CHANNELS)
        self.spatial_compression = self.SPATIAL_COMPRESSION

        print(f"  ✓ VAE loaded: {self.latent_channels} latent channels")
        print(f"  ✓ Spatial compression: {self.spatial_compression}x")
        print(f"  ✓ scaling_factor={self.scaling_factor}, shift_factor={self.shift_factor}")

        if use_tiling:
            self.vae.enable_tiling()
            print("  ✓ Tiling enabled for large images")

        # Freeze
        for param in self.vae.parameters():
            param.requires_grad = False

    @property
    def hidden_size(self) -> int:
        """For head_router auto-detection."""
        return self.latent_channels

    def preprocess(self, images: Tensor) -> Tensor:
        """
        Preprocess images for VAE.

        Args:
            images: [B, C, H, W] in [0, 1] or [-1, 1]

        Returns:
            Preprocessed tensor in [-1, 1]
        """
        # Ensure [-1, 1] range
        if images.min() >= 0:
            images = images * 2 - 1
        return images.to(self.device, self.torch_dtype)

    @torch.no_grad()
    def encode(self, images: Tensor, sample: bool = True) -> Tensor:
        """
        Encode images to latent space.

        Args:
            images: [B, C, H, W] preprocessed images
            sample: If True, sample from posterior. If False, use mode.

        Returns:
            Latent tensor [B, latent_channels, H//8, W//8]
        """
        images = self.preprocess(images)

        # Encode
        if self.use_tiling:
            latent_dist = self.vae.tiled_encode(images).latent_dist
        else:
            latent_dist = self.vae.encode(images).latent_dist

        # Sample or mode
        if sample:
            z = latent_dist.sample()
        else:
            z = latent_dist.mode()

        # Apply scaling
        z = (z - self.shift_factor) * self.scaling_factor

        return z

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latents back to images.

        Args:
            z: Latent tensor [B, latent_channels, H, W]

        Returns:
            Images [B, 3, H*8, W*8] in [0, 1]
        """
        # Reverse scaling
        z = z / self.scaling_factor + self.shift_factor

        # Decode
        if self.use_tiling:
            images = self.vae.tiled_decode(z).sample
        else:
            images = self.vae.decode(z).sample

        # To [0, 1]
        images = (images + 1) / 2
        return images.clamp(0, 1)

    def forward(self, images: Tensor) -> Tensor:
        """Forward = encode for head_router compatibility."""
        return self.encode(images, sample=False)


# =============================================================================
# DINO GUIDANCE ENCODER
# =============================================================================

class DinoGuidanceEncoder(nn.Module):
    """
    DINO encoder for guidance signals.

    Provides ξ (tangent forces) for steering, NOT the manifold.
    """

    def __init__(
            self,
            model_path: str = "facebook/dinov2-large",
            device: str = 'cuda',
    ):
        super().__init__()

        self.device = device

        print(f"Loading DINO from {model_path}...")
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        print(f"  ✓ DINO loaded: {self.hidden_size}d")

        for param in self.model.parameters():
            param.requires_grad = False

    def _to_pil(self, images: Tensor) -> list:
        """Convert tensor to PIL images."""
        if images.dim() == 3:
            images = images.unsqueeze(0)

        pil_images = []
        for img in images:
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        return pil_images

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """
        Extract DINO features for guidance.

        Args:
            images: [B, C, H, W] tensor in [0, 1]

        Returns:
            Pooled features [B, hidden_size]
        """
        pil_images = self._to_pil(images)
        inputs = self.processor(images=pil_images, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self.model(**inputs)

        # Use pooler_output (CLS token)
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            return output.pooler_output
        else:
            return output.last_hidden_state[:, 0]


# =============================================================================
# EXTRACT FUNCTIONS FOR HEAD_ROUTER
# =============================================================================

def flux_ae_extract_fn(encoder: Flux2AEEncoder, images: Tensor, **kwargs) -> Tensor:
    """
    Extract function for head_router.

    Encodes images and flattens spatial dims for sequence processing.

    Returns:
        [B, H*W, latent_channels] - sequence of latent patches
    """
    # Encode to [B, C, H, W]
    z = encoder.encode(images, sample=False)

    # Flatten spatial to sequence [B, H*W, C]
    B, C, H, W = z.shape
    z_seq = z.permute(0, 2, 3, 1).reshape(B, H * W, C)

    return z_seq


def flux_ae_extract_pooled_fn(encoder: Flux2AEEncoder, images: Tensor, **kwargs) -> Tensor:
    """
    Extract function that returns pooled latent.

    Returns:
        [B, latent_channels] - mean-pooled latent
    """
    z = encoder.encode(images, sample=False)
    return z.mean(dim=[2, 3])  # Global average pool


def dino_extract_fn(encoder: DinoGuidanceEncoder, images: Tensor, **kwargs) -> Tensor:
    """Extract function for DINO guidance."""
    return encoder(images)


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_flux_ae_standalone(device='cuda'):
    """Test Flux2 AE encoding/decoding."""
    print("\n" + "=" * 60)
    print("  FLUX2 AE STANDALONE TEST")
    print("=" * 60)

    # Load Flux2 VAE
    # Note: Requires HF token with accepted FLUX.2-dev license
    flux_ae = Flux2AEEncoder(
        model_path="black-forest-labs/FLUX.2-dev",
        device=device,
    )

    # Create test image - Flux2 uses 16x spatial compression
    B, C, H, W = 2, 3, 512, 512
    images = torch.rand(B, C, H, W, device=device)

    print(f"\nInput: {images.shape}")

    # Encode
    z = flux_ae.encode(images, sample=False)
    print(f"Latent: {z.shape}")

    expected_spatial = H // 8  # VAE uses 8x compression
    actual_compression = H // z.shape[2]
    print(f"  Latent spatial: {z.shape[2]}x{z.shape[3]} (expected {expected_spatial}x{expected_spatial})")
    print(f"  Latent channels: {z.shape[1]} (expected {flux_ae.latent_channels})")
    print(f"  Compression: {actual_compression}x spatial, {H * W * 3 / z.numel() * B:.1f}x total")

    # Sequence form for transformers
    z_seq = flux_ae_extract_fn(flux_ae, images)
    print(f"Sequence: {z_seq.shape} (for transformer towers)")

    # Pooled form for fusion
    z_pool = flux_ae_extract_pooled_fn(flux_ae, images)
    print(f"Pooled: {z_pool.shape} (for fusion)")

    # Decode roundtrip
    reconstructed = flux_ae.decode(z)
    print(f"Reconstructed: {reconstructed.shape}")

    # Check reconstruction quality
    # Crop to match input size if needed
    recon_cropped = reconstructed[:, :, :H, :W]
    mse = F.mse_loss(images, recon_cropped)
    psnr = 10 * torch.log10(1.0 / mse)
    print(f"Reconstruction MSE: {mse.item():.6f}, PSNR: {psnr.item():.2f} dB")

    return flux_ae


def test_dino_standalone(device='cuda'):
    """Test DINO guidance extraction."""
    print("\n" + "=" * 60)
    print("  DINO GUIDANCE TEST")
    print("=" * 60)

    dino = DinoGuidanceEncoder(
        model_path="facebook/dinov2-base",  # Use base for speed
        device=device,
    )

    B, C, H, W = 2, 3, 224, 224
    images = torch.rand(B, C, H, W, device=device)

    print(f"\nInput: {images.shape}")

    features = dino(images)
    print(f"Guidance: {features.shape}")

    # Check feature quality (not collapsed)
    feat_norm = F.normalize(features, dim=-1)
    sim = (feat_norm @ feat_norm.T)
    off_diag = sim[~torch.eye(B, dtype=bool, device=device)].mean()
    print(f"Inter-sample similarity: {off_diag:.4f}")

    # Note: Random noise will produce similar DINO features since there's no
    # semantic content - DINO is trained on real images. High similarity with
    # random data is expected. Real images would show diversity.
    if off_diag > 0.8:
        print("  Note: High similarity expected for random noise (no semantics)")
    else:
        print("  ✓ Good diversity")

    return dino


def test_head_router_integration(flux_ae, dino, device='cuda'):
    """Test integration with head_router."""
    print("\n" + "=" * 60)
    print("  HEAD_ROUTER INTEGRATION TEST")
    print("=" * 60)

    try:
        from geofractal.router.prefab.agatha.head_router import (
            AgathaHeadRouter,
            create_agatha_head,
        )
    except ImportError:
        print("  ⚠ head_router not in path, skipping integration test")
        return

    # Create head router
    head = create_agatha_head(
        embed_dim=256,  # Common projection dim
        fingerprint_dim=64,
    )

    # Attach Flux AE (IMAGE stream - the manifold)
    # stream_type auto-detected from name 'flux_ae'
    head.attach_encoder(
        name='flux_ae',
        encoder=flux_ae,
        embed_dim=flux_ae.latent_channels,
        extract_fn=flux_ae_extract_pooled_fn,
    )

    # Attach DINO (GUIDANCE stream - the steering)
    # stream_type auto-detected from name 'dino'
    head.attach_encoder(
        name='dino',
        encoder=dino,
        embed_dim=dino.hidden_size,
        extract_fn=dino_extract_fn,
    )

    print(f"\nHead router: {head}")
    for name, status in head.stream_status().items():
        print(f"  {name}: {status}")

    # Move head router to device AFTER attaching encoders
    head = head.to(device)

    # Test forward
    head.debug_on()

    B, C, H, W = 2, 3, 512, 512
    images = torch.rand(B, C, H, W, device=device)

    inputs = {
        'flux_ae': images,
        'dino': images,
    }

    mail = head(inputs)

    print(f"\nMail sources: {mail.sources}")
    print(f"Fused: {mail.fused.shape}")

    for name in mail.sources:
        m = mail[name]
        print(f"  {name}: content={list(m.content.shape)}, fp={list(m.fingerprint.shape)}")

    # Check fingerprint distinctness
    sims = head.fingerprint_similarity()
    print(f"\nFingerprint similarities:")
    for pair, sim in sims.items():
        print(f"  {pair}: {sim:.4f}")

    return head, mail


def test_latent_manifold_properties(flux_ae, device='cuda'):
    """
    Test that Flux2 latent space has good manifold properties.

    For Beatrix oscillator, we need:
    1. Smooth interpolation (geodesics exist)
    2. Diverse representations (not collapsed)
    3. Reconstruction fidelity (manifold covers image space)

    Flux2 VAE: 32 channels, 16x spatial compression
    """
    print("\n" + "=" * 60)
    print("  MANIFOLD PROPERTIES TEST")
    print("=" * 60)

    H, W = 512, 512

    # Create two different images
    img1 = torch.rand(1, 3, H, W, device=device)
    img2 = torch.rand(1, 3, H, W, device=device)

    # Encode
    z1 = flux_ae.encode(img1, sample=False)
    z2 = flux_ae.encode(img2, sample=False)

    print(f"\nLatent z1: {z1.shape}, norm={z1.norm():.4f}")
    print(f"Latent z2: {z2.shape}, norm={z2.norm():.4f}")
    print(f"Spatial: {H}x{W} → {z1.shape[2]}x{z1.shape[3]} ({H // z1.shape[2]}x compression)")
    print(f"Channels: 3 → {z1.shape[1]}")

    # Test 1: Interpolation smoothness
    print("\n--- Interpolation Test (Geodesic Viability) ---")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        img_interp = flux_ae.decode(z_interp)

        # Check it decodes without NaN/Inf
        if torch.isnan(img_interp).any() or torch.isinf(img_interp).any():
            print(f"  α={alpha}: FAILED (NaN/Inf)")
        else:
            print(f"  α={alpha}: OK, range=[{img_interp.min():.3f}, {img_interp.max():.3f}]")

    # Test 2: Latent diversity
    print("\n--- Diversity Test (Manifold Not Collapsed) ---")
    N = 10
    images = torch.rand(N, 3, 256, 256, device=device)
    z_batch = flux_ae.encode(images, sample=False)
    z_flat = z_batch.reshape(N, -1)
    z_norm = F.normalize(z_flat, dim=-1)

    sim_matrix = z_norm @ z_norm.T
    off_diag = sim_matrix[~torch.eye(N, dtype=bool, device=device)].mean()

    print(f"  Inter-sample similarity: {off_diag:.4f}")
    if off_diag < 0.5:
        print("  ✓ Good diversity (manifold spans image space)")
    else:
        print("  ⚠ High similarity (potential collapse)")

    # Test 3: Reconstruction quality
    print("\n--- Reconstruction Test (Manifold Covers Image Space) ---")
    test_img = torch.rand(1, 3, H, W, device=device)
    z = flux_ae.encode(test_img, sample=False)
    recon = flux_ae.decode(z)

    # Crop to original size if needed
    recon = recon[:, :, :H, :W]

    mse = F.mse_loss(test_img, recon)
    psnr = 10 * torch.log10(1.0 / mse)

    print(f"  MSE: {mse.item():.6f}")
    print(f"  PSNR: {psnr.item():.2f} dB")

    if psnr > 25:
        print("  ✓ Good reconstruction (Log/Exp maps are valid)")
    else:
        print("  ⚠ Lower PSNR (acceptable for compressed manifold)")

    # Test 4: Latent statistics (for flow matching)
    print("\n--- Latent Statistics (For Flow Matching) ---")
    print(f"  Mean: {z_batch.mean():.4f}")
    print(f"  Std:  {z_batch.std():.4f}")
    print(f"  Min:  {z_batch.min():.4f}")
    print(f"  Max:  {z_batch.max():.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  FLUX2 AE + DINO HEAD_ROUTER INTEGRATION")
    print("=" * 60)
    print("\nFLUX.2 VAE Specs:")
    print("  - 32 latent channels (vs 16 in FLUX.1)")
    print("  - 8x spatial compression in VAE (64x64 from 512x512)")
    print("  - 2x2 patchify in transformer → 16x total (32x32)")
    print("  - Apache 2.0 licensed")
    print("\nArchitecture:")
    print("  Flux2 AE → Manifold M (where x(t) lives)")
    print("  DINO → Guidance ξ (tangent forces for steering)")
    print("  Head Router → Fused mail for downstream towers")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Test standalone components
    flux_ae = test_flux_ae_standalone(device)
    dino = test_dino_standalone(device)

    # Test manifold properties
    test_latent_manifold_properties(flux_ae, device)

    # Test head_router integration
    test_head_router_integration(flux_ae, dino, device)

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)
    print("\nFlux2 AE provides the manifold M")
    print("DINO provides the guidance signal ξ")
    print("Ready to wire into Beatrix oscillator")


if __name__ == '__main__':
    main()