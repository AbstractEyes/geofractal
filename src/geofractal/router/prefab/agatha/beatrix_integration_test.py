"""
BEATRIX INTEGRATION TEST
========================
geofractal.router.prefab.agatha.beatrix_integration_test

Wire together:
    - Flux2 AE (manifold M)
    - DINO (guidance ξ)
    - Head Router (stream fusion)
    - Tower Collectives (force generators)
    - Oscillator (dynamics engine)

This is the complete Tesla translation matrix.

Author: AbstractPhil + Claude
Date: December 2025
License: Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# Import components (adjust paths as needed)
from beatrix_oscillator import (
    BeatrixOscillator,
    BeatrixCore,
    OscillatorState,
    ScheduleType,
    create_beatrix_oscillator,
)


# These would come from your geofractal package
# from geofractal.router.prefab.agatha.head_router import create_agatha_head
# from geofractal.tower.geometric_tower_builder import ConfigurableCollective
# from geofractal.tower.geometric_conv_tower_builder import WideResNetTower


class IntegratedBeatrix(nn.Module):
    """
    Full Beatrix system with all components wired together.

    Flow:
        Image → Flux2 AE → z (manifold position)
        Image → DINO → ξ (guidance force)
        (z, ξ) → Head Router → fused mail
        fused → Towers → opinions y_i
        (z, y_i, ξ) → Oscillator → z'
        z' → Flux2 AE decode → Image'

    Training:
        1. Encode real image to z_target
        2. Add noise: z_noisy = z_target + σ·ε
        3. Run oscillator: z_pred = oscillate(z_noisy, condition, towers, guidance)
        4. Loss = MSE(z_pred, z_target) or flow matching loss

    Inference:
        1. Start from pure noise z_0
        2. Run oscillator with guidance
        3. Decode z_final → image
    """

    def __init__(
            self,
            # Manifold config (Flux2)
            latent_channels: int = 32,
            latent_height: int = 64,
            latent_width: int = 64,
            # Embedding config
            embed_dim: int = 256,
            fingerprint_dim: int = 64,
            # Tower config
            num_tower_pairs: int = 4,  # Cantor, Simplex, Shape, Wide
            num_theta_probes: int = 4,
            tower_dim: int = 256,
            # Oscillator config
            num_steps: int = 50,
            beta_range: Tuple[float, float] = (0.1, 2.0),
            omega_range: Tuple[float, float] = (1.0, 0.1),
            schedule_type: str = "tesla_369",
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.latent_dim = latent_channels * latent_height * latent_width
        self.embed_dim = embed_dim
        self.num_steps = num_steps

        # Oscillator
        schedule = ScheduleType(schedule_type)
        self.oscillator = BeatrixOscillator(
            manifold_dim=self.latent_dim,
            tower_dim=tower_dim,
            num_tower_pairs=num_tower_pairs,
            num_theta_probes=num_theta_probes,
            beta_start=beta_range[0],
            beta_end=beta_range[1],
            omega_start=omega_range[0],
            omega_end=omega_range[1],
            kappa_schedule=schedule,
        )

        # Projections
        self.condition_proj = nn.Linear(embed_dim, self.latent_dim)
        self.guidance_proj = nn.Linear(embed_dim, self.latent_dim)

        # Tower placeholders (would be real towers in full system)
        num_towers = num_tower_pairs * 2 + num_theta_probes
        self.tower_projs = nn.ModuleList([
            nn.Linear(embed_dim, tower_dim) for _ in range(num_towers)
        ])

    def encode_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """Project text embedding to manifold anchor."""
        return self.condition_proj(condition)

    def encode_guidance(self, guidance: torch.Tensor) -> torch.Tensor:
        """Project DINO features to manifold guidance."""
        return self.guidance_proj(guidance)

    def compute_tower_outputs(
            self,
            fused: torch.Tensor,  # [B, embed_dim] from head router
    ) -> List[torch.Tensor]:
        """
        Compute tower opinions from fused representation.

        In the full system, this would run through actual tower collectives.
        Here we use simple projections as placeholders.
        """
        return [proj(fused) for proj in self.tower_projs]

    def forward(
            self,
            z_noisy: torch.Tensor,  # [B, C, H, W] noisy latent
            condition: torch.Tensor,  # [B, embed_dim] text embedding
            fused: torch.Tensor,  # [B, embed_dim] from head router
            guidance: Optional[torch.Tensor] = None,  # [B, embed_dim] DINO
            fingerprint: Optional[torch.Tensor] = None,
            num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Denoise z_noisy toward clean latent.

        Args:
            z_noisy: Noisy latent from VAE encoding + noise
            condition: Text conditioning
            fused: Fused representation from head router
            guidance: DINO guidance
            fingerprint: For tower routing

        Returns:
            Denoised latent [B, C, H, W]
        """
        B = z_noisy.shape[0]
        num_steps = num_steps or self.num_steps

        # Flatten latent
        z_flat = z_noisy.reshape(B, -1)

        # Project condition to anchor
        x_ref = self.encode_condition(condition)

        # Project guidance
        guidance_flat = None
        if guidance is not None:
            guidance_flat = self.encode_guidance(guidance)

        # Compute tower outputs
        tower_outputs = self.compute_tower_outputs(fused)

        # Run oscillator
        z_final = self.oscillator(
            x_init=z_flat,
            x_ref=x_ref,
            tower_outputs=tower_outputs,
            guidance=guidance_flat,
            state_fingerprint=fingerprint,
            num_steps=num_steps,
        )

        # Reshape to spatial
        z_out = z_final.reshape(B, self.latent_channels, self.latent_height, self.latent_width)

        return z_out

    def generate(
            self,
            condition: torch.Tensor,  # [B, embed_dim]
            fused: torch.Tensor,  # [B, embed_dim]
            guidance: Optional[torch.Tensor] = None,
            fingerprint: Optional[torch.Tensor] = None,
            num_steps: Optional[int] = None,
            noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate from pure noise.

        Returns:
            Generated latent [B, C, H, W]
        """
        B = condition.shape[0]
        device = condition.device

        # Start from noise
        z_noise = torch.randn(
            B, self.latent_channels, self.latent_height, self.latent_width,
            device=device
        ) * noise_scale

        return self.forward(
            z_noisy=z_noise,
            condition=condition,
            fused=fused,
            guidance=guidance,
            fingerprint=fingerprint,
            num_steps=num_steps,
        )


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def flow_matching_loss(
        model: IntegratedBeatrix,
        z_clean: torch.Tensor,  # Clean latent from VAE
        condition: torch.Tensor,  # Text embedding
        fused: torch.Tensor,  # From head router
        guidance: Optional[torch.Tensor] = None,
        sigma_min: float = 0.001,
        sigma_max: float = 1.0,
) -> torch.Tensor:
    """
    Flow matching training loss.

    1. Sample noise level σ ~ U[σ_min, σ_max]
    2. Add noise: z_noisy = z_clean + σ·ε
    3. Predict clean: z_pred = model(z_noisy, condition, ...)
    4. Loss = ||z_pred - z_clean||²
    """
    B = z_clean.shape[0]
    device = z_clean.device

    # Sample noise level
    sigma = torch.rand(B, 1, 1, 1, device=device) * (sigma_max - sigma_min) + sigma_min

    # Add noise
    noise = torch.randn_like(z_clean)
    z_noisy = z_clean + sigma * noise

    # Predict clean
    z_pred = model(z_noisy, condition, fused, guidance)

    # MSE loss
    loss = F.mse_loss(z_pred, z_clean)

    return loss


def velocity_matching_loss(
        model: IntegratedBeatrix,
        z_0: torch.Tensor,  # Clean latent (target)
        z_1: torch.Tensor,  # Noise
        condition: torch.Tensor,
        fused: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Velocity field matching loss (rectified flow style).

    Target velocity: v* = z_0 - z_1 (straight line from noise to clean)
    Model predicts: v = oscillator output direction

    Loss = ||v - v*||²
    """
    B = z_0.shape[0]
    device = z_0.device

    # Sample time
    if t is None:
        t = torch.rand(B, 1, 1, 1, device=device)

    # Interpolate
    z_t = t * z_0 + (1 - t) * z_1

    # Target velocity (straight path)
    v_target = z_0 - z_1

    # Model prediction (single step from z_t)
    z_pred = model(z_t, condition, fused, guidance, num_steps=1)

    # Predicted velocity
    v_pred = z_pred - z_t

    # Loss
    loss = F.mse_loss(v_pred, v_target * (1.0 / model.num_steps))

    return loss


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  INTEGRATED BEATRIX TEST")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create model
    model = IntegratedBeatrix(
        latent_channels=32,
        latent_height=64,
        latent_width=64,
        embed_dim=256,
        num_tower_pairs=4,
        num_theta_probes=4,
        num_steps=20,
        schedule_type="tesla_369",
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test inputs
    B = 2
    z_clean = torch.randn(B, 32, 64, 64, device=device)
    condition = torch.randn(B, 256, device=device)
    fused = torch.randn(B, 256, device=device)
    guidance = torch.randn(B, 256, device=device)

    print("\nTesting forward pass...")
    z_noisy = z_clean + 0.5 * torch.randn_like(z_clean)
    z_pred = model(z_noisy, condition, fused, guidance)
    print(f"  Input: {z_noisy.shape}")
    print(f"  Output: {z_pred.shape}")

    print("\nTesting generation...")
    z_gen = model.generate(condition, fused, guidance)
    print(f"  Generated: {z_gen.shape}")

    print("\nTesting flow matching loss...")
    loss = flow_matching_loss(model, z_clean, condition, fused, guidance)
    print(f"  Loss: {loss.item():.6f}")

    print("\nTesting velocity matching loss...")
    z_noise = torch.randn_like(z_clean)
    loss_v = velocity_matching_loss(model, z_clean, z_noise, condition, fused, guidance)
    print(f"  Loss: {loss_v.item():.6f}")

    print("\n" + "=" * 60)
    print("  ✓ ALL TESTS PASSED")
    print("=" * 60)
    print("\nThe oscillator is ready.")
    print("Tesla's resonant architecture lives in silicon.")