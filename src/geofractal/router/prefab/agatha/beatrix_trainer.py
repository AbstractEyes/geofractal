"""
BEATRIX TRAINER (Full Integration)
==================================

Complete training pipeline using:
- Head Router (Flux2 AE + DINO)
- Tower Builders (geometric + conv collectives)
- Beatrix Oscillator

Author: AbstractPhil + Claude
Date: December 2024
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Local imports - adjust paths as needed
from geofractal.router.prefab.agatha.beatrix_oscillator import (
    BeatrixOscillator,
    ScheduleType,
)
from geofractal.router.prefab.agatha.beatrix_collective import (
    BeatrixCollective,
    BeatrixCollectiveConfig,
    create_beatrix_collective,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BeatrixTrainingConfig:
    """Complete configuration for Beatrix training."""

    # Manifold dimensions (Flux2 VAE)
    latent_channels: int = 32
    latent_height: int = 64
    latent_width: int = 64

    # Embedding dimensions
    embed_dim: int = 256
    fingerprint_dim: int = 64

    # Collective config
    geometric_types: List[str] = field(default_factory=lambda: [
        'cantor', 'beatrix', 'simplex', 'helix'
    ])
    conv_types: List[str] = field(default_factory=lambda: [
        'wide_resnet', 'frequency'
    ])
    num_theta_probes: int = 4
    use_signed_pairs: bool = True

    # Oscillator config
    num_steps: int = 50
    beta_start: float = 0.1
    beta_end: float = 2.0
    omega_start: float = 1.0
    omega_end: float = 0.1
    kappa_start: float = 1.0
    kappa_end: float = 0.5
    gamma_start: float = 1.0
    gamma_end: float = 0.0
    schedule_type: str = "tesla_369"

    # Training config
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Loss weights
    flow_weight: float = 1.0
    velocity_weight: float = 0.5
    consistency_weight: float = 0.1

    # Noise schedule
    sigma_min: float = 0.001
    sigma_max: float = 1.0

    @property
    def latent_dim(self) -> int:
        return self.latent_channels * self.latent_height * self.latent_width

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return (self.latent_channels, self.latent_height, self.latent_width)


# =============================================================================
# TIMESTEP EMBEDDING
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# =============================================================================
# BEATRIX MODEL
# =============================================================================

class Beatrix(nn.Module):
    """
    Complete Beatrix system.

    Flow:
        fused (from head router) → Collective → Tower forces
                                               ↓
        z_noisy ──────────────────────→ Oscillator → z_pred
                                               ↑
        condition ─────────────────────────────┘
        guidance ──────────────────────────────┘
    """

    def __init__(self, config: BeatrixTrainingConfig):
        super().__init__()
        self.config = config

        # Build collective config
        collective_config = BeatrixCollectiveConfig(
            dim=config.embed_dim,
            fingerprint_dim=config.fingerprint_dim,
            geometric_types=config.geometric_types,
            conv_types=config.conv_types,
            num_theta_probes=config.num_theta_probes,
            use_signed_pairs=config.use_signed_pairs,
        )

        # Tower collective
        self.collective = BeatrixCollective(collective_config)

        # Count towers for oscillator
        num_tower_pairs = (
                len(config.geometric_types) + len(config.conv_types)
        ) if config.use_signed_pairs else 0

        # Oscillator
        self.oscillator = BeatrixOscillator(
            manifold_dim=config.latent_dim,
            tower_dim=config.embed_dim,
            num_tower_pairs=num_tower_pairs,
            num_theta_probes=config.num_theta_probes,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            omega_start=config.omega_start,
            omega_end=config.omega_end,
            kappa_start=config.kappa_start,
            kappa_end=config.kappa_end,
            gamma_start=config.gamma_start,
            gamma_end=config.gamma_end,
            kappa_schedule=ScheduleType(config.schedule_type),
        )

        # Projections
        self.condition_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.embed_dim * 2, config.latent_dim),
        )

        self.guidance_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.embed_dim * 2, config.latent_dim),
        )

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.embed_dim * 2, config.embed_dim),
        )

    def forward(
            self,
            z_noisy: Tensor,  # [B, C, H, W] noisy latent
            fused: Tensor,  # [B, D] from head router
            condition: Tensor,  # [B, D] text/conditioning embedding
            guidance: Optional[Tensor] = None,  # [B, D] DINO guidance
            t: Optional[Tensor] = None,  # [B] timestep
            num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Denoise z_noisy toward clean latent.

        Returns:
            z_pred: Predicted clean latent [B, C, H, W]
        """
        B = z_noisy.shape[0]
        device = z_noisy.device
        num_steps = num_steps or self.config.num_steps

        # Add timestep info to condition if provided
        if t is not None:
            t_emb = self.time_embed(t)
            condition = condition + t_emb
            fused = fused + t_emb  # Also condition the towers

        # Flatten latent for oscillator
        z_flat = z_noisy.reshape(B, -1)

        # Project condition to anchor
        x_ref = self.condition_proj(condition)

        # Project guidance
        guidance_flat = None
        if guidance is not None:
            guidance_flat = self.guidance_proj(guidance)

        # Get tower outputs from collective
        collective_result = self.collective(fused, return_all=True)
        tower_outputs = collective_result['outputs']
        fingerprint = collective_result['fingerprint']

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
        z_pred = z_final.reshape(
            B,
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
        )

        return z_pred

    def generate(
            self,
            fused: Tensor,
            condition: Tensor,
            guidance: Optional[Tensor] = None,
            num_steps: Optional[int] = None,
            noise_scale: float = 1.0,
    ) -> Tensor:
        """Generate from pure noise."""
        B = fused.shape[0]
        device = fused.device

        z_noise = torch.randn(
            B,
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
            device=device,
        ) * noise_scale

        return self.forward(
            z_noisy=z_noise,
            fused=fused,
            condition=condition,
            guidance=guidance,
            num_steps=num_steps,
        )

    def network_to(self, device: torch.device):
        """Move to device (WideRouter compatibility)."""
        self.to(device)
        self.collective.network_to(device)
        return self


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def flow_matching_loss(
        model: Beatrix,
        z_clean: Tensor,
        fused: Tensor,
        condition: Tensor,
        guidance: Optional[Tensor] = None,
        sigma_min: float = 0.001,
        sigma_max: float = 1.0,
        num_steps: int = 10,  # Fewer steps during training
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Flow matching loss: predict clean from noisy.
    """
    B = z_clean.shape[0]
    device = z_clean.device

    # Sample noise level
    sigma = torch.rand(B, 1, 1, 1, device=device) * (sigma_max - sigma_min) + sigma_min

    # Add noise
    noise = torch.randn_like(z_clean)
    z_noisy = z_clean + sigma * noise

    # Timestep for conditioning
    t = sigma.squeeze() / sigma_max

    # Predict clean
    z_pred = model(z_noisy, fused, condition, guidance, t=t, num_steps=num_steps)

    # MSE loss
    loss = F.mse_loss(z_pred, z_clean)

    # Metrics
    with torch.no_grad():
        mse = loss.item()
        psnr = -10 * math.log10(mse) if mse > 0 else float('inf')

    return loss, {'flow_mse': mse, 'flow_psnr': psnr, 'sigma_mean': sigma.mean().item()}


def velocity_matching_loss(
        model: Beatrix,
        z_clean: Tensor,
        fused: Tensor,
        condition: Tensor,
        guidance: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Velocity field matching (rectified flow style).
    """
    B = z_clean.shape[0]
    device = z_clean.device

    # Sample time
    t = torch.rand(B, device=device)

    # Noise
    z_noise = torch.randn_like(z_clean)

    # Interpolate
    t_expand = t.view(B, 1, 1, 1)
    z_t = t_expand * z_clean + (1 - t_expand) * z_noise

    # Target velocity (straight path)
    v_target = z_clean - z_noise

    # Model prediction (single step)
    z_pred = model(z_t, fused, condition, guidance, t=t, num_steps=1)

    # Predicted velocity
    v_pred = z_pred - z_t

    # Normalize by expected step size
    v_target_scaled = v_target / model.config.num_steps

    # Loss
    loss = F.mse_loss(v_pred, v_target_scaled)

    return loss, {'velocity_mse': loss.item()}


# =============================================================================
# TRAINER
# =============================================================================

class BeatrixTrainer:
    """Training manager for Beatrix."""

    def __init__(
            self,
            model: Beatrix,
            config: BeatrixTrainingConfig,
            device: str = 'cuda',
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device)

        # Move model
        self.model.network_to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
        )

        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

    def train_step(
            self,
            z_clean: Tensor,
            fused: Tensor,
            condition: Tensor,
            guidance: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        z_clean = z_clean.to(self.device)
        fused = fused.to(self.device)
        condition = condition.to(self.device)
        if guidance is not None:
            guidance = guidance.to(self.device)

        # Flow matching loss
        loss_flow, metrics_flow = flow_matching_loss(
            self.model,
            z_clean,
            fused,
            condition,
            guidance,
            self.config.sigma_min,
            self.config.sigma_max,
            num_steps=10,  # Fewer steps during training
        )

        # Velocity matching loss
        loss_vel, metrics_vel = velocity_matching_loss(
            self.model,
            z_clean,
            fused,
            condition,
            guidance,
        )

        # Combined loss
        loss = (
                self.config.flow_weight * loss_flow +
                self.config.velocity_weight * loss_vel
        )

        # Backward
        loss.backward()

        # Gradient clipping
        if self.config.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )
        else:
            grad_norm = 0.0

        # Step
        self.optimizer.step()
        self.global_step += 1

        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'loss_flow': loss_flow.item(),
            'loss_velocity': loss_vel.item(),
            'grad_norm': grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
            'lr': self.optimizer.param_groups[0]['lr'],
            **metrics_flow,
            **metrics_vel,
        }

        return metrics

    def train_epoch(
            self,
            dataloader: DataLoader,
            log_fn: Optional[Callable[[Dict], None]] = None,
            log_interval: int = 10,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {}

        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            z_clean = batch['latent']
            fused = batch['fused']
            condition = batch['condition']
            guidance = batch.get('guidance', None)

            metrics = self.train_step(z_clean, fused, condition, guidance)

            # Accumulate
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

            # Log
            if batch_idx % log_interval == 0:
                print(f"  Step {self.global_step}: "
                      f"loss={metrics['loss']:.6f}, "
                      f"psnr={metrics.get('flow_psnr', 0):.2f}dB, "
                      f"lr={metrics['lr']:.2e}")

                if log_fn:
                    log_fn(metrics)

        # Average
        avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

        # Scheduler step
        self.scheduler.step()
        self.epoch += 1

        # Track best
        if avg_metrics['loss'] < self.best_loss:
            self.best_loss = avg_metrics['loss']
            avg_metrics['is_best'] = True
        else:
            avg_metrics['is_best'] = False

        return avg_metrics

    def save_checkpoint(self, path: str, extra: Dict = None):
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
        }
        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> Dict:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        print(f"Loaded checkpoint: {path} (epoch {self.epoch}, step {self.global_step})")
        return checkpoint


# =============================================================================
# DUMMY DATASET
# =============================================================================

class DummyBeatrixDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, config: BeatrixTrainingConfig, num_samples: int = 1000):
        self.config = config
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'latent': torch.randn(*self.config.latent_shape),
            'fused': torch.randn(self.config.embed_dim),
            'condition': torch.randn(self.config.embed_dim),
            'guidance': torch.randn(self.config.embed_dim),
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  BEATRIX TRAINING TEST")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Config
    config = BeatrixTrainingConfig(
        batch_size=4,
        num_steps=10,
        num_epochs=2,
        geometric_types=['cantor', 'beatrix'],
        conv_types=['wide_resnet'],
        num_theta_probes=2,
    )

    print(f"\nConfig:")
    print(f"  Latent shape: {config.latent_shape}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Oscillator steps: {config.num_steps}")

    # Model
    print("\nBuilding model...")
    model = Beatrix(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    trainer = BeatrixTrainer(model, config, device)

    # Dataset
    dataset = DummyBeatrixDataset(config, num_samples=32)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    print("\n--- Training ---")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        metrics = trainer.train_epoch(dataloader, log_interval=2)
        print(f"  Avg loss: {metrics['loss']:.6f}")
        print(f"  Avg PSNR: {metrics.get('flow_psnr', 0):.2f}dB")
        if metrics.get('is_best'):
            print("  ★ New best!")

    # Test generation
    print("\n--- Generation Test ---")
    model.eval()
    with torch.no_grad():
        fused = torch.randn(2, config.embed_dim, device=device)
        condition = torch.randn(2, config.embed_dim, device=device)
        guidance = torch.randn(2, config.embed_dim, device=device)

        z_gen = model.generate(fused, condition, guidance, num_steps=20)
        print(f"Generated: {z_gen.shape}")
        print(f"  Norm: {z_gen.norm():.4f}")
        print(f"  Range: [{z_gen.min():.4f}, {z_gen.max():.4f}]")

    print("\n" + "=" * 60)
    print("  ✓ BEATRIX TRAINING READY")
    print("=" * 60)