"""
BEATRIX TRAINER
===============

Training infrastructure for Beatrix diffusion.

Contains ONLY:
    - BeatrixTrainer (optimizer, scheduler, training loop)
    - TensorBoard logging
    - Checkpoint save/load (training state)
    - Dataset utilities

NO model code. Model is in beatrix.py.

Author: AbstractPhil + Claude
Date: December 2024
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from geofractal.router.prefab.agatha.beatrix import Beatrix, BeatrixConfig


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainerConfig:
    """Training hyperparameters."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    num_epochs: int = 100
    save_every: int = 10
    log_interval: int = 10


# =============================================================================
# DATASET WRAPPER
# =============================================================================

class BeatrixDataset(Dataset):
    """
    Wraps any image dataset to return dict format for Beatrix.

    Beatrix expects: {'flux_ae': images, 'dino': images}
    """

    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle (image, label) tuple
        if isinstance(item, tuple):
            image = item[0]
        else:
            image = item

        if self.transform:
            image = self.transform(image)

        # Return dict format for Beatrix
        return {'flux_ae': image, 'dino': image}


def collate_beatrix(batch):
    """Collate function for Beatrix datasets."""
    flux_ae = torch.stack([b['flux_ae'] for b in batch])
    dino = torch.stack([b['dino'] for b in batch])
    return {'flux_ae': flux_ae, 'dino': dino}


# =============================================================================
# TRAINER
# =============================================================================

class BeatrixTrainer:
    """
    Training manager for Beatrix.

    Handles:
        - Optimizer and scheduler
        - Training loop
        - TensorBoard logging
        - Checkpoint save/load (training state only)
    """

    def __init__(
        self,
        model: Beatrix,
        config: Optional[TrainerConfig] = None,
        device: str = 'cuda',
        log_dir: Optional[str] = None,
    ):
        self.model = model
        self.config = config or TrainerConfig()
        self.device = torch.device(device)

        model.network_to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        self.scheduler = None
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # TensorBoard
        self.writer = None
        self.log_dir = log_dir
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        batch = {k: v.to(self.device) for k, v in batch.items()}

        loss, metrics = self.model.compute_loss(batch)
        loss.backward()

        if self.config.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(),
                self.config.gradient_clip,
            )
            metrics['grad_norm'] = grad_norm.item()

        self.optimizer.step()
        self.global_step += 1

        # TensorBoard step logging
        if self.writer:
            self.writer.add_scalar('step/loss', metrics['loss'], self.global_step)
            self.writer.add_scalar('step/cos_sim', metrics.get('cos_sim', 0), self.global_step)
            if 'grad_norm' in metrics:
                self.writer.add_scalar('step/grad_norm', metrics['grad_norm'], self.global_step)

        return metrics

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {}

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)

            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'cos': f"{metrics.get('cos_sim', 0):.3f}",
                })

        avg = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

        if avg['loss'] < self.best_loss:
            self.best_loss = avg['loss']
            avg['is_best'] = True

        return avg

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: Optional[int] = None,
        save_dir: str = "./checkpoints",
    ):
        """Full training loop."""
        num_epochs = num_epochs or self.config.num_epochs
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )

        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            metrics = self.train_epoch(dataloader)

            # Console output
            print(f"  Loss: {metrics['loss']:.6f}, Cos: {metrics.get('cos_sim', 0):.3f}")

            # TensorBoard epoch logging
            if self.writer:
                self.writer.add_scalar('epoch/loss', metrics['loss'], epoch)
                self.writer.add_scalar('epoch/cos_sim', metrics.get('cos_sim', 0), epoch)
                self.writer.add_scalar('epoch/lr', self.scheduler.get_last_lr()[0], epoch)

            if metrics.get('is_best'):
                print("  ★ New best!")
                self.save_checkpoint(f"{save_dir}/beatrix_best.pt")

            self.scheduler.step()

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"{save_dir}/beatrix_epoch_{epoch + 1}.pt")

        self.save_checkpoint(f"{save_dir}/beatrix_final.pt")

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, path: str):
        """Save training checkpoint (training state + denoiser weights)."""
        torch.save({
            'denoiser': self.model.denoiser.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.model.config,
            'trainer_config': self.config,
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.denoiser.load_state_dict(ckpt['denoiser'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler') and self.scheduler:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.global_step = ckpt['global_step']
        self.epoch = ckpt['epoch']
        self.best_loss = ckpt['best_loss']
        print(f"Checkpoint loaded: {path} (epoch {self.epoch}, step {self.global_step})")

    def get_log_dir(self) -> Optional[str]:
        """Return tensorboard log directory."""
        return self.log_dir


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_trainer(
    model: Beatrix,
    learning_rate: float = 1e-4,
    num_epochs: int = 100,
    device: str = 'cuda',
    log_dir: Optional[str] = None,
) -> BeatrixTrainer:
    """Create trainer with common settings."""
    config = TrainerConfig(
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
    return BeatrixTrainer(model, config, device, log_dir)