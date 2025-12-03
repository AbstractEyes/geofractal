"""
geofractal.router.collective
============================
High-level API for router collectives.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0

RouterCollective is the main interface for:
- Building multi-stream architectures
- Training with coordination
- Inference with emergent specialization

Usage:
    collective = RouterCollective.from_streams([
        FrozenStream.from_pretrained("openai/clip-vit-base-patch32", config),
        FrozenStream.from_pretrained("openai/clip-vit-large-patch14", config),
    ], config)

    collective.fit(train_loader, val_loader)

    logits = collective(batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional, Any, Union
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from tqdm.auto import tqdm

from geofractal.router.config import CollectiveConfig
from geofractal.router.core import GlobalFractalRouterConfig
from geofractal.router.registry import RouterMailbox, get_registry
from geofractal.router.streams.base import BaseStream


class RouterCollective(nn.Module):
    """
    Collective of streams coordinated via GlobalFractalRouter.

    Takes multiple expert streams (frozen, trainable, or feature-based)
    and coordinates them through fingerprint-based routing and mailbox
    communication.

    Key insight: Individual streams don't need to be accurate classifiers.
    They need to provide divergent perspectives that the collective
    can triangulate into accurate predictions.

    Proven Results:
        - ImageNet: 5 streams at 0.1% → 84.68% collective
        - FashionMNIST: 10% + 10% + 10% = 93.4%
        - Dual CLIP: 98.6% frozen → 92.6% accuracy

    Usage:
        # From streams
        collective = RouterCollective.from_streams(streams, config)

        # Training
        history = collective.fit(train_loader, val_loader)

        # Inference
        logits, info = collective(batch)
    """

    def __init__(
            self,
            streams: List[BaseStream],
            config: CollectiveConfig,
            cooperation_group: str = "default_collective",
    ):
        super().__init__()

        self.config = config
        self.cooperation_group = cooperation_group

        # Store streams
        self.streams = nn.ModuleList(streams)
        self.stream_names = [s.name for s in streams]

        # Shared mailbox
        router_config = config.to_router_config()
        self.mailbox = RouterMailbox(router_config)

        # Fusion layer
        num_streams = len(streams)
        self.fusion = nn.Sequential(
            nn.Linear(config.feature_dim * num_streams, config.feature_dim * 2),
            nn.LayerNorm(config.feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.feature_dim * 2, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
        )

        # Classification head
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)

        # Per-stream classifiers (for measuring individual contribution)
        self.stream_classifiers = nn.ModuleDict({
            name: nn.Linear(config.feature_dim, config.num_classes)
            for name in self.stream_names
        })

    def forward(
            self,
            x: Union[torch.Tensor, Dict[str, torch.Tensor]],
            return_individual: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through collective.

        Args:
            x: Input tensor [B, ...] or dict of features per stream
            return_individual: Return per-stream predictions

        Returns:
            logits: [B, num_classes] collective prediction
            info: Dict with routing metrics
        """
        # Clear mailbox at start
        self.mailbox.clear()

        # Process each stream
        stream_features = {}
        stream_infos = {}

        for i, stream in enumerate(self.streams):
            # Get input for this stream
            if isinstance(x, dict):
                stream_input = x[stream.name]
            else:
                stream_input = x

            # Get target fingerprint (next stream or None)
            if i < len(self.streams) - 1:
                target_fp = self.streams[i + 1].fingerprint
            else:
                target_fp = None

            # Forward through stream
            routed, info = stream(stream_input, self.mailbox, target_fp)

            # Pool across slots
            pooled = stream.pool(routed)  # [B, feature_dim]
            stream_features[stream.name] = pooled
            stream_infos[stream.name] = info

        # Fuse all streams
        fused = torch.cat(
            [stream_features[n] for n in self.stream_names],
            dim=-1
        )
        fused = self.fusion(fused)

        # Classify
        logits = self.classifier(fused)

        # Build info
        info = {
            'stream_infos': stream_infos,
            'mailbox_messages': len(self.mailbox.messages),
            'mean_route_entropy': np.mean([
                i['route_entropy'] for i in stream_infos.values()
            ]),
        }

        # Individual predictions
        if return_individual:
            individual_logits = {
                name: self.stream_classifiers[name](stream_features[name])
                for name in self.stream_names
            }
            info['individual_logits'] = individual_logits

        return logits, info

    def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: Optional[int] = None,
            lr: Optional[float] = None,
            callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the collective.

        Only trainable parameters are optimized (frozen streams excluded).
        Uses AMP by default on CUDA.

        Args:
            train_loader: Training data
            val_loader: Validation data (optional)
            epochs: Override config.epochs
            lr: Override config.lr
            callbacks: Optional training callbacks

        Returns:
            history: Dict with training metrics
        """
        epochs = epochs or self.config.epochs
        lr = lr or self.config.lr

        # Only trainable parameters
        params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

        # Warmup + cosine schedule
        total_steps = len(train_loader) * epochs
        warmup_steps = len(train_loader) * self.config.warmup_epochs

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = GradScaler() if self.config.use_amp else None

        history = defaultdict(list)
        best_acc = 0

        for epoch in range(epochs):
            # Training
            self.train()
            epoch_loss = 0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in pbar:
                # Handle different input formats
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, labels = batch
                else:
                    x, labels = batch, None

                # Move to device
                if isinstance(x, dict):
                    x = {k: v.to(self.config.device, non_blocking=True)
                         for k, v in x.items()}
                else:
                    x = x.to(self.config.device, non_blocking=True)

                if labels is not None:
                    labels = labels.to(self.config.device, non_blocking=True)

                optimizer.zero_grad()

                if self.config.use_amp:
                    with autocast():
                        logits, info = self(x)
                        loss = F.cross_entropy(logits, labels)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, info = self(x)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
                    optimizer.step()

                scheduler.step()

                epoch_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct / total * 100:.1f}%",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                })

            # Validation
            if val_loader is not None:
                val_acc, stream_accs = self.evaluate(val_loader)
                history['val_acc'].append(val_acc)
                history['stream_accs'].append(stream_accs)

                if val_acc > best_acc:
                    best_acc = val_acc
                    tqdm.write(f"  ★ New best: {best_acc * 100:.2f}%")

            history['train_loss'].append(epoch_loss / total)
            history['train_acc'].append(correct / total)

            # Log
            stream_str = ' | '.join([
                f"{k[:8]}: {v * 100:.1f}%"
                for k, v in (stream_accs.items() if val_loader else {})
            ])
            tqdm.write(f"Epoch {epoch + 1:3d} | Loss: {epoch_loss / total:.4f} | "
                       f"Val: {val_acc * 100:.2f}% | {stream_str}" if val_loader else "")

        return dict(history)

    def evaluate(
            self,
            loader: DataLoader,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate collective and per-stream accuracy.

        Args:
            loader: Evaluation data

        Returns:
            accuracy: Collective accuracy
            stream_accs: Per-stream accuracy dict
        """
        self.eval()
        correct = 0
        total = 0
        stream_correct = defaultdict(int)

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval", leave=False):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, labels = batch
                else:
                    x, labels = batch, None

                if isinstance(x, dict):
                    x = {k: v.to(self.config.device, non_blocking=True)
                         for k, v in x.items()}
                else:
                    x = x.to(self.config.device, non_blocking=True)

                labels = labels.to(self.config.device, non_blocking=True)

                if self.config.use_amp:
                    with autocast():
                        logits, info = self(x, return_individual=True)
                else:
                    logits, info = self(x, return_individual=True)

                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

                for name, ind_logits in info['individual_logits'].items():
                    stream_correct[name] += (
                            ind_logits.argmax(dim=1) == labels
                    ).sum().item()

        accuracy = correct / total
        stream_accs = {k: v / total for k, v in stream_correct.items()}

        return accuracy, stream_accs

    @classmethod
    def from_streams(
            cls,
            streams: List[BaseStream],
            config: CollectiveConfig,
    ) -> "RouterCollective":
        """
        Create collective from list of streams.

        Automatically builds hierarchical chain topology.
        """
        # Reset registry
        get_registry().reset()

        return cls(streams, config)

    @classmethod
    def from_pretrained_models(
            cls,
            model_names: List[str],
            config: CollectiveConfig,
            use_fp16: bool = True,
    ) -> "RouterCollective":
        """
        Create collective from HuggingFace model names.

        All models are loaded frozen with FP16 by default.

        Args:
            model_names: List of HuggingFace model identifiers
            config: Collective configuration
            use_fp16: Use FP16 for inference
        """
        from geofractal.router.streams.frozen import FrozenStream

        get_registry().reset()

        streams = []
        parent_id = None

        for model_name in model_names:
            stream = FrozenStream.from_pretrained(
                model_name,
                config=config,
                parent_id=parent_id,
                use_fp16=use_fp16,
            )
            streams.append(stream)
            parent_id = stream.module_id

        return cls(streams, config)

    @classmethod
    def from_feature_dims(
            cls,
            feature_configs: Dict[str, int],
            config: CollectiveConfig,
    ) -> "RouterCollective":
        """
        Create collective for pre-extracted features.

        Args:
            feature_configs: Dict mapping name to feature dimension
            config: Collective configuration

        Example:
            collective = RouterCollective.from_feature_dims({
                'clip_vit_b32': 512,
                'clip_vit_l14': 768,
                'dinov2_base': 768,
            }, config)
        """
        from geofractal.router.streams.feature import FeatureStream

        get_registry().reset()

        streams = []
        parent_id = None

        for name, input_dim in feature_configs.items():
            stream = FeatureStream(
                config=config,
                name=name,
                input_dim=input_dim,
                parent_id=parent_id,
            )
            streams.append(stream)
            parent_id = stream.module_id

        return cls(streams, config)