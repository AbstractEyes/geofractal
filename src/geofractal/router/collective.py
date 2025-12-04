"""
geofractal.router.collective
============================
High-level API for router collectives with training utilities.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0

RouterCollective provides:
- Building multi-stream architectures
- Training with coordination
- Inference with emergent specialization
- Emergence metrics (ρ = collective / max individual)

Usage:
    # From specs (recommended)
    collective = RouterCollective.from_specs(
        stream_specs=[
            StreamSpec.feature_vector("clip_b32", input_dim=512),
            StreamSpec.feature_vector("clip_l14", input_dim=768),
        ],
        config=CollectiveConfig(feature_dim=512, num_classes=1000),
    )

    # Training
    history = collective.fit(train_loader, val_loader)

    # Inference
    logits, info = collective(batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional, Any, Union
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm

from geofractal.router.config import CollectiveConfig, GlobalFractalRouterConfig
from geofractal.router.registry import RouterMailbox, get_registry
from geofractal.router.head import HeadBuilder, HeadConfig, ComposedHead
from geofractal.router.fusion import FusionBuilder, FusionStrategy
from geofractal.router.streams import (
    BaseStream,
    StreamBuilder,
    FeatureVectorStream,
    TrainableVectorStream,
    SequenceStream,
    InputShape,
)
from geofractal.router.factory import StreamSpec, HeadSpec, FusionSpec


class RouterCollective(nn.Module):
    """
    Collective of streams coordinated through fingerprint-based routing.

    A collective consists of multiple streams, each with its own head,
    that communicate through a shared mailbox and fuse their perspectives
    into a unified prediction.

    Key insight: Individual streams don't need to be accurate classifiers.
    They need to provide divergent perspectives that the collective
    can triangulate into accurate predictions.

    Proven Results:
        - ImageNet: 5 streams at 0.1% → 84.68% collective (ρ = 847)
        - FashionMNIST: 10% + 10% + 10% = 93.4% (ρ = 9.34)
        - Dual CLIP: 98.6% frozen → 92.6% accuracy

    The emergence ratio ρ = collective_acc / max(individual_acc) measures
    how much the collective exceeds its best individual. ρ > 1 indicates
    emergence.
    """

    def __init__(
            self,
            streams: nn.ModuleDict,
            heads: nn.ModuleDict,
            fusion: nn.Module,
            classifier: nn.Module,
            config: CollectiveConfig,
            stream_dims: Optional[Dict[str, int]] = None,
            stream_input_shapes: Optional[Dict[str, str]] = None,
    ):
        super().__init__()

        self.config = config
        self.streams = streams
        self.heads = heads
        self.fusion = fusion
        self.classifier = classifier

        self.stream_names = list(streams.keys())
        self._stream_dims = stream_dims or {}
        self._stream_input_shapes = stream_input_shapes or {}

        # Shared mailbox for inter-stream coordination
        self.mailbox = RouterMailbox(config.to_router_config())

        # Projections to fusion dimension
        self.projections = nn.ModuleDict()
        for name in self.stream_names:
            stream_dim = self._stream_dims.get(name, config.feature_dim)
            if stream_dim != config.feature_dim:
                self.projections[name] = nn.Linear(stream_dim, config.feature_dim)
            else:
                self.projections[name] = nn.Identity()

        # Per-stream classifiers for measuring individual contribution
        self.stream_classifiers = nn.ModuleDict({
            name: nn.Linear(config.feature_dim, config.num_classes)
            for name in self.stream_names
        })

    def forward(
            self,
            x: Union[torch.Tensor, Dict[str, torch.Tensor]],
            return_individual: bool = False,
            return_emergence: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through collective.

        Args:
            x: Input tensor [B, ...] or dict of inputs per stream
            return_individual: Return per-stream predictions
            return_emergence: Compute emergence metrics (requires labels)

        Returns:
            logits: [B, num_classes] collective prediction
            info: Dict with routing metrics, individual predictions, emergence
        """
        # Clear mailbox at start of forward pass
        self.mailbox.clear()

        stream_outputs = {}
        stream_infos = {}

        for i, name in enumerate(self.stream_names):
            # Get input for this stream
            if isinstance(x, dict):
                stream_input = x[name]
            else:
                stream_input = x

            # Encode through stream
            encoded = self.streams[name](stream_input)

            # Get target fingerprint for adjacent gating
            if i < len(self.stream_names) - 1:
                next_name = self.stream_names[i + 1]
                target_fp = self.heads[next_name].fingerprint
            else:
                target_fp = None

            # Route through head
            head = self.heads[name]
            routed, info = head(encoded, target_fingerprint=target_fp)

            # Post to mailbox (detached - no gradient flow)
            self.mailbox.post(
                module_id=id(head),
                name=name,
                content=info.get('routing_state', routed.mean(dim=1)).detach(),
            )

            # Pool and project
            pooled = self._pool(routed, name)
            projected = self.projections[name](pooled)

            stream_outputs[name] = projected
            stream_infos[name] = info

        # Fuse all streams
        fused = self.fusion(stream_outputs)

        # Classify
        logits = self.classifier(fused)

        # Build info dict
        info = {
            'stream_infos': stream_infos,
            'mailbox_messages': len(self.mailbox.messages),
        }

        # Aggregate routing metrics
        route_entropies = [
            i.get('route_entropy', 0) for i in stream_infos.values()
        ]
        if route_entropies:
            info['mean_route_entropy'] = np.mean(route_entropies)

        # Individual predictions
        if return_individual:
            individual_logits = {
                name: self.stream_classifiers[name](stream_outputs[name])
                for name in self.stream_names
            }
            info['individual_logits'] = individual_logits

        return logits, info

    def _pool(self, x: torch.Tensor, stream_name: str) -> torch.Tensor:
        """Pool stream output to [B, D]."""
        input_shape = self._stream_input_shapes.get(stream_name, 'vector')

        if input_shape == 'vector' or x.dim() == 2:
            return x
        elif x.dim() == 3:
            # [B, S, D] -> [B, D] via mean pooling
            return x.mean(dim=1)
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")

    def compute_emergence(
            self,
            collective_acc: float,
            individual_accs: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute emergence metrics.

        Args:
            collective_acc: Accuracy of the collective
            individual_accs: Per-stream accuracies

        Returns:
            Dict with emergence metrics:
                - rho: emergence ratio (collective / max individual)
                - max_individual: best individual accuracy
                - min_individual: worst individual accuracy
                - spread: max - min individual
        """
        if not individual_accs:
            return {'rho': 1.0}

        max_ind = max(individual_accs.values())
        min_ind = min(individual_accs.values())

        # Avoid division by zero
        rho = collective_acc / max_ind if max_ind > 0 else float('inf')

        return {
            'rho': rho,
            'max_individual': max_ind,
            'min_individual': min_ind,
            'spread': max_ind - min_ind,
            'emergence': collective_acc - max_ind,
        }

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
        Tracks emergence ratio ρ throughout training.

        Args:
            train_loader: Training data
            val_loader: Validation data (optional)
            epochs: Override config.epochs
            lr: Override config.lr
            callbacks: Optional training callbacks

        Returns:
            history: Dict with training metrics including emergence
        """
        epochs = epochs or self.config.epochs
        lr = lr or self.config.lr
        device = self.config.device

        self.to(device)

        # Only trainable parameters
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters. Check stream/head freezing.")

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
        best_rho = 0

        for epoch in range(epochs):
            # Training
            self.train()
            epoch_loss = 0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in pbar:
                x, labels = self._unpack_batch(batch)
                x = self._to_device(x, device)
                labels = labels.to(device, non_blocking=True)

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

            train_acc = correct / total
            train_loss = epoch_loss / total

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validation
            if val_loader is not None:
                val_acc, stream_accs, val_loss = self.evaluate(val_loader, return_loss=True)
                emergence = self.compute_emergence(val_acc, stream_accs)

                history['val_acc'].append(val_acc)
                history['val_loss'].append(val_loss)
                history['rho'].append(emergence['rho'])
                history['stream_accs'].append(stream_accs)

                # Track best
                if val_acc > best_acc:
                    best_acc = val_acc
                if emergence['rho'] > best_rho:
                    best_rho = emergence['rho']

                # Log
                stream_str = ' | '.join([
                    f"{k[:6]}: {v * 100:.1f}%"
                    for k, v in stream_accs.items()
                ])
                tqdm.write(
                    f"Epoch {epoch + 1:3d} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Val: {val_acc * 100:.2f}% | "
                    f"ρ: {emergence['rho']:.3f} | "
                    f"{stream_str}"
                )

                if emergence['rho'] > 1.0:
                    tqdm.write(f"  ★ EMERGENCE: ρ = {emergence['rho']:.3f}")

            # Callbacks
            if callbacks:
                for cb in callbacks:
                    cb(epoch, history)

        history['best_acc'] = best_acc
        history['best_rho'] = best_rho

        return dict(history)

    def evaluate(
            self,
            loader: DataLoader,
            return_loss: bool = False,
    ) -> Union[Tuple[float, Dict[str, float]], Tuple[float, Dict[str, float], float]]:
        """
        Evaluate collective and per-stream accuracy.

        Args:
            loader: Evaluation data
            return_loss: Also return average loss

        Returns:
            accuracy: Collective accuracy
            stream_accs: Per-stream accuracy dict
            loss: Average loss (if return_loss=True)
        """
        self.eval()
        device = self.config.device

        correct = 0
        total = 0
        total_loss = 0
        stream_correct = defaultdict(int)

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval", leave=False):
                x, labels = self._unpack_batch(batch)
                x = self._to_device(x, device)
                labels = labels.to(device, non_blocking=True)

                if self.config.use_amp:
                    with autocast():
                        logits, info = self(x, return_individual=True)
                        loss = F.cross_entropy(logits, labels)
                else:
                    logits, info = self(x, return_individual=True)
                    loss = F.cross_entropy(logits, labels)

                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item() * labels.size(0)

                for name, ind_logits in info['individual_logits'].items():
                    stream_correct[name] += (
                            ind_logits.argmax(dim=1) == labels
                    ).sum().item()

        accuracy = correct / total
        stream_accs = {k: v / total for k, v in stream_correct.items()}

        if return_loss:
            return accuracy, stream_accs, total_loss / total
        return accuracy, stream_accs

    def _unpack_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        """Unpack batch into inputs and labels."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        elif isinstance(batch, dict):
            labels = batch.pop('labels', batch.pop('label', None))
            return batch, labels
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")

    def _to_device(self, x, device) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Move inputs to device."""
        if isinstance(x, dict):
            return {k: v.to(device, non_blocking=True) for k, v in x.items()}
        return x.to(device, non_blocking=True)

    @property
    def num_parameters(self) -> int:
        """Total parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Trainable parameters only."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_streams(self):
        """Freeze all stream parameters."""
        for stream in self.streams.values():
            for param in stream.parameters():
                param.requires_grad = False

    def unfreeze_streams(self):
        """Unfreeze all stream parameters."""
        for stream in self.streams.values():
            for param in stream.parameters():
                param.requires_grad = True

    def freeze_heads(self):
        """Freeze all head parameters (including fingerprints)."""
        for head in self.heads.values():
            for param in head.parameters():
                param.requires_grad = False

    def unfreeze_heads(self):
        """Unfreeze all head parameters."""
        for head in self.heads.values():
            for param in head.parameters():
                param.requires_grad = True

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"RouterCollective",
            f"  Streams: {len(self.stream_names)}",
            f"  Feature dim: {self.config.feature_dim}",
            f"  Classes: {self.config.num_classes}",
            f"  Total params: {self.num_parameters:,}",
            f"  Trainable: {self.num_trainable_parameters:,}",
            "",
        ]

        for name in self.stream_names:
            stream = self.streams[name]
            head = self.heads[name]
            shape = self._stream_input_shapes.get(name, 'vector')
            dim = self._stream_dims.get(name, self.config.feature_dim)

            stream_params = sum(p.numel() for p in stream.parameters())
            head_params = sum(p.numel() for p in head.parameters())

            lines.append(f"  [{name}]")
            lines.append(f"    Input: {shape} @ {dim}D")
            lines.append(f"    Stream: {stream_params:,} params")
            lines.append(f"    Head: {head_params:,} params")

        return "\n".join(lines)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_specs(
            cls,
            stream_specs: List[StreamSpec],
            config: CollectiveConfig,
            head_spec: Optional[HeadSpec] = None,
            fusion_spec: Optional[FusionSpec] = None,
    ) -> "RouterCollective":
        """
        Create collective from stream specifications.

        This is the recommended way to build a collective.

        Args:
            stream_specs: List of StreamSpec defining each stream
            config: Collective configuration
            head_spec: Head specification (default: standard)
            fusion_spec: Fusion specification (default: gated)

        Example:
            collective = RouterCollective.from_specs(
                stream_specs=[
                    StreamSpec.feature_vector("clip_b32", input_dim=512),
                    StreamSpec.feature_vector("clip_l14", input_dim=768),
                    StreamSpec.sequence("t5", input_dim=768),
                ],
                config=CollectiveConfig(feature_dim=512, num_classes=1000),
            )
        """
        head_spec = head_spec or HeadSpec.standard(feature_dim=config.feature_dim)
        fusion_spec = fusion_spec or FusionSpec.gated(output_dim=config.feature_dim)

        # Reset registry
        get_registry().reset()

        # Build streams
        streams = nn.ModuleDict()
        stream_dims = {}
        stream_input_shapes = {}

        for spec in stream_specs:
            stream = cls._build_stream_from_spec(spec)
            streams[spec.name] = stream
            stream_dims[spec.name] = spec.feature_dim
            stream_input_shapes[spec.name] = spec.input_shape

        # Build heads (one per stream, each with unique fingerprint)
        heads = nn.ModuleDict()
        for spec in stream_specs:
            head_config = HeadConfig(
                feature_dim=spec.feature_dim,
                fingerprint_dim=head_spec.fingerprint_dim,
                num_heads=head_spec.num_heads,
                num_anchors=head_spec.num_anchors,
                num_routes=head_spec.num_routes,
                use_cantor=head_spec.use_cantor,
            )
            head = HeadBuilder(head_config).build()
            heads[spec.name] = head

        # Build fusion
        stream_dims_for_fusion = {
            spec.name: config.feature_dim  # After projection
            for spec in stream_specs
        }
        fusion = cls._build_fusion_from_spec(fusion_spec, stream_dims_for_fusion)

        # Classifier
        classifier = nn.Sequential(
            nn.LayerNorm(fusion_spec.output_dim),
            nn.Dropout(0.1),
            nn.Linear(fusion_spec.output_dim, config.num_classes),
        )

        return cls(
            streams=streams,
            heads=heads,
            fusion=fusion,
            classifier=classifier,
            config=config,
            stream_dims=stream_dims,
            stream_input_shapes=stream_input_shapes,
        )

    @classmethod
    def from_feature_dims(
            cls,
            feature_configs: Dict[str, int],
            config: CollectiveConfig,
            head_spec: Optional[HeadSpec] = None,
            fusion_spec: Optional[FusionSpec] = None,
    ) -> "RouterCollective":
        """
        Create collective for pre-extracted features.

        Args:
            feature_configs: Dict mapping name to input dimension
            config: Collective configuration

        Example:
            collective = RouterCollective.from_feature_dims({
                'clip_b32': 512,
                'clip_l14': 768,
                'dino_b16': 768,
            }, config)
        """
        stream_specs = [
            StreamSpec.feature_vector(name, input_dim=dim, feature_dim=config.feature_dim)
            for name, dim in feature_configs.items()
        ]
        return cls.from_specs(stream_specs, config, head_spec, fusion_spec)

    @classmethod
    def from_streams(
            cls,
            streams: List[BaseStream],
            config: CollectiveConfig,
            head_spec: Optional[HeadSpec] = None,
            fusion_spec: Optional[FusionSpec] = None,
    ) -> "RouterCollective":
        """
        Create collective from existing stream objects.

        For backward compatibility with manually constructed streams.
        """
        head_spec = head_spec or HeadSpec.standard(feature_dim=config.feature_dim)
        fusion_spec = fusion_spec or FusionSpec.gated(output_dim=config.feature_dim)

        get_registry().reset()

        stream_dict = nn.ModuleDict({s.name: s for s in streams})
        stream_dims = {s.name: getattr(s, 'feature_dim', config.feature_dim) for s in streams}
        stream_input_shapes = {
            s.name: getattr(s, 'input_shape', InputShape.VECTOR).value
            if hasattr(s, 'input_shape') else 'vector'
            for s in streams
        }

        # Build heads
        heads = nn.ModuleDict()
        for s in streams:
            dim = stream_dims[s.name]
            head_config = HeadConfig(
                feature_dim=dim,
                fingerprint_dim=head_spec.fingerprint_dim,
                num_heads=head_spec.num_heads,
                num_anchors=head_spec.num_anchors,
                num_routes=head_spec.num_routes,
                use_cantor=head_spec.use_cantor,
            )
            heads[s.name] = HeadBuilder(head_config).build()

        # Fusion
        stream_dims_for_fusion = {s.name: config.feature_dim for s in streams}
        fusion = cls._build_fusion_from_spec(fusion_spec, stream_dims_for_fusion)

        # Classifier
        classifier = nn.Sequential(
            nn.LayerNorm(fusion_spec.output_dim),
            nn.Dropout(0.1),
            nn.Linear(fusion_spec.output_dim, config.num_classes),
        )

        return cls(
            streams=stream_dict,
            heads=heads,
            fusion=fusion,
            classifier=classifier,
            config=config,
            stream_dims=stream_dims,
            stream_input_shapes=stream_input_shapes,
        )

    @staticmethod
    def _build_stream_from_spec(spec: StreamSpec) -> nn.Module:
        """Build stream module from specification."""
        stream_type = spec.stream_type

        if stream_type in ('feature', 'feature_vector'):
            if spec.input_dim != spec.feature_dim:
                return nn.Sequential(
                    nn.Linear(spec.input_dim, spec.feature_dim),
                    nn.LayerNorm(spec.feature_dim),
                    nn.GELU(),
                )
            return nn.Identity()

        elif stream_type == 'trainable_vector':
            return nn.Sequential(
                nn.Linear(spec.input_dim, spec.feature_dim),
                nn.LayerNorm(spec.feature_dim),
                nn.GELU(),
                nn.Linear(spec.feature_dim, spec.feature_dim),
                nn.LayerNorm(spec.feature_dim),
            )

        elif stream_type in ('sequence', 'frozen'):
            if spec.input_dim != spec.feature_dim:
                return nn.Linear(spec.input_dim, spec.feature_dim)
            return nn.Identity()

        elif stream_type == 'transformer_sequence':
            layers = []
            if spec.input_dim != spec.feature_dim:
                layers.append(nn.Linear(spec.input_dim, spec.feature_dim))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=spec.feature_dim,
                nhead=getattr(spec, 'num_heads', 8),
                dim_feedforward=spec.feature_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            )
            num_layers = getattr(spec, 'num_layers', 2)
            layers.append(nn.TransformerEncoder(encoder_layer, num_layers))

            return nn.Sequential(*layers) if layers else nn.Identity()

        elif stream_type == 'conv_sequence':
            kernel_sizes = getattr(spec, 'kernel_sizes', [3, 5, 7])
            return MultiScaleConv1d(spec.input_dim, spec.feature_dim, kernel_sizes)

        else:
            raise ValueError(f"Unknown stream type: {stream_type}")

    @staticmethod
    def _build_fusion_from_spec(
            spec: FusionSpec,
            stream_dims: Dict[str, int],
    ) -> nn.Module:
        """Build fusion module from specification."""
        strategy_map = {
            'concat': FusionStrategy.CONCAT,
            'weighted': FusionStrategy.WEIGHTED,
            'gated': FusionStrategy.GATED,
            'attention': FusionStrategy.ATTENTION,
            'fingerprint': FusionStrategy.FINGERPRINT,
            'residual': FusionStrategy.RESIDUAL,
            'moe': FusionStrategy.MOE,
            'hierarchical': FusionStrategy.HIERARCHICAL,
        }

        strategy = strategy_map.get(spec.strategy, FusionStrategy.GATED)

        builder = (FusionBuilder()
            .with_streams(stream_dims)
            .with_output_dim(spec.output_dim)
            .with_strategy(strategy))

        if spec.strategy == 'fingerprint':
            builder.with_extra_kwargs(fingerprint_dim=spec.fingerprint_dim)
        elif spec.strategy == 'moe':
            builder.with_extra_kwargs(
                num_experts=getattr(spec, 'num_experts', 4),
                top_k=getattr(spec, 'top_k', 2),
            )
        elif spec.strategy == 'attention':
            builder.with_extra_kwargs(
                num_heads=getattr(spec, 'num_heads', 8),
            )

        return builder.build()


class MultiScaleConv1d(nn.Module):
    """Multi-scale 1D convolution for sequence streams."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            kernel_sizes: List[int] = [3, 5, 7],
    ):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, output_dim // len(kernel_sizes), k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        x_t = x.transpose(1, 2)  # [B, D, S]
        outs = [conv(x_t) for conv in self.convs]
        out = torch.cat(outs, dim=1)  # [B, D_out, S]
        out = out.transpose(1, 2)  # [B, S, D_out]
        return self.norm(out)