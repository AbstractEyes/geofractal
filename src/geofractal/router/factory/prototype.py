"""
geofractal.router.factory.prototype
===================================
Assembled router prototype - the complete system.

A prototype combines:
- Multiple streams (divergent feature extractors)
- Per-stream heads (routing decision makers)
- Fusion layer (combination strategy)
- Classifier head (final predictions)

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

from .protocols import (
    BasePrototype,
    PrototypeInfo,
    StreamSpec,
    HeadSpec,
    FusionSpec,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PrototypeConfig:
    """
    Complete configuration for a router prototype.
    """
    # Core settings
    num_classes: int = 1000
    prototype_name: str = "gfr_prototype"

    # Stream specifications
    stream_specs: List[StreamSpec] = field(default_factory=list)

    # Head specification (shared across streams by default)
    head_spec: HeadSpec = field(default_factory=HeadSpec.standard)
    per_stream_heads: bool = True  # Each stream gets its own head

    # Fusion specification
    fusion_spec: FusionSpec = field(default_factory=FusionSpec.standard)

    # Classifier settings
    classifier_hidden: int = 512
    classifier_dropout: float = 0.1

    # Pooling
    pool_type: str = 'cls'  # 'cls', 'mean', 'max'

    # Training
    freeze_streams: bool = True

    def __post_init__(self):
        # Ensure fusion output matches classifier input
        if self.fusion_spec.output_dim != self.classifier_hidden:
            self.classifier_hidden = self.fusion_spec.output_dim

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_classes': self.num_classes,
            'prototype_name': self.prototype_name,
            'stream_specs': [s.to_dict() for s in self.stream_specs],
            'head_spec': self.head_spec.to_dict(),
            'per_stream_heads': self.per_stream_heads,
            'fusion_spec': self.fusion_spec.to_dict(),
            'classifier_hidden': self.classifier_hidden,
            'classifier_dropout': self.classifier_dropout,
            'pool_type': self.pool_type,
            'freeze_streams': self.freeze_streams,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PrototypeConfig':
        d = d.copy()
        d['stream_specs'] = [StreamSpec.from_dict(s) for s in d.get('stream_specs', [])]
        d['head_spec'] = HeadSpec.from_dict(d.get('head_spec', {}))
        d['fusion_spec'] = FusionSpec.from_dict(d.get('fusion_spec', {}))
        return cls(**d)


# =============================================================================
# ASSEMBLED PROTOTYPE
# =============================================================================

class AssembledPrototype(BasePrototype):
    """
    Complete router prototype assembled from components.

    This is the main class for building and running router prototypes.
    It composes streams, heads, fusion, and classifier into a
    trainable end-to-end system.

    Architecture:
        Input
          │
          ├─→ Stream₁ ─→ Head₁ ─→ Pool ─┐
          ├─→ Stream₂ ─→ Head₂ ─→ Pool ─┼─→ Fusion ─→ Classifier ─→ Logits
          └─→ Stream₃ ─→ Head₃ ─→ Pool ─┘

    Usage:
        config = PrototypeConfig(
            num_classes=1000,
            stream_specs=[
                StreamSpec.frozen_clip("clip_b32", "openai/clip-vit-base-patch32"),
                StreamSpec.frozen_clip("clip_l14", "openai/clip-vit-large-patch14"),
            ],
            head_spec=HeadSpec.standard(),
            fusion_spec=FusionSpec.attention(),
        )

        prototype = AssembledPrototype(config)
        logits = prototype(images)
    """

    def __init__(self, config: PrototypeConfig):
        super().__init__(
            num_classes=config.num_classes,
            prototype_name=config.prototype_name,
        )
        self.config = config

        # Build components
        self.streams = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.projections = nn.ModuleDict()  # Project to common dim if needed

        self._build_streams()
        self._build_heads()
        self._build_fusion()
        self._build_classifier()

        # Freeze streams if configured
        if config.freeze_streams:
            self.freeze_streams()

    def _build_streams(self):
        """Build stream modules from specs."""
        from geofractal.router.streams import FrozenStream, FeatureStream, TrainableStream

        for spec in self.config.stream_specs:
            if spec.stream_type == 'frozen':
                # Placeholder - actual loading happens in from_pretrained
                self.streams[spec.name] = nn.Identity()
                self._stream_dims = getattr(self, '_stream_dims', {})
                self._stream_dims[spec.name] = spec.feature_dim

            elif spec.stream_type == 'feature':
                input_dim = spec.kwargs.get('input_dim', spec.feature_dim)
                self.streams[spec.name] = nn.Sequential(
                    nn.Linear(input_dim, spec.feature_dim),
                    nn.LayerNorm(spec.feature_dim),
                    nn.GELU(),
                )
                self._stream_dims = getattr(self, '_stream_dims', {})
                self._stream_dims[spec.name] = spec.feature_dim

            elif spec.stream_type == 'trainable':
                # Placeholder for trainable backbone
                self.streams[spec.name] = nn.Identity()
                self._stream_dims = getattr(self, '_stream_dims', {})
                self._stream_dims[spec.name] = spec.feature_dim

        self.stream_names = list(self.streams.keys())

    def _build_heads(self):
        """Build head modules from spec."""
        from geofractal.router.head import HeadBuilder, HeadConfig
        from geofractal.router.head import (
            CantorAttention, StandardAttention,
            TopKRouter, SoftRouter,
            ConstitutiveAnchorBank, AttentiveAnchorBank,
        )

        head_spec = self.config.head_spec

        # Map spec strings to classes
        attention_map = {
            'cantor': CantorAttention,
            'standard': StandardAttention,
        }
        router_map = {
            'topk': TopKRouter,
            'soft': SoftRouter,
        }
        anchor_map = {
            'constitutive': ConstitutiveAnchorBank,
            'attentive': AttentiveAnchorBank,
        }

        for name in self.stream_names:
            stream_dim = self._stream_dims[name]

            # Create head config for this stream
            head_config = HeadConfig(
                feature_dim=stream_dim,
                fingerprint_dim=head_spec.fingerprint_dim,
                num_heads=head_spec.num_heads,
                num_anchors=head_spec.num_anchors,
                num_routes=head_spec.num_routes,
                use_cantor=head_spec.use_cantor,
            )

            # Build head
            head = (HeadBuilder(head_config)
                    .with_attention(attention_map.get(head_spec.attention_type, CantorAttention))
                    .with_router(router_map.get(head_spec.router_type, TopKRouter))
                    .with_anchors(anchor_map.get(head_spec.anchor_type, ConstitutiveAnchorBank))
                    .build())

            self.heads[name] = head

            # Projection to common dimension if needed
            fusion_dim = self.config.fusion_spec.output_dim
            if stream_dim != fusion_dim:
                self.projections[name] = nn.Linear(stream_dim, fusion_dim)
            else:
                self.projections[name] = nn.Identity()

    def _build_fusion(self):
        """Build fusion layer from spec."""
        from geofractal.router.fusion import (
            FusionBuilder, FusionStrategy,
            ConcatFusion, WeightedFusion, GatedFusion,
            AttentionFusion, FingerprintGuidedFusion,
            ResidualFusion, MoEFusion, HierarchicalTreeFusion,
        )

        fusion_spec = self.config.fusion_spec
        fusion_dim = fusion_spec.output_dim

        # Stream dims after projection
        stream_dims = {name: fusion_dim for name in self.stream_names}

        # Map strategy string to enum
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

        strategy = strategy_map.get(fusion_spec.strategy, FusionStrategy.CONCAT)

        self.fusion = (FusionBuilder()
                       .with_streams(stream_dims)
                       .with_output_dim(fusion_dim)
                       .with_strategy(strategy)
                       .with_extra_kwargs(
            num_heads=fusion_spec.num_heads,
            temperature=fusion_spec.temperature,
        )
                       .build())

    def _build_classifier(self):
        """Build classifier head."""
        hidden = self.config.classifier_hidden

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(hidden, self.num_classes),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence to single vector."""
        if self.config.pool_type == 'cls':
            return x[:, 0]  # CLS token
        elif self.config.pool_type == 'mean':
            return x.mean(dim=1)
        elif self.config.pool_type == 'max':
            return x.max(dim=1).values
        else:
            return x[:, 0]

    def get_stream_outputs(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get raw outputs from each stream."""
        outputs = {}
        for name in self.stream_names:
            stream_out = self.streams[name](x)
            outputs[name] = stream_out
        return outputs

    def forward(
            self,
            x: torch.Tensor,
            return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PrototypeInfo]]:
        """
        Full forward pass.

        Args:
            x: [B, ...] input (images or features)
            return_info: Return routing info

        Returns:
            logits: [B, num_classes] predictions
            info: Optional PrototypeInfo with routing details
        """
        # Get stream outputs
        stream_outputs = self.get_stream_outputs(x)

        # Process through heads
        head_outputs = {}
        fingerprints = {}
        routing_info = {}

        for name in self.stream_names:
            stream_feat = stream_outputs[name]

            # Ensure 3D: [B, S, D]
            if stream_feat.dim() == 2:
                stream_feat = stream_feat.unsqueeze(1)

            # Through head
            head = self.heads[name]
            if return_info:
                head_out, head_info = head(stream_feat, return_info=True)
                routing_info[name] = head_info
            else:
                head_out = head(stream_feat)

            # Pool to [B, D]
            pooled = self._pool(head_out)

            # Project to fusion dimension
            projected = self.projections[name](pooled)
            head_outputs[name] = projected

            # Collect fingerprint
            fingerprints[name] = head.fingerprint.detach()

        # Fusion
        fused, fusion_info = self.fusion(
            head_outputs,
            stream_fingerprints=fingerprints if return_info else None,
            return_weights=return_info,
        )

        # Classify
        logits = self.classifier(fused)

        # Build info
        info = None
        if return_info:
            info = PrototypeInfo(
                stream_outputs=stream_outputs,
                head_outputs=head_outputs,
                fusion_weights=fusion_info.weights if fusion_info else None,
                routing_info=routing_info,
                fingerprints=fingerprints,
            )

        return logits, info

    def get_emergence_ratio(
            self,
            dataloader,
            device: torch.device = None,
    ) -> float:
        """
        Compute emergence ratio on dataset.

        ρ = collective_accuracy / max(individual_accuracies)
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # Collect predictions
        all_labels = []
        collective_preds = []
        individual_preds = {name: [] for name in self.stream_names}

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, labels = batch[0], batch[1]
                else:
                    x, labels = batch, None

                x = x.to(device)
                if labels is not None:
                    all_labels.extend(labels.cpu().tolist())

                # Collective prediction
                logits, info = self(x, return_info=True)
                collective_preds.extend(logits.argmax(dim=-1).cpu().tolist())

                # Individual predictions (from head outputs through classifier)
                for name in self.stream_names:
                    # Simple classifier on individual stream
                    head_out = info.head_outputs[name]
                    # Use same classifier (rough approximation)
                    ind_logits = self.classifier(head_out)
                    individual_preds[name].extend(ind_logits.argmax(dim=-1).cpu().tolist())

        if not all_labels:
            return 0.0

        # Compute accuracies
        all_labels = torch.tensor(all_labels)
        collective_preds = torch.tensor(collective_preds)
        collective_acc = (collective_preds == all_labels).float().mean().item()

        individual_accs = {}
        for name in self.stream_names:
            preds = torch.tensor(individual_preds[name])
            individual_accs[name] = (preds == all_labels).float().mean().item()

        max_individual = max(individual_accs.values()) if individual_accs else 0.0

        # Emergence ratio
        if max_individual > 0:
            emergence_ratio = collective_acc / max_individual
        else:
            emergence_ratio = float('inf') if collective_acc > 0 else 0.0

        return emergence_ratio


# =============================================================================
# LIGHTWEIGHT PROTOTYPE (Minimal overhead)
# =============================================================================

class LightweightPrototype(BasePrototype):
    """
    Lightweight prototype for fast experimentation.

    Simpler architecture with minimal routing overhead.
    Good for quick iteration and baseline comparisons.
    """

    def __init__(
            self,
            stream_dims: Dict[str, int],
            num_classes: int,
            hidden_dim: int = 512,
            dropout: float = 0.1,
    ):
        super().__init__(num_classes, "lightweight")

        self.stream_dims = stream_dims
        self.stream_names = list(stream_dims.keys())
        self.hidden_dim = hidden_dim

        # Simple projections per stream
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for name, dim in stream_dims.items()
        })

        # Learnable stream weights
        self.stream_weights = nn.Parameter(torch.ones(len(stream_dims)))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def get_stream_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x should be dict of stream features
        if isinstance(x, dict):
            return x
        # Otherwise return empty (streams handled externally)
        return {}

    def forward(
            self,
            stream_features: Dict[str, torch.Tensor],
            return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PrototypeInfo]]:
        """
        Args:
            stream_features: {name: [B, D]} features from each stream
        """
        # Project each stream
        projected = {}
        for name in self.stream_names:
            feat = stream_features[name]
            if feat.dim() == 3:
                feat = feat[:, 0]  # Take CLS
            projected[name] = self.projections[name](feat)

        # Weighted combination
        weights = F.softmax(self.stream_weights, dim=0)

        fused = None
        for i, name in enumerate(self.stream_names):
            weighted = weights[i] * projected[name]
            fused = weighted if fused is None else fused + weighted

        # Classify
        logits = self.classifier(fused)

        info = None
        if return_info:
            info = PrototypeInfo(
                head_outputs=projected,
                fusion_weights=weights.detach(),
            )

        return logits, info


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PrototypeConfig',
    'AssembledPrototype',
    'LightweightPrototype',
]