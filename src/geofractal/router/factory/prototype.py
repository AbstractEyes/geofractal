"""
geofractal.router.factory.prototype
===================================
Assembled router prototype - the complete system.

Updated to use new stream architecture with proper input shape handling.

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
    InputShape,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PrototypeConfig:
    """Complete configuration for a router prototype."""
    num_classes: int = 1000
    prototype_name: str = "gfr_prototype"

    stream_specs: List[StreamSpec] = field(default_factory=list)
    head_spec: HeadSpec = field(default_factory=HeadSpec.standard)
    per_stream_heads: bool = True
    fusion_spec: FusionSpec = field(default_factory=FusionSpec.standard)

    classifier_hidden: int = 512
    classifier_dropout: float = 0.1
    pool_type: str = 'cls'  # 'cls', 'mean', 'max'
    freeze_streams: bool = True

    def __post_init__(self):
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

    Supports both vector [B, D] and sequence [B, S, D] inputs.

    Architecture:
        Input
          │
          ├─→ Stream₁ ─→ Head₁ ─→ Pool ─┐
          ├─→ Stream₂ ─→ Head₂ ─→ Pool ─┼─→ Fusion ─→ Classifier ─→ Logits
          └─→ Stream₃ ─→ Head₃ ─→ Pool ─┘
    """

    def __init__(self, config: PrototypeConfig):
        super().__init__(
            num_classes=config.num_classes,
            prototype_name=config.prototype_name,
        )
        self.config = config

        self.streams = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.projections = nn.ModuleDict()

        self._stream_dims: Dict[str, int] = {}
        self._stream_input_shapes: Dict[str, str] = {}

        self._build_streams()
        self._build_heads()
        self._build_fusion()
        self._build_classifier()

        if config.freeze_streams:
            self.freeze_streams()

    def _build_streams(self):
        """Build stream modules from specs using new stream architecture."""
        from geofractal.router.streams import (
            FeatureVectorStream,
            TrainableVectorStream,
            SequenceStream,
            TransformerSequenceStream,
            ConvSequenceStream,
            StreamBuilder,
        )
        from geofractal.router.config import CollectiveConfig

        for spec in self.config.stream_specs:
            # Build CollectiveConfig for stream
            collective_config = CollectiveConfig(
                feature_dim=spec.feature_dim,
                fingerprint_dim=self.config.head_spec.fingerprint_dim,
                num_anchors=self.config.head_spec.num_anchors,
                num_routes=self.config.head_spec.num_routes,
            )

            stream_type = spec.stream_type

            # === VECTOR STREAMS ===
            if stream_type in ('feature', 'feature_vector'):
                # Simple feature projection (no backbone)
                self.streams[spec.name] = nn.Sequential(
                    nn.Linear(spec.input_dim, spec.feature_dim),
                    nn.LayerNorm(spec.feature_dim),
                    nn.GELU(),
                )
                self._stream_input_shapes[spec.name] = InputShape.VECTOR

            elif stream_type == 'trainable_vector':
                # Trainable backbone (must be provided in spec)
                if spec.backbone is not None:
                    self.streams[spec.name] = spec.backbone
                else:
                    self.streams[spec.name] = nn.Sequential(
                        nn.Linear(spec.input_dim, spec.feature_dim),
                        nn.LayerNorm(spec.feature_dim),
                        nn.GELU(),
                    )
                self._stream_input_shapes[spec.name] = InputShape.VECTOR

            elif stream_type == 'frozen':
                # Placeholder - actual loading handled separately
                self.streams[spec.name] = nn.Identity()
                self._stream_input_shapes[spec.name] = InputShape.IMAGE

            # === SEQUENCE STREAMS ===
            elif stream_type == 'sequence':
                # Basic sequence projection
                if spec.input_dim != spec.feature_dim:
                    self.streams[spec.name] = nn.Linear(spec.input_dim, spec.feature_dim)
                else:
                    self.streams[spec.name] = nn.Identity()
                self._stream_input_shapes[spec.name] = InputShape.SEQUENCE

            elif stream_type == 'transformer_sequence':
                # Transformer backbone for sequences
                layers = []
                if spec.input_dim != spec.feature_dim:
                    layers.append(nn.Linear(spec.input_dim, spec.feature_dim))

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=spec.feature_dim,
                    nhead=spec.num_heads,
                    dim_feedforward=spec.feature_dim * 4,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True,
                )
                layers.append(nn.TransformerEncoder(encoder_layer, spec.num_layers))

                self.streams[spec.name] = nn.Sequential(*layers) if len(layers) > 1 else layers[0]
                self._stream_input_shapes[spec.name] = InputShape.SEQUENCE

            elif stream_type == 'conv_sequence':
                # Multi-scale conv for sequences
                self.streams[spec.name] = ConvSequenceBackbone(
                    input_dim=spec.input_dim,
                    feature_dim=spec.feature_dim,
                    kernel_sizes=spec.kernel_sizes,
                )
                self._stream_input_shapes[spec.name] = InputShape.SEQUENCE

            else:
                raise ValueError(f"Unknown stream type: {stream_type}")

            self._stream_dims[spec.name] = spec.feature_dim

        self.stream_names = list(self.streams.keys())

    def _build_heads(self):
        """Build head modules from spec."""
        from geofractal.router.head import HeadBuilder, HeadConfig
        from geofractal.router.head import (
            CantorAttention, StandardAttention,
            TopKRouter, SoftRouter,
            ConstitutiveAnchorBank, AttentiveAnchorBank,
            FingerprintGate, ChannelGate,
            LearnableWeightCombiner, GatedCombiner,
            FFNRefinement, MixtureOfExpertsRefinement,
        )

        head_spec = self.config.head_spec

        attention_map = {'cantor': CantorAttention, 'standard': StandardAttention}
        router_map = {'topk': TopKRouter, 'soft': SoftRouter}
        anchor_map = {'constitutive': ConstitutiveAnchorBank, 'attentive': AttentiveAnchorBank}
        gate_map = {'fingerprint': FingerprintGate, 'channel': ChannelGate}
        combiner_map = {'learnable_weight': LearnableWeightCombiner, 'gated': GatedCombiner}
        refinement_map = {'ffn': FFNRefinement, 'moe': MixtureOfExpertsRefinement}

        for name in self.stream_names:
            stream_dim = self._stream_dims[name]

            head_config = HeadConfig(
                feature_dim=stream_dim,
                fingerprint_dim=head_spec.fingerprint_dim,
                num_heads=head_spec.num_heads,
                num_anchors=head_spec.num_anchors,
                num_routes=head_spec.num_routes,
                use_cantor=head_spec.use_cantor,
            )

            builder = HeadBuilder(head_config)
            builder.with_attention(attention_map.get(head_spec.attention_type, CantorAttention))
            builder.with_router(router_map.get(head_spec.router_type, TopKRouter))
            builder.with_anchors(anchor_map.get(head_spec.anchor_type, ConstitutiveAnchorBank))
            builder.with_gate(gate_map.get(getattr(head_spec, 'gate_type', 'fingerprint'), FingerprintGate))
            builder.with_combiner(combiner_map.get(getattr(head_spec, 'combiner_type', 'learnable_weight'), LearnableWeightCombiner))
            builder.with_refinement(refinement_map.get(getattr(head_spec, 'refinement_type', 'ffn'), FFNRefinement))

            self.heads[name] = builder.build()

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
        stream_dims = {name: self.config.fusion_spec.output_dim for name in self.stream_names}

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

        builder = (FusionBuilder()
            .with_streams(stream_dims)
            .with_output_dim(fusion_spec.output_dim)
            .with_strategy(strategy))

        if fusion_spec.strategy == 'fingerprint':
            builder.with_extra_kwargs(fingerprint_dim=fusion_spec.fingerprint_dim)
        elif fusion_spec.strategy == 'moe':
            builder.with_extra_kwargs(num_experts=fusion_spec.num_experts, top_k=fusion_spec.top_k)

        self.fusion = builder.build()

    def _build_classifier(self):
        """Build classifier head."""
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config.classifier_hidden),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.classifier_hidden, self.config.num_classes),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence to vector."""
        if x.dim() == 2:
            return x
        if self.config.pool_type == 'cls':
            return x[:, 0]
        elif self.config.pool_type == 'mean':
            return x.mean(dim=1)
        elif self.config.pool_type == 'max':
            return x.max(dim=1).values
        return x[:, 0]

    def get_stream_outputs(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Get outputs from each stream."""
        outputs = {}

        if isinstance(x, dict):
            # Input is already per-stream
            for name in self.stream_names:
                if name in x:
                    outputs[name] = self.streams[name](x[name])
        else:
            # Shared input - route to all streams
            for name in self.stream_names:
                outputs[name] = self.streams[name](x)

        return outputs

    def forward(
            self,
            x: Union[torch.Tensor, Dict[str, torch.Tensor]],
            return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PrototypeInfo]]:
        """Full forward pass."""
        stream_outputs = self.get_stream_outputs(x)

        head_outputs = {}
        fingerprints = {}
        routing_info = {}

        for name in self.stream_names:
            stream_feat = stream_outputs[name]

            # Ensure 3D: [B, S, D]
            if stream_feat.dim() == 2:
                stream_feat = stream_feat.unsqueeze(1)

            head = self.heads[name]
            if return_info:
                head_out, head_info = head(stream_feat, return_info=True)
                routing_info[name] = head_info
            else:
                head_out = head(stream_feat)

            pooled = self._pool(head_out)
            projected = self.projections[name](pooled)
            head_outputs[name] = projected
            fingerprints[name] = head.fingerprint.detach()

        fused, fusion_info = self.fusion(
            head_outputs,
            stream_fingerprints=fingerprints if return_info else None,
            return_weights=return_info,
        )

        logits = self.classifier(fused)

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


# =============================================================================
# HELPER: Conv Sequence Backbone
# =============================================================================

class ConvSequenceBackbone(nn.Module):
    """Multi-scale conv backbone for sequences."""

    def __init__(self, input_dim: int, feature_dim: int, kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()

        if input_dim != feature_dim:
            self.projection = nn.Linear(input_dim, feature_dim)
        else:
            self.projection = nn.Identity()

        self.convs = nn.ModuleList([
            nn.Conv1d(feature_dim, feature_dim, k, padding=k//2)
            for k in kernel_sizes
        ])
        self.gate = nn.Linear(feature_dim * len(kernel_sizes), feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)  # [B, S, D]
        x_t = x.transpose(1, 2)  # [B, D, S]

        outs = [torch.relu(c(x_t)) for c in self.convs]
        concat = torch.cat(outs, dim=1).transpose(1, 2)  # [B, S, D*K]

        gated = self.gate(concat)  # [B, S, D]
        return self.norm(gated + x)


# =============================================================================
# LIGHTWEIGHT PROTOTYPE (unchanged)
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
    'ConvSequenceBackbone',
]