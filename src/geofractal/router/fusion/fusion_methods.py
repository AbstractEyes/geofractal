"""
geofractal.router.fusion.methods
================================
Concrete fusion implementations.

Each method represents a different philosophy for combining
divergent stream outputs into collective intelligence.

Methods:
- ConcatFusion: Simple concatenation (baseline)
- WeightedFusion: Learnable static weights
- GatedFusion: Input-adaptive gating
- AttentionFusion: Cross-stream attention
- FingerprintGuidedFusion: Fingerprint-conditioned fusion
- ResidualFusion: Additive with learned residuals
- MoEFusion: Mixture-of-experts sparse fusion
- HierarchicalTreeFusion: Tree-structured progressive fusion

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass

from .fusion_protocols import BaseFusion, BaseAdaptiveFusion, FusionInfo


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FusionConfig:
    """Configuration for fusion layers."""
    output_dim: int = 512
    dropout: float = 0.1
    num_heads: int = 8
    expansion: int = 2
    temperature: float = 1.0
    use_layer_norm: bool = True


# =============================================================================
# METHOD 1: CONCATENATION FUSION (Baseline)
# =============================================================================

class ConcatFusion(BaseFusion):
    """
    Simple concatenation followed by projection.

    This is the baseline method - reliable but not adaptive.
    Proven on ImageNet (84.68% collective accuracy).

    Flow:
        [stream_1, stream_2, ..., stream_n] → concat → project → output
    """

    def __init__(
            self,
            stream_dims: Dict[str, int],
            output_dim: int,
            config: Optional[FusionConfig] = None,
    ):
        super().__init__(stream_dims, output_dim)
        config = config or FusionConfig(output_dim=output_dim)

        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(self.total_input_dim, output_dim * config.expansion),
            nn.LayerNorm(output_dim * config.expansion) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(output_dim * config.expansion, output_dim),
            nn.LayerNorm(output_dim) if config.use_layer_norm else nn.Identity(),
        )

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[FusionInfo]]:
        # Concatenate all streams
        concat = self._concat_outputs(stream_outputs)  # [B, sum(D)]

        # Project to output dimension
        fused = self.projection(concat)  # [B, output_dim]

        info = None
        if return_weights:
            # Equal implicit weights
            weights = torch.ones(len(self.stream_names), device=fused.device) / len(self.stream_names)
            info = FusionInfo(weights=weights, method="concat")

        return fused, info


# =============================================================================
# METHOD 2: WEIGHTED FUSION (Learnable Static)
# =============================================================================

class WeightedFusion(BaseFusion):
    """
    Learnable static weights per stream.

    Each stream gets a learned importance weight.
    Weights are normalized via softmax.

    Flow:
        weights = softmax(learnable_w)
        output = Σ weights[i] * project(stream[i])
    """

    def __init__(
            self,
            stream_dims: Dict[str, int],
            output_dim: int,
            config: Optional[FusionConfig] = None,
            init_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(stream_dims, output_dim)
        config = config or FusionConfig(output_dim=output_dim)

        # Per-stream projections (handle different dimensions)
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in stream_dims.items()
        })

        # Learnable weights
        if init_weights:
            w = [init_weights.get(name, 1.0) for name in self.stream_names]
        else:
            w = [1.0] * self.num_streams
        self.weights = nn.Parameter(torch.tensor(w))

        # Output norm
        self.norm = nn.LayerNorm(output_dim) if config.use_layer_norm else nn.Identity()

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[FusionInfo]]:
        # Normalize weights
        normalized = F.softmax(self.weights, dim=0)

        # Project and weight each stream
        fused = None
        for i, name in enumerate(self.stream_names):
            projected = self.projections[name](stream_outputs[name])
            weighted = normalized[i] * projected
            fused = weighted if fused is None else fused + weighted

        fused = self.norm(fused)

        info = None
        if return_weights:
            info = FusionInfo(weights=normalized.detach(), method="weighted")

        return fused, info


# =============================================================================
# METHOD 3: GATED FUSION (Input-Adaptive)
# =============================================================================

class GatedFusion(BaseAdaptiveFusion):
    """
    Input-adaptive gating for fusion weights.

    Each sample gets different fusion weights based on
    the content of the stream outputs.

    Flow:
        context = aggregate(streams)
        gates = softmax(gate_net(context))
        output = Σ gates[i] * project(stream[i])
    """

    def __init__(
            self,
            stream_dims: Dict[str, int],
            output_dim: int,
            config: Optional[FusionConfig] = None,
    ):
        super().__init__(stream_dims, output_dim)
        config = config or FusionConfig(output_dim=output_dim)
        self.config = config

        # Per-stream projections
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in stream_dims.items()
        })

        # Gate network: takes concatenated streams, outputs per-stream weights
        self.gate_net = nn.Sequential(
            nn.Linear(self.total_input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, self.num_streams),
        )

        self.norm = nn.LayerNorm(output_dim) if config.use_layer_norm else nn.Identity()
        self.temperature = config.temperature

    def compute_weights(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use concatenated streams as context if not provided
        if context is None:
            context = self._concat_outputs(stream_outputs)

        # Compute gates
        logits = self.gate_net(context)  # [B, N]
        weights = F.softmax(logits / self.temperature, dim=-1)  # [B, N]

        return weights

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[FusionInfo]]:
        B = next(iter(stream_outputs.values())).shape[0]

        # Compute adaptive weights
        weights = self.compute_weights(stream_outputs)  # [B, N]

        # Project and weight each stream
        fused = torch.zeros(B, self.output_dim, device=weights.device)
        for i, name in enumerate(self.stream_names):
            projected = self.projections[name](stream_outputs[name])  # [B, D]
            fused = fused + weights[:, i:i + 1] * projected

        fused = self.norm(fused)

        info = None
        if return_weights:
            info = FusionInfo(weights=weights.detach(), method="gated")

        return fused, info


# =============================================================================
# METHOD 4: ATTENTION FUSION (Cross-Stream)
# =============================================================================

class AttentionFusion(BaseFusion):
    """
    Multi-head attention across streams.

    Each stream attends to all other streams, enabling
    dynamic information exchange during fusion.

    Flow:
        stacked = stack(streams)  # [B, N, D]
        attended = multi_head_attention(stacked)
        output = pool(attended)
    """

    def __init__(
            self,
            stream_dims: Dict[str, int],
            output_dim: int,
            config: Optional[FusionConfig] = None,
    ):
        super().__init__(stream_dims, output_dim)
        config = config or FusionConfig(output_dim=output_dim)

        # Project all streams to common dimension first
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in stream_dims.items()
        })

        # Multi-head self-attention across streams
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim * self.num_streams, output_dim * config.expansion),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(output_dim * config.expansion, output_dim),
        )

        self.norm = nn.LayerNorm(output_dim) if config.use_layer_norm else nn.Identity()

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[FusionInfo]]:
        # Project all streams to common dimension
        projected = [
            self.projections[name](stream_outputs[name])
            for name in self.stream_names
        ]
        stacked = torch.stack(projected, dim=1)  # [B, N, D]

        # Cross-stream attention
        attended, attn_weights = self.attention(
            stacked, stacked, stacked,
            need_weights=return_weights,
        )  # [B, N, D]

        # Flatten and project
        flat = attended.reshape(attended.shape[0], -1)  # [B, N*D]
        fused = self.output_proj(flat)  # [B, output_dim]
        fused = self.norm(fused)

        info = None
        if return_weights:
            info = FusionInfo(attention=attn_weights, method="attention")

        return fused, info


# =============================================================================
# METHOD 5: FINGERPRINT-GUIDED FUSION
# =============================================================================

class FingerprintGuidedFusion(BaseFusion):
    """
    Fusion guided by stream fingerprints.

    Uses fingerprint similarity to determine how streams
    should be combined - similar fingerprints get combined
    more closely.

    Flow:
        affinity = fingerprint_similarity_matrix
        weights = softmax(affinity @ query)
        output = weighted_combine(streams, weights)
    """

    def __init__(
            self,
            stream_dims: Dict[str, int],
            output_dim: int,
            fingerprint_dim: int = 64,
            config: Optional[FusionConfig] = None,
    ):
        super().__init__(stream_dims, output_dim)
        config = config or FusionConfig(output_dim=output_dim)
        self.fingerprint_dim = fingerprint_dim

        # Per-stream projections
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in stream_dims.items()
        })

        # Fingerprint processing
        self.fp_key = nn.Linear(fingerprint_dim, output_dim)
        self.fp_query = nn.Linear(fingerprint_dim, output_dim)

        # Learnable fusion query
        self.fusion_query = nn.Parameter(torch.randn(output_dim) * 0.02)

        # Output
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim) if config.use_layer_norm else nn.Identity()
        self.temperature = config.temperature

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[FusionInfo]]:
        B = next(iter(stream_outputs.values())).shape[0]
        device = next(iter(stream_outputs.values())).device

        # Project streams
        projected = torch.stack([
            self.projections[name](stream_outputs[name])
            for name in self.stream_names
        ], dim=1)  # [B, N, D]

        if stream_fingerprints is not None:
            # Compute fingerprint-based weights
            fp_stack = torch.stack([
                stream_fingerprints[name] for name in self.stream_names
            ], dim=0)  # [N, F]

            # Fingerprint keys
            fp_keys = self.fp_key(fp_stack)  # [N, D]

            # Query with fusion query
            query = self.fusion_query.unsqueeze(0)  # [1, D]
            scores = torch.matmul(query, fp_keys.T) / math.sqrt(self.output_dim)  # [1, N]
            weights = F.softmax(scores / self.temperature, dim=-1)  # [1, N]
            weights = weights.expand(B, -1)  # [B, N]
        else:
            # Fallback to uniform
            weights = torch.ones(B, self.num_streams, device=device) / self.num_streams

        # Weighted combination
        fused = (projected * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        fused = self.output_proj(fused)
        fused = self.norm(fused)

        info = None
        if return_weights:
            info = FusionInfo(weights=weights.detach(), method="fingerprint_guided")

        return fused, info


# =============================================================================
# METHOD 6: RESIDUAL FUSION
# =============================================================================

class ResidualFusion(BaseFusion):
    """
    Additive fusion with learned residuals.

    Starts with mean of streams, then adds learned
    residual corrections based on stream content.

    Flow:
        base = mean(project(streams))
        residual = residual_net(concat(streams))
        output = base + residual
    """

    def __init__(
            self,
            stream_dims: Dict[str, int],
            output_dim: int,
            config: Optional[FusionConfig] = None,
    ):
        super().__init__(stream_dims, output_dim)
        config = config or FusionConfig(output_dim=output_dim)

        # Per-stream projections
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in stream_dims.items()
        })

        # Residual network
        self.residual_net = nn.Sequential(
            nn.Linear(self.total_input_dim, output_dim * config.expansion),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(output_dim * config.expansion, output_dim),
        )

        # Residual scale (starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self.norm = nn.LayerNorm(output_dim) if config.use_layer_norm else nn.Identity()

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[FusionInfo]]:
        # Project and average
        projected = [
            self.projections[name](stream_outputs[name])
            for name in self.stream_names
        ]
        base = torch.stack(projected, dim=0).mean(dim=0)  # [B, D]

        # Compute residual correction
        concat = self._concat_outputs(stream_outputs)
        residual = self.residual_net(concat)  # [B, D]

        # Combine with scaled residual
        fused = base + self.residual_scale * residual
        fused = self.norm(fused)

        info = None
        if return_weights:
            info = FusionInfo(
                weights=torch.tensor([1.0 / self.num_streams] * self.num_streams),
                method="residual",
            )

        return fused, info


# =============================================================================
# METHOD 7: MIXTURE OF EXPERTS FUSION
# =============================================================================

class MoEFusion(BaseFusion):
    """
    Mixture-of-Experts fusion with sparse activation.

    Multiple expert fusion networks, with a router that
    selects top-K experts per sample.

    Flow:
        expert_outputs = [expert_i(streams) for i in range(E)]
        gates = router(streams)
        top_k = select_top_k(gates)
        output = weighted_sum(expert_outputs[top_k])
    """

    def __init__(
            self,
            stream_dims: Dict[str, int],
            output_dim: int,
            num_experts: int = 4,
            top_k: int = 2,
            config: Optional[FusionConfig] = None,
    ):
        super().__init__(stream_dims, output_dim)
        config = config or FusionConfig(output_dim=output_dim)
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.total_input_dim, output_dim * config.expansion),
                nn.GELU(),
                nn.Linear(output_dim * config.expansion, output_dim),
            )
            for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Sequential(
            nn.Linear(self.total_input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, num_experts),
        )

        self.norm = nn.LayerNorm(output_dim) if config.use_layer_norm else nn.Identity()

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[FusionInfo]]:
        B = next(iter(stream_outputs.values())).shape[0]

        concat = self._concat_outputs(stream_outputs)  # [B, total_dim]

        # Route to experts
        logits = self.router(concat)  # [B, E]

        # Top-K selection
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)  # [B, K]
        topk_weights = F.softmax(topk_logits, dim=-1)  # [B, K]

        # Compute expert outputs and combine
        expert_outputs = torch.stack([e(concat) for e in self.experts], dim=1)  # [B, E, D]

        # Gather selected experts
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, self.output_dim)  # [B, K, D]
        selected = torch.gather(expert_outputs, 1, topk_indices_exp)  # [B, K, D]

        # Weighted sum
        fused = (selected * topk_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        fused = self.norm(fused)

        info = None
        if return_weights:
            # Reconstruct full weight distribution
            full_weights = torch.zeros(B, self.num_experts, device=fused.device)
            full_weights.scatter_(1, topk_indices, topk_weights)
            info = FusionInfo(weights=full_weights.detach(), method="moe")

        return fused, info


# =============================================================================
# METHOD 8: HIERARCHICAL TREE FUSION
# =============================================================================

class HierarchicalTreeFusion(BaseFusion):
    """
    Tree-structured progressive fusion.

    Streams are organized into a hierarchy where related
    streams are fused first at leaf level, then progressively
    combined up the tree.

    Flow:
        level_0: [s1+s2, s3+s4, s5]  # Pairwise fusion
        level_1: [l0_1+l0_2, l0_3]    # Higher fusion
        level_2: final                 # Root
    """

    def __init__(
            self,
            stream_dims: Dict[str, int],
            output_dim: int,
            config: Optional[FusionConfig] = None,
            hierarchy: Optional[List[List[str]]] = None,
    ):
        super().__init__(stream_dims, output_dim)
        config = config or FusionConfig(output_dim=output_dim)

        # Default hierarchy: pairwise bottom-up
        if hierarchy is None:
            hierarchy = self._build_default_hierarchy()
        self.hierarchy = hierarchy

        # Project all streams to common dimension
        self.stream_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in stream_dims.items()
        })

        # Fusion layers for each level
        self.level_fusions = nn.ModuleList()
        for level_idx, level_groups in enumerate(hierarchy):
            level_modules = nn.ModuleDict()
            for group_idx, group in enumerate(level_groups):
                group_size = len(group) if isinstance(group, list) else 1
                if group_size > 1:
                    level_modules[f"group_{group_idx}"] = nn.Sequential(
                        nn.Linear(output_dim * group_size, output_dim * config.expansion),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(output_dim * config.expansion, output_dim),
                        nn.LayerNorm(output_dim),
                    )
            self.level_fusions.append(level_modules)

        # Final projection
        self.final = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim) if config.use_layer_norm else nn.Identity()

    def _build_default_hierarchy(self) -> List[List[List[str]]]:
        """Build default pairwise hierarchy."""
        names = self.stream_names.copy()
        hierarchy = []

        while len(names) > 1:
            level = []
            for i in range(0, len(names), 2):
                if i + 1 < len(names):
                    level.append([names[i], names[i + 1]])
                else:
                    level.append([names[i]])
            hierarchy.append(level)
            names = [f"level_{len(hierarchy) - 1}_group_{i}" for i in range(len(level))]

        return hierarchy

    def forward(
            self,
            stream_outputs: Dict[str, torch.Tensor],
            stream_fingerprints: Optional[Dict[str, torch.Tensor]] = None,
            return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[FusionInfo]]:
        B = next(iter(stream_outputs.values())).shape[0]

        # Project all streams
        current = {
            name: self.stream_projections[name](stream_outputs[name])
            for name in self.stream_names
        }

        intermediate = {}

        # Process hierarchy levels
        for level_idx, (level_groups, level_modules) in enumerate(zip(self.hierarchy, self.level_fusions)):
            next_level = {}

            for group_idx, group in enumerate(level_groups):
                group_name = f"level_{level_idx}_group_{group_idx}"

                if len(group) == 1:
                    # Single item, pass through
                    next_level[group_name] = current[group[0]]
                else:
                    # Fuse group
                    group_concat = torch.cat([current[name] for name in group], dim=-1)
                    fused = level_modules[f"group_{group_idx}"](group_concat)
                    next_level[group_name] = fused

                intermediate[group_name] = next_level[group_name]

            current = next_level

        # Get final output (should be single tensor)
        final_key = list(current.keys())[0]
        fused = self.final(current[final_key])
        fused = self.norm(fused)

        info = None
        if return_weights:
            info = FusionInfo(intermediate=intermediate, method="hierarchical")

        return fused, info


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'FusionConfig',
    'ConcatFusion',
    'WeightedFusion',
    'GatedFusion',
    'AttentionFusion',
    'FingerprintGuidedFusion',
    'ResidualFusion',
    'MoEFusion',
    'HierarchicalTreeFusion',
]