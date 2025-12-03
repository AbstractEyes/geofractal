"""
geofractal.router.head.builder
==============================
Builder pattern for composing router heads.

Enables easy construction of custom head configurations
with static injection of components.

Usage:
    head = (HeadBuilder(config)
        .with_attention(CantorAttention)
        .with_router(TopKRouter)
        .with_anchors(ConstitutiveAnchorBank)
        .with_gate(FingerprintGate)
        .with_combiner(LearnableWeightCombiner,
                       signal_names=['attention', 'routing', 'anchors'])
        .with_refinement(FFNRefinement)
        .build())

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional, Dict, Any, Callable
from dataclasses import dataclass, field

from .protocols import (
    AttentionHead,
    Router,
    AnchorProvider,
    GatingMechanism,
    Combiner,
    Refinement,
    HeadComponents,
)
from .components import (
    HeadConfig,
    CantorAttention,
    StandardAttention,
    TopKRouter,
    SoftRouter,
    ConstitutiveAnchorBank,
    AttentiveAnchorBank,
    FingerprintGate,
    ChannelGate,
    LearnableWeightCombiner,
    GatedCombiner,
    FFNRefinement,
    MixtureOfExpertsRefinement,
)


# =============================================================================
# PRESETS
# =============================================================================

@dataclass
class HeadPreset:
    """Preset configuration for common head types."""
    attention_cls: Type[nn.Module]
    router_cls: Type[nn.Module]
    anchor_cls: Type[nn.Module]
    gate_cls: Type[nn.Module]
    combiner_cls: Type[nn.Module]
    refinement_cls: Type[nn.Module]
    combiner_kwargs: Dict[str, Any] = field(default_factory=dict)


# Standard GFR head (proven on ImageNet)
STANDARD_HEAD = HeadPreset(
    attention_cls=CantorAttention,
    router_cls=TopKRouter,
    anchor_cls=ConstitutiveAnchorBank,
    gate_cls=FingerprintGate,
    combiner_cls=LearnableWeightCombiner,
    refinement_cls=FFNRefinement,
    combiner_kwargs={'signal_names': ['attention', 'routing', 'anchors']},
)

# Lightweight head (faster, fewer params)
LIGHTWEIGHT_HEAD = HeadPreset(
    attention_cls=StandardAttention,
    router_cls=SoftRouter,
    anchor_cls=ConstitutiveAnchorBank,
    gate_cls=FingerprintGate,
    combiner_cls=LearnableWeightCombiner,
    refinement_cls=FFNRefinement,
    combiner_kwargs={'signal_names': ['attention', 'routing', 'anchors']},
)

# Heavy head (maximum capacity)
HEAVY_HEAD = HeadPreset(
    attention_cls=CantorAttention,
    router_cls=TopKRouter,
    anchor_cls=AttentiveAnchorBank,
    gate_cls=ChannelGate,
    combiner_cls=GatedCombiner,
    refinement_cls=MixtureOfExpertsRefinement,
    combiner_kwargs={},
)


# =============================================================================
# BUILDER
# =============================================================================

class HeadBuilder:
    """
    Builder for composing router heads.

    Supports:
    - Fluent API for component selection
    - Static injection of custom components
    - Preset configurations
    - Validation before build

    Example:
        head = (HeadBuilder(config)
            .with_attention(CantorAttention)
            .with_router(TopKRouter)
            .with_anchors(ConstitutiveAnchorBank)
            .build())
    """

    def __init__(self, config: HeadConfig):
        self.config = config

        # Component classes (will be instantiated on build)
        self._attention_cls: Optional[Type[nn.Module]] = None
        self._router_cls: Optional[Type[nn.Module]] = None
        self._anchor_cls: Optional[Type[nn.Module]] = None
        self._gate_cls: Optional[Type[nn.Module]] = None
        self._combiner_cls: Optional[Type[nn.Module]] = None
        self._refinement_cls: Optional[Type[nn.Module]] = None

        # Extra kwargs for components
        self._attention_kwargs: Dict[str, Any] = {}
        self._router_kwargs: Dict[str, Any] = {}
        self._anchor_kwargs: Dict[str, Any] = {}
        self._gate_kwargs: Dict[str, Any] = {}
        self._combiner_kwargs: Dict[str, Any] = {}
        self._refinement_kwargs: Dict[str, Any] = {}

        # Pre-built instances (for static injection)
        self._attention_instance: Optional[nn.Module] = None
        self._router_instance: Optional[nn.Module] = None
        self._anchor_instance: Optional[nn.Module] = None
        self._gate_instance: Optional[nn.Module] = None
        self._combiner_instance: Optional[nn.Module] = None
        self._refinement_instance: Optional[nn.Module] = None

    # -------------------------------------------------------------------------
    # Fluent setters (by class)
    # -------------------------------------------------------------------------

    def with_attention(
        self,
        cls: Type[nn.Module],
        **kwargs
    ) -> 'HeadBuilder':
        """Set attention component by class."""
        self._attention_cls = cls
        self._attention_kwargs = kwargs
        return self

    def with_router(
        self,
        cls: Type[nn.Module],
        **kwargs
    ) -> 'HeadBuilder':
        """Set router component by class."""
        self._router_cls = cls
        self._router_kwargs = kwargs
        return self

    def with_anchors(
        self,
        cls: Type[nn.Module],
        **kwargs
    ) -> 'HeadBuilder':
        """Set anchor bank by class."""
        self._anchor_cls = cls
        self._anchor_kwargs = kwargs
        return self

    def with_gate(
        self,
        cls: Type[nn.Module],
        **kwargs
    ) -> 'HeadBuilder':
        """Set gating mechanism by class."""
        self._gate_cls = cls
        self._gate_kwargs = kwargs
        return self

    def with_combiner(
        self,
        cls: Type[nn.Module],
        **kwargs
    ) -> 'HeadBuilder':
        """Set signal combiner by class."""
        self._combiner_cls = cls
        self._combiner_kwargs = kwargs
        return self

    def with_refinement(
        self,
        cls: Type[nn.Module],
        **kwargs
    ) -> 'HeadBuilder':
        """Set refinement (FFN) by class."""
        self._refinement_cls = cls
        self._refinement_kwargs = kwargs
        return self

    # -------------------------------------------------------------------------
    # Static injection (by instance)
    # -------------------------------------------------------------------------

    def inject_attention(self, instance: nn.Module) -> 'HeadBuilder':
        """Inject pre-built attention instance."""
        self._attention_instance = instance
        return self

    def inject_router(self, instance: nn.Module) -> 'HeadBuilder':
        """Inject pre-built router instance."""
        self._router_instance = instance
        return self

    def inject_anchors(self, instance: nn.Module) -> 'HeadBuilder':
        """Inject pre-built anchor bank instance."""
        self._anchor_instance = instance
        return self

    def inject_gate(self, instance: nn.Module) -> 'HeadBuilder':
        """Inject pre-built gate instance."""
        self._gate_instance = instance
        return self

    def inject_combiner(self, instance: nn.Module) -> 'HeadBuilder':
        """Inject pre-built combiner instance."""
        self._combiner_instance = instance
        return self

    def inject_refinement(self, instance: nn.Module) -> 'HeadBuilder':
        """Inject pre-built refinement instance."""
        self._refinement_instance = instance
        return self

    # -------------------------------------------------------------------------
    # Preset loading
    # -------------------------------------------------------------------------

    def from_preset(self, preset: HeadPreset) -> 'HeadBuilder':
        """Load all components from a preset."""
        self._attention_cls = preset.attention_cls
        self._router_cls = preset.router_cls
        self._anchor_cls = preset.anchor_cls
        self._gate_cls = preset.gate_cls
        self._combiner_cls = preset.combiner_cls
        self._refinement_cls = preset.refinement_cls
        self._combiner_kwargs = preset.combiner_kwargs.copy()
        return self

    @classmethod
    def standard(cls, config: HeadConfig) -> 'HeadBuilder':
        """Create builder with standard (proven) preset."""
        return cls(config).from_preset(STANDARD_HEAD)

    @classmethod
    def lightweight(cls, config: HeadConfig) -> 'HeadBuilder':
        """Create builder with lightweight preset."""
        return cls(config).from_preset(LIGHTWEIGHT_HEAD)

    @classmethod
    def heavy(cls, config: HeadConfig) -> 'HeadBuilder':
        """Create builder with heavy (max capacity) preset."""
        return cls(config).from_preset(HEAVY_HEAD)

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def _build_component(
        self,
        cls: Optional[Type[nn.Module]],
        instance: Optional[nn.Module],
        kwargs: Dict[str, Any],
        default_cls: Type[nn.Module],
        needs_config: bool = True,
    ) -> nn.Module:
        """Build a single component."""
        # Injected instance takes priority
        if instance is not None:
            return instance

        # Use provided class or default
        component_cls = cls if cls is not None else default_cls

        # Instantiate
        if needs_config:
            return component_cls(self.config, **kwargs)
        else:
            return component_cls(**kwargs)

    def build(self) -> 'ComposedHead':
        """Build the composed head."""
        # Build each component
        attention = self._build_component(
            self._attention_cls, self._attention_instance,
            self._attention_kwargs, CantorAttention
        )

        router = self._build_component(
            self._router_cls, self._router_instance,
            self._router_kwargs, TopKRouter
        )

        anchors = self._build_component(
            self._anchor_cls, self._anchor_instance,
            self._anchor_kwargs, ConstitutiveAnchorBank
        )

        gate = self._build_component(
            self._gate_cls, self._gate_instance,
            self._gate_kwargs, FingerprintGate
        )

        # Combiner - FIX: pass config properly
        combiner_cls = self._combiner_cls or LearnableWeightCombiner
        combiner_kwargs = self._combiner_kwargs.copy()

        if self._combiner_instance is not None:
            combiner = self._combiner_instance
        else:
            combiner = combiner_cls(self.config, **combiner_kwargs)

        refinement = self._build_component(
            self._refinement_cls, self._refinement_instance,
            self._refinement_kwargs, FFNRefinement
        )

        return ComposedHead(
            config=self.config,
            attention=attention,
            router=router,
            anchors=anchors,
            gate=gate,
            combiner=combiner,
            refinement=refinement,
        )

# =============================================================================
# COMPOSED HEAD
# =============================================================================

class ComposedHead(nn.Module):
    """
    A head composed from individual components.

    This is the runtime representation of a built head.
    All components are accessible and can be individually
    modified or analyzed.
    """

    def __init__(
        self,
        config: HeadConfig,
        attention: nn.Module,
        router: nn.Module,
        anchors: nn.Module,
        gate: nn.Module,
        combiner: nn.Module,
        refinement: nn.Module,
    ):
        super().__init__()
        self.config = config

        # Store components
        self.attention = attention
        self.router = router
        self.anchors = anchors
        self.gate = gate
        self.combiner = combiner
        self.refinement = refinement

        # Fingerprint (unique identity)
        self.fingerprint = nn.Parameter(
            torch.randn(config.fingerprint_dim) * 0.02
        )

        # Normalization
        self.input_norm = nn.LayerNorm(config.feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        target_fingerprint: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through composed head.

        Args:
            x: [B, S, D] input
            target_fingerprint: Optional adjacent router's fingerprint
            return_info: Return routing info dict

        Returns:
            output: [B, S, D] routed output
            info: Optional dict with routing details
        """
        B, S, D = x.shape

        # Normalize input
        x_norm = self.input_norm(x)

        # === ATTENTION ===
        attn_out, attn_weights = self.attention(x_norm, return_weights=True)

        # === ROUTING ===
        q = attn_out
        k = x_norm
        v_gated = self.gate.gate_values(x_norm, self.fingerprint)

        routes, route_weights, routed_out = self.router(
            q, k, v_gated, self.fingerprint
        )

        # === ANCHORS ===
        anchor_out, anchor_affinities = self.anchors(x_norm, self.fingerprint)

        # === ADJACENT GATING ===
        if target_fingerprint is not None:
            gate_value = self.gate.compute_similarity(
                self.fingerprint, target_fingerprint
            )
            routed_out = routed_out * gate_value

        # === COMBINE ===
        signals = {
            'attention': attn_out,
            'routing': routed_out,
            'anchors': anchor_out,
        }
        combined = self.combiner(signals)

        # === RESIDUAL + REFINE ===
        output = x + combined
        output = self.refinement(output)

        if return_info:
            info = {
                'routes': routes,
                'route_weights': route_weights,
                'anchor_affinities': anchor_affinities,
                'attn_weights': attn_weights,
            }
            return output, info

        return output

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_component(self, name: str) -> nn.Module:
        """Get a component by name."""
        return getattr(self, name)

    def replace_component(self, name: str, new_component: nn.Module):
        """Replace a component (for experimentation)."""
        setattr(self, name, new_component)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def build_standard_head(config: HeadConfig) -> ComposedHead:
    """Build a standard GFR head."""
    return HeadBuilder.standard(config).build()


def build_lightweight_head(config: HeadConfig) -> ComposedHead:
    """Build a lightweight head."""
    return HeadBuilder.lightweight(config).build()


def build_custom_head(
    config: HeadConfig,
    attention_cls: Type[nn.Module] = CantorAttention,
    router_cls: Type[nn.Module] = TopKRouter,
    anchor_cls: Type[nn.Module] = ConstitutiveAnchorBank,
    gate_cls: Type[nn.Module] = FingerprintGate,
    combiner_cls: Type[nn.Module] = LearnableWeightCombiner,
    refinement_cls: Type[nn.Module] = FFNRefinement,
    **kwargs,
) -> ComposedHead:
    """Build a custom head with specified components."""
    builder = HeadBuilder(config)
    builder.with_attention(attention_cls)
    builder.with_router(router_cls)
    builder.with_anchors(anchor_cls)
    builder.with_gate(gate_cls)
    builder.with_combiner(combiner_cls, **kwargs.get('combiner_kwargs', {}))
    builder.with_refinement(refinement_cls)
    return builder.build()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Builder
    'HeadBuilder',
    'HeadPreset',
    # Presets
    'STANDARD_HEAD',
    'LIGHTWEIGHT_HEAD',
    'HEAVY_HEAD',
    # Composed head
    'ComposedHead',
    # Factory functions
    'build_standard_head',
    'build_lightweight_head',
    'build_custom_head',
]