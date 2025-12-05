"""
geofractal.router.head.builder
==============================
Builder pattern for composing router heads.

ComposedHead now includes mailbox support for inter-router coordination,
matching the behavior of the original GlobalFractalRouter.

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field

from .head_protocols import (
    AttentionHead,
    Router,
    AnchorProvider,
    GatingMechanism,
    Combiner,
    Refinement,
)
from .head_components import (
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
)

# Lightweight head (faster, fewer params)
LIGHTWEIGHT_HEAD = HeadPreset(
    attention_cls=StandardAttention,
    router_cls=SoftRouter,
    anchor_cls=ConstitutiveAnchorBank,
    gate_cls=FingerprintGate,
    combiner_cls=LearnableWeightCombiner,
    refinement_cls=FFNRefinement,
)

# Heavy head (maximum capacity)
HEAVY_HEAD = HeadPreset(
    attention_cls=CantorAttention,
    router_cls=TopKRouter,
    anchor_cls=AttentiveAnchorBank,
    gate_cls=ChannelGate,
    combiner_cls=GatedCombiner,
    refinement_cls=MixtureOfExpertsRefinement,
)


# =============================================================================
# COMPOSED HEAD (with mailbox support)
# =============================================================================

class ComposedHead(nn.Module):
    """
    A head composed from individual components with mailbox coordination.

    This matches the behavior of GlobalFractalRouter:
    - Reads from mailbox for router context
    - Posts routing state to mailbox
    - Uses fingerprint for identity and coordination
    - Supports adjacent gating via target_fingerprint

    Forward signature matches original:
        forward(x, mailbox, target_fingerprint, skip_first, return_info)
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
        comm_dim: int = 128,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.name = name or f"head_{id(self)}"

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

        # === MAILBOX COMMUNICATION ===
        self.comm_dim = comm_dim
        self.comm_encoder = nn.Linear(config.feature_dim, comm_dim)
        self.comm_decoder = nn.Linear(comm_dim, config.feature_dim)

        # Module ID for mailbox (set by registry or manually)
        self._module_id: Optional[str] = None

    @property
    def module_id(self) -> str:
        """Get module ID for mailbox communication."""
        if self._module_id is None:
            self._module_id = f"head_{id(self)}"
        return self._module_id

    @module_id.setter
    def module_id(self, value: str):
        self._module_id = value

    def forward(
        self,
        x: torch.Tensor,
        mailbox: Optional[Any] = None,
        target_fingerprint: Optional[torch.Tensor] = None,
        skip_first: bool = False,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through composed head with mailbox coordination.

        Args:
            x: [B, S, D] input
            mailbox: Optional RouterMailbox for inter-head communication
            target_fingerprint: Optional adjacent router's fingerprint for gating
            skip_first: Skip first token (CLS) in routing
            return_info: Return routing info dict

        Returns:
            output: [B, S, D] routed output
            info: Optional dict with routing details (if return_info=True)
        """
        B, S, D = x.shape

        # Handle skip_first (CLS token)
        if skip_first and S > 1:
            x_route = x[:, 1:, :]
            x_cls = x[:, :1, :]
        else:
            x_route = x
            x_cls = None

        B, P, D = x_route.shape  # P = S or S-1

        # Normalize input
        x_norm = self.input_norm(x_route)

        # === READ FROM MAILBOX ===
        router_context = None
        if mailbox is not None and hasattr(mailbox, 'read'):
            messages = mailbox.read(self.module_id, self.fingerprint)
            if messages:
                # Stack routing states from other routers
                states = torch.stack([
                    m.routing_state.to(x.device) if hasattr(m, 'routing_state')
                    else m.content.to(x.device)
                    for m in messages
                ])
                router_context = self.comm_decoder(states.mean(dim=0))

        # === ATTENTION ===
        attn_out, attn_weights = self.attention(x_norm, return_weights=True)

        # === ROUTING ===
        q = attn_out
        k = x_norm
        v_gated = self.gate.gate_values(x_norm, self.fingerprint)

        # Integrate router context into routing if available
        if router_context is not None:
            # Add context bias to queries
            context_bias = torch.einsum('d,bpd->bp', router_context, q)
            # This will be used by router - for now just modulate v
            v_gated = v_gated + 0.1 * router_context.unsqueeze(0).unsqueeze(0)

        routes, route_weights, routed_out = self.router(
            q, k, v_gated, self.fingerprint
        )

        # === ANCHORS ===
        anchor_out, anchor_affinities = self.anchors(x_norm, self.fingerprint)

        # === ADJACENT GATING ===
        gate_value = None
        if target_fingerprint is not None:
            gate_value = self.gate.compute_similarity(
                self.fingerprint, target_fingerprint
            )
            routed_out = routed_out * gate_value

        # === POST TO MAILBOX ===
        if mailbox is not None and hasattr(mailbox, 'post'):
            # Encode routing state for other routers
            routing_state = self.comm_encoder(routed_out.mean(dim=(0, 1)))
            mailbox.post(
                sender_id=self.module_id,
                sender_name=self.name,
                content=routing_state.detach(),
            )

        # === COMBINE ===
        signals = {
            'attention': attn_out,
            'routing': routed_out,
            'anchors': anchor_out,
        }
        combined = self.combiner(signals)

        # === RESIDUAL + REFINE ===
        output = x_route + combined
        output = self.refinement(output)

        # Reconstruct with CLS if skipped
        if x_cls is not None:
            output = torch.cat([x_cls, output], dim=1)

        if return_info:
            # Compute route entropy for monitoring
            route_entropy = -(route_weights * torch.log(route_weights + 1e-8)).sum(dim=-1).mean()

            info = {
                'routes': routes,
                'route_weights': route_weights,
                'route_entropy': route_entropy.item(),
                'anchor_affinities': anchor_affinities,
                'attn_weights': attn_weights,
                'gate_value': gate_value,
                'router_context_used': router_context is not None,
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

        # Component classes
        self._attention_cls: Optional[Type[nn.Module]] = None
        self._router_cls: Optional[Type[nn.Module]] = None
        self._anchor_cls: Optional[Type[nn.Module]] = None
        self._gate_cls: Optional[Type[nn.Module]] = None
        self._combiner_cls: Optional[Type[nn.Module]] = None
        self._refinement_cls: Optional[Type[nn.Module]] = None

        # Extra kwargs
        self._attention_kwargs: Dict[str, Any] = {}
        self._router_kwargs: Dict[str, Any] = {}
        self._anchor_kwargs: Dict[str, Any] = {}
        self._gate_kwargs: Dict[str, Any] = {}
        self._combiner_kwargs: Dict[str, Any] = {}
        self._refinement_kwargs: Dict[str, Any] = {}

        # Pre-built instances
        self._attention_instance: Optional[nn.Module] = None
        self._router_instance: Optional[nn.Module] = None
        self._anchor_instance: Optional[nn.Module] = None
        self._gate_instance: Optional[nn.Module] = None
        self._combiner_instance: Optional[nn.Module] = None
        self._refinement_instance: Optional[nn.Module] = None

        # Head-level config
        self._comm_dim: int = 128
        self._name: Optional[str] = None

    # -------------------------------------------------------------------------
    # Fluent setters
    # -------------------------------------------------------------------------

    def with_attention(self, cls: Type[nn.Module], **kwargs) -> 'HeadBuilder':
        self._attention_cls = cls
        self._attention_kwargs = kwargs
        return self

    def with_router(self, cls: Type[nn.Module], **kwargs) -> 'HeadBuilder':
        self._router_cls = cls
        self._router_kwargs = kwargs
        return self

    def with_anchors(self, cls: Type[nn.Module], **kwargs) -> 'HeadBuilder':
        self._anchor_cls = cls
        self._anchor_kwargs = kwargs
        return self

    def with_gate(self, cls: Type[nn.Module], **kwargs) -> 'HeadBuilder':
        self._gate_cls = cls
        self._gate_kwargs = kwargs
        return self

    def with_combiner(self, cls: Type[nn.Module], **kwargs) -> 'HeadBuilder':
        self._combiner_cls = cls
        self._combiner_kwargs = kwargs
        return self

    def with_refinement(self, cls: Type[nn.Module], **kwargs) -> 'HeadBuilder':
        self._refinement_cls = cls
        self._refinement_kwargs = kwargs
        return self

    def with_comm_dim(self, dim: int) -> 'HeadBuilder':
        """Set communication dimension for mailbox."""
        self._comm_dim = dim
        return self

    def with_name(self, name: str) -> 'HeadBuilder':
        """Set head name for identification."""
        self._name = name
        return self

    # -------------------------------------------------------------------------
    # Static injection
    # -------------------------------------------------------------------------

    def inject_attention(self, instance: nn.Module) -> 'HeadBuilder':
        self._attention_instance = instance
        return self

    def inject_router(self, instance: nn.Module) -> 'HeadBuilder':
        self._router_instance = instance
        return self

    def inject_anchors(self, instance: nn.Module) -> 'HeadBuilder':
        self._anchor_instance = instance
        return self

    def inject_gate(self, instance: nn.Module) -> 'HeadBuilder':
        self._gate_instance = instance
        return self

    def inject_combiner(self, instance: nn.Module) -> 'HeadBuilder':
        self._combiner_instance = instance
        return self

    def inject_refinement(self, instance: nn.Module) -> 'HeadBuilder':
        self._refinement_instance = instance
        return self

    # -------------------------------------------------------------------------
    # Preset loading
    # -------------------------------------------------------------------------

    def from_preset(self, preset: HeadPreset) -> 'HeadBuilder':
        """Load from a preset configuration."""
        self._attention_cls = preset.attention_cls
        self._router_cls = preset.router_cls
        self._anchor_cls = preset.anchor_cls
        self._gate_cls = preset.gate_cls
        self._combiner_cls = preset.combiner_cls
        self._refinement_cls = preset.refinement_cls
        self._combiner_kwargs = preset.combiner_kwargs
        return self

    @classmethod
    def standard(cls, config: HeadConfig) -> 'HeadBuilder':
        """Create builder with standard preset."""
        return cls(config).from_preset(STANDARD_HEAD)

    @classmethod
    def lightweight(cls, config: HeadConfig) -> 'HeadBuilder':
        """Create builder with lightweight preset."""
        return cls(config).from_preset(LIGHTWEIGHT_HEAD)

    @classmethod
    def heavy(cls, config: HeadConfig) -> 'HeadBuilder':
        """Create builder with heavy preset."""
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
    ) -> nn.Module:
        """Build a single component."""
        if instance is not None:
            return instance

        component_cls = cls if cls is not None else default_cls
        return component_cls(self.config, **kwargs)

    def build(self) -> ComposedHead:
        """Build the composed head."""
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

        combiner_cls = self._combiner_cls or LearnableWeightCombiner
        if self._combiner_instance is not None:
            combiner = self._combiner_instance
        else:
            combiner = combiner_cls(self.config, **self._combiner_kwargs)

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
            comm_dim=self._comm_dim,
            name=self._name,
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def build_standard_head(config: HeadConfig, name: Optional[str] = None) -> ComposedHead:
    """Build a standard GFR head."""
    return HeadBuilder.standard(config).with_name(name).build()


def build_lightweight_head(config: HeadConfig, name: Optional[str] = None) -> ComposedHead:
    """Build a lightweight head."""
    return HeadBuilder.lightweight(config).with_name(name).build()


def build_custom_head(
    config: HeadConfig,
    attention_cls: Type[nn.Module] = CantorAttention,
    router_cls: Type[nn.Module] = TopKRouter,
    anchor_cls: Type[nn.Module] = ConstitutiveAnchorBank,
    gate_cls: Type[nn.Module] = FingerprintGate,
    combiner_cls: Type[nn.Module] = LearnableWeightCombiner,
    refinement_cls: Type[nn.Module] = FFNRefinement,
    name: Optional[str] = None,
    **kwargs,
) -> ComposedHead:
    """Build a custom head with specified components."""
    return (HeadBuilder(config)
        .with_attention(attention_cls)
        .with_router(router_cls)
        .with_anchors(anchor_cls)
        .with_gate(gate_cls)
        .with_combiner(combiner_cls, **kwargs.get('combiner_kwargs', {}))
        .with_refinement(refinement_cls)
        .with_name(name)
        .build())


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