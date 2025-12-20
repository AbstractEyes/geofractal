"""
geofractal.router.prefab.agatha.head_router
====================================

Agatha Head Router - Block 1 vector extraction with fingerprinting.

Extracts from pre-trained encoders (QWEN, FluxAE, Lyra, Dino) and produces
fingerprinted mail to teach downstream UNET experts. Uses existing
DataComponent, AddressComponent, and FusionComponent infrastructure.

Architecture:
    Block 1 (Encoders):
        - QWEN 2.5 Instruct (text)
        - Flux AE (image latent)
        - Lyra Bottleneck (KL-divergence)
        - Dino 3 (guidance/fusion)

    Each encoder stream:
        Encoder -> DataComponent (move) -> AddressComponent (fingerprint)

    Fusion:
        All streams -> FusionComponent (gated/adaptive) -> Teaching mail

Design Philosophy:
    - Reuse existing components (no reinvention)
    - Each stream gets geometric identity via AddressComponent
    - Mail carries content + fingerprint for downstream teaching
    - Attach/detach for debugging incorrect learned behaviors

Copyright 2025 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto

import torch
import torch.nn as nn
from torch import Tensor

from geofractal.router.base_router import BaseRouter
from geofractal.router.components.torch_component import TorchComponent
from geofractal.router.components.data_component import DataComponent
from geofractal.router.components.address_component import AddressComponent
from geofractal.router.components.fusion_component import (
    FusionComponent,
    GatedFusion,
    AdaptiveFusion,
    ConcatFusion,
    AttentionFusion,
)


# =============================================================================
# STREAM TYPES
# =============================================================================

class StreamType(Enum):
    """Encoder stream types for Agatha Block 1."""
    TEXT = auto()      # QWEN - text encoding
    IMAGE = auto()     # Flux AE - image latents
    LATENT = auto()    # Lyra - bottleneck representation
    GUIDANCE = auto()  # Dino - structural guidance
    FUSED = auto()     # Post-fusion combined


# =============================================================================
# MAIL STRUCTURES
# =============================================================================

@dataclass
class EncoderMail:
    """
    Mail from a single encoder stream.

    Carries extracted content + geometric fingerprint for routing.
    """
    content: Tensor                          # Extracted vectors [B, L, D] or [B, D]
    fingerprint: Tensor                      # Geometric identity [B, fp_dim]
    stream_type: StreamType                  # Which encoder type
    source: str                              # Encoder name
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> torch.Size:
        return self.content.shape

    @property
    def device(self) -> torch.device:
        return self.content.device

    def __repr__(self) -> str:
        return (f"EncoderMail('{self.source}', {self.stream_type.name}, "
                f"content={list(self.content.shape)}, fp={list(self.fingerprint.shape)})")


@dataclass
class HeadMail:
    """
    Aggregated mail from Agatha head router.

    Contains all encoder outputs + fusion for downstream teaching.
    """
    streams: Dict[str, EncoderMail]                    # Individual encoder mails
    fused: Optional[Tensor] = None                     # Fused representation
    fused_fingerprint: Optional[Tensor] = None         # Fusion fingerprint
    timestep: Optional[Tensor] = None                  # Diffusion timestep
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> EncoderMail:
        return self.streams[key]

    def __contains__(self, key: str) -> bool:
        return key in self.streams

    @property
    def sources(self) -> List[str]:
        return list(self.streams.keys())

    @property
    def batch_size(self) -> int:
        if self.streams:
            return next(iter(self.streams.values())).content.shape[0]
        return 0

    def get_all_fingerprints(self) -> Dict[str, Tensor]:
        """Get all fingerprints including fusion."""
        fps = {name: mail.fingerprint for name, mail in self.streams.items()}
        if self.fused_fingerprint is not None:
            fps['fused'] = self.fused_fingerprint
        return fps


# =============================================================================
# ENCODER STREAM
# =============================================================================

class EncoderStream(TorchComponent):
    """
    Single encoder stream with fingerprinting.

    Wraps a pre-trained encoder and adds:
        - DataComponent for device movement
        - AddressComponent for geometric fingerprint
        - Optional projection to common dimension
        - Learnable stream identity embedding
    """

    def __init__(
        self,
        name: str,
        stream_type: StreamType,
        embed_dim: int,
        fingerprint_dim: int = 64,
        project_dim: Optional[int] = None,
        frozen: bool = True,
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, uuid, **kwargs)

        self.stream_type = stream_type
        self.embed_dim = embed_dim
        self.fingerprint_dim = fingerprint_dim
        self.output_dim = project_dim or embed_dim
        self.frozen = frozen

        # Encoder (attached later)
        self.encoder: Optional[nn.Module] = None
        self.extract_fn: Optional[Callable] = None

        # Data mover
        self.data = DataComponent(f'{name}_data')

        # Learned fingerprint identity
        self.address = AddressComponent(
            f'{name}_address',
            fingerprint_dim=fingerprint_dim,
        )

        # Projection to common dim (if needed)
        if project_dim and project_dim != embed_dim:
            self.projection = nn.Linear(embed_dim, project_dim)
        else:
            self.projection = None

        # Learnable stream identity - distinguishes this stream geometrically
        self.stream_identity = nn.Parameter(
            torch.randn(1, fingerprint_dim) * 0.02
        )

    def attach_encoder(
        self,
        encoder: nn.Module,
        extract_fn: Optional[Callable] = None,
    ) -> 'EncoderStream':
        """
        Attach a pre-trained encoder.

        Args:
            encoder: The encoder module.
            extract_fn: Optional fn(encoder, x, **kwargs) -> features.
                        Defaults to encoder.forward().

        Returns:
            Self for chaining.
        """
        self.encoder = encoder
        self.extract_fn = extract_fn

        if self.frozen:
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()

        return self

    def detach_encoder(self) -> Optional[nn.Module]:
        """Detach and return the encoder."""
        encoder = self.encoder
        self.encoder = None
        self.extract_fn = None
        return encoder

    @property
    def has_encoder(self) -> bool:
        return self.encoder is not None

    def extract(self, x: Any, **kwargs) -> Tensor:
        """
        Extract features from input using attached encoder.

        Args:
            x: Input data (tokens, latents, etc.)
            **kwargs: Passed to encoder.

        Returns:
            Extracted features tensor.
        """
        if self.encoder is None:
            raise RuntimeError(f"EncoderStream '{self.name}' has no encoder attached.")

        with torch.set_grad_enabled(not self.frozen):
            if self.extract_fn is not None:
                features = self.extract_fn(self.encoder, x, **kwargs)
            else:
                features = self.encoder(x, **kwargs)

        # Handle various output types
        if isinstance(features, tuple):
            features = features[0]
        if isinstance(features, dict):
            # Common transformer output keys
            for key in ['last_hidden_state', 'pooler_output', 'hidden_states', 'logits']:
                if key in features:
                    features = features[key]
                    if isinstance(features, tuple):
                        features = features[-1]  # Last layer
                    break

        return features

    def forward(self, x: Any, **kwargs) -> EncoderMail:
        """
        Extract features and generate fingerprinted mail.

        Args:
            x: Input data.
            **kwargs: Passed to extraction.

        Returns:
            EncoderMail with content and fingerprint.
        """
        # Extract
        features = self.extract(x, **kwargs)

        # Ensure 3D: [B, L, D]
        if features.dim() == 2:
            features = features.unsqueeze(1)

        # Project if needed
        if self.projection is not None:
            features = self.projection(features.to(self.projection.weight.dtype))

        # Get learned fingerprint (expand to batch)
        B = features.shape[0]
        fingerprint = self.address.fingerprint.unsqueeze(0).expand(B, -1)  # [B, fp_dim]

        # Add stream identity
        fingerprint = fingerprint + self.stream_identity

        return EncoderMail(
            content=features,
            fingerprint=fingerprint,
            stream_type=self.stream_type,
            source=self.name,
            metadata={'frozen': self.frozen, 'projected': self.projection is not None},
        )

    def __repr__(self) -> str:
        status = "attached" if self.encoder else "empty"
        frozen = "frozen" if self.frozen else "trainable"
        return (f"EncoderStream('{self.name}', {self.stream_type.name}, "
                f"dim={self.embed_dim}->{self.output_dim}, {status}, {frozen})")


# =============================================================================
# AGATHA HEAD ROUTER
# =============================================================================

class AgathaHeadRouter(BaseRouter):
    """
    Agatha Block 1 head router.

    Extracts from pre-trained encoders and produces fingerprinted
    mail to teach downstream UNET experts.

    Default streams:
        - qwen: TEXT (QWEN 2.5 Instruct)
        - flux_ae: IMAGE (Flux autoencoder)
        - lyra: LATENT (Lyra bottleneck)
        - dino: GUIDANCE (Dino v3)

    Usage:
        head = AgathaHeadRouter('agatha', embed_dim=1024)

        # Attach encoders
        head.attach_encoder('qwen', qwen_model, embed_dim=4096)
        head.attach_encoder('flux_ae', flux_encoder, embed_dim=64)
        head.attach_encoder('lyra', lyra_model, embed_dim=256)
        head.attach_encoder('dino', dino_model, embed_dim=1024)

        # Forward
        mail = head({
            'qwen': text_tokens,
            'flux_ae': image_latents,
            'lyra': lyra_input,
            'dino': dino_input,
        })

        # mail.streams['qwen'].content, mail.streams['qwen'].fingerprint
        # mail.fused, mail.fused_fingerprint
    """

    STREAM_TYPES = {
        'qwen': StreamType.TEXT,
        'flux_ae': StreamType.IMAGE,
        'lyra': StreamType.LATENT,
        'dino': StreamType.GUIDANCE,
    }

    def __init__(
        self,
        name: str = 'agatha_head',
        embed_dim: int = 1024,
        fingerprint_dim: int = 64,
        fusion_type: str = 'adaptive',  # 'adaptive', 'gated', 'concat', 'attention'
        uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, uuid=uuid, strict=False, **kwargs)

        self.embed_dim = embed_dim
        self.fingerprint_dim = fingerprint_dim
        self.fusion_type = fusion_type

        # Encoder streams (populated on attach)
        self.streams: Dict[str, EncoderStream] = {}

        # Fusion component (rebuilt when streams change)
        self._fusion: Optional[FusionComponent] = None

        # Fusion fingerprint (learned)
        self.fusion_address = AddressComponent(
            f'{name}_fusion_address',
            fingerprint_dim=fingerprint_dim,
        )

        # Data component for batch operations
        self.data = DataComponent(f'{name}_data')

        # Teaching hooks for monitoring
        self._teaching_hooks: Dict[str, Callable[[HeadMail], None]] = {}

        # Debug state
        self._debug = False
        self._last_mail: Optional[HeadMail] = None

    # =========================================================================
    # ENCODER MANAGEMENT
    # =========================================================================

    def attach_encoder(
        self,
        name: str,
        encoder: nn.Module,
        embed_dim: Optional[int] = None,
        stream_type: Optional[StreamType] = None,
        extract_fn: Optional[Callable] = None,
        frozen: bool = True,
    ) -> 'AgathaHeadRouter':
        """
        Attach an encoder to a named stream.

        Args:
            name: Stream name ('qwen', 'flux_ae', 'lyra', 'dino', or custom).
            encoder: The encoder module.
            embed_dim: Encoder output dim. Auto-detected if not provided.
            stream_type: Type of stream. Auto-detected from name if not provided.
            extract_fn: Custom extraction fn(encoder, x, **kwargs) -> features.
            frozen: Whether to freeze encoder parameters.

        Returns:
            Self for chaining.
        """
        # Auto-detect stream type
        if stream_type is None:
            stream_type = self.STREAM_TYPES.get(name, StreamType.FUSED)

        # Auto-detect embed_dim
        if embed_dim is None:
            embed_dim = self._detect_embed_dim(encoder)

        # Create stream
        stream = EncoderStream(
            name=name,
            stream_type=stream_type,
            embed_dim=embed_dim,
            fingerprint_dim=self.fingerprint_dim,
            project_dim=self.embed_dim,
            frozen=frozen,
        )
        stream.attach_encoder(encoder, extract_fn)

        # Move to same device as router
        device = self.fusion_address.fingerprint.device
        if device.type != 'cpu':
            stream = stream.to(device)

        # Register
        self.streams[name] = stream
        self.attach(name, stream)

        # Rebuild fusion
        self._rebuild_fusion()

        return self

    def detach_encoder(self, name: str) -> Optional[nn.Module]:
        """
        Detach an encoder from a stream.

        Args:
            name: Stream name.

        Returns:
            The detached encoder or None.
        """
        if name not in self.streams:
            return None

        stream = self.streams.pop(name)
        self.detach(name)
        encoder = stream.detach_encoder()

        self._rebuild_fusion()

        return encoder

    def _detect_embed_dim(self, encoder: nn.Module) -> int:
        """Try to auto-detect encoder output dimension."""
        # Check common attributes
        for attr in ['config.hidden_size', 'hidden_size', 'embed_dim', 'd_model', 'num_features']:
            try:
                obj = encoder
                for part in attr.split('.'):
                    obj = getattr(obj, part)
                if isinstance(obj, int):
                    return obj
            except AttributeError:
                continue

        # Fallback
        return self.embed_dim

    def _rebuild_fusion(self):
        """Rebuild fusion component when streams change."""
        n = len(self.streams)

        if n < 2:
            self._fusion = None
            return

        # Create appropriate fusion
        if self.fusion_type == 'adaptive':
            self._fusion = AdaptiveFusion(
                f'{self.name}_fusion',
                num_inputs=n,
                in_features=self.embed_dim,
            )
        elif self.fusion_type == 'gated':
            self._fusion = GatedFusion(
                f'{self.name}_fusion',
                num_inputs=n,
                in_features=self.embed_dim,
            )
        elif self.fusion_type == 'concat':
            self._fusion = ConcatFusion(
                f'{self.name}_fusion',
                num_inputs=n,
                in_features=self.embed_dim,
                out_features=self.embed_dim,
            )
        elif self.fusion_type == 'attention':
            self._fusion = AttentionFusion(
                f'{self.name}_fusion',
                num_inputs=n,
                in_features=self.embed_dim,
                num_heads=8,
            )
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # Move to same device as other components
        try:
            device = self.fusion_address.fingerprint.device
            if device.type != 'cpu':
                self._fusion = self._fusion.to(device)
        except (StopIteration, AttributeError):
            pass

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(
        self,
        inputs: Dict[str, Any],
        timestep: Optional[Tensor] = None,
    ) -> HeadMail:
        """
        Extract from all encoders and produce teaching mail.

        Args:
            inputs: Dict mapping stream name to input data.
            timestep: Optional diffusion timestep [B].

        Returns:
            HeadMail with all streams and fusion.
        """
        stream_mails: Dict[str, EncoderMail] = {}

        # Extract from each available stream
        for name, stream in self.streams.items():
            if name in inputs:
                mail = stream(inputs[name])
                stream_mails[name] = mail

        # Fuse if multiple streams and fusion available
        fused = None
        fused_fingerprint = None

        if len(stream_mails) > 1 and self._fusion is not None:
            # Pool each stream to [B, D] for fusion
            pooled = []
            for name in self.streams.keys():
                if name in stream_mails:
                    content = stream_mails[name].content
                    if content.dim() == 3:
                        content = content.mean(dim=1)
                    pooled.append(content)

            # Cast to fusion dtype and fuse
            fusion_dtype = next(self._fusion.parameters()).dtype
            pooled = [p.to(fusion_dtype) for p in pooled]
            fused = self._fusion(*pooled)

            # Get fusion fingerprint (expand to batch)
            B = fused.shape[0]
            fused_fingerprint = self.fusion_address.fingerprint.unsqueeze(0).expand(B, -1)

        # Build mail
        mail = HeadMail(
            streams=stream_mails,
            fused=fused,
            fused_fingerprint=fused_fingerprint,
            timestep=timestep,
        )

        # Debug storage
        if self._debug:
            self._last_mail = mail

        # Call teaching hooks
        for hook in self._teaching_hooks.values():
            hook(mail)

        return mail

    # =========================================================================
    # TEACHING UTILITIES
    # =========================================================================

    def register_hook(self, name: str, hook: Callable[[HeadMail], None]) -> 'AgathaHeadRouter':
        """Register a teaching hook called after each forward."""
        self._teaching_hooks[name] = hook
        return self

    def remove_hook(self, name: str) -> 'AgathaHeadRouter':
        """Remove a teaching hook."""
        self._teaching_hooks.pop(name, None)
        return self

    def debug_on(self) -> 'AgathaHeadRouter':
        """Enable debug mode (stores last mail)."""
        self._debug = True
        return self

    def debug_off(self) -> 'AgathaHeadRouter':
        """Disable debug mode."""
        self._debug = False
        self._last_mail = None
        return self

    @property
    def last_mail(self) -> Optional[HeadMail]:
        """Get last mail (debug mode only)."""
        return self._last_mail

    def fingerprint_similarity(self) -> Dict[str, float]:
        """
        Compute pairwise fingerprint cosine similarities.

        Useful for verifying streams have distinct geometric identities.
        Requires debug mode with at least one forward pass.
        """
        if self._last_mail is None:
            raise RuntimeError("No mail. Enable debug and run forward first.")

        fps = self._last_mail.get_all_fingerprints()
        names = list(fps.keys())

        sims = {}
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                f1 = fps[n1].mean(dim=0)
                f2 = fps[n2].mean(dim=0)
                sim = nn.functional.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
                sims[f'{n1}-{n2}'] = sim

        return sims

    # =========================================================================
    # STATE
    # =========================================================================

    def stream_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all streams."""
        return {
            name: {
                'type': stream.stream_type.name,
                'embed_dim': stream.embed_dim,
                'output_dim': stream.output_dim,
                'frozen': stream.frozen,
                'attached': stream.has_encoder,
            }
            for name, stream in self.streams.items()
        }

    def __repr__(self) -> str:
        streams = ', '.join(self.streams.keys()) if self.streams else 'none'
        return (f"AgathaHeadRouter('{self.name}', streams=[{streams}], "
                f"embed_dim={self.embed_dim}, fusion='{self.fusion_type}')")


# =============================================================================
# CONVENIENCE
# =============================================================================

def create_agatha_head(
    embed_dim: int = 1024,
    fingerprint_dim: int = 64,
    fusion_type: str = 'adaptive',
) -> AgathaHeadRouter:
    """Create standard Agatha head router."""
    return AgathaHeadRouter(
        name='agatha_head',
        embed_dim=embed_dim,
        fingerprint_dim=fingerprint_dim,
        fusion_type=fusion_type,
    )


# =============================================================================
# MOCK FOR TESTING
# =============================================================================

class MockEncoder(nn.Module):
    """Mock encoder for testing."""

    def __init__(self, in_dim: int, out_dim: int, seq_len: int = 1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.seq_len = seq_len
        self.hidden_size = out_dim  # For auto-detection

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2 and self.seq_len > 1:
            x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        return self.proj(x)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':

    def section(title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------------------------------------------
    section("ENCODER STREAM")
    # -------------------------------------------------------------------------

    stream = EncoderStream(
        'test',
        stream_type=StreamType.TEXT,
        embed_dim=256,
        fingerprint_dim=64,
        project_dim=512,
    )
    print(f"Stream: {stream}")

    encoder = MockEncoder(128, 256, seq_len=16)
    stream.attach_encoder(encoder)
    print(f"After attach: {stream}")

    x = torch.randn(4, 128)
    mail = stream(x)
    print(f"Mail: {mail}")

    # -------------------------------------------------------------------------
    section("AGATHA HEAD ROUTER")
    # -------------------------------------------------------------------------

    head = create_agatha_head(embed_dim=512, fingerprint_dim=64)
    print(f"Head: {head}")

    # Attach encoders
    head.attach_encoder('qwen', MockEncoder(256, 768, seq_len=32), embed_dim=768)
    head.attach_encoder('flux_ae', MockEncoder(64, 1024), embed_dim=1024)
    head.attach_encoder('lyra', MockEncoder(128, 256), embed_dim=256)
    head.attach_encoder('dino', MockEncoder(128, 384, seq_len=16), embed_dim=384)

    print(f"\nWith encoders: {head}")
    for name, status in head.stream_status().items():
        print(f"  {name}: {status}")

    # -------------------------------------------------------------------------
    section("FORWARD")
    # -------------------------------------------------------------------------

    head.debug_on()

    inputs = {
        'qwen': torch.randn(4, 256),
        'flux_ae': torch.randn(4, 64),
        'lyra': torch.randn(4, 128),
        'dino': torch.randn(4, 128),
    }

    mail = head(inputs)

    print(f"Sources: {mail.sources}")
    print(f"Fused: {mail.fused.shape}")
    print(f"Fused FP: {mail.fused_fingerprint.shape}")

    for name in mail.sources:
        m = mail[name]
        print(f"  {name}: content={list(m.content.shape)}, fp={list(m.fingerprint.shape)}")

    # -------------------------------------------------------------------------
    section("FINGERPRINT ANALYSIS")
    # -------------------------------------------------------------------------

    fps = mail.get_all_fingerprints()
    for name, fp in fps.items():
        print(f"  {name}: norm={fp.norm(dim=-1).mean():.4f}")

    sims = head.fingerprint_similarity()
    print("\nPairwise similarities:")
    for pair, sim in sims.items():
        print(f"  {pair}: {sim:.4f}")

    # -------------------------------------------------------------------------
    section("PARTIAL INPUTS")
    # -------------------------------------------------------------------------

    partial = {'qwen': torch.randn(4, 256), 'flux_ae': torch.randn(4, 64)}
    mail = head(partial)
    print(f"Partial sources: {mail.sources}")
    print(f"Fused: {mail.fused.shape if mail.fused is not None else None}")

    # -------------------------------------------------------------------------
    section("TEACHING HOOK")
    # -------------------------------------------------------------------------

    log = []
    head.register_hook('logger', lambda m: log.append(len(m.sources)))

    for _ in range(3):
        head(inputs)

    print(f"Hook logged: {log}")
    head.remove_hook('logger')

    # -------------------------------------------------------------------------
    section("DETACH / REATTACH")
    # -------------------------------------------------------------------------

    dino = head.detach_encoder('dino')
    print(f"After detach: {list(head.streams.keys())}")

    head.attach_encoder('dino', dino, embed_dim=384)
    print(f"After reattach: {list(head.streams.keys())}")

    # -------------------------------------------------------------------------
    section("GPU")
    # -------------------------------------------------------------------------

    if torch.cuda.is_available():
        head_gpu = create_agatha_head(embed_dim=512).to(device)
        head_gpu.attach_encoder('qwen', MockEncoder(256, 768).to(device), embed_dim=768)
        head_gpu.attach_encoder('flux_ae', MockEncoder(64, 1024).to(device), embed_dim=1024)

        mail = head_gpu({
            'qwen': torch.randn(4, 256, device=device),
            'flux_ae': torch.randn(4, 64, device=device),
        })
        print(f"Device: {mail.fused.device}")
    else:
        print("No CUDA")

    # -------------------------------------------------------------------------
    section("DONE")
    # -------------------------------------------------------------------------

    print("\nAgathaHeadRouter ready.")
    print("Uses: DataComponent, AddressComponent, FusionComponent")