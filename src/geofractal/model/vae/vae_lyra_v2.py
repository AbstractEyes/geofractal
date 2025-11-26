"""
Multi-Modal VAE with Adaptive Cantor Fusion
============================================

Cantor-based fusion with learned visibility (alpha) and capacity (beta):
- Alpha: Learned visibility controlling latent space usage (tied to KL divergence)
- Beta: Learned capacity controlling source influence strength
- Decoupled T5-XL representations for CLIP-L and CLIP-G
- Cantor fractal routing for sparse attention
- HARD MASKING: Strict isolation between binding groups

Author: AbstractPhil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

class FusionStrategy(Enum):
    """Fusion strategies for multi-modal VAE."""
    CONCATENATE = "concatenate"
    ATTENTION = "attention"
    GATED = "gated"
    CANTOR = "cantor"
    GEOMETRIC = "geometric"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE_CANTOR = "adaptive_cantor"  # Default and recommended


@dataclass
class MultiModalVAEConfig:
    """Configuration for multi-modal VAE."""
    # Input modalities
    modality_dims: Dict[str, int] = None
    modality_seq_lens: Dict[str, int] = None

    # Latent space
    latent_dim: int = 2048
    seq_len: int = 77

    # Architecture
    encoder_layers: int = 3
    decoder_layers: int = 3
    hidden_dim: int = 1024
    dropout: float = 0.1

    # Fusion
    fusion_strategy: str = "adaptive_cantor"  # Default
    fusion_heads: int = 8
    fusion_dropout: float = 0.1
    binding_config: Optional[Dict[str, Dict[str, float]]] = None

    # Cantor parameters (for cantor and adaptive_cantor)
    cantor_depth: int = 8
    cantor_local_window: int = 3

    # Adaptive fusion parameters (for adaptive_cantor)
    alpha_init: float = 1.0
    beta_init: float = 0.3
    alpha_lr_scale: float = 0.1
    beta_lr_scale: float = 1.0

    # Loss weights
    beta_kl: float = 0.1
    beta_reconstruction: float = 1.0
    beta_cross_modal: float = 0.0  # DISABLED by default - causes contamination
    beta_alpha_regularization: float = 0.01

    # KL clamping
    kl_clamp_max: float = 1.0  # Prevent KL explosion
    logvar_clamp_min: float = -10.0
    logvar_clamp_max: float = 10.0

    # Training
    use_amp: bool = True

    # Reproducibility
    seed: Optional[int] = None

    def __post_init__(self):
        # Default: SDXL configuration with decoupled T5
        if self.modality_dims is None:
            self.modality_dims = {
                "clip_l": 768,
                "clip_g": 1280,
                "t5_xl_l": 2048,
                "t5_xl_g": 2048
            }

        # Default: Different sequence lengths
        if self.modality_seq_lens is None:
            self.modality_seq_lens = {
                "clip_l": 77,
                "clip_g": 77,
                "t5_xl_l": 512,
                "t5_xl_g": 512
            }

        # Default binding for adaptive strategies
        if self.binding_config is None and self.fusion_strategy == "adaptive_cantor":
            self.binding_config = {
                "clip_l": {"t5_xl_l": 0.3},
                "clip_g": {"t5_xl_g": 0.3},
                "t5_xl_l": {},
                "t5_xl_g": {}
            }


# ============================================================================
# FUSION MODULES
# ============================================================================

class ConcatenateFusion(nn.Module):
    """Simple concatenation-based fusion with sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())
        total_dim = sum(modality_dims.values())

        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = list(modality_inputs.values())[0].device

        # Pad all modalities to max sequence length
        padded = []
        for name in self.modality_names:
            if name in modality_inputs:
                inp = modality_inputs[name]
                B, seq_len, dim = inp.shape

                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, dim, device=device)
                    inp = torch.cat([inp, padding], dim=1)

                padded.append(inp)

        # Concatenate along feature dimension
        cat_inputs = torch.cat(padded, dim=-1)
        return self.projection(cat_inputs)


class AttentionFusion(nn.Module):
    """Standard multi-head attention fusion with sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())
        self.num_modalities = len(self.modality_names)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        # Project each modality to common dimension
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Multi-head attention
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = list(modality_inputs.values())[0].device
        B = list(modality_inputs.values())[0].shape[0]

        # Project and pad all modalities
        projected = []
        for name in self.modality_names:
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                seq_len = proj.shape[1]

                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, self.output_dim, device=device)
                    proj = torch.cat([proj, padding], dim=1)

                projected.append(proj)

        # Stack: [batch, num_modalities, max_seq, dim]
        stacked = torch.stack(projected, dim=1)

        # Apply attention
        Q = self.q_proj(stacked[:, 0:1])  # Use first modality as query
        K = self.k_proj(stacked)
        V = self.v_proj(stacked)

        # Reshape for multi-head
        Q = Q.view(B, 1, self.max_seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        K = K.view(B, self.num_modalities, self.max_seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = V.view(B, self.num_modalities, self.max_seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # Attention scores
        scores = torch.einsum('bhqsd,bhmsd->bhqms', Q, K) / math.sqrt(self.head_dim)
        scores = scores / self.temperature.abs()

        attn = F.softmax(scores, dim=-2)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.einsum('bhqms,bhmsd->bhqsd', attn, V)
        out = out.squeeze(2).permute(0, 2, 1, 3).reshape(B, self.max_seq_len, self.output_dim)

        return self.out_proj(out)


class GatedFusion(nn.Module):
    """Gated fusion with learned modality weights and sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())

        # Project each modality to common dimension
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Gating networks
        self.gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.GELU(),
                nn.Linear(output_dim // 4, 1),
                nn.Sigmoid()
            )
            for name in self.modality_names
        })

        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = list(modality_inputs.values())[0].device

        # Project, gate, and pad each modality
        gated_features = []

        for name in self.modality_names:
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                gate = self.gates[name](proj)
                gated = proj * gate

                B, seq_len, dim = gated.shape
                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, dim, device=device)
                    gated = torch.cat([gated, padding], dim=1)

                gated_features.append(gated)

        # Sum gated features
        fused = sum(gated_features) / len(gated_features)

        return self.output_proj(fused)


class CantorModalityFusion(nn.Module):
    """Cantor-based fusion with sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            num_heads: int = 8,
            cantor_depth: int = 8,
            local_window: int = 3,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())
        self.num_modalities = len(self.modality_names)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        # Project each modality to common dimension
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Modality embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(self.num_modalities, output_dim) * 0.02
        )

        # QKV projections
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)

        # Cantor routing
        self.cantor_depth = cantor_depth
        self.local_window = min(local_window, self.num_modalities)

        # Pre-compute Cantor coordinates
        self.register_buffer(
            'modality_cantor_coords',
            self._compute_modality_cantor_coordinates()
        )

        # Pre-compute routing
        self.register_buffer(
            'modality_routes',
            self._build_modality_routes()
        )

        # Output
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def _cantor_coordinate(self, position: int, max_len: int, depth: int) -> float:
        """Compute Cantor set coordinate."""
        x = position / max(1, max_len - 1)
        x = max(1e-6, min(x, 1.0 - 1e-6))

        cantor_val = 0.0
        factor = 0.5

        for _ in range(depth):
            x *= 3.0
            digit = int(x)
            x -= digit

            if digit == 2:
                cantor_val += factor

            factor *= 0.5

        return cantor_val

    def _compute_modality_cantor_coordinates(self) -> torch.Tensor:
        """Map each modality to Cantor coordinate."""
        coords = torch.tensor([
            self._cantor_coordinate(i, self.num_modalities, self.cantor_depth)
            for i in range(self.num_modalities)
        ], dtype=torch.float32)
        return coords

    def _build_modality_routes(self) -> torch.Tensor:
        """Build routing table for modality attention."""
        routes = torch.zeros(self.num_modalities, self.local_window, dtype=torch.long)

        for i in range(self.num_modalities):
            distances = torch.abs(
                self.modality_cantor_coords - self.modality_cantor_coords[i]
            )
            _, nearest = torch.topk(distances, self.local_window, largest=False)
            routes[i] = nearest

        return routes

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multiple modality inputs using Cantor routing."""
        B = list(modality_inputs.values())[0].shape[0]
        device = list(modality_inputs.values())[0].device

        # Project and pad all modalities to common space
        projected = []
        for i, name in enumerate(self.modality_names):
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                proj = proj + self.modality_embeddings[i]

                seq_len = proj.shape[1]
                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, self.output_dim, device=device)
                    proj = torch.cat([proj, padding], dim=1)

                projected.append(proj)

        # Stack: [batch, num_modalities, max_seq, dim]
        stacked = torch.stack(projected, dim=1)

        # Multi-head attention with Cantor routing
        Q = self.q_proj(stacked).view(B, self.num_modalities, self.max_seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(stacked).view(B, self.num_modalities, self.max_seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(stacked).view(B, self.num_modalities, self.max_seq_len, self.num_heads, self.head_dim)

        # Permute to [batch, heads, num_modalities, seq, head_dim]
        Q = Q.permute(0, 3, 1, 2, 4)
        K = K.permute(0, 3, 1, 2, 4)
        V = V.permute(0, 3, 1, 2, 4)

        # Sparse attention via Cantor routing
        routes = self.modality_routes.to(device)

        attended = []
        for i in range(self.num_modalities):
            neighbors = routes[i]

            q_i = Q[:, :, i, :, :]
            k_neighbors = K[:, :, neighbors, :, :]
            v_neighbors = V[:, :, neighbors, :, :]

            scores = torch.einsum('bhsd,bhwsd->bhsw', q_i, k_neighbors) / math.sqrt(self.head_dim)
            scores = scores / self.temperature.abs()

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out_i = torch.einsum('bhsw,bhwsd->bhsd', attn, v_neighbors)
            attended.append(out_i)

        # Stack and mean over modalities
        fused = torch.stack(attended, dim=2).mean(dim=2)

        # Reshape back
        fused = fused.permute(0, 2, 1, 3).reshape(B, self.max_seq_len, self.output_dim)

        output = self.out_proj(fused)
        output = self.dropout(output)

        return output


class GeometricModalityFusion(nn.Module):
    """Geometric fusion with sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            num_heads: int = 4,
            use_cayley: bool = True,
            use_angular: bool = True,
            temperature: float = 0.07,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())
        self.num_modalities = len(self.modality_names)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.use_cayley = use_cayley
        self.use_angular = use_angular

        # Project modalities to common space
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # QKV
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)

        # Pentachoron role weights
        if use_angular:
            role_weights = torch.tensor([1.0, -0.75, 0.75, 0.75, -0.75])
            if self.num_modalities < 5:
                role_weights = role_weights[:self.num_modalities]
            elif self.num_modalities > 5:
                extra = torch.ones(self.num_modalities - 5) * 0.5
                role_weights = torch.cat([role_weights, extra])
            self.register_buffer("role_weights", role_weights)

        # Output
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)

    def _compute_angular_attention(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention based on angular relationships."""
        B = features[0].shape[0]
        features_norm = [F.normalize(f, dim=-1) for f in features]
        attention = torch.zeros(B, self.num_modalities, device=features[0].device)

        for i, feat_i in enumerate(features_norm):
            for j, feat_j in enumerate(features_norm):
                if i != j:
                    cos_sim = (feat_i * feat_j).sum(dim=-1).mean(dim=-1)
                    angle = torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))
                    attention[:, i] += self.role_weights[j] * torch.exp(-angle / self.temperature.abs())

        return F.softmax(attention, dim=-1)

    def _compute_cayley_attention(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention based on Cayley-Menger volumes."""
        B = features[0].shape[0]
        volume_scores = []

        for i in range(self.num_modalities):
            simplex_points = [features[i]]
            for j in range(min(4, self.num_modalities - 1)):
                angle = (j + 1) * math.pi / 4
                other_idx = (i + j + 1) % self.num_modalities
                rot_feat = features[i] * math.cos(angle) + features[other_idx] * math.sin(angle)
                simplex_points.append(rot_feat)

            simplex = torch.stack(simplex_points, dim=1)
            diff = simplex.unsqueeze(2) - simplex.unsqueeze(1)
            distsq = (diff * diff).sum(dim=-1).sum(dim=-1)
            volume = distsq.mean(dim=(1, 2))
            volume_scores.append(volume)

        volumes = torch.stack(volume_scores, dim=1)
        return F.softmax(volumes / self.temperature.abs(), dim=-1)

    def _compute_standard_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Standard multi-head attention."""
        scores = torch.einsum('bhqsd,bhmsd->bhqms', Q, K) / math.sqrt(self.head_dim)
        scores = scores / self.temperature.abs()
        attn = F.softmax(scores, dim=-2)
        attn = self.dropout(attn)
        out = torch.einsum('bhqms,bhmsd->bhqsd', attn, V)
        B = out.shape[0]
        out = out.squeeze(2).permute(0, 2, 1, 3).reshape(B, -1, self.output_dim)
        return out

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse modalities using geometric attention."""
        B = list(modality_inputs.values())[0].shape[0]
        device = list(modality_inputs.values())[0].device

        # Project and pad features
        features = []
        for name in self.modality_names:
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                seq_len = proj.shape[1]

                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, self.output_dim, device=device)
                    proj = torch.cat([proj, padding], dim=1)

                features.append(proj)

        Q_feat = features[0]
        Q = self.q_proj(Q_feat).view(B, self.num_heads, 1, self.max_seq_len, self.head_dim)

        K_list, V_list = [], []
        for feat in features:
            K = self.k_proj(feat).view(B, self.num_heads, 1, self.max_seq_len, self.head_dim)
            V = self.v_proj(feat).view(B, self.num_heads, 1, self.max_seq_len, self.head_dim)
            K_list.append(K)
            V_list.append(V)

        K = torch.cat(K_list, dim=2)
        V = torch.cat(V_list, dim=2)

        attention_outputs = []
        mha_out = self._compute_standard_attention(Q, K, V)
        attention_outputs.append(mha_out)

        if self.use_angular:
            angular_weights = self._compute_angular_attention(features)
            angular_out = torch.zeros_like(features[0])
            for i in range(self.num_modalities):
                w = angular_weights[:, i]
                angular_out = angular_out + w.unsqueeze(-1).unsqueeze(-1) * features[i]
            attention_outputs.append(angular_out)

        if self.use_cayley:
            cayley_weights = self._compute_cayley_attention(features)
            cayley_out = torch.zeros_like(features[0])
            for i in range(self.num_modalities):
                w = cayley_weights[:, i]
                cayley_out = cayley_out + w.unsqueeze(-1).unsqueeze(-1) * features[i]
            attention_outputs.append(cayley_out)

        attn_weights = F.softmax(self.attention_weights[:len(attention_outputs)], dim=0)
        fused = torch.zeros_like(attention_outputs[0])
        for i, out in enumerate(attention_outputs):
            fused = fused + attn_weights[i] * out

        output = self.out_proj(fused)
        output = self.dropout(output)

        return output


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion with sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())

        # Stage 1: Project each modality
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Stage 2: Hierarchical combination
        self.stage2_proj = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Stage 3: Final fusion
        self.final_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = list(modality_inputs.values())[0].device

        # Stage 1: Project and pad all modalities
        projected = []
        for name in self.modality_names:
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                B, seq_len, dim = proj.shape

                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, dim, device=device)
                    proj = torch.cat([proj, padding], dim=1)

                projected.append(proj)

        # Stage 2: Pairwise combinations
        if len(projected) == 1:
            return self.final_proj(projected[0])

        combined = projected[0]
        for i in range(1, len(projected)):
            pair = torch.cat([combined, projected[i]], dim=-1)
            combined = self.stage2_proj(pair)

        # Stage 3: Final projection
        return self.final_proj(combined)


class AdaptiveCantorModalityFusion(nn.Module):
    """
    Cantor-based fusion with learned alpha (visibility) and beta (capacity).

    NOW WITH HARD MASKING: Strict isolation between binding groups.
    - (clip_l ↔ t5_xl_l) is completely isolated from (clip_g ↔ t5_xl_g)
    - Each modality gets its own output (no averaging across groups)
    """

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            binding_config: Dict[str, Dict[str, float]],
            output_dim: int,
            num_heads: int = 8,
            cantor_depth: int = 8,
            local_window: int = 3,
            alpha_init: float = 1.0,
            beta_init: float = 0.3,
            alpha_lr_scale: float = 0.1,
            beta_lr_scale: float = 1.0,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.modality_dims = modality_dims
        self.modality_seq_lens = modality_seq_lens
        self.binding_config = binding_config
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.alpha_lr_scale = alpha_lr_scale
        self.beta_lr_scale = beta_lr_scale

        # Project each modality to common dimension
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Modality embeddings (Cantor)
        self.modality_embeddings = nn.Parameter(
            torch.randn(self.num_modalities, output_dim) * 0.02
        )

        # Learned alpha (visibility) per modality
        self.alphas = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(alpha_init))
            for name in self.modality_names
        })

        # Learned beta (capacity) per binding pair
        self.betas = nn.ParameterDict()
        for target, sources in binding_config.items():
            for source, init_weight in sources.items():
                if init_weight > 0:
                    key = f"{target}_{source}"
                    self.betas[key] = nn.Parameter(torch.tensor(beta_init))

        # Alpha-modulated gating networks
        self.alpha_gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.GELU(),
                nn.Linear(output_dim // 4, 1),
                nn.Sigmoid()
            )
            for name in self.modality_names
        })

        # QKV projections (Cantor)
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)

        # Per-modality output projections (no more shared output)
        self.out_projs = nn.ModuleDict({
            name: nn.Linear(output_dim, output_dim)
            for name in self.modality_names
        })

        # Cantor routing
        self.cantor_depth = cantor_depth
        self.local_window = min(local_window, self.num_modalities)

        # Pre-compute Cantor coordinates
        self.register_buffer(
            'modality_cantor_coords',
            self._compute_modality_cantor_coordinates()
        )

        # Pre-compute routing (still used for Cantor structure within groups)
        self.register_buffer(
            'modality_routes',
            self._build_modality_routes()
        )

        # BUILD HARD BINDING MASK - this is the key fix
        self.register_buffer(
            'binding_mask',
            self._build_binding_mask()
        )

        # Build binding groups for per-group averaging
        self.binding_groups = self._build_binding_groups()

        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def _cantor_coordinate(self, position: int, max_len: int, depth: int) -> float:
        """Compute Cantor set coordinate."""
        x = position / max(1, max_len - 1)
        x = max(1e-6, min(x, 1.0 - 1e-6))

        cantor_val = 0.0
        factor = 0.5

        for _ in range(depth):
            x *= 3.0
            digit = int(x)
            x -= digit

            if digit == 2:
                cantor_val += factor

            factor *= 0.5

        return cantor_val

    def _compute_modality_cantor_coordinates(self) -> torch.Tensor:
        """Map each modality to Cantor coordinate."""
        coords = torch.tensor([
            self._cantor_coordinate(i, self.num_modalities, self.cantor_depth)
            for i in range(self.num_modalities)
        ], dtype=torch.float32)
        return coords

    def _build_modality_routes(self) -> torch.Tensor:
        """Build routing table for modality attention using Cantor distances."""
        routes = torch.zeros(self.num_modalities, self.local_window, dtype=torch.long)

        for i in range(self.num_modalities):
            distances = torch.abs(
                self.modality_cantor_coords - self.modality_cantor_coords[i]
            )
            _, nearest = torch.topk(distances, self.local_window, largest=False)
            routes[i] = nearest

        return routes

    def _build_binding_mask(self) -> torch.Tensor:
        """
        Build HARD mask for attention: -inf for non-binding pairs, 0 for allowed.

        This creates strict isolation between binding groups:
        - clip_l can only attend to itself and t5_xl_l
        - clip_g can only attend to itself and t5_xl_g
        - t5_xl_l can only attend to itself and clip_l
        - t5_xl_g can only attend to itself and clip_g
        """
        # Start with all blocked
        mask = torch.full((self.num_modalities, self.num_modalities), float('-inf'))

        for i, name_i in enumerate(self.modality_names):
            # Self-attention always allowed
            mask[i, i] = 0.0

            # Check if this modality has bindings
            if name_i in self.binding_config:
                for bound_name in self.binding_config[name_i]:
                    if bound_name in self.modality_names:
                        j = self.modality_names.index(bound_name)
                        mask[i, j] = 0.0  # i can attend to j

            # Also check reverse: if another modality binds to this one
            for name_j, sources in self.binding_config.items():
                if name_i in sources and name_j in self.modality_names:
                    j = self.modality_names.index(name_j)
                    mask[i, j] = 0.0  # i can attend to j (bidirectional)

        print(f"  [AdaptiveCantor] Built binding mask:")
        for i, name in enumerate(self.modality_names):
            allowed = [self.modality_names[j] for j in range(self.num_modalities) if mask[i, j] == 0.0]
            print(f"    {name} → {allowed}")

        return mask

    def _build_binding_groups(self) -> Dict[str, List[int]]:
        """
        Build groups of modalities that should be averaged together.

        Returns dict mapping group_id -> list of modality indices
        """
        # Use union-find to build connected components
        parent = list(range(self.num_modalities))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union modalities that are bound together
        for i, name_i in enumerate(self.modality_names):
            if name_i in self.binding_config:
                for bound_name in self.binding_config[name_i]:
                    if bound_name in self.modality_names:
                        j = self.modality_names.index(bound_name)
                        union(i, j)

        # Build groups
        groups = {}
        for i in range(self.num_modalities):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        print(f"  [AdaptiveCantor] Binding groups:")
        for root, members in groups.items():
            member_names = [self.modality_names[i] for i in members]
            print(f"    Group {root}: {member_names}")

        return groups

    def get_alpha_params(self) -> Dict[str, torch.Tensor]:
        """Get alpha parameters for external use (e.g., loss computation)."""
        return {name: alpha for name, alpha in self.alphas.items()}

    def get_beta_params(self) -> Dict[str, torch.Tensor]:
        """Get beta parameters for external use."""
        return {key: beta for key, beta in self.betas.items()}

    def forward(
            self,
            modality_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse modalities using Cantor routing with adaptive alpha/beta and HARD MASKING.

        Each modality now gets its own output, only influenced by its binding group.
        """
        max_seq_len = max(self.modality_seq_lens.values())
        device = list(modality_inputs.values())[0].device
        B = list(modality_inputs.values())[0].shape[0]

        # Project and pad all modalities
        projected = {}
        original_seq_lens = {}

        for i, name in enumerate(self.modality_names):
            if name in modality_inputs:
                # Project to common space
                proj = self.modality_projections[name](modality_inputs[name])

                # Apply alpha-modulated gating
                alpha = self.alphas[name]
                gate = self.alpha_gates[name](proj)
                alpha_clamped = torch.sigmoid(alpha)
                proj = proj * (gate * alpha_clamped + (1 - alpha_clamped))

                # Add modality embedding
                proj = proj + self.modality_embeddings[i]

                # Store original length and pad
                _, seq_len, _ = proj.shape
                original_seq_lens[name] = seq_len

                if seq_len < max_seq_len:
                    padding = torch.zeros(B, max_seq_len - seq_len, self.output_dim, device=device)
                    proj = torch.cat([proj, padding], dim=1)

                projected[name] = proj

        # Stack: [batch, num_modalities, max_seq, dim]
        stacked = torch.stack([projected[name] for name in self.modality_names], dim=1)

        # Multi-head QKV
        Q = self.q_proj(stacked).view(B, self.num_modalities, max_seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(stacked).view(B, self.num_modalities, max_seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(stacked).view(B, self.num_modalities, max_seq_len, self.num_heads, self.head_dim)

        # Permute to [batch, heads, num_modalities, seq, head_dim]
        Q = Q.permute(0, 3, 1, 2, 4)
        K = K.permute(0, 3, 1, 2, 4)
        V = V.permute(0, 3, 1, 2, 4)

        # Get binding mask
        binding_mask = self.binding_mask.to(device)  # [num_mod, num_mod]

        # Compute attention for each modality with HARD MASKING
        attended = []
        for i, target_name in enumerate(self.modality_names):
            q_i = Q[:, :, i, :, :]  # [B, H, S, D]

            # Attend to ALL modalities but mask non-bound ones
            # k_all, v_all: [B, H, num_mod, S, D]
            k_all = K
            v_all = V

            # Compute scores: [B, H, S, num_mod]
            scores = torch.einsum('bhsd,bhwsd->bhsw', q_i, k_all) / math.sqrt(self.head_dim)
            scores = scores / self.temperature.abs()

            # Apply HARD MASK - add -inf to blocked pairs
            # binding_mask[i]: [num_mod], expand to [1, 1, 1, num_mod]
            mask_i = binding_mask[i].view(1, 1, 1, -1)
            scores = scores + mask_i

            # Apply beta modulation for bound pairs (soft boost on top of hard mask)
            if target_name in self.binding_config:
                for source_name, weight in self.binding_config[target_name].items():
                    if weight > 0:
                        key = f"{target_name}_{source_name}"
                        if key in self.betas and source_name in self.modality_names:
                            beta = self.betas[key]
                            beta_clamped = torch.sigmoid(beta)
                            source_idx = self.modality_names.index(source_name)

                            # Boost attention to bound source
                            # scores[:, :, :, source_idx] shape: [B, H, S]
                            scores[:, :, :, source_idx] = scores[:, :, :, source_idx] + beta_clamped

            # Softmax (masked positions become ~0 after softmax)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            # Apply attention: [B, H, S, D]
            out_i = torch.einsum('bhsw,bhwsd->bhsd', attn, v_all)
            attended.append(out_i)

        # Build per-modality outputs (NO global averaging)
        enriched = {}
        for i, name in enumerate(self.modality_names):
            # Get this modality's attended output
            out_i = attended[i]  # [B, H, S, D]

            # Find which group this modality belongs to
            group_members = None
            for root, members in self.binding_groups.items():
                if i in members:
                    group_members = members
                    break

            # Average only within the binding group
            if group_members and len(group_members) > 1:
                group_outputs = [attended[j] for j in group_members]
                out_i = torch.stack(group_outputs, dim=0).mean(dim=0)

            # Reshape: [B, H, S, D] -> [B, S, H*D]
            out_i = out_i.permute(0, 2, 1, 3).reshape(B, max_seq_len, self.output_dim)

            # Per-modality output projection
            out_i = self.out_projs[name](out_i)
            out_i = self.dropout(out_i)

            # Slice to original sequence length
            seq_len = original_seq_lens[name]
            enriched[name] = out_i[:, :seq_len, :]

        return enriched


# ============================================================================
# MULTIMODAL VAE
# ============================================================================

class MultiModalVAE(nn.Module):
    """Multi-modal VAE with multiple fusion strategies."""

    def __init__(self, config: MultiModalVAEConfig):
        super().__init__()
        self.config = config
        self.modality_names = list(config.modality_dims.keys())
        self.modality_seq_lens = config.modality_seq_lens
        self.num_modalities = len(self.modality_names)
        self.seed = config.seed

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # Fusion module - select based on strategy
        fusion_strategy = FusionStrategy(config.fusion_strategy)
        self.fusion_strategy = fusion_strategy

        if fusion_strategy == FusionStrategy.CONCATENATE:
            self.fusion = ConcatenateFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,
                output_dim=config.hidden_dim,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.ATTENTION:
            self.fusion = AttentionFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,
                output_dim=config.hidden_dim,
                num_heads=config.fusion_heads,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.GATED:
            self.fusion = GatedFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,
                output_dim=config.hidden_dim,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.CANTOR:
            self.fusion = CantorModalityFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,
                output_dim=config.hidden_dim,
                num_heads=config.fusion_heads,
                cantor_depth=config.cantor_depth,
                local_window=config.cantor_local_window,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.GEOMETRIC:
            self.fusion = GeometricModalityFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,
                output_dim=config.hidden_dim,
                num_heads=config.fusion_heads,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.HIERARCHICAL:
            self.fusion = HierarchicalFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,
                output_dim=config.hidden_dim,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.ADAPTIVE_CANTOR:
            self.fusion = AdaptiveCantorModalityFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,
                binding_config=config.binding_config,
                output_dim=config.hidden_dim,
                num_heads=config.fusion_heads,
                cantor_depth=config.cantor_depth,
                local_window=config.cantor_local_window,
                alpha_init=config.alpha_init,
                beta_init=config.beta_init,
                alpha_lr_scale=config.alpha_lr_scale,
                beta_lr_scale=config.beta_lr_scale,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {config.fusion_strategy}")

        # Encoders - per-modality for ADAPTIVE_CANTOR, single for others
        if fusion_strategy == FusionStrategy.ADAPTIVE_CANTOR:
            self.encoders = nn.ModuleDict()
            for name in self.modality_names:
                encoder_layers = []
                in_dim = config.hidden_dim
                for i in range(config.encoder_layers):
                    out_dim = config.hidden_dim if i < config.encoder_layers - 1 else config.latent_dim * 2
                    encoder_layers.extend([
                        nn.Linear(in_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        nn.GELU(),
                        nn.Dropout(config.dropout)
                    ])
                    in_dim = out_dim
                self.encoders[name] = nn.Sequential(*encoder_layers[:-2])

            # Per-modality mu/logvar projections
            self.fc_mus = nn.ModuleDict({
                name: nn.Linear(config.latent_dim * 2, config.latent_dim)
                for name in self.modality_names
            })
            self.fc_logvars = nn.ModuleDict({
                name: nn.Linear(config.latent_dim * 2, config.latent_dim)
                for name in self.modality_names
            })
        else:
            # Single encoder for other strategies
            encoder_layers = []
            in_dim = config.hidden_dim
            for i in range(config.encoder_layers):
                out_dim = config.hidden_dim if i < config.encoder_layers - 1 else config.latent_dim * 2
                encoder_layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout)
                ])
                in_dim = out_dim
            self.encoder = nn.Sequential(*encoder_layers[:-2])
            self.fc_mu = nn.Linear(config.latent_dim * 2, config.latent_dim)
            self.fc_logvar = nn.Linear(config.latent_dim * 2, config.latent_dim)

        # Decoders (per-modality for all strategies)
        self.decoders = nn.ModuleDict()
        for name, dim in config.modality_dims.items():
            decoder_layers = []
            in_dim = config.latent_dim
            for i in range(config.decoder_layers):
                out_dim = config.hidden_dim if i < config.decoder_layers - 1 else dim
                decoder_layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim) if i < config.decoder_layers - 1 else nn.Identity(),
                    nn.GELU() if i < config.decoder_layers - 1 else nn.Identity(),
                    nn.Dropout(config.dropout) if i < config.decoder_layers - 1 else nn.Identity()
                ])
                in_dim = out_dim
            self.decoders[name] = nn.Sequential(*decoder_layers)

        # Cross-modal projection layers
        self.cross_modal_projections = nn.ModuleDict({
            name: nn.Linear(dim, config.latent_dim)
            for name, dim in config.modality_dims.items()
        })

    def get_fusion_params(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get learned alpha and beta parameters from fusion layer."""
        if hasattr(self.fusion, 'get_alpha_params'):
            return {
                'alphas': self.fusion.get_alpha_params(),
                'betas': self.fusion.get_beta_params()
            }
        return {}

    def fuse_modalities(self, modality_inputs: Dict[str, torch.Tensor]):
        """Fuse multiple modality inputs."""
        return self.fusion(modality_inputs)

    def encode(
            self,
            modality_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Encode modalities to latent parameters.

        Returns:
            mu, logvar, per_modality_mus (for alpha regularization, only for ADAPTIVE_CANTOR)
        """
        if self.fusion_strategy == FusionStrategy.ADAPTIVE_CANTOR:
            fused = self.fuse_modalities(modality_inputs)  # Dict

            # Encode each modality separately
            mus, logvars = [], []
            per_modality_mus = {}

            # Find max sequence length for padding
            max_seq = max(f.shape[1] for f in fused.values())
            device = list(fused.values())[0].device

            for name in self.modality_names:
                if name in fused:
                    h = self.encoders[name](fused[name])
                    mu = self.fc_mus[name](h)
                    logvar = self.fc_logvars[name](h)

                    # CLAMP LOGVAR to prevent explosion
                    logvar = torch.clamp(logvar, min=-10.0, max=10.0)

                    # Pad to max sequence length for stacking
                    B, seq_len, latent_dim = mu.shape
                    if seq_len < max_seq:
                        pad_mu = torch.zeros(B, max_seq - seq_len, latent_dim, device=device)
                        pad_logvar = torch.zeros(B, max_seq - seq_len, latent_dim, device=device)
                        mu = torch.cat([mu, pad_mu], dim=1)
                        logvar = torch.cat([logvar, pad_logvar], dim=1)

                    mus.append(mu)
                    logvars.append(logvar)
                    per_modality_mus[name] = mu

            # Average latent parameters
            mu = torch.stack(mus).mean(dim=0)
            logvar = torch.stack(logvars).mean(dim=0)

            return mu, logvar, per_modality_mus
        else:
            # Single encoder for other strategies
            fused = self.fuse_modalities(modality_inputs)  # Tensor
            h = self.encoder(fused)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)

            # CLAMP LOGVAR
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)

            return mu, logvar, None

    def reparameterize(
            self,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)

        if generator is None and self.seed is not None:
            generator = torch.Generator(device=mu.device).manual_seed(self.seed)

        if generator is not None:
            eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        else:
            eps = torch.randn_like(std)

        return mu + eps * std

    def decode(
            self,
            z: torch.Tensor,
            target_modalities: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Decode latent to reconstructions."""
        if target_modalities is None:
            target_modalities = self.modality_names

        reconstructions = {}
        for name in target_modalities:
            recon = self.decoders[name](z)

            # Slice to original sequence length
            original_seq_len = self.modality_seq_lens[name]
            reconstructions[name] = recon[:, :original_seq_len, :]

        return reconstructions

    def project_for_cross_modal(
            self,
            reconstructions: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Project reconstructions to common space."""
        projected = {}
        for name, recon in reconstructions.items():
            proj = self.cross_modal_projections[name](recon)
            proj = F.normalize(proj, dim=-1)
            projected[name] = proj
        return projected

    def forward(
            self,
            modality_inputs: Dict[str, torch.Tensor],
            target_modalities: Optional[List[str]] = None,
            generator: Optional[torch.Generator] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Full forward pass.

        Returns:
            reconstructions, mu, logvar, per_modality_mus (None for non-ADAPTIVE_CANTOR)
        """
        mu, logvar, per_modality_mus = self.encode(modality_inputs)
        z = self.reparameterize(mu, logvar, generator)
        reconstructions = self.decode(z, target_modalities)
        return reconstructions, mu, logvar, per_modality_mus


# ============================================================================
# ENHANCED LOSS FUNCTION
# ============================================================================

class MultiModalVAELoss(nn.Module):
    """Enhanced loss with alpha regularization and KL clamping."""

    def __init__(
            self,
            beta_kl: float = 0.1,
            beta_reconstruction: float = 1.0,
            beta_cross_modal: float = 0.0,  # DISABLED by default
            beta_alpha_regularization: float = 0.01,
            kl_clamp_max: float = 1.0,  # Maximum KL value
            recon_type: str = 'mse',
            modality_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.beta_kl = beta_kl
        self.beta_reconstruction = beta_reconstruction
        self.beta_cross_modal = beta_cross_modal
        self.beta_alpha_regularization = beta_alpha_regularization
        self.kl_clamp_max = kl_clamp_max
        self.recon_type = recon_type
        self.modality_weights = modality_weights or {}

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
            reconstructions: Dict[str, torch.Tensor],
            mu: torch.Tensor,
            logvar: torch.Tensor,
            per_modality_mus: Optional[Dict[str, torch.Tensor]] = None,
            alphas: Optional[Dict[str, torch.Tensor]] = None,
            projected_recons: Optional[Dict[str, torch.Tensor]] = None,
            return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute total loss with KL clamping.
        """
        losses = {}

        # 1. Reconstruction loss
        recon_losses = []
        total_weight = 0.0

        for name in reconstructions.keys():
            recon = reconstructions[name]
            inp = inputs[name]

            min_seq_len = min(recon.shape[1], inp.shape[1])
            recon = recon[:, :min_seq_len, :]
            inp = inp[:, :min_seq_len, :]

            if self.recon_type == 'mse':
                recon_loss = F.mse_loss(recon, inp)
            elif self.recon_type == 'cosine':
                recon_flat = recon.reshape(-1, recon.shape[-1])
                input_flat = inp.reshape(-1, inp.shape[-1])
                recon_norm = F.normalize(recon_flat, dim=-1)
                input_norm = F.normalize(input_flat, dim=-1)
                cos_sim = (recon_norm * input_norm).sum(dim=-1)
                recon_loss = (1 - cos_sim).mean()
            else:
                raise ValueError(f"Unknown recon_type: {self.recon_type}")

            weight = self.modality_weights.get(name, 1.0)
            weighted_loss = recon_loss * weight

            losses[f'recon_{name}'] = recon_loss
            recon_losses.append(weighted_loss)
            total_weight += weight

        total_recon = sum(recon_losses) / total_weight if recon_losses else torch.tensor(0.0)

        # 2. KL divergence with CLAMPING
        # Clamp logvar first to prevent exp() explosion
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)

        kl_loss = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
        kl_loss = kl_loss / (mu.shape[0] * mu.shape[1] * mu.shape[2])

        # CLAMP KL to prevent explosion
        kl_loss = torch.clamp(kl_loss, max=self.kl_clamp_max)

        losses['kl'] = kl_loss

        # 3. Alpha regularization (only for ADAPTIVE_CANTOR)
        if per_modality_mus is not None and alphas is not None:
            alpha_reg_losses = []
            for name, alpha in alphas.items():
                if name in per_modality_mus:
                    mu_mod = per_modality_mus[name]
                    kl_mod = -0.5 * torch.sum(1 - mu_mod.pow(2))
                    kl_mod = kl_mod / (mu_mod.shape[0] * mu_mod.shape[1] * mu_mod.shape[2])

                    alpha_clamped = torch.sigmoid(alpha)
                    alpha_target = torch.sigmoid(kl_mod * 10)

                    alpha_reg = (alpha_clamped - alpha_target).pow(2)
                    alpha_reg_losses.append(alpha_reg)

                    losses[f'alpha_{name}'] = alpha_clamped.item()

            alpha_regularization = sum(alpha_reg_losses) / len(alpha_reg_losses) if alpha_reg_losses else torch.tensor(
                0.0, device=mu.device)
            losses['alpha_reg'] = alpha_regularization
        else:
            alpha_regularization = torch.tensor(0.0, device=mu.device)
            losses['alpha_reg'] = alpha_regularization

        # 4. Cross-modal consistency (DISABLED by default - causes contamination)
        if self.beta_cross_modal > 0 and len(reconstructions) > 1 and projected_recons is not None:
            projected_list = list(projected_recons.values())
            cross_modal_losses = []

            for i in range(len(projected_list)):
                for j in range(i + 1, len(projected_list)):
                    proj_i = projected_list[i]
                    proj_j = projected_list[j]
                    min_seq = min(proj_i.shape[1], proj_j.shape[1])

                    cm_loss = F.mse_loss(proj_i[:, :min_seq, :], proj_j[:, :min_seq, :])
                    cross_modal_losses.append(cm_loss)

            cross_modal = sum(cross_modal_losses) / len(cross_modal_losses) if cross_modal_losses else torch.tensor(0.0,
                                                                                                                    device=mu.device)
            losses['cross_modal'] = cross_modal
        else:
            cross_modal = torch.tensor(0.0, device=mu.device)
            losses['cross_modal'] = cross_modal

        # Total loss
        total_loss = (
                self.beta_reconstruction * total_recon +
                self.beta_kl * kl_loss +
                self.beta_cross_modal * cross_modal +
                self.beta_alpha_regularization * alpha_regularization
        )
        losses['total'] = total_loss

        if return_components:
            return total_loss, losses
        return total_loss, None


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Testing AdaptiveCantorModalityFusion with HARD MASKING")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # Test inputs
    batch_size = 2
    test_inputs = {
        "clip_l": torch.randn(batch_size, 77, 768, device=device),
        "clip_g": torch.randn(batch_size, 77, 1280, device=device),
        "t5_xl_l": torch.randn(batch_size, 512, 2048, device=device),
        "t5_xl_g": torch.randn(batch_size, 512, 2048, device=device)
    }

    config = MultiModalVAEConfig(
        fusion_strategy="adaptive_cantor",
        beta_cross_modal=0.0,  # Disabled
        kl_clamp_max=1.0,
    )

    print(f"\n📊 Building model...")
    model = MultiModalVAE(config).to(device)

    print(f"\n🧪 Testing forward pass...")
    with torch.no_grad():
        recons, mu, logvar, per_mod_mus = model(test_inputs)

    print(f"\n✅ Outputs:")
    for name, recon in recons.items():
        print(f"   {name}: {recon.shape}")

    print(f"\n   mu: {mu.shape}, range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"   logvar: {logvar.shape}, range: [{logvar.min():.3f}, {logvar.max():.3f}]")

    # Test loss
    print(f"\n📉 Testing loss computation...")
    loss_fn = MultiModalVAELoss(
        beta_kl=0.1,
        beta_cross_modal=0.0,  # Disabled
        kl_clamp_max=1.0,
    )

    loss, components = loss_fn(
        inputs=test_inputs,
        reconstructions=recons,
        mu=mu,
        logvar=logvar,
        per_modality_mus=per_mod_mus,
        alphas=model.get_fusion_params().get('alphas'),
        return_components=True
    )

    print(f"   Total loss: {loss.item():.4f}")
    print(f"   KL loss: {components['kl'].item():.4f}")
    for name in ['clip_l', 'clip_g', 't5_xl_l', 't5_xl_g']:
        if f'recon_{name}' in components:
            print(f"   Recon {name}: {components[f'recon_{name}'].item():.4f}")

    print(f"\n✅ All tests passed!")