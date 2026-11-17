"""
geofractal.router.prefab.agatha_v2.config
=========================================

Configuration for AgathaFlow, a flow-matching latent denoiser whose velocity is
read out of a damped, banded second-order integrator driven by a collective of
per-token force towers.

Two ready-made configurations:
    AgathaConfig()        the default: a 32 x 32 x 32 VAE latent (256 px with an
                          8x, 32-channel VAE), patch 2 -> 256 tokens of width 128,
                          model width 384, 6 heads, a T5-base text sequence.
    smoke_config()        a CPU-sized copy for tests (latent 4 x 8 x 8, width 64).

`arch` selects the model built from a configuration:
    "agatha"  AgathaFlow: trunk, force towers, banded oscillator (the default).
    "plain"   PlainFlow, the control: the same patch embedding, time embedding,
              text projection and trunk blocks, no towers and no oscillator; the
              velocity is a zero-initialized linear head on LayerNorm(h). The tower,
              band and oscillator fields are ignored. plain_counterpart() builds the
              plain configuration of an Agatha configuration at a given trunk depth.

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

GEOMETRY_TOWER_KINDS = ("cantor", "beatrix", "simplex", "helix", "plain")
CONV_TOWER_KINDS = ("wide_resnet", "frequency")
ARCHITECTURES = ("agatha", "plain")


@dataclass
class AgathaConfig:
    # ---- which model is built ("agatha" or the "plain" control)
    arch: str = "agatha"

    # ---- latent and tokens
    latent_channels: int = 32
    latent_height: int = 32
    latent_width: int = 32
    patch_size: int = 2

    # ---- model width
    dim: int = 384
    num_heads: int = 6
    mlp_ratio: float = 4.0
    time_freq_dim: int = 256
    time_scale: float = 1000.0      # t in [0, 1] is multiplied by this before the sinusoid

    # ---- text condition (a frozen encoder's token sequence, computed offline)
    text_dim: int = 768
    null_text_len: int = 1          # tokens of the empty caption (T5: the end token only)
    cond_dropout: float = 0.1       # probability of the null sequence per training sample

    # ---- trunk and towers
    trunk_depth: int = 1
    geometry_towers: Tuple[str, ...] = ("cantor", "beatrix", "simplex", "helix")
    tower_depth: int = 2
    conv_towers: Tuple[str, ...] = ("wide_resnet", "frequency")
    conv_depth: int = 2
    rope_theta: float = 10000.0

    # ---- noise bands (on the shift-warped sigma = 1 - t)
    band_edges: Tuple[float, float] = (0.35, 0.75)
    band_half_width: float = 0.06   # cosine crossfade half-width around each edge
    shift: float = 1.0

    # ---- oscillator
    num_osc_steps: int = 4
    damping_step_min: float = 0.05  # lower bound of h * beta, h = 1 / num_osc_steps
    damping_step_max: float = 0.5   # upper bound of h * beta
    omega_step_max: float = 1.0     # upper bound of h * omega
    kappa_init: float = 0.0         # raw kappa at initialization; kappa = softplus(raw) = log 2 ~ 0.693

    # ---- optional signed dispatch (experimental, off by default)
    signed_dispatch: bool = False
    dispatch_dim: int = 64
    dispatch_tau: float = 1.0

    # ---- training-time choices read by loss()
    t_sampling: str = "uniform"     # "uniform" or "logit_normal"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0

    # ---- sampling default
    sample_steps: int = 50

    # ---- metadata only: the frozen encoders this model was trained against
    text_encoder_name: str = "t5-base"
    text_encoder_revision: Optional[str] = None
    vae_name: str = "flux2-vae"
    vae_revision: Optional[str] = None

    def __post_init__(self) -> None:
        self.geometry_towers = tuple(self.geometry_towers)
        self.conv_towers = tuple(self.conv_towers)
        self.band_edges = tuple(float(e) for e in self.band_edges)
        self.validate()

    # ------------------------------------------------------------------ derived sizes
    @property
    def grid_height(self) -> int:
        return self.latent_height // self.patch_size

    @property
    def grid_width(self) -> int:
        return self.latent_width // self.patch_size

    @property
    def num_tokens(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def token_dim(self) -> int:
        return self.latent_channels * self.patch_size * self.patch_size

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def num_towers(self) -> int:
        return len(self.geometry_towers) + len(self.conv_towers)

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return (self.latent_channels, self.latent_height, self.latent_width)

    @property
    def osc_step(self) -> float:
        return 1.0 / self.num_osc_steps

    @property
    def beta_min(self) -> float:
        return self.damping_step_min / self.osc_step

    @property
    def beta_max(self) -> float:
        return self.damping_step_max / self.osc_step

    @property
    def omega_max(self) -> float:
        return self.omega_step_max / self.osc_step

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        if self.arch not in ARCHITECTURES:
            raise ValueError(f"unknown arch {self.arch!r}; known: {ARCHITECTURES}")
        if self.trunk_depth < 0:
            raise ValueError("trunk_depth must be >= 0")
        if self.latent_height % self.patch_size or self.latent_width % self.patch_size:
            raise ValueError("latent height and width must be divisible by patch_size")
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if self.head_dim % 4:
            raise ValueError("head width (dim / num_heads) must be divisible by 4 (2-D rotary in the trunk)")
        for k in self.geometry_towers:
            if k not in GEOMETRY_TOWER_KINDS:
                raise ValueError(f"unknown geometry tower kind {k!r}; known: {GEOMETRY_TOWER_KINDS}")
        for k in self.conv_towers:
            if k not in CONV_TOWER_KINDS:
                raise ValueError(f"unknown conv tower kind {k!r}; known: {CONV_TOWER_KINDS}")
        if self.arch == "agatha" and self.num_towers < 1:
            raise ValueError("at least one tower is required")
        lo, hi = self.band_edges
        if not (0.0 < lo < hi < 1.0):
            raise ValueError("band_edges must satisfy 0 < lo < hi < 1")
        if lo + self.band_half_width >= hi - self.band_half_width:
            raise ValueError("crossfade windows of the two edges overlap")
        # Stability of the semi-implicit step with p = (h omega)^2, q = 2 h beta:
        # both roots inside the unit circle iff 0 < q < 2, p > 0 and p + 2 q < 4.
        q_lo, q_hi = 2 * self.damping_step_min, 2 * self.damping_step_max
        p_hi = self.omega_step_max ** 2
        if not (0.0 < q_lo < q_hi < 2.0):
            raise ValueError("need 0 < damping_step_min < damping_step_max < 1")
        if not (p_hi > 0.0 and p_hi + 2 * q_hi < 4.0):
            raise ValueError("omega_step_max^2 + 4 * damping_step_max must stay below 4 (stability)")
        if self.num_osc_steps < 1:
            raise ValueError("num_osc_steps must be >= 1")
        if self.shift <= 0:
            raise ValueError("shift must be positive")
        if self.t_sampling not in ("uniform", "logit_normal"):
            raise ValueError("t_sampling must be 'uniform' or 'logit_normal'")
        if self.null_text_len < 1:
            raise ValueError("null_text_len must be >= 1")

    # ------------------------------------------------------------------ (de)serialization
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["geometry_towers"] = list(self.geometry_towers)
        d["conv_towers"] = list(self.conv_towers)
        d["band_edges"] = list(self.band_edges)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgathaConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)


def plain_counterpart(config: AgathaConfig, trunk_depth: int) -> AgathaConfig:
    """The plain control of `config`: every shared field kept, arch "plain", the given
    trunk depth, and no towers listed (the plain model has none)."""
    d = config.to_dict()
    d.update(arch="plain", trunk_depth=int(trunk_depth), geometry_towers=[], conv_towers=[])
    return AgathaConfig.from_dict(d)


def smoke_config(**overrides: Any) -> AgathaConfig:
    """A CPU-sized configuration: latent 4 x 8 x 8, patch 2 -> 16 tokens of width 16."""
    base = dict(
        latent_channels=4,
        latent_height=8,
        latent_width=8,
        patch_size=2,
        dim=64,
        num_heads=1,
        time_freq_dim=64,
        text_dim=32,
        trunk_depth=1,
        tower_depth=1,
        conv_depth=1,
        num_osc_steps=4,
        dispatch_dim=16,
        sample_steps=4,
    )
    base.update(overrides)
    return AgathaConfig(**base)
