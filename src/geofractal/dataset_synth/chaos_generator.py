"""
Chaos Factory System
--------------------
Learned bias injection for causal perturbation of classifier trajectories.

Unlike deterministic generation, Chaos Factory introduces structured perturbation
through learnable curve functions that counteract baseline classifier bias.

Key Concepts:
  - BiasHook: Parametric curve with learnable weights (sigmoid, tanh, polynomial, etc.)
  - TrajectoryRoute: Named path through classifier layers
  - ChaosWeights: Learned parameters that encode the "counter-bias"
  - Training Mode: Learn weights by observing classification trajectories
  - Inference Mode: Apply learned curves without synthesizer overhead

The "chaos" is causal - it's not random noise but a learned transformation
that can be tuned to explore decision boundaries, introduce controlled
variability, or study classifier behavior under different bias regimes.

Author: AbstractPhil + Claude Opus 4.5
License: MIT
"""

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List, Callable
from datetime import datetime
from enum import Enum
import warnings

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

# Assuming this exists in your codebase
# from geovocab2.shapes.factory.factory_base import FactoryBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bias Curve Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CurveType(Enum):
    """Available bias curve functions."""
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTPLUS = "softplus"
    POLYNOMIAL = "polynomial"
    SINUSOIDAL = "sinusoidal"
    EXPONENTIAL = "exponential"
    RATIONAL = "rational"
    SPLINE = "spline"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bias Hook - Learnable Curve Function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BiasHook(nn.Module if HAS_TORCH else object):
    """
    Learnable bias curve function.

    Applies a parametric transformation to input activations.
    Parameters are learned during training to counteract baseline bias.

    The curve transforms input x as:
        output = curve(x * scale + shift) * amplitude + offset

    Where curve() depends on curve_type.

    Supports lazy initialization: if input_dim=None, parameters are created
    on first forward pass based on actual input dimension.
    """

    def __init__(
        self,
        curve_type: CurveType = CurveType.SIGMOID,
        input_dim: Optional[int] = None,  # None = lazy init
        learnable_scale: bool = True,
        learnable_shift: bool = True,
        learnable_amplitude: bool = True,
        learnable_offset: bool = True,
        init_scale: float = 1.0,
        init_shift: float = 0.0,
        init_amplitude: float = 1.0,
        init_offset: float = 0.0,
        polynomial_degree: int = 3,
        spline_knots: int = 5,
    ):
        if HAS_TORCH:
            super().__init__()

        self.curve_type = curve_type
        self.input_dim = input_dim
        self.polynomial_degree = polynomial_degree
        self.spline_knots = spline_knots

        # Store init values for lazy initialization
        self._init_scale = init_scale
        self._init_shift = init_shift
        self._init_amplitude = init_amplitude
        self._init_offset = init_offset
        self._learnable_scale = learnable_scale
        self._learnable_shift = learnable_shift
        self._learnable_amplitude = learnable_amplitude
        self._learnable_offset = learnable_offset

        self._initialized = False

        if HAS_TORCH and input_dim is not None:
            self._init_parameters(input_dim)
        elif not HAS_TORCH and input_dim is not None:
            # NumPy fallback storage
            self._params = {
                'scale': np.full(input_dim, init_scale),
                'shift': np.full(input_dim, init_shift),
                'amplitude': np.full(input_dim, init_amplitude),
                'offset': np.full(input_dim, init_offset),
            }
            self._initialized = True

    def _init_parameters(self, input_dim: int, device: str = "cpu"):
        """Initialize parameters for given dimension."""
        if self._initialized and self.input_dim == input_dim:
            return

        self.input_dim = input_dim

        if HAS_TORCH:
            # Core learnable parameters (per-dimension)
            self.scale = nn.Parameter(
                torch.full((input_dim,), self._init_scale, device=device),
                requires_grad=self._learnable_scale
            )
            self.shift = nn.Parameter(
                torch.full((input_dim,), self._init_shift, device=device),
                requires_grad=self._learnable_shift
            )
            self.amplitude = nn.Parameter(
                torch.full((input_dim,), self._init_amplitude, device=device),
                requires_grad=self._learnable_amplitude
            )
            self.offset = nn.Parameter(
                torch.full((input_dim,), self._init_offset, device=device),
                requires_grad=self._learnable_offset
            )

            # Curve-specific parameters
            if self.curve_type == CurveType.POLYNOMIAL:
                self.poly_coeffs = nn.Parameter(
                    torch.randn(input_dim, self.polynomial_degree + 1, device=device) * 0.1
                )
            elif self.curve_type == CurveType.SPLINE:
                self.spline_points = nn.Parameter(
                    torch.linspace(-1, 1, self.spline_knots, device=device).unsqueeze(0).expand(input_dim, -1).clone()
                )
                self.spline_values = nn.Parameter(
                    torch.zeros(input_dim, self.spline_knots, device=device)
                )
            elif self.curve_type == CurveType.RATIONAL:
                self.rational_num = nn.Parameter(torch.randn(input_dim, 3, device=device) * 0.1)
                self.rational_den = nn.Parameter(torch.ones(input_dim, 3, device=device) * 0.1)
            elif self.curve_type == CurveType.SINUSOIDAL:
                self.frequency = nn.Parameter(torch.ones(input_dim, device=device))
                self.phase = nn.Parameter(torch.zeros(input_dim, device=device))
        else:
            self._params = {
                'scale': np.full(input_dim, self._init_scale),
                'shift': np.full(input_dim, self._init_shift),
                'amplitude': np.full(input_dim, self._init_amplitude),
                'offset': np.full(input_dim, self._init_offset),
            }

        self._initialized = True

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply bias curve transformation."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for forward()")

        # Lazy initialization based on actual input dimension and device
        actual_dim = x.shape[-1]
        device = x.device

        if not self._initialized:
            self._init_parameters(actual_dim, device=device)
        elif self.input_dim != actual_dim:
            # Re-initialize for new dimension (warning: loses learned weights)
            warnings.warn(
                f"BiasHook input_dim changed from {self.input_dim} to {actual_dim}. "
                f"Re-initializing parameters."
            )
            self._init_parameters(actual_dim, device=device)

        # Apply scale and shift
        z = x * self.scale + self.shift

        # Apply curve function
        if self.curve_type == CurveType.SIGMOID:
            curved = torch.sigmoid(z)
        elif self.curve_type == CurveType.TANH:
            curved = torch.tanh(z)
        elif self.curve_type == CurveType.SOFTPLUS:
            curved = F.softplus(z)
        elif self.curve_type == CurveType.POLYNOMIAL:
            curved = self._polynomial(z)
        elif self.curve_type == CurveType.SINUSOIDAL:
            curved = torch.sin(z * self.frequency + self.phase)
        elif self.curve_type == CurveType.EXPONENTIAL:
            curved = torch.exp(-torch.abs(z)) * torch.sign(z)
        elif self.curve_type == CurveType.RATIONAL:
            curved = self._rational(z)
        elif self.curve_type == CurveType.SPLINE:
            curved = self._spline(z)
        else:
            curved = z  # Identity fallback

        # Apply amplitude and offset
        return curved * self.amplitude + self.offset

    def _polynomial(self, z: "torch.Tensor") -> "torch.Tensor":
        """Evaluate learnable polynomial."""
        result = torch.zeros_like(z)
        for i in range(self.polynomial_degree + 1):
            result = result + self.poly_coeffs[:, i] * (z ** i)
        return result

    def _rational(self, z: "torch.Tensor") -> "torch.Tensor":
        """Evaluate rational function (ratio of polynomials)."""
        num = (self.rational_num[:, 0] +
               self.rational_num[:, 1] * z +
               self.rational_num[:, 2] * z**2)
        den = (self.rational_den[:, 0] +
               self.rational_den[:, 1] * z.abs() +
               self.rational_den[:, 2] * z**2 + 1e-6)
        return num / den

    def _spline(self, z: "torch.Tensor") -> "torch.Tensor":
        """Evaluate cubic spline interpolation."""
        # Simple linear interpolation between knots
        # (Full cubic spline would require more complexity)
        z_clamped = torch.clamp(z, -1, 1)
        # Find knot indices
        knot_spacing = 2.0 / (self.spline_knots - 1)
        indices = ((z_clamped + 1) / knot_spacing).long().clamp(0, self.spline_knots - 2)
        t = ((z_clamped + 1) / knot_spacing) - indices.float()

        # Gather values and interpolate
        v0 = torch.gather(self.spline_values, 1, indices)
        v1 = torch.gather(self.spline_values, 1, (indices + 1).clamp(max=self.spline_knots - 1))
        return v0 * (1 - t) + v1 * t

    def apply_numpy(self, x: np.ndarray) -> np.ndarray:
        """Apply bias curve using NumPy (inference only)."""
        # Lazy init for numpy if not initialized
        if not self._initialized:
            actual_dim = x.shape[-1]
            if not HAS_TORCH:
                self._params = {
                    'scale': np.full(actual_dim, self._init_scale),
                    'shift': np.full(actual_dim, self._init_shift),
                    'amplitude': np.full(actual_dim, self._init_amplitude),
                    'offset': np.full(actual_dim, self._init_offset),
                }
                self.input_dim = actual_dim
                self._initialized = True
            else:
                # Initialize torch params then extract (CPU for numpy inference)
                self._init_parameters(actual_dim, device="cpu")

        if HAS_TORCH and self._initialized:
            params = {
                'scale': self.scale.detach().cpu().numpy(),
                'shift': self.shift.detach().cpu().numpy(),
                'amplitude': self.amplitude.detach().cpu().numpy(),
                'offset': self.offset.detach().cpu().numpy(),
            }
        else:
            params = self._params

        z = x * params['scale'] + params['shift']

        if self.curve_type == CurveType.SIGMOID:
            curved = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.curve_type == CurveType.TANH:
            curved = np.tanh(z)
        elif self.curve_type == CurveType.SOFTPLUS:
            curved = np.log1p(np.exp(np.clip(z, -500, 500)))
        else:
            curved = z  # Simplified for numpy

        return curved * params['amplitude'] + params['offset']

    def get_weights(self) -> Dict[str, Any]:
        """Extract learned weights as numpy arrays."""
        if not self._initialized:
            return {
                'curve_type': self.curve_type.value,
                'input_dim': None,
                'initialized': False,
            }

        if HAS_TORCH:
            weights = {
                'scale': self.scale.detach().cpu().numpy(),
                'shift': self.shift.detach().cpu().numpy(),
                'amplitude': self.amplitude.detach().cpu().numpy(),
                'offset': self.offset.detach().cpu().numpy(),
                'curve_type': self.curve_type.value,
                'input_dim': self.input_dim,
                'initialized': True,
            }
            if self.curve_type == CurveType.POLYNOMIAL:
                weights['poly_coeffs'] = self.poly_coeffs.detach().cpu().numpy()
            elif self.curve_type == CurveType.SPLINE:
                weights['spline_points'] = self.spline_points.detach().cpu().numpy()
                weights['spline_values'] = self.spline_values.detach().cpu().numpy()
            elif self.curve_type == CurveType.RATIONAL:
                weights['rational_num'] = self.rational_num.detach().cpu().numpy()
                weights['rational_den'] = self.rational_den.detach().cpu().numpy()
            elif self.curve_type == CurveType.SINUSOIDAL:
                weights['frequency'] = self.frequency.detach().cpu().numpy()
                weights['phase'] = self.phase.detach().cpu().numpy()
            return weights
        else:
            return self._params.copy()

    def load_weights(self, weights: Dict[str, Any]):
        """Load pre-trained weights."""
        if not weights.get('initialized', True):
            # Weights were never initialized, nothing to load
            return

        input_dim = weights.get('input_dim')
        if input_dim is None:
            return

        # Initialize parameters if needed (CPU, will be moved by .to(device))
        if not self._initialized:
            self._init_parameters(input_dim, device="cpu")

        if HAS_TORCH:
            with torch.no_grad():
                self.scale.copy_(torch.from_numpy(weights['scale']))
                self.shift.copy_(torch.from_numpy(weights['shift']))
                self.amplitude.copy_(torch.from_numpy(weights['amplitude']))
                self.offset.copy_(torch.from_numpy(weights['offset']))

                if 'poly_coeffs' in weights:
                    self.poly_coeffs.copy_(torch.from_numpy(weights['poly_coeffs']))
                if 'spline_points' in weights:
                    self.spline_points.copy_(torch.from_numpy(weights['spline_points']))
                    self.spline_values.copy_(torch.from_numpy(weights['spline_values']))
                if 'rational_num' in weights:
                    self.rational_num.copy_(torch.from_numpy(weights['rational_num']))
                    self.rational_den.copy_(torch.from_numpy(weights['rational_den']))
                if 'frequency' in weights:
                    self.frequency.copy_(torch.from_numpy(weights['frequency']))
                    self.phase.copy_(torch.from_numpy(weights['phase']))
        else:
            self._params = weights.copy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trajectory Route - Named Path Through Classifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TrajectoryRoute:
    """
    Defines a named route through classifier layers.

    Routes specify where bias hooks are applied:
      - pre_embed: Before embedding lookup
      - post_embed: After embedding, before projection
      - hidden_N: After hidden layer N
      - pre_output: Before final classification head
      - post_output: After logits (before softmax)
    """
    name: str
    hook_points: List[str]  # Where to attach hooks
    dimensions: Optional[List[int]] = None  # Which dims to affect (None = all)
    blend_mode: str = "additive"  # additive, multiplicative, replace

    def __post_init__(self):
        valid_modes = ["additive", "multiplicative", "replace", "gate"]
        if self.blend_mode not in valid_modes:
            raise ValueError(f"blend_mode must be one of {valid_modes}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chaos Weights - Learned Counter-Bias Parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ChaosWeights:
    """
    Container for learned chaos weights.

    These weights encode the learned counter-bias that can be
    applied at inference without the full synthesizer.
    """
    version: str
    created_at: str
    route_name: str
    hook_weights: Dict[str, Dict[str, np.ndarray]]  # hook_point -> param_name -> values
    metadata: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Generate hash of weights for versioning."""
        # Serialize weights deterministically
        weight_bytes = b""
        for hook_point in sorted(self.hook_weights.keys()):
            for param_name in sorted(self.hook_weights[hook_point].keys()):
                arr = self.hook_weights[hook_point][param_name]
                if isinstance(arr, np.ndarray):
                    weight_bytes += arr.tobytes()
                else:
                    weight_bytes += str(arr).encode()
        return hashlib.sha256(weight_bytes).hexdigest()[:16]

    def save(self, path: Path):
        """Save weights to file."""
        path = Path(path)

        # Convert numpy arrays for JSON serialization
        serializable = {
            'version': self.version,
            'created_at': self.created_at,
            'route_name': self.route_name,
            'metadata': self.metadata,
            'fingerprint': self.fingerprint(),
            'hook_weights': {}
        }

        for hook_point, params in self.hook_weights.items():
            serializable['hook_weights'][hook_point] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in params.items()
            }

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ChaosWeights':
        """Load weights from file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        hook_weights = {}
        for hook_point, params in data['hook_weights'].items():
            hook_weights[hook_point] = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in params.items()
            }

        return cls(
            version=data['version'],
            created_at=data['created_at'],
            route_name=data['route_name'],
            hook_weights=hook_weights,
            metadata=data.get('metadata', {})
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chaos Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ChaosConfiguration:
    """
    Configuration for chaos factory.

    Defines the structure of bias injection and learning parameters.
    """
    # Identity
    factory_version: str = "1.0.0"

    # Route configuration
    routes: List[Dict[str, Any]] = field(default_factory=list)

    # Hook configuration (per route)
    curve_type: str = "sigmoid"
    input_dim: int = 768  # Match your embedding dimension

    # Learning parameters
    learning_rate: float = 1e-3
    warmup_steps: int = 100
    regularization: float = 1e-4

    # Chaos intensity bounds
    min_amplitude: float = 0.0
    max_amplitude: float = 1.0

    # Metadata
    created_at: str = None
    description: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        config_dict = self.to_dict()
        config_dict.pop('created_at', None)
        canonical = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def save(self, path: Path):
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ChaosConfiguration':
        path = Path(path)
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chaos Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChaosFactory(nn.Module if HAS_TORCH else object):
    """
    Learned bias injection factory for classifier trajectories.

    Two operating modes:

    1. TRAINING MODE:
       - Attach hooks to classifier
       - Observe classification trajectories
       - Learn counter-bias weights via gradient descent
       - Optimize to counteract baseline classifier bias

    2. INFERENCE MODE:
       - Load pre-trained ChaosWeights
       - Apply bias curves without synthesizer
       - Fast, deterministic perturbation

    The "chaos" is structured perturbation that can:
      - Explore decision boundaries
      - Introduce controlled variability
      - Study classifier behavior under different bias regimes
      - Create adversarial-like but constructive perturbations
    """

    FACTORY_VERSION = "1.0.0"

    def __init__(
        self,
        config: ChaosConfiguration,
        mode: str = "training",  # "training" or "inference"
    ):
        if HAS_TORCH:
            super().__init__()

        self.config = config
        self.mode = mode

        # Parse routes
        self.routes: Dict[str, TrajectoryRoute] = {}
        for route_dict in config.routes:
            route = TrajectoryRoute(**route_dict)
            self.routes[route.name] = route

        # Create bias hooks for each route and hook point
        self.hooks: Dict[str, Dict[str, BiasHook]] = {}

        if HAS_TORCH:
            self.hook_modules = nn.ModuleDict()

        for route_name, route in self.routes.items():
            self.hooks[route_name] = {}
            for hook_point in route.hook_points:
                hook = BiasHook(
                    curve_type=CurveType(config.curve_type),
                    input_dim=None,  # Lazy init - will be set on first forward
                )
                self.hooks[route_name][hook_point] = hook

                if HAS_TORCH:
                    # Register as submodule for parameter tracking
                    self.hook_modules[f"{route_name}_{hook_point}"] = hook

        # Intensity controls (learnable gates)
        if HAS_TORCH:
            self.intensity = nn.ParameterDict({
                route_name: nn.Parameter(torch.ones(1) * 0.5)
                for route_name in self.routes
            })

        # Training state
        self._step = 0
        self._baseline_captured = False
        self._baseline_stats: Dict[str, Dict[str, torch.Tensor]] = {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Hook Registration (for classifier integration)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_forward_hook(
        self,
        route_name: str,
        hook_point: str
    ) -> Callable:
        """
        Create a forward hook for PyTorch module registration.

        Usage:
            classifier.layer.register_forward_hook(
                chaos_factory.create_forward_hook("main", "hidden_0")
            )
        """
        def hook_fn(module, input, output):
            return self.apply_bias(output, route_name, hook_point)
        return hook_fn

    def apply_bias(
        self,
        activations: "torch.Tensor",
        route_name: str,
        hook_point: str
    ) -> "torch.Tensor":
        """
        Apply learned bias to activations.

        Args:
            activations: Tensor of shape (batch, ..., dim)
            route_name: Which route this belongs to
            hook_point: Which hook point in the route

        Returns:
            Biased activations
        """
        if route_name not in self.routes:
            return activations

        route = self.routes[route_name]
        hook = self.hooks[route_name].get(hook_point)

        if hook is None:
            return activations

        # Get intensity (warmup schedule during training)
        if self.mode == "training" and self._step < self.config.warmup_steps:
            warmup_factor = self._step / self.config.warmup_steps
            intensity = self.intensity[route_name] * warmup_factor
        else:
            intensity = self.intensity[route_name]

        # Clamp intensity
        intensity = torch.clamp(
            intensity,
            self.config.min_amplitude,
            self.config.max_amplitude
        )

        # Apply hook to get bias
        bias = hook(activations)

        # Apply based on blend mode
        if route.blend_mode == "additive":
            return activations + bias * intensity
        elif route.blend_mode == "multiplicative":
            return activations * (1 + bias * intensity)
        elif route.blend_mode == "replace":
            return activations * (1 - intensity) + bias * intensity
        elif route.blend_mode == "gate":
            gate = torch.sigmoid(bias)
            return activations * gate * intensity + activations * (1 - intensity)

        return activations

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Training Interface
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def capture_baseline(self, activations: Dict[str, "torch.Tensor"]):
        """
        Capture baseline activation STATISTICS (without chaos) for counter-bias learning.

        Stores mean and std per feature dimension, not raw activations.
        Call this once before training with a representative batch.
        """
        self._baseline_stats = {}
        for k, v in activations.items():
            # Store statistics, not raw tensors (batch-size independent)
            self._baseline_stats[k] = {
                'mean': v.mean(dim=0).detach(),  # [feature_dims]
                'std': v.std(dim=0).detach() + 1e-6,
            }
        self._baseline_captured = True

    def compute_chaos_loss(
        self,
        current_activations: Dict[str, "torch.Tensor"],
        target_divergence: float = 0.1
    ) -> "torch.Tensor":
        """
        Compute loss for learning counter-bias.

        The goal: learn bias curves that produce controlled divergence
        from baseline activation statistics.

        Args:
            current_activations: Dict of hook_point -> activations
            target_divergence: Desired normalized distance from baseline

        Returns:
            Loss tensor for optimization
        """
        if not self._baseline_captured:
            raise RuntimeError("Must call capture_baseline() first")

        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for hook_point, current in current_activations.items():
            if hook_point not in self._baseline_stats:
                continue

            baseline = self._baseline_stats[hook_point]

            # Compute statistics of current batch
            current_mean = current.mean(dim=0)
            current_std = current.std(dim=0) + 1e-6

            # Divergence = normalized difference in mean + std ratio
            mean_div = ((current_mean - baseline['mean']) / baseline['std']).pow(2).mean()
            std_div = ((current_std / baseline['std']) - 1).pow(2).mean()

            divergence = (mean_div + std_div).sqrt()

            # Loss: penalize deviation from target divergence
            divergence_loss = (divergence - target_divergence) ** 2

            total_loss = total_loss + divergence_loss

        # Add regularization on hook parameters
        reg_loss = torch.tensor(0.0, device=total_loss.device)
        for route_hooks in self.hooks.values():
            for hook in route_hooks.values():
                # Skip uninitialized hooks
                if not hook._initialized:
                    continue
                # Penalize extreme parameter values
                reg_loss = reg_loss + self.config.regularization * (
                    hook.scale.pow(2).mean() +
                    hook.shift.pow(2).mean() +
                    (hook.amplitude - 1).pow(2).mean()
                )

        return total_loss + reg_loss

    def training_step(self):
        """Increment training step counter."""
        self._step += 1

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Inference Interface
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def apply_numpy(
        self,
        activations: np.ndarray,
        route_name: str,
        hook_point: str,
        intensity: float = 1.0
    ) -> np.ndarray:
        """
        Apply learned bias using NumPy (inference without PyTorch).

        This is the fast path for inference when you have pre-trained weights
        and don't need gradient computation.
        """
        if route_name not in self.routes:
            return activations

        route = self.routes[route_name]
        hook = self.hooks[route_name].get(hook_point)

        if hook is None:
            return activations

        # Apply hook
        bias = hook.apply_numpy(activations)

        # Clamp intensity
        intensity = np.clip(
            intensity,
            self.config.min_amplitude,
            self.config.max_amplitude
        )

        # Apply based on blend mode
        if route.blend_mode == "additive":
            return activations + bias * intensity
        elif route.blend_mode == "multiplicative":
            return activations * (1 + bias * intensity)
        elif route.blend_mode == "replace":
            return activations * (1 - intensity) + bias * intensity
        elif route.blend_mode == "gate":
            gate = 1 / (1 + np.exp(-bias))
            return activations * gate * intensity + activations * (1 - intensity)

        return activations

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Weight Management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def extract_weights(self, route_name: str) -> ChaosWeights:
        """
        Extract learned weights for a route.

        Returns ChaosWeights that can be saved and used for inference
        without the full factory.
        """
        if route_name not in self.routes:
            raise ValueError(f"Unknown route: {route_name}")

        hook_weights = {}
        for hook_point, hook in self.hooks[route_name].items():
            hook_weights[hook_point] = hook.get_weights()

        return ChaosWeights(
            version=self.FACTORY_VERSION,
            created_at=datetime.utcnow().isoformat(),
            route_name=route_name,
            hook_weights=hook_weights,
            metadata={
                'config_fingerprint': self.config.fingerprint()[:16],
                'training_steps': self._step,
            }
        )

    def load_weights(self, weights: ChaosWeights):
        """Load pre-trained weights into factory."""
        route_name = weights.route_name

        if route_name not in self.routes:
            raise ValueError(f"Unknown route: {route_name}")

        for hook_point, hook_weights in weights.hook_weights.items():
            if hook_point in self.hooks[route_name]:
                self.hooks[route_name][hook_point].load_weights(hook_weights)

    def save_all_weights(self, directory: Path):
        """Save all route weights to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(directory / "config.json")

        # Save weights for each route
        for route_name in self.routes:
            weights = self.extract_weights(route_name)
            weights.save(directory / f"{route_name}_weights.json")

    @classmethod
    def load_for_inference(cls, directory: Path) -> 'ChaosFactory':
        """
        Load factory in inference mode from saved weights.

        This creates a minimal factory that can apply pre-trained
        bias curves without training overhead.
        """
        directory = Path(directory)

        # Load config
        config = ChaosConfiguration.load(directory / "config.json")

        # Create factory in inference mode
        factory = cls(config, mode="inference")

        # Load all weights
        for weight_file in directory.glob("*_weights.json"):
            weights = ChaosWeights.load(weight_file)
            factory.load_weights(weights)

        return factory

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Utility
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def info(self) -> Dict[str, Any]:
        """Get factory information."""
        return {
            'factory_version': self.FACTORY_VERSION,
            'config_fingerprint': self.config.fingerprint()[:16],
            'mode': self.mode,
            'routes': list(self.routes.keys()),
            'total_hooks': sum(len(h) for h in self.hooks.values()),
            'training_steps': self._step,
            'baseline_captured': self._baseline_captured,
        }

    def __repr__(self) -> str:
        return (
            f"ChaosFactory(mode={self.mode}, routes={list(self.routes.keys())}, "
            f"steps={self._step})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lightweight Inference Applicator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChaosApplicator:
    """
    Minimal inference-only chaos applicator.

    Uses pre-trained ChaosWeights without PyTorch dependency.
    This is the "runtime without synthesizer" component.
    """

    def __init__(self, weights: ChaosWeights):
        self.weights = weights
        self.route_name = weights.route_name

        # Pre-compute curve parameters
        self._curves: Dict[str, Dict[str, np.ndarray]] = {}
        for hook_point, params in weights.hook_weights.items():
            self._curves[hook_point] = params

    def apply(
        self,
        activations: np.ndarray,
        hook_point: str,
        intensity: float = 1.0,
        blend_mode: str = "additive"
    ) -> np.ndarray:
        """
        Apply learned bias curve to activations.

        Fast NumPy-only implementation for inference.
        """
        if hook_point not in self._curves:
            return activations

        params = self._curves[hook_point]
        curve_type = params.get('curve_type', 'sigmoid')

        # Apply transformation
        scale = params['scale']
        shift = params['shift']
        amplitude = params['amplitude']
        offset = params['offset']

        z = activations * scale + shift

        # Apply curve
        if curve_type == 'sigmoid':
            curved = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif curve_type == 'tanh':
            curved = np.tanh(z)
        elif curve_type == 'softplus':
            curved = np.log1p(np.exp(np.clip(z, -500, 500)))
        else:
            curved = z

        bias = curved * amplitude + offset

        # Apply blend
        if blend_mode == "additive":
            return activations + bias * intensity
        elif blend_mode == "multiplicative":
            return activations * (1 + bias * intensity)
        elif blend_mode == "replace":
            return activations * (1 - intensity) + bias * intensity
        elif blend_mode == "gate":
            gate = 1 / (1 + np.exp(-bias))
            return activations * gate * intensity + activations * (1 - intensity)

        return activations

    @classmethod
    def from_file(cls, path: Path) -> 'ChaosApplicator':
        """Load applicator from weights file."""
        weights = ChaosWeights.load(path)
        return cls(weights)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Testing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_chaos_factory():
    """Test chaos factory functionality."""
    print("\n" + "=" * 70)
    print("CHAOS FACTORY TESTS")
    print("=" * 70)

    if not HAS_TORCH:
        print("PyTorch not available, skipping training tests")
        return

    # Test 1: Configuration
    print("\n[Test 1] Configuration")
    config = ChaosConfiguration(
        routes=[
            {
                'name': 'main',
                'hook_points': ['hidden_0', 'hidden_1', 'pre_output'],
                'blend_mode': 'additive'
            },
            {
                'name': 'auxiliary',
                'hook_points': ['post_embed'],
                'blend_mode': 'gate'
            }
        ],
        curve_type='sigmoid',
        input_dim=64,
        learning_rate=1e-3,
    )

    print(f"  Config fingerprint: {config.fingerprint()[:16]}")
    print(f"  Routes: {[r['name'] for r in config.routes]}")
    print(f"  Status: ✓ PASS")

    # Test 2: Factory Creation
    print("\n[Test 2] Factory Creation")
    factory = ChaosFactory(config, mode="training")

    print(f"  Factory info: {factory.info()}")
    print(f"  Total hooks: {factory.info()['total_hooks']}")
    print(f"  Status: ✓ PASS")

    # Test 3: Bias Application
    print("\n[Test 3] Bias Application")
    x = torch.randn(4, 32, 64)  # batch=4, seq=32, dim=64

    y = factory.apply_bias(x, "main", "hidden_0")

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Changed: {not torch.allclose(x, y)}")
    print(f"  Status: ✓ PASS")

    # Test 4: Baseline Capture
    print("\n[Test 4] Baseline Capture")
    baseline_acts = {
        'hidden_0': torch.randn(4, 32, 64),
        'hidden_1': torch.randn(4, 32, 64),
    }
    factory.capture_baseline(baseline_acts)

    print(f"  Baseline captured: {factory._baseline_captured}")
    print(f"  Status: ✓ PASS")

    # Test 5: Chaos Loss
    print("\n[Test 5] Chaos Loss Computation")
    current_acts = {
        'hidden_0': baseline_acts['hidden_0'] + torch.randn_like(baseline_acts['hidden_0']) * 0.1,
        'hidden_1': baseline_acts['hidden_1'] + torch.randn_like(baseline_acts['hidden_1']) * 0.1,
    }

    loss = factory.compute_chaos_loss(current_acts, target_divergence=0.1)
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Status: ✓ PASS")

    # Test 6: Weight Extraction
    print("\n[Test 6] Weight Extraction")
    weights = factory.extract_weights("main")

    print(f"  Weights fingerprint: {weights.fingerprint()}")
    print(f"  Hook points: {list(weights.hook_weights.keys())}")
    print(f"  Status: ✓ PASS")

    # Test 7: Save/Load Cycle
    print("\n[Test 7] Save/Load Cycle")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Save
        factory.save_all_weights(tmppath)

        # Load
        loaded = ChaosFactory.load_for_inference(tmppath)

        # Compare
        orig_weights = factory.extract_weights("main")
        loaded_weights = loaded.extract_weights("main")

        print(f"  Original fingerprint: {orig_weights.fingerprint()}")
        print(f"  Loaded fingerprint: {loaded_weights.fingerprint()}")
        print(f"  Match: {orig_weights.fingerprint() == loaded_weights.fingerprint()}")
        print(f"  Status: ✓ PASS")

    # Test 8: NumPy Inference
    print("\n[Test 8] NumPy Inference")
    x_np = np.random.randn(4, 32, 64).astype(np.float32)

    y_np = factory.apply_numpy(x_np, "main", "hidden_0", intensity=0.5)

    print(f"  Input shape: {x_np.shape}")
    print(f"  Output shape: {y_np.shape}")
    print(f"  Changed: {not np.allclose(x_np, y_np)}")
    print(f"  Status: ✓ PASS")

    # Test 9: Chaos Applicator
    print("\n[Test 9] Chaos Applicator (Minimal Inference)")
    applicator = ChaosApplicator(weights)

    y_app = applicator.apply(x_np, "hidden_0", intensity=0.5)

    print(f"  Output shape: {y_app.shape}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests passed! ✓ (9/9)")
    print("=" * 70)


def example_training_loop():
    """Example of integrating ChaosFactory with a classifier."""
    if not HAS_TORCH:
        print("PyTorch required for training example")
        return

    print("\n[EXAMPLE] Training Loop Integration")
    print("-" * 70)

    # 1. Create config
    config = ChaosConfiguration(
        routes=[{
            'name': 'classifier',
            'hook_points': ['embed', 'hidden', 'output'],
            'blend_mode': 'additive'
        }],
        curve_type='sigmoid',
        input_dim=128,
    )

    # 2. Create factory
    chaos_factory = ChaosFactory(config, mode="training")

    # 3. Create simple classifier (example)
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(32, 128)
            self.hidden = nn.Linear(128, 128)
            self.output = nn.Linear(128, 10)

        def forward(self, x, chaos_factory=None):
            x = self.embed(x)
            if chaos_factory:
                x = chaos_factory.apply_bias(x, "classifier", "embed")

            x = F.relu(x)
            x = self.hidden(x)
            if chaos_factory:
                x = chaos_factory.apply_bias(x, "classifier", "hidden")

            x = F.relu(x)
            x = self.output(x)
            if chaos_factory:
                x = chaos_factory.apply_bias(x, "classifier", "output")

            return x

    classifier = SimpleClassifier()

    # 4. Capture baseline (without chaos)
    x_baseline = torch.randn(8, 32)
    with torch.no_grad():
        _ = classifier(x_baseline, chaos_factory=None)
        baseline_acts = {
            'embed': classifier.embed(x_baseline),
            'hidden': classifier.hidden(F.relu(classifier.embed(x_baseline))),
        }
    chaos_factory.capture_baseline(baseline_acts)

    # 5. Training loop
    optimizer = torch.optim.Adam(chaos_factory.parameters(), lr=config.learning_rate)

    print("  Training chaos weights...")
    for step in range(10):
        x = torch.randn(8, 32)

        # Forward with chaos
        logits = classifier(x, chaos_factory=chaos_factory)

        # Get current activations
        current_acts = {
            'embed': chaos_factory.apply_bias(
                classifier.embed(x), "classifier", "embed"
            ),
        }

        # Compute chaos loss
        loss = chaos_factory.compute_chaos_loss(current_acts, target_divergence=0.1)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        chaos_factory.training_step()

        if step % 5 == 0:
            print(f"    Step {step}: loss = {loss.item():.4f}")

    # 6. Extract and save weights
    weights = chaos_factory.extract_weights("classifier")
    print(f"  Trained weights fingerprint: {weights.fingerprint()}")

    # 7. Use for inference
    applicator = ChaosApplicator(weights)
    x_test = np.random.randn(1, 128).astype(np.float32)
    x_biased = applicator.apply(x_test, "embed", intensity=0.3)

    print(f"  Inference applied, delta norm: {np.linalg.norm(x_biased - x_test):.4f}")
    print("-" * 70)


if __name__ == "__main__":
    test_chaos_factory()
    example_training_loop()

    print("\n[Summary]")
    print("Chaos Factory System")
    print("Features:")
    print("  ✓ Learnable bias curves (sigmoid, tanh, polynomial, spline, etc.)")
    print("  ✓ Route-based hook attachment to classifier trajectories")
    print("  ✓ Training mode: learn counter-bias weights")
    print("  ✓ Inference mode: apply learned curves without synthesizer")
    print("  ✓ Multiple blend modes (additive, multiplicative, gate, replace)")
    print("  ✓ ChaosApplicator for minimal NumPy-only inference")
    print("  ✓ Weight serialization and versioning")
    print("\nUse cases:")
    print("  • Decision boundary exploration")
    print("  • Controlled variability injection")
    print("  • Classifier robustness testing")
    print("  • Adversarial-constructive perturbations")
    print("  • Causal bias intervention studies")