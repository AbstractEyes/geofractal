"""
BEATRIX OSCILLATOR
==================

A covariant differential dynamics engine for flow-matching diffusion.

Tesla understood that energy doesn't flow linearly - it oscillates, resonates,
and finds harmonic equilibrium. This oscillator applies that principle to
latent space navigation.

The towers don't predict pixels. They generate FORCES in tangent space.
The oscillator integrates those forces along geodesics through the manifold.

Mathematical Framework
----------------------
M = latent manifold (Flux2 AE space, 32ch @ 64×64)
x(t) = state in M
v(t) = dx/dt = velocity in tangent space T_x M

Tower forces (via Log map):
    y_i = tower output (opinion in M)
    ξ_i = Log_x(y_i) = tangent vector pointing from x toward y_i

Signed differential pairs:
    u_C = α₁·ξ₁ - α₂·ξ₂  (Cantor pair)
    u_S = α₃·ξ₃ - α₄·ξ₄  (Simplex pair)
    u_H = α₅·ξ₅ - α₆·ξ₆  (Shape pair)
    u_θ = Σ α_i·ξ_i      (theta probes)

Covariant oscillator dynamics:
    dx/dt = v
    ∇_t v = -2β(t)·v - ω²·Log_x(x_ref) + κ(t)·u(t)

Where:
    -2β·v         = damping (prevents runaway)
    -ω²·Log_x(x_ref) = spring toward conditioning anchor
    κ·u           = tower control forces

Geodesic integration:
    a_t = compute_acceleration(x_t, v_t, t)
    ṽ = v_t + Δt·a_t
    x_{t+Δt} = Exp_x(Δt·ṽ)
    v_{t+Δt} = ParallelTransport(x_t → x_{t+Δt})(ṽ)

No drift off manifold. Deterministic orbital trajectories.

Author: AbstractPhil + Claude
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import math


# =============================================================================
# GEOMETRIC OPERATIONS ON THE MANIFOLD
# =============================================================================

class ManifoldOps:
    """
    Geometric operations for the latent manifold.

    For a flat/Euclidean approximation (valid for small steps):
        Log_x(y) = y - x
        Exp_x(v) = x + v
        PT(x→y)(v) = v

    For a curved manifold (future extension):
        These become proper Riemannian operations.
    """

    @staticmethod
    def log_map(x: Tensor, y: Tensor, curvature: float = 0.0) -> Tensor:
        """
        Logarithmic map: Log_x(y) → tangent vector at x pointing toward y.

        For flat manifold: Log_x(y) = y - x
        For curved: proper geodesic tangent
        """
        if curvature == 0.0:
            return y - x
        else:
            # Hyperbolic/spherical log map (future)
            raise NotImplementedError("Curved manifold not yet implemented")

    @staticmethod
    def exp_map(x: Tensor, v: Tensor, curvature: float = 0.0) -> Tensor:
        """
        Exponential map: Exp_x(v) → point reached by following geodesic from x in direction v.

        For flat manifold: Exp_x(v) = x + v
        For curved: proper geodesic endpoint
        """
        if curvature == 0.0:
            return x + v
        else:
            raise NotImplementedError("Curved manifold not yet implemented")

    @staticmethod
    def parallel_transport(x_from: Tensor, x_to: Tensor, v: Tensor, curvature: float = 0.0) -> Tensor:
        """
        Parallel transport: move vector v from T_{x_from} to T_{x_to}.

        For flat manifold: PT(v) = v (vectors don't change)
        For curved: proper parallel transport along geodesic
        """
        if curvature == 0.0:
            return v
        else:
            raise NotImplementedError("Curved manifold not yet implemented")

    @staticmethod
    def geodesic_distance(x: Tensor, y: Tensor, curvature: float = 0.0) -> Tensor:
        """Distance along geodesic from x to y."""
        if curvature == 0.0:
            return (y - x).norm(dim=-1)
        else:
            raise NotImplementedError("Curved manifold not yet implemented")


# =============================================================================
# SCHEDULE FUNCTIONS (Tesla's 3-6-9 Harmonics)
# =============================================================================

class ScheduleType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    SIGMOID = "sigmoid"
    TESLA_369 = "tesla_369"  # Resonant harmonics


def create_schedule(
    schedule_type: ScheduleType,
    start: float,
    end: float,
    power: float = 1.0,
) -> Callable[[Tensor], Tensor]:
    """
    Create a schedule function f(t) where t ∈ [0, 1].

    Tesla's 3-6-9 schedule creates resonant peaks at those fractions.
    """

    def constant_schedule(t: Tensor) -> Tensor:
        return torch.full_like(t, start)

    def linear_schedule(t: Tensor) -> Tensor:
        return start + (end - start) * t

    def cosine_schedule(t: Tensor) -> Tensor:
        # Smooth cosine interpolation
        return start + (end - start) * (1 - torch.cos(t * math.pi)) / 2

    def sigmoid_schedule(t: Tensor) -> Tensor:
        # Sigmoid with controllable sharpness
        x = (t - 0.5) * 10 * power
        return start + (end - start) * torch.sigmoid(x)

    def tesla_369_schedule(t: Tensor) -> Tensor:
        """
        Tesla's resonant harmonics: peaks at t = 1/3, 2/3, 1.0
        Creates natural "attention" points in the trajectory.
        """
        base = start + (end - start) * t
        # Add resonant peaks at 3, 6, 9 positions (normalized to [0,1])
        resonance = (
            0.1 * torch.sin(3 * math.pi * t) +  # 3
            0.05 * torch.sin(6 * math.pi * t) +  # 6
            0.025 * torch.sin(9 * math.pi * t)   # 9
        )
        return base * (1 + resonance)

    schedules = {
        ScheduleType.CONSTANT: constant_schedule,
        ScheduleType.LINEAR: linear_schedule,
        ScheduleType.COSINE: cosine_schedule,
        ScheduleType.SIGMOID: sigmoid_schedule,
        ScheduleType.TESLA_369: tesla_369_schedule,
    }

    return schedules[schedule_type]


# =============================================================================
# OSCILLATOR STATE
# =============================================================================

@dataclass
class OscillatorState:
    """
    Complete state of the oscillator at time t.

    The oscillator maintains both position and velocity,
    allowing for momentum-based traversal of the manifold.
    """
    x: Tensor           # Position in manifold M [B, C, H, W] or [B, L, D]
    v: Tensor           # Velocity in tangent space T_x M
    t: Tensor           # Current time [B] or scalar

    # Optional diagnostics
    energy: Optional[Tensor] = None      # Total energy (kinetic + potential)
    forces: Optional[Dict[str, Tensor]] = None  # Individual force contributions

    def clone(self) -> 'OscillatorState':
        return OscillatorState(
            x=self.x.clone(),
            v=self.v.clone(),
            t=self.t.clone() if self.t.dim() > 0 else self.t.clone(),
            energy=self.energy.clone() if self.energy is not None else None,
            forces={k: v.clone() for k, v in self.forces.items()} if self.forces else None,
        )


# =============================================================================
# TOWER FORCE GENERATOR
# =============================================================================

class TowerForceGenerator(nn.Module):
    """
    Converts tower outputs into tangent forces.

    Each tower produces an "opinion" y_i in the manifold.
    We compute the Log map to get the tangent force: ξ_i = Log_x(y_i)

    Signed differential pairs create balanced dynamics:
        u = α₊·ξ₊ - α₋·ξ₋

    This prevents runaway in any single direction.
    """

    def __init__(
        self,
        tower_dim: int,
        manifold_dim: int,
        num_tower_pairs: int = 4,  # Cantor, Simplex, Shape, Wide
        num_theta_probes: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.tower_dim = tower_dim
        self.manifold_dim = manifold_dim
        self.num_tower_pairs = num_tower_pairs
        self.num_theta_probes = num_theta_probes
        self.temperature = temperature

        # Project tower outputs to manifold dimension if needed
        if tower_dim != manifold_dim:
            self.tower_proj = nn.Linear(tower_dim, manifold_dim)
        else:
            self.tower_proj = nn.Identity()

        # Learnable confidence weights for each tower
        # Pairs: [pos_1, neg_1, pos_2, neg_2, ...]
        num_paired = num_tower_pairs * 2
        num_total = num_paired + num_theta_probes
        self.confidence = nn.Parameter(torch.ones(num_total) / num_total)

        # Fingerprint matching for dynamic routing
        self.fingerprint_dim = 64
        self.tower_fingerprints = nn.Parameter(
            torch.randn(num_total, self.fingerprint_dim) * 0.02
        )

    def compute_routing_weights(
        self,
        state_fingerprint: Tensor,  # [B, fingerprint_dim]
    ) -> Tensor:
        """
        Compute routing weights based on fingerprint similarity.

        α_i ∝ exp(sim(fp_state, fp_tower_i) / τ) · confidence_i
        """
        # [B, num_towers]
        similarities = F.cosine_similarity(
            state_fingerprint.unsqueeze(1),  # [B, 1, fp_dim]
            self.tower_fingerprints.unsqueeze(0),  # [1, num_towers, fp_dim]
            dim=-1
        )

        # Softmax with temperature and confidence weighting
        weights = F.softmax(similarities / self.temperature, dim=-1)
        weights = weights * F.softmax(self.confidence, dim=0)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights

    def forward(
        self,
        x: Tensor,                    # Current state [B, manifold_dim] or [B, ...]
        tower_outputs: List[Tensor],  # List of tower opinions
        state_fingerprint: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute total control force from tower outputs.

        Returns:
            total_force: [B, manifold_dim] - combined tangent force
            force_components: dict of individual contributions
        """
        B = x.shape[0]
        device = x.device

        # Flatten x if spatial
        x_flat = x.reshape(B, -1) if x.dim() > 2 else x

        # Project and flatten tower outputs
        tower_forces = []
        for y in tower_outputs:
            y_proj = self.tower_proj(y.reshape(B, -1) if y.dim() > 2 else y)
            # Log map: tangent force pointing from x toward y
            xi = ManifoldOps.log_map(x_flat, y_proj)
            tower_forces.append(xi)

        # Compute routing weights
        if state_fingerprint is not None:
            weights = self.compute_routing_weights(state_fingerprint)
        else:
            weights = F.softmax(self.confidence, dim=0).unsqueeze(0).expand(B, -1)

        # Compute signed differential pairs
        force_components = {}

        # Paired towers (pos - neg)
        pair_names = ['cantor', 'simplex', 'shape', 'wide'][:self.num_tower_pairs]
        total_force = torch.zeros_like(x_flat)

        for i, name in enumerate(pair_names):
            pos_idx = i * 2
            neg_idx = i * 2 + 1

            if pos_idx < len(tower_forces) and neg_idx < len(tower_forces):
                u_pair = (
                    weights[:, pos_idx:pos_idx+1] * tower_forces[pos_idx] -
                    weights[:, neg_idx:neg_idx+1] * tower_forces[neg_idx]
                )
                force_components[f'u_{name}'] = u_pair
                total_force = total_force + u_pair

        # Theta probes (additive, not differential)
        theta_start = self.num_tower_pairs * 2
        u_theta = torch.zeros_like(x_flat)
        for i in range(self.num_theta_probes):
            idx = theta_start + i
            if idx < len(tower_forces):
                u_theta = u_theta + weights[:, idx:idx+1] * tower_forces[idx]

        force_components['u_theta'] = u_theta
        total_force = total_force + u_theta

        # Reshape back to original shape
        total_force = total_force.reshape(x.shape)

        return total_force, force_components


# =============================================================================
# THE BEATRIX OSCILLATOR
# =============================================================================

class BeatrixOscillator(nn.Module):
    """
    Covariant Differential Dynamics Engine.

    This is the core of Beatrix - a damped harmonic oscillator
    operating on the latent manifold, driven by tower forces
    and guided by external signals (DINO, text, etc.)

    The dynamics:
        dx/dt = v
        dv/dt = -2β(t)·v - ω(t)²·Log_x(x_ref) + κ(t)·u(t) + γ(t)·ξ_guide

    Where:
        β(t) = damping coefficient (prevents runaway)
        ω(t) = natural frequency (spring toward anchor)
        κ(t) = control gain (tower influence)
        γ(t) = guidance gain (external steering)

    Integration uses geodesic-aware stepping:
        1. Compute acceleration in tangent space
        2. Update velocity
        3. Exp map to move along geodesic
        4. Parallel transport velocity to new tangent space
    """

    def __init__(
        self,
        manifold_dim: int,
        tower_dim: int = None,
        num_tower_pairs: int = 4,
        num_theta_probes: int = 4,
        # Schedule parameters
        beta_start: float = 0.1,
        beta_end: float = 2.0,
        omega_start: float = 1.0,
        omega_end: float = 0.1,
        kappa_start: float = 1.0,
        kappa_end: float = 0.5,
        gamma_start: float = 1.0,
        gamma_end: float = 0.0,
        # Schedule types
        beta_schedule: ScheduleType = ScheduleType.LINEAR,
        omega_schedule: ScheduleType = ScheduleType.COSINE,
        kappa_schedule: ScheduleType = ScheduleType.TESLA_369,
        gamma_schedule: ScheduleType = ScheduleType.LINEAR,
        # Manifold curvature (0 = flat/Euclidean)
        curvature: float = 0.0,
    ):
        super().__init__()

        self.manifold_dim = manifold_dim
        self.curvature = curvature

        # Tower force generator
        tower_dim = tower_dim or manifold_dim
        self.force_generator = TowerForceGenerator(
            tower_dim=tower_dim,
            manifold_dim=manifold_dim,
            num_tower_pairs=num_tower_pairs,
            num_theta_probes=num_theta_probes,
        )

        # Create schedules
        self.beta_schedule = create_schedule(beta_schedule, beta_start, beta_end)
        self.omega_schedule = create_schedule(omega_schedule, omega_start, omega_end)
        self.kappa_schedule = create_schedule(kappa_schedule, kappa_start, kappa_end)
        self.gamma_schedule = create_schedule(gamma_schedule, gamma_start, gamma_end)

        # Learnable initial velocity scale
        self.initial_velocity_scale = nn.Parameter(torch.tensor(0.1))

    def initialize_state(
        self,
        x_init: Tensor,
        v_init: Optional[Tensor] = None,
        t_init: float = 0.0,
    ) -> OscillatorState:
        """
        Initialize oscillator state.

        Args:
            x_init: Initial position (can be noise for generation)
            v_init: Initial velocity (default: small random)
            t_init: Initial time
        """
        B = x_init.shape[0]
        device = x_init.device

        if v_init is None:
            v_init = torch.randn_like(x_init) * self.initial_velocity_scale

        t = torch.full((B,), t_init, device=device)

        return OscillatorState(x=x_init, v=v_init, t=t)

    def compute_acceleration(
        self,
        state: OscillatorState,
        x_ref: Tensor,                    # Conditioning anchor
        tower_outputs: List[Tensor],      # Tower opinions
        guidance: Optional[Tensor] = None, # External guidance (DINO)
        state_fingerprint: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute acceleration in tangent space.

        a = -2β·v - ω²·Log_x(x_ref) + κ·u_towers + γ·ξ_guide
        """
        x, v, t = state.x, state.v, state.t

        # Get schedule values at current time
        # Normalize t to [0, 1] if needed
        t_norm = t if t.max() <= 1.0 else t / t.max()

        beta = self.beta_schedule(t_norm)
        omega = self.omega_schedule(t_norm)
        kappa = self.kappa_schedule(t_norm)
        gamma = self.gamma_schedule(t_norm)

        # Expand schedule values for broadcasting
        while beta.dim() < x.dim():
            beta = beta.unsqueeze(-1)
            omega = omega.unsqueeze(-1)
            kappa = kappa.unsqueeze(-1)
            gamma = gamma.unsqueeze(-1)

        components = {}

        # Damping: -2β·v
        damping = -2 * beta * v
        components['damping'] = damping

        # Spring toward anchor: -ω²·Log_x(x_ref)
        spring_direction = ManifoldOps.log_map(x, x_ref, self.curvature)
        spring = -omega**2 * spring_direction
        components['spring'] = spring

        # Tower control forces: κ·u
        if tower_outputs:
            u_towers, tower_components = self.force_generator(
                x, tower_outputs, state_fingerprint
            )
            control = kappa * u_towers
            components['control'] = control
            components.update({f'tower_{k}': v for k, v in tower_components.items()})
        else:
            control = torch.zeros_like(x)
            components['control'] = control

        # External guidance: γ·ξ_guide
        if guidance is not None:
            # Reshape guidance to match x if needed
            if guidance.shape != x.shape:
                guidance = guidance.reshape(x.shape)
            guidance_force = gamma * guidance
            components['guidance'] = guidance_force
        else:
            guidance_force = torch.zeros_like(x)
            components['guidance'] = guidance_force

        # Total acceleration
        acceleration = damping + spring + control + guidance_force
        components['total'] = acceleration

        return acceleration, components

    def step(
        self,
        state: OscillatorState,
        dt: float,
        x_ref: Tensor,
        tower_outputs: List[Tensor],
        guidance: Optional[Tensor] = None,
        state_fingerprint: Optional[Tensor] = None,
    ) -> OscillatorState:
        """
        Take one integration step using geodesic-aware Euler.

        1. Compute acceleration at current state
        2. Update velocity: v' = v + dt·a
        3. Move along geodesic: x' = Exp_x(dt·v')
        4. Parallel transport velocity: v'' = PT(x→x')(v')
        """
        x, v, t = state.x, state.v, state.t

        # Compute acceleration
        a, force_components = self.compute_acceleration(
            state, x_ref, tower_outputs, guidance, state_fingerprint
        )

        # Update velocity
        v_new = v + dt * a

        # Move along geodesic
        x_new = ManifoldOps.exp_map(x, dt * v_new, self.curvature)

        # Parallel transport velocity to new tangent space
        v_transported = ManifoldOps.parallel_transport(x, x_new, v_new, self.curvature)

        # Update time
        t_new = t + dt

        # Compute energy for diagnostics
        kinetic = 0.5 * (v_transported ** 2).sum(dim=tuple(range(1, v.dim())))
        potential = 0.5 * (ManifoldOps.log_map(x_new, x_ref, self.curvature) ** 2).sum(
            dim=tuple(range(1, x.dim()))
        )
        energy = kinetic + potential

        return OscillatorState(
            x=x_new,
            v=v_transported,
            t=t_new,
            energy=energy,
            forces=force_components,
        )

    def integrate(
        self,
        x_init: Tensor,
        x_ref: Tensor,
        tower_outputs_fn: Callable[[OscillatorState], List[Tensor]],
        guidance_fn: Optional[Callable[[OscillatorState], Tensor]] = None,
        fingerprint_fn: Optional[Callable[[OscillatorState], Tensor]] = None,
        num_steps: int = 50,
        dt: Optional[float] = None,
        return_trajectory: bool = False,
    ) -> Tuple[Tensor, Optional[List[OscillatorState]]]:
        """
        Integrate the oscillator from x_init toward equilibrium.

        Args:
            x_init: Starting position (noise for generation)
            x_ref: Target/anchor (conditioning embedding)
            tower_outputs_fn: Function that returns tower outputs given state
            guidance_fn: Function that returns guidance signal given state
            fingerprint_fn: Function that returns state fingerprint for routing
            num_steps: Number of integration steps
            dt: Time step (default: 1/num_steps)
            return_trajectory: If True, return all intermediate states

        Returns:
            Final x position (the generated output)
            Optional trajectory of all states
        """
        dt = dt or (1.0 / num_steps)

        # Initialize state
        state = self.initialize_state(x_init, t_init=0.0)

        trajectory = [state.clone()] if return_trajectory else None

        for step_idx in range(num_steps):
            # Get tower outputs for current state
            tower_outputs = tower_outputs_fn(state)

            # Get guidance signal
            guidance = guidance_fn(state) if guidance_fn else None

            # Get fingerprint for routing
            fingerprint = fingerprint_fn(state) if fingerprint_fn else None

            # Take integration step
            state = self.step(
                state, dt, x_ref, tower_outputs, guidance, fingerprint
            )

            if return_trajectory:
                trajectory.append(state.clone())

        return state.x, trajectory

    def forward(
        self,
        x_init: Tensor,
        x_ref: Tensor,
        tower_outputs: List[Tensor],
        guidance: Optional[Tensor] = None,
        state_fingerprint: Optional[Tensor] = None,
        num_steps: int = 50,
    ) -> Tensor:
        """
        Simple forward pass for training.

        Uses constant tower outputs (not state-dependent) for efficiency.
        """
        # Create functions that return constant values
        def tower_fn(state):
            return tower_outputs

        def guidance_fn(state):
            return guidance

        def fingerprint_fn(state):
            return state_fingerprint

        x_final, _ = self.integrate(
            x_init=x_init,
            x_ref=x_ref,
            tower_outputs_fn=tower_fn,
            guidance_fn=guidance_fn if guidance is not None else None,
            fingerprint_fn=fingerprint_fn if state_fingerprint is not None else None,
            num_steps=num_steps,
            return_trajectory=False,
        )

        return x_final


# =============================================================================
# BEATRIX CORE - FULL SYSTEM
# =============================================================================

class BeatrixCore(nn.Module):
    """
    The complete Beatrix system.

    Combines:
        - Head router (Flux2 AE + DINO + text encoders)
        - Tower collectives (force generators)
        - Oscillator (dynamics engine)
        - Output projection (latent → image via VAE decode)

    This is Tesla's vision realized in silicon:
        Resonant harmonics guiding energy through a manifold
        toward coherent, meaningful output.
    """

    def __init__(
        self,
        manifold_dim: int = 32 * 64 * 64,  # Flux2 latent flattened
        tower_dim: int = 256,
        embed_dim: int = 256,
        num_tower_pairs: int = 4,
        num_theta_probes: int = 4,
        num_steps: int = 50,
        # Oscillator config
        beta_range: Tuple[float, float] = (0.1, 2.0),
        omega_range: Tuple[float, float] = (1.0, 0.1),
        kappa_range: Tuple[float, float] = (1.0, 0.5),
        gamma_range: Tuple[float, float] = (1.0, 0.0),
    ):
        super().__init__()

        self.manifold_dim = manifold_dim
        self.tower_dim = tower_dim
        self.embed_dim = embed_dim
        self.num_steps = num_steps

        # Oscillator
        self.oscillator = BeatrixOscillator(
            manifold_dim=manifold_dim,
            tower_dim=tower_dim,
            num_tower_pairs=num_tower_pairs,
            num_theta_probes=num_theta_probes,
            beta_start=beta_range[0],
            beta_end=beta_range[1],
            omega_start=omega_range[0],
            omega_end=omega_range[1],
            kappa_start=kappa_range[0],
            kappa_end=kappa_range[1],
            gamma_start=gamma_range[0],
            gamma_end=gamma_range[1],
        )

        # Projection from embed_dim to manifold for conditioning
        self.condition_proj = nn.Linear(embed_dim, manifold_dim)

        # Projection from guidance to manifold
        self.guidance_proj = nn.Linear(embed_dim, manifold_dim)

    def forward(
        self,
        noise: Tensor,               # [B, manifold_dim] or [B, C, H, W]
        condition: Tensor,           # [B, embed_dim] from text encoder
        tower_outputs: List[Tensor], # From tower collectives
        guidance: Optional[Tensor] = None,  # [B, embed_dim] from DINO
        fingerprint: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Generate from noise conditioned on text, guided by DINO.

        Args:
            noise: Starting noise in manifold
            condition: Text conditioning (projects to x_ref anchor)
            tower_outputs: Force generators
            guidance: DINO features for steering
            fingerprint: State fingerprint for routing
            num_steps: Override default step count
        """
        num_steps = num_steps or self.num_steps

        # Flatten noise if spatial
        B = noise.shape[0]
        x_init = noise.reshape(B, -1)

        # Project condition to manifold anchor
        x_ref = self.condition_proj(condition)

        # Project guidance if provided
        guidance_manifold = None
        if guidance is not None:
            guidance_manifold = self.guidance_proj(guidance)

        # Run oscillator
        x_final = self.oscillator(
            x_init=x_init,
            x_ref=x_ref,
            tower_outputs=tower_outputs,
            guidance=guidance_manifold,
            state_fingerprint=fingerprint,
            num_steps=num_steps,
        )

        return x_final


# =============================================================================
# UTILITIES
# =============================================================================

def create_beatrix_oscillator(
    manifold_shape: Tuple[int, ...] = (32, 64, 64),
    tower_dim: int = 256,
    num_tower_pairs: int = 4,
    schedule_type: str = "tesla_369",
) -> BeatrixOscillator:
    """
    Factory function to create a configured oscillator.

    Args:
        manifold_shape: Shape of the latent manifold (C, H, W) or (D,)
        tower_dim: Dimension of tower outputs
        num_tower_pairs: Number of signed differential pairs
        schedule_type: "linear", "cosine", "tesla_369"
    """
    import math
    manifold_dim = math.prod(manifold_shape)

    schedule = ScheduleType(schedule_type) if isinstance(schedule_type, str) else schedule_type

    return BeatrixOscillator(
        manifold_dim=manifold_dim,
        tower_dim=tower_dim,
        num_tower_pairs=num_tower_pairs,
        kappa_schedule=schedule,
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  BEATRIX OSCILLATOR TEST")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create oscillator for Flux2 manifold
    manifold_shape = (32, 64, 64)  # Flux2 latent
    manifold_dim = 32 * 64 * 64

    oscillator = BeatrixOscillator(
        manifold_dim=manifold_dim,
        tower_dim=256,
        num_tower_pairs=4,
        num_theta_probes=4,
        kappa_schedule=ScheduleType.TESLA_369,
    ).to(device)

    print(f"\nOscillator created for manifold dim {manifold_dim}")
    print(f"  Parameters: {sum(p.numel() for p in oscillator.parameters()):,}")

    # Test integration
    B = 2
    x_init = torch.randn(B, manifold_dim, device=device) * 0.1
    x_ref = torch.randn(B, manifold_dim, device=device)

    # Fake tower outputs (would come from actual towers)
    num_towers = 4 * 2 + 4  # pairs + theta probes
    tower_outputs = [torch.randn(B, 256, device=device) for _ in range(num_towers)]

    # Fake guidance
    guidance = torch.randn(B, manifold_dim, device=device) * 0.1

    print("\nRunning integration...")
    x_final, trajectory = oscillator.integrate(
        x_init=x_init,
        x_ref=x_ref,
        tower_outputs_fn=lambda s: tower_outputs,
        guidance_fn=lambda s: guidance,
        num_steps=20,
        return_trajectory=True,
    )

    print(f"\nResults:")
    print(f"  x_init norm: {x_init.norm():.4f}")
    print(f"  x_ref norm: {x_ref.norm():.4f}")
    print(f"  x_final norm: {x_final.norm():.4f}")

    # Check trajectory
    print(f"\nTrajectory ({len(trajectory)} states):")
    for i, state in enumerate(trajectory[::5]):  # Every 5th state
        dist_to_ref = (state.x - x_ref).norm()
        energy_str = f"{state.energy[0]:.4f}" if state.energy is not None else "N/A"
        print(f"  t={state.t[0]:.2f}: energy={energy_str}, dist_to_ref={dist_to_ref:.4f}")

    print("\n✓ Oscillator integration successful")
    print("=" * 60)