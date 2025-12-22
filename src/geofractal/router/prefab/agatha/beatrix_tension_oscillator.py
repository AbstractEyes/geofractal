"""
BEATRIX Tension OSCILLATOR
==================
geofractal.router.prefab.agatha.beatrix_tension_oscillator

A covariant differential dynamics engine for flow-matching diffusion.

This version is specifically formatted with some predominantly physics operations and notations for clarity.
Additionally, it includes a hardcoded mathematical framework in the docstring for reference.

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

Covariant oscillator dynamics with intrinsic tension:
    dx/dt = v
    ∇_t v = -2β(t)·v - (1-τ)·ω²·Log_x(x_ref) + τ·κ(t)·u(t)

Where:
    -2β·v              = damping (prevents runaway)
    -(1-τ)·ω²·Log_x    = spring toward anchor (attenuated by tension)
    τ·κ·u              = tower control forces (amplified by tension)
    τ                  = intrinsic tension ∈ [0,1] (learned from geometric invariants)

Geometric Invariants for Tension:
    The tension τ emerges from STATE relationships, not raw 131k coordinates:
    1. ||x - x_ref||           - distance from anchor
    2. ||v||                   - velocity magnitude
    3. v̂ · d̂                   - velocity toward anchor
    4. F̂_spring · F̂_tower     - force alignment
    5. v̂ · F̂_spring           - velocity-spring alignment
    6. v̂ · F̂_tower            - velocity-tower alignment
    7. E_kinetic + E_potential - energy proxy
    8. ||F_tower|| / ||F_spring|| - force balance

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

from geofractal.router.base_router import BaseRouter
from geofractal.router.components.torch_component import TorchComponent


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
    TESLA_369 = "tesla_369"
    SURGE = "surge"
    DELTA = "delta"
    GAMMA = "gamma"
    MULTI = "multi"
    ADDITIVE = "additive"


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
        return start + (end - start) * (1 - torch.cos(t * math.pi)) / 2

    def sigmoid_schedule(t: Tensor) -> Tensor:
        x = (t - 0.5) * 10 * power
        return start + (end - start) * torch.sigmoid(x)

    def tesla_369_schedule(t: Tensor) -> Tensor:
        """
        Tesla's resonant harmonics: peaks at t = 1/3, 2/3, 1.0
        Creates natural "attention" points in the trajectory.
        """
        base = start + (end - start) * t
        resonance = (
                0.1 * torch.sin(3 * math.pi * t) +
                0.05 * torch.sin(6 * math.pi * t) +
                0.025 * torch.sin(9 * math.pi * t)
        )
        return base * (1 + resonance)

    schedules = {
        ScheduleType.CONSTANT: constant_schedule,
        ScheduleType.LINEAR: linear_schedule,
        ScheduleType.COSINE: cosine_schedule,
        ScheduleType.SIGMOID: sigmoid_schedule,
        ScheduleType.TESLA_369: tesla_369_schedule,
    }

    return schedules.get(schedule_type, constant_schedule)


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
    x: Tensor  # Position in manifold M [B, C, H, W] or [B, L, D]
    v: Tensor  # Velocity in tangent space T_x M
    t: Tensor  # Current time [B] or scalar

    energy: Optional[Tensor] = None
    forces: Optional[Dict[str, Tensor]] = None
    tension: Optional[Tensor] = None  # Intrinsic tension τ

    def clone(self) -> 'OscillatorState':
        return OscillatorState(
            x=self.x.clone(),
            v=self.v.clone(),
            t=self.t.clone() if self.t.dim() > 0 else self.t.clone(),
            energy=self.energy.clone() if self.energy is not None else None,
            forces={k: v.clone() for k, v in self.forces.items()} if self.forces else None,
            tension=self.tension.clone() if self.tension is not None else None,
        )


# =============================================================================
# INTRINSIC TENSION (Invariant-Based - ~17k params)
# =============================================================================

class IntrinsicTension(nn.Module):
    """
    Tension from geometric invariants - NOT raw coordinates.

    The 131k dimensions are the SPACE. Tension emerges from
    relationships WITHIN that space, which are naturally low-dimensional.

    Invariants computed:
        1. log(||x - x_ref||)         - distance from anchor
        2. log(||v||)                 - velocity magnitude
        3. v̂ · d̂                      - velocity toward anchor (-1=fleeing, +1=approaching)
        4. F̂_spring · F̂_tower        - force alignment (-1=fighting, +1=cooperating)
        5. v̂ · F̂_spring              - velocity-spring alignment
        6. v̂ · F̂_tower               - velocity-tower alignment
        7. log(E_kinetic + E_potential) - energy proxy
        8. log(||F_tower|| / ||F_spring||) - force balance

    ~17k params instead of 10B. The physics is in the invariants, not coordinates.
    """

    def __init__(
        self,
        manifold_dim: int,  # kept for API compatibility, not used internally
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.manifold_dim = manifold_dim  # stored but not used for sizing

        num_invariants = 8

        self.invariant_net = nn.Sequential(
            nn.Linear(num_invariants, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learnable importance weighting per invariant
        self.invariant_scales = nn.Parameter(torch.ones(num_invariants))

        # Learnable equilibrium (bias) and gain (sensitivity)
        self.equilibrium = nn.Parameter(torch.tensor(0.0))
        self.gain = nn.Parameter(torch.tensor(1.0))

    def compute_invariants(
        self,
        x: Tensor,
        x_ref: Tensor,
        v: Tensor,
        spring_force: Tensor,
        tower_force: Tensor,
    ) -> Tensor:
        """Extract geometric invariants from state. All outputs are [B, 8]."""
        eps = 1e-8

        # Compute norms
        displacement = x_ref - x
        disp_norm = displacement.norm(dim=-1, keepdim=True) + eps
        vel_norm = v.norm(dim=-1, keepdim=True) + eps
        spring_norm = spring_force.norm(dim=-1, keepdim=True) + eps
        tower_norm = tower_force.norm(dim=-1, keepdim=True) + eps

        # Normalized directions
        disp_dir = displacement / disp_norm
        vel_dir = v / vel_norm
        spring_dir = spring_force / spring_norm
        tower_dir = tower_force / tower_norm

        invariants = torch.cat([
            # 1. Distance from anchor (log scale for numerical stability)
            torch.log1p(disp_norm),

            # 2. Velocity magnitude (log scale)
            torch.log1p(vel_norm),

            # 3. Velocity toward anchor: +1 = approaching, -1 = fleeing
            (vel_dir * disp_dir).sum(dim=-1, keepdim=True),

            # 4. Spring-tower alignment: +1 = cooperating, -1 = fighting
            (spring_dir * tower_dir).sum(dim=-1, keepdim=True),

            # 5. Velocity-spring alignment
            (vel_dir * spring_dir).sum(dim=-1, keepdim=True),

            # 6. Velocity-tower alignment
            (vel_dir * tower_dir).sum(dim=-1, keepdim=True),

            # 7. Energy proxy (log scale)
            torch.log1p(vel_norm**2 + disp_norm**2),

            # 8. Force balance: >0 = tower dominates, <0 = spring dominates
            torch.log(tower_norm / spring_norm),

        ], dim=-1)  # [B, 8]

        return invariants

    def forward(
        self,
        x: Tensor,
        x_ref: Tensor,
        spring_force: Tensor,
        tower_force: Tensor,
        velocity: Tensor,
    ) -> Tensor:
        """
        Compute intrinsic tension τ ∈ [0, 1].

        High τ → trust towers (geometric structure dominates)
        Low τ  → trust spring (anchor pull dominates)
        """
        # Compute geometric invariants
        invariants = self.compute_invariants(
            x, x_ref, velocity, spring_force, tower_force
        )  # [B, 8]

        # Scale by learned importance weights
        scaled = invariants * self.invariant_scales

        # Predict tension from invariants
        raw_tension = self.invariant_net(scaled).squeeze(-1)  # [B]

        # Add equilibrium bias
        raw_tension = raw_tension + self.equilibrium

        # Sigmoid with learned gain
        tau = torch.sigmoid(self.gain * raw_tension)

        return tau


# =============================================================================
# TOWER FORCE GENERATOR (TorchComponent)
# =============================================================================

class TowerForceGenerator(TorchComponent):
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
            name: str,
            tower_dim: int,
            manifold_dim: int,
            num_tower_pairs: int = 4,
            num_theta_probes: int = 4,
            temperature: float = 1.0,
            fingerprint_dim: int = 64,
    ):
        super().__init__(name)

        self.tower_dim = tower_dim
        self.manifold_dim = manifold_dim
        self.num_tower_pairs = num_tower_pairs
        self.num_theta_probes = num_theta_probes
        self.temperature = temperature
        self.fingerprint_dim = fingerprint_dim

        # Project tower outputs to manifold dimension if needed
        if tower_dim != manifold_dim:
            self.tower_proj = nn.Linear(tower_dim, manifold_dim)
        else:
            self.tower_proj = nn.Identity()

        # Learnable confidence weights for each tower
        num_paired = num_tower_pairs * 2
        num_total = num_paired + num_theta_probes
        self.confidence = nn.Parameter(torch.ones(num_total) / num_total)

        # Fingerprint matching for dynamic routing
        self.tower_fingerprints = nn.Parameter(
            torch.randn(num_total, fingerprint_dim) * 0.02
        )

    def compute_routing_weights(
            self,
            state_fingerprint: Tensor,
    ) -> Tensor:
        """
        Compute routing weights based on fingerprint similarity.

        α_i ∝ exp(sim(fp_state, fp_tower_i) / τ) · confidence_i
        """
        similarities = F.cosine_similarity(
            state_fingerprint.unsqueeze(1),
            self.tower_fingerprints.unsqueeze(0),
            dim=-1
        )

        weights = F.softmax(similarities / self.temperature, dim=-1)
        weights = weights * F.softmax(self.confidence, dim=0)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights

    def forward(
            self,
            x: Tensor,
            tower_outputs: List[Tensor],
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

        x_flat = x.reshape(B, -1) if x.dim() > 2 else x

        # Project and flatten tower outputs
        tower_forces = []
        for y in tower_outputs:
            y_proj = self.tower_proj(y.reshape(B, -1) if y.dim() > 2 else y)
            xi = ManifoldOps.log_map(x_flat, y_proj)
            tower_forces.append(xi)

        # Compute routing weights
        if state_fingerprint is not None:
            weights = self.compute_routing_weights(state_fingerprint)
        else:
            weights = F.softmax(self.confidence, dim=0).unsqueeze(0).expand(B, -1)

        # Compute signed differential pairs
        force_components = {}
        pair_names = ['cantor', 'simplex', 'shape', 'wide'][:self.num_tower_pairs]
        total_force = torch.zeros_like(x_flat)

        for i, name in enumerate(pair_names):
            pos_idx = i * 2
            neg_idx = i * 2 + 1

            if pos_idx < len(tower_forces) and neg_idx < len(tower_forces):
                u_pair = (
                        weights[:, pos_idx:pos_idx + 1] * tower_forces[pos_idx] -
                        weights[:, neg_idx:neg_idx + 1] * tower_forces[neg_idx]
                )
                force_components[f'u_{name}'] = u_pair
                total_force = total_force + u_pair

        # Theta probes (additive, not differential)
        theta_start = self.num_tower_pairs * 2
        u_theta = torch.zeros_like(x_flat)
        for i in range(self.num_theta_probes):
            idx = theta_start + i
            if idx < len(tower_forces):
                u_theta = u_theta + weights[:, idx:idx + 1] * tower_forces[idx]

        force_components['u_theta'] = u_theta
        total_force = total_force + u_theta

        total_force = total_force.reshape(x.shape)

        return total_force, force_components


# =============================================================================
# BEATRIX OSCILLATOR (TorchComponent)
# =============================================================================

class BeatrixOscillator(TorchComponent):
    """
    Covariant Differential Dynamics Engine with Intrinsic Tension.

    This is the core of Beatrix - a damped harmonic oscillator
    operating on the latent manifold, driven by tower forces
    and guided by external signals (DINO, text, etc.)

    The dynamics with intrinsic tension:
        dx/dt = v
        dv/dt = -2β(t)·v - (1-τ)·ω(t)²·Log_x(x_ref) + τ·κ(t)·u(t) + γ(t)·ξ_guide

    Where:
        β(t) = damping coefficient (prevents runaway)
        ω(t) = natural frequency (spring toward anchor)
        κ(t) = control gain (tower influence)
        γ(t) = guidance gain (external steering)
        τ    = intrinsic tension (learned from geometric invariants)

    Integration uses geodesic-aware stepping:
        1. Compute raw forces (spring, tower)
        2. Compute intrinsic tension from current state
        3. Modulate forces by tension
        4. Update velocity and position
    """

    def __init__(
            self,
            name: str = 'beatrix_oscillator',
            manifold_dim: int = 131072,
            tower_dim: int = 256,
            num_tower_pairs: int = 4,
            num_theta_probes: int = 4,
            fingerprint_dim: int = 64,
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
            # Tension configuration
            use_intrinsic_tension: bool = True,
            tension_hidden_dim: int = 128,
    ):
        super().__init__(name)

        self.manifold_dim = manifold_dim
        self.curvature = curvature
        self.use_intrinsic_tension = use_intrinsic_tension

        # Tower force generator (TorchComponent)
        self.force_generator = TowerForceGenerator(
            name=f'{name}_force_gen',
            tower_dim=tower_dim,
            manifold_dim=manifold_dim,
            num_tower_pairs=num_tower_pairs,
            num_theta_probes=num_theta_probes,
            fingerprint_dim=fingerprint_dim,
        )

        # Intrinsic tension module (lightweight, invariant-based)
        if use_intrinsic_tension:
            self.tension = IntrinsicTension(
                manifold_dim=manifold_dim,
                hidden_dim=tension_hidden_dim,
            )
        else:
            self.tension = None

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
        """Initialize oscillator state."""
        B = x_init.shape[0]
        device = x_init.device

        if v_init is None:
            v_init = torch.randn_like(x_init) * self.initial_velocity_scale

        t = torch.full((B,), t_init, device=device)

        return OscillatorState(x=x_init, v=v_init, t=t)

    def compute_acceleration(
            self,
            state: OscillatorState,
            x_ref: Tensor,
            tower_outputs: List[Tensor],
            guidance: Optional[Tensor] = None,
            state_fingerprint: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """
        Compute acceleration in tangent space with intrinsic tension.

        a = -2β·v - (1-τ)·ω²·Log_x(x_ref) + τ·κ·u_towers + γ·ξ_guide

        Returns:
            acceleration: [B, D]
            components: dict of force components
            tau: tension values [B]
        """
        x, v, t = state.x, state.v, state.t

        t_norm = t if t.max() <= 1.0 else t / t.max()

        beta = self.beta_schedule(t_norm)
        omega = self.omega_schedule(t_norm)
        kappa = self.kappa_schedule(t_norm)
        gamma = self.gamma_schedule(t_norm)

        while beta.dim() < x.dim():
            beta = beta.unsqueeze(-1)
            omega = omega.unsqueeze(-1)
            kappa = kappa.unsqueeze(-1)
            gamma = gamma.unsqueeze(-1)

        components = {}

        # Damping: -2β·v (not modulated by tension)
        damping = -2 * beta * v
        components['damping'] = damping

        # Compute raw spring force: -ω²·Log_x(x_ref)
        spring_direction = ManifoldOps.log_map(x, x_ref, self.curvature)
        raw_spring = -omega ** 2 * spring_direction
        components['raw_spring'] = raw_spring

        # Compute raw tower control force: κ·u
        if tower_outputs:
            u_towers, tower_components = self.force_generator(
                x, tower_outputs, state_fingerprint
            )
            raw_control = kappa * u_towers
            components['raw_control'] = raw_control
            components.update({f'tower_{k}': val for k, val in tower_components.items()})
        else:
            raw_control = torch.zeros_like(x)
            components['raw_control'] = raw_control

        # Compute intrinsic tension from geometric invariants
        if self.use_intrinsic_tension and self.tension is not None:
            # Flatten for tension computation if needed
            x_flat = x.reshape(x.shape[0], -1) if x.dim() > 2 else x
            x_ref_flat = x_ref.reshape(x_ref.shape[0], -1) if x_ref.dim() > 2 else x_ref
            spring_flat = raw_spring.reshape(raw_spring.shape[0], -1) if raw_spring.dim() > 2 else raw_spring
            control_flat = raw_control.reshape(raw_control.shape[0], -1) if raw_control.dim() > 2 else raw_control
            v_flat = v.reshape(v.shape[0], -1) if v.dim() > 2 else v

            tau = self.tension(
                x=x_flat,
                x_ref=x_ref_flat,
                spring_force=spring_flat,
                tower_force=control_flat,
                velocity=v_flat,
            )  # [B]

            # Expand tau for broadcasting
            tau_expanded = tau
            while tau_expanded.dim() < x.dim():
                tau_expanded = tau_expanded.unsqueeze(-1)

            # Modulate forces by tension
            spring = (1 - tau_expanded) * raw_spring
            control = tau_expanded * raw_control
        else:
            tau = torch.ones(x.shape[0], device=x.device) * 0.5  # Default balanced
            spring = raw_spring
            control = raw_control

        components['spring'] = spring
        components['control'] = control
        components['tension'] = tau

        # External guidance: γ·ξ_guide (not modulated by tension)
        if guidance is not None:
            if guidance.shape != x.shape:
                guidance = guidance.reshape(x.shape)
            guidance_force = gamma * guidance
            components['guidance'] = guidance_force
        else:
            guidance_force = torch.zeros_like(x)
            components['guidance'] = guidance_force

        acceleration = damping + spring + control + guidance_force
        components['total'] = acceleration

        return acceleration, components, tau

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

        1. Compute acceleration at current state (with intrinsic tension)
        2. Update velocity: v' = v + dt·a
        3. Move along geodesic: x' = Exp_x(dt·v')
        4. Parallel transport velocity: v'' = PT(x→x')(v')
        """
        x, v, t = state.x, state.v, state.t

        a, force_components, tau = self.compute_acceleration(
            state, x_ref, tower_outputs, guidance, state_fingerprint
        )

        v_new = v + dt * a
        x_new = ManifoldOps.exp_map(x, dt * v_new, self.curvature)
        v_transported = ManifoldOps.parallel_transport(x, x_new, v_new, self.curvature)
        t_new = t + dt

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
            tension=tau,
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

        state = self.initialize_state(x_init, t_init=0.0)

        trajectory = [state.clone()] if return_trajectory else None

        for step_idx in range(num_steps):
            tower_outputs = tower_outputs_fn(state)
            guidance = guidance_fn(state) if guidance_fn else None
            fingerprint = fingerprint_fn(state) if fingerprint_fn else None

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
# PROJECTION COMPONENT
# =============================================================================

class ProjectionBlock(TorchComponent):
    """MLP projection as a proper TorchComponent."""

    def __init__(
            self,
            name: str,
            in_dim: int,
            out_dim: int,
            hidden_dim: Optional[int] = None,
    ):
        super().__init__(name)
        hidden_dim = hidden_dim or in_dim * 2

        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# =============================================================================
# BEATRIX CORE (BaseRouter)
# =============================================================================

class BeatrixCore(BaseRouter):
    """
    The complete Beatrix oscillator system as a BaseRouter.

    Combines:
        - Oscillator (dynamics engine with intrinsic tension)
        - Condition projection (embed_dim → manifold)
        - Guidance projection (embed_dim → manifold)

    This is Tesla's vision realized in silicon:
        Resonant harmonics guiding energy through a manifold
        toward coherent, meaningful output.
    """

    def __init__(
            self,
            name: str = 'beatrix_core',
            manifold_dim: int = 32 * 64 * 64,
            tower_dim: int = 256,
            embed_dim: int = 256,
            num_tower_pairs: int = 4,
            num_theta_probes: int = 4,
            num_steps: int = 50,
            fingerprint_dim: int = 64,
            # Oscillator config
            beta_range: Tuple[float, float] = (0.1, 2.0),
            omega_range: Tuple[float, float] = (1.0, 0.1),
            kappa_range: Tuple[float, float] = (1.0, 0.5),
            gamma_range: Tuple[float, float] = (1.0, 0.0),
            kappa_schedule: ScheduleType = ScheduleType.TESLA_369,
            # Tension config
            use_intrinsic_tension: bool = True,
    ):
        super().__init__(name, strict=False)

        self.manifold_dim = manifold_dim
        self.tower_dim = tower_dim
        self.embed_dim = embed_dim
        self.num_steps = num_steps

        # Store config
        self.objects['config'] = {
            'manifold_dim': manifold_dim,
            'tower_dim': tower_dim,
            'embed_dim': embed_dim,
            'num_steps': num_steps,
            'use_intrinsic_tension': use_intrinsic_tension,
        }

        # Oscillator (TorchComponent)
        self.attach('oscillator', BeatrixOscillator(
            name='oscillator',
            manifold_dim=manifold_dim,
            tower_dim=tower_dim,
            num_tower_pairs=num_tower_pairs,
            num_theta_probes=num_theta_probes,
            fingerprint_dim=fingerprint_dim,
            beta_start=beta_range[0],
            beta_end=beta_range[1],
            omega_start=omega_range[0],
            omega_end=omega_range[1],
            kappa_start=kappa_range[0],
            kappa_end=kappa_range[1],
            gamma_start=gamma_range[0],
            gamma_end=gamma_range[1],
            kappa_schedule=kappa_schedule,
            use_intrinsic_tension=use_intrinsic_tension,
        ))

        # Projection from embed_dim to manifold for conditioning
        self.attach('condition_proj', ProjectionBlock(
            'condition_proj',
            in_dim=embed_dim,
            out_dim=manifold_dim,
        ))

        # Projection from guidance to manifold
        self.attach('guidance_proj', ProjectionBlock(
            'guidance_proj',
            in_dim=embed_dim,
            out_dim=manifold_dim,
        ))

    def forward(
            self,
            noise: Tensor,
            condition: Tensor,
            tower_outputs: List[Tensor],
            guidance: Optional[Tensor] = None,
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

        B = noise.shape[0]
        x_init = noise.reshape(B, -1)

        x_ref = self['condition_proj'](condition)

        guidance_manifold = None
        if guidance is not None:
            guidance_manifold = self['guidance_proj'](guidance)

        x_final = self['oscillator'](
            x_init=x_init,
            x_ref=x_ref,
            tower_outputs=tower_outputs,
            guidance=guidance_manifold,
            state_fingerprint=fingerprint,
            num_steps=num_steps,
        )

        return x_final


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_beatrix_oscillator(
        manifold_shape: Tuple[int, ...] = (32, 64, 64),
        tower_dim: int = 256,
        num_tower_pairs: int = 4,
        schedule_type: str = "tesla_369",
        use_intrinsic_tension: bool = True,
        tension_hidden_dim: int = 128,
) -> BeatrixOscillator:
    """
    Factory function to create a configured oscillator.

    Args:
        manifold_shape: Shape of the latent manifold (C, H, W) or (D,)
        tower_dim: Dimension of tower outputs
        num_tower_pairs: Number of signed differential pairs
        schedule_type: "linear", "cosine", "tesla_369"
        use_intrinsic_tension: Whether to use learned tension modulation
        tension_hidden_dim: Hidden dimension for tension network
    """
    manifold_dim = math.prod(manifold_shape)

    schedule = ScheduleType(schedule_type) if isinstance(schedule_type, str) else schedule_type

    return BeatrixOscillator(
        name='beatrix_oscillator',
        manifold_dim=manifold_dim,
        tower_dim=tower_dim,
        num_tower_pairs=num_tower_pairs,
        kappa_schedule=schedule,
        use_intrinsic_tension=use_intrinsic_tension,
        tension_hidden_dim=tension_hidden_dim,
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  BEATRIX OSCILLATOR TEST (with Intrinsic Tension)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    manifold_shape = (32, 64, 64)
    manifold_dim = 32 * 64 * 64

    oscillator = BeatrixOscillator(
        name='test_oscillator',
        manifold_dim=manifold_dim,
        tower_dim=256,
        num_tower_pairs=4,
        num_theta_probes=4,
        kappa_schedule=ScheduleType.TESLA_369,
        use_intrinsic_tension=True,
        tension_hidden_dim=128,
    ).to(device)

    print(f"\nOscillator created for manifold dim {manifold_dim}")
    total_params = sum(p.numel() for p in oscillator.parameters())
    tension_params = sum(p.numel() for p in oscillator.tension.parameters()) if oscillator.tension else 0
    print(f"  Total parameters: {total_params:,}")
    print(f"  Tension parameters: {tension_params:,}")
    print(f"  Intrinsic tension: {oscillator.use_intrinsic_tension}")

    B = 2
    x_init = torch.randn(B, manifold_dim, device=device) * 0.1
    x_ref = torch.randn(B, manifold_dim, device=device)

    num_towers = 4 * 2 + 4
    tower_outputs = [torch.randn(B, 256, device=device) for _ in range(num_towers)]
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

    print(f"\nTrajectory ({len(trajectory)} states):")
    for i, state in enumerate(trajectory[::5]):
        dist_to_ref = (state.x - x_ref).norm()
        energy_str = f"{state.energy[0]:.4f}" if state.energy is not None else "N/A"
        tau_str = f"{state.tension.mean():.3f}" if state.tension is not None else "N/A"
        print(f"  t={state.t[0]:.2f}: energy={energy_str}, dist_to_ref={dist_to_ref:.4f}, τ={tau_str}")

    # Show invariant scales if learned
    if oscillator.tension is not None:
        print(f"\nLearned invariant scales:")
        scale_names = ['disp', 'vel', 'v→anchor', 'F_align', 'v→spring', 'v→tower', 'energy', 'F_ratio']
        for name, scale in zip(scale_names, oscillator.tension.invariant_scales):
            print(f"  {name}: {scale.item():.4f}")
        print(f"  equilibrium: {oscillator.tension.equilibrium.item():.4f}")
        print(f"  gain: {oscillator.tension.gain.item():.4f}")

    print("\n✓ Oscillator integration successful")
    print("=" * 60)