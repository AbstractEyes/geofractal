import math
import types
import torch
from dataclasses import dataclass
from typing import Tuple

from beatrix_oscillator import BeatrixOscillator, ScheduleType, OscillatorState


# ============================================================
# Config
# ============================================================

@dataclass
class ExpConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    manifold_dim: int = 2048
    tower_dim: int = 256
    B: int = 8

    num_tower_pairs: int = 4
    num_theta_probes: int = 0

    num_steps: int = 64
    dt: float | None = None

    beta: Tuple[float, float] = (0.1, 1.0)
    omega: Tuple[float, float] = (1.0, 0.1)
    kappa: Tuple[float, float] = (1.0, 0.5)
    gamma: Tuple[float, float] = (0.0, 0.0)


cfg = ExpConfig()


# ============================================================
# Monkey-patched dual-opinion control force
# ============================================================

def dual_opinion_control(self, state: OscillatorState, t_frac: float):
    """
    Monkey-patched replacement for compute_control_force().
    Implements dual-opinion ± effectiveness alignment.
    """

    if not state.tower_outputs:
        return torch.zeros_like(state.x)

    # unpack tower outputs: (p0,n0,p1,n1,...)
    pairs = []
    it = iter(state.tower_outputs)
    for _ in range(cfg.num_tower_pairs):
        p = next(it)
        n = next(it)
        pairs.append((p, n))

    # desired direction toward reference
    u = state.x_ref - state.x
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-6)

    forces = []
    for p, n in pairs:
        d = p - n
        d_hat = d / (d.norm(dim=-1, keepdim=True) + 1e-6)

        # effectiveness = alignment with trajectory
        cos = (d_hat * u).sum(dim=-1)
        e = torch.sigmoid(6.0 * cos)   # sharp but stable

        # project along desired direction (trajectory aligner)
        proj = (d_hat * u).sum(dim=-1, keepdim=True) * u

        forces.append(proj * e.unsqueeze(-1))

    control = torch.stack(forces, dim=0).sum(dim=0)
    return control


# ============================================================
# Experiment runner
# ============================================================

def make_oscillator(schedule: ScheduleType) -> BeatrixOscillator:
    osc = BeatrixOscillator(
        name="patched",
        manifold_dim=cfg.manifold_dim,
        tower_dim=cfg.tower_dim,
        num_tower_pairs=cfg.num_tower_pairs,
        num_theta_probes=cfg.num_theta_probes,
        beta_start=cfg.beta[0],
        beta_end=cfg.beta[1],
        omega_start=cfg.omega[0],
        omega_end=cfg.omega[1],
        kappa_start=cfg.kappa[0],
        kappa_end=cfg.kappa[1],
        gamma_start=cfg.gamma[0],
        gamma_end=cfg.gamma[1],
        beta_schedule=ScheduleType.LINEAR,
        omega_schedule=ScheduleType.COSINE,
        kappa_schedule=schedule,
        gamma_schedule=ScheduleType.CONSTANT,
        curvature=0.0,
    ).to(cfg.device)

    osc = osc.to(dtype=cfg.dtype)

    # monkey-patch control force
    osc.compute_control_force = types.MethodType(dual_opinion_control, osc)

    return osc


@torch.no_grad()
def run_case(schedule, use_towers):
    osc = make_oscillator(schedule)

    B = cfg.B
    x_init = 0.1 * torch.randn(B, cfg.manifold_dim, device=cfg.device, dtype=cfg.dtype)
    x_ref = torch.randn(B, cfg.manifold_dim, device=cfg.device, dtype=cfg.dtype)

    tower_outputs = []
    if use_towers:
        for _ in range(cfg.num_tower_pairs * 2):
            tower_outputs.append(
                torch.randn(B, cfg.tower_dim, device=cfg.device, dtype=cfg.dtype)
            )

    x_final, traj = osc.integrate(
        x_init=x_init,
        x_ref=x_ref,
        tower_outputs_fn=(lambda s: tower_outputs),
        guidance_fn=None,
        fingerprint_fn=None,
        num_steps=cfg.num_steps,
        dt=cfg.dt,
        return_trajectory=True,
    )

    # skip initial state (energy undefined)
    traj = traj[1:]

    d0 = (traj[0].x - x_ref).norm(dim=-1).mean().item()
    d1 = (traj[-1].x - x_ref).norm(dim=-1).mean().item()

    return d0, d1, d1 / (d0 + 1e-9)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    schedules = [
        ScheduleType.CONSTANT,
        ScheduleType.LINEAR,
        ScheduleType.COSINE,
        ScheduleType.TESLA_369,
    ]

    print("\nBeatrixOscillator — Dual-Opinion Alignment Test\n")
    print("schedule     towers   dist ratio")
    print("-" * 40)

    for sch in schedules:
        for towers in [False, True]:
            d0, d1, ratio = run_case(sch, towers)
            print(f"{sch.value:<12} T{int(towers)}      {ratio:.4f}")
