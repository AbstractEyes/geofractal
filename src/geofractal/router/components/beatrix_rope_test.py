"""
geofractal.router.components.fusion_shape_seq_experiment
========================================================

Synthetic multi-view shape classification:
    label ∈ {cube, pyramid}

Long-context analog:
    context length = number of views (frames) of the same object.

Core goals:
    - Cohesively bounded, traceable task (cube vs pyramid)
    - Batched tensor ops (no B×L×E Python loops in hot path)
    - Progress bars with live loss + accuracy (EMA)

Streams:
    1) raw wireframe render
    2) Sobel edge magnitude
    3) blurred image (avgpool)

Pipeline:
    render -> streams -> per-frame encoders -> per-frame fusion (your FusionComponent)
    -> mean over time -> classify

Usage:
    from geofractal.router.components.fusion_shape_seq_experiment import run
    run("gated")  # or "concat", "sum", "adaptive", "residual_gated"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# tqdm (graceful fallback)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it

# Import your fusion components
from geofractal.router.components.fusion_component import (
    GatedFusion,
    ConcatFusion,
    SumFusion,
    AdaptiveFusion,
    ResidualFusion,
)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class BaseConfig:
    device: str = "cuda"
    seed: int = 42

    # data
    img_size: int = 64
    batch_size: int = 64

    # training
    train_steps: int = 300
    eval_batches: int = 50
    lr: float = 2e-3

    # long-context analog (views per sample)
    context_lengths: Tuple[int, ...] = (2, 4, 8, 16, 32)

    # model
    feat_dim: int = 128

    # render randomness
    noise_std: float = 0.08
    line_thickness: float = 1.3      # pixels (soft thickness)
    jitter_px: float = 1.0           # vertex jitter in pixels
    dropout_prob: float = 0.10       # randomly drop some edges per (B,L,E)
    scale_range: Tuple[float, float] = (0.85, 1.15)

    # camera
    cam_dist: float = 3.0
    fov: float = 1.2                 # projection scale

    # progress / metrics
    ema_alpha: float = 0.1


CFG = BaseConfig()


# =============================================================================
# EMA METER
# =============================================================================

class EMAMeter:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return float(self.value)


# =============================================================================
# SHAPES (Unified padded representation)
# =============================================================================

def _build_unified_shapes(device: torch.device):
    """
    Returns:
        verts: [2, Vmax, 3]
        edges: [2, Emax, 2] (indices into Vmax, invalid edges have index 0 but masked off)
        emask: [2, Emax] boolean mask for valid edges
    """
    # ---- cube ----
    v_cube = torch.tensor([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], dtype=torch.float32, device=device)

    e_cube = torch.tensor([
        [0,1],[1,2],[2,3],[3,0],   # bottom
        [4,5],[5,6],[6,7],[7,4],   # top
        [0,4],[1,5],[2,6],[3,7],   # verticals
    ], dtype=torch.long, device=device)  # 12 edges

    # ---- pyramid ----
    v_pyr = torch.tensor([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [ 0,  0,  1.2],  # apex
    ], dtype=torch.float32, device=device)

    e_pyr = torch.tensor([
        [0,1],[1,2],[2,3],[3,0],   # base (4)
        [0,4],[1,4],[2,4],[3,4],   # sides (4)
    ], dtype=torch.long, device=device)  # 8 edges

    Vmax = 8
    Emax = 12

    verts = torch.zeros((2, Vmax, 3), dtype=torch.float32, device=device)
    edges = torch.zeros((2, Emax, 2), dtype=torch.long, device=device)
    emask = torch.zeros((2, Emax), dtype=torch.bool, device=device)

    # cube fill
    verts[0, :8] = v_cube
    edges[0, :12] = e_cube
    emask[0, :12] = True

    # pyramid fill (pad vertices to 8)
    verts[1, :5] = v_pyr
    edges[1, :8] = e_pyr
    emask[1, :8] = True
    # remaining edges masked off (indices already 0)

    return verts, edges, emask


# =============================================================================
# BATCHED ROTATIONS / PROJECTION
# =============================================================================

def batched_rotations(B: int, L: int, device: torch.device) -> torch.Tensor:
    """
    Returns R: [B, L, 3, 3]
    """
    base_y = torch.rand((B,), device=device) * (2.0 * math.pi)
    base_x = (torch.rand((B,), device=device) * 1.2 - 0.6)  # [-0.6, 0.6]

    t_lin = torch.linspace(0.0, 2.0 * math.pi, steps=L, device=device)  # [L]
    theta = base_y[:, None] + t_lin[None, :]                             # [B,L]
    phi = base_x[:, None] + 0.15 * torch.sin(theta)

    c, s = torch.cos(theta), torch.sin(theta)
    cp, sp = torch.cos(phi), torch.sin(phi)

    # R_y
    Ry = torch.zeros((B, L, 3, 3), device=device, dtype=torch.float32)
    Ry[..., 0, 0] = c
    Ry[..., 0, 2] = s
    Ry[..., 1, 1] = 1.0
    Ry[..., 2, 0] = -s
    Ry[..., 2, 2] = c

    # R_x
    Rx = torch.zeros_like(Ry)
    Rx[..., 0, 0] = 1.0
    Rx[..., 1, 1] = cp
    Rx[..., 1, 2] = -sp
    Rx[..., 2, 1] = sp
    Rx[..., 2, 2] = cp

    return Ry @ Rx


def batched_project(
    v: torch.Tensor,              # [B,V,3]
    R: torch.Tensor,              # [B,L,3,3]
    cam_dist: float,
    fov: float
) -> torch.Tensor:
    """
    Returns: [B, L, V, 2] in normalized-ish coords
    """
    # rotate: [B,L,V,3] = einsum("blij, bvj -> blvi")
    p = torch.einsum("blij,bvj->blvi", R, v)
    p[..., 2] += cam_dist

    z = p[..., 2].clamp(min=1e-3)
    x = p[..., 0] / z * fov
    y = p[..., 1] / z * fov
    return torch.stack([x, y], dim=-1)


# =============================================================================
# BATCHED RASTERIZATION (Vectorized line distance field)
# =============================================================================

def batched_line_field(
    pts_px: torch.Tensor,         # [B,L,V,2] pixel coords
    edges: torch.Tensor,          # [B,L,E,2] edge indices (safe indices)
    edge_valid: torch.Tensor,     # [B,L,E] bool
    H: int,
    W: int,
    thickness: float
) -> torch.Tensor:
    """
    Returns: [B,L,H,W] float in [0,1]
    """
    device = pts_px.device
    B, L, V, _ = pts_px.shape
    E = edges.shape[2]

    # gather endpoints: [B,L,E,2]
    idx0 = edges[..., 0].clamp(0, V - 1)
    idx1 = edges[..., 1].clamp(0, V - 1)

    p0 = pts_px.gather(2, idx0[..., None].expand(-1, -1, -1, 2))
    p1 = pts_px.gather(2, idx1[..., None].expand(-1, -1, -1, 2))

    # pixel grid [H,W,2] -> [1,1,1,H,W,2]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).view(1, 1, 1, H, W, 2)

    # segment vectors
    vseg = (p1 - p0).view(B, L, E, 1, 1, 2)  # [B,L,E,1,1,2]
    p0e = p0.view(B, L, E, 1, 1, 2)
    w = grid - p0e

    denom = (vseg[..., 0] ** 2 + vseg[..., 1] ** 2).clamp(min=1e-6)  # [B,L,E,1,1]
    t = (w[..., 0] * vseg[..., 0] + w[..., 1] * vseg[..., 1]) / denom
    t = t.clamp(0.0, 1.0)

    proj = p0e + t[..., None] * vseg
    dist = torch.sqrt(((grid - proj) ** 2).sum(dim=-1) + 1e-6)  # [B,L,E,H,W]

    lines = torch.exp(-((dist / thickness) ** 2))               # [B,L,E,H,W]

    # mask invalid edges by forcing them to -inf before max
    # (so they cannot win the max)
    if edge_valid is not None:
        m = edge_valid.view(B, L, E, 1, 1)
        lines = torch.where(m, lines, torch.full_like(lines, -1e9))

    img = lines.max(dim=2).values  # [B,L,H,W]
    return img.clamp(0, 1)


# =============================================================================
# RENDER SEQUENCE (Fully batched)
# =============================================================================

def render_sequence(
    labels: torch.Tensor,  # [B] 0=cube, 1=pyramid
    L: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns: [B, L, 1, H, W]
    """
    B = labels.shape[0]
    H = W = CFG.img_size

    verts_all, edges_all, emask_all = _build_unified_shapes(device)

    # select per-sample shape: verts [B,V,3], edges [B,E,2], emask [B,E]
    verts = verts_all[labels]    # [B,V,3]
    edges = edges_all[labels]    # [B,E,2]
    emask = emask_all[labels]    # [B,E]

    # per-sample scale
    smin, smax = CFG.scale_range
    scale = (torch.rand((B, 1, 1), device=device) * (smax - smin) + smin)
    verts = verts * scale

    # rotations
    R = batched_rotations(B, L, device)  # [B,L,3,3]

    # project -> normalized coords [B,L,V,2]
    pts = batched_project(verts, R, CFG.cam_dist, CFG.fov)

    # normalized [-1,1] -> pixel coords [0..W-1], [0..H-1]
    pts01 = (pts + 1.0) * 0.5
    pts_px = pts01 * (CFG.img_size - 1)

    # jitter per-vertex per-frame
    if CFG.jitter_px > 0:
        pts_px = pts_px + torch.randn_like(pts_px) * CFG.jitter_px

    # expand edges/edge mask over L (and optionally drop edges per frame)
    E = edges.shape[1]
    edges_bl = edges[:, None, :, :].expand(B, L, E, 2).contiguous()   # [B,L,E,2]
    valid_bl = emask[:, None, :].expand(B, L, E).contiguous()         # [B,L,E]

    # edge dropout per (B,L,E)
    if CFG.dropout_prob > 0:
        keep = torch.rand((B, L, E), device=device) > CFG.dropout_prob
        valid_bl = valid_bl & keep

    img = batched_line_field(
        pts_px=pts_px,
        edges=edges_bl,
        edge_valid=valid_bl,
        H=H, W=W,
        thickness=CFG.line_thickness
    )  # [B,L,H,W]

    img = img.unsqueeze(2)  # [B,L,1,H,W]

    # noise
    if CFG.noise_std > 0:
        img = (img + torch.randn_like(img) * CFG.noise_std).clamp(0, 1)

    return img


# =============================================================================
# STREAMS (RAW / EDGES / BLUR) - Batched
# =============================================================================

def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,L,1,H,W] -> [B,L,1,H,W]
    """
    B, L, C, H, W = x.shape
    x2 = x.view(B * L, 1, H, W)

    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                      dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                      dtype=torch.float32, device=x.device).view(1,1,3,3)

    gx = F.conv2d(x2, kx, padding=1)
    gy = F.conv2d(x2, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-6).clamp(0, 1)
    return g.view(B, L, 1, H, W)


def blur_avg(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,L,1,H,W] -> [B,L,1,H,W]
    """
    B, L, C, H, W = x.shape
    x2 = x.view(B * L, 1, H, W)
    y = F.avg_pool2d(x2, kernel_size=3, stride=1, padding=1)
    return y.view(B, L, 1, H, W)


# =============================================================================
# MODEL (Batched frame encoding + batched fusion)
# =============================================================================

class TinyFrameEncoder(nn.Module):
    """
    Encodes frames [B*L,1,H,W] -> [B*L,D]
    """
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.proj(h)


class ShapeSeqFusionClassifier(nn.Module):
    """
    Encodes all frames in batch:
        x_* : [B,L,1,H,W]
    and fuses per-frame in a single batched call (B*L, D).
    """
    def __init__(self, feat_dim: int, fusion_kind: str = "gated"):
        super().__init__()
        self.feat_dim = feat_dim
        self.enc_raw = TinyFrameEncoder(feat_dim)
        self.enc_edge = TinyFrameEncoder(feat_dim)
        self.enc_blur = TinyFrameEncoder(feat_dim)

        if fusion_kind == "gated":
            self.fusion = GatedFusion("gated", num_inputs=3, in_features=feat_dim)
        elif fusion_kind == "concat":
            self.fusion = ConcatFusion("concat", num_inputs=3, in_features=feat_dim, out_features=feat_dim)
        elif fusion_kind == "sum":
            self.fusion = SumFusion("sum", num_inputs=3, in_features=feat_dim, normalize=True)
        elif fusion_kind == "adaptive":
            self.fusion = AdaptiveFusion("adaptive", num_inputs=3, in_features=feat_dim, temperature=1.5)
        elif fusion_kind == "residual_gated":
            self.fusion = ResidualFusion("resid", num_inputs=3, in_features=feat_dim, fusion_type="gated")
        else:
            raise ValueError(f"Unknown fusion_kind: {fusion_kind}")

        self.head = nn.Linear(feat_dim, 2)

    def forward(self, x_raw: torch.Tensor, x_edge: torch.Tensor, x_blur: torch.Tensor) -> torch.Tensor:
        B, L, _, H, W = x_raw.shape
        BL = B * L

        raw = x_raw.view(BL, 1, H, W)
        edg = x_edge.view(BL, 1, H, W)
        blu = x_blur.view(BL, 1, H, W)

        fr = self.enc_raw(raw)   # [BL,D]
        fe = self.enc_edge(edg)
        fb = self.enc_blur(blu)

        fused = self.fusion(fr, fe, fb)  # [BL,D]
        fused = fused.view(B, L, self.feat_dim).mean(dim=1)  # [B,D]
        return self.head(fused)  # [B,2]


# =============================================================================
# TRAIN / EVAL (with progress bars + accuracy)
# =============================================================================

def sample_batch(L: int, device: torch.device):
    y = torch.randint(0, 2, (CFG.batch_size,), device=device, dtype=torch.long)
    x = render_sequence(y, L=L, device=device)  # [B,L,1,H,W]
    xe = sobel_edges(x)
    xb = blur_avg(x)
    return x, xe, xb, y


@torch.no_grad()
def eval_acc(model: nn.Module, L: int, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    pbar = tqdm(range(CFG.eval_batches), desc=f"Eval  (views={L})", leave=False)
    for _ in pbar:
        x, xe, xb, y = sample_batch(L, device)
        logits = model(x, xe, xb)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        pbar.set_postfix(acc=f"{correct / max(1, total):.3f}")

    return correct / total


def train_steps(model: nn.Module, L: int, device: torch.device) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    acc_ema = EMAMeter(alpha=CFG.ema_alpha)

    pbar = tqdm(range(CFG.train_steps), desc=f"Train (views={L})", leave=False)
    for _ in pbar:
        x, xe, xb, y = sample_batch(L, device)
        logits = model(x, xe, xb)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean().item()
            ema = acc_ema.update(acc)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{acc:.3f}",
            ema_acc=f"{ema:.3f}",
        )


# =============================================================================
# RUNNER
# =============================================================================

def run(fusion_kind: str = "gated") -> None:
    device = torch.device(CFG.device if (CFG.device == "cuda" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(CFG.seed)

    print("\n" + "=" * 78)
    print(" SHAPE SEQ EXPERIMENT — Predict Cube vs Pyramid (BATCHED)")
    print(f" device={device} | img={CFG.img_size} | feat_dim={CFG.feat_dim} | fusion={fusion_kind}")
    print(f" context_lengths={list(CFG.context_lengths)} | train_steps={CFG.train_steps} | eval_batches={CFG.eval_batches}")
    print("=" * 78)

    outer = tqdm(CFG.context_lengths, desc="Context sweep")
    for L in outer:
        model = ShapeSeqFusionClassifier(CFG.feat_dim, fusion_kind=fusion_kind).to(device)
        train_steps(model, L, device)
        acc = eval_acc(model, L, device)
        outer.set_postfix(last_acc=f"{acc:.3f}")
        print(f"Views {L:3d} | Acc: {acc:.4f}")


if __name__ == "__main__":
    run("gated")
