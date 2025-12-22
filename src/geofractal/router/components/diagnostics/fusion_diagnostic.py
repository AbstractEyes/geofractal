"""
geofractal.router.components.diagnostics.fusion_diagnostic
========================================================

Batched multi-view shape classification experiment:
    Predict: cube vs pyramid

Purpose:
    Compare FusionComponent variants on a bounded, traceable task
    where "long context" == number of views.

Fusion variants tested:
    - gated
    - concat
    - sum
    - adaptive
    - residual_gated

Guarantees:
    - No Python loops over batch, time, frames, or edges in the hot path
    - Fully batched torch ops
    - Clean stdout, no progress bars
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    img_size: int = 64
    batch_size: int = 64

    train_steps: int = 300
    eval_batches: int = 50
    lr: float = 2e-3

    context_lengths: Tuple[int, ...] = (2, 4, 8, 16, 32)
    feat_dim: int = 128

    noise_std: float = 0.08
    line_thickness: float = 1.3
    jitter_px: float = 1.0
    dropout_prob: float = 0.10
    scale_range: Tuple[float, float] = (0.85, 1.15)

    cam_dist: float = 3.0
    fov: float = 1.2


CFG = BaseConfig()

# =============================================================================
# SHAPES (PADDED)
# =============================================================================

def build_shapes(device):
    v_cube = torch.tensor(
        [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
         [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
        device=device
    )
    e_cube = torch.tensor(
        [[0,1],[1,2],[2,3],[3,0],
         [4,5],[5,6],[6,7],[7,4],
         [0,4],[1,5],[2,6],[3,7]],
        device=device
    )

    v_pyr = torch.tensor(
        [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[0,0,1.2]],
        device=device
    )
    e_pyr = torch.tensor(
        [[0,1],[1,2],[2,3],[3,0],
         [0,4],[1,4],[2,4],[3,4]],
        device=device
    )

    V, E = 8, 12
    verts = torch.zeros((2,V,3), device=device)
    edges = torch.zeros((2,E,2), dtype=torch.long, device=device)
    mask  = torch.zeros((2,E), dtype=torch.bool, device=device)

    verts[0] = v_cube
    edges[0] = e_cube
    mask[0]  = True

    verts[1,:5] = v_pyr
    edges[1,:8] = e_pyr
    mask[1,:8]  = True

    return verts, edges, mask

# =============================================================================
# RENDERING (FULLY BATCHED)
# =============================================================================

def batched_rot(B, L, device):
    t = torch.linspace(0, 2*math.pi, L, device=device)
    t = t[None,:].expand(B,L)

    c, s = torch.cos(t), torch.sin(t)

    R = torch.zeros((B,L,3,3), device=device)
    R[...,0,0] = c
    R[...,0,2] = s
    R[...,1,1] = 1
    R[...,2,0] = -s
    R[...,2,2] = c
    return R

def project(verts, R):
    p = torch.einsum("blij,bvj->blvi", R, verts)
    p[...,2] += CFG.cam_dist
    z = p[...,2].clamp(1e-3)
    return torch.stack([p[...,0]/z*CFG.fov, p[...,1]/z*CFG.fov], -1)

def rasterize(pts, edges, mask):
    B,L,V,_ = pts.shape
    E = edges.shape[2]
    H=W=CFG.img_size

    idx0, idx1 = edges[...,0], edges[...,1]
    p0 = pts.gather(2, idx0[...,None].expand(-1,-1,-1,2))
    p1 = pts.gather(2, idx1[...,None].expand(-1,-1,-1,2))

    yy,xx = torch.meshgrid(
        torch.arange(H,device=pts.device),
        torch.arange(W,device=pts.device),
        indexing="ij"
    )
    grid = torch.stack([xx,yy],-1)[None,None,None,:,:,:]

    v = p1-p0
    w = grid - p0[...,None,None,:]
    t = (w*v[...,None,None,:]).sum(-1)/(v.pow(2).sum(-1)[...,None,None]+1e-6)
    t = t.clamp(0,1)
    proj = p0[...,None,None,:] + t[...,None]*v[...,None,None,:]
    d = ((grid-proj)**2).sum(-1).sqrt()

    lines = torch.exp(-(d/CFG.line_thickness)**2)
    lines = torch.where(mask[...,None,None], lines, -1e9)
    return lines.max(2).values

def render(labels, L, device):
    B = labels.shape[0]
    verts_all, edges_all, mask_all = build_shapes(device)

    verts = verts_all[labels]
    edges = edges_all[labels]
    mask  = mask_all[labels]

    scale = torch.rand((B,1,1),device=device)*(CFG.scale_range[1]-CFG.scale_range[0])+CFG.scale_range[0]
    verts = verts*scale

    R = batched_rot(B,L,device)
    pts = project(verts,R)
    pts = (pts+1)*0.5*(CFG.img_size-1)

    edges = edges[:,None].expand(B,L,-1,2)
    mask  = mask[:,None].expand(B,L,-1)

    img = rasterize(pts,edges,mask)
    img = img.unsqueeze(2)
    if CFG.noise_std>0:
        img = (img+torch.randn_like(img)*CFG.noise_std).clamp(0,1)
    return img

# =============================================================================
# STREAMS
# =============================================================================

def sobel(x):
    B, L, _, H, W = x.shape
    x = x.view(B * L, 1, H, W)

    k = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=x.dtype,          # ← IMPORTANT
        device=x.device
    ).view(1, 1, 3, 3)

    g = F.conv2d(x, k, padding=1)
    return g.view(B, L, 1, H, W).abs().clamp(0, 1)

def blur(x):
    B,L,_,H,W = x.shape
    x = x.view(B*L,1,H,W)
    y = F.avg_pool2d(x,3,1,1)
    return y.view(B,L,1,H,W)

# =============================================================================
# MODEL
# =============================================================================

class Encoder(nn.Module):
    def __init__(self,D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,5,2,2),nn.GELU(),
            nn.Conv2d(16,32,3,2,1),nn.GELU(),
            nn.Conv2d(32,64,3,2,1),nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(64,D)

    def forward(self,x):
        return self.proj(self.net(x).flatten(1))

class ShapeModel(nn.Module):
    def __init__(self,fusion_kind):
        super().__init__()
        D=CFG.feat_dim
        self.er = Encoder(D)
        self.ee = Encoder(D)
        self.eb = Encoder(D)

        if fusion_kind=="gated":
            self.fusion = GatedFusion("g",3,D)
        elif fusion_kind=="concat":
            self.fusion = ConcatFusion("c",3,D,D)
        elif fusion_kind=="sum":
            self.fusion = SumFusion("s",3,D)
        elif fusion_kind=="adaptive":
            self.fusion = AdaptiveFusion("a",3,D)
        elif fusion_kind=="residual_gated":
            self.fusion = ResidualFusion("r",3,D,"gated")
        else:
            raise ValueError(fusion_kind)

        self.head = nn.Linear(D,2)

    def forward(self,x,xe,xb):
        B,L,_,H,W = x.shape
        BL=B*L
        fr = self.er(x.view(BL,1,H,W))
        fe = self.ee(xe.view(BL,1,H,W))
        fb = self.eb(xb.view(BL,1,H,W))
        f  = self.fusion(fr,fe,fb)
        f  = f.view(B,L,-1).mean(1)
        return self.head(f)

# =============================================================================
# TRAIN / EVAL
# =============================================================================

def sample(L,device):
    y = torch.randint(0,2,(CFG.batch_size,),device=device)
    x = render(y,L,device)
    return x, sobel(x), blur(x), y

def train(model,L,device):
    opt = torch.optim.Adam(model.parameters(),CFG.lr)
    model.train()
    for _ in range(CFG.train_steps):
        x,xe,xb,y = sample(L,device)
        loss = F.cross_entropy(model(x,xe,xb),y)
        opt.zero_grad()
        loss.backward()
        opt.step()

@torch.no_grad()
def evaluate(model,L,device):
    model.eval()
    c=t=0
    for _ in range(CFG.eval_batches):
        x,xe,xb,y = sample(L,device)
        p = model(x,xe,xb).argmax(-1)
        c += (p==y).sum().item()
        t += y.numel()
    return c/t

# =============================================================================
# RUN ALL FUSIONS
# =============================================================================

def run_all_fusions():
    device = torch.device(CFG.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG.seed)

    fusions = ["gated","concat","sum","adaptive","residual_gated"]
    results: Dict[str,Dict[int,float]] = {}

    print("\n"+"="*90)
    print(" SHAPE SEQ EXPERIMENT — ALL FUSIONS")
    print("="*90)

    for fk in fusions:
        print("\n"+"-"*90)
        print(f" Fusion: {fk}")
        print("-"*90)
        results[fk]={}
        for L in CFG.context_lengths:
            model = ShapeModel(fk).to(device)
            train(model,L,device)
            acc = evaluate(model,L,device)
            results[fk][L]=acc
            print(f"  Views {L:3d} | Acc: {acc:.4f}")

    print("\n"+"="*90)
    print(" FINAL SUMMARY")
    print("="*90)
    header="Fusion".ljust(18)+ "".join(f"{L:>10}" for L in CFG.context_lengths)
    print(header)
    print("-"*len(header))
    for fk in fusions:
        row=fk.ljust(18)
        for L in CFG.context_lengths:
            row+=f"{results[fk][L]:10.4f}"
        print(row)

if __name__=="__main__":
    run_all_fusions()