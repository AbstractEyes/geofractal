# ============================================================
# CIFAR100: Bi-directional Opinions -> Oscillator Denoiser + Classifier
# ============================================================
# Goal: teach an oscillation system to align a noisy latent to a clean latent
# using dual-opinion tower signals produced by a tiny transformer.
#
# Loss:
#   L = mse(x_hat, x_ref) + ce(classifier(x_hat), y)
#
# This is the minimal stepping stone toward diffusion/flow teaching.
# ============================================================

from dataclasses import dataclass
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -------------------------
# Config
# -------------------------
@dataclass
class BaseConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # data
    batch_size: int = 256
    num_workers: int = 4

    # latent/dynamics
    latent_dim: int = 512
    towers: int = 4              # number of +/- pairs
    steps: int = 8               # oscillator integration steps
    dt: float = 1.0 / 8

    # dynamics scalars
    beta: float = 0.20           # damping
    omega: float = 1.00          # spring to x_ref
    kappa: float = 1.00          # tower gain
    gamma: float = 0.00          # (optional) external guidance

    # noise schedule
    sigma_min: float = 0.05
    sigma_max: float = 1.00

    # model sizes
    op_hidden: int = 512
    op_layers: int = 4
    op_heads: int = 8

    # train
    lr: float = 3e-4
    epochs: int = 10
    denoise_weight: float = 1.0
    cls_weight: float = 1.0

cfg = BaseConfig()


# -------------------------
# Utils
# -------------------------
def sample_sigma(B, device):
    # log-uniform noise like diffusion
    u = torch.rand(B, device=device)
    log_s = math.log(cfg.sigma_min) + u * (math.log(cfg.sigma_max) - math.log(cfg.sigma_min))
    return torch.exp(torch.tensor(log_s, device=device))

def timestep_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    # sinusoidal embedding, t in [0,1]
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=t.device).float() / half)
    ang = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    return emb if dim % 2 == 0 else torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)


# -------------------------
# Image -> latent teacher target
# -------------------------
class TinyEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),   # 16x16
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(),  # 8x8
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.GELU(),  # 4x4
        )
        self.proj = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.flatten(1)
        return self.proj(h)


# -------------------------
# Opinion Transformer: (x_noisy, t) -> (p_i, n_i) for i in towers
# -------------------------
class OpinionTransformer(nn.Module):
    def __init__(self, latent_dim: int, towers: int, hidden: int, layers: int, heads: int):
        super().__init__()
        self.towers = towers
        self.in_proj = nn.Linear(latent_dim + hidden, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)

        # we produce 2*towers vectors in latent space: p0,n0,p1,n1,...
        self.out = nn.Linear(hidden, (2 * towers) * latent_dim)

        self.t_hidden = hidden

    def forward(self, x_noisy: torch.Tensor, t01: torch.Tensor) -> torch.Tensor:
        # treat latent as a "single token" sequence for now (simple + fast)
        t_emb = timestep_embed(t01, self.t_hidden).to(x_noisy.dtype)
        h = torch.cat([x_noisy, t_emb], dim=-1)
        tok = self.in_proj(h).unsqueeze(1)   # [B,1,H]
        tok = self.tr(tok)                   # [B,1,H]
        out = self.out(tok[:, 0])            # [B, 2*towers*D]
        return out.view(x_noisy.size(0), 2 * self.towers, x_noisy.size(1))  # [B,2T,D]


# -------------------------
# Oscillator Core: differentiable dual-opinion aligner
# -------------------------
class OscillatorCore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_init, x_ref, tower_pairs, t01):
        B, D = x_init.shape
        device = x_init.device
        dtype = x_init.dtype
        eps = 1e-6

        beta = cfg.beta
        omega = cfg.omega
        kappa = cfg.kappa

        # -------------------------
        # Cache tower deltas & norms ONCE
        # -------------------------
        if tower_pairs:
            deltas = torch.stack([p - n for (p, n) in tower_pairs], dim=1)  # [B,T,D]
            delta_norm = deltas.norm(dim=-1, keepdim=True) + eps
            deltas_hat = deltas / delta_norm                                # [B,T,D]
        else:
            deltas_hat = None

        # -------------------------
        # Init state
        # -------------------------
        x = x_init
        v = torch.zeros_like(x)

        # -------------------------
        # Integrate
        # -------------------------
        for _ in range(cfg.steps):
            # Desired direction (cached per step only)
            u = x_ref - x
            u_norm = u.norm(dim=-1, keepdim=True) + eps
            u_hat = u / u_norm

            # Spring + damping
            spring = (omega ** 2) * u
            damp = -beta * v

            # -------------------------
            # Cached dual-opinion control
            # -------------------------
            if deltas_hat is not None:
                # cosine similarity [B,T]
                cos = (deltas_hat * u_hat.unsqueeze(1)).sum(dim=-1)

                # effectiveness gate
                eff = torch.sigmoid(6.0 * cos)

                # projection along trajectory [B,T,1] * [B,1,D]
                proj = cos.unsqueeze(-1) * u_hat.unsqueeze(1)

                # sum forces
                control = (proj * eff.unsqueeze(-1)).sum(dim=1)
            else:
                control = torch.zeros_like(x)

            # Acceleration
            a = damp + spring + (kappa * control)

            # Integrate
            v = v + cfg.dt * a
            x = x + cfg.dt * v

        return x


# -------------------------
# Full model: encode -> noisy -> opinions -> oscillator -> heads
# -------------------------
class CIFAR100OscModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = TinyEncoder(cfg.latent_dim)
        self.op = OpinionTransformer(cfg.latent_dim, cfg.towers, cfg.op_hidden, cfg.op_layers, cfg.op_heads)
        self.osc = OscillatorCore()
        self.cls = nn.Linear(cfg.latent_dim, 100)

    def forward(self, img, y):
        x_ref = self.enc(img)  # clean latent target

        B = x_ref.size(0)
        sigma = sample_sigma(B, x_ref.device).to(x_ref.dtype)
        t01 = (sigma - cfg.sigma_min) / (cfg.sigma_max - cfg.sigma_min)
        t01 = t01.clamp(0, 1)

        noise = torch.randn_like(x_ref)
        x_noisy = x_ref + sigma.unsqueeze(-1) * noise

        # opinions
        opinions = self.op(x_noisy, t01)  # [B,2T,D]
        tower_pairs = [(opinions[:, 2*i], opinions[:, 2*i + 1]) for i in range(cfg.towers)]

        # align
        x_hat = self.osc(x_noisy, x_ref, tower_pairs, t01)

        # losses
        logits = self.cls(x_hat)
        loss_cls = F.cross_entropy(logits.float(), y)
        loss_den = F.mse_loss(x_hat.float(), x_ref.float())

        return (cfg.cls_weight * loss_cls) + (cfg.denoise_weight * loss_den), loss_cls.detach(), loss_den.detach(), logits.detach()


# -------------------------
# Train
# -------------------------
def main():
    device = torch.device(cfg.device)

    tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR100(root="./data", train=True, download=True, transform=tf)
    test  = datasets.CIFAR100(root="./data", train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = CIFAR100OscModel().to(device=device).to(dtype=cfg.dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot, totc, totd, correct, seen = 0.0, 0.0, 0.0, 0, 0

        for img, y in train_loader:
            img = img.to(device=device, dtype=cfg.dtype, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            loss, lc, ld, logits = model(img, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot += float(loss.item())
            totc += float(lc.item())
            totd += float(ld.item())

            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            seen += y.numel()

        train_acc = 100.0 * correct / max(1, seen)

        # eval
        model.eval()
        correct, seen = 0, 0
        with torch.no_grad():
            for img, y in test_loader:
                img = img.to(device=device, dtype=cfg.dtype, non_blocking=True)
                y = y.to(device=device, non_blocking=True)

                x_ref = model.enc(img)
                logits = model.cls(x_ref)  # cheap baseline: classifier on clean latent
                pred = logits.argmax(dim=-1)
                correct += int((pred == y).sum().item())
                seen += y.numel()

        test_acc_clean_latent = 100.0 * correct / max(1, seen)

        print(
            f"Epoch {epoch:02d} | loss {tot/len(train_loader):.4f} "
            f"(cls {totc/len(train_loader):.4f}, den {totd/len(train_loader):.4f}) "
            f"| train_acc {train_acc:.2f}% | test_acc(clean-latent) {test_acc_clean_latent:.2f}%"
        )

if __name__ == "__main__":
    main()
