"""
geofractal.router.prefab.agatha_v2.sphere_check
===============================================

A check that a set of generated images is not a set of centred round blobs.

For a set of RGB images in [0, 1] (square; a non-square image is cropped to its
central square), on the grayscale image g (0.299 R + 0.587 G + 0.114 B):

  (a) rotational self-similarity: per image, the mean of corr(g, rot90(g)) and
      corr(g, rot180(g)) (Pearson correlation over pixels, rotation about the image
      centre); the set value is the mean over images. a90 and a180 are kept too.
  (b) radial dominance: per image, R^2 of the best function of the distance from the
      image centre, i.e. of the ring means (rings one pixel wide):
          R^2 = 1 - sum (g - ring_mean(r))^2 / sum (g - mean(g))^2
      the set value is the mean over images.
  (c) sameness across prompts: images downsampled to s x s (area average, RGB), each
      standardized; the mean Pearson correlation over all pairs of images of different
      prompts.

A perfectly centred, frontally lit sphere reads near 1 on (a) and (b); independent
noise reads near 0 on both; a left-right gradient reads -1 on the 180-degree
correlation.

Calibration controls (render_spheres, render_blobs) and calibrate(): the per-measure
means of a set of real photos, of rendered shaded spheres on flat backgrounds (random
colours, positions near the centre) and of blurred noise blobs; the thresholds on (a)
and (b) are the midpoints between the real-photo mean and the sphere mean. A set is
sphere-like when it is above the threshold on (a) AND (b).

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

LUMA = (0.299, 0.587, 0.114)


# =============================================================================
# INPUT HANDLING
# =============================================================================

def as_images(images: Any) -> Tensor:
    """float64 [N, 3, S, S] in [0, 1] from a tensor [N, 3, H, W] (uint8 or float in [0, 1]) or
    a numpy array [N, H, W, 3] uint8; non-square images are cropped to their central square."""
    if isinstance(images, np.ndarray):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"numpy images must be [N, H, W, 3], got {images.shape}")
        images = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be [N, 3, H, W], got {tuple(images.shape)}")
    x = images.double() / 255.0 if images.dtype == torch.uint8 else images.double()
    h, w = x.shape[-2:]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return x[:, :, top:top + s, left:left + s].clamp(0.0, 1.0)


def to_gray(images: Tensor) -> Tensor:
    """[N, 3, S, S] -> [N, S, S] luminance."""
    w = torch.tensor(LUMA, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images * w).sum(dim=1)


def _corr_rows(a: Tensor, b: Tensor, eps: float = 1e-12) -> Tensor:
    """Pearson correlation of matching rows of a and b ([N, P]); 0 where a row is constant."""
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = torch.sqrt((a * a).sum(dim=1) * (b * b).sum(dim=1))
    return torch.where(den > eps, num / den.clamp_min(eps), torch.zeros_like(num))


# =============================================================================
# THE THREE MEASURES
# =============================================================================

def rotational_self_similarity(gray: Tensor) -> Dict[str, Tensor]:
    """Per-image correlations with the 90 and 180 degree rotations about the centre, [N] each."""
    n = gray.shape[0]
    flat = gray.reshape(n, -1)
    r90 = torch.rot90(gray, 1, dims=(1, 2)).reshape(n, -1)
    r180 = torch.rot90(gray, 2, dims=(1, 2)).reshape(n, -1)
    a90 = _corr_rows(flat, r90)
    a180 = _corr_rows(flat, r180)
    return {"a90": a90, "a180": a180, "a": 0.5 * (a90 + a180)}


def _ring_index(size: int, device=None) -> Tuple[Tensor, int]:
    c = (size - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float64, device=device),
                            torch.arange(size, dtype=torch.float64, device=device), indexing="ij")
    r = torch.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    idx = torch.round(r).long().reshape(-1)
    return idx, int(idx.max()) + 1


def radial_dominance(gray: Tensor, eps: float = 1e-12) -> Tensor:
    """Per-image R^2 of the ring means (rings one pixel wide around the centre), [N]."""
    n, s = gray.shape[0], gray.shape[-1]
    idx, rings = _ring_index(s, gray.device)
    flat = gray.reshape(n, -1)
    sums = torch.zeros(n, rings, dtype=flat.dtype, device=flat.device).index_add_(1, idx, flat)
    counts = torch.bincount(idx, minlength=rings).to(flat.dtype)
    fitted = (sums / counts.clamp_min(1.0))[:, idx]
    ss_res = ((flat - fitted) ** 2).sum(dim=1)
    ss_tot = ((flat - flat.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
    return torch.where(ss_tot > eps, 1.0 - ss_res / ss_tot.clamp_min(eps), torch.zeros_like(ss_tot))


def sameness_across_prompts(images: Tensor, size: int = 32, prompt_ids: Optional[Sequence[int]] = None,
                            eps: float = 1e-12) -> Dict[str, float]:
    """Mean Pearson correlation of downsampled RGB images over pairs of different prompts."""
    n = images.shape[0]
    if n < 2:
        return {"c": float("nan"), "pairs": 0}
    small = F.adaptive_avg_pool2d(images, (size, size)).reshape(n, -1)
    z = small - small.mean(dim=1, keepdim=True)
    norm = torch.sqrt((z * z).sum(dim=1, keepdim=True))
    z = torch.where(norm > eps, z / norm.clamp_min(eps), torch.zeros_like(z))
    corr = z @ z.T
    ids = torch.arange(n) if prompt_ids is None else torch.as_tensor(list(prompt_ids))
    if ids.numel() != n:
        raise ValueError("prompt_ids must give one id per image")
    other = ids[:, None] != ids[None, :]
    pairs = int(other.sum()) // 2
    if pairs == 0:
        return {"c": float("nan"), "pairs": 0}
    return {"c": float(corr[other].mean()), "pairs": pairs}


def measure(images: Any, prompt_ids: Optional[Sequence[int]] = None, sameness_size: int = 32,
            per_image: bool = False) -> Dict[str, Any]:
    """The set means of (a), its two parts, (b) and (c); per-image values on request."""
    x = as_images(images)
    g = to_gray(x)
    rot = rotational_self_similarity(g)
    rad = radial_dominance(g)
    same = sameness_across_prompts(x, sameness_size, prompt_ids)
    out: Dict[str, Any] = {"n": int(x.shape[0]), "size": int(x.shape[-1]),
                           "a": float(rot["a"].mean()), "a90": float(rot["a90"].mean()),
                           "a180": float(rot["a180"].mean()), "b": float(rad.mean()),
                           "c": same["c"], "c_pairs": same["pairs"]}
    if per_image:
        out["per_image"] = {"a": rot["a"].tolist(), "a90": rot["a90"].tolist(), "a180": rot["a180"].tolist(),
                            "b": rad.tolist()}
    return out


def is_sphere_like(measures: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    """Above the threshold on (a) AND on (b)."""
    return bool(measures["a"] > thresholds["a"] and measures["b"] > thresholds["b"])


# =============================================================================
# CALIBRATION CONTROLS
# =============================================================================

def _grid(size: int) -> Tuple[Tensor, Tensor]:
    yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float64), torch.arange(size, dtype=torch.float64),
                            indexing="ij")
    return yy, xx


def render_spheres(n: int, size: int = 256, seed: int = 0, radius: Tuple[float, float] = (0.18, 0.38),
                   jitter: float = 0.08, light: Optional[Sequence[float]] = None,
                   ambient: float = 0.2, specular: float = 0.35) -> Tensor:
    """Shaded spheres on flat backgrounds, float [n, 3, size, size] in [0, 1].

    Random background and sphere colours; radius a random fraction of the side in `radius`;
    centre within +-jitter * size of the image centre; Lambertian shading with an ambient
    term and a specular highlight, light direction random over the front hemisphere unless
    `light` is given (light=(0, 0, 1) with jitter=0 gives a radially symmetric sphere). The
    edge is antialiased over one pixel.
    """
    g = torch.Generator().manual_seed(int(seed))

    def u(lo: float, hi: float, k: int = n) -> Tensor:
        return lo + (hi - lo) * torch.rand(k, generator=g, dtype=torch.float64)

    yy, xx = _grid(size)
    c0 = (size - 1) / 2.0
    bg = torch.rand(n, 3, generator=g, dtype=torch.float64)
    col = torch.rand(n, 3, generator=g, dtype=torch.float64)
    rad = u(*radius) * size
    cx = c0 + u(-jitter, jitter) * size
    cy = c0 + u(-jitter, jitter) * size
    if light is None:
        lx, ly = u(-0.7, 0.7), u(-0.7, 0.7)
        lz = torch.ones(n, dtype=torch.float64)
    else:
        lx, ly, lz = (torch.full((n,), float(v), dtype=torch.float64) for v in light)
    ln = torch.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
    lx, ly, lz = lx / ln, ly / ln, lz / ln
    dx = (xx[None] - cx[:, None, None]) / rad[:, None, None]
    dy = (yy[None] - cy[:, None, None]) / rad[:, None, None]
    rr = dx ** 2 + dy ** 2
    nz = torch.sqrt((1.0 - rr).clamp_min(0.0))
    lam = (dx * lx[:, None, None] - dy * ly[:, None, None] + nz * lz[:, None, None]).clamp_min(0.0)
    # specular: reflection of the light about the normal, against the view direction (0, 0, 1)
    ndl = lam
    rz = 2.0 * ndl * nz - lz[:, None, None]
    spec = specular * rz.clamp_min(0.0) ** 24
    shade = ambient + (1.0 - ambient) * lam
    sphere = (col[:, :, None, None] * shade[:, None] + spec[:, None]).clamp(0.0, 1.0)
    alpha = ((1.0 - torch.sqrt(rr)) * rad[:, None, None] + 0.5).clamp(0.0, 1.0)[:, None]
    return (alpha * sphere + (1.0 - alpha) * bg[:, :, None, None]).float()


def render_blobs(n: int, size: int = 256, seed: int = 1, sigma: Tuple[float, float] = (0.12, 0.28),
                 jitter: float = 0.08, texture: float = 0.35, blur_cells: int = 8) -> Tensor:
    """Blurred noise blobs, float [n, 3, size, size] in [0, 1]: a soft round envelope (Gaussian
    in the distance from a centre near the image centre) of a random colour with a smooth noise
    texture (noise on a blur_cells x blur_cells grid, bicubic upsampled), on a flat background."""
    g = torch.Generator().manual_seed(int(seed))

    def u(lo: float, hi: float) -> Tensor:
        return lo + (hi - lo) * torch.rand(n, generator=g, dtype=torch.float64)

    yy, xx = _grid(size)
    c0 = (size - 1) / 2.0
    bg = torch.rand(n, 3, generator=g, dtype=torch.float64)
    col = torch.rand(n, 3, generator=g, dtype=torch.float64)
    sig = u(*sigma) * size
    cx = c0 + u(-jitter, jitter) * size
    cy = c0 + u(-jitter, jitter) * size
    r2 = (xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2
    env = torch.exp(-r2 / (2.0 * sig[:, None, None] ** 2))[:, None]
    coarse = torch.randn(n, 3, blur_cells, blur_cells, generator=g, dtype=torch.float64)
    smooth = F.interpolate(coarse, size=(size, size), mode="bicubic", align_corners=False)
    blob = col[:, :, None, None] + texture * smooth
    return (env * blob + (1.0 - env) * bg[:, :, None, None]).clamp(0.0, 1.0).float()


def calibrate(real: Any, n_controls: int = 256, seed: int = 0, spheres: Any = None, blobs: Any = None,
              size: Optional[int] = None) -> Dict[str, Any]:
    """Means of the real set, the sphere set and the blob set; thresholds on (a) and (b) at the
    midpoints between the real-photo mean and the sphere mean. `valid` is True when the sphere
    set reads above the real set on both (a) and (b), so the thresholds separate them."""
    real_x = as_images(real)
    s = int(size or real_x.shape[-1])
    sph = render_spheres(n_controls, s, seed=seed) if spheres is None else spheres
    blb = render_blobs(n_controls, s, seed=seed + 1) if blobs is None else blobs
    m_real, m_sph, m_blob = measure(real_x), measure(sph), measure(blb)
    thresholds = {"a": 0.5 * (m_real["a"] + m_sph["a"]), "b": 0.5 * (m_real["b"] + m_sph["b"])}
    valid = bool(m_sph["a"] > m_real["a"] and m_sph["b"] > m_real["b"])
    return {"real": m_real, "sphere": m_sph, "blob": m_blob, "thresholds": thresholds, "valid": valid,
            "flags": {"real": is_sphere_like(m_real, thresholds), "sphere": is_sphere_like(m_sph, thresholds),
                      "blob": is_sphere_like(m_blob, thresholds)},
            "rule": "sphere-like = a above thresholds['a'] AND b above thresholds['b']"}
