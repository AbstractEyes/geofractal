"""
geofractal.router.prefab.agatha_v2.metrics
==========================================

Image metrics for evaluation reads. The heavy packages are imported only when a real
feature extractor or judge is built, so the plumbing runs (and is tested) without them.

FID and KID (FidKid). Features of real and generated images come from a feature
extractor (a callable: uint8 images [n, 3, h, w] -> features [n, d]); from them, in
float64,
    FID = |mu_r - mu_g|^2 + tr(S_r) + tr(S_g) - 2 sum_i sqrt(lambda_i(S_r S_g))
          (unbiased covariances; the square roots of the eigenvalues of the product)
    KID = mean and population std over `subsets` random subsets of `subset_size`
          features of the unbiased MMD^2 with the kernel k(x, y) = (x . y / d + 1)^3
These are the formulas of torchmetrics 1.9.0 (FrechetInceptionDistance and
KernelInceptionDistance; 100 subsets of 1000 by default); here the subsets are drawn
from a seeded generator. The standard extractor, built by inception_extractor(), is the
one torchmetrics uses: the 2048-d pool features of the FID Inception network
(torchmetrics.image.fid.NoTrainInceptionV3 on top of torch-fidelity 0.4.0; uint8
input, resized internally). torchmetrics_fid_kid() computes torchmetrics' own values
on the same images, so the two can be compared once where the packages are installed.
Images enter as float in [0, 1] (rounded to uint8) or as uint8. The reference
statistics are computed once and saved with save_reference() / load_reference().

CLIP (ClipJudge; openai/clip-vit-large-patch14 through transformers by default; any
image and text embedders can be passed in instead):
    CLIP score       100 * max(cos(E_image, E_text), 0) per image and caption (the
                     torchmetrics CLIPScore definition); mean and standard error
    grounding        for samples made with the same noise from an item's own caption
                     (own) and from another item's caption (shuffled), and the item's
                     real photo: g_i = cos(E(own_i), E(photo_i)) - cos(E(shuffled_i), E(photo_i));
                     mean and standard error. Identical own and shuffled images give
                     exactly 0 (they are embedded in the same batch layout).

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor

FEATURE_PACKAGES = "torchmetrics==1.9.0 and torch-fidelity==0.4.0"
CLIP_MODEL = "openai/clip-vit-large-patch14"


def to_uint8(images: Tensor) -> Tensor:
    """uint8 [n, 3, h, w]: float images in [0, 1] are clamped, scaled by 255 and rounded."""
    if images.dtype == torch.uint8:
        return images
    return (images.float().clamp(0, 1) * 255.0).round().to(torch.uint8)


def to_unit(images: Tensor) -> Tensor:
    """float [n, 3, h, w] in [0, 1] from uint8 or float images."""
    if images.dtype == torch.uint8:
        return images.float() / 255.0
    return images.float().clamp(0, 1)


# =============================================================================
# FID / KID
# =============================================================================

def gaussian_stats(features: Tensor):
    """(mean [d], unbiased covariance [d, d]) in float64."""
    x = features.double()
    n = x.shape[0]
    if n < 2:
        raise ValueError("at least two feature vectors are needed")
    mu = x.mean(dim=0)
    xc = x - mu
    return mu, xc.T @ xc / (n - 1)


def frechet_distance(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> float:
    a = (mu1 - mu2).square().sum()
    b = torch.trace(sigma1) + torch.trace(sigma2)
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum()
    return float(a + b - 2.0 * c)


def _poly_kernel(f1: Tensor, f2: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0) -> Tensor:
    gamma = 1.0 / f1.shape[1] if gamma is None else gamma
    return (f1 @ f2.T * gamma + coef) ** degree


def _unbiased_mmd2(k_xx: Tensor, k_xy: Tensor, k_yy: Tensor) -> Tensor:
    m = k_xx.shape[0]
    kt_xx = k_xx.sum() - torch.diagonal(k_xx).sum()
    kt_yy = k_yy.sum() - torch.diagonal(k_yy).sum()
    return (kt_xx + kt_yy) / (m * (m - 1)) - 2.0 * k_xy.sum() / (m * m)


def kernel_inception_distance(f_real: Tensor, f_fake: Tensor, subsets: int = 100, subset_size: int = 1000,
                              seed: int = 0, degree: int = 3, coef: float = 1.0) -> Dict[str, float]:
    """KID mean and population std over random subsets (seeded); the subset size is cut to the
    smaller set when a set is smaller than `subset_size` (reported as subset_size_used)."""
    fr, ff = f_real.double(), f_fake.double()
    m = int(min(subset_size, fr.shape[0], ff.shape[0]))
    if m < 2:
        raise ValueError("KID needs at least two features in each set")
    g = torch.Generator().manual_seed(int(seed))
    scores = []
    for _ in range(int(subsets)):
        a = fr[torch.randperm(fr.shape[0], generator=g)[:m]]
        b = ff[torch.randperm(ff.shape[0], generator=g)[:m]]
        scores.append(_unbiased_mmd2(_poly_kernel(a, a, degree, None, coef), _poly_kernel(a, b, degree, None, coef),
                                     _poly_kernel(b, b, degree, None, coef)))
    s = torch.stack(scores)
    return {"kid_mean": float(s.mean()), "kid_std": float(s.std(unbiased=False)), "subset_size_used": m,
            "subsets": int(subsets)}


def inception_extractor(device: str = "cuda") -> Callable[[Tensor], Tensor]:
    """The FID Inception network's 2048-d pool features (torchmetrics' NoTrainInceptionV3).
    Imported here, on first use; needs torchmetrics with torch-fidelity installed."""
    try:
        from torchmetrics.image.fid import NoTrainInceptionV3
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(f"the standard feature extractor needs {FEATURE_PACKAGES}: {e}") from e
    net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()

    @torch.no_grad()
    def extract(x: Tensor) -> Tensor:
        out = net(x.to(device))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(x.shape[0], -1)

    return extract


class FidKid:
    """FID and KID against fixed reference statistics; reset() / update(images) / compute().

    extractor: callable uint8 [n, 3, h, w] -> [n, d]; None builds inception_extractor(device)
    on first use. The reference keeps its mean, covariance and features (KID needs them).
    """

    def __init__(self, extractor: Optional[Callable[[Tensor], Tensor]] = None, device: str = "cpu",
                 kid_subsets: int = 100, kid_subset_size: int = 1000, seed: int = 0, batch_size: int = 200):
        self._extractor = extractor
        self.device = device
        self.kid_subsets, self.kid_subset_size = int(kid_subsets), int(kid_subset_size)
        self.seed, self.batch_size = int(seed), int(batch_size)
        self.reference: Optional[Dict[str, Any]] = None
        self._fake: List[Tensor] = []

    @property
    def extractor(self) -> Callable[[Tensor], Tensor]:
        if self._extractor is None:
            self._extractor = inception_extractor(self.device)
        return self._extractor

    @torch.no_grad()
    def features(self, images: Tensor) -> Tensor:
        """float32 features [n, d] on the CPU, computed in batches."""
        x = to_uint8(images)
        out = [self.extractor(x[i:i + self.batch_size]).float().cpu() for i in range(0, x.shape[0], self.batch_size)]
        return torch.cat(out)

    # ---- reference
    def set_reference_features(self, features: Tensor, note: str = "") -> Dict[str, Any]:
        mu, sigma = gaussian_stats(features)
        self.reference = {"mu": mu, "sigma": sigma, "features": features.float().cpu(),
                          "n": int(features.shape[0]), "note": note}
        return {"n": self.reference["n"], "dim": int(features.shape[1]), "note": note}

    def set_reference_images(self, batches: Iterable[Tensor], note: str = "") -> Dict[str, Any]:
        return self.set_reference_features(torch.cat([self.features(b) for b in batches]), note)

    def save_reference(self, path: str) -> None:
        if self.reference is None:
            raise RuntimeError("no reference set")
        torch.save(self.reference, path)

    def load_reference(self, path: str) -> Dict[str, Any]:
        ref = torch.load(path, map_location="cpu", weights_only=True)
        self.reference = ref
        return {"n": int(ref["n"]), "dim": int(ref["features"].shape[1]), "note": ref.get("note", "")}

    # ---- generated images
    def reset(self) -> None:
        self._fake = []

    def update(self, images: Tensor) -> None:
        self._fake.append(self.features(images))

    def update_features(self, features: Tensor) -> None:
        self._fake.append(features.float().cpu())

    def compute(self) -> Dict[str, Any]:
        if self.reference is None:
            raise RuntimeError("no reference set")
        if not self._fake:
            raise RuntimeError("no generated images")
        fake = torch.cat(self._fake)
        mu, sigma = gaussian_stats(fake)
        out = {"fid": frechet_distance(self.reference["mu"], self.reference["sigma"], mu, sigma),
               "n_real": int(self.reference["n"]), "n_fake": int(fake.shape[0])}
        out.update(kernel_inception_distance(self.reference["features"], fake, self.kid_subsets,
                                             self.kid_subset_size, self.seed))
        return out


def torchmetrics_fid_kid(real_batches: Iterable[Tensor], fake_batches: Iterable[Tensor], device: str = "cuda",
                         kid_subset_size: int = 1000, kid_subsets: int = 100) -> Dict[str, float]:
    """torchmetrics' own FID and KID (feature 2048) on uint8 batches, for a one-off comparison."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(feature=2048, subsets=kid_subsets, subset_size=kid_subset_size,
                                  normalize=False).to(device)
    for b in real_batches:
        x = to_uint8(b).to(device)
        fid.update(x, real=True)
        kid.update(x, real=True)
    for b in fake_batches:
        x = to_uint8(b).to(device)
        fid.update(x, real=False)
        kid.update(x, real=False)
    k_mean, k_std = kid.compute()
    return {"fid": float(fid.compute()), "kid_mean": float(k_mean), "kid_std": float(k_std)}


# =============================================================================
# CLIP
# =============================================================================

def _mean_se(values: Tensor) -> Dict[str, float]:
    v = values.double()
    n = v.numel()
    se = float(v.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(v.mean()), "se": se, "n": int(n)}


def _unit_rows(e: Tensor) -> Tensor:
    e = e.double()
    return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class ClipJudge:
    """CLIP embeddings, the CLIP score and the paired grounding read.

    image_embedder: callable float images [n, 3, h, w] in [0, 1] -> [n, d];
    text_embedder: callable list of str -> [n, d]. When both are None the CLIP model
    `model_name` is loaded through transformers on first use (image tower + projection,
    text tower + projection). Embeddings are returned L2-normalized, float64, on the CPU.
    """

    def __init__(self, model_name: str = CLIP_MODEL, device: str = "cpu",
                 image_embedder: Optional[Callable[[Tensor], Tensor]] = None,
                 text_embedder: Optional[Callable[[List[str]], Tensor]] = None, batch_size: int = 64):
        self.model_name, self.device, self.batch_size = model_name, device, int(batch_size)
        self._image_embedder, self._text_embedder = image_embedder, text_embedder
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_name)

    @torch.no_grad()
    def _clip_images(self, images: Tensor) -> Tensor:
        self._load()
        arrs = [a for a in to_uint8(images).permute(0, 2, 3, 1).contiguous().cpu().numpy()]
        pix = self._processor.image_processor(images=arrs, return_tensors="pt")["pixel_values"].to(self.device)
        out = self._model.vision_model(pixel_values=pix)
        pooled = out.pooler_output if hasattr(out, "pooler_output") else out[1]
        return self._model.visual_projection(pooled)

    @torch.no_grad()
    def _clip_texts(self, captions: List[str]) -> Tensor:
        self._load()
        tok = self._processor.tokenizer(list(captions), padding=True, truncation=True, max_length=77,
                                        return_tensors="pt").to(self.device)
        out = self._model.text_model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        pooled = out.pooler_output if hasattr(out, "pooler_output") else out[1]
        return self._model.text_projection(pooled)

    def embed_images(self, images: Tensor) -> Tensor:
        fn = self._image_embedder or self._clip_images
        x = to_unit(images)
        return torch.cat([_unit_rows(fn(x[i:i + self.batch_size]).float().cpu())
                          for i in range(0, x.shape[0], self.batch_size)])

    def embed_texts(self, captions: Sequence[str]) -> Tensor:
        fn = self._text_embedder or self._clip_texts
        caps = list(captions)
        return torch.cat([_unit_rows(fn(caps[i:i + self.batch_size]).float().cpu())
                          for i in range(0, len(caps), self.batch_size)])

    # ---- reads from embeddings
    @staticmethod
    def clip_score_from_embeddings(e_images: Tensor, e_texts: Tensor) -> Dict[str, float]:
        cos = (_unit_rows(e_images) * _unit_rows(e_texts)).sum(dim=-1)
        return _mean_se(100.0 * cos.clamp_min(0.0))

    @staticmethod
    def grounding_from_embeddings(e_own: Tensor, e_shuffled: Tensor, e_photos: Tensor) -> Dict[str, float]:
        p = _unit_rows(e_photos)
        c_own = (_unit_rows(e_own) * p).sum(dim=-1)
        c_shuf = (_unit_rows(e_shuffled) * p).sum(dim=-1)
        out = _mean_se(c_own - c_shuf)
        out.update(own_cos=float(c_own.mean()), shuffled_cos=float(c_shuf.mean()))
        return out

    # ---- reads from images
    def clip_score(self, images: Tensor, captions: Sequence[str]) -> Dict[str, float]:
        return self.clip_score_from_embeddings(self.embed_images(images), self.embed_texts(captions))

    def paired_grounding(self, own: Tensor, shuffled: Tensor, photos: Tensor) -> Dict[str, float]:
        if not (own.shape[0] == shuffled.shape[0] == photos.shape[0]):
            raise ValueError("own, shuffled and photos must have the same length")
        return self.grounding_from_embeddings(self.embed_images(own), self.embed_images(shuffled),
                                              self.embed_images(photos))
