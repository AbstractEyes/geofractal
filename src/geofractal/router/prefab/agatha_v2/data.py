"""
geofractal.router.prefab.agatha_v2.data
=======================================

Data over cached tensors. The model never holds a VAE or a text encoder:
latents and text states are computed once, offline, by build_cache(), and
read back by CachedLatentTextDataset.

Cache layout (one directory):
    meta.pt                {'latent_shape', 'text_dim', 'max_text_len', 'num_items',
                            'shards', 'null_len', 'text_encoder', 'vae'}
    null_text.pt           [null_len, text_dim] states of the empty caption
    shard_00000.pt ...     {'latents': [n, C, H, W] fp16,
                            'text': [n, L, text_dim] fp16 (zero padded),
                            'mask': [n, L] bool}
Tensors are stored in float16 and cast to float32 when read.

build_cache() takes the encoders as callables, so this module imports no
encoder library:
    encode_images(list_of_items) -> latents [n, C, H, W]
    encode_texts(list_of_str)    -> (states [n, L, text_dim], mask [n, L] bool)
For example, encode_texts can wrap a frozen T5 encoder's last hidden states
and attention mask; encode_images can wrap a VAE encoder's (scaled) latent.

The memory-mapped cache (build_memmap_cache, MemmapLatentTextDataset) is the form
for large sets: one flat file per field, random access with no decoding.
    latents.bin          [N, C, H, W] float16, raw, image order
    text_tokens.bin      [T, D] float16, raw: every caption's token states, true
                         length (no padding), caption after caption
    text_offsets.npy     int64 [M + 1]: caption j is rows text_offsets[j]:text_offsets[j + 1]
    caption_image.npy    int64 [M]: the image of caption j
    image_captions.npy   int64 [N + 1]: the captions of image i are image_captions[i]:image_captions[i + 1]
    images.jsonl         one line per image: index, id, file name, license id, caption count
    captions.jsonl       one line per caption: index, image index, id, text
    null_text.npy        float16 [null_len, D]: the states of the empty caption
    meta.json            counts, shapes, dtypes, per-channel latent mean and std, the null
                         caption's length, encoder and VAE names and revisions, preprocessing
                         (written last: a cache without it is incomplete)
StepOrderSampler gives the (image, caption) pairs of every global step from a seed (a
fresh permutation of the images per epoch, one caption drawn per image and epoch), so a
run resumed at step k reads exactly what an unbroken run reads from step k on.
collate_latent_text pads the token states to the longest caption of the batch.

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

EncodeImages = Callable[[List], Tensor]
EncodeTexts = Callable[[List[str]], Tuple[Tensor, Tensor]]


def _pad_text(states: Tensor, mask: Tensor, length: int) -> Tuple[Tensor, Tensor]:
    n, l, d = states.shape
    if l > length:
        raise ValueError(f"text length {l} exceeds max_text_len {length}")
    if l == length:
        return states, mask
    pad = length - l
    return (torch.cat([states, states.new_zeros(n, pad, d)], dim=1),
            torch.cat([mask, mask.new_zeros(n, pad)], dim=1))


@torch.no_grad()
def build_cache(
    items: Iterable[Tuple[object, str]],
    encode_images: EncodeImages,
    encode_texts: EncodeTexts,
    out_dir: str,
    max_text_len: int = 128,
    batch_size: int = 32,
    shard_size: int = 4096,
    text_encoder: str = "t5-base",
    vae: str = "flux2-vae",
    progress: Optional[Callable[[int], None]] = None,
) -> Dict:
    """Encode (image, caption) pairs once and write shards. Returns the meta dict.

    The caption is encoded exactly as given (no rewriting).
    """
    os.makedirs(out_dir, exist_ok=True)
    null_states, null_mask = encode_texts([""])
    null_len = int(null_mask[0].sum())
    torch.save(null_states[0, :null_len].to(torch.float16).clone(), os.path.join(out_dir, "null_text.pt"))

    buf_lat: List[Tensor] = []
    buf_txt: List[Tensor] = []
    buf_msk: List[Tensor] = []
    shards: List[str] = []
    count = 0
    latent_shape = None
    text_dim = None

    def flush():
        if not buf_lat:
            return
        name = f"shard_{len(shards):05d}.pt"
        torch.save({"latents": torch.cat(buf_lat), "text": torch.cat(buf_txt), "mask": torch.cat(buf_msk)},
                   os.path.join(out_dir, name))
        shards.append(name)
        buf_lat.clear(); buf_txt.clear(); buf_msk.clear()

    batch: List[Tuple[object, str]] = []

    def run(batch_items):
        nonlocal count, latent_shape, text_dim
        imgs = [b[0] for b in batch_items]
        caps = [b[1] for b in batch_items]
        lat = encode_images(imgs).float().cpu()
        st, mk = encode_texts(caps)
        st, mk = _pad_text(st.float().cpu(), mk.bool().cpu(), max_text_len)
        latent_shape = tuple(lat.shape[1:])
        text_dim = st.shape[-1]
        buf_lat.append(lat.to(torch.float16))
        buf_txt.append(st.to(torch.float16))
        buf_msk.append(mk)
        count += lat.shape[0]
        if progress is not None:
            progress(count)
        if sum(x.shape[0] for x in buf_lat) >= shard_size:
            flush()

    for it in items:
        batch.append(it)
        if len(batch) == batch_size:
            run(batch)
            batch = []
    if batch:
        run(batch)
    flush()

    meta = {"latent_shape": latent_shape, "text_dim": text_dim, "max_text_len": max_text_len,
            "num_items": count, "shards": shards, "null_len": null_len,
            "text_encoder": text_encoder, "vae": vae}
    torch.save(meta, os.path.join(out_dir, "meta.pt"))
    return meta


class CachedLatentTextDataset(Dataset):
    """Reads a cache written by build_cache(). Items: (latent [C,H,W], text [L,D], mask [L]) in float32."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.meta = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=True)
        lat, txt, msk = [], [], []
        for name in self.meta["shards"]:
            s = torch.load(os.path.join(cache_dir, name), weights_only=True)
            lat.append(s["latents"]); txt.append(s["text"]); msk.append(s["mask"])
        self.latents = torch.cat(lat)
        self.text = torch.cat(txt)
        self.mask = torch.cat(msk)

    def null_text(self) -> Tensor:
        return torch.load(os.path.join(self.cache_dir, "null_text.pt"), weights_only=True).float()

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, i: int):
        return self.latents[i].float(), self.text[i].float(), self.mask[i]


class TensorLatentTextDataset(Dataset):
    """The same item format over in-memory tensors (tests, small experiments)."""

    def __init__(self, latents: Tensor, text: Tensor, mask: Optional[Tensor] = None):
        if mask is None:
            mask = torch.ones(text.shape[:2], dtype=torch.bool)
        if not (latents.shape[0] == text.shape[0] == mask.shape[0]):
            raise ValueError("latents, text and mask must have the same length")
        self.latents, self.text, self.mask = latents, text, mask

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, i: int):
        return self.latents[i].float(), self.text[i].float(), self.mask[i]


# =============================================================================
# SEEDS
# =============================================================================

_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def mix_seed(*parts: int) -> int:
    """A 63-bit seed from integers (splitmix64 chained over the parts); equal parts give equal seeds."""
    h = 0
    for p in parts:
        h = _splitmix64(h ^ (int(p) & _MASK64))
    return h & ((1 << 63) - 1)


# =============================================================================
# MEMORY-MAPPED CACHE
# =============================================================================

MEMMAP_FORMAT = "agatha_v2_memmap_v1"
MEMMAP_FILES = {
    "latents": "latents.bin",
    "text_tokens": "text_tokens.bin",
    "text_offsets": "text_offsets.npy",
    "caption_image": "caption_image.npy",
    "image_captions": "image_captions.npy",
    "images": "images.jsonl",
    "captions": "captions.jsonl",
    "null_text": "null_text.npy",
}


class _ChannelStats:
    """Per-channel mean and standard deviation over [n, C, H, W] batches (float64, pairwise merge)."""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def add(self, x: np.ndarray) -> None:
        x = x.astype(np.float64)
        nb = x.shape[0] * x.shape[2] * x.shape[3]
        mb = x.mean(axis=(0, 2, 3))
        m2b = ((x - mb[None, :, None, None]) ** 2).sum(axis=(0, 2, 3))
        if self.mean is None:
            self.n, self.mean, self.m2 = nb, mb, m2b
            return
        tot = self.n + nb
        delta = mb - self.mean
        self.mean = self.mean + delta * (nb / tot)
        self.m2 = self.m2 + m2b + delta ** 2 * (self.n * nb / tot)
        self.n = tot

    def result(self) -> Tuple[List[float], List[float]]:
        return self.mean.tolist(), np.sqrt(self.m2 / self.n).tolist()


def _write_json(path: str, obj: Any) -> None:
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


@torch.no_grad()
def build_memmap_cache(
    records: Iterable[Dict[str, Any]],
    encode_images: EncodeImages,
    encode_texts: EncodeTexts,
    out_dir: str,
    *,
    batch_size: int = 64,
    text_encoder: str = "",
    text_encoder_revision: Optional[str] = None,
    vae: str = "",
    vae_revision: Optional[str] = None,
    preprocessing: str = "",
    extra_meta: Optional[Dict[str, Any]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Encode images and all their captions once and write the memory-mapped cache.

    records yields dicts with keys
        "id", "file_name", "license"   written to images.jsonl as given
        "image"                        any object encode_images accepts
        "captions"                     a list of one or more strings, encoded exactly as given
        "caption_ids" (optional)       ids written to captions.jsonl
    encode_images(list of images) -> latents [n, C, H, W]; encode_texts(list of str) ->
    (states [m, L, D], mask [m, L]); a caption's stored length is its mask count.
    Work is done `batch_size` images at a time and appended to the files, so memory stays
    bounded by one batch. Returns the meta dict (also written to meta.json, last).
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)
    paths = {k: os.path.join(out_dir, v) for k, v in MEMMAP_FILES.items()}

    null_states, null_mask = encode_texts([""])
    null_keep = null_mask[0].bool().cpu()
    null_np = null_states[0].float().cpu()[null_keep].numpy().astype(np.float16)
    if null_np.shape[0] < 1:
        raise ValueError("the empty caption must keep at least one token")
    np.save(paths["null_text"], null_np)
    text_dim = int(null_np.shape[1])

    stats = _ChannelStats()
    text_offsets: List[int] = [0]
    caption_image: List[int] = []
    image_captions: List[int] = [0]
    counts = {"images": 0, "captions": 0, "tokens": 0}
    lengths = {"min": None, "max": 0}
    latent_shape: Optional[Tuple[int, ...]] = None

    with open(paths["latents"], "wb") as f_lat, open(paths["text_tokens"], "wb") as f_txt, \
            open(paths["images"], "w", encoding="utf-8") as f_img, \
            open(paths["captions"], "w", encoding="utf-8") as f_cap:

        def run(batch: List[Dict[str, Any]]) -> None:
            nonlocal latent_shape
            lat = encode_images([r["image"] for r in batch]).float().cpu()
            if lat.dim() != 4 or lat.shape[0] != len(batch):
                raise ValueError(f"encode_images returned {tuple(lat.shape)} for {len(batch)} images")
            if latent_shape is None:
                latent_shape = tuple(lat.shape[1:])
            elif tuple(lat.shape[1:]) != latent_shape:
                raise ValueError(f"latent shape changed from {latent_shape} to {tuple(lat.shape[1:])}")
            lat16 = np.ascontiguousarray(lat.numpy().astype(np.float16))
            f_lat.write(lat16.tobytes())
            stats.add(lat16)

            caps = [c for r in batch for c in r["captions"]]
            states, mask = encode_texts(caps)
            states, mask = states.float().cpu(), mask.bool().cpu()
            if states.shape[0] != len(caps) or states.shape[2] != text_dim:
                raise ValueError(f"encode_texts returned {tuple(states.shape)} for {len(caps)} captions")
            k = 0
            for r in batch:
                i = counts["images"]
                ids = r.get("caption_ids") or [None] * len(r["captions"])
                if not r["captions"]:
                    raise ValueError(f"image {r.get('id')} has no caption")
                for cid, text in zip(ids, r["captions"]):
                    tok = states[k][mask[k]].numpy().astype(np.float16)
                    if tok.shape[0] < 1:
                        raise ValueError(f"caption {text!r} kept no token")
                    f_txt.write(np.ascontiguousarray(tok).tobytes())
                    counts["tokens"] += tok.shape[0]
                    text_offsets.append(counts["tokens"])
                    caption_image.append(i)
                    f_cap.write(json.dumps({"index": counts["captions"], "image_index": i, "id": cid,
                                            "text": text}, ensure_ascii=False) + "\n")
                    counts["captions"] += 1
                    lengths["max"] = max(lengths["max"], tok.shape[0])
                    lengths["min"] = tok.shape[0] if lengths["min"] is None else min(lengths["min"], tok.shape[0])
                    k += 1
                image_captions.append(counts["captions"])
                f_img.write(json.dumps({"index": i, "id": r.get("id"), "file_name": r.get("file_name"),
                                        "license": r.get("license"), "num_captions": len(r["captions"])},
                                       ensure_ascii=False) + "\n")
                counts["images"] += 1
            if progress is not None:
                progress(counts["images"], counts["captions"])

        batch: List[Dict[str, Any]] = []
        for rec in records:
            batch.append(rec)
            if len(batch) == batch_size:
                run(batch)
                batch = []
        if batch:
            run(batch)

    if counts["images"] == 0:
        raise ValueError("no records")
    np.save(paths["text_offsets"], np.asarray(text_offsets, dtype=np.int64))
    np.save(paths["caption_image"], np.asarray(caption_image, dtype=np.int64))
    np.save(paths["image_captions"], np.asarray(image_captions, dtype=np.int64))
    mean, std = stats.result()
    meta = {
        "format": MEMMAP_FORMAT,
        "num_images": counts["images"],
        "num_captions": counts["captions"],
        "num_text_tokens": counts["tokens"],
        "latent_shape": list(latent_shape),
        "latent_dtype": "float16",
        "text_dim": text_dim,
        "text_dtype": "float16",
        "caption_tokens_min": lengths["min"],
        "caption_tokens_max": lengths["max"],
        "latent_mean": mean,
        "latent_std": std,
        "normalization": "z = (latent - latent_mean[c]) / latent_std[c] per channel; decoding undoes it",
        "null_text": {"file": MEMMAP_FILES["null_text"], "length": int(null_np.shape[0]), "dtype": "float16"},
        "files": dict(MEMMAP_FILES),
        "text_encoder": {"name": text_encoder, "revision": text_encoder_revision},
        "vae": {"name": vae, "revision": vae_revision},
        "preprocessing": preprocessing,
    }
    if extra_meta:
        meta["extra"] = extra_meta
    _write_json(meta_path, meta)
    return meta


def read_memmap_meta(cache_dir: str) -> Dict[str, Any]:
    with open(os.path.join(cache_dir, "meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("format") != MEMMAP_FORMAT:
        raise ValueError(f"not a {MEMMAP_FORMAT} cache: {cache_dir}")
    return meta


class MemmapLatentTextDataset(Dataset):
    """Reads a cache written by build_memmap_cache().

    Index with an (image, caption) pair, or with an image index alone (its first caption).
    An item is (latent [C, H, W] float32, normalized per channel; token states [L, D]
    float32 of that caption, true length). The maps open lazily in the process that reads
    (each DataLoader worker opens its own, also after a fork) and are never pickled.
    `stats` = (mean, std) per channel overrides the cache's own statistics (an evaluation
    cache is normalized with the training cache's statistics).
    """

    def __init__(self, cache_dir: str, stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                 normalize: bool = True):
        self.cache_dir = cache_dir
        self.meta = read_memmap_meta(cache_dir)
        files = self.meta["files"]
        self.text_offsets = np.load(os.path.join(cache_dir, files["text_offsets"]))
        self.caption_image = np.load(os.path.join(cache_dir, files["caption_image"]))
        self.image_captions = np.load(os.path.join(cache_dir, files["image_captions"]))
        mean, std = stats if stats is not None else (self.meta["latent_mean"], self.meta["latent_std"])
        c = self.meta["latent_shape"][0]
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(c, 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(c, 1, 1)
        if bool((self.std <= 0).any()):
            raise ValueError("latent std must be positive")
        self.normalize = bool(normalize)
        self._latents = None
        self._tokens = None
        self._pid = None

    # ---- maps
    def _maps(self):
        pid = os.getpid()
        if self._latents is None or self._pid != pid:
            files = self.meta["files"]
            n, shape = self.meta["num_images"], tuple(self.meta["latent_shape"])
            self._latents = np.memmap(os.path.join(self.cache_dir, files["latents"]), dtype=np.float16,
                                      mode="r", shape=(n,) + shape)
            self._tokens = np.memmap(os.path.join(self.cache_dir, files["text_tokens"]), dtype=np.float16,
                                     mode="r", shape=(self.meta["num_text_tokens"], self.meta["text_dim"]))
            self._pid = pid
        return self._latents, self._tokens

    def close(self) -> None:
        """Drop the maps (they reopen on the next read)."""
        self._latents = None
        self._tokens = None
        self._pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_latents"] = None
        state["_tokens"] = None
        state["_pid"] = None
        return state

    # ---- sizes and lookups
    @property
    def num_images(self) -> int:
        return int(self.meta["num_images"])

    @property
    def num_captions(self) -> int:
        return int(self.meta["num_captions"])

    def __len__(self) -> int:
        return self.num_images

    def captions_of(self, image: int) -> range:
        return range(int(self.image_captions[image]), int(self.image_captions[image + 1]))

    def latent_stats(self) -> Tuple[Tensor, Tensor]:
        return self.mean.flatten().clone(), self.std.flatten().clone()

    def normalize_latents(self, z: Tensor) -> Tensor:
        """(latent - mean) / std per channel, for [C, H, W] or [B, C, H, W]."""
        return (z - self.mean.to(z.device, z.dtype)) / self.std.to(z.device, z.dtype)

    def denormalize(self, z: Tensor) -> Tensor:
        """Undo the normalization of [C, H, W] or [B, C, H, W] latents."""
        return z * self.std.to(z.device, z.dtype) + self.mean.to(z.device, z.dtype)

    def null_text(self) -> Tensor:
        arr = np.load(os.path.join(self.cache_dir, self.meta["null_text"]["file"]))
        return torch.from_numpy(arr.astype(np.float32))

    def raw_latent(self, image: int) -> Tensor:
        lat, _ = self._maps()
        return torch.from_numpy(np.array(lat[image], dtype=np.float32))

    def token_states(self, caption: int) -> Tensor:
        _, tok = self._maps()
        a, b = int(self.text_offsets[caption]), int(self.text_offsets[caption + 1])
        return torch.from_numpy(np.array(tok[a:b], dtype=np.float32))

    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            image, caption = int(key[0]), int(key[1])
            if int(self.caption_image[caption]) != image:
                raise IndexError(f"caption {caption} belongs to image {int(self.caption_image[caption])}, not {image}")
        else:
            image = int(key)
            caption = int(self.image_captions[image])
        z = self.raw_latent(image)
        if self.normalize:
            z = (z - self.mean) / self.std
        return z, self.token_states(caption)


class StepOrderSampler(Sampler):
    """The (image, caption) pairs of every global step, from a seed alone.

    Sample position p = step * batch_size + k (k < batch_size) lies in epoch e = p // N at
    rank r = p % N; its image is perm_e[r], where perm_e is a fresh permutation of the N
    images for each epoch, and its caption is image_captions[img] + draw_e[r] mod (the
    image's caption count). Batches may span two epochs. Iterating yields the batches of
    steps start_step, start_step + 1, ... (to end_step, or without end), so a run resumed at
    step k reads exactly the batches an unbroken run reads from step k on. Use it as a
    DataLoader batch_sampler.
    """

    PERM_SALT = 0x7065726D
    DRAW_SALT = 0x63617074

    def __init__(self, image_captions: Sequence[int], batch_size: int, seed: int, start_step: int = 0,
                 end_step: Optional[int] = None):
        offsets = np.asarray(image_captions, dtype=np.int64)
        if offsets.ndim != 1 or offsets.shape[0] < 2:
            raise ValueError("image_captions must be the offsets [N + 1]")
        self.offsets = offsets
        self.counts = np.diff(offsets)
        if bool((self.counts < 1).any()):
            raise ValueError("every image needs at least one caption")
        self.num_images = int(offsets.shape[0] - 1)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.end_step = None if end_step is None else int(end_step)
        self._epochs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def epoch_order(self, epoch: int) -> Tuple[np.ndarray, np.ndarray]:
        """(image permutation [N], caption draws [N]) of an epoch."""
        if epoch not in self._epochs:
            gp = torch.Generator().manual_seed(mix_seed(self.seed, epoch, self.PERM_SALT))
            gd = torch.Generator().manual_seed(mix_seed(self.seed, epoch, self.DRAW_SALT))
            perm = torch.randperm(self.num_images, generator=gp).numpy()
            draw = torch.randint(0, 2 ** 31 - 1, (self.num_images,), generator=gd).numpy()
            if len(self._epochs) >= 2:
                self._epochs.pop(min(self._epochs))
            self._epochs[epoch] = (perm, draw)
        return self._epochs[epoch]

    def batch(self, step: int) -> List[Tuple[int, int]]:
        """The (image, caption) pairs of global step `step` (0-based)."""
        out = []
        n = self.num_images
        for k in range(self.batch_size):
            p = step * self.batch_size + k
            epoch, r = divmod(p, n)
            perm, draw = self.epoch_order(epoch)
            img = int(perm[r])
            cap = int(self.offsets[img] + draw[r] % self.counts[img])
            out.append((img, cap))
        return out

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        step = self.start_step
        while self.end_step is None or step < self.end_step:
            yield self.batch(step)
            step += 1

    def __len__(self) -> int:
        if self.end_step is None:
            raise TypeError("an open-ended StepOrderSampler has no length")
        return max(self.end_step - self.start_step, 0)

    def position(self, next_step: int) -> Dict[str, Any]:
        """The data position to store in a checkpoint taken before step `next_step`."""
        return {"sampler": "step_order", "seed": self.seed, "batch_size": self.batch_size,
                "num_images": self.num_images, "next_step": int(next_step)}


def collate_latent_text(items: Sequence[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
    """Stack latents; pad token states to the batch's longest caption; return the mask."""
    z = torch.stack([it[0] for it in items])
    lengths = [int(it[1].shape[0]) for it in items]
    width = int(items[0][1].shape[1])
    text = torch.zeros(len(items), max(lengths), width, dtype=items[0][1].dtype)
    mask = torch.zeros(len(items), max(lengths), dtype=torch.bool)
    for b, (it, n) in enumerate(zip(items, lengths)):
        text[b, :n] = it[1]
        mask[b, :n] = True
    return z, text, mask


def fixed_caption_pairs(image_captions: Sequence[int], images: Sequence[int], seed: int) -> List[Tuple[int, int]]:
    """One caption per listed image, chosen from a seed: [(image, caption), ...] in the given order."""
    offsets = np.asarray(image_captions, dtype=np.int64)
    g = torch.Generator().manual_seed(mix_seed(seed, 0x6576616C))
    draws = torch.randint(0, 2 ** 31 - 1, (len(images),), generator=g).numpy()
    out = []
    for img, d in zip(images, draws):
        img = int(img)
        n = int(offsets[img + 1] - offsets[img])
        out.append((img, int(offsets[img] + d % n)))
    return out


def gather_pairs(dataset: MemmapLatentTextDataset, pairs: Sequence[Tuple[int, int]]) -> Tuple[Tensor, Tensor, Tensor]:
    """(latents [n, C, H, W], text [n, L, D], mask [n, L]) of the listed pairs (normalized latents)."""
    return collate_latent_text([dataset[p] for p in pairs])
