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

Copyright 2026 AbstractPhil
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

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
