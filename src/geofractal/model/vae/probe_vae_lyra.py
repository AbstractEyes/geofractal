# 🔬 VAE Lyra Latent Space Probe - Single Cell Colab
# Just paste this entire cell and run!

# !pip install - q transformers accelerate safetensors huggingface_hub scipy scikit - learn seaborn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Any
import json
import random

from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, T5EncoderModel, T5Tokenizer
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import hf_hub_download
from scipy import stats as scipy_stats

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Device: {device}")


@dataclass
class ModalityEmbeddings:
    clip_l: torch.Tensor
    clip_g: torch.Tensor
    t5_xl: torch.Tensor
    texts: List[str]


class LyraVAEShell(nn.Module):
    def __init__(self, state_dict, config):
        super().__init__()
        self.modality_dims = config.get('modality_dims',
                                        {'clip_l': 768, 'clip_g': 1280, 't5_xl_l': 2048, 't5_xl_g': 2048})
        self.latent_dim = config.get('latent_dim', 2048)
        self.modality_encoders = nn.ModuleDict()
        self.modality_decoders = nn.ModuleDict()

        for mod, dim in self.modality_dims.items():
            enc_w = {k.replace(f'modality_encoders.{mod}.', ''): v for k, v in state_dict.items() if
                     k.startswith(f'modality_encoders.{mod}')}
            if enc_w and '0.weight' in enc_w:
                layers, idx = [], 0
                while f'{idx}.weight' in enc_w:
                    w = enc_w[f'{idx}.weight']
                    lin = nn.Linear(w.shape[1], w.shape[0])
                    lin.weight, lin.bias = nn.Parameter(w), nn.Parameter(
                        enc_w.get(f'{idx}.bias', torch.zeros(w.shape[0])))
                    layers.append(lin)
                    idx += 1
                    if f'{idx}.weight' not in enc_w: break
                    layers.append(nn.GELU());
                    idx += 1
                self.modality_encoders[mod] = nn.Sequential(*layers)
            else:
                self.modality_encoders[mod] = nn.Linear(dim, self.latent_dim)

            dec_w = {k.replace(f'modality_decoders.{mod}.', ''): v for k, v in state_dict.items() if
                     k.startswith(f'modality_decoders.{mod}')}
            if dec_w and '0.weight' in dec_w:
                layers, idx = [], 0
                while f'{idx}.weight' in dec_w:
                    w = dec_w[f'{idx}.weight']
                    lin = nn.Linear(w.shape[1], w.shape[0])
                    lin.weight, lin.bias = nn.Parameter(w), nn.Parameter(
                        dec_w.get(f'{idx}.bias', torch.zeros(w.shape[0])))
                    layers.append(lin)
                    idx += 1
                    if f'{idx}.weight' not in dec_w: break
                    layers.append(nn.GELU());
                    idx += 1
                self.modality_decoders[mod] = nn.Sequential(*layers)
            else:
                self.modality_decoders[mod] = nn.Linear(self.latent_dim, dim)

        self.fc_mu = nn.Linear(state_dict['fc_mu.weight'].shape[1],
                               state_dict['fc_mu.weight'].shape[0]) if 'fc_mu.weight' in state_dict else nn.Identity()
        self.fc_logvar = nn.Linear(state_dict['fc_logvar.weight'].shape[1], state_dict['fc_logvar.weight'].shape[
            0]) if 'fc_logvar.weight' in state_dict else nn.Identity()
        if 'fc_mu.weight' in state_dict: self.fc_mu.weight, self.fc_mu.bias = nn.Parameter(
            state_dict['fc_mu.weight']), nn.Parameter(
            state_dict.get('fc_mu.bias', torch.zeros(state_dict['fc_mu.weight'].shape[0])))
        if 'fc_logvar.weight' in state_dict: self.fc_logvar.weight, self.fc_logvar.bias = nn.Parameter(
            state_dict['fc_logvar.weight']), nn.Parameter(
            state_dict.get('fc_logvar.bias', torch.zeros(state_dict['fc_logvar.weight'].shape[0])))

    def forward(self, inputs):
        encoded = {m: self.modality_encoders[m](x) for m, x in inputs.items() if m in self.modality_encoders}
        fused = torch.stack(list(encoded.values())).mean(0)
        mu, logvar = self.fc_mu(fused), self.fc_logvar(fused)
        recon = {m: self.modality_decoders[m](mu) for m in inputs if m in self.modality_decoders}
        return recon, mu, logvar, encoded


class LyraProbe:
    def __init__(self, model, config, device="cuda"):
        self.model = model.to(device).eval()
        self.config, self.device = config, torch.device(device)
        self._clip_l = self._clip_g = self._t5 = None

    @classmethod
    def from_hub(cls, repo_id, checkpoint=None, device="cuda"):
        """
        Load from HuggingFace Hub.

        Args:
            repo_id: HF repo (e.g. "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious")
            checkpoint: Specific checkpoint file in weights/ folder (e.g. "lyra_step_9000.safetensors")
                        If None, auto-discovers best checkpoint
            device: cuda or cpu
        """
        print(f"🔬 Loading from {repo_id}...")
        config = json.load(open(hf_hub_download(repo_id=repo_id, filename="config.json")))

        if checkpoint:
            # User specified checkpoint - look in weights/ subfolder
            ckpt_path = f"weights/{checkpoint}" if not checkpoint.startswith("weights/") else checkpoint
            weights_path = hf_hub_download(repo_id=repo_id, filename=ckpt_path)
            print(f"  ✓ {ckpt_path}")
        else:
            # Auto-discover
            model_name = config.get('model_name', 'lyra')
            for f in [f"weights/{model_name}_best.safetensors", f"weights/{model_name}_illustrious_best.safetensors",
                      "model.safetensors"]:
                try:
                    weights_path = hf_hub_download(repo_id=repo_id, filename=f); print(f"  ✓ {f}"); break
                except:
                    continue

        sd = load_safetensors(weights_path, device=device) if weights_path.endswith('.safetensors') else torch.load(
            weights_path, map_location=device)
        if isinstance(sd, dict) and 'model_state_dict' in sd: sd = sd['model_state_dict']
        print(f"  ✓ {len(sd)} tensors")
        return cls(LyraVAEShell(sd, config), config, device)

    def _load_enc(self):
        if self._clip_l: return
        print("📥 Loading encoders...")
        self._clip_l_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self._clip_l = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).eval()
        self._clip_g_tok = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        self._clip_g = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(
            self.device).eval()
        self._t5_tok = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        self._t5 = T5EncoderModel.from_pretrained("google/flan-t5-xl").to(self.device).eval()
        print("  ✓ Done")

    @torch.no_grad()
    def encode(self, texts, skip=2):
        self._load_enc()
        cl, cg, t5 = [], [], []
        for t in texts:
            tok = self._clip_l_tok(t, max_length=77, padding='max_length', truncation=True, return_tensors='pt').to(
                self.device)
            cl.append(self._clip_l(**tok, output_hidden_states=True).hidden_states[-skip])
            tok = self._clip_g_tok(t, max_length=77, padding='max_length', truncation=True, return_tensors='pt').to(
                self.device)
            cg.append(self._clip_g(**tok, output_hidden_states=True).hidden_states[-skip])
            tok = self._t5_tok(t, max_length=512, padding='max_length', truncation=True, return_tensors='pt').to(
                self.device)
            t5.append(self._t5(**tok).last_hidden_state)
        return ModalityEmbeddings(torch.cat(cl), torch.cat(cg), torch.cat(t5), texts)

    @torch.no_grad()
    def analyze(self, prompts):
        print(f"\n🔬 Analyzing {len(prompts)} prompts...")
        emb = self.encode(prompts)
        inputs = {'clip_l': emb.clip_l, 'clip_g': emb.clip_g, 't5_xl_l': emb.t5_xl, 't5_xl_g': emb.t5_xl}
        recon, mu, logvar, _ = self.model(inputs)

        cos, dist, qual = {}, {}, {}
        lat = mu.mean(1)
        for n, t in [('clip_l', emb.clip_l), ('clip_g', emb.clip_g), ('t5_xl', emb.t5_xl)]:
            m, d = t.mean(1), min(lat.shape[1], t.shape[2])
            cos[n] = {'global': F.cosine_similarity(lat[:, :d], m[:, :d], dim=1).mean().item(),
                      'curve': [F.cosine_similarity(mu[:, p, :d], t[:, p, :d], dim=1).mean().item() for p in
                                range(min(mu.shape[1], t.shape[1]))]}
            dist[n] = torch.norm(lat[:, :d] - m[:, :d], p=2, dim=1).mean().item() / d
        for m in inputs:
            if m in recon:
                qual[m] = {'mse': F.mse_loss(recon[m], inputs[m]).item(),
                           'cos': F.cosine_similarity(inputs[m].flatten(1), recon[m].flatten(1), dim=1).mean().item()}
        print("  ✓ Done")
        return {'emb': emb, 'mu': mu, 'cos': cos, 'dist': dist, 'qual': qual, 'recon': recon}

    @torch.no_grad()
    def shuffle_test(self, prompts, n=10):
        print(f"\n🔀 Shuffle test ({n} shuffles)...")
        results = {'prompts': [], 'lyra': [], 'lyra_std': [], 'clip_l': [], 'clip_g': [], 't5': []}
        for p in prompts:
            tags = [t.strip() for t in p.split(", ") if t.strip()]
            if len(tags) < 3: continue
            shuffled = [p] + [", ".join(random.sample(tags, len(tags))) for _ in range(n)]
            emb = self.encode(shuffled)
            inputs = {'clip_l': emb.clip_l, 'clip_g': emb.clip_g, 't5_xl_l': emb.t5_xl, 't5_xl_g': emb.t5_xl}
            _, mu, _, _ = self.model(inputs)

            ref = mu[0:1].mean(1)
            lc = [F.cosine_similarity(ref, mu[i:i + 1].mean(1), dim=1).item() for i in range(1, len(shuffled))]
            ref_cl, ref_cg, ref_t5 = emb.clip_l[0:1].mean(1), emb.clip_g[0:1].mean(1), emb.t5_xl[0:1].mean(1)
            cl = [F.cosine_similarity(ref_cl, emb.clip_l[i:i + 1].mean(1), dim=1).item() for i in
                  range(1, len(shuffled))]
            cg = [F.cosine_similarity(ref_cg, emb.clip_g[i:i + 1].mean(1), dim=1).item() for i in
                  range(1, len(shuffled))]
            t5 = [F.cosine_similarity(ref_t5, emb.t5_xl[i:i + 1].mean(1), dim=1).item() for i in
                  range(1, len(shuffled))]

            results['prompts'].append(p[:35])
            results['lyra'].append(np.mean(lc));
            results['lyra_std'].append(np.std(lc))
            results['clip_l'].append(np.mean(cl));
            results['clip_g'].append(np.mean(cg));
            results['t5'].append(np.mean(t5))
            print(f"  {p[:25]}... | Lyra:{np.mean(lc):.3f} CLIP-L:{np.mean(cl):.3f} T5:{np.mean(t5):.3f}")

        results['summary'] = {'lyra': np.mean(results['lyra']), 'clip_l': np.mean(results['clip_l']),
                              'clip_g': np.mean(results['clip_g']), 't5': np.mean(results['t5']),
                              'gain': np.mean(results['lyra']) - np.mean(results['clip_l'])}
        print(
            f"\n📊 Lyra={results['summary']['lyra']:.3f} CLIP-L={results['summary']['clip_l']:.3f} Gain={results['summary']['gain']:+.3f}")
        return results


def visualize(r):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🔬 VAE Lyra Probe', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    mods, vals = ['clip_l', 'clip_g', 't5_xl'], [r['cos'][m]['global'] for m in ['clip_l', 'clip_g', 't5_xl']]
    bars = ax.bar(mods, vals, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel('Cosine');
    ax.set_title('Latent ↔ Modality Similarity');
    ax.set_ylim(0, 1)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.3f}', ha='center',
                                         fontweight='bold')

    ax = axes[0, 1]
    mods = list(r['qual'].keys())
    ax.bar(mods, [r['qual'][m]['mse'] for m in mods], color='coral')
    ax.set_ylabel('MSE');
    ax.set_title('Reconstruction MSE');
    ax.tick_params(axis='x', rotation=45)

    ax = axes[1, 0]
    for n in ['clip_l', 'clip_g', 't5_xl']:
        ax.plot(r['cos'][n]['curve'][:77], label=n, lw=2)
    ax.set_xlabel('Position');
    ax.set_ylabel('Cosine');
    ax.set_title('Per-Position Similarity');
    ax.legend();
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.bar(['clip_l', 'clip_g', 't5_xl'], [r['dist'][m] for m in ['clip_l', 'clip_g', 't5_xl']],
           color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel('L2 (norm)');
    ax.set_title('Distance to Modality')

    plt.tight_layout();
    plt.show()


def visualize_shuffle(r):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('🔀 Shuffle Invariance', fontsize=14, fontweight='bold')

    cats, vals = ['Lyra', 'CLIP-L', 'CLIP-G', 'T5'], [r['summary']['lyra'], r['summary']['clip_l'],
                                                      r['summary']['clip_g'], r['summary']['t5']]
    bars = ax1.bar(cats, vals, color=['#9B59B6', '#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_ylabel('Cosine (shuffled vs orig)');
    ax1.set_ylim(0, 1);
    ax1.axhline(1, color='green', ls='--', alpha=0.5)
    ax1.set_title('Stability (Higher = More Invariant)')
    for b, v in zip(bars, vals): ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.3f}', ha='center',
                                          fontweight='bold')

    x, w = np.arange(len(r['prompts'])), 0.2
    ax2.bar(x - 1.5 * w, r['lyra'], w, label='Lyra', color='#9B59B6', yerr=r['lyra_std'], capsize=3)
    ax2.bar(x - 0.5 * w, r['clip_l'], w, label='CLIP-L', color='#FF6B6B')
    ax2.bar(x + 0.5 * w, r['clip_g'], w, label='CLIP-G', color='#4ECDC4')
    ax2.bar(x + 1.5 * w, r['t5'], w, label='T5', color='#45B7D1')
    ax2.set_ylabel('Cosine');
    ax2.set_title('Per-Prompt');
    ax2.set_xticks(x);
    ax2.set_xticklabels([f'P{i + 1}' for i in x]);
    ax2.legend();
    ax2.set_ylim(0, 1)
    plt.tight_layout();
    plt.show()


# ═══════════════════════════════════════════════════════════════
# 🚀 CONFIGURATION - EDIT THESE
# ═══════════════════════════════════════════════════════════════

REPO_ID = "AbstractPhil/vae-lyra-xl-adaptive-cantor-illustrious"

# Specify checkpoint in weights/ folder, or None for auto-detect
# Examples:
#   CHECKPOINT = "lyra_step_9000.safetensors"
#   CHECKPOINT = "lyra_illustrious_step_21000.safetensors"
#   CHECKPOINT = None  # auto-detect best
CHECKPOINT = None

TEST_PROMPTS = [
    "masterpiece, 1girl, blue hair, school uniform, smile, looking at viewer",
    "landscape, mountains, sunset, dramatic lighting, cinematic",
    "1boy, armor, sword, fantasy, dark background, epic",
    "2girls, sisters, holding hands, matching outfits, happy",
]

SHUFFLE_PROMPTS = [
    "1girl, blue hair, smile, school uniform, looking at viewer",
    "landscape, mountains, clouds, sunset, beautiful scenery",
    "1boy, 1girl, couple, holding hands, romantic, outdoors",
]

# ═══════════════════════════════════════════════════════════════
# 🚀 RUN THE PROBE
# ═══════════════════════════════════════════════════════════════

probe = LyraProbe.from_hub(REPO_ID, checkpoint=CHECKPOINT, device=device)

# Core analysis
results = probe.analyze(TEST_PROMPTS)
visualize(results)

print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)
print("\n🎯 SIMILARITY (higher = latent closer to modality):")
for m in ['clip_l', 'clip_g', 't5_xl']: print(f"   {m:8s}: {results['cos'][m]['global']:.4f}")
best = max(results['cos'].items(), key=lambda x: x[1]['global'])
print(f"\n   → Lyra latent most similar to: {best[0].upper()}")

# Shuffle test
shuffle_results = probe.shuffle_test(SHUFFLE_PROMPTS)
visualize_shuffle(shuffle_results)

print("\n" + "=" * 60)
print("🔀 INTERPRETATION")
print("=" * 60)
g, s = shuffle_results['summary']['gain'], shuffle_results['summary']['lyra']
if s > 0.95:
    print(f"\n✨ EXCELLENT! Lyra={s:.3f} - learned SEMANTIC structure!")
elif s > 0.85:
    print(f"\n✓ Good! Lyra={s:.3f}")
else:
    print(f"\n⚠️ Lyra={s:.3f} - some positional bias")
if g > 0.1:
    print(f"🚀 {g:.1%} MORE stable than CLIP-L! Cantor chaos worked!")
elif g > 0:
    print(f"📈 {g:.1%} more stable than CLIP-L")
else:
    print(f"📉 Similar to CLIP-L ({g:+.1%})")