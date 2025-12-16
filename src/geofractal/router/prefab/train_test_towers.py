"""
CIFAR-100 Classifier using GeometricTowers from Codebase
=========================================================

Uses the actual AgathaTowerCollective + DINOv3 cached latents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import time
import os

# Import the tower builder
from geofractal.router.prefab.geometric_tower_builder import (
    TowerConfig,
    ConfigurableCollective,
    ConfigurableTower,
    CollectiveOpinion,
    TowerOpinion,
    RoPEType,
    AddressType,
    FusionType,
    build_tower_collective,
    preset_pos_neg_pairs,
    preset_all_six,
)


# =============================================================================
# DINO CACHING (same as before)
# =============================================================================

class DinoCacher:
    """Precaches DINOv3 ConvNeXt-Small outputs for entire dataset."""

    DINO_MODEL = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
    DINO_DIM = 768  # ConvNeXt-Small embedding dimension

    def __init__(self, cache_dir: str = "./dino_cache", device: str = "cuda"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.device = device
        self.dino_dim = self.DINO_DIM

    def _get_cache_path(self, split: str) -> Path:
        return self.cache_dir / f"dinov3_convnext_small_cifar100_{split}.pt"

    def cache_exists(self, split: str) -> bool:
        return self._get_cache_path(split).exists()

    def load_cache(self, split: str) -> Tensor:
        path = self._get_cache_path(split)
        print(f"Loading cached DINOv3 ConvNeXt-S latents from {path}")
        return torch.load(path, weights_only=True)

    @torch.no_grad()
    def build_cache(self, dataset: Dataset, split: str, batch_size: int = 64) -> Tensor:
        path = self._get_cache_path(split)

        if path.exists():
            print(f"Cache exists at {path}")
            return self.load_cache(split)

        print(f"Building DINOv3 ConvNeXt-S cache for {split} ({len(dataset)} images)...")

        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(self.DINO_MODEL)
        backbone = AutoModel.from_pretrained(self.DINO_MODEL).to(self.device)
        backbone.eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        all_latents = []
        for images, _ in tqdm(loader, desc=f"Caching ConvNeXt-S {split}"):
            images = images.to(self.device)
            if images.shape[-1] != 224:
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            outputs = backbone(pixel_values=images)
            # ConvNeXt uses pooler_output
            latents = outputs.pooler_output.cpu()  # [B, 768]
            all_latents.append(latents)

        all_latents = torch.cat(all_latents, dim=0)
        torch.save(all_latents, path)
        print(f"Saved cache to {path} - shape: {all_latents.shape}")

        del backbone
        torch.cuda.empty_cache()

        return all_latents


class DinoVitLCacher:
    """Precaches DINOv3 ViT-L outputs for entire dataset.

    ViT-L is distilled from the 7B teacher using multi-distillation
    which includes CLIP-L knowledge.
    """

    DINO_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    DINO_DIM = 1024  # ViT-L embedding dimension

    def __init__(self, cache_dir: str = "./dino_cache", device: str = "cuda"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.device = device
        self.dino_dim = self.DINO_DIM

    def _get_cache_path(self, split: str) -> Path:
        return self.cache_dir / f"dinov3_vitl16_cifar100_{split}.pt"

    def cache_exists(self, split: str) -> bool:
        return self._get_cache_path(split).exists()

    def load_cache(self, split: str) -> Tensor:
        path = self._get_cache_path(split)
        print(f"Loading cached DINOv3 ViT-L latents from {path}")
        return torch.load(path, weights_only=True)

    @torch.no_grad()
    def build_cache(self, dataset: Dataset, split: str, batch_size: int = 64) -> Tensor:
        path = self._get_cache_path(split)

        if path.exists():
            print(f"Cache exists at {path}")
            return self.load_cache(split)

        print(f"Building DINOv3 ViT-L cache for {split} ({len(dataset)} images)...")

        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(self.DINO_MODEL)
        backbone = AutoModel.from_pretrained(self.DINO_MODEL).to(self.device)
        backbone.eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        all_latents = []
        for images, _ in tqdm(loader, desc=f"Caching ViT-L {split}"):
            images = images.to(self.device)
            if images.shape[-1] != 224:
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            outputs = backbone(pixel_values=images)
            # ViT uses pooler_output (CLS token projected)
            latents = outputs.pooler_output.cpu()  # [B, 1024]
            all_latents.append(latents)

        all_latents = torch.cat(all_latents, dim=0)
        torch.save(all_latents, path)
        print(f"Saved cache to {path} - shape: {all_latents.shape}")

        del backbone
        torch.cuda.empty_cache()

        return all_latents


class CachedCIFARDataset(Dataset):
    """CIFAR dataset with precached DINOv3 ConvNeXt-S and ViT-L latents."""

    def __init__(self, base_dataset: Dataset, convnext_latents: Tensor, vitl_latents: Tensor):
        self.base_dataset = base_dataset
        self.convnext_latents = convnext_latents
        self.vitl_latents = vitl_latents

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        convnext_latent = self.convnext_latents[idx]
        vitl_latent = self.vitl_latents[idx]
        return image, convnext_latent, vitl_latent, label


# =============================================================================
# DINO PROJECTION WITH SCHEDULING
# =============================================================================

class ScheduledDinoProjection(nn.Module):
    """Projects cached DINO latents with scheduled scale and dropout."""

    def __init__(
            self,
            dino_dim: int = 768,
            out_dim: int = 256,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        # Runtime parameters set by scheduler
        self.current_scale = 1.0
        self.current_dropout = 0.0

    def set_schedule(self, scale: float, dropout: float):
        """Called by trainer each epoch."""
        self.current_scale = scale
        self.current_dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        # Apply scheduled dropout
        if self.training and self.current_dropout > 0:
            mask = torch.bernoulli(torch.full_like(x, 1.0 - self.current_dropout))
            x = x * mask / (1.0 - self.current_dropout + 1e-8)
        return x * self.current_scale


class ExpertScheduler:
    """
    Curriculum: Rapidly introduce experts, then fade them out.

    Phase 1 (warmup): scale 0→1 over warmup_epochs
    Phase 2 (full): scale=1, dropout=0 
    Phase 3 (fadeout): dropout 0→1 over fadeout_epochs

    This forces towers to learn WITH experts, then WITHOUT.
    """

    def __init__(
            self,
            total_epochs: int,
            warmup_epochs: int = 3,  # Rapid introduction
            plateau_epochs: int = 10,  # Full power plateau
            fadeout_epochs: int = 37,  # Gradual fadeout
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.plateau_epochs = plateau_epochs
        self.fadeout_epochs = fadeout_epochs

        # Phases
        self.warmup_end = warmup_epochs
        self.plateau_end = warmup_epochs + plateau_epochs
        # Remaining epochs are fadeout

    def get_schedule(self, epoch: int) -> Tuple[float, float]:
        """Returns (scale, dropout) for current epoch."""
        if epoch < self.warmup_end:
            # Phase 1: Rapid warmup
            progress = (epoch + 1) / self.warmup_epochs
            scale = progress  # 0 → 1
            dropout = 0.0
        elif epoch < self.plateau_end:
            # Phase 2: Full power plateau
            scale = 1.0
            dropout = 0.0
        else:
            # Phase 3: Fadeout via dropout
            fadeout_progress = (epoch - self.plateau_end) / self.fadeout_epochs
            fadeout_progress = min(1.0, fadeout_progress)  # Cap at 1.0
            scale = 1.0
            dropout = fadeout_progress  # 0 → 1

        return scale, dropout

    def __repr__(self):
        return (f"ExpertScheduler(warmup={self.warmup_epochs}, "
                f"plateau={self.plateau_epochs}, fadeout={self.fadeout_epochs})")


# =============================================================================
# CIFAR CLASSIFIER WITH GEOMETRIC TOWERS
# =============================================================================

class CIFARGeometricClassifier(nn.Module):
    """
    CIFAR classifier using ConfigurableTower builder + dual DINOv3 experts.

    Two DINOv3 experts provide complementary signals:
    - ConvNeXt-Small (768d): Dense spatial features, efficient
    - ViT-L (1024d): Distilled from 7B with CLIP-L knowledge

    Architecture:
        Image → PatchEmbed → ConfigurableCollective → Fused Opinion (256)
                              ↓
        DINOv3 ConvNeXt-S (768) → ConvNextProj → ConvNeXt Opinion (256)
        DINOv3 ViT-L (1024) → ViTLProj → ViT-L Opinion (256)
                              ↓
        [Collective + ConvNeXt + ViT-L] (768) → Classifier → Logits
    """

    CIFAR_CLASSES = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

    def __init__(
            self,
            dim: int = 256,
            tower_depth: int = 1,
            num_heads: int = 4,
            patch_size: int = 2,
            num_classes: int = 100,
            fingerprint_dim: int = 64,
            convnext_dim: int = 768,  # ConvNeXt-Small
            vitl_dim: int = 1024,  # ViT-L
            tower_configs: Optional[List[TowerConfig]] = None,
            fusion_type: str = 'adaptive',
    ):
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size

        # Patch embedding
        num_patches = (32 // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)

        # Default tower configs: pos/neg pairs for 4 geometries
        if tower_configs is None:
            tower_configs = preset_pos_neg_pairs(['cantor', 'beatrix', 'helix', 'simplex'])

        # Build tower collective
        self.collective = build_tower_collective(
            configs=tower_configs,
            dim=dim,
            default_depth=tower_depth,
            num_heads=num_heads,
            ffn_mult=4.0,
            dropout=0.1,
            fingerprint_dim=fingerprint_dim,
            fusion_type=fusion_type,
        )

        # DINOv3 ConvNeXt-Small projection (768d → 256d)
        self.convnext_proj = ScheduledDinoProjection(
            dino_dim=convnext_dim,
            out_dim=dim,
        )

        # DINOv3 ViT-L projection (1024d → 256d)
        self.vitl_proj = ScheduledDinoProjection(
            dino_dim=vitl_dim,
            out_dim=dim,
        )

        # Classifier: [collective_fused + convnext + vitl] → logits
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim * 3),  # 256 * 3 = 768
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_classes),
        )

        self.num_patches = num_patches
        self._tower_names = self.collective.tower_names + ['convnext', 'vitl']
        self._tower_configs = tower_configs

    def set_expert_schedule(self, scale: float, dropout: float):
        """Update expert projections with current schedule."""
        self.convnext_proj.set_schedule(scale, dropout)
        self.vitl_proj.set_schedule(scale, dropout)

    @property
    def tower_names(self) -> List[str]:
        return self._tower_names

    def forward(self, images: Tensor, convnext_latents: Tensor, vitl_latents: Tensor) -> Tensor:
        """Forward pass returning logits."""
        # Patch embed
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # Tower collective
        collective_out: CollectiveOpinion = self.collective(x)

        # DINOv3 projections
        convnext_opinion = self.convnext_proj(convnext_latents)
        vitl_opinion = self.vitl_proj(vitl_latents)

        # Combine and classify
        combined = torch.cat([collective_out.fused, convnext_opinion, vitl_opinion], dim=-1)
        logits = self.classifier(combined)

        return logits

    def forward_with_opinions(
            self, images: Tensor, convnext_latents: Tensor, vitl_latents: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward returning logits and all opinions."""
        # Patch embed
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # Tower collective
        collective_out: CollectiveOpinion = self.collective(x)

        # Extract individual tower opinions
        opinions = {}
        for name, tower_op in collective_out.opinions.items():
            opinions[name] = tower_op.opinion
        opinions['collective_fused'] = collective_out.fused
        opinions['collective_weights'] = collective_out.weights

        # DINOv3 experts
        convnext_opinion = self.convnext_proj(convnext_latents)
        vitl_opinion = self.vitl_proj(vitl_latents)
        opinions['convnext'] = convnext_opinion
        opinions['vitl'] = vitl_opinion

        # Combine and classify
        combined = torch.cat([collective_out.fused, convnext_opinion, vitl_opinion], dim=-1)
        logits = self.classifier(combined)

        return logits, opinions


# =============================================================================
# ANALYZER
# =============================================================================

class ClassAnalyzer:
    """Tracks per-class performance."""

    CIFAR_CLASSES = CIFARGeometricClassifier.CIFAR_CLASSES

    def __init__(self, tower_names: List[str]):
        self.tower_names = tower_names
        self.reset()

    def reset(self):
        self.tower_class_norms = {name: defaultdict(list) for name in self.tower_names}
        self.class_correct = defaultdict(int)
        self.class_total = defaultdict(int)

    @torch.no_grad()
    def update(self, opinions: Dict[str, Tensor], logits: Tensor, labels: Tensor):
        _, predicted = logits.max(1)

        for i, (pred, label) in enumerate(zip(predicted, labels)):
            label_idx = label.item()
            self.class_total[label_idx] += 1
            if pred == label:
                self.class_correct[label_idx] += 1

            for name in self.tower_names:
                if name in opinions and name not in ['collective_fused', 'collective_weights']:
                    norm = opinions[name][i].norm().item()
                    self.tower_class_norms[name][label_idx].append(norm)

    def get_class_accuracy(self) -> Dict[str, float]:
        num_classes = len(self.CIFAR_CLASSES)
        return {
            self.CIFAR_CLASSES[c]: 100. * self.class_correct[c] / max(1, self.class_total[c])
            for c in range(num_classes)
        }

    def print_report(self):
        num_classes = len(self.CIFAR_CLASSES)

        print("\n" + "=" * 80)
        print("CLASS-SPECIFIC ANALYSIS")
        print("=" * 80)

        class_acc = self.get_class_accuracy()
        sorted_acc = sorted(class_acc.items(), key=lambda x: -x[1])

        print("\nPer-Class Accuracy (Top 10):")
        print("-" * 50)
        for cls, acc in sorted_acc[:10]:
            bar = "█" * int(acc / 5)
            print(f"  {cls:16s}: {acc:5.1f}% {bar}")

        print("\nPer-Class Accuracy (Bottom 10):")
        print("-" * 50)
        for cls, acc in sorted_acc[-10:]:
            bar = "█" * int(acc / 5)
            print(f"  {cls:16s}: {acc:5.1f}% {bar}")

        avg_acc = sum(class_acc.values()) / num_classes
        print(f"\n  Average across {num_classes} classes: {avg_acc:.1f}%")

        # Tower norms
        print("\nTower Opinion Norms (avg across all classes):")
        print("-" * 50)
        for name in self.tower_names:
            if name in ['collective_fused', 'collective_weights']:
                continue
            all_norms = []
            for c in range(num_classes):
                all_norms.extend(self.tower_class_norms[name][c])
            avg_norm = sum(all_norms) / len(all_norms) if all_norms else 0
            print(f"  {name:12s}: {avg_norm:.2f}")


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, epoch=0, scheduler: Optional[ExpertScheduler] = None):
    model.train()

    # Apply schedule for this epoch
    if scheduler:
        scale, dropout = scheduler.get_schedule(epoch)
        model.set_expert_schedule(scale, dropout)

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
    for images, convnext_latents, vitl_latents, labels in pbar:
        images = images.to(device)
        convnext_latents = convnext_latents.to(device)
        vitl_latents = vitl_latents.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, convnext_latents, vitl_latents)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{total_loss / total:.3f}', 'acc': f'{100. * correct / total:.1f}%'})

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate_with_analysis(model, loader, device, analyzer: ClassAnalyzer):
    model.eval()
    analyzer.reset()
    correct = 0
    total = 0

    for images, convnext_latents, vitl_latents, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        convnext_latents = convnext_latents.to(device)
        vitl_latents = vitl_latents.to(device)
        labels = labels.to(device)

        logits, opinions = model.forward_with_opinions(images, convnext_latents, vitl_latents)

        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        analyzer.update(opinions, logits, labels)

    return 100. * correct / total


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("CIFAR-100: Geometric Towers + Dual DINOv3 [CURRICULUM]")
    print("=" * 80)
    print("DINOv3 ConvNeXt-S (768d): Dense spatial features")
    print("DINOv3 ViT-L (1024d): Distilled from 7B with CLIP-L knowledge")
    print("CURRICULUM: Introduce → Full Power → Fadeout (towers inherit)")
    print("=" * 80)

    # Config
    BATCH_SIZE = 128
    EPOCHS = 50
    DIM = 256
    TOWER_DEPTH = 1
    NUM_HEADS = 4
    PATCH_SIZE = 2
    CACHE_DIR = "./dino_cache"
    NUM_CLASSES = 100

    # DINOv3 expert config
    CONVNEXT_DIM = 768  # ConvNeXt-Small
    VITL_DIM = 1024  # ViT-L

    # Expert curriculum schedule
    WARMUP_EPOCHS = 3  # Rapid introduction (scale 0→1)
    PLATEAU_EPOCHS = 12  # Full power
    FADEOUT_EPOCHS = 35  # Gradual dropout (0→1)

    # Tower configs - pos/neg pairs for each geometry
    TOWER_CONFIGS = [
        # Cantor pair
        TowerConfig('cantor_pos', rope='cantor', address='cantor', inverted=False),
        TowerConfig('cantor_neg', rope='cantor', address='cantor', inverted=True),
        # Beatrix pair
        TowerConfig('beatrix_pos', rope='beatrix', address='beatrix', inverted=False),
        TowerConfig('beatrix_neg', rope='beatrix', address='beatrix', inverted=True),
        # Helix pair
        TowerConfig('helix_pos', rope='helix', address='helix', inverted=False),
        TowerConfig('helix_neg', rope='helix', address='helix', inverted=True),
        # Simplex pair
        TowerConfig('simplex_pos', rope='simplex', address='simplex', inverted=False),
        TowerConfig('simplex_neg', rope='simplex', address='simplex', inverted=True),
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Patch size: {PATCH_SIZE} → {(32 // PATCH_SIZE) ** 2} patches")
    print(f"Tower depth: {TOWER_DEPTH}")
    print(f"Optimizer: Adafactor (relative_step=True, warmup_init=True)")
    print(f"ConvNeXt-S: dim={CONVNEXT_DIM}")
    print(f"ViT-L: dim={VITL_DIM}")
    print(f"Schedule: warmup={WARMUP_EPOCHS}, plateau={PLATEAU_EPOCHS}, fadeout={FADEOUT_EPOCHS}")

    print(f"\nTower Configs ({len(TOWER_CONFIGS)} towers):")
    for cfg in TOWER_CONFIGS:
        print(f"  {cfg.name}: rope={cfg.rope.value}, addr={cfg.address.value}, inv={cfg.inverted}")

    # Transforms
    transform_base = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Datasets
    train_dataset_base = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_base)
    test_dataset_base = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_base)

    print(f"\nTrain: {len(train_dataset_base)}, Test: {len(test_dataset_base)}")

    # Cache DINOv3 ConvNeXt-Small
    convnext_cacher = DinoCacher(cache_dir=CACHE_DIR, device=device)
    train_convnext = convnext_cacher.build_cache(train_dataset_base, split="train")
    test_convnext = convnext_cacher.build_cache(test_dataset_base, split="test")

    # Cache DINOv3 ViT-L  
    vitl_cacher = DinoVitLCacher(cache_dir=CACHE_DIR, device=device)
    train_vitl = vitl_cacher.build_cache(train_dataset_base, split="train")
    test_vitl = vitl_cacher.build_cache(test_dataset_base, split="test")

    # Augmented training
    train_dataset_aug = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)

    train_dataset = CachedCIFARDataset(train_dataset_aug, train_convnext, train_vitl)
    test_dataset = CachedCIFARDataset(test_dataset_base, test_convnext, test_vitl)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model
    print("\nBuilding model with ConfigurableTowers + Dual DINOv3 Experts...")
    model = CIFARGeometricClassifier(
        dim=DIM,
        tower_depth=TOWER_DEPTH,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        convnext_dim=CONVNEXT_DIM,
        vitl_dim=VITL_DIM,
        tower_configs=TOWER_CONFIGS,
    ).to(device)

    print(f"\nTowers: {model.collective.tower_names}")
    print(f"Expert heads: ConvNeXt-S (768→{DIM}), ViT-L (1024→{DIM})")
    print(f"Classifier input: {DIM * 3} (collective + convnext + vitl)")
    print(f"Patches: {model.num_patches}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    for name in model.collective.tower_names:
        tower = model.collective[name]
        tower_params = sum(p.numel() for p in tower.parameters())
        cfg = tower.config
        print(f"  {name}: {tower_params:,} params, rope={cfg.rope.value}, inv={cfg.inverted}")

    # Optimizer - Adafactor with internal scheduling (best for tower convergence)
    from transformers import Adafactor

    optimizer = Adafactor(
        model.parameters(),
        lr=None,  # Let Adafactor compute LR
        scale_parameter=True,  # Scale by RMS of params
        relative_step=True,  # LR = 1/sqrt(step) decay
        warmup_init=True,  # Linear warmup from 0
    )
    # No external scheduler - Adafactor handles it internally

    # Analyzer
    analyzer = ClassAnalyzer(model.tower_names)

    # Expert Scheduler - curriculum learning
    expert_scheduler = ExpertScheduler(
        total_epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        plateau_epochs=PLATEAU_EPOCHS,
        fadeout_epochs=FADEOUT_EPOCHS,
    )
    print(f"\nExpert Scheduler: {expert_scheduler}")

    # Training
    print("\n" + "-" * 80)
    print("Training with Expert Curriculum")
    print("Phase 1: Warmup (scale 0→1)")
    print("Phase 2: Plateau (full power)")
    print("Phase 3: Fadeout (dropout 0→1, towers inherit)")
    print("-" * 80)

    best_acc = 0
    for epoch in range(EPOCHS):
        start = time.time()

        # Get current schedule
        scale, dropout = expert_scheduler.get_schedule(epoch)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch, expert_scheduler)

        # Set to full scale for eval (no dropout)
        model.set_expert_schedule(scale=1.0, dropout=0.0)
        test_acc = evaluate_with_analysis(model, test_loader, device, analyzer)

        elapsed = time.time() - start

        marker = " *" if test_acc > best_acc else ""
        if test_acc > best_acc:
            best_acc = test_acc

        # Phase indicator
        if epoch < WARMUP_EPOCHS:
            phase = f"WARMUP s={scale:.2f}"
        elif epoch < WARMUP_EPOCHS + PLATEAU_EPOCHS:
            phase = "PLATEAU"
        else:
            phase = f"FADE d={dropout:.2f}"

        print(f"Epoch {epoch + 1:2d}/{EPOCHS} | "
              f"{phase:14s} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Test: {test_acc:.1f}%{marker} | "
              f"Time: {elapsed:.1f}s")

    print("\n" + "=" * 80)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("=" * 80)

    # Final analysis
    print("\nFinal evaluation with class analysis...")
    final_acc = evaluate_with_analysis(model, test_loader, device, analyzer)
    analyzer.print_report()


if __name__ == '__main__':
    main()