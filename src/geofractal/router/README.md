# geofractal.router

**Collective intelligence through geometric routing.**

## Overview

The GlobalFractalRouter is infrastructure for coordinating multiple models/experts via fingerprint-based divergence and mailbox communication.

**Key Insight:** Individual streams don't need to be accurate classifiers. They need to provide *divergent perspectives* that the collective can triangulate into accurate predictions.

## Proven Results

| Experiment | Individual Accuracy | Collective Accuracy |
|------------|---------------------|---------------------|
| ImageNet (5 CLIP variants) | 0.1% each | **84.68%** |
| FashionMNIST (CLIP + 2 Conv) | 10% + 10% + 10% | **93.4%** |
| Dual Frozen CLIP | 98.6% frozen | **92.6%** |
| Math Collective (5 models) | Random | **+30%** over baseline |

## Quick Start

```python
from geofractal.router import (
    RouterCollective,
    CollectiveConfig,
    FrozenStream,
    FeatureStream,
)

# Configuration
config = CollectiveConfig(
    feature_dim=512,
    num_classes=1000,
    epochs=20,
)

# From pretrained models (frozen)
collective = RouterCollective.from_pretrained_models([
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
], config)

# Train (only routing learns)
history = collective.fit(train_loader, val_loader)

# Inference
logits, info = collective(batch)
```

## For Pre-extracted Features

```python
# Fastest - no model inference
collective = RouterCollective.from_feature_dims({
    'clip_vit_b32': 512,
    'clip_vit_b16': 512,
    'clip_vit_l14': 768,
    'dinov2_base': 768,
}, config)

# features: Dict[str, Tensor] mapping name to [B, dim]
logits, info = collective(features)
```

## Architecture

```
Stream 0 ──┐
           │    Fingerprint (divergence)
Stream 1 ──┼──▶ Router ──▶ Mailbox ──┐
           │                         │
Stream 2 ──┤                         ├──▶ Fusion ──▶ Classifier
           │    Shared Anchors       │
Stream N ──┘    (behavioral modes)   ┘
```

### Core Components

| Component | Role |
|-----------|------|
| **Fingerprint** | Unique identity per stream → creates divergent behavior |
| **Mailbox** | Inter-stream communication → enables coordination |
| **Anchors** | Shared behavioral modes → collective alignment |
| **Cantor Prior** | Self-similar attention structure → geometric routing |

## Stream Types

### FrozenStream
Wraps frozen pretrained models (CLIP, DINO, etc.)
```python
stream = FrozenStream.from_pretrained(
    "openai/clip-vit-large-patch14",
    config=config,
)
```

### FeatureStream
For pre-extracted features (fastest throughput)
```python
stream = FeatureStream(
    config=config,
    name="clip_features",
    input_dim=768,
)
```

### TrainableStream
Fully learnable backbone + router
```python
stream = TrainableStream.conv_stream(
    config=config,
    name="conv_expert",
    channels=[32, 64],
)
```

## Configuration

### CollectiveConfig
```python
config = CollectiveConfig(
    # Architecture
    feature_dim=512,      # Internal routing dimension
    fingerprint_dim=64,   # Identity dimension
    num_classes=1000,     # Output classes
    
    # Router
    num_anchors=16,       # Behavioral modes
    num_routes=8,         # Top-K routes per position
    num_slots=16,         # Sequence length
    
    # Training
    batch_size=256,
    epochs=20,
    lr=3e-4,
    use_amp=True,         # Mixed precision
    
    # DataLoader (A100 optimized)
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
```

### Preset Configs
```python
from geofractal.router.config import (
    IMAGENET_COLLECTIVE_CONFIG,
    FASHIONMNIST_COLLECTIVE_CONFIG,
    CIFAR_COLLECTIVE_CONFIG,
)
```

## Package Structure

```
geofractal/router/
├── __init__.py          # Main exports
├── config.py            # Configuration classes
├── core.py              # GlobalFractalRouter
├── registry.py          # RouterRegistry, RouterMailbox
├── collective.py        # RouterCollective (high-level API)
└── streams/
    ├── __init__.py
    ├── base.py          # BaseStream (abstract)
    ├── frozen.py        # FrozenStream
    ├── feature.py       # FeatureStream
    └── trainable.py     # TrainableStream
```

## The Paradigm

**Old thinking:**
- Make each model accurate
- Ensemble averages predictions
- More parameters = better

**What this proves:**
- Make each model *different*
- Router fuses *perspectives*, not predictions
- Coordination > capacity

```
0.1% + 0.1% + 0.1% + 0.1% + 0.1% = 84.68%
```

That's not math. That's emergence.

## License

Apache License 2.0

**Attribution Required:** If you use this architecture in research or production, please cite:

```bibtex
@software{globalfractalrouter2025,
  author       = {AbstractPhil},
  title        = {GlobalFractalRouter: Collective Intelligence through 
                  Geometric Routing},
  year         = {2025},
  url          = {https://github.com/AbstractPhil/geofractal}
}
```

See [NOTICE](NOTICE) for full attribution requirements.

## Author

AbstractPhil  
December 2025