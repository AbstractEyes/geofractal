# GlobalFractalRouter Documentation

**Collective Intelligence through Geometric Routing**

---

## Documentation Index

### Core Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [PROTOCOL.md](PROTOCOL.md) | Complete protocol specification | Implementers, Researchers |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Component deep dive | Engineers, Contributors |
| [MATHEMATICS.md](MATHEMATICS.md) | Formal mathematical foundations | Researchers, Theorists |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Roadmap and expansion paths | Contributors, Planners |
| [METRICS.md](METRICS.md) | Evaluation methodology | Evaluators, Benchmarkers |

### Head Decomposition

| Document | Description | Audience |
|----------|-------------|----------|
| [RATIONALITY.md](RATIONALITY.md) | **Why** each component exists | Everyone |
| [INIT_BREAKDOWN.md](INIT_BREAKDOWN.md) | Detailed init decomposition | Engineers, Hackers |

### Fusion Layer

| Document | Description | Audience |
|----------|-------------|----------|
| [FUSION.md](FUSION.md) | Fusion strategies and selection guide | Engineers, Researchers |

---

## Quick Reference

### Core Concept

```
Traditional: Make each model accurate → ensemble averages predictions
GFR:        Make each model DIFFERENT → router fuses perspectives
```

### Proven Results

| Experiment | Individual | Collective | Emergence |
|------------|------------|------------|-----------|
| ImageNet (5 CLIP) | 0.1% each | 84.68% | 847× |
| FashionMNIST | 10% each | 93.4% | 9.34× |
| Dual CLIP | 7-18% | 92.6% | 5-13× |

### Key Components

| Component | Purpose | Key Insight |
|-----------|---------|-------------|
| **Fingerprint** | Unique identity | Creates divergence, not accuracy |
| **Cantor Attention** | Geometric structure | Self-similar spatial relationships |
| **Anchor Bank** | Behavioral modes | MUST be constitutive, not additive |
| **TopK Router** | Sparse routing | Specialization through selection |
| **Mailbox** | Coordination | Detached to prevent collapse |

### Minimum Viable Collective

```python
from geofractal.router import (
    RouterCollective,
    CollectiveConfig,
    FeatureStream,
)

# Configuration
config = CollectiveConfig(
    feature_dim=512,
    num_classes=1000,
)

# Create streams
streams = [
    FeatureStream(config, "clip_b32", input_dim=512),
    FeatureStream(config, "clip_l14", input_dim=768),
]

# Build collective
collective = RouterCollective(streams, config)

# Train
history = collective.fit(train_loader, val_loader)

# Inference
logits, info = collective(batch)
```

### Configuration Presets

```python
from geofractal.router.config import (
    IMAGENET_COLLECTIVE_CONFIG,      # 512D, 16 anchors, 8 routes
    FASHIONMNIST_COLLECTIVE_CONFIG,  # 128D, 8 anchors, 4 routes
    CIFAR_COLLECTIVE_CONFIG,         # 256D, 12 anchors, 6 routes
)
```

---

## Reading Path

### For Implementers

1. Start with **PROTOCOL.md** §1-3 (Principles, Overview, Components)
2. Read **ARCHITECTURE.md** for component details
3. Reference **MATHEMATICS.md** for formulas
4. Use **METRICS.md** for evaluation

### For Researchers

1. Start with **MATHEMATICS.md** for formal foundations
2. Read **PROTOCOL.md** §6 (Mathematical Foundations)
3. Explore **DEVELOPMENT.md** for research directions
4. Use **METRICS.md** for experimental design

### For Contributors

1. Read **ARCHITECTURE.md** for code organization
2. Review **DEVELOPMENT.md** for roadmap
3. Follow **PROTOCOL.md** §7-8 for conformance
4. Add metrics per **METRICS.md**

---

## Frequently Asked Questions

### Why do individual streams have low accuracy?

**By design.** Streams provide *perspectives*, not *classifications*. The collective triangulates from divergent viewpoints.

### Why detach mailbox content?

Prevents gradient flow between streams, which would cause collapse to trivial solutions. Coordination emerges through observation, not optimization.

### Why Cantor pairing?

Creates self-similar attention structure that encodes geometric relationships. Points along diagonals receive consecutive indices, building hierarchical patterns.

### Why constitutive anchors?

Early experiments with additive anchors (biasing attention) failed because gradients died. Constitutive contribution ensures gradients flow.

### How many streams are optimal?

Depends on task complexity. Empirically:
- Simple (MNIST): 2-3 streams
- Medium (CIFAR): 3-5 streams
- Complex (ImageNet): 5-8 streams

More streams add capacity but increase coordination complexity.

### What makes good stream diversity?

Streams should differ in:
1. Training data (LAION vs WebImageText)
2. Architecture (ViT-B vs ViT-L)
3. Objective (contrastive vs self-supervised)
4. Resolution (patch32 vs patch14)

Same architecture trained differently often works well.

---

## Glossary

| Term | Definition |
|------|------------|
| **Collective** | Group of streams coordinated by router |
| **Stream** | Expert model + translation + router |
| **Fingerprint** | Unique learnable identity vector |
| **Anchor** | Shared behavioral prototype |
| **Mailbox** | Inter-router message passing |
| **Registry** | Singleton tracking all routers |
| **Emergence** | Collective >> max(individuals) |
| **Constitutive** | Directly contributing to output |
| **Adjacent Gating** | Parent-child fingerprint modulation |
| **Cantor Pairing** | Bijection ℕ² → ℕ with diagonal structure |

---

## Citation

```bibtex
@software{globalfractalrouter2025,
  author       = {AbstractPhil},
  title        = {GlobalFractalRouter: Collective Intelligence through 
                  Geometric Routing},
  year         = {2025},
  url          = {https://github.com/AbstractPhil/geofractal},
  note         = {Apache License 2.0}
}
```

---

## License

Apache License 2.0 with Attribution

See [LICENSE](../LICENSE) and [NOTICE](../NOTICE) for details.

---

*GlobalFractalRouter Documentation v1.0.0*