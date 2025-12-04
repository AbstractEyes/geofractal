# GeoFractal Router

**Collective Intelligence through Geometric Routing**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## What Is This?

GeoFractal Router is a coordination architecture for **collective intelligence**. Instead of building one smart model, you build multiple *different* models that communicate and triangulate truth together.

**The key insight:** Individual models don't need to be accurate. They need to *see differently*. The collective triangulates from divergent perspectives.
```
Traditional Ensemble:    Smart Model + Smart Model + Smart Model → Average
GeoFractal Collective:   Different View + Different View + Different View → Triangulate
```

**Proven emergence:**

| Experiment | Individual Accuracy | Collective Accuracy | Multiplier |
|------------|---------------------|---------------------|------------|
| ImageNet (5 CLIP streams) | 0.1% each | 84.68% | **847×** |
| FashionMNIST (3 streams) | 10% each | 93.4% | **9.34×** |
| Dual CLIP (frozen) | 7-18% | 92.6% | **5-13×** |

The collective achieves what no individual can.

---

## How It Works

### The Collective

A GeoFractal collective consists of multiple **streams** that process inputs through **heads**, coordinate through a **mailbox**, and fuse their perspectives:
```
                    ┌─────────────────────────────────────┐
                    │           Collective                 │
                    ├─────────────────────────────────────┤
   Input A ────────►│  Stream A ──► Head A ──┐            │
                    │                        │            │
   Input B ────────►│  Stream B ──► Head B ──┼──► Fusion ──► Output
                    │                        │            │
   Input C ────────►│  Stream C ──► Head C ──┘            │
                    │         ▲      │                    │
                    │         └──────┴─── Mailbox ◄───────│
                    └─────────────────────────────────────┘
```

Each stream sees the same problem from a different angle. The mailbox lets them observe each other's routing decisions. The fusion layer triangulates their divergent views into a unified answer.

### Why Divergence Matters

Traditional ensembles train each model to be accurate, then average their predictions. This works, but you're limited by what each model can learn individually.

GeoFractal inverts this: each stream is trained to be **different**, not accurate. When streams provide orthogonal projections of the input space, the routing fabric can recover the true answer even when no single stream knows it.

This is how sensor fusion works: multiple imperfect measurements → accurate state estimation.

### The Fingerprint

Every head has a unique **fingerprint** - a learnable identity vector that makes it see differently:
```python
# Each head's fingerprint creates divergent behavior through:
anchor_affinities = sigmoid(W @ fingerprint)      # Different behavioral modes
value_gating = v * sigmoid(W @ fingerprint)       # Different attention patterns  
score_bias = scores + (q @ W @ fingerprint) * 0.1 # Different routing decisions
```

Fingerprints don't encode "what to look for" - they encode "how to look."

### The Mailbox

Streams coordinate through a shared mailbox where they post their routing states:
```python
# After each head processes its input:
mailbox.post(head_id, routing_state.detach())

# Other heads can observe (content is detached - no gradient flow):
peer_states = mailbox.read_all(exclude=my_id)
```

**Critical:** Mailbox content is detached. Gradients don't flow through coordination. This prevents collapse to trivial solutions - specialization must emerge from observation, not optimization.

### Constitutive Contribution

All components must contribute **constitutively** to the output:
```python
# WRONG - gradients die:
attention_scores = scores + learned_bias

# RIGHT - gradients flow:
output = w0*attention + w1*routing + w2*anchor_contribution
```

This emerged from experimental failure. Additive-only biases become ignorable noise.

---

## Quick Start

### Installation
```bash
pip install geofractal
```

Or from source:
```bash
git clone https://github.com/AbstractPhil/geofractal.git
cd geofractal
pip install -e .
```

### Build a Collective
```python
from geofractal.router import (
    PrototypeBuilder,
    StreamSpec,
    HeadSpec,
    FusionSpec,
)

# Define streams with different input sources
prototype = (PrototypeBuilder("vision_collective")
    .add_stream(StreamSpec.feature_vector("clip_b32", input_dim=512))
    .add_stream(StreamSpec.feature_vector("clip_l14", input_dim=768))
    .add_stream(StreamSpec.feature_vector("dino_b16", input_dim=768))
    .with_head(HeadSpec.standard(feature_dim=512))
    .with_fusion(FusionSpec.gated(output_dim=512))
    .with_classifier(num_classes=1000)
    .build())

# Forward pass with dict of features
logits = prototype({
    "clip_b32": clip_b32_features,   # [B, 512]
    "clip_l14": clip_l14_features,   # [B, 768]  
    "dino_b16": dino_features,       # [B, 768]
})
```

### Stream Types

**Vector Streams** - for pooled embeddings `[B, D]`:
```python
StreamSpec.feature_vector("name", input_dim=512)      # Pre-extracted
StreamSpec.trainable_vector("name", input_dim=512)    # Learnable backbone
```

**Sequence Streams** - for token sequences `[B, S, D]`:
```python
StreamSpec.sequence("name", input_dim=768)                    # Pass-through
StreamSpec.transformer_sequence("name", input_dim=768)        # +Transformer layers
StreamSpec.conv_sequence("name", input_dim=768)               # +Conv layers
```

### Head Presets
```python
HeadSpec.lightweight(feature_dim=512)  # Minimal params
HeadSpec.standard(feature_dim=512)     # Balanced
HeadSpec.heavy(feature_dim=512)        # Maximum capacity
```

### Fusion Strategies
```python
FusionSpec.concat(output_dim=512)       # Concatenate + project
FusionSpec.gated(output_dim=512)        # Input-adaptive gates
FusionSpec.attention(output_dim=512)    # Cross-stream attention
FusionSpec.moe(output_dim=512)          # Mixture of experts
```

---

## Architecture

### Package Structure
```
geofractal/router/
├── __init__.py
├── collective.py
├── registry.py
├── config.py
├── run_tst_full.py
├── run_tst_gradient.py
├── run_tst_seq.py
├── factory/
│   ├── __init__.py
│   ├── builder.py
│   ├── protocols.py
│   ├── prototype.py
│   └── registry.py
├── fusion/
│   ├── __init__.py
│   ├── builder.py
│   ├── methods.py
│   └── protocols.py
├── getting_started/
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   ├── INDEX.md
│   ├── MATHEMATICS.md
│   └── PROTOCOL.md
├── head/
│   ├── __init__.py
│   ├── builder.py
│   ├── components.py
│   └── protocols.py
├── streams/
│   ├── __init__.py
│   ├── base.py
│   ├── builder.py
│   ├── feature.py
│   ├── frozen.py
│   ├── protocols.py
│   ├── sequence.py
│   ├── trainable.py
│   └── vector.py
```

### Head Components

| Component | Purpose |
|-----------|---------|
| **Fingerprint** | Unique identity that creates divergent behavior |
| **CantorAttention** | Geometric attention via Cantor pairing |
| **TopKRouter** | Sparse routing with fingerprint modulation |
| **ConstitutiveAnchorBank** | Shared behavioral modes (gradients flow) |
| **FingerprintGate** | Identity-based value gating |
| **LearnableWeightCombiner** | Weighted combination of pathways |
| **FFNRefinement** | Post-combination refinement |

### The Cantor Prior

Attention includes a geometric bias from Cantor pairing - a bijection ℕ² → ℕ where diagonal positions receive consecutive indices:
```
Position:     Cantor Index:
(0,0) (0,1)     0   2
(1,0) (1,1)     1   4
```

This encodes self-similar spatial relationships without learned parameters.

---

## Documentation

Detailed documentation is in `getting_started/`:

| Document | Description |
|----------|-------------|
| [INDEX.md](getting_started/INDEX.md) | Documentation overview and quick reference |
| [PROTOCOL.md](getting_started/PROTOCOL.md) | Complete protocol specification |
| [ARCHITECTURE.md](getting_started/ARCHITECTURE.md) | Component deep dive |
| [MATHEMATICS.md](getting_started/MATHEMATICS.md) | Formal foundations |
| [RATIONALITY.md](getting_started/RATIONALITY.md) | Why each component exists |
| [FUSION.md](getting_started/FUSION.md) | Fusion strategy guide |
| [FACTORY.md](getting_started/FACTORY.md) | Prototype building guide |

---

## Key Principles

### 1. Divergence Over Accuracy

Don't optimize individual streams for accuracy. Optimize the collective for emergence.

### 2. Coordination Without Gradients

Mailbox content is detached. Streams learn to coordinate through observation, not backpropagation. This prevents collapse.

### 3. Constitutive Contribution

Every learned component must contribute directly to output. Additive biases die.

### 4. Fingerprint Identity

Each head's fingerprint creates its unique perspective. Same architecture, different identity, different behavior.

---

## When to Use GeoFractal

**Good fit:**
- Multiple pre-trained models available (CLIP, DINO, BERT, T5, etc.)
- Task benefits from diverse perspectives
- Individual models underperform but see different things
- You want emergence, not just averaging

**Not ideal:**
- Single model already solves the task well
- Inputs are homogeneous (no diversity to exploit)
- Latency-critical inference (collective adds overhead)

---

## Citation
```bibtex
@software{geofractalrouter2025,
  author       = {AbstractPhil},
  title        = {GeoFractal Router: Collective Intelligence through 
                  Geometric Routing},
  year         = {2025},
  url          = {https://github.com/AbstractPhil/geofractal}
}
```

---

## License

Apache License 2.0 with Attribution

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

---

*"Individual streams don't need to classify accurately. They need to see differently. The routing fabric triangulates truth from divergent viewpoints."*