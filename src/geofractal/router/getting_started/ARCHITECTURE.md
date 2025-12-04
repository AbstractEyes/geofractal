# Architecture Deep Dive

**GlobalFractalRouter Component Architecture**

---

## Overview

This document provides an exhaustive examination of each architectural component, its design rationale, internal mechanics, and integration points.

---

## 1. Package Structure
```
geofractal/router/
├── __init__.py              # Public API exports
├── config.py                # GlobalFractalRouterConfig, presets
├── registry.py              # RouterRegistry, RouterMailbox
├── factory/
│   ├── __init__.py
│   ├── builder.py           # PrototypeBuilder (fluent API)
│   ├── protocols.py         # StreamSpec, HeadSpec, FusionSpec
│   ├── prototype.py         # AssembledPrototype
│   └── registry.py          # PrototypeRegistry, ComponentSwapper
├── fusion/
│   ├── __init__.py
│   ├── builder.py           # FusionBuilder
│   ├── methods.py           # Concat, Gated, Attention, MoE, etc.
│   └── protocols.py         # FusionStrategy, FusionConfig
├── head/
│   ├── __init__.py
│   ├── builder.py           # HeadBuilder, ComposedHead
│   ├── components.py        # CantorAttention, TopKRouter, Anchors, Gates
│   └── protocols.py         # HeadConfig, component interfaces
├── streams/
│   ├── __init__.py
│   ├── base.py              # BaseStream (abstract)
│   ├── builder.py           # StreamBuilder
│   ├── feature.py           # FeatureStream (legacy)
│   ├── frozen.py            # FrozenStream (legacy)
│   ├── protocols.py         # StreamProtocol, InputShape
│   ├── sequence.py          # SequenceStream, TransformerSequenceStream
│   ├── trainable.py         # TrainableStream (legacy)
│   └── vector.py            # VectorStream, FeatureVectorStream
└── getting_started/
    ├── INDEX.md
    ├── ARCHITECTURE.md      # (this file)
    ├── PROTOCOL.md
    ├── MATHEMATICS.md
    ├── DEVELOPMENT.md
    ├── METRICS.md
    ├── INIT_BREAKDOWN.md
    └── RATIONALITY.md
```

---

## 2. ComposedHead

### 2.1 Purpose

The ComposedHead is the assembled routing head built from modular components. It transforms stream outputs through geometric attention, sparse routing, and anchor-based behavioral modes.

### 2.2 Architecture Diagram
```
                            ┌─────────────────────────────────────┐
                            │          ComposedHead               │
                            ├─────────────────────────────────────┤
Input x ─────────────────────►  input_norm (LayerNorm)            │
[B, S, D]                   │           │                         │
                            │           ▼                         │
                            │  ┌─────────────────────────────┐   │
                            │  │      CantorAttention        │   │
                            │  │  • Q, K, V projections      │   │
                            │  │  • Cantor bias per head     │   │
                            │  │  • Multi-head attention     │   │
                            │  └──────────────┬──────────────┘   │
                            │                 │ attn_out          │
                            │                 ▼                   │
                            │  ┌─────────────────────────────┐   │
                            │  │     FingerprintGate         │   │
                            │  │  • gate_values(v, fp)       │   │
                            │  └──────────────┬──────────────┘   │
                            │                 │ v_gated           │
                            │                 ▼                   │
                            │  ┌─────────────────────────────┐   │
                            │  │       TopKRouter            │   │
                            │  │  • Score computation        │   │
                            │  │  • Fingerprint bias         │   │
                            │  │  • Top-K selection          │   │
                            │  │  • Weighted gathering       │   │
                            │  └──────────────┬──────────────┘   │
                            │                 │ routed_out        │
                            │                 ▼                   │
                            │  ┌─────────────────────────────┐   │
                            │  │  ConstitutiveAnchorBank     │   │
                            │  │  • Compute affinities       │   │
                            │  │  • Weight anchors           │   │
                            │  │  • Project to output        │   │
                            │  └──────────────┬──────────────┘   │
                            │                 │ anchor_out        │
                            │                 ▼                   │
                            │  ┌─────────────────────────────┐   │
                            │  │  LearnableWeightCombiner    │   │
                            │  │  w₀·attn + w₁·routed +      │   │
                            │  │  w₂·anchor                  │   │
                            │  └──────────────┬──────────────┘   │
                            │                 │ combined          │
                            │                 ▼                   │
                            │     x + combined (residual)         │
                            │                 │                   │
                            │                 ▼                   │
                            │  ┌─────────────────────────────┐   │
                            │  │      FFNRefinement          │   │
                            │  │  Linear(D, 4D) → GELU →     │   │
                            │  │  Dropout → Linear(4D, D)    │   │
                            │  └──────────────┬──────────────┘   │
                            │                 │                   │
                            │                 ▼                   │
                            │     + residual (second residual)    │
                            │                 │                   │
                            └─────────────────┼───────────────────┘
                                              │
                                              ▼
                                    Output [B, S, D]
```

### 2.3 Building a Head
```python
from geofractal.router.head import HeadBuilder, HeadConfig

config = HeadConfig(
    feature_dim=512,
    fingerprint_dim=64,
    num_heads=8,
    num_anchors=16,
    num_routes=4,
)

head = (HeadBuilder(config)
    .with_cantor_attention()
    .with_topk_router()
    .with_constitutive_anchors()
    .with_fingerprint_gate()
    .with_learnable_combiner()
    .with_ffn_refinement()
    .build())
```

### 2.4 Parameter Breakdown

For feature_dim=512, fingerprint_dim=64, num_anchors=16, num_routes=4, num_heads=8:

| Component | Parameters | Location |
|-----------|------------|----------|
| Fingerprint | 64 | `head.fingerprint` |
| CantorAttention | ~1.05M | `head/components.py` |
| TopKRouter | ~295K | `head/components.py` |
| ConstitutiveAnchorBank | ~274K | `head/components.py` |
| FingerprintGate | ~41K | `head/components.py` |
| LearnableWeightCombiner | 3 | `head/components.py` |
| FFNRefinement | ~2.1M | `head/components.py` |
| **Total per head** | **~3.76M** |

### 2.5 Head Presets
```python
from geofractal.router.factory import HeadSpec

# Lightweight (fewer params)
HeadSpec.lightweight(feature_dim=512)

# Standard (balanced)
HeadSpec.standard(feature_dim=512)

# Heavy (maximum capacity)
HeadSpec.heavy(feature_dim=512)
```

---

## 3. Head Components

### 3.1 CantorAttention

**Location:** `head/components.py`

Multi-head self-attention with geometric prior from Cantor pairing.

**Cantor Bias Structure:**

For a 4×4 grid, Cantor indices form diagonal patterns:
```
Position (y,x):    Cantor Index:
(0,0) (0,1) (0,2) (0,3)     0   2   5   9
(1,0) (1,1) (1,2) (1,3)     1   4   8  13
(2,0) (2,1) (2,2) (2,3)     3   7  12  18
(3,0) (3,1) (3,2) (3,3)     6  11  17  24
```

**Per-Head Scale:**
```python
cantor_scale = nn.Parameter(torch.ones(num_heads) * 0.1)
```

Each head learns how much to weight geometric structure vs content.

### 3.2 ConstitutiveAnchorBank

**Location:** `head/components.py`

Shared behavioral modes that MUST contribute constitutively to output.

**WRONG (gradients die):**
```python
scores = attention_scores + anchor_bias
```

**RIGHT (gradients flow):**
```python
anchor_out = linear(weighted_anchors)
combined = w0*attn + w1*routed + w2*anchor_out
```

Post-training, anchors often specialize:
- Texture patterns (fur, scales)
- Shape patterns (circular, elongated)
- Composition patterns (centered, scattered)

### 3.3 FingerprintGate

**Location:** `head/components.py`

**Value Gating:**
```python
gate = sigmoid(linear(fingerprint))  # [D]
gated_v = v * gate  # Different view per fingerprint
```

Creates unique perspective per stream via identity-based modulation.

### 3.4 TopKRouter

**Location:** `head/components.py`

**Algorithm:**
1. Compute attention-like scores
2. Add fingerprint bias (scale 0.1)
3. Select top-K positions
4. Softmax over selected
5. Weighted gather of values

**Benefits:**
- Implicit regularization (limited gradient paths)
- Specialization pressure (must choose what to attend)
- Memory efficiency for long sequences

### 3.5 LearnableWeightCombiner

**Location:** `head/components.py`
```python
combine_weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.1]))
weights = F.softmax(combine_weights, dim=0)
combined = weights[0]*attn + weights[1]*routed + weights[2]*anchor
```

Empirically converges to ~0.45 attention, ~0.45 routing, ~0.1 anchors.

### 3.6 FFNRefinement

**Location:** `head/components.py`
```python
ffn = nn.Sequential(
    nn.Linear(D, 4*D),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(4*D, D),
)
```

---

## 4. Stream Types

### 4.1 Input Shape Categories

| Shape | Dimensions | Description |
|-------|------------|-------------|
| VECTOR | `[B, D]` | Single vector per sample |
| SEQUENCE | `[B, S, D]` | Token sequence |

### 4.2 Vector Streams

**Location:** `streams/vector.py`

For `[B, D]` inputs (pooled embeddings, CLS tokens).

| Class | Use Case |
|-------|----------|
| `FeatureVectorStream` | Pre-extracted features (CLIP, DINO) |
| `TrainableVectorStream` | Learnable backbone |

**Slot Expansion:**
```python
# [B, D] → [B, num_slots, D]
translations = nn.Parameter(torch.randn(num_slots, D))
x_expanded = x.unsqueeze(1) + translations
```

### 4.3 Sequence Streams

**Location:** `streams/sequence.py`

For `[B, S, D]` inputs (token sequences, hidden states).

| Class | Use Case |
|-------|----------|
| `SequenceStream` | Pass-through (already sequential) |
| `TransformerSequenceStream` | Adds transformer layers |
| `ConvSequenceStream` | Multi-scale convolutions |

**No artificial expansion** - routes tokens directly.

### 4.4 Legacy Aliases
```python
# For backward compatibility
FrozenStream = FeatureVectorStream
FeatureStream = FeatureVectorStream
TrainableStream = TrainableVectorStream
```

---

## 5. Fusion Strategies

**Location:** `fusion/methods.py`

| Strategy | When to Use | Mechanism |
|----------|-------------|-----------|
| `concat` | Baseline | Concatenate + project |
| `weighted` | Known quality differences | Learnable scalar weights |
| `gated` | Input-adaptive | Per-sample gates |
| `attention` | Cross-stream relations | Multi-head cross-attention |
| `fingerprint` | Identity-guided | Fingerprint similarity weighting |
| `residual` | Stable training | Sum with learnable scale |
| `moe` | Sparse selection | Top-K expert routing |
| `hierarchical` | Many streams | Tree-structured merging |

**Building Fusion:**
```python
from geofractal.router.fusion import FusionBuilder, FusionStrategy

fusion = (FusionBuilder()
    .with_streams({"clip": 512, "dino": 768})
    .with_output_dim(512)
    .with_strategy(FusionStrategy.GATED)
    .build())
```

---

## 6. Factory System

### 6.1 PrototypeBuilder

**Location:** `factory/builder.py`

Fluent API for assembling multi-stream prototypes.
```python
from geofractal.router.factory import (
    PrototypeBuilder,
    StreamSpec,
    HeadSpec,
    FusionSpec,
)

prototype = (PrototypeBuilder("my_prototype")
    .add_stream(StreamSpec.feature_vector("clip", input_dim=512))
    .add_stream(StreamSpec.sequence("t5", input_dim=768))
    .with_head(HeadSpec.standard(feature_dim=512))
    .with_fusion(FusionSpec.gated(output_dim=512))
    .with_classifier(num_classes=1000)
    .build())
```

### 6.2 StreamSpec

**Location:** `factory/protocols.py`

| Factory Method | Input Shape | Description |
|----------------|-------------|-------------|
| `.feature_vector()` | `[B, D]` | Pre-extracted features |
| `.trainable_vector()` | `[B, D]` | Learnable backbone |
| `.sequence()` | `[B, S, D]` | Pass-through sequence |
| `.transformer_sequence()` | `[B, S, D]` | With transformer layers |
| `.conv_sequence()` | `[B, S, D]` | With conv layers |

### 6.3 ComponentSwapper

**Location:** `factory/registry.py`

Runtime modification of assembled prototypes.
```python
from geofractal.router.factory import ComponentSwapper

swapper = ComponentSwapper(prototype)
swapper.swap_fusion(new_fusion)
swapper.add_stream("new_stream", stream, head, input_shape="sequence")
swapper.remove_stream("old_stream")
```

---

## 7. Mailbox & Registry

### 7.1 Mailbox Protocol

**Location:** `registry.py`
```python
# Start of forward
mailbox.clear()

# Each stream posts after routing
mailbox.post(id, name, routing_state.detach())

# Streams can read (future use)
peer_states = mailbox.read_all(exclude=self.id)
```

**Detached content prevents gradient flow** → emergent coordination without collapse.

### 7.2 Registry

**Location:** `registry.py`

Singleton tracking all active routers for collective coordination.

---

## 8. Integration Patterns

### Custom Head Component
```python
from geofractal.router.head import HeadBuilder

class CustomGate(nn.Module):
    def forward(self, x, fingerprint):
        ...

head = (HeadBuilder(config)
    .with_gate(CustomGate)
    .build())
```

### Custom Stream
```python
from geofractal.router.streams import BaseStream

class CustomStream(BaseStream):
    @property
    def input_shape(self):
        return InputShape.SEQUENCE
    
    def encode(self, x):
        return self.custom_encoder(x)
```

### Custom Fusion
```python
from geofractal.router.fusion import FusionBuilder

class CustomFusion(nn.Module):
    def forward(self, stream_outputs):
        ...

builder.with_custom_fusion(CustomFusion(...))
```

---

*End of Architecture Deep Dive*