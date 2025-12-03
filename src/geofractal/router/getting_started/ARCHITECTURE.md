# Architecture Deep Dive

**GlobalFractalRouter Component Architecture**

---

## Overview

This document provides an exhaustive examination of each architectural component, its design rationale, internal mechanics, and integration points.

---

## 1. GlobalFractalRouter

### 1.1 Purpose

The GlobalFractalRouter is the core routing module that transforms input sequences through geometric attention, sparse routing, and anchor-based behavioral modes. It is designed to:

1. Create divergent behavior through unique fingerprints
2. Encode geometric structure through Cantor pairing
3. Enable collective coordination through mailbox posting
4. Provide constitutive output through weighted combination

### 1.2 Architecture Diagram

```
                            ┌─────────────────────────────────────┐
                            │       GlobalFractalRouter           │
                            ├─────────────────────────────────────┤
Input x ─────────────────────►  input_norm (LayerNorm)            │
[B, S, D]                   │           │                         │
                            │           ▼                         │
                            │  ┌─────────────────────────────┐   │
                            │  │  CantorMultiHeadAttention   │   │
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
                            │  │       AnchorBank            │   │
                            │  │  • Compute affinities       │   │
                            │  │  • Weight anchors           │   │
                            │  │  • Project to output        │   │
                            │  └──────────────┬──────────────┘   │
                            │                 │ anchor_out        │
                            │                 ▼                   │
                            │  ┌─────────────────────────────┐   │
                            │  │     Adjacent Gating         │   │
                            │  │  (if target_fingerprint)    │   │
                            │  │  routed *= similarity_gate  │   │
                            │  └──────────────┬──────────────┘   │
                            │                 │                   │
                            │                 ▼                   │
                            │  ┌─────────────────────────────┐   │
                            │  │     Combination             │   │
                            │  │  w₀·attn + w₁·routed +      │   │
                            │  │  w₂·anchor                  │   │
                            │  └──────────────┬──────────────┘   │
                            │                 │ combined          │
                            │                 ▼                   │
                            │     x + combined (residual)         │
                            │                 │                   │
                            │                 ▼                   │
                            │        output_norm (LayerNorm)      │
                            │                 │                   │
                            │                 ▼                   │
                            │  ┌─────────────────────────────┐   │
                            │  │          FFN                │   │
                            │  │  Linear(D, 4D) → GELU →     │   │
                            │  │  Dropout → Linear(4D, D)    │   │
                            │  └──────────────┬──────────────┘   │
                            │                 │                   │
                            │                 ▼                   │
                            │     + residual (second residual)    │
                            │                 │                   │
                            │                 ▼                   │
                            │           Mailbox Post              │
                            │     (routing_state, affinities)     │
                            │                 │                   │
                            └─────────────────┼───────────────────┘
                                              │
                                              ▼
                                    Output [B, S, D]
```

### 1.3 Parameter Breakdown

For a router with feature_dim=512, fingerprint_dim=64, num_anchors=16, num_routes=4, num_heads=8:

| Component | Parameters |
|-----------|------------|
| Fingerprint | 64 |
| CantorMultiHeadAttention | ~1.05M |
| TopKRouter | ~295K |
| AnchorBank | ~274K |
| FingerprintGate | ~41K |
| Norms + FFN | ~2.1M |
| **Total per router** | **~3.76M** |

### 1.4 Critical Design Decisions

**Learnable Combination Weights:**
```python
combine_weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.1]))
weights = F.softmax(combine_weights, dim=0)
combined = weights[0]*attn + weights[1]*routed + weights[2]*anchor
```

Allows network to learn optimal balance. Empirically converges to ~0.45 for attention, ~0.45 for routing, ~0.1 for anchors.

**Two Residual Connections:**
```python
output = x + combined                    # First: preserve input
output = output + ffn(output_norm(...))  # Second: refine
```

**Detached Mailbox Content:**
```python
mailbox.post(id, name, routing_state.detach())
```

Prevents gradient flow → emergent coordination without collapse.

---

## 2. CantorMultiHeadAttention

### 2.1 Purpose

Multi-head self-attention with geometric prior from Cantor pairing.

### 2.2 Cantor Bias Structure

For a 4×4 grid, Cantor indices form diagonal patterns:

```
Position (y,x):    Cantor Index:
(0,0) (0,1) (0,2) (0,3)     0   2   5   9
(1,0) (1,1) (1,2) (1,3)     1   4   8  13
(2,0) (2,1) (2,2) (2,3)     3   7  12  18
(3,0) (3,1) (3,2) (3,3)     6  11  17  24
```

The bias matrix has higher values for positions with similar Cantor indices.

### 2.3 Per-Head Scale

```python
cantor_scale = nn.Parameter(torch.ones(num_heads) * 0.1)
```

Each head learns how much to weight geometric structure vs content.

---

## 3. AnchorBank

### 3.1 Purpose

Shared behavioral modes across all streams in a collective.

### 3.2 Constitutive Requirement

**WRONG (gradients die):**
```python
scores = attention_scores + anchor_bias
```

**RIGHT (gradients flow):**
```python
anchor_out = linear(weighted_anchors)
combined = w0*attn + w1*routed + w2*anchor_out
```

Anchors MUST contribute to output, not just bias attention.

### 3.3 Anchor Specialization

Post-training, anchors often specialize:
- Texture patterns (fur, scales)
- Shape patterns (circular, elongated)
- Composition patterns (centered, scattered)

---

## 4. FingerprintGate

### 4.1 Value Gating

```python
gate = sigmoid(linear(fingerprint))  # [D]
gated_v = v * gate  # Different view per fingerprint
```

### 4.2 Adjacent Gating

```python
similarity = sigmoid(mlp(cat([fp_self, fp_target])))
routed_out = routed_out * similarity
```

Creates implicit curriculum between adjacent streams.

---

## 5. TopKRouter

### 5.1 Algorithm

1. Compute attention-like scores
2. Add fingerprint bias (scale 0.1)
3. Select top-K positions
4. Softmax over selected
5. Weighted gather of values

### 5.2 Sparsity Benefits

- Implicit regularization (limited gradient paths)
- Specialization pressure (must choose what to attend)
- Memory efficiency for long sequences

---

## 6. Mailbox & Registry

### 6.1 Mailbox Protocol

```python
# Start of forward
mailbox.clear()

# Each stream posts after routing
mailbox.post(id, name, routing_state.detach())

# Streams can read (future use)
peer_states = mailbox.read_all(exclude=self.id)
```

### 6.2 Registry Hierarchy

```
Stream_0 ──parent──► Stream_1 ──parent──► Stream_2
    │                    │                    │
    └──children──────────┘                    │
                         └──children──────────┘
```

---

## 7. Stream Types

| Type | Backbone | Training | Use Case |
|------|----------|----------|----------|
| FeatureStream | None | Router only | Pre-extracted features |
| FrozenStream | Frozen | Router only | CLIP, DINO, etc. |
| TrainableStream | Learnable | Full | Custom encoders |

---

## 8. Integration Patterns

### Adding Custom Stream

```python
class CustomStream(BaseStream):
    def encode(self, x):
        return self.custom_encoder(x)
```

### Custom Topology

Override parent_id assignments in stream construction.

### Custom Coordination

Subclass GlobalFractalRouter and override forward to read mailbox.

---

*End of Architecture Deep Dive*