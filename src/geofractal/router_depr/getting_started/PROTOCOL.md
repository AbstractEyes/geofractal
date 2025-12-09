# GlobalFractalRouter Protocol Specification

**Version:** 2.0.0  
**Author:** AbstractPhil  
**Date:** December 2025  
**License:** Apache 2.0 with Attribution

---

## Abstract

The GlobalFractalRouter (GFR) is a coordination infrastructure that enables emergent collective intelligence through geometric routing. Unlike traditional ensemble methods that average predictions from individually competent models, GFR coordinates *divergent perspectives* from individually incompetent streams to achieve collective mastery.

**Core Insight:** Individual streams don't need to classify accurately. They need to *see differently*. The routing fabric triangulates truth from divergent viewpoints.

**Proven Results:**
- ImageNet: 5 streams at 0.1% individual → 84.68% collective
- FashionMNIST: 10% + 10% + 10% = 93.4%
- Dual CLIP: 98.6% frozen parameters → 92.6% accuracy

---

## Table of Contents

1. [Foundational Principles](#1-foundational-principles)
2. [Protocol Overview](#2-protocol-overview)
3. [Component Specification](#3-component-specification)
4. [Information Flow](#4-information-flow)
5. [Coordination Mechanisms](#5-coordination-mechanisms)
6. [Mathematical Foundations](#6-mathematical-foundations)
7. [Implementation Requirements](#7-implementation-requirements)
8. [Conformance Criteria](#8-conformance-criteria)

---

## 1. Foundational Principles

### 1.1 The Divergence Principle

Traditional machine learning optimizes individual model accuracy:
```
Model_i → max P(correct | input)
Ensemble = average(Model_1, Model_2, ..., Model_n)
```

GFR inverts this paradigm:
```
Stream_i → max Divergence(Stream_i, Stream_j) for all j ≠ i
Collective = Route(Stream_1, Stream_2, ..., Stream_n)
```

**Why this works:** When streams provide orthogonal projections of the input space, the routing fabric can triangulate the true answer even when no individual stream knows it. This is analogous to how multiple imperfect sensors can be fused into accurate state estimation.

### 1.2 The Fingerprint Axiom

Every stream in a GFR collective MUST possess a unique fingerprint:
```
∀ Stream_i, Stream_j where i ≠ j:
    Fingerprint_i ≢ Fingerprint_j
```

Fingerprints serve three functions:
1. **Identity:** Distinguish streams in the registry
2. **Divergence:** Create different routing behaviors
3. **Coordination:** Enable fingerprint-based gating between streams

### 1.3 The Coordination Imperative

Streams MUST be able to observe and respond to collective state:
```
Output_i = f(Input_i, Fingerprint_i, CollectiveState)
```

This is achieved through the Mailbox mechanism, where streams post their routing states for others to read.

### 1.4 The Constitutive Requirement

All learned components MUST contribute constitutively to the output, not merely additively to attention scores:
```
WRONG:  attention_scores = scores + learned_bias  (gradients die)
RIGHT:  output = f(input) + g(learned_component)  (gradients flow)
```

This requirement emerged from experimental failure: additive-only components become ignorable noise.

---

## 2. Protocol Overview

### 2.1 System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    AssembledPrototype                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐       │
│  │ Stream  │  │ Stream  │  │ Stream  │  ...  │ Stream  │       │
│  │    0    │  │    1    │  │    2    │       │    N    │       │
│  └────┬────┘  └────┬────┘  └────┬────┘       └────┬────┘       │
│       │            │            │                  │            │
│       ▼            ▼            ▼                  ▼            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐       │
│  │  Head   │  │  Head   │  │  Head   │  ...  │  Head   │       │
│  │    0    │  │    1    │  │    2    │       │    N    │       │
│  └────┬────┘  └────┬────┘  └────┬────┘       └────┬────┘       │
│       │            │            │                  │            │
│       ▼            ▼            ▼                  ▼            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Mailbox                              │   │
│  │  (Shared coordination fabric - routing states)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │            │            │                  │            │
│       ▼            ▼            ▼                  ▼            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Fusion Layer                           │   │
│  │  (Gated / Attention / MoE / Concat+Project)              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Classifier                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Stream + Head Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                      Stream + Head                               │
├─────────────────────────────────────────────────────────────────┤
│  Input ──► [Encoder] ──► [Prepare for Head]                     │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    ComposedHead                          │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │   │
│  │  │Fingerprint│  │  Cantor   │  │Constitutive│           │   │
│  │  │   [F]     │  │ Attention │  │AnchorBank  │           │   │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘            │   │
│  │        │              │              │                   │   │
│  │        ▼              ▼              ▼                   │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │         LearnableWeightCombiner              │    │   │
│  │  │  w₀·Attention + w₁·Routing + w₂·Anchors         │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │                         │                                │   │
│  │                         ▼                                │   │
│  │                  [FFNRefinement]                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│                         [Pool] ──► Output                       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Protocol Messages

The GFR protocol defines the following message types:

| Message | Direction | Content | Purpose |
|---------|-----------|---------|---------|
| `REGISTER` | Stream → Registry | name, parent_id, fingerprint_dim | Join collective |
| `POST` | Stream → Mailbox | routing_state, anchor_affinities | Share state |
| `READ` | Stream ← Mailbox | List[routing_states] | Observe collective |
| `GATE` | Stream → Stream | fingerprint_similarity | Adjacent coordination |

---

## 3. Component Specification

### 3.1 Fingerprint

**Location:** `head/builder.py` (ComposedHead.fingerprint)

**Purpose:** Create unique identity that induces divergent behavior.

**Specification:**
```python
fingerprint: nn.Parameter  # Shape: [fingerprint_dim]
initialization: torch.randn(fingerprint_dim) * 0.02
```

**Requirements:**
- MUST be learnable (requires_grad=True)
- MUST be unique per stream (initialized randomly)
- SHOULD have dimension 64 for most applications
- MUST influence routing decisions (see FingerprintGate)

**Divergence Mechanism:**

The fingerprint creates divergence through three pathways:

1. **Anchor Affinities:** `affinities = sigmoid(W_anchor @ fingerprint)`
2. **Value Gating:** `gated_v = v * sigmoid(W_gate @ fingerprint)`
3. **Score Bias:** `scores = scores + (q @ W_bias @ fingerprint) * 0.1`

### 3.2 Cantor Pairing

**Location:** `head/components.py` (CantorAttention, cantor_pair, build_cantor_bias)

**Purpose:** Encode self-similar geometric structure in attention patterns.

**Definition:**
```
π(x, y) = ((x + y)(x + y + 1))/2 + y
```

The Cantor pairing function creates a bijection ℕ² → ℕ with the property that points along diagonals receive consecutive indices. This encodes a natural hierarchical structure.

**Attention Bias:**
```python
cantor_indices = cantor_pair(positions[:, 0], positions[:, 1])
diff = |cantor_indices[i] - cantor_indices[j]|
bias = 1.0 - (diff / max_diff)  # Normalized similarity
```

**Why Cantor:**
- Encodes 2D spatial relationships in 1D sequence
- Creates self-similar attention patterns at multiple scales
- Provides geometric prior without learned parameters
- Enables reasoning about spatial hierarchies

### 3.3 Anchor Bank

**Location:** `head/components.py` (ConstitutiveAnchorBank)

**Purpose:** Shared behavioral modes that streams align to.

**Specification:**
```python
anchors: nn.Parameter      # Shape: [num_anchors, feature_dim]
fp_to_anchor: MLP          # fingerprint_dim → num_anchors
anchor_out: nn.Linear      # feature_dim → feature_dim
```

**Forward Pass:**
```python
affinities = sigmoid(fp_to_anchor(fingerprint))  # [A]
weighted = (anchors * affinities.unsqueeze(-1)).sum(dim=0)  # [D]
output = anchor_out(weighted)  # [D] - CONSTITUTIVE
```

**Requirements:**
- MUST contribute constitutively to output (not just bias attention)
- SHOULD have 8-16 anchors for most applications
- Anchors are shared across all streams (collective behavioral modes)

**Interpretation:**

Anchors can be understood as:
- **Behavioral prototypes:** Common patterns the collective learns
- **Coordination signals:** Shared vocabulary for inter-stream communication
- **Regularization:** Prevents streams from diverging too far

### 3.4 TopK Router

**Location:** `head/components.py` (TopKRouter)

**Purpose:** Sparse routing that selects relevant positions.

**Specification:**
```python
K: int                     # Number of routes (default: 4-8)
score_proj: nn.Linear      # feature_dim → feature_dim
fp_to_bias: nn.Linear      # fingerprint_dim → feature_dim
```

**Forward Pass:**
```python
scores = (score_proj(q) @ k.T) / sqrt(D)
scores += (q @ fp_to_bias(fingerprint)) * 0.1
topk_scores, routes = topk(scores / temperature, K)
weights = softmax(topk_scores)
output = sum(weights * gather(v, routes))
```

**Requirements:**
- K SHOULD be much smaller than sequence length (sparse)
- Temperature SHOULD be 1.0 unless tuning for specific behavior
- Fingerprint bias MUST be small (0.1 scale) to not dominate

### 3.5 FingerprintGate

**Location:** `head/components.py` (FingerprintGate)

**Purpose:** Create unique perspective per stream via identity-based modulation.

**Value Gating:**
```python
gate = sigmoid(linear(fingerprint))  # [D]
gated_v = v * gate  # Different view per fingerprint
```

**Adjacent Gating:**
```python
similarity = sigmoid(mlp(cat([fp_self, fp_target])))
routed_out = routed_out * similarity
```

### 3.6 Combiner

**Location:** `head/components.py` (LearnableWeightCombiner)

**Purpose:** Combine attention, routing, and anchor outputs.

**Specification:**
```python
combine_weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.1]))
weights = F.softmax(combine_weights, dim=0)
combined = weights[0]*attn + weights[1]*routed + weights[2]*anchor
```

Empirically converges to ~0.45 attention, ~0.45 routing, ~0.1 anchors.

### 3.7 Mailbox

**Location:** `registry.py` (RouterMailbox)

**Purpose:** Inter-router communication for collective coordination.

**Specification:**
```python
messages: Dict[module_id, RouterMessage]
RouterMessage:
    sender_id: str
    sender_name: str
    content: Tensor  # Detached (no gradients through mailbox)
    timestamp: int
```

**Protocol:**
```python
# Post (after routing)
routing_state = cat([route_weights.mean(), anchor_affinities])
mailbox.post(module_id, name, routing_state)

# Read (before routing, optional)
peer_states = mailbox.read_all(exclude=module_id)
```

**Requirements:**
- Content MUST be detached (no gradient flow through mailbox)
- Mailbox MUST be cleared at start of each collective forward pass
- Timestamps MUST be monotonically increasing within a forward pass

### 3.8 Registry

**Location:** `registry.py` (RouterRegistry)

**Purpose:** Track all routers and their relationships.

**Specification:**
```python
routers: Dict[module_id, RouterInfo]
groups: Dict[group_name, Set[module_id]]
name_to_id: Dict[name, module_id]

RouterInfo:
    module_id: str
    name: str
    parent_id: Optional[str]
    cooperation_group: str
    fingerprint_dim: int
    feature_dim: int
    children: Set[str]
```

**Requirements:**
- Registry MUST be singleton (one per process)
- Registry SHOULD be reset before building new collective
- Parent-child relationships define information flow hierarchy

---

## 4. Information Flow

### 4.1 Forward Pass Sequence
```
1. Prototype.forward(inputs)
   │
   ├─► mailbox.clear()
   │
   ├─► for name in stream_names:
   │       │
   │       ├─► encoded = streams[name](inputs[name])
   │       │
   │       ├─► head_out = heads[name](encoded)
   │       │
   │       ├─► projected = projections[name](head_out)
   │       │
   │       └─► stream_outputs[name] = projected.mean(dim=1)  # pool
   │
   ├─► fused = fusion(stream_outputs)
   │
   └─► logits = classifier(fused)
```

### 4.2 Head Internal Flow
```
ComposedHead.forward(x)
   │
   ├─► x_norm = input_norm(x)
   │
   ├─► attn_out = attention(x_norm)  # CantorAttention
   │
   ├─► v_gated = gate.gate_values(x_norm, fingerprint)
   │
   ├─► routed = router(attn_out, x_norm, v_gated, fingerprint)
   │
   ├─► anchor_out, affinities = anchors(x_norm, fingerprint)
   │
   ├─► combined = combiner(attn_out, routed, anchor_out)
   │
   ├─► output = x + combined
   │
   └─► output = output + refinement(output)
```

### 4.3 Gradient Flow
```
Loss
  │
  ▼
Classifier ◄── gradients flow
  │
  ▼
Fusion ◄────── gradients flow
  │
  ├─────────────────────────────────┐
  ▼                                 ▼
Head[0]                        Head[N]
  │                                 │
  ├─► fingerprint ◄─ gradients     ├─► fingerprint ◄─ gradients
  ├─► anchors ◄──── gradients      ├─► anchors ◄──── gradients
  ├─► attention ◄── gradients      ├─► attention ◄── gradients
  │                                 │
  ▼                                 ▼
Stream[0]                      Stream[N]
  │                                 │
  ▼                                 ▼
(frozen or trainable)         (frozen or trainable)

NOTE: Mailbox content is DETACHED - no gradients flow through mailbox
```

---

## 5. Coordination Mechanisms

### 5.1 Adjacent Gating

**Purpose:** Enable parent-child information flow in stream hierarchy.

**Mechanism:**
```python
# In head forward, when target_fingerprint is provided:
gate = sigmoid(MLP(cat([self.fingerprint, target_fingerprint])))
routed_output = routed_output * gate
```

**Effect:**
- Parent streams modulate their output based on child fingerprint
- Creates implicit curriculum: earlier streams condition later ones
- Enables hierarchical specialization

### 5.2 Mailbox Coordination

**Purpose:** Enable streams to observe and respond to collective state.

**Emergent Specialization:**

Even without explicit reading, the mailbox ordering creates implicit coordination:
- Stream 0 always goes first (no peer states available)
- Stream N sees all prior states
- This asymmetry induces role differentiation

### 5.3 Fingerprint Similarity

**Purpose:** Measure and utilize relationships between streams.

**Mechanism:**
```python
similarity = sigmoid(MLP(cat([fp_self, fp_target])))
```

**Applications:**
1. **Adjacent gating:** Modulate output based on next stream
2. **Routing influence:** Route based on stream similarity
3. **Analysis:** Visualize stream relationships post-training

---

## 6. Mathematical Foundations

### 6.1 Cantor Pairing Function

**Definition:**
$$\pi(x, y) = \frac{(x + y)(x + y + 1)}{2} + y$$

**Properties:**
- Bijection: $\pi: \mathbb{N}^2 \rightarrow \mathbb{N}$
- Diagonal structure: Points $(0,k), (1,k-1), ..., (k,0)$ map to consecutive integers
- Inverse:
  $$w = \lfloor \frac{\sqrt{8z + 1} - 1}{2} \rfloor$$
  $$t = \frac{w^2 + w}{2}$$
  $$y = z - t, \quad x = w - y$$

**Attention Bias:**

For positions $p_i = (x_i, y_i)$ and $p_j = (x_j, y_j)$:

$$\text{bias}_{ij} = 1 - \frac{|\pi(p_i) - \pi(p_j)|}{\max_{k,l} |\pi(p_k) - \pi(p_l)|}$$

### 6.2 Multi-Head Attention with Cantor Prior

**Standard Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Cantor-Augmented Attention:**
$$\text{CantorAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \alpha \cdot B_{\text{cantor}}\right)V$$

Where:
- $B_{\text{cantor}} \in \mathbb{R}^{S \times S}$ is the Cantor bias matrix
- $\alpha \in \mathbb{R}^H$ is a learnable per-head scale

### 6.3 Fingerprint-Modulated Routing

**Standard Top-K Routing:**
$$\text{scores}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}$$
$$\text{routes}_i = \text{argtopk}(\text{scores}_i, K)$$
$$\text{output}_i = \sum_{j \in \text{routes}_i} \text{softmax}(\text{scores}_{i,\text{routes}_i})_j \cdot v_j$$

**Fingerprint-Modulated:**
$$\text{scores}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}} + \beta \cdot (q_i \cdot W_{\text{bias}} \cdot f)$$

Where:
- $f \in \mathbb{R}^{F}$ is the fingerprint
- $W_{\text{bias}} \in \mathbb{R}^{D \times F}$ is a learned projection
- $\beta = 0.1$ (small to prevent domination)

### 6.4 Anchor Bank Contribution

**Affinity Computation:**
$$a = \sigma(W_2 \cdot \text{GELU}(W_1 \cdot f))$$

Where $a \in \mathbb{R}^A$ are anchor affinities.

**Weighted Anchor:**
$$\bar{A} = \sum_{i=1}^{A} a_i \cdot A_i$$

**Output Projection:**
$$\text{anchor\_out} = W_{\text{out}} \cdot \bar{A}$$

This is added CONSTITUTIVELY to the combination:
$$\text{combined} = w_0 \cdot \text{attn} + w_1 \cdot \text{routed} + w_2 \cdot \text{anchor\_out}$$

### 6.5 Collective Emergence

**Individual Stream Accuracy:**
$$\text{Acc}_i = P(\hat{y}_i = y | x)$$

**Collective Accuracy:**
$$\text{Acc}_{\text{collective}} = P(\hat{y}_{\text{fusion}} = y | x_1, ..., x_N)$$

**Emergence Condition:**
$$\text{Acc}_{\text{collective}} \gg \max_i \text{Acc}_i$$

**Observed Ratios:**
- ImageNet: $84.68\% \gg 0.1\%$ (847× improvement)
- FashionMNIST: $93.4\% \gg 10\%$ (9.34× improvement)

---

## 7. Implementation Requirements

### 7.1 Minimum Viable Implementation

A conformant GFR implementation MUST include:

1. **Fingerprint:** Learnable parameter [fingerprint_dim] in ComposedHead
2. **Cantor Attention:** Multi-head attention with Cantor bias
3. **Anchor Bank:** Constitutive contribution from weighted anchors
4. **TopK Router:** Sparse routing with fingerprint modulation
5. **Mailbox:** Message passing infrastructure (even if unused)
6. **Registry:** Tracking of router relationships

### 7.2 Numerical Stability

- Use LayerNorm before attention and FFN
- Initialize fingerprints with small scale (0.02)
- Clip fingerprint bias contribution (0.1 scale)
- Use gradient clipping during training (1.0 norm)

### 7.3 Memory Efficiency

For large collectives:
- Mailbox content MUST be detached (no gradient storage)
- Cantor bias can be computed once and cached
- Anchor bank is shared (not duplicated per stream)

### 7.4 Computational Complexity

Per head, per forward pass:
- Attention: $O(S^2 \cdot D)$
- TopK Routing: $O(S^2 \cdot D + S \cdot K \cdot \log S)$
- Anchor Bank: $O(A \cdot D + F \cdot A)$
- Total: $O(S^2 \cdot D)$ (attention-dominated)

---

## 8. Conformance Criteria

### 8.1 Required Behaviors

A GFR implementation is CONFORMANT if:

1. ✓ Each head has a unique, learnable fingerprint
2. ✓ Attention includes Cantor pairing bias (can be disabled)
3. ✓ Anchor bank contributes constitutively to output
4. ✓ TopK routing uses fingerprint modulation
5. ✓ Mailbox is present and cleared each forward pass
6. ✓ Registry tracks stream relationships

### 8.2 Required Emergence

A GFR collective DEMONSTRATES EMERGENCE if:

$$\text{Acc}_{\text{collective}} > 2 \times \max_i \text{Acc}_i$$

That is, collective accuracy exceeds twice the best individual.

### 8.3 Recommended Configurations

| Dataset | feature_dim | fingerprint_dim | num_anchors | num_routes |
|---------|-------------|-----------------|-------------|------------|
| MNIST/Fashion | 128 | 64 | 8 | 4 |
| CIFAR-10/100 | 256 | 64 | 12 | 6 |
| ImageNet | 512 | 64 | 16 | 8 |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Prototype** | Assembled multi-stream model via factory |
| **Stream** | Input processor (VectorStream or SequenceStream) |
| **ComposedHead** | Assembled routing head with fingerprint and all components |
| **Fingerprint** | Unique learnable vector that creates divergent behavior |
| **Cantor Pairing** | Bijection ℕ² → ℕ with diagonal structure |
| **Anchor** | Shared behavioral prototype in ConstitutiveAnchorBank |
| **Mailbox** | Message-passing infrastructure for coordination |
| **Registry** | Singleton tracking all routers and relationships |
| **Adjacent Gating** | Parent-child fingerprint-based modulation |
| **Emergence** | Collective accuracy >> individual accuracy |
| **Constitutive** | Contributing directly to output (not just biasing) |

---

## Appendix B: Reference Implementation

See `geofractal.router` package:
```python
from geofractal.router import (
    # Config
    GlobalFractalRouterConfig,
    
    # Streams
    FeatureVectorStream,
    TrainableVectorStream,
    SequenceStream,
    TransformerSequenceStream,
    StreamBuilder,
    
    # Head
    HeadBuilder,
    HeadConfig,
    ComposedHead,
    CantorAttention,
    TopKRouter,
    FingerprintGate,
    ConstitutiveAnchorBank,
    
    # Fusion
    FusionBuilder,
    FusionStrategy,
    
    # Factory
    PrototypeBuilder,
    StreamSpec,
    HeadSpec,
    FusionSpec,
    AssembledPrototype,
)

# Build a prototype
prototype = (PrototypeBuilder("dual_clip")
    .add_stream(StreamSpec.feature_vector("clip_b32", input_dim=512))
    .add_stream(StreamSpec.feature_vector("clip_l14", input_dim=768))
    .with_head(HeadSpec.standard(feature_dim=512))
    .with_fusion(FusionSpec.gated(output_dim=512))
    .with_classifier(num_classes=1000)
    .build())
```

---

## Appendix C: Citation
```bibtex
@software{globalfractalrouter2025,
  author       = {AbstractPhil},
  title        = {GlobalFractalRouter: Collective Intelligence through 
                  Geometric Routing},
  year         = {2025},
  url          = {https://github.com/AbstractPhil/geofractal}
}
```

---

*End of Protocol Specification*