# geofractal.router Package Analysis

**Author:** Claude Analysis
**Date:** December 2025
**Package Version:** 0.2.0

---

## Executive Summary

The `geofractal.router` package implements a **collective intelligence framework** that coordinates multiple independent processing streams through geometric routing and fingerprint-based divergence. The core philosophy is that accuracy emerges not from individually perfect models, but from the triangulation of divergent perspectives.

**Proven Results:**
- ImageNet: 5 streams at 0.1% → **84.68%** collective (847× emergence)
- FashionMNIST: 10% + 10% + 10% = **93.4%** (ρ = 9.34)
- Dual CLIP: 7-18% frozen → **92.6%** (5-13× emergence)

---

## Table of Contents

1. [Package Architecture](#1-package-architecture)
2. [Core Components](#2-core-components)
   - [Registry & Mailbox System](#21-registry--mailbox-system)
   - [Configuration System](#22-configuration-system)
   - [RouterCollective](#23-routercollective)
3. [Subpackages](#3-subpackages)
   - [Streams](#31-streams-subpackage)
   - [Head](#32-head-subpackage)
   - [Fusion](#33-fusion-subpackage)
   - [Factory](#34-factory-subpackage)
4. [Key Algorithms](#4-key-algorithms)
   - [Cantor Pairing](#41-cantor-pairing)
   - [Fingerprint-Based Routing](#42-fingerprint-based-routing)
   - [Detached Mailbox Coordination](#43-detached-mailbox-coordination)
5. [Design Implications](#5-design-implications)
6. [Usage Patterns](#6-usage-patterns)
7. [Performance Considerations](#7-performance-considerations)

---

## 1. Package Architecture

```
geofractal/router/
├── __init__.py           # Public API exports (207 lines)
├── config.py             # Configuration dataclasses (206 lines)
├── registry.py           # Global registry + mailbox (242 lines)
├── collective.py         # RouterCollective (705 lines)
├── streams/              # Input processing units
│   ├── stream_base.py    # Abstract base class
│   ├── stream_vector.py  # Vector streams (CLIP, features)
│   ├── stream_sequence.py # Sequence streams (tokens)
│   └── stream_builder.py # Builder pattern
├── head/                 # Routing decision engine
│   ├── head_components.py # Attention, Router, Anchors, Gates
│   ├── head_builder.py   # Builder + ComposedHead
│   └── head_protocols.py # Abstract interfaces
├── fusion/               # Multi-stream combination
│   ├── fusion_methods.py # 8 fusion strategies
│   └── fusion_builder.py # Builder pattern
└── factory/              # Prototype assembly
    ├── factory_prototype.py # AssembledPrototype
    ├── factory_builder.py   # PrototypeBuilder
    └── factory_registry.py  # Experiment tracking
```

**Total: ~8,400 lines of code across 32+ Python files**

---

## 2. Core Components

### 2.1 Registry & Mailbox System

**File:** `registry.py` (lines 1-243)

The registry and mailbox form the coordination infrastructure enabling emergent collective behavior.

#### RouterRegistry (Singleton)

```python
class RouterRegistry:
    """Global registry for router coordination."""
    routers: Dict[str, RouterInfo]      # module_id → info
    groups: Dict[str, Set[str]]         # group_name → module_ids
    name_to_id: Dict[str, str]          # name → module_id
```

**Purpose:** Track all routers in the system, their relationships, and cooperation groups.

**Key Methods:**
| Method | Returns | Purpose |
|--------|---------|---------|
| `register()` | `str` (module_id) | Register new router, return UUID |
| `get_children()` | `List[str]` | Get child router IDs |
| `get_siblings()` | `List[str]` | Get routers in same cooperation group |
| `get_hierarchy()` | `Dict` | Full tree from a router |

**Implications:**
- **Thread-safe singleton** ensures consistent global state
- **UUID-based identification** prevents naming collisions
- **Hierarchy tracking** enables parent→child fingerprint gating
- **Cooperation groups** allow flexible router clustering

#### RouterMailbox

```python
class RouterMailbox:
    """Shared mailbox for inter-router communication."""
    messages: Dict[str, RouterMessage]  # sender_id → message
    step_counter: int                   # Ordering
```

**Critical Design Decision:** `content.detach()` at line 207

```python
def post(self, sender_id, sender_name, content):
    self.messages[sender_id] = RouterMessage(
        ...
        content=content.detach(),  # Don't backprop through mailbox
        ...
    )
```

**Implications:**
1. **No gradient flow between routers** - Routers cannot directly optimize each other
2. **Prevents collapse** - Without detachment, routers would converge to identical behavior
3. **Emergent coordination** - Routers learn to coordinate through observation, not gradient descent
4. **Computational efficiency** - Breaks autograd graph, reducing memory

**Warning:** If you modify this to allow gradient flow, expect:
- Rapid convergence to degenerate solutions
- Loss of divergence between streams
- Collapse of emergence ratio ρ toward 1.0

---

### 2.2 Configuration System

**File:** `config.py` (lines 1-206)

Three primary configuration classes form a hierarchy:

#### GlobalFractalRouterConfig
```python
@dataclass
class GlobalFractalRouterConfig:
    # Core dimensions
    feature_dim: int = 512           # Internal routing dimension
    fingerprint_dim: int = 64        # Identity/divergence dimension

    # Routing
    num_anchors: int = 16            # Shared behavioral modes
    num_routes: int = 4              # Top-K routes per position
    num_heads: int = 8               # Attention heads
    temperature: float = 1.0         # Softmax temperature

    # Coordination
    use_adjacent_gating: bool = True # Parent→child fingerprint gating
    use_cantor_prior: bool = True    # Cantor diagonal structure
    use_mailbox: bool = True         # Inter-router communication
```

**Implications of Key Parameters:**

| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| `num_anchors` | Simpler behavior modes, faster | More expressive, risk of unused anchors |
| `num_routes` | More focused routing | More diversity, higher compute |
| `fingerprint_dim` | Less identity capacity | More divergence potential |
| `temperature` | Sharper distributions | Softer, more exploration |

#### CollectiveConfig

Extends router config with training hyperparameters:

```python
@dataclass
class CollectiveConfig:
    # Training
    batch_size: int = 256            # A100-optimized
    epochs: int = 20
    lr: float = 3e-4
    warmup_epochs: int = 2

    # DataLoader (A100 optimized)
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
```

**Proven Presets:**

| Preset | feature_dim | num_anchors | num_routes | Use Case |
|--------|-------------|-------------|------------|----------|
| ImageNet | 512 | 16 | 8 | Large-scale classification |
| FashionMNIST | 128 | 8 | 4 | Small-scale, rapid iteration |
| CIFAR | 256 | 12 | 6 | Medium complexity |

---

### 2.3 RouterCollective

**File:** `collective.py` (lines 1-705)

The orchestrator that coordinates streams, heads, and fusion.

#### Architecture Flow

```
Input(s)
    │
    ├─→ Stream₁ ─→ Head₁ ─→ Pool ─┐
    ├─→ Stream₂ ─→ Head₂ ─→ Pool ─┼─→ Fusion ─→ Classifier ─→ Logits
    └─→ Stream₃ ─→ Head₃ ─→ Pool ─┘
```

#### StreamWrapper

Internal class handling input normalization:

```python
class StreamWrapper(nn.Module):
    """Wraps streams to ensure consistent [B, S, D] output."""

    def __init__(self, name, input_dim, feature_dim, num_slots, input_type):
        if input_type == "vector":
            # Expansion: [B, D_in] → [B, num_slots, D]
            self.expansion = nn.Sequential(
                nn.Linear(input_dim, feature_dim * 2),
                nn.LayerNorm(feature_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 2, feature_dim * num_slots),
            )
            self.slot_embed = nn.Parameter(torch.randn(1, num_slots, feature_dim) * 0.02)
```

**Implications:**
1. **Slot expansion** converts 1D features into pseudo-sequences for routing
2. **Learnable slot embeddings** add position-like identity to each slot
3. **Dimension-agnostic** - streams with different input dims normalized to `feature_dim`

#### Adjacent Fingerprint Coordination

```python
def forward(self, x, return_individual=False, return_info=False):
    for i, name in enumerate(self.stream_names):
        # Circular adjacent fingerprint
        next_idx = (i + 1) % len(self.stream_names)
        next_name = self.stream_names[next_idx]
        target_fp = self.heads[next_name].fingerprint

        # Head routes with adjacent fingerprint awareness
        routed = head(encoded, target_fingerprint=target_fp)
```

**Implications:**
- Creates a **circular dependency** between streams
- Each stream's routing is influenced by its neighbor's identity
- Prevents streams from becoming completely independent
- Encourages complementary behavior

#### Emergence Computation

```python
def compute_emergence(self, collective_acc, individual_accs):
    max_ind = max(individual_accs.values())
    rho = collective_acc / max_ind if max_ind > 0 else float('inf')

    return {
        'rho': rho,                    # Emergence ratio
        'max_individual': max_ind,
        'emergence': collective_acc - max_ind,  # Absolute gain
    }
```

**The emergence ratio ρ is the key metric:**
- ρ = 1.0: No emergence (collective = best individual)
- ρ > 1.0: Positive emergence (collective > best individual)
- ρ = 847: Extreme emergence (as seen in ImageNet results)

---

## 3. Subpackages

### 3.1 Streams Subpackage

**Location:** `geofractal/router/streams/`

Streams are the primary input processing units. They:
1. Encode input to features
2. Expand to slots (vectors) or pass through (sequences)
3. Route through internal head with mailbox coordination
4. Pool output for fusion

#### Stream Types

| Type | Input Shape | Output Shape | Use Case |
|------|-------------|--------------|----------|
| `FeatureVectorStream` | [B, D] | [B, S, D] | Pre-extracted features (CLIP, DINO) |
| `TrainableVectorStream` | [B, D] | [B, S, D] | End-to-end learned features |
| `SequenceStream` | [B, S, D] | [B, S, D] | Token sequences (NLP) |
| `TransformerSequenceStream` | [B, S, D] | [B, S, D] | With transformer encoder |
| `ConvSequenceStream` | [B, S, D] | [B, S, D] | Multi-scale 1D convolutions |

#### BaseStream Abstract Interface

```python
class BaseStream(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: Any) -> torch.Tensor:
        """Encode input to features."""
        pass

    @abstractmethod
    def prepare_for_head(self, features: torch.Tensor) -> torch.Tensor:
        """Prepare for head routing (slot expansion or passthrough)."""
        pass

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool [B, S, D] → [B, D]."""
        return x.mean(dim=1)
```

**Implications:**
- **Streams own their heads** - not separate components
- **Consistent interface** regardless of input type
- **Registry integration** - each stream registers globally

---

### 3.2 Head Subpackage

**Location:** `geofractal/router/head/`

The routing decision engine with modular, swappable components.

#### Component Architecture

```
           ┌───────────────────────────────────────────┐
           │              ComposedHead                 │
           │  ┌─────────────────────────────────────┐  │
Input ────→│  │  Attention → Router → Anchors      │  │────→ Output
           │  │      ↓         ↓         ↓          │  │
           │  │   Gate   →  Combiner  → Refinement │  │
           │  └─────────────────────────────────────┘  │
           └───────────────────────────────────────────┘
```

#### Component Deep Dive

**1. CantorAttention** (lines 92-153)

```python
class CantorAttention(BaseAttention):
    """Multi-head attention with Cantor pairing positional bias."""

    def _cantor_pair(self, i, j):
        """π(i,j) = ((i+j)(i+j+1))/2 + j"""
        return ((i + j) * (i + j + 1)) // 2 + j

    def _build_cantor_bias(self, S, device):
        idx = torch.arange(S, device=device)
        i, j = torch.meshgrid(idx, idx, indexing='ij')
        paired = self._cantor_pair(i, j).float()
        return torch.log1p(paired) / math.log(S + 1)
```

**Implications:**
- Creates **self-similar diagonal structure** in attention
- Positions along diagonals have related attention patterns
- **No learned parameters** for the bias structure
- Per-head learnable scale allows adaptation

**2. TopKRouter** (lines 160-208)

```python
class TopKRouter(BaseRouter):
    """Top-K sparse router with fingerprint-guided key biasing."""

    def forward(self, q, k, v, fingerprint):
        # Base scores
        scores = torch.bmm(q_proj, k.transpose(-2, -1)) / (D ** 0.5)

        # CRITICAL: Fingerprint bias per-KEY (not per-query)
        fp_bias = self.fp_to_bias(fingerprint)  # [D]
        key_bias = torch.einsum('bsd,d->bs', k, fp_bias)  # [B, S]
        scores = scores + key_bias.unsqueeze(1) * 0.1

        # Top-K selection
        topk_scores, routes = torch.topk(scores / temp, K, dim=-1)
```

**Critical Design Decision:** Fingerprint bias is applied per-KEY, not per-query.

**Why this matters:**
- Per-key bias affects **relative ranking within softmax**
- Enables gradients to flow through `fp_to_bias`
- If applied per-query, all keys shifted equally → no relative change → zero gradient

**3. ConstitutiveAnchorBank** (lines 252-295)

```python
class ConstitutiveAnchorBank(BaseAnchorBank):
    """Anchor bank that constitutively contributes to output."""

    def __init__(self, config):
        self.anchors = nn.Parameter(
            torch.randn(config.num_anchors, config.feature_dim) * 0.02
        )
        self.fp_to_anchor = nn.Sequential(
            nn.Linear(fingerprint_dim, num_anchors * 2),
            nn.GELU(),
            nn.Linear(num_anchors * 2, num_anchors),
        )
```

**Implications:**
- **Anchors are learned, not additive** - avoids gradient zeroing
- Fingerprint selects anchor combination via sigmoid affinities
- Shared across all positions within a stream
- Provides "behavioral modes" the router can activate

**4. FingerprintGate** (lines 343-398)

```python
class FingerprintGate(BaseGate):
    def forward(self, x, fingerprint, target_fingerprint=None):
        gated = self.gate_values(x, fingerprint)

        if target_fingerprint is not None:
            similarity = self.compute_similarity(fingerprint, target_fingerprint)
            gated = gated * similarity  # Adjacent gating

        return gated
```

**Implications:**
- **Self-gating:** Stream's fingerprint controls its own output
- **Adjacent gating:** Similarity to neighbor's fingerprint modulates output
- Creates coupling between streams without gradient flow

#### Head Presets

```python
STANDARD_HEAD = HeadPreset(
    attention_cls=CantorAttention,
    router_cls=TopKRouter,
    anchor_cls=ConstitutiveAnchorBank,
    gate_cls=FingerprintGate,
    combiner_cls=LearnableWeightCombiner,
    refinement_cls=FFNRefinement,
)

LIGHTWEIGHT_HEAD = HeadPreset(
    attention_cls=StandardAttention,  # No Cantor
    router_cls=SoftRouter,            # No hard selection
    ...
)

HEAVY_HEAD = HeadPreset(
    attention_cls=CantorAttention,
    router_cls=TopKRouter,
    anchor_cls=AttentiveAnchorBank,   # Input-dependent anchors
    gate_cls=ChannelGate,             # SE-style gating
    combiner_cls=GatedCombiner,       # Input-adaptive combination
    refinement_cls=MixtureOfExpertsRefinement,  # MoE FFN
)
```

---

### 3.3 Fusion Subpackage

**Location:** `geofractal/router/fusion/`

Eight strategies for combining divergent stream outputs.

#### Strategy Comparison

| Strategy | When to Use | Parameters | Adaptivity |
|----------|------------|------------|------------|
| **CONCAT** | Baseline, maximum info | `O(n × D²)` | None |
| **WEIGHTED** | Known quality differences | `O(n)` weights | Static |
| **GATED** | Input-adaptive combination | `O(n × D)` | Per-sample |
| **ATTENTION** | Cross-stream relationships | `O(n² × D)` | Per-sample |
| **FINGERPRINT** | Identity-guided fusion | `O(n × F)` | Fingerprint-based |
| **RESIDUAL** | Conservative changes | `O(D²)` | Residual only |
| **MOE** | Sparse expert selection | `O(E × D²)` | Top-K experts |
| **HIERARCHICAL** | Tree-structured | `O(log n × D²)` | Hierarchical |

#### GatedFusion Deep Dive

```python
class GatedFusion(BaseAdaptiveFusion):
    def compute_weights(self, stream_outputs, context=None):
        if context is None:
            context = self._concat_outputs(stream_outputs)

        logits = self.gate_net(context)  # [B, N]
        weights = F.softmax(logits / self.temperature, dim=-1)
        return weights

    def forward(self, stream_outputs, ...):
        weights = self.compute_weights(stream_outputs)  # [B, N]

        fused = torch.zeros(B, self.output_dim, device=weights.device)
        for i, name in enumerate(self.stream_names):
            projected = self.projections[name](stream_outputs[name])
            fused = fused + weights[:, i:i + 1] * projected

        return self.norm(fused), info
```

**Implications:**
- **Per-sample weights:** Different inputs get different stream importance
- **Temperature control:** Lower temperature → sharper selection
- **Projection per stream:** Handles different input dimensions

#### Choosing a Fusion Strategy

```
Do streams have different qualities?
├─ Yes, and you know them → WEIGHTED (static)
├─ Yes, but input-dependent → GATED (adaptive)
└─ No/Unknown → Continue

Do streams need to interact during fusion?
├─ Yes → ATTENTION
└─ No → Continue

Do you have fingerprint information?
├─ Yes → FINGERPRINT_GUIDED
└─ No → Continue

Is sparsity important?
├─ Yes → MOE
└─ No → CONCAT (default)
```

---

### 3.4 Factory Subpackage

**Location:** `geofractal/router/factory/`

Fluent builder API for assembling complete router systems.

#### PrototypeBuilder Usage

```python
prototype = (PrototypeBuilder()
    .with_name("my_model")
    .with_num_classes(1000)
    .add_stream(StreamSpec.frozen_clip("clip_b32", "openai/clip-vit-base-patch32"))
    .add_stream(StreamSpec.frozen_clip("clip_l14", "openai/clip-vit-large-patch14"))
    .with_head(HeadSpec.standard(feature_dim=512))
    .with_fusion(FusionSpec.attention(output_dim=512, num_heads=8))
    .with_classifier(hidden_dim=512, dropout=0.1)
    .freeze_streams(True)
    .build())
```

#### Available Presets

| Preset | Description | Streams | Fusion |
|--------|-------------|---------|--------|
| `imagenet` | ImageNet-proven (84.68%) | 2 frozen CLIP | Concat |
| `cifar` | CIFAR-10/100 | 1 frozen CLIP | Concat |
| `fashion` | FashionMNIST (ρ=9.34) | 3 feature streams | Concat |
| `multimodal` | Multi-modal | 2 frozen CLIP | Attention |
| `research` | Full features | 2 frozen CLIP | Attention |

#### AssembledPrototype Architecture

```python
class AssembledPrototype(BasePrototype):
    """
    Complete router prototype assembled from components.

    Architecture:
        Input
          │
          ├─→ Stream₁ ─→ Head₁ ─→ Pool ─┐
          ├─→ Stream₂ ─→ Head₂ ─→ Pool ─┼─→ Fusion ─→ Classifier
          └─→ Stream₃ ─→ Head₃ ─→ Pool ─┘
    """

    def __init__(self, config: PrototypeConfig):
        self._build_streams()     # From StreamSpecs
        self._build_heads()       # Per-stream heads from HeadSpec
        self._build_fusion()      # From FusionSpec
        self._build_classifier()  # Simple linear
```

---

## 4. Key Algorithms

### 4.1 Cantor Pairing

The Cantor pairing function bijectively maps 2D coordinates to 1D indices:

```
π(x, y) = ((x + y)(x + y + 1)) / 2 + y
```

**Visual representation:**

```
     y=0  y=1  y=2  y=3
x=0   0    2    5    9
x=1   1    4    8   13
x=2   3    7   12   18
x=3   6   11   17   24
```

**Key Properties:**
1. **Self-similar diagonal structure:** Values increase along diagonals
2. **Unique mapping:** Every (x,y) pair gets unique index
3. **Reversible:** Can recover (x,y) from index

**Use in Attention:**
```python
def _build_cantor_bias(self, S, device):
    idx = torch.arange(S, device=device)
    i, j = torch.meshgrid(idx, idx, indexing='ij')
    paired = self._cantor_pair(i, j).float()
    return torch.log1p(paired) / math.log(S + 1)  # Normalized
```

The result is an [S, S] bias matrix where:
- Positions close on the diagonal attend more to each other
- Creates implicit spatial/structural relationships
- No learned parameters (purely geometric)

**Implications:**
- Introduces inductive bias for diagonal relationships
- Useful when positions have inherent 2D structure (images → patches)
- Alternative to learned positional embeddings

---

### 4.2 Fingerprint-Based Routing

Each stream has a learnable fingerprint vector:

```python
self.fingerprint = nn.Parameter(
    torch.randn(config.fingerprint_dim) * 0.02
)
```

**How fingerprints affect routing:**

1. **Router key biasing:**
```python
fp_bias = self.fp_to_bias(fingerprint)  # [D]
key_bias = torch.einsum('bsd,d->bs', k, fp_bias)  # Per-key bias
scores = scores + key_bias.unsqueeze(1) * 0.1
```

2. **Anchor selection:**
```python
affinities = torch.sigmoid(self.fp_to_anchor(fingerprint))  # [num_anchors]
weighted = (self.anchors * affinities.unsqueeze(-1)).sum(dim=0)
```

3. **Gating:**
```python
gate = self.fp_to_gate(fingerprint)  # [D]
return x * gate.unsqueeze(0).unsqueeze(0)  # Channel-wise gating
```

**Implications:**
- **Same architecture, different behavior:** Identical head architectures diverge via fingerprints
- **Learnable identity:** Fingerprint learns what makes this stream unique
- **Coordination without coupling:** Adjacent fingerprint gating creates dependencies

---

### 4.3 Detached Mailbox Coordination

The mailbox enables inter-router communication **without gradient flow**:

```python
def post(self, sender_id, sender_name, content):
    self.messages[sender_id] = RouterMessage(
        ...
        content=content.detach(),  # CRITICAL: No backprop
        ...
    )
```

**Why detachment is essential:**

Without detachment:
```
Stream A loss ─→ Stream A params ─→ Mailbox ─→ Stream B params
                         └───────────────────────────────────┘
                                    Gradient flow
```

This creates **optimization coupling** where:
- Stream A optimizes Stream B through mailbox
- Leads to degenerate solutions (all streams converge)
- Destroys divergence

With detachment:
```
Stream A loss ─→ Stream A params ─╳─→ Mailbox ─→ Stream B
                         └───────────────────────────┘
                                    Information flow only
```

**Result:**
- Streams observe each other but cannot directly optimize each other
- Coordination must emerge through indirect learning
- Maintains divergence while enabling cooperation

---

## 5. Design Implications

### 5.1 Divergence Over Accuracy

**Principle:** Streams don't need to be individually accurate; they must see differently.

**Implementation Evidence:**
- ImageNet streams at 0.1% individual accuracy achieve 84.68% collectively
- Fingerprint-based divergence ensures different perspectives
- No shared parameters between streams (except mailbox observation)

**Practical Implication:**
- Don't waste effort making individual streams perfect
- Focus on ensuring streams capture orthogonal information
- Measure divergence via fingerprint distance/correlation

### 5.2 Constitutive Contribution

**Principle:** Every component must directly contribute to output.

**Implementation Evidence:**
- Anchors are learned through weighted combination, not added as bias
- Gates multiply (not add) to ensure gradient flow
- Combiner weights are normalized via softmax

**What to avoid:**
```python
# BAD: Additive-only anchor (gradient ≈ 0)
output = x + anchor_bias

# GOOD: Constitutive anchor (gradient flows)
weighted = (anchors * affinities).sum()
output = transform(weighted)
```

### 5.3 Gradient Health

**Problem areas identified in code comments:**

1. **TopKRouter fingerprint bias:**
   - Must be per-KEY, not per-query
   - Per-query shifts all keys equally → no relative change → zero gradient

2. **ConstitutiveAnchorBank LayerNorm:**
   - Requires proper shape handling via unsqueeze/squeeze
   - LayerNorm needs at least 2D input

3. **Mailbox detachment:**
   - Without detachment, optimization collapse occurs
   - `content.detach()` is intentional and critical

### 5.4 Scalability Considerations

**Stream count scaling:**
- Registry O(n) for n streams
- Mailbox O(n) messages per forward pass
- Fusion complexity depends on strategy (O(n) to O(n²))

**Sequence length scaling:**
- CantorAttention: O(S²) bias computation (cached per S)
- TopKRouter: O(S × K) per position
- Pooling: O(S) reduction

---

## 6. Usage Patterns

### 6.1 Quick Start with Presets

```python
from geofractal.router import PrototypeBuilder

# ImageNet-ready prototype
prototype = PrototypeBuilder.imagenet().build()

# CIFAR with custom classes
prototype = PrototypeBuilder.cifar(num_classes=10).build()

# FashionMNIST (proven ρ=9.34)
prototype = PrototypeBuilder.fashion().build()
```

### 6.2 Custom Configuration

```python
from geofractal.router import (
    PrototypeBuilder, StreamSpec, HeadSpec, FusionSpec
)

prototype = (PrototypeBuilder()
    .with_name("custom_model")
    .with_num_classes(100)

    # Add diverse streams
    .add_stream(StreamSpec.feature_stream("cnn", input_dim=2048, feature_dim=512))
    .add_stream(StreamSpec.feature_stream("vit", input_dim=768, feature_dim=512))
    .add_stream(StreamSpec.feature_stream("clip", input_dim=512, feature_dim=512))

    # Configure routing
    .with_custom_head(
        feature_dim=512,
        fingerprint_dim=64,
        num_anchors=16,
        num_routes=4,
        use_cantor=True,
        attention_type='cantor',
        router_type='topk',
        anchor_type='constitutive',
    )

    # Configure fusion
    .with_attention_fusion(output_dim=512, num_heads=8)

    # Training settings
    .freeze_streams(False)
    .build())
```

### 6.3 Using RouterCollective Directly

```python
from geofractal.router import RouterCollective, CollectiveConfig

config = CollectiveConfig(
    feature_dim=512,
    num_classes=1000,
    num_slots=16,
    epochs=20,
    lr=3e-4,
)

collective = RouterCollective.from_feature_dims({
    'clip_b32': 512,
    'clip_l14': 768,
    'dino_b16': 768,
}, config, fusion_strategy='gated')

# Train
history = collective.fit(train_loader, val_loader)
print(f"Final ρ: {history['rho'][-1]:.3f}")
```

### 6.4 Monitoring Emergence

```python
# During training
val_acc, stream_accs, val_loss = collective.evaluate(val_loader, return_loss=True)
emergence = collective.compute_emergence(val_acc, stream_accs)

print(f"Collective: {val_acc*100:.2f}%")
print(f"Best individual: {emergence['max_individual']*100:.2f}%")
print(f"ρ = {emergence['rho']:.3f}")
print(f"Emergence: {emergence['emergence']*100:.2f}%")

# Interpretation
if emergence['rho'] > 1.5:
    print("Strong emergence detected")
elif emergence['rho'] > 1.0:
    print("Positive emergence")
else:
    print("No emergence - check stream divergence")
```

---

## 7. Performance Considerations

### 7.1 Memory Usage

| Component | Memory | Scaling |
|-----------|--------|---------|
| Fingerprints | O(n × F) | Linear in streams |
| Cantor bias | O(S²) | Quadratic in sequence length |
| Mailbox | O(n × D) | Linear in streams |
| Anchors | O(A × D) | Fixed per head |

### 7.2 Compute Bottlenecks

1. **CantorAttention:** O(S²) attention computation
   - Cached bias helps but attention itself is quadratic

2. **TopKRouter:** O(S² × K) scoring and selection
   - K typically small (4-8), S dominates

3. **Fusion:** Depends on strategy
   - Concat: O(n × D)
   - Attention: O(n² × D)
   - MoE: O(E × D²)

### 7.3 Optimization Tips

```python
# 1. Use AMP for memory efficiency
config = CollectiveConfig(use_amp=True)

# 2. Freeze streams when using pretrained encoders
prototype = builder.freeze_streams(True).build()

# 3. Use lightweight head for iteration
builder.with_lightweight_head()

# 4. Reduce num_slots for faster training
config = CollectiveConfig(num_slots=8)  # vs default 16

# 5. Use concat fusion for speed (vs attention)
builder.with_concat_fusion()
```

---

## Appendix: File Reference

| File | Lines | Key Contents |
|------|-------|--------------|
| `__init__.py` | 207 | Public API exports |
| `config.py` | 206 | `GlobalFractalRouterConfig`, `CollectiveConfig` |
| `registry.py` | 243 | `RouterRegistry`, `RouterMailbox` |
| `collective.py` | 705 | `RouterCollective`, `StreamWrapper` |
| `head/head_components.py` | 651 | `CantorAttention`, `TopKRouter`, `ConstitutiveAnchorBank` |
| `head/head_builder.py` | 570 | `HeadBuilder`, `ComposedHead` |
| `fusion/fusion_methods.py` | 729 | 8 fusion strategies |
| `factory/factory_builder.py` | 545 | `PrototypeBuilder` |
| `factory/factory_prototype.py` | 513 | `AssembledPrototype` |

---

*This analysis is based on geofractal.router v0.2.0*
