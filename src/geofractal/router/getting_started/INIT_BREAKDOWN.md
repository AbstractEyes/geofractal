# Init Breakdown

**Decomposition of GlobalFractalRouter Initialization**

---

## Overview

The `GlobalFractalRouter.__init__` creates a complex system of interacting components. This document breaks down each initialization step, explains its purpose, and shows how the decomposed `head` package enables customization.

---

## 1. Original Init Structure

```python
def __init__(
    self,
    config: GlobalFractalRouterConfig,
    parent_id: Optional[str] = None,
    cooperation_group: str = "default",
    name: str = "router",
):
    super().__init__()
    
    # 1. IDENTITY & REGISTRATION
    self.config = config
    self.parent_id = parent_id
    self.cooperation_group = cooperation_group
    self.name = name
    self.module_id = get_registry().register(...)
    
    # 2. FINGERPRINT
    self.fingerprint = nn.Parameter(torch.randn(F) * 0.02)
    
    # 3. ATTENTION
    self.attention = CantorMultiHeadAttention(config)
    
    # 4. ROUTING
    self.router = TopKRouter(config)
    
    # 5. ANCHORS
    self.anchor_bank = AnchorBank(config)
    
    # 6. GATING
    self.fp_gate = FingerprintGate(config)
    
    # 7. NORMALIZATION
    self.input_norm = nn.LayerNorm(D)
    self.output_norm = nn.LayerNorm(D)
    
    # 8. REFINEMENT
    self.ffn = nn.Sequential(...)
    
    # 9. COMBINATION
    self.combine_weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.1]))
```

---

## 2. Component-by-Component Breakdown

### 2.1 Identity & Registration

```python
self.config = config
self.parent_id = parent_id
self.cooperation_group = cooperation_group
self.name = name
self.module_id = get_registry().register(
    name=name,
    parent_id=parent_id,
    cooperation_group=cooperation_group,
    fingerprint_dim=config.fingerprint_dim,
    feature_dim=config.feature_dim,
)
```

**Purpose:** Establish router identity in the collective.

**What it provides:**
- Unique `module_id` for mailbox addressing
- Parent-child relationships for adjacent gating
- Group membership for cooperation
- Metadata for introspection

**Registry singleton:**
```python
# Global registry tracks all routers
registry = get_registry()  # Returns singleton

# Registration creates entry
info = RouterInfo(
    module_id=uuid4(),
    name="expert_1",
    parent_id="expert_0",
    cooperation_group="default",
    fingerprint_dim=64,
    feature_dim=512,
    children=set(),
)
```

**Adjunctive capacity:** The registry enables:
- Hierarchy queries (get_children, get_siblings)
- Group coordination
- Dynamic topology changes

---

### 2.2 Fingerprint

```python
self.fingerprint = nn.Parameter(torch.randn(F) * 0.02)
```

**Purpose:** Unique learnable identity that creates divergent behavior.

**Initialization analysis:**

```python
# Why randn?
torch.randn(F)  # Gaussian distribution → fingerprints start different

# Why * 0.02?
* 0.02  # Small scale → gradients can push fingerprints apart
        # Large scale would dominate before learning occurs
```

**Dimensional choice:**
```
fingerprint_dim = 64 (default)

Why 64?
- Large enough to encode unique identity
- Small enough to not dominate parameter count
- Powers of 2 for GPU efficiency
```

**Adjunctive capacity:** Fingerprint influences:
1. Value gating → different views of data
2. Anchor affinities → different behavioral modes
3. Score bias → different routing decisions
4. Adjacent gating → different inter-stream relationships

---

### 2.3 Attention (CantorMultiHeadAttention)

```python
self.attention = CantorMultiHeadAttention(config)
```

**Internal structure:**

```python
class CantorMultiHeadAttention:
    def __init__(self, config):
        D, H = config.feature_dim, config.num_heads
        head_dim = D // H
        
        # Standard MHA projections
        self.q_proj = nn.Linear(D, H * head_dim)  # 512 × 512 = 262K
        self.k_proj = nn.Linear(D, H * head_dim)  # 262K
        self.v_proj = nn.Linear(D, H * head_dim)  # 262K
        self.out_proj = nn.Linear(H * head_dim, D)  # 262K
        
        # Cantor-specific
        self.cantor_scale = nn.Parameter(torch.ones(H) * 0.1)  # 8 params
        self.register_buffer('cantor_bias', None)  # Built lazily
```

**Parameter breakdown:**
```
q_proj:       262,144
k_proj:       262,144
v_proj:       262,144
out_proj:     262,144
cantor_scale:       8
───────────────────────
Total:      1,048,584 (~1M)
```

**Cantor bias construction (lazy):**
```python
def _ensure_cantor_bias(self, S, device):
    if self.cantor_bias is None:
        # Build once, reuse forever
        self.cantor_bias = build_cantor_bias(H, W, device)
```

**Adjunctive capacity:** Attention provides:
- Global context aggregation
- Geometric structure via Cantor
- Multi-head diversity via per-head cantor_scale

---

### 2.4 Routing (TopKRouter)

```python
self.router = TopKRouter(config)
```

**Internal structure:**

```python
class TopKRouter:
    def __init__(self, config):
        D, F, K = config.feature_dim, config.fingerprint_dim, config.num_routes
        
        self.K = K  # Sparsity level
        
        # Score computation
        self.score_proj = nn.Linear(D, D)  # 512 × 512 = 262K
        
        # Fingerprint bias
        self.fp_to_bias = nn.Linear(F, D)  # 64 × 512 = 32K
```

**Parameter breakdown:**
```
score_proj:   262,144
fp_to_bias:    32,768
───────────────────────
Total:        294,912 (~295K)
```

**K selection logic:**
```python
# K = num_routes = 4 (default)
# For sequence length S:
K = min(self.K, S)  # Can't route to more positions than exist

# Top-K is applied per query position
topk_scores, routes = torch.topk(scores, K, dim=-1)  # [B, S, K]
```

**Adjunctive capacity:** Router provides:
- Sparse selection pressure
- Fingerprint-modulated routing
- Position-specific attention patterns

---

### 2.5 Anchors (AnchorBank)

```python
self.anchor_bank = AnchorBank(config)
```

**Internal structure:**

```python
class AnchorBank:
    def __init__(self, config):
        D, A, F = config.feature_dim, config.num_anchors, config.fingerprint_dim
        
        # Anchor embeddings (shared prototypes)
        self.anchors = nn.Parameter(torch.randn(A, D) * 0.02)  # 16 × 512 = 8K
        
        # Fingerprint → anchor affinities
        self.fp_to_anchor = nn.Sequential(
            nn.Linear(F, A * 2),     # 64 → 32 = 2K
            nn.GELU(),
            nn.Linear(A * 2, A),     # 32 → 16 = 0.5K
        )
        
        # CONSTITUTIVE output projection
        self.anchor_out = nn.Linear(D, D)  # 512 × 512 = 262K
        self.norm = nn.LayerNorm(D)         # 1K
```

**Parameter breakdown:**
```
anchors:        8,192
fp_to_anchor:   2,608
anchor_out:   262,144
norm:           1,024
───────────────────────
Total:        273,968 (~274K)
```

**The constitutive requirement:**
```python
# WRONG: Additive (gradients die)
scores = attention_scores + anchor_bias

# RIGHT: Constitutive (gradients flow)
anchor_out = self.anchor_out(weighted_anchors)  # Direct projection
combined = w₀*attn + w₁*routed + w₂*anchor_out  # Added to output
```

**Adjunctive capacity:** Anchors provide:
- Shared behavioral vocabulary
- Fingerprint-conditioned activation
- Collective coordination reference points

---

### 2.6 Gating (FingerprintGate)

```python
self.fp_gate = FingerprintGate(config)
```

**Internal structure:**

```python
class FingerprintGate:
    def __init__(self, config):
        F, D = config.fingerprint_dim, config.feature_dim
        
        # Fingerprint comparison (for adjacent gating)
        self.fp_compare = nn.Sequential(
            nn.Linear(F * 2, F),  # 128 → 64 = 8K
            nn.GELU(),
            nn.Linear(F, 1),     # 64 → 1 = 65
            nn.Sigmoid(),
        )
        
        # Value gating from fingerprint
        self.fp_to_gate = nn.Sequential(
            nn.Linear(F, D),     # 64 → 512 = 32K
            nn.Sigmoid(),
        )
```

**Parameter breakdown:**
```
fp_compare:     8,321
fp_to_gate:    32,768
───────────────────────
Total:         41,089 (~41K)
```

**Two gating modes:**
```python
# 1. Value gating (within router)
gate = sigmoid(linear(fingerprint))  # [D]
v_gated = v * gate  # Different dimensions emphasized

# 2. Adjacent gating (between routers)
gate = sigmoid(mlp(cat([fp_self, fp_target])))  # Scalar
output = output * gate  # Modulate based on compatibility
```

**Adjunctive capacity:** Gates provide:
- Fingerprint-based divergence
- Cross-router coordination
- Learnable compatibility

---

### 2.7 Normalization

```python
self.input_norm = nn.LayerNorm(D)
self.output_norm = nn.LayerNorm(D)
```

**Parameter breakdown:**
```
input_norm:   1,024 (512 weight + 512 bias)
output_norm:  1,024
───────────────────────
Total:        2,048
```

**Placement:**
```python
# Pre-norm architecture
x_norm = self.input_norm(x)      # Normalize before attention/routing
# ... attention, routing, anchors ...
output = self.output_norm(combined)  # Normalize before FFN
```

---

### 2.8 Refinement (FFN)

```python
self.ffn = nn.Sequential(
    nn.Linear(D, D * 4),    # 512 → 2048 = 1M
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(D * 4, D),    # 2048 → 512 = 1M
    nn.Dropout(0.1),
)
```

**Parameter breakdown:**
```
expand:    1,048,576 (512 × 2048)
contract:  1,048,576 (2048 × 512)
───────────────────────
Total:     2,097,152 (~2.1M)
```

**Design choices:**
```
Expansion factor: 4× (standard transformer)
Activation: GELU (smooth, gradient-friendly)
Dropout: 0.1 (regularization)
```

---

### 2.9 Combination Weights

```python
self.combine_weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.1]))
```

**Purpose:** Learnable balance between attention, routing, and anchors.

**Usage:**
```python
weights = F.softmax(self.combine_weights, dim=0)
combined = weights[0]*attn + weights[1]*routed + weights[2]*anchor
```

**Initialization rationale:**
```
[1.0, 1.0, 0.1]
  ↑    ↑    ↑
  |    |    └── Anchors start smaller (refinement role)
  |    └─────── Routing equal to attention
  └──────────── Attention (primary signal)
```

**Typical convergence:**
```
After training: [0.45, 0.45, 0.10] (roughly)
```

---

## 3. Total Parameter Count

```
Component           Parameters
─────────────────────────────────
Fingerprint              64
Attention         1,048,584
Router              294,912
Anchors             273,968
Gate                 41,089
Input Norm            1,024
Output Norm           1,024
FFN               2,097,152
Combine Weights           3
─────────────────────────────────
TOTAL             3,757,820 (~3.76M)
```

---

## 4. Decomposed Head Mapping

The `head` package maps to original components:

| Original | Decomposed | Protocol |
|----------|------------|----------|
| `attention` | `CantorAttention` | `AttentionHead` |
| `router` | `TopKRouter` | `Router` |
| `anchor_bank` | `ConstitutiveAnchorBank` | `AnchorProvider` |
| `fp_gate` | `FingerprintGate` | `GatingMechanism` |
| (implicit) | `LearnableWeightCombiner` | `Combiner` |
| `ffn` | `FFNRefinement` | `Refinement` |

---

## 5. Using the Decomposed Head

### Standard Build

```python
from geofractal.router.head import HeadConfig, build_standard_head

config = HeadConfig(feature_dim=512, num_anchors=16)
head = build_standard_head(config)

# Use like original
output = head(x, target_fingerprint=next_fp)
```

### Custom Composition

```python
from geofractal.router.head import (
    HeadBuilder,
    HeadConfig,
    CantorAttention,
    SoftRouter,  # Different router!
    AttentiveAnchorBank,  # Different anchors!
    FingerprintGate,
    GatedCombiner,  # Different combiner!
    FFNRefinement,
)

config = HeadConfig(feature_dim=512)

head = (HeadBuilder(config)
    .with_attention(CantorAttention)
    .with_router(SoftRouter)  # Swap to soft routing
    .with_anchors(AttentiveAnchorBank)  # Swap to attentive anchors
    .with_gate(FingerprintGate)
    .with_combiner(GatedCombiner, feature_dim=512)
    .with_refinement(FFNRefinement)
    .build())
```

### Static Injection

```python
# Create custom component externally
class MyCustomGate(BaseGate):
    def __init__(self):
        super().__init__()
        # Custom implementation
    
    def gate_values(self, v, fingerprint):
        # Custom gating
        pass
    
    def compute_similarity(self, fp_self, fp_target):
        # Custom similarity
        pass

my_gate = MyCustomGate()

head = (HeadBuilder(config)
    .from_preset(STANDARD_HEAD)
    .inject_gate(my_gate)  # Inject custom instance
    .build())
```

### Component Access

```python
head = build_standard_head(config)

# Access components
attention = head.attention
router = head.router
anchors = head.anchors

# Replace at runtime (for experiments)
head.replace_component('router', MySoftRouter(config))

# Inspect
print(f"Router type: {type(head.router).__name__}")
print(f"Total params: {head.num_parameters:,}")
```

---

## 6. Adjunctive Capacities Summary

The decomposed head enables:

| Capacity | Mechanism |
|----------|-----------|
| **Swap attention** | Replace CantorAttention with StandardAttention or custom |
| **Change routing** | TopK → Soft → Custom sparse pattern |
| **Modify anchors** | Constitutive → Attentive → Custom learned modes |
| **Custom gating** | Per-channel, per-position, learned compatibility |
| **Fusion strategy** | Learned weights → Gated → Custom MoE |
| **Refinement** | FFN → MoE → Custom transformation |

Each swap is protocol-compatible, ensuring the head still functions correctly while allowing experimentation.

---

*End of Init Breakdown*