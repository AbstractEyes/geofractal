# Mathematical Foundations

**Formal Mathematics of GlobalFractalRouter**

---

## 1. Notation

| Symbol | Definition |
|--------|------------|
| $B$ | Batch size |
| $S$ | Sequence length (slots) |
| $D$ | Feature dimension |
| $F$ | Fingerprint dimension |
| $H$ | Number of attention heads |
| $K$ | Number of routes (top-K) |
| $A$ | Number of anchors |
| $N$ | Number of streams |
| $C$ | Number of classes |
| $\sigma$ | Sigmoid function |
| $\odot$ | Element-wise multiplication |

---

## 2. Cantor Pairing

### 2.1 Definition

The Cantor pairing function is a bijection $\pi: \mathbb{N}^2 \rightarrow \mathbb{N}$:

$$\pi(x, y) = \frac{(x + y)(x + y + 1)}{2} + y$$

### 2.2 Properties

**Bijectivity:** For every $z \in \mathbb{N}$, there exists a unique $(x, y) \in \mathbb{N}^2$ such that $\pi(x, y) = z$.

**Diagonal Structure:** Points along anti-diagonals $\{(k, 0), (k-1, 1), ..., (0, k)\}$ map to consecutive integers:
$$\pi(k, 0) = \frac{k(k+1)}{2}, \quad \pi(k-1, 1) = \frac{k(k+1)}{2} + 1, \quad ...$$

**Self-Similarity:** The function exhibits fractal-like nesting: subregions map to similar patterns at different scales.

### 2.3 Inverse

Given $z$, recover $(x, y)$:

$$w = \left\lfloor \frac{\sqrt{8z + 1} - 1}{2} \right\rfloor$$

$$t = \frac{w^2 + w}{2}$$

$$y = z - t, \quad x = w - y$$

### 2.4 Attention Bias Matrix

For a grid of positions $\{(i, j) : 0 \leq i < H, 0 \leq j < W\}$, the Cantor bias matrix $B_C \in \mathbb{R}^{S \times S}$ where $S = H \times W$:

$$B_C[p, q] = 1 - \frac{|\pi(p) - \pi(q)|}{\max_{r,s} |\pi(r) - \pi(s)|}$$

This creates similarity between positions with nearby Cantor indices.

---

## 3. Multi-Head Attention

### 3.1 Standard Formulation

Given input $X \in \mathbb{R}^{B \times S \times D}$:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$.

For head $h$ with head dimension $d = D/H$:

$$Q_h = Q[:,:,hd:(h+1)d], \quad \text{similarly for } K_h, V_h$$

$$\text{Attention}_h(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d}}\right) V_h$$

### 3.2 Cantor-Augmented Attention

$$\text{CantorAttn}_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d}} + \alpha_h \cdot B_C\right) V_h$$

where $\alpha_h \in \mathbb{R}$ is a learnable per-head scale.

### 3.3 Output Projection

$$\text{Output} = \text{Concat}(\text{CantorAttn}_1, ..., \text{CantorAttn}_H) W_O$$

where $W_O \in \mathbb{R}^{D \times D}$.

---

## 4. Fingerprint Mechanics

### 4.1 Fingerprint Definition

Each router has a learnable fingerprint $f \in \mathbb{R}^F$:

$$f \sim \mathcal{N}(0, 0.02^2 \cdot I_F)$$

### 4.2 Value Gating

$$g = \sigma(W_g f), \quad W_g \in \mathbb{R}^{D \times F}$$

$$V_{\text{gated}} = V \odot g$$

where $g$ is broadcast across batch and sequence dimensions.

### 4.3 Anchor Affinities

$$a = \sigma(W_2 \cdot \text{GELU}(W_1 f))$$

where $W_1 \in \mathbb{R}^{2A \times F}$, $W_2 \in \mathbb{R}^{A \times 2A}$, and $a \in \mathbb{R}^A$.

### 4.4 Score Bias

$$\text{bias}_i = \frac{1}{\sqrt{D}} Q_i \cdot (W_b f)$$

where $W_b \in \mathbb{R}^{D \times F}$ and the bias is added with scale $\beta = 0.1$:

$$\text{scores} = \text{scores} + \beta \cdot \text{bias}$$

### 4.5 Adjacent Similarity

For fingerprints $f_{\text{self}}$ and $f_{\text{target}}$:

$$\text{sim}(f_{\text{self}}, f_{\text{target}}) = \sigma\left(W_c^{(2)} \cdot \text{GELU}(W_c^{(1)} [f_{\text{self}}; f_{\text{target}}])\right)$$

where $[;]$ denotes concatenation.

---

## 5. TopK Routing

### 5.1 Score Computation

$$\text{scores}_{ij} = \frac{(W_s Q_i) \cdot K_j}{\sqrt{D}} + \beta \cdot \text{bias}_i$$

### 5.2 Route Selection

$$\text{routes}_i = \text{argtop}_K\left(\frac{\text{scores}_i}{\tau}\right)$$

where $\tau$ is temperature (default 1.0).

### 5.3 Weight Computation

$$w_{ik} = \frac{\exp(\text{scores}_{i, \text{routes}_{ik}} / \tau)}{\sum_{k'} \exp(\text{scores}_{i, \text{routes}_{ik'}} / \tau)}$$

### 5.4 Output

$$\text{routed}_i = \sum_{k=1}^{K} w_{ik} \cdot V_{\text{routes}_{ik}}$$

---

## 6. Anchor Bank

### 6.1 Anchor Matrix

$$\mathcal{A} \in \mathbb{R}^{A \times D}$$

Initialized as $\mathcal{A}_{ij} \sim \mathcal{N}(0, 0.02^2)$.

### 6.2 Weighted Anchor

Given affinities $a \in \mathbb{R}^A$:

$$\bar{\mathcal{A}} = \sum_{i=1}^{A} a_i \cdot \mathcal{A}_i \in \mathbb{R}^D$$

### 6.3 Output Projection

$$\text{anchor\_out} = W_a \bar{\mathcal{A}}, \quad W_a \in \mathbb{R}^{D \times D}$$

This is broadcast to $\mathbb{R}^{B \times S \times D}$.

---

## 7. Combination and Output

### 7.1 Learnable Combination

$$\tilde{w} = \text{softmax}([w_0, w_1, w_2])$$

where $[w_0, w_1, w_2]$ are learnable scalars initialized to $[1, 1, 0.1]$.

### 7.2 Combined Output

$$\text{combined} = \tilde{w}_0 \cdot \text{attn\_out} + \tilde{w}_1 \cdot \text{routed} + \tilde{w}_2 \cdot \text{anchor\_out}$$

### 7.3 Residual and FFN

$$Y^{(1)} = X + \text{combined}$$

$$Y^{(2)} = Y^{(1)} + \text{FFN}(\text{LayerNorm}(Y^{(1)}))$$

where:

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x) + b_2$$

with $W_1 \in \mathbb{R}^{4D \times D}$, $W_2 \in \mathbb{R}^{D \times 4D}$.

---

## 8. Collective Dynamics

### 8.1 Stream Processing

For stream $n$ with encoder $E_n$ and router $R_n$:

$$h_n = E_n(x_n) \in \mathbb{R}^{B \times D_n}$$

$$\tilde{h}_n = \text{Translation}_n(h_n) \in \mathbb{R}^{B \times S \times D}$$

$$\tilde{h}_n = \tilde{h}_n + \text{SlotEmbed}_n$$

$$y_n = R_n(\tilde{h}_n, \text{mailbox}, f_{n+1}) \in \mathbb{R}^{B \times S \times D}$$

$$\bar{y}_n = \text{Pool}(y_n) = \frac{1}{S} \sum_{s=1}^{S} y_n[:, s, :] \in \mathbb{R}^{B \times D}$$

### 8.2 Fusion

$$\text{fused} = \text{Fusion}([\bar{y}_1; \bar{y}_2; ...; \bar{y}_N])$$

where $[;]$ denotes concatenation along feature dimension.

$$\text{Fusion}(z) = W_f^{(2)} \cdot \text{GELU}(W_f^{(1)} z)$$

with $W_f^{(1)} \in \mathbb{R}^{2D \times ND}$, $W_f^{(2)} \in \mathbb{R}^{D \times 2D}$.

### 8.3 Classification

$$\hat{y} = \text{softmax}(W_c \cdot \text{fused} + b_c)$$

where $W_c \in \mathbb{R}^{C \times D}$.

---

## 9. Loss Functions

### 9.1 Classification Loss

$$\mathcal{L}_{\text{CE}} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}$$

### 9.2 Routing Entropy (Optional Regularization)

$$\mathcal{H}_{\text{route}} = -\frac{1}{BS} \sum_{b,s} \sum_{k=1}^{K} w_{bsk} \log(w_{bsk} + \epsilon)$$

Minimizing encourages sharper routing (specialization).
Maximizing encourages uniform routing (exploration).

### 9.3 Fingerprint Diversity (Optional Regularization)

$$\mathcal{L}_{\text{div}} = \frac{1}{N(N-1)} \sum_{i \neq j} \text{cosine\_sim}(f_i, f_j)$$

Minimizing encourages fingerprint diversity.

---

## 10. Emergence Analysis

### 10.1 Individual Accuracy

For stream $n$ with its own classifier $W_c^{(n)}$:

$$\text{Acc}_n = \frac{1}{|D_{\text{val}}|} \sum_{(x,y) \in D_{\text{val}}} \mathbf{1}[\arg\max(W_c^{(n)} \bar{y}_n) = y]$$

### 10.2 Collective Accuracy

$$\text{Acc}_{\text{collective}} = \frac{1}{|D_{\text{val}}|} \sum_{(x,y) \in D_{\text{val}}} \mathbf{1}[\arg\max(\hat{y}) = y]$$

### 10.3 Emergence Ratio

$$\rho = \frac{\text{Acc}_{\text{collective}}}{\max_n \text{Acc}_n}$$

**Emergence is demonstrated when $\rho > 2$.**

Observed ratios:
- ImageNet: $\rho = 84.68\% / 0.1\% = 847$
- FashionMNIST: $\rho = 93.4\% / 10\% = 9.34$

### 10.4 Information Theoretic Interpretation

Let $I(Y; X | S_n)$ be mutual information between output and input given stream $n$'s representation.

**Hypothesis:** The collective achieves high mutual information through complementary partial information:

$$I(Y; X | S_1, ..., S_N) > \sum_n I(Y; X | S_n)$$

This super-additivity suggests streams capture orthogonal aspects of the input.

---

## 11. Complexity Analysis

### 11.1 Time Complexity

Per router forward pass:

| Operation | Complexity |
|-----------|------------|
| QKV Projection | $O(BSD^2)$ |
| Attention Scores | $O(BS^2D)$ |
| Cantor Bias | $O(S^2)$ (cached) |
| TopK Selection | $O(BS^2 + BSK\log S)$ |
| Value Gathering | $O(BSKD)$ |
| FFN | $O(BSD^2)$ |
| **Total** | $O(BS^2D + BSD^2)$ |

### 11.2 Space Complexity

| Component | Memory |
|-----------|--------|
| Attention weights | $O(BHS^2)$ |
| Cantor bias (cached) | $O(S^2)$ |
| Intermediate activations | $O(BSD)$ |
| Parameters per router | $O(D^2 + AD + FD)$ |

### 11.3 Scaling Properties

For $N$ streams with sequence length $S$ and dimension $D$:

- **Parameter count:** $O(N \cdot D^2)$
- **Forward time:** $O(N \cdot S^2 \cdot D)$
- **Memory:** $O(N \cdot B \cdot S \cdot D)$

The quadratic attention term $S^2$ dominates for long sequences.

---

## 12. Gradient Flow

### 12.1 Through Combination

$$\frac{\partial \mathcal{L}}{\partial \text{attn}} = \tilde{w}_0 \cdot \frac{\partial \mathcal{L}}{\partial \text{combined}}$$

$$\frac{\partial \mathcal{L}}{\partial \text{routed}} = \tilde{w}_1 \cdot \frac{\partial \mathcal{L}}{\partial \text{combined}}$$

$$\frac{\partial \mathcal{L}}{\partial \text{anchor}} = \tilde{w}_2 \cdot \frac{\partial \mathcal{L}}{\partial \text{combined}}$$

All three paths receive gradients proportional to their combination weights.

### 12.2 Through Fingerprint

$$\frac{\partial \mathcal{L}}{\partial f} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial f} + \frac{\partial \mathcal{L}}{\partial g} \cdot \frac{\partial g}{\partial f} + \frac{\partial \mathcal{L}}{\partial \text{bias}} \cdot \frac{\partial \text{bias}}{\partial f}$$

Fingerprints receive gradients from all three pathways, enabling end-to-end learning of divergent behavior.

### 12.3 Mailbox Detachment

$$\frac{\partial \mathcal{L}}{\partial \text{mailbox.content}} = 0$$

No gradients flow through mailbox → coordination emerges without explicit optimization.

---

## Appendix: Derivations

### A.1 Cantor Inverse Derivation

Given $z = \pi(x, y) = \frac{(x+y)(x+y+1)}{2} + y$:

Let $w = x + y$ (the anti-diagonal index).

Then $z = \frac{w(w+1)}{2} + y$.

The triangular number $T_w = \frac{w(w+1)}{2}$ satisfies $T_w \leq z < T_{w+1}$.

From $T_w \leq z$:
$$\frac{w(w+1)}{2} \leq z$$
$$w^2 + w - 2z \leq 0$$
$$w \leq \frac{-1 + \sqrt{1 + 8z}}{2}$$

Taking the floor: $w = \lfloor \frac{\sqrt{8z+1} - 1}{2} \rfloor$.

Then $y = z - T_w$ and $x = w - y$.

### A.2 Softmax Temperature Effect

For scores $s$ and temperature $\tau$:

$$\lim_{\tau \to 0} \text{softmax}(s/\tau) = \text{one\_hot}(\arg\max(s))$$

$$\lim_{\tau \to \infty} \text{softmax}(s/\tau) = \text{uniform}$$

Temperature $\tau = 1$ balances sharpness and smoothness.

---

*End of Mathematical Foundations*