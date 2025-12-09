# Metrics and Evaluation

**Measuring Success in GlobalFractalRouter**

---

## 1. Primary Metrics

### 1.1 Emergence Ratio (ρ)

The fundamental metric for collective intelligence.

**Definition:**
$$\rho = \frac{\text{Acc}_{\text{collective}}}{\max_n \text{Acc}_n}$$

**Interpretation:**
| ρ Value | Interpretation |
|---------|----------------|
| ρ < 1 | Collective worse than best individual (failure) |
| ρ = 1-2 | Marginal improvement (ensemble-like) |
| ρ = 2-10 | Strong emergence (10%→90%) |
| ρ > 10 | Extraordinary emergence (0.1%→85%) |

**Benchmark Results:**
| Dataset | ρ (Observed) |
|---------|--------------|
| ImageNet | 847 (0.1% → 84.68%) |
| FashionMNIST | 9.34 (10% → 93.4%) |
| CIFAR-100 | TBD |

**How to Measure:**
```python
def compute_emergence_ratio(collective, val_loader, device):
    collective.eval()
    
    collective_correct = 0
    individual_correct = defaultdict(int)
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            logits, info = collective(x, return_individual=True)
            
            # Collective accuracy
            collective_correct += (logits.argmax(1) == y).sum().item()
            
            # Individual accuracies
            for name, ind_logits in info['individual_logits'].items():
                individual_correct[name] += (ind_logits.argmax(1) == y).sum().item()
            
            total += y.size(0)
    
    collective_acc = collective_correct / total
    individual_accs = {k: v/total for k, v in individual_correct.items()}
    best_individual = max(individual_accs.values())
    
    emergence_ratio = collective_acc / (best_individual + 1e-8)
    
    return {
        'emergence_ratio': emergence_ratio,
        'collective_acc': collective_acc,
        'individual_accs': individual_accs,
        'best_individual': best_individual,
    }
```

### 1.2 Collective Accuracy

Raw classification accuracy of the fused collective.

**Target Benchmarks:**
| Dataset | Baseline | Target | State-of-Art |
|---------|----------|--------|--------------|
| MNIST | 99% | 99.5% | 99.8% |
| FashionMNIST | 90% | 94% | 96% |
| CIFAR-10 | 90% | 96% | 99% |
| CIFAR-100 | 70% | 82% | 91% |
| ImageNet | 75% | 87% | 91% |

### 1.3 Parameter Efficiency

Accuracy per million parameters.

**Definition:**
$$\text{Efficiency} = \frac{\text{Acc}_{\text{collective}}}{\text{Params (M)}}$$

**Example:**
```
ImageNet: 84.68% / 60M params = 1.41% per M params
vs CLIP ViT-L: 76% / 428M params = 0.18% per M params
```

**7.8× more efficient** (with caveat that CLIP features are used)

---

## 2. Diagnostic Metrics

### 2.1 Routing Entropy

Measures how spread out the routing distribution is.

**Definition:**
$$H_{\text{route}} = -\sum_{k=1}^{K} w_k \log(w_k + \epsilon)$$

**Interpretation:**
| Entropy | Meaning |
|---------|---------|
| Low (< 0.5) | Sharp routing, strong specialization |
| Medium (0.5-1.5) | Balanced routing |
| High (> 1.5) | Diffuse routing, weak selection |

**Desired Behavior:**
- Training start: High entropy (exploration)
- Training end: Medium-low entropy (specialization)

```python
def compute_routing_entropy(weights):
    """
    Args:
        weights: [B, S, K] routing weights
    Returns:
        entropy: Scalar average entropy
    """
    eps = 1e-8
    entropy = -(weights * (weights + eps).log()).sum(dim=-1)
    return entropy.mean().item()
```

### 2.2 Anchor Utilization

Measures how anchors are being used across streams.

**Per-Anchor Activation:**
$$u_a = \frac{1}{N} \sum_{n=1}^{N} \mathbb{E}_{x}[\text{affinity}_{n,a}(x)]$$

**Anchor Entropy:**
$$H_{\text{anchor}} = -\sum_{a=1}^{A} u_a \log(u_a + \epsilon)$$

**Interpretation:**
| Pattern | Meaning |
|---------|---------|
| Uniform utilization | All anchors contribute equally |
| Sparse utilization | Few dominant anchors |
| Dead anchors ($u_a ≈ 0$) | Wasted capacity |

```python
def compute_anchor_utilization(collective, val_loader, device):
    """Measure anchor activation patterns."""
    anchor_activations = defaultdict(list)
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            _, info = collective(x, return_individual=True)
            
            for stream in collective.streams:
                affinities = torch.sigmoid(
                    stream.router.anchor_bank.fp_to_anchor(stream.fingerprint)
                )
                anchor_activations[stream.name].append(affinities.cpu())
    
    # Average per stream
    utilization = {}
    for name, acts in anchor_activations.items():
        utilization[name] = torch.stack(acts).mean(dim=0).numpy()
    
    return utilization
```

### 2.3 Fingerprint Diversity

Measures how different fingerprints are across streams.

**Pairwise Cosine Similarity:**
$$\text{sim}_{ij} = \frac{f_i \cdot f_j}{\|f_i\| \|f_j\|}$$

**Diversity Score:**
$$D = 1 - \frac{1}{N(N-1)} \sum_{i \neq j} |\text{sim}_{ij}|$$

**Target:** $D > 0.8$ (high diversity)

```python
def compute_fingerprint_diversity(collective):
    """Measure fingerprint diversity."""
    fingerprints = torch.stack([
        s.fingerprint for s in collective.streams
    ])  # [N, F]
    
    # Normalize
    fp_norm = F.normalize(fingerprints, dim=-1)
    
    # Pairwise similarity
    sim_matrix = fp_norm @ fp_norm.T  # [N, N]
    
    # Exclude diagonal
    N = len(collective.streams)
    mask = ~torch.eye(N, dtype=bool)
    avg_sim = sim_matrix[mask].abs().mean().item()
    
    diversity = 1 - avg_sim
    
    return {
        'diversity': diversity,
        'similarity_matrix': sim_matrix.cpu().numpy(),
        'avg_pairwise_similarity': avg_sim,
    }
```

### 2.4 Stream Contribution

Measures how much each stream contributes to the final decision.

**Gradient-based Contribution:**
$$C_n = \mathbb{E}_{x,y}\left[\left\|\frac{\partial \mathcal{L}}{\partial \bar{y}_n}\right\|\right]$$

**Ablation-based Contribution:**
$$C_n^{\text{abl}} = \text{Acc}_{\text{collective}} - \text{Acc}_{\text{collective} \setminus n}$$

```python
def compute_stream_contributions(collective, val_loader, device):
    """Measure each stream's contribution via ablation."""
    collective.eval()
    
    # Full collective accuracy
    full_acc = evaluate_accuracy(collective, val_loader, device)
    
    contributions = {}
    for i, stream in enumerate(collective.streams):
        # Create ablated collective (remove stream i)
        ablated = create_ablated_collective(collective, exclude_idx=i)
        ablated_acc = evaluate_accuracy(ablated, val_loader, device)
        
        contributions[stream.name] = full_acc - ablated_acc
    
    return contributions
```

---

## 3. Training Dynamics

### 3.1 Loss Curves

**Expected Behavior:**

```
Loss
│
│   ╲
│    ╲╲
│      ╲╲___
│          ╲__
│             ╲___________
└─────────────────────────► Epoch
    Fast early   Gradual refinement
    convergence
```

**Warning Signs:**
- Loss increasing: Learning rate too high
- Loss flat: Dead fingerprints or anchors
- Loss oscillating: Batch size too small

### 3.2 Emergence Curve

Plot emergence ratio over training:

```python
def track_emergence(collective, val_loader, device, epochs, train_fn):
    history = {'emergence': [], 'collective_acc': [], 'individual_accs': []}
    
    for epoch in range(epochs):
        train_fn(collective, epoch)
        
        metrics = compute_emergence_ratio(collective, val_loader, device)
        history['emergence'].append(metrics['emergence_ratio'])
        history['collective_acc'].append(metrics['collective_acc'])
        history['individual_accs'].append(metrics['individual_accs'])
    
    return history
```

**Expected Pattern:**
- Early: ρ ≈ 1 (collective similar to average)
- Middle: ρ increasing rapidly (emergence begins)
- Late: ρ plateaus at high value (stable emergence)

### 3.3 Specialization Dynamics

Track how routing patterns evolve:

```python
def track_specialization(collective, val_loader, device, sample_epochs):
    """Track routing patterns at specific epochs."""
    patterns = {}
    
    for epoch in sample_epochs:
        # Collect routing decisions
        routing_decisions = collect_routing_decisions(collective, val_loader, device)
        patterns[epoch] = routing_decisions
    
    # Compute overlap between streams
    for epoch, decisions in patterns.items():
        overlap = compute_routing_overlap(decisions)
        print(f"Epoch {epoch}: Routing overlap = {overlap:.3f}")
    
    return patterns
```

---

## 4. Robustness Metrics

### 4.1 Distribution Shift

Test on shifted distributions:

```python
SHIFT_TRANSFORMS = {
    'gaussian_noise': transforms.GaussianNoise(std=0.1),
    'contrast': transforms.ColorJitter(contrast=0.5),
    'blur': transforms.GaussianBlur(kernel_size=5),
    'rotation': transforms.RandomRotation(30),
}

def evaluate_robustness(collective, val_dataset, device):
    results = {}
    
    # Clean accuracy
    results['clean'] = evaluate_accuracy(collective, val_dataset, device)
    
    # Shifted accuracies
    for name, transform in SHIFT_TRANSFORMS.items():
        shifted_dataset = apply_transform(val_dataset, transform)
        results[name] = evaluate_accuracy(collective, shifted_dataset, device)
    
    # Robustness score
    results['avg_shifted'] = np.mean([v for k, v in results.items() if k != 'clean'])
    results['robustness'] = results['avg_shifted'] / results['clean']
    
    return results
```

**Hypothesis:** Collectives are more robust due to diverse perspectives.

### 4.2 Adversarial Robustness

Test against adversarial attacks:

```python
def evaluate_adversarial(collective, val_loader, device, epsilon=0.03):
    """Evaluate against FGSM attack."""
    collective.eval()
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        x.requires_grad = True
        
        # Clean prediction
        logits, _ = collective(x)
        clean_correct += (logits.argmax(1) == y).sum().item()
        
        # FGSM attack
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        x_adv = x + epsilon * x.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # Adversarial prediction
        logits_adv, _ = collective(x_adv)
        adv_correct += (logits_adv.argmax(1) == y).sum().item()
        
        total += y.size(0)
    
    return {
        'clean_acc': clean_correct / total,
        'adv_acc': adv_correct / total,
        'robustness': (adv_correct / total) / (clean_correct / total + 1e-8),
    }
```

### 4.3 Stream Failure Recovery

Test collective when streams fail:

```python
def evaluate_stream_failure(collective, val_loader, device):
    """Test collective with streams disabled."""
    results = {}
    
    # All streams
    results['all'] = evaluate_accuracy(collective, val_loader, device)
    
    # Remove each stream
    for i, stream in enumerate(collective.streams):
        # Disable stream
        original_forward = stream.forward
        stream.forward = lambda *args, **kwargs: zero_output(original_forward, *args, **kwargs)
        
        results[f'without_{stream.name}'] = evaluate_accuracy(collective, val_loader, device)
        
        # Restore
        stream.forward = original_forward
    
    # Remove random 50%
    results['half_streams'] = evaluate_with_random_subset(collective, val_loader, device, 0.5)
    
    return results
```

---

## 5. Visualization

### 5.1 Emergence Plot

```python
def plot_emergence(history, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Collective vs individuals
    ax = axes[0]
    ax.plot(history['collective_acc'], label='Collective', linewidth=2)
    for name in history['individual_accs'][0].keys():
        ax.plot([h[name] for h in history['individual_accs']], '--', label=name, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Collective vs Individual Accuracy')
    ax.legend()
    
    # Emergence ratio
    ax = axes[1]
    ax.plot(history['emergence'], linewidth=2, color='red')
    ax.axhline(y=2, color='gray', linestyle='--', label='Emergence threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Emergence Ratio (ρ)')
    ax.set_title('Emergence Ratio Over Training')
    ax.legend()
    
    # Final comparison
    ax = axes[2]
    final_accs = history['individual_accs'][-1]
    final_accs['Collective'] = history['collective_acc'][-1]
    ax.bar(final_accs.keys(), final_accs.values())
    ax.set_ylabel('Accuracy')
    ax.set_title('Final Accuracy Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
```

### 5.2 Routing Visualization

```python
def visualize_routing(collective, x, save_path=None):
    """Visualize routing patterns for a single input."""
    fig, axes = plt.subplots(1, len(collective.streams), figsize=(4*len(collective.streams), 4))
    
    with torch.no_grad():
        collective.mailbox.clear()
        
        for i, stream in enumerate(collective.streams):
            features = stream.encode(x)
            translated = stream.translation(features)
            slots = translated.view(1, collective.config.num_slots, -1)
            
            routes, weights, _ = stream.router(slots, collective.mailbox, None)
            
            ax = axes[i]
            im = ax.imshow(weights[0].cpu().numpy(), cmap='viridis')
            ax.set_xlabel('Route Index')
            ax.set_ylabel('Slot Index')
            ax.set_title(f'{stream.name}\nRouting Weights')
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
```

### 5.3 Fingerprint Space

```python
def visualize_fingerprints(collective, save_path=None):
    """Visualize fingerprints in 2D via PCA."""
    fingerprints = torch.stack([s.fingerprint for s in collective.streams]).detach().cpu().numpy()
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    fp_2d = pca.fit_transform(fingerprints)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i, (x, y) in enumerate(fp_2d):
        ax.scatter(x, y, s=200)
        ax.annotate(collective.streams[i].name, (x, y), fontsize=12)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Fingerprint Space (PCA)')
    
    if save_path:
        plt.savefig(save_path)
    return fig
```

---

## 6. Benchmarking Protocol

### 6.1 Standard Evaluation

```python
def full_evaluation(collective, train_loader, val_loader, test_loader, device):
    """Complete evaluation suite."""
    results = {}
    
    # Core metrics
    results['emergence'] = compute_emergence_ratio(collective, val_loader, device)
    results['test_acc'] = evaluate_accuracy(collective, test_loader, device)
    
    # Diagnostics
    results['fingerprint_diversity'] = compute_fingerprint_diversity(collective)
    results['anchor_utilization'] = compute_anchor_utilization(collective, val_loader, device)
    results['stream_contributions'] = compute_stream_contributions(collective, val_loader, device)
    
    # Robustness
    results['robustness'] = evaluate_robustness(collective, test_loader.dataset, device)
    
    # Efficiency
    results['params'] = sum(p.numel() for p in collective.parameters())
    results['efficiency'] = results['test_acc'] / (results['params'] / 1e6)
    
    return results
```

### 6.2 Reporting Template

```markdown
## Results for [Dataset]

### Core Metrics
| Metric | Value |
|--------|-------|
| Collective Accuracy | X.XX% |
| Best Individual Accuracy | X.XX% |
| Emergence Ratio (ρ) | X.XX |
| Parameters | X.XM |
| Efficiency | X.XX% per M params |

### Per-Stream Breakdown
| Stream | Individual Acc | Contribution |
|--------|---------------|--------------|
| stream_1 | X.XX% | +X.XX% |
| stream_2 | X.XX% | +X.XX% |
| ... | ... | ... |

### Robustness
| Condition | Accuracy | Δ from Clean |
|-----------|----------|--------------|
| Clean | X.XX% | - |
| Gaussian Noise | X.XX% | -X.XX% |
| Blur | X.XX% | -X.XX% |
| Adversarial (ε=0.03) | X.XX% | -X.XX% |

### Diagnostics
- Fingerprint Diversity: X.XX
- Anchor Utilization Entropy: X.XX
- Average Routing Entropy: X.XX
```

---

## 7. Continuous Monitoring

### 7.1 Training Dashboard Metrics

Real-time tracking during training:

```python
DASHBOARD_METRICS = [
    'train_loss',
    'val_loss',
    'collective_acc',
    'emergence_ratio',
    'routing_entropy',
    'fingerprint_diversity',
    'learning_rate',
    'gradient_norm',
]

def log_to_wandb(metrics, step):
    import wandb
    wandb.log(metrics, step=step)
```

### 7.2 Alert Conditions

```python
ALERTS = {
    'emergence_drop': lambda m: m['emergence_ratio'] < m['prev_emergence'] * 0.9,
    'dead_stream': lambda m: any(v < 0.01 for v in m['individual_accs'].values()),
    'collapsed_fingerprints': lambda m: m['fingerprint_diversity'] < 0.5,
    'high_routing_entropy': lambda m: m['routing_entropy'] > 2.0,
}

def check_alerts(metrics):
    triggered = []
    for name, condition in ALERTS.items():
        if condition(metrics):
            triggered.append(name)
    return triggered
```

---

*End of Metrics and Evaluation*