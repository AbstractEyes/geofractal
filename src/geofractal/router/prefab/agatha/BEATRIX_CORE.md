# BEATRIX CORE
Covariant Differential Oscillation Integrator  
for Geofractal Router + Flow-Matching Diffusion


Author: AbstractPhil + Claude

Date: December 2025

License: Apache-2.0

---

## Overview

Beatrix Core is a **covariant differential dynamics engine**.

It aggregates **10 expert tower opinions** (including mirrored inverses and
theta probes) into a **controlled state update** suitable for downstream
**differential-geometric flow-matching diffusion**.

This document defines the mathematics in a format that is:
- Plain-text safe
- Markdown-renderable
- Implementation-agnostic

---

## 1. State, Space, and Objects

### Manifold

```
M : latent manifold
g : metric on M
```

### State

```
x(t) : state in M
v(t) = dx/dt : velocity in tangent space T_x M
```

### Tower outputs (10 experts)

For each expert i in {1..10}:

```
y_i(t) : tower opinion projected into M
f_i(t) : fingerprint / address vector in R^d
c_i(t) : optional confidence in [0,1]
```

State fingerprint:

```
f_x(t) : fingerprint of current state
```

### Manifold operators

```
Log_x(y) : map point y in M to tangent vector at x
Exp_x(v) : map tangent vector v at x back to M
PT(a -> b) : parallel transport between tangent spaces
∇_t v : covariant (intrinsic) acceleration
```

Metric operations:

```
||u||_g : norm under metric g
cos_g(u, v) : cosine similarity under metric g
```

---

## 2. Opinion → Tangent Force (Covariant Conversion)

Each tower opinion becomes a tangent-space force:

```
xi_i(t) = Log_{x(t)}( y_i(t) )
```

Where:
- `xi_i` lives in `T_{x(t)} M`
- It represents the intrinsic direction suggested by expert `i`

---

## 3. Geometric Routing Weights

### Similarity

```
s_i(t) = dot(f_x(t), f_i(t)) / (||f_x(t)|| * ||f_i(t)||)
```

### Soft routing (temperature tau)

```
alpha_tilde_i = exp(s_i / tau) / sum_j exp(s_j / tau)
```

### Confidence-weighted routing

```
alpha_i = alpha_tilde_i * c_i
alpha_i = alpha_i / sum_j alpha_j
```

Properties:

```
alpha_i >= 0
sum_i alpha_i = 1
alpha lies on the 10-simplex
```

---

## 4. Mirrored / Inverse Towers (Signed Differential Pairs)

### Tower roles (example indexing)

```
1  Cantor
2  Cantor-inverse

3  Simplex
4  Simplex-inverse

5  Shape
6  Shape-inverse

7–10  Theta probes (four unsupervised theta RoPE towers)
```

### Signed pair contributions

```
u_C = alpha_1 * xi_1  -  alpha_2 * xi_2
u_S = alpha_3 * xi_3  -  alpha_4 * xi_4
u_H = alpha_5 * xi_5  -  alpha_6 * xi_6
```

### Theta probe mixture

```
u_theta = alpha_7 * xi_7
        + alpha_8 * xi_8
        + alpha_9 * xi_9
        + alpha_10 * xi_10
```

### Total routed control force

```
u(t) = u_C + u_S + u_H + u_theta
```

Notes:
- Mirrored towers act as **counter-gradients**
- Theta towers provide **spectral stability probes**

---

## 5. Beatrix Core Dynamics (Covariant Oscillator)

### Continuous dynamics

```
dx/dt = v
```

```
∇_t v = -2 * beta(t) * v
        - omega(t)^2 * Log_x( x_ref(t) )
        + kappa(t) * u(t)
```

---

## 6. Discrete Geodesic Integration

Given time step Δt:

```
a_t = -2 * beta_t * v_t
      - omega_t^2 * Log_{x_t}(x_ref_t)
      + kappa_t * u_t
```

```
v_tilde = v_t + Δt * a_t
```

```
x_{t+Δt} = Exp_{x_t}( Δt * v_tilde )
```

```
v_{t+Δt} = PT(x_t -> x_{t+Δt})( v_tilde )
```

---

## 7. Interface to Flow-Matching Diffusion

### Additive control

```
dx/dt = F_theta(x, t) + lambda(t) * u(x, t)
```

### Conditioned control

```
dx/dt = F_theta(x, t ; u(x, t))
```

---

## 8. Differential Rose-Style Stability Terms (Optional)

### Dispersion

```
L_disp = sum_{i<j} alpha_i * alpha_j * (1 - cos_g(xi_i, xi_j))
```

### Jerk

```
L_jerk ≈ || v_{t+Δt} - v_t ||_g^2
```

### Mirror consistency

```
L_mirror = 1 + cos_g(xi_i, xi_j)
```

---

## 9. Compact Core Summary

```
xi_i = Log_x(y_i)

alpha_i ∝ exp(s_i / tau) * c_i
sum_i alpha_i = 1

u = sum_i alpha_i * sigma_i * xi_i

dx/dt = v
∇_t v = -2 * beta * v - omega^2 * Log_x(x_ref) + kappa * u
```

---

## Notes

- Beatrix Core is **not a classifier**
- Mirrored towers are **required counter-terms**
- Theta towers act as **frequency-domain diagnostics**
- Rose is expressed as **differential stability**, not static similarity

---

Copyright 2025 AbstractPhil  
License: Apache-2.0
