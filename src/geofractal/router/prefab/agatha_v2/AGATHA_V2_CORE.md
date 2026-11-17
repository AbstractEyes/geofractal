# AGATHA V2 CORE

A flow-matching velocity model over VAE latent tokens, with a collective of
per-token force towers integrated by a damped, banded second-order step.
This document states the math exactly as `model.py`, `bands.py`,
`oscillator.py`, `towers.py` and `blocks.py` compute it.

License: Apache-2.0

---

## 1. Inputs and objective

```
z_1 : clean latent [B, C, H, W]          z_0 ~ N(0, I)
t   ~ U(0, 1)  (or logit-normal)          t = 1 is data
z_t = (1 - t) z_0 + t z_1
v*  = z_1 - z_0
L   = mean (v_hat(z_t, t, E) - v*)^2      squared error only
```

The model's inputs are exactly `(z_t, t, E, mask)`, where `E` [B, L, d_txt] is
a frozen text encoder's last hidden states (all tokens, padding masked) and is
computed offline. `z_1` reaches the model only through `z_t`. With probability
`cond_dropout` per training sample, `E` is replaced by the stored states of the
empty caption.

## 2. Tokens, time, text

```
X  = patchify(z_t)               [B, N, P]   N = (H/p)(W/p), P = C p^2, raster order
h0 = W_in X + b_in               [B, N, d]
C  = W_c E + b_c                 [B, L, d]
e  = MLP([sin(f t'), cos(f t')]) [B, d]      t' = 1000 t, f_j = 10000^(-j / (F/2))
```

## 3. Blocks

Transformer block (trunk and geometry towers), three residual branches, no gates:

```
mod_i(y) = y (1 + ds_i(e)) + dh_i(e)        (ds_i, dh_i) from one linear e -> 6d, zero at initialization
x = x + SelfAttn(mod_1(LN(x)))              rotary position code on q, k
x = x + CrossAttn(mod_2(LN(x)), C, mask)
x = x + MLP(mod_3(LN(x)))
```

The trunk uses a plain 2-D rotary code (first half of each head rotated by the
row index, second half by the column index). Conv towers use residual blocks
on the (H/p) x (W/p) grid:

```
y = conv3x3(SiLU(FiLM(GN(conv3x3(SiLU(GN(x_in)))), e)));  x = x + y
FiLM(y, e) = y (1 + ds(e)) + dh(e)          last layer zero at initialization
x_in = x                       ('wide_resnet')
x_in = x - avgpool3x3(x)       ('frequency': a fixed high-pass of the branch input)
```

## 4. Anchor and tower forces

```
x_ref = X + D_ref(h)                        D_ref = Linear(LN(h)), zero at initialization
Y_k   = Head_k(Tower_k(h; C, e))            k = 1..K, [B, N, P], heads zero at initialization
```

Geometry towers each own a position code at the true head width:
cantor (CantorRoPE, hybrid), beatrix (BeatrixRoPE), simplex (QuadRoPE),
helix (TriRoPE). Each code is built for the exact token count N.

## 5. Noise bands

```
sigma = 1 - t
s     = shift sigma / (1 + (shift - 1) sigma)
ramp(x) = 0.5 - 0.5 cos(pi (clamp(x / 0.06, -1, 1) + 1) / 2)
b_0 = 1 - ramp(s - 0.35)
b_1 = ramp(s - 0.35) (1 - ramp(s - 0.75))
b_2 = ramp(s - 0.35) ramp(s - 0.75)         b_j >= 0, sum_j b_j = 1
```

Per band j (raw parameters rho_j, phi_j initialized to 0, psi_j initialized to 0):

```
beta_j  = 0.05 N_osc + (0.5 N_osc - 0.05 N_osc) sigmoid(rho_j)
omega_j = 1.0 N_osc sigmoid(phi_j)
kappa_j = softplus(psi_j)                   initialized to log 2 ~ 0.693
w_{j,k} real, signed, initialized to 1/K
q(t)    = sum_j b_j(s) q_j                  for q in {beta, omega, kappa, w_k}
u       = sum_k w_k(t) Y_k                  [B, N, P]
```

Nothing is compared against anything: the windows are fixed functions of `s`
and the weights are plain numbers.

Optional (off by default, experimental): signed per-token dispatch replaces
`w_k(t)` by `s_{n,k} = sinh(u_{n,k}) / sum_j cosh(u_{n,j})`,
`u_{n,k} = cos(W_q h_n, a_k) / tau`.

## 6. The oscillator

Per token, gains constant over the steps, `h = 1 / N_osc`, `x_0 = X`, `v_0 = 0`:

```
a  = -2 beta v + omega^2 (x_ref - x) + kappa u
v' = v + h a
x' = x + h v'
```

Exact form (used in the forward pass): with `Omega = omega^2`, scalars
`(A_n, G_n)` from the recurrence

```
av' = (1 - 2 beta h) av - h Omega ax ;  ax' = ax + h av'      (ax, av) = (1, 0) at n = 0
gv' = (1 - 2 beta h) gv - h Omega gx + h ;  gx' = gx + h gv'  (gx, gv) = (0, 0) at n = 0
```

and one combination

```
x_N = x_ref + A_N (X - x_ref) + G_N kappa u
```

which equals `A_N X + G_N (Omega x_ref + kappa u)` because `A_N + G_N Omega = 1`
(x_ref is the fixed point when u = 0). The stepped loop is kept as the
reference; the two agree to about 1e-16 relative in float64.

Stability: with `p = (h omega)^2` and `q = 2 h beta`, the step matrix of
`(x, h v)` is `[[1 - p, 1 - q], [-p, 1 - q]]` with characteristic polynomial
`lambda^2 - (2 - p - q) lambda + (1 - q)`. Both roots lie inside the unit circle
iff `0 < q < 2`, `p > 0`, `p + 2q < 4`. The bounds above give `q` in `(0.1, 1.0)`
and `p` in `(0, 1.0)`, inside that region for every raw parameter value.

## 7. Velocity head and the closed form

```
v_hat = unpatchify( W_v (x_N - X) + b_v )     W_v initialized to the identity, b_v initialized to 0
```

Because `x_ref = X + D_ref(h)` and `A_N + G_N omega^2 = 1`, the input tokens
cancel:

```
v_hat = W_v G_N(t) [ omega(t)^2 D_ref(h) + kappa(t) sum_k w_k(t) Y_k ] + b_v
```

So the oscillator's exact effect is two bounded scalar gains per noise band:
`G_N omega^2` on the anchor correction and `G_N kappa` on the summed tower
forces. It buys bounded, interpretable, banded gains, not extra expressivity;
the modelling is done by the trunk, the anchor head and the towers. At initialization
`D_ref = 0` and every `Y_k = 0`, so `v_hat = 0` exactly.

## 8. Sampling

```
z(0) ~ N(0, I);  z <- z + dt v_hat(z, t_i, E)      t_i = i / N_s   (Euler)
midpoint: v1 = v_hat(z, t_i), v2 = v_hat(z + dt/2 v1, t_i + dt/2), z <- z + dt v2
guidance (optional): v = v_null + g (v_c - v_null)
```

`loss()`, `sample()` and `sample_midpoint()` all call the same `velocity()`,
with the same `N_osc`. Evaluation is deterministic given `z(0)`.

## 9. Training

Adam over every trained parameter, weight decay 0, float32 parameters
(bfloat16 autocast optional; the banded combination of the forces, the
oscillator and the velocity head always run in float32). After backward, each tower's gradients are
scaled to the median tower gradient norm; the per-tower norms are logged.
A checkpoint holds the config, the whole model state, and optionally the
optimizer state and step. Encoders are recorded by name and revision only.

## 10. Differences from the original folder's core document (BEATRIX_CORE.md)

| topic | original | this version |
|---|---|---|
| spring | `a = ... - omega^2 Log_x(x_ref)`, i.e. `- omega^2 (x_ref - x)` in flat space: pushes away from `x_ref` | `+ omega^2 (x_ref - x)`: pulls toward `x_ref` |
| initial velocity | random, `v_0 = 0.1 randn` | `v_0 = 0`; evaluation deterministic |
| conditioning | fused head streams that include the clean latent (and, in training, the target image) | text token sequence only, by cross-attention; the clean latent only through `z_t` |
| towers | 16 outputs from one token; some never reach the force | K towers over all N tokens, every one summed into the force by index |
| routing | `softmax(cos(fingerprints)) x softmax(confidence)`, input independent | plain signed weights per noise band (or the optional signed dispatch) |
| schedules | fixed beta / omega / kappa / gamma curves over an internal clock, including a 3-6-9 schedule | learned per-band gains on the flow time, bounded for stability |
| steps | 10 oscillator steps in training, 1 per call in sampling | one `velocity()` with `N_osc` steps everywhere |
| integration | stepped Exp / Log / transport calls (flat: plain addition and subtraction) | exact affine form; stepped form kept as reference |
| velocity head | MLP on the flattened latent | one P x P matrix shared over tokens, initialized to the identity |
| guidance force | `gamma xi_guide` from image features | none |
| geometry | Riemannian vocabulary for a flat space | flat space, written as the arithmetic it is |
| optimizer | AdamW | Adam, weight decay 0 |
| checkpoint | denoiser only | whole model + config |
