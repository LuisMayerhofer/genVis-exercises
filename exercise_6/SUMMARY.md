# Exercise 6 — Normalizing Flows (RealNVP)

> Course: Generative AI and Visual Synthesis (SoSe 2026)

## 1. What this exercise is about

The fourth generative family: **normalizing flows**. The idea: build an INVERTIBLE
neural network `f` that maps data `x` to a latent `z` with a simple known distribution
(standard Gaussian). Because `f` is invertible, you can go both ways:

- `x -> z = f(x)`     (encode; used for training via likelihood)
- `z -> x = f^{-1}(z)`  (sample: draw `z ~ N(0,I)`, invert to get a new image)

Unlike VAEs and GANs, a flow can compute the EXACT likelihood `p(x)` of any data
point. You implement RealNVP (a flow built from "affine coupling" layers) and
apply it to 2D toy data (two moons) and 8×8 digit images.

## 2. Key concepts / theory

**Change of variables (why flows work):** if `z = f(x)`, then

```
log p_X(x) = log p_Z(z) + log|det(Jacobian of f)|
```

To train, MAXIMIZE this likelihood = MINIMIZE the negative log-likelihood (NLL).
With a standard-Gaussian latent the loss simplifies (dropping constants) to:

```
NLL = mean( 0.5 * ||z||^2  -  log_det )
```

**Affine coupling layer** (the trick that makes `f` both expressive AND invertible):

- Split the input into two halves `x1, x2`.
- Keep `x1` unchanged: `y1 = x1`.
- Transform `x2` using functions of `x1` only:
  `y2 = x2 * exp(s) + t`, where `(s, t) = subnet(x1)`, `s = tanh(s_tilde)`.
- This is trivially invertible: `x2 = (y2 - t) * exp(-s)`.
- The Jacobian is triangular, so `log|det| = sum(s)` — cheap to compute!
- tanh on the log-scale keeps training numerically stable.

**Swap layers:** since each coupling leaves one half untouched, you SWAP the halves
between layers so that, after several blocks, every dimension gets transformed.

**Stacking:** more coupling blocks + bigger hidden subnets = more expressive flow.

**Conditioning (Task 3):** feed the one-hot class label into each coupling subnet so
the flow models `p(x | y)` and you can generate a chosen digit.

> **Libraries new here:** `sklearn.datasets.make_moons` and `load_digits` (toy data),
> `torch.chunk` (split halves), `F.one_hot` (labels for the conditional flow).

## 3. The core building blocks (what the code does and why)

### The subnet — a small MLP that predicts the transformation

Each coupling layer owns a tiny neural network (a multi-layer perceptron) that
reads one half of the data and outputs the scale and shift for the other half:

```python
def subnet(in_dim, out_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),     # outputs concatenated (s, t)
    )
```

Bigger `hidden` = a more expressive transformation. Note this subnet itself does
**not** need to be invertible — the coupling structure below provides invertibility.

### The affine coupling layer — expressive *and* invertible

The clever trick: split the input in two, leave one half alone, and transform the
other half using only the untouched half.

```python
def forward(self, x):
    x1, x2 = torch.chunk(x, 2, dim=1)        # split into two halves
    s_tilde, t = self.subnet(x1).chunk(2, dim=1)
    s = torch.tanh(s_tilde)                   # bound the log-scale for stability
    y2 = x2 * torch.exp(s) + t                # scale-and-shift the second half
    log_det = s.sum(dim=1)                    # Jacobian is triangular -> sum(s)
    return torch.cat([x1, y2], dim=1), log_det

def inverse(self, y):
    y1, y2 = torch.chunk(y, 2, dim=1)
    s_tilde, t = self.subnet(y1).chunk(2, dim=1)
    s = torch.tanh(s_tilde)
    x2 = (y2 - t) * torch.exp(-s)             # exactly undo the forward
    return torch.cat([y1, x2], dim=1)
```

- **`torch.chunk(x, 2, dim=1)`** splits a tensor into two equal halves.
- Because `y1 = x1` is copied and `y2` depends on `x1` only, the inverse is exact:
  recover `s, t` from `y1`, then `x2 = (y2 - t) * exp(-s)`.
- The Jacobian is **triangular**, so its log-determinant is just **`sum(s)`** — cheap
  to compute, which is exactly what the likelihood loss needs.
- **`tanh`** keeps the log-scale `s` in a safe range so `exp(s)` never explodes.

### Stacking layers with swaps

One coupling layer never transforms its first half. So you stack several and
**swap** the halves between them, and the inverse must run everything backwards:

```python
def forward(self, x):
    log_det_total = 0
    for layer in self.layers:
        x, ld = layer(x)
        x = x.flip(dims=[1])          # swap halves so every dim gets transformed
        log_det_total += ld
    return x, log_det_total

def inverse(self, z):
    for layer in reversed(self.layers):   # REVERSE order
        z = z.flip(dims=[1])              # undo the swap first
        z = layer.inverse(z)
    return z
```

The forward pass **accumulates** every layer's `log_det`; the inverse undoes each
step in reverse order. To sample a new image you draw `z ~ N(0, I)` and call
`inverse(z)`.

### The negative-log-likelihood loss

```python
def nll_loss(z, log_det):
    return (0.5 * (z**2).sum(dim=1) - log_det).mean()
```

This is the change-of-variables formula with a standard-Gaussian latent, constants
dropped: `0.5*||z||^2` rewards mapping data to the center of the Gaussian, and
subtracting `log_det` accounts for how much the flow stretched space. Minimizing it
*maximizes* the exact likelihood of the data.

**Sanity check:** because the flow is invertible by construction,
`flow.inverse(flow.forward(x))` should return `x` with near-zero error. Sweeping
the hidden size and number of blocks shows hidden width matters most for fitting
the two-moons shape, and that `z = f(x)` indeed looks Gaussian afterwards.

### Making the flow conditional

To choose *which* digit to generate, feed the class label into every subnet:

```python
y_onehot = F.one_hot(labels, num_classes).float()
s_tilde, t = self.subnet(torch.cat([x1, y_onehot], dim=1)).chunk(2, dim=1)
```

**`F.one_hot`** turns an integer label into a vector of 0s with a single 1, which is
concatenated onto the coupling input. Now the flow models `p(x | y)`, and sampling
with a fixed label produces that specific digit — the same conditioning idea used
in the VAE (Ex3) and GAN (Ex5), and the foundation for the cINN in Ex7.

## 4. How it fits the big picture

- Completes the four generative families:
  - **VAE (Ex3):** approximate likelihood, samples blurry.
  - **GAN (Ex5):** no likelihood, samples sharp but unstable.
  - **Flow (Ex6):** EXACT likelihood + exact invertibility, by construction.
- Conditioning via labels is the same theme as the conditional VAE (Ex3) and
  conditional GAN (Ex5), now applied to a flow → `p(x|y)`.
- The affine-coupling machinery here is the EXACT prerequisite for Ex7, which
  reuses coupling layers for disentanglement and for an image-space conditional
  flow (cINN), and exploits the exact-likelihood property for classification and
  anomaly detection.

> **Exam takeaway:** State the change-of-variables formula and the resulting NLL loss;
> explain the affine coupling layer (why it is invertible and why `log|det|=sum(s)`);
> explain why swap layers are needed; know that flows give EXACT likelihood, unlike
> VAEs/GANs.
