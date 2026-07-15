# Exercise 8 — Energy-Based Models & Score Matching

> Course: Generative AI and Visual Synthesis (SoSe 2026)

## 1. What this exercise is about

The fifth generative family and the direct run-up to diffusion models. Instead of
mapping to a latent (VAE, flow) or playing a game (GAN), you model an **unnormalized
energy** `E(x)` where low energy = high probability, and you *sample* by walking
downhill in energy with noise. Three parts:

- **Task 1 — Sampling from a known EBM:** on a fixed 2-D two-mode energy, implement
  **Langevin dynamics** and study how step size `α` and number of steps `N` control
  convergence.
- **Task 2 — Training an EBM:** learn `E_θ(x)` on MNIST with **contrastive
  divergence**, using a short Langevin chain (initialized from a noisy real image)
  to produce the "fake" negative samples. The result is a denoiser.
- **Task 3 — Score matching:** skip the energy and learn the **score**
  `∇_x log p(x)` directly with a **denoising score matching** objective, train a
  small U-Net conditioned on the noise level, and generate digits with **annealed
  Langevin dynamics** — essentially a score-based diffusion model.

## 2. Key concepts / theory

**Energy-based model.** Write `p(x) = exp(−E(x)) / Z` with `Z = ∫ exp(−E(x)) dx` an
intractable normalizer. The key escape: the **score** does not depend on `Z`,

```
∇_x log p(x) = −∇_x E(x)          (Z is constant in x, so its gradient is 0)
```

so you can move toward likely samples without ever computing `Z`.

**Langevin dynamics (the sampler).** Take gradient steps toward lower energy and add
Gaussian noise:

```
x_{t+1} = x_t − (α²/2) ∇_x E(x_t) + α z,     z ~ N(0, I)
```

As `t → ∞` (and step size → 0) the chain's stationary distribution is exactly
`p(x) ∝ e^{−E(x)}`. The **drift** pulls samples into low-energy basins; the **noise**
is what turns mode-seeking gradient descent into proper *sampling* (without it every
point collapses onto the nearest minimum — zero variance, only modes recovered).
`α` must be big enough to move but small enough to stay accurate; `N` big enough to
converge.

**Training an EBM via contrastive divergence (Task 2).** Maximizing log-likelihood
gives two competing terms:

```
∇_θ log p_θ(x) ≈ −∇_θ E_θ(x)  +  ∇_θ E_θ(x̃),     x̃ ~ p_θ
```

i.e. **push down** the energy of real data and **push up** the energy of model
samples `x̃`. Since sampling `x̃` from scratch is slow, contrastive divergence
initializes the Langevin chain from a **noisy real image** and runs only a few steps.
An `L2` penalty on the energies keeps them from diverging.

**Score matching (Task 3).** Instead of energy, learn the score field directly:
`s_θ(x) ≈ ∇_x log p(x)`. The true score is unknown, so **denoising score matching**
supervises against the *conditional* score of a Gaussian-corrupted sample. With
`x_t = x + σ_t ε`, the conditional score has a closed form:

```
∇_{x_t} log N(x_t | x, σ_t² I) = (x − x_t) / σ_t²  =  −ε / σ_t
```

Train `s_θ(x_t, σ_t)` to match this with a σ-weighted MSE. Sampling uses **annealed
Langevin dynamics**: run Langevin at a decreasing schedule of noise levels
`σ = 2 → 0.01`, so the chain first explores coarsely then refines — exactly the
score-based generative modeling recipe behind diffusion.

> **Libraries new here:** `torch.autograd.grad(energy, x)[0]` (score by
> autodiff w.r.t. the *input*), `torch.logsumexp` (numerically stable
> log-sum-of-exponentials for the mixture energy), `nn.GroupNorm`,
> `nn.ConvTranspose2d` (U-Net upsampling), `torchvision.datasets.MNIST`.

## 3. The core building blocks (what the code does and why)

### Task 1 — Langevin sampling on a known 2-D energy

The target is a two-bump energy (a mixture of two Gaussians at `(±2, 0)`):

```python
def energy(x):
    d1_sq = ((x - mu1) ** 2).sum(dim=1); d2_sq = ((x - mu2) ** 2).sum(dim=1)
    logits = torch.stack([-0.5 * d1_sq, -0.5 * d2_sq], dim=1)
    return -torch.logsumexp(logits, dim=1)      # −log( e^-d1²/2 + e^-d2²/2 )

def score(x):
    x = x.detach().requires_grad_(True)
    grad = torch.autograd.grad(energy(x).sum(), x)[0]
    return -grad.detach()                        # score = −∇E
```

- **`torch.logsumexp`** computes `log(Σ exp(·))` stably — the natural form of a
  log-mixture energy, and it matches the `−log(e^… + e^…)` in the task.
- **`torch.autograd.grad`** differentiates the energy w.r.t. the **input** `x` (not
  parameters) — this is how you get the score of *any* differentiable energy.

The sampler is the Langevin update, run for `N` steps from `x_0 ~ N(0, I)`:

```python
def langevin_sampling(x0, n_steps, step_size):
    x = x0.clone().requires_grad_(True)
    alpha = step_size
    for _ in range(n_steps):
        s = score(x)                              # −∇E  (toward higher probability)
        x = x + 0.5 * alpha**2 * s + alpha * torch.randn_like(x)
    return x
```

Plotting the 500 final samples over both the energy contours and the density shows
them concentrating in the two low-energy / high-density basins, split between the two
modes, with the noise keeping them spread *within* each mode. The **ablation grid**
(`N ∈ {1,10,100,1000}` × `α ∈ {0.001,0.01,0.1,1}`) makes the trade-off visible:
tiny `α` never leaves the origin, `α = 0.1` with `N ≥ 100` converges cleanly onto
both modes, and `α = 1` overshoots into a diffuse blob (large step-size / discretization bias).

### Task 2 — a CNN energy model trained with contrastive divergence

The energy network is a 3-layer conv stack (each block: GroupNorm → conv → ReLU →
2×2 maxpool, shrinking 28→14→7→3) followed by a 2-layer MLP to a **scalar**:

```python
class EnergyCNN(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            CNNLayer(1, 16), CNNLayer(16, 32), CNNLayer(32, 64),
            nn.Flatten(), nn.Linear(64*3*3, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(1)
```

The image-space Langevin sampler mirrors Task 1 but uses the *model's* gradient and
**clamps pixels** to the valid `[0,1]` range each step:

```python
def langevin_sample_cnn(model, x, n_steps=30, step_size=0.1, noise_scale=0.01):
    x = x.clone().detach().requires_grad_(True)
    for _ in range(n_steps):
        grad = torch.autograd.grad(model(x).sum(), x)[0]
        x = x - step_size * grad + noise_scale * torch.randn_like(x)  # downhill + noise
        x = x.clamp(0.0, 1.0).detach().requires_grad_(True)
    return x.detach()
```

The **contrastive divergence** training step: build negatives by adding noise
(`σ² = 0.3`) to real images and refining with 30 Langevin steps, then push real
energy down and fake energy up (plus the `L2` energy regularizer):

```python
noisy = (real + math.sqrt(0.3) * torch.randn_like(real)).clamp(0, 1)
fake  = langevin_sample_cnn(model, noisy, n_steps=30, step_size=0.2, noise_scale=0.01)
cd_loss  = model(real).mean() - model(fake).mean()          # ↓ real energy, ↑ fake energy
reg_loss = real_energy.pow(2).mean() + fake_energy.pow(2).mean()
loss = cd_loss + reg_weight * reg_loss                       # reg_weight = 0.001
```

Visualizing real / noisy / denoised rows every 1000 steps shows the trained EBM
acting as a **denoiser**: the Langevin chain pulls corrupted digits back toward the
learned low-energy manifold.

### Task 2.3 — evaluation via *relative* log-probability

Absolute `log p(x) = −E(x) − log Z` is uncomputable, but **differences** kill the
constant: `log p(x) − log p(real) = E(real) − E(x)`. Averaged over 1000 samples the
ordering comes out as expected:

```
ground_truth (0) ≳ denoised (−105) > noisy (−290) ≫ random_noise (−983)
```

Denoised samples land close to real (the sampler works); random noise is off the
manifold and gets huge energy. The matching "relative *probability*" column
(`exp` of those) underflows to `0.0000` for everything but real — which is exactly
the point of the write-up: **report log-probabilities** because (1) probabilities of
high-dim data underflow to 0, (2) the intractable `log Z` cancels additively in
log-space, and (3) log-likelihoods of independent samples add, giving stable
gradients.

### Task 3 — denoising score matching with a U-Net

**3.1 — the derivation.** Starting from `q(x_t|x) = N(x_t; x, σ_t² I)`, take
`log`, drop `x`-independent constants, and differentiate w.r.t. `x_t`:

```
log q = −‖x_t − x‖² / (2σ_t²) + const
∇_{x_t} log q = −(x_t − x)/σ_t² = (x − x_t)/σ_t²      (= −ε/σ_t for x_t = x + σ_t ε)
```

**3.2 — model & training.** The score net is a small **U-Net** (`ConvBlock`s,
maxpool down / `ConvTranspose2d` up, skip connections) conditioned on the noise
level. A **sigmoidal/sinusoidal σ-embedding** is projected per resolution and *added*
to the feature maps so the same network handles every noise scale. Training draws a
random `σ ~ U(0.01, 2)` per batch, corrupts `x_t = x + σε`, and regresses the
predicted score onto the closed-form target with a **σ²-weighted MSE**:

```python
x = 2*x - 1                                   # MNIST -> [-1, 1]
sigma = torch.rand(...) * (2 - 0.01) + 0.01   # σ ~ U(0.01, 2)
x_t = x + sigma * torch.randn_like(x)
target = (x - x_t) / sigma**2                 # conditional score
loss = (sigma**2 * (model(x_t, sigma) - target)**2).mean()   # weighted MSE
```

The `σ²` weight balances the loss across noise levels (the target score scales like
`1/σ`, so unweighted MSE would be dominated by small-σ samples).

**3.3 — annealed Langevin sampling.** Generate from pure Gaussian noise by running
Langevin at 50 **decreasing** noise levels (`σ = 2 → 0.01`), 20 updates each, with a
σ-scaled step `α = β (σ/σ_max)²`, `β = 2e-5`:

```python
for sigma in sigmas:                          # high -> low noise
    alpha = step_lr * (sigma / sigmas[-1])**2
    for _ in range(steps_for_sigma):
        x = x + 0.5 * alpha * model(x, sigma_batch) + torch.sqrt(alpha) * torch.randn_like(x)
```

Coarse (high-σ) steps place global structure, fine (low-σ) steps sharpen details —
the annealing schedule that makes score-based generation work, and the conceptual
core of diffusion models.

## 4. How it fits the big picture

- **Completes the family tour:** VAE (Ex3, approximate likelihood), GAN (Ex5, no
  likelihood), Flow (Ex6/7, exact likelihood by construction), and now **EBM /
  score models** — unnormalized density, sampled by Langevin dynamics.
- **The score `∇_x log p(x)` is the unifying object:** in Ex6/7 the flow gave exact
  likelihood; here the score alone is enough to *sample*, sidestepping the
  intractable `Z`. `torch.autograd.grad` w.r.t. the input is the recurring tool.
- **Denoising as a theme:** the EBM Langevin chain (Task 2) is literally a denoiser,
  and denoising score matching (Task 3) trains on the denoising objective — the same
  intuition the noisy-latent VAE and the corruption-based tasks used earlier.
- **Direct precursor to diffusion:** annealed Langevin over a σ-schedule with a
  noise-conditioned U-Net *is* score-based diffusion. Everything here (score, noise
  schedule, U-Net conditioning, reparametrized corruption) reappears in DDPM-style
  models.

> **Exam takeaway:** Define an EBM (`p ∝ e^{−E}`, `Z` intractable) and why the score
> `−∇_x E` avoids `Z`. Write the Langevin update and explain the roles of drift and
> noise (noise ⇒ sampling, not mode collapse; makes `p ∝ e^{−E}` the stationary
> distribution). State the contrastive-divergence objective (push real energy down,
> Langevin-fake energy up, init from noisy reals). Derive the denoising-score-matching
> target `(x − x_t)/σ_t² = −ε/σ_t` and know the σ²-weighted MSE and annealed Langevin
> sampling. Know why papers report log-probabilities (underflow, `log Z` cancels,
> additivity).
