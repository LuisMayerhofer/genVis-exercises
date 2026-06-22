# Exercise 5 — Generative Adversarial Networks (GANs)

> Course: Generative AI and Visual Synthesis (SoSe 2026) · [40 points, 2 weeks]

## 1. What this exercise is about

GANs, the third big generative-model family. A GAN is a GAME between two networks:

- **Generator (G):** turns random noise `z` (+ class label) into a fake image.
- **Discriminator (D):** tries to tell real images from G's fakes.

They train against each other: D gets better at spotting fakes, forcing G to make
better fakes, until fakes are indistinguishable from real. You derive the theory
(Task 1), build a class-conditional DCGAN (Task 2), study how to STABILIZE the
notoriously unstable training (Task 3), then explore control/interpolation
(Task 4) and evaluation (Task 5). Dataset: Fashion-MNIST resized to 32×32.

## 2. Key concepts / theory

**The minimax value function:**

```
V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

**Optimal discriminator (Task 1a):** for fixed G,

```
D*(x) = p_data(x) / (p_data(x) + p_G(x))
```

Derived by maximizing `f(y)=a log y + b log(1-y)` → `y = a/(a+b)`.

**Generator optimum (Task 1b):** substituting `D*` gives

```
V(D*,G) = -log 4 + 2 * JSD(p_data || p_G)
```

Global min when `p_G = p_data`, where `D*(x)=1/2` (D is reduced to coin-flipping) and
`V = -log 4`. So the GAN minimizes the Jensen-Shannon Divergence between real and
fake distributions.

**Non-saturating generator loss (Task 1c):** the "saturating" loss `log(1-D(G(z)))`
gives almost NO gradient early on when `D(G(z))~0` (G learns slowest exactly when it
needs to learn most). Replace with the non-saturating loss `-log D(G(z))`, whose
gradient EXPLODES (large) when `D(G(z))~0` → strong early learning signal. Same
optimum, better optimization.

**Why GANs are unstable:** if D learns too fast it overpowers G → D loss collapses to
~0, G loss diverges, G stops improving (often MODE COLLAPSE: G outputs little
variety). Three fixes (Task 3):

- **(A) Spectral normalization on D:** divide weights by their largest singular value,
  bounding D's Lipschitz constant (smoothness).
- **(B) One-sided label smoothing:** use real target 0.9 instead of 1.0 so D can't
  become over-confident and starve G of gradients.
- **(C) R1 gradient penalty:** penalize D's gradient magnitude on real images
  (`gamma/2 * ||grad_x D(x)||^2`) to avoid overly sharp decision boundaries.

(In this submission C/R1 worked best: ~91% classifier accuracy on generated
samples; B helped ~55%; A failed without LR tuning.)

**DCGAN design choices:**

- Generator uses Upsample + Conv (not ConvTranspose) to avoid checkerboard
  artifacts; BatchNorm + ReLU; Tanh output for `[-1,1]`.
- Discriminator uses strided Conv + LeakyReLU; outputs a single real/fake logit.
- Class conditioning via `nn.Embedding` concatenated into both G and D.
- Loss: `BCEWithLogitsLoss`; optimizer `Adam(lr=1e-4, betas=(0.2,0.95))`.

> **Libraries new here:** `nn.BCEWithLogitsLoss`, `nn.utils.parametrizations.spectral_norm`,
> `torch.autograd.grad` (for R1), `sklearn.manifold.TSNE` (visualize feature clusters).

## 3. Core code blocks (the TODOs)

### Task 1 (theory, written in LaTeX)

Derive `D*`, the JSD result, and the non-saturating loss argument (see section 2).

### Task 2 — Conditional DCGAN

- `ConditionalGenerator`: TODO build the upsample→conv→BatchNorm→ReLU stack
  (4×4 → 8×8 → 16×16 → 32×32) ending in Conv→Tanh.
- `ConditionalDiscriminator`: TODO build the strided conv stack
  (32 → 16 → 8 → 4) with LeakyReLU; the `sn()` wrapper enables spectral norm.
- `train_classifier`: TODO the forward + CrossEntropy loss (a frozen CNN
  classifier used later to score generated images).
- `train_gan`: TODO the D step (real loss vs real_target, fake loss vs 0,
  detach fakes) and the G step (non-saturating: fakes vs target 1).
- Reporting TODOs: generate random samples across classes; t-SNE of
  train/test/generated features; classifier accuracy on generated images.

### Task 3 — Stabilization (pick any two variants)

- **Variant A** TODO: set `sn = spectral_norm` and `use_spectral_norm=True`.
- **Variant B** TODO: `real_label_value = 0.9` and use it as the real target when
  `label_smoothing` is on.
- **Variant C:** R1 penalty code is GIVEN; just run and discuss.
- Written discussion comparing loss curves, D accuracy, sample quality.

### Task 4 — Control & interpolation (uses the best generator)

- **4a:** fix `z`, vary label `y=0..9` (does the label control the class?).
- **4b:** fix class, SLERP (spherical interpolation, given) between two latents `z1,z2`
  → smooth within-class morph. (Slerp preserves norm; plain LERP shrinks it and
  leaves the `N(0,I)` shell the generator was trained on.)
- **4c:** fix `z`, LERP between two class EMBEDDINGS → continuous class-to-class morph.
- **4d:** discuss controllability and smoothness.

### Task 5 — Confusion matrix

Generate balanced per-class samples, classify them with the frozen classifier,
plot the 10×10 confusion matrix + per-class accuracy. Most confusion: T-shirt vs
Shirt (same in real data).

**Bonus:** combine variants A + C.

## 4. How it fits the big picture

- Third generative family: VAE (Ex3, max-likelihood, blurry) vs GAN (Ex5,
  adversarial, sharp but unstable) vs Flows (Ex6/7, exact likelihood).
- Reuses Ex2/Ex3 building blocks: conv encoder/decoder, ConvTranspose/upsampling,
  `nn.Embedding` conditioning (like the conditional VAE).
- R1 gradient penalty connects back to the gradient-w.r.t.-input idea from the
  Ex1 saliency map.
- EVALUATION methodology (classifier accuracy on generated samples, t-SNE,
  confusion matrix) is a transferable skill for judging any generator.

> **Exam takeaway:** Derive `D*` and the JSD/-log4 result; explain the non-saturating
> loss gradient argument; describe GAN instability/mode collapse and the three
> stabilization tricks (spectral norm, label smoothing, R1). Know G uses
> upsample+conv+Tanh, D uses strided conv+LeakyReLU+single logit, with embedding-
> based class conditioning.
