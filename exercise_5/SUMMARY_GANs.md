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

## 3. The core building blocks (what the code does and why)

### The generator — growing noise into an image

The generator turns a random latent `z` plus a class label into a 32×32 image by
*upsampling* a tiny feature map step by step:

```python
nn.Embedding(num_classes, embed_dim)        # turn the label into a vector
# concat that vector with z, then repeatedly:
nn.Upsample(scale_factor=2)                 # double the spatial size (4->8->16->32)
nn.Conv2d(in_ch, out_ch, 3, padding=1)      # learnable filters refine the detail
nn.BatchNorm2d(out_ch)                      # stabilizes/centres activations
nn.ReLU()
# final layer:
nn.Conv2d(..., out_channels=1); nn.Tanh()   # 1 grayscale channel, pixels in [-1,1]
```

- **`nn.Upsample` + `nn.Conv2d`** is used *instead of* `ConvTranspose2d` to avoid
  "checkerboard" artifacts: upsample enlarges, then conv fills in detail.
- **`nn.BatchNorm2d`** normalizes each layer's outputs across the batch, which keeps
  GAN training from blowing up.
- **`nn.Embedding`** converts the integer class label into a learnable vector that is
  concatenated in, so the generator can produce a *chosen* class.
- **`Tanh`** final activation matches the `[-1, 1]` image normalization.

### The discriminator — shrinking an image into one verdict

The discriminator does the reverse: **strided** convolutions halve the size each
step until a single real/fake score remains.

```python
nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)   # stride 2 -> halves H,W
nn.LeakyReLU(0.2)                                   # like ReLU but small negative slope
# ... 32 -> 16 -> 8 -> 4 ... then flatten to a single logit
```

- **stride=2** makes the convolution *downsample* (no separate pooling needed).
- **`LeakyReLU`** lets a little gradient through for negative inputs, which the
  discriminator needs to keep learning.
- The output is a single **logit** (raw score, not yet a probability) — paired with
  `BCEWithLogitsLoss` below.

### The adversarial training step

The two networks are trained alternately. Note the **`.detach()`** when training D:

```python
criterion = nn.BCEWithLogitsLoss()

# --- Discriminator step: learn to separate real from fake ---
d_real = D(real_imgs, labels)
d_fake = D(fake_imgs.detach(), labels)             # detach: don't update G here
d_loss = criterion(d_real, real_target) + criterion(d_fake, zeros)

# --- Generator step: fool the discriminator (non-saturating) ---
d_fake = D(fake_imgs, labels)
g_loss = criterion(d_fake, ones)                   # push fakes toward "real"=1
```

- **`BCEWithLogitsLoss`** combines a sigmoid + binary cross-entropy in one
  numerically-stable step, so D can output raw logits.
- **`fake_imgs.detach()`** in the D step stops gradients from flowing into the
  generator — D should only update *itself* from the fakes.
- The G step targets label **1** for its fakes: this is the **non-saturating** loss
  `-log D(G(z))`, giving strong gradients early when fakes are obviously fake.

### Stabilization tricks (in code)

```python
# A) Spectral normalization: wrap D's conv layers
from torch.nn.utils.parametrizations import spectral_norm
sn = spectral_norm if use_spectral_norm else (lambda m: m)
layer = sn(nn.Conv2d(...))                          # bounds D's "sharpness"

# B) One-sided label smoothing: soften the real target
real_target = torch.full_like(d_real, 0.9)         # 0.9 instead of 1.0

# C) R1 gradient penalty: punish steep gradients on real images
grad = torch.autograd.grad(d_real.sum(), real_imgs, create_graph=True)[0]
r1 = (gamma / 2) * grad.pow(2).sum(dim=[1,2,3]).mean()
```

Each tames an over-confident discriminator (the usual cause of GAN collapse):
**spectral norm** caps how fast D's output can change, **label smoothing** stops D
from being 100% certain, and the **R1 penalty** (`torch.autograd.grad` takes the
gradient of D's real score w.r.t. its input — the same idea as the Ex1 saliency
map) discourages sharp decision boundaries. In this submission R1 worked best.

### Control & interpolation

```python
# vary the label with z fixed  -> does y control the class?
imgs = [G(z, torch.tensor([c])) for c in range(10)]

# SLERP between two latents (spherical) keeps them on the N(0,I) shell:
z = (sin((1-t)*omega)*z1 + sin(t*omega)*z2) / sin(omega)
```

Plain linear interpolation (LERP) between two latents would shrink their length and
leave the unit "shell" the generator was trained on, giving blurry midpoints;
**SLERP** interpolates along the sphere and preserves the norm, so morphs stay
sharp. You can also LERP between two class **embeddings** to morph one class into
another.

### Evaluating a generator

There is no pixel-wise ground truth for generated images, so you judge them
indirectly with a **separately trained, frozen classifier**:

```python
preds = classifier(generated_imgs).argmax(1)
acc = (preds == intended_labels).float().mean()    # are samples recognizable?
```

High accuracy means the samples look like their intended class. You also plot a
**confusion matrix** (which classes get mixed up) and a **t-SNE** projection
(`sklearn.manifold.TSNE`) that squashes high-dimensional features to 2-D to check
that generated samples cluster with real ones. These evaluation tools transfer to
*any* generative model.

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
