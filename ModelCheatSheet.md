# Generative Models — Final Cheat Sheet

> Course: Generative AI and Visual Synthesis (SoSe 2026)
> A one-page overview of every model concept covered across the exercises.
> For code-level detail see `CodeGlossary.md` and each `exercise_*/SUMMARY_*.md`.

---

## The big picture at a glance

| Model | Latent | Likelihood | Sampling | Sample quality |
|---|---|---|---|---|
| Autoencoder (Ex2) | continuous, *unstructured* | ✗ | ✗ (cannot generate) | — |
| VAE (Ex3) | continuous Gaussian | approximate (ELBO) | draw `z~N(0,I)`, decode | blurry |
| VQ-VAE (Ex4) | **discrete** codebook | ✗ (needs a prior) | via a 2nd prior model | sharp (with prior) |
| GAN (Ex5) | continuous noise | ✗ | one forward pass of G | sharp, but unstable |
| Normalizing Flow (Ex6/7) | continuous, same dim as data | **exact** | invert `z~N(0,I)` | good, exactly invertible |
| Energy-Based Model (Ex8) | none (energy `E(x)`) | unnormalized (`Z` intractable) | Langevin dynamics | slow to sample |
| Score / Diffusion (Ex8) | noise schedule | implicit (via score) | annealed Langevin | state-of-the-art |

---

## 0. Convolutional Neural Network (CNN) — *Exercise 1*
*Not generative, but the backbone of every model below.*

**Core concept:** Slide learnable filters (kernels) over an image so the network learns spatial feature detectors (edges, textures, shapes) directly from data instead of hand-designing them.

- **✅ Advantages:** Exploits spatial locality → far fewer parameters than fully-connected; translation-equivariant; learns a hierarchy of features.
- **❌ Weaknesses:** Not generative on its own; fixed receptive field; needs labels for classification tasks; interpretability requires extra tools (saliency maps).
- **🔧 Example use:** Image classification, and as the encoder/decoder building block inside all the generative models here.

---

## 1. Autoencoder (AE) — *Exercise 2*

**Core concept:** Compress an image through a narrow bottleneck (encoder) and rebuild it (decoder), forcing the network to keep only the essential structure.

- **✅ Advantages:** Simple to train (just MSE reconstruction); learns compact representations; the denoising variant learns robust features; conv version is parameter-efficient.
- **❌ Weaknesses:** **Cannot generate new images** — the latent space has no known distribution to sample from; latent space has gaps/holes; only reconstructs, does not create.
- **🔧 Example use:** Dimensionality reduction, image compression, denoising (denoising AE), anomaly detection via reconstruction error, feature pretraining.

---

## 2. Variational Autoencoder (VAE) — *Exercise 3*

**Core concept:** An autoencoder whose encoder outputs a *distribution* (mean + variance) and whose latent space is regularized toward `N(0, I)`, so you can sample a random latent and decode a brand-new image.

- **✅ Advantages:** Truly generative (sample `z~N(0,I)` → decode); smooth, structured latent space good for interpolation; probabilistic/principled (ELBO); easy to make conditional; stable training.
- **❌ Weaknesses:** Samples are **blurry** (MSE/BCE reconstruction averages detail); only an *approximate* likelihood; the recon-vs-KL balance (`kl_weight`) needs tuning; posterior collapse possible.
- **🔧 Example use:** Generating new class-conditioned digits, latent-space interpolation, anomaly/outlier detection (high reconstruction loss = out-of-distribution).

---

## 3. VQ-VAE (Vector-Quantized VAE) — *Exercise 4*

**Core concept:** An autoencoder whose latent is **discrete** — each spatial location is snapped to the nearest vector in a learned codebook, turning images into grids of "tokens".

- **✅ Advantages:** Discrete tokens enable powerful token-based generators (DALL-E, VQGAN); avoids blurriness of continuous VAEs; codebook is a compact, reusable vocabulary.
- **❌ Weaknesses:** Quantization is non-differentiable → needs the **straight-through estimator**; prone to **codebook collapse** (few codes used); **cannot sample directly** — you need a second "prior" model (e.g. a Transformer) to generate realistic token arrangements.
- **🔧 Example use:** Foundation of token-based image generation; compressing images into discrete tokens for a downstream autoregressive/Transformer prior.

### 3b. Masked Image Modeling (MIM) — *Exercise 4, Task 2*
**Core concept:** Hide random patches of an image and train a network to fill them back in — the vision analogue of masked-word prediction (BERT/MAE).
- **✅ Advantages:** Self-supervised (no labels); learns strong general-purpose features; scalable.
- **❌ Weaknesses:** Reconstructions blur at high mask ratios (model defaults to the mean); a pretext task, not a generator by itself.
- **🔧 Example use:** Self-supervised pretraining of vision encoders; inpainting.

---

## 4. Generative Adversarial Network (GAN) — *Exercise 5*

**Core concept:** A game between a Generator (turns noise into fakes) and a Discriminator (tells real from fake); they compete until fakes are indistinguishable from real data (minimizes Jensen-Shannon divergence).

- **✅ Advantages:** **Sharp, realistic** samples; fast single-pass generation; no explicit likelihood needed; flexible conditioning and latent-space control (interpolation, SLERP).
- **❌ Weaknesses:** **Notoriously unstable** training; **mode collapse** (low variety); no likelihood estimate; hard to evaluate (needs proxy metrics); requires stabilization tricks (spectral norm, label smoothing, R1 penalty).
- **🔧 Example use:** High-fidelity image synthesis, class-conditional generation, image editing/morphing via latent interpolation, super-resolution.

---

## 5. Normalizing Flow (RealNVP) — *Exercise 6*

**Core concept:** Build an *invertible* neural network `f` mapping data `x` ↔ a simple latent `z ~ N(0,I)`; the change-of-variables formula gives the **exact** likelihood, and inverting `f` generates samples.

- **✅ Advantages:** **Exact likelihood** (unlike VAE/GAN); exactly invertible by construction; stable maximum-likelihood training; cheap Jacobian via affine coupling layers.
- **❌ Weaknesses:** Latent must have the **same dimension** as the data (memory-heavy); architecture is constrained (every layer must be invertible); many coupling blocks needed for expressiveness; samples often less sharp than GANs.
- **🔧 Example use:** Exact density estimation, likelihood-based generation, any task needing a tractable `p(x)`.

---

## 6. Conditional INN / cINN — *Exercise 7*

**Core concept:** An invertible network conditioned on a label, `p(x|y)`, where the leftover latent/noise variable stores exactly the information the condition throws away (what makes inversion possible).

- **✅ Advantages:** One model does generation, exact density, **classification** (argmin NLL over conditions), and **anomaly detection**; enables **disentanglement** — invertibly reorganize a latent into named factors (digit, color, style) and edit one factor exactly.
- **❌ Weaknesses:** Inherits all flow constraints (invertibility, dimension matching); requires careful factor design + a residual channel to stay bijective; conditioning must be fed into every coupling layer.
- **🔧 Example use:** Controllable/disentangled generation and image editing (recolor, swap identity), likelihood-based classification, outlier detection.

---

## 7. Energy-Based Model (EBM) — *Exercise 8, Tasks 1–2*

**Core concept:** Model an *unnormalized* energy `E(x)` where `p(x) ∝ e^{-E(x)}`; low energy = high probability. You never compute the intractable normalizer `Z` — you sample by walking downhill in energy with noise (Langevin dynamics) and train with contrastive divergence.

- **✅ Advantages:** Very flexible (any scalar network is a valid energy); sidesteps `Z` via the score `-∇E`; naturally acts as a denoiser; relative likelihoods are comparable.
- **❌ Weaknesses:** **Slow sampling** (long Langevin chains); training is finicky (needs energy regularization, tuned step sizes); no absolute likelihood; contrastive divergence is a biased approximation.
- **🔧 Example use:** Denoising, out-of-distribution / anomaly detection (real data = low energy), unconditional image generation.

---

## 8. Score-Based Model / Denoising Score Matching — *Exercise 8, Task 3*
*The direct precursor to diffusion models.*

**Core concept:** Skip the energy and learn the **score** `∇_x log p(x)` directly; denoising score matching supervises it against the closed-form score of Gaussian-corrupted data, and generation runs *annealed* Langevin dynamics from high to low noise.

- **✅ Advantages:** **State-of-the-art sample quality**; stable regression (MSE) training; the noise schedule (coarse→fine) explores globally then refines; one noise-conditioned U-Net handles all scales; foundation of modern diffusion models.
- **❌ Weaknesses:** **Slow iterative sampling** (many noise levels × many Langevin steps); requires a well-tuned noise schedule and step sizes; no explicit likelihood readout.
- **🔧 Example use:** High-quality image generation (diffusion), inpainting, conditional generation — the basis of Stable Diffusion / DALL-E-style systems.

---

## Quick decision guide

- **Need exact likelihood?** → Normalizing Flow / cINN.
- **Need the sharpest fast samples?** → GAN.
- **Need best-quality samples & don't mind slow sampling?** → Score/Diffusion.
- **Need a smooth, samplable latent + easy conditioning?** → VAE.
- **Need discrete tokens for a Transformer prior?** → VQ-VAE.
- **Need anomaly/outlier detection?** → VAE (recon error), cINN or EBM (likelihood/energy).
- **Just compressing / denoising, no generation?** → (Denoising) Autoencoder.
