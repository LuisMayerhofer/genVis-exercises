# Exercise 3 — Variational Autoencoders (VAE)

> Course: Generative AI and Visual Synthesis (SoSe 2026)

## 1. What this exercise is about

The VAE turns the autoencoder from Ex2 into a real **generative** model. Instead of
encoding an image to a single point, the encoder outputs a small probability
distribution (a Gaussian: a mean and a variance) in latent space. Because the
latent space is now shaped like a standard normal distribution, you can **sample** a
random latent vector and decode it into a brand-new image — something a plain
autoencoder cannot do.

The model here is also **conditional**: it takes the class label `y` as an extra input,
so you can ask it to generate a specific digit. Dataset: MNIST (handwritten
digits 0-9).

## 2. Key concepts / theory

**Two-part loss (the heart of the VAE):**

```
loss = reconstruction_loss + kl_weight * KL_loss
```

- **Reconstruction loss** (here binary cross-entropy): make the output look like the
  input. Pulls the model toward copying data accurately.
- **KL divergence loss**: pulls each encoded distribution toward the standard normal
  `N(0, I)`. This **regularizes** the latent space so it is smooth and samplable.

These two FIGHT each other; `kl_weight` balances them (here 0.0001, small).

**Reparameterization trick:** you cannot backpropagate through random sampling. Trick:

```
z = mean + exp(0.5*log_var) * eps,   eps ~ N(0, I)
```

The randomness lives in `eps` (no gradient needed), while `mean` and `log_var` get
gradients. This is THE key idea that makes VAEs trainable.

**Why `log_var` (not `var`):** variance must be positive; predicting log-variance lets the
network output any real number and `exp()` makes it positive + numerically stable.

**Conditioning:** the class label is turned into a vector via `nn.Embedding` and
concatenated into the encoder and decoder, so generation can be class-controlled.

> **Libraries new here:** `nn.Embedding` (turn an integer label into a learned vector),
> `F.binary_cross_entropy`, `F.interpolate` (upsampling inside the decoder).

## 3. Core code blocks (the TODOs / key given code)

### VAE class (given, but understand it)

- `encode(x, y)`: conv stack → flatten → concat class embedding →
  `fc_mean` and `fc_var` produce `mean` and `log_var`.
- `forward`: encode → reparameterize (`z = mean + exp(0.5*log_var)*eps`) → decode.
- `decode(z, y)`: concat class embedding, project, reshape, upsample+conv stack,
  Sigmoid to get pixels in `[0,1]`.
- `sample(y)`: draw `z ~ N(0,I)` and decode → a NEW image of class `y`.

### Task 1 — Train the VAE

- `vae_loss()`: `recon_loss = BCE(x_recon, x, sum)`;
  `kl_loss = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))` (closed-form KL);
  return `recon_loss + kl_weight * kl_loss`.
- Standard training loop (forward, loss, backward, optimizer step); plot
  generated samples per class each epoch.

### Task 2 — What each loss term DOES (ablation study)

Train three variants and visualize the 2D latent space:

- `'recon_only'` (KL off): latent space is well-separated by class BUT has gaps /
  is not shaped like a Gaussian → sampling random `z` gives bad images.
- `'kl_only'` (recon off): latent collapses to a perfect Gaussian blob but classes
  are NOT separated and reconstructions are meaningless.
- `'full'`: a good compromise — Gaussian-shaped AND class structure preserved.

Lesson: you NEED both terms.

### Task 3 — Anomaly detection

- Reconstruct MNIST (in-distribution) vs Fashion-MNIST (out-of-distribution)
  images with the digit-trained VAE; print per-image reconstruction loss.
- Fashion images get HIGH reconstruction loss → the model only reconstructs
  well what it was trained on, so high loss flags an anomaly/outlier.

## 4. How it fits the big picture

- Direct upgrade of Ex2: same encoder/decoder skeleton, now PROBABILISTIC so it
  can generate by sampling `z ~ N(0,I)`.
- Introduces three themes that recur for the rest of the course:
  - a structured/known latent distribution you can sample (vs Ex2's unusable
    latent),
  - **conditioning** on a class label via `nn.Embedding` (reused in Ex5 GAN, Ex6/Ex7
    conditional flows),
  - using likelihood/reconstruction error to detect outliers (mirrored by the
    likelihood-based detection in Ex7 Task 3.3).
- The VAE is one of the "big four" generative model families in this course:
  **VAE (Ex3) | VQ-VAE/discrete (Ex4) | GAN (Ex5) | Normalizing Flows (Ex6/7)**.

> **Exam takeaway:** Write the two-term VAE loss and explain what each term does, the
> reparameterization trick (`z = mean + exp(0.5 log_var)*eps` and why), why we
> predict log-variance, and how conditioning works. Know the Task-2 ablation
> results by heart (recon-only = gaps, KL-only = no class structure, full = good).
