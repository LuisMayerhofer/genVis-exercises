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
  input. Pulls the model toward copying data accurately. => Good Reconstruction
- **KL divergence loss**: pulls each encoded distribution toward the standard normal
  `N(0, I)`. This **regularizes** the latent space so it is smooth and samplable. => Good latent space

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

## 3. The core building blocks (what the code does and why)

### The encoder: outputting a distribution, not a point

A plain autoencoder's encoder outputs one latent vector. A VAE's encoder instead
outputs **two** vectors — a mean and a log-variance — describing a Gaussian:

```python
def encode(self, x, y):
    h = self.conv_layers(x)               # convolutions extract features
    h = torch.flatten(h, start_dim=1)     # (B,C,H,W) -> (B, features)
    h = torch.cat([h, self.embed(y)], 1)  # append the class-label embedding
    return self.fc_mean(h), self.fc_var(h)  # -> mean, log_var (two Linear heads)
```

Two separate `Linear` heads (`fc_mean`, `fc_var`) read the same features and
produce the mean and `log_var`. **`torch.cat([..., embed(y)], dim=1)`** glues the
class information onto the features so the model knows which digit it is handling.

### The reparameterization trick (the key trainable step)

You cannot backpropagate through `torch.randn` (random sampling has no gradient).
The trick moves the randomness into an independent variable `eps`:

```python
def reparameterize(self, mean, log_var):
    std = torch.exp(0.5 * log_var)        # convert log-variance -> std-dev
    eps = torch.randn_like(std)           # random noise, same shape, NO gradient
    return mean + std * eps               # gradients still flow into mean & log_var
```

Because `eps` is just fixed noise during a backward pass, gradients flow cleanly
into `mean` and `log_var`. Predicting **`log_var`** instead of variance lets the
network output any real number while `exp()` guarantees a positive std and keeps
training numerically stable.

### The decoder and sampling new images

```python
def decode(self, z, y):
    h = torch.cat([z, self.embed(y)], 1)  # condition the latent on the class
    h = self.project_and_reshape(h)       # vector -> small feature map
    h = self.upsample_conv(h)             # grow back to image size
    return torch.sigmoid(h)               # Sigmoid -> pixels in [0, 1]

def sample(self, y):
    z = torch.randn(n, latent_dim)        # draw a fresh latent from N(0, I)
    return self.decode(z, y)              # -> a brand-new image of class y
```

`sample` is what makes a VAE *generative*: since the latent space is trained to
look like `N(0, I)`, you can draw a random `z` and decode it into a never-seen
image. **`Sigmoid`** is the final activation here (not Tanh) because these images
are scaled to `[0, 1]`, matching the binary-cross-entropy loss below.

### The two-term loss in code

```python
def vae_loss(x_recon, x, mean, log_var, kl_weight):
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
    return recon + kl_weight * kl
```

- **Reconstruction term** (`binary_cross_entropy`) pushes the output to match the
  input pixel-for-pixel. `reduction='sum'` adds the error over all pixels.
- **KL term** uses a *closed-form formula* (no sampling needed) that measures how
  far the encoded Gaussian is from `N(0, I)`, pulling the latent space into the
  standard-normal shape you need for sampling.
- `kl_weight` balances the two; they pull in opposite directions.

The training loop is the same `zero_grad → forward → loss → backward → step` rhythm
from Ex2.

### Task 2 — the ablation that proves you need both terms

By training with each term switched off and plotting the 2-D latent space, you see:

- **recon-only** (no KL): classes separate well, but the cloud has gaps and isn't
  Gaussian → random `z` decodes to garbage.
- **KL-only** (no recon): a perfect Gaussian blob, but classes overlap and
  reconstructions are meaningless.
- **full**: Gaussian-shaped *and* class-structured → good samples. You need both.

### Task 3 — anomaly detection via reconstruction error

Feed images the VAE was *not* trained on (Fashion-MNIST) into the digit-trained
VAE and measure each one's reconstruction loss:

```python
loss_per_image = F.binary_cross_entropy(recon, x, reduction='none').sum(dim=[1,2,3])
```

A model only reconstructs well what resembles its training data, so out-of-
distribution images get a **high** reconstruction loss. Thresholding that loss
flags anomalies — the same likelihood-based outlier idea returns in Ex7.

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
