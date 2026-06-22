# Exercise 4 — Discrete & Masked Image Models (VQ-VAE + MIM)

> Course: Generative AI and Visual Synthesis (SoSe 2026)

## 1. What this exercise is about

Two modern ideas, both on Fashion-MNIST:

- **Task 1 — VQ-VAE (Vector-Quantized VAE):** an autoencoder whose latent code is
  **discrete**. Instead of continuous numbers, each spatial location of the
  latent is snapped to the nearest entry in a learned "codebook" (a
  dictionary of vectors). This is the foundation of token-based image
  generators (DALL-E, VQGAN, etc.).
- **Task 2 — Masked Image Modeling (MIM):** hide random square patches of an image
  and train a network to fill them back in. This is the vision version
  of how language models learn by predicting masked words (BERT/MAE).

## 2. Key concepts / theory

**VQ-VAE pipeline:** encoder → `z_e` (continuous) → QUANTIZE to nearest codebook
vector → `z_q` (discrete) → decoder → image.

**Three loss terms (memorize):**

- **reconstruction loss (MSE):** output should match input.
- **codebook loss:** move the codebook vectors toward the encoder outputs.
  `= MSE(z_q, stop_grad(z_e))`
- **commitment loss** (× beta, default 0.25): keep the encoder output from drifting
  away from the codebook. `= MSE(z_e, stop_grad(z_q))`

`vq_loss = codebook_loss + beta * commitment_loss; total = recon + vq_loss`.

**Straight-through estimator (the crucial trick):** picking the nearest code with
argmin is non-differentiable (no gradient). Fix:

```
z_q = z_e + (z_q - z_e).detach()
```

Forward value is still `z_q`, but gradients flow straight through to `z_e` as if it
were the identity. This lets the encoder train despite the hard, discrete lookup.

**Codebook collapse:** a failure mode where the model only ever uses a few codes;
the usage histogram shows a few tall spikes and many empty bins.

**Why random code samples look bad:** real images have spatial structure between
patches; picking codes independently at random ignores that, giving incoherent
images. To sample properly you need a SECOND model (a "prior", e.g. a
Transformer) that learns which code arrangements are realistic.

**MIM concepts:**

- **Patch mask:** split the image into patches, randomly mark some as "masked".
- **Masked loss:** compute MSE ONLY on the masked pixels (the model is graded only
  on what it had to guess).
- **Mask ratio:** higher ratio = less context = harder = blurrier reconstructions
  (model defaults to a mean/average guess when uncertain).

> **Libraries new here:** `torch.nn.Embedding` as a codebook, `torch.argmin`, `torch.bincount`
> (usage histogram), `repeat_interleave` (expand patch mask to pixels), `torch.randperm`.

## 3. The core building blocks (what the code does and why)

### The codebook — a learnable dictionary of vectors

The discrete latent is stored in an `nn.Embedding`, which is simply a lookup table
of `K` learnable vectors (the "codes"):

```python
self.codebook = nn.Embedding(num_codes, embedding_dim)   # K vectors of size D
```

Given an integer index, `self.codebook(index)` returns that row's vector. Training
adjusts these vectors like any other weights.

### Vector quantization — snapping to the nearest code

This is the heart of the exercise. The encoder produces a continuous map `z_e` of
shape `(B, D, 7, 7)`; quantization replaces each spatial location with its nearest
codebook vector:

```python
flat = z_e.permute(0,2,3,1).reshape(-1, D)          # (N, D): one row per location
dist = torch.cdist(flat, self.codebook.weight)       # distance to every code
indices = dist.argmin(dim=1)                          # nearest code per location
z_q = self.codebook(indices).view_as(z_e)            # look up -> quantized map
```

- **`torch.cdist`** computes the (Euclidean) distance from each encoder vector to
  every codebook vector at once.
- **`argmin`** picks the index of the closest code — this is the discrete "token".
- looking those indices up in the codebook gives `z_q`, the quantized latent.

### The two VQ losses and the straight-through estimator

```python
codebook_loss   = F.mse_loss(z_q, z_e.detach())      # move codes toward encoder
commitment_loss = F.mse_loss(z_e, z_q.detach())      # keep encoder near codes
vq_loss = codebook_loss + beta * commitment_loss     # beta ≈ 0.25

z_q = z_e + (z_q - z_e).detach()                      # STRAIGHT-THROUGH trick
```

- **`.detach()`** cuts the gradient: `z_e.detach()` means "treat `z_e` as a constant
  here". So the codebook loss only updates the *codes*, and the commitment loss
  only updates the *encoder* — each term trains one side toward the other.
- The **straight-through estimator** is the crucial line. `argmin` is not
  differentiable, so no gradient could reach the encoder. By writing
  `z_q = z_e + (z_q - z_e).detach()`, the *value* passed forward is still `z_q`, but
  the `(z_q - z_e)` part is detached, so during backprop the gradient flows
  straight through to `z_e` as if quantization were the identity.

### Putting it together and detecting collapse

The full `VQVAE.forward` is Encoder → VectorQuantizer → Decoder, and the total loss
is `F.mse_loss(recon, images) + vq_loss`. To inspect codebook usage:

```python
counts = torch.bincount(indices, minlength=num_codes)   # how often each code used
```

**`torch.bincount`** tallies how many times each index was chosen. A healthy
codebook spreads usage; **codebook collapse** shows up as a few huge bars and many
zeros. Decoding *random* index grids (instead of encoder-produced ones) gives
incoherent images, because real images need spatially-consistent code arrangements
— which a separate "prior" model would have to learn.

### Masked Image Modeling — building and using a mask

```python
n_masked = int(num_patches * mask_ratio)
masked = torch.randperm(num_patches)[:n_masked]         # randomly choose patches
patch_mask[masked] = 1
pixel_mask = patch_mask.repeat_interleave(ph, 0).repeat_interleave(pw, 1)
image[pixel_mask == 1] = mask_value                     # blank out masked pixels
```

- **`torch.randperm`** shuffles the patch indices so a random subset gets masked.
- **`repeat_interleave`** expands the small per-patch mask up to full pixel
  resolution (each patch becomes a `ph×pw` block of pixels).

**The masked loss** grades the model only on what it had to guess:

```python
loss = F.mse_loss(recon[pixel_mask == 1], target[pixel_mask == 1])
```

By indexing with `pixel_mask == 1`, the MSE is computed **only over masked pixels**.
A higher mask ratio leaves less surrounding context, so the network has to guess
more and reconstructions get blurrier (it falls back to an average guess).

## 4. How it fits the big picture

- VQ-VAE = the Ex2/Ex3 autoencoder with a DISCRETE latent. It sits alongside the
  VAE (continuous latent, Ex3) and contrasts with GANs (Ex5) and flows (Ex6/7).
- The straight-through estimator is a general trick for training through
  non-differentiable operations — a likely exam favorite.
- "You need a second model to sample the latent prior" (Task 1.9) is the key
  insight behind today's token-based image generators.
- MIM is self-supervised learning for vision; conceptually it's the masking idea
  you also see in Ex7 (storing/recovering missing information) and is the modern
  alternative to GAN/VAE pretraining.

> **Exam takeaway:** Draw the VQ-VAE pipeline, write all three loss terms, and explain
> the straight-through estimator and codebook collapse. For MIM, explain masked-only
> loss and the effect of mask ratio.
