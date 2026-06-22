# Exercise 2 — Autoencoders on Fashion-MNIST

> Course: Generative AI and Visual Synthesis (SoSe 2026)

## 1. What this exercise is about

The first real generative-style models: **autoencoders**. An autoencoder squeezes an
image down into a tiny "latent code" (the bottleneck) and then tries to rebuild
the original image from that code. If it can reconstruct well, the small code
must capture the essential content. You build three flavors and compare them:

1. Fully-connected (FC) autoencoder
2. Denoising autoencoder (same FC model, trained to clean up noise)
3. Convolutional autoencoder

Dataset: Fashion-MNIST = 28×28 grayscale images of clothing (10 classes).

## 2. Key concepts / theory

**Encoder → bottleneck (latent code) → decoder → reconstruction.**

- Encoder shrinks 784 pixels down to e.g. 8 numbers.
- Decoder expands 8 numbers back to 784 pixels.
- Forcing data through the narrow bottleneck makes the network keep only the
  important structure (compression / representation learning).

**Loss:** Mean Squared Error (MSE) between input and reconstruction. Lower = better.

**Why Tanh at the end:** images are normalized to `[-1, 1]`, so the final activation
must output `[-1, 1]` (Tanh) to match.

**Denoising autoencoder:** feed a NOISY image in, but compare the output to the CLEAN
image. The network must learn to remove noise → learns more robust features.

**Convolutional vs fully-connected:** conv layers respect spatial structure (nearby
pixels relate), so the conv autoencoder reconstructs better at fewer parameters.
Observed losses in this submission: FC ~59.99, denoising ~59.06, conv ~40.61
(conv clearly best).

> **Libraries new here:** `torch.nn` layers (Linear, Conv2d, ConvTranspose2d, Flatten,
> Unflatten, ReLU, Tanh, Softmax), `torch.optim.Adam`, DataLoader + datasets.

## 3. Core code blocks (the TODOs)

### Task 1 — Fully-connected autoencoder (class `Autoencoder`)

- `encoder` = Sequential of Linear+ReLU shrinking 784 → 128 → 64 → 32 → 16 → 8.
- `decoder` = Linear+ReLU expanding 8 → ... → 784, final Tanh.
- `forward`: x → encoder → latent → decoder → reconstruction.

Train with `run_experiment(...)` (given), saves best weights to `.pth`.

### Task 2 — Denoising autoencoder

- SAME `Autoencoder` class, but run with `noisy=True` so noisy inputs are fed and
  the clean image is the target (the `prepare_batch` helper adds the noise).
- **2.1** Written: compare denoising vs plain (slightly lower loss; visibly removes
  noise, producing a blurred clean image).

### Task 3 — Convolutional autoencoder (class `Conv_Autoencoder`)

- `encoder`: Conv2d(1→4,k5)+ReLU, Conv2d(4→8,k5)+ReLU, Flatten,
  Linear(3200→10), Softmax (a 10-dim latent code).
- `decoder`: Linear(10→400)+ReLU, Linear(400→4000)+ReLU,
  Unflatten to (10,20,20), ConvTranspose2d(10→10,k5)+ReLU,
  ConvTranspose2d(10→1,k5)+Tanh.
- `forward` reshapes flat input to `(B,1,28,28)` and flattens output back.

> **Key new layer:** `ConvTranspose2d` = "deconvolution" / learnable **upsampling** — the
> decoder's way of growing a small feature map back into a full image.

- **3.4** Written: conv beats FC on both loss and visual detail.

## 4. How it fits the big picture

- This is the encoder/decoder **template** reused by almost every later model:
  - Ex3 VAE = autoencoder + probabilistic latent (mean/variance + KL).
  - Ex4 VQ-VAE = autoencoder + DISCRETE codebook latent.
  - Ex7 Task2 = trains an INN on top of an autoencoder's latent code.
- `ConvTranspose2d` (upsampling) reappears in the GAN generator (Ex5).
- The autoencoder CANNOT generate truly new images by itself — its latent space
  has no known distribution to sample from. Fixing that is exactly what the VAE
  (Ex3) and GAN (Ex5) do. So Ex2 sets up the central question of the course:
  *"How do we make the latent space something we can SAMPLE from?"*

> **Exam takeaway:** Explain encoder/bottleneck/decoder, why MSE + Tanh, what a
> denoising AE adds, and why a convolutional AE beats a fully-connected one. Know
> that a plain AE compresses but cannot generate.
