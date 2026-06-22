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

## 3. The core building blocks (what the code does and why)

### How you define a network in PyTorch

Every model is a class inheriting from `nn.Module` with two parts: `__init__`
(declare the layers) and `forward` (describe how data flows through them).

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(...)   # layers that shrink the image
        self.decoder = nn.Sequential(...)   # layers that rebuild it
    def forward(self, x):
        z = self.encoder(x)                 # x -> small latent code z
        return self.decoder(z)              # z -> reconstruction
```

`nn.Sequential` chains layers so the output of one is the input of the next. The
**latent code** `z` is whatever comes out of the encoder — the narrow bottleneck.

### The fully-connected (FC) layers

```python
nn.Linear(784, 128)   # a fully-connected layer: 784 inputs -> 128 outputs
nn.ReLU()             # activation: replaces negatives with 0 (adds non-linearity)
nn.Tanh()             # squashes outputs into [-1, 1]
```

- **`nn.Linear(in, out)`** is a matrix multiply + bias; it connects every input to
  every output. The encoder stacks these to shrink 784 → 128 → 64 → … → 8, the
  decoder mirrors it to grow 8 → … → 784.
- **`nn.ReLU`** between layers lets the network learn non-linear functions (without
  it, stacking Linear layers would collapse into one big Linear layer).
- **`nn.Tanh`** is used as the *final* layer because the images were normalized to
  `[-1, 1]`, and Tanh's output range matches exactly.

### Training loop ingredients

```python
loss_fn = nn.MSELoss()                          # mean squared error
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for images, _ in dataloader:                    # DataLoader yields mini-batches
    recon = model(images)                       # forward pass
    loss = loss_fn(recon, images)               # how wrong is the reconstruction?
    optimizer.zero_grad()                        # clear old gradients
    loss.backward()                              # compute new gradients
    optimizer.step()                             # nudge weights to reduce loss
```

These four lines — `zero_grad → forward/loss → backward → step` — are the **universal
PyTorch training rhythm** you will see in every later exercise. `MSELoss` measures
the average squared pixel difference; `Adam` is the optimizer that updates weights.
The `DataLoader` feeds images in shuffled batches.

### Denoising variant — same model, different inputs

A denoising autoencoder uses the **identical** network, but you corrupt the input
while keeping the *clean* image as the target:

```python
noisy = images + noise_level * torch.randn_like(images)   # add random noise
recon = model(noisy)
loss = loss_fn(recon, images)        # compare to the CLEAN image, not the noisy one
```

Because the target is clean, the network is forced to *remove* noise rather than
copy it — which teaches it more robust features.

### The convolutional autoencoder and the key new layer

Instead of `Linear`, the conv version uses layers that respect 2-D image structure:

```python
nn.Conv2d(1, 4, kernel_size=5)        # encoder: slide learnable filters over image
nn.Flatten()                          # (B, C, H, W) -> (B, C*H*W) flat vector
nn.Unflatten(1, (10, 20, 20))         # flat vector -> (B, 10, 20, 20) image shape
nn.ConvTranspose2d(10, 1, kernel_size=5)   # decoder: learnable UPSAMPLING
```

- **`nn.Conv2d(in_ch, out_ch, k)`** is the learnable version of the convolution you
  coded by hand in Ex1: it learns its filters from data. It *shrinks* spatial size
  and changes the channel count.
- **`nn.Flatten` / `nn.Unflatten`** convert between image shape and a flat vector so
  you can plug a `Linear` layer into the bottleneck.
- **`nn.ConvTranspose2d`** (the important new layer) is the inverse of `Conv2d`: a
  *learnable upsampling* ("deconvolution") that **grows** a small feature map back
  into a larger image. It is how decoders enlarge the bottleneck back to full size,
  and it reappears in the GAN generator (Ex5).

Because conv layers exploit the fact that nearby pixels are related, the conv
autoencoder reconstructs more sharply than the FC one at fewer parameters.

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
