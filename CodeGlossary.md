
## Convolution
Slide a small weight grid (the **kernel**) over the image. The core
loop, with no padding, produces an output of size `(H-n+1, W-n+1)`:

```python
def conv2d_gray(image, kernel):
    n = kernel.shape[0]
    out = np.zeros((H - n + 1, W - n + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = image[i:i+n, j:j+n]      # the n×n window under the kernel
            out[i, j] = np.sum(patch * kernel)  # multiply & sum -> one pixel
    return out
```

**Saliency map**

```python
#batch = weights.transforms()(img).unsqueeze(0)  # preprocess + add batch dim
batch.requires_grad_()                           # ask torch to track gradients here
#score = model(batch)                             # forward pass -> class scores
#pred = score.argmax()                            # the predicted class index
#model.zero_grad()
score[0, pred].backward()                        # gradient of that score w.r.t. input
#saliency = batch.grad.abs().max(dim=1)[0]        # strongest gradient across channels
```

## Building a Model in PyTorch
Every model is a class inheriting from `nn.Module` with two parts: `__init__`
(declare the layers) and `forward` (describe how data flows through them).

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            [
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                ...
            ])   # layers that shrink the image
        self.decoder = nn.Sequential(...)   # layers that rebuild it

    def forward(self, x):
        z = self.encoder(x)                 # x -> small latent code z
        return self.decoder(z)              # z -> reconstruction
```

`nn.Sequential` chains layers so the output of one is the input of the next. The
**latent code** `z` is whatever comes out of the encoder — the narrow bottleneck.

## Training Loop
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
## Reparameterization trick for VAEs 
You cannot backpropagate through random sampling. Trick:
```
z = mean + exp(0.5*log_var) * eps,   eps ~ N(0, I)
```
The randomness lives in `eps` (no gradient needed), while `mean` and `log_var` get
gradients. This is THE key idea that makes VAEs trainable.

## Straight-through estimator
The argmin operation used in VQ-VAE quantization is non-differentiable, so gradients cannot flow back to the encoder. The straight-through estimator fixes this with one line:
```
z_q = z_e + (z_q - z_e).detach()
```
The forward pass sees `z_q` (correct, discrete), but the backward pass ignores the detached `(z_q - z_e)` term and passes gradients straight through to `z_e` as if quantization were the identity. This lets the encoder train despite the hard discrete lookup.

## GAN Formulas

**Minimax objective:**
$$V(D,G) = \mathbb{E}_x[\log D(x)] + \mathbb{E}_z[\log(1 - D(G(z)))]$$

**Optimal discriminator** (maximize over D, fix G):
$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}$$
*Derived from:* $\max_y\ a\log y + b\log(1-y) \Rightarrow y = \frac{a}{a+b}$

**Generator optimum** (substitute $D^*$):
$$V(D^*, G) = -\log 4 + 2 \cdot \text{JSD}(p_{data} \| p_G)$$
Global min when $p_G = p_{data}$: $D^*(x) = \frac{1}{2}$, $V = -\log 4$

**Saturating vs. non-saturating generator loss:**

| | Formula | Problem |
|---|---|---|
| Saturating | $\log(1 - D(G(z)))$ | near-zero gradient when $D(G(z)) \approx 0$ |
| **Non-saturating** | $-\log D(G(z))$ | large gradient when $D(G(z)) \approx 0$ ✓ |

**R1 gradient penalty:**
$$\frac{\gamma}{2} \|\nabla_x D(x)\|^2$$

**Label smoothing:** real target $= 0.9$ instead of $1.0$

**Spherical interpolation (SLERP):**
$$z(t) = \frac{\sin((1-t)\,\omega)\,z_1 + \sin(t\,\omega)\,z_2}{\sin(\omega)}$$


