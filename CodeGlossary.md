
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