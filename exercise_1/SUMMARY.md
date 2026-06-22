# Exercise 1 — Images, Convolution, and CNN Interpretation

> Course: Generative AI and Visual Synthesis (SoSe 2026)

## 1. What this exercise is about

Three big ideas, building on the tensor basics from Ex0:

- **(A)** An image is just a tensor of numbers, and you can manipulate it with array
  slicing and indexing.
- **(B) Convolution** — the single most important operation in computer vision. You
  implement it by hand so you understand what a "filter/kernel" really does.
- **(C)** Convolutional Neural Networks (CNNs) **learn** their filters from data, and you
  can look inside a trained network to see what it learned (filter
  visualization) and what it "looks at" when deciding (saliency maps).

This is the bridge from "raw pixels" to "neural networks that understand images".

## 2. Key concepts / theory

**Convolution:** slide a small grid of weights (the **kernel**, e.g. 3×3) over the image.
At each position, multiply overlapping pixels by the kernel weights and sum them
into one output pixel. Different kernels do different things:

- **Gaussian kernel** → blurs (averages neighbours, weighted by distance).
- **1×1 kernel** → no neighbours, just mixes the color channels (here:
  RGB → grayscale via weights 0.299, 0.587, 0.114).

Two knobs of a Gaussian: `n` = kernel size (how many neighbours), `sigma` = spread
(how strong the blur).

**Why learn filters instead of designing them:** hand-designed filters get
impossibly hard for complex patterns. A CNN learns thousands of filters
automatically from data, discovering edge/color/texture detectors no human would
hand-code (Task 2.5 discussion).

**Saliency map:** which input pixels most influenced the prediction? Take the
gradient of the predicted class score w.r.t. the input image; large `|gradient|`
= high influence. This is "explainable AI".

> **Libraries new here:** `torchvision.models` (pretrained AlexNet + its weights and
> preprocessing transforms), `PIL.Image` (load the photo).

## 3. The core techniques (what the code does and why)

### Images as arrays — slicing and indexing

An image loaded with PIL and converted to a NumPy array has shape `(H, W, 3)`:
height, width, and 3 color channels (R, G, B). Every pixel is a number (0–255 or
0–1). Because it is just an array, you manipulate it with **slicing**:

```python
img[y0:y1, x0:x1]          # CROP: keep only rows y0..y1 and columns x0..x1
img[:, :, perm]            # REORDER CHANNELS: perm is e.g. [2,0,1] -> swaps RGB
img[::2, ::4]              # RESIZE by skipping: every 2nd row, every 4th column
```

The slice `start:stop:step` selects a range; a bare `:` means "all of this axis".
A `step` of 2 throws away every other pixel — a crude but instant downscale.

**Grayscale as a weighted channel sum:**

```python
gray = np.dot(img, [0.299, 0.587, 0.114])   # collapse 3 channels into 1
```

`np.dot` here multiplies each channel by its weight and sums them. Green is
weighted most because human eyes are most sensitive to it. The result has shape
`(H, W)` — the color dimension is gone.

### Convolution by hand

Convolution slides a small weight grid (the **kernel**) over the image. The core
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

The key line is `np.sum(patch * kernel)`: element-wise multiply the window by the
kernel, then add everything up into a single output pixel. That single operation,
repeated everywhere, *is* convolution.

**Building a Gaussian (blur) kernel:**

```python
ax = np.arange(n) - n // 2                  # coordinates centered on 0
xx, yy = np.meshgrid(ax, ax)                # 2D grid of x and y offsets
kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
kernel /= kernel.sum()                      # NORMALIZE so brightness is preserved
```

`np.meshgrid` turns two 1-D coordinate arrays into the full 2-D grid of (x, y)
positions. The Gaussian formula gives the center the highest weight, neighbours
less. **Normalizing** (dividing by the sum) is essential — otherwise the image
would get brighter or darker. Larger `sigma` = wider, stronger blur.

**Multi-channel convolution** adds a channel dimension to both the patch and the
kernel, and sums over space *and* channels into one output value. A special case:
a **1×1×3 kernel** has no spatial neighbours — it only mixes the 3 color channels.
Using the grayscale weights `[0.299, 0.587, 0.114]` as a 1×1×3 kernel reproduces
the grayscale conversion, proving grayscale *is* a convolution.

### Looking inside a pretrained CNN (AlexNet)

```python
from torchvision.models import alexnet, AlexNet_Weights
weights = AlexNet_Weights.DEFAULT
model = alexnet(weights=weights)            # download a network trained on ImageNet
```

`torchvision.models` gives you famous networks **already trained**, so you can
inspect what they learned without training anything. The first convolution layer's
weights have shape `(64, 3, 11, 11)` = 64 separate filters, each looking at 3 RGB
channels with an 11×11 window. Plotting these 64 filters as small images reveals
edge detectors, color-contrast detectors, and gradients — the primitive features a
network builds on.

**Saliency map — what the network "looks at":**

```python
batch = weights.transforms()(img).unsqueeze(0)  # preprocess + add batch dim
batch.requires_grad_()                           # ask torch to track gradients here
score = model(batch)                             # forward pass -> class scores
pred = score.argmax()                            # the predicted class index
model.zero_grad()
score[0, pred].backward()                        # gradient of that score w.r.t. input
saliency = batch.grad.abs().max(dim=1)[0]        # strongest gradient across channels
```

The new idea is `requires_grad_()` + `backward()`: instead of training weights, we
compute the gradient of the predicted class score **with respect to the input
pixels**. `weights.transforms()` applies the exact resizing/normalization the
network was trained with. Pixels with a large `|gradient|` are the ones that would
most change the prediction if nudged — i.e. the pixels the model relied on. Taking
the max over the channel dimension collapses RGB into one saliency heatmap.

## 4. How it fits the big picture

- Uses Ex0 shape skills constantly (permute for plotting, unsqueeze for batch).
- The convolution you build by hand is the SAME operation done by `nn.Conv2d` in
  every later model (Ex2 conv autoencoder, Ex3 VAE, Ex4 VQ-VAE, Ex5 GAN).
- "Filters are learned from data" is the seed of generative modeling: the whole
  rest of the course is about networks learning representations of images.
- Saliency/gradient-w.r.t-input reappears conceptually in Ex5 (R1 gradient
  penalty) and Ex7 (gradients/likelihood w.r.t. data).

> **Exam takeaway:** Be able to (a) explain convolution and the role of kernel size `n`
> and `sigma`, (b) state why a 1×1 conv = channel mixing, (c) describe how a saliency
> map is computed (gradient of class score w.r.t. input pixels).
