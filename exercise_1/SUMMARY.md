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

## 3. Core code blocks (the TODOs)

### Task 1 — Basic image manipulation (pure NumPy slicing)

- **1.1** Load `Wombat.jpg` → `np.array`; shape is `(H, W, 3)`.
- **1.2** Explain shape (height, width, color channels) and show it.
- **1.3** Center crop via slicing `img[y0:y1, x0:x1]`.
- **1.4** Permute RGB channels: `img[:,:,perm]` with a random permutation.
- **1.5** Grayscale via weighted sum: `np.dot(img, [0.299,0.587,0.114])`.
- **1.6** Paste color crop back into the (3-channel) gray image.
- **1.7** Resize by strided slicing `img[::2, ::4]` (1/2 height, 1/4 width).

### Task 2 — Convolution by hand (NO built-in conv allowed)

- **2.1** `conv2d_gray(image, kernel)`: double loop over pixels; at each pixel take the
  surrounding patch, multiply by kernel, sum → output pixel. Output is
  smaller (no padding): `(H-n+1, W-n+1)`.
- **2.2** `gaussian_kernel(n, sigma)`: build an `(n,n)` grid with meshgrid, apply the
  Gaussian formula `exp(-(x^2+y^2)/(2 sigma^2))`, then **normalize** (divide by
  sum so brightness is preserved). Apply it → blurred image.
- **2.3** `conv2d_multi`: same idea but the patch and kernel have a channel dimension;
  sum over space AND channels into one output channel.
- **2.4** 1×1×3 kernel = the grayscale weights → proves grayscale IS a convolution.
- **2.5** Written: limits of manual filters vs. learned filters.

### Task 3 — Looking inside a real CNN (AlexNet)

- **3.1** Load pretrained AlexNet; first conv layer weight shape `(64, 3, 11, 11)` =
  64 filters, 3 input channels (RGB), 11×11 each.
- **3.2** Visualize all 64 first-layer filters in an 8×8 grid (normalize each for
  display).
- **3.3** Written: filters show edges, color contrasts, intensity gradients — the
  primitive building blocks for recognizing objects.
- **3.4–3.7 Saliency map:**
  - preprocess image with `weights.transforms()`; `unsqueeze(0)` to add batch.
  - forward pass → `argmax` = predicted class.
  - `input_batch.requires_grad_()`; `zero_grad()`; `score.backward()` → gradient.
  - `saliency` = max over channels of `|gradient|`; visualize next to the image.
  - Written: high saliency = pixels the model relied on most.

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
