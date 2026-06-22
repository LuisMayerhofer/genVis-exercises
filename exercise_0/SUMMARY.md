# Exercise 0 — Very Short PyTorch Intro

> Course: Generative AI and Visual Synthesis (SoSe 2026)

## 1. What this exercise is about

This is the warm-up. It teaches the single data structure that **every** later
exercise is built on: the PyTorch **tensor**. A tensor is just a multi-dimensional
array of numbers (like a NumPy array) that can additionally:

- live on a GPU for fast computation, and
- automatically track gradients for training neural networks.

If you understand tensors and their shapes, the rest of the course becomes
"plug layers together and watch shapes flow through them". Almost every bug you
will hit in later exercises is a **shape bug**, so this sheet is the foundation.

## 2. Library / Module Glossary (used throughout the WHOLE course)

> Read this once — later summaries assume you know these.

| Library | What it is |
| --- | --- |
| `torch` | The core deep-learning library. Provides tensors, math ops, autograd (automatic differentiation), and GPU support. |
| `torch.nn` (`nn`) | Building blocks of neural networks: layers (Linear, Conv2d), activations (ReLU, Tanh), containers (Sequential, ModuleList), and loss functions (MSELoss, CrossEntropyLoss). |
| `torch.nn.functional` (`F`) | The same operations as `nn` but as plain functions (e.g. `F.mse_loss`, `F.cross_entropy`, `F.interpolate`, `F.one_hot`). |
| `torch.optim` | Optimizers that update weights from gradients (mostly Adam). |
| `torch.utils.data` | `DataLoader` = feeds the model data in shuffled mini-batches. |
| `torchvision` | Vision add-on: ready datasets (MNIST, FashionMNIST), image transforms, pretrained models (AlexNet), and `make_grid`. |
| `numpy` (`np`) | CPU array math. Used for non-learning number crunching. |
| `matplotlib` (`plt`) | Plotting (images, loss curves, scatter plots). |
| `PIL` (`Image`) | Loading/saving image files. |
| `sklearn` | Classic ML toolbox; here used for toy datasets (`make_moons`, `load_digits`), PCA, t-SNE, `accuracy_score`, `confusion_matrix`. |

**Key vocabulary:**

- **Batch (B)** = how many samples processed at once.
- **Channels (C)** = e.g. 1 for grayscale, 3 for RGB, or "feature maps" inside a net.
- **H, W** = image height and width.
- **Shape** = the tuple describing a tensor, e.g. `(B, C, H, W)`.

## 3. The core tensor toolkit (what every operation does)

These are the fundamental tensor operations you will use in every exercise.
Learn what each one does and *why* you reach for it.

### Creating tensors

```python
t = torch.tensor([[1, 2, 3], [4, 5, 6]])   # build from a Python list
o = torch.ones(2, 3)                        # a 2×3 tensor full of ones
t.shape                                     # -> torch.Size([2, 3])
```

A tensor is created either from existing data (`torch.tensor`) or from a "shape
recipe" (`torch.ones`, `torch.zeros`, `torch.randn` for random normal values).
**`.shape` is the most important attribute** — it tells you the size along every
dimension. Always print it when confused; most bugs are shape mismatches.

### Element-wise math and broadcasting

```python
a + b      # adds matching positions
a * b      # multiplies matching positions (NOT matrix multiply)
a * 2      # multiplies every element by 2
```

Arithmetic happens **element by element**. If shapes don't match exactly, PyTorch
tries to **broadcast**: it stretches a smaller dimension (size 1) to match the
larger one. E.g. a `(3,1)` tensor and a `(1,4)` tensor combine into `(3,4)`. This
is how you add a single bias value to a whole image without a loop.

### Moving between NumPy and PyTorch

```python
t = torch.from_numpy(arr)   # NumPy array -> tensor (shares memory!)
arr = t.numpy()             # tensor -> NumPy array
```

NumPy is the universal CPU array format used by data loaders and plotting; PyTorch
adds gradients and GPU support. You convert constantly: load/prepare data in NumPy,
compute in torch, then convert back to NumPy to plot with matplotlib.

### Reshape — same numbers, new layout

```python
t = torch.arange(12)        # a flat vector of 12 numbers: shape (12,)
t.reshape(3, 4)             # rearrange into a 3×4 grid
t.reshape(3, 4).reshape(12) # and back to flat
```

`reshape` (or `view`) keeps **the same numbers in the same order** but reinterprets
the shape. The total count must stay equal (3·4 = 12). You use it to **flatten** an
image into a vector before a fully-connected layer, and to unflatten afterwards.

### Permute — reorder the axes

```python
img.permute(1, 2, 0)   # (C, H, W) -> (H, W, C)
```

`permute` **swaps dimensions** (the arguments are the new order of the old axis
indices). Unlike reshape, it changes *which axis means what*, not the grouping.
The classic use: PyTorch stores images as `(C, H, W)`, but matplotlib's `imshow`
expects `(H, W, C)` — so you permute every time you want to display an image.

### Squeeze / unsqueeze — remove or add a size-1 dimension

```python
x = img.unsqueeze(0)   # (C, H, W) -> (1, C, H, W)   add a batch dim of size 1
y = x.squeeze(0)       # (1, C, H, W) -> (C, H, W)    remove it again
```

Models always expect a **batch** dimension first, even for a single image. So
before feeding one image to a network you `unsqueeze(0)` to make a batch of one;
afterwards you `squeeze` it away. `squeeze()` with no argument removes *all*
size-1 dimensions.

## 4. How it fits the big picture

This sheet has no "model" yet — it is pure groundwork. The skills here recur in
literally every later exercise:

- **reshape/view**: flatten images for fully-connected layers (Ex2, Ex7).
- **permute**: convert between `(C,H,W)` and `(H,W,C)` for plotting (Ex1, Ex4, Ex5).
- **unsqueeze**: add the batch dimension before feeding a single image to a model
  (Ex1 saliency map).
- **numpy ↔ torch**: prepare data and post-process results (everywhere).

> **Exam takeaway:** Be fluent in reading a shape `(B, C, H, W)` and predicting how it
> changes after reshape/permute/squeeze. This single skill unlocks the whole
> course.
