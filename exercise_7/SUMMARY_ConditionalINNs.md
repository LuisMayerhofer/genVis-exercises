# Exercise 7 — Conditional INNs & Disentanglement

> Course: Generative AI and Visual Synthesis (SoSe 2026)

## 1. What this exercise is about

The capstone on invertible networks (INNs = the flow machinery from Ex6). Three
parts:

- **Task 1** — A formal warm-up: WHY do invertible models need an extra latent/noise
  variable? (Information that the forward map throws away.)
- **Task 2 — Disentanglement:** train an invertible map on TOP of an autoencoder's
  latent code so that separate, named coordinates control separate
  factors (digit identity, color, leftover "style"). Then EDIT one
  factor at a time.
- **Task 3 — A conditional normalizing flow** directly on the images, `p(x|y)`, used
  for three things: density estimation, likelihood-based CLASSIFICATION,
  and outlier (noise) detection.

Dataset: colorized 8×8 digits (a grayscale digit tinted a random RGB foreground
color) — chosen so there are two clear factors: which digit, and which color.

## 2. Key concepts / theory

**Why a latent/noise variable is needed (Task 1, the circle example):**

- The map `x -> y = sqrt(x1^2 + x2^2)` (radius) is MANY-TO-ONE: it discards the
  angle, so no deterministic inverse can recover `x`.
- Best constant guess minimizing `E||x - x_hat||^2` is the MEAN `E[x]` (bias-
  variance decomposition); for `x` uniform on the unit circle that mean is `(0,0)`
  and the minimal error is 1.
- Fix: store the lost info in a latent `z = atan2(x2, x1)` (the angle). Then
  `(y, z) <-> (x1, x2)` is exactly invertible (polar ↔ Cartesian).
- **Moral:** the "noise" variable in an INN stores exactly the information the
  condition does not — this is what makes inversion possible.

**Disentanglement via an INN on latents (Task 2):**

- First an autoencoder compresses images to a 16-D latent `z_ae`.
- Then an INVERTIBLE map `T` reorganizes `z_ae` into `u = [digit(10), color(3),
  residual(3)]`. Trained so the digit block predicts the label (cross-entropy),
  the color block predicts RGB (MSE), and the residual is kept small.
- Because `T` is invertible, `z_ae = T^{-1}(u)` EXACTLY, so you can EDIT a factor
  (swap in a new color or a new digit prototype) and invert+decode to get the
  edited image, with the other factors untouched.
- The RESIDUAL factor is essential: it stores style/nuisance info (stroke
  thickness, slant, noise) so the map stays a lossless bijection. Same role as
  the angle `z` in Task 1.

**Conditional flow on images (Task 3):**

- A conditional INN models `p(x | y)` via change of variables (as in Ex6), with
  the label fed into every coupling layer.
- NLL loss per sample: `-(log N(z;0,I) + log_det)`.
- **Likelihood classification:** score a test image under all 10 label conditions;
  predict the label giving the LOWEST NLL (highest likelihood). A generative
  model used as a classifier!
- **Anomaly detection:** real digits get low best-NLL; random color-noise images
  get much higher NLL → likelihood separates in-distribution from garbage.

> **Libraries new here:** sklearn PCA (2D latent projection), `accuracy_score` &
> `confusion_matrix` (evaluation), `F.one_hot`, `math.log/pi` (Gaussian log-density).

## 3. The core building blocks (what the code does and why)

### Task 1 — why an invertible map needs a noise variable (the math)

This part is a written derivation around the radius map `y = sqrt(x1² + x2²)`:

- The map is **many-to-one** (every point on a circle of radius `y` maps to the same
  `y`), so no function `x = f⁻¹(y)` can exist — the angle is lost.
- The best *constant* reconstruction minimizing `E‖x − x̂‖²` is the **mean** `E[x]`
  (the bias-variance decomposition: squared error is minimized at the mean). For
  points uniform on the unit circle that mean is the origin, and the minimal error
  is 1.
- The fix is to **store the lost information** in a latent `z = atan2(x2, x1)` (the
  angle). Then `(y, z) ↔ (x1, x2)` is exactly invertible — it is just polar ↔
  Cartesian coordinates. **`atan2`** recovers the angle from the two components.

This is the conceptual core of the whole sheet: an INN's noise/latent variable
carries exactly the information the condition throws away, which is what *allows*
inversion.

### Task 2 — disentangling an autoencoder's latent with an INN

An autoencoder first compresses each image to a 16-D latent `z_ae`. An invertible
map `T` then *reorganizes* those 16 numbers into **named blocks**:

```python
u = model(z_ae)                              # invertible reshuffle of the latent
digit    = u[:, 0:10]    # 10 numbers meant to encode the class
color    = u[:, 10:13]   # 3 numbers meant to encode the RGB color
residual = u[:, 13:16]   # 3 leftover numbers for everything else (style)
```

The training loss *forces* each block to mean what we want:

```python
loss = F.cross_entropy(digit, labels) \        # digit block must predict the class
     + F.mse_loss(color, rgb_targets) \        # color block must predict the RGB
     + 0.05 * (residual**2).mean()             # keep the residual small
```

- **`F.cross_entropy`** is the standard classification loss; it pushes the 10 "digit"
  numbers to behave like class logits.
- **`F.mse_loss`** ties the 3 "color" numbers to the true foreground RGB.
- The small penalty on the **residual** stops it from absorbing everything, while
  still letting it store leftover style. Crucially, the residual is *needed*: since
  `T` is invertible it cannot throw information away, so style/nuisance detail must
  live somewhere — exactly like the angle `z` in Task 1.

Because `T` is invertible, **editing is exact**:

```python
u_edited = u.clone()
u_edited[:, 10:13] = new_color        # change ONE factor, leave the rest
z_new = model.inverse(u_edited)       # exact inverse back to AE latent
img   = autoencoder.decode(z_new)     # decode the edited latent
```

Swapping the color block recolors the digit; swapping the digit block (using the
average learned logit vector per class as a "prototype", since a raw one-hot is too
weak) changes the digit — each without disturbing the other factors. Plotting a 2-D
**PCA** projection of `z_ae` before vs. the named factors after shows the structure
the INN imposed.

### Task 3 — a conditional flow directly on images

This reuses the RealNVP machinery from Ex6, but the label is fed into every
coupling layer so the model learns `p(x | y)`.

**The NLL loss** (per sample, then averaged):

```python
def flow_nll(z, log_det):
    log_pz = -0.5 * (z**2).sum(dim=1)        # log-density under N(0, I) (const dropped)
    return -(log_pz + log_det).mean()        # negative log-likelihood
```

**Classification by likelihood** — the headline trick. Score one image under *every*
possible label and pick the most likely:

```python
nlls = []
for d in range(10):
    cond = F.one_hot(torch.full((N,), d), 10).float()
    z, log_det = flow(images, cond)
    nlls.append(per_sample_nll(z, log_det))   # (N,) NLL assuming the digit is d
nlls = torch.stack(nlls, dim=1)               # (N, 10) matrix
pred = nlls.argmin(dim=1)                      # lowest NLL = highest likelihood
```

A generative model used as a classifier: the condition under which the image is
*most probable* is the predicted class. You evaluate with `accuracy_score` and a
`confusion_matrix` (both from sklearn).

**Anomaly detection** uses the same scores: take each image's *best* (minimum) NLL
over all conditions. Real digits achieve a low best-NLL; random color-noise images
score far higher. Plotting the two histograms shows likelihood cleanly separates
real data from garbage.

**Probing what `y` vs `z` control:**

```python
# fix the label, vary z  -> different styles of the SAME digit
imgs = [flow.inverse(torch.randn(1, D), cond_d) for _ in range(k)]
# fix z, swap the label across 0..9 -> same style, different digit
imgs = [flow.inverse(z_fixed, F.one_hot(torch.tensor([d]),10).float()) for d in range(10)]
```

Holding `z` and changing the label, then holding the label and changing `z`,
demonstrates the Task-1 lesson concretely: the **condition `y`** carries the digit
identity, while the **latent `z`** carries the leftover style/color.

## 4. How it fits the big picture

- Direct continuation of Ex6: reuses affine coupling layers, the swap/mask
  pattern, exact invertibility, and the NLL/change-of-variables loss.
- Ties together MANY course threads:
  - autoencoder latent (Ex2) reused as the thing the INN reorganizes;
  - conditioning on labels (Ex3 VAE, Ex5 GAN, Ex6 flow) here enables both
    controllable generation AND likelihood classification;
  - likelihood-as-anomaly-detector echoes the VAE anomaly detection (Ex3 Task3);
  - factor editing / interpolation echoes the GAN control study (Ex5 Task4).
- Big conceptual payoff: a single invertible model gives you generation, exact
  density, classification, and outlier detection — showing why invertibility and
  exact likelihood are powerful.

> **Exam takeaway:** Explain why an INN needs a latent variable (the circle example:
> many-to-one map, mean minimizes MSE, store the angle). Explain disentanglement
> via an invertible map on a latent with named factors + a residual, and why the
> residual preserves invertibility. Explain how a conditional flow does likelihood
> classification (argmin NLL over conditions) and noise detection (real << noise
> NLL).
