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

## 3. Core code blocks (the TODOs)

### Task 1 (written, 7P)

1.1 why no deterministic inverse; 1.2 prove `x_hat*=E[x]` via bias-variance and
that min error = 1; 1.3 define `z = atan2(x2,x1)`; 1.4 forward `F(x)=(y,z)` and
inverse `F^{-1}(y,z)=(x1,x2)` (polar coords); 1.5 `z` stores the angle.

### Task 2 — INN on autoencoder latents

- **2.1** `train_iin` TODO: `u = model(z)`; split into digit/color/residual; loss =
  `cross_entropy(digit, labels) + mse(color, rgb) + 0.05*mean(residual^2)`.
  Report digit-factor accuracy, color MSE, and inverse error (~0). Plot
  PCA-of-`z_ae` (before) vs digit-vs-color factors (after).
- **2.2** written: what the residual stores and why invertibility needs it.
- **2.3** `change_color_factor` / `change_digit_factor` TODO: hold the other factors
  fixed, swap in new color values (or per-class digit PROTOTYPES — the mean
  learned digit-logit vector, since raw one-hot is too weak), then
  `T^{-1}` → decode → edited image.

### Task 3 — Conditional flow on images

- **3.1** `flow_nll` TODO: mean of `-(log_pz + log_det)`; plug into `train_cinn`
  (forward, loss, optimizer step). Check test NLL and inverse error.
- **3.2** `nll_for_all_conditions` TODO: for each digit `d`, run the flow with that
  one-hot condition, collect per-sample NLL → `(N,10)` matrix; argmin = predicted
  digit. Report accuracy + confusion matrix.
- **3.3** TODO: best (min over conditions) NLL for real vs random-noise images;
  plot two histograms (real << noise).
- **3.4** TODO: `sample_images_for_digit` (fix label, vary `z` → within-class variety)
  and `transfer_z_to_all_digits` (fix `z` from one image, swap the label across
  0..9 → shows what `y` controls vs. what `z` carries: style/color).

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
