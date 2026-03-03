# Keras 3 Compatibility Guide

This project was originally written for **Keras 2 / TensorFlow 1.x–2.x**. Running it on modern environments with **Keras 3** (bundled with TF 2.16+) requires several fixes. This document captures every issue encountered and the correct fix, to be applied consistently across all chapters.

---

## Environment

- TensorFlow: 2.16+
- Keras: 3.x (standalone, `import keras`)
- Python: 3.12

---

## Issue 1: `keras.backend` math functions removed

**Error:**
```
AttributeError: module 'keras.backend' has no attribute 'random_normal'
AttributeError: module 'keras.backend' has no attribute 'exp'
AttributeError: module 'keras.backend' has no attribute 'sum'
# etc.
```

**Cause:** Keras 3 stripped most math functions from `keras.backend` (`K`). The `K.*` API is essentially gone.

**Fix:** Replace all `K.*` math calls with `keras.ops.*` equivalents. Add `import keras` to imports.

| Old (`K.*`) | New (`keras.ops.*`) |
|---|---|
| `K.random_normal(shape=..., mean=..., stddev=...)` | `keras.random.normal(shape=..., stddev=...)` |
| `K.shape(x)` | `keras.ops.shape(x)` |
| `K.exp(x)` | `keras.ops.exp(x)` |
| `K.sum(x, axis=...)` | `keras.ops.sum(x, axis=...)` |
| `K.square(x)` | `keras.ops.square(x)` |
| `K.mean(x)` | `keras.ops.mean(x)` |
| `K.log(x)` | `keras.ops.log(x)` |
| `K.abs(x)` | `keras.ops.abs(x)` |
| `K.sigmoid(x)` | `keras.ops.sigmoid(x)` |
| `K.tanh(x)` | `keras.ops.tanh(x)` |
| `K.dot(x, y)` | `keras.ops.matmul(x, y)` |
| `K.flatten(x)` | `keras.ops.reshape(x, (-1,))` |
| `K.ones_like(x)` | `keras.ops.ones_like(x)` |
| `K.zeros_like(x)` | `keras.ops.zeros_like(x)` |

Also replace:
| Old | New |
|---|---|
| `from keras import metrics` then `metrics.binary_crossentropy(...)` | `keras.losses.binary_crossentropy(...)` |
| `tf.random.normal(...)` on KerasTensors | `keras.random.normal(...)` |

---

## Issue 2: `tf.*` functions cannot be used on KerasTensors

**Error:**
```
ValueError: A KerasTensor cannot be used as input to a TensorFlow function.
A KerasTensor is a symbolic placeholder...
```

**Cause:** In Keras 3, KerasTensors (symbolic tensors used to build Functional models) cannot be passed to raw `tf.*` functions. Only `keras.ops.*` and `keras.layers.*` accept them.

**Fix:** Replace any `tf.*` math operations that touch Keras tensors with `keras.ops.*`. Use `keras.random.*` for random ops.

```python
# Wrong
epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
result = tf.exp(x)

# Correct
epsilon = keras.random.normal(shape=(keras.ops.shape(z_mean)[0], latent_dim))
result = keras.ops.exp(x)
```

---

## Issue 3: `Lambda` layers with stateful/complex functions

**Cause:** Keras 3 strongly discourages `Lambda` layers for anything beyond trivial element-wise ops. Lambda layers that call `keras.backend`, `tf.*`, or capture external state are problematic and hard to serialize.

**Fix:** Replace `Lambda` layers with a proper subclassed `keras.layers.Layer`.

```python
# Old — Keras 2
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# New — Keras 3
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = keras.ops.shape(z_mean)[0]
        epsilon = keras.random.normal(shape=(batch, latent_dim))
        return z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])
```

---

## Issue 4: Custom loss functions that capture KerasTensors

**Error:**
```
ValueError: A KerasTensor cannot be used as input to a TensorFlow function.
# or
NotImplementedError  (from Functional.add_loss)
```

**Cause:** A common Keras 2 VAE pattern captures encoder outputs (`z_mean`, `z_log_var`) as Python default arguments in a loss function, making them KerasTensors inside the loss. This breaks in Keras 3. Additionally, `Functional.add_loss()` is not implemented in Keras 3.

```python
# Old — Keras 2 (broken in Keras 3)
def vae_loss(x, x_decoded, z_log_var=z_log_var, z_mean=z_mean):
    ...
vae.compile(loss=vae_loss)
```

**Fix:** Use a **subclassed `keras.Model`** and call `self.add_loss()` inside `call()`, where the tensors are real (not symbolic).

```python
# New — Keras 3
class VAE(keras.Model):
    def __init__(self, encoder, decoder, original_dim, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.original_dim = original_dim

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        # KL loss added here, on real tensors
        kl_loss = -0.5 * keras.ops.sum(
            1 + z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var),
            axis=-1)
        self.add_loss(keras.ops.mean(kl_loss))
        return reconstruction

vae = VAE(encoder, decoder, original_dim, name="vae")

# Compile with only the reconstruction loss; KL is handled by add_loss
def xent_loss(x, x_decoded):
    return original_dim * keras.losses.binary_crossentropy(x, x_decoded)

vae.compile(optimizer='rmsprop', loss=xent_loss)
```

---

## Issue 5: `Functional.add_loss()` not implemented

**Error:**
```
NotImplementedError
# from keras/src/models/functional.py — add_loss raises NotImplementedError with "# Symbolic only. TODO"
```

**Cause:** Keras 3's Functional model does not support `add_loss()` called after model construction with a KerasTensor.

**Fix:** Same as Issue 4 — use a subclassed `keras.Model` and call `self.add_loss()` inside `call()`.

---

## Issue 6: `discriminator.trainable = False` no longer freezes weights at compile time (GAN training broken)

**Symptom:** GAN produces only noise. Training metrics show discriminator failing completely:
```
100  [D loss: 5.27, acc.: 25.38%] [G loss: 0.010]
6000 [D loss: 7.27, acc.: 25.24%] [G loss: 0.0001]
```
D accuracy stuck at ~25% (worse than random), D loss rising, G loss → 0.

**Cause:** The classic Keras 2 GAN trick relied on `trainable` being **baked into the compiled graph**:

```python
# Old trick — worked in Keras 2 / old tf.keras
discriminator.compile(...)          # compiled with trainable=True
discriminator.trainable = False     # set AFTER compile
gan.compile(...)                    # GAN compiled with discriminator frozen

# discriminator.train_on_batch() → used its own compiled graph (trainable=True) ✓
# gan.train_on_batch()           → used GAN's compiled graph (trainable=False) ✓
```

In **Keras 3**, `trainable` is evaluated **dynamically at every training step**, not at compile time. Setting `discriminator.trainable = False` freezes the discriminator globally — including when calling `discriminator.train_on_batch()` directly. The discriminator never learns, the generator trivially fools it, and training collapses.

**Fix:** Replace `compile()` + `train_on_batch()` with `tf.GradientTape`, passing explicit variable lists to each optimizer so each step only updates the intended model's weights.

```python
import tensorflow as tf

discriminator = build_discriminator(img_shape)
generator = build_generator(img_shape, z_dim)

d_optimizer = Adam()
g_optimizer = Adam()
cross_entropy = tf.keras.losses.BinaryCrossentropy()


def train_discriminator_step(real_imgs, batch_size):
    z = tf.random.normal((batch_size, z_dim))
    real_labels = tf.ones((batch_size, 1))
    fake_labels = tf.zeros((batch_size, 1))

    with tf.GradientTape() as tape:
        fake_imgs = generator(z, training=False)          # generator frozen
        real_output = discriminator(real_imgs, training=True)
        fake_output = discriminator(fake_imgs, training=True)
        d_loss = (cross_entropy(real_labels, real_output)
                  + cross_entropy(fake_labels, fake_output)) / 2

    # Only discriminator variables updated
    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    return d_loss


def train_generator_step(batch_size):
    z = tf.random.normal((batch_size, z_dim))
    real_labels = tf.ones((batch_size, 1))

    with tf.GradientTape() as tape:
        fake_imgs = generator(z, training=True)
        fake_output = discriminator(fake_imgs, training=False)  # discriminator frozen
        g_loss = cross_entropy(real_labels, fake_output)

    # Only generator variables updated
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return g_loss
```

**Chapters affected:** Any chapter that trains a GAN using the `discriminator.trainable = False` + `compile()` pattern (chapters 3, 4, 5, 6, 7, 8, 9).

---

## Issue 7: Removed sub-module import paths for layers

**Error:**
```
ImportError: cannot import name 'LeakyReLU' from 'keras.layers.advanced_activations'
ImportError: cannot import name 'Conv2D' from 'keras.layers.convolutional'
```

**Cause:** Keras 3 removed the internal sub-module structure. All layers are now imported directly from `keras.layers`.

**Fix:** Consolidate all layer imports into a single `from keras.layers import ...` statement.

| Old | New |
|---|---|
| `from keras.layers.advanced_activations import LeakyReLU` | `from keras.layers import LeakyReLU` |
| `from keras.layers.convolutional import Conv2D, Conv2DTranspose` | `from keras.layers import Conv2D, Conv2DTranspose` |
| `from keras.layers.normalization import BatchNormalization` | `from keras.layers import BatchNormalization` |

---

## Issue 8: `LeakyReLU(alpha=...)` parameter renamed

**Warning:**
```
UserWarning: Argument `alpha` is deprecated. Use `negative_slope` instead.
```

**Fix:** Replace `alpha` with `negative_slope` in all `LeakyReLU` instantiations.

```python
# Old
LeakyReLU(alpha=0.01)

# New
LeakyReLU(negative_slope=0.01)
```

---

## Issue 9: BatchNorm `training` flag mismatch in GAN generator step

**Symptom:** With BatchNormalization in the discriminator, D loss collapses to ~0, D accuracy locks at 100%, and G loss also collapses to ~0 — a paradoxical state impossible in a correctly functioning GAN.

**Cause:** The `training` flag controls two **independent** things:

| Aspect | Controlled by |
|---|---|
| BatchNorm behaviour (batch stats vs. running stats) | `training=True/False` passed to model call |
| Which weights get updated | Which variables are passed to `apply_gradients` |

In the GradientTape implementation, calling `discriminator(fake_imgs, training=False)` inside `train_generator_step` makes the discriminator's BatchNorm use **running statistics** (accumulated over previous batches), while the discriminator training step uses **batch statistics** (`training=True`). These differ significantly, especially early in training, causing the discriminator to behave inconsistently between its two contexts. This creates the paradox: D classifies correctly during its own step but appears fooled during the generator step.

The original `gan.train_on_batch()` always called everything in `training=True` mode, so this inconsistency never arose.

**Fix:** In `train_generator_step`, call the discriminator with `training=True`. The discriminator weights are still not updated — that is controlled by only passing `generator.trainable_variables` to `apply_gradients`, independent of the `training` flag.

```python
def train_generator_step(batch_size):
    z = tf.random.normal((batch_size, z_dim))
    real_labels = tf.ones((batch_size, 1))

    with tf.GradientTape() as tape:
        fake_imgs = generator(z, training=True)
        # training=True: BatchNorm uses batch statistics (consistent with D training step)
        # Discriminator weights NOT updated: only generator.trainable_variables passed below
        fake_output = discriminator(fake_imgs, training=True)
        g_loss = cross_entropy(real_labels, fake_output)

    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return float(g_loss)
```

**Note:** This only manifests in networks with BatchNormalization in the discriminator. Chapter 3 (no BatchNorm) was unaffected. Chapters 4+ all have BatchNorm in the discriminator and require this fix.

---

## Issue 10: Redundant `input_shape` in non-first Conv2D layers

**Warning:**
```
UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer.
When using Sequential models, prefer using an Input(shape) object as the first layer.
```

**Cause:** The original code passed `input_shape=img_shape` to every `Conv2D` layer in the discriminator, but only the first layer needs it. Keras 3 warns on all subsequent ones.

**Fix:** Only keep `input_shape` on the **first** layer of a Sequential model; remove it from all subsequent layers.


```python
# Old — input_shape on every Conv2D (wrong)
model.add(Conv2D(32,  kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
model.add(Conv2D(64,  kernel_size=3, strides=2, input_shape=img_shape, padding='same'))  # redundant
model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))  # redundant

# New — input_shape only on first layer
model.add(Conv2D(32,  kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
model.add(Conv2D(64,  kernel_size=3, strides=2, padding='same'))
model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
```

---

## General Checklist for Each Chapter

Before running any chapter notebook, scan for and fix:

- [ ] `from keras import backend as K` and any `K.*` calls → replace with `keras.ops.*`
- [ ] `from keras import metrics` and `metrics.*` loss/metric calls → use `keras.losses.*` or `keras.metrics.*` proper classes
- [ ] `from keras.layers.advanced_activations import ...` or `from keras.layers.convolutional import ...` → consolidate into `from keras.layers import ...` (Issue 7)
- [ ] `LeakyReLU(alpha=...)` → `LeakyReLU(negative_slope=...)` (Issue 8)
- [ ] `input_shape` on non-first layers in Sequential models → remove (Issue 9)
- [ ] `Lambda(fn)` layers → replace with subclassed `keras.layers.Layer`
- [ ] `tf.*` ops applied to model inputs/outputs inside model-building code → replace with `keras.ops.*`
- [ ] Loss functions that capture KerasTensors as closures/defaults → refactor to subclassed model with `self.add_loss()`
- [ ] GAN training using `discriminator.trainable = False` + `compile()` trick → replace with `tf.GradientTape` and per-model variable lists (Issue 6)
- [ ] In `train_generator_step`, call `discriminator(fake_imgs, training=True)` not `training=False` when discriminator has BatchNorm (Issue 9)
- [ ] Add `import keras` and `import tensorflow as tf` to imports if not present
- [ ] After any code changes: **restart the kernel and run all cells from the top**

---

## Chapter 2 — Applied Fixes Summary

| Cell | Change |
|---|---|
| imports | Added `import keras`; removed `from keras import metrics` |
| `sampling` function / `Lambda` | Replaced with `class Sampling(keras.layers.Layer)` |
| Encoder | Changed `Lambda(sampling)(...)` → `Sampling()(...)` |
| VAE model | Changed Functional model + closure loss → `class VAE(keras.Model)` with `self.add_loss()` in `call()` |
| `vae_loss` | Replaced with `xent_loss` (reconstruction only); KL moved into VAE `call()` |

---

## Chapter 3 — Applied Fixes Summary

| Cell | Change |
|---|---|
| imports | Added `import tensorflow as tf` |
| Build/compile cell | Removed `discriminator.compile()`, `discriminator.trainable = False`, `build_gan()`, `gan.compile()`. Added `d_optimizer`, `g_optimizer`, `cross_entropy` |
| Training loop | Replaced `discriminator.train_on_batch()` and `gan.train_on_batch()` with `train_discriminator_step()` and `train_generator_step()` using `tf.GradientTape` |
| Compatibility note (markdown) | Updated to explain the Keras 3 dynamic `trainable` breaking change and the GradientTape fix |

---

## Chapter 4 — Applied Fixes Summary

| Cell | Change |
|---|---|
| imports | Removed `from keras.layers.advanced_activations import LeakyReLU` and `from keras.layers.convolutional import Conv2D, Conv2DTranspose`; consolidated into single `from keras.layers import (...)`. Added `import tensorflow as tf` |
| Generator (`build_generator`) | Removed `input_dim=z_dim` from `Dense`; changed `LeakyReLU(alpha=0.01)` → `LeakyReLU(negative_slope=0.01)` |
| Discriminator (`build_discriminator`) | Removed redundant `input_shape=img_shape` from 2nd and 3rd `Conv2D` layers; changed `LeakyReLU(alpha=0.01)` → `LeakyReLU(negative_slope=0.01)` |
| Build/compile cell | Removed `discriminator.compile()`, `discriminator.trainable = False`, `build_gan()`, `gan.compile()`. Added `d_optimizer`, `g_optimizer`, `cross_entropy` |
| Training loop | Replaced `discriminator.train_on_batch()` and `gan.train_on_batch()` with `train_discriminator_step()` and `train_generator_step()` using `tf.GradientTape` |
| Training loop (generator step) | Fixed `discriminator(fake_imgs, training=False)` → `training=True` to avoid BatchNorm running-stats mismatch (Issue 9) |
| Compatibility note (markdown) | Updated to reference KERAS3_COMPAT.md Issue 6 |
