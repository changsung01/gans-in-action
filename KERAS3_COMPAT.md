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

## General Checklist for Each Chapter

Before running any chapter notebook, scan for and fix:

- [ ] `from keras import backend as K` and any `K.*` calls → replace with `keras.ops.*`
- [ ] `from keras import metrics` and `metrics.*` loss/metric calls → use `keras.losses.*` or `keras.metrics.*` proper classes
- [ ] `Lambda(fn)` layers → replace with subclassed `keras.layers.Layer`
- [ ] `tf.*` ops applied to model inputs/outputs inside model-building code → replace with `keras.ops.*`
- [ ] Loss functions that capture KerasTensors as closures/defaults → refactor to subclassed model with `self.add_loss()`
- [ ] Add `import keras` to imports if not present
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
