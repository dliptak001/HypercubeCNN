# HypercubeCNN — Implementation notes

**Audience:** contributors, power users, and anyone debugging training or performance.  
**Not** the first document for new users — start with [README.md](../README.md) and [CPP_SDK.md](CPP_SDK.md).

This page describes **how the core is built**, not how to call it. Public API contracts live in the SDK guide; when this file and a header disagree, **the header and `.cpp` win**.

| | |
|--|--|
| Library | `HypercubeCNNCore` (C++23, CMake ≥ 3.21) |
| Front door | `hcnn::HCNN` |
| Orchestrator | `hcnn::HCNNNetwork` (owned by `HCNN`) |
| Version (CMake) | 0.2.0 |

---

## 1. Module graph

```text
                    HCNN  (public, PIMPL)
                      │
                      ▼
                 HCNNNetwork
           ┌──────────┼──────────┐
           ▼          ▼          ▼
       HCNNConv   HCNNPool   HCNNReadout
           │          │
           └──── ThreadPool ────┘   (shared, non-reentrant)

Outside the graph (optional SDK headers):
  HCNNSpatialAug / HCNNSpatialEmbed   — image preprocess
  HCNNTrainHelpers                      — metrics, LR, checkpoints
```

| Class | Files | Owns |
|-------|-------|------|
| `HCNN` | `HCNN.h/cpp` | `unique_ptr<HCNNNetwork>`, epoch shuffle gather buffers |
| `HCNNNetwork` | `HCNNNetwork.h/cpp` | conv/pool stacks, readout, `ThreadPool`, train/infer scratch |
| `HCNNConv` | `HCNNConv.h/cpp` | kernel (`c_out×c_in×K`), bias, BN params, optimizer moments |
| `HCNNPool` | `HCNNPool.h/cpp` | no parameters (stateless reduce) |
| `HCNNReadout` | `HCNNReadout.h/cpp` | dense weights + bias + moments |
| `ThreadPool` | `ThreadPool.h` | worker threads; caller participates as thread 0 |

Enums (via `HCNN.h`): `Activation`, `OptimizerType`, `PoolType`, `TaskType`, `LossType`.

---

## 2. Convolution implementation (`HCNNConv`)

### 2.1 Math and layout

```text
out[co, v] = bias[co]
           + Σ_ci  w[co,ci,SELF] * in[ci, v]              // SELF = DIM
           + Σ_ci,k w[co,ci,k]   * in[ci, v ^ (1<<k)]    // k = 0 .. DIM-1

kernel layout:  [co * c_in * K + ci * K + k]
K = DIM + 1
activations:    data[c * N + v]   (channel-major)
```

Self is **never** implemented as `1 << DIM` (that shift is undefined for large DIM). It is a contiguous multiply-add over `in[v]`.

Constraints: `3 ≤ DIM ≤ 30` for both `HCNNNetwork` and `HCNNConv`
(`N = 2^DIM` fits in signed 32-bit int). BN backward requires the
`bn_save` buffer from the matching forward (`get_bn_save_size()` =
`3 * c_out`: inv_std, mean, var).

### 2.2 Forward path structure

Per output channel:

1. **Accumulate** self + neighbor weighted sums (+ bias).
2. Optional **batch norm** (if enabled for the layer).
3. **Activation** (`NONE` / `RELU` / `LEAKY_RELU` / `TANH`).

**BN is sample-local, not mini-batch-local:** mean/variance are over the **N vertices of the current sample** for that channel. Running mean/var (EMA, momentum 0.1, eps 1e-5) are used in eval mode. This is easy to confuse with “batch” BN over images.

Two structural variants:

- **No BN:** accumulate (full-N or tiled), then activate.
- **BN:** accumulate → BN (needs full-channel stats) → activate.

### 2.3 Two execution strategies (DIM threshold)

| Path | When | Pattern |
|------|------|---------|
| **Block-pair full-N** | No per-layer pool **or** DIM &lt; 12 | Contiguous half-block loads for each bit mask; auto-vectorizer friendly |
| **Tiled XOR** | Per-layer pool active **and** DIM ≥ 12 | Vertex tiles `T = 64`; `in[v ^ m]` style indexing over `[v_begin, v_end)` |

**Why block-pair?** For mask `m = 1<<k`, pairs `(v, v^m)` sit in blocks of size `2^(k+1)`. Scanning low/high halves yields contiguous loads instead of gathers. That is the **primary hot path** for typical demos (DIM 6–11).

**Why tile at large DIM?** Threaded ranges may not align to block boundaries, so the threaded path keeps XOR indexing inside tiles.

Self tap is always a straight `out[v] += w_self * in[v]` on both paths.

### 2.4 Backward variants

| API | Role |
|-----|------|
| `backward(...)` | Gradients + optimizer step in one call (`TrainStep` path) |
| `compute_gradients(...)` + `apply_gradients(...)` | Write raw grads to caller buffers; apply after batch reduction (`TrainBatch` path) |

Math is the same. XOR is self-inverse, so input grads reuse the same neighbor structure. Self contributes:

```text
∂L/∂w_self = Σ_v grad_pre[v] * in[v]
∂L/∂in[v]  += w_self * grad_pre[v]
```

Optional **`post_act`**: for `TANH`, derivative is `1 - y²` from cached post-activation (avoids a second `tanh` in backward).

### 2.5 TANH speed paths

1. **Post-activation derivative** — as above; numerics match exact `tanh` derivative when `y = tanh(x)` is consistent with forward.
2. **`HCNN_FAST_TANH` (CMake, default ON)** — Padé-style rational approx in forward (and derivative fallback). Private compile definition on the library; consumers inherit whatever was baked into `libHypercubeCNNCore`. Disable with `-DHCNN_FAST_TANH=OFF`.

### 2.6 Weight init (per conv layer)

| Condition | Scheme |
|-----------|--------|
| `scale > 0` | Uniform `[-scale, scale]` |
| ReLU / LeakyReLU and `c_in > 1` | He/Kaiming: `sqrt(6 / fan_in)` |
| Otherwise (NONE, TANH, first layer `c_in=1`, …) | Xavier: `sqrt(6 / (fan_in + fan_out))` |

`fan_in = c_in * K`, `fan_out = c_out * K`, `K = DIM + 1`. Biases → 0; optimizer moments cleared; BN γ→1, β→0, running stats reset when present.

---

## 3. Antipodal pooling (`HCNNPool`)

```text
anti_mask = (1 << DIM) - 1
for each channel, for v in [0, N/2):
  MAX: out[v] = max(in[v], in[v ^ anti_mask]); store argmax
  AVG: out[v] = 0.5 * (in[v] + in[v ^ anti_mask])
```

- Output: perfect (DIM−1)-cube, `N/2` vertices, **same** channel count.
- No learnable parameters.
- **Not** spatial 2×2 neighborhood pool — pairs **maximally distant** addresses.
- Channel-parallel when DIM ≥ 14 and a thread pool is attached (disabled under batch-parallel dispatch).

---

## 4. Readout (`HCNNReadout` — FLATTEN linear only)

No global average pool. After the last conv/pool, activations stay channel-major
`[c * N + v]`. The head treats that buffer as a flat feature vector:

```text
// randomize_all_weights:
num_features = final_channels * final_N
readout = HCNNReadout(num_outputs, num_features)

// every forward / backward:
out[o] = bias[o] + sum_f W[o, f] * in[f]
```

Parameter count of the head:

```text
num_outputs * (c_final * N_final) + num_outputs
```

Often dominates total params when pools are few and N is large (MNIST-style demos).

### 4.1 `grad_in` loop A/B (`ReadoutGradInLoop`)

Backprop into features is `grad_in = W^T * grad_logits`. Two loop nests (same
math; pick via `HCNN::SetReadoutGradInLoop` or `HCNNReadout::set_grad_in_loop`):

| Enum | Nest | W access |
|------|------|----------|
| `OutputOuter` (default) | for o: stream row into `grad_in` | sequential rows |
| `FeatureOuter` | for f: sum over o | column-strided |

Default is `OutputOuter` (faster on MNIST-scale heads in Release A/B). Setting survives `RandomizeWeights`.

---

## 5. Training architecture (`HCNNNetwork`)

### 5.1 Shared cores

Public train entry points are thin wrappers:

| Public | Core | Loss gradient |
|--------|------|----------------|
| `train_step` | `train_step_impl` | classification lambda |
| `train_batch` | `train_batch_impl` | classification lambda per sample |
| `train_step_regression` | `train_step_impl` | regression lambda |
| `train_batch_regression` | `train_batch_impl` | regression lambda per sample |

Shared cores own forward → loss grad → backward → weight update. Adding a loss is a new `LossType` case in `compute_classification_grad` / `compute_regression_grad`, not a new train path.

**Classification CE grad** (after stable softmax):  
`dL/d logits[i] = class_weight * (p[i] - 1[i==target])`.

**Regression MSE-style grad in code (sum-style, not mean):**  
`dL/d pred[i] = pred[i] - target[i]` (LR absorbs the usual 2/K mean-MSE scale).

### 5.2 Optimizers

Configured on all layers via `HCNN::SetOptimizer` (resets global Adam timestep `t`).

**SGD + momentum** (default `OptimizerType`):

```text
g = grad + weight_decay * w     // kernels / readout weights; not typically biases
v = momentum * v + g
w -= lr * v
```

**Adam** with decoupled weight decay on kernels:

```text
m = β1 m + (1-β1) g
v = β2 v + (1-β2) g²
m̂ = m / (1-β1^t),  v̂ = v / (1-β2^t)
w -= lr * (m̂ / (√v̂ + ε) + weight_decay * w)
```

`t` increments once per `train_step` / `train_batch` call (not per sample inside the batch).

### 5.3 Learning rate

Not owned by the network. Every train call takes `lr` explicitly. Optional helper: `hcnn::cosine_lr` in `HCNNTrainHelpers.h`.

### 5.4 Mini-batch parallel training

1. `LayerThreadGuard` — null out per-layer thread pools (no nested `ForEach`).
2. `BNStatsGuard` — suppress per-sample running-stat EMA races.
3. Each sample: forward + `compute_gradients` into **thread-local** accumulators.
4. Reduce (sum) → average → single `apply_gradients` pass.
5. Recompute BN running stats from batch accumulators where needed.

Buffers: `prepare_batch_buffers()` lazy, then reused (allocation-free steady state).
`add_conv` / `add_pool` / `randomize_all_weights` invalidate step, batch, and
inference caches so a later prepare matches the current arch and head size.
`set_optimizer` is stored on the network and re-applied to new convs and to the
rebuilt readout after randomize.

`start_dim` is in **[3, 30]** (`N = 2^dim` fits in signed 32-bit int). `add_pool`
requires `current_dim >= 2`.

### 5.5 Inference

| Path | Notes |
|------|--------|
| `forward` | Ping-pong `fwd_buf1_` / `fwd_buf2_`; **not** concurrent with another forward/train on the same instance |
| `forward_batch` | Per-thread inference buffers; sample parallel |

Both use **`EvalModeGuard`**: force BN eval for the call, restore prior train flags on exit (including exceptions). Safe to `Forward` mid-training without permanently flipping BN mode.

### 5.6 Epoch + shuffle (`HCNN`)

`TrainEpoch` / `TrainEpochRegression` chunk into `TrainBatch*`:

| `shuffle_seed` | Behavior |
|----------------|----------|
| `0` | Sequential slices; zero-copy into caller buffer |
| nonzero | Deterministic shuffle of indices; gather into persistent scratch |

Scratch grows on demand, never shrinks.

### 5.7 Contiguous data model

All batch/epoch APIs take one base pointer + uniform `input_length` (and targets). Row-major: sample `i` at `base + i * input_length`. This is intentional — pointer-per-sample APIs were rejected to avoid silent stride bugs.

---

## 6. Threading model

| Strategy | Scope | Active when |
|----------|--------|-------------|
| Batch sample parallel | Samples in `TrainBatch` / `ForwardBatch` | pool present, batch size > 1 |
| Vertex parallel | Vertices in conv F/B | DIM ≥ 12, not inside batch parallel |
| Channel parallel | Channels in pool F/B | DIM ≥ 14, not inside batch parallel |

`ThreadPool` is **not reentrant**: nested `ForEach` on the same pool deadlocks. Batch paths disable layer pools via `LayerThreadGuard`.

Constructor `num_threads`:

| Value | Meaning |
|-------|---------|
| `0` | Auto (`hardware_concurrency − 1` workers) |
| `1` | No background workers (host may outer-parallelize nets) |
| `N > 1` | `N` workers (+ caller as thread 0 during `ForEach`) |

---

## 7. Weight blob (`GetWeights` / `SetWeights`)

Requires `RandomizeWeights` first (`WeightsInitialized()`).

**Included (in order), per conv:**

```text
kernel[c_out * c_in * K]
bias[c_out]                         // if bias enabled
if BN: gamma, beta, running_mean, running_var   // each c_out
```

then readout weights + readout bias.

**Not included:** SGD/Adam moments, Adam timestep.

`SetWeights(blob, reset_optimizer_moments=false)` restores params only (eval/export).
Pass `true` (or call `SetOptimizer`) to clear moments before continuing training.

---

## 8. Embedding (network-level)

```text
capacity = input_channels * 2^start_dim
copy raw[0 .. input_length)
zero-fill [input_length .. capacity)     // always 0, not a custom pad_value
```

Spatial preprocess may pad with `pad_value = -1`. After spatial embed, train/infer with **`input_length = N`** or the network zero-pad will wipe intentional background. Details: [spatial_preprocess.md](spatial_preprocess.md).

---

## 9. CMake / build options (core-relevant)

| Option | Default | Effect |
|--------|---------|--------|
| `HCNN_FAST_TANH` | ON | Rational tanh in conv activate path |
| `HCNN_NATIVE_ARCH` | ON | `-march=native` style host tuning (non-MSVC) |
| `HCNN_FAST_MATH` | ON | Relaxed float flags without full associative-math chaos |
| `HCNN_BUILD_EXAMPLES` | ON if top-level | Demos + smoke test |

When consumed via FetchContent as a subproject, examples are typically skipped.

---

## 10. File inventory (core library)

```text
HCNN.h / HCNN.cpp
HCNNNetwork.h / HCNNNetwork.cpp
HCNNConv.h / HCNNConv.cpp
HCNNPool.h / HCNNPool.cpp
HCNNReadout.h / HCNNReadout.cpp
ThreadPool.h
HCNNSpatialAug.h / .cpp      (optional)
HCNNSpatialEmbed.h / .cpp    (optional)
HCNNTrainHelpers.h / .cpp    (optional)
```

In-tree only (not install surface): `examples/`, `tests/`, `dataloader/`, `examples/demo_arch.h`.

---

## 11. Related docs

| Doc | Use it for |
|-----|------------|
| [../README.md](../README.md) | Project hook and first path |
| [CPP_SDK.md](CPP_SDK.md) | Public API, educational train loops, train helpers |
| [spatial_preprocess.md](spatial_preprocess.md) | Image aug/embed contracts |
| Headers in `HCNN*.h` | Authoritative signatures and contracts |
