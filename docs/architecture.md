# HypercubeCNN Architecture

## Core idea

HypercubeCNN performs convolutions on binary hypercubes instead of spatial grids. The substrate is a DIM-dimensional binary hypercube with N = 2^DIM vertices. All geometry is bitwise -- neighbor lookup is a single XOR, there are no adjacency lists, no padding, no border cases.

The key insight: on a binary hypercube, every vertex has exactly DIM nearest neighbors at Hamming distance 1 (single-bit flips). The convolution kernel learns one weight per neighbor direction (per bit position), shared across all vertices -- the hypercube analogue of a 3x3 spatial kernel shared across all pixel positions.

All public symbols live in `namespace hcnn`. Code samples in this document omit the prefix for brevity (`using namespace hcnn;` is implied).

## Data representation

### Vertices

Each vertex is identified by a DIM-bit integer in [0, N). The vertex index IS the coordinate -- bit k of the index corresponds to dimension k of the hypercube.

### Channels

One channel = one scalar value per vertex = N floats. Multiple channels are independent copies of the same hypercube geometry, stored contiguously in a channel-major layout:

```
activations[c * N + v]  →  channel c, vertex v
```

Example at DIM=10: one channel = 1024 floats. 64 channels = 65,536 floats in one flat array.

### Memory layout

All data flows through the pipeline as flat `float*` arrays. No tensors, no multidimensional containers. The channel-major convention is universal: conv input, conv output, pool input/output, readout input.

## The convolution (`HCNNConv`)

### What it computes

For each output channel co, each vertex v:

```
out[co, v] = bias[co] + sum over (ci, k) of w[co, ci, k] * in[ci, v ^ (1 << k)]
```

where:
- `ci` iterates over input channels
- `k` iterates over [0, DIM) -- the DIM bit-flip directions
- `v ^ (1 << k)` is the neighbor of v obtained by flipping bit k (XOR)
- `w[co, ci, k]` is the learned weight for output channel co, input channel ci, direction k

This is a direct analogue of a spatial convolution: instead of sliding a 3x3 kernel across a 2D grid, we apply a DIM-direction kernel at every hypercube vertex. The kernel is indexed by bit-flip direction, not by spatial offset.

### Kernel shape

```
kernel[co * c_in * K + ci * K + k]     where K = DIM
```

Total kernel parameters per layer: `c_out * c_in * K`. At DIM=10 with 64→128 channels: 64 * 128 * 10 = 81,920 weights.

For comparison, a standard 3x3 conv with the same channel dimensions: 64 * 128 * 9 = 73,728. Similar ballpark -- the hypercube kernel has DIM directions where a spatial kernel has 9 (or 25 for 5x5).

### Why this works

The binary hypercube is vertex-transitive: every vertex looks the same from every other vertex. This means weight sharing is mathematically principled, not just a heuristic -- the same kernel applied at every vertex respects the symmetry group (Z_2^n).

Each bit-flip direction k is a distinct geometric axis of the hypercube. The kernel learns how much each axis matters for each input→output channel pair. After multiple layers, information from vertices at Hamming distance 2, 3, ... arrives through composition of single-bit-flip convolutions (like how stacked 3x3 convolutions see larger receptive fields).

### Activation, bias, batch normalization

After the weighted sum, each output vertex optionally gets:
1. A per-channel bias term (one scalar per output channel, added to all vertices).
2. Batch normalization (optional, per-layer): per-channel mean/variance computed across all N vertices of the current sample, with learnable scale (γ) and shift (β) parameters and an EMA of running stats for inference.
3. Activation function: `Activation::NONE`, `Activation::RELU`, `Activation::LEAKY_RELU` (slope 0.01), or `Activation::TANH` (smooth, symmetric, bounded in (-1, 1)).

All three are configured per-layer at construction time:

```cpp
net.AddConv(c_out, Activation::RELU,
            /*use_bias=*/true,
            /*use_batchnorm=*/false);
```

The forward path has two compiled variants -- fused accumulate+activate (no BN) and a split accumulate → BN → activate (with BN) -- selected by the per-layer flag.

### Cache tiling

The inner vertex loop is tiled with T=64 vertices per tile. Within a tile:
- Bit-flip masks with `k < log2(T)` stay within the same tile (cache-hot)
- Higher-bit masks map to exactly one other tile (predictable prefetch)

The tile loop structure is: `for tile: for ci: for k: for v_in_tile`, keeping the output tile in L1 for the entire accumulation + activation sequence.

## Pooling (`HCNNPool`)

### Antipodal pooling

Pairs each vertex v with its bitwise complement (antipode) `v ^ (2^DIM - 1)`, the maximally distant vertex on the hypercube. Reduces DIM by 1: vertices in [0, N/2) survive, forming a perfect (DIM-1)-dimensional sub-hypercube.

```
For each channel, each vertex v in [0, N/2):
    v_anti = v ^ ((1 << DIM) - 1)
    MAX:  out[v] = max(in[v], in[v_anti])
    AVG:  out[v] = (in[v] + in[v_anti]) / 2
```

The pooled geometry is exact -- no approximation, no interpolation, no overlap.

### Why antipodal?

On a binary hypercube, the antipodal vertex is the most information-rich pairing: it's at maximum Hamming distance, so combining it with the original vertex captures the widest possible context in a single reduction. This is the hypercube analogue of max-pooling over a 2x2 spatial patch, but instead of collapsing adjacent pixels, it collapses maximally distant vertices.

## Readout (`HCNNReadout`)

Selected at construction time via `ReadoutType`:

- **`ReadoutType::GAP`** (default): global average pooling per channel → one scalar per channel → linear layer `[c_final] → [num_outputs]` with bias. Translation-invariant across hypercube vertices, identical to the GAP + FC readout used in modern spatial CNNs (ResNet, etc.).
- **`ReadoutType::FLATTEN`**: every (channel, vertex) activation is treated as an independent feature → linear layer `[c_final * N_final] → [num_outputs]` with bias. Position-sensitive -- the readout learns per-vertex weights. Larger parameter count, useful when vertex identity carries information.

Internally `HCNNReadout` is a single class; the FLATTEN mode is implemented by sizing the input as `c_final * N_final` and passing `N = 1` to the channel-wise average step (so the average is a no-op and the linear layer sees every activation directly).

The readout is **loss-agnostic and task-agnostic**. It produces `num_outputs` raw real-valued scalars and takes an upstream `grad_logits` in its backward pass. Classification and regression share the same readout class and the same forward/backward math; only the loss-gradient computation differs (see [Task type and loss](#task-type-and-loss)).

## Task type and loss

The network is built for a specific task at construction time via `TaskType`, which controls the training API shape and the default loss:

- **`TaskType::Classification`** (default): integer class targets, softmax + cross-entropy loss, `TrainStep` / `TrainBatch` / `TrainEpoch` API.
- **`TaskType::Regression`**: `const float*` target vectors of length `num_outputs`, MSE loss, `TrainStepRegression` / `TrainBatchRegression` / `TrainEpochRegression` API.

The loss function is an independent axis, selected via `LossType`:

- **`LossType::Default`**: resolve at construction to the natural pairing for the task (CrossEntropy for Classification, MSE for Regression).
- **`LossType::CrossEntropy`**: softmax + cross-entropy. Only valid with `TaskType::Classification`.
- **`LossType::MSE`**: mean squared error. Only valid with `TaskType::Regression`.

Invalid pairings (Classification + MSE, Regression + CrossEntropy) are rejected in the `HCNNNetwork` constructor. Mixing task-mismatched training methods (e.g., calling `TrainStep` on a Regression net) throws `std::logic_error`.

Adding new loss functions (Huber, L1, focal, ...) is a localized change: a new `LossType` enum value plus a new case in `HCNNNetwork::compute_classification_grad` or `compute_regression_grad`. The readout layer, forward pass, backward pass, and public training API are untouched.

The forward path is **identical** in both task types. For classification, the `num_outputs` raw outputs are interpreted as logits (softmax is applied only inside the loss gradient during training, never during inference). For regression, the same `num_outputs` raw outputs are interpreted as predictions directly. `Forward` and `ForwardBatch` know nothing about the task type.

## Network assembly (`HCNN`)

`HCNN` is the canonical SDK front door -- a single class wrapping the entire pipeline. The network is built by stacking conv and pool layers sequentially:

```cpp
hcnn::HCNN net(10);                       // DIM=10, N=1024
net.AddConv(32);                          // 1→32 channels,   K=10 (DIM=10)
net.AddPool(hcnn::PoolType::MAX);         // DIM 10→9,        N 1024→512
net.AddConv(64);                          // 32→64 channels,  K=9  (DIM=9)
net.AddPool(hcnn::PoolType::MAX);         // DIM 9→8,         N 512→256
net.AddConv(128);                         // 64→128 channels, K=8  (DIM=8)
net.AddPool(hcnn::PoolType::MAX);         // DIM 8→7,         N 256→128
net.AddConv(128);                         // 128→128 channels, K=7  (DIM=7)
net.AddPool(hcnn::PoolType::MAX);         // DIM 7→6,         N 128→64
net.RandomizeWeights();                   // Xavier/He init
```

Internally `HCNN` owns an `HCNNNetwork` and forwards every call through a thin PIMPL-style wrapper. The layer headers (`HCNNNetwork`, `HCNNConv`, `HCNNPool`, `HCNNReadout`, `ThreadPool`) are re-exported transitively via `HCNN.h` for power users who need direct weight access, but ordinary consumers should never need to reach for them.

Key properties:
- DIM shrinks only at pool layers. Conv layers preserve dimensionality.
- K = DIM at each conv layer, so deeper layers have fewer kernel directions (but operate on more abstract features).
- Channel count typically increases with depth (same as spatial CNNs).
- The readout is automatically configured from the final channel count and the chosen `ReadoutType`.

`HCNN` is non-copyable and non-movable (it owns a `HCNNNetwork`, which owns a `ThreadPool` with live worker threads and persistent per-thread scratch). Wrap it in `std::unique_ptr<HCNN>` if you need transfer-of-ownership semantics.

## Input embedding

The embedding maps external data onto hypercube vertices. Currently uses **Direct Linear Assignment**: the first min(input_length, N) scalars are assigned to vertices 0, 1, 2, ... in index order. Remaining vertices are zero-padded.

For MNIST (784 pixels → 1024 vertices): pixels land on vertices 0-783, vertices 784-1023 are zero. The mapping is simple and intentionally structure-agnostic -- no spatial locality is encoded. The network must learn all useful relationships from the hypercube topology alone.

## Weight initialization

`RandomizeWeights(scale, seed)` selects per layer based on `scale` and the layer's activation:

- `scale > 0` -- uniform `[-scale, +scale]` (deterministic, primarily for testing).
- `scale <= 0` (default) -- auto:
  - **He/Kaiming uniform** for ReLU/LeakyReLU layers with `c_in > 1`: `s = sqrt(6 / fan_in)`. Accounts for the variance-halving effect of ReLU.
  - **Xavier/Glorot uniform** otherwise (NONE, TANH, or first layer with `c_in = 1` whose input is raw data, not post-activation): `s = sqrt(6 / (fan_in + fan_out))`.
  - `fan_in = c_in * K`, `fan_out = c_out * K`. Computed per-layer since K = DIM varies after pooling.

Biases are reset to zero. Optimizer state buffers (SGD velocity / Adam first and second moments) are cleared. Batch-norm parameters are reset to `γ = 1, β = 0` and running stats to `mean = 0, var = 1`.

## Training

### Optimizer

Two optimizers are available; choose per-network with `HCNN::SetOptimizer`:

- **`OptimizerType::SGD`** (default) -- SGD with momentum and optional L2 weight decay. Decay is applied to kernel and readout weights only (not biases). The update rule:

  ```
  g = gradient + weight_decay * weight
  velocity = momentum * velocity + g
  weight -= learning_rate * velocity
  ```

- **`OptimizerType::ADAM`** -- Adam with decoupled weight decay (AdamW). Per-parameter first and second moment estimates with bias correction:

  ```
  m  = beta1 * m + (1 - beta1) * g
  v  = beta2 * v + (1 - beta2) * g * g
  m_hat = m / (1 - beta1^t)
  v_hat = v / (1 - beta2^t)
  weight -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * weight)
  ```

  The global timestep `t` is owned by `HCNNNetwork` and incremented per training call (`train_step` or `train_batch`); `SetOptimizer` resets it to zero.

### Learning rate

HCNN does not own a learning rate schedule. The learning rate is a parameter on every `TrainStep`, `TrainBatch`, and `TrainEpoch` call -- the caller is responsible for computing it. Both shipped examples use cosine annealing with a floor (`lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch / total_epochs))`), but any schedule (or constant LR) works.

### Backward pass and shared training core

The forward path through the conv/pool stack is identical for both task types. The backward path is also shared -- the only thing that differs between classification and regression is the computation of dL/d(logits) for each sample. This is captured in a single architectural pattern:

1. The four public training methods (`train_step`, `train_batch`, `train_step_regression`, `train_batch_regression`) are thin wrappers. Each validates its task-specific arguments, builds a small lambda that computes the loss gradient for one sample, and delegates to a shared core.
2. The shared cores (`train_step_impl`, `train_batch_impl`) own the entire forward/backward/weight-update pipeline. They call the loss-gradient lambda at exactly one point -- after the readout forward, before the readout backward -- and are otherwise task-agnostic.
3. Adding a new loss function (Huber, L1, focal) requires only a new `LossType` enum value and a new case in `compute_classification_grad` or `compute_regression_grad`. No changes to the forward pass, backward pass, or public API.

Both `HCNNConv` and `HCNNReadout` expose two backward variants:

- **`backward()`**: computes gradients and applies the weight update in a single call. Used by `train_step_impl` (single-sample path).
- **`compute_gradients()` + `apply_gradients()`**: writes raw gradients into caller-provided buffers without touching weights, then applies averaged gradients in a separate call. Used by `train_batch_impl`, where per-sample gradients are accumulated across threads before a single weight update.

The gradient math is identical in both variants -- only the destination differs (internal state vs caller buffers).

### Mini-batch parallelism

`train_batch` processes B samples in parallel across threads. Each thread runs full forward + backward, writing per-sample gradients into thread-local accumulators. After all samples complete, the accumulators are reduced (summed), averaged, and applied in a single weight update.

Per-layer vertex threading is automatically disabled during batch parallelism via `LayerThreadGuard` (RAII), preventing nested `ForEach` on the non-reentrant ThreadPool. Per-sample batch-norm running-stats EMA updates are suppressed during the parallel forward pass via `BNStatsGuard` (RAII) to avoid a data race on `bn_running_mean` / `bn_running_var`; the running stats are recomputed from the per-thread accumulators and applied once after the reduction, race-free.

All per-thread training buffers (forward caches, gradient accumulators, work buffers) are allocated lazily on the first `train_batch` call (`prepare_batch_buffers`) and reused across all subsequent calls -- the steady-state hot path is allocation-free.

### Inference

`HCNN::Forward(embedded, logits)` runs the conv/pool/readout stack on a single pre-embedded sample. It is observably const w.r.t. batch-norm training state: an `EvalModeGuard` (RAII) saves the prior per-layer training flag, forces eval mode (so BN uses its running stats and never updates them), and restores the prior flag on scope exit -- even on exception. This means a `Forward` call mid-training does not silently disable BN running-stats updates for subsequent training calls.

The single-sample forward path uses persistent ping-pong scratch (`fwd_buf1_` / `fwd_buf2_`) sized to the largest layer in the network and grown on demand. After the first call no further allocation occurs.

`HCNN::ForwardBatch(flat_inputs, input_length, batch_size, logits_out)` parallelizes inference across samples using lazily-allocated per-thread buffers (`prepare_inference_buffers`). It uses the same `EvalModeGuard` semantics, plus the `LayerThreadGuard` to disable per-layer vertex threading during the parallel dispatch. Throws `std::invalid_argument` if `batch_size <= 0`.

### Epoch dispatch and shuffling

`HCNN::TrainEpoch` is the convenience wrapper that drives a full pass over a dataset: it walks `sample_count` samples and dispatches `TrainBatch` in chunks of `batch_size`, with the last chunk possibly smaller. Two modes:

- `shuffle_seed = 0` -- samples are processed in input order. Zero-copy: each batch is a direct slice of the caller's buffer via pointer arithmetic.
- `shuffle_seed != 0` -- HCNN permutes a persistent index array (Mersenne Twister + `std::shuffle`), then gathers each batch into persistent scratch buffers before dispatch. Pass a different seed each epoch (e.g. epoch index + 1) for a fresh reproducible permutation.

The gather buffers grow on demand and are reused across epochs, so the steady-state shuffle path is allocation-free as long as `sample_count` does not grow.

## Contiguous data model

All batch and epoch methods accept data as contiguous row-major `const float*` buffers -- one base pointer plus a uniform `int input_length`. This eliminates a class of bugs inherent in pointer-per-sample interfaces (off-by-one stride, mismatched lengths, dangling pointers after reallocation) that produce silent data corruption rather than compiler errors.

The shuffle path gathers each batch into a scratch buffer sized to `batch_size * input_length` floats (classification) or `batch_size * input_length` + `batch_size * num_outputs` floats (regression). These buffers grow on demand but never shrink.

## Threading

Three threading strategies coexist but never nest:

| Strategy | Scope | When active |
|----------|-------|-------------|
| **Batch parallelism** | Samples within a `TrainBatch` or `ForwardBatch` call | Always (when ThreadPool available and batch_size > 1) |
| **Vertex parallelism** | Vertices within a single `HCNNConv::forward` / `backward` call | DIM >= 12 and not inside a batch-parallel dispatch |
| **Channel parallelism** | Channels within `HCNNPool::forward` / `backward` | DIM >= 14 and not inside a batch-parallel dispatch |

All three use the same `ThreadPool` (fork-join, caller-as-thread-0, non-reentrant). Three RAII guards ensure correctness:

- **`LayerThreadGuard`**: disables per-layer vertex/channel threading during batch dispatch, preventing nested `ForEach` calls on the non-reentrant pool. Restores on scope exit.
- **`BNStatsGuard`**: suppresses per-sample running-stats EMA updates during batch-parallel forward passes, preventing a data race on `bn_running_mean` / `bn_running_var`. Running stats are recomputed from per-thread accumulators after the reduction.
- **`EvalModeGuard`**: forces eval mode during inference (`Forward` / `ForwardBatch`), making these calls observably const w.r.t. BN training state. Restores the prior per-layer training flag on scope exit.

All per-thread buffers (training caches, gradient accumulators, inference scratch) are allocated lazily on first use and reused across calls. The steady-state hot path is allocation-free.

## Parameter count

Reference configuration (DIM=10, 4 conv+pool stages, MNIST classification):

| Layer | Shape | Kernel | Bias | Total |
|-------|-------|--------|------|-------|
| Conv1 | 1→32, K=10 | 320 | 32 | 352 |
| Conv2 | 32→64, K=9 | 18,432 | 64 | 18,496 |
| Conv3 | 64→128, K=8 | 65,536 | 128 | 65,664 |
| Conv4 | 128→128, K=7 | 114,688 | 128 | 114,816 |
| Readout | 128→10 | 1,280 | 10 | 1,290 |
| **Total** | | | | **~200K** |

## Implementation

All core code is in the `HypercubeCNNCore` static library (pure C++23, no external dependencies). All public symbols live in `namespace hcnn`. The library exports:

| Class | File | Role |
|-------|------|------|
| `hcnn::HCNN` | HCNN.h/cpp | **Top-level SDK API** -- wraps the entire pipeline |
| `hcnn::HCNNNetwork` | HCNNNetwork.h/cpp | Internal pipeline orchestrator (re-exported via `HCNN.h`) |
| `hcnn::HCNNConv` | HCNNConv.h/cpp | Single conv layer (re-exported) |
| `hcnn::HCNNPool` | HCNNPool.h/cpp | Antipodal pooling layer (re-exported) |
| `hcnn::HCNNReadout` | HCNNReadout.h/cpp | GAP / FLATTEN linear readout (re-exported) |
| `hcnn::ThreadPool` | ThreadPool.h | Header-only fork-join pool (re-exported) |

Public enums (all in `namespace hcnn`): `PoolType` {MAX, AVG} (HCNNPool.h), `ReadoutType` {GAP, FLATTEN} (HCNNNetwork.h), `TaskType` {Classification, Regression} (HCNNNetwork.h), `LossType` {Default, CrossEntropy, MSE} (HCNNNetwork.h), `Activation` {NONE, RELU, LEAKY_RELU, TANH} (HCNNConv.h), `OptimizerType` {SGD, ADAM} (HCNNConv.h).

Executables are thin wrappers that link the library. This separation is intentional -- the library is the C++ SDK surface.
