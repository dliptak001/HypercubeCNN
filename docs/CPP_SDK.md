# HypercubeCNN C++ SDK

Canonical C++ SDK guide for **HypercubeCNN** (`HypercubeCNNCore` **v0.2.0**). Aligned with the core headers and in-tree teaching demos.

**Audience:** undergrad / grad students who know C++ and the basics of neural nets (forward, loss, SGD/Adam). No framework background required.

**Package:** pure C++23 static library, `namespace hcnn`, no third-party deps beyond the standard library and OS threads.

---

## 1. What you are building

HypercubeCNN is a **CNN whose “grid” is a Boolean hypercube**, not a 2D pixel lattice.

| Idea | Meaning in code |
|------|-----------------|
| Dimension `DIM` | Integer ≥ 3 and ≤ 30 (`N = 2^DIM` fits in 32-bit int). |
| Vertices | `N = 2^DIM` addresses `0 … N−1`. |
| Neighbor of `v` along bit `k` | `v ^ (1 << k)` (XOR). |
| Activation values | Ordinary `float`s (conventionally in `[-1, 1]`). The cube is topology, not bit-valued data. |

**Pipeline:**

```text
raw floats
  → Embed (index assignment + zero-pad to capacity)
  → [ Conv → optional antipodal Pool ]*
  → FLATTEN linear readout  →  num_outputs floats
```

- **Body** = stacked hypercube convolutions (+ optional pools). Multilayer feature mixing lives here.
- **Head** = **one linear layer** over every final `(channel, vertex)` feature (no GAP, no MLP).
- **Task** chooses how those outputs are trained (classification CE vs regression MSE), not the forward graph.

---

## 2. Mental model of one conv layer

For each output channel `co` and vertex `v`:

```text
out[co, v] = bias[co]
           + Σ_ci  w[co,ci,SELF] * in[ci, v]                 // self / center
           + Σ_ci,k w[co,ci,k]    * in[ci, v ^ (1<<k)]      // k = 0 .. DIM-1
```

- Kernel width **`K = DIM + 1`**: neighbors `k ∈ [0, DIM)`, self at index `DIM`.
- Weights are **shared across all vertices** (exact on a vertex-transitive graph).
- Layout of activations is always **channel-major**: `data[c * N + v]`.

**Antipodal pool** (optional): pair `v` with `v ^ (2^DIM − 1)` (maximum Hamming distance), reduce with MAX or AVG, drop DIM by 1 → exact subcube of size `N/2`. This is **not** 2×2 spatial neighborhood pool.

**Readout:** after the last layer, features = `c_final * N_final`. Linear map → `num_outputs`. Softmax is **only** inside the classification loss, never in `Forward`.

---

## 3. What’s in the SDK

### Install tree

```text
<prefix>/
  include/HypercubeCNN/
    HCNN.h                 # front door — start here
    HCNNNetwork.h          # re-exported internals
    HCNNConv.h / HCNNPool.h / HCNNReadout.h
    HCNNSpatialAug.h       # optional 2D aug (not part of the graph)
    HCNNSpatialEmbed.h     # optional 2D → length-N pack
    HCNNTrainHelpers.h     # optional metrics / LR / checkpoints
    ThreadPool.h
  lib/libHypercubeCNNCore.a   # (name may be .lib on MSVC)
  lib/cmake/HypercubeCNN/…
```

| Layer of the product | Include | Required? |
|----------------------|---------|-----------|
| Core train / infer | `HCNN.h` | Yes |
| Image preprocess | `HCNNSpatialAug.h`, `HCNNSpatialEmbed.h` | Optional |
| Thin training loop | `HCNNTrainHelpers.h` | Optional |
| Demo-only scaffolding | `examples/demo_arch.h` | **Not installed** (in-tree teaching only) |

Link target: **`HypercubeCNNCore`** (or imported `HypercubeCNN::HypercubeCNNCore`).

---

## 4. Build and consume

**Needs:** C++23, CMake ≥ 3.21.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/sdk   # optional
```

Useful CMake options (library): `HCNN_FAST_TANH` (default ON), `HCNN_NATIVE_ARCH`, `HCNN_FAST_MATH`, `HCNN_BUILD_EXAMPLES`.

### FetchContent (typical coursework project)

```cmake
cmake_minimum_required(VERSION 3.21)
project(MyApp)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    HypercubeCNN
    GIT_REPOSITORY https://github.com/dliptak001/HypercubeCNN.git
    GIT_TAG        v0.2.0   # or a commit / branch you pin
)
FetchContent_MakeAvailable(HypercubeCNN)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeCNNCore)
```

When HypercubeCNN is **not** the top-level project, examples/tests are skipped automatically.

### find_package (after install)

```cmake
find_package(HypercubeCNN REQUIRED)
target_link_libraries(my_app PRIVATE HypercubeCNN::HypercubeCNNCore)
```

---

## 5. First program (forward only)

```cpp
#include "HCNN.h"
#include <iostream>
#include <random>
#include <vector>

int main() {
    using namespace hcnn;

    // DIM=6 → N=64 vertices; 4 output logits
    HCNN net(/*start_dim=*/6, /*num_outputs=*/4);
    net.AddConv(16, Activation::RELU);
    net.AddPool(PoolType::MAX);          // DIM 6→5, N 64→32
    net.AddConv(32, Activation::RELU);
    net.RandomizeWeights(/*scale=*/0.f, /*seed=*/42);

    const int N = net.GetStartN();
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> x(N), emb(N), logits(net.GetNumOutputs());
    for (float& v : x) v = U(rng);

    net.Embed(x.data(), N, emb.data());   // copy + zero-pad if short
    net.Forward(emb.data(), logits.data());  // raw logits (no softmax)

    for (float z : logits) std::cout << z << ' ';
    std::cout << '\n';
}
```

**Habits this teaches:**

1. Build with `AddConv` / `AddPool`, then **`RandomizeWeights`** (sizes the readout).
2. Caller owns `embedded` and `logits` buffers and reuses them.
3. `Forward` does not apply softmax.

---

## 6. Core API (`hcnn::HCNN`)

### Enums (all via `HCNN.h`)

| Enum | Values | Role |
|------|--------|------|
| `TaskType` | `Classification`, `Regression` | Which train API + default loss |
| `LossType` | `Default`, `CrossEntropy`, `MSE` | `Default` → CE or MSE by task; invalid pairs throw at construct |
| `Activation` | `NONE`, `RELU`, `LEAKY_RELU`, `TANH` | After conv (+ optional BN) |
| `PoolType` | `MAX`, `AVG` | Antipodal reduction |
| `OptimizerType` | `SGD`, `ADAM` | Via `SetOptimizer` (AdamW-style decoupled decay on kernels) |

Constraints: `3 ≤ start_dim ≤ 32`, `num_outputs ≥ 1`, `input_channels ≥ 1`.  
`Classification` only pairs with CE; `Regression` only with MSE.

### Construct and build

```cpp
explicit HCNN(int start_dim,
              int num_outputs = 10,
              int input_channels = 1,
              TaskType task_type = TaskType::Classification,
              LossType loss_type = LossType::Default,
              size_t num_threads = 0);
// num_threads: 0 = auto, 1 = no worker pool, N = N workers

void AddConv(int c_out,
             Activation activation = Activation::RELU,
             bool use_bias = true,
             bool use_batchnorm = false);
void AddPool(PoolType type = PoolType::MAX);   // DIM -= 1
void RandomizeWeights(float scale = 0.f, unsigned seed = 42);
void SetOptimizer(OptimizerType type, float beta1 = 0.9f,
                  float beta2 = 0.999f, float eps = 1e-8f);
void SetTraining(bool training);   // BN train/eval flag
void PrepareBuffers();             // optional: allocate scratch up front
```

- Default optimizer is **SGD** (+ optional momentum/weight decay on train calls). Prefer **`SetOptimizer(ADAM)`** for demos and regression.
- `RandomizeWeights`: `scale > 0` → uniform `[-scale, scale]`; else He (ReLU/Leaky, `c_in > 1`) or Xavier. Rebuilds readout to match final `c * N`. Clears optimizer moments / Adam timestep.
- **Non-copyable, non-movable** (live thread pool). Use `std::unique_ptr<HCNN>` if ownership must move.

### Inference

```cpp
void Embed(const float* raw, int input_length, float* embedded_out) const;
void Forward(const float* embedded, float* outputs) const;
void ForwardBatch(const float* flat_inputs, int input_length,
                  int batch_size, float* outputs_out);
```

| | |
|--|--|
| Capacity | `input_channels * GetStartN()` |
| Short input | Copied; **remainder zero-filled** (always 0, not a custom pad) |
| Over-long input | Throws |
| `Forward` outputs | Raw logits (classif.) or predictions (regress.); no softmax |
| BN during `Forward*` | Forced eval for the call (safe mid-training) |

### Training — classification

Targets: `int` class indices. Loss: softmax + cross-entropy.

```cpp
void TrainStep(const float* raw, int len, int target_class,
               float lr, float momentum = 0, float weight_decay = 0,
               const float* class_weights = nullptr);

void TrainBatch(const float* flat_inputs, int input_length,
                const int* targets, int batch_size,
                float lr, float momentum = 0, float weight_decay = 0,
                const float* class_weights = nullptr);

void TrainEpoch(const float* flat_inputs, int input_length,
                const int* targets, int sample_count, int batch_size,
                float lr, float momentum = 0, float weight_decay = 0,
                const float* class_weights = nullptr,
                unsigned shuffle_seed = 0);
```

- Contiguous **row-major** inputs: sample `i` starts at `flat_inputs + i * input_length`.
- `shuffle_seed == 0`: sequential, zero-copy slices. Nonzero: deterministic shuffle (use a new seed each epoch, e.g. `epoch + 1`).
- **You pass `lr` every call** — HCNN does not own a schedule (use `hcnn::cosine_lr` helper if desired).

### Training — regression

Construct with `TaskType::Regression`. Targets: `float` vectors of length `GetNumOutputs()`.

```cpp
void TrainStepRegression(...);
void TrainBatchRegression(...);   // flat_targets: batch * num_outputs
void TrainEpochRegression(...);   // flat_targets: samples * num_outputs
```

Calling the wrong family’s train methods throws `std::logic_error`.

**Regression tips (from the teaching demo):** center targets on the **train** mean; prefer Adam; mix activations as needed (demo often uses RELU then TANH); full-N FLATTEN without pool keeps vertex identity (useful for reservoir-like inputs).

### Sizing and weights

| Method | Meaning |
|--------|---------|
| `GetStartDim()` / `GetStartN()` | `DIM` and `N = 2^DIM` |
| `GetInputChannels()` / `GetNumOutputs()` | Buffer sizes |
| `GetTaskType()` / `GetLossType()` | Resolved enums (`Default` already expanded) |
| `GetWeightCount()` / `GetWeights()` / `SetWeights()` | Kernel + bias + readout only |

**Weight blob layout:**

```text
for each conv:
  kernel[c_out * c_in * K]   // K = DIM_layer + 1
  bias[c_out]                // if enabled
readout weights[num_outputs * (c_final * N_final)]
readout bias[num_outputs]
```

**Not in the blob:** BN γ/β, BN running stats, optimizer moments, Adam timestep. Checkpoints based on `GetWeights` are for **eval/export**, not perfect mid-train resume (call `SetOptimizer` again if you continue training after restore).

---

## 7. Educational training loop (pattern)

The shipped demos keep a single **`DemoConfig`** struct at the top of the `.cpp` and a thin loop. Reproduce that structure in coursework:

```cpp
// 1) Config: dim, layers, lr, batch, seeds, epochs
// 2) Build net from config; RandomizeWeights; SetOptimizer(ADAM)
// 3) Pack data into contiguous float arrays (+ int labels or float targets)
// 4) for epoch:
//      lr = cosine_lr(lr_max, lr_min, epoch, num_epochs);
//      TrainEpoch[Regression](..., lr, ..., shuffle_seed = epoch+1);
//      evaluate_*(...);
//      checkpoint.observe(...);
// 5) checkpoint.restore_*(net);
```

### Classification sketch

```cpp
#include "HCNN.h"
#include "HCNNTrainHelpers.h"

using namespace hcnn;

HCNN net(dim, /*classes=*/10);
net.AddConv(16);
net.AddConv(16);
net.RandomizeWeights(0.f, weight_seed);
net.SetOptimizer(OptimizerType::ADAM);

// flat: sample_count * input_length floats; labels: sample_count ints
HCNNDualCheckpoint ckpt;
for (int e = 0; e < epochs; ++e) {
    float lr = cosine_lr(1e-3f, 1e-4f, e, epochs);
    net.TrainEpoch(train_x, input_length, train_y, n_train, batch,
                   lr, /*mom=*/0.f, /*wd=*/1e-3f, nullptr,
                   /*shuffle_seed=*/static_cast<unsigned>(e + 1));
    auto r = evaluate_classification(net, test_x, input_length, test_y, n_test);
    ckpt.observe(net, r.loss, r.accuracy, e + 1);
}
ckpt.restore_best_acc(net);
```

### Regression sketch

```cpp
HCNN net(dim, /*num_outputs=*/1, 1, TaskType::Regression);
net.AddConv(16, Activation::RELU);
net.AddConv(16, Activation::TANH);
net.RandomizeWeights(0.f, seed);
net.SetOptimizer(OptimizerType::ADAM);

HCNNBestMetricCheckpoint best;
for (int e = 0; e < epochs; ++e) {
    float lr = cosine_lr(lr_max, lr_min, e, epochs);
    net.TrainEpochRegression(train_x, N, train_t, n_train, batch, lr,
                             0.f, 0.f, static_cast<unsigned>(e + 1));
    auto r = evaluate_regression(net, test_x, N, test_t, n_test);
    best.observe(net, static_cast<float>(r.mse), e + 1);
}
best.restore(net);
// r.r2() = 1 - mse / target_var
```

In-tree references (not part of the install):

| Example | Target | Story |
|---------|--------|--------|
| `examples/mnist_train.cpp` + `.md` | Classification | Spatial aug → DualPlane embed → `TrainEpoch` + dual checkpoint |
| `examples/regression_timeseries.cpp` + `.md` | Regression | Synthetic length-N state → next-step sine; best-MSE checkpoint |
| `examples/demo_arch.h` | Both | `ArchLayer` list, param count vs `GetWeightCount`, print helpers |
| `tests/CoreSmokeTest.cpp` | API | Canonical behavior contract for the front door |

---

## 8. Optional: spatial preprocess (images)

**Not** part of the conv graph. Headers: `HCNNSpatialAug.h`, `HCNNSpatialEmbed.h`.

```text
H×W image  →  HCNNSpatialAugmenter (train only)  →  HCNNSpatialEmbedder  →  float[N]
                                                                         →  HCNN
```

Embed modes (summary): `RowMajorPad`, `ResizeToFit`, `DualPlaneResize` — see the spatial guide for layouts and capacity.

**Pad contract (important):**

1. Spatial embed may pad with **`pad_value`** (MNIST-like data: use **−1** for background).
2. `HCNN::Embed` / train paths **zero-pad** any short tail (`input_length < capacity`).
3. After spatial embed, pass **`input_length = emb.capacity()` (= N)**.  
   A short `P` **overwrites** nonzero spatial pad with **0**.

Depth (modes, capacity tables, aug knobs, API sketches): **[`spatial_preprocess.md`](spatial_preprocess.md)**.  
End-to-end image demo: [`examples/mnist_train.md`](../examples/mnist_train.md).

---

## 9. Optional: train helpers

Header: `HCNNTrainHelpers.h`. **Not** part of the conv/pool graph; does not change `HCNN` math. Include it when you want a thin teaching loop instead of re-implementing CE, cosine LR, or weight snapshots. Native cube apps that already own their loop can ignore this header.

| Utility | Role |
|---------|------|
| `argmax`, `softmax_cross_entropy` | Building blocks for custom eval |
| `evaluate_classification` / `HCNNClassEval` | Mean CE + accuracy % over a flat batch |
| `evaluate_regression` / `HCNNRegEval` | MSE, target variance, `r2()` |
| `HCNNFlatDataset` | Contiguous `inputs` + int labels (classification only) |
| `cosine_lr(lr_max, lr_min, epoch, num_epochs)` | Cosine anneal; epoch 0 → max, last → min |
| `HCNNDualCheckpoint` | Best test loss **and** best test accuracy (`GetWeights` blobs) |
| `HCNNBestMetricCheckpoint` | Best (lowest) scalar, e.g. test MSE |

### Cosine LR

`HCNN` does not own a schedule — you pass `lr` into every train call.

```text
progress = epoch / max(num_epochs - 1, 1)     # clamped
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
```

`num_epochs <= 1` returns `lr_max`. Typical floor: `lr_min = 0.1 * lr_max`.

```cpp
float lr = cosine_lr(1e-3f, 1e-4f, epoch, /*num_epochs=*/60);
net.TrainEpoch(..., lr, ...);
```

### Metrics and checkpoints (sketch)

```cpp
#include "HCNNTrainHelpers.h"
using namespace hcnn;

// Classification eval
HCNNClassEval r = evaluate_classification(net, flat_x, input_length, labels, count);
// r.loss, r.accuracy (percent), r.correct, r.count

// Regression eval
HCNNRegEval re = evaluate_regression(net, flat_x, input_length, flat_t, count);
// re.mse, re.target_var, re.r2()

// Dual checkpoint (MNIST-style): best CE loss and best accuracy
HCNNDualCheckpoint ckpt;
ckpt.observe(net, r.loss, r.accuracy, /*epoch_1based=*/epoch + 1);
ckpt.restore_best_acc(net);

// Best-metric checkpoint (regression-style): minimize test MSE
HCNNBestMetricCheckpoint best;
best.observe(net, static_cast<float>(re.mse), epoch + 1);
best.restore(net);
```

**Weights only.** Checkpoints use `GetWeights` / `SetWeights`: **no** BN γ/β, **no** optimizer state. Fine for eval/export on no-BN demos. To continue training after restore, call `SetOptimizer` again (or accept stale moments).

### Flat classification dataset

```cpp
HCNNFlatDataset ds;
ds.reset(n, input_length);   // inputs = n*len, targets = n
// fill ds.sample_input(i) and ds.targets[i]
net.TrainEpoch(ds.inputs.data(), ds.input_length,
               ds.targets.data(), ds.count, batch, lr, ...);
```

Regression demos keep their own `float` target buffers; there is no regression twin of `HCNNFlatDataset`.

### Image demo pipeline

```text
H×W → optional SpatialAug → SpatialEmbed (length N)
    → HCNNFlatDataset → TrainEpoch + cosine_lr
    → evaluate_classification + HCNNDualCheckpoint
```

See [`spatial_preprocess.md`](spatial_preprocess.md) and `examples/mnist_train.cpp` / `examples/regression_timeseries.cpp`.

---

## 10. Memory, threading, performance (student-relevant)

**Layout:** `activations[c * N + v]` at every stage.

**Threading** (internal `ThreadPool`; strategies never nest):

| Strategy | When |
|----------|------|
| Batch sample parallel | `TrainBatch` / `ForwardBatch`, batch > 1 |
| Vertex parallel (conv) | DIM ≥ 12 and not inside batch parallel |
| Channel parallel (pool) | DIM ≥ 14 and not inside batch parallel |

`num_threads = 1` disables worker threads (use when *you* parallelize across many nets).

**Steady state:** after warm-up / `PrepareBuffers()`, training and inference avoid per-call heap traffic (lazy per-thread buffers, ping-pong forward scratch, shuffle gather).

**Cost scaling:** activations and FLATTEN head grow with `N = 2^DIM` and channels. Demos often use DIM 6–12. Skipping pool keeps full N into a large linear head (high capacity, higher param count).

---

## 11. Pitfalls checklist

| Pitfall | Fix |
|---------|-----|
| Forgot `RandomizeWeights` | Readout not sized; weights zero / unusable |
| Softmax in `Forward` | Don’t; use logits + `argmax` / CE helper |
| Wrong train family for `TaskType` | `logic_error` — match Classification vs Regression APIs |
| Short `input_length` after spatial pad −1 | Use `input_length = N` |
| Expect neighborhood pool | Only **antipodal** pool exists today |
| `K = DIM` in param math | **`K = DIM + 1`** (self + neighbors) |
| Resume train from checkpoint blob | Weights only; reset optimizer; BN γ/β not in blob |
| Copy/move `HCNN` | Deleted — use `unique_ptr` |
| Treat MNIST pack as spatial CNN prior | Row-major DualPlane is **not** Hamming-local |
| Hypercube = binary values | Topology is binary; activations are float |

---

## 12. Power-user internals (optional reading)

`HCNN.h` re-exports layer types for inspection and custom loops:

- `HCNNConv` — kernels, BN, `compute_gradients` / `apply_gradients`
- `HCNNPool` — antipodal MAX/AVG
- `HCNNReadout` — FLATTEN linear head (`num_features = c_final * N_final`; no GAP)
- `HCNNNetwork` — orchestrator behind the PIMPL
- `ThreadPool` — non-reentrant fork-join

Coursework and apps should stay on **`HCNN`** unless you are writing tests or research instrumentation.

How training cores, threading, block-pair kernels, and weight blobs actually work: **[internals.md](internals.md)**.

---

## 13. Further reading in this repo

| Doc / path | Content |
|------------|---------|
| [`internals.md`](internals.md) | Implementation notes (train cores, RAII, optimizers) |
| [`spatial_preprocess.md`](spatial_preprocess.md) | Aug + embed contracts |
| `examples/mnist_train.md` | Classification teaching write-up |
| `examples/regression_timeseries.md` | Regression teaching write-up |

---

## 14. One-page cheat sheet

```text
HCNN net(DIM, outputs [, c_in, TaskType, LossType, threads]);
net.AddConv(c_out [, act, bias, bn]);
net.AddPool([MAX|AVG]);          // optional; DIM -= 1
net.RandomizeWeights([scale], [seed]);
net.SetOptimizer(ADAM);          // recommended for demos

// Inference
net.Embed(raw, len, emb);        // emb size GetStartN(); short → zero pad
net.Forward(emb, out);           // out size GetNumOutputs()
net.ForwardBatch(flat, len, B, out);

// Train (pick one family)
net.TrainEpoch(...);             // classification, int labels
net.TrainEpochRegression(...);   // regression, float targets

// You own: contiguous float buffers, learning rate each call, metrics/checkpoints
```

**Dependencies:** C++23 standard library + threads only.
