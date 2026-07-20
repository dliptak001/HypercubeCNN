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
    HypercubeCNN.h         # umbrella — core + arch + helpers + spatial
    HCNN.h                 # front door (core only)
    HCNNTypes.h            # public enums
    HCNNInput.h            # full-capacity HCNNInputView / HCNNInputBatch
    HCNNArch.h             # LayerSpec, apply_arch, HCNNConfig::Build
    HCNNSpatialAug.h       # optional 2D aug (not part of the graph)
    HCNNSpatialEmbed.h     # optional 2D → length-N pack
    HCNNTrainHelpers.h     # optional metrics / LR / checkpoints
  lib/libHypercubeCNNCore.a   # (name may be .lib on MSVC)
  lib/cmake/HypercubeCNN/…
```

Layer/orchestrator headers (`HCNNNetwork`, `HCNNConv`, `HCNNPool`, `HCNNReadout`,
`ThreadPool`) ship in the **source tree** for tests and research instrumentation;
they are **not** part of the installed teaching surface.

| Layer of the product | Include | Required? |
|----------------------|---------|-----------|
| Full teaching stack | `HypercubeCNN.h` | Recommended for demos |
| Core train / infer only | `HCNN.h` | Yes (minimal) |
| Architecture list / Build | `HCNNArch.h` (via umbrella) | Optional |
| Image preprocess | spatial headers (via umbrella) | Optional |
| Thin training loop | `HCNNTrainHelpers.h` | Optional |
| Demo-only aliases | `examples/demo_arch.h` | **Not installed** (`hcnn_demo::` → `hcnn::`) |

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

    // DIM=6 → N=64 vertices; 4 output logits; default optimizer = Adam
    HCNN net(/*start_dim=*/6, /*num_outputs=*/4);
    net.AddConv(16, Activation::RELU);
    net.AddPool(PoolType::MAX);          // DIM 6→5, N 64→32
    net.AddConv(32, Activation::RELU);
    net.RandomizeWeights(/*scale=*/0.f, /*seed=*/42);

    const int N = net.GetStartN();
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> x(N), logits(net.GetNumOutputs());
    for (float& v : x) v = U(rng);

    net.Predict(x.data(), N, logits.data());  // embed + forward (raw logits)
    // Or: Embed into a caller buffer, then Forward — same math.

    for (float z : logits) std::cout << z << ' ';
    std::cout << '\n';
}
```

**Habits this teaches:**

1. Build with `AddConv` / `AddPool`, then **`RandomizeWeights`** (sizes the readout).
2. Prefer **`Predict`** for single-sample inference; use `Embed`+`Forward` when you cache embeddings.
3. `Predict` / `Forward` do not apply softmax — use `PredictClass` or `argmax` for a label.

---

## 6. Core API (`hcnn::HCNN`)

### Enums (via `HCNNTypes.h` / `HCNN.h`)

| Enum | Values | Role |
|------|--------|------|
| `TaskType` | `Classification`, `Regression` | Train API + fixed loss (CE / MSE) |
| `Activation` | `NONE`, `RELU`, `LEAKY_RELU`, `TANH` | After conv (+ optional BN) |
| `PoolType` | `MAX`, `AVG` | Antipodal reduction |
| `OptimizerType` | `SGD`, `ADAM` | Default **Adam**; override with `SetOptimizer` |

Loss is **fixed by task** (no separate loss enum): Classification → softmax CE; Regression → MSE.

Constraints: **`3 ≤ start_dim ≤ 30`**, `num_outputs ≥ 1`, `input_channels ≥ 1`.  

### Construct and build

```cpp
explicit HCNN(int start_dim,
              int num_outputs = 10,
              int input_channels = 1,
              TaskType task_type = TaskType::Classification,
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

- Default optimizer is **Adam** (AdamW-style decoupled decay on kernels when `weight_decay > 0`). Use `SetOptimizer(SGD)` if you want classical momentum SGD.
- `RandomizeWeights`: `scale > 0` → uniform `[-scale, scale]`; else He (ReLU/Leaky, `c_in > 1`) or Xavier. Rebuilds readout to match final `c * N`. Clears optimizer moments / Adam timestep.
- **Non-copyable, non-movable** (live thread pool). Use `std::unique_ptr<HCNN>` if ownership must move.

### Inference

```cpp
void Predict(const float* raw, int input_length, float* outputs) const;
int  PredictClass(const float* raw, int input_length) const;  // classif. only

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
| `Predict` | Embed into internal scratch + Forward (single-sample happy path) |
| `PredictClass` | `Predict` + argmax; throws on Regression nets |
| `Forward` outputs | Raw logits (classif.) or predictions (regress.); no softmax |
| BN during `Forward*` / `Predict*` | Forced eval for the call (safe mid-training) |

### Training — classification

Targets: `int` class indices. Loss: softmax + cross-entropy.

**Preferred (TrainParams):**

```cpp
struct TrainParams {
    float learning_rate = 1e-3f;
    float momentum = 0.0f;          // SGD only; ignored by Adam
    float weight_decay = 0.0f;
    const float* class_weights = nullptr;  // optional, length num_outputs
    unsigned shuffle_seed = 0;      // epoch only; 0 = sequential
};

void TrainStep(const float* raw, int len, int target, const TrainParams& p);
void TrainBatch(const float* flat_inputs, int input_length,
                const int* targets, int batch_size, const TrainParams& p);
void TrainEpoch(const float* flat_inputs, int input_length,
                const int* targets, int sample_count, int batch_size,
                const TrainParams& p);
```

**Positional overloads** (same math; kept for compatibility) take
`lr, momentum=0, weight_decay=0, class_weights=nullptr[, shuffle_seed=0]`.

- Contiguous **row-major** inputs: sample `i` starts at `flat_inputs + i * input_length`.
- `shuffle_seed == 0`: sequential, zero-copy slices. Nonzero: deterministic shuffle (use a new seed each epoch, e.g. `epoch + 1`).
- **You pass `lr` every call** (or via `TrainParams`) — HCNN does not own a schedule (use `hcnn::cosine_lr` if desired).

### Training — regression

Construct with `TaskType::Regression`. Targets: `float` vectors of length `GetNumOutputs()`.

```cpp
void TrainStepRegression(..., const TrainParams& p);
void TrainBatchRegression(...);   // flat_targets: batch * num_outputs
void TrainEpochRegression(...);   // flat_targets: samples * num_outputs
// + positional overloads
```

Calling the wrong family’s train methods throws `std::logic_error`.

**Regression tips (from the teaching demo):** center targets on the **train** mean; Adam is already the default; mix activations as needed (demo often uses RELU then TANH); full-N FLATTEN without pool keeps vertex identity (useful for reservoir-like inputs).

### Sizing and weights

| Method | Meaning |
|--------|---------|
| `GetStartDim()` / `GetStartN()` / `GetCurrentDim()` | Start DIM, `N`, DIM after pools |
| `GetInputChannels()` / `GetNumOutputs()` | Buffer sizes |
| `GetNumConv()` / `GetNumPool()` | Layer counts |
| `GetTaskType()` / `GetOptimizerType()` | Task and optimizer |
| `WeightsInitialized()` | True after `RandomizeWeights` |
| `GetWeightCount()` / `GetWeights` / `SetWeights` | Full param blob (vector or `float*` + n; incl. BN when used) |

**Weight blob layout** (requires `RandomizeWeights` first):

```text
for each conv:
  kernel[c_out * c_in * K]   // K = DIM_layer + 1
  bias[c_out]                // if enabled
  if BN: gamma, beta, running_mean, running_var  // each c_out
readout weights[num_outputs * (c_final * N_final)]
readout bias[num_outputs]
```

**In the blob when BN is enabled:** γ, β, running mean, running var (each `c_out`).  
**Not in the blob:** optimizer moments, Adam timestep.  
`SetWeights(blob)` — eval restore.  
`SetWeights(blob, /*reset_optimizer_moments=*/true)` — safe train resume.

---

## 7. Architecture product (`HCNNArch.h`)

Describe the body as a list of **`LayerSpec`**, apply it, print params, or build in one shot.

```cpp
#include "HCNNArch.h"   // or HypercubeCNN.h
using namespace hcnn;

// Layer list
std::vector<LayerSpec> layers = {
    LayerSpec::Conv(16),
    LayerSpec::Pool(PoolType::MAX),
    LayerSpec::Conv(32, Activation::TANH),
};

// Option A — incremental HCNN + apply
HCNN net(10, /*classes=*/10);
apply_arch(net, layers);
net.RandomizeWeights();

// Param count matches GetWeightCount (incl. BN blob floats when use_bn)
ArchParamSummary sum = summarize_arch(10, 10, 1, layers);
print_arch(std::cout, 10, 10, 1, layers, sum);

// Option B — one-shot Build (returns unique_ptr; HCNN is non-movable)
HCNNConfig cfg;
cfg.start_dim = 10;
cfg.num_outputs = 10;
cfg.layers = layers;
cfg.weight_seed = 42;
auto built = cfg.Build();   // Apply + RandomizeWeights + SetOptimizer(Adam)
```

| API | Role |
|-----|------|
| `LayerSpec::Conv` / `Pool` | Factory helpers for body steps |
| `summarize_arch` | Walk stack; total == `GetWeightCount` after init |
| `apply_arch(net, layers)` | Append layers (validates first) |
| `print_arch` | Human-readable stack + param breakdown |
| `HCNNConfig::Build()` | Construct + apply + optional randomize + optimizer |

Pool floor matches the network: cannot pool when `current_dim < 2`. Need ≥1 conv.

---

## 8. Educational training loop (pattern)

The shipped demos keep a single **`DemoConfig`** struct at the top of the `.cpp` and a thin loop. Reproduce that structure in coursework:

```cpp
// 1) Config: dim, layers, lr, batch, seeds, epochs
// 2) Build net from config; RandomizeWeights (Adam is default)
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
#include "HypercubeCNN.h"

using namespace hcnn;

HCNN net(dim, /*classes=*/10);  // Adam by default
net.AddConv(16);
net.AddConv(16);
net.RandomizeWeights(0.f, weight_seed);

TrainParams tp;
tp.weight_decay = 1e-3f;

HCNNDualCheckpoint ckpt;
for (int e = 0; e < epochs; ++e) {
    tp.learning_rate = cosine_lr(1e-3f, 1e-4f, e, epochs);
    tp.shuffle_seed  = static_cast<unsigned>(e + 1);
    net.TrainEpoch(train_x, input_length, train_y, n_train, batch, tp);
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

TrainParams tp;
HCNNBestMetricCheckpoint best;
for (int e = 0; e < epochs; ++e) {
    tp.learning_rate = cosine_lr(lr_max, lr_min, e, epochs);
    tp.shuffle_seed  = static_cast<unsigned>(e + 1);
    net.TrainEpochRegression(train_x, N, train_t, n_train, batch, tp);
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
| `examples/demo_arch.h` | Both | Thin `hcnn_demo::` aliases to `HCNNArch.h` (not installed) |
| `tests/CoreSmokeTest.cpp` | API | Canonical behavior contract for the front door |

---

## 9. Optional: spatial preprocess (images)

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

## 10. Optional: train helpers

Header: `HCNNTrainHelpers.h`. **Not** part of the conv/pool graph; does not change `HCNN` math. Include it when you want a thin teaching loop instead of re-implementing CE, cosine LR, or weight snapshots. Native cube apps that already own their loop can ignore this header.

| Utility | Role |
|---------|------|
| `argmax`, `softmax_cross_entropy` | Building blocks for custom eval |
| `evaluate_classification` / `HCNNClassEval` | Mean CE + accuracy % over a flat batch |
| `evaluate_regression` / `HCNNRegEval` | MSE, target variance, `r2()` |
| `HCNNFlatDataset` | Contiguous `inputs` + class `targets` and/or `float_targets` |
| `cosine_lr(lr_max, lr_min, epoch, num_epochs)` | Cosine anneal; epoch 0 → max, last → min |
| `HCNNDualCheckpoint` | Best test loss **and** best test accuracy (`GetWeights` blobs) |
| `HCNNBestMetricCheckpoint` | Best (lowest) scalar, e.g. test MSE |
| `save_weights` / `load_weights` | Versioned binary weight files (architecture-checked) |

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

Checkpoints use `GetWeights` / `SetWeights` (kernels, biases, BN γ/β + running stats when present). **No** optimizer moments. Dual-checkpoint restore is eval-oriented; for train resume after restore use `SetWeights(blob, true)` or `SetOptimizer`.

### Flat dataset (classification and regression)

```cpp
// Classification
HCNNFlatDataset ds;
ds.reset(n, input_length);   // inputs = n*len, targets = n ints
// fill ds.sample_input(i) and ds.targets[i]
net.TrainEpoch(ds.inputs.data(), ds.input_length,
               ds.targets.data(), ds.count, batch, tp);
auto r = evaluate_classification(net, ds);

// Regression
HCNNFlatDataset reg;
reg.reset_regression(n, input_length, /*num_outputs=*/1);
// fill reg.sample_input(i) and reg.sample_float_target(i)[0..]
net.TrainEpochRegression(reg.inputs.data(), reg.input_length,
                         reg.float_targets.data(), reg.count, batch, tp);
auto re = evaluate_regression(net, reg);
```

### Weight blob I/O (core + helpers)

```cpp
// In-memory (HCNN) — pointer form avoids extra allocation
size_t n = net.GetWeightCount();
std::vector<float> buf(n);
net.GetWeights(buf.data(), n);
net.SetWeights(buf.data(), n, /*reset_optimizer_moments=*/false);

// Versioned file (helpers) — checks dim / task / layer counts / weight_count
save_weights(net, "model.hcnw");
load_weights(net, "model.hcnw", /*reset_optimizer_moments=*/true);  // train resume
```

### Image demo pipeline

```text
H×W → optional SpatialAug → SpatialEmbed (length N)
    → HCNNFlatDataset → TrainEpoch + cosine_lr
    → evaluate_classification + HCNNDualCheckpoint
```

See [`spatial_preprocess.md`](spatial_preprocess.md) and `examples/mnist_train.cpp` / `examples/regression_timeseries.cpp`.

---

## 11. Memory, threading, performance (student-relevant)

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

## 12. Pitfalls checklist

| Pitfall | Fix |
|---------|-----|
| Forgot `RandomizeWeights` | Readout not sized; weights zero / unusable |
| Softmax in `Forward` | Don’t; use logits + `argmax` / CE helper |
| Wrong train family for `TaskType` | `logic_error` — match Classification vs Regression APIs |
| Short `input_length` after spatial pad −1 | Use `HCNNInputView` / `input_length = N` |
| Expect neighborhood pool | Only **antipodal** pool exists today |
| `K = DIM` in param math | **`K = DIM + 1`** (self + neighbors) |
| Resume train from checkpoint blob | Weights + BN stats when present; **not** optimizer moments — use `SetWeights(blob, true)` or `SetOptimizer` |
| Copy/move `HCNN` | Deleted — use `unique_ptr` |
| Treat MNIST pack as spatial CNN prior | Row-major DualPlane is **not** Hamming-local |
| Hypercube = binary values | Topology is binary; activations are float |

---

## 13. Advanced / internal surfaces (optional)

The installed SDK is **`HCNN` + types + helpers + spatial**.  In the source
tree (not the install public set), advanced headers exist for tests and
instrumentation:

- `HCNNNetwork` — orchestrator behind the PIMPL
- `HCNNConv` / `HCNNPool` / `HCNNReadout` — layer implementations
- `ThreadPool` — non-reentrant fork-join

Coursework and apps should stay on **`HCNN`** (or `HypercubeCNN.h`).

**Research only (not on `HCNN`):** `HCNNReadout::set_grad_in_loop` selects the
`grad_in = W^T * g` loop nest (FeatureOuter vs OutputOuter; same math).

How training cores, threading, block-pair kernels, and weight blobs actually work: **[internals.md](internals.md)**.

---

## 14. Further reading in this repo

| Doc / path | Content |
|------------|---------|
| [`internals.md`](internals.md) | Implementation notes (train cores, RAII, optimizers) |
| [`spatial_preprocess.md`](spatial_preprocess.md) | Aug + embed contracts |
| `examples/mnist_train.md` | Classification teaching write-up |
| `examples/regression_timeseries.md` | Regression teaching write-up |

---

## 15. One-page cheat sheet

```text
#include "HypercubeCNN.h"        // or HCNN.h / HCNNArch.h for subsets

// Architecture list or one-shot Build
HCNNConfig cfg;
cfg.start_dim = DIM; cfg.num_outputs = K;
cfg.layers = { LayerSpec::Conv(16), LayerSpec::Pool(), LayerSpec::Conv(32) };
auto net = cfg.Build();          // or HCNN + apply_arch + RandomizeWeights

// Inference
net->Predict(raw, len, out);     // happy path (embed + forward)
net->PredictClass(raw, len);     // classification only

// Train (pick one family) — prefer TrainParams
TrainParams p{ .learning_rate = 1e-3f, .weight_decay = 1e-3f, .shuffle_seed = e+1 };
net->TrainEpoch(x, len, y, n, batch, p);

// Helpers: cosine_lr, evaluate_*, FlatDataset, save/load_weights, checkpoints
```

**Dependencies:** C++23 standard library + threads only.
