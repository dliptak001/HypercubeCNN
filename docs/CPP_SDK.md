# HypercubeCNN C++ SDK

HypercubeCNN is a **dependency-free C++23 hypercube CNN core** for research and systems integration (including HypercubeESN). The public surface is **small and contract-driven** so it stays usable in production hosts and legible for engineers learning the stack.

**Examples stay examples. Helpers stay optional. Neither is the product’s reason for existing** — the core (`HCNN`) is.

Canonical API guide for **`HypercubeCNNCore` v1.0.2**. Aligned with the public headers and in-tree recipes. Release notes: **[ChangeLog.md](../ChangeLog.md)**.

**Primary audience:** engineers integrating HCNN into a host binary (e.g. HypercubeESN, custom train/infer pipelines).  
**Secondary:** engineers learning the stack — same API, progressive disclosure, worked recipes.

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

### Capacity is topological (power of two)

Per input channel, capacity is **always** `N = 2^DIM`. That is the size of the Boolean hypercube — not an artificial “max length” the library could relax.

| What the core does | What it does **not** do |
|--------------------|-------------------------|
| Own length-N (or multi-channel × N) activations | Accept a free “input_size = 784” that is not a power of two |
| Zero-pad a **short** raw vector to capacity | Choose how your domain data should sit on the cube |
| Throw if raw length **exceeds** capacity | Guarantee that a packing preserves 2D / sequence locality |

**Non–power-of-two data is normal.** Handle it **outside** the graph with any scheme you prefer: zero or background pad, resize, hash/scatter to vertices, dual-plane image pack, ESN state already at N, etc. Optional `HCNNSpatial*` helpers are one image-oriented recipe (`P ≤ N`); hosts may ignore them entirely. Once packed, pass **full capacity** (or typed full-capacity views) so intentional pad values are not wiped by network zero-pad.

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

## 3. SDK layers and install tree

| Layer | Role |
|-------|------|
| **Core (`HCNN`)** | The product. Integration surface — document like a library another binary links. |
| **Arch / spatial / train helpers** | Optional products for hosts that want them. |
| **Examples** | Proof of contracts + recipes (MNIST, regression). Not the definition of the SDK. |
| **Internals** | Research / maintainers; private headers are not installed and not for apps. |

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
they are **not** part of the installed public surface.

| Product layer | Include | Required? |
|---------------|---------|-----------|
| Full public stack | `HypercubeCNN.h` | Convenient for hosts that want helpers + spatial |
| Core train / infer only | `HCNN.h` | Yes (minimal integration) |
| Architecture list / Build | `HCNNArch.h` (via umbrella) | Optional |
| Image preprocess | spatial headers (via umbrella) | Optional |
| Thin training loop | `HCNNTrainHelpers.h` | Optional |

Link target: **`HypercubeCNNCore`** (or imported `HypercubeCNN::HypercubeCNNCore`).

### Host contracts (integrators and future language bindings)

These rules are intentional product contracts, not implementation accidents.

| Contract | Rule |
|----------|------|
| **Capacity** | Always `input_channels * GetStartN()` with `GetStartN() = 2^DIM`. Power of two is **topology**, not a tunable limit. Map arbitrary lengths onto the cube in the host (pad / pack / embed); the core does not invent that map |
| **Pad (HCNN)** | Short raw inputs **zero-fill** the tail; over-long inputs throw. Spatial embed may use non-zero `pad_value` — always pass full `N` (or typed `HCNNInputView`) after spatial pack |
| **Task / loss** | Classification → softmax CE; Regression → **sum-style** MSE grad (`pred − target`, no `/K` factor). Loss is fixed by `TaskType` |
| **BatchNorm** | Stats are over **vertices of one sample** (per channel), not over the mini-batch |
| **Outputs** | `Forward` / `Predict` return raw logits or predictions — **never** softmax |
| **Arch lifecycle** | `AddConv` / `AddPool` after `RandomizeWeights` **invalidate** weights. Train/infer/`GetWeights` require a successful `RandomizeWeights` for the current stack |
| **Weights blob** | Kernels, biases, BN γ/β + running stats when present; **not** optimizer moments or Adam timestep |
| **Model I/O** | `save_weights` / HCNW store **parameters + coarse checks** (dims, task, layer counts). They do **not** serialize the layer graph. Keep `LayerSpec` / `HCNNConfig` (or equivalent) as the arch sidecar; rebuild the net, then `load_weights` |
| **Concurrency** | One `HCNN` instance is exclusive-use (no concurrent train/infer on the same object). Use `num_threads = 1` when the host parallelizes across many nets |
| **Ownership** | `HCNN` is non-copyable, movable (`unique_ptr` PIMPL) |

**Canonical train/infer surface for new hosts and language bindings** (do not grow beyond this without design review):

```text
construct → AddConv/AddPool or HCNNConfig::Build
RandomizeWeights / SetOptimizer / SetTrainDefaults
Embed / Forward / Predict / PredictClass / ForwardBatch
TrainStep / TrainBatch / TrainEpoch  (+ TrainParams preferred)
GetWeights / SetWeights / GetWeightCount + sizing getters
optional: spatial pack (full N), evaluate_*, cosine_lr, checkpoints, save/load
```

Prefer `TrainParams` (or session defaults) over long positional train argument lists. Prefer full-capacity typed inputs after spatial embed.

### Integration contracts (quick table)

Same rules as **Host contracts** above in compact form:

| Contract | Rule |
|----------|------|
| **Capacity** | `input_channels * GetStartN()` |
| **Pad (HCNN)** | Short → zero-fill; over-long → throw |
| **Task / loss** | CE or sum-style MSE by `TaskType` |
| **Outputs** | Raw logits / preds — no softmax |
| **Arch** | Randomize after stack changes |
| **Weights** | Params + BN stats; no optimizer state |
| **Threading** | Exclusive instance; `num_threads=1` for multi-net |
| **Ownership** | Non-copyable, movable |

---

## 4. Build and consume

**Needs:** C++23, CMake ≥ 3.21.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/sdk   # optional
```

Useful CMake options (library):

| Option | Top-level default | Subproject default | Notes |
|--------|-------------------|--------------------|-------|
| `HCNN_FAST_TANH` | ON | ON | Baked into the static lib; wheels should pick one and document it |
| `HCNN_NATIVE_ARCH` | ON | **OFF** | Host-tuned codegen; **must be OFF** for wheels / redistributable packages |
| `HCNN_FAST_MATH` | ON | ON | Relaxed float flags (not full associative-math) |
| `HCNN_BUILD_EXAMPLES` | ON | OFF | Examples/smoke only when this repo is top-level |

**Packagers / language bindings:** configure with at least
`-DHCNN_NATIVE_ARCH=OFF`. Decide `HCNN_FAST_TANH` once per binary distribution.

### FetchContent (typical consumer project)

```cmake
cmake_minimum_required(VERSION 3.21)
project(MyApp)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    HypercubeCNN
    GIT_REPOSITORY https://github.com/dliptak001/HypercubeCNN.git
    GIT_TAG        v1.0.2   # pin a release tag (see GitHub Releases)
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

**Integration habits:**

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
void RandomizeWeights(float scale = 0.f, uint64_t seed = 42);
void SetOptimizer(OptimizerType type, float beta1 = 0.9f,
                  float beta2 = 0.999f, float eps = 1e-8f);
void SetTrainDefaults(const TrainParams& p);  // session knobs for no-param train*
const TrainParams& GetTrainDefaults() const;
void SetTraining(bool training);   // BN train/eval flag
void PrepareBuffers();             // optional: allocate scratch up front
```

- Default optimizer is **Adam** (AdamW-style decoupled decay on kernels when `weight_decay > 0`). Use `SetOptimizer(SGD)` if you want classical momentum SGD.
- **Session train knobs:** `SetTrainDefaults` holds lr / momentum / weight_decay / shuffle_seed / class_weights for overloads that omit `TrainParams` (e.g. `TrainEpoch(view, y, batch)`). Explicit `TrainParams` args still win.
- `RandomizeWeights`: `scale > 0` → uniform `[-scale, scale]`; else He (ReLU/Leaky, `c_in > 1`) or Xavier. Rebuilds readout to match final `c * N`. Clears optimizer moments / Adam timestep. `seed` is a full 64-bit master seed (`HCNNConfig::weight_seed` too): high half zero keeps the historical `mt19937(seed32)` path (bit-identical for small seeds); wider seeds expand both halves via `seed_seq` (no silent low-32 truncation).
- **Non-copyable, movable.**  Move transfers the heap-owned network (and its
  thread pool) via `unique_ptr`; no worker relocation.  Prefer `unique_ptr<HCNN>`
  for optional ownership; value move is fine for returns and containers.

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

Full-capacity typed views (`HCNNInputView` / `HCNNInputBatch` in `HCNNInput.h`)
avoid short-`input_length` wiping a non-zero spatial pad. Prefer them after
spatial embed when the sample is already length `capacity`.

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

### Training — one vocabulary (classif vs regress by target type)

```cpp
// Classification: int / const int*
net.TrainEpoch(x, len, int_labels, n, batch, params);

// Regression: const float* (num_outputs per sample)
net.TrainEpoch(x, len, float_targets, n, batch, params);

// Same for TrainStep / TrainBatch; HCNNInputView overloads; SetTrainDefaults.
// Wrong TaskType → std::logic_error.
```

**Regression tips (from the regression recipe):** center targets on the **train** mean; Adam is already the default; mix activations as needed (recipe often uses RELU then TANH); full-N FLATTEN without pool keeps vertex identity (useful for reservoir-like inputs). Remember train grads are **sum-style** MSE (`pred − target`), not mean over outputs.

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

Optional public product: describe the body as a list of **`LayerSpec`**, apply it, print params, or build in one shot.

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

// Option B — one-shot Build (unique_ptr; HCNN is also value-movable)
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

## 8. Training loop pattern (recipes)

The shipped examples keep a single **`DemoConfig`** struct at the top of the `.cpp` and a thin loop. Hosts can mirror that structure or drive `HCNN` from their own orchestrator:

```cpp
// 1) Config: dim, layers, lr, batch, seeds, epochs
// 2) Build net from config; RandomizeWeights (Adam is default)
// 3) Pack data into contiguous float arrays (+ int labels or float targets)
// 4) for epoch:
//      lr = cosine_lr(lr_max, lr_min, epoch, num_epochs);
//      TrainEpoch(..., lr, ..., shuffle_seed = epoch+1);
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
    net.TrainEpoch(train_x, N, train_t, n_train, batch, tp);  // float* targets
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
| `tests/CoreSmokeTest.cpp` | API | Canonical behavior contract for the front door |

---

## 9. Optional: spatial preprocess (images)

**Not** part of the conv graph. Headers: `HCNNSpatialAug.h`, `HCNNSpatialEmbed.h`.

```text
H×W image  →  HCNNSpatialAugmenter (train only)  →  HCNNSpatialEmbedder  →  float[N]
                                                                         →  HCNN
```

Embed modes (summary): `PadLow`, `PadLowCenter`, `ResizeToFit`, `DualPlaneResize` — see the spatial guide for layouts and capacity.

**Pad contract (important):**

1. Spatial embed may pad with **`pad_value`** (MNIST-like data: use **−1** for background).
2. `HCNN::Embed` / train paths **zero-pad** any short tail (`input_length < capacity`).
3. After spatial embed, pass **`input_length = emb.capacity()` (= N)**.  
   A short `P` **overwrites** nonzero spatial pad with **0**.

Depth (modes, capacity tables, aug knobs, API sketches): **[`spatial_preprocess.md`](spatial_preprocess.md)**.  
End-to-end image recipe: [`examples/mnist_train.md`](../examples/mnist_train.md).

---

## 10. Optional: train helpers

Header: `HCNNTrainHelpers.h`. **Not** part of the conv/pool graph; does not change `HCNN` math. Include it when you want a thin loop instead of re-implementing CE, cosine LR, or weight snapshots. Hosts that already own their loop can ignore this header.

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
| `HCNNTrainer` | Thin session: `TrainParams` + optional cosine + `train_epoch` |

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
net.TrainEpoch(reg.inputs.data(), reg.input_length,
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

// Versioned file (helpers) — portable little-endian (ints + IEEE float32)
// Checks dim / task / layer counts / weight_count against the live net.
// Does NOT reconstruct architecture: rebuild from LayerSpec/HCNNConfig first.
save_weights(net, "model.hcnw");
load_weights(net, "model.hcnw", /*reset_optimizer_moments=*/true);  // train resume
```

### Image pipeline (with helpers)

```text
H×W → optional SpatialAug → SpatialEmbed (length N)
    → HCNNFlatDataset → TrainEpoch + cosine_lr
    → evaluate_classification + HCNNDualCheckpoint
```

See [`spatial_preprocess.md`](spatial_preprocess.md) and `examples/mnist_train.cpp` / `examples/regression_timeseries.cpp`.

---

## 11. Memory, threading, performance

**Layout:** `activations[c * N + v]` at every stage.

**Threading** (internal `ThreadPool`; strategies never nest):

| Strategy | When |
|----------|------|
| Batch sample parallel | `TrainBatch` / `ForwardBatch`, batch > 1 |
| Vertex parallel (conv) | DIM ≥ 12 and not inside batch parallel |
| Channel parallel (pool) | DIM ≥ 14 and not inside batch parallel |

`num_threads = 1` disables worker threads (use when *you* parallelize across many nets).

**Steady state:** after warm-up / `PrepareBuffers()`, training and inference avoid per-call heap traffic (lazy per-thread buffers, ping-pong forward scratch, shuffle gather).

**Cost scaling:** activations and FLATTEN head grow with `N = 2^DIM` and channels. Examples often use DIM 6–12. Skipping pool keeps full N into a large linear head (high capacity, higher param count).

---

## 12. Pitfalls checklist

| Pitfall | Fix |
|---------|-----|
| Forgot `RandomizeWeights` | Train/infer/`GetWeights` throw until weights match the stack |
| `AddConv`/`AddPool` after randomize | Weights invalidated — call `RandomizeWeights` again |
| Expect free input length (not 2^DIM) | Capacity is the cube; pack/pad in the host (see §1) |
| Softmax in `Forward` | Don’t; use logits + `argmax` / CE helper |
| Wrong train family for `TaskType` | `logic_error` — match Classification vs Regression APIs |
| Short `input_length` after spatial pad −1 | Use `HCNNInputView` / `input_length = N` |
| Expect neighborhood pool | Only **antipodal** pool exists today |
| `K = DIM` in param math | **`K = DIM + 1`** (self + neighbors) |
| Resume train from checkpoint blob | Weights + BN stats when present; **not** optimizer moments — use `SetWeights(blob, true)` or `SetOptimizer` |
| Copy `HCNN` | Deleted — move or `unique_ptr` |
| Treat MNIST pack as spatial CNN prior | Row-major DualPlane is **not** Hamming-local |
| Hypercube = binary values | Topology is binary; activations are float |

---

## 13. Private implementation (not installed — not an app API)

The **installed** SDK is only:

`HCNN.h`, `HCNNTypes.h`, `HCNNInput.h`, `HCNNArch.h`, `HypercubeCNN.h`,
spatial headers, `HCNNTrainHelpers.h`.

These exist **only** in the source tree (for `HCNN.cpp` and in-tree tests):

| Header | Role |
|--------|------|
| `HCNNNetwork` | PIMPL orchestrator behind `HCNN` |
| `HCNNConv` / `HCNNPool` / `HCNNReadout` | Layer implementations |
| `ThreadPool` | Non-reentrant fork-join |

**Boundary policy:**

1. **Hard private** — never install the headers above; apps never include them.
2. **Not a second SDK** — do not build applications against Network/layers.
3. **Features land on `HCNN` first** — Network gains only what the facade needs.

Integrators and applications use **`HCNN`** or **`HypercubeCNN.h`** only.

How the private core is built: **[internals.md](internals.md)**.

---

## 14. Further reading in this repo

| Doc / path | Content |
|------------|---------|
| [`internals.md`](internals.md) | Implementation notes (train cores, RAII, optimizers) |
| [`spatial_preprocess.md`](spatial_preprocess.md) | Aug + embed contracts |
| `examples/mnist_train.md` | Classification recipe write-up |
| `examples/regression_timeseries.md` | Regression recipe write-up |

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

// Helpers: cosine_lr, HCNNTrainer, evaluate_*, FlatDataset, save/load, checkpoints
// Session: net.SetTrainDefaults(p);  or  HCNNTrainer tr(net); tr.set_cosine(...);
```

**Dependencies:** C++23 standard library + threads only.
