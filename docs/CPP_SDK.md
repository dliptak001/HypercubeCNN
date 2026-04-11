# HypercubeCNN C++ SDK

Static C++ library for convolutional neural networks on Boolean hypercube graphs.

## Contents

- [What's in the SDK](#whats-in-the-sdk)
- [Building from source](#building-from-source)
- [Using the SDK](#using-the-sdk)
  - [CMake FetchContent (recommended)](#cmake-fetchcontent-recommended)
  - [Installed SDK (find_package)](#installed-sdk-find_package)
- [Minimal example](#minimal-example)
- [API Reference](#api-reference)
  - [Enums](#enums)
  - [HCNN](#hcnn)
  - [Internals (re-exported)](#internals-re-exported)
- [Memory layout](#memory-layout)
- [Threading](#threading)
- [Dependencies](#dependencies)

## What's in the SDK

After installation, the SDK contains:

```
<prefix>/
  include/HypercubeCNN/
    HCNN.h             -- Primary public API (top-level pipeline wrapper)
    HCNNNetwork.h      -- Internal orchestrator (re-exported via HCNN.h)
    HCNNConv.h         -- Conv layer (re-exported via HCNNNetwork.h)
    HCNNPool.h         -- Pooling layer (re-exported via HCNNNetwork.h)
    HCNNReadout.h      -- Readout layer (re-exported via HCNNNetwork.h)
    ThreadPool.h       -- Internal threading (re-exported via HCNNNetwork.h)
  lib/
    libHypercubeCNNCore.a
  lib/cmake/HypercubeCNN/
    HypercubeCNNConfig.cmake
    HypercubeCNNTargets.cmake
    HypercubeCNNConfigVersion.cmake
```

Consumers include `"HCNN.h"` and link against `HypercubeCNNCore`. `HCNN` is the canonical front door for the entire pipeline; the underlying layer headers are re-exported transitively for power users who need direct weight access or custom training loops.

All public symbols live in the `hcnn::` namespace (`hcnn::HCNN`, `hcnn::PoolType`, etc.).

## Building from source

Requirements: C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+), CMake 3.21+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/sdk
```

## Using the SDK

### CMake FetchContent (recommended)

The simplest way to use HypercubeCNN in a CMake project. No installation, no manual downloads -- CMake pulls the source from GitHub and builds it alongside your project.

```cmake
cmake_minimum_required(VERSION 3.21)
project(MyApp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    HypercubeCNN
    GIT_REPOSITORY https://github.com/dliptak001/HypercubeCNN.git
    GIT_TAG        v0.1.0
)
FetchContent_MakeAvailable(HypercubeCNN)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeCNNCore)
```

When pulled via FetchContent, only the library is built -- examples and diagnostics are skipped automatically.

### Installed SDK (find_package)

If you installed the SDK with `cmake --install`:

```cmake
find_package(HypercubeCNN REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeCNN::HypercubeCNNCore)
```

## Minimal example

A self-contained forward pass on synthetic data (no MNIST required):

```cpp
#include "HCNN.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    using namespace hcnn;

    // Build a small network: DIM=6, N=64 vertices, 4 classes
    HCNN net(6, /*num_outputs=*/4);
    net.AddConv(16);
    net.AddPool(PoolType::MAX);   // DIM 6->5, N 64->32
    net.AddConv(32);
    net.AddPool(PoolType::MAX);   // DIM 5->4, N 32->16
    net.RandomizeWeights();       // Xavier/He init per layer

    // Generate random input in [-1, 1]
    const int N = net.GetStartN();  // 64
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> input(N);
    for (auto& v : input) v = dist(rng);

    // Forward pass -- both buffers caller-owned, designed for reuse.
    std::vector<float> embedded(N);
    std::vector<float> logits(net.GetNumOutputs());
    net.Embed(input.data(), N, embedded.data());
    net.Forward(embedded.data(), logits.data());

    std::cout << "Logits:";
    for (float v : logits) std::cout << " " << v;
    std::cout << "\n";

    return 0;
}
```

## API Reference

### Enums

All enums live in `namespace hcnn`. They are scattered across the layer headers but all reachable through `HCNN.h` (transitively).

| Enum | Values | Defined in | Description |
|------|--------|------------|-------------|
| `hcnn::PoolType`      | `MAX`, `AVG`              | HCNNPool.h     | Antipodal pooling reduction. MAX: keep the larger value. AVG: average the pair. |
| `hcnn::ReadoutType`   | `GAP`, `FLATTEN`          | HCNNNetwork.h  | Readout strategy. GAP: global average pooling per channel (translation-invariant). FLATTEN: concatenate all channel x vertex activations (position-sensitive). |
| `hcnn::TaskType`      | `Classification`, `Regression` | HCNNNetwork.h | Task the network is trained for. Controls the training API (integer class targets vs. float regression targets) and the interpretation of raw readout outputs. |
| `hcnn::LossType`      | `Default`, `CrossEntropy`, `MSE` | HCNNNetwork.h | Loss function. `Default` resolves to the natural pairing for the task (CrossEntropy for Classification, MSE for Regression). Invalid pairings throw at construction. |
| `hcnn::Activation`    | `NONE`, `RELU`, `LEAKY_RELU`, `TANH` | HCNNConv.h  | Activation function applied after conv (and optional batch normalization). `TANH` is smooth, symmetric, and bounded in (-1, 1) -- the standard activation for reservoir-computing readouts and other bounded-output architectures. |
| `hcnn::OptimizerType` | `SGD`, `ADAM`             | HCNNConv.h     | Weight-update rule. Configured per-network via `HCNN::SetOptimizer`. |

### HCNN

The canonical SDK front door. Owns the full pipeline: input embedding → conv/pool stack → readout. Non-copyable, non-movable.

All public methods avoid hidden per-call allocations in steady state:

- **Single-sample inference** (`Forward`): caller owns the embed/logits scratch and reuses it; HCNN keeps a persistent ping-pong scratch internally for the conv/pool ladder, sized to the largest layer and grown on demand.
- **Batch inference** (`ForwardBatch`) and **batch training** (`TrainBatch`, `TrainEpoch`): per-thread work buffers are allocated lazily on the first call and reused thereafter.
- **TrainEpoch shuffle**: persistent gather buffers grow on demand (and never shrink); the steady-state shuffle is allocation-free.

`Forward` and `ForwardBatch` are observably const w.r.t. batch-norm running statistics: they internally force eval mode for the duration of the call and restore the prior per-layer training flag on exit (RAII-safe, including on exception). You do not need to call `SetTraining(false)` before inference.

#### Constructor

```cpp
explicit HCNN(int start_dim, int num_outputs = 10,
              int input_channels = 1,
              ReadoutType readout_type = ReadoutType::GAP,
              TaskType task_type = TaskType::Classification,
              LossType loss_type = LossType::Default,
              size_t num_threads = 0);
```

| Parameter | Description |
|-----------|-------------|
| `start_dim` | Hypercube dimension. The input has N = 2^start_dim vertices. |
| `num_outputs` | Number of readout outputs. For `TaskType::Classification` this is the class count; for `TaskType::Regression` it is the dimensionality of the target vector. |
| `input_channels` | Number of input channels (typically 1). |
| `readout_type` | `GAP` (default) or `FLATTEN`. See [Enums](#enums). |
| `task_type` | `Classification` (default) or `Regression`. See [Enums](#enums) and [Task types and losses](#task-types-and-losses). |
| `loss_type` | `Default` (default) resolves to CrossEntropy for Classification or MSE for Regression. Explicit values are `CrossEntropy` (Classification only) and `MSE` (Regression only). Invalid pairings throw `std::runtime_error` at construction. |
| `num_threads` | Thread pool size. 0 (default) = auto-detect from hardware. |

#### Architecture (incremental builder)

```cpp
void AddConv(int c_out, Activation activation = Activation::RELU,
             bool use_bias = true, bool use_batchnorm = false);
void AddPool(PoolType type = PoolType::MAX);
void RandomizeWeights(float scale = 0.0f, unsigned seed = 42);
```

| Method | Description |
|--------|-------------|
| `AddConv` | Append a convolutional layer with `c_out` output channels. K = current DIM (one weight per Hamming-distance-1 neighbor). Optional per-channel bias and batch normalization. Activation is `RELU` by default; pass `Activation::LEAKY_RELU` for LeakyReLU (slope 0.01), `Activation::TANH` for tanh, or `Activation::NONE` for a linear layer. |
| `AddPool` | Append an antipodal pooling layer. Reduces DIM by 1. |
| `RandomizeWeights` | Initialize all weights. `scale > 0`: uniform `[-scale, +scale]` (deterministic, primarily for testing). `scale <= 0` (default): per-layer auto-init -- He/Kaiming uniform for ReLU/LeakyReLU layers with `c_in > 1`, Xavier/Glorot uniform otherwise. Resets biases to zero, optimizer state to zero, and BN parameters to (γ=1, β=0). |

Call `AddConv` and `AddPool` to build the architecture, then `RandomizeWeights` before training. Optional: `SetOptimizer(OptimizerType::ADAM, ...)` to switch from the default SGD-with-momentum to Adam (with decoupled weight decay).

#### Mode / optimizer

```cpp
void SetTraining(bool training);
void SetOptimizer(OptimizerType type, float beta1 = 0.9f,
                  float beta2 = 0.999f, float eps = 1e-8f);
```

`SetTraining` flips all batch-norm layers between training (running-stat updates) and eval mode. `SetOptimizer` reconfigures all layers' optimizer (`SGD` or `ADAM`) and resets the timestep.

#### Inference

```cpp
void Embed(const float* raw_input, int input_length, float* embedded_out) const;
void Forward(const float* embedded, float* logits) const;
void ForwardBatch(const float* flat_inputs, int input_length,
                  int batch_size, float* logits_out);
```

| Method | Description |
|--------|-------------|
| `Embed` | Map a flat scalar array onto N = 2^DIM hypercube vertices via Direct Linear Assignment. Values must be in [-1.0, 1.0]. `embedded_out` must hold `GetStartN()` floats. Caller-owned buffer (designed for reuse). Throws `std::runtime_error` if `input_length` exceeds capacity (`input_channels * GetStartN()`). |
| `Forward` | Run all conv/pool/readout layers from already-embedded activations. Input: `GetStartN()` floats. Output: `GetNumOutputs()` floats -- raw logits for Classification, raw predictions for Regression (no softmax either way). Steady-state allocation-free (uses persistent ping-pong scratch on the network). Internally forces BN eval mode for the duration of the call and restores the prior per-layer training flag on exit (RAII-safe). |
| `ForwardBatch` | Batch inference from a contiguous row-major input matrix. `flat_inputs` is `batch_size * input_length` contiguous floats. `logits_out` must hold `batch_size * GetNumOutputs()` floats. Per-thread buffers are lazily allocated and reused. Same eval-mode RAII semantics as `Forward`. Throws `std::invalid_argument` if `batch_size <= 0`. |

For single-sample inference, allocate two scratch vectors once and reuse them across calls -- see the [Minimal example](#minimal-example).

#### Training -- classification

```cpp
void TrainStep(const float* raw_input, int input_length, int target_class,
               float learning_rate, float momentum = 0.0f,
               float weight_decay = 0.0f,
               const float* class_weights = nullptr);

void TrainBatch(const float* flat_inputs, int input_length,
                const int* targets, int batch_size,
                float learning_rate, float momentum = 0.0f,
                float weight_decay = 0.0f,
                const float* class_weights = nullptr);

void TrainEpoch(const float* flat_inputs, int input_length,
                const int* targets, int sample_count, int batch_size,
                float learning_rate, float momentum = 0.0f,
                float weight_decay = 0.0f,
                const float* class_weights = nullptr,
                unsigned shuffle_seed = 0);
```

| Method | Description |
|--------|-------------|
| `TrainStep` | Single-sample step: forward + backward + weight update via the configured optimizer. Classification only -- throws `std::logic_error` if the network was built with `TaskType::Regression`. Throws `std::runtime_error` if `target_class` is out of range. |
| `TrainBatch` | Mini-batch parallel step from contiguous data. `flat_inputs` is `batch_size * input_length` contiguous floats. Forward+backward run in parallel for each sample, gradients are reduced (averaged), then a single weight update is applied via the configured optimizer. Per-thread buffers are lazily allocated and reused. Classification only. Throws `std::invalid_argument` if `batch_size <= 0`, `std::logic_error` on Regression nets, or `std::runtime_error` if any target is out of range. |
| `TrainEpoch` | Full pass over `sample_count` samples, dispatching `TrainBatch` in chunks of `batch_size` (last chunk may be smaller). `shuffle_seed = 0`: input order, zero-copy. Nonzero `shuffle_seed`: deterministic permutation (pass a different seed per epoch for a fresh shuffle). Classification only. Throws `std::invalid_argument` if `batch_size <= 0` or `sample_count < 0`. |

`class_weights` (optional, length `GetNumOutputs()`) scales the per-class loss; pass `nullptr` for uniform weighting.

The optimizer (`SGD` or `ADAM`) and the per-layer batch-norm flag (set when calling `AddConv`) are honored automatically by all three training methods.

#### Training -- regression

```cpp
void TrainStepRegression(const float* raw_input, int input_length,
                         const float* target, float learning_rate,
                         float momentum = 0.0f,
                         float weight_decay = 0.0f);

void TrainBatchRegression(const float* flat_inputs, int input_length,
                          const float* flat_targets, int batch_size,
                          float learning_rate, float momentum = 0.0f,
                          float weight_decay = 0.0f);

void TrainEpochRegression(const float* flat_inputs, int input_length,
                          const float* flat_targets,
                          int sample_count, int batch_size,
                          float learning_rate, float momentum = 0.0f,
                          float weight_decay = 0.0f,
                          unsigned shuffle_seed = 0);
```

Regression counterparts of `TrainStep` / `TrainBatch` / `TrainEpoch`. The only differences from the classification methods are:

- `target` / `flat_targets` are `const float*` pointers to contiguous real-valued target data instead of integer class indices. `flat_targets` is `batch_size * GetNumOutputs()` (for `TrainBatchRegression`) or `sample_count * GetNumOutputs()` (for `TrainEpochRegression`) contiguous floats.
- The loss is MSE (the default for `TaskType::Regression`) instead of softmax + cross-entropy.
- No `class_weights` parameter.

All three methods throw `std::logic_error` if called on a Classification network. Forward pass, backward pass, optimizer, and batch-parallel reduction are identical to the classification path -- only the loss-gradient computation and target type differ.

**Regression best practices:**

- **Center targets** on the train-set mean before training, add it back at inference. Even nearly-zero-mean targets (e.g. sine) have small but nonzero empirical mean that slows early convergence.
- **Standardize inputs** if per-vertex distributions vary (e.g. reservoir state with different timescales). The conv kernel shares weights across vertices, so uniform input scale matters.
- **Prefer `Activation::TANH`** for bounded-output regression (reservoir readout, time-series prediction). It matches common upstream nonlinearities and produces smooth gradients that interact well with antipodal max-pool.
- **Prefer `OptimizerType::ADAM`** for regression tasks. The adaptive per-parameter scaling navigates the max-pool's non-smooth gradient landscape more effectively than SGD.

See [examples/regression_timeseries.md](../examples/regression_timeseries.md) for an end-to-end walkthrough.

#### Task types and losses

`TaskType::Classification` (default) gives you the integer-class-index API and softmax + cross-entropy loss.

`TaskType::Regression` gives you the `*Regression` API and MSE loss. Build the network with:

```cpp
hcnn::HCNN net(DIM, /*num_outputs=*/3, /*input_channels=*/1,
               hcnn::ReadoutType::GAP,
               hcnn::TaskType::Regression);
```

and train with `TrainEpochRegression` (or the `TrainStepRegression` / `TrainBatchRegression` lower-level primitives). The forward path (`Embed`, `Forward`, `ForwardBatch`) is identical to classification; the raw `num_outputs` readout outputs are simply interpreted as real-valued predictions instead of logits. No softmax is applied in either direction.

Invalid task/loss pairings are rejected in the constructor:

- `Classification` + `MSE` → `std::runtime_error`
- `Regression` + `CrossEntropy` → `std::runtime_error`

Mixing training APIs (e.g., calling `TrainStep` on a Regression net or `TrainStepRegression` on a Classification net) throws `std::logic_error`.

The `LossType` enum is designed for extension: adding a new loss (Huber, L1, focal, ...) requires a new enum value and a new case in the internal gradient dispatch -- no public API change.

#### Sizing accessors

| Method | Returns |
|--------|---------|
| `GetStartDim()` | Initial hypercube dimension. |
| `GetStartN()` | Initial vertex count (2^start_dim). Use to size embed/input buffers. |
| `GetInputChannels()` | Number of input channels. |
| `GetNumOutputs()` | Number of readout outputs. Use to size logits / prediction buffers. |
| `GetTaskType()` | `TaskType::Classification` or `TaskType::Regression`. |
| `GetLossType()` | The resolved loss type (never `LossType::Default` -- resolved at construction). |

### Internals (re-exported)

`HCNN.h` transitively re-exports `HCNNNetwork.h`, which in turn re-exports `HCNNConv.h`, `HCNNPool.h`, `HCNNReadout.h`, and `ThreadPool.h`. All of these symbols live in `namespace hcnn`. They remain reachable for power users who need:

- Direct kernel/bias inspection (`hcnn::HCNNConv::get_kernel_data()`, etc.)
- Custom gradient pipelines (`hcnn::HCNNConv::compute_gradients` / `apply_gradients`)
- Layer-by-layer diagnostics (e.g. gradient checking)
- A custom training loop that drives `hcnn::HCNNNetwork` directly instead of going through `HCNN`

Typical SDK consumers should not need to touch them. The full inventory of these classes is documented in their respective headers and is not duplicated here -- read the header you need.

## Memory layout

All activations use channel-major layout:

```
activations[c * N + v]    // channel c, vertex v
```

where N = 2^DIM at the current layer. Conv input/output, pool input/output, and readout input all follow this convention. Readout output is a flat array of `num_outputs` floats -- logits (for Classification) or raw predictions (for Regression).

Input values must be in [-1.0, 1.0]. The embedding maps the first min(input_length, N) scalars to vertices 0, 1, 2, ...; remaining vertices are zero-padded.

## Threading

`HCNNNetwork` owns a fork-join `ThreadPool` (auto-sized to `hardware_concurrency() - 1` workers, or caller-specified via the `num_threads` constructor parameter). Three threading strategies coexist but never nest:

| Strategy | Scope | When active |
|----------|-------|-------------|
| **Batch parallelism** | Samples within `TrainBatch` / `ForwardBatch` | Always (when ThreadPool available and batch_size > 1) |
| **Vertex parallelism** | Vertices within a single `HCNNConv` forward/backward | DIM >= 12 and not inside a batch-parallel dispatch |
| **Channel parallelism** | Channels within a single `HCNNPool` forward/backward | DIM >= 14 and not inside a batch-parallel dispatch |

Three RAII guards prevent nesting and data races:

- **`LayerThreadGuard`**: disables per-layer vertex/channel threading during batch dispatch. Restores on scope exit.
- **`BNStatsGuard`**: suppresses per-sample running-stats EMA updates during batch-parallel forward passes. Running stats are recomputed from per-thread accumulators after the reduction.
- **`EvalModeGuard`**: forces eval mode during inference (`Forward` / `ForwardBatch`), making these calls observably const w.r.t. BN training state. Restores on scope exit.

All per-thread buffers are allocated lazily on first use and reused across calls. The thread pool is internal and not part of the public API surface, though it is reachable as `hcnn::ThreadPool` for power users.

## Dependencies

No external dependencies beyond the C++ standard library.
