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
    HCNN net(6, /*num_classes=*/4);
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
    std::vector<float> logits(net.GetNumClasses());
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
| `hcnn::Activation`    | `NONE`, `RELU`, `LEAKY_RELU` | HCNNConv.h  | Activation function applied after conv (and optional batch normalization). |
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
explicit HCNN(int start_dim, int num_classes = 10,
              int input_channels = 1,
              ReadoutType readout_type = ReadoutType::GAP,
              size_t num_threads = 0);
```

| Parameter | Description |
|-----------|-------------|
| `start_dim` | Hypercube dimension. The input has N = 2^start_dim vertices. |
| `num_classes` | Number of output classes for classification. |
| `input_channels` | Number of input channels (typically 1). |
| `readout_type` | `GAP` (default) or `FLATTEN`. See [Enums](#enums). |
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
| `AddConv` | Append a convolutional layer with `c_out` output channels. K = current DIM (one weight per Hamming-distance-1 neighbor). Optional per-channel bias and batch normalization. Activation is `RELU` by default; pass `Activation::LEAKY_RELU` for LeakyReLU (slope 0.01) or `Activation::NONE` for a linear layer. |
| `AddPool` | Append an antipodal pooling layer. Reduces DIM by 1. |
| `RandomizeWeights` | Initialize all weights. `scale > 0`: uniform `[-scale, +scale]` (deterministic, primarily for testing). `scale <= 0` (default): per-layer auto-init — He/Kaiming uniform for ReLU/LeakyReLU layers with `c_in > 1`, Xavier/Glorot uniform otherwise. Resets biases to zero, optimizer state to zero, and BN parameters to (γ=1, β=0). |

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
void ForwardBatch(const float* const* raw_inputs, const int* input_lengths,
                  int batch_size, float* logits_out);
```

| Method | Description |
|--------|-------------|
| `Embed` | Map a flat scalar array onto N = 2^DIM hypercube vertices via Direct Linear Assignment. Values must be in [-1.0, 1.0]. `embedded_out` must hold `GetStartN()` floats. Caller-owned buffer (designed for reuse). Throws `std::runtime_error` if `input_length` exceeds capacity (`input_channels * GetStartN()`). |
| `Forward` | Run all conv/pool/readout layers from already-embedded activations. Input: `GetStartN()` floats. Output: `GetNumClasses()` floats. Steady-state allocation-free (uses persistent ping-pong scratch on the network). Internally forces BN eval mode for the duration of the call and restores the prior per-layer training flag on exit (RAII-safe). |
| `ForwardBatch` | Embed + forward for multiple samples in parallel using the internal thread pool. `logits_out` must hold `batch_size * GetNumClasses()` floats. Per-thread buffers are lazily allocated and reused. Same eval-mode RAII semantics as `Forward`. Throws `std::invalid_argument` if `batch_size <= 0`. |

For single-sample inference, allocate two scratch vectors once and reuse them across calls — see the [Minimal example](#minimal-example).

#### Training

```cpp
void TrainStep(const float* raw_input, int input_length, int target_class,
               float learning_rate, float momentum = 0.0f,
               float weight_decay = 0.0f,
               const float* class_weights = nullptr);

void TrainBatch(const float* const* inputs, const int* input_lengths,
                const int* targets, int batch_size,
                float learning_rate, float momentum = 0.0f,
                float weight_decay = 0.0f,
                const float* class_weights = nullptr);

void TrainEpoch(const float* const* inputs, const int* input_lengths,
                const int* targets, int sample_count, int batch_size,
                float learning_rate, float momentum = 0.0f,
                float weight_decay = 0.0f,
                const float* class_weights = nullptr,
                unsigned shuffle_seed = 0);
```

| Method | Description |
|--------|-------------|
| `TrainStep` | Single-sample step: forward + backward + weight update via the configured optimizer. Throws `std::runtime_error` if `target_class` is out of range. |
| `TrainBatch` | Mini-batch parallel step. Forward+backward run in parallel for each sample, gradients are reduced (averaged), then a single weight update is applied via the configured optimizer. Per-thread buffers are lazily allocated and reused. Throws `std::invalid_argument` if `batch_size <= 0`, or `std::runtime_error` if any target is out of range. |
| `TrainEpoch` | Iterate `sample_count` samples and dispatch `TrainBatch` in chunks of `batch_size` (the final chunk may be smaller). With `shuffle_seed = 0`, samples are processed in input order. With nonzero `shuffle_seed`, the epoch is permuted deterministically (pass a different seed each epoch — e.g. epoch index + 1 — for a fresh reproducible shuffle). HCNN owns persistent gather buffers for the shuffle path; after the first shuffled epoch no further allocations occur unless `sample_count` grows. Throws `std::invalid_argument` if `batch_size <= 0` or `sample_count < 0`. |

`class_weights` (optional, length `GetNumClasses()`) scales the per-class loss; pass `nullptr` for uniform weighting.

The optimizer (`SGD` or `ADAM`) and the per-layer batch-norm flag (set when calling `AddConv`) are honored automatically by all three training methods.

#### Sizing accessors

| Method | Returns |
|--------|---------|
| `GetStartDim()` | Initial hypercube dimension. |
| `GetStartN()` | Initial vertex count (2^start_dim). Use to size embed/input buffers. |
| `GetInputChannels()` | Number of input channels. |
| `GetNumClasses()` | Number of output classes. Use to size logits buffers. |

### Internals (re-exported)

`HCNN.h` transitively re-exports `HCNNNetwork.h`, which in turn re-exports `HCNNConv.h`, `HCNNPool.h`, `HCNNReadout.h`, and `ThreadPool.h`. All of these symbols live in `namespace hcnn`. They remain reachable for power users who need:

- Direct kernel/bias inspection (`hcnn::HCNNConv::get_kernel_data()`, etc.)
- Custom gradient pipelines (`hcnn::HCNNConv::compute_gradients` / `apply_gradients`)
- Layer-by-layer diagnostics (e.g. gradient checking)
- A custom training loop that drives `hcnn::HCNNNetwork` directly instead of going through `HCNN`

Typical SDK consumers should not need to touch them. The full inventory of these classes is documented in their respective headers and is not duplicated here — read the header you need.

## Memory layout

All activations use channel-major layout:

```
activations[c * N + v]    // channel c, vertex v
```

where N = 2^DIM at the current layer. Conv input/output, pool input/output, and readout input all follow this convention. Logits output from readout is a flat array of `num_classes` floats.

Input values must be in [-1.0, 1.0]. The embedding maps the first min(input_length, N) scalars to vertices 0, 1, 2, ...; remaining vertices are zero-padded.

## Threading

`HCNNNetwork` owns a fork-join `ThreadPool` (auto-sized to `hardware_concurrency() - 1` workers, or caller-specified via the `num_threads` constructor parameter). Three threading strategies are used, never nested:

- **Batch training** (`TrainBatch`, `TrainEpoch`): samples run forward+backward in parallel, with each thread accumulating gradients into thread-local buffers. After all samples complete the gradients are reduced (summed), averaged, and applied in a single weight update.
- **Batch inference** (`ForwardBatch`): samples run forward in parallel using lazily-allocated per-thread inference buffers.
- **Per-layer vertex threading**: parallelizes the inner vertex loop within each conv layer. Only activates at `DIM >= 12` (below that, fork-join overhead exceeds the per-vertex work). Automatically disabled during batch dispatch via an RAII guard, preventing nested `ForEach` on the non-reentrant pool.

The thread pool is internal and not part of the public API surface, though it is reachable as `hcnn::ThreadPool` for power users.

## Dependencies

No external dependencies beyond the C++ standard library.
