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
  - [HCNNNetwork](#hcnnnetwork)
  - [HCNN (conv layer)](#hcnn-conv-layer)
  - [HCNNPool](#hcnnpool)
  - [HCNNReadout](#hcnnreadout)
- [Memory layout](#memory-layout)
- [Threading](#threading)
- [Dependencies](#dependencies)

## What's in the SDK

After installation, the SDK contains:

```
<prefix>/
  include/HypercubeCNN/
    HCNNNetwork.h      -- Primary public API (network orchestrator)
    HCNN.h             -- Conv layer (included by HCNNNetwork.h)
    HCNNPool.h         -- Pooling layer (included by HCNNNetwork.h)
    HCNNReadout.h      -- Readout layer (included by HCNNNetwork.h)
    ThreadPool.h       -- Internal threading (included by HCNNNetwork.h)
    HCNNDataset.h      -- MNIST dataloader (example utility, not core API)
  lib/
    libHypercubeCNNCore.a
  lib/cmake/HypercubeCNN/
    HypercubeCNNConfig.cmake
    HypercubeCNNTargets.cmake
    HypercubeCNNConfigVersion.cmake
```

Consumers include `"HCNNNetwork.h"` and link against `HypercubeCNNCore`. The remaining headers are pulled in transitively -- most consumers only interact with `HCNNNetwork` directly.

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
    GIT_REPOSITORY https://github.com/dliptak001/HypercubeCNNStaging.git
    GIT_TAG        main
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
#include "HCNNNetwork.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    // Build a small network: DIM=6, N=64 vertices, 4 classes
    HCNNNetwork net(6, /*num_classes=*/4);
    net.add_conv(16, /*use_relu=*/true, /*use_bias=*/true);
    net.add_pool(PoolType::MAX);   // DIM 6->5, N 64->32
    net.add_conv(32, true, true);
    net.add_pool(PoolType::MAX);   // DIM 5->4, N 32->16
    net.randomize_all_weights();   // Xavier/Glorot init

    // Generate random input in [-1, 1]
    const int N = net.get_start_N();  // 64
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> input(N);
    for (auto& v : input) v = dist(rng);

    // Forward pass
    std::vector<float> embedded(N);
    net.embed_input(input.data(), N, embedded.data());

    int K = net.get_num_classes();  // 4
    std::vector<float> logits(K);
    net.forward(embedded.data(), logits.data());

    std::cout << "Logits:";
    for (int i = 0; i < K; ++i) std::cout << " " << logits[i];
    std::cout << "\n";

    return 0;
}
```

## API Reference

### Enums

| Enum | Values | Defined in | Description |
|------|--------|------------|-------------|
| `ReadoutType` | `GAP`, `FLATTEN` | HCNNNetwork.h | Readout strategy. GAP: global average pooling per channel (translation-invariant). FLATTEN: concatenate all channel x vertex activations (position-sensitive). |
| `PoolType` | `MAX`, `AVG` | HCNNPool.h | Antipodal pooling reduction. MAX: keep the larger value. AVG: average the pair. |

### HCNNNetwork

The primary public API. Orchestrates the full pipeline: input embedding, conv layers, pool layers, and readout.

Non-copyable, non-movable.

#### Constructor

```cpp
HCNNNetwork(int start_dim, int num_classes = 10,
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

#### Network construction

```cpp
void add_conv(int c_out, bool use_relu = true, bool use_bias = true);
void add_pool(PoolType type = PoolType::MAX);
void randomize_all_weights(float scale = 0.0f);
```

| Method | Description |
|--------|-------------|
| `add_conv` | Append a convolutional layer with `c_out` output channels. K = current DIM (one weight per Hamming-distance-1 neighbor). |
| `add_pool` | Append an antipodal pooling layer. Reduces DIM by 1. |
| `randomize_all_weights` | Initialize all weights. `scale > 0`: uniform [-scale, +scale]. `scale <= 0` (default): Xavier/Glorot uniform per layer. |

Call `add_conv` and `add_pool` to build the architecture, then `randomize_all_weights` before training.

#### Single-sample inference

```cpp
void embed_input(const float* raw_input, int input_length,
                 float* first_layer_activations) const;
void forward(const float* first_layer_activations, float* logits) const;
```

| Method | Description |
|--------|-------------|
| `embed_input` | Map a flat scalar array onto N = 2^DIM hypercube vertices via Direct Linear Assignment. Values must be in [-1.0, 1.0]. Output buffer must hold N floats. |
| `forward` | Run all conv/pool/readout layers. Input: embedded activations (N floats). Output: raw logits (num_classes floats). |

#### Batch inference (parallel)

```cpp
void forward_batch(const float* const* raw_inputs, const int* input_lengths,
                   int batch_size, float* logits_out);
```

Embed + forward for multiple samples in parallel using the thread pool. `logits_out` must hold `batch_size * num_classes` floats. Per-thread buffers are lazily allocated on first call and reused.

#### Single-sample training

```cpp
void train_step(const float* raw_input, int input_length,
                int target_class, float learning_rate, float momentum = 0.0f,
                float weight_decay = 0.0f,
                const float* class_weights = nullptr);
```

Full forward + backward + SGD update for one sample. `class_weights` is an optional array of length `num_classes` for per-class loss scaling (nullptr = uniform).

#### Mini-batch training (parallel)

```cpp
void train_batch(const float* const* inputs, const int* input_lengths,
                 const int* targets, int batch_size,
                 float learning_rate, float momentum = 0.0f,
                 float weight_decay = 0.0f,
                 const float* class_weights = nullptr);
```

Process `batch_size` samples in parallel. Each thread runs forward + backward independently; gradients are reduced (averaged) then applied in a single weight update. Per-thread buffers are lazily allocated and reused.

#### Accessors

| Method | Returns |
|--------|---------|
| `get_start_dim()` | Initial hypercube dimension. |
| `get_start_N()` | Initial vertex count (2^start_dim). |
| `get_input_channels()` | Number of input channels. |
| `get_num_classes()` | Number of output classes. |
| `get_conv(i)` | Reference to the i-th conv layer (`HCNN&`). |
| `get_readout()` | Reference to the readout layer (`HCNNReadout&`). |
| `get_num_conv()` | Number of conv layers. |
| `get_num_pool()` | Number of pool layers. |
| `get_layer_types()` | Layer ordering: true = conv, false = pool. |
| `get_channel_counts()` | Channel count after each layer (including input). |

### HCNN (conv layer)

Hypercube convolutional layer. Each output vertex is a weighted sum of K = DIM nearest neighbors (single-bit XOR flips) from each input channel, plus optional bias and ReLU.

Most consumers interact with conv layers only through `HCNNNetwork`. Direct access via `get_conv(i)` is available for weight inspection, serialization, or custom training loops.

#### Constructor

```cpp
HCNN(int dim, int c_in, int c_out, bool use_relu = true, bool use_bias = true);
```

Requires `dim >= 3`. Kernel weights are initialized to zero; call `randomize_weights()` or use `HCNNNetwork::randomize_all_weights()`.

#### Forward / backward

```cpp
void forward(const float* in, float* out, float* pre_act = nullptr) const;
void backward(const float* grad_out, const float* in, const float* pre_act,
              float* grad_in, float learning_rate, float momentum = 0.0f,
              float weight_decay = 0.0f);
```

`forward`: input [c_in * N], output [c_out * N], optional pre_act [c_out * N] (needed by backward).

`backward`: computes grad_in (if non-null) and updates weights via momentum SGD in one call.

#### Gradient computation (for custom training)

```cpp
void compute_gradients(const float* grad_out, const float* in, const float* pre_act,
                       float* grad_in, float* kernel_grad, float* bias_grad,
                       float* work_buf = nullptr) const;
void apply_gradients(const float* kernel_grad, const float* bias_grad,
                     float learning_rate, float momentum, float weight_decay = 0.0f);
```

`compute_gradients`: writes raw gradients to caller buffers without modifying weights. Used by mini-batch training (gradients averaged across samples before applying).

`apply_gradients`: apply externally computed gradients via momentum SGD.

#### Accessors

| Method | Returns |
|--------|---------|
| `get_dim()` | Hypercube dimension. |
| `get_N()` | Vertex count (2^DIM). |
| `get_c_in()` | Input channels. |
| `get_c_out()` | Output channels. |
| `get_K()` | Number of connection masks (= DIM). |
| `get_kernel_data()` | Raw pointer to kernel weights [c_out * c_in * K]. |
| `get_kernel_size()` | Total kernel weight count. |
| `get_bias_data()` | Raw pointer to bias [c_out] (or empty if bias disabled). |
| `get_bias_size()` | Bias element count (0 if disabled). |

### HCNNPool

Antipodal pooling layer. Pairs each vertex v with its bitwise complement `v ^ (2^DIM - 1)`, reducing DIM by 1.

```cpp
HCNNPool(int input_dim, PoolType type = PoolType::MAX);
void forward(const float* in, float* out, int num_channels,
             std::vector<int>* max_indices = nullptr) const;
void backward(const float* grad_out, float* grad_in, int num_channels,
              const std::vector<int>* max_indices) const;
```

| Method | Returns |
|--------|---------|
| `get_input_dim()` | Input dimension. |
| `get_output_dim()` | Output dimension (input_dim - 1). |
| `get_input_N()` | Input vertex count. |
| `get_output_N()` | Output vertex count (input_N / 2). |

### HCNNReadout

Global average pooling per channel followed by a linear layer to class logits.

```cpp
HCNNReadout(int num_classes, int input_channels);
void forward(const float* in, float* out, int N,
             float* work_buf = nullptr) const;
void backward(const float* grad_logits, const float* in, int N,
              float* grad_in, float learning_rate, float momentum = 0.0f,
              float weight_decay = 0.0f);
void compute_gradients(const float* grad_logits, const float* in, int N,
                       float* grad_in, float* weight_grad, float* bias_grad,
                       float* work_buf = nullptr) const;
void apply_gradients(const float* weight_grad, const float* bias_grad,
                     float learning_rate, float momentum, float weight_decay = 0.0f);
```

| Method | Returns |
|--------|---------|
| `get_num_classes()` | Number of output classes. |
| `get_input_channels()` | Number of input channels. |
| `get_weight_data()` | Raw pointer to weights [num_classes * input_channels]. |
| `get_weight_size()` | Total weight count. |
| `get_bias_data()` | Raw pointer to bias [num_classes]. |
| `get_bias_size()` | Bias element count. |

## Memory layout

All activations use channel-major layout:

```
activations[c * N + v]    // channel c, vertex v
```

where N = 2^DIM at the current layer. Conv input/output, pool input/output, and readout input all follow this convention. Logits output from readout is a flat array of `num_classes` floats.

Input values must be in [-1.0, 1.0]. The embedding maps the first min(input_length, N) scalars to vertices 0, 1, 2, ...; remaining vertices are zero-padded.

## Threading

`HCNNNetwork` owns a fork-join thread pool (auto-sized or caller-specified). Three threading strategies are used, never nested:

- **Batch training** (`train_batch`): samples run forward+backward in parallel, gradients reduced and applied.
- **Batch inference** (`forward_batch`): samples run forward in parallel.
- **Per-layer vertex threading**: parallelizes the vertex loop within each conv layer. Only activates at DIM >= 12. Automatically disabled during batch dispatch.

The thread pool is internal and not part of the public API.

## Dependencies

No external dependencies beyond the C++ standard library.
