# HypercubeCNN

[![CI](https://github.com/dliptak001/HypercubeCNN/actions/workflows/ci.yml/badge.svg)](https://github.com/dliptak001/HypercubeCNN/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)]()
[![CMake](https://img.shields.io/badge/CMake-3.21+-blue.svg)]()

A convolutional neural network that operates on Boolean hypercubes instead of spatial grids -- the same XOR-addressed topology used by [HypercubeRC](https://github.com/dliptak001/HypercubeRC) and [HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield), with learned convolution kernels and end-to-end backpropagation.

## What is HypercubeCNN?

Standard CNNs convolve over 2D pixel grids, exploiting spatial locality with sliding kernels. HypercubeCNN replaces the grid with a DIM-dimensional binary hypercube (N = 2^DIM vertices) and replaces the spatial kernel with a Hamming-distance kernel: each vertex has exactly DIM nearest neighbors, reached by flipping a single bit (one XOR), and the convolution learns one weight per flip direction. This is the direct analogue of a 3x3 kernel shared across all pixel positions -- except the geometry is bitwise, not spatial.

A clarification on terminology: "Boolean hypercube" refers to the *topology* -- vertices are addressed by DIM-bit binary indices, and connectivity is defined by bitwise operations on those indices. The *values* stored at each vertex are ordinary floating-point scalars (activations in [-1, 1]), not bits. The hypercube is the graph that data lives on, not a constraint on the data itself.

Why this topology? The binary hypercube is vertex-transitive: every vertex looks structurally identical to every other. Weight sharing is not an approximation forced by implementation convenience (as it arguably is at image boundaries in spatial CNNs) -- it is mathematically exact, respecting the symmetry group Z_2^n. All topology is implicit in the bit representation of vertex indices. There are no adjacency lists, no padding, no border effects, and neighbor lookup is a single XOR instruction.

Pooling pairs each vertex with its bitwise complement -- the maximally distant point on the hypercube -- and reduces DIM by 1, producing a perfect (DIM-1)-dimensional sub-hypercube. Stacking conv + pool stages builds a feature hierarchy analogous to standard CNN architectures, with DIM shrinking and channel count growing at each stage.

The architecture supports both classification (softmax + cross-entropy) and regression (MSE) via a unified conv/pool/readout pipeline -- only the loss gradient differs. Activations include ReLU, LeakyReLU, and tanh (the natural choice for bounded-output regression and reservoir-computing readouts).

## Quick start (C++)

```cpp
#include "HCNN.h"

using namespace hcnn;

HCNN net(10);                       // DIM=10, N=1024
net.AddConv(32);                    // 1->32 channels, K=10
net.AddPool(PoolType::MAX);         // DIM 10->9, N 1024->512
net.AddConv(64);                    // 32->64 channels, K=9
net.AddPool(PoolType::MAX);         // DIM 9->8, N 512->256
net.RandomizeWeights();             // Xavier/He init

// Forward pass -- caller-owned scratch buffers, designed for reuse.
std::vector<float> embedded(net.GetStartN());
std::vector<float> logits(net.GetNumOutputs());
net.Embed(input_data, input_len, embedded.data());
net.Forward(embedded.data(), logits.data());
```

`hcnn::HCNN` is the canonical SDK front door -- a single class that wraps the entire pipeline (embed → conv/pool → readout). All public symbols live in `namespace hcnn`. Available as a CMake static library via `FetchContent` or `find_package`. See [docs/CPP_SDK.md](docs/CPP_SDK.md) for full API reference and integration guide.

## Pipeline

```
Input (flat scalars in [-1, 1])
  |
  v
Embed onto 2^DIM hypercube vertices (Direct Linear Assignment)
  |
  v
Conv (HCNNConv) -- K=DIM XOR masks, one weight per neighbor direction
  |
  v
Pool (HCNNPool) -- antipodal pairing, DIM -> DIM-1
  |
  v
[repeat conv + pool stages]
  |
  v
Readout (HCNNReadout) -- flatten all (channel, vertex) activations -> linear -> output
```

## Readout-throughput optimization roadmap

When HCNN is used as the learned readout for [HypercubeRC](https://github.com/dliptak001/HypercubeRC), HCNN consumes >95% of the combined system's training compute -- the reservoir itself is essentially free in comparison. Thread parallelism inside HCNN is already saturated (block-pair auto-vectorization, vertex- and channel-level threading, per-thread gradient accumulators in `train_batch_impl`), so further wall-clock speedup must come from **algorithmic** changes rather than more threading.

The RC workload that drives this roadmap: `Conv(c_in=1 -> ch, TANH, bias) + MaxPool` x `min(DIM-2, 2)` stages, Adam optimizer, regression task via `TrainEpochRegression` -> `TrainBatchRegression` -> `train_batch_impl`. DIM range 5-16, scaling target DIM=14-16. The hot path is `compute_gradients` (per-sample, parallel) followed by `apply_gradients` (once per batch, serial). At DIM=14 the second conv layer (32 ch x 16 in x K=13 x N=8192) dominates wall time, and inside that, `std::tanh` on every output element in both forward and backward is the single most expensive primitive.

The options below are grouped by numerical risk. Each is independently selectable.

### Group A -- Safe (bit-identical or trivially close numerics)

**1. Use post-activation for the TANH derivative (the `1 - y^2` identity).** *Implemented.*
For `Activation::TANH`, `HCNNConv::backward` and `HCNNConv::compute_gradients` previously called `activate_derivative(pre_act[i])`, which internally re-ran `std::tanh`. The post-activation `y = tanh(x)` is already alive in `cache[i+1].activation` throughout backward, so the derivative is obtained as `1 - y*y` with no transcendental call. An optional `const float* post_act = nullptr` parameter on the two backward entry points selects the identity when supplied and the activation is TANH; the public `HCNN` API is unchanged.
- **Measured gain:** ~15% per-epoch wall-clock reduction at DIM=12 (3.55 s -> 3.03 s on `RegressionTimeseries`). Numerics bit-identical to the prior baseline.
- **Files:** `HCNNConv.h`, `HCNNConv.cpp`, `HCNNNetwork.cpp`.


### Group B -- Tradeoff (numerical changes; require A/B accuracy validation)

**3b. Compile-time fast TANH approximation.** *Implemented.*
Wrapped both `std::tanh` call sites in `HCNNConv.cpp` (forward `activate` and the `post_act==nullptr` fallback in `activate_derivative`) with a static `hcnn_tanh` helper guarded by `#ifdef HCNN_FAST_TANH`. When the macro is defined the helper uses a rational Padé approximation `tanh(x) ~= x * (27 + x^2) / (27 + 9*x^2)` with the output clamped to `[-1, 1]` (the rational form's asymptote is `x/9`, not 1, so a clamp is required for `|x| >~ 3`). The CMake option `HCNN_FAST_TANH` (default **ON**) attaches the macro as a `PRIVATE` `target_compile_definitions` on `HypercubeCNNCore`, so the public include interface is untouched and downstream consumers (e.g. HypercubeRC) inherit whichever variant is built into `libHypercubeCNNCore.a`. Pass `-DHCNN_FAST_TANH=OFF` to fall back to exact `std::tanh`. Because option 1's backward branch reuses the cached post-activation, the approximation propagates from forward to backward consistently with no separate code path.
- **Measured gain:** ~9% additional per-epoch wall-clock reduction at DIM=12 on top of option 1 (3.03 s -> 2.76 s on `RegressionTimeseries`); cumulative ~22% vs the pre-roadmap baseline.
- **Numerics:** final test 1-R^2 shifted from 9.92e-08 to 1.20e-07 on the DIM=12 timeseries fit -- still "seven nines of variance explained", well past any practical regression threshold.
- **Files:** `HCNNConv.cpp`, `CMakeLists.txt`.

**4. fp32 + Kahan compensation for kernel-gradient reduction.**
`conv_kernel_grad_one` (`HCNNConv.cpp:97-117`) currently accumulates in fp64 throughout to match the original numerical semantics bit-for-bit. Replacing the fp64 accumulator with Kahan-compensated fp32 halves SIMD-lane cost and roughly doubles effective vector width on AVX2/AVX-512. Kahan error grows as `O(eps)` independent of N -- substantially better than naive fp32 (`O(eps * N)`) and competitive with fp64 for the bounded reservoir-state magnitudes seen in RC training. Validate with `CoreSmokeTest` and `RegressionTimeseries` before merging.
- **Expected gain:** up to 2x on kernel-grad compute at DIM=14-16 where the fp64 reduction is hot.
- **Files:** `HCNNConv.cpp`.

### Group C -- Out of scope for this roadmap

Documented so the deferral is intentional, not forgotten:
- Simplifying `HCNNReadout` to a pure matmul (eliminating the `channel_avg` abstraction now that `N=1` always in the FLATTEN configuration).
- Hand-rolled AVX2/AVX-512 intrinsics on the block-pair inner loops -- the auto-vectorizer is already producing tight code on these paths (see the comment block at `HCNNConv.cpp:25-45, 81-92`); manual intrinsics risk regression for marginal gain.
- Loop fusion to reuse `grad_pre` across `c_in` in kernel-grad -- only beneficial when `grad_pre` spills L1 (`DIM >= 15` with large `c_out`); revisit if profiler shows climbing L1 miss rate.

### Status and measured gains (RC, DIM=12)

| # | Optimization | Group | Status | Per-epoch wall-clock |
|---|---|---|---|---|
| 1 | TANH `1 - y^2` backward | A | **Shipped** | -14.8% vs baseline (3.55 s -> 3.03 s) |
| 3b | Compile-time fast tanh (`HCNN_FAST_TANH`, default ON) | B | **Shipped** | -8.8% on top of 1 (3.03 s -> 2.76 s); cumulative -22% |
| 2 | Drop redundant zero-fills | A | Tried, reverted | No measurable gain at DIM=12 |
| 4 | fp32 + Kahan kernel-grad | B | Open | Predicted up to 2x on kernel-grad at DIM=14-16 |

## Build targets

| Target | Purpose |
|--------|---------|
| `HypercubeCNNCore` | Static library (HCNN front door + core layers) |
| `HypercubeCNN` | Quick check runner (main.cpp) |
| `MNISTTrain` | MNIST classification demo (examples/mnist_train.cpp) |
| `RegressionTimeseries` | Regression demo -- next-step prediction (examples/regression_timeseries.cpp) |
| `CoreSmokeTest` | HCNN SDK smoke test (tests/CoreSmokeTest.cpp) |

## Building from source

Requirements: C++23 compiler, CMake 3.21+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run the smoke test:

```bash
./build/CoreSmokeTest
```

## Project structure

```
HypercubeCNN/
  HCNN.h/cpp              Top-level pipeline wrapper (canonical SDK API)
  HCNNNetwork.h/cpp       Internal orchestrator (re-exported via HCNN.h)
  HCNNConv.h/cpp          Conv layer
  HCNNPool.h/cpp          Antipodal pooling
  HCNNReadout.h/cpp       Linear readout
  ThreadPool.h            Header-only fork-join pool
  main.cpp                Quick check runner
  dataloader/             MNIST dataset loader (in-tree example utility)
  examples/               Training demos
  tests/                  HCNN SDK smoke test
  docs/                   Architecture and SDK reference
  cmake/                  Package config template
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/CPP_SDK.md](docs/CPP_SDK.md) | C++ SDK API reference and integration guide |
| [docs/architecture.md](docs/architecture.md) | Full technical architecture |
| [examples/mnist_train.md](examples/mnist_train.md) | MNIST classification example, benchmark results, and analysis |
| [examples/regression_timeseries.md](examples/regression_timeseries.md) | Regression example, DIM=12 results, and HypercubeRC integration notes |

## Results

Both benchmarks validate that hypercube convolution learns meaningful features via standard backpropagation -- they are not leaderboard targets.

**Classification -- MNIST** (no spatial inductive bias): **98.07%** test accuracy with ~84K parameters, 2 conv layers + 1 pool stage, Adam optimizer, cosine LR annealing. The network learns digit features from hypercube topology alone -- no 2D spatial locality is encoded. See [examples/mnist_train.md](examples/mnist_train.md).

**Regression -- time-series prediction** (DIM=12, N=4,096 vertices): **1-R² = 9.9e-8** (seven nines of variance explained) predicting the next value of a sine wave from a 4,096-dimensional synthetic reservoir state, with 19,425 parameters. Validates HCNN as a learned readout layer for reservoir computing. See [examples/regression_timeseries.md](examples/regression_timeseries.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
