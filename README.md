# HypercubeCNN

[![CI](https://github.com/dliptak001/HypercubeCNN/actions/workflows/ci.yml/badge.svg)](https://github.com/dliptak001/HypercubeCNN/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)]()
[![CMake](https://img.shields.io/badge/CMake-3.21+-blue.svg)]()

A convolutional neural network that operates on Boolean hypercubes instead of spatial grids -- the same XOR-addressed topology used by [HypercubeRC](https://github.com/dliptak001/HypercubeRC) and [HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield), now with learned convolution kernels and end-to-end backpropagation.

## What is HypercubeCNN?

Standard CNNs convolve over 2D pixel grids, exploiting spatial locality with sliding kernels. HypercubeCNN replaces the grid with a DIM-dimensional binary hypercube (N = 2^DIM vertices) and replaces the spatial kernel with a Hamming-distance kernel: each vertex has exactly DIM nearest neighbors, reached by flipping a single bit (one XOR), and the convolution learns one weight per flip direction. This is the direct analogue of a 3x3 kernel shared across all pixel positions -- except the geometry is bitwise, not spatial.

A clarification on terminology: "Boolean hypercube" refers to the *topology* -- vertices are addressed by DIM-bit binary indices, and connectivity is defined by bitwise operations on those indices. The *values* stored at each vertex are ordinary floating-point scalars (activations in [-1, 1]), not bits. The hypercube is the graph that data lives on, not a constraint on the data itself.

Why this topology? The binary hypercube is vertex-transitive: every vertex looks structurally identical to every other. Weight sharing is not an approximation forced by implementation convenience (as it arguably is at image boundaries in spatial CNNs) -- it is mathematically exact, respecting the symmetry group Z_2^n. All topology is implicit in the bit representation of vertex indices. There are no adjacency lists, no padding, no border effects, and neighbor lookup is a single XOR instruction.

Pooling pairs each vertex with its bitwise complement -- the maximally distant point on the hypercube -- and reduces DIM by 1, producing a perfect (DIM-1)-dimensional sub-hypercube. Stacking conv + pool stages builds a feature hierarchy analogous to standard CNN architectures, with DIM shrinking and channel count growing at each stage.

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
std::vector<float> logits(net.GetNumClasses());
net.Embed(input_data, input_len, embedded.data());
net.Forward(embedded.data(), logits.data());
```

`hcnn::HCNN` is the canonical SDK front door — a single class that wraps the entire pipeline (embed → conv/pool → readout). All public symbols live in `namespace hcnn`. Available as a CMake static library via `FetchContent` or `find_package`. See [docs/CPP_SDK.md](docs/CPP_SDK.md) for full API reference and integration guide.

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
Readout (HCNNReadout) -- global average per channel -> linear -> logits
```

## Build targets

| Target | Purpose |
|--------|---------|
| `HypercubeCNNCore` | Static library (HCNN front door + core layers) |
| `HypercubeCNN` | Quick check runner (main.cpp) |
| `MNISTTrain` | MNIST training demo (examples/mnist_train.cpp) |
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
  HCNNReadout.h/cpp       GAP + linear readout
  ThreadPool.h            Header-only fork-join pool
  main.cpp                Quick check runner
  dataloader/             MNIST dataset loader (in-tree example utility)
  examples/               Training demos
  tests/                  HCNN SDK smoke test
  docs/                   Architecture, SDK reference, concept
  cmake/                  Package config template
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/CPP_SDK.md](docs/CPP_SDK.md) | C++ SDK API reference and integration guide |
| [docs/architecture.md](docs/architecture.md) | Full technical architecture |
| [docs/concept.md](docs/concept.md) | Design motivation and research context |
| [docs/original_work.md](docs/original_work.md) | Prior art survey |
| [docs/mnist.md](docs/mnist.md) | MNIST benchmark results and analysis |
| [examples/mnist_train.md](examples/mnist_train.md) | MNIST example walkthrough |

## Results

MNIST is used as a validation benchmark, not a leaderboard target -- the goal is to confirm that the hypercube convolution learns meaningful features via standard backpropagation, not to compete with spatial CNNs on a task that inherently favors 2D locality.

**MNIST** (no spatial inductive bias): **98.10%** test accuracy with ~200K parameters, 4 conv+pool stages, SGD with momentum, cosine LR annealing. The network learns digit features from hypercube topology alone -- no 2D spatial locality is encoded.

See [docs/mnist.md](docs/mnist.md) for full results and analysis.

## License

Apache 2.0. See [LICENSE](LICENSE).
