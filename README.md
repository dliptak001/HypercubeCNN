# HypercubeCNN

[![CI](https://github.com/dliptak001/HypercubeCNN/actions/workflows/ci.yml/badge.svg)](https://github.com/dliptak001/HypercubeCNN/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)]()
[![CMake](https://img.shields.io/badge/CMake-3.21+-blue.svg)]()

HypercubeCNN is a **convolutional neural network that lives on a Boolean hypercube** instead of a 2D pixel grid. It keeps the familiar CNN story — shared local kernels, stacked layers, end-to-end backpropagation — but replaces spatial neighborhoods with **bitwise (XOR) geometry**. The library is **pure C++23**, ships as a static SDK (`HypercubeCNNCore`), and is meant for research, coursework, and as a **learned readout** beside [HypercubeESN](https://github.com/dliptak001/HypercubeESN) and [HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield).

---

## What is HypercubeCNN?

A standard CNN slides a small kernel (e.g. 3×3) across an image, reusing the same weights at every location. **HypercubeCNN does the same thing on a different domain:** a `DIM`-dimensional binary hypercube with **N = 2^DIM** vertices. Each vertex has a **self** (center) contribution plus exactly **DIM** nearest neighbors, reached by flipping one bit of its address (`v ^ (1 << k)`). The layer learns one weight per tap — **`K = DIM + 1`** total — shared across all vertices.

| Spatial CNN | HypercubeCNN |
|-------------|--------------|
| Pixels on a rectangle | Vertices of a `DIM`-cube (`N = 2^DIM`) |
| Neighbors by grid offset | Neighbors by **bit flip** + **self** |
| Shared 3×3 (center + edges) | Shared **`K = DIM + 1`** taps (self + one weight per bit axis) |
| Borders, padding, edge cases | No borders — the cube is **vertex-transitive** |

**Terminology:** “Boolean hypercube” names the **topology** — indices are DIM-bit integers and connectivity is bitwise. Values on vertices are ordinary **floats** (typically in `[-1, 1]`), not bits. The cube is the graph data lives on, not a constraint that activations be binary.

Because every vertex looks the same under the symmetry group of the cube, weight sharing is mathematically exact (not a convenience that breaks at image borders). Neighbor lookup is a single XOR; there are no adjacency lists.

You stack convolutions (and optional **antipodal** pooling) into a feature body, then a **single linear readout** over every final `(channel, vertex)`. Classification (softmax cross-entropy) and regression (MSE) share that forward path; only the training loss and target type change.

**A good fit when:**

- Inputs already live at length `2^D` (reservoir / ESN state, bit-indexed fingerprints, product-space features)
- You want a small, dependency-free C++ CNN core with a clean teaching surface
- You are exploring what “convolution” means outside Euclidean grids

**Not a drop-in replacement for spatial vision stacks:** mapping images onto the cube is an explicit packing step (the MNIST demo does this for you). Hamming neighbors are **not** automatic 2D adjacency unless packing makes them so.

---

## 60-second tour

```text
raw floats  →  Embed onto N vertices
            →  Conv*  (optional antipodal Pool*)
            →  FLATTEN linear head  →  logits / predictions
```

- **Body** can be multilayer (stack `AddConv` / `AddPool`).
- **Head** is a **single linear** map over every final `(channel, vertex)` — no MLP classifier.
- **Pool** (optional) is **antipodal**: pair each vertex with its complement at max Hamming distance, reduce DIM by 1. Not 2×2 spatial pooling.
- **Tasks:** classification (softmax CE) or regression (MSE); same forward path, different loss + train API.

```cpp
#include "HCNN.h"
using namespace hcnn;

HCNN net(/*DIM=*/10, /*num_outputs=*/10);  // N = 1024
net.AddConv(32);                           // K = 11 (self + 10 bit axes)
net.AddPool(PoolType::MAX);                // DIM 10→9, N 1024→512
net.AddConv(64);                           // K = 10 at DIM 9
net.RandomizeWeights();
net.SetOptimizer(OptimizerType::ADAM);     // recommended for demos

std::vector<float> emb(net.GetStartN()), logits(net.GetNumOutputs());
net.Embed(input, input_len, emb.data());
net.Forward(emb.data(), logits.data());    // raw logits — no softmax
```

Full student-oriented API: **[docs/CPP_SDK.md](docs/CPP_SDK.md)**.

---

## Quick start

**Needs:** C++23, CMake ≥ 3.21.

```bash
git clone https://github.com/dliptak001/HypercubeCNN.git
cd HypercubeCNN
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/CoreSmokeTest          # or build/CoreSmokeTest.exe on Windows
```

### Use as a dependency (FetchContent)

```cmake
include(FetchContent)
FetchContent_Declare(
    HypercubeCNN
    GIT_REPOSITORY https://github.com/dliptak001/HypercubeCNN.git
    GIT_TAG        v0.1.0          # pin a tag or commit you trust
)
FetchContent_MakeAvailable(HypercubeCNN)
target_link_libraries(my_app PRIVATE HypercubeCNNCore)
```

Include `"HCNN.h"`, symbols in `namespace hcnn`.

---

## Teaching demos

Config lives at the top of each example (`DemoConfig`); thin train loops use optional helpers (`HCNNTrainHelpers`, spatial preprocess for images).

| Target | What it teaches | Write-up |
|--------|-----------------|----------|
| **`CoreSmokeTest`** | Front-door API contract | `tests/CoreSmokeTest.cpp` |
| **`MNISTTrain`** | Image → DualPlane pack → classify | [examples/mnist_train.md](examples/mnist_train.md) |
| **`RegressionTimeseries`** | Length-N state → scalar next-step | [examples/regression_timeseries.md](examples/regression_timeseries.md) |

```bash
cmake --build build --target MNISTTrain RegressionTimeseries
# MNIST: place IDX files under data/ (see mnist_train.md)
```

**How to read the numbers:** demos prove the stack **learns** end-to-end. They are **not** leaderboard claims. MNIST uses engineered packing + aug + a large FLATTEN head — not “free” 2D CNN structure. Regression uses a synthetic uncoupled reservoir; near-perfect R² is an API smoke signal, not HypercubeESN production skill. Details and current recipes live in the example docs (they track the code more tightly than this page).

---

## Repository map

```text
HCNN.h / .cpp              SDK front door (start here)
HCNNConv / Pool / Readout  Layers (re-exported via HCNN.h)
HCNNSpatial*               Optional 2D aug + embed (images)
HCNNTrainHelpers.*         Optional metrics, cosine LR, checkpoints
examples/                  Teaching demos + demo_arch.h
tests/CoreSmokeTest.cpp    Smoke tests for the public API
docs/CPP_SDK.md            Canonical SDK guide
docs/architecture.md       Implementation depth
docs/report.md             Concept, applications, pitfalls
```

CMake library target: **`HypercubeCNNCore`**. Optional targets: `MNISTTrain`, `RegressionTimeseries`, `CoreSmokeTest`, `HypercubeCNN` (quick runner).

---

## Documentation

| Doc | Role |
|-----|------|
| **[docs/CPP_SDK.md](docs/CPP_SDK.md)** | Onboarding + API + educational train loops |
| [docs/architecture.md](docs/architecture.md) | Geometry, training cores, threading |
| [docs/report.md](docs/report.md) | Concept, where it shines / fails |
| [docs/spatial_preprocess.md](docs/spatial_preprocess.md) | Image aug/embed (pad contracts) |
| [docs/train_helpers.md](docs/train_helpers.md) | Metrics and checkpoints |
| [examples/mnist_train.md](examples/mnist_train.md) | Classification demo |
| [examples/regression_timeseries.md](examples/regression_timeseries.md) | Regression demo |

---

## Ecosystem

- **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)** — echo-state / reservoir computing on the same topology; HCNN is a natural **learned readout**.
- **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)** — Hopfield-style dynamics on the cube.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
