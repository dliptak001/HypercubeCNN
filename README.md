# HypercubeCNN

[![CI](https://github.com/dliptak001/HypercubeCNN/actions/workflows/ci.yml/badge.svg)](https://github.com/dliptak001/HypercubeCNN/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)]()
[![CMake](https://img.shields.io/badge/CMake-3.21+-blue.svg)]()

HypercubeCNN is a **dependency-free C++23 hypercube CNN core** for research and systems integration (including [HypercubeESN](https://github.com/dliptak001/HypercubeESN)). The public surface is **small and contract-driven** so it stays usable in production hosts and legible for engineers learning the stack.

It is a **convolutional neural network that lives on a Boolean hypercube** instead of a 2D pixel grid: familiar CNN structure (shared local kernels, stacked layers, end-to-end backpropagation) with **bitwise (XOR) geometry** in place of spatial neighborhoods. Ships as the static library **`HypercubeCNNCore`**. Also used beside [HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield).

**Examples stay examples. Helpers stay optional. Neither is the product’s reason for existing** — the core (`HCNN`) is.

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

**Capacity is always a power of two.** That is not a convenience limit you can turn off — it is the hypercube. For each channel the network holds **N = 2^DIM** vertices. Arbitrary-length or non-square data is a **host packing problem**: pad, resize, hash, dual-plane layout, reservoir indexing, or any other map you invent into length ≤ N (then pass full capacity or use the built-in zero-pad for short tails). Optional spatial helpers are one recipe for images; the core does not invent a packing for you and will not grow a non–power-of-two “input size” knob.

**A good fit when:**

- Inputs already live at length `2^D` (reservoir / ESN state, bit-indexed fingerprints, product-space features), or you are willing to own a pack into N
- You need a small, dependency-free C++ CNN core with a stable public API
- You are exploring what “convolution” means outside Euclidean grids

**Not a drop-in replacement for spatial vision stacks:** mapping images onto the cube is an explicit packing step (the MNIST example does this for you). Hamming neighbors are **not** automatic 2D adjacency unless packing makes them so.

---

## SDK layers

| Layer | Role |
|-------|------|
| **Core (`HCNN`)** | The product. Integration surface — documented and versioned like a library another binary links. |
| **Arch / spatial / train helpers** | Optional products for hosts that want them. |
| **Examples** | Proof of contracts + recipes (MNIST, regression). Not the definition of the SDK. |
| **Internals** | Research / maintainers; private headers are not installed and not for apps. |

Include `"HCNN.h"` for the core only, or `"HypercubeCNN.h"` for the full public stack (arch + helpers + spatial). Link target: **`HypercubeCNNCore`**.

Full API guide: **[docs/CPP_SDK.md](docs/CPP_SDK.md)**.

### Python SDK

Package **`hypercube-cnn`** (`import hypercube_cnn`): same core contracts, NumPy
train/infer, arch JSON + HCNW model I/O, spatial embed/aug, metrics / cosine LR,
pickle secondary.

```bash
pip install hypercube-cnn   # when published; or pip install . from this repo
```

- API: **[docs/Python_SDK.md](docs/Python_SDK.md)**
- In-repo recipes: **[examples/python/](examples/python/)**
- Wheels workflow: **[.github/workflows/wheels.yml](.github/workflows/wheels.yml)**

---

## 60-second tour

```text
raw floats  →  Embed onto N vertices
            →  Conv*  (optional antipodal Pool*)
            →  FLATTEN linear head  →  logits / predictions
```

- **Body** can be multilayer (stack `AddConv` / `AddPool`).
- **Head** is a **single linear** map over every final `(channel, vertex)` (FLATTEN; no GAP, no MLP).
- **Pool** (optional) is **antipodal**: pair each vertex with its complement at max Hamming distance, reduce DIM by 1. Not 2×2 spatial pooling.
- **Tasks:** classification (softmax CE) or regression (MSE); same forward path, different loss + train API.

```cpp
#include "HypercubeCNN.h"   // or just "HCNN.h" for the core only
using namespace hcnn;

HCNN net(/*DIM=*/10, /*num_outputs=*/10);  // N = 1024; default optimizer = Adam
net.AddConv(32);                           // K = 11 (self + 10 bit axes)
net.AddPool(PoolType::MAX);                // DIM 10→9, N 1024→512
net.AddConv(64);                           // K = 10 at DIM 9
net.RandomizeWeights();

std::vector<float> logits(net.GetNumOutputs());
net.Predict(input, input_len, logits.data());  // embed + forward (raw logits)
int pred = net.PredictClass(input, input_len); // classification only
```

### Minimal train loop (helpers)

```cpp
#include "HypercubeCNN.h"
using namespace hcnn;

HCNN net(6, /*classes=*/4);
net.AddConv(16);
net.RandomizeWeights();

// Contiguous row-major train_x / train_y, test_x / test_y already packed...
TrainParams tp;
tp.learning_rate = 1e-3f;
tp.weight_decay  = 1e-3f;

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
    GIT_TAG        v1.0.0          # pin a release tag (see GitHub Releases)
)
FetchContent_MakeAvailable(HypercubeCNN)
target_link_libraries(my_app PRIVATE HypercubeCNNCore)
```

Include `"HypercubeCNN.h"` (full public stack) or `"HCNN.h"` (core only). Symbols live in `namespace hcnn`.

---

## Examples (recipes)

Config lives at the top of each example (`DemoConfig`); optional helpers thin the train loop (`HCNNTrainHelpers`, spatial preprocess for images).

| Target | What it demonstrates | Write-up |
|--------|----------------------|----------|
| **`CoreSmokeTest`** | Front-door API contract | `tests/CoreSmokeTest.cpp` |
| **`MNISTTrain`** | Image → DualPlane pack → classify | [examples/mnist_train.md](examples/mnist_train.md) |
| **`RegressionTimeseries`** | Length-N state → scalar next-step | [examples/regression_timeseries.md](examples/regression_timeseries.md) |

**MNIST (closed recipe):** ~**99.44%** mean best test accuracy (3 weight seeds; peak **99.42–99.46%**) with DualPlane embed + train aug — not a leaderboard claim; full recipe and tables in [`examples/mnist_train.md`](examples/mnist_train.md).

```bash
cmake --build build --target MNISTTrain RegressionTimeseries
# MNIST: place IDX files under data/ (see mnist_train.md)
```

**How to read the numbers:** examples prove the stack **learns** end-to-end. They are **not** SOTA claims. MNIST uses engineered packing + aug + a large FLATTEN head — not “free” 2D CNN structure. Regression uses a synthetic uncoupled reservoir; near-perfect R² is an API smoke signal, not HypercubeESN production skill. Details live in the example docs.

---

## Repository map

```text
HypercubeCNN.h             Umbrella include (core + arch + helpers + spatial)
HCNN.h / HCNNTypes.h       Public front door + enums
HCNNInput.h                Full-capacity HCNNInputView / Batch (spatial-safe)
HCNNArch.h                 LayerSpec, apply_arch, HCNNConfig::Build
HCNNTrainHelpers.*         Metrics, flat dataset, cosine LR, checkpoints, weight files
HCNNSpatial*               Optional 2D aug + embed (images)
HCNNNetwork / Conv / …     Private impl (not installed; not for apps)
examples/                  Recipes (MNIST, regression)
tests/CoreSmokeTest.cpp    Smoke tests for the public API
docs/CPP_SDK.md            Canonical SDK guide (integration-first)
docs/internals.md          Implementation depth (contributors)
```

CMake library target: **`HypercubeCNNCore`**. Optional targets: `MNISTTrain`, `RegressionTimeseries`, `CoreSmokeTest`.

---

## Documentation

| Doc | Role |
|-----|------|
| **[docs/CPP_SDK.md](docs/CPP_SDK.md)** | Public SDK: contracts, API, optional helpers, recipes |
| **[ChangeLog.md](ChangeLog.md)** | Release notes (v1.0.0 public SDK arc) |
| [docs/internals.md](docs/internals.md) | Training cores, threading, optimizers, build options |
| [docs/spatial_preprocess.md](docs/spatial_preprocess.md) | Image aug/embed (pad contracts) |
| [examples/mnist_train.md](examples/mnist_train.md) | Classification recipe |
| [examples/regression_timeseries.md](examples/regression_timeseries.md) | Regression recipe |

---

## Ecosystem

- **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)** — echo-state / reservoir computing on the same topology; HCNN is a natural **learned readout**.
- **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)** — Hopfield-style dynamics on the cube.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
