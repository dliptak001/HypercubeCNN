# HypercubeCNN

[![Build wheels](https://github.com/dliptak001/HypercubeCNN/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeCNN/actions/workflows/wheels.yml)

Python bindings for **HypercubeCNN** — a dependency-free convolutional neural
network whose feature map is a Boolean hypercube. Choose a dimension **DIM**; each channel then lives on
exactly `N = 2^DIM` vertices (for example DIM 6 → 64 sites, DIM 10 → 1024).
A local filter at a vertex reaches only that site and its nearest neighbors,
and every neighbor index is a single XOR on the binary address — no spatial
grid, no adjacency list, no stencil table to store. The activations stay
ordinary real-valued units (ReLU, tanh, …); only the *topology* is binary, so
capacity is power-of-two by construction and packing non-cube data is host work.

You stack local layers, train for classification or regression, and save weights
with a small architecture sidecar. Optional helpers map images onto length N;
data that already lives at `2^D` (reservoir or ESN state, fingerprints, product
features) can drive the network with no packing step.

Compared with a standard vision CNN, the shared-weight stack and end-to-end
training stay familiar — only the domain changes. A usual network slides a
window on a rectangle and pads the borders; here every site has the same
neighbors under the cube’s symmetry, with no image edge and no stencil table.
Pixels are not native: pack them onto the N sites first, then train on those
length-N inputs.

---

<p align="center">
  <strong>HypercubeAI ecosystem</strong><br/>
</p>

<p align="center">
  <a href="https://github.com/dliptak001/HypercubeESN"><strong>HypercubeESN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeCNN"><strong>HypercubeCNN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeHopfield"><strong>HypercubeHopfield</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWTF"><strong>HypercubeWTF</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeEtalon"><strong>HypercubeEtalon</strong></a>
</p>

HypercubeCNN is an experiment in the **HypercubeAI** project — our quest to
systematically re-implement classical neural architectures on a Boolean
hypercube topology instead of Euclidean grids or random graphs. The central
thesis is “topology-native intelligence”: the hypercube’s algebraic structure
(vertex-transitive symmetry, Hamming geometry, bitwise addressing) can serve
as a first-class computational substrate.

- **A topology you don’t store** — the graph is specified: connectivity is
  implicit in the vertex indices; with a seed and a few config scalars the whole
  reservoir reconstructs mathematically.
- **Perfect homogeneity** — every vertex has the same degree and the same local
  world, so local dynamics mean the same thing everywhere — no structural
  favorites baked in by a random graph.
- **Cheap navigation** — each neighbor is a few bit operations on the vertex
  index, not a pointer chase through a stored edge list, so walks stay
  arithmetic and cache-friendly.
- **Topology-native pairing** — the readout consumes the reservoir’s output with
  zero geometric distortion, and the learned kernels exploit the same locality
  that generated the dynamics. The data never leaves the hypercube it was born
  on.

Each product in the family is a different architecture on that same foundation.

---

## Installation

```bash
pip install hypercube-cnn
```

Pre-built wheels for Python 3.10–3.13 on Windows (x64), Linux (x86_64,
aarch64), and macOS (x86_64, arm64). No compiler required.

### From source

```bash
git clone https://github.com/dliptak001/HypercubeCNN.git
cd HypercubeCNN
pip install .
```

Requires Python 3.10+, a C++23 compiler, and CMake ≥ 3.21. On Windows with
CLion’s bundled MinGW, put that toolchain’s `bin` (and Ninja) on `PATH`, set
`CMAKE_GENERATOR=Ninja`, and point `CC`/`CXX` at the MinGW gcc/g++ — exact
install paths change with the CLion version. Then:

```bash
pip install . --no-build-isolation --force-reinstall --no-deps
```

## Quick start

```python
import numpy as np
import hypercube_cnn as hc

net = hc.HCNNConfig(
    dim=6,
    num_outputs=3,
    layers=[
        hc.LayerSpec.conv(8, bn=True),
        hc.LayerSpec.pool("max"),
        hc.LayerSpec.conv(8),
    ],
    weight_seed=1,
).build()

x = np.random.randn(net.N).astype(np.float32)  # full capacity N = 2**dim
logits = net.predict(x)
cls = net.predict_class(x)
net.train_step(x, target=0, params=hc.TrainParams(learning_rate=1e-3))
net.save("model")  # model.hcnw + model.arch.json
```

## Features

- **Core train/infer** — classification (CE) and regression (MSE); NumPy float32
- **Architecture product** — `LayerSpec` / `HCNNConfig`, export/import arch JSON
- **Model I/O** — HCNW weights + arch sidecar (C++ interop); pickle as secondary
- **Spatial pack** — `SpatialEmbedder` / `SpatialAugmenter` for H×W → length N
- **Train helpers** — `evaluate_classification` / `evaluate_regression`, `cosine_lr`
- **Contracts** — capacity `input_channels * 2**dim`; after packing, pass length-N inputs

## Documentation

Full API reference: [docs/Python_SDK.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/Python_SDK.md)

C++ contracts: [docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/CPP_SDK.md)

In-repo recipes: [examples/python/](https://github.com/dliptak001/HypercubeCNN/tree/main/examples/python)

Project repository: [github.com/dliptak001/HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)

## License

Apache-2.0
