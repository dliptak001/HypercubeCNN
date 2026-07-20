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

You build a stack of local layers, train for classification or regression, and
save a weight file with a small architecture sidecar so another host can rebuild
the same network. Optional helpers turn ordinary images into length-N inputs
without pretending the cube is a pixel grid. Practical demos use a few dozen to
a few thousand vertices per channel; larger cubes are available when you need
them.

Compared with a standard vision CNN, the familiar pieces stay: shared local
weights, stacked layers, end-to-end training. What changes is the domain. A
usual network slides a small window across a rectangle and has to invent padding
at the borders. Here every site has the same number of neighbors, found by
flipping one bit of its address, so there is no image edge and no stencil table —
only a power-of-two number of sites and geometry that is exact under the cube’s
symmetry.

How you feed the network is part of that story. Pixels are not the native layout:
if you have an image, you (or the optional spatial helpers) first lay it out onto
the N vertices — pad, resize, dual-plane ink-and-gradient packs, or any scheme you
prefer — then train and infer on full-capacity vectors. Many hosts never touch
images at all. Reservoir or ESN state, bit-indexed fingerprints, product-space
features, and other signals that already sit at length 2^D can drive the network
directly, with no packing step and no pretense that the cube is a camera sensor.

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

Requires Python 3.10+, C++23, CMake ≥ 3.21. Windows + CLion MinGW local rebuild:

```powershell
$env:PATH = "C:\Program Files\JetBrains\CLion 2026.1\bin\mingw\bin;C:\Program Files\JetBrains\CLion 2026.1\bin\ninja\win\x64;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\Program Files\JetBrains\CLion 2026.1\bin\ninja\win\x64\ninja.exe"
$env:CC = "C:\Program Files\JetBrains\CLion 2026.1\bin\mingw\bin\gcc.exe"
$env:CXX = "C:\Program Files\JetBrains\CLion 2026.1\bin\mingw\bin\g++.exe"
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
- **Contracts** — capacity `input_channels * 2**dim`; full-capacity inputs after packing

## Documentation

Full API reference: [docs/Python_SDK.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/Python_SDK.md)

C++ contracts: [docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/CPP_SDK.md)

In-repo recipes: [examples/python/](https://github.com/dliptak001/HypercubeCNN/tree/main/examples/python)

Project repository: [github.com/dliptak001/HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)

## License

Apache-2.0
