# hypercube-cnn

Python bindings for **HypercubeCNN** — a dependency-free C++23 CNN on Boolean
hypercube topology (`N = 2^DIM` vertices per channel).

## Installation

```bash
pip install hypercube-cnn
```

Pre-built wheels (when published) target Python 3.10–3.13 on Windows (x64),
Linux (x86_64, aarch64), and macOS (x86_64, arm64). No compiler required.

### From source

```bash
git clone https://github.com/dliptak001/HypercubeCNN.git
cd HypercubeCNN
pip install .
```

Windows + CLion MinGW (local rebuild):

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

- Core `HCNN` train/infer (classification CE, regression MSE)
- `LayerSpec` / `HCNNConfig` architecture product surface
- HCNW weights + arch JSON sidecar (C++-interop); pickle as secondary
- Spatial embed/aug (`SpatialEmbedder`, `SpatialAugmenter`)
- Metrics: `evaluate_classification` / `evaluate_regression`, `cosine_lr`
- NumPy float32 integration

## Documentation

- Full API: [docs/Python_SDK.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/Python_SDK.md)
- C++ contracts: [docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/CPP_SDK.md)
- Repo examples: [examples/python/](https://github.com/dliptak001/HypercubeCNN/tree/main/examples/python)

## License

Apache-2.0
