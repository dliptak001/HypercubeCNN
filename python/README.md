# hypercube-cnn

Python bindings for **HypercubeCNN** — a dependency-free C++23 CNN on Boolean
hypercube topology (`N = 2^DIM` vertices per channel).

## Installation

```bash
pip install hypercube-cnn
```

Pre-built wheels (when published) target Python 3.10–3.13 on Windows (x64),
Linux (x86_64, aarch64), and macOS (x86_64, arm64).

### From source

Requirements: Python 3.10+, a C++23 compiler, CMake 3.21+, Ninja recommended.

```bash
git clone https://github.com/dliptak001/HypercubeCNN.git
cd HypercubeCNN
pip install .
```

On Windows with CLion MinGW (local development):

```powershell
$env:PATH = "C:\Program Files\JetBrains\CLion 2026.1\bin\mingw\bin;C:\Program Files\JetBrains\CLion 2026.1\bin\ninja\win\x64;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\Program Files\JetBrains\CLion 2026.1\bin\ninja\win\x64\ninja.exe"
$env:CC = "C:\Program Files\JetBrains\CLion 2026.1\bin\mingw\bin\gcc.exe"
$env:CXX = "C:\Program Files\JetBrains\CLion 2026.1\bin\mingw\bin\g++.exe"
pip install . --no-build-isolation --force-reinstall --no-deps
```

## Status

**Phase 2:** core train/infer plus architecture product surface (`LayerSpec`,
`HCNNConfig`, `export_arch` / `from_arch`). HCNW file I/O is Phase 3 — see
[docs/python_sdk_plan.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/python_sdk_plan.md)
and
[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/CPP_SDK.md).

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

arch = net.export_arch()           # JSON-serializable sidecar
w = net.get_weights()
net2 = hc.HCNN.from_arch(arch)
net2.set_weights(w)                # bit-identical forward
```

## License

Apache-2.0
