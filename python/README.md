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

**Phase 1:** core `HCNN` train/infer surface is available (construct, stack,
`randomize_weights`, predict, train_*, weights). Arch JSON / HCNW file I/O and
docs polish land in later phases — see
[docs/python_sdk_plan.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/python_sdk_plan.md)
and
[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/CPP_SDK.md).

```python
import numpy as np
import hypercube_cnn as hc

net = hc.HCNN(dim=6, num_outputs=3, task=hc.TaskType.Classification)
net.add_conv(8)
net.add_pool(hc.PoolType.MAX)
net.add_conv(8)
net.randomize_weights(seed=1)

x = np.random.randn(net.N).astype(np.float32)  # full capacity N = 2**dim
logits = net.predict(x)
cls = net.predict_class(x)
net.train_step(x, target=0, params=hc.TrainParams(learning_rate=1e-3))
```

## License

Apache-2.0
