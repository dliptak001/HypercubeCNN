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

**Phase 0 scaffold:** the native extension loads and reports a version. The
full train/infer API is not yet exposed — see
[docs/python_sdk_plan.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/python_sdk_plan.md)
and the C++ guide
[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/CPP_SDK.md).

```python
import hypercube_cnn as hc
print(hc.__version__)
```

## License

Apache-2.0
