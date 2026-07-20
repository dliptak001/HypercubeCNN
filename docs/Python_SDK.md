# HypercubeCNN Python SDK

Python bindings for the **HypercubeCNN** C++ core: a dependency-free hypercube
CNN (`N = 2^DIM` vertices per channel). Same host contracts as
[CPP_SDK.md](CPP_SDK.md).

**Package:** `hypercube-cnn` · **import:** `hypercube_cnn` · **version:** aligned
with the C++ project (currently 1.0.0).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Host contracts](#host-contracts)
- [NumPy layout](#numpy-layout)
- [API](#api)
- [Model I/O](#model-io)
- [Examples](#examples)
- [Build notes](#build-notes)

---

## Installation

### From PyPI (when published)

```bash
pip install hypercube-cnn
```

Pre-built wheels: Python 3.10–3.13 on Windows (x64), Linux (x86_64, aarch64),
macOS (x86_64, arm64). No compiler required.

### From source

```bash
git clone https://github.com/dliptak001/HypercubeCNN.git
cd HypercubeCNN
pip install .
```

Requires Python 3.10+, C++23 compiler, CMake ≥ 3.21. On Windows with CLion MinGW,
see [python/README.md](../python/README.md) for the toolchain env vars and
`--no-build-isolation`.

### Tests

```bash
pip install ".[test]"
pytest python/tests/ -v --import-mode=importlib
```

Use a working directory **outside** `python/` so the source tree does not shadow
the installed `_core` extension. Wheel CI runs only
`python/tests/test_wheel.py` (import + one predict/step).

---

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

x = np.random.randn(net.N).astype(np.float32)  # full capacity
logits = net.predict(x)                         # (num_outputs,)
cls = net.predict_class(x)

net.train_step(x, target=0, params=hc.TrainParams(learning_rate=1e-3))
net.save("model")                               # model.hcnw + model.arch.json
net2 = hc.HCNN.load("model")
```

Incremental stack (same product):

```python
net = hc.HCNN(dim=6, num_outputs=3, task=hc.TaskType.Classification)
net.add_conv(8)
net.add_pool(hc.PoolType.MAX)
net.add_conv(8)
net.randomize_weights(seed=1)
```

---

## Host contracts

Same rules as the C++ SDK (language-agnostic):

| Contract | Rule |
|----------|------|
| Capacity | Always `input_channels * 2**dim`. Power of two is topology. Pack non-power-of-two data in the host. |
| Pad | Short inputs zero-fill the tail; over-long raise. Prefer full capacity after packing. |
| Task / loss | Classification → CE; Regression → sum-style MSE. Fixed by `task` at construct. |
| BatchNorm | Stats over vertices of **one sample** (per channel), not the mini-batch. Use `set_training`. |
| Outputs | Raw logits / preds — never softmax in forward. |
| Arch lifecycle | Stack changes invalidate weights; train/infer need `randomize_weights` for the current stack. |
| Weights | Params + BN stats; **not** optimizer moments. |
| Model I/O | HCNW does **not** store the layer graph; keep arch JSON beside weights. |
| Concurrency | One instance exclusive-use; expose `num_threads`. Not thread-safe across threads. |
| GIL | Released during long C++ train/infer calls; do not touch the same net from another thread. |

---

## NumPy layout

| Kind | Preferred shape |
|------|-----------------|
| Single sample, 1 ch | `(N,)` with `N = 2**dim` |
| Single sample, C ch | `(C, N)` or flat `(C*N,)` channel-major |
| Batch | `(B, input_length)` or `(B, C, N)` |
| Class targets | `(B,)` int |
| Regression targets | `(num_outputs,)` or `(B, num_outputs)` float32 |
| Weights blob | `(weight_count,)` float32 |

`dim` is in **[3, 30]**. Demos typically use 5–8.

Contiguous **float32** (inputs/weights) and **int32** (class labels) are produced
automatically when needed.

---

## API

### Enums

- `Activation`: `NONE`, `RELU`, `LEAKY_RELU`, `TANH`
- `PoolType`: `MAX`, `AVG`
- `TaskType`: `Classification`, `Regression`
- `OptimizerType`: `SGD`, `ADAM` (default at construction is Adam)

### `TrainParams`

```python
hc.TrainParams(
    learning_rate=1e-3,
    momentum=0.0,       # SGD only
    weight_decay=0.0,
    shuffle_seed=0,     # epoch: 0 = sequential; nonzero = shuffle
    class_weights=None, # optional (num_outputs,) float32, classification
)
```

### `LayerSpec` / `HCNNConfig` / arch helpers

```python
hc.LayerSpec.conv(c_out, activation="relu", use_bias=True, bn=False)
hc.LayerSpec.pool("max")  # or PoolType.MAX

cfg = hc.HCNNConfig(dim=6, num_outputs=3, layers=[...], weight_seed=1)
net = cfg.build()

hc.summarize_arch(dim, num_outputs, input_channels, layers)  # -> ArchParamSummary
hc.apply_arch(net, layers)
```

### `HCNN`

| Area | Methods / properties |
|------|----------------------|
| Construct | `HCNN(dim, num_outputs=10, input_channels=1, task=..., num_threads=0)` |
| Stack | `add_conv`, `add_pool`, `apply_layers`, `layers` |
| Init | `randomize_weights(scale=0.0, seed=42)`, `weights_initialized` |
| Mode | `set_training`, `set_optimizer`, `set_train_defaults` |
| Infer | `predict`, `predict_class`, `forward`, `forward_batch` |
| Train | `train_step`, `train_batch`, `train_epoch` (target form from `task`) |
| Weights | `weight_count`, `get_weights`, `set_weights` |
| Arch I/O | `export_arch()`, `from_arch(dict)`, `from_layers([...])` |
| File I/O | `save_weights`, `load_weights`, `save`, `load` |
| Sizing | `dim`, `N`, `capacity`, `num_outputs`, `input_channels`, `num_conv`, `num_pool`, `task`, `optimizer` |

**Classification** targets: int class index (step) or `(B,)` int (batch/epoch).  
**Regression** targets: float array of length `num_outputs` per sample.

---

## Model I/O

HCNW stores **parameters + coarse checks** only (same binary format as C++
`hcnn::save_weights`). The layer graph is **not** in the weight file.

| Artifact | Content |
|----------|---------|
| `model.hcnw` | Weights (little-endian IEEE float32 blob) |
| `model.arch.json` | Versioned arch sidecar (`format: hcnn_arch`, `version: 1`) |

```python
net.save("model")              # both files
net2 = hc.HCNN.load("model")   # or load("model.hcnw")

# explicit
net.save_weights("model.hcnw")
# write export_arch() to model.arch.json yourself if needed
```

`from_arch` / `load` rebuild the stack, randomize (required before loading
weights), then restore parameters. Optimizer defaults to Adam; moments are not
in HCNW.

Arch JSON future versions raise `ValueError` with an upgrade message. HCNW
architecture mismatch or bad magic raise `RuntimeError`.

---

## Examples

In-repo recipes (not installed with the wheel):

| Script | Role |
|--------|------|
| [examples/python/synthetic_classification.py](../examples/python/synthetic_classification.py) | Short CE train loop |
| [examples/python/synthetic_regression.py](../examples/python/synthetic_regression.py) | Short MSE train loop |
| [examples/python/arch_and_weights_io.py](../examples/python/arch_and_weights_io.py) | Arch JSON + HCNW identity |

See [examples/python/README.md](../examples/python/README.md).

C++ MNIST / timeseries examples remain the heavy recipes; Python Tier 1 stays
synthetic and offline.

---

## Build notes

- Extension compiles core + train-helpers into `_core` (scikit-build-core).
- Wheels: `HCNN_NATIVE_ARCH` off; portable `HYPERCUBE_ARCH` (`x86-64-v2` / `none`).
- `HCNN_FAST_TANH` defaults **ON** (matches C++ library default).
- MinGW: static libgcc/libstdc++; ships `libwinpthread-1.dll` on Windows wheels.
- Do not reconfigure CLion `cmake-build-*` for the Python package.

Living implementation tracker: [python_sdk_plan.md](python_sdk_plan.md).  
C++ contracts: [CPP_SDK.md](CPP_SDK.md).
