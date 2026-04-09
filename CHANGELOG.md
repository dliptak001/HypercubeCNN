# Changelog

All notable changes to HypercubeCNN are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — pre-1.0, every
minor bump is potentially breaking.

## [0.1.0] — Initial public release

The first public release of HypercubeCNN: a convolutional neural network that
operates on Boolean hypercube topology instead of spatial grids, with end-to-end
backpropagation in pure C++23.

### Public C++ SDK

- `hcnn::HCNN` — top-level pipeline wrapper, single class for the entire
  embedding → conv/pool stack → readout flow. PIMPL-style wrapper around
  `hcnn::HCNNNetwork`. Non-copyable, non-movable.
- All public symbols live in `namespace hcnn`.
- `HCNN.h` is the canonical front door; layer headers (`HCNNNetwork`,
  `HCNNConv`, `HCNNPool`, `HCNNReadout`, `ThreadPool`) are re-exported
  transitively for power users.

### Architecture features

- Convolution: K = DIM nearest-neighbor XOR kernels, optional per-channel
  bias, optional batch normalization.
- Activations: `NONE`, `RELU`, `LEAKY_RELU`.
- Pooling: antipodal `MAX` or `AVG`, exact dimension reduction.
- Readout: `GAP` (translation-invariant) or `FLATTEN` (position-sensitive).
- Optimizer: `SGD` with momentum or `ADAM` with decoupled weight decay.
- Weight init: per-layer auto He/Kaiming or Xavier/Glorot uniform based on
  activation, plus a manual scale override.

### Training APIs

- `TrainStep` — single-sample SGD/Adam step.
- `TrainBatch` — mini-batch parallel step (samples in parallel across threads,
  gradients reduced + averaged + applied once).
- `TrainEpoch` — drives `TrainBatch` over a full dataset, optional deterministic
  shuffle via `shuffle_seed` with persistent gather buffers.
- `Embed` + `Forward` — single-sample inference (caller-owned scratch).
- `ForwardBatch` — parallel inference across samples.

### Performance

- Single-sample `Forward` is allocation-free in steady state (persistent
  ping-pong scratch on `HCNNNetwork`).
- `ForwardBatch` and `TrainBatch` use lazily-allocated per-thread buffers
  reused across calls.
- `TrainEpoch` shuffle path uses persistent gather buffers (grow on demand,
  never shrink).
- Conv inner vertex loop is cache-tiled (T = 64 vertices per tile).
- Per-layer vertex threading kicks in at DIM ≥ 12; auto-disabled during
  batch-parallel dispatch (RAII `LayerThreadGuard`).

### Correctness

- `Forward` and `ForwardBatch` are observably const w.r.t. batch-norm
  training state — they force eval mode for the duration of the call and
  restore the prior per-layer training flag on exit (RAII `EvalModeGuard`).
- Batch-norm running-stats updates are race-free during batch-parallel
  forward passes (RAII `BNStatsGuard` defers EMA updates until after the
  per-thread reduction).
- All batch APIs throw `std::invalid_argument` on `batch_size <= 0`.
- `Embed` throws `std::runtime_error` on over-capacity input.
- `train_step` / `train_batch` throw `std::runtime_error` on out-of-range
  target classes.

### Build / packaging

- Pure C++23, no external dependencies beyond the standard library.
- CMake static library (`HypercubeCNNCore`) with full install support.
- Configurable via `find_package(HypercubeCNN)` (imported target
  `HypercubeCNN::HypercubeCNNCore`) or via `FetchContent_MakeAvailable`.
- Build options: `HCNN_NATIVE_ARCH`, `HCNN_FAST_MATH`, `HCNN_WERROR`,
  `HCNN_BUILD_EXAMPLES`.
- GitHub Actions CI: Linux (GCC + Clang), macOS, Windows MSVC.

### Examples + tests

- `main.cpp` — quick smoke runner.
- `examples/mnist_train.cpp` — full MNIST training demo.
- `tests/CoreSmokeTest.cpp` — 47-assertion smoke test exercising every
  documented code path. Wired to ctest.

### Validated benchmark

MNIST (no spatial inductive bias): **98.10%** test accuracy with ~200K
parameters, 4 conv+pool stages, SGD + cosine LR + L2 weight decay.
See [docs/mnist.md](docs/mnist.md).
