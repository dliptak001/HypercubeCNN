# HypercubeCNN — Change Log

## v1.0.0 (Jul 20, 2026)

First full public SDK release. Package version and FetchContent pin: **v1.0.0**
(create the git tag and GitHub Release when pushing).

### Highlights

- Single public front door: `hcnn::HCNN` (+ umbrella `HypercubeCNN.h`)
- Unified train API, session defaults, architecture product, full-capacity inputs
- Portable weight files, movable networks, examples as living facade recipes
- Pre-binding polish: arch lifecycle, host contracts, smaller train surface

### Breaking / public API

- **Constructor:** drop `LossType` argument; loss is fixed by `TaskType`
  (Classification → softmax CE, Regression → MSE). `num_threads` is the 5th arg.
- **Train vocabulary:** `TrainStep` / `TrainBatch` / `TrainEpoch` overload by
  target type (`int` / `const int*` vs `const float*`).  
  **`Train*Regression` aliases removed** — use the `float*` overloads.
- **Arch lifecycle:** `AddConv` / `AddPool` clear `WeightsInitialized`; train,
  infer, and weight I/O require `RandomizeWeights` for the current stack.
- **Default optimizer:** Adam (was SGD). Explicit `SetOptimizer` still supported.
- **Install surface:** only public headers are installed (`HCNN`, types, input,
  arch, helpers, spatial). `HCNNNetwork` / layers / `ThreadPool` are private
  (source-tree + in-tree tests only).
- **Removed:** `SetReadoutGradInLoop` from `HCNN` (private on `HCNNReadout`);
  `main.cpp` quick-check exe target; `examples/demo_arch.h` shim.

### Features

- **Inference:** `Predict` / `PredictClass`; batch path unchanged
- **Train session:** `TrainParams`, `SetTrainDefaults` / `GetTrainDefaults`,
  no-param train overloads; optional `HCNNTrainer` (cosine LR + shuffle)
- **Architecture:** `LayerSpec`, `summarize_arch` / `apply_arch` / `print_arch`,
  `HCNNConfig::Build()`
- **Inputs:** `HCNNInputView` / `HCNNInputBatch` (full capacity); spatial
  `pack_spatial` / `pack_spatial_batch`; `HCNNFlatDataset::input_view()`
- **Data / weights:** unified flat dataset (classif + regress);
  pointer `GetWeights` / `SetWeights`; versioned `save_weights` / `load_weights`
  (HCNW, little-endian ints **and** IEEE float32)
- **Ownership:** `HCNN` is **movable** (heap PIMPL); still non-copyable
- **Demos:** MNIST and regression use `HCNNConfig` + `HCNNTrainer` + public arch helpers
- **Core math / quality (earlier in arc):** self/center kernel tap `K = DIM + 1`;
  FLATTEN-only readout; BN + full weight blob; lifecycle/buffer hardening;
  spatial aug (shear/elastic) + embed; streamlined CoreSmokeTest

### Notes for integrators (e.g. HypercubeESN)

- Pin FetchContent / re-vendor to **v1.0.0** (or the release commit) after publish
- Update ctor (no `LossType`); use unified `Train*` with `float*` targets for regression
- Re-call `RandomizeWeights` after any post-init `AddConv` / `AddPool`
- Model files: keep arch (`LayerSpec` / `HCNNConfig`) beside HCNW weights
- See HypercubeESN `docs/adapt_HypercubeCNN.md` for a full host checklist
- Redistributable / wheels: `-DHCNN_NATIVE_ARCH=OFF` (default when not top-level)

### Python SDK (`hypercube-cnn`)

- scikit-build-core + pybind11 package at repo root (`import hypercube_cnn`)
- Core surface: `HCNN`, `TrainParams`, enums; train/infer/weights; GIL released on long ops
- Arch product: `LayerSpec`, `HCNNConfig`, `export_arch` / `from_arch`, versioned JSON sidecar
- Model I/O: HCNW `save_weights` / `load_weights`; `HCNN.save` / `load` (`.hcnw` + `.arch.json`)
- Lean tests: `python/tests/test_wheel.py` (cibuildwheel), `test_basic.py` (local)
- Tier 1 recipes: `examples/python/` (synthetic cls/reg + arch I/O)
- Docs: `docs/Python_SDK.md`; wheels: `.github/workflows/wheels.yml` (OIDC PyPI on `v*`)

### Docs

- README / CPP_SDK host contracts / internals aligned with the public facade
- Python_SDK + python_sdk_plan; capacity-as-topology packaging docs

---

## v0.1.0

Initial public tag on GitHub (pre–full-SDK facade arc).
