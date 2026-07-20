# HypercubeCNN — Change Log

## v1.0.0 (Jul 20, 2026)

First full public SDK release. Package version and FetchContent pin: **v1.0.0**
(create the git tag and GitHub Release when pushing).

### Highlights

- Single public front door: `hcnn::HCNN` (+ umbrella `HypercubeCNN.h`)
- Unified train API, session defaults, architecture product, full-capacity inputs
- Portable weight files, movable networks, examples as living facade recipes

### Breaking / public API

- **Constructor:** drop `LossType` argument; loss is fixed by `TaskType`
  (Classification → softmax CE, Regression → MSE). `num_threads` is the 5th arg.
- **Train vocabulary:** prefer `TrainStep` / `TrainBatch` / `TrainEpoch` with
  target-type overloads (`int` / `const int*` vs `const float*`).  
  `Train*Regression` remain as **thin aliases** for transition.
- **Default optimizer:** Adam (was SGD). Explicit `SetOptimizer` still supported.
- **Install surface:** only public headers are installed (`HCNN`, types, input,
  arch, helpers, spatial). `HCNNNetwork` / layers / `ThreadPool` are private
  (source-tree + in-tree tests only).
- **Removed:** `SetReadoutGradInLoop` from `HCNN` (private on `HCNNReadout`);
  `main.cpp` quick-check exe target.

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
- Update ctor (no `LossType`); prefer unified `Train*` overloads
- See HypercubeESN `docs/adapt_HypercubeCNN.md` for a full host checklist
- Redistributable binaries: consider `-DHCNN_NATIVE_ARCH=OFF`

### Docs

- README / CPP_SDK / internals aligned with the public facade and v1.0.0 pin

---

## v0.1.0

Initial public tag on GitHub (pre–full-SDK facade arc).
