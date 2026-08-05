# HypercubeCNN — Change Log

## Unreleased

### Spatial embed

- **Rename** `HCNNSpatialEmbedMode::RowMajorPad` → **`PadLow`** (same layout:
  full H×W in low verts, pad tail). Clean break — no alias (pre-user-base).
- **Add** **`PadLowCenter`**: full H×W in low verts + largest near-square
  centered crop in the remaining budget. MNIST 28×28 @ dim=10 → 15×16 @ (6,6),
  full N=1024 occupancy. Plan exposes `crop_h` / `crop_w` / `crop_row0` /
  `crop_col0`.
- **Keep** `ResizeToFit` and `DualPlaneResize`.
- Python: `SpatialEmbedMode.PadLow` / `PadLowCenter`; plan crop fields.
- Docs + CoreSmokeTest + Python tests updated.

---

## v1.0.2 (Aug 4, 2026)

Package version and FetchContent pin: **v1.0.2**.

### Documentation

- Project and Python package READMEs: HypercubeAI ecosystem positioning
  (ESN · CNN · Hopfield) so PyPI long description matches the GitHub project page.

---

## v1.0.1 (Aug 4, 2026)

Package version and FetchContent pin: **v1.0.1**.

### Fixed

- **`RandomizeWeights` / `HCNNConfig::weight_seed` seed width:** was `unsigned`
  (32-bit on typical hosts). A 64-bit master seed silently truncated before
  `mt19937` init. Now `uint64_t` end-to-end (`HCNN::RandomizeWeights`,
  `HCNNNetwork::randomize_all_weights`, `HCNNConfig::weight_seed`). Seeds with
  high half zero keep the historical `mt19937(seed32)` path (bit-identical to
  v1.0.0 for small seeds); wider seeds expand both halves via `seed_seq`.
  Python `randomize_weights(..., seed=)`, `HCNNConfig.weight_seed`,
  `from_arch` / `from_layers` accept full 64-bit ints (and `np.uint64`);
  façade rejects values outside `[0, 2**64-1]` instead of truncating.

### Notes for integrators (e.g. HypercubeESN)

- Re-vendor / FetchContent pin to **v1.0.1**.
- Hosts that already promoted their own `readout.seed` to `uint64_t` can pass
  full trial seeds without truncation.

---

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
- **Phase 5:** spatial embed/aug (`SpatialEmbedder`, `SpatialAugmenter`); metrics
  (`evaluate_classification` / `evaluate_regression`, `cosine_lr`); pickle secondary
  (`pickle.dumps` of arch + weights; HCNW remains primary)
- Lean tests: `python/tests/test_wheel.py` (cibuildwheel), `test_basic.py` (local)
- Tier 1 recipes: `examples/python/` (synthetic cls/reg, arch I/O, spatial smoke)
- Docs: `docs/Python_SDK.md`; wheels: `.github/workflows/wheels.yml` (OIDC PyPI on `v*`)

### Docs

- README / CPP_SDK host contracts / internals aligned with the public facade
- Python_SDK; capacity-as-topology packaging docs

---

## v0.1.0

Initial public tag on GitHub (pre–full-SDK facade arc).
