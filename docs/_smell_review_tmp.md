# Project-scope smell review (high level)

**Date:** 2026-07-19 (from session review; items 10–11 updated after follow-up)  
**Scope:** HypercubeCNN as a whole — layout, API product, build/release, tests, docs, integration risk.  
**Not:** line-by-line correctness of every kernel.

**Overall:** For a research/teaching C++ SDK this is in good shape: clear front door, intentional data model, solid smoke suite, no TODO/BUGBUG litter, and the P0–P4 pass tightened the public surface. Smells below are mostly **product maturity, dual surfaces, and operational** issues — not “rewrite the math.”

**Status legend:** open · **done** · won't fix · partial

---

## What’s healthy

| Area | Note |
|------|------|
| **Front door** | `HCNN` + `HypercubeCNN.h` is a coherent story; layers are demoted |
| **Data contract** | Contiguous row-major batches; hard task/API mismatch throws |
| **Lifecycle** | Buffer invalidation, BN guards, pool floor, weight blob incl. BN |
| **Tests** | One fast smoke (~0.2s, 200+ checks) covering facade + contracts |
| **Docs** | README / CPP_SDK / internals / spatial / demo write-ups are real |
| **Hygiene** | No TODO/FIXME/BUGBUG markers; Apache headers consistent |

---

## Medium–high smells

### 1. Dual dialect still lives in the tree — **won't fix**
Public API is PascalCase (`AddConv`, `TrainEpoch`). Internals remain snake_case (`add_conv`, `train_batch`). Accepted: teaching surface stays PascalCase; advanced/internal headers keep existing snake_case. Not worth a churn rename or forcing one style across both layers.

**Type:** soft public/private boundary (by design).

### 2. Classification vs regression still doubles the train surface — **done**
Unified vocabulary: `TrainStep` / `TrainBatch` / `TrainEpoch` overload by target
type (`int`/`const int*` vs `const float*`). Same for TrainParams, defaults, and
`HCNNInputView`. `Train*Regression` kept as thin aliases for transition (e.g.
HypercubeESN re-vendor). Wrong task still throws.

**Type:** combinatorial API growth (names unified; overload styles remain).

### 3. `HCNNNetwork` is still a second full orchestrator API — **done** (boundary, not blend)
Kept two-type PIMPL. Tightened boundary:
1. **Hard private** — Network/layers/ThreadPool never installed (`HCNN_PUBLIC_HEADERS` only).
2. **Not a second SDK** — docs + smoke framed as private impl / in-tree tests only.
3. **Policy** — features land on `HCNN` first; Network only what facade needs.
`friend class HCNN` on Network. Optional later: shrink Network into `HCNN.cpp`.

**Type:** thin facade over fat twin (boundary fixed; merge not desired).

### 4. Version / release story lags the work — **open**
CMake/docs still **v0.2.0** after a large public-API arc (Predict, TrainParams, Arch, FlatDataset, save/load, Adam default, install surface, LossType removal). Branch was ~30+ commits ahead of origin when reviewed.

**Type:** version and remote don’t match product reality.

### 5. Single smoke binary is both strength and gap — **won't fix**
`CoreSmokeTest` is the sole fast behavioral contract (~200+ checks, sub-second). Accepted: one well-maintained smoke suite is enough for this stage; no separate grad-check / golden / fuzz suite required. Revisit only if kernel/optimizer refactors demand numerical gates.

**Type:** one basket for tests (by design for now).

### 6. Spatial pad dual-contract remains a structural footgun — **done** (option C)
Introduced `HCNNInputView` / `HCNNInputBatch` (`HCNNInput.h`): full-capacity per sample.
Typed `Predict` / `ForwardBatch` / `Train*` overloads require `capacity == c_in * N`.
Spatial helpers: `pack_spatial` / `pack_spatial_batch`. FlatDataset: `input_view()`.
Explicit short→zero: `from_short_zero_pad`. Raw pointer APIs kept for power users.
MNIST demo uses `TrainEpoch(train_ds.input_view(), ...)`.

**Type:** structural guard on the typed happy path (raw short length still possible).

### 7. Optimizer / training knobs are still “caller-owned and scattered” — **done** (small A/C)
- **A:** `HCNN::SetTrainDefaults` / `GetTrainDefaults`; train overloads that omit `TrainParams` use them.
- **C:** `HCNNTrainer` in TrainHelpers — holds params, optional cosine LR + shuffle_seed per epoch, `train_epoch` / `train_epoch_regression`, optional sync to net defaults.
Explicit `TrainParams` / positional APIs remain. Optimizer type still via `SetOptimizer`.

**Type:** light session knobs without a full Trainer framework.

---

## Medium smells

### 8. Flat root layout — **won't fix**
Core `.h/.cpp` at repo root next to examples/tests/docs. Accepted: install already separates public headers; source-tree split is cosmetic churn (ESN re-vendor cost) with no API/behavior win at current size. Revisit only with a packaging overhaul or multi-target split.

**Type:** source aesthetics / scale hygiene.

### 9. Optional modules vs one static lib — **open**
Spatial + train helpers + arch are “optional by include” but always linked into `HypercubeCNNCore`.

### 10. `LossType` is almost a ghost API — **done** (commit `acad57e`)
Removed public enum and ctor param. Loss fixed by `TaskType` only (CE / MSE). `GetLossType` removed. `num_threads` is now the 5th constructor argument.

### 11. `SetReadoutGradInLoop` still on the facade — **done** (commit `acad57e`)
Removed from `HCNN`. `ReadoutGradInLoop` lives on `HCNNReadout` only (advanced / smoke).

### 12. Demo vs product config duplication — **open**
Demos keep fat `DemoConfig` + local loops; SDK has `HCNNConfig` / `TrainParams` / helpers but demos don’t fully live on that stack.

### 13. `main.cpp` “quick check” vs CoreSmokeTest — **done**
Removed `main.cpp` and the `HypercubeCNN` exe target. **CoreSmokeTest** is the sole in-tree smoke path (CTest `smoke`).

### 14. Weight file format is host-float endian — **open** (intentional-ish)
Ints LE; floats host IEEE. Fine for coursework on x86_64; not a portable model zoo.

### 15. Build-tree cruft risk — **open**
One-off experiments under build dirs; gitignore covers them. Process smell, not tree pollution in git.

---

## Lower-priority / polish

| # | Item | Status | Note |
|---|------|--------|------|
| — | Non-movable `HCNN` | open / by design | Forces `unique_ptr`; `Build()` already does |
| — | No multi-channel spatial | open / by design | Documented limit |
| — | C++23 hard requirement | open | Narrows coursework environments |
| — | Native/fast-math defaults ON | open | Demo-friendly; packaging risk |
| — | README FetchContent tag | open | Easy to show stale tag |
| — | No Python / bindings | open | Product choice |
| — | No ChangeLog for big arc | open | Release hygiene |

---

## Architecture map (smell lens)

```text
Public teach path          Advanced / dual surface
─────────────────          ───────────────────────
HypercubeCNN.h             HCNNNetwork (full twin API)
  HCNN  ──PIMPL──►         HCNNConv / Pool / Readout
  HCNNArch                 ThreadPool
  TrainHelpers             (snake_case, source-only install)
  Spatial*
```

Smell is not the split — it’s that **value still flows through both**.

---

## Prioritized “if you only fix a few” (updated)

| Priority | Action | Status |
|----------|--------|--------|
| 1 | Cut a release — bump 0.3.0, tag, push, align docs | **open** |
| 2 | Harden public boundary — Network/layers non-API forever | **done** |
| 3 | One train-entry design — `TrainParams` in demos; deprecate long positional later | **open** |
| 4 | Typed length-N input or spatial→Train helper (pad footgun) | **done** (option C) |
| 5 | Complement smoke with opt-in numerical/grad or golden test | **won't fix** |
| 6 | Optional `src/` + `include/HypercubeCNN/` layout | **won't fix** |
| — | Drop LossType ghost API | **done** |
| — | Drop facade grad_in loop knob | **done** |
| — | Unify PascalCase vs snake_case dialects | **won't fix** |

---

## Bottom line

No emergency codebase smell (dead TODOs, missing errors, broken install narrative). Remaining work is mid-maturity:

1. **version/release lag**  
2. **dual internal/public surfaces**  
3. **API multiplicity** (task × method × param style)  
4. **one test executable as oracle**  
5. **docs-correct, API-loose contracts** (pad/length)

---

## Related work already shipped (context)

| Arc | What |
|-----|------|
| P0–P1–P4 | Docs hygiene, Predict/TrainParams/Adam default/umbrella, demote install surface |
| P2 | `HCNNArch.h` — LayerSpec, apply_arch, HCNNConfig::Build |
| P3 | Unified FlatDataset, pointer Get/SetWeights, save/load_weights |
| Smells 10–11 | LossType removed; ReadoutGradInLoop off HCNN |

---

*This is a temporary tracking doc. Delete or promote into a real design note when no longer needed.*
