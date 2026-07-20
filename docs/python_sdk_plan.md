# HypercubeCNN Python SDK — Plan & Progress

Living plan for the Python bindings. Capture decisions, reference notes, and phase status here as work lands. When the product ships, keep a short “as built” section and move the long-form API into **`docs/Python_SDK.md`**.

| Field | Value |
|-------|--------|
| **Status** | **Phase 1 complete** — core train/infer façade green; Phase 2 (arch) next |
| **Last updated** | 2026-07-20 |
| **C++ peer** | [CPP_SDK.md](CPP_SDK.md) (v1.0.0 surface) |
| **References** | `C:\CLion\HypercubeESN\python`, `C:\CLion\HypercubeHopfield\python` (patterns, not clones) |

---

## Goal

Ship a **best-possible product** for both the C++ and Python SDKs: same host contracts, same core lifecycle, packaging that is honest for sdists and wheels, and a Python surface that is ergonomic without hiding architecture.

- **Core is the product** (`HCNN`). Arch helpers, train helpers, and spatial are optional products.
- Integration-first docs and API. No teaching-primary voice; no hype.
- Prefer approaches that improve on ESN/Hopfield packaging and persistence where CNN’s contracts demand it.
- Python is a **first-class host**, not a thin demo wrapper: if a contract matters in C++, it matters in Python.

### Definition of done (Python v1)

All of the following before calling the Python SDK “shipped”:

1. Installable wheel/sdist path (local MinGW green; cibuildwheel workflow in tree).
2. Core train/infer + arch build/export + HCNW save/load on the public API.
3. Lean pytest suite suitable for wheel smoke tests.
4. **Tier 0** docs quickstart + **Tier 1** in-repo scripts all exit 0.
5. `docs/Python_SDK.md` documents contracts (capacity, lifecycle, layout, I/O) without inventing a second API.
6. Version and ChangeLog entry aligned with the C++ release story (see Versioning).

PyPI publish can follow the first green tag; it is not required to *develop* v1, but the workflow must be ready.

---

## Decisions

Update this table when a choice is locked or reversed. **Bold** = locked for v1 unless blocked.

| Topic | Decision | Notes |
|-------|----------|--------|
| PyPI / import | **`hypercube-cnn` / `hypercube_cnn`** | Pending trusted publisher on pypi.org (repo `dliptak001/HypercubeCNN`, workflow `wheels.yml`, env any). First successful tag publish creates the project |
| Build stack | **scikit-build-core + pybind11 + NumPy** | Same stack as ESN/Hopfield |
| Packaging root | **Repo-root `pyproject.toml`** | Honest sdists. Package code under `python/hypercube_cnn/`. Extension CMake under `python/`. Only fall back to sibling-style `python/pyproject.toml` if root proves painful — then exhaustive `sdist.include` is mandatory |
| Bindings style | **Thin `_core` + rich Python façade** | Hopfield pattern; keep C++ boring |
| Construction | **Explicit stack and `HCNNConfig` / layers** | Not one fat ESN-style constructor |
| High-level `fit` | **Not in v1 public API** | Demos use explicit loops; optional helper only later |
| Persistence primary | **HCNW + arch sidecar (JSON)** | Interop with C++; pickle secondary if at all |
| Persistence bundle | **Separate `*.hcnw` + `*.arch.json`** | No zip/archive in v1 |
| Spatial in v1 | **Core-only** | No spatial bind in Phases 0–4; Phase 5 optional |
| Metrics / cosine LR helpers | **Defer** | Not required for Tier 1; bind later if cheap |
| Pickle | **Defer (Phase 5 or never)** | Prefer HCNW + JSON only for v1; avoids dual persistence stories |
| Wheel arch | **`HCNN_NATIVE_ARCH=OFF` + `HYPERCUBE_ARCH`** | Portable wheels (`x86-64-v2` / `none` like ESN) |
| Fast tanh | **`HCNN_FAST_TANH=ON` in wheels; document it** | Match C++ default |
| Fast math | **Follow extension Release flags / lib options; document** | Same spirit as C++ `HCNN_FAST_MATH` |
| GIL | **Release on long train/infer** | Do not release while holding borrowed NumPy buffers without a copy or clear ownership |
| Python versions | **3.10–3.13** for first matrix | Add 3.14 when ESN-stable and cheap |
| Data types | **float32 inputs/weights; int class labels** | Contiguous C-order NumPy |
| Python demos | **Tier 0 + Tier 1 in v1** | Not in the installed wheel; MNIST etc. deferred |
| Error mapping | **C++ `std::invalid_argument` / `std::runtime_error` → Python `ValueError` / `RuntimeError`** | Validate shapes in Python first where cheap |
| License | **Apache-2.0** | Match repo |

---

## Non-goals (v1)

- Torch / JAX tensor types or autograd interop.
- Replacing or wrapping the C++ examples as the Python package.
- A second “Pythonic” training framework (callbacks, loggers, Hydra).
- Bit-identical training trajectories vs C++ examples (different loops/seeds are fine); **bit-identical forward** after weight load *is* a goal.
- Binding every C++ overload (positional train lists, rare Embed-only paths).
- Reintroducing removed C++ APIs (`Train*Regression` as a separate surface).
- Self-vendoring a second copy of sources (unlike ESN’s `third_party/HypercubeCNN`).

---

## Reference review (ESN / Hopfield)

### Shared skeleton (keep)

| Layer | Pattern |
|-------|---------|
| Layout | Extension + pure-Python package + tests under `python/` |
| Build | Compile core `.cpp` **into** the extension (PIC); do not ship by linking only a prebuilt `.a` |
| API | Private `_core._Thing` + public class with validation and NumPy |
| Deploy | cibuildwheel (Linux x86_64/aarch64, Windows AMD64, macOS x86_64/arm64); PyPI on `v*` tags |
| Docs | Package README for PyPI + `docs/Python_SDK.md` for API |

### What to take from each

| Source | Take |
|--------|------|
| **Hopfield** | Thin bindings; enums; clear result types; GIL release on long ops; validation in Python |
| **ESN** | Mature wheel matrix (cibuildwheel v4, QEMU aarch64); MinGW packaging care; local rebuild skill (`pybuild`); importlib pytest mode |

### What to improve for CNN

1. **Sdist honesty** — sibling packages use `python/` as project root with parent-relative sources; wheel CI builds from full checkout and can hide broken sdists. Prefer root `pyproject` or exhaustive `sdist.include` of every public + private compile unit.
2. **Persistence** — siblings are pickle-first. CNN already has HCNW + arch sidecar; Python treats that as primary.
3. **API shape** — ESN’s large constructor does not map to CNN’s stack + `RandomizeWeights` lifecycle.
4. **No self-vendoring** — sources are this repo.
5. **Demos** — Hopfield/ESN under-invest in Python recipes; CNN commits Tier 0+1 because arch + I/O need a runnable proof.

---

## Host contracts (must survive the language boundary)

Same rules as [CPP_SDK.md](CPP_SDK.md) host contracts:

| Contract | Rule |
|----------|------|
| Capacity | Always `input_channels * 2^DIM`. Power of two is topology. Host packing is host work. |
| Pad | Short raw → zero-fill tail; over-long → error. After spatial pack, pass full capacity. |
| Task / loss | Classification → CE; Regression → sum-style MSE. Fixed by task. |
| BatchNorm | Stats over **vertices of one sample** (per channel), not the mini-batch. `set_training(True/False)` matters when BN is used. |
| Outputs | Raw logits / preds — never softmax in forward. |
| Arch lifecycle | Stack changes invalidate weights; train/infer/weights require successful `randomize_weights` for the current stack. |
| Weights blob | Params + BN running stats when present; **not** optimizer moments or Adam timestep. |
| Model I/O | HCNW stores parameters + coarse checks; **not** the layer graph. Keep arch JSON beside weights; rebuild, then load. |
| Concurrency | One instance exclusive-use; expose `num_threads`. Use `num_threads=1` when the host parallelizes across many nets. |
| Ownership | Non-copyable net; Python holds one implementation object (no copy, no share across threads). |

**Canonical surface to bind** (do not grow without design review):

```text
construct → add_conv/add_pool or HCNNConfig.build
randomize_weights / set_optimizer / set_train_defaults
predict / predict_class / forward / forward_batch
train_step / train_batch / train_epoch  (+ TrainParams)
get_weights / set_weights / weight_count + sizing getters
save_weights / load_weights  (+ export_arch / from_arch)
optional later: spatial, evaluate_*, cosine_lr, checkpoints
```

Prefer `TrainParams` (or session defaults) over long positional train argument lists. Prefer **full-capacity** arrays so intentional pad values are not wiped by short-input zero-fill.

---

## NumPy data layout (v1 convention)

Document this in `Python_SDK.md` and enforce in the façade.

| Kind | Preferred shape | Notes |
|------|-----------------|-------|
| Single sample, 1 channel | `(N,)` or `(1, N)` | `N = 2**dim` |
| Single sample, C channels | `(C, N)` or flat `(C*N,)` channel-major | Channel-major matches C++ (`c * N + v`) |
| Batch | `(B, C, N)` or flat `(B, C*N)` row-major samples | Same sample stride as C++ `ForwardBatch` / `TrainBatch` |
| Class targets (batch) | `(B,)` int | Classification |
| Regression targets | `(num_outputs,)` or `(B, num_outputs)` float32 | Task fixed at construct |
| Weights blob | `(weight_count,)` float32 | Layout matches C++ `GetWeights` |

**DIM range:** C++ allows start_dim in **[3, 30]**. Python validates the same. Practical demos use 6–10.

**Short inputs:** allowed only where C++ allows (zero-fill tail). Façade should prefer documenting full capacity; Tier 1 scripts always pass full `N`.

**Do not bind (v1) unless needed:** low-level `Embed` as a separate public method — `predict` / `forward` / train paths already embed. Add later only if a host requires split embed/forward.

---

## Public API inventory (v1)

Illustrative names; tighten during implementation. Grouped by module.

### Package exports (`hypercube_cnn`)

- `__version__`
- Enums: `Activation`, `PoolType`, `TaskType`, `OptimizerType`
- `TrainParams` (dataclass)
- `LayerSpec`, `HCNNConfig` (or builders on `LayerSpec`)
- `HCNN`
- `summarize_arch` (optional free function)

### `HCNN` methods / properties

| Area | Surface |
|------|---------|
| Construct | `HCNN(dim, num_outputs=…, input_channels=1, task=…, num_threads=0)` |
| Stack | `add_conv(...)`, `add_pool(...)` |
| Init | `randomize_weights(scale=0.0, seed=42)`, `weights_initialized` |
| Opt / mode | `set_optimizer(...)`, `set_training(bool)`, `set_train_defaults(TrainParams)` |
| Infer | `predict`, `predict_class` (classification only), `forward`, `forward_batch` |
| Train | `train_step`, `train_batch`, `train_epoch` (target form from `task`) |
| Weights | `weight_count`, `get_weights()`, `set_weights(arr)` |
| Sizing | `dim`, `N` / `start_n`, `num_outputs`, `input_channels`, `num_conv`, `num_pool`, `task`, `optimizer` |
| Arch I/O | `export_arch() -> dict`, `HCNN.from_arch(dict)`, maybe `apply_layers(list)` |
| File I/O | `save_weights(path)`, `load_weights(path, reset_optimizer_moments=True)` |
| Repr | `__repr__` including dim, N, task, layer counts, weights flag |

### Arch JSON (sidecar) sketch

Versioned, JSON-serializable, sufficient to rebuild then `load_weights`:

```text
{
  "format": "hcnn_arch",
  "version": 1,
  "dim": 10,
  "num_outputs": 10,
  "input_channels": 1,
  "task": "classification",   // or "regression"
  "layers": [
    {"kind": "conv", "c_out": 32, "activation": "relu", "use_bias": true, "use_bn": true},
    {"kind": "pool", "pool_type": "max"},
    {"kind": "conv", "c_out": 64, "activation": "relu", "use_bias": true, "use_bn": false}
  ]
}
```

Optimizer choice and `TrainParams` are **session** state, not required in the arch sidecar (match C++: HCNW is params, not optimizer). Document whether `from_arch` resets optimizer to Adam default (yes — same as fresh `Build`).

---

## Target API sketch (v1)

```python
import numpy as np
import hypercube_cnn as hc

net = hc.HCNN(
    dim=10,
    num_outputs=10,
    input_channels=1,
    task=hc.TaskType.Classification,
    num_threads=0,
)
net.add_conv(32, activation=hc.Activation.RELU, use_bn=True)
net.add_pool(hc.PoolType.MAX)
net.add_conv(64)
net.randomize_weights(seed=42)
# Default optimizer is Adam (C++ default); set_optimizer only when changing.

x = np.random.randn(1024).astype(np.float32)  # full capacity N
logits = net.predict(x)                         # (num_outputs,)
cls = net.predict_class(x)

params = hc.TrainParams(learning_rate=1e-3, weight_decay=1e-4)
net.train_step(x, target_class=3, params=params)

net2 = hc.HCNNConfig(
    dim=10,
    num_outputs=10,
    layers=[
        hc.LayerSpec.conv(32, bn=True),
        hc.LayerSpec.pool("max"),
        hc.LayerSpec.conv(64),
    ],
).build()

net.save_weights("model.hcnw")
with open("model.arch.json", "w", encoding="utf-8") as f:
    json.dump(net.export_arch(), f)
net3 = hc.HCNN.from_arch(json.load(open("model.arch.json", encoding="utf-8")))
net3.load_weights("model.hcnw")
```

---

## Target tree

```text
HypercubeCNN/
  pyproject.toml                 # scikit-build root
  CMakeLists.txt                 # existing C++ lib (unchanged role)
  python/
    CMakeLists.txt               # extension only (own build dir via scikit-build)
    bindings.cpp
    hypercube_cnn/
      __init__.py
      _hcnn.py                   # optional split for façade
      arch.py                    # LayerSpec / Config / export
    tests/
      test_basic.py              # lean; cibuildwheel
    README.md                    # PyPI readme (Tier 0 quickstart)
  examples/
    python/                      # Tier 1 (repo only; not the wheel)
      synthetic_classification.py
      synthetic_regression.py
      arch_and_weights_io.py
      README.md
  docs/
    CPP_SDK.md
    Python_SDK.md                # Phase 4
    python_sdk_plan.md           # this file
  .github/workflows/
    ci.yml                       # existing C++
    wheels.yml                   # Python wheels (new)
```

**Important:** scikit-build uses its own build directory. Do **not** reconfigure or overwrite CLion `cmake-build-*` trees (global project rule).

---

## Demos / examples

**Posture:** Examples prove contracts and give recipes. They are not the product (same rule as C++ `examples/`).

| Rule | Detail |
|------|--------|
| Not in the wheel | Scripts under `examples/python/`. Pip package = core + façade only. |
| Not a second API | Public surface only. |
| No flagship `fit()` | build stack → randomize → train loop → predict → save_weights + export_arch. |
| Packing stays visible | Tier 2 image demos must show length-N packing. |
| Tests ≠ demos | `python/tests/` lean for wheels; demos optional CI only. |

### Tier 0 — Docs only (v1 committed)

| Deliverable | Content |
|-------------|---------|
| `python/README.md` | Install + 15–30 line quickstart |
| `docs/Python_SDK.md` | Full API + recipes + capacity/packing; links to Tier 1 |

### Tier 1 — In-repo contract scripts (v1 committed)

Location: **`examples/python/`**. Fast, no dataset download, fixed seeds, offline.

| Script | Why | Constraints |
|--------|-----|-------------|
| `synthetic_classification.py` | Train/infer shapes + CE path | DIM 6–8, few epochs, NumPy only |
| `synthetic_regression.py` | MSE / regression path | Fixed seed; print MSE; exit 0 |
| `arch_and_weights_io.py` | Arch JSON + HCNW round-trip | Rebuild; **bit-identical logits** (or max abs diff == 0) |
| `README.md` | Install + how to run | Link `docs/Python_SDK.md` |

**Exit:** each script runs after local install and exits 0 with a one-line success metric.

Optional later: one CI job / `workflow_dispatch` for Tier 1 — **not** inside the cibuildwheel matrix.

### Tier 2 — Deferred

`mnist.py`, `regression_timeseries.py` — packing explicit; data path configurable; nothing vendored into the package.

### Tier 3 — Avoid unless requested

Notebooks as primary demos; plotting galleries; MLOps frameworks; torch/jax demos before NumPy is solid.

### Sequencing

| When | What |
|------|------|
| Phase 0–3 | No demos required |
| Phase 4 | **Tier 0 + Tier 1** |
| Phase 5+ | Tier 2 only if needed |

---

## Phases and checklist

Mark items `[x]` as they complete. Note blockers under **Log**.  
Phases are ordered by dependency; keep PRs reviewable (prefer one phase per PR when practical).

### Phase 0 — Packaging scaffold

**Exit:** `pip install .` produces a loadable `_core` on the local MinGW toolchain (empty or minimal module OK).

- [x] Add root `pyproject.toml` (scikit-build-core, pybind11, numpy, pytest; cibuildwheel stubs)
- [x] Add `python/CMakeLists.txt` compiling core sources into `_core`
- [x] Always compile **core + train-helpers** units needed for later HCNW (see Sources); spatial `.cpp` only if Phase 5 binds them
- [x] MinGW static / winpthread handling — full `-static` failed on CLion MinGW 15.2; **ESN pattern** used (`-static-libgcc/stdc++` + static pthread + ship `libwinpthread-1.dll`)
- [x] Portable arch via `HYPERCUBE_ARCH` (no host-only forced native in wheels; local default `native`)
- [x] Pin/document `HCNN_FAST_TANH=ON` for the extension (CMake option default ON)
- [x] Local rebuild notes in `python/README.md` (CLion MinGW + Ninja; `--no-build-isolation`)
- [x] Smoke: `import hypercube_cnn` → `__version__ == 1.0.0` (local MinGW wheel ~545 KB)
- [ ] Confirm sdist includes every file needed to compile (manual `python -m build --sdist` when `build` is available)

### Phase 1 — Core bindings + façade

**Exit:** Synthetic classification loss moves the right way; regression smoke; clear errors for bad shapes / uninitialized weights.

- [x] Enums: `Activation`, `PoolType`, `TaskType`, `OptimizerType`
- [x] `HCNN` construct, exclusive ownership, `add_conv` / `add_pool`, `randomize_weights`
- [x] Sizing getters + `weights_initialized`
- [x] `set_optimizer`, `set_training`, `TrainParams` / `set_train_defaults`
- [x] Infer: `predict`, `predict_class`, `forward` / `forward_batch`
- [x] Train: `train_step` / `train_batch` / `train_epoch` (dispatch on task; no parallel Regression API names)
- [x] Weights: `get_weights` / `set_weights` / `weight_count` as NumPy (enables Phase 2 identity checks without HCNW)
- [x] Contiguous float32 / int conversion; capacity and task checks
- [x] GIL release on long ops (with safe buffer ownership)
- [x] `__repr__` + exclusive-use note in class docstring
- [x] C++ exceptions surface as `RuntimeError` / `ValueError` (pybind11 default + Python validation)
- [x] Local smoke: cls acc 0.41→1.00; reg MSE drop; weight blob identity; invalid dim + uninit weights

### Phase 2 — Arch product surface

**Exit:** `export_arch` → `from_arch` → `set_weights` (same blob) → **bit-identical logits** on a fixed input. (File HCNW is Phase 3; in-memory weights are enough here.)

- [ ] `LayerSpec.conv` / `.pool` (dataclasses)
- [ ] `apply_arch` / `summarize_arch` (params / final dim)
- [ ] `HCNNConfig.build()` / layer-list construction
- [ ] `export_arch()` / `from_arch(dict)` with version field
- [ ] Reject unknown arch format versions with a clear upgrade message

### Phase 3 — Model I/O (HCNW)

**Exit:** Round-trip `save_weights` / `load_weights` in Python; ideally load a file produced by C++ smoke/tools on the same arch (or document manual check).

- [ ] Bind `save_weights` / `load_weights` from train helpers
- [ ] Document pair: `model.hcnw` + `model.arch.json`
- [ ] Arch mismatch / magic / version errors surface cleanly
- [ ] No pickle in v1 (deferred)

### Phase 4 — Tests, docs, CI, demos (Tier 0 + 1)

**Exit:** Local pytest green; wheels workflow present; `Python_SDK.md` usable; all Tier 1 scripts exit 0.

- [ ] `python/tests/test_basic.py` — construct, invalid dim, shapes, train/infer smoke, arch round-trip, HCNW round-trip, uninitialized-weights error (keep **fast**)
- [ ] `docs/Python_SDK.md` — contracts, layout table, API, recipes, Tier 1 links
- [ ] `python/README.md` — install + Tier 0 quickstart
- [ ] `examples/python/synthetic_classification.py`
- [ ] `examples/python/synthetic_regression.py`
- [ ] `examples/python/arch_and_weights_io.py`
- [ ] `examples/python/README.md`
- [x] `.github/workflows/wheels.yml` (cibuildwheel v4; QEMU aarch64; OIDC publish on `v*`; `skip-existing`) — green builds still need Phase 0 package
- [ ] Top-level README pointer (Python SDK + examples)
- [ ] ChangeLog entry
- [ ] pytest `import-mode=importlib` (avoid source-tree shadowing of `_core`)

### Phase 5 — Polish / extras

- [ ] Spatial module if needed; link [spatial_preprocess.md](spatial_preprocess.md)
- [ ] Tier 2 demos if needed
- [ ] Optional pickle wrapper **or** leave unsupported
- [ ] Optional metrics / cosine LR
- [ ] Type stubs (`.pyi`) if users need them
- [ ] Align HypercubeESN vendored HCNN (downstream; not a CNN v1 gate)
- [ ] First PyPI publish on tag when ready

---

## Scope cuts (v1)

| In v1 | Later / out of scope |
|-------|----------------------|
| Core train/infer + arch + HCNW | End-to-end `fit()` as public API |
| NumPy float32 | torch / jax |
| `num_threads` on construct | Multi-net process pool utilities |
| Contract docs + layout rules | Full spatial product |
| Wheel CI workflow | Perfect parity with every C++ overload |
| Tier 0 + Tier 1 demos | Tier 2 MNIST/timeseries; notebooks; viz |
| Separate `.hcnw` + `.arch.json` | Single archive bundle; pickle-first I/O |

---

## Versioning

| Artifact | Scheme |
|----------|--------|
| C++ project | `CMakeLists.txt` `VERSION` (currently 1.0.0) |
| Python package | `pyproject.toml` / `__version__` — **align major.minor with C++** when releasing together; patch may diverge for binding-only fixes |
| Arch JSON | `version` field inside sidecar (start at 1) |
| HCNW | C++ `kHCNNWeightFileVersion` — Python must not invent a parallel weight format |

Document dual-version in `Python_SDK.md` (“library X, arch format Y, HCNW Z”).

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| C++23 on manylinux / macOS CI | Use recent images + gcc-toolset (ESN pattern); fail early in Phase 0 on one Linux runner if possible |
| MinGW runtime DLLs on Windows wheels | Static link or ship `libwinpthread`; test import on clean PATH |
| Sdist missing private headers | Root packaging + explicit include list; test `build --sdist` then install from tarball once |
| Phase 2/3 confusion | Phase 2 uses in-memory weights; Phase 3 is file HCNW only |
| Over-binding C++ overloads | One NumPy-friendly path per operation; expand only on demand |
| GIL + internal thread pool | Document exclusive instance; release GIL only around full C++ calls that own their buffers |
| Float / fast-math differences across platforms | Wheel tests use loose smoke thresholds; identity checks use save/load same binary |
| Name collision on PyPI | Check `hypercube-cnn` before first publish |
| CLion build dirs clobbered | scikit-build isolated build dir; never `-B cmake-build-*` for the extension |

---

## Implementation notes

### Local Windows build (CLion toolchain)

Paths track installed CLion (see global build instructions). Rebuild must put MinGW + Ninja on `PATH`. Prefer `--no-build-isolation` when the isolated env cannot see those tools.

```text
PATH += CLion mingw/bin + ninja
CMAKE_GENERATOR=Ninja
CC/CXX = CLion gcc/g++
pip install . --no-build-isolation --force-reinstall --no-deps
pytest ... --import-mode=importlib
# Run pytest / demos from a cwd outside python/ so the source package does not shadow _core
```

### Sources that must compile into `_core`

Apps never include private headers; the **extension** must compile them.

**Always (v1 core + I/O):**

- `HCNN.cpp`, `HCNNConv.cpp`, `HCNNPool.cpp`, `HCNNNetwork.cpp`, `HCNNReadout.cpp`
- `HCNNTrainHelpers.cpp` (HCNW + any bound helpers)
- Headers: public set + `HCNNNetwork.h`, `HCNNConv.h`, `HCNNPool.h`, `HCNNReadout.h`, `ThreadPool.h`

**Phase 5 only if spatial is bound:**

- `HCNNSpatialAug.cpp`, `HCNNSpatialEmbed.cpp`

Link `Threads` as the core library does.

### Existing C++ CI

Keep `ci.yml` for C++/smoke. Add `wheels.yml` for Python; do not overload C++ CI with full wheel matrices on every PR if cost is high — match ESN (wheels on push/PR/tag) only if runtime is acceptable; otherwise wheels on tag + `workflow_dispatch` with PR smoke on one platform.

### Open items

- [x] PyPI project name `hypercube-cnn` — pending trusted publisher registered (does not reserve the name until first upload)
- [x] `wheels.yml` added (triggers: main push/PR, tags `v*`, `workflow_dispatch`)
- [x] Land Phase 0 package so local install works (CI wheels still need push + green matrix)
- [ ] Optional sdist tarball install smoke
- [ ] Optional: one C++-written HCNW fixture in `tests/data/` for cross-language load (nice-to-have in Phase 3–4)
- [x] Packaging root default: repo-root `pyproject.toml` (workflow `package-dir: .`)
- [x] Persistence: separate `.hcnw` + `.arch.json`; pickle deferred
- [x] Spatial: not in v1 core path
- [x] Demos: Tier 0 + Tier 1 in Phase 4

### PyPI trusted publishing checklist

| Step | Status |
|------|--------|
| Pending publisher on pypi.org for `hypercube-cnn` → this repo / `wheels.yml` | Done (user) |
| Workflow file `.github/workflows/wheels.yml` on default branch | In tree (push to `main` to activate) |
| Python package (`pyproject.toml` + extension) builds | **Local MinGW green** (Phase 0); CI unproven until push |
| Tag `v*` after green wheels → OIDC publish creates project | Later (need API + green CI first) |

---

## Log

| Date | Note |
|------|------|
| 2026-07-20 | Plan written from ESN/Hopfield review + CNN C++ contracts. |
| 2026-07-20 | Demos: Tier 0 + Tier 1 locked for v1 Phase 4; Tier 2 deferred. |
| 2026-07-20 | Refine pass: locked decisions; non-goals; NumPy layout; API inventory; arch JSON sketch; Phase 2/3 split (in-memory vs HCNW); pickle deferred; risks; versioning; DoD; train-helpers always in `_core` for HCNW. |
| 2026-07-20 | Added `.github/workflows/wheels.yml` (ESN-style cibuildwheel v4 + OIDC publish, `skip-existing`). Matches pending PyPI trusted publisher. `package-dir: .` for repo-root packaging. Workflow will fail until Phase 0 lands. |
| 2026-07-20 | **Phase 0 done:** root `pyproject.toml`, `python/CMakeLists.txt` + stub `bindings.cpp`, `hypercube_cnn` package. Local `pip install .` → import `1.0.0`; ships `libwinpthread-1.dll`. Full MinGW `-static` rejected; ESN link recipe used. |
| 2026-07-20 | **Phase 1 done:** `_HCNN` + enums + train/infer/weights; Python `HCNN` / `TrainParams`. Smoke: classification train, regression MSE, in-memory weight round-trip. MinGW: dynamic winpthread + ship DLL (`-Bstatic pthread` broke CLion 15.2). Use `==` not `is` for pybind enums. |

---

## Related docs

| Doc | Role |
|-----|------|
| [CPP_SDK.md](CPP_SDK.md) | C++ contracts and canonical surface |
| [spatial_preprocess.md](spatial_preprocess.md) | Optional 2D packing (host-side) |
| [internals.md](internals.md) | Maintainers / research |
| `docs/Python_SDK.md` | Public Python API (Phase 4) |
| [ChangeLog.md](../ChangeLog.md) | Release notes |
| [README.md](../README.md) | Product overview; will link Python when present |
