# Regression Timeseries -- Next-Step Prediction

End-to-end **regression** recipe for HypercubeCNN
(`examples/regression_timeseries.cpp`). Trains a config-driven stack with
`TaskType::Regression` and MSE loss to predict the next value of a sine wave
from a length-N synthetic reservoir state.

### What it proves vs what it does not

Matches the binary banner (`Proves:` / `Does not:`):

| Proves | Does **not** prove |
|--------|---------------------|
| Regression API (`TrainEpoch` with float targets, MSE) | Real HypercubeESN readout dynamics |
| RELU/TANH + full-N FLATTEN at DIM=10 | Hard multi-step / chaotic forecasting |
| Cosine LR, target centering, best-MSE restore | Production RC skill |
| Thin train loop aligned with `mnist_train` | That near-perfect R² transfers off this sine task |

The synthetic reservoir is **uncoupled** (independent leaky tanh units). That is
intentional for an API smoke test; it is a weak stand-in for live RC state.
Near-perfect R² on next-step sine is an expected smoke signal once capacity
is enough -- not a HypercubeRC accuracy claim.

No external data files -- the example synthesizes its own series every run.

Pair with [`mnist_train.md`](mnist_train.md): both use top-of-file **`DemoConfig`**,
public **`HCNNArch.h`** (demo still includes thin `examples/demo_arch.h` aliases), and auto-printed architecture / param counts.

## What this example shows

- **`DemoConfig` at the top of `regression_timeseries.cpp`**: dim, layers,
  data sizes, seeds, schedule, logging -- one place to edit; architecture print
  + param counts follow
- **Public facade:** `HCNNConfig::Build`, `LayerSpec`, `HCNNTrainer`; demo knobs in `DemoConfig`
- `hcnn::HCNN` with `TaskType::Regression` (default loss MSE)
- Param count checked against `HCNN::GetWeightCount()` at startup
- Contiguous `TrainEpoch` (float targets) + `hcnn::evaluate_regression` (MSE / R²)
- `hcnn::cosine_lr` and **`HCNNBestMetricCheckpoint`** (minimize test MSE)
- Target centering (train-set mean subtracted from train **and** test targets)
- Default DIM=10 (N=1024; **19,137** parameters)

## Developer config (`DemoConfig`)

All developer-facing knobs live in one struct near the top of the `.cpp`.
`main` uses `const DemoConfig cfg{}`.

| Group | Fields (defaults) | Notes |
|-------|-------------------|--------|
| Hypercube / task | `dim=10`, `num_outputs=1`, `input_channels=1` | `N = 1 << dim` |
| Layers | `Conv(16, RELU)`, `Conv(16, TANH)` | No pool; full N into FLATTEN |
| Synthetic data | `n_warmup=200`, `n_train=4096`, `n_test=1024`, `horizon=1`, `input_freq=0.1`, `reservoir_seed=77` | Time-ordered: train then test (no series shuffle) |
| Init / opt | `weight_seed=42`, `optimizer=ADAM` | `RandomizeWeights(0.0, weight_seed)` |
| Schedule | `epochs=50`, `lr_max=0.002`, `lr_min_ratio=0.1`, `batch_size=32`, `weight_decay=0`, `momentum=0` | `lr_min = lr_max * lr_min_ratio` → **2e-4** |
| Logging | `log_first_epochs=5`, `log_every=10`, `n_sample_preds=8` | See [Logging](#logging) |

Helpers on the struct: `cfg.N()`, `cfg.lr_min()`.

Examples of edits: raise `dim` to 12/14; uncomment `ArchLayer::Pool` in the
layer list; swap activations; shrink `n_train` for a faster smoke run.

## The synthetic task

`make_reservoir` draws per-vertex parameters (seed `reservoir_seed`):

| Param | Distribution |
|-------|----------------|
| leak `α` | U[0.05, 0.45] |
| `w_in` | U[-1, 1] |
| `bias` | U[-0.5, 0.5] |

Drive and state update (same as the binary):

```text
u(t)     = sin(input_freq * t)
drive_i  = tanh(u(t) * w_in_i + bias_i)
state_i ← (1 - α_i) * state_i + α_i * drive_i
```

After `n_warmup` discarded steps, each collected sample is:

```text
input  = state(t)          // length N
target = sin(input_freq * (t + horizon))   // default horizon=1
```

**No exact closed form** maps the uncoupled state vector to the next sine
sample; the net must learn which timescale combinations predict the target.

**Uncoupled reservoir (intentional).** Vertices do not see each other -- only
the scalar drive. The readout test is "combine timescales," not "undo
cross-vertex coupling."

## Architecture

Edit `DemoConfig::layers` / `dim` in the `.cpp`. **Default** matches
`print_arch` / the binary:

```text
Architecture: Conv(1->16, RELU, bias)  DIM=10  N=1024
              -> Conv(16->16, TANH, bias)  DIM=10  N=1024
              -> FLATTEN
              -> Linear(16384 -> 1)
Parameters:   19409 (192 conv1 + 2832 conv2 + 16385 readout)  # K=DIM+1 (self+neighbors)
```

| Component | Count |
|-----------|------:|
| Conv1 (1×16×11 kernel + 16 bias) | 192 |
| Conv2 (16×16×11 kernel + 16 bias) | 2,832 |
| Readout (16×1024 → 1 + bias) | 16,385 |
| **Total** | **19,409** |

Startup throws if `summarize_arch` total ≠ `GetWeightCount()` (kernel + bias + BN blob floats when enabled;
BN γ/β and running stats are included in the weight blob when BN is enabled; optimizer moments are not).

FLATTEN treats every (channel, vertex) activation as an independent feature.
Vertex identity is informative: each reservoir unit encodes a different
timescale. No antipodal pool on the default recipe -- DIM and N stay 1024
through both convs into the head.

### Architectural choices

| Choice | Reason |
|--------|--------|
| Two convs, no pool | Full-N features into FLATTEN; pool is optional via `ArchLayer::Pool` |
| First-layer `RELU` | Sparse half-wave features; **beats dual-TANH** on seed 42 for this task |
| Second-layer `TANH` | Smooth, bounded bottleneck before the linear head |
| FLATTEN readout | Per-vertex weights for timescale identity |
| 16 channels | Capacity beyond a single linear combination of units |
| DIM=10 (N=1024) | Enough capacity for this sine smoke; ~1 min wall on 32 threads |

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Task | `TaskType::Regression` | MSE (fixed by task) |
| Optimizer | Adam (`SetOptimizer`) | Default betas |
| `lr_max` / floor | 0.002 / **2e-4** | `lr_min_ratio = 0.1` |
| LR schedule | `hcnn::cosine_lr(lr_max, lr_min, e, epochs)` | Progress `e/(epochs-1)`; last epoch hits `lr_min` |
| Batch size | 32 | |
| Weight decay | 0.0 | |
| Momentum | 0.0 | Passed through; Adam ignores |
| Epochs | 50 | |
| Shuffle | per-epoch | `shuffle_seed = e + 1` (train set only) |
| Target centering | Train mean on train **and** test | Mean from train only; added back for sample preds |
| Weight init seed | **42** | Printed as `Weight init seed:` |
| Reservoir seed | **77** | Printed in `Reservoir:` line |
| Checkpoint | `HCNNBestMetricCheckpoint` | Every epoch: observe test MSE; restore best at end |

### Logging

`should_log_epoch` prints on epochs **1..log_first_epochs**, every
**log_every**, and the **last** epoch (defaults: 1–5, 10, 20, 30, 40, 50).

On **every** epoch (logged or not):

1. `TrainEpoch` (float targets) on the centered train set
2. `evaluate_regression` on the **test** set
3. `best_mse.observe(net, test_mse, epoch)` for the best-MSE snapshot

On **logged** epochs only: also evaluate **train** MSE and print
`lr`, `train_mse`, `test_mse`, `test_R^2`, wall seconds, samples/s, and
`[best-mse]` when the snapshot updates.

End of run:

- Print best epoch + metric; `restore` those weights
- Re-eval test (`Restored best-mse` / `Final`)
- MSE reduction vs initial, total train wall time
- `n_sample_preds` evenly spaced predictions on **original** target scale
  (mean added back)

### Exit code

Returns **0** if final (restored) test **R² > 0.9**, else **1** (CI smoke).

## Key API patterns

### Contiguous data (as used by the demo)

```cpp
// Training
net.TrainEpoch(train_flat.inputs.data(), N,
                         train_flat.targets.data(),
                         train_flat.count, batch_size,
                         lr, momentum, weight_decay,
                         shuffle_seed);

// Metrics (core helper -> ForwardBatch under the hood)
hcnn::HCNNRegEval r = hcnn::evaluate_regression(
    net, flat.inputs.data(), flat.input_length,
    flat.targets.data(), flat.count, /*num_outputs=*/1);

// Best test MSE
hcnn::HCNNBestMetricCheckpoint best_mse;
best_mse.observe(net, static_cast<float>(r.mse), epoch_1based);
best_mse.restore(net);
```

Local packing: `FlatRegDataset` in the example. Architecture types live in
`HCNNArch.h` (installed) via `examples/demo_arch.h` aliases. Core helpers:
`evaluate_regression`, `cosine_lr`, `HCNNBestMetricCheckpoint`.

### Classification vs regression

| Step | Classification (`mnist_train`) | Regression (this example) |
|------|--------------------------------|---------------------------|
| Construction | `HCNN(dim, 10)` (class count) | `HCNN(dim, 1, 1, TaskType::Regression)` |
| Targets | `const int*` class indices | `const float*` contiguous |
| Training | `TrainEpoch` (int labels) | `TrainEpoch` (float targets) |
| Loss | Softmax + CE | MSE |
| Forward output | Logits | Raw predictions |
| Metrics helper | `evaluate_classification` | `evaluate_regression` |
| Checkpoint | `HCNNDualCheckpoint` | `HCNNBestMetricCheckpoint` |
| Shared arch product | `HCNNArch.h` (+ demo shim) | same |
| Flat data | `HCNNFlatDataset::reset_regression` | public helper |

Conv/pool stack, forward, weight init, optimizer, and batch parallelism are
shared; only loss gradient and target type differ.

## How to run

MinGW runtime on PATH (CLion toolchain), Release build:

```bash
cmake --build cmake-build-release --target RegressionTimeseries
./cmake-build-release/RegressionTimeseries
```

No data download. At default DIM=10, expect ~**55 s** for 50 epochs on a
32-thread box (hardware-dependent; samples/s printed on logged epochs).

## Results

Documented default after DemoConfig wiring. Hardware: Windows, MinGW g++,
**32 threads**. Weight seed **42**, reservoir seed **77**. Train target mean
subtracted: **2.674e-03**. Exit code **0**.

### Startup (architecture)

```text
Architecture: Conv(1->16, RELU, bias)  DIM=10  N=1024
              -> Conv(16->16, TANH, bias)  DIM=10  N=1024
              -> FLATTEN
              -> Linear(16384 -> 1)
Parameters:   19409 (192 conv1 + 2832 conv2 + 16385 readout)  # K=DIM+1 (self+neighbors)
```

### Curve (logged epochs)

```text
Initial test: mse=5.2473e-01  target_var=4.9564e-01  R^2=-0.0587  (1-R^2=1.0587)

Epoch  1  lr=0.002000  train_mse=7.5631e-05  test_mse=7.5625e-05  R^2=0.9998  [best-mse]
Epoch  2  lr=0.001998  train_mse=1.9695e-05  test_mse=1.9764e-05  R^2=1.0000  [best-mse]
Epoch  3  lr=0.001993  train_mse=8.0038e-06  test_mse=8.0079e-06  R^2=1.0000  [best-mse]
Epoch  4  lr=0.001983  train_mse=4.2062e-06  test_mse=4.1957e-06  R^2=1.0000  [best-mse]
Epoch  5  lr=0.001971  train_mse=3.8355e-06  test_mse=3.8264e-06  R^2=1.0000  [best-mse]
Epoch 10  lr=0.001854  train_mse=3.0068e-06  test_mse=2.9931e-06  R^2=1.0000
Epoch 20  lr=0.001411  train_mse=1.7928e-05  test_mse=1.7880e-05  R^2=1.0000   ← mid-run bump
Epoch 30  lr=0.000844  train_mse=8.7874e-08  test_mse=8.7727e-08  R^2=1.0000  [best-mse]
Epoch 40  lr=0.000379  train_mse=9.5826e-08  test_mse=9.7693e-08  R^2=1.0000
Epoch 50  lr=0.000200  train_mse=7.0045e-08  test_mse=7.0764e-08  R^2=1.0000

Best test MSE: epoch 49  mse=3.2532e-08
Restored best-mse: mse=3.2532e-08  target_var=4.9564e-01  R^2=1.0000  (1-R^2=0.0000)
Total train wall time: 54.72s
```

Throughput on logged epochs: ~3.7k–4.5k samples/s (~0.9–1.1 s/epoch).

### Summary table

| Metric | Value |
|--------|-------|
| Parameters | **19,137** |
| Initial test MSE | 5.247e-01 |
| Initial test R² | −0.0587 |
| Target variance (centered) | 4.956e-01 |
| **Best** test MSE | **3.253e-08** @ epoch **49** |
| Final (restored) R² | **1.0000** (1-R² ≈ 0) |
| MSE reduction vs initial | 100.00% |
| Total train wall | **54.72 s** |

### Sample predictions (test, original scale)

| step | target | pred | err |
|-----:|-------:|-----:|----:|
| 0 | +0.642826 | +0.642853 | +2.8e-05 |
| 128 | +0.448033 | +0.447714 | −3.2e-04 |
| 256 | +0.228870 | +0.228954 | +8.4e-05 |
| 384 | −0.002701 | −0.003048 | −3.5e-04 |
| 512 | −0.234124 | −0.234301 | −1.8e-04 |
| 640 | −0.452856 | −0.452838 | +1.8e-05 |
| 768 | −0.646954 | −0.647101 | −1.5e-04 |
| 896 | −0.805903 | −0.805813 | +9.1e-05 |

Absolute errors stay within ~3.5e-4 across the sine cycle.

### vs prior recipes (same task / weight seed 42)

| Recipe | Test MSE | Notes |
|--------|----------|--------|
| **DIM=10, RELU→TANH, no pool** (current) | **3.25e-08** best | Documented default |
| DIM=10, TANH→TANH, no pool | worse than above | Same seed A/B |
| Older DIM=12, TANH + MaxPool ×2 | ~7.75e-08 final | Heavier N; no best-MSE restore in that older writeup |

### Key observations

1. **Best-MSE restore matters** -- last epoch (50, test MSE 7.08e-08) is not
   the best; epoch **49** holds **3.25e-08**.
2. **Train and test MSE track closely** at convergence -- no weight decay needed
   on this task.
3. **Small-batch noise** can raise MSE mid-run (epoch 20); cosine floor recovers.
4. **Near-perfect fit is possible** on this synthetic task -- high R² is a
   smoke signal, not a claim about real RC workloads.
5. **First-layer RELU > dual TANH** on this seed for the no-pool DIM=10 stack;
   second-layer TANH remains a smooth head-facing nonlinearity.

## Adapting for your own data

1. Replace the synthetic reservoir with your state source. Inputs should
   respect HCNN's **[-1, 1]** embedding contract (or pad carefully).
2. Set `dim` so `N = 2^dim` matches (or exceeds) your state length; short
   inputs are zero-padded by `Embed`.
3. Set `num_outputs` for scalar or multi-output targets.
4. Keep time-series **causal** splits if you care about leakage (this demo
   trains on earlier timesteps, tests on later ones).

## Implications for HypercubeRC

This example supports the Phase B premise: HCNN can learn a useful projection
from a high-dimensional reservoir-like state to a continuous target without
hand-crafted features. Production RC coupling will still need real reservoir
dynamics, multi-output heads, and domain metrics beyond synthetic R².
