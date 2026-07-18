# Regression Timeseries -- Next-Step Prediction

End-to-end **regression** teaching demo for HypercubeCNN. Trains a config-driven
conv+pool stack with `TaskType::Regression` and MSE loss to predict the next
value of a sine wave from a high-dimensional synthetic reservoir state vector.

### What it proves vs what it does not

| Proves | Does **not** prove |
|--------|---------------------|
| Regression API (MSE, `TrainEpochRegression`) | Real HypercubeRC / ESN dynamics |
| TANH + antipodal pool + FLATTEN at DIM=12 scale | Hard multi-step / chaotic forecasting |
| Cosine LR, target centering, **best test-MSE restore** | That near-perfect R^2 transfers to production RC |
| Thin teaching loop aligned with `mnist_train` | Coupled reservoir / spectral-radius stories |

The synthetic reservoir is **uncoupled** (independent leaky tanh units). That is
intentional for an API smoke test; it is a weak stand-in for live RC state.
Near-perfect R^2 on next-step sine is an expected smoke signal once capacity
is enough -- not a HypercubeRC accuracy claim.

No external data files -- the example synthesizes its own series every run.

Pair with [`mnist_train.md`](mnist_train.md): both use top-of-file **`DemoConfig`**,
shared **`examples/demo_arch.h`**, and auto-printed architecture / param counts.

## What this example shows

- **`DemoConfig` at the top of `regression_timeseries.cpp`**: dim, layers,
  data sizes, seeds, schedule, logging -- edit knobs there only
- Shared **`demo_arch.h`** (`ArchLayer`, `summarize_arch`, `print_arch`, `apply_arch`)
- `hcnn::HCNN` with `TaskType::Regression` and default MSE loss
- Config-driven stack; param counts checked against `GetWeightCount`
- FLATTEN readout (position-sensitive per-vertex weights)
- Contiguous `TrainEpochRegression` + `hcnn::evaluate_regression` (MSE / R^2)
- `hcnn::cosine_lr` and **`HCNNBestMetricCheckpoint`** (best test MSE)
- Target centering (subtract train-set mean before training)
- Default DIM=12 (N=4,096; **19,425** parameters)

## Developer config (`DemoConfig`)

All developer-facing knobs live in one struct near the top of the `.cpp`.
`main` is `const DemoConfig cfg{}`.

| Group | Fields (defaults) | Notes |
|-------|-------------------|--------|
| Hypercube / task | `dim=12`, `num_outputs=1`, `input_channels=1` | `N = 2^dim` is the state length |
| Layers | Conv16 TANH, MaxPool, Conv16 TANH, MaxPool | Edit `layers` vector |
| Synthetic data | `n_warmup=200`, `n_train=4096`, `n_test=1024`, `horizon=1`, `input_freq=0.1`, `reservoir_seed=77` | Time-ordered train then test (no shuffle of the series) |
| Init / opt | `weight_seed=42`, `optimizer=ADAM` | |
| Schedule | `epochs=50`, `lr_max=0.002`, `lr_min_ratio=0.1`, `batch_size=32`, `weight_decay=0`, `momentum=0` | `lr_min = lr_max * lr_min_ratio` |
| Logging | `log_first_epochs=5`, `log_every=10`, `n_sample_preds=8` | Full train+test eval on logged epochs only |

Examples of edits: raise `dim` to 14/16 for scale stress; add a third
conv+pool pair; change `n_train` for a faster smoke run.

## The synthetic task

A synthetic reservoir of **N independent** leaky tanh integrators is driven by
`sin(input_freq * t)`. Each vertex has its own leak rate (U[0.05, 0.45]),
input weight (U[-1, 1]), and bias (U[-0.5, 0.5]), so different vertices capture
the input at different timescales. After `n_warmup` steps of burn-in, the
example collects (state, target) pairs where

```text
target = sin(input_freq * (t + horizon))   // default: one step ahead
```

**No exact solution exists.** The target is a nonlinear function of input
history; the model must learn which timescale combinations predict the next
step. This is a real regression problem, not sparse recovery.

**Uncoupled reservoir (intentional).** Vertices do not see each other -- only
the scalar drive. That simplifies the readout test: learn combinations of
timescales, not how to undo cross-vertex coupling.

## Architecture

Edit `DemoConfig::layers` / `dim` in the `.cpp`. **Default** (`dim=12`):

```text
Input: N=4096 floats -> N vertices (DIM=12, 1 channel)
  |
Conv1: 1 -> 16, K=12, TANH, bias     DIM=12  N=4096
Pool1: MAX antipodal                 DIM 12->11  N 4096->2048
  |
Conv2: 16 -> 16, K=11, TANH, bias    DIM=11  N=2048
Pool2: MAX antipodal                 DIM 11->10  N 2048->1024
  |
Readout: FLATTEN -> Linear(16384 -> 1) -> prediction
```

| Component | Parameters |
|-----------|------------|
| Conv1 kernel (1 x 16 x 12) + bias | 208 |
| Conv2 kernel (16 x 16 x 11) + bias | 2,832 |
| Readout weights (16,384 -> 1) + bias | 16,385 |
| **Total** | **19,425** |

Startup checks that this total equals `HCNN::GetWeightCount()` (kernel + bias;
BN gamma/beta are not in the weight blob if you enable BN).

FLATTEN treats every (channel, vertex) activation as an independent feature.
Vertex identity is informative: each reservoir unit encodes a different
timescale.

### Architectural choices

| Choice | Reason |
|--------|--------|
| Two conv+pool stages | 2-hop receptive field on the hypercube |
| `Activation::TANH` | Smooth, symmetric, bounded; matches reservoir nonlinearity; Xavier/Glorot init |
| `PoolType::MAX` (antipodal) | Keep stronger of complement pair; reduces DIM by 1 per pool |
| FLATTEN readout | Per-vertex weights for timescale identity |
| 16 channels | Capacity for a target that is not exactly one-channel expressible |

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Task | `TaskType::Regression` | Loss defaults to `LossType::MSE` |
| Optimizer | Adam | Default betas |
| `lr_max` / floor | 0.002 / **2e-4** (10%) | `lr_min_ratio = 0.1` |
| LR schedule | `hcnn::cosine_lr` | Progress `epoch / (epochs - 1)`; last epoch hits `lr_min` |
| Batch size | 32 | |
| Weight decay | 0.0 | No L2 by default |
| Epochs | 50 | |
| Shuffle | per-epoch | `shuffle_seed = epoch + 1` (within train set only) |
| Target centering | Train-set mean subtracted from train **and** test | Mean taken on train only |
| Weight init seed | **42** | Printed at startup |
| Reservoir seed | **77** | Fixed synthetic dynamics |

### Logging

- Epochs **1..log_first_epochs**, every **log_every**, and the **last** epoch
  print train MSE, test MSE, test R^2, wall time, and samples/s
- Other epochs train only (no full eval) to keep wall time down
- End: final test metrics, MSE reduction vs initial, and evenly spaced sample
  predictions on **original** target scale (mean added back)

### Exit code

Process returns **0** if final test **R^2 > 0.9**, else **1** (CI smoke).

## Key API patterns

### Contiguous data

```cpp
// Training: flat_inputs is sample_count * N floats, flat_targets is sample_count floats
net.TrainEpochRegression(flat_inputs.data(), N,
                         flat_targets.data(),
                         sample_count, batch_size,
                         lr, momentum, weight_decay,
                         shuffle_seed);

// Inference: flat_inputs is count * N floats, preds is count floats
net.ForwardBatch(flat_inputs.data(), N, count, preds.data());
```

Local packing: `FlatRegDataset` in the example. Core helpers used:
`evaluate_regression`, `cosine_lr`, `HCNNBestMetricCheckpoint`. Architecture
types live in `examples/demo_arch.h` (not installed SDK).

### Classification vs regression

| Step | Classification (`mnist_train`) | Regression (this example) |
|------|--------------------------------|---------------------------|
| Construction | `HCNN(DIM, 10)` | `HCNN(DIM, 1, 1, TaskType::Regression)` |
| Targets | `const int*` class indices | `const float*` contiguous |
| Training | `TrainEpoch` | `TrainEpochRegression` |
| Loss | Softmax + CE | MSE |
| Forward output | Logits | Raw predictions |
| Shared helpers | class metrics, dual-ckpt, Spatial* | reg metrics, best-MSE ckpt, `cosine_lr` |
| Shared demo arch | `demo_arch.h` | `demo_arch.h` |

Conv/pool stack, forward, weight init, optimizer, and batch parallelism are
shared; only loss gradient and target type differ.

## How to run

MinGW runtime on PATH (CLion toolchain), Release build:

```bash
cmake --build cmake-build-release --target RegressionTimeseries
./cmake-build-release/RegressionTimeseries
```

No data download. At default DIM=12, expect on the order of a few minutes for
50 epochs (hardware-dependent; throughput is printed per logged epoch).

## Results

Default recipe after DemoConfig + `hcnn::cosine_lr` wiring. Hardware: Windows,
MinGW g++, 32 threads. Wall time ~**3.8 min** (50 epochs). Exit code 0.

```text
Initial test: mse=5.45e-01  R^2=-0.10
Epoch  1  test_mse=2.19e-04  test_R^2=0.9996  (~4.6s)
Epoch  5  test_mse=3.56e-06  test_R^2=1.0000
Epoch 10  test_mse=8.45e-07
Epoch 20  test_mse=5.75e-07
Epoch 30  test_mse=5.84e-06   (small-batch bump; recovered)
Epoch 40  test_mse=8.56e-08
Epoch 50  test_mse=7.75e-08  test_R^2=1.0000  lr=2e-4
Final: mse=7.75e-08  1-R^2~0  MSE reduction ~100%
```

| Metric | Value |
|--------|-------|
| Parameters | 19,425 |
| Initial test MSE | 5.452e-01 |
| Final test MSE | **7.754e-08** |
| Final test R^2 | **~1.0** (1-R^2 ~ 1.6e-7) |
| MSE reduction vs initial | 100.00% |
| Throughput (logged epochs) | ~780–1100 samples/s |

Sample predictions (test, original scale) stay within ~6e-4 absolute error
across the sine cycle (see binary output for the full table).

### Key observations

1. **Train and test MSE track closely** at convergence -- hypercube inductive
   bias regularizes without weight decay on this task.
2. **Small-batch noise** can produce non-monotone MSE mid-run (e.g. epoch 30);
   cosine floor damps late epochs.
3. **Near-perfect fit is possible** on this synthetic task -- high R^2 is a
   smoke signal, not a claim about real RC workloads.
4. **Sample predictions** cover peaks, troughs, and zeros across the test
   window on original scale.

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
dynamics, multi-output heads, and domain metrics beyond synthetic R^2.
