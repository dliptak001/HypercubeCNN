# Regression Timeseries -- Next-Step Prediction

End-to-end regression example for HypercubeCNN. Trains a small network with the `TaskType::Regression` API and MSE loss to predict the next value of a sine wave from a high-dimensional synthetic reservoir state vector.

This is the canonical regression example for HCNN and the primary validation for using HCNN as a readout layer in reservoir computing (the HypercubeRC integration). The task mirrors what HypercubeRC's `CNNReadout` does in production: take a length-N state vector from a reservoir and predict a scalar target.

## What this example shows

- Constructing an `hcnn::HCNN` with `TaskType::Regression` and `Activation::TANH`
- Feeding contiguous `const float*` data via `TrainEpochRegression`
- Using `OptimizerType::ADAM` with cosine LR annealing and a floor
- Target centering (subtract train-set mean before training)
- Evaluating via `ForwardBatch` for efficient parallel inference
- Scaling to DIM=16 (N=65,536 vertices, 289 parameters)

No external data is required -- the example synthesizes its own reservoir and data on every run.

## The synthetic task

A synthetic reservoir of N independent leaky tanh integrators is driven by sin(0.1*t). Each vertex has its own leak rate (U[0.05, 0.45]), input weight (U[-1, 1]), and bias (U[-0.5, 0.5]), so different vertices capture the input at different timescales. After a 200-step warmup, the example collects (state, target) pairs where the target is sin(0.1*(t+1)) -- one step ahead.

**No exact solution exists.** The target is a nonlinear function of the input history, and the model must learn which timescale combinations predict the next step. This is a real regression problem, not a sparse-recovery demo.

## Architecture

```
Input: N floats -> N vertices (DIM configurable, default 16)
  |
Conv1: 1 -> 16 channels, K=DIM, Activation::TANH, bias
Pool1: MAX (antipodal), DIM -> DIM-1, N -> N/2
  |
Readout: GAP per channel -> Linear(16 -> 1) -> prediction
```

| Component | Parameters |
|-----------|------------|
| Conv kernel (1 x 16 x DIM) | 16 * DIM |
| Conv bias | 16 |
| Readout weights (16 -> 1) | 16 |
| Readout bias | 1 |
| **Total** | **16 * DIM + 33** |

At DIM=16: **289 parameters** predicting from a 65,536-dimensional state (227:1 compression).

### Architectural choices

| Choice | Reason |
|--------|--------|
| `Activation::TANH` | Smooth, symmetric, bounded in (-1, 1). Matches the reservoir's own nonlinearity. The gradient is everywhere smooth -- no kink at zero (unlike ReLU) which interacts better with the max-pool's non-smooth gradient. Uses Xavier/Glorot initialization. |
| `PoolType::MAX` (antipodal) | Pairs each vertex with its bitwise complement and keeps the larger activation. For tanh-bounded conv output this picks the more strongly responding member of each pair -- a useful inductive bias for extracting the strongest signal across phase-related vertices. |
| `ReadoutType::GAP` | Averages post-pool activations across vertices, producing one scalar per channel. Translation-invariant -- the readout doesn't depend on which specific vertex has a given activation, only on the per-channel distribution. |
| 16 conv channels | Gives the model meaningful capacity for a target that is not exactly expressible by a single channel. |

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Task | `TaskType::Regression` | Loss defaults to `LossType::MSE` |
| Optimizer | Adam | Adaptive per-parameter LR handles the max-pool's non-smooth gradient |
| `lr_max` | 0.005 | |
| `lr_min` | 5e-4 (10% of lr_max) | Floor prevents learning from stalling in final epochs |
| LR schedule | Cosine annealing with floor | `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch/total))` |
| Batch size | 32 | |
| Weight decay | 0.0 | No L2 -- the architecture's inductive bias provides sufficient regularization |
| Epochs | 400 | |
| Shuffle | per-epoch | `shuffle_seed = epoch + 1` |
| Target centering | Train-set mean subtracted | Standard regression hygiene |

## Key API patterns

### Contiguous data

All batch and epoch methods accept contiguous row-major buffers:

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

### Classification vs regression differences

| Step | Classification | Regression |
|------|---------------|------------|
| Construction | `HCNN net(DIM, num_outputs)` | `HCNN net(DIM, 1, 1, ReadoutType::GAP, TaskType::Regression)` |
| Target type | `const int*` class indices | `const float*` contiguous targets |
| Training | `TrainEpoch(inputs, N, targets, ...)` | `TrainEpochRegression(inputs, N, targets, ...)` |
| Loss | Softmax + cross-entropy | MSE |
| Forward output | Logits (apply softmax for probs) | Raw predictions (no transform) |

The conv/pool stack, forward pass, weight init, optimizer, and batch-parallelism machinery are **identical** between the two task types. Only the loss gradient and target type differ.

## How to run

Build and run (requires MinGW on PATH for runtime DLLs):

```bash
cmake --build cmake-build-release
./cmake-build-release/RegressionTimeseries
```

No data files, no environment setup -- the example generates its own reservoir and data. At DIM=16 the default run takes ~3 hours (400 epochs x ~26s/epoch). For a quick sanity check, edit the source to set DIM=10 (~1 minute total).

## DIM=16 results (N=65,536 vertices, 289 parameters)

Run date: 2026-04-10. Hardware: Windows 11, MinGW g++, 32 threads.

| Metric | Value |
|--------|-------|
| Initial test MSE | 5.416e-1 |
| Initial target_var | 4.956e-1 |
| Initial R^2 | -0.093 |
| Final test MSE | **3.999e-3** |
| Final test R^2 | **0.9919** |
| MSE reduction | 99.26% |
| Per-epoch wall time | ~26s |
| Total training time | ~3h |

### Convergence trajectory

| Epoch | lr | Test MSE | R^2 |
|-------|-----|----------|------|
| 1 | 0.005000 | 4.733e-1 | 0.045 |
| 10 | 0.004994 | 2.189e-1 | 0.558 |
| 50 | 0.004835 | 8.210e-2 | 0.834 |
| 100 | 0.004353 | 7.214e-2 | 0.854 |
| 150 | 0.003627 | 4.470e-2 | 0.910 |
| 200 | 0.002768 | 2.578e-2 | 0.948 |
| 250 | 0.001905 | 1.595e-2 | 0.968 |
| 300 | 0.001172 | 8.150e-3 | 0.984 |
| 350 | 0.000678 | 5.392e-3 | 0.989 |
| 400 | 0.000500 | 3.999e-3 | 0.992 |

### Key observations

**1. No overfitting.** Train MSE (3.953e-3) and test MSE (3.999e-3) differ by ~1%. The hypercube convolution's inductive bias constrains the function class enough to regularize without explicit weight decay.

**2. The lr_min floor matters.** With `lr_min = 0`, performance plateaus as cosine annealing drives the learning rate toward zero. The 10% floor keeps learning through the final epochs. The model was still improving at epoch 400 -- longer training would push R^2 higher.

**3. 289 parameters predict a 65,536-dimensional state.** The 227:1 compression ratio demonstrates that HCNN is learning a highly efficient projection, extracting only the timescale combinations that predict the next step.

**4. Per-sample errors show systematic phase lag.** Predictions at the start of the test set show errors of ~0.05-0.11 with a consistent positive bias (predictions lag the true target). This is expected for a 1-step-ahead predictor trained on a finite window -- the model is slightly "behind" the sine wave. This is a property of the task, not a deficiency of the architecture.

**5. Target centering is essential.** Even though sine targets are nearly zero-mean in expectation, the empirical train-set mean is small but nonzero. Centering improves early-epoch convergence and is always the right preprocessing for regression.

## Extending this example

To adapt for your own regression task:

1. **Replace the synthetic reservoir with your data source.** Input values must be in [-1, 1] (HCNN's embedding contract).
2. **Pick `DIM` to match your input dimensionality.** The embedding zero-pads if `input_length < 2^DIM`.
3. **Pick `num_outputs` for your target dimension** -- 1 for scalar, K for multi-output. HCNN trains all outputs jointly.
4. **For time-series tasks, use TANH activation.** It matches typical reservoir nonlinearities and produces smooth gradients.
5. **For tasks with non-smooth gradients (max-pool), prefer Adam over SGD.** Adam's per-parameter adaptive scaling navigates the max-pool gradient landscape more effectively.
6. **Always center your targets.** Subtract the train-set mean before training, add it back at inference time. This is what HypercubeRC's `CNNReadout` does automatically.

## Implications for HypercubeRC integration

This result validates the core premise of Phase B: HCNN can learn a useful projection from a high-dimensional reservoir state to a scalar prediction target, with no hand-crafted feature engineering. The 0.9919 R^2 at DIM=16 with only 289 parameters demonstrates that the architecture's inductive bias (hypercube convolution + antipodal pooling + GAP) is appropriate for the readout task.

Key takeaways:

1. **DIM=16 works.** No architecture changes needed to scale from DIM=10 to DIM=16.
2. **Budget 400 epochs at DIM=16.** Per-epoch cost is ~26s. Total ~3h is acceptable for offline readout fitting.
3. **Target centering and input standardization are essential.** This example handles centering; the `CNNReadout` adapter handles both centering and per-vertex standardization.
