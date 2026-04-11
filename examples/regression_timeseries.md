# Regression Timeseries -- Next-Step Prediction

End-to-end regression example for HypercubeCNN. Trains a two-layer deep network with the `TaskType::Regression` API and MSE loss to predict the next value of a sine wave from a high-dimensional synthetic reservoir state vector.

This is the canonical regression example for HCNN and the primary validation for using HCNN as a readout layer in reservoir computing (the HypercubeRC integration). The task mirrors what HypercubeRC's `CNNReadout` does in production: take a length-N state vector from a reservoir and predict a scalar target.

## What this example shows

- Constructing an `hcnn::HCNN` with `TaskType::Regression` and `Activation::TANH`
- Stacking two conv+pool stages for a deeper architecture with 2-hop receptive field
- Feeding contiguous `const float*` data via `TrainEpochRegression`
- Using `OptimizerType::ADAM` with cosine LR annealing and a floor
- Target centering (subtract train-set mean before training)
- Evaluating via `ForwardBatch` for efficient parallel inference
- DIM=12 (N=4,096 vertices, 3,057 parameters)

No external data is required -- the example synthesizes its own reservoir and data on every run.

## The synthetic task

A synthetic reservoir of N independent leaky tanh integrators is driven by sin(0.1*t). Each vertex has its own leak rate (U[0.05, 0.45]), input weight (U[-1, 1]), and bias (U[-0.5, 0.5]), so different vertices capture the input at different timescales. After a 200-step warmup, the example collects (state, target) pairs where the target is sin(0.1*(t+1)) -- one step ahead.

**No exact solution exists.** The target is a nonlinear function of the input history, and the model must learn which timescale combinations predict the next step. This is a real regression problem, not a sparse-recovery demo.

## Architecture

```
Input: N floats -> N vertices (DIM configurable, default 12)
  |
Conv1: 1 -> 16 channels, K=DIM, Activation::TANH, bias     DIM=12
Pool1: MAX (antipodal), DIM -> DIM-1, N -> N/2              DIM=11
  |
Conv2: 16 -> 16 channels, K=DIM-1, Activation::TANH, bias  DIM=11
Pool2: MAX (antipodal), DIM-1 -> DIM-2, N/2 -> N/4         DIM=10
  |
Readout: GAP per channel -> Linear(16 -> 1) -> prediction
```

| Component | Parameters |
|-----------|------------|
| Conv1 kernel (1 x 16 x DIM) | 16 * DIM |
| Conv1 bias | 16 |
| Conv2 kernel (16 x 16 x (DIM-1)) | 16 * 16 * (DIM-1) |
| Conv2 bias | 16 |
| Readout weights (16 -> 1) | 16 |
| Readout bias | 1 |
| **Total** | **16 * DIM + 256 * (DIM-1) + 49** |

At DIM=12: **3,057 parameters** predicting from a 4,096-dimensional state (1.3:1 compression). The second conv layer dominates the parameter count (2,832 of 3,057).

### Architectural choices

| Choice | Reason |
|--------|--------|
| Two conv+pool stages | 2-hop receptive field on the hypercube. The first conv sees DIM immediate neighbors; the second sees (DIM-1) neighbors in the pooled graph, integrating information from a 2-hop neighborhood in the original hypercube. More realistic template for HypercubeRC readout. |
| `Activation::TANH` | Smooth, symmetric, bounded in (-1, 1). Matches the reservoir's own nonlinearity. The gradient is everywhere smooth -- no kink at zero (unlike ReLU) which interacts better with the max-pool's non-smooth gradient. Uses Xavier/Glorot initialization. |
| `PoolType::MAX` (antipodal) | Pairs each vertex with its bitwise complement and keeps the larger activation. For tanh-bounded conv output this picks the more strongly responding member of each pair -- a useful inductive bias for extracting the strongest signal across phase-related vertices. |
| `ReadoutType::GAP` | Averages post-pool activations across vertices, producing one scalar per channel. Translation-invariant -- the readout doesn't depend on which specific vertex has a given activation, only on the per-channel distribution. |
| 16 conv channels | Gives the model meaningful capacity for a target that is not exactly expressible by a single channel. |

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Task | `TaskType::Regression` | Loss defaults to `LossType::MSE` |
| Optimizer | Adam | Adaptive per-parameter LR handles the max-pool's non-smooth gradient |
| `lr_max` | 0.002 | Conservative for deeper net -- higher LR causes oscillation with two conv layers |
| `lr_min` | 2e-4 (10% of lr_max) | Floor prevents learning from stalling in final epochs |
| LR schedule | Cosine annealing with floor | `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch/total))` |
| Batch size | 32 | |
| Weight decay | 0.0 | No L2 -- the architecture's inductive bias provides sufficient regularization |
| Epochs | 300 | |
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

No data files, no environment setup -- the example generates its own reservoir and data. At DIM=12 the default run completes in ~30 minutes (300 epochs x ~6s/epoch). For higher scale, bump DIM to 14 or 16.

## DIM=12 results (N=4,096 vertices, 3,057 parameters)

Run date: 2026-04-11. Hardware: Windows 11, MinGW g++, 32 threads.

| Metric | Value |
|--------|-------|
| Initial test MSE | 5.073e-1 |
| Initial target_var | 4.956e-1 |
| Initial 1-R^2 | 1.023 |
| Final test MSE | **4.247e-7** |
| Final test 1-R^2 | **8.569e-7** |
| MSE reduction | 100.00% |
| Per-epoch wall time | ~6s |
| Total training time | ~30 min |

### Key observations

**1. No overfitting.** Train MSE and test MSE track closely throughout training. The hypercube convolution's inductive bias constrains the function class enough to regularize without explicit weight decay.

**2. Stochastic noise from small batches settles as LR decays.** With batch_size=32 and 4,096 training samples, gradient estimates are noisy. This causes epoch-to-epoch oscillation early in training. The cosine LR schedule naturally tames this -- by epoch 200+ the oscillation amplitude is negligible and convergence is clean.

**3. The lr_min floor matters.** The 10% floor keeps learning through the final epochs rather than stalling as cosine annealing approaches zero.

**4. Sub-millionth 1-R^2.** The final 1-R^2 of 8.6e-7 with 3,057 parameters demonstrates that HCNN can learn an extremely precise projection from reservoir state to prediction target.

**5. Sample predictions are tight across the full sine cycle.** Peak errors of ~1e-3 at sine peaks/troughs, with no systematic phase lag. Earlier (shallower, fewer epochs) versions showed phase lag at peaks -- the deeper architecture and longer training eliminate it.

**6. Target centering is essential.** Even though sine targets are nearly zero-mean in expectation, the empirical train-set mean is small but nonzero. Centering improves early-epoch convergence and is always the right preprocessing for regression.

## Extending this example

To adapt for your own regression task:

1. **Replace the synthetic reservoir with your data source.** Input values must be in [-1, 1] (HCNN's embedding contract).
2. **Pick `DIM` to match your input dimensionality.** The embedding zero-pads if `input_length < 2^DIM`.
3. **Pick `num_outputs` for your target dimension** -- 1 for scalar, K for multi-output. HCNN trains all outputs jointly.
4. **For time-series tasks, use TANH activation.** It matches typical reservoir nonlinearities and produces smooth gradients.
5. **For tasks with non-smooth gradients (max-pool), prefer Adam over SGD.** Adam's per-parameter adaptive scaling navigates the max-pool gradient landscape more effectively.
6. **Always center your targets.** Subtract the train-set mean before training, add it back at inference time. This is what HypercubeRC's `CNNReadout` does automatically.
7. **For deeper nets, use a conservative LR.** Two conv+pool stages need lower LR (0.002) than a single-layer net (0.01) to avoid oscillation.

## Implications for HypercubeRC integration

This result validates the core premise of Phase B: HCNN can learn a useful projection from a high-dimensional reservoir state to a scalar prediction target, with no hand-crafted feature engineering. The 1-R^2 of 8.6e-7 at DIM=12 with 3,057 parameters demonstrates that the architecture's inductive bias (hypercube convolution + antipodal pooling + GAP) is appropriate for the readout task.

Key takeaways:

1. **Two conv+pool stages work well.** The deeper architecture converges to much tighter predictions than a single layer, at the cost of more training time per epoch.
2. **Budget 300 epochs at DIM=12.** Per-epoch cost is ~6s. Total ~30 min is acceptable for offline readout fitting.
3. **Target centering and input standardization are essential.** This example handles centering; the `CNNReadout` adapter handles both centering and per-vertex standardization.
