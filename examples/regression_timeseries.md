# Regression Timeseries -- Next-Step Prediction

End-to-end regression example for HypercubeCNN. Trains a two-layer network with the `TaskType::Regression` API and MSE loss to predict the next value of a sine wave from a high-dimensional synthetic reservoir state vector.

This is the canonical regression example for HCNN and the primary validation for using HCNN as a readout layer in reservoir computing (the HypercubeRC integration). The task mirrors what HypercubeRC's `CNNReadout` does in production: take a length-N state vector from a reservoir and predict a scalar target.

## What this example shows

- Constructing an `hcnn::HCNN` with `TaskType::Regression` and `Activation::TANH`
- Stacking two conv+pool stages for a deeper architecture with 2-hop receptive field
- Using FLATTEN readout for position-sensitive per-vertex weights
- Feeding contiguous `const float*` data via `TrainEpochRegression`
- Using `OptimizerType::ADAM` with cosine LR annealing and a floor
- Target centering (subtract train-set mean before training)
- Evaluating via `ForwardBatch` for efficient parallel inference
- DIM=12 (N=4,096 vertices, 19,425 parameters)

No external data is required -- the example synthesizes its own reservoir and data on every run.

## The synthetic task

A synthetic reservoir of N independent leaky tanh integrators is driven by sin(0.1*t). Each vertex has its own leak rate (U[0.05, 0.45]), input weight (U[-1, 1]), and bias (U[-0.5, 0.5]), so different vertices capture the input at different timescales. After a 200-step warmup, the example collects (state, target) pairs where the target is sin(0.1*(t+1)) -- one step ahead.

**No exact solution exists.** The target is a nonlinear function of the input history, and the model must learn which timescale combinations predict the next step. This is a real regression problem, not a sparse-recovery demo.

## Architecture

```
Input: N floats -> N vertices (DIM configurable, default 12)
  |
Conv1: 1 -> 16 channels, K=12, Activation::TANH, bias       DIM=12
Pool1: MAX (antipodal), DIM 12->11, N 4096->2048
  |
Conv2: 16 -> 16 channels, K=11, Activation::TANH, bias     DIM=11
Pool2: MAX (antipodal), DIM 11->10, N 2048->1024
  |
Readout: FLATTEN -> Linear(16384 -> 1) -> prediction
```

| Component | Parameters |
|-----------|------------|
| Conv1 kernel (1 x 16 x 12) + bias | 208 |
| Conv2 kernel (16 x 16 x 11) + bias | 2,832 |
| Readout weights (16,384 -> 1) + bias | 16,385 |
| **Total** | **19,425** |

FLATTEN treats every (channel, vertex) activation as an independent feature. The readout learns per-vertex weights -- well-suited to reservoir state where each vertex encodes the input at a distinct timescale and vertex identity carries information.

### Architectural choices

| Choice | Reason |
|--------|--------|
| Two conv+pool stages | 2-hop receptive field on the hypercube. The first conv sees DIM immediate neighbors; the second sees (DIM-1) neighbors in the pooled graph, integrating information from a 2-hop neighborhood in the original hypercube. |
| `Activation::TANH` | Smooth, symmetric, bounded in (-1, 1). Matches the reservoir's own nonlinearity. The gradient is everywhere smooth -- no kink at zero (unlike ReLU) which interacts better with the max-pool's non-smooth gradient. Uses Xavier/Glorot initialization. |
| `PoolType::MAX` (antipodal) | Pairs each vertex with its bitwise complement and keeps the larger activation. For tanh-bounded conv output this picks the more strongly responding member of each pair -- a useful inductive bias for extracting the strongest signal across phase-related vertices. |
| FLATTEN readout | Position-sensitive -- the readout learns per-vertex weights. Each reservoir vertex encodes the input at a distinct timescale, so vertex identity is informative. |
| 16 conv channels | Gives the model meaningful capacity for a target that is not exactly expressible by a single channel. |

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Task | `TaskType::Regression` | Loss defaults to `LossType::MSE` |
| Optimizer | Adam | Adaptive per-parameter LR handles the max-pool's non-smooth gradient |
| `lr_max` | 0.002 | Higher LR causes oscillation with two conv layers |
| `lr_min` | 2e-4 (10% of lr_max) | Floor prevents learning from stalling in final epochs |
| LR schedule | Cosine annealing with floor | `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch/total))` |
| Batch size | 32 | |
| Weight decay | 0.0 | No L2 -- the architecture's inductive bias provides sufficient regularization |
| Epochs | 50 | |
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
| Construction | `HCNN net(DIM, num_outputs)` | `HCNN net(DIM, 1, 1, TaskType::Regression)` |
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

No data files, no environment setup -- the example generates its own reservoir and data. At DIM=12 the default run completes in ~3 minutes (50 epochs x ~3.5s/epoch).

## Results

DIM=12, N=4,096 vertices, 19,425 parameters. 4,096 train / 1,024 test samples. Hardware: Windows 11, MinGW g++, 32 threads.

```
Epoch  Train MSE    Test MSE     1-R^2        LR        Time
  1    2.239e-04    2.229e-04    4.498e-04    0.00200   ~3s
  5    3.709e-06    3.731e-06    7.528e-06    0.00197   ~4s
 10    8.338e-07    8.315e-07    1.678e-06    0.00186   ~4s
 20    2.189e-07    2.218e-07    4.476e-07    0.00143   ~3s
 40    9.835e-08    9.850e-08    1.987e-07    0.00041   ~3s
 50    4.968e-08    4.918e-08    9.923e-08    0.00020   ~4s
```

| Metric | Value |
|--------|-------|
| Initial test MSE | 5.440e-1 |
| Final test MSE | **4.918e-8** |
| Final test 1-R^2 | **9.923e-8** |
| MSE reduction | 100.00% |
| Per-epoch wall time | ~3.5s |
| Total training time | ~3 min |

### Sample predictions (test set, original scale)

```
step  target      pred       err
   0    +0.642826   +0.643125  +2.998e-04
 128    +0.448033   +0.448306  +2.725e-04
 256    +0.228870   +0.228833  -3.715e-05
 384    -0.002701   -0.002576  +1.249e-04
 512    -0.234124   -0.234278  -1.538e-04
 640    -0.452856   -0.453215  -3.595e-04
 768    -0.646954   -0.646723  +2.313e-04
 896    -0.805903   -0.806068  -1.647e-04
```

### Key observations

**1. No overfitting.** Train MSE and test MSE track closely at convergence. The hypercube convolution's inductive bias constrains the function class enough to regularize without explicit weight decay.

**2. Non-monotone convergence.** MSE spikes at epoch 30 (2.3e-6, up from 2.2e-7 at epoch 20) before recovering to 4.9e-8 by epoch 50. This is stochastic noise from small batches (32 samples from 4,096 training points). The cosine LR schedule tames it -- by the final epochs the lower LR damps the oscillation and convergence is clean.

**3. Near-perfect fit.** The final 1-R^2 of 9.9e-8 (seven nines of variance explained) demonstrates that HCNN can learn an extremely precise projection from reservoir state to prediction target.

**4. Sample predictions are tight across the full sine cycle.** Peak errors of ~3e-4 at sine peaks/troughs, with no systematic phase lag.

## Adapting for your own data

1. **Replace the synthetic reservoir with your data source.** Input values must be in [-1, 1] (HCNN's embedding contract).
2. **Pick `DIM` to match your input dimensionality.** The embedding zero-pads if `input_length < 2^DIM`.
3. **Set `num_outputs` for your target dimension** -- 1 for scalar, K for multi-output. HCNN trains all outputs jointly.

## Implications for HypercubeRC integration

This result validates the core premise of Phase B: HCNN can learn a useful projection from a high-dimensional reservoir state to a scalar prediction target, with no hand-crafted feature engineering. The 1-R^2 of 9.9e-8 at DIM=12 with 19,425 parameters demonstrates that the architecture's inductive bias (hypercube convolution + antipodal pooling + FLATTEN readout) is appropriate for the readout task.
