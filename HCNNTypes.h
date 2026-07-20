// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

/**
 * @file HCNNTypes.h
 * @brief Public enums for the HypercubeCNN SDK front door.
 *
 * Included by `HCNN.h`.  Ordinary apps should not need layer headers
 * (`HCNNNetwork`, `HCNNConv`, …) — those are advanced / internal surfaces.
 */

namespace hcnn {

/// Activation after convolution (and optional batch-norm).
///
/// - `NONE`: identity
/// - `RELU` / `LEAKY_RELU`: rectified linear (He init when c_in > 1)
/// - `TANH`: smooth, bounded (-1, 1) (Xavier init)
enum class Activation { NONE, RELU, LEAKY_RELU, TANH };

/// Weight-update rule (set via `HCNN::SetOptimizer`).  Default is Adam.
enum class OptimizerType { SGD, ADAM };

/// Antipodal pool reduction.
enum class PoolType { MAX, AVG };

/// Task the network is trained for.  Selects train API family and default loss.
///
/// - `Classification`: int class targets; softmax + cross-entropy
/// - `Regression`: float targets of length `num_outputs`; MSE
enum class TaskType { Classification, Regression };

/// Loss function.  Prefer `Default` (resolved from `TaskType`).
/// Explicit `CrossEntropy` / `MSE` must match the task or construction throws.
/// Future losses (Huber, …) can extend this enum without an API break.
enum class LossType { Default, CrossEntropy, MSE };

/**
 * Loop nest for readout `grad_in = W^T * grad_logits` (advanced A/B knob).
 * Same math; different memory traffic.  Default `OutputOuter`.
 * Ordinary apps never need to set this.
 */
enum class ReadoutGradInLoop {
    FeatureOuter,
    OutputOuter
};

} // namespace hcnn
