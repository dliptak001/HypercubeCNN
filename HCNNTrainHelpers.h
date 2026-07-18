// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include "HCNN.h"

#include <limits>
#include <vector>

namespace hcnn {

// =============================================================================
// Optional training-loop helpers (not part of the conv/pool graph).
//
// Extracted from the shipped demos so teaching examples stay thin:
//   - classification metrics (softmax CE, argmax, batch evaluate)
//   - cosine LR schedule with floor
//   - dual weight checkpoints (best test loss + best test accuracy)
//   - contiguous flat classification dataset for TrainEpoch / ForwardBatch
//
// Include this header only when you need these utilities.  HCNN itself does
// not depend on them.
// =============================================================================

// -----------------------------------------------------------------------------
// Metrics
// -----------------------------------------------------------------------------

/// Index of the maximum value in `v[0 .. n)`.  Requires n > 0.
[[nodiscard]] int argmax(const float* v, int n);

/// Numerically stable softmax cross-entropy for one sample.
/// `target` must be in [0, num_classes).
[[nodiscard]] float softmax_cross_entropy(const float* logits,
                                          int num_classes,
                                          int target);

/// Aggregate classification metrics over a dataset.
struct HCNNClassEval {
    float loss = 0.0f;       ///< Mean softmax CE
    float accuracy = 0.0f;   ///< Percent correct in [0, 100]
    int   correct = 0;
    int   count = 0;
};

/// Run `ForwardBatch` then compute mean CE and accuracy.
/// `flat_inputs` is `count * input_length` row-major; `targets` is `count` labels.
[[nodiscard]] HCNNClassEval evaluate_classification(
    HCNN& net,
    const float* flat_inputs,
    int input_length,
    const int* targets,
    int count);

// -----------------------------------------------------------------------------
// Flat classification dataset
// -----------------------------------------------------------------------------

/// Contiguous buffers for HCNN's flat TrainEpoch / ForwardBatch APIs.
struct HCNNFlatDataset {
    std::vector<float> inputs;   ///< count * input_length
    std::vector<int>   targets;  ///< count class indices
    int count = 0;
    int input_length = 0;

    /// Resize/reallocate for `n` samples of length `len`.  Contents undefined.
    void reset(int n, int len);

    /// Pointer to sample `i`'s input (length `input_length`).  No bounds check.
    [[nodiscard]] float* sample_input(int i) {
        return inputs.data() + static_cast<size_t>(i) * static_cast<size_t>(input_length);
    }
    [[nodiscard]] const float* sample_input(int i) const {
        return inputs.data() + static_cast<size_t>(i) * static_cast<size_t>(input_length);
    }
};

/// Convenience overload for HCNNFlatDataset.
[[nodiscard]] HCNNClassEval evaluate_classification(HCNN& net,
                                                    const HCNNFlatDataset& ds);

// -----------------------------------------------------------------------------
// Cosine LR schedule
// -----------------------------------------------------------------------------

/**
 * Cosine annealing from `lr_max` (epoch 0) down to `lr_min` (last epoch).
 *
 *   progress = epoch / max(num_epochs - 1, 1)   for epoch in [0, num_epochs)
 *   lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
 *
 * When `num_epochs <= 1`, returns `lr_max`.  Negative epoch is treated as 0;
 * epoch >= num_epochs is clamped to the last step (lr_min when num_epochs > 1).
 *
 * Typical MNIST schedule: lr_max = 1e-3, lr_min = 1e-4 (10% floor).
 */
[[nodiscard]] float cosine_lr(float lr_max, float lr_min,
                              int epoch, int num_epochs);

// -----------------------------------------------------------------------------
// Dual weight checkpoints
// -----------------------------------------------------------------------------

/// Flags returned by HCNNDualCheckpoint::observe.
struct HCNNDualCheckpointUpdate {
    bool new_best_loss = false;
    bool new_best_acc  = false;

    [[nodiscard]] bool any() const { return new_best_loss || new_best_acc; }
};

/**
 * Tracks two independent weight snapshots during training:
 *   - best (lowest) loss, with higher accuracy as tie-break
 *   - best (highest) accuracy, with lower loss as tie-break
 *
 * Uses HCNN::GetWeights / SetWeights.  The blob is **weights only**:
 *   - BN scale/shift (gamma/beta) are not currently included
 *   - optimizer state (SGD velocity / Adam m,v and network timestep) is not
 *     included
 *
 * Intended for **eval / export** of the best snapshot (as in the MNIST demo).
 * Restoring then continuing training reuses stale optimizer moments against
 * the restored weights unless the caller also resets optimizer state
 * (e.g. SetOptimizer) or otherwise re-inits moments.
 */
class HCNNDualCheckpoint {
public:
    /// Record metrics for this epoch; copy weights if a new best is found.
    /// `epoch` is 1-based for reporting (matches typical "Epoch e/E" logs).
    HCNNDualCheckpointUpdate observe(const HCNN& net,
                                     float loss,
                                     float accuracy,
                                     int epoch);

    [[nodiscard]] bool has_best_loss() const { return !best_loss_weights_.empty(); }
    [[nodiscard]] bool has_best_acc()  const { return !best_acc_weights_.empty(); }

    void restore_best_loss(HCNN& net) const;
    void restore_best_acc(HCNN& net) const;

    [[nodiscard]] float best_loss() const { return best_loss_; }
    [[nodiscard]] float best_loss_acc() const { return best_loss_acc_; }
    [[nodiscard]] int   best_loss_epoch() const { return best_loss_epoch_; }

    [[nodiscard]] float best_acc() const { return best_acc_; }
    [[nodiscard]] float best_acc_loss() const { return best_acc_loss_; }
    [[nodiscard]] int   best_acc_epoch() const { return best_acc_epoch_; }

    /// Clear stored weights and reset metrics to initial sentinels.
    void reset();

private:
    std::vector<float> best_loss_weights_;
    std::vector<float> best_acc_weights_;

    float best_loss_ = std::numeric_limits<float>::infinity();
    float best_loss_acc_ = -1.0f;
    int   best_loss_epoch_ = 0;

    float best_acc_ = -1.0f;
    float best_acc_loss_ = std::numeric_limits<float>::infinity();
    int   best_acc_epoch_ = 0;
};

} // namespace hcnn
