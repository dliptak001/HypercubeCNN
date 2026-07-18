// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNNTrainHelpers.h"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace hcnn {

// -----------------------------------------------------------------------------
// Metrics
// -----------------------------------------------------------------------------

int argmax(const float* v, int n) {
    if (v == nullptr || n <= 0)
        throw std::invalid_argument("hcnn::argmax: need non-null v and n > 0");
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

float softmax_cross_entropy(const float* logits, int num_classes, int target) {
    if (logits == nullptr || num_classes <= 0)
        throw std::invalid_argument(
            "hcnn::softmax_cross_entropy: need non-null logits and num_classes > 0");
    if (target < 0 || target >= num_classes)
        throw std::invalid_argument(
            "hcnn::softmax_cross_entropy: target out of range");

    double max_l = logits[0];
    for (int i = 1; i < num_classes; ++i)
        if (logits[i] > max_l) max_l = logits[i];

    double sum_exp = 0.0;
    for (int i = 0; i < num_classes; ++i)
        sum_exp += std::exp(static_cast<double>(logits[i]) - max_l);

    return static_cast<float>(
        -(static_cast<double>(logits[target]) - max_l) + std::log(sum_exp));
}

HCNNClassEval evaluate_classification(HCNN& net,
                                      const float* flat_inputs,
                                      int input_length,
                                      const int* targets,
                                      int count) {
    if (flat_inputs == nullptr || targets == nullptr)
        throw std::invalid_argument(
            "hcnn::evaluate_classification: null inputs or targets");
    if (count <= 0)
        throw std::invalid_argument(
            "hcnn::evaluate_classification: count must be > 0");
    if (input_length <= 0)
        throw std::invalid_argument(
            "hcnn::evaluate_classification: input_length must be > 0");

    const int K = net.GetNumOutputs();
    std::vector<float> all_logits(static_cast<size_t>(count) * static_cast<size_t>(K));
    net.ForwardBatch(flat_inputs, input_length, count, all_logits.data());

    float total_loss = 0.0f;
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        const float* logits = all_logits.data() + static_cast<size_t>(i) * K;
        total_loss += softmax_cross_entropy(logits, K, targets[i]);
        if (argmax(logits, K) == targets[i]) ++correct;
    }

    HCNNClassEval r;
    r.loss = total_loss / static_cast<float>(count);
    r.correct = correct;
    r.count = count;
    r.accuracy = 100.0f * static_cast<float>(correct) / static_cast<float>(count);
    return r;
}

HCNNClassEval evaluate_classification(HCNN& net, const HCNNFlatDataset& ds) {
    if (ds.count <= 0 || ds.input_length <= 0)
        throw std::invalid_argument(
            "hcnn::evaluate_classification: empty HCNNFlatDataset");

    // Public fields can drift from the vectors if callers mutate them by hand.
    // Size-check here so we never pass undersized buffers into ForwardBatch.
    const size_t need_in =
        static_cast<size_t>(ds.count) * static_cast<size_t>(ds.input_length);
    if (ds.inputs.size() < need_in)
        throw std::invalid_argument(
            "hcnn::evaluate_classification: inputs.size() < count * input_length");
    if (ds.targets.size() < static_cast<size_t>(ds.count))
        throw std::invalid_argument(
            "hcnn::evaluate_classification: targets.size() < count");

    return evaluate_classification(net,
                                   ds.inputs.data(),
                                   ds.input_length,
                                   ds.targets.data(),
                                   ds.count);
}

// -----------------------------------------------------------------------------
// Flat dataset
// -----------------------------------------------------------------------------

void HCNNFlatDataset::reset(int n, int len) {
    if (n < 0 || len < 0)
        throw std::invalid_argument(
            "HCNNFlatDataset::reset: n and len must be >= 0");
    // Build temps first so a throw leaves *this fully unchanged (strong guarantee).
    std::vector<float> new_inputs(
        static_cast<size_t>(n) * static_cast<size_t>(len));
    std::vector<int> new_targets(static_cast<size_t>(n));
    inputs.swap(new_inputs);
    targets.swap(new_targets);
    count = n;
    input_length = len;
}

// -----------------------------------------------------------------------------
// Cosine LR
// -----------------------------------------------------------------------------

float cosine_lr(float lr_max, float lr_min, int epoch, int num_epochs) {
    if (num_epochs <= 1)
        return lr_max;
    if (epoch < 0)
        epoch = 0;
    if (epoch >= num_epochs)
        epoch = num_epochs - 1;

    const float progress =
        static_cast<float>(epoch) / static_cast<float>(num_epochs - 1);
    return lr_min + 0.5f * (lr_max - lr_min)
        * (1.0f + std::cos(static_cast<float>(std::numbers::pi) * progress));
}

// -----------------------------------------------------------------------------
// Dual checkpoint
// -----------------------------------------------------------------------------

void HCNNDualCheckpoint::reset() {
    best_loss_weights_.clear();
    best_acc_weights_.clear();
    best_loss_ = std::numeric_limits<float>::infinity();
    best_loss_acc_ = -1.0f;
    best_loss_epoch_ = 0;
    best_acc_ = -1.0f;
    best_acc_loss_ = std::numeric_limits<float>::infinity();
    best_acc_epoch_ = 0;
}

HCNNDualCheckpointUpdate HCNNDualCheckpoint::observe(const HCNN& net,
                                                     float loss,
                                                     float accuracy,
                                                     int epoch) {
    HCNNDualCheckpointUpdate u;

    if (loss < best_loss_
        || (loss == best_loss_ && accuracy > best_loss_acc_)) {
        best_loss_ = loss;
        best_loss_acc_ = accuracy;
        best_loss_epoch_ = epoch;
        best_loss_weights_ = net.GetWeights();
        u.new_best_loss = true;
    }

    if (accuracy > best_acc_
        || (accuracy == best_acc_ && loss < best_acc_loss_)) {
        best_acc_ = accuracy;
        best_acc_loss_ = loss;
        best_acc_epoch_ = epoch;
        best_acc_weights_ = net.GetWeights();
        u.new_best_acc = true;
    }

    return u;
}

void HCNNDualCheckpoint::restore_best_loss(HCNN& net) const {
    if (best_loss_weights_.empty())
        throw std::logic_error(
            "HCNNDualCheckpoint::restore_best_loss: no best-loss snapshot");
    net.SetWeights(best_loss_weights_);
}

void HCNNDualCheckpoint::restore_best_acc(HCNN& net) const {
    if (best_acc_weights_.empty())
        throw std::logic_error(
            "HCNNDualCheckpoint::restore_best_acc: no best-acc snapshot");
    net.SetWeights(best_acc_weights_);
}

} // namespace hcnn
