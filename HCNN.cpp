// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNN.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace hcnn {

HCNN::HCNN(int start_dim, int num_outputs, int input_channels,
           TaskType task_type, LossType loss_type,
           size_t num_threads)
    : net_(std::make_unique<HCNNNetwork>(start_dim, num_outputs, input_channels,
                                         task_type, loss_type,
                                         num_threads)) {}

HCNN::~HCNN() = default;

// ---------------------------------------------------------------------------
//  Architecture
// ---------------------------------------------------------------------------
void HCNN::AddConv(int c_out, Activation activation,
                   bool use_bias, bool use_batchnorm) {
    net_->add_conv(c_out, activation, use_bias, use_batchnorm);
}

void HCNN::AddPool(PoolType type) {
    net_->add_pool(type);
}

void HCNN::RandomizeWeights(float scale, unsigned seed) {
    net_->randomize_all_weights(scale, seed);
}

// ---------------------------------------------------------------------------
//  Mode / optimizer
// ---------------------------------------------------------------------------
void HCNN::SetTraining(bool training) {
    net_->set_training(training);
}

void HCNN::SetOptimizer(OptimizerType type, float beta1, float beta2, float eps) {
    net_->set_optimizer(type, beta1, beta2, eps);
}

void HCNN::SetReadoutGradInLoop(ReadoutGradInLoop loop) {
    net_->get_readout().set_grad_in_loop(loop);
}

ReadoutGradInLoop HCNN::GetReadoutGradInLoop() const {
    return net_->get_readout().get_grad_in_loop();
}

void HCNN::PrepareBuffers() {
    net_->prepare_all_buffers();
}

// ---------------------------------------------------------------------------
//  Inference
// ---------------------------------------------------------------------------
void HCNN::Embed(const float* raw_input, int input_length,
                 float* embedded_out) const {
    net_->embed_input(raw_input, input_length, embedded_out);
}

void HCNN::Forward(const float* embedded, float* logits) const {
    net_->forward(embedded, logits);
}

void HCNN::ForwardBatch(const float* flat_inputs, int input_length,
                        int batch_size, float* logits_out) {
    net_->forward_batch(flat_inputs, input_length, batch_size, logits_out);
}

// ---------------------------------------------------------------------------
//  Training — classification
// ---------------------------------------------------------------------------
void HCNN::TrainStep(const float* raw_input, int input_length, int target_class,
                     float learning_rate, float momentum, float weight_decay,
                     const float* class_weights) {
    net_->train_step(raw_input, input_length, target_class, learning_rate,
                     momentum, weight_decay, class_weights);
}

void HCNN::TrainBatch(const float* flat_inputs, int input_length,
                      const int* targets, int batch_size,
                      float learning_rate, float momentum, float weight_decay,
                      const float* class_weights) {
    net_->train_batch(flat_inputs, input_length, targets, batch_size,
                      learning_rate, momentum, weight_decay, class_weights);
}

template <typename GatherTargets, typename TrainChunk>
void HCNN::train_epoch_impl_(const float* flat_inputs, int input_length,
                             int sample_count, int batch_size,
                             unsigned shuffle_seed,
                             GatherTargets&& gather_targets,
                             TrainChunk&& train_chunk) {
    if (batch_size <= 0) {
        throw std::invalid_argument("HCNN::TrainEpoch*: batch_size must be > 0");
    }
    if (sample_count < 0) {
        throw std::invalid_argument("HCNN::TrainEpoch*: sample_count must be >= 0");
    }
    if (sample_count == 0) return;

    const auto n  = static_cast<size_t>(sample_count);
    const auto il = static_cast<size_t>(input_length);
    const auto bs = static_cast<size_t>(batch_size);

    if (shuffle_seed != 0) {
        if (shuffle_idx_.size() < n) shuffle_idx_.resize(n);
        std::iota(shuffle_idx_.begin(),
                  shuffle_idx_.begin() + static_cast<std::ptrdiff_t>(n), 0);
        std::mt19937 rng(shuffle_seed);
        std::shuffle(shuffle_idx_.begin(),
                     shuffle_idx_.begin() + static_cast<std::ptrdiff_t>(n), rng);
        if (shuffle_inputs_.size() < bs * il)
            shuffle_inputs_.resize(bs * il);
    }

    for (int start = 0; start < sample_count; start += batch_size) {
        const int chunk = std::min(batch_size, sample_count - start);

        if (shuffle_seed != 0) {
            for (int i = 0; i < chunk; ++i) {
                const int j = shuffle_idx_[static_cast<size_t>(start + i)];
                std::memcpy(shuffle_inputs_.data() + static_cast<size_t>(i) * il,
                            flat_inputs + static_cast<size_t>(j) * il,
                            il * sizeof(float));
                gather_targets(i, j);
            }
            train_chunk(shuffle_inputs_.data(), chunk, start, /*shuffled=*/true);
        } else {
            train_chunk(flat_inputs + static_cast<size_t>(start) * il,
                        chunk, start, /*shuffled=*/false);
        }
    }
}

void HCNN::TrainEpoch(const float* flat_inputs, int input_length,
                      const int* targets, int sample_count, int batch_size,
                      float learning_rate, float momentum, float weight_decay,
                      const float* class_weights, unsigned shuffle_seed) {
    if (shuffle_seed != 0) {
        const auto bs = static_cast<size_t>(batch_size);
        if (shuffle_targets_.size() < bs) shuffle_targets_.resize(bs);
    }

    train_epoch_impl_(
        flat_inputs, input_length, sample_count, batch_size, shuffle_seed,
        [&](int chunk_i, int sample_j) {
            shuffle_targets_[static_cast<size_t>(chunk_i)] = targets[sample_j];
        },
        [&](const float* inputs, int chunk, int start, bool shuffled) {
            const int* tgt = shuffled ? shuffle_targets_.data()
                                      : (targets + start);
            net_->train_batch(inputs, input_length, tgt, chunk,
                              learning_rate, momentum, weight_decay,
                              class_weights);
        });
}

// ---------------------------------------------------------------------------
//  Training — regression
// ---------------------------------------------------------------------------
void HCNN::TrainStepRegression(const float* raw_input, int input_length,
                               const float* target, float learning_rate,
                               float momentum, float weight_decay) {
    net_->train_step_regression(raw_input, input_length, target, learning_rate,
                                momentum, weight_decay);
}

void HCNN::TrainBatchRegression(const float* flat_inputs, int input_length,
                                const float* flat_targets, int batch_size,
                                float learning_rate, float momentum,
                                float weight_decay) {
    net_->train_batch_regression(flat_inputs, input_length, flat_targets,
                                 batch_size, learning_rate, momentum,
                                 weight_decay);
}

void HCNN::TrainEpochRegression(const float* flat_inputs, int input_length,
                                const float* flat_targets,
                                int sample_count, int batch_size,
                                float learning_rate, float momentum,
                                float weight_decay, unsigned shuffle_seed) {
    const auto K  = static_cast<size_t>(net_->get_num_outputs());
    const auto bs = static_cast<size_t>(batch_size);

    if (shuffle_seed != 0) {
        if (shuffle_targets_f_.size() < bs * K)
            shuffle_targets_f_.resize(bs * K);
    }

    train_epoch_impl_(
        flat_inputs, input_length, sample_count, batch_size, shuffle_seed,
        [&](int chunk_i, int sample_j) {
            std::memcpy(shuffle_targets_f_.data() + static_cast<size_t>(chunk_i) * K,
                        flat_targets + static_cast<size_t>(sample_j) * K,
                        K * sizeof(float));
        },
        [&](const float* inputs, int chunk, int start, bool shuffled) {
            const float* tgt = shuffled
                ? shuffle_targets_f_.data()
                : (flat_targets + static_cast<size_t>(start) * K);
            net_->train_batch_regression(inputs, input_length, tgt, chunk,
                                         learning_rate, momentum, weight_decay);
        });
}

// ---------------------------------------------------------------------------
//  Sizing accessors
// ---------------------------------------------------------------------------
int HCNN::GetStartDim() const       { return net_->get_start_dim(); }
int HCNN::GetStartN() const         { return net_->get_start_N(); }
int HCNN::GetCurrentDim() const     { return net_->get_current_dim(); }
int HCNN::GetInputChannels() const  { return net_->get_input_channels(); }
int HCNN::GetNumOutputs() const     { return net_->get_num_outputs(); }
size_t HCNN::GetNumConv() const     { return net_->get_num_conv(); }
size_t HCNN::GetNumPool() const     { return net_->get_num_pool(); }
TaskType HCNN::GetTaskType() const  { return net_->get_task_type(); }
LossType HCNN::GetLossType() const  { return net_->get_loss_type(); }
OptimizerType HCNN::GetOptimizerType() const { return net_->get_optimizer_type(); }
bool HCNN::WeightsInitialized() const { return net_->weights_initialized(); }

void HCNN::require_weights_initialized_(const char* api) const {
    if (!net_->weights_initialized()) {
        throw std::logic_error(
            std::string(api) + ": call RandomizeWeights() first "
            "(weight blob requires a sized FLATTEN head)");
    }
}

// ---------------------------------------------------------------------------
//  Weight serialization
// ---------------------------------------------------------------------------

size_t HCNN::GetWeightCount() const {
    require_weights_initialized_("HCNN::GetWeightCount");
    size_t total = 0;
    for (size_t i = 0; i < net_->get_num_conv(); ++i) {
        const auto& conv = net_->get_conv(i);
        total += static_cast<size_t>(conv.get_kernel_size());
        total += static_cast<size_t>(conv.get_bias_size());
        if (conv.has_batchnorm()) {
            const size_t p = static_cast<size_t>(conv.get_bn_param_size());
            total += 4 * p;  // gamma, beta, running_mean, running_var
        }
    }
    const auto& ro = net_->get_readout();
    total += static_cast<size_t>(ro.get_weight_size());
    total += static_cast<size_t>(ro.get_bias_size());
    return total;
}

std::vector<float> HCNN::GetWeights() const {
    require_weights_initialized_("HCNN::GetWeights");
    std::vector<float> blob;
    blob.reserve(GetWeightCount());

    for (size_t i = 0; i < net_->get_num_conv(); ++i) {
        const auto& conv = net_->get_conv(i);
        const float* k = conv.get_kernel_data();
        blob.insert(blob.end(), k, k + conv.get_kernel_size());
        const float* b = conv.get_bias_data();
        blob.insert(blob.end(), b, b + conv.get_bias_size());
        if (conv.has_batchnorm()) {
            const int p = conv.get_bn_param_size();
            const float* g = conv.get_bn_gamma_data();
            const float* bt = conv.get_bn_beta_data();
            const float* rm = conv.get_bn_running_mean_data();
            const float* rv = conv.get_bn_running_var_data();
            blob.insert(blob.end(), g, g + p);
            blob.insert(blob.end(), bt, bt + p);
            blob.insert(blob.end(), rm, rm + p);
            blob.insert(blob.end(), rv, rv + p);
        }
    }

    const auto& ro = net_->get_readout();
    const float* w = ro.get_weight_data();
    blob.insert(blob.end(), w, w + ro.get_weight_size());
    const float* b = ro.get_bias_data();
    blob.insert(blob.end(), b, b + ro.get_bias_size());

    return blob;
}

void HCNN::SetWeights(const std::vector<float>& blob,
                      bool reset_optimizer_moments) {
    require_weights_initialized_("HCNN::SetWeights");
    if (blob.size() != GetWeightCount()) {
        throw std::invalid_argument(
            "HCNN::SetWeights: blob size " + std::to_string(blob.size()) +
            " != weight count " + std::to_string(GetWeightCount()));
    }

    size_t offset = 0;

    for (size_t i = 0; i < net_->get_num_conv(); ++i) {
        auto& conv = net_->get_conv(i);
        int ks = conv.get_kernel_size();
        std::memcpy(conv.get_kernel_data(), blob.data() + offset,
                    static_cast<size_t>(ks) * sizeof(float));
        offset += static_cast<size_t>(ks);
        int bs = conv.get_bias_size();
        if (bs > 0) {
            std::memcpy(conv.get_bias_data(), blob.data() + offset,
                        static_cast<size_t>(bs) * sizeof(float));
            offset += static_cast<size_t>(bs);
        }
        if (conv.has_batchnorm()) {
            const int p = conv.get_bn_param_size();
            const size_t bytes = static_cast<size_t>(p) * sizeof(float);
            std::memcpy(conv.get_bn_gamma_data(), blob.data() + offset, bytes);
            offset += static_cast<size_t>(p);
            std::memcpy(conv.get_bn_beta_data(), blob.data() + offset, bytes);
            offset += static_cast<size_t>(p);
            std::memcpy(conv.get_bn_running_mean_data(), blob.data() + offset, bytes);
            offset += static_cast<size_t>(p);
            std::memcpy(conv.get_bn_running_var_data(), blob.data() + offset, bytes);
            offset += static_cast<size_t>(p);
        }
    }

    auto& ro = net_->get_readout();
    int ws = ro.get_weight_size();
    std::memcpy(ro.get_weight_data(), blob.data() + offset,
                static_cast<size_t>(ws) * sizeof(float));
    offset += static_cast<size_t>(ws);
    int rbs = ro.get_bias_size();
    std::memcpy(ro.get_bias_data(), blob.data() + offset,
                static_cast<size_t>(rbs) * sizeof(float));
    offset += static_cast<size_t>(rbs);

    if (offset != blob.size()) {
        throw std::logic_error(
            "HCNN::SetWeights: internal layout mismatch (offset "
            + std::to_string(offset) + " vs blob " + std::to_string(blob.size())
            + ")");
    }

    if (reset_optimizer_moments) {
        net_->reset_optimizer_moments();
    }
}

} // namespace hcnn
