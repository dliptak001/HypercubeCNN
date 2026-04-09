#include "HCNN.h"

#include <algorithm>
#include <numeric>
#include <random>

HCNN::HCNN(int start_dim, int num_classes, int input_channels,
           ReadoutType readout_type, size_t num_threads)
    : net_(std::make_unique<HCNNNetwork>(start_dim, num_classes, input_channels,
                                         readout_type, num_threads)) {}

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

void HCNN::ForwardBatch(const float* const* raw_inputs, const int* input_lengths,
                        int batch_size, float* logits_out) {
    net_->forward_batch(raw_inputs, input_lengths, batch_size, logits_out);
}

// ---------------------------------------------------------------------------
//  Training
// ---------------------------------------------------------------------------
void HCNN::TrainStep(const float* raw_input, int input_length, int target_class,
                     float learning_rate, float momentum, float weight_decay,
                     const float* class_weights) {
    net_->train_step(raw_input, input_length, target_class, learning_rate,
                     momentum, weight_decay, class_weights);
}

void HCNN::TrainBatch(const float* const* inputs, const int* input_lengths,
                      const int* targets, int batch_size,
                      float learning_rate, float momentum, float weight_decay,
                      const float* class_weights) {
    net_->train_batch(inputs, input_lengths, targets, batch_size,
                      learning_rate, momentum, weight_decay, class_weights);
}

void HCNN::TrainEpoch(const float* const* inputs, const int* input_lengths,
                      const int* targets, int sample_count, int batch_size,
                      float learning_rate, float momentum, float weight_decay,
                      const float* class_weights, unsigned shuffle_seed) {
    if (batch_size <= 0) batch_size = 1;
    if (sample_count <= 0) return;

    // Shuffle path: gather inputs/lengths/targets into persistent scratch
    // buffers in a freshly permuted order, then iterate the gathered arrays
    // in contiguous chunks.  Buffers grow on demand and are reused across
    // calls -- after the first shuffled epoch, no allocations occur unless
    // sample_count grows.
    if (shuffle_seed != 0) {
        const auto n = static_cast<size_t>(sample_count);
        if (shuffle_idx_.size() < n) {
            shuffle_idx_.resize(n);
            shuffle_inputs_.resize(n);
            shuffle_lengths_.resize(n);
            shuffle_targets_.resize(n);
        }
        std::iota(shuffle_idx_.begin(), shuffle_idx_.begin() + n, 0);

        std::mt19937 rng(shuffle_seed);
        std::shuffle(shuffle_idx_.begin(), shuffle_idx_.begin() + n, rng);

        for (size_t i = 0; i < n; ++i) {
            int j = shuffle_idx_[i];
            shuffle_inputs_[i]  = inputs[j];
            shuffle_lengths_[i] = input_lengths[j];
            shuffle_targets_[i] = targets[j];
        }
        inputs        = shuffle_inputs_.data();
        input_lengths = shuffle_lengths_.data();
        targets       = shuffle_targets_.data();
    }

    for (int start = 0; start < sample_count; start += batch_size) {
        int chunk = std::min(batch_size, sample_count - start);
        net_->train_batch(inputs + start, input_lengths + start, targets + start,
                          chunk, learning_rate, momentum, weight_decay, class_weights);
    }
}

// ---------------------------------------------------------------------------
//  Sizing accessors
// ---------------------------------------------------------------------------
int HCNN::GetStartDim() const       { return net_->get_start_dim(); }
int HCNN::GetStartN() const         { return net_->get_start_N(); }
int HCNN::GetInputChannels() const  { return net_->get_input_channels(); }
int HCNN::GetNumClasses() const     { return net_->get_num_classes(); }
