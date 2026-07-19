// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include <vector>
#include <cstdint>
#include <random>

// OptimizerType enum — defined in HCNNConv.h (no circular dependency)
#include "HCNNConv.h"

namespace hcnn {

/**
 * Loop nest used when forming grad_in = W^T * grad_logits.
 *
 * Both orders compute the same sums (same add order per feature when
 * OutputOuter zeros then accumulates o = 0 .. O-1).  They differ only in
 * memory traffic / vectorization — for A/B timing of the linear head.
 *
 *   OutputOuter  — for each output o, stream weight row o into grad_in.
 *                  Sequential W reads; grad_in is RMW'd O times. (default)
 *   FeatureOuter — for each feature f, sum over outputs (legacy A/B).
 *                  Touches W column-strided (stride = num_features).
 */
enum class ReadoutGradInLoop {
    FeatureOuter,
    OutputOuter
};

/**
 * @class HCNNReadout
 * @brief Final pipeline stage: linear map from a flat feature vector to
 *        `num_outputs` real-valued scalars (FLATTEN head).
 *
 * The orchestrator builds features as every final (channel, vertex) activation
 * laid out channel-major and contiguous:
 *
 *   num_features = c_final * N_final
 *   in[c * N + v]  is feature index (c * N + v)
 *
 * Then:
 *
 *   out[o] = bias[o] + sum_f  weights[o, f] * in[f]
 *
 * No global average pool, no activation, no softmax — raw linear outputs.
 * Classification CE / regression MSE live upstream of this class.
 *
 * Owns: weight matrix [num_outputs × num_features] + bias + optimizer moments.
 *
 * Two backward paths mirror HCNNConv:
 *   - backward(): gradients + in-place optimizer step (TrainStep)
 *   - compute_gradients() + apply_gradients(): batch accumulate then apply
 *
 * Power-user class: ordinary SDK consumers should use HCNN.
 */
class HCNNReadout {
public:
    /// @param num_outputs  Output dimension (classes or regression dims).
    /// @param num_features Flat feature count (typically c_final * N_final).
    HCNNReadout(int num_outputs, int num_features);

    void randomize_weights(float scale, std::mt19937& rng);

    /// @param in   Length `num_features` (channel-major flatten of final map).
    /// @param out  Length `num_outputs`.
    void forward(const float* in, float* out) const;

    /// Gradients + optimizer step.  `grad_in` may be null (first-layer-like).
    void backward(const float* grad_logits, const float* in,
                  float* grad_in, float learning_rate, float momentum = 0.0f,
                  float weight_decay = 0.0f, int timestep = 0);

    /// Write raw weight/bias/input gradients; no weight update.
    /// `grad_in`, `weight_grad`, and `bias_grad` may each be null if unused.
    void compute_gradients(const float* grad_logits, const float* in,
                           float* grad_in, float* weight_grad, float* bias_grad) const;

    /// Apply averaged gradients via the configured optimizer.
    void apply_gradients(const float* weight_grad, const float* bias_grad,
                         float learning_rate, float momentum, float weight_decay = 0.0f,
                         int timestep = 0);

    /// Configure the optimizer. Allocates second-moment buffers for Adam.
    void set_optimizer(OptimizerType type, float beta1 = 0.9f,
                       float beta2 = 0.999f, float eps = 1e-8f);

    /// Select grad_in loop nest (A/B). Default OutputOuter. Does not affect
    /// forward, dW, or optimizer math — only how W^T * grad_logits is formed.
    void set_grad_in_loop(ReadoutGradInLoop loop) { grad_in_loop_ = loop; }
    ReadoutGradInLoop get_grad_in_loop() const { return grad_in_loop_; }

    OptimizerType get_optimizer_type() const { return optimizer_type_; }

    /// Zero first/second moments without changing weights or optimizer type.
    void clear_optimizer_moments();

    int get_num_outputs() const { return num_outputs; }
    int get_num_features() const { return num_features; }

    float* get_weight_data() { return weights.data(); }
    const float* get_weight_data() const { return weights.data(); }
    int get_weight_size() const { return static_cast<int>(weights.size()); }
    float* get_bias_data() { return bias.data(); }
    const float* get_bias_data() const { return bias.data(); }
    int get_bias_size() const { return static_cast<int>(bias.size()); }

private:
    /// Fill grad_in[0..num_features) = W^T * grad_logits using grad_in_loop_.
    void fill_grad_in(const float* grad_logits, float* grad_in) const;

    int num_outputs;
    int num_features;
    std::vector<float> weights;     // [num_outputs * num_features], row = output
    std::vector<float> bias;
    std::vector<float> weight_m;    // first moment (SGD velocity / Adam m)
    std::vector<float> bias_m;
    std::vector<float> weight_m2;   // second moment (Adam only)
    std::vector<float> bias_m2;
    OptimizerType optimizer_type_ = OptimizerType::SGD;
    float adam_beta1_ = 0.9f, adam_beta2_ = 0.999f, adam_eps_ = 1e-8f;
    ReadoutGradInLoop grad_in_loop_ = ReadoutGradInLoop::OutputOuter;
};

} // namespace hcnn
