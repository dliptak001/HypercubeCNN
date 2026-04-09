#pragma once

#include <vector>
#include <cstdint>
#include <random>

// OptimizerType enum — defined in HCNNConv.h (no circular dependency)
#include "HCNNConv.h"

namespace hcnn {

class HCNNReadout {
public:
    HCNNReadout(int num_classes, int input_channels);

    void randomize_weights(float scale, std::mt19937& rng);

    // work_buf: optional pre-allocated buffer of at least input_channels floats.
    void forward(const float* in, float* out, int N,
                 float* work_buf = nullptr) const;

    // Backward: computes grad_in (if non-null) and updates weights via SGD with optional momentum.
    void backward(const float* grad_logits, const float* in, int N,
                  float* grad_in, float learning_rate, float momentum = 0.0f,
                  float weight_decay = 0.0f, int timestep = 0);

    // Compute gradients without applying SGD update.
    // work_buf: optional pre-allocated buffer of at least input_channels floats.
    void compute_gradients(const float* grad_logits, const float* in, int N,
                           float* grad_in, float* weight_grad, float* bias_grad,
                           float* work_buf = nullptr) const;

    // Apply externally computed (averaged) gradients via momentum SGD.
    void apply_gradients(const float* weight_grad, const float* bias_grad,
                         float learning_rate, float momentum, float weight_decay = 0.0f,
                         int timestep = 0);

    /// Configure the optimizer. Allocates second-moment buffers for Adam.
    void set_optimizer(OptimizerType type, float beta1 = 0.9f,
                       float beta2 = 0.999f, float eps = 1e-8f);

    int get_num_classes() const { return num_classes; }
    int get_input_channels() const { return input_channels; }

    float* get_weight_data() { return weights.data(); }
    int get_weight_size() const { return static_cast<int>(weights.size()); }
    float* get_bias_data() { return bias.data(); }
    int get_bias_size() const { return static_cast<int>(bias.size()); }

private:
    int num_classes;
    int input_channels;
    std::vector<float> weights;
    std::vector<float> bias;
    std::vector<float> weight_m;    // first moment (SGD velocity / Adam m)
    std::vector<float> bias_m;      // first moment for bias
    std::vector<float> weight_m2;   // second moment (Adam only)
    std::vector<float> bias_m2;     // second moment (Adam only)
    OptimizerType optimizer_type_ = OptimizerType::SGD;
    float adam_beta1_ = 0.9f, adam_beta2_ = 0.999f, adam_eps_ = 1e-8f;
};

} // namespace hcnn
