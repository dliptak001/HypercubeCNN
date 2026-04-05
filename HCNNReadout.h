#pragma once

#include <vector>
#include <cstdint>
#include <random>

class HCNNReadout {
public:
    HCNNReadout(int num_classes, int input_channels);

    void randomize_weights(float scale, std::mt19937& rng);

    void forward(const float* in, float* out, int N) const;

    // Backward: computes grad_in (if non-null) and updates weights via SGD with optional momentum.
    void backward(const float* grad_logits, const float* in, int N,
                  float* grad_in, float learning_rate, float momentum = 0.0f,
                  float weight_decay = 0.0f);

    // Compute gradients without applying SGD update.
    void compute_gradients(const float* grad_logits, const float* in, int N,
                           float* grad_in, float* weight_grad, float* bias_grad) const;

    // Apply externally computed (averaged) gradients via momentum SGD.
    void apply_gradients(const float* weight_grad, const float* bias_grad,
                         float learning_rate, float momentum, float weight_decay = 0.0f);

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
    std::vector<float> weight_vel;  // momentum velocity for weights
    std::vector<float> bias_vel;    // momentum velocity for bias
};
