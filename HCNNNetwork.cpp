#include "HCNNNetwork.h"

HCNNNetwork::HCNNNetwork(int dim)
    : start_dim(dim), current_dim(dim), readout(10, 16) {
    channel_counts.push_back(1);
}

void HCNNNetwork::add_conv(int radius, int c_out, bool use_relu, bool use_bias) {
    conv_layers.emplace_back(current_dim, radius, use_relu, use_bias);
    channel_counts.push_back(c_out);
    is_conv_layer.push_back(true);
}

void HCNNNetwork::add_pool(int reduce_by, PoolType type) {
    pool_layers.emplace_back(current_dim, reduce_by, type);
    current_dim -= reduce_by;
    is_conv_layer.push_back(false);
}

void HCNNNetwork::set_kernel(int layer_idx, const float* weights, int size) {
    int conv_idx = 0;
    for (size_t i = 0; i < is_conv_layer.size(); ++i) {
        if (is_conv_layer[i]) {
            if (conv_idx == layer_idx) {
                conv_layers[conv_idx].set_kernel(weights, size);
                return;
            }
            ++conv_idx;
        }
    }
}

void HCNNNetwork::set_bias(int layer_idx, const float* biases, int size) {
    int conv_idx = 0;
    for (size_t i = 0; i < is_conv_layer.size(); ++i) {
        if (is_conv_layer[i]) {
            if (conv_idx == layer_idx) {
                conv_layers[conv_idx].set_bias(biases, size);
                return;
            }
            ++conv_idx;
        }
    }
}

void HCNNNetwork::randomize_all_weights(float scale) {
    for (auto& layer : conv_layers) layer.randomize_weights(scale);
    readout.randomize_weights(scale);
}

void HCNNNetwork::embed_input(const float* raw_input, int input_length, float* first_layer_activations) const {
    int N = 1 << start_dim;
    if (input_length > N) {
        throw std::runtime_error("Input length exceeds hypercube size N = " + std::to_string(N));
    }
    for (int i = 0; i < input_length; ++i) {
        float val = raw_input[i];
        if (val < -1.0f || val > 1.0f) {
            throw std::runtime_error("Input value out of required range [-1.0, 1.0]. Value = " + std::to_string(val));
        }
        first_layer_activations[i] = val;
    }
    for (int i = input_length; i < N; ++i) {
        first_layer_activations[i] = 0.0f;
    }
}

void HCNNNetwork::forward(const float* first_layer_activations, float* logits) const {
    if (conv_layers.empty()) return;

    int current_N = 1 << start_dim;
    std::vector<float> buf1(current_N * 64);
    std::vector<float> buf2(current_N * 64);
    float* current = buf1.data();
    float* next_buf = buf2.data();

    for (int i = 0; i < current_N; ++i) current[i] = first_layer_activations[i];

    size_t conv_idx = 0, pool_idx = 0;
    int current_channels = 1;

    for (size_t i = 0; i < is_conv_layer.size(); ++i) {
        if (is_conv_layer[i]) {
            int c_out = channel_counts[conv_idx + 1];
            conv_layers[conv_idx].forward(current, next_buf, current_channels, c_out);
            current_channels = c_out;
            ++conv_idx;
        } else {
            pool_layers[pool_idx].forward(current, next_buf, current_channels);
            current_N = pool_layers[pool_idx].get_output_N();
            ++pool_idx;
        }
        std::swap(current, next_buf);
    }

    readout.forward(current, logits, current_N);
}

void HCNNNetwork::train_step(const float* raw_input, int input_length,
                             const float* target, float learning_rate) {
    int N = 1 << start_dim;
    std::vector<float> embedded(N, 0.0f);
    embed_input(raw_input, input_length, embedded.data());

    std::vector<float> logits(10, 0.0f);
    forward(embedded.data(), logits.data());

    // Simple MSE gradient w.r.t. logits
    std::vector<float> grad_logits(10);
    for (int i = 0; i < 10; ++i) {
        grad_logits[i] = 2.0f * (logits[i] - target[i]);
    }

    // Apply SGD update to readout weights
    readout.apply_sgd_update(grad_logits, learning_rate);
}