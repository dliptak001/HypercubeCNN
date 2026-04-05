#include "HCNNNetwork.h"
#include <algorithm>
#include <cmath>
#include <random>

HCNNNetwork::HCNNNetwork(int dim, int num_classes)
    : start_dim(dim), current_dim(dim), num_classes(num_classes),
      readout(num_classes, 1) {
    channel_counts.push_back(1);
}

void HCNNNetwork::add_conv(int c_out, bool use_relu, bool use_bias) {
    int c_in = channel_counts.back();
    conv_layers.emplace_back(current_dim, c_in, c_out, use_relu, use_bias);
    channel_counts.push_back(c_out);
    is_conv_layer.push_back(true);
}

void HCNNNetwork::add_pool(int reduce_by, PoolType type) {
    pool_layers.emplace_back(current_dim, reduce_by, type);
    current_dim -= reduce_by;
    channel_counts.push_back(channel_counts.back());
    is_conv_layer.push_back(false);
}

void HCNNNetwork::randomize_all_weights(float scale) {
    // Rebuild readout with correct final channel count
    int final_channels = channel_counts.back();
    readout = HCNNReadout(num_classes, final_channels);

    std::mt19937 rng(42);
    for (auto& layer : conv_layers) {
        layer.randomize_weights(scale, rng);
    }
    readout.randomize_weights(scale, rng);
}

void HCNNNetwork::embed_input(const float* raw_input, int input_length,
                              float* out) const {
    int N = 1 << start_dim;
    if (input_length > N) {
        throw std::runtime_error("Input length exceeds hypercube size N = "
                                 + std::to_string(N));
    }
    for (int i = 0; i < input_length; ++i) {
        float val = raw_input[i];
        if (val < -1.0f || val > 1.0f) {
            throw std::runtime_error("Input value out of range [-1.0, 1.0]: "
                                     + std::to_string(val));
        }
        out[i] = val;
    }
    for (int i = input_length; i < N; ++i) {
        out[i] = 0.0f;
    }
}

void HCNNNetwork::forward(const float* first_layer_activations, float* logits) const {
    if (conv_layers.empty()) return;

    // Compute max buffer size
    int cur_N = 1 << start_dim;
    int max_size = cur_N;
    size_t ci = 0, pi = 0;
    for (size_t i = 0; i < is_conv_layer.size(); ++i) {
        if (is_conv_layer[i]) {
            max_size = std::max(max_size, conv_layers[ci].get_c_out() * cur_N);
            ++ci;
        } else {
            cur_N = pool_layers[pi].get_output_N();
            ++pi;
        }
    }

    std::vector<float> buf1(max_size);
    std::vector<float> buf2(max_size);
    float* current = buf1.data();
    float* next_buf = buf2.data();

    cur_N = 1 << start_dim;
    for (int i = 0; i < cur_N; ++i) current[i] = first_layer_activations[i];

    ci = 0; pi = 0;

    for (size_t i = 0; i < is_conv_layer.size(); ++i) {
        if (is_conv_layer[i]) {
            conv_layers[ci].forward(current, next_buf);
            ++ci;
        } else {
            pool_layers[pi].forward(current, next_buf, channel_counts[i]);
            cur_N = pool_layers[pi].get_output_N();
            ++pi;
        }
        std::swap(current, next_buf);
    }

    readout.forward(current, logits, cur_N);
}

void HCNNNetwork::train_step(const float* raw_input, int input_length,
                             int target_class, float learning_rate, float momentum) {
    int N = 1 << start_dim;
    std::vector<float> embedded(N, 0.0f);
    embed_input(raw_input, input_length, embedded.data());

    int num_layers = static_cast<int>(is_conv_layer.size());

    // Per-layer cache for backprop
    struct LayerCache {
        std::vector<float> activation;
        std::vector<float> pre_act;       // conv layers only
        std::vector<int> max_indices;     // max-pool layers only
        int N;
        int channels;
    };

    std::vector<LayerCache> cache(num_layers + 1);

    // Cache[0] = embedded input
    cache[0].N = N;
    cache[0].channels = 1;
    cache[0].activation.assign(embedded.begin(), embedded.end());

    int cur_N = N;
    size_t ci = 0, pi = 0;

    // Forward pass — store all intermediate activations
    for (int i = 0; i < num_layers; ++i) {
        auto& c = cache[i + 1];
        if (is_conv_layer[i]) {
            c.N = cur_N;
            c.channels = conv_layers[ci].get_c_out();
            c.activation.resize(c.channels * cur_N);
            c.pre_act.resize(c.channels * cur_N);
            conv_layers[ci].forward(cache[i].activation.data(),
                                    c.activation.data(), c.pre_act.data());
            ++ci;
        } else {
            c.N = pool_layers[pi].get_output_N();
            c.channels = cache[i].channels;
            c.activation.resize(c.channels * c.N);
            pool_layers[pi].forward(cache[i].activation.data(),
                                    c.activation.data(), c.channels,
                                    &c.max_indices);
            cur_N = c.N;
            ++pi;
        }
    }

    // Readout forward
    auto& final_c = cache[num_layers];
    std::vector<float> logits(num_classes, 0.0f);
    readout.forward(final_c.activation.data(), logits.data(), final_c.N);

    // Softmax + cross-entropy gradient
    // softmax: p[i] = exp(logits[i] - max) / sum(exp(logits[j] - max))
    // cross-entropy loss: L = -log(p[target_class])
    // gradient: dL/d(logits[i]) = p[i] - (i == target_class ? 1 : 0)
    float max_logit = logits[0];
    for (int i = 1; i < num_classes; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    std::vector<float> probs(num_classes);
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < num_classes; ++i) {
        probs[i] /= sum_exp;
    }

    std::vector<float> grad_logits(num_classes);
    for (int i = 0; i < num_classes; ++i) {
        grad_logits[i] = probs[i] - (i == target_class ? 1.0f : 0.0f);
    }

    // Backward through readout
    std::vector<float> grad_current(final_c.channels * final_c.N);
    readout.backward(grad_logits.data(), final_c.activation.data(),
                     final_c.N, grad_current.data(), learning_rate, momentum);

    // Backward through layers in reverse
    ci = conv_layers.size();
    pi = pool_layers.size();

    for (int i = num_layers - 1; i >= 0; --i) {
        std::vector<float> grad_prev(cache[i].channels * cache[i].N, 0.0f);

        if (is_conv_layer[i]) {
            --ci;
            conv_layers[ci].backward(grad_current.data(),
                                     cache[i].activation.data(),
                                     cache[i + 1].pre_act.data(),
                                     (i > 0) ? grad_prev.data() : nullptr,
                                     learning_rate, momentum);
        } else {
            --pi;
            pool_layers[pi].backward(grad_current.data(), grad_prev.data(),
                                     cache[i].channels, &cache[i + 1].max_indices);
        }

        grad_current = std::move(grad_prev);
    }
}
