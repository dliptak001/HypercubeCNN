#pragma once

#include "HCNN.h"
#include "HCNNPool.h"
#include "HCNNReadout.h"
#include <vector>
#include <stdexcept>

class HCNNNetwork {
public:
    HCNNNetwork(int start_dim);

    // Core layers
    void add_conv(int radius, bool use_relu = true, bool use_bias = true);
    void add_pool(int reduce_by, PoolType type = PoolType::MAX);

    // Configuration
    void set_kernel(int layer_idx, const float* weights, int size);
    void set_bias(int layer_idx, const float* biases, int size);
    void randomize_all_weights(float scale = 0.1f);

    // Core input embedding (Direct Linear Assignment)
    // input_length must be <= N; throws std::runtime_error otherwise
    void embed_input(const float* raw_input, int input_length, float* first_layer_activations) const;

    // Forward pass (after embedding)
    void forward(const float* first_layer_activations, float* logits) const;

    int get_start_dim() const { return start_dim; }
    int get_start_N() const { return 1 << start_dim; }

private:
    int start_dim;
    int current_dim;
    std::vector<HCNN> conv_layers;
    std::vector<HCNNPool> pool_layers;
    std::vector<bool> is_conv_layer;
    std::vector<int> channel_counts;
    HCNNReadout readout;
};