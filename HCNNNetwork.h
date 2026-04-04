#pragma once

#include "HCNN.h"
#include "HCNNPool.h"
#include "HCNNReadout.h"
#include <vector>
#include <stdexcept>

class HCNNNetwork {
public:
    HCNNNetwork(int start_dim, int num_classes = 10);

    void add_conv(int radius, int c_out, bool use_relu = true, bool use_bias = true);
    void add_pool(int reduce_by, PoolType type = PoolType::MAX);

    void randomize_all_weights(float scale = 0.1f);

    void embed_input(const float* raw_input, int input_length,
                     float* first_layer_activations) const;

    void forward(const float* first_layer_activations, float* logits) const;

    void train_step(const float* raw_input, int input_length,
                    int target_class, float learning_rate);

    int get_start_dim() const { return start_dim; }
    int get_start_N() const { return 1 << start_dim; }

private:
    int start_dim;
    int current_dim;
    int num_classes;
    std::vector<HCNN> conv_layers;
    std::vector<HCNNPool> pool_layers;
    std::vector<bool> is_conv_layer;
    std::vector<int> channel_counts;
    HCNNReadout readout;
};
