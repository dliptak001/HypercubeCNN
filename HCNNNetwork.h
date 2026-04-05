#pragma once

#include "HCNN.h"
#include "HCNNPool.h"
#include "HCNNReadout.h"
#include <vector>
#include <stdexcept>

class HCNNNetwork {
public:
    HCNNNetwork(int start_dim, int num_classes = 10);

    void add_conv(int c_out, bool use_relu = true, bool use_bias = true);
    void add_pool(int reduce_by, PoolType type = PoolType::MAX);

    void randomize_all_weights(float scale = 0.1f);

    void embed_input(const float* raw_input, int input_length,
                     float* first_layer_activations) const;

    void forward(const float* first_layer_activations, float* logits) const;

    void train_step(const float* raw_input, int input_length,
                    int target_class, float learning_rate, float momentum = 0.0f);

    int get_start_dim() const { return start_dim; }
    int get_start_N() const { return 1 << start_dim; }
    int get_num_classes() const { return num_classes; }

    HCNN& get_conv(size_t i) { return conv_layers[i]; }
    HCNNReadout& get_readout() { return readout; }
    size_t get_num_conv() const { return conv_layers.size(); }
    size_t get_num_pool() const { return pool_layers.size(); }
    const std::vector<bool>& get_layer_types() const { return is_conv_layer; }
    const std::vector<int>& get_channel_counts() const { return channel_counts; }

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
