#pragma once

#include "HCNN.h"
#include "HCNNPool.h"
#include "HCNNReadout.h"
#include <memory>
#include <vector>
#include <stdexcept>

class ThreadPool;

/// Readout strategy after the final conv/pool layer.
/// GAP: global average pooling per channel → linear (translation-invariant).
/// FLATTEN: concatenate all channel×vertex activations → linear (position-sensitive).
enum class ReadoutType { GAP, FLATTEN };

class HCNNNetwork {
public:
    HCNNNetwork(int start_dim, int num_classes = 10,
                ReadoutType readout_type = ReadoutType::GAP,
                size_t num_threads = 0);
    ~HCNNNetwork();

    HCNNNetwork(const HCNNNetwork&) = delete;
    HCNNNetwork& operator=(const HCNNNetwork&) = delete;
    HCNNNetwork(HCNNNetwork&&) = delete;
    HCNNNetwork& operator=(HCNNNetwork&&) = delete;

    void add_conv(int c_out, bool use_relu = true, bool use_bias = true);
    void add_pool(PoolType type = PoolType::MAX);

    /// Initialize all weights.  scale > 0: uniform [-scale, +scale].
    /// scale <= 0 (default): Xavier/Glorot uniform per layer.
    void randomize_all_weights(float scale = 0.0f);

    void embed_input(const float* raw_input, int input_length,
                     float* first_layer_activations) const;

    void forward(const float* first_layer_activations, float* logits) const;

    void train_step(const float* raw_input, int input_length,
                    int target_class, float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f);

    /// Mini-batch training: process batch_size samples in parallel, average
    /// gradients, then apply a single weight update. Requires ThreadPool.
    void train_batch(const float* const* inputs, const int* input_lengths,
                     const int* targets, int batch_size,
                     float learning_rate, float momentum = 0.0f,
                     float weight_decay = 0.0f);

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
    ReadoutType readout_type;
    int readout_N{1};          // N passed to readout: 1 for FLATTEN, final_N for GAP
    std::vector<HCNN> conv_layers;
    std::vector<HCNNPool> pool_layers;
    std::vector<bool> is_conv_layer;
    std::vector<int> channel_counts;
    HCNNReadout readout;
    std::unique_ptr<ThreadPool> thread_pool;
};
