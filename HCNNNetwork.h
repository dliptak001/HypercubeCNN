#pragma once

#include "HCNNConv.h"
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
                int input_channels = 1,
                ReadoutType readout_type = ReadoutType::GAP,
                size_t num_threads = 0);
    ~HCNNNetwork();

    HCNNNetwork(const HCNNNetwork&) = delete;
    HCNNNetwork& operator=(const HCNNNetwork&) = delete;
    HCNNNetwork(HCNNNetwork&&) = delete;
    HCNNNetwork& operator=(HCNNNetwork&&) = delete;

    void add_conv(int c_out, Activation activation = Activation::RELU,
                  bool use_bias = true, bool use_batchnorm = false);
    void add_pool(PoolType type = PoolType::MAX);

    /// Set training mode (true) or eval mode (false) for all layers with BN.
    void set_training(bool training) const;

    /// Configure the optimizer for all layers. Resets timestep.
    void set_optimizer(OptimizerType type, float beta1 = 0.9f,
                       float beta2 = 0.999f, float eps = 1e-8f);

    /// Initialize all weights.  scale > 0: uniform [-scale, +scale].
    /// scale <= 0 (default): Xavier/Glorot uniform per layer.
    void randomize_all_weights(float scale = 0.0f, unsigned seed = 42);

    void embed_input(const float* raw_input, int input_length,
                     float* first_layer_activations) const;

    void forward(const float* first_layer_activations, float* logits) const;

    /// Batch inference: embed + forward for multiple samples in parallel.
    /// logits_out must have batch_size * num_classes floats.
    void forward_batch(const float* const* raw_inputs, const int* input_lengths,
                       int batch_size, float* logits_out);

    void train_step(const float* raw_input, int input_length,
                    int target_class, float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f,
                    const float* class_weights = nullptr);

    /// Mini-batch training: process batch_size samples in parallel, average
    /// gradients, then apply a single weight update. Requires ThreadPool.
    /// class_weights: optional per-class loss scaling (length num_classes).
    void train_batch(const float* const* inputs, const int* input_lengths,
                     const int* targets, int batch_size,
                     float learning_rate, float momentum = 0.0f,
                     float weight_decay = 0.0f,
                     const float* class_weights = nullptr);

    int get_start_dim() const { return start_dim; }
    int get_start_N() const { return 1 << start_dim; }
    int get_input_channels() const { return input_channels; }
    int get_num_classes() const { return num_classes; }

    HCNNConv& get_conv(size_t i) { return conv_layers[i]; }
    HCNNReadout& get_readout() { return readout; }
    size_t get_num_conv() const { return conv_layers.size(); }
    size_t get_num_pool() const { return pool_layers.size(); }
    const std::vector<bool>& get_layer_types() const { return is_conv_layer; }
    const std::vector<int>& get_channel_counts() const { return channel_counts; }

private:
    int start_dim;
    int current_dim;
    int num_classes;
    int input_channels;
    ReadoutType readout_type;
    int readout_N{1};          // N passed to readout: 1 for FLATTEN, final_N for GAP
    int adam_timestep_{0};     // Global optimizer timestep (incremented per train_step/train_batch)
    std::vector<HCNNConv> conv_layers;
    std::vector<HCNNPool> pool_layers;
    std::vector<bool> is_conv_layer;
    std::vector<int> channel_counts;
    HCNNReadout readout;
    std::unique_ptr<ThreadPool> thread_pool;

    // --- Persistent batch-training buffers (allocated once, reused every train_batch) ---
    struct LayerInfo { int N; int channels; };

    struct ThreadAccum {
        std::vector<std::vector<float>> conv_kernel_grad;
        std::vector<std::vector<float>> conv_bias_grad;
        std::vector<std::vector<float>> conv_bn_gamma_grad;
        std::vector<std::vector<float>> conv_bn_beta_grad;
        std::vector<std::vector<float>> conv_bn_mean;   // per-conv BN mean accumulator
        std::vector<std::vector<float>> conv_bn_var;    // per-conv BN var accumulator
        std::vector<float> readout_weight_grad;
        std::vector<float> readout_bias_grad;
    };

    struct ThreadBuf {
        struct LayerCache {
            std::vector<float> activation;
            std::vector<float> pre_act;
            std::vector<int> max_indices;
        };
        std::vector<LayerCache> cache;
        std::vector<float> logits, probs, grad_logits;
        std::vector<float> grad_a, grad_b;
        std::vector<float> rw_grad, rb_grad;
        std::vector<std::vector<float>> kg, bg;
        std::vector<std::vector<float>> bn_gg, bn_bg;  // per-conv BN gamma/beta grads
        std::vector<std::vector<float>> bn_save;        // per-conv BN inv_std cache
        std::vector<float> conv_work;     // work buf for HCNNConv::compute_gradients
        std::vector<float> readout_work;  // work buf for HCNNReadout::compute_gradients
    };

    bool batch_bufs_ready{false};
    std::vector<LayerInfo> layer_info_;
    std::vector<ThreadAccum> accum_;
    std::vector<ThreadBuf> tbufs_;

    void prepare_batch_buffers();
    void zero_accumulators();

    // --- Persistent inference buffers (allocated once, reused every forward_batch) ---
    struct InferenceBuf {
        std::vector<float> buf1, buf2;
        std::vector<float> embedded;
    };
    bool infer_bufs_ready{false};
    std::vector<InferenceBuf> ibufs_;
    int infer_max_layer_size_{0};

    void prepare_inference_buffers();

    // RAII guard to disable per-layer threading during batch dispatch
    // and restore it when the scope exits (including on exception).
    struct LayerThreadGuard {
        std::vector<HCNNConv>& layers;
        ThreadPool* pool;
        LayerThreadGuard(std::vector<HCNNConv>& l, ThreadPool* p) : layers(l), pool(p) {
            for (auto& layer : layers) layer.set_thread_pool(nullptr);
        }
        ~LayerThreadGuard() {
            for (auto& layer : layers) layer.set_thread_pool(pool);
        }
    };

    // RAII guard to suppress per-sample running-stats EMA updates during
    // batch-parallel forward passes, and restore on scope exit (including on exception).
    struct BNStatsGuard {
        std::vector<HCNNConv>& layers;
        BNStatsGuard(std::vector<HCNNConv>& l) : layers(l) {
            for (auto& layer : layers)
                if (layer.has_batchnorm()) layer.set_skip_running_stats(true);
        }
        ~BNStatsGuard() {
            for (auto& layer : layers)
                if (layer.has_batchnorm()) layer.set_skip_running_stats(false);
        }
    };
};
