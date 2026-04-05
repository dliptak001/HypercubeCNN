#include "HCNNNetwork.h"
#include "ThreadPool.h"
#include <algorithm>
#include <cmath>
#include <random>

HCNNNetwork::HCNNNetwork(int dim, int num_classes, size_t num_threads)
    : start_dim(dim), current_dim(dim), num_classes(num_classes),
      readout(num_classes, 1),
      thread_pool(std::make_unique<ThreadPool>(num_threads)) {
    channel_counts.push_back(1);
}

HCNNNetwork::~HCNNNetwork() = default;

void HCNNNetwork::add_conv(int c_out, bool use_relu, bool use_bias) {
    int c_in = channel_counts.back();
    conv_layers.emplace_back(current_dim, c_in, c_out, use_relu, use_bias);
    conv_layers.back().set_thread_pool(thread_pool.get());
    channel_counts.push_back(c_out);
    is_conv_layer.push_back(true);
}

void HCNNNetwork::add_pool(PoolType type) {
    pool_layers.emplace_back(current_dim, type);
    current_dim -= 1;
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

// ---------------------------------------------------------------------------
// Mini-batch training: samples are processed in parallel, gradients averaged,
// then a single weight update is applied.
// ---------------------------------------------------------------------------
void HCNNNetwork::train_batch(const float* const* inputs, const int* input_lengths,
                              const int* targets, int batch_size,
                              float learning_rate, float momentum) {
    if (batch_size <= 0) return;

    int N = 1 << start_dim;
    int num_layers = static_cast<int>(is_conv_layer.size());
    size_t num_conv = conv_layers.size();

    // Per-thread gradient accumulators
    size_t nt = thread_pool ? thread_pool->NumThreads() : 1;

    struct ThreadAccum {
        std::vector<std::vector<float>> conv_kernel_grad; // [conv_idx][kernel_size]
        std::vector<std::vector<float>> conv_bias_grad;   // [conv_idx][c_out]
        std::vector<float> readout_weight_grad;
        std::vector<float> readout_bias_grad;
    };

    std::vector<ThreadAccum> accum(nt);
    for (size_t t = 0; t < nt; ++t) {
        auto& a = accum[t];
        a.conv_kernel_grad.resize(num_conv);
        a.conv_bias_grad.resize(num_conv);
        for (size_t ci = 0; ci < num_conv; ++ci) {
            a.conv_kernel_grad[ci].assign(conv_layers[ci].get_kernel_size(), 0.0f);
            a.conv_bias_grad[ci].assign(conv_layers[ci].get_bias_size(), 0.0f);
        }
        a.readout_weight_grad.assign(readout.get_weight_size(), 0.0f);
        a.readout_bias_grad.assign(readout.get_bias_size(), 0.0f);
    }

    // Process samples in parallel, each thread accumulates into its own accum
    auto process_sample = [&](size_t tid, int sample_idx) {
        auto& a = accum[tid];

        std::vector<float> embedded(N, 0.0f);
        embed_input(inputs[sample_idx], input_lengths[sample_idx], embedded.data());

        // Per-layer cache
        struct LayerCache {
            std::vector<float> activation;
            std::vector<float> pre_act;
            std::vector<int> max_indices;
            int N;
            int channels;
        };
        std::vector<LayerCache> cache(num_layers + 1);

        cache[0].N = N;
        cache[0].channels = 1;
        cache[0].activation.assign(embedded.begin(), embedded.end());

        int cur_N = N;
        size_t ci = 0, pi = 0;

        // Forward pass
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
        float max_logit = logits[0];
        for (int i = 1; i < num_classes; ++i)
            if (logits[i] > max_logit) max_logit = logits[i];

        std::vector<float> probs(num_classes);
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            probs[i] = std::exp(logits[i] - max_logit);
            sum_exp += probs[i];
        }
        for (int i = 0; i < num_classes; ++i) probs[i] /= sum_exp;

        std::vector<float> grad_logits(num_classes);
        for (int i = 0; i < num_classes; ++i)
            grad_logits[i] = probs[i] - (i == targets[sample_idx] ? 1.0f : 0.0f);

        // Readout backward (compute_gradients only)
        std::vector<float> grad_current(final_c.channels * final_c.N);
        std::vector<float> rw_grad(readout.get_weight_size());
        std::vector<float> rb_grad(readout.get_bias_size());
        readout.compute_gradients(grad_logits.data(), final_c.activation.data(),
                                  final_c.N, grad_current.data(),
                                  rw_grad.data(), rb_grad.data());

        // Accumulate readout gradients
        for (int i = 0; i < readout.get_weight_size(); ++i)
            a.readout_weight_grad[i] += rw_grad[i];
        for (int i = 0; i < readout.get_bias_size(); ++i)
            a.readout_bias_grad[i] += rb_grad[i];

        // Backward through layers
        ci = conv_layers.size();
        pi = pool_layers.size();

        for (int i = num_layers - 1; i >= 0; --i) {
            std::vector<float> grad_prev(cache[i].channels * cache[i].N, 0.0f);

            if (is_conv_layer[i]) {
                --ci;
                std::vector<float> kg(conv_layers[ci].get_kernel_size());
                std::vector<float> bg(conv_layers[ci].get_bias_size());
                conv_layers[ci].compute_gradients(
                    grad_current.data(),
                    cache[i].activation.data(),
                    cache[i + 1].pre_act.data(),
                    (i > 0) ? grad_prev.data() : nullptr,
                    kg.data(), bg.empty() ? nullptr : bg.data());

                // Accumulate conv gradients
                for (int j = 0; j < conv_layers[ci].get_kernel_size(); ++j)
                    a.conv_kernel_grad[ci][j] += kg[j];
                for (int j = 0; j < conv_layers[ci].get_bias_size(); ++j)
                    a.conv_bias_grad[ci][j] += bg[j];
            } else {
                --pi;
                pool_layers[pi].backward(grad_current.data(), grad_prev.data(),
                                         cache[i].channels, &cache[i + 1].max_indices);
            }

            grad_current = std::move(grad_prev);
        }
    };

    if (thread_pool && batch_size > 1) {
        // Disable per-layer vertex threading during batch parallelism to
        // prevent nested ForEach on the same pool (ThreadPool is not reentrant).
        for (auto& layer : conv_layers)
            layer.set_thread_pool(nullptr);

        thread_pool->ForEach(static_cast<size_t>(batch_size),
            [&](size_t tid, size_t begin, size_t end) {
                for (size_t s = begin; s < end; ++s)
                    process_sample(tid, static_cast<int>(s));
            });

        // Restore per-layer threading for inference
        for (auto& layer : conv_layers)
            layer.set_thread_pool(thread_pool.get());
    } else {
        for (int s = 0; s < batch_size; ++s)
            process_sample(0, s);
    }

    // Reduce across threads and average
    float scale = 1.0f / static_cast<float>(batch_size);

    // Conv layers
    for (size_t ci = 0; ci < num_conv; ++ci) {
        int ks = conv_layers[ci].get_kernel_size();
        int bs = conv_layers[ci].get_bias_size();
        auto& base_kg = accum[0].conv_kernel_grad[ci];
        auto& base_bg = accum[0].conv_bias_grad[ci];

        // Sum thread accumulators into thread 0
        for (size_t t = 1; t < nt; ++t) {
            for (int j = 0; j < ks; ++j) base_kg[j] += accum[t].conv_kernel_grad[ci][j];
            for (int j = 0; j < bs; ++j) base_bg[j] += accum[t].conv_bias_grad[ci][j];
        }

        // Average
        for (int j = 0; j < ks; ++j) base_kg[j] *= scale;
        for (int j = 0; j < bs; ++j) base_bg[j] *= scale;

        conv_layers[ci].apply_gradients(base_kg.data(),
                                        base_bg.empty() ? nullptr : base_bg.data(),
                                        learning_rate, momentum);
    }

    // Readout
    auto& base_rw = accum[0].readout_weight_grad;
    auto& base_rb = accum[0].readout_bias_grad;
    for (size_t t = 1; t < nt; ++t) {
        for (size_t j = 0; j < base_rw.size(); ++j) base_rw[j] += accum[t].readout_weight_grad[j];
        for (size_t j = 0; j < base_rb.size(); ++j) base_rb[j] += accum[t].readout_bias_grad[j];
    }
    for (auto& g : base_rw) g *= scale;
    for (auto& g : base_rb) g *= scale;
    readout.apply_gradients(base_rw.data(), base_rb.data(), learning_rate, momentum);
}
