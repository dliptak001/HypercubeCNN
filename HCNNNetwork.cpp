#include "HCNNNetwork.h"
#include "ThreadPool.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>

HCNNNetwork::HCNNNetwork(int dim, int num_classes, int input_channels,
                         ReadoutType readout_type, size_t num_threads)
    : start_dim(dim), current_dim(dim), num_classes(num_classes),
      input_channels(input_channels),
      readout_type(readout_type),
      readout(num_classes, 1),
      thread_pool(std::make_unique<ThreadPool>(num_threads)) {
    if (dim < 3) {
        throw std::runtime_error("HCNNNetwork requires start_dim >= 3");
    }
    if (input_channels < 1) {
        throw std::runtime_error("HCNNNetwork requires input_channels >= 1");
    }
    channel_counts.push_back(input_channels);
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
    int final_channels = channel_counts.back();
    int final_N = 1 << current_dim;

    if (readout_type == ReadoutType::FLATTEN) {
        // Flatten: readout sees all channel*vertex activations as independent inputs.
        // Pass N=1 to readout so GAP is a no-op (average of 1 value = identity).
        readout = HCNNReadout(num_classes, final_channels * final_N);
        readout_N = 1;
    } else {
        // GAP: readout sees one averaged scalar per channel.
        readout = HCNNReadout(num_classes, final_channels);
        readout_N = final_N;
    }

    std::mt19937 rng(42);
    for (auto& layer : conv_layers) {
        layer.randomize_weights(scale, rng);
    }
    readout.randomize_weights(scale, rng);
}

void HCNNNetwork::embed_input(const float* raw_input, int input_length,
                              float* out) const {
    int N = 1 << start_dim;
    int total = input_channels * N;
    if (input_length > total) {
        throw std::runtime_error("Input length exceeds capacity ("
                                 + std::to_string(input_channels) + " channels × "
                                 + std::to_string(N) + " vertices = "
                                 + std::to_string(total) + ")");
    }
    for (int i = 0; i < input_length; ++i) {
        assert(raw_input[i] >= -1.0f && raw_input[i] <= 1.0f &&
               "Input value out of range [-1.0, 1.0]");
        out[i] = raw_input[i];
    }
    for (int i = input_length; i < total; ++i) {
        out[i] = 0.0f;
    }
}

void HCNNNetwork::forward(const float* first_layer_activations, float* logits) const {
    if (conv_layers.empty()) {
        throw std::runtime_error("HCNNNetwork::forward called with no conv layers");
    }

    // Compute max buffer size across all layers (including multi-channel input)
    int cur_N = 1 << start_dim;
    int max_size = input_channels * cur_N;
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
    int input_size = input_channels * cur_N;
    for (int i = 0; i < input_size; ++i) current[i] = first_layer_activations[i];

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

    readout.forward(current, logits, readout_N);
}

void HCNNNetwork::prepare_inference_buffers() {
    if (infer_bufs_ready) return;

    int N = 1 << start_dim;
    int total = input_channels * N;
    size_t nt = thread_pool ? thread_pool->NumThreads() : 1;

    int cur_N = N;
    int max_size = input_channels * cur_N;
    size_t ci0 = 0, pi0 = 0;
    for (size_t i = 0; i < is_conv_layer.size(); ++i) {
        if (is_conv_layer[i]) {
            max_size = std::max(max_size, conv_layers[ci0].get_c_out() * cur_N);
            ++ci0;
        } else {
            cur_N = pool_layers[pi0].get_output_N();
            ++pi0;
        }
    }
    infer_max_layer_size_ = max_size;

    ibufs_.resize(nt);
    for (size_t t = 0; t < nt; ++t) {
        ibufs_[t].buf1.resize(max_size);
        ibufs_[t].buf2.resize(max_size);
        ibufs_[t].embedded.resize(total);
    }
    infer_bufs_ready = true;
}

void HCNNNetwork::forward_batch(const float* const* raw_inputs,
                                const int* input_lengths,
                                int batch_size, float* logits_out) {
    if (batch_size <= 0) return;
    prepare_inference_buffers();

    int total = input_channels * (1 << start_dim);
    int K = num_classes;

    auto process_one = [&](size_t tid, int s) {
        auto& ib = ibufs_[tid];
        std::fill(ib.embedded.begin(), ib.embedded.end(), 0.0f);
        embed_input(raw_inputs[s], input_lengths[s], ib.embedded.data());

        float* current = ib.buf1.data();
        float* next_buf = ib.buf2.data();
        for (int i = 0; i < total; ++i) current[i] = ib.embedded[i];

        size_t ci = 0, pi = 0;
        for (size_t i = 0; i < is_conv_layer.size(); ++i) {
            if (is_conv_layer[i]) {
                conv_layers[ci].forward(current, next_buf);
                ++ci;
            } else {
                pool_layers[pi].forward(current, next_buf, channel_counts[i]);
                ++pi;
            }
            std::swap(current, next_buf);
        }
        readout.forward(current, logits_out + s * K, readout_N);
    };

    if (thread_pool && batch_size > 1) {
        LayerThreadGuard guard(conv_layers, thread_pool.get());

        thread_pool->ForEach(static_cast<size_t>(batch_size),
            [&](size_t tid, size_t begin, size_t end) {
                for (size_t s = begin; s < end; ++s)
                    process_one(tid, static_cast<int>(s));
            });
    } else {
        for (int s = 0; s < batch_size; ++s)
            process_one(0, s);
    }
}

void HCNNNetwork::train_step(const float* raw_input, int input_length,
                             int target_class, float learning_rate, float momentum,
                             float weight_decay, const float* class_weights) {
    int N = 1 << start_dim;
    int total = input_channels * N;
    std::vector<float> embedded(total, 0.0f);
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
    cache[0].channels = input_channels;
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
    readout.forward(final_c.activation.data(), logits.data(), readout_N);

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
    float cw = (class_weights != nullptr) ? class_weights[target_class] : 1.0f;
    for (int i = 0; i < num_classes; ++i) {
        grad_logits[i] = cw * (probs[i] - (i == target_class ? 1.0f : 0.0f));
    }

    // Backward through readout
    std::vector<float> grad_current(final_c.channels * final_c.N);
    readout.backward(grad_logits.data(), final_c.activation.data(),
                     readout_N, grad_current.data(), learning_rate, momentum,
                     weight_decay);

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
                                     learning_rate, momentum, weight_decay);
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
// ---------------------------------------------------------------------------
// Lazily allocate persistent per-thread buffers on first train_batch call.
// Subsequent calls reuse the same memory — only accumulators are zeroed.
// ---------------------------------------------------------------------------
void HCNNNetwork::prepare_batch_buffers() {
    if (batch_bufs_ready) return;

    int N = 1 << start_dim;
    int num_layers = static_cast<int>(is_conv_layer.size());
    size_t num_conv = conv_layers.size();
    size_t nt = thread_pool ? thread_pool->NumThreads() : 1;

    // Layer info (sizes are architecture-fixed)
    layer_info_.resize(num_layers + 1);
    layer_info_[0] = {N, input_channels};
    {
        int cur = N;
        size_t ci2 = 0, pi2 = 0;
        for (int i = 0; i < num_layers; ++i) {
            if (is_conv_layer[i]) {
                layer_info_[i + 1] = {cur, conv_layers[ci2].get_c_out()};
                ++ci2;
            } else {
                int out_N = pool_layers[pi2].get_output_N();
                layer_info_[i + 1] = {out_N, layer_info_[i].channels};
                cur = out_N;
                ++pi2;
            }
        }
    }

    // Gradient accumulators
    accum_.resize(nt);
    for (size_t t = 0; t < nt; ++t) {
        auto& a = accum_[t];
        a.conv_kernel_grad.resize(num_conv);
        a.conv_bias_grad.resize(num_conv);
        for (size_t ci = 0; ci < num_conv; ++ci) {
            a.conv_kernel_grad[ci].resize(conv_layers[ci].get_kernel_size());
            a.conv_bias_grad[ci].resize(conv_layers[ci].get_bias_size());
        }
        a.readout_weight_grad.resize(readout.get_weight_size());
        a.readout_bias_grad.resize(readout.get_bias_size());
    }

    // Work buffers
    tbufs_.resize(nt);
    for (size_t t = 0; t < nt; ++t) {
        auto& b = tbufs_[t];
        b.cache.resize(num_layers + 1);
        for (int i = 0; i <= num_layers; ++i) {
            auto& li = layer_info_[i];
            b.cache[i].activation.resize(li.channels * li.N);
            b.cache[i].pre_act.resize(li.channels * li.N);
            b.cache[i].max_indices.resize(li.channels * li.N);
        }
        b.logits.resize(num_classes);
        b.probs.resize(num_classes);
        b.grad_logits.resize(num_classes);
        int max_layer_size = 0;
        for (int i = 0; i <= num_layers; ++i)
            max_layer_size = std::max(max_layer_size,
                                      layer_info_[i].channels * layer_info_[i].N);
        b.grad_a.resize(max_layer_size);
        b.grad_b.resize(max_layer_size);
        b.rw_grad.resize(readout.get_weight_size());
        b.rb_grad.resize(readout.get_bias_size());
        b.kg.resize(num_conv);
        b.bg.resize(num_conv);
        for (size_t ci = 0; ci < num_conv; ++ci) {
            b.kg[ci].resize(conv_layers[ci].get_kernel_size());
            b.bg[ci].resize(conv_layers[ci].get_bias_size());
        }
        // Work buffers for compute_gradients (avoid per-call heap allocs)
        b.conv_work.resize(max_layer_size); // HCNN needs c_out*N, max_layer_size >= that
        int final_ch = layer_info_[num_layers].channels;
        b.readout_work.resize(final_ch);    // HCNNReadout needs input_channels floats
    }

    batch_bufs_ready = true;
}

void HCNNNetwork::zero_accumulators() {
    for (auto& a : accum_) {
        for (auto& v : a.conv_kernel_grad) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : a.conv_bias_grad)   std::fill(v.begin(), v.end(), 0.0f);
        std::fill(a.readout_weight_grad.begin(), a.readout_weight_grad.end(), 0.0f);
        std::fill(a.readout_bias_grad.begin(), a.readout_bias_grad.end(), 0.0f);
    }
}

void HCNNNetwork::train_batch(const float* const* inputs, const int* input_lengths,
                              const int* targets, int batch_size,
                              float learning_rate, float momentum,
                              float weight_decay, const float* class_weights) {
    if (batch_size <= 0) return;

    prepare_batch_buffers();
    zero_accumulators();

    int num_layers = static_cast<int>(is_conv_layer.size());
    size_t num_conv = conv_layers.size();
    size_t nt = accum_.size();

    // Process samples in parallel, each thread accumulates into its own accum
    auto process_sample = [&](size_t tid, int sample_idx) {
        auto& a = accum_[tid];
        auto& b = tbufs_[tid];

        // Embed directly into cache[0]
        auto& c0 = b.cache[0];
        std::fill(c0.activation.begin(), c0.activation.end(), 0.0f);
        embed_input(inputs[sample_idx], input_lengths[sample_idx], c0.activation.data());

        size_t ci = 0, pi = 0;

        // Forward pass
        for (int i = 0; i < num_layers; ++i) {
            auto& c = b.cache[i + 1];
            if (is_conv_layer[i]) {
                conv_layers[ci].forward(b.cache[i].activation.data(),
                                        c.activation.data(), c.pre_act.data());
                ++ci;
            } else {
                pool_layers[pi].forward(b.cache[i].activation.data(),
                                        c.activation.data(),
                                        layer_info_[i].channels,
                                        &c.max_indices);
                ++pi;
            }
        }

        // Readout forward
        auto& final_c = b.cache[num_layers];
        std::fill(b.logits.begin(), b.logits.end(), 0.0f);
        readout.forward(final_c.activation.data(), b.logits.data(), readout_N,
                        b.readout_work.data());

        // Softmax + cross-entropy gradient
        float max_logit = b.logits[0];
        for (int i = 1; i < num_classes; ++i)
            if (b.logits[i] > max_logit) max_logit = b.logits[i];

        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            b.probs[i] = std::exp(b.logits[i] - max_logit);
            sum_exp += b.probs[i];
        }
        for (int i = 0; i < num_classes; ++i) b.probs[i] /= sum_exp;

        float cw = (class_weights != nullptr) ? class_weights[targets[sample_idx]] : 1.0f;
        for (int i = 0; i < num_classes; ++i)
            b.grad_logits[i] = cw * (b.probs[i] - (i == targets[sample_idx] ? 1.0f : 0.0f));

        // Readout backward
        int final_size = layer_info_[num_layers].channels * layer_info_[num_layers].N;
        std::fill(b.grad_a.begin(), b.grad_a.begin() + final_size, 0.0f);
        readout.compute_gradients(b.grad_logits.data(), final_c.activation.data(),
                                  readout_N, b.grad_a.data(),
                                  b.rw_grad.data(), b.rb_grad.data(),
                                  b.readout_work.data());

        for (int i = 0; i < readout.get_weight_size(); ++i)
            a.readout_weight_grad[i] += b.rw_grad[i];
        for (int i = 0; i < readout.get_bias_size(); ++i)
            a.readout_bias_grad[i] += b.rb_grad[i];

        // Backward through layers — ping-pong between grad_a and grad_b
        ci = conv_layers.size();
        pi = pool_layers.size();

        for (int i = num_layers - 1; i >= 0; --i) {
            int prev_size = layer_info_[i].channels * layer_info_[i].N;
            std::fill(b.grad_b.begin(), b.grad_b.begin() + prev_size, 0.0f);

            if (is_conv_layer[i]) {
                --ci;
                conv_layers[ci].compute_gradients(
                    b.grad_a.data(),
                    b.cache[i].activation.data(),
                    b.cache[i + 1].pre_act.data(),
                    (i > 0) ? b.grad_b.data() : nullptr,
                    b.kg[ci].data(),
                    b.bg[ci].empty() ? nullptr : b.bg[ci].data(),
                    b.conv_work.data());

                for (int j = 0; j < conv_layers[ci].get_kernel_size(); ++j)
                    a.conv_kernel_grad[ci][j] += b.kg[ci][j];
                for (int j = 0; j < conv_layers[ci].get_bias_size(); ++j)
                    a.conv_bias_grad[ci][j] += b.bg[ci][j];
            } else {
                --pi;
                pool_layers[pi].backward(b.grad_a.data(), b.grad_b.data(),
                                         layer_info_[i].channels,
                                         &b.cache[i + 1].max_indices);
            }

            std::swap(b.grad_a, b.grad_b);
        }
    };

    if (thread_pool && batch_size > 1) {
        LayerThreadGuard guard(conv_layers, thread_pool.get());

        thread_pool->ForEach(static_cast<size_t>(batch_size),
            [&](size_t tid, size_t begin, size_t end) {
                for (size_t s = begin; s < end; ++s)
                    process_sample(tid, static_cast<int>(s));
            });
    } else {
        for (int s = 0; s < batch_size; ++s)
            process_sample(0, s);
    }

    // Reduce across threads and average
    float scale = 1.0f / static_cast<float>(batch_size);

    for (size_t ci = 0; ci < num_conv; ++ci) {
        int ks = conv_layers[ci].get_kernel_size();
        int bs = conv_layers[ci].get_bias_size();
        auto& base_kg = accum_[0].conv_kernel_grad[ci];
        auto& base_bg = accum_[0].conv_bias_grad[ci];

        for (size_t t = 1; t < nt; ++t) {
            for (int j = 0; j < ks; ++j) base_kg[j] += accum_[t].conv_kernel_grad[ci][j];
            for (int j = 0; j < bs; ++j) base_bg[j] += accum_[t].conv_bias_grad[ci][j];
        }

        for (int j = 0; j < ks; ++j) base_kg[j] *= scale;
        for (int j = 0; j < bs; ++j) base_bg[j] *= scale;

        conv_layers[ci].apply_gradients(base_kg.data(),
                                        base_bg.empty() ? nullptr : base_bg.data(),
                                        learning_rate, momentum, weight_decay);
    }

    auto& base_rw = accum_[0].readout_weight_grad;
    auto& base_rb = accum_[0].readout_bias_grad;
    for (size_t t = 1; t < nt; ++t) {
        for (size_t j = 0; j < base_rw.size(); ++j) base_rw[j] += accum_[t].readout_weight_grad[j];
        for (size_t j = 0; j < base_rb.size(); ++j) base_rb[j] += accum_[t].readout_bias_grad[j];
    }
    for (auto& g : base_rw) g *= scale;
    for (auto& g : base_rb) g *= scale;
    readout.apply_gradients(base_rw.data(), base_rb.data(), learning_rate, momentum,
                            weight_decay);
}
