#include "HCNN.h"
#include "ThreadPool.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Minimum DIM at which per-layer threading kicks in.
// Below this, fork-join overhead exceeds the per-vertex work.
static constexpr int THREAD_DIM_THRESHOLD = 12;

HCNN::HCNN(int dim, int c_in, int c_out, bool use_relu, bool use_bias)
    : DIM(dim), N(1 << dim), c_in(c_in), c_out(c_out),
      K(dim),
      use_relu(use_relu), use_bias(use_bias),
      kernel(c_out * c_in * K, 0.0f),
      bias(use_bias ? c_out : 0, 0.0f),
      kernel_vel(c_out * c_in * K, 0.0f),
      bias_vel(use_bias ? c_out : 0, 0.0f) {
    if (DIM < 3) {
        throw std::runtime_error("HCNN requires DIM >= 3");
    }
}

void HCNN::randomize_weights(float scale, std::mt19937& rng) {
    // Xavier/Glorot uniform: scale = sqrt(6 / (fan_in + fan_out)).
    // fan_in = c_in * K, fan_out = c_out * K.
    if (scale <= 0.0f) {
        float fan_in  = static_cast<float>(c_in * K);
        float fan_out = static_cast<float>(c_out * K);
        scale = std::sqrt(6.0f / (fan_in + fan_out));
    }
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& w : kernel) w = dist(rng);
    if (use_bias) {
        for (auto& b : bias) b = 0.0f;
    }
    std::fill(kernel_vel.begin(), kernel_vel.end(), 0.0f);
    std::fill(bias_vel.begin(), bias_vel.end(), 0.0f);
}

// ---------------------------------------------------------------------------
// Forward: vertex-level threading within each output channel.
// Each thread handles a contiguous vertex range — no write conflicts.
// ---------------------------------------------------------------------------
void HCNN::forward(const float* in, float* out, float* pre_act) const {
    const bool use_threads = thread_pool && DIM >= THREAD_DIM_THRESHOLD;

    for (int co = 0; co < c_out; ++co) {
        float* out_co = out + co * N;
        float b = use_bias ? bias[co] : 0.0f;

        auto do_vertices = [&](size_t v_begin, size_t v_end) {
            for (size_t v = v_begin; v < v_end; ++v)
                out_co[v] = b;
            for (int ci = 0; ci < c_in; ++ci) {
                const float* in_ci = in + ci * N;
                for (int k = 0; k < K; ++k) {
                    float w = kernel[kernel_idx(co, ci, k)];
                    uint32_t m = 1u << k;
                    for (size_t v = v_begin; v < v_end; ++v)
                        out_co[v] += w * in_ci[v ^ m];
                }
            }
            if (pre_act) {
                float* pa = pre_act + co * N;
                for (size_t v = v_begin; v < v_end; ++v) {
                    pa[v] = out_co[v];
                    out_co[v] = activate(out_co[v]);
                }
            } else {
                for (size_t v = v_begin; v < v_end; ++v)
                    out_co[v] = activate(out_co[v]);
            }
        };

        if (use_threads) {
            thread_pool->ForEach(static_cast<size_t>(N),
                [&](size_t, size_t begin, size_t end) { do_vertices(begin, end); });
        } else {
            do_vertices(0, static_cast<size_t>(N));
        }
    }
}

// ---------------------------------------------------------------------------
// Backward: vertex-level threading for input gradients (same structure as
// forward — each thread writes its own vertex range, no conflicts).
// Channel-level threading for weight gradients (each c_out owns its own
// kernel/bias slice, and the v reduction is kept sequential per thread).
// ---------------------------------------------------------------------------
void HCNN::backward(const float* grad_out, const float* in, const float* pre_act,
                    float* grad_in, float learning_rate, float momentum) {
    const bool use_threads = thread_pool && DIM >= THREAD_DIM_THRESHOLD;

    // Pre-activation gradients
    std::vector<float> grad_pre(c_out * N);
    for (int i = 0; i < c_out * N; ++i)
        grad_pre[i] = grad_out[i] * activate_derivative(pre_act[i]);

    // Input gradient: vertex-level parallelism
    if (grad_in) {
        for (int ci = 0; ci < c_in; ++ci) {
            float* gi = grad_in + ci * N;

            auto do_vertices = [&](size_t v_begin, size_t v_end) {
                for (size_t v = v_begin; v < v_end; ++v) gi[v] = 0.0f;
                for (int co = 0; co < c_out; ++co) {
                    const float* gp = grad_pre.data() + co * N;
                    for (int k = 0; k < K; ++k) {
                        float w = kernel[kernel_idx(co, ci, k)];
                        uint32_t m = 1u << k;
                        for (size_t v = v_begin; v < v_end; ++v)
                            gi[v] += w * gp[v ^ m];
                    }
                }
            };

            if (use_threads) {
                thread_pool->ForEach(static_cast<size_t>(N),
                    [&](size_t, size_t b, size_t e) { do_vertices(b, e); });
            } else {
                do_vertices(0, static_cast<size_t>(N));
            }
        }
    }

    // Weight update: channel-level parallelism (each c_out is independent)
    auto do_weight_update = [&](int co) {
        const float* gp = grad_pre.data() + co * N;
        for (int ci = 0; ci < c_in; ++ci) {
            const float* in_ci = in + ci * N;
            for (int k = 0; k < K; ++k) {
                uint32_t m = 1u << k;
                float grad_k = 0.0f;
                for (int v = 0; v < N; ++v)
                    grad_k += gp[v] * in_ci[v ^ m];
                int ki = kernel_idx(co, ci, k);
                kernel_vel[ki] = momentum * kernel_vel[ki] + grad_k;
                kernel[ki] -= learning_rate * kernel_vel[ki];
            }
        }
        if (use_bias) {
            float grad_b = 0.0f;
            for (int v = 0; v < N; ++v) grad_b += gp[v];
            bias_vel[co] = momentum * bias_vel[co] + grad_b;
            bias[co] -= learning_rate * bias_vel[co];
        }
    };

    if (use_threads) {
        thread_pool->ForEach(static_cast<size_t>(c_out),
            [&](size_t, size_t b, size_t e) {
                for (size_t co = b; co < e; ++co) do_weight_update(static_cast<int>(co));
            });
    } else {
        for (int co = 0; co < c_out; ++co) do_weight_update(co);
    }
}

// ---------------------------------------------------------------------------
// compute_gradients: same threading strategy as backward, but writes raw
// gradients to caller buffers instead of updating weights.
// ---------------------------------------------------------------------------
void HCNN::compute_gradients(const float* grad_out, const float* in, const float* pre_act,
                             float* grad_in, float* kernel_grad, float* bias_grad) const {
    const bool use_threads = thread_pool && DIM >= THREAD_DIM_THRESHOLD;

    std::vector<float> grad_pre(c_out * N);
    for (int i = 0; i < c_out * N; ++i)
        grad_pre[i] = grad_out[i] * activate_derivative(pre_act[i]);

    // Input gradient: vertex-level
    if (grad_in) {
        for (int ci = 0; ci < c_in; ++ci) {
            float* gi = grad_in + ci * N;

            auto do_vertices = [&](size_t v_begin, size_t v_end) {
                for (size_t v = v_begin; v < v_end; ++v) gi[v] = 0.0f;
                for (int co = 0; co < c_out; ++co) {
                    const float* gp = grad_pre.data() + co * N;
                    for (int k = 0; k < K; ++k) {
                        float w = kernel[kernel_idx(co, ci, k)];
                        uint32_t m = 1u << k;
                        for (size_t v = v_begin; v < v_end; ++v)
                            gi[v] += w * gp[v ^ m];
                    }
                }
            };

            if (use_threads) {
                thread_pool->ForEach(static_cast<size_t>(N),
                    [&](size_t, size_t b, size_t e) { do_vertices(b, e); });
            } else {
                do_vertices(0, static_cast<size_t>(N));
            }
        }
    }

    // Kernel + bias gradient: channel-level
    auto do_kernel_grad = [&](int co) {
        const float* gp = grad_pre.data() + co * N;
        for (int ci = 0; ci < c_in; ++ci) {
            const float* in_ci = in + ci * N;
            for (int k = 0; k < K; ++k) {
                uint32_t m = 1u << k;
                float grad_k = 0.0f;
                for (int v = 0; v < N; ++v)
                    grad_k += gp[v] * in_ci[v ^ m];
                kernel_grad[kernel_idx(co, ci, k)] = grad_k;
            }
        }
        if (bias_grad && use_bias) {
            float grad_b = 0.0f;
            for (int v = 0; v < N; ++v) grad_b += gp[v];
            bias_grad[co] = grad_b;
        }
    };

    if (use_threads) {
        thread_pool->ForEach(static_cast<size_t>(c_out),
            [&](size_t, size_t b, size_t e) {
                for (size_t co = b; co < e; ++co) do_kernel_grad(static_cast<int>(co));
            });
    } else {
        for (int co = 0; co < c_out; ++co) do_kernel_grad(co);
    }
}

// ---------------------------------------------------------------------------
// apply_gradients: apply pre-computed (averaged) gradients with momentum SGD.
// ---------------------------------------------------------------------------
void HCNN::apply_gradients(const float* kernel_grad, const float* bias_grad,
                           float learning_rate, float momentum) {
    int total_k = c_out * c_in * K;
    for (int i = 0; i < total_k; ++i) {
        kernel_vel[i] = momentum * kernel_vel[i] + kernel_grad[i];
        kernel[i] -= learning_rate * kernel_vel[i];
    }
    if (use_bias && bias_grad) {
        for (int co = 0; co < c_out; ++co) {
            bias_vel[co] = momentum * bias_vel[co] + bias_grad[co];
            bias[co] -= learning_rate * bias_vel[co];
        }
    }
}

float HCNN::activate(float x) const {
    if (!use_relu) return x;
    return (x > 0.0f) ? x : 0.0f;
}

float HCNN::activate_derivative(float x) const {
    if (!use_relu) return 1.0f;
    return (x > 0.0f) ? 1.0f : 0.0f;
}
