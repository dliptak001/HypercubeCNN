#include "HCNN.h"
#include <algorithm>
#include <stdexcept>

HCNN::HCNN(int dim, int c_in, int c_out, bool use_relu, bool use_bias)
    : DIM(dim), N(1 << dim), c_in(c_in), c_out(c_out),
      K(2 * dim - 2),
      use_relu(use_relu), use_bias(use_bias),
      kernel(c_out * c_in * K, 0.0f),
      bias(use_bias ? c_out : 0, 0.0f),
      kernel_vel(c_out * c_in * K, 0.0f),
      bias_vel(use_bias ? c_out : 0, 0.0f) {
    if (DIM < 3) {
        throw std::runtime_error("HCNN requires DIM >= 3 (K = 2*DIM-2 must be >= 4)");
    }

    // Build mask table: shell masks first, then nearest-neighbor masks
    masks.resize(K);

    // Shell masks: (1 << (i+1)) - 1 for i in [0, DIM-3]
    // Produces cumulative-bit patterns 3, 7, 15, 31, ...
    for (int i = 0; i < DIM - 2; ++i) {
        masks[i] = (1u << (i + 1)) - 1;
    }

    // Nearest-neighbor masks: 1 << i for i in [0, DIM-1]
    // Produces single-bit flips 1, 2, 4, 8, ...
    for (int i = 0; i < DIM; ++i) {
        masks[DIM - 2 + i] = 1u << i;
    }
}

void HCNN::randomize_weights(float scale, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& w : kernel) {
        w = dist(rng);
    }
    if (use_bias) {
        for (auto& b : bias) b = 0.0f;
    }
    std::fill(kernel_vel.begin(), kernel_vel.end(), 0.0f);
    std::fill(bias_vel.begin(), bias_vel.end(), 0.0f);
}

void HCNN::forward(const float* in, float* out, float* pre_act) const {
    for (int co = 0; co < c_out; ++co) {
        float* out_co = out + co * N;

        // Initialize with bias
        float b = use_bias ? bias[co] : 0.0f;
        for (int v = 0; v < N; ++v) out_co[v] = b;

        // Accumulate: w * in[v ^ mask] for each input channel and mask
        for (int ci = 0; ci < c_in; ++ci) {
            const float* in_ci = in + ci * N;
            for (int k = 0; k < K; ++k) {
                float w = kernel[kernel_idx(co, ci, k)];
                uint32_t m = masks[k];
                for (int v = 0; v < N; ++v) {
                    out_co[v] += w * in_ci[v ^ m];
                }
            }
        }

        // Store pre-activation and apply activation
        if (pre_act) {
            float* pa = pre_act + co * N;
            for (int v = 0; v < N; ++v) {
                pa[v] = out_co[v];
                out_co[v] = activate(out_co[v]);
            }
        } else {
            for (int v = 0; v < N; ++v) {
                out_co[v] = activate(out_co[v]);
            }
        }
    }
}

void HCNN::backward(const float* grad_out, const float* in, const float* pre_act,
                    float* grad_in, float learning_rate, float momentum) {
    // Pre-activation gradients (through activation function)
    std::vector<float> grad_pre(c_out * N);
    for (int i = 0; i < c_out * N; ++i) {
        grad_pre[i] = grad_out[i] * activate_derivative(pre_act[i]);
    }

    // Input gradient: gi[v] += w * gp[v ^ m]  (XOR is self-inverse)
    if (grad_in) {
        for (int ci = 0; ci < c_in; ++ci) {
            float* gi = grad_in + ci * N;
            for (int v = 0; v < N; ++v) gi[v] = 0.0f;
            for (int co = 0; co < c_out; ++co) {
                const float* gp = grad_pre.data() + co * N;
                for (int k = 0; k < K; ++k) {
                    float w = kernel[kernel_idx(co, ci, k)];
                    uint32_t m = masks[k];
                    for (int v = 0; v < N; ++v) {
                        gi[v] += w * gp[v ^ m];
                    }
                }
            }
        }
    }

    // Kernel update with momentum: vel = mu*vel + grad; w -= lr*vel
    for (int co = 0; co < c_out; ++co) {
        const float* gp = grad_pre.data() + co * N;
        for (int ci = 0; ci < c_in; ++ci) {
            const float* in_ci = in + ci * N;
            for (int k = 0; k < K; ++k) {
                uint32_t m = masks[k];
                float grad_k = 0.0f;
                for (int v = 0; v < N; ++v) {
                    grad_k += gp[v] * in_ci[v ^ m];
                }
                int ki = kernel_idx(co, ci, k);
                kernel_vel[ki] = momentum * kernel_vel[ki] + grad_k;
                kernel[ki] -= learning_rate * kernel_vel[ki];
            }
        }
    }

    // Bias update with momentum
    if (use_bias) {
        for (int co = 0; co < c_out; ++co) {
            float grad_b = 0.0f;
            const float* gp = grad_pre.data() + co * N;
            for (int v = 0; v < N; ++v) {
                grad_b += gp[v];
            }
            bias_vel[co] = momentum * bias_vel[co] + grad_b;
            bias[co] -= learning_rate * bias_vel[co];
        }
    }
}

void HCNN::compute_gradients(const float* grad_out, const float* in, const float* pre_act,
                             float* grad_in, float* kernel_grad, float* bias_grad) const {
    std::vector<float> grad_pre(c_out * N);
    for (int i = 0; i < c_out * N; ++i) {
        grad_pre[i] = grad_out[i] * activate_derivative(pre_act[i]);
    }

    // Input gradient
    if (grad_in) {
        for (int ci = 0; ci < c_in; ++ci) {
            float* gi = grad_in + ci * N;
            for (int v = 0; v < N; ++v) gi[v] = 0.0f;
            for (int co = 0; co < c_out; ++co) {
                const float* gp = grad_pre.data() + co * N;
                for (int k = 0; k < K; ++k) {
                    float w = kernel[kernel_idx(co, ci, k)];
                    uint32_t m = masks[k];
                    for (int v = 0; v < N; ++v) {
                        gi[v] += w * gp[v ^ m];
                    }
                }
            }
        }
    }

    // Kernel gradient
    for (int co = 0; co < c_out; ++co) {
        const float* gp = grad_pre.data() + co * N;
        for (int ci = 0; ci < c_in; ++ci) {
            const float* in_ci = in + ci * N;
            for (int k = 0; k < K; ++k) {
                uint32_t m = masks[k];
                float grad_k = 0.0f;
                for (int v = 0; v < N; ++v) {
                    grad_k += gp[v] * in_ci[v ^ m];
                }
                kernel_grad[kernel_idx(co, ci, k)] = grad_k;
            }
        }
    }

    // Bias gradient
    if (bias_grad && use_bias) {
        for (int co = 0; co < c_out; ++co) {
            float grad_b = 0.0f;
            const float* gp = grad_pre.data() + co * N;
            for (int v = 0; v < N; ++v) {
                grad_b += gp[v];
            }
            bias_grad[co] = grad_b;
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
