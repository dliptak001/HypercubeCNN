#include "HCNN.h"
#include <algorithm>

// Generate all bitmasks with exactly 'bits' bits set out of 'dim' total bits.
static void generate_masks(int dim, int bits, int current_mask, int start_bit,
                           int bits_set, std::vector<int>& out) {
    if (bits_set == bits) {
        out.push_back(current_mask);
        return;
    }
    int remaining = bits - bits_set;
    for (int b = start_bit; b <= dim - remaining; ++b) {
        generate_masks(dim, bits, current_mask | (1 << b), b + 1, bits_set + 1, out);
    }
}

HCNN::HCNN(int dim, int c_in, int c_out, int radius, bool use_relu, bool use_bias)
    : DIM(dim), N(1 << dim), c_in(c_in), c_out(c_out), radius(radius),
      use_relu(use_relu), use_bias(use_bias),
      kernel(c_out * c_in * (radius + 1), 0.0f),
      bias(use_bias ? c_out : 0, 0.0f),
      kernel_vel(c_out * c_in * (radius + 1), 0.0f),
      bias_vel(use_bias ? c_out : 0, 0.0f) {
    // Precompute shell counts: C(DIM, d)
    shell_count.resize(DIM + 1);
    shell_count[0] = 1;
    for (int d = 1; d <= DIM; ++d) {
        shell_count[d] = shell_count[d - 1] * (DIM - d + 1) / d;
    }

    // Precompute shell masks for each distance 0..radius
    shell_masks.resize(radius + 1);
    for (int d = 0; d <= radius; ++d) {
        shell_masks[d].reserve(shell_count[d]);
        generate_masks(DIM, d, 0, 0, 0, shell_masks[d]);
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
    // Precompute shell means for all input channels in bulk
    std::vector<float> sm_buf(c_in * (radius + 1) * N);
    precompute_shell_means(in, c_in, sm_buf.data());

    for (int co = 0; co < c_out; ++co) {
        float* out_co = out + co * N;

        // Initialize with bias
        float b = use_bias ? bias[co] : 0.0f;
        for (int v = 0; v < N; ++v) out_co[v] = b;

        // Accumulate kernel * shell_mean — inner loop is vectorizable
        for (int ci = 0; ci < c_in; ++ci) {
            for (int d = 0; d <= radius; ++d) {
                float w = kernel[kernel_idx(co, ci, d)];
                const float* sm = sm_buf.data() + (ci * (radius + 1) + d) * N;
                for (int v = 0; v < N; ++v) {
                    out_co[v] += w * sm[v];
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

    // Precompute shell means of input (for kernel gradient)
    std::vector<float> in_sm(c_in * (radius + 1) * N);
    precompute_shell_means(in, c_in, in_sm.data());

    // Input gradient: grad_in[ci*N+u] = sum_co sum_d kernel[co,ci,d] * shell_mean(u, d, grad_pre+co*N)
    if (grad_in) {
        std::vector<float> gp_sm(c_out * (radius + 1) * N);
        precompute_shell_means(grad_pre.data(), c_out, gp_sm.data());

        for (int ci = 0; ci < c_in; ++ci) {
            float* gi = grad_in + ci * N;
            for (int v = 0; v < N; ++v) gi[v] = 0.0f;
            for (int co = 0; co < c_out; ++co) {
                for (int d = 0; d <= radius; ++d) {
                    float w = kernel[kernel_idx(co, ci, d)];
                    const float* gsm = gp_sm.data() + (co * (radius + 1) + d) * N;
                    for (int v = 0; v < N; ++v) {
                        gi[v] += w * gsm[v];
                    }
                }
            }
        }
    }

    // Kernel update with momentum: v = mu*v + grad; w -= lr*v
    for (int co = 0; co < c_out; ++co) {
        const float* gp = grad_pre.data() + co * N;
        for (int ci = 0; ci < c_in; ++ci) {
            for (int d = 0; d <= radius; ++d) {
                const float* sm = in_sm.data() + (ci * (radius + 1) + d) * N;
                float grad_k = 0.0f;
                for (int v = 0; v < N; ++v) {
                    grad_k += gp[v] * sm[v];
                }
                int ki = kernel_idx(co, ci, d);
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

    std::vector<float> in_sm(c_in * (radius + 1) * N);
    precompute_shell_means(in, c_in, in_sm.data());

    if (grad_in) {
        std::vector<float> gp_sm(c_out * (radius + 1) * N);
        precompute_shell_means(grad_pre.data(), c_out, gp_sm.data());

        for (int ci = 0; ci < c_in; ++ci) {
            float* gi = grad_in + ci * N;
            for (int v = 0; v < N; ++v) gi[v] = 0.0f;
            for (int co = 0; co < c_out; ++co) {
                for (int d = 0; d <= radius; ++d) {
                    float w = kernel[kernel_idx(co, ci, d)];
                    const float* gsm = gp_sm.data() + (co * (radius + 1) + d) * N;
                    for (int v = 0; v < N; ++v) {
                        gi[v] += w * gsm[v];
                    }
                }
            }
        }
    }

    for (int co = 0; co < c_out; ++co) {
        const float* gp = grad_pre.data() + co * N;
        for (int ci = 0; ci < c_in; ++ci) {
            for (int d = 0; d <= radius; ++d) {
                const float* sm = in_sm.data() + (ci * (radius + 1) + d) * N;
                float grad_k = 0.0f;
                for (int v = 0; v < N; ++v) {
                    grad_k += gp[v] * sm[v];
                }
                kernel_grad[kernel_idx(co, ci, d)] = grad_k;
            }
        }
    }

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

void HCNN::precompute_shell_means(const float* data, int channels, float* buf) const {
    for (int c = 0; c < channels; ++c) {
        const float* chan = data + c * N;
        for (int d = 0; d <= radius; ++d) {
            float* dst = buf + (c * (radius + 1) + d) * N;
            if (d == 0) {
                for (int v = 0; v < N; ++v) dst[v] = chan[v];
            } else {
                for (int v = 0; v < N; ++v) dst[v] = 0.0f;
                for (int m : shell_masks[d]) {
                    for (int v = 0; v < N; ++v) {
                        dst[v] += chan[v ^ m];
                    }
                }
                float inv = 1.0f / static_cast<float>(shell_count[d]);
                for (int v = 0; v < N; ++v) dst[v] *= inv;
            }
        }
    }
}

float HCNN::shell_mean(int v, int d, const float* data) const {
    if (d == 0) return data[v];
    float s = 0.0f;
    for (int m : shell_masks[d]) {
        s += data[v ^ m];
    }
    return s / static_cast<float>(shell_count[d]);
}

float HCNN::activate(float x) const {
    if (!use_relu) return x;
    return (x > 0.0f) ? x : 0.0f;
}

float HCNN::activate_derivative(float x) const {
    if (!use_relu) return 1.0f;
    return (x > 0.0f) ? 1.0f : 0.0f;
}
