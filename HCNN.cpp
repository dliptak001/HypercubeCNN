#include "HCNN.h"

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
      bias(use_bias ? c_out : 0, 0.0f) {
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
}

void HCNN::forward(const float* in, float* out, float* pre_act) const {
    for (int co = 0; co < c_out; ++co) {
        for (int v = 0; v < N; ++v) {
            float sum = 0.0f;
            for (int ci = 0; ci < c_in; ++ci) {
                const float* chan = in + ci * N;
                for (int d = 0; d <= radius; ++d) {
                    sum += kernel[kernel_idx(co, ci, d)] * shell_mean(v, d, chan);
                }
            }
            if (use_bias) sum += bias[co];
            if (pre_act) pre_act[co * N + v] = sum;
            out[co * N + v] = activate(sum);
        }
    }
}

void HCNN::backward(const float* grad_out, const float* in, const float* pre_act,
                    float* grad_in, float learning_rate) {
    // Pre-activation gradients (through activation function)
    std::vector<float> grad_pre(c_out * N);
    for (int i = 0; i < c_out * N; ++i) {
        grad_pre[i] = grad_out[i] * activate_derivative(pre_act[i]);
    }

    // Input gradient BEFORE weight update (uses current weights)
    if (grad_in) {
        for (int i = 0; i < c_in * N; ++i) grad_in[i] = 0.0f;

        for (int co = 0; co < c_out; ++co) {
            for (int v = 0; v < N; ++v) {
                float gp = grad_pre[co * N + v];
                if (gp == 0.0f) continue;
                for (int ci = 0; ci < c_in; ++ci) {
                    for (int d = 0; d <= radius; ++d) {
                        float w = kernel[kernel_idx(co, ci, d)]
                                  / static_cast<float>(shell_count[d]);
                        for (int m : shell_masks[d]) {
                            grad_in[ci * N + (v ^ m)] += gp * w;
                        }
                    }
                }
            }
        }
    }

    // Kernel weight update
    for (int co = 0; co < c_out; ++co) {
        for (int ci = 0; ci < c_in; ++ci) {
            const float* chan = in + ci * N;
            for (int d = 0; d <= radius; ++d) {
                float grad_k = 0.0f;
                for (int v = 0; v < N; ++v) {
                    grad_k += grad_pre[co * N + v] * shell_mean(v, d, chan);
                }
                kernel[kernel_idx(co, ci, d)] -= learning_rate * grad_k;
            }
        }
    }

    // Bias update
    if (use_bias) {
        for (int co = 0; co < c_out; ++co) {
            float grad_b = 0.0f;
            for (int v = 0; v < N; ++v) {
                grad_b += grad_pre[co * N + v];
            }
            bias[co] -= learning_rate * grad_b;
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
