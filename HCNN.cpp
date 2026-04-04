#include "HCNN.h"

HCNN::HCNN(int dim, int radius, bool use_relu, bool use_bias)
    : DIM(dim), N(1 << dim), radius(radius), use_relu(use_relu), use_bias(use_bias),
      kernel(radius + 1, 0.0f) {}

void HCNN::set_kernel(const float* weights, int size) {
    if (size != radius + 1) return;
    for (int i = 0; i < size; ++i) kernel[i] = weights[i];
}

void HCNN::set_bias(const float* b, int size) {
    if (!use_bias) return;
    bias.resize(size);
    for (int i = 0; i < size; ++i) bias[i] = b[i];
}

void HCNN::randomize_weights(float scale) {
    uint64_t seed = 12345;
    for (auto& w : kernel) {
        seed = seed * 6364136223846793005ULL + 1;
        w = scale * (static_cast<float>(seed & 0xFFFF) / 32768.0f - 1.0f);
    }
    if (use_bias) bias.assign(16, 0.0f);
}

void HCNN::forward(const float* in, float* out, int c_in, int c_out) const {
    const int stride = N;
    if (use_bias && bias.empty()) const_cast<std::vector<float>&>(bias).assign(c_out, 0.0f);

    for (int c_out_idx = 0; c_out_idx < c_out; ++c_out_idx) {
        for (int v = 0; v < N; ++v) {
            float sum = 0.0f;
            for (int c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
                const float* chan = in + c_in_idx * stride;
                for (int d = 0; d <= radius; ++d) {
                    float shell = sum_shell(v, d, chan);   // stride removed
                    sum += kernel[d] * shell;
                }
            }
            if (use_bias) sum += bias[c_out_idx];
            out[c_out_idx * stride + v] = activate(sum);
        }
    }
}

float HCNN::sum_shell(int v, int d, const float* data) const {   // stride removed
    if (d == 0) return data[v];
    float s = 0.0f;
    for (int u = 0; u < N; ++u) {
        int dist = __builtin_popcountll(static_cast<uint64_t>(v) ^ static_cast<uint64_t>(u));
        if (dist == d) s += data[u];
    }
    return s;
}

float HCNN::activate(float x) const {
    if (!use_relu) return x;
    return (x > 0.0f) ? x : 0.0f;
}