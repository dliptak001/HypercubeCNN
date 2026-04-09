#include "HCNNPool.h"
#include <cassert>

namespace hcnn {

HCNNPool::HCNNPool(int input_dim, PoolType type)
    : input_dim(input_dim), output_dim(input_dim - 1),
      input_N(1 << input_dim), output_N(1 << (input_dim - 1)),
      type(type) {}

void HCNNPool::forward(const float* in, float* out, int num_channels,
                       std::vector<int>* max_indices) const {
    if (max_indices && type == PoolType::MAX) {
        max_indices->resize(num_channels * output_N);
    }

    uint32_t anti_mask = (1u << input_dim) - 1;

    for (int c = 0; c < num_channels; ++c) {
        const float* chan_in = in + c * input_N;
        float* chan_out = out + c * output_N;

        for (int v = 0; v < output_N; ++v) {
            int v_anti = v ^ anti_mask;
            if (type == PoolType::MAX) {
                if (chan_in[v] >= chan_in[v_anti]) {
                    chan_out[v] = chan_in[v];
                    if (max_indices) (*max_indices)[c * output_N + v] = v;
                } else {
                    chan_out[v] = chan_in[v_anti];
                    if (max_indices) (*max_indices)[c * output_N + v] = v_anti;
                }
            } else {
                chan_out[v] = (chan_in[v] + chan_in[v_anti]) * 0.5f;
            }
        }
    }
}

void HCNNPool::backward(const float* grad_out, float* grad_in, int num_channels,
                        const std::vector<int>* max_indices) const {
    for (int i = 0; i < num_channels * input_N; ++i) grad_in[i] = 0.0f;

    uint32_t anti_mask = (1u << input_dim) - 1;

    for (int c = 0; c < num_channels; ++c) {
        const float* g_out = grad_out + c * output_N;
        float* g_in = grad_in + c * input_N;

        if (type == PoolType::MAX) {
            assert(max_indices && "MAX pool backward requires max_indices from forward pass");
            for (int v = 0; v < output_N; ++v) {
                int src = (*max_indices)[c * output_N + v];
                g_in[src] = g_out[v];
            }
        } else {
            for (int v = 0; v < output_N; ++v) {
                g_in[v] += g_out[v] * 0.5f;
                g_in[v ^ anti_mask] += g_out[v] * 0.5f;
            }
        }
    }
}

} // namespace hcnn
