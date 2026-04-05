#include "HCNNPool.h"
#include <stdexcept>

HCNNPool::HCNNPool(int input_dim, int reduce_by, PoolType type, PoolGrouping grouping)
    : input_dim(input_dim), output_dim(input_dim - reduce_by), reduce_by(reduce_by),
      input_N(1 << input_dim), output_N(1 << (input_dim - reduce_by)),
      type(type), grouping(grouping) {
    if (grouping == PoolGrouping::ANTIPODAL && reduce_by != 1) {
        throw std::runtime_error("Antipodal pooling requires reduce_by == 1");
    }
}

void HCNNPool::forward(const float* in, float* out, int num_channels,
                       std::vector<int>* max_indices) const {
    if (max_indices && type == PoolType::MAX) {
        max_indices->resize(num_channels * output_N);
    }

    for (int c = 0; c < num_channels; ++c) {
        const float* chan_in = in + c * input_N;
        float* chan_out = out + c * output_N;

        if (grouping == PoolGrouping::ANTIPODAL) {
            // Pair each vertex v (lower half) with its antipodal v' (upper half)
            uint32_t anti_mask = (1u << input_dim) - 1;
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
        } else {
            // Subcube pooling: collapse local subcubes via bit-shift grouping
            const int group_size = 1 << reduce_by;

            if (type == PoolType::MAX) {
                for (int i = 0; i < output_N; ++i) {
                    chan_out[i] = -std::numeric_limits<float>::infinity();
                    if (max_indices) (*max_indices)[c * output_N + i] = -1;
                }
            } else {
                for (int i = 0; i < output_N; ++i) chan_out[i] = 0.0f;
            }

            for (int u = 0; u < input_N; ++u) {
                int new_v = u >> reduce_by;
                float val = chan_in[u];
                if (type == PoolType::MAX) {
                    if (val > chan_out[new_v]) {
                        chan_out[new_v] = val;
                        if (max_indices) (*max_indices)[c * output_N + new_v] = u;
                    }
                } else {
                    chan_out[new_v] += val;
                }
            }

            if (type == PoolType::AVG) {
                float scale = 1.0f / static_cast<float>(group_size);
                for (int i = 0; i < output_N; ++i) chan_out[i] *= scale;
            }
        }
    }
}

void HCNNPool::backward(const float* grad_out, float* grad_in, int num_channels,
                        const std::vector<int>* max_indices) const {
    for (int i = 0; i < num_channels * input_N; ++i) grad_in[i] = 0.0f;

    for (int c = 0; c < num_channels; ++c) {
        const float* g_out = grad_out + c * output_N;
        float* g_in = grad_in + c * input_N;

        if (grouping == PoolGrouping::ANTIPODAL) {
            uint32_t anti_mask = (1u << input_dim) - 1;
            if (type == PoolType::MAX) {
                for (int v = 0; v < output_N; ++v) {
                    int src = (*max_indices)[c * output_N + v];
                    g_in[src] = g_out[v];
                }
            } else {
                for (int v = 0; v < output_N; ++v) {
                    g_in[v] = g_out[v] * 0.5f;
                    g_in[v ^ anti_mask] = g_out[v] * 0.5f;
                }
            }
        } else {
            // Subcube pooling backward
            const int group_size = 1 << reduce_by;
            if (type == PoolType::MAX) {
                for (int i = 0; i < output_N; ++i) {
                    int src = (*max_indices)[c * output_N + i];
                    if (src >= 0) g_in[src] = g_out[i];
                }
            } else {
                float scale = 1.0f / static_cast<float>(group_size);
                for (int i = 0; i < output_N; ++i) {
                    int base = i << reduce_by;
                    for (int j = 0; j < group_size; ++j) {
                        g_in[base + j] = g_out[i] * scale;
                    }
                }
            }
        }
    }
}
