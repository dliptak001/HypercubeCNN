#include "HCNNPool.h"

HCNNPool::HCNNPool(int input_dim, int reduce_by, PoolType type)
    : input_dim(input_dim), output_dim(input_dim - reduce_by), reduce_by(reduce_by),
      input_N(1 << input_dim), output_N(1 << (input_dim - reduce_by)), type(type) {}

void HCNNPool::forward(const float* in, float* out, int num_channels) const {
    const int group_size = 1 << reduce_by;
    for (int c = 0; c < num_channels; ++c) {
        const float* chan_in = in + c * input_N;
        float* chan_out = out + c * output_N;

        if (type == PoolType::MAX) {
            for (int i = 0; i < output_N; ++i) chan_out[i] = -std::numeric_limits<float>::infinity();
        } else {
            for (int i = 0; i < output_N; ++i) chan_out[i] = 0.0f;
        }

        for (int u = 0; u < input_N; ++u) {
            int new_v = u >> reduce_by;
            float val = chan_in[u];
            if (type == PoolType::MAX) {
                if (val > chan_out[new_v]) chan_out[new_v] = val;
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