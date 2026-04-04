#pragma once

#include <vector>
#include <cstdint>
#include <limits>

enum class PoolType { MAX, AVG };

class HCNNPool {
public:
    HCNNPool(int input_dim, int reduce_by, PoolType type = PoolType::MAX);

    void forward(const float* in, float* out, int num_channels) const;

    int get_input_dim() const { return input_dim; }
    int get_output_dim() const { return output_dim; }
    int get_input_N() const { return input_N; }
    int get_output_N() const { return output_N; }

private:
    int input_dim, output_dim, reduce_by, input_N, output_N;
    PoolType type;
};