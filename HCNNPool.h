#pragma once

#include <vector>
#include <cstdint>

namespace hcnn {

enum class PoolType { MAX, AVG };

/// Antipodal pooling: pairs each vertex v with its bitwise complement v' (maximally
/// distant vertex), reduces DIM by 1. The lower-half vertex survives.
class HCNNPool {
public:
    HCNNPool(int input_dim, PoolType type = PoolType::MAX);

    void forward(const float* in, float* out, int num_channels,
                 std::vector<int>* max_indices = nullptr) const;

    void backward(const float* grad_out, float* grad_in, int num_channels,
                  const std::vector<int>* max_indices) const;

    int get_input_dim() const { return input_dim; }
    int get_output_dim() const { return output_dim; }
    int get_input_N() const { return input_N; }
    int get_output_N() const { return output_N; }

private:
    int input_dim, output_dim, input_N, output_N;
    PoolType type;
};

} // namespace hcnn
