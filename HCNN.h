#pragma once

#include <vector>
#include <cstdint>
#include <random>

class HCNN {
public:
    HCNN(int dim, int c_in, int c_out, int radius, bool use_relu = true, bool use_bias = true);

    void randomize_weights(float scale, std::mt19937& rng);

    // Forward pass. If pre_act is non-null, stores pre-activation values for backprop.
    void forward(const float* in, float* out, float* pre_act = nullptr) const;

    // Backward pass: computes grad_in and updates weights via SGD.
    // grad_in may be null if input gradients are not needed (first layer).
    void backward(const float* grad_out, const float* in, const float* pre_act,
                  float* grad_in, float learning_rate);

    int get_dim() const { return DIM; }
    int get_N() const { return N; }
    int get_c_in() const { return c_in; }
    int get_c_out() const { return c_out; }
    int get_radius() const { return radius; }

private:
    int DIM, N, c_in, c_out, radius;
    bool use_relu, use_bias;
    std::vector<float> kernel;      // [c_out * c_in * (radius+1)]
    std::vector<float> bias;        // [c_out]
    std::vector<int> shell_count;   // [DIM+1] precomputed C(DIM, d)

    // Precomputed shell masks: shell_masks[d] contains all C(DIM,d) bitmasks
    // with exactly d bits set. XOR any vertex v with a mask to get a neighbor
    // at Hamming distance d.
    std::vector<std::vector<int>> shell_masks;

    int kernel_idx(int co, int ci, int d) const {
        return (co * c_in + ci) * (radius + 1) + d;
    }
    float shell_mean(int v, int d, const float* data) const;
    float activate(float x) const;
    float activate_derivative(float x) const;
};
