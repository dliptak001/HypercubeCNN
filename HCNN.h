#pragma once

#include <vector>
#include <cstdint>

class HCNN {
public:
    HCNN(int dim, int radius, bool use_relu = true, bool use_bias = true);

    void set_kernel(const float* weights, int size);   // size == radius + 1
    void set_bias(const float* biases, int size);      // size == c_out
    void randomize_weights(float scale = 0.1f);

    void forward(const float* in, float* out, int c_in, int c_out) const;

    int get_dim() const { return DIM; }
    int get_N() const { return N; }
    int get_radius() const { return radius; }

private:
    int DIM, N, radius;
    bool use_relu, use_bias;
    std::vector<float> kernel;
    std::vector<float> bias;

    float sum_shell(int v, int d, const float* data) const;   // stride removed
    float activate(float x) const;
};