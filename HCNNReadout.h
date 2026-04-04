#pragma once

#include <vector>
#include <cstdint>
#include <random>

class HCNNReadout {
public:
    HCNNReadout(int num_classes, int input_channels);

    void randomize_weights(float scale, std::mt19937& rng);

    void forward(const float* in, float* out, int N) const;

    // Backward: computes grad_in (if non-null) and updates weights via SGD.
    void backward(const float* grad_logits, const float* in, int N,
                  float* grad_in, float learning_rate);

    int get_num_classes() const { return num_classes; }
    int get_input_channels() const { return input_channels; }

private:
    int num_classes;
    int input_channels;
    std::vector<float> weights;
};
