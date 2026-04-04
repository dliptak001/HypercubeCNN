#pragma once

#include <vector>

class HCNNReadout {
public:
    HCNNReadout(int num_classes, int input_channels);

    void set_weights(const float* w, int size);
    void randomize_weights(float scale = 0.1f);

    void forward(const float* in, float* out, int N) const;

    // Minimal SGD update for training stub
    void apply_sgd_update(const std::vector<float>& grad_logits, float learning_rate);

private:
    int num_classes;
    int input_channels;
    std::vector<float> weights;
};