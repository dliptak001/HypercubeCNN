#pragma once

#include <vector>

class HCNNReadout {
public:
    HCNNReadout(int num_classes, int input_channels);

    void set_weights(const float* w, int size);
    void randomize_weights(float scale = 0.1f);

    void forward(const float* in, float* out, int N) const;

private:
    int num_classes;
    int input_channels;
    std::vector<float> weights;
};