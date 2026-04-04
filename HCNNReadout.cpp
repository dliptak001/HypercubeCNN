#include "HCNNReadout.h"

#include <cstdint>

HCNNReadout::HCNNReadout(int nc, int ic)
    : num_classes(nc), input_channels(ic), weights(nc * ic, 0.0f) {}

void HCNNReadout::set_weights(const float* w, int size) {
    if (size != num_classes * input_channels) return;
    for (int i = 0; i < size; ++i) weights[i] = w[i];
}

void HCNNReadout::randomize_weights(float scale) {
    uint64_t seed = 54321;
    for (auto& w : weights) {
        seed = seed * 6364136223846793005ULL + 1;
        w = scale * (static_cast<float>(seed & 0xFFFF) / 32768.0f - 1.0f);
    }
}

void HCNNReadout::forward(const float* in, float* out, int N) const {
    // First compute global average per final channel
    std::vector<float> channel_avg(input_channels, 0.0f);
    for (int c = 0; c < input_channels; ++c) {
        const float* chan = in + c * N;
        float sum = 0.0f;
        for (int v = 0; v < N; ++v) sum += chan[v];
        channel_avg[c] = sum / static_cast<float>(N);
    }

    // Now apply linear weights for each class
    for (int cls = 0; cls < num_classes; ++cls) {
        float sum = 0.0f;
        for (int c = 0; c < input_channels; ++c) {
            sum += weights[cls * input_channels + c] * channel_avg[c];
        }
        out[cls] = sum;
    }
}