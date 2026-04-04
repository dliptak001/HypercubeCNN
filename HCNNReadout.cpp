#include "HCNNReadout.h"

HCNNReadout::HCNNReadout(int nc, int ic)
    : num_classes(nc), input_channels(ic), weights(nc * ic, 0.0f) {}

void HCNNReadout::randomize_weights(float scale, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& w : weights) {
        w = dist(rng);
    }
}

void HCNNReadout::forward(const float* in, float* out, int N) const {
    std::vector<float> channel_avg(input_channels, 0.0f);
    for (int c = 0; c < input_channels; ++c) {
        const float* chan = in + c * N;
        float sum = 0.0f;
        for (int v = 0; v < N; ++v) sum += chan[v];
        channel_avg[c] = sum / static_cast<float>(N);
    }

    for (int cls = 0; cls < num_classes; ++cls) {
        float sum = 0.0f;
        for (int c = 0; c < input_channels; ++c) {
            sum += weights[cls * input_channels + c] * channel_avg[c];
        }
        out[cls] = sum;
    }
}

void HCNNReadout::backward(const float* grad_logits, const float* in, int N,
                           float* grad_in, float learning_rate) {
    // Recompute channel averages
    std::vector<float> channel_avg(input_channels);
    for (int c = 0; c < input_channels; ++c) {
        float sum = 0.0f;
        for (int v = 0; v < N; ++v) sum += in[c * N + v];
        channel_avg[c] = sum / static_cast<float>(N);
    }

    // Input gradient BEFORE weight update (uses current weights)
    // logits[cls] = sum_c(w[cls,c] * avg_c), avg_c = (1/N)*sum_v(in[c*N+v])
    // d(logits[cls])/d(in[c*N+v]) = w[cls,c] / N
    if (grad_in) {
        for (int c = 0; c < input_channels; ++c) {
            float g = 0.0f;
            for (int cls = 0; cls < num_classes; ++cls) {
                g += grad_logits[cls] * weights[cls * input_channels + c];
            }
            g /= static_cast<float>(N);
            for (int v = 0; v < N; ++v) {
                grad_in[c * N + v] = g;
            }
        }
    }

    // Weight update
    for (int cls = 0; cls < num_classes; ++cls) {
        for (int c = 0; c < input_channels; ++c) {
            weights[cls * input_channels + c] -= learning_rate * grad_logits[cls] * channel_avg[c];
        }
    }
}
