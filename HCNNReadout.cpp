#include "HCNNReadout.h"
#include <algorithm>
#include <cmath>

HCNNReadout::HCNNReadout(int nc, int ic)
    : num_classes(nc), input_channels(ic),
      weights(nc * ic, 0.0f), bias(nc, 0.0f),
      weight_vel(nc * ic, 0.0f), bias_vel(nc, 0.0f) {}

void HCNNReadout::randomize_weights(float scale, std::mt19937& rng) {
    // Xavier/Glorot uniform for the linear layer.
    if (scale <= 0.0f) {
        scale = std::sqrt(6.0f / static_cast<float>(input_channels + num_classes));
    }
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& w : weights) {
        w = dist(rng);
    }
    for (auto& b : bias) b = 0.0f;
    std::fill(weight_vel.begin(), weight_vel.end(), 0.0f);
    std::fill(bias_vel.begin(), bias_vel.end(), 0.0f);
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
        float sum = bias[cls];
        for (int c = 0; c < input_channels; ++c) {
            sum += weights[cls * input_channels + c] * channel_avg[c];
        }
        out[cls] = sum;
    }
}

void HCNNReadout::backward(const float* grad_logits, const float* in, int N,
                           float* grad_in, float learning_rate, float momentum) {
    std::vector<float> channel_avg(input_channels);
    for (int c = 0; c < input_channels; ++c) {
        float sum = 0.0f;
        for (int v = 0; v < N; ++v) sum += in[c * N + v];
        channel_avg[c] = sum / static_cast<float>(N);
    }

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

    // Weight update with momentum: v = mu*v + grad; w -= lr*v
    for (int cls = 0; cls < num_classes; ++cls) {
        for (int c = 0; c < input_channels; ++c) {
            int wi = cls * input_channels + c;
            float g = grad_logits[cls] * channel_avg[c];
            weight_vel[wi] = momentum * weight_vel[wi] + g;
            weights[wi] -= learning_rate * weight_vel[wi];
        }
        bias_vel[cls] = momentum * bias_vel[cls] + grad_logits[cls];
        bias[cls] -= learning_rate * bias_vel[cls];
    }
}

void HCNNReadout::compute_gradients(const float* grad_logits, const float* in, int N,
                                    float* grad_in, float* weight_grad, float* bias_grad) const {
    std::vector<float> channel_avg(input_channels);
    for (int c = 0; c < input_channels; ++c) {
        float sum = 0.0f;
        for (int v = 0; v < N; ++v) sum += in[c * N + v];
        channel_avg[c] = sum / static_cast<float>(N);
    }

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

    for (int cls = 0; cls < num_classes; ++cls) {
        for (int c = 0; c < input_channels; ++c) {
            weight_grad[cls * input_channels + c] = grad_logits[cls] * channel_avg[c];
        }
        if (bias_grad) bias_grad[cls] = grad_logits[cls];
    }
}

void HCNNReadout::apply_gradients(const float* weight_grad, const float* bias_grad,
                                  float learning_rate, float momentum) {
    int total_w = num_classes * input_channels;
    for (int i = 0; i < total_w; ++i) {
        weight_vel[i] = momentum * weight_vel[i] + weight_grad[i];
        weights[i] -= learning_rate * weight_vel[i];
    }
    if (bias_grad) {
        for (int cls = 0; cls < num_classes; ++cls) {
            bias_vel[cls] = momentum * bias_vel[cls] + bias_grad[cls];
            bias[cls] -= learning_rate * bias_vel[cls];
        }
    }
}
