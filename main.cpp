#include "HCNNNetwork.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>   // for std::accumulate

int main() {
    HCNNNetwork net(5);

    net.add_conv(1, 8, true, true);
    net.add_pool(1, PoolType::MAX);
    net.add_conv(2, 16, true, true);

    net.randomize_all_weights(0.3f);

    // Dummy input in [-1.0, 1.0]
    std::vector<float> raw_input = {
        0.17f, 0.33f, 0.50f, 0.67f, 0.83f, 1.00f, 0.08f, 0.25f,
        0.42f, 0.58f, 0.75f, 0.92f, 0.13f, 0.30f, 0.47f, 0.63f
    };

    // Dummy one-hot target for class 0
    std::vector<float> target(10, 0.0f);
    target[0] = 1.0f;

    int N = net.get_start_N();
    std::vector<float> embedded(N, 0.0f);
    std::vector<float> logits(10, 0.0f);

    // Print initial loss
    net.embed_input(raw_input.data(), static_cast<int>(raw_input.size()), embedded.data());
    net.forward(embedded.data(), logits.data());
    float loss = 0.0f;
    for (int i = 0; i < 10; ++i) {
        float e = logits[i] - target[i];
        loss += e * e;
    }
    loss /= 10.0f;
    std::cout << "Initial MSE loss: " << loss << "\n";

    // Simple training loop
    const int epochs = 100;
    const float lr = 0.05f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        net.train_step(raw_input.data(), static_cast<int>(raw_input.size()), target.data(), lr);
    }

    // Print final loss
    net.forward(embedded.data(), logits.data());
    loss = 0.0f;
    for (int i = 0; i < 10; ++i) {
        float e = logits[i] - target[i];
        loss += e * e;
    }
    loss /= 10.0f;
    std::cout << "Final MSE loss after " << epochs << " steps: " << loss << "\n";
    std::cout << "Final logits: ";
    for (float v : logits) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}