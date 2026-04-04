#include "HCNNNetwork.h"
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // DIM=5 (N=32)
    HCNNNetwork net(5);

    // Conv (8 out) → Pool → Conv (16 out)
    net.add_conv(1, 8, true, true);
    net.add_pool(1, PoolType::MAX);
    net.add_conv(2, 16, true, true);

    net.randomize_all_weights(0.3f);

    // Dummy raw input — strictly in [-1.0, 1.0]
    std::vector<float> raw_input = {
        0.17f, 0.33f, 0.50f, 0.67f, 0.83f, 1.00f, 0.08f, 0.25f,
        0.42f, 0.58f, 0.75f, 0.92f, 0.13f, 0.30f, 0.47f, 0.63f
    };

    int N = net.get_start_N();
    std::vector<float> embedded(N, 0.0f);

    net.embed_input(raw_input.data(), static_cast<int>(raw_input.size()), embedded.data());

    float max_act = *std::max_element(embedded.begin(), embedded.end());
    std::cout << "Max activation after embedding: " << max_act << "\n";

    std::vector<float> logits(10, 0.0f);

    net.forward(embedded.data(), logits.data());

    std::cout << "HypercubeCNN test with pooling complete.\n";
    std::cout << "Input length: " << raw_input.size() << " (embedded into N=" << N << ")\n";
    std::cout << "Final logits: ";
    for (float v : logits) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}