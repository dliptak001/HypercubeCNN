#include "HCNNNetwork.h"
#include <iostream>
#include <vector>
#include <algorithm>   // for std::max_element

int main() {
    // DIM=5 (N=32) gives room for one pooling step
    HCNNNetwork net(5);

    // Conv → Pool → Conv
    net.add_conv(1, true, true);       // radius 1, ReLU + bias
    net.add_pool(1, PoolType::MAX);    // reduce DIM by 1 → now DIM=4
    net.add_conv(2, true, true);       // radius 2, ReLU + bias

    net.randomize_all_weights(1.0f);   // larger scale for visible logit variation

    // Dummy raw input (length <= N=32)
    std::vector<float> raw_input = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 0.5f, 1.5f,
        2.5f, 3.5f, 4.5f, 5.5f, 0.8f, 1.8f, 2.8f, 3.8f
    };

    int N = net.get_start_N();
    std::vector<float> embedded(N, 0.0f);

    net.embed_input(raw_input.data(), static_cast<int>(raw_input.size()), embedded.data());

    // Debug print: max activation after embedding (should be non-zero)
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