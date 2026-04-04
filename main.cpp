#include "HCNNNetwork.h"
#include "dataloader/HCNNMNISTDataset.h"   // <-- updated path for new folder
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    HCNNNetwork net(5);

    net.add_conv(1, 8, true, true);
    net.add_pool(1, PoolType::MAX);
    net.add_conv(2, 16, true, true);

    net.randomize_all_weights(0.3f);

    HCNNMNISTDataset dataset = create_toy_mnist_like_dataset();

    // Initial loss on first sample
    const auto& first = dataset.get(0);
    std::vector<float> logits(10, 0.0f);
    int N = net.get_start_N();
    std::vector<float> embedded(N, 0.0f);
    net.embed_input(first.input.data(), static_cast<int>(first.input.size()), embedded.data());
    net.forward(embedded.data(), logits.data());

    float loss = 0.0f;
    for (int i = 0; i < 10; ++i) {
        float e = logits[i] - first.target[i];
        loss += e * e;
    }
    loss /= 10.0f;
    std::cout << "Initial MSE loss: " << loss << "\n";

    // Training loop
    const int epochs = 50;
    const float lr = 0.05f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        dataset.train_epoch(net, lr);
    }

    // Final loss on first sample
    net.embed_input(first.input.data(), static_cast<int>(first.input.size()), embedded.data());
    net.forward(embedded.data(), logits.data());
    loss = 0.0f;
    for (int i = 0; i < 10; ++i) {
        float e = logits[i] - first.target[i];
        loss += e * e;
    }
    loss /= 10.0f;
    std::cout << "Final MSE loss after " << epochs << " epochs: " << loss << "\n";
    std::cout << "Final logits for sample 0: ";
    for (float v : logits) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}