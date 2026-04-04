#include "HCNNNetwork.h"
#include "dataloader/HCNNMNISTDataset.h"
#include <cmath>
#include <iostream>
#include <vector>

static float cross_entropy_loss(const float* logits, int num_classes, int target_class) {
    float max_logit = logits[0];
    for (int i = 1; i < num_classes; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        sum_exp += std::exp(logits[i] - max_logit);
    }
    float log_prob = (logits[target_class] - max_logit) - std::log(sum_exp);
    return -log_prob;
}

int main() {
    HCNNNetwork net(5);

    net.add_conv(1, 8, true, true);
    net.add_pool(1, PoolType::MAX);
    net.add_conv(2, 16, true, true);

    net.randomize_all_weights(0.2f);

    HCNNMNISTDataset dataset = create_toy_mnist_like_dataset();

    const auto& first = dataset.get(0);
    const int num_classes = 10;
    std::vector<float> logits(num_classes, 0.0f);
    int N = net.get_start_N();
    std::vector<float> embedded(N, 0.0f);

    // Initial loss
    net.embed_input(first.input.data(), static_cast<int>(first.input.size()), embedded.data());
    net.forward(embedded.data(), logits.data());
    float loss = cross_entropy_loss(logits.data(), num_classes, first.target_class);
    std::cout << "Initial cross-entropy loss: " << loss << "\n";

    // Training with progress
    const int epochs = 500;
    const float lr = 0.01f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        dataset.train_epoch(net, lr);

        if ((epoch + 1) % 100 == 0) {
            net.embed_input(first.input.data(), static_cast<int>(first.input.size()), embedded.data());
            net.forward(embedded.data(), logits.data());
            loss = cross_entropy_loss(logits.data(), num_classes, first.target_class);
            std::cout << "Epoch " << (epoch + 1) << " cross-entropy loss: " << loss << "\n";
        }
    }

    // Final
    net.embed_input(first.input.data(), static_cast<int>(first.input.size()), embedded.data());
    net.forward(embedded.data(), logits.data());
    std::cout << "Final logits for sample 0: ";
    for (float v : logits) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
