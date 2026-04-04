#include "HCNNMNISTDataset.h"

#include <algorithm>
#include <random>

HCNNMNISTDataset create_toy_mnist_like_dataset() {
    HCNNMNISTDataset ds;
    ds.samples.resize(10);

    for (int cls = 0; cls < 10; ++cls) {
        auto& s = ds.samples[cls];
        s.input.resize(16, -1.0f);
        s.target.resize(10, 0.0f);
        s.target[cls] = 1.0f;

        // Simple distinct pattern: class k lights up one unique position
        int hot = cls % 16;
        s.input[hot] = 1.0f;
    }
    return ds;
}

void HCNNMNISTDataset::train_epoch(HCNNNetwork& net, float learning_rate) {
    static std::mt19937 rng(42);
    std::vector<size_t> order(samples.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    for (size_t i : order) {
        const auto& s = samples[i];
        net.train_step(s.input.data(), static_cast<int>(s.input.size()),
                       s.target.data(), learning_rate);
    }
}