#include "HCNNNetwork.h"
#include "dataloader/HCNNMNISTDataset.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

static float cross_entropy_loss(const float* logits, int K, int target) {
    double max_l = logits[0];
    for (int i = 1; i < K; ++i) if (logits[i] > max_l) max_l = logits[i];
    double sum_exp = 0.0;
    for (int i = 0; i < K; ++i) sum_exp += std::exp(logits[i] - max_l);
    return static_cast<float>(-(logits[target] - max_l) + std::log(sum_exp));
}

static int argmax(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
    return best;
}

// Test 1: Minimal network — single conv + readout, no pool.
static void test_single_conv() {
    std::cout << "\n=== Test 2: Single conv layer + readout (no pool) ===\n";

    HCNNNetwork net(4);  // DIM=4, N=16 — matches toy data exactly
    net.add_conv(2, 8, true, true);
    net.randomize_all_weights(0.3f);

    HCNNMNISTDataset dataset = create_toy_mnist_like_dataset();

    int K = net.get_num_classes();
    int N = net.get_start_N();

    auto evaluate = [&]() -> std::pair<float, float> {
        float total_loss = 0.0f;
        int correct = 0;
        for (size_t i = 0; i < dataset.size(); ++i) {
            const auto& s = dataset.get(i);
            std::vector<float> embedded(N), logits(K);
            net.embed_input(s.input.data(), static_cast<int>(s.input.size()), embedded.data());
            net.forward(embedded.data(), logits.data());
            total_loss += cross_entropy_loss(logits.data(), K, s.target_class);
            if (argmax(logits.data(), K) == s.target_class) ++correct;
        }
        return {total_loss / dataset.size(), static_cast<float>(correct) / dataset.size()};
    };

    auto [loss0, acc0] = evaluate();
    std::cout << "  Initial: loss=" << loss0 << " acc=" << acc0 << "\n";

    for (int epoch = 0; epoch < 1000; ++epoch) {
        dataset.train_epoch(net, 0.01f);
        if ((epoch + 1) % 200 == 0) {
            auto [loss, acc] = evaluate();
            std::cout << "  Epoch " << (epoch + 1) << ": loss=" << loss << " acc=" << acc << "\n";
        }
    }
}

// Test 2: Readout-only on a trivially separable problem.
// Create a net with random conv (frozen), train only readout.
static void test_readout_only_simple() {
    std::cout << "\n=== Test 2: Readout-only on trivial data ===\n";

    HCNNNetwork net(4);  // DIM=4, N=16
    net.add_conv(2, 32, true, true);
    net.randomize_all_weights(0.5f);

    HCNNMNISTDataset dataset = create_toy_mnist_like_dataset();

    int K = net.get_num_classes();
    int N = net.get_start_N();

    // Precompute features through frozen conv
    struct CachedSample {
        std::vector<float> features; // post-conv activations
        int target_class;
    };

    std::vector<CachedSample> cached(dataset.size());
    int out_channels = net.get_conv(0).get_c_out();
    int out_N = N; // no pool

    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& s = dataset.get(i);
        std::vector<float> embedded(N);
        net.embed_input(s.input.data(), static_cast<int>(s.input.size()), embedded.data());
        cached[i].features.resize(out_channels * out_N);
        net.get_conv(0).forward(embedded.data(), cached[i].features.data());
        cached[i].target_class = s.target_class;
    }

    // Print channel averages for first 3 samples to check separability
    std::cout << "  Channel averages (first 3 samples, first 8 channels):\n";
    for (int s = 0; s < std::min(3, (int)cached.size()); ++s) {
        std::cout << "    Sample " << s << " (class " << cached[s].target_class << "): ";
        for (int c = 0; c < std::min(8, out_channels); ++c) {
            float avg = 0.0f;
            for (int v = 0; v < out_N; ++v) avg += cached[s].features[c * out_N + v];
            avg /= out_N;
            std::cout << avg << " ";
        }
        std::cout << "\n";
    }

    // Train only readout
    auto& readout = net.get_readout();
    float lr = 0.5f;

    auto evaluate = [&]() -> std::pair<float, float> {
        float total_loss = 0.0f;
        int correct = 0;
        for (auto& cs : cached) {
            std::vector<float> logits(K);
            readout.forward(cs.features.data(), logits.data(), out_N);
            total_loss += cross_entropy_loss(logits.data(), K, cs.target_class);
            if (argmax(logits.data(), K) == cs.target_class) ++correct;
        }
        return {total_loss / cached.size(), static_cast<float>(correct) / cached.size()};
    };

    auto [loss0, acc0] = evaluate();
    std::cout << "  Initial: loss=" << loss0 << " acc=" << acc0 << "\n";

    std::mt19937 rng(77);
    for (int epoch = 0; epoch < 2000; ++epoch) {
        std::vector<size_t> order(cached.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        for (size_t idx : order) {
            auto& cs = cached[idx];
            std::vector<float> logits(K);
            readout.forward(cs.features.data(), logits.data(), out_N);

            // Softmax + CE gradient
            double max_l = logits[0];
            for (int i = 1; i < K; ++i) if (logits[i] > max_l) max_l = logits[i];
            std::vector<float> grad(K);
            float sum_exp = 0.0f;
            for (int i = 0; i < K; ++i) {
                grad[i] = std::exp(static_cast<float>(logits[i] - max_l));
                sum_exp += grad[i];
            }
            for (int i = 0; i < K; ++i) {
                grad[i] = grad[i] / sum_exp - (i == cs.target_class ? 1.0f : 0.0f);
            }

            readout.backward(grad.data(), cs.features.data(), out_N, nullptr, lr);
        }

        if ((epoch + 1) % 500 == 0) {
            auto [loss, acc] = evaluate();
            std::cout << "  Epoch " << (epoch + 1) << ": loss=" << loss << " acc=" << acc << "\n";
        }
    }
}

int main() {
    test_single_conv();
    test_readout_only_simple();
    return 0;
}
