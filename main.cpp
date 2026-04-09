#include "HCNN.h"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::cout << "HypercubeCNN v0.1.0 -- Quick Check\n\n";
    int failures = 0;

    // Build a small network: DIM=5, N=32, 4 classes
    hcnn::HCNN net(5, 4);
    net.AddConv(16);
    net.AddPool(hcnn::PoolType::MAX);
    net.AddConv(16);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumClasses();

    // Generate synthetic data
    const int num_samples = 16;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<float>> inputs(num_samples, std::vector<float>(N));
    std::vector<int> targets(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        for (auto& v : inputs[i]) v = dist(rng);
        targets[i] = i % K;
    }

    // Reusable scratch buffers (no per-call allocation in the loss loop).
    std::vector<float> embedded(N);
    std::vector<float> logits(K);

    auto compute_loss = [&]() {
        double total = 0.0;
        for (int i = 0; i < num_samples; ++i) {
            net.Embed(inputs[i].data(), N, embedded.data());
            net.Forward(embedded.data(), logits.data());
            double mx = logits[0];
            for (int j = 1; j < K; ++j) if (logits[j] > mx) mx = logits[j];
            double se = 0.0;
            for (int j = 0; j < K; ++j) se += std::exp(logits[j] - mx);
            total += -(logits[targets[i]] - mx) + std::log(se);
        }
        return total / num_samples;
    };

    // Check 1: loss decreases after training
    double loss_before = compute_loss();
    for (int step = 0; step < 50; ++step) {
        int idx = step % num_samples;
        net.TrainStep(inputs[idx].data(), N, targets[idx], 0.01f);
    }
    double loss_after = compute_loss();

    if (loss_after < loss_before) {
        std::cout << "PASS  Loss decreased: " << loss_before << " -> " << loss_after << "\n";
    } else {
        std::cout << "FAIL  Loss did not decrease: " << loss_before << " -> " << loss_after << "\n";
        ++failures;
    }

    // Check 2: batch inference produces finite logits
    std::vector<const float*> ptrs(num_samples);
    std::vector<int> lengths(num_samples, N);
    for (int i = 0; i < num_samples; ++i) ptrs[i] = inputs[i].data();

    std::vector<float> all_logits(num_samples * K);
    net.ForwardBatch(ptrs.data(), lengths.data(), num_samples, all_logits.data());

    bool all_ok = true;
    for (int i = 0; i < num_samples * K; ++i)
        if (!std::isfinite(all_logits[i])) { all_ok = false; break; }

    if (all_ok) {
        std::cout << "PASS  Batch inference: all logits finite\n";
    } else {
        std::cout << "FAIL  Batch inference: non-finite logits\n";
        ++failures;
    }

    std::cout << "\n" << (failures == 0 ? "All checks passed." : "Some checks FAILED.") << "\n";
    return failures > 0 ? 1 : 0;
}
