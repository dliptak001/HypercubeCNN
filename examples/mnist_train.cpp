// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNN.h"
#include "HCNNDataset.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <thread>
#include <vector>

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

// Contiguous flat-buffer view of a dataset, suitable for HCNN's flat
// training and inference APIs.  Built once per dataset and reused across
// epochs.  Flattens the per-sample vectors in HCNNDataset into a single
// contiguous buffer.
struct FlatDataset {
    std::vector<float> inputs;   // count * input_length contiguous floats
    std::vector<int>   targets;  // count class indices
    int count = 0;
    int input_length = 0;

    explicit FlatDataset(const HCNNDataset& ds) {
        count = static_cast<int>(ds.size());
        if (count == 0) return;
        input_length = static_cast<int>(ds.get(0).input.size());
        inputs.resize(static_cast<size_t>(count) * input_length);
        targets.resize(count);
        for (int i = 0; i < count; ++i) {
            const auto& s = ds.get(i);
            std::copy(s.input.begin(), s.input.end(),
                      inputs.begin() + i * input_length);
            targets[i] = s.target_class;
        }
    }
};

static void evaluate(hcnn::HCNN& net, const FlatDataset& ds, const char* label) {
    int K = net.GetNumOutputs();
    int count = ds.count;

    std::vector<float> all_logits(static_cast<size_t>(count) * K);
    net.ForwardBatch(ds.inputs.data(), ds.input_length, count, all_logits.data());

    float total_loss = 0.0f;
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        const float* logits = all_logits.data() + i * K;
        total_loss += cross_entropy_loss(logits, K, ds.targets[i]);
        if (argmax(logits, K) == ds.targets[i]) ++correct;
    }

    float avg_loss = total_loss / count;
    float accuracy = 100.0f * correct / count;
    std::cout << label << ": loss=" << avg_loss
              << " acc=" << correct << "/" << count
              << " (" << accuracy << "%)\n";
}

static void train_and_evaluate(const char* name, hcnn::HCNN& net,
                               const FlatDataset& train_ds,
                               const FlatDataset& test_ds,
                               float lr = 0.01f, int batch_size = 32,
                               float weight_decay = 0.0f) {
    std::cout << "\n=== " << name << " (lr=" << lr
              << ", batch=" << batch_size
              << ", wd=" << weight_decay << ") ===\n";
    evaluate(net, test_ds, "Initial test");

    const int epochs = 40;
    const float momentum = 0.9f;
    const float lr_min = 1e-3f;  // 10% of lr_max
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Cosine annealing: lr decays smoothly from lr to lr_min
        float progress = static_cast<float>(epoch) / static_cast<float>(epochs);
        float current_lr = lr_min + 0.5f * (lr - lr_min)
                           * (1.0f + std::cos(static_cast<float>(std::numbers::pi) * progress));

        auto t0 = std::chrono::steady_clock::now();
        net.TrainEpoch(train_ds.inputs.data(), train_ds.input_length,
                       train_ds.targets.data(), train_ds.count, batch_size,
                       current_lr, momentum, weight_decay,
                       /*class_weights=*/nullptr,
                       /*shuffle_seed=*/static_cast<unsigned>(epoch + 1));
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::string label = "Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs);
        evaluate(net, test_ds, label.c_str());
        std::cout << "  (lr=" << current_lr << ", " << secs << "s, "
                  << train_ds.count / secs << " samples/s)\n";
    }
}

int main() {
    // Resolve data path relative to source file location
    auto src_dir = std::filesystem::path(__FILE__).parent_path().parent_path();
    auto data_dir = src_dir / "data";

    std::cout << "Loading MNIST from " << data_dir << "...\n";
    auto train_data = load_mnist((data_dir / "train-images-idx3-ubyte").string(),
                                 (data_dir / "train-labels-idx1-ubyte").string(), 20000);
    auto test_data  = load_mnist((data_dir / "t10k-images-idx3-ubyte").string(),
                                 (data_dir / "t10k-labels-idx1-ubyte").string(), 2000);
    std::cout << "Train: " << train_data.size() << " samples, "
              << "Test: " << test_data.size() << " samples\n";
    std::cout << "Threads: " << std::thread::hardware_concurrency() << "\n";

    FlatDataset train_flat(train_data);
    FlatDataset test_flat(test_data);

    hcnn::HCNN net(10);
    net.AddConv(32);                          // 1->32 ch,    K=10 (DIM=10)
    net.AddPool(hcnn::PoolType::MAX);         // DIM 10->9,   N 1024->512
    net.AddConv(64);                          // 32->64 ch,   K=9  (DIM=9)
    net.AddPool(hcnn::PoolType::MAX);         // DIM 9->8,    N 512->256
    net.AddConv(128);                         // 64->128 ch,  K=8  (DIM=8)
    net.AddPool(hcnn::PoolType::MAX);         // DIM 8->7,    N 256->128
    net.AddConv(128);                         // 128->128 ch, K=7  (DIM=7)
    net.AddPool(hcnn::PoolType::MAX);         // DIM 7->6,    N 128->64
    net.RandomizeWeights();

    train_and_evaluate("HCNN", net, train_flat, test_flat, 0.06f, 256, 1e-4f);

    return 0;
}
