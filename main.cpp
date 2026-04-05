#include "HCNNNetwork.h"
#include "dataloader/HCNNMNISTDataset.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
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

static void evaluate(HCNNNetwork& net, const HCNNMNISTDataset& dataset,
                     const char* label) {
    int K = net.get_num_classes();
    int N = net.get_start_N();
    float total_loss = 0.0f;
    int correct = 0;
    int count = static_cast<int>(dataset.size());

    std::vector<float> embedded(N);
    std::vector<float> logits(K);

    for (int i = 0; i < count; ++i) {
        const auto& s = dataset.get(i);
        net.embed_input(s.input.data(), static_cast<int>(s.input.size()), embedded.data());
        net.forward(embedded.data(), logits.data());
        total_loss += cross_entropy_loss(logits.data(), K, s.target_class);
        if (argmax(logits.data(), K) == s.target_class) ++correct;
    }

    float avg_loss = total_loss / count;
    float accuracy = 100.0f * correct / count;
    std::cout << label << ": loss=" << avg_loss
              << " acc=" << correct << "/" << count
              << " (" << accuracy << "%)\n";
}

static void train_and_evaluate(const char* name, HCNNNetwork& net,
                               HCNNMNISTDataset& train_data,
                               const HCNNMNISTDataset& test_data,
                               float lr = 0.01f) {
    std::cout << "\n=== " << name << " (lr=" << lr << ") ===\n";
    evaluate(net, test_data, "Initial test");

    const int epochs = 15;
    const float momentum = 0.9f;
    float current_lr = lr;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto t0 = std::chrono::steady_clock::now();
        train_data.train_epoch(net, current_lr, momentum);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::string label = "Epoch " + std::to_string(epoch + 1);
        evaluate(net, test_data, label.c_str());
        std::cout << "  (lr=" << current_lr << ", " << secs << "s, "
                  << train_data.size() / secs << " samples/s)\n";

        // Halve LR every 5 epochs
        if ((epoch + 1) % 5 == 0) {
            current_lr *= 0.5f;
        }
    }
}

int main() {
    // Resolve data path relative to source file location
    auto src_dir = std::filesystem::path(__FILE__).parent_path();
    auto data_dir = src_dir / "data";

    std::cout << "Loading MNIST from " << data_dir << "...\n";
    auto train_data = load_mnist((data_dir / "train-images-idx3-ubyte").string(),
                                 (data_dir / "train-labels-idx1-ubyte").string(), 10000);
    auto test_data = load_mnist((data_dir / "t10k-images-idx3-ubyte").string(),
                                (data_dir / "t10k-labels-idx1-ubyte").string(), 1000);
    std::cout << "Train: " << train_data.size() << " samples, "
              << "Test: " << test_data.size() << " samples\n";

    HCNNNetwork net(10);
    net.add_conv(16, true, true);     // K=18 (DIM=10)
    net.add_pool(PoolType::MAX);      // DIM 10->9, N 1024->512
    net.add_pool(PoolType::MAX);      // DIM 9->8, N 512->256
    net.add_conv(32, true, true);     // K=14 (DIM=8)
    net.randomize_all_weights(0.1f);
    train_and_evaluate("HCNN", net, train_data, test_data, 0.005f);

    return 0;
}
