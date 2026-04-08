#include "HCNNNetwork.h"
#include "dataloader/HCNNDataset.h"
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

static void evaluate(HCNNNetwork& net, const HCNNDataset& dataset,
                     const char* label) {
    int K = net.get_num_classes();
    int count = static_cast<int>(dataset.size());

    // Batch-parallel inference
    std::vector<const float*> inputs(count);
    std::vector<int> lengths(count);
    std::vector<int> targets(count);
    for (int i = 0; i < count; ++i) {
        const auto& s = dataset.get(i);
        inputs[i] = s.input.data();
        lengths[i] = static_cast<int>(s.input.size());
        targets[i] = s.target_class;
    }

    std::vector<float> all_logits(count * K);
    net.forward_batch(inputs.data(), lengths.data(), count, all_logits.data());

    float total_loss = 0.0f;
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        const float* logits = all_logits.data() + i * K;
        total_loss += cross_entropy_loss(logits, K, targets[i]);
        if (argmax(logits, K) == targets[i]) ++correct;
    }

    float avg_loss = total_loss / count;
    float accuracy = 100.0f * correct / count;
    std::cout << label << ": loss=" << avg_loss
              << " acc=" << correct << "/" << count
              << " (" << accuracy << "%)\n";
}

static void train_and_evaluate(const char* name, HCNNNetwork& net,
                               HCNNDataset& train_data,
                               const HCNNDataset& test_data,
                               float lr = 0.01f, int batch_size = 32,
                               float weight_decay = 0.0f) {
    std::cout << "\n=== " << name << " (lr=" << lr
              << ", batch=" << batch_size
              << ", wd=" << weight_decay << ") ===\n";
    evaluate(net, test_data, "Initial test");

    const int epochs = 40;
    const float momentum = 0.9f;
    const float lr_min = 1e-5f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Cosine annealing: lr decays smoothly from lr to lr_min
        float progress = static_cast<float>(epoch) / static_cast<float>(epochs);
        float current_lr = lr_min + 0.5f * (lr - lr_min)
                           * (1.0f + std::cos(static_cast<float>(std::numbers::pi) * progress));

        auto t0 = std::chrono::steady_clock::now();
        train_data.train_epoch(net, current_lr, momentum, batch_size, weight_decay);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::string label = "Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs);
        evaluate(net, test_data, label.c_str());
        std::cout << "  (lr=" << current_lr << ", " << secs << "s, "
                  << train_data.size() / secs << " samples/s)\n";
    }
}

int main() {
    // Resolve data path relative to source file location
    auto src_dir = std::filesystem::path(__FILE__).parent_path().parent_path();
    auto data_dir = src_dir / "data";

    std::cout << "Loading MNIST from " << data_dir << "...\n";
    auto train_data = load_mnist((data_dir / "train-images-idx3-ubyte").string(),
                                 (data_dir / "train-labels-idx1-ubyte").string(), 60000);
    auto test_data = load_mnist((data_dir / "t10k-images-idx3-ubyte").string(),
                                (data_dir / "t10k-labels-idx1-ubyte").string(), 10000);
    std::cout << "Train: " << train_data.size() << " samples, "
              << "Test: " << test_data.size() << " samples\n";

    HCNNNetwork net(10);               // auto-detect thread count
    net.add_conv(32, true, true);         // 1->32 ch,   K=10 (DIM=10)
    net.add_pool(PoolType::MAX);          // DIM 10->9,  N 1024->512
    net.add_conv(64, true, true);         // 32->64 ch,  K=9  (DIM=9)
    net.add_pool(PoolType::MAX);          // DIM 9->8,   N 512->256
    net.add_conv(128, true, true);        // 64->128 ch, K=8  (DIM=8)
    net.add_pool(PoolType::MAX);          // DIM 8->7,   N 256->128
    net.add_conv(128, true, true);        // 128->128 ch, K=7  (DIM=7)
    net.add_pool(PoolType::MAX);          // DIM 7->6,   N 128->64
    net.randomize_all_weights();          // Xavier/Glorot init
    std::cout << "Threads: " << std::thread::hardware_concurrency() << "\n";
    train_and_evaluate("HCNN", net, train_data, test_data, 0.06f, 256, 1e-4f);

    return 0;
}
