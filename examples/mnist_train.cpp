#include "HCNN.h"
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

// Flat-array view of a dataset, suitable for HCNN's raw-pointer training
// and inference APIs.  Built once per dataset and reused across epochs.
struct DatasetView {
    std::vector<const float*> inputs;
    std::vector<int>          lengths;
    std::vector<int>          targets;

    explicit DatasetView(const HCNNDataset& ds) {
        const int n = static_cast<int>(ds.size());
        inputs.resize(n);
        lengths.resize(n);
        targets.resize(n);
        for (int i = 0; i < n; ++i) {
            const auto& s = ds.get(i);
            inputs[i]  = s.input.data();
            lengths[i] = static_cast<int>(s.input.size());
            targets[i] = s.target_class;
        }
    }

    int size() const { return static_cast<int>(inputs.size()); }
};

static void evaluate(HCNN& net, const DatasetView& view, const char* label) {
    int K = net.GetNumClasses();
    int count = view.size();

    std::vector<float> all_logits(static_cast<size_t>(count) * K);
    net.ForwardBatch(view.inputs.data(), view.lengths.data(), count, all_logits.data());

    float total_loss = 0.0f;
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        const float* logits = all_logits.data() + i * K;
        total_loss += cross_entropy_loss(logits, K, view.targets[i]);
        if (argmax(logits, K) == view.targets[i]) ++correct;
    }

    float avg_loss = total_loss / count;
    float accuracy = 100.0f * correct / count;
    std::cout << label << ": loss=" << avg_loss
              << " acc=" << correct << "/" << count
              << " (" << accuracy << "%)\n";
}

static void train_and_evaluate(const char* name, HCNN& net,
                               const DatasetView& train_view,
                               const DatasetView& test_view,
                               float lr = 0.01f, int batch_size = 32,
                               float weight_decay = 0.0f) {
    std::cout << "\n=== " << name << " (lr=" << lr
              << ", batch=" << batch_size
              << ", wd=" << weight_decay << ") ===\n";
    evaluate(net, test_view, "Initial test");

    const int epochs = 40;
    const float momentum = 0.9f;
    const float lr_min = 1e-5f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Cosine annealing: lr decays smoothly from lr to lr_min
        float progress = static_cast<float>(epoch) / static_cast<float>(epochs);
        float current_lr = lr_min + 0.5f * (lr - lr_min)
                           * (1.0f + std::cos(static_cast<float>(std::numbers::pi) * progress));

        auto t0 = std::chrono::steady_clock::now();
        // Pass `epoch + 1` as the shuffle seed -- nonzero, distinct per epoch,
        // and reproducible across runs.
        net.TrainEpoch(train_view.inputs.data(), train_view.lengths.data(),
                       train_view.targets.data(), train_view.size(), batch_size,
                       current_lr, momentum, weight_decay,
                       /*class_weights=*/nullptr,
                       /*shuffle_seed=*/static_cast<unsigned>(epoch + 1));
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::string label = "Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs);
        evaluate(net, test_view, label.c_str());
        std::cout << "  (lr=" << current_lr << ", " << secs << "s, "
                  << train_view.size() / secs << " samples/s)\n";
    }
}

int main() {
    // Resolve data path relative to source file location
    auto src_dir = std::filesystem::path(__FILE__).parent_path().parent_path();
    auto data_dir = src_dir / "data";

    std::cout << "Loading MNIST from " << data_dir << "...\n";
    auto train_data = load_mnist((data_dir / "train-images-idx3-ubyte").string(),
                                 (data_dir / "train-labels-idx1-ubyte").string(), 60000);
    auto test_data  = load_mnist((data_dir / "t10k-images-idx3-ubyte").string(),
                                 (data_dir / "t10k-labels-idx1-ubyte").string(), 10000);
    std::cout << "Train: " << train_data.size() << " samples, "
              << "Test: " << test_data.size() << " samples\n";
    std::cout << "Threads: " << std::thread::hardware_concurrency() << "\n";

    DatasetView train_view(train_data);
    DatasetView test_view(test_data);

    HCNN net(10);
    net.AddConv(32);                  // 1->32 ch,    K=10 (DIM=10)
    net.AddPool(PoolType::MAX);       // DIM 10->9,   N 1024->512
    net.AddConv(64);                  // 32->64 ch,   K=9  (DIM=9)
    net.AddPool(PoolType::MAX);       // DIM 9->8,    N 512->256
    net.AddConv(128);                 // 64->128 ch,  K=8  (DIM=8)
    net.AddPool(PoolType::MAX);       // DIM 8->7,    N 256->128
    net.AddConv(128);                 // 128->128 ch, K=7  (DIM=7)
    net.AddPool(PoolType::MAX);       // DIM 7->6,    N 128->64
    net.RandomizeWeights();

    train_and_evaluate("HCNN", net, train_view, test_view, 0.06f, 256, 1e-4f);

    return 0;
}
