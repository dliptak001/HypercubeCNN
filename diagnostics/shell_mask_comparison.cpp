/// Comparison test: shell+nn masks (K=2*DIM-2) vs nn-only (K=DIM).
/// Trains two identical architectures on MNIST for a few epochs,
/// prints per-epoch timing and test accuracy side by side.

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

struct EvalResult {
    float loss;
    float accuracy;
};

static EvalResult evaluate(const HCNNNetwork& net, const HCNNMNISTDataset& dataset) {
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

    return { total_loss / count, 100.0f * correct / count };
}

struct RunResult {
    std::vector<double> epoch_secs;
    std::vector<float> epoch_acc;
    std::vector<float> epoch_loss;
};

static RunResult train_and_eval(const char* label, HCNNNetwork& net,
                                HCNNMNISTDataset& train_data,
                                const HCNNMNISTDataset& test_data,
                                float lr, int batch_size, int epochs) {
    RunResult result;

    // Print K per conv layer
    std::cout << label << ": ";
    for (size_t i = 0; i < net.get_num_conv(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "conv" << i << ".K=" << net.get_conv(i).get_K();
    }
    std::cout << "\n";

    auto r0 = evaluate(net, test_data);
    std::cout << "  init  loss=" << r0.loss << "  acc=" << r0.accuracy << "%\n";

    float current_lr = lr;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto t0 = std::chrono::steady_clock::now();
        train_data.train_epoch(net, current_lr, 0.9f, batch_size);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        auto r = evaluate(net, test_data);
        result.epoch_secs.push_back(secs);
        result.epoch_acc.push_back(r.accuracy);
        result.epoch_loss.push_back(r.loss);

        std::cout << "  ep " << (epoch + 1) << "  loss=" << r.loss
                  << "  acc=" << r.accuracy << "%  " << secs << "s\n";

        if ((epoch + 1) % 5 == 0) current_lr *= 0.5f;
    }
    return result;
}

int main() {
    auto src_dir = std::filesystem::path(__FILE__).parent_path().parent_path();
    auto data_dir = src_dir / "data";

    std::cout << "Loading MNIST from " << data_dir << "...\n";
    auto train_data = load_mnist((data_dir / "train-images-idx3-ubyte").string(),
                                 (data_dir / "train-labels-idx1-ubyte").string(), 20000);
    auto test_data  = load_mnist((data_dir / "t10k-images-idx3-ubyte").string(),
                                 (data_dir / "t10k-labels-idx1-ubyte").string(), 2000);
    std::cout << "Train: " << train_data.size() << "  Test: " << test_data.size() << "\n\n";

    const float lr = 0.16f;
    const int batch_size = 32;
    const int epochs = 20;

    // --- Run A: full masks (shell + nn) ---
    HCNNNetwork net_full(10);
    net_full.add_conv(16, true, true, true);   // K=18
    net_full.add_pool(PoolType::MAX);
    net_full.add_conv(32, true, true, true);   // K=16
    net_full.add_pool(PoolType::MAX);
    net_full.add_conv(64, true, true, true);   // K=14
    net_full.add_pool(PoolType::MAX);
    net_full.randomize_all_weights();          // He init

    auto result_full = train_and_eval("FULL (shell+nn)", net_full,
                                      train_data, test_data, lr, batch_size, epochs);

    std::cout << "\n";

    // --- Run B: nn-only masks ---
    HCNNNetwork net_nn(10);
    net_nn.add_conv(16, true, true, false);    // K=10
    net_nn.add_pool(PoolType::MAX);
    net_nn.add_conv(32, true, true, false);    // K=9
    net_nn.add_pool(PoolType::MAX);
    net_nn.add_conv(64, true, true, false);    // K=8
    net_nn.add_pool(PoolType::MAX);
    net_nn.randomize_all_weights();            // He init

    auto result_nn = train_and_eval("NN-ONLY", net_nn,
                                    train_data, test_data, lr, batch_size, epochs);

    // --- Summary ---
    std::cout << "\n=== COMPARISON SUMMARY ===\n";
    std::cout << "Epoch  Full_acc  NN_acc   Full_s  NN_s   Speedup\n";
    double total_full = 0, total_nn = 0;
    for (int i = 0; i < epochs; ++i) {
        double spd = result_full.epoch_secs[i] / result_nn.epoch_secs[i];
        total_full += result_full.epoch_secs[i];
        total_nn += result_nn.epoch_secs[i];
        printf("  %2d   %5.1f%%   %5.1f%%   %5.2fs  %5.2fs  %.2fx\n",
               i + 1,
               result_full.epoch_acc[i], result_nn.epoch_acc[i],
               result_full.epoch_secs[i], result_nn.epoch_secs[i], spd);
    }
    printf("\nTotal: full=%.1fs  nn=%.1fs  speedup=%.2fx\n",
           total_full, total_nn, total_full / total_nn);
    printf("Final: full=%.1f%%  nn=%.1f%%  delta=%+.1f%%\n",
           result_full.epoch_acc.back(), result_nn.epoch_acc.back(),
           result_full.epoch_acc.back() - result_nn.epoch_acc.back());

    return 0;
}
