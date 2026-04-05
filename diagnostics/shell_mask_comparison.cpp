/// Comparison test: shell+nn masks (K=2*DIM-2) vs nn-only (K=DIM).
/// Tests multiple LR schedules to find nn-only configuration that matches
/// full-mask convergence speed while retaining the wall-clock speedup.

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
    std::string label;
    std::vector<double> epoch_secs;
    std::vector<float> epoch_acc;
    std::vector<float> epoch_loss;
    std::vector<float> epoch_lr;
};

/// LR schedule types
enum class LRSchedule { STEP_DECAY, WARMUP_STEP_DECAY };

struct RunConfig {
    const char* label;
    bool use_shell_masks;
    float start_lr;
    LRSchedule schedule;
    int warmup_epochs;     // only used for WARMUP_STEP_DECAY
    float peak_lr;         // only used for WARMUP_STEP_DECAY
};

static RunResult train_and_eval(RunConfig cfg, HCNNMNISTDataset& train_data,
                                const HCNNMNISTDataset& test_data,
                                int batch_size, int epochs) {
    HCNNNetwork net(10);
    net.add_conv(16, true, true, cfg.use_shell_masks);
    net.add_pool(PoolType::MAX);
    net.add_conv(32, true, true, cfg.use_shell_masks);
    net.add_pool(PoolType::MAX);
    net.add_conv(64, true, true, cfg.use_shell_masks);
    net.add_pool(PoolType::MAX);
    net.randomize_all_weights();

    RunResult result;
    result.label = cfg.label;

    // Print K per conv layer
    std::cout << cfg.label << ": ";
    for (size_t i = 0; i < net.get_num_conv(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "conv" << i << ".K=" << net.get_conv(i).get_K();
    }
    std::cout << "\n";

    auto r0 = evaluate(net, test_data);
    std::cout << "  init  loss=" << r0.loss << "  acc=" << r0.accuracy << "%\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Compute LR for this epoch
        float lr;
        if (cfg.schedule == LRSchedule::WARMUP_STEP_DECAY) {
            if (epoch < cfg.warmup_epochs) {
                // Linear warmup from start_lr to peak_lr
                float t = static_cast<float>(epoch + 1) / static_cast<float>(cfg.warmup_epochs);
                lr = cfg.start_lr + t * (cfg.peak_lr - cfg.start_lr);
            } else {
                // Step decay from peak_lr, halving every 5 epochs after warmup
                lr = cfg.peak_lr;
                int decay_steps = (epoch - cfg.warmup_epochs) / 5;
                for (int d = 0; d < decay_steps; ++d) lr *= 0.5f;
            }
        } else {
            // Simple step decay: halve every 5 epochs
            lr = cfg.start_lr;
            int decay_steps = epoch / 5;
            for (int d = 0; d < decay_steps; ++d) lr *= 0.5f;
        }

        auto t0 = std::chrono::steady_clock::now();
        train_data.train_epoch(net, lr, 0.9f, batch_size);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        auto r = evaluate(net, test_data);
        result.epoch_secs.push_back(secs);
        result.epoch_acc.push_back(r.accuracy);
        result.epoch_loss.push_back(r.loss);
        result.epoch_lr.push_back(lr);

        printf("  ep %2d  loss=%.4f  acc=%5.1f%%  lr=%.4f  %.2fs\n",
               epoch + 1, r.loss, r.accuracy, lr, secs);
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

    const int batch_size = 32;
    const int epochs = 20;

    // --- Configurations to compare ---
    std::vector<RunConfig> configs = {
        // Baseline: full masks with original LR
        { "A: Full  lr=0.16 step",    true,  0.16f, LRSchedule::STEP_DECAY, 0, 0.0f },
        // NN-only with same LR (shows plateau)
        { "B: NN    lr=0.16 step",    false, 0.16f, LRSchedule::STEP_DECAY, 0, 0.0f },
        // NN-only with lower starting LR
        { "C: NN    lr=0.04 step",    false, 0.04f, LRSchedule::STEP_DECAY, 0, 0.0f },
        // NN-only with warmup: start low, ramp to 0.16 over 3 epochs, then decay
        { "D: NN    warmup->0.16",    false, 0.01f, LRSchedule::WARMUP_STEP_DECAY, 3, 0.16f },
        // NN-only with warmup: start low, ramp to 0.08 over 3 epochs, then decay
        { "E: NN    warmup->0.08",    false, 0.01f, LRSchedule::WARMUP_STEP_DECAY, 3, 0.08f },
    };

    std::vector<RunResult> results;
    for (auto& cfg : configs) {
        results.push_back(train_and_eval(cfg, train_data, test_data, batch_size, epochs));
        std::cout << "\n";
    }

    // --- Summary table ---
    std::cout << "=== COMPARISON SUMMARY ===\n";
    printf("%-24s", "Epoch");
    for (auto& r : results) printf("  %s", r.label.c_str());
    printf("\n");

    for (int i = 0; i < epochs; ++i) {
        printf("  %2d  ", i + 1);
        for (auto& r : results) {
            printf("  %5.1f%% %5.1fs", r.epoch_acc[i], r.epoch_secs[i]);
        }
        printf("\n");
    }

    // Totals
    printf("\n%-24s", "Total time");
    for (auto& r : results) {
        double total = 0;
        for (auto s : r.epoch_secs) total += s;
        printf("  %8.1fs     ", total);
    }

    printf("\n%-24s", "Final accuracy");
    for (auto& r : results) {
        printf("  %8.1f%%     ", r.epoch_acc.back());
    }

    printf("\n%-24s", "Best accuracy");
    for (auto& r : results) {
        float best = 0;
        for (auto a : r.epoch_acc) if (a > best) best = a;
        printf("  %8.1f%%     ", best);
    }

    printf("\n%-24s", "Epochs to 80%%");
    for (auto& r : results) {
        int ep = -1;
        for (int i = 0; i < epochs; ++i) {
            if (r.epoch_acc[i] >= 80.0f) { ep = i + 1; break; }
        }
        if (ep > 0) printf("  %8d      ", ep);
        else printf("       n/a      ");
    }

    printf("\n%-24s", "Epochs to 90%%");
    for (auto& r : results) {
        int ep = -1;
        for (int i = 0; i < epochs; ++i) {
            if (r.epoch_acc[i] >= 90.0f) { ep = i + 1; break; }
        }
        if (ep > 0) printf("  %8d      ", ep);
        else printf("       n/a      ");
    }

    printf("\n%-24s", "Speedup vs A");
    double total_a = 0;
    for (auto s : results[0].epoch_secs) total_a += s;
    for (auto& r : results) {
        double total = 0;
        for (auto s : r.epoch_secs) total += s;
        printf("  %8.2fx     ", total_a / total);
    }
    printf("\n");

    return 0;
}
