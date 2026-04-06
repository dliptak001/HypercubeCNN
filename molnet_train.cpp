/// Molecular fingerprint classification with HypercubeCNN.
///
/// Input: 1024-bit ECFP4 fingerprint (1 channel, each vertex = 0.0 or 1.0).
/// This is native hypercube data: the full binary vector IS an activation map
/// over 2^10 vertices. No encoding gymnastics needed.

#include "HCNNNetwork.h"
#include "dataloader/MoleculeNetDataset.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

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
    float accuracy;
    float loss;
    float auc_roc;  // approximate via sorted-threshold method
};

/// Compute accuracy, loss, and AUC-ROC on a dataset split.
static EvalResult evaluate(const HCNNNetwork& net,
                           const std::vector<MolSample>& samples) {
    int K = net.get_num_classes();
    int correct = 0;
    float total_loss = 0.0f;

    // For AUC-ROC: collect (score_for_class_1, true_label) pairs
    std::vector<std::pair<float, int>> score_label;
    score_label.reserve(samples.size());

    std::vector<float> logits(K);

    for (const auto& s : samples) {
        if (s.label < 0) continue;  // skip missing labels
        net.forward(s.fingerprint.data(), logits.data());
        total_loss += cross_entropy_loss(logits.data(), K, s.label);
        if (argmax(logits.data(), K) == s.label) ++correct;

        // Softmax to get probability for class 1
        float max_l = std::max(logits[0], logits[1]);
        float p1 = std::exp(logits[1] - max_l) /
                    (std::exp(logits[0] - max_l) + std::exp(logits[1] - max_l));
        score_label.emplace_back(p1, s.label);
    }

    int n = static_cast<int>(score_label.size());
    if (n == 0) return {0.0f, 0.0f, 0.5f};
    EvalResult result{};
    result.accuracy = 100.0f * correct / static_cast<float>(n);
    result.loss = total_loss / static_cast<float>(n);

    // AUC-ROC via trapezoidal rule on sorted scores
    std::sort(score_label.begin(), score_label.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    int pos = 0, neg = 0;
    for (auto& [s, l] : score_label) {
        if (l == 1) ++pos; else ++neg;
    }

    if (pos == 0 || neg == 0) {
        result.auc_roc = 0.5f;  // undefined, return chance
    } else {
        float tp = 0, fp = 0;
        float auc = 0.0f;
        float prev_fpr = 0.0f, prev_tpr = 0.0f;
        for (auto& [s, l] : score_label) {
            if (l == 1) tp += 1.0f; else fp += 1.0f;
            float tpr = tp / pos;
            float fpr = fp / neg;
            auc += 0.5f * (tpr + prev_tpr) * (fpr - prev_fpr);
            prev_tpr = tpr;
            prev_fpr = fpr;
        }
        result.auc_roc = auc;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Parse arguments
    std::string data_path = "data/bbbp_ecfp4_1024.hcfp";
    bool no_conv = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-conv") no_conv = true;
        else data_path = arg;
    }

    std::cout << "Loading " << data_path << "...\n";
    auto ds = load_hcfp(data_path, "BBBP");
    std::cout << "BBBP: train=" << ds.train.size()
              << " val=" << ds.val.size()
              << " test=" << ds.test.size()
              << " bits=" << ds.num_bits << "\n";

    // Count label balance
    int pos = 0;
    for (auto& s : ds.train) if (s.label == 1) ++pos;
    printf("Train label balance: %d/%d positive (%.0f%%)\n",
           pos, static_cast<int>(ds.train.size()),
           100.0f * pos / ds.train.size());

    // Network configuration
    const int DIM = 10;  // 2^10 = 1024 bits
    const int num_classes = 2;
    const int epochs = 60;
    const float lr_max = 0.03f;
    const float lr_min = 1e-5f;
    const float weight_decay = 1e-3f;  // stronger regularization for small dataset
    const float momentum = 0.9f;
    const int batch_size = 32;

    HCNNNetwork net(DIM, num_classes);
    if (no_conv) {
        // Ablation: single conv (1 channel, no ReLU, no bias) + no pooling.
        // This is the minimal pipeline: a linear combination of Hamming-1
        // neighbors → GAP → linear readout. Tests whether stacked conv+pool
        // adds value beyond a single-hop linear filter.
        net.add_conv(1, /*use_relu=*/false, /*use_bias=*/false);  // K=10, 10 params
    } else {
        // Full model: 2 conv+pool stages (~5K params)
        net.add_conv(16, true, true);     // K=10, 1->16
        net.add_pool(PoolType::MAX);      // DIM 10->9
        net.add_conv(32, true, true);     // K=9, 16->32
        net.add_pool(PoolType::MAX);      // DIM 9->8
    }
    net.randomize_all_weights();

    std::cout << "\nArchitecture: " << (no_conv ? "NO CONV (GAP -> linear ablation)" :
              "2 conv+pool stages (16->32)") << "\n";
    std::cout << "Optimizer: SGD, momentum=" << momentum
              << " wd=" << weight_decay
              << " lr=" << lr_max << "->cosine->" << lr_min
              << " batch=" << batch_size
              << " epochs=" << epochs << "\n\n";

    // Prepare input pointers and lengths for train_batch
    int N = net.get_start_N();
    std::vector<const float*> input_ptrs(ds.train.size());
    std::vector<int> input_lens(ds.train.size(), N);
    std::vector<int> targets(ds.train.size());
    for (size_t i = 0; i < ds.train.size(); ++i) {
        input_ptrs[i] = ds.train[i].fingerprint.data();
        targets[i] = ds.train[i].label;
    }

    std::mt19937 shuffle_rng(42);
    float best_val_auc = 0.0f;
    int best_epoch = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Cosine annealing
        float progress = static_cast<float>(epoch) / static_cast<float>(epochs);
        float current_lr = lr_min + 0.5f * (lr_max - lr_min)
                           * (1.0f + std::cos(static_cast<float>(std::numbers::pi) * progress));

        auto t0 = std::chrono::steady_clock::now();

        // Shuffle and train mini-batches
        std::vector<size_t> order(ds.train.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), shuffle_rng);

        int n = static_cast<int>(ds.train.size());
        for (int start = 0; start < n; start += batch_size) {
            int actual = std::min(batch_size, n - start);
            std::vector<const float*> batch_ptrs(actual);
            std::vector<int> batch_lens(actual, N);
            std::vector<int> batch_targets(actual);
            for (int j = 0; j < actual; ++j) {
                size_t idx = order[start + j];
                batch_ptrs[j] = input_ptrs[idx];
                batch_targets[j] = targets[idx];
            }
            net.train_batch(batch_ptrs.data(), batch_lens.data(),
                            batch_targets.data(), actual,
                            current_lr, momentum, weight_decay);
        }

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        // Evaluate on val set
        auto val = evaluate(net, ds.val);

        if (val.auc_roc > best_val_auc) {
            best_val_auc = val.auc_roc;
            best_epoch = epoch + 1;
        }

        printf("ep %2d  val_acc=%5.1f%%  val_auc=%.4f  val_loss=%.4f  lr=%.5f  %.1fs\n",
               epoch + 1, val.accuracy, val.auc_roc, val.loss, current_lr, secs);
    }

    // Final evaluation on test set
    std::cout << "\n--- Final evaluation ---\n";
    auto train_r = evaluate(net, ds.train);
    auto val_r = evaluate(net, ds.val);
    auto test_r = evaluate(net, ds.test);

    printf("Train:  acc=%5.1f%%  auc=%.4f  loss=%.4f\n",
           train_r.accuracy, train_r.auc_roc, train_r.loss);
    printf("Val:    acc=%5.1f%%  auc=%.4f  loss=%.4f  (best auc=%.4f @ ep %d)\n",
           val_r.accuracy, val_r.auc_roc, val_r.loss, best_val_auc, best_epoch);
    printf("Test:   acc=%5.1f%%  auc=%.4f  loss=%.4f\n",
           test_r.accuracy, test_r.auc_roc, test_r.loss);

    // Published baselines for comparison
    std::cout << "\nPublished BBBP baselines (scaffold split):\n";
    std::cout << "  Random Forest:  AUC ~0.71\n";
    std::cout << "  MLP:            AUC ~0.67\n";
    std::cout << "  GCN:            AUC ~0.69\n";
    std::cout << "  MPNN:           AUC ~0.71\n";

    return 0;
}
