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
    auto ds = load_hcfp(data_path, "dataset");
    std::cout << ds.name << ": train=" << ds.train.size()
              << " val=" << ds.val.size()
              << " test=" << ds.test.size()
              << " bits=" << ds.num_bits
              << " tasks=" << ds.num_tasks << "\n";

    if (ds.num_tasks > 1) {
        std::cerr << "Multi-task datasets not yet supported (got "
                  << ds.num_tasks << " tasks). Use single-task datasets.\n";
        return 1;
    }

    // Count label balance (skip missing)
    int pos = 0, valid_train = 0;
    for (auto& s : ds.train) {
        if (s.label < 0) continue;
        ++valid_train;
        if (s.label == 1) ++pos;
    }
    int neg = valid_train - pos;
    float pos_frac = static_cast<float>(pos) / valid_train;
    printf("Train label balance: %d/%d positive (%.0f%%)\n",
           pos, valid_train, 100.0f * pos_frac);

    // For extreme imbalance, oversample minority class to ~50/50.
    // This ensures every batch sees positives, which is more effective than
    // loss weighting when the minority is <10%.
    bool oversample = (pos_frac < 0.2f || pos_frac > 0.8f);
    std::vector<size_t> pos_indices, neg_indices;
    for (size_t i = 0; i < ds.train.size(); ++i) {
        if (ds.train[i].label == 1) pos_indices.push_back(i);
        else if (ds.train[i].label == 0) neg_indices.push_back(i);
    }
    if (oversample) {
        printf("Oversampling: %d pos -> %d (to match %d neg)\n",
               pos, neg, neg);
    }

    // Scale architecture and hyperparameters to dataset size
    const int DIM = 10;
    const int num_classes = 2;
    const float momentum = 0.9f;
    const int batch_size = 32;

    bool large_dataset = (ds.train.size() > 5000);
    int epochs       = large_dataset ? 40 : 60;
    float lr_max     = large_dataset ? 0.03f : 0.03f;
    float lr_min     = 1e-5f;
    float weight_decay = large_dataset ? 1e-4f : 1e-3f;

    HCNNNetwork net(DIM, num_classes);
    if (no_conv) {
        net.add_conv(1, /*use_relu=*/false, /*use_bias=*/false);
    } else if (large_dataset) {
        // Larger model for >5K samples (~200K params)
        net.add_conv(32, true, true);
        net.add_pool(PoolType::MAX);
        net.add_conv(64, true, true);
        net.add_pool(PoolType::MAX);
        net.add_conv(128, true, true);
        net.add_pool(PoolType::MAX);
        net.add_conv(128, true, true);
        net.add_pool(PoolType::MAX);
    } else {
        // Small model for <5K samples (~5K params)
        net.add_conv(16, true, true);
        net.add_pool(PoolType::MAX);
        net.add_conv(32, true, true);
        net.add_pool(PoolType::MAX);
    }
    net.randomize_all_weights();

    std::cout << "\nArchitecture: ";
    if (no_conv) std::cout << "NO CONV (ablation)";
    else if (large_dataset) std::cout << "4 conv+pool (32->64->128->128) ~200K params";
    else std::cout << "2 conv+pool (16->32) ~5K params";
    std::cout << "\n";
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

        // Build epoch training order: oversample minority if needed
        std::vector<size_t> order;
        if (oversample) {
            // All negatives + positives repeated to match
            order = neg_indices;
            std::shuffle(order.begin(), order.end(), shuffle_rng);
            // Resample positives with replacement to match neg count
            std::vector<size_t> pos_resampled(neg_indices.size());
            std::uniform_int_distribution<size_t> pos_dist(0, pos_indices.size() - 1);
            for (size_t i = 0; i < pos_resampled.size(); ++i)
                pos_resampled[i] = pos_indices[pos_dist(shuffle_rng)];
            order.insert(order.end(), pos_resampled.begin(), pos_resampled.end());
        } else {
            order.resize(ds.train.size());
            std::iota(order.begin(), order.end(), 0);
        }
        std::shuffle(order.begin(), order.end(), shuffle_rng);

        int n = static_cast<int>(order.size());
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
                            current_lr, momentum, weight_decay,
                            nullptr);
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

    // Published MoleculeNet baselines (scaffold split, 2018 paper):
    // BBBP:  RF ~0.71, MLP ~0.67, GCN ~0.69, MPNN ~0.71
    // BACE:  RF ~0.68, MLP ~0.67, GCN ~0.78, MPNN ~0.81
    // HIV:   RF ~0.79, MLP ~0.78, GCN ~0.76, MPNN ~0.77

    return 0;
}
