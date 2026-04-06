/// Boolean function learning experiment.
/// Tests HCNN on parity, majority, threshold, and DNF functions where
/// the input IS a hypercube vertex (no embedding distortion).
///
/// Each sample: vertex v → bipolar activation (+1 at v, -1 elsewhere), label ∈ {0,1}.
/// Binary classification with 2-class softmax, flatten readout (not GAP).

#include "HCNNNetwork.h"
#include "BooleanData.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

// ---------------------------------------------------------------------------
// Training and evaluation
// ---------------------------------------------------------------------------

static int argmax(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
    return best;
}

static float evaluate(const HCNNNetwork& net,
                      const std::vector<BooleanDataset::Sample>& samples) {
    int N = net.get_start_N();
    int K = net.get_num_classes();
    int correct = 0;
    std::vector<float> embedded(N);
    std::vector<float> logits(K);

    for (const auto& s : samples) {
        // Bipolar encoding: +1 at input vertex, -1 elsewhere
        std::fill(embedded.begin(), embedded.end(), -1.0f);
        embedded[s.vertex] = 1.0f;
        net.forward(embedded.data(), logits.data());
        if (argmax(logits.data(), K) == s.label) ++correct;
    }
    return 100.0f * correct / static_cast<float>(samples.size());
}

struct RunResult {
    std::string name;
    float train_acc;
    float test_acc;
    int epochs_to_95;   // -1 if never reached
    int epochs_to_99;   // -1 if never reached
    double total_secs;
};

static RunResult train_and_eval(const BooleanDataset& ds,
                                int num_channels, int num_conv_layers,
                                bool use_pooling,
                                float lr, int epochs,
                                float weight_decay = 1e-4f) {
    int dim = ds.dim;
    int num_classes = 2;  // binary classification

    // FLATTEN readout: conv+GAP is translation-invariant on the hypercube,
    // so it provably cannot distinguish bipolar-encoded vertices.
    // Flatten preserves positional information needed for vertex classification.
    HCNNNetwork net(dim, num_classes, ReadoutType::FLATTEN);

    // Build conv (+ optional pool) layers
    int c_out = num_channels;
    for (int i = 0; i < num_conv_layers; ++i) {
        net.add_conv(c_out, true, true);
        if (use_pooling) {
            if (dim - 1 >= 3) {
                net.add_pool(PoolType::MAX);
                --dim;
            }
        }
    }
    net.randomize_all_weights();

    // Prepare bipolar training inputs: +1 at input vertex, -1 elsewhere
    int N = net.get_start_N();
    std::vector<std::vector<float>> train_inputs(ds.train.size(), std::vector<float>(N, -1.0f));
    std::vector<const float*> input_ptrs(ds.train.size());
    std::vector<int> targets(ds.train.size());

    for (size_t i = 0; i < ds.train.size(); ++i) {
        train_inputs[i][ds.train[i].vertex] = 1.0f;
        input_ptrs[i] = train_inputs[i].data();
        targets[i] = ds.train[i].label;
    }

    RunResult result;
    result.name = ds.name;
    result.epochs_to_95 = -1;
    result.epochs_to_99 = -1;
    result.total_secs = 0;

    float lr_min = 1e-5f;
    int batch_size = std::min(32, static_cast<int>(ds.train.size()));
    std::mt19937 shuffle_rng(123);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Cosine annealing
        float progress = static_cast<float>(epoch) / static_cast<float>(epochs);
        float current_lr = lr_min + 0.5f * (lr - lr_min)
                           * (1.0f + std::cos(static_cast<float>(std::numbers::pi) * progress));

        auto t0 = std::chrono::steady_clock::now();

        // Mini-batch training
        // Shuffle order each epoch
        std::vector<size_t> order(ds.train.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), shuffle_rng);

        int n = static_cast<int>(ds.train.size());
        for (int start = 0; start < n; start += batch_size) {
            int actual = std::min(batch_size, n - start);
            std::vector<const float*> batch_ptrs(actual);
            std::vector<int> batch_lens(actual);
            std::vector<int> batch_targets(actual);
            for (int j = 0; j < actual; ++j) {
                size_t idx = order[start + j];
                batch_ptrs[j] = input_ptrs[idx];
                batch_lens[j] = N;
                batch_targets[j] = targets[idx];
            }
            net.train_batch(batch_ptrs.data(), batch_lens.data(),
                            batch_targets.data(), actual,
                            current_lr, 0.9f, weight_decay);
        }

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        result.total_secs += secs;

        // Evaluate test set every epoch (small dataset — cheap)
        float test_acc = evaluate(net, ds.test);

        // Print periodically
        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == epochs - 1) {
            float train_acc = evaluate(net, ds.train);
            printf("  ep %3d  train=%5.1f%%  test=%5.1f%%  lr=%.5f  %.2fs\n",
                   epoch + 1, train_acc, test_acc, current_lr, secs);
            result.train_acc = train_acc;
            result.test_acc = test_acc;
        }

        // Track milestones
        if (test_acc >= 95.0f && result.epochs_to_95 < 0)
            result.epochs_to_95 = epoch + 1;
        if (test_acc >= 99.0f && result.epochs_to_99 < 0)
            result.epochs_to_99 = epoch + 1;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Main: run experiments across function types
// ---------------------------------------------------------------------------

int main() {
    const int DIM = 10;
    const float train_fraction = 0.7f;  // 70% train, 30% test
    const int epochs = 100;
    const float lr = 0.01f;           // lower LR for flatten readout (more params)
    const int channels = 16;          // fewer channels to limit flatten readout size
    const int conv_layers = 3;
    const bool use_pooling = true;    // pool to reduce N before flatten

    std::cout << "Boolean Function Learning Experiment\n";
    std::cout << "DIM=" << DIM << " (N=" << (1 << DIM) << ")"
              << " train=" << static_cast<int>(train_fraction * 100) << "%"
              << " epochs=" << epochs
              << " channels=" << channels
              << " conv_layers=" << conv_layers
              << " pooling=" << (use_pooling ? "yes" : "no") << "\n\n";

    // Build datasets
    struct Experiment {
        std::string name;
        std::vector<int> table;
    };

    std::vector<Experiment> experiments = {
        {"Parity (full)",     generate_parity(DIM)},
        {"Parity (k=3)",     generate_k_parity(DIM, 3)},
        {"Majority",         generate_majority(DIM)},
        {"Threshold (k=3)",  generate_threshold(DIM, 3)},
        {"DNF (4 terms, w=3)", generate_random_dnf(DIM, 4, 3)},
    };

    std::vector<RunResult> results;

    for (auto& exp : experiments) {
        auto ds = make_dataset(exp.name, DIM, exp.table, train_fraction);

        // Count label balance
        int pos = 0;
        for (auto& s : ds.train) pos += s.label;
        int total = static_cast<int>(ds.train.size());
        printf("--- %s (train: %d/%d positive = %.0f%%) ---\n",
               ds.name.c_str(), pos, total,
               100.0f * pos / total);

        auto result = train_and_eval(ds, channels, conv_layers, use_pooling,
                                     lr, epochs);
        results.push_back(result);
        std::cout << "\n";
    }

    // Summary table
    std::cout << "=== SUMMARY ===\n";
    printf("%-24s  Train%%  Test%%  ep→95%%  ep→99%%  Time\n", "Function");
    for (auto& r : results) {
        printf("%-24s  %5.1f  %5.1f  ", r.name.c_str(), r.train_acc, r.test_acc);
        if (r.epochs_to_95 >= 0) printf("%5d   ", r.epochs_to_95);
        else printf("  n/a   ");
        if (r.epochs_to_99 >= 0) printf("%5d   ", r.epochs_to_99);
        else printf("  n/a   ");
        printf("%.1fs\n", r.total_secs);
    }

    return 0;
}
