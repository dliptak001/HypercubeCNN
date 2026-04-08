#include "HCNNNetwork.h"
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const std::string& name) {
    if (condition) {
        std::cout << "  PASS  " << name << "\n";
    } else {
        std::cout << "  FAIL  " << name << "\n";
        ++failures;
    }
}

static bool is_finite(float v) {
    return std::isfinite(v);
}

static bool all_finite(const float* v, int n) {
    for (int i = 0; i < n; ++i)
        if (!is_finite(v[i])) return false;
    return true;
}

// --- Test functions ---

static void test_construction() {
    std::cout << "\n[Construction]\n";

    HCNNNetwork net(5, 4);  // DIM=5, N=32, 4 classes
    check(net.get_start_dim() == 5, "start_dim = 5");
    check(net.get_start_N() == 32, "start_N = 32");
    check(net.get_num_classes() == 4, "num_classes = 4");
    check(net.get_num_conv() == 0, "no conv layers initially");
    check(net.get_num_pool() == 0, "no pool layers initially");

    net.add_conv(8, true, true);
    net.add_pool(PoolType::MAX);
    net.add_conv(16, true, true);

    check(net.get_num_conv() == 2, "2 conv layers after add");
    check(net.get_num_pool() == 1, "1 pool layer after add");
    check(net.get_conv(0).get_c_out() == 8, "conv[0] c_out = 8");
    check(net.get_conv(1).get_c_out() == 16, "conv[1] c_out = 16");
    check(net.get_conv(0).get_K() == 5, "conv[0] K = DIM = 5");
    check(net.get_conv(1).get_K() == 4, "conv[1] K = 4 (after pool)");
}

static void test_forward_pass() {
    std::cout << "\n[Forward pass]\n";

    HCNNNetwork net(5, 4);
    net.add_conv(8, true, true);
    net.add_pool(PoolType::MAX);
    net.add_conv(16, true, true);
    net.randomize_all_weights();

    int N = net.get_start_N();
    int K = net.get_num_classes();

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> input(N);
    for (auto& v : input) v = dist(rng);

    std::vector<float> embedded(N);
    net.embed_input(input.data(), N, embedded.data());
    check(all_finite(embedded.data(), N), "embedded values are finite");

    std::vector<float> logits(K);
    net.forward(embedded.data(), logits.data());
    check(all_finite(logits.data(), K), "logits are finite after forward");

    // Softmax should produce valid probabilities
    float max_l = logits[0];
    for (int i = 1; i < K; ++i) if (logits[i] > max_l) max_l = logits[i];
    float sum_exp = 0.0f;
    for (int i = 0; i < K; ++i) sum_exp += std::exp(logits[i] - max_l);
    check(sum_exp > 0.0f, "softmax denominator is positive");
}

static void test_training_step() {
    std::cout << "\n[Training step]\n";

    HCNNNetwork net(5, 4);
    net.add_conv(16, true, true);
    net.randomize_all_weights();

    int N = net.get_start_N();
    int K = net.get_num_classes();

    // Generate synthetic data: 20 samples, 4 classes
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int num_samples = 20;
    std::vector<std::vector<float>> inputs(num_samples, std::vector<float>(N));
    std::vector<int> targets(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        for (auto& v : inputs[i]) v = dist(rng);
        targets[i] = i % K;
    }

    // Compute initial loss
    auto compute_loss = [&]() {
        double total = 0.0;
        for (int i = 0; i < num_samples; ++i) {
            std::vector<float> emb(N), logits(K);
            net.embed_input(inputs[i].data(), N, emb.data());
            net.forward(emb.data(), logits.data());
            double max_l = logits[0];
            for (int j = 1; j < K; ++j) if (logits[j] > max_l) max_l = logits[j];
            double se = 0.0;
            for (int j = 0; j < K; ++j) se += std::exp(logits[j] - max_l);
            total += -(logits[targets[i]] - max_l) + std::log(se);
        }
        return total / num_samples;
    };

    double loss_before = compute_loss();
    check(is_finite(static_cast<float>(loss_before)), "initial loss is finite");

    for (int step = 0; step < 100; ++step) {
        int idx = step % num_samples;
        net.train_step(inputs[idx].data(), N, targets[idx], 0.01f);
    }

    double loss_after = compute_loss();
    check(is_finite(static_cast<float>(loss_after)), "loss after training is finite");
    check(loss_after < loss_before, "loss decreased after 100 train_step calls");
}

static void test_batch_training() {
    std::cout << "\n[Batch training]\n";

    HCNNNetwork net(5, 4);
    net.add_conv(16, true, true);
    net.randomize_all_weights();

    int N = net.get_start_N();
    int K = net.get_num_classes();
    const int batch_size = 8;

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<float>> inputs(batch_size, std::vector<float>(N));
    std::vector<const float*> input_ptrs(batch_size);
    std::vector<int> lengths(batch_size, N);
    std::vector<int> targets(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        for (auto& v : inputs[i]) v = dist(rng);
        input_ptrs[i] = inputs[i].data();
        targets[i] = i % K;
    }

    // Should not crash
    net.train_batch(input_ptrs.data(), lengths.data(), targets.data(),
                    batch_size, 0.01f);

    // Verify logits are still finite after batch training
    std::vector<float> logits(K);
    std::vector<float> emb(N);
    net.embed_input(inputs[0].data(), N, emb.data());
    net.forward(emb.data(), logits.data());
    check(all_finite(logits.data(), K), "logits finite after train_batch");
}

static void test_batch_inference() {
    std::cout << "\n[Batch inference]\n";

    HCNNNetwork net(5, 4);
    net.add_conv(16, true, true);
    net.add_pool(PoolType::MAX);
    net.randomize_all_weights();

    int N = net.get_start_N();
    int K = net.get_num_classes();
    const int batch_size = 8;

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<float>> inputs(batch_size, std::vector<float>(N));
    std::vector<const float*> input_ptrs(batch_size);
    std::vector<int> lengths(batch_size, N);
    for (int i = 0; i < batch_size; ++i) {
        for (auto& v : inputs[i]) v = dist(rng);
        input_ptrs[i] = inputs[i].data();
    }

    std::vector<float> all_logits(batch_size * K);
    net.forward_batch(input_ptrs.data(), lengths.data(), batch_size, all_logits.data());
    check(all_finite(all_logits.data(), batch_size * K),
          "all logits finite from forward_batch");

    // Verify batch results match single-sample results
    bool match = true;
    for (int i = 0; i < batch_size; ++i) {
        std::vector<float> emb(N), single_logits(K);
        net.embed_input(inputs[i].data(), N, emb.data());
        net.forward(emb.data(), single_logits.data());
        for (int j = 0; j < K; ++j) {
            if (std::fabs(single_logits[j] - all_logits[i * K + j]) > 1e-4f) {
                match = false;
                break;
            }
        }
    }
    check(match, "batch inference matches single-sample inference");
}

static void test_readout_types() {
    std::cout << "\n[Readout types]\n";

    int N = 32; // DIM=5

    // GAP readout
    {
        HCNNNetwork net(5, 4, 1, ReadoutType::GAP);
        net.add_conv(8, true, true);
        net.randomize_all_weights();
        std::mt19937 rng(111);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> input(N);
        for (auto& v : input) v = dist(rng);
        std::vector<float> emb(N), logits(4);
        net.embed_input(input.data(), N, emb.data());
        net.forward(emb.data(), logits.data());
        check(all_finite(logits.data(), 4), "GAP readout produces finite logits");
    }

    // FLATTEN readout
    {
        HCNNNetwork net(5, 4, 1, ReadoutType::FLATTEN);
        net.add_conv(8, true, true);
        net.randomize_all_weights();
        std::mt19937 rng(222);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> input(N);
        for (auto& v : input) v = dist(rng);
        std::vector<float> emb(N), logits(4);
        net.embed_input(input.data(), N, emb.data());
        net.forward(emb.data(), logits.data());
        check(all_finite(logits.data(), 4), "FLATTEN readout produces finite logits");
    }
}

static void test_pool_types() {
    std::cout << "\n[Pool types]\n";

    auto test_pool = [](PoolType type, const char* name) {
        HCNNNetwork net(5, 4);
        net.add_conv(8, true, true);
        net.add_pool(type);
        net.randomize_all_weights();

        int N = net.get_start_N();
        std::mt19937 rng(333);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> input(N);
        for (auto& v : input) v = dist(rng);
        std::vector<float> emb(N), logits(4);
        net.embed_input(input.data(), N, emb.data());
        net.forward(emb.data(), logits.data());
        check(all_finite(logits.data(), 4),
              std::string(name) + " pool produces finite logits");
    };

    test_pool(PoolType::MAX, "MAX");
    test_pool(PoolType::AVG, "AVG");
}

int main() {
    std::cout << "HypercubeCNN Core Smoke Test\n";
    std::cout << "============================\n";

    test_construction();
    test_forward_pass();
    test_training_step();
    test_batch_training();
    test_batch_inference();
    test_readout_types();
    test_pool_types();

    std::cout << "\n============================\n";
    if (failures == 0) {
        std::cout << "All tests PASSED\n";
    } else {
        std::cout << failures << " test(s) FAILED\n";
    }
    return failures > 0 ? 1 : 0;
}
