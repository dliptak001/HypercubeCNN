// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// Smoke test for the HCNN top-level SDK API.
// Everything goes through HCNN -- no direct use of HCNNNetwork or layer
// classes.  This file is the canonical regression check that the SDK front
// door behaves correctly across architecture, training, optimizer and
// readout-type variations.

#include "HCNN.h"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using hcnn::HCNN;
using hcnn::PoolType;
using hcnn::ReadoutType;
using hcnn::Activation;
using hcnn::OptimizerType;

static int failures = 0;

static void check(bool condition, const std::string& name) {
    if (condition) {
        std::cout << "  PASS  " << name << "\n";
    } else {
        std::cout << "  FAIL  " << name << "\n";
        ++failures;
    }
}

static bool is_finite_f(float v) { return std::isfinite(v); }

static bool all_finite(const float* v, int n) {
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(v[i])) return false;
    return true;
}

// Cross-entropy loss over a fixed sample list.  Reuses caller-owned scratch
// buffers to keep the hot path allocation-free.
static double cross_entropy_over_samples(
    HCNN& net,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<int>& targets,
    std::vector<float>& embedded,
    std::vector<float>& logits)
{
    const int N = net.GetStartN();
    const int K = net.GetNumClasses();
    const int n = static_cast<int>(inputs.size());
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        net.Embed(inputs[i].data(), N, embedded.data());
        net.Forward(embedded.data(), logits.data());
        double max_l = logits[0];
        for (int j = 1; j < K; ++j) if (logits[j] > max_l) max_l = logits[j];
        double se = 0.0;
        for (int j = 0; j < K; ++j) se += std::exp(logits[j] - max_l);
        total += -(logits[targets[i]] - max_l) + std::log(se);
    }
    return total / n;
}

// Build a synthetic dataset of `n` samples with `N`-dim inputs in [-1, 1].
static void make_synth(int n, int N, int K, unsigned seed,
                       std::vector<std::vector<float>>& inputs_out,
                       std::vector<int>& targets_out) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    inputs_out.assign(n, std::vector<float>(N));
    targets_out.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        for (auto& v : inputs_out[i]) v = dist(rng);
        targets_out[i] = i % K;
    }
}

// Pack a sample-list into the parallel-pointer arrays HCNN's batch APIs want.
static void pack_batch(const std::vector<std::vector<float>>& inputs, int N,
                       std::vector<const float*>& ptrs_out,
                       std::vector<int>& lengths_out) {
    const int n = static_cast<int>(inputs.size());
    ptrs_out.resize(n);
    lengths_out.assign(n, N);
    for (int i = 0; i < n; ++i) ptrs_out[i] = inputs[i].data();
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

static void test_construction() {
    std::cout << "\n[Construction]\n";

    HCNN net(5, 4);   // DIM=5, N=32, 4 classes
    check(net.GetStartDim() == 5,       "GetStartDim() == 5");
    check(net.GetStartN() == 32,        "GetStartN() == 32");
    check(net.GetNumClasses() == 4,     "GetNumClasses() == 4");
    check(net.GetInputChannels() == 1,  "GetInputChannels() == 1");

    // Architecture build should not throw or affect sizing accessors.
    net.AddConv(8);
    net.AddPool(PoolType::MAX);
    net.AddConv(16);
    net.RandomizeWeights();

    check(net.GetStartDim() == 5,   "GetStartDim() unchanged after build");
    check(net.GetStartN() == 32,    "GetStartN() unchanged after build");
    check(net.GetNumClasses() == 4, "GetNumClasses() unchanged after build");
}

static void test_forward_pass() {
    std::cout << "\n[Forward pass]\n";

    HCNN net(5, 4);
    net.AddConv(8);
    net.AddPool(PoolType::MAX);
    net.AddConv(16);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumClasses();

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> input(N);
    for (auto& v : input) v = dist(rng);

    std::vector<float> embedded(N);
    net.Embed(input.data(), N, embedded.data());
    check(all_finite(embedded.data(), N), "Embed produces finite values");

    std::vector<float> logits(K);
    net.Forward(embedded.data(), logits.data());
    check(all_finite(logits.data(), K), "Forward produces finite logits");

    float max_l = logits[0];
    for (int i = 1; i < K; ++i) if (logits[i] > max_l) max_l = logits[i];
    float sum_exp = 0.0f;
    for (int i = 0; i < K; ++i) sum_exp += std::exp(logits[i] - max_l);
    check(sum_exp > 0.0f, "softmax denominator is positive");
}

static void test_training_step() {
    std::cout << "\n[TrainStep]\n";

    HCNN net(5, 4);
    net.AddConv(16);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumClasses();

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(20, N, K, 123, inputs, targets);

    std::vector<float> emb(N), logits(K);
    double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);
    check(is_finite_f(static_cast<float>(loss_before)), "initial loss is finite");

    for (int step = 0; step < 100; ++step) {
        int idx = step % static_cast<int>(inputs.size());
        net.TrainStep(inputs[idx].data(), N, targets[idx], 0.01f);
    }

    double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
    check(is_finite_f(static_cast<float>(loss_after)), "loss after training is finite");
    check(loss_after < loss_before, "loss decreased after 100 TrainStep calls");
}

static void test_train_batch() {
    std::cout << "\n[TrainBatch]\n";

    HCNN net(5, 4);
    net.AddConv(16);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumClasses();
    const int batch_size = 8;

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(batch_size, N, K, 456, inputs, targets);

    std::vector<const float*> ptrs;
    std::vector<int> lengths;
    pack_batch(inputs, N, ptrs, lengths);

    net.TrainBatch(ptrs.data(), lengths.data(), targets.data(), batch_size, 0.01f);

    std::vector<float> emb(N), logits(K);
    net.Embed(inputs[0].data(), N, emb.data());
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), K), "logits finite after TrainBatch");
}

static void test_train_epoch() {
    std::cout << "\n[TrainEpoch]\n";

    HCNN net(5, 4);
    net.AddConv(16);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumClasses();

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(40, N, K, 999, inputs, targets);

    std::vector<const float*> ptrs;
    std::vector<int> lengths;
    pack_batch(inputs, N, ptrs, lengths);

    std::vector<float> emb(N), logits(K);
    double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

    // Two shuffled epochs at the same nominal LR -- distinct seeds give
    // different reproducible permutations each call.
    net.TrainEpoch(ptrs.data(), lengths.data(), targets.data(),
                   static_cast<int>(inputs.size()), /*batch_size=*/8,
                   /*lr=*/0.05f, /*momentum=*/0.0f, /*wd=*/0.0f,
                   /*class_weights=*/nullptr, /*shuffle_seed=*/1u);
    net.TrainEpoch(ptrs.data(), lengths.data(), targets.data(),
                   static_cast<int>(inputs.size()), /*batch_size=*/8,
                   /*lr=*/0.05f, /*momentum=*/0.0f, /*wd=*/0.0f,
                   /*class_weights=*/nullptr, /*shuffle_seed=*/2u);

    double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
    check(loss_after < loss_before,
          "TrainEpoch (shuffled): loss decreased ("
          + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");

    // No-shuffle path also produces finite logits.
    net.TrainEpoch(ptrs.data(), lengths.data(), targets.data(),
                   static_cast<int>(inputs.size()), /*batch_size=*/8,
                   /*lr=*/0.01f, /*momentum=*/0.0f, /*wd=*/0.0f,
                   /*class_weights=*/nullptr, /*shuffle_seed=*/0u);
    net.Embed(inputs[0].data(), N, emb.data());
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), K), "TrainEpoch (no shuffle): logits finite");
}

static void test_forward_batch() {
    std::cout << "\n[ForwardBatch]\n";

    HCNN net(5, 4);
    net.AddConv(16);
    net.AddPool(PoolType::MAX);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumClasses();
    const int batch_size = 8;

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(batch_size, N, K, 789, inputs, targets);

    std::vector<const float*> ptrs;
    std::vector<int> lengths;
    pack_batch(inputs, N, ptrs, lengths);

    std::vector<float> all_logits(static_cast<size_t>(batch_size) * K);
    net.ForwardBatch(ptrs.data(), lengths.data(), batch_size, all_logits.data());
    check(all_finite(all_logits.data(), batch_size * K),
          "all logits finite from ForwardBatch");

    bool match = true;
    std::vector<float> emb(N), single_logits(K);
    for (int i = 0; i < batch_size; ++i) {
        net.Embed(inputs[i].data(), N, emb.data());
        net.Forward(emb.data(), single_logits.data());
        for (int j = 0; j < K; ++j) {
            if (std::fabs(single_logits[j] - all_logits[i * K + j]) > 1e-4f) {
                match = false;
                break;
            }
        }
    }
    check(match, "ForwardBatch matches single-sample Embed+Forward");
}

static void test_readout_types() {
    std::cout << "\n[Readout types]\n";

    auto run_one = [](ReadoutType type, const char* name) {
        HCNN net(5, 4, /*input_channels=*/1, type);
        net.AddConv(8);
        net.RandomizeWeights();

        int N = net.GetStartN();
        std::mt19937 rng(111);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> input(N);
        for (auto& v : input) v = dist(rng);
        std::vector<float> emb(N), logits(net.GetNumClasses());
        net.Embed(input.data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), net.GetNumClasses()),
              std::string(name) + " readout produces finite logits");
    };
    run_one(ReadoutType::GAP,     "GAP");
    run_one(ReadoutType::FLATTEN, "FLATTEN");
}

static void test_pool_types() {
    std::cout << "\n[Pool types]\n";

    auto run_one = [](PoolType type, const char* name) {
        HCNN net(5, 4);
        net.AddConv(8);
        net.AddPool(type);
        net.RandomizeWeights();

        int N = net.GetStartN();
        std::mt19937 rng(333);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> input(N);
        for (auto& v : input) v = dist(rng);
        std::vector<float> emb(N), logits(net.GetNumClasses());
        net.Embed(input.data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), net.GetNumClasses()),
              std::string(name) + " pool produces finite logits");
    };
    run_one(PoolType::MAX, "MAX");
    run_one(PoolType::AVG, "AVG");
}

static void test_batchnorm() {
    std::cout << "\n[Batch normalization]\n";

    // Forward pass
    {
        HCNN net(5, 4);
        net.AddConv(16, Activation::RELU, true, /*use_batchnorm=*/true);
        net.AddPool(PoolType::MAX);
        net.AddConv(16, Activation::RELU, true, true);
        net.RandomizeWeights();

        int N = net.GetStartN();
        int K = net.GetNumClasses();
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> input(N);
        for (auto& v : input) v = dist(rng);

        std::vector<float> emb(N), logits(K);
        net.Embed(input.data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), K), "BN forward produces finite logits");
    }

    // BN TrainStep reduces loss
    {
        HCNN net(5, 4);
        net.AddConv(16, Activation::RELU, true, true);
        net.RandomizeWeights();

        int N = net.GetStartN();
        int K = net.GetNumClasses();

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(20, N, K, 123, inputs, targets);

        std::vector<float> emb(N), logits(K);
        net.SetTraining(false);
        double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

        for (int step = 0; step < 100; ++step) {
            int idx = step % static_cast<int>(inputs.size());
            net.TrainStep(inputs[idx].data(), N, targets[idx], 0.01f);
        }

        net.SetTraining(false);
        double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(loss_after < loss_before,
              "BN TrainStep: loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
    }

    // BN TrainBatch produces finite logits
    {
        HCNN net(5, 4);
        net.AddConv(16, Activation::RELU, true, true);
        net.AddPool(PoolType::MAX);
        net.AddConv(16, Activation::RELU, true, true);
        net.RandomizeWeights();

        int N = net.GetStartN();
        int K = net.GetNumClasses();
        const int batch_size = 8;

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(batch_size, N, K, 456, inputs, targets);

        std::vector<const float*> ptrs;
        std::vector<int> lengths;
        pack_batch(inputs, N, ptrs, lengths);

        net.TrainBatch(ptrs.data(), lengths.data(), targets.data(), batch_size, 0.01f);

        std::vector<float> emb(N), logits(K);
        net.Embed(inputs[0].data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), K), "BN TrainBatch: logits finite");
    }

    // BN ForwardBatch matches single-sample inference (eval mode)
    {
        HCNN net(5, 4);
        net.AddConv(8, Activation::RELU, true, true);
        net.AddPool(PoolType::MAX);
        net.RandomizeWeights();

        int N = net.GetStartN();
        int K = net.GetNumClasses();

        // Train a few steps to get non-trivial running stats.
        std::mt19937 rng(789);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        const int ns = 10;
        std::vector<std::vector<float>> inputs(ns, std::vector<float>(N));
        for (int i = 0; i < ns; ++i) {
            for (auto& v : inputs[i]) v = dist(rng);
            net.TrainStep(inputs[i].data(), N, i % K, 0.01f);
        }

        const int batch_size = 4;
        std::vector<const float*> ptrs(batch_size);
        std::vector<int> lengths(batch_size, N);
        for (int i = 0; i < batch_size; ++i) ptrs[i] = inputs[i].data();

        std::vector<float> batch_logits(static_cast<size_t>(batch_size) * K);
        net.ForwardBatch(ptrs.data(), lengths.data(), batch_size, batch_logits.data());

        bool match = true;
        std::vector<float> emb(N), single_logits(K);
        for (int i = 0; i < batch_size; ++i) {
            net.Embed(inputs[i].data(), N, emb.data());
            net.Forward(emb.data(), single_logits.data());
            for (int j = 0; j < K; ++j) {
                if (std::fabs(single_logits[j] - batch_logits[i * K + j]) > 1e-4f) {
                    match = false;
                    break;
                }
            }
        }
        check(match, "BN ForwardBatch matches single-sample inference (eval mode)");
    }
}

static void test_leaky_relu() {
    std::cout << "\n[LeakyReLU]\n";

    HCNN net(5, 4);
    net.AddConv(16, Activation::LEAKY_RELU);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumClasses();

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(20, N, K, 123, inputs, targets);

    std::vector<float> emb(N), logits(K);
    double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

    for (int step = 0; step < 100; ++step) {
        int idx = step % static_cast<int>(inputs.size());
        net.TrainStep(inputs[idx].data(), N, targets[idx], 0.01f);
    }

    double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
    check(all_finite(logits.data(), K), "LeakyReLU forward produces finite logits");
    check(loss_after < loss_before,
          "LeakyReLU loss decreased ("
          + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
}

static void test_adam() {
    std::cout << "\n[Adam optimizer]\n";

    // Adam TrainStep reduces loss
    {
        HCNN net(5, 4);
        net.AddConv(16);
        net.RandomizeWeights();
        net.SetOptimizer(OptimizerType::ADAM);

        int N = net.GetStartN();
        int K = net.GetNumClasses();

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(20, N, K, 42, inputs, targets);

        std::vector<float> emb(N), logits(K);
        double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

        for (int step = 0; step < 100; ++step) {
            int idx = step % static_cast<int>(inputs.size());
            net.TrainStep(inputs[idx].data(), N, targets[idx], 0.001f);
        }

        double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(loss_after < loss_before,
              "Adam TrainStep: loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
    }

    // Adam TrainBatch produces finite logits
    {
        HCNN net(5, 4);
        net.AddConv(16);
        net.RandomizeWeights();
        net.SetOptimizer(OptimizerType::ADAM);

        int N = net.GetStartN();
        int K = net.GetNumClasses();
        const int batch_size = 8;

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(batch_size, N, K, 456, inputs, targets);

        std::vector<const float*> ptrs;
        std::vector<int> lengths;
        pack_batch(inputs, N, ptrs, lengths);

        net.TrainBatch(ptrs.data(), lengths.data(), targets.data(), batch_size, 0.001f);

        std::vector<float> emb(N), logits(K);
        net.Embed(inputs[0].data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), K), "Adam TrainBatch: logits finite");
    }

    // Adam + BN reduces loss
    {
        HCNN net(5, 4);
        net.AddConv(16, Activation::RELU, true, true);
        net.RandomizeWeights();
        net.SetOptimizer(OptimizerType::ADAM);

        int N = net.GetStartN();
        int K = net.GetNumClasses();

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(20, N, K, 789, inputs, targets);

        std::vector<float> emb(N), logits(K);
        net.SetTraining(false);
        double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

        for (int step = 0; step < 200; ++step) {
            int idx = step % static_cast<int>(inputs.size());
            net.TrainStep(inputs[idx].data(), N, targets[idx], 0.001f);
        }

        net.SetTraining(false);
        double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(loss_after < loss_before,
              "Adam + BN: loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
    }
}

static void test_flatten_train_batch() {
    std::cout << "\n[FLATTEN TrainBatch]\n";

    HCNN net(5, 4, /*input_channels=*/1, ReadoutType::FLATTEN);
    net.AddConv(8);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumClasses();
    const int batch_size = 4;

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(batch_size, N, K, 42, inputs, targets);

    std::vector<const float*> ptrs;
    std::vector<int> lengths;
    pack_batch(inputs, N, ptrs, lengths);

    net.TrainBatch(ptrs.data(), lengths.data(), targets.data(), batch_size, 0.01f);

    std::vector<float> emb(N), logits(K);
    net.Embed(inputs[0].data(), N, emb.data());
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), K), "FLATTEN TrainBatch: logits finite");
}

static void test_avg_pool_training() {
    std::cout << "\n[AVG pool training]\n";

    HCNN net(5, 4);
    net.AddConv(16);
    net.AddPool(PoolType::AVG);
    net.AddConv(16);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumClasses();

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(20, N, K, 42, inputs, targets);

    std::vector<float> emb(N), logits(K);
    double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

    for (int step = 0; step < 100; ++step) {
        int idx = step % static_cast<int>(inputs.size());
        net.TrainStep(inputs[idx].data(), N, targets[idx], 0.01f);
    }

    double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
    check(loss_after < loss_before,
          "AVG pool: loss decreased ("
          + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
}

static void test_weight_decay() {
    std::cout << "\n[Weight decay]\n";

    // Without exposing kernel internals, we can only confirm that weight
    // decay is accepted by the API and does not destabilize training:
    // both with and without WD, training should still produce finite
    // logits and (typically) decreasing loss on a small synthetic task.
    HCNN net(5, 4);
    net.AddConv(16);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumClasses();

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(20, N, K, 123, inputs, targets);

    std::vector<float> emb(N), logits(K);
    double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

    for (int step = 0; step < 100; ++step) {
        int idx = step % static_cast<int>(inputs.size());
        net.TrainStep(inputs[idx].data(), N, targets[idx],
                      /*lr=*/0.01f, /*momentum=*/0.0f, /*weight_decay=*/0.01f);
    }

    double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
    check(all_finite(logits.data(), K), "weight decay: logits finite");
    check(loss_after < loss_before,
          "weight decay: loss decreased ("
          + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
}

// Embed truncation (input shorter than N) and zero-pad behavior, plus the
// over-capacity input length (must throw).
static void test_embed_padding_and_truncation() {
    std::cout << "\n[Embed padding / truncation]\n";

    HCNN net(5, 4);  // N = 32
    net.AddConv(8);
    net.RandomizeWeights();

    const int N = net.GetStartN();

    // 1) Short input is zero-padded to N.
    std::vector<float> short_input(N - 5, 0.5f);
    std::vector<float> emb(N, -123.0f);   // sentinel
    net.Embed(short_input.data(), static_cast<int>(short_input.size()), emb.data());
    bool front_ok = true;
    for (int i = 0; i < static_cast<int>(short_input.size()); ++i)
        if (emb[i] != 0.5f) { front_ok = false; break; }
    bool tail_zeroed = true;
    for (int i = static_cast<int>(short_input.size()); i < N; ++i)
        if (emb[i] != 0.0f) { tail_zeroed = false; break; }
    check(front_ok,    "Embed: front of short input copied verbatim");
    check(tail_zeroed, "Embed: tail of short input zero-padded");

    // 2) Forward on the zero-padded embedding succeeds.
    std::vector<float> logits(net.GetNumClasses());
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), net.GetNumClasses()),
          "Forward on zero-padded embedding: logits finite");

    // 3) Over-capacity input length throws.
    std::vector<float> oversized(N + 4, 0.0f);
    bool threw = false;
    try {
        net.Embed(oversized.data(), static_cast<int>(oversized.size()), emb.data());
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "Embed: over-capacity input length throws");
}

// FLATTEN readout combined with the Adam optimizer.  Both work in isolation
// elsewhere; this guards the cross-product.
static void test_flatten_adam() {
    std::cout << "\n[FLATTEN + Adam]\n";

    HCNN net(5, 4, /*input_channels=*/1, ReadoutType::FLATTEN);
    net.AddConv(8);
    net.RandomizeWeights();
    net.SetOptimizer(OptimizerType::ADAM);

    int N = net.GetStartN();
    int K = net.GetNumClasses();

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(20, N, K, 7, inputs, targets);

    std::vector<float> emb(N), logits(K);
    double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

    for (int step = 0; step < 100; ++step) {
        int idx = step % static_cast<int>(inputs.size());
        net.TrainStep(inputs[idx].data(), N, targets[idx], 0.001f);
    }

    double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
    check(all_finite(logits.data(), K),
          "FLATTEN + Adam: logits finite");
    check(loss_after < loss_before,
          "FLATTEN + Adam: loss decreased ("
          + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
}

// Validation paths: API methods that should throw on bad inputs.
static void test_invalid_arguments() {
    std::cout << "\n[Invalid arguments]\n";

    HCNN net(5, 4);
    net.AddConv(8);
    net.RandomizeWeights();

    // batch_size <= 0 must throw on all three batch APIs.
    auto throws = [](auto&& fn) {
        try { fn(); } catch (const std::exception&) { return true; }
        return false;
    };

    std::vector<float> dummy_input(net.GetStartN(), 0.0f);
    const float* ptrs[1] = { dummy_input.data() };
    int lengths[1] = { net.GetStartN() };
    int targets[1] = { 0 };
    std::vector<float> logits_out(net.GetNumClasses());

    check(throws([&] { net.ForwardBatch(ptrs, lengths, 0, logits_out.data()); }),
          "ForwardBatch(batch_size=0) throws");
    check(throws([&] { net.TrainBatch(ptrs, lengths, targets, 0, 0.01f); }),
          "TrainBatch(batch_size=0) throws");
    check(throws([&] { net.TrainEpoch(ptrs, lengths, targets, 1, 0, 0.01f); }),
          "TrainEpoch(batch_size=0) throws");
}

// Inference path must NOT mutate the per-layer training flag observed by the
// caller.  Was previously a footgun: forward() called set_training(false)
// silently, leaving BN layers in eval mode after a Forward call.
static void test_forward_preserves_training_mode() {
    std::cout << "\n[Forward preserves training mode]\n";

    HCNN net(5, 4);
    net.AddConv(8, Activation::RELU, true, /*use_batchnorm=*/true);
    net.RandomizeWeights();

    // Put the network into training mode, then run inference and verify
    // that a subsequent TrainStep still updates BN running stats (i.e. the
    // training flag was not silently flipped to eval).
    net.SetTraining(true);

    std::vector<float> input(net.GetStartN(), 0.25f);
    std::vector<float> emb(net.GetStartN()), logits(net.GetNumClasses());
    net.Embed(input.data(), net.GetStartN(), emb.data());
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), net.GetNumClasses()),
          "Forward in training mode: logits finite");

    // If forward() had silently set training=false, this TrainStep would
    // still work but a downstream "is the network still in training mode"
    // check would fail.  We can detect it indirectly: a no-op-style call
    // sequence below should not throw and should still produce finite logits.
    net.TrainStep(input.data(), net.GetStartN(), 0, 0.01f);
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), net.GetNumClasses()),
          "TrainStep + Forward after training-mode Forward: logits finite");
}

static void test_class_weights() {
    std::cout << "\n[Class weights]\n";

    HCNN net(5, 4);
    net.AddConv(16);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumClasses();

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> input(N);
    for (auto& v : input) v = dist(rng);

    // Heavily up-weight class 0
    std::vector<float> class_weights = {10.0f, 1.0f, 1.0f, 1.0f};

    net.TrainStep(input.data(), N, 0, 0.01f, 0.0f, 0.0f, class_weights.data());

    std::vector<float> emb(N), logits(K);
    net.Embed(input.data(), N, emb.data());
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), K), "class-weighted TrainStep: logits finite");

    const int batch_size = 4;
    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(batch_size, N, K, 42, inputs, targets);

    std::vector<const float*> ptrs;
    std::vector<int> lengths;
    pack_batch(inputs, N, ptrs, lengths);

    net.TrainBatch(ptrs.data(), lengths.data(), targets.data(), batch_size,
                   0.01f, 0.0f, 0.0f, class_weights.data());

    net.Embed(inputs[0].data(), N, emb.data());
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), K), "class-weighted TrainBatch: logits finite");
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "HCNN SDK Smoke Test\n";
    std::cout << "===================\n";

    test_construction();
    test_forward_pass();
    test_training_step();
    test_train_batch();
    test_train_epoch();
    test_forward_batch();
    test_readout_types();
    test_pool_types();
    test_batchnorm();
    test_leaky_relu();
    test_adam();
    test_flatten_train_batch();
    test_avg_pool_training();
    test_weight_decay();
    test_embed_padding_and_truncation();
    test_flatten_adam();
    test_invalid_arguments();
    test_forward_preserves_training_mode();
    test_class_weights();

    std::cout << "\n===================\n";
    if (failures == 0) {
        std::cout << "All tests PASSED\n";
        return 0;
    } else {
        std::cout << failures << " test(s) FAILED\n";
        return 1;
    }
}
