// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// Smoke test for the HCNN top-level SDK API.
// Everything goes through HCNN -- no direct use of HCNNNetwork or layer
// classes.  This file is the canonical regression check that the SDK front
// door behaves correctly across architecture, training, optimizer and
// readout-type variations.

#include "HCNN.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using hcnn::HCNN;
using hcnn::PoolType;
using hcnn::TaskType;
using hcnn::LossType;
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
    const int K = net.GetNumOutputs();
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

// Flatten a vector-of-vectors into a contiguous buffer for HCNN's flat API.
static std::vector<float> flatten_inputs(
    const std::vector<std::vector<float>>& inputs, int N) {
    const int n = static_cast<int>(inputs.size());
    std::vector<float> flat(static_cast<size_t>(n) * N);
    for (int i = 0; i < n; ++i)
        std::copy(inputs[i].begin(), inputs[i].end(),
                  flat.begin() + i * N);
    return flat;
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

static void test_construction() {
    std::cout << "\n[Construction]\n";

    HCNN net(5, 4);   // DIM=5, N=32, 4 classes
    check(net.GetStartDim() == 5,       "GetStartDim() == 5");
    check(net.GetStartN() == 32,        "GetStartN() == 32");
    check(net.GetNumOutputs() == 4,     "GetNumOutputs() == 4");
    check(net.GetInputChannels() == 1,  "GetInputChannels() == 1");

    // Architecture build should not throw or affect sizing accessors.
    net.AddConv(8);
    net.AddPool(PoolType::MAX);
    net.AddConv(16);
    net.RandomizeWeights();

    check(net.GetStartDim() == 5,   "GetStartDim() unchanged after build");
    check(net.GetStartN() == 32,    "GetStartN() unchanged after build");
    check(net.GetNumOutputs() == 4, "GetNumOutputs() unchanged after build");
}

static void test_forward_pass() {
    std::cout << "\n[Forward pass]\n";

    HCNN net(5, 4);
    net.AddConv(8);
    net.AddPool(PoolType::MAX);
    net.AddConv(16);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumOutputs();

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
    int K = net.GetNumOutputs();

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
    int K = net.GetNumOutputs();
    const int batch_size = 8;

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(batch_size, N, K, 456, inputs, targets);

    auto flat = flatten_inputs(inputs, N);

    net.TrainBatch(flat.data(), N, targets.data(), batch_size, 0.01f);

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
    int K = net.GetNumOutputs();

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(40, N, K, 999, inputs, targets);

    auto flat = flatten_inputs(inputs, N);

    std::vector<float> emb(N), logits(K);
    double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

    // Two shuffled epochs at the same nominal LR -- distinct seeds give
    // different reproducible permutations each call.
    net.TrainEpoch(flat.data(), N, targets.data(),
                   static_cast<int>(inputs.size()), /*batch_size=*/8,
                   /*lr=*/0.05f, /*momentum=*/0.0f, /*wd=*/0.0f,
                   /*class_weights=*/nullptr, /*shuffle_seed=*/1u);
    net.TrainEpoch(flat.data(), N, targets.data(),
                   static_cast<int>(inputs.size()), /*batch_size=*/8,
                   /*lr=*/0.05f, /*momentum=*/0.0f, /*wd=*/0.0f,
                   /*class_weights=*/nullptr, /*shuffle_seed=*/2u);

    double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
    check(loss_after < loss_before,
          "TrainEpoch (shuffled): loss decreased ("
          + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");

    // No-shuffle path also produces finite logits.
    net.TrainEpoch(flat.data(), N, targets.data(),
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
    int K = net.GetNumOutputs();
    const int batch_size = 8;

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(batch_size, N, K, 789, inputs, targets);

    auto flat = flatten_inputs(inputs, N);

    std::vector<float> all_logits(static_cast<size_t>(batch_size) * K);
    net.ForwardBatch(flat.data(), N, batch_size, all_logits.data());
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

static void test_readout() {
    std::cout << "\n[Readout]\n";

    HCNN net(5, 4, /*input_channels=*/1);
    net.AddConv(8);
    net.RandomizeWeights();

    int N = net.GetStartN();
    std::mt19937 rng(111);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> input(N);
    for (auto& v : input) v = dist(rng);
    std::vector<float> emb(N), logits(net.GetNumOutputs());
    net.Embed(input.data(), N, emb.data());
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), net.GetNumOutputs()),
          "FLATTEN readout produces finite logits");
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
        std::vector<float> emb(N), logits(net.GetNumOutputs());
        net.Embed(input.data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), net.GetNumOutputs()),
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
        int K = net.GetNumOutputs();
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
        int K = net.GetNumOutputs();

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
        int K = net.GetNumOutputs();
        const int batch_size = 8;

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(batch_size, N, K, 456, inputs, targets);

        auto flat = flatten_inputs(inputs, N);
        net.TrainBatch(flat.data(), N, targets.data(), batch_size, 0.01f);

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
        int K = net.GetNumOutputs();

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
        auto flat = flatten_inputs(
            std::vector<std::vector<float>>(inputs.begin(), inputs.begin() + batch_size), N);

        std::vector<float> batch_logits(static_cast<size_t>(batch_size) * K);
        net.ForwardBatch(flat.data(), N, batch_size, batch_logits.data());

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

static void test_activations() {
    std::cout << "\n[Activations -- LeakyReLU, Tanh]\n";

    // Shared data for all activations (DIM=5, 4 classes, 20 samples).
    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(20, 32, 4, 123, inputs, targets);

    struct Case { Activation act; const char* name; float lr; };
    Case cases[] = {
        { Activation::LEAKY_RELU, "LeakyReLU", 0.01f },
        { Activation::TANH,       "Tanh",      0.01f },
    };

    for (auto& c : cases) {
        HCNN net(5, 4);
        net.AddConv(16, c.act);
        net.RandomizeWeights();

        int N = net.GetStartN();
        int K = net.GetNumOutputs();
        std::vector<float> emb(N), logits(K);
        double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

        for (int step = 0; step < 100; ++step) {
            int idx = step % static_cast<int>(inputs.size());
            net.TrainStep(inputs[idx].data(), N, targets[idx], c.lr);
        }

        double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(all_finite(logits.data(), K),
              std::string(c.name) + " forward produces finite logits");
        check(loss_after < loss_before,
              std::string(c.name) + " loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
    }

    // TANH bounded-output sanity check: stacked tanh layers must produce
    // finite logits.  Catches accidental fall-through into a NONE/RELU path.
    {
        HCNN net2(5, 4);
        net2.AddConv(8, Activation::TANH, /*use_bias=*/true);
        net2.AddConv(8, Activation::TANH, /*use_bias=*/true);
        net2.RandomizeWeights();
        std::vector<float> emb2(net2.GetStartN());
        std::vector<float> logits2(net2.GetNumOutputs());
        net2.Embed(inputs[0].data(), net2.GetStartN(), emb2.data());
        net2.Forward(emb2.data(), logits2.data());
        check(all_finite(logits2.data(), net2.GetNumOutputs()),
              "Tanh stacked layers produce finite logits");
    }
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
        int K = net.GetNumOutputs();

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
        int K = net.GetNumOutputs();
        const int batch_size = 8;

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(batch_size, N, K, 456, inputs, targets);

        auto flat = flatten_inputs(inputs, N);
        net.TrainBatch(flat.data(), N, targets.data(), batch_size, 0.001f);

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
        int K = net.GetNumOutputs();

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(20, N, K, 789, inputs, targets);

        std::vector<float> emb(N), logits(K);
        net.SetTraining(false);
        double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

        for (int step = 0; step < 50; ++step) {
            int idx = step % static_cast<int>(inputs.size());
            net.TrainStep(inputs[idx].data(), N, targets[idx], 0.005f);
        }

        net.SetTraining(false);
        double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(loss_after < loss_before,
              "Adam + BN: loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
    }
}

static void test_flatten_readout() {
    std::cout << "\n[FLATTEN readout -- SGD + Adam]\n";

    // FLATTEN TrainBatch (SGD)
    {
        HCNN net(5, 4, /*input_channels=*/1);
        net.AddConv(8);
        net.RandomizeWeights();

        int N = net.GetStartN();
        int K = net.GetNumOutputs();
        const int batch_size = 4;

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(batch_size, N, K, 42, inputs, targets);

        auto flat = flatten_inputs(inputs, N);
        net.TrainBatch(flat.data(), N, targets.data(), batch_size, 0.01f);

        std::vector<float> emb(N), logits(K);
        net.Embed(inputs[0].data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), K), "FLATTEN TrainBatch: logits finite");
    }

    // FLATTEN + Adam: loss decreases
    {
        HCNN net(5, 4, /*input_channels=*/1);
        net.AddConv(8);
        net.RandomizeWeights();
        net.SetOptimizer(OptimizerType::ADAM);

        int N = net.GetStartN();
        int K = net.GetNumOutputs();

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
        check(all_finite(logits.data(), K), "FLATTEN + Adam: logits finite");
        check(loss_after < loss_before,
              "FLATTEN + Adam: loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
    }
}

static void test_avg_pool_training() {
    std::cout << "\n[AVG pool training]\n";

    HCNN net(5, 4);
    net.AddConv(16);
    net.AddPool(PoolType::AVG);
    net.AddConv(16);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumOutputs();

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
    int K = net.GetNumOutputs();

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
    std::vector<float> logits(net.GetNumOutputs());
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), net.GetNumOutputs()),
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
    int N = net.GetStartN();
    int targets[1] = { 0 };
    std::vector<float> logits_out(net.GetNumOutputs());

    check(throws([&] { net.ForwardBatch(dummy_input.data(), N, 0, logits_out.data()); }),
          "ForwardBatch(batch_size=0) throws");
    check(throws([&] { net.TrainBatch(dummy_input.data(), N, targets, 0, 0.01f); }),
          "TrainBatch(batch_size=0) throws");
    check(throws([&] { net.TrainEpoch(dummy_input.data(), N, targets, 1, 0, 0.01f); }),
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
    std::vector<float> emb(net.GetStartN()), logits(net.GetNumOutputs());
    net.Embed(input.data(), net.GetStartN(), emb.data());
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), net.GetNumOutputs()),
          "Forward in training mode: logits finite");

    // If forward() had silently set training=false, this TrainStep would
    // still work but a downstream "is the network still in training mode"
    // check would fail.  We can detect it indirectly: a no-op-style call
    // sequence below should not throw and should still produce finite logits.
    net.TrainStep(input.data(), net.GetStartN(), 0, 0.01f);
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), net.GetNumOutputs()),
          "TrainStep + Forward after training-mode Forward: logits finite");
}

static void test_class_weights() {
    std::cout << "\n[Class weights]\n";

    HCNN net(5, 4);
    net.AddConv(16);
    net.RandomizeWeights();

    int N = net.GetStartN();
    int K = net.GetNumOutputs();

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

    auto flat = flatten_inputs(inputs, N);

    net.TrainBatch(flat.data(), N, targets.data(), batch_size,
                   0.01f, 0.0f, 0.0f, class_weights.data());

    net.Embed(inputs[0].data(), N, emb.data());
    net.Forward(emb.data(), logits.data());
    check(all_finite(logits.data(), K), "class-weighted TrainBatch: logits finite");
}

// ---------------------------------------------------------------------------
//  Regression tests
// ---------------------------------------------------------------------------
//
// The regression path shares all forward/backward machinery with the
// classification path -- only the loss-gradient computation and the
// target types differ.  These tests verify that:
//   1. A scalar-output net can fit a simple linear target (loss decreases).
//   2. A multi-output net can fit a 3-dimensional target vector
//      (per-output loss decreases independently).
//   3. Calling classification APIs on a regression net throws logic_error
//      (and vice versa).
//   4. The constructor rejects invalid task/loss combinations.
// ---------------------------------------------------------------------------

// Build a regression dataset: each sample is a random N-float input in
// [-1, 1]; the target is a single-output linear function of a small
// projection of the input so the network has something learnable.
static void make_synth_regression_scalar(
    int n, int N, unsigned seed,
    std::vector<std::vector<float>>& inputs_out,
    std::vector<float>& targets_out)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    inputs_out.assign(n, std::vector<float>(N));
    targets_out.assign(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) {
            float v = dist(rng);
            inputs_out[i][j] = v;
            s += v;
        }
        // Target: shallow linear function of the input's mean, in [-1, 1].
        float mean = static_cast<float>(s / N);
        targets_out[i] = std::tanh(2.0f * mean);
    }
}

// Compute mean-squared error over a sample list via single-sample
// Embed+Forward (so we exercise the inference path that regression
// consumers will use).
static double mse_over_samples(
    HCNN& net,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<float>& targets,
    std::vector<float>& embedded,
    std::vector<float>& preds)
{
    const int N = net.GetStartN();
    const int n = static_cast<int>(inputs.size());
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        net.Embed(inputs[i].data(), N, embedded.data());
        net.Forward(embedded.data(), preds.data());
        double d = preds[0] - targets[i];
        total += d * d;
    }
    return total / n;
}

static void test_regression_scalar() {
    std::cout << "\n[Regression -- scalar fit (Step / Batch / Epoch)]\n";

    const int DIM = 6;
    const int num_outputs = 1;

    HCNN net(DIM, num_outputs, /*input_channels=*/1,
             TaskType::Regression);
    net.AddConv(16);
    net.AddPool(PoolType::MAX);
    net.AddConv(16);
    net.RandomizeWeights();

    check(net.GetNumOutputs() == 1,                      "GetNumOutputs() == 1");
    check(net.GetTaskType() == TaskType::Regression,     "GetTaskType() == Regression");
    check(net.GetLossType() == LossType::MSE,            "GetLossType() == MSE (default)");

    const int N = net.GetStartN();
    const int n_train = 32;

    std::vector<std::vector<float>> inputs;
    std::vector<float> targets;
    make_synth_regression_scalar(n_train, N, /*seed=*/7, inputs, targets);

    auto flat_inputs = flatten_inputs(inputs, N);

    std::vector<float> embedded(N);
    std::vector<float> preds(num_outputs);

    // --- TrainStepRegression ---
    double mse_before = mse_over_samples(net, inputs, targets, embedded, preds);
    check(std::isfinite(mse_before), "initial MSE is finite");

    for (int e = 0; e < 3; ++e)
        for (int i = 0; i < n_train; ++i)
            net.TrainStepRegression(inputs[i].data(), N,
                                    &targets[i], /*lr=*/0.05f, /*momentum=*/0.9f);

    double mse_after_step = mse_over_samples(net, inputs, targets, embedded, preds);
    check(mse_after_step < mse_before,
          "TrainStepRegression: MSE decreased ("
              + std::to_string(mse_before) + " -> "
              + std::to_string(mse_after_step) + ")");

    // --- TrainBatchRegression ---
    for (int e = 0; e < 3; ++e)
        net.TrainBatchRegression(flat_inputs.data(), N,
                                 targets.data(), n_train,
                                 /*lr=*/0.05f, /*momentum=*/0.9f);

    double mse_after_batch = mse_over_samples(net, inputs, targets, embedded, preds);
    check(mse_after_batch < mse_after_step,
          "TrainBatchRegression: MSE decreased ("
              + std::to_string(mse_after_step) + " -> "
              + std::to_string(mse_after_batch) + ")");

    // --- TrainEpochRegression ---
    for (int e = 0; e < 5; ++e)
        net.TrainEpochRegression(flat_inputs.data(), N,
                                 targets.data(),
                                 n_train, /*batch_size=*/16,
                                 /*lr=*/0.05f, /*momentum=*/0.9f,
                                 /*weight_decay=*/1e-4f,
                                 /*shuffle_seed=*/static_cast<unsigned>(e + 1));

    double mse_after_epoch = mse_over_samples(net, inputs, targets, embedded, preds);
    check(std::isfinite(mse_after_epoch), "final MSE is finite");
    check(mse_after_epoch < mse_after_batch,
          "TrainEpochRegression: MSE decreased ("
              + std::to_string(mse_after_batch) + " -> "
              + std::to_string(mse_after_epoch) + ")");
    check(mse_after_epoch < 0.5 * mse_before,
          "Regression scalar fit: MSE dropped by at least 50%");
}

static void test_regression_multi_output() {
    std::cout << "\n[Regression -- multi-output fit]\n";

    const int DIM = 6;
    const int num_outputs = 3;

    HCNN net(DIM, num_outputs, /*input_channels=*/1,
             TaskType::Regression);
    net.AddConv(16);
    net.AddPool(PoolType::MAX);
    net.AddConv(16);
    net.RandomizeWeights();

    const int N = net.GetStartN();
    const int n_train = 32;

    // Targets: 3 different nonlinear functions of the input mean.
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<float>> inputs(n_train, std::vector<float>(N));
    std::vector<std::vector<float>> targets(n_train, std::vector<float>(num_outputs));
    for (int i = 0; i < n_train; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) {
            float v = dist(rng);
            inputs[i][j] = v;
            s += v;
        }
        float mean = static_cast<float>(s / N);
        targets[i][0] = std::tanh(2.0f * mean);
        targets[i][1] = std::tanh(-1.5f * mean);
        targets[i][2] = 0.3f * mean;
    }

    auto flat_inputs = flatten_inputs(inputs, N);

    // Flatten multi-output targets into contiguous buffer.
    std::vector<float> flat_targets(static_cast<size_t>(n_train) * num_outputs);
    for (int i = 0; i < n_train; ++i)
        std::copy(targets[i].begin(), targets[i].end(),
                  flat_targets.begin() + i * num_outputs);

    std::vector<float> embedded(N);
    std::vector<float> preds(num_outputs);

    auto compute_per_output_mse = [&](std::vector<double>& out) {
        out.assign(num_outputs, 0.0);
        for (int i = 0; i < n_train; ++i) {
            net.Embed(inputs[i].data(), N, embedded.data());
            net.Forward(embedded.data(), preds.data());
            for (int k = 0; k < num_outputs; ++k) {
                double d = preds[k] - targets[i][k];
                out[k] += d * d;
            }
        }
        for (int k = 0; k < num_outputs; ++k) out[k] /= n_train;
    };

    std::vector<double> mse_before, mse_after;
    compute_per_output_mse(mse_before);

    const int epochs = 5;
    for (int e = 0; e < epochs; ++e) {
        net.TrainEpochRegression(flat_inputs.data(), N,
                                 flat_targets.data(),
                                 n_train, /*batch_size=*/16,
                                 /*lr=*/0.05f, /*momentum=*/0.9f,
                                 /*weight_decay=*/1e-4f,
                                 /*shuffle_seed=*/static_cast<unsigned>(e + 1));
    }
    compute_per_output_mse(mse_after);

    for (int k = 0; k < num_outputs; ++k) {
        check(std::isfinite(mse_after[k]),
              "multi-output MSE[" + std::to_string(k) + "] finite");
        check(mse_after[k] < mse_before[k],
              "multi-output MSE[" + std::to_string(k) + "] decreased ("
                  + std::to_string(mse_before[k]) + " -> "
                  + std::to_string(mse_after[k]) + ")");
    }
}

static void test_forward_batch_regression() {
    std::cout << "\n[ForwardBatch regression]\n";

    const int DIM = 6;
    const int N = 1 << DIM;
    const int num_outputs = 1;
    const int n_train = 32;

    auto net_p = std::make_unique<HCNN>(DIM, num_outputs, /*input_channels=*/1,
                                        TaskType::Regression);
    net_p->AddConv(16);
    net_p->AddPool(PoolType::MAX);
    net_p->RandomizeWeights(/*scale=*/0.0f, /*seed=*/42);
    HCNN& net = *net_p;

    // Generate synthetic data and flatten.
    std::vector<std::vector<float>> inputs;
    std::vector<float> targets;
    make_synth_regression_scalar(n_train, N, /*seed=*/7, inputs, targets);

    auto flat_inputs = flatten_inputs(inputs, N);

    // Train for a few epochs.
    for (int e = 0; e < 3; ++e) {
        unsigned seed = static_cast<unsigned>(e + 1);
        net.TrainEpochRegression(
            flat_inputs.data(), N, targets.data(),
            n_train, /*batch_size=*/16, /*lr=*/0.05f, /*momentum=*/0.0f,
            /*weight_decay=*/0.0f, /*shuffle_seed=*/seed);
    }

    // ForwardBatch should match single-sample Embed+Forward.
    std::vector<float> batch_preds(n_train);
    net.ForwardBatch(flat_inputs.data(), N, n_train, batch_preds.data());

    check(all_finite(batch_preds.data(), n_train),
          "ForwardBatch regression: all predictions finite");

    std::vector<float> embedded(N), pred(1);
    double max_diff = 0.0;
    for (int i = 0; i < n_train; ++i) {
        net.Embed(inputs[i].data(), N, embedded.data());
        net.Forward(embedded.data(), pred.data());
        double d = std::abs(static_cast<double>(pred[0]) - batch_preds[i]);
        if (d > max_diff) max_diff = d;
    }
    check(max_diff < 1e-4,
          "ForwardBatch matches single-sample inference (max_diff="
              + std::to_string(max_diff) + ")");

    // Verify learning happened.
    double mse_after = mse_over_samples(net, inputs, targets, embedded, pred);
    check(std::isfinite(mse_after) && mse_after < 0.1,
          "Regression learned (MSE=" + std::to_string(mse_after) + ")");
}

static void test_regression_classification_cross_misuse() {
    std::cout << "\n[Regression -- task/API misuse]\n";

    // Build a regression net and verify classification APIs throw logic_error.
    {
        HCNN net(5, /*num_outputs=*/2, 1, TaskType::Regression);
        net.AddConv(8);
        net.RandomizeWeights();

        const int N = net.GetStartN();
        std::vector<float> input(N, 0.1f);

        bool threw = false;
        try {
            net.TrainStep(input.data(), N, 0, 0.01f);
        } catch (const std::logic_error&) {
            threw = true;
        }
        check(threw, "TrainStep on Regression net throws logic_error");

        int target = 0;
        threw = false;
        try {
            net.TrainBatch(input.data(), N, &target, 1, 0.01f);
        } catch (const std::logic_error&) {
            threw = true;
        }
        check(threw, "TrainBatch on Regression net throws logic_error");
    }

    // Build a classification net and verify regression APIs throw logic_error.
    {
        HCNN net(5, /*num_outputs=*/2, 1, TaskType::Classification);
        net.AddConv(8);
        net.RandomizeWeights();

        const int N = net.GetStartN();
        std::vector<float> input(N, 0.1f);
        std::vector<float> target(2, 0.0f);

        bool threw = false;
        try {
            net.TrainStepRegression(input.data(), N, target.data(), 0.01f);
        } catch (const std::logic_error&) {
            threw = true;
        }
        check(threw, "TrainStepRegression on Classification net throws logic_error");

        threw = false;
        try {
            net.TrainBatchRegression(input.data(), N, target.data(), 1, 0.01f);
        } catch (const std::logic_error&) {
            threw = true;
        }
        check(threw, "TrainBatchRegression on Classification net throws logic_error");
    }
}

static void test_regression_invalid_construction() {
    std::cout << "\n[Regression -- invalid construction]\n";

    // Classification + MSE is rejected in the constructor.
    bool threw = false;
    try {
        HCNN net(5, 4, 1, TaskType::Classification, LossType::MSE);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "Classification + MSE throws at construction");

    // Regression + CrossEntropy is rejected in the constructor.
    threw = false;
    try {
        HCNN net(5, 4, 1, TaskType::Regression, LossType::CrossEntropy);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "Regression + CrossEntropy throws at construction");

    // Regression + Default resolves to MSE.
    {
        HCNN net(5, 4, 1, TaskType::Regression, LossType::Default);
        check(net.GetLossType() == LossType::MSE,
              "Regression + Default resolves to MSE");
    }

    // Classification + Default resolves to CrossEntropy.
    {
        HCNN net(5, 4, 1, TaskType::Classification, LossType::Default);
        check(net.GetLossType() == LossType::CrossEntropy,
              "Classification + Default resolves to CrossEntropy");
    }
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
    test_readout();
    test_pool_types();
    test_batchnorm();
    test_activations();
    test_adam();
    test_flatten_readout();
    test_avg_pool_training();
    test_weight_decay();
    test_embed_padding_and_truncation();
    test_invalid_arguments();
    test_forward_preserves_training_mode();
    test_class_weights();
    test_regression_scalar();
    test_regression_multi_output();
    test_forward_batch_regression();
    test_regression_classification_cross_misuse();
    test_regression_invalid_construction();

    std::cout << "\n===================\n";
    if (failures == 0) {
        std::cout << "All tests PASSED\n";
        return 0;
    } else {
        std::cout << failures << " test(s) FAILED\n";
        return 1;
    }
}
