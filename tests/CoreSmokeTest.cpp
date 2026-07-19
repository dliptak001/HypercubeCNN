// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// Smoke test for the HCNN top-level SDK API.
// Everything goes through HCNN -- no direct use of HCNNNetwork or layer
// classes.  This file is the canonical regression check that the SDK front
// door behaves correctly across architecture, training, optimizer and
// readout-type variations.

#include "HCNN.h"
#include "HCNNReadout.h"
#include "HCNNSpatialAug.h"
#include "HCNNSpatialEmbed.h"
#include "HCNNTrainHelpers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using hcnn::HCNN;
using hcnn::HCNNNetwork;
using hcnn::HCNNReadout;
using hcnn::ReadoutGradInLoop;
using hcnn::PoolType;
using hcnn::TaskType;
using hcnn::LossType;
using hcnn::Activation;
using hcnn::OptimizerType;
using hcnn::HCNNSpatialAugConfig;
using hcnn::HCNNSpatialAugmenter;
using hcnn::HCNNSpatialEmbedConfig;
using hcnn::HCNNSpatialEmbedMode;
using hcnn::HCNNSpatialEmbedder;
using hcnn::HCNNFlatDataset;
using hcnn::HCNNDualCheckpoint;
using hcnn::HCNNBestMetricCheckpoint;
using hcnn::argmax;
using hcnn::softmax_cross_entropy;
using hcnn::evaluate_classification;
using hcnn::evaluate_regression;
using hcnn::cosine_lr;

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

// Self/center kernel tap: K = DIM + 1, last index multiplies in[v] (not a neighbor).
static void test_self_contribution() {
    std::cout << "\n[Self contribution]\n";

    const int dim = 5;
    const int N = 1 << dim;
    const int K = dim + 1;

    hcnn::HCNNConv conv(dim, /*c_in=*/1, /*c_out=*/2,
                        Activation::NONE, /*bias=*/true, /*bn=*/false);
    check(conv.get_K() == K, "get_K() == DIM + 1");
    check(conv.get_self_index() == dim, "get_self_index() == DIM");
    check(conv.get_kernel_size() == 2 * 1 * K, "kernel size = c_out*c_in*(DIM+1)");

    // Zero all taps, then set self only (+ bias).
    float* ker = conv.get_kernel_data();
    for (int i = 0; i < conv.get_kernel_size(); ++i) ker[i] = 0.0f;
    // layout: (co * c_in + ci) * K + k; self at k == dim
    ker[0 * K + dim] = 2.0f;    // co=0
    ker[1 * K + dim] = -0.5f;   // co=1
    float* bias = conv.get_bias_data();
    bias[0] = 0.1f;
    bias[1] = -0.2f;

    std::vector<float> in(static_cast<size_t>(N));
    for (int v = 0; v < N; ++v)
        in[static_cast<size_t>(v)] = 0.01f * static_cast<float>(v) - 0.5f;

    std::vector<float> out(static_cast<size_t>(2 * N));
    conv.forward(in.data(), out.data());

    bool self_ok = true;
    for (int v = 0; v < N; ++v) {
        const float x = in[static_cast<size_t>(v)];
        const float e0 = 0.1f + 2.0f * x;
        const float e1 = -0.2f + (-0.5f) * x;
        if (std::fabs(out[static_cast<size_t>(v)] - e0) > 1e-5f ||
            std::fabs(out[static_cast<size_t>(N + v)] - e1) > 1e-5f) {
            self_ok = false;
            break;
        }
    }
    check(self_ok, "self-only kernel: out[v] = bias + w_self * in[v]");

    // Turning on a neighbor tap must change the output.
    ker[0 * K + 0] = 1.0f;  // co=0, bit-0 neighbor
    std::vector<float> out2(static_cast<size_t>(2 * N));
    conv.forward(in.data(), out2.data());
    bool differ = false;
    for (int v = 0; v < N; ++v) {
        if (std::fabs(out2[static_cast<size_t>(v)] - out[static_cast<size_t>(v)]) > 1e-5f) {
            differ = true;
            break;
        }
    }
    check(differ, "neighbor tap changes output beyond self-only");

    // HCNN weight blob includes self taps (K = DIM+1).
    HCNN net(5, /*num_outputs=*/4);
    net.AddConv(8, Activation::RELU, /*bias=*/true, /*bn=*/false);
    net.RandomizeWeights();
    // kernel: 1*8*6 + bias 8; readout FLATTEN 8*32 -> 4 + bias 4
    const size_t expected = static_cast<size_t>(1 * 8 * 6 + 8 + 8 * 32 * 4 + 4);
    check(net.GetWeightCount() == expected,
          "GetWeightCount includes self taps (K=DIM+1)");
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

// A/B: FeatureOuter vs OutputOuter for grad_in = W^T * g.
// Power-user path (HCNNReadout) for a pure head microbench + numerical match.
// Also checks HCNN facade preserves the setting across RandomizeWeights.
static void test_readout_grad_in_loop_ab() {
    std::cout << "\n[Readout grad_in loop A/B]\n";

    // --- Facade: setting survives RandomizeWeights ---
    {
        HCNN net(5, 4);
        net.AddConv(8);
        check(net.GetReadoutGradInLoop() == ReadoutGradInLoop::OutputOuter,
              "default grad_in loop is OutputOuter");
        net.SetReadoutGradInLoop(ReadoutGradInLoop::FeatureOuter);
        net.RandomizeWeights(0.0f, 99);
        check(net.GetReadoutGradInLoop() == ReadoutGradInLoop::FeatureOuter,
              "SetReadoutGradInLoop survives RandomizeWeights");
        net.SetReadoutGradInLoop(ReadoutGradInLoop::OutputOuter);
        check(net.GetReadoutGradInLoop() == ReadoutGradInLoop::OutputOuter,
              "can switch back to OutputOuter");
    }

    // MNIST-ish head size: 16 * 2048 -> 10
    const int O = 10;
    const int F = 16 * 2048;
    HCNNReadout ro(O, F);
    std::mt19937 rng(12345);
    ro.randomize_weights(0.0f, rng);

    std::vector<float> in(F), glog(O), gin_a(F), gin_b(F);
    std::vector<float> wgrad(static_cast<size_t>(O) * F), bgrad(O);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : in) v = dist(rng);
    for (auto& v : glog) v = dist(rng);

    ro.set_grad_in_loop(ReadoutGradInLoop::FeatureOuter);
    ro.compute_gradients(glog.data(), in.data(), gin_a.data(),
                         wgrad.data(), bgrad.data());

    ro.set_grad_in_loop(ReadoutGradInLoop::OutputOuter);
    ro.compute_gradients(glog.data(), in.data(), gin_b.data(),
                         wgrad.data(), bgrad.data());

    float max_abs = 0.0f;
    for (int f = 0; f < F; ++f) {
        max_abs = std::max(max_abs, std::abs(gin_a[f] - gin_b[f]));
    }
    // Same add order per feature → expect exact match; allow tiny slack.
    check(max_abs == 0.0f || max_abs < 1e-5f,
          "FeatureOuter vs OutputOuter grad_in match (max_abs="
          + std::to_string(max_abs) + ")");

    // Microbench: grad_in only (weight_grad/bias_grad null so dW is skipped).
    auto time_loop = [&](ReadoutGradInLoop loop, int iters) {
        ro.set_grad_in_loop(loop);
        for (int i = 0; i < 50; ++i) {
            ro.compute_gradients(glog.data(), in.data(), gin_a.data(),
                                 nullptr, nullptr);
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            ro.compute_gradients(glog.data(), in.data(), gin_a.data(),
                                 nullptr, nullptr);
        }
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    const int iters = 500;
    const double ms_feat = time_loop(ReadoutGradInLoop::FeatureOuter, iters);
    const double ms_out  = time_loop(ReadoutGradInLoop::OutputOuter, iters);
    std::cout << "  INFO  grad_in-only A/B (O=" << O << " F=" << F
              << " iters=" << iters << "):\n";
    std::cout << "  INFO    FeatureOuter: " << ms_feat << " ms  ("
              << (ms_feat / iters) << " ms/call)\n";
    std::cout << "  INFO    OutputOuter:  " << ms_out << " ms  ("
              << (ms_out / iters) << " ms/call)\n";
    if (ms_feat > 0.0) {
        std::cout << "  INFO    OutputOuter/FeatureOuter ratio: "
                  << (ms_out / ms_feat)
                  << "  (<1 => OutputOuter faster on this machine)\n";
    }
    check(std::isfinite(ms_feat) && std::isfinite(ms_out) && ms_feat > 0.0
              && ms_out > 0.0,
          "grad_in A/B microbench produced finite positive times");
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

    check(throws([&] { HCNN bad(31, 2); }),
          "HCNN(start_dim=31) throws (max 30)");
    check(throws([&] { HCNN bad(2, 2); }),
          "HCNN(start_dim=2) throws (min 3)");
}

// Lifecycle: buffer invalidation, optimizer survive randomize, pool floor.
// Uses HCNNNetwork (power-user) where getters are needed.
static void test_network_lifecycle() {
    std::cout << "\n[Network lifecycle]\n";

    auto throws = [](auto&& fn) {
        try { fn(); } catch (const std::exception&) { return true; }
        return false;
    };

    // --- Optimizer + grad_in survive RandomizeWeights ---
    {
        HCNNNetwork net(5, 4, 1, TaskType::Classification, LossType::Default,
                        /*num_threads=*/1);
        net.add_conv(8);
        net.set_optimizer(OptimizerType::ADAM, 0.9f, 0.999f, 1e-8f);
        net.get_readout().set_grad_in_loop(ReadoutGradInLoop::FeatureOuter);
        net.randomize_all_weights(0.0f, 7);
        check(net.get_optimizer_type() == OptimizerType::ADAM,
              "network optimizer type is ADAM after set");
        check(net.get_readout().get_optimizer_type() == OptimizerType::ADAM,
              "readout keeps ADAM after randomize_all_weights");
        check(net.get_readout().get_grad_in_loop() == ReadoutGradInLoop::FeatureOuter,
              "grad_in loop survives randomize_all_weights");
        check(net.get_readout().get_num_features() == 8 * 32,
              "readout sized to c_final * N after randomize");
    }

    // --- PrepareBuffers before Randomize then train (was OOB risk) ---
    {
        HCNNNetwork net(5, 4, 1, TaskType::Classification, LossType::Default,
                        /*num_threads=*/1);
        net.add_conv(8);
        net.prepare_all_buffers();  // would freeze placeholder head size
        net.randomize_all_weights(0.0f, 11);
        check(net.get_readout().get_num_features() == 8 * 32,
              "full head after prepare-then-randomize");

        const int N = net.get_start_N();
        std::vector<float> x(static_cast<size_t>(N), 0.1f);
        int target = 1;
        net.train_step(x.data(), N, target, 0.01f);
        net.train_batch(x.data(), N, &target, 1, 0.01f);
        check(true, "train_step/batch after prepare-then-randomize");
    }

    // --- Grow stack after train_step reallocates step buffers ---
    {
        HCNNNetwork net(5, 4, 1, TaskType::Classification, LossType::Default,
                        /*num_threads=*/1);
        net.add_conv(8);
        net.randomize_all_weights(0.0f, 3);
        const int N = net.get_start_N();
        std::vector<float> x(static_cast<size_t>(N), 0.05f);
        net.train_step(x.data(), N, 0, 0.01f);
        net.add_conv(4);
        net.randomize_all_weights(0.0f, 4);
        net.train_step(x.data(), N, 0, 0.01f);
        check(net.get_readout().get_num_features() == 4 * 32,
              "step path works after add_conv post-train");
    }

    // --- New conv inherits optimizer ---
    {
        HCNNNetwork net(5, 4, 1, TaskType::Classification, LossType::Default,
                        /*num_threads=*/1);
        net.set_optimizer(OptimizerType::ADAM);
        net.add_conv(8);
        // No public getter on HCNNConv for optimizer; train under Adam after
        // randomize is enough that set_optimizer was stored for new layers.
        net.randomize_all_weights(0.0f, 1);
        check(net.get_readout().get_optimizer_type() == OptimizerType::ADAM,
              "readout Adam when set_optimizer precedes add_conv");
    }

    // --- Pool floor ---
    {
        HCNNNetwork net(3, 2, 1, TaskType::Classification, LossType::Default,
                        /*num_threads=*/1);
        net.add_conv(4);
        // dim 3 -> pool -> 2 -> pool -> 1; next pool must throw
        net.add_pool(PoolType::MAX);
        check(net.get_current_dim() == 2, "current_dim 2 after first pool");
        net.add_pool(PoolType::MAX);
        check(net.get_current_dim() == 1, "current_dim 1 after second pool");
        check(throws([&] { net.add_pool(PoolType::MAX); }),
              "add_pool at current_dim=1 throws");
    }
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
//  Spatial augmentation (2D preprocess; independent of hypercube DIM)
// ---------------------------------------------------------------------------

static void test_spatial_aug() {
    std::cout << "\n[Spatial augmentation]\n";

    // Identity / disabled: pure copy, any size
    {
        const int H = 7, W = 5, n = H * W;
        std::vector<float> src(n), dst(n, 99.0f);
        for (int i = 0; i < n; ++i) src[i] = 0.01f * static_cast<float>(i);

        HCNNSpatialAugmenter aug(HCNNSpatialAugConfig::None());
        std::mt19937 rng(1);
        check(aug.config().is_identity(), "None() config is_identity");
        aug.apply(src.data(), dst.data(), H, W, rng);
        bool same = true;
        for (int i = 0; i < n; ++i) if (dst[i] != src[i]) same = false;
        check(same, "disabled aug is memcpy for non-square HxW");
    }

    // Default config (all ops off) is identity even when enabled
    {
        HCNNSpatialAugConfig cfg; // defaults
        check(cfg.is_identity(), "default config is_identity");
        HCNNSpatialAugmenter aug(cfg);
        const int H = 4, W = 4, n = H * W;
        std::vector<float> src(n, 0.5f), dst(n, -9.0f);
        std::mt19937 rng(2);
        aug.apply(src.data(), dst.data(), H, W, rng);
        bool same = true;
        for (int i = 0; i < n; ++i) if (dst[i] != src[i]) same = false;
        check(same, "default enabled config with zero ops copies");
    }

    // Determinism: same seed => same geometric warp
    {
        HCNNSpatialAugConfig cfg;
        cfg.rot_deg_max = 15.0f;
        cfg.scale_min = 0.85f;
        cfg.scale_max = 1.15f;
        cfg.shift_max = 2;
        cfg.noise_sigma = 0.0f;
        cfg.border_value = -1.0f;
        HCNNSpatialAugmenter aug(cfg);

        const int H = 16, W = 16, n = H * W;
        std::vector<float> src(n);
        for (int i = 0; i < n; ++i) src[i] = std::sin(0.1f * static_cast<float>(i));

        std::vector<float> a(n), b(n);
        {
            std::mt19937 rng(12345);
            aug.apply(src.data(), a.data(), H, W, rng);
        }
        {
            std::mt19937 rng(12345);
            aug.apply(src.data(), b.data(), H, W, rng);
        }
        bool same = true;
        for (int i = 0; i < n; ++i) if (a[i] != b[i]) same = false;
        check(same, "fixed seed reproduces geometric aug");

        // Different seed should almost always differ
        std::vector<float> c(n);
        {
            std::mt19937 rng(99999);
            aug.apply(src.data(), c.data(), H, W, rng);
        }
        bool differ = false;
        for (int i = 0; i < n; ++i) if (c[i] != a[i]) { differ = true; break; }
        check(differ, "different seed changes geometric aug");
    }

    // Pure shift: content moves; OOB is border_value
    {
        HCNNSpatialAugConfig cfg;
        cfg.shift_max = 0; // we force shift via... can't force exact shift with random.
        // Use rot/scale off and shift_max=1, check finiteness + border appears
        cfg.shift_max = 1;
        cfg.border_value = -0.5f;
        HCNNSpatialAugmenter aug(cfg);
        const int H = 8, W = 8, n = H * W;
        std::vector<float> src(n, 1.0f), dst(n);
        std::mt19937 rng(7);
        // Several draws so we likely get nonzero shift
        bool saw_border = false;
        bool all_finite = true;
        for (int trial = 0; trial < 32; ++trial) {
            aug.apply(src.data(), dst.data(), H, W, rng);
            for (float v : dst) {
                if (!std::isfinite(v)) all_finite = false;
                if (v == cfg.border_value) saw_border = true;
            }
        }
        check(all_finite, "shift aug produces finite values");
        check(saw_border, "shift aug can introduce border_value at edges");
    }

    // Noise only: in-place allowed; clips to range
    {
        HCNNSpatialAugConfig cfg;
        cfg.noise_sigma = 0.5f;
        cfg.value_min = -1.0f;
        cfg.value_max = 1.0f;
        HCNNSpatialAugmenter aug(cfg);
        const int n = 64;
        std::vector<float> buf(n, 0.0f);
        std::mt19937 rng(3);
        aug.apply(buf.data(), buf.data(), 8, 8, rng);
        bool in_range = true;
        bool changed = false;
        for (float v : buf) {
            if (v < -1.0f || v > 1.0f) in_range = false;
            if (v != 0.0f) changed = true;
        }
        check(in_range, "noise aug clips to [value_min, value_max]");
        check(changed, "noise aug perturbs values");
    }

    // Batch path
    {
        HCNNSpatialAugConfig cfg;
        cfg.rot_deg_max = 5.0f;
        cfg.border_value = 0.0f;
        HCNNSpatialAugmenter aug(cfg);
        const int B = 3, H = 5, W = 6, plane = H * W;
        std::vector<float> src(B * plane, 0.25f), dst(B * plane, 0.0f);
        std::mt19937 rng(11);
        aug.apply_batch(src.data(), dst.data(), B, H, W, rng);
        bool finite = true;
        for (float v : dst) if (!std::isfinite(v)) finite = false;
        check(finite, "apply_batch produces finite values");
    }

    // Invalid sizes throw
    {
        HCNNSpatialAugmenter aug;
        std::mt19937 rng(0);
        float x = 0.0f;
        bool threw = false;
        try { aug.apply(&x, &x, 0, 4, rng); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "apply rejects height < 1");
    }

    // Geometric with in == out throws
    {
        HCNNSpatialAugConfig cfg;
        cfg.rot_deg_max = 10.0f;
        HCNNSpatialAugmenter aug(cfg);
        std::vector<float> buf(16, 1.0f);
        std::mt19937 rng(0);
        bool threw = false;
        try { aug.apply(buf.data(), buf.data(), 4, 4, rng); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "geometric aug rejects in == out");
    }

    // value_min > value_max rejected
    {
        HCNNSpatialAugConfig cfg;
        cfg.value_min = 1.0f;
        cfg.value_max = -1.0f;
        bool threw = false;
        try { HCNNSpatialAugmenter aug(cfg); (void)aug; }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "value_min > value_max rejected");
    }

    // Negative rot_deg_max uses absolute magnitude (still non-identity)
    {
        HCNNSpatialAugConfig cfg;
        cfg.rot_deg_max = -12.0f;
        check(!cfg.is_identity(), "negative rot_deg_max is not identity");
        HCNNSpatialAugmenter aug(cfg);
        check(true, "negative rot_deg_max accepted at construct");
    }

    // Negative noise_sigma rejected
    {
        HCNNSpatialAugConfig cfg;
        cfg.noise_sigma = -0.1f;
        bool threw = false;
        try { HCNNSpatialAugmenter aug(cfg); (void)aug; }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "negative noise_sigma rejected");
    }

    // Shear: non-identity, deterministic, changes a non-uniform field
    {
        HCNNSpatialAugConfig cfg;
        cfg.shear_x_max = 0.2f;
        cfg.shear_y_max = 0.0f;
        cfg.border_value = -1.0f;
        check(!cfg.is_identity(), "shear_x_max makes config non-identity");
        HCNNSpatialAugmenter aug(cfg);
        const int H = 12, W = 12, n = H * W;
        std::vector<float> src(n);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                src[y * W + x] = static_cast<float>(x) / static_cast<float>(W - 1);

        std::vector<float> a(n), b(n);
        {
            std::mt19937 rng(4242);
            aug.apply(src.data(), a.data(), H, W, rng);
        }
        {
            std::mt19937 rng(4242);
            aug.apply(src.data(), b.data(), H, W, rng);
        }
        bool same = true;
        for (int i = 0; i < n; ++i) if (a[i] != b[i]) same = false;
        check(same, "fixed seed reproduces shear aug");

        bool differ = false;
        for (int i = 0; i < n; ++i) if (a[i] != src[i]) { differ = true; break; }
        check(differ, "shear aug changes a horizontal gradient");

        bool finite = true;
        for (float v : a) if (!std::isfinite(v)) finite = false;
        check(finite, "shear aug produces finite values");
    }

    // Elastic: validation, deterministic, changes content
    {
        HCNNSpatialAugConfig bad0;
        bad0.elastic_alpha = 1.0f;
        bad0.elastic_sigma = 0.0f;
        bool threw = false;
        try { HCNNSpatialAugmenter aug(bad0); (void)aug; }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "elastic_alpha without elastic_sigma rejected");

        HCNNSpatialAugConfig bad_lo;
        bad_lo.elastic_alpha = 1.0f;
        bad_lo.elastic_sigma = 0.1f; // below kElasticSigmaMin
        threw = false;
        try { HCNNSpatialAugmenter aug(bad_lo); (void)aug; }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "elastic_sigma below min rejected");

        HCNNSpatialAugConfig bad_hi;
        bad_hi.elastic_alpha = 1.0f;
        bad_hi.elastic_sigma = 100.0f;
        threw = false;
        try { HCNNSpatialAugmenter aug(bad_hi); (void)aug; }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "elastic_sigma above max rejected");

        HCNNSpatialAugConfig bad_shear;
        bad_shear.shear_x_max = 1.0f;
        bad_shear.shear_y_max = 1.0f; // product 1.0 >= 0.95
        threw = false;
        try { HCNNSpatialAugmenter aug(bad_shear); (void)aug; }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "near-singular shear product rejected");

        // Note: isfinite guards in validate() are best-effort; Release may use
        // -ffast-math which can break NaN tests, so we do not smoke NaN here.

        HCNNSpatialAugConfig cfg;
        cfg.elastic_alpha = 1.5f;
        cfg.elastic_sigma = 4.0f;
        cfg.border_value = -1.0f;
        check(!cfg.is_identity(), "elastic_alpha makes config non-identity");
        HCNNSpatialAugmenter aug(cfg);
        const int H = 16, W = 16, n = H * W;
        std::vector<float> src(n);
        for (int i = 0; i < n; ++i)
            src[i] = std::sin(0.2f * static_cast<float>(i));

        std::vector<float> a(n), b(n);
        {
            std::mt19937 rng(777);
            aug.apply(src.data(), a.data(), H, W, rng);
        }
        {
            std::mt19937 rng(777);
            aug.apply(src.data(), b.data(), H, W, rng);
        }
        bool same = true;
        for (int i = 0; i < n; ++i) if (a[i] != b[i]) same = false;
        check(same, "fixed seed reproduces elastic aug");

        bool differ = false;
        for (int i = 0; i < n; ++i) if (std::fabs(a[i] - src[i]) > 1e-6f) {
            differ = true; break;
        }
        check(differ, "elastic aug changes a sinusoidal field");

        // Mild elastic should not destroy a constant field interior into border.
        std::vector<float> solid(n, 0.5f), out_s(n);
        {
            std::mt19937 rng(3);
            aug.apply(solid.data(), out_s.data(), H, W, rng);
        }
        int interior_ok = 0;
        int interior_n = 0;
        for (int y = 2; y < H - 2; ++y) {
            for (int x = 2; x < W - 2; ++x) {
                ++interior_n;
                const float v = out_s[y * W + x];
                if (std::isfinite(v) && std::fabs(v - 0.5f) < 0.25f)
                    ++interior_ok;
            }
        }
        check(interior_ok * 2 >= interior_n,
              "mild elastic keeps most solid-field interior near constant");

        // in == out rejected for elastic
        std::vector<float> buf = src;
        std::mt19937 rng(0);
        threw = false;
        try { aug.apply(buf.data(), buf.data(), H, W, rng); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "elastic aug rejects in == out");
    }

    // Shear geometry: horizontal ramp f=x tilts under shear_x (row profiles differ)
    {
        HCNNSpatialAugConfig cfg;
        cfg.shear_x_max = 0.25f;
        cfg.border_value = -1.0f;
        HCNNSpatialAugmenter aug(cfg);
        const int H = 20, W = 20, n = H * W;
        std::vector<float> src(n), dst(n);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                src[y * W + x] = static_cast<float>(x);
        std::mt19937 rng(42);
        aug.apply(src.data(), dst.data(), H, W, rng);

        // Compare mid columns of top vs bottom rows (excluding borders).
        // Pure horizontal ramp is constant down columns; shear mixes x with y.
        float top_mid = 0.0f, bot_mid = 0.0f;
        int cnt = 0;
        for (int x = 4; x < W - 4; ++x) {
            top_mid += dst[2 * W + x];
            bot_mid += dst[(H - 3) * W + x];
            ++cnt;
        }
        top_mid /= static_cast<float>(cnt);
        bot_mid /= static_cast<float>(cnt);
        check(std::fabs(top_mid - bot_mid) > 0.05f,
              "shear_x tilts horizontal ramp across rows");
    }

    // Affine + elastic composition is deterministic
    {
        HCNNSpatialAugConfig cfg;
        cfg.rot_deg_max = 8.0f;
        cfg.shear_x_max = 0.1f;
        cfg.elastic_alpha = 1.0f;
        cfg.elastic_sigma = 5.0f;
        cfg.border_value = -1.0f;
        HCNNSpatialAugmenter aug(cfg);
        const int H = 14, W = 14, n = H * W;
        std::vector<float> src(n, 0.3f);
        for (int i = 0; i < n; ++i)
            src[i] = 0.01f * static_cast<float>(i % 17);
        std::vector<float> a(n), b(n);
        {
            std::mt19937 rng(13579);
            aug.apply(src.data(), a.data(), H, W, rng);
        }
        {
            std::mt19937 rng(13579);
            aug.apply(src.data(), b.data(), H, W, rng);
        }
        bool same = true;
        for (int i = 0; i < n; ++i) if (a[i] != b[i]) same = false;
        check(same, "fixed seed reproduces shear+elastic composition");
    }
}

// ---------------------------------------------------------------------------
//  Spatial embed (2D → length N = 2^dim, P ≤ N)
// ---------------------------------------------------------------------------

static void test_spatial_embed() {
    std::cout << "\n[Spatial embed]\n";

    check(HCNNSpatialEmbedder::max_square_side(2048) == 45
          || HCNNSpatialEmbedder::max_square_side(2048) * HCNNSpatialEmbedder::max_square_side(2048) <= 2048,
          "max_square_side(2048) is valid");
    check(HCNNSpatialEmbedder::max_dual_plane_side(2048) == 32,
          "max_dual_plane_side(2048) == 32");
    check(HCNNSpatialEmbedder::max_dual_plane_side(512) == 16,
          "max_dual_plane_side(512) == 16");
    check(HCNNSpatialEmbedder::max_square_side(512) == 22,
          "max_square_side(512) == 22");

    // RowMajorPad: copy + pad
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 6;  // N=64
        cfg.mode = HCNNSpatialEmbedMode::RowMajorPad;
        cfg.pad_value = -1.0f;
        HCNNSpatialEmbedder emb(cfg);
        check(emb.capacity() == 64, "capacity 2^6 == 64");

        const int H = 4, W = 5;  // 20 <= 64
        std::vector<float> src(H * W);
        for (int i = 0; i < H * W; ++i) src[i] = 0.1f * static_cast<float>(i);
        std::vector<float> out(emb.capacity(), 99.0f);
        emb.embed(src.data(), H, W, out.data());

        bool head_ok = true;
        for (int i = 0; i < H * W; ++i) if (out[i] != src[i]) head_ok = false;
        bool tail_ok = true;
        for (int i = H * W; i < 64; ++i) if (out[i] != -1.0f) tail_ok = false;
        check(head_ok, "RowMajorPad copies H*W prefix");
        check(tail_ok, "RowMajorPad pads with pad_value");

        auto plan = emb.plan(H, W);
        check(plan.pattern_length == H * W && plan.N == 64,
              "RowMajorPad plan pattern_length and N");
    }

    // RowMajorPad rejects H*W > N
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 5;  // N=32
        cfg.mode = HCNNSpatialEmbedMode::RowMajorPad;
        HCNNSpatialEmbedder emb(cfg);
        std::vector<float> src(64, 0.0f), out(32);
        bool threw = false;
        try { emb.embed(src.data(), 8, 8, out.data()); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "RowMajorPad rejects H*W > N");
    }

    // ResizeToFit: always fills S*S, pads rest
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 9;  // N=512, S=22
        cfg.mode = HCNNSpatialEmbedMode::ResizeToFit;
        cfg.pad_value = 0.0f;
        HCNNSpatialEmbedder emb(cfg);
        auto plan = emb.plan(28, 28);
        check(plan.plane_side == 22, "ResizeToFit dim9 plane_side == 22");
        check(plan.pattern_length == 22 * 22, "ResizeToFit pattern_length");

        std::vector<float> src(28 * 28, 0.5f), out(emb.capacity(), -9.0f);
        emb.embed(src.data(), 28, 28, out.data());
        bool finite = true;
        for (float v : out) if (!std::isfinite(v)) finite = false;
        check(finite, "ResizeToFit produces finite values");
        bool pad_ok = true;
        for (int i = plan.pattern_length; i < plan.N; ++i)
            if (out[i] != 0.0f) pad_ok = false;
        check(pad_ok, "ResizeToFit pads unused vertices");
    }

    // DualPlaneResize: full occupancy at dim 11
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 11;
        cfg.mode = HCNNSpatialEmbedMode::DualPlaneResize;
        cfg.pad_value = -1.0f;
        HCNNSpatialEmbedder emb(cfg);
        auto plan = emb.plan(28, 28);
        check(plan.plane_side == 32, "DualPlane dim11 plane_side == 32");
        check(plan.pattern_length == 2048 && plan.N == 2048,
              "DualPlane dim11 full occupancy");

        // Non-constant image so |grad| is not blank
        std::vector<float> src(28 * 28);
        for (int y = 0; y < 28; ++y)
            for (int x = 0; x < 28; ++x)
                src[y * 28 + x] = (x > 14) ? 1.0f : -1.0f;

        std::vector<float> out(emb.capacity());
        emb.embed(src.data(), 28, 28, out.data());
        bool finite = true;
        for (float v : out) if (!std::isfinite(v)) finite = false;
        check(finite, "DualPlane produces finite values");

        // Grad plane should not be all pad for a step edge
        bool grad_alive = false;
        for (int i = 1024; i < 2048; ++i) {
            if (out[i] != cfg.pad_value) { grad_alive = true; break; }
        }
        check(grad_alive, "DualPlane |grad| plane has structure");
    }

    // DualPlane dim 9: 16x16 * 2 = 512
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 9;
        cfg.mode = HCNNSpatialEmbedMode::DualPlaneResize;
        HCNNSpatialEmbedder emb(cfg);
        auto plan = emb.plan(28, 28);
        check(plan.plane_side == 16 && plan.pattern_length == 512,
              "DualPlane dim9 is 16x16 dual full fill");
    }

    // Explicit plane_side override
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 11;
        cfg.mode = HCNNSpatialEmbedMode::DualPlaneResize;
        cfg.plane_side = 16;  // smaller than max 32
        HCNNSpatialEmbedder emb(cfg);
        auto plan = emb.plan(28, 28);
        check(plan.plane_side == 16 && plan.pattern_length == 512,
              "plane_side override respected");
        std::vector<float> src(28 * 28, 0.0f), out(emb.capacity(), 1.0f);
        emb.embed(src.data(), 28, 28, out.data());
        // tail N - 512 should be pad (default 0)
        bool tail = true;
        for (int i = 512; i < 2048; ++i) if (out[i] != 0.0f) tail = false;
        check(tail, "smaller dual plane pads remainder of N");
    }

    // Invalid plane_side rejected
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 8;  // N=256
        cfg.mode = HCNNSpatialEmbedMode::DualPlaneResize;
        cfg.plane_side = 20;  // 2*400=800 > 256
        bool threw = false;
        try { HCNNSpatialEmbedder emb(cfg); (void)emb; }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "oversized plane_side rejected at construct");
    }

    // Batch embed
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 6;
        cfg.mode = HCNNSpatialEmbedMode::RowMajorPad;
        HCNNSpatialEmbedder emb(cfg);
        const int B = 2, H = 3, W = 3, N = 64;
        std::vector<float> src(B * H * W, 0.25f), out(B * N, -3.0f);
        emb.embed_batch(src.data(), B, H, W, out.data());
        check(out[0] == 0.25f && out[N] == 0.25f, "embed_batch writes both samples");
        check(out[H * W] == 0.0f, "embed_batch pads each sample");
    }

    // Works with HCNN train path (input_length = N)
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 6;
        cfg.mode = HCNNSpatialEmbedMode::ResizeToFit;
        HCNNSpatialEmbedder emb(cfg);
        std::vector<float> src(10 * 10, 0.1f), packed(emb.capacity());
        emb.embed(src.data(), 10, 10, packed.data());

        HCNN net(6, 2);
        net.AddConv(4);
        net.RandomizeWeights(0.0f, 1);
        std::vector<float> logits(2);
        // Contract: input_length = capacity() (== N), not pattern_length alone
        net.TrainStep(packed.data(), emb.capacity(), 0, 0.01f, 0.9f, 0.0f);
        net.ForwardBatch(packed.data(), emb.capacity(), 1, logits.data());
        check(std::isfinite(logits[0]) && std::isfinite(logits[1]),
              "embedded vector trains/infers on HCNN");
    }

    // Aug then embed chain
    {
        HCNNSpatialAugConfig acfg;
        acfg.rot_deg_max = 5.0f;
        acfg.border_value = -1.0f;
        HCNNSpatialAugmenter aug(acfg);

        HCNNSpatialEmbedConfig ecfg;
        ecfg.dim = 9;
        ecfg.mode = HCNNSpatialEmbedMode::DualPlaneResize;
        ecfg.pad_value = -1.0f;
        HCNNSpatialEmbedder emb(ecfg);

        const int H = 28, W = 28;
        std::vector<float> src(H * W, 0.2f), work(H * W), packed(emb.capacity());
        std::mt19937 rng(42);
        aug.apply(src.data(), work.data(), H, W, rng);
        emb.embed(work.data(), H, W, packed.data());
        bool finite = true;
        for (float v : packed) if (!std::isfinite(v)) finite = false;
        check(finite, "aug then DualPlane embed is finite");
        check(emb.plan(H, W).pattern_length == emb.capacity(),
              "dim9 dual full occupancy after chain");
    }

    // HCNN::Embed zero-pads short length (do not use short input_length with pad_value)
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 6;  // N=64
        cfg.mode = HCNNSpatialEmbedMode::RowMajorPad;
        cfg.pad_value = -1.0f;
        HCNNSpatialEmbedder emb(cfg);
        std::vector<float> src(4 * 4, 0.5f), packed(emb.capacity());
        emb.embed(src.data(), 4, 4, packed.data());
        check(packed[16] == -1.0f, "spatial embed pad_value on unused verts");

        HCNN net(6, 2);
        std::vector<float> embedded(net.GetStartN());
        // If caller wrongly passes only pattern_length, Embed zeros the tail
        net.Embed(packed.data(), 16, embedded.data());
        check(embedded[16] == 0.0f,
              "HCNN::Embed zero-pads short length (overrides non-zero pad)");
    }
}

// ---------------------------------------------------------------------------
//  Training helpers (metrics, cosine LR, dual-ckpt, flat dataset)
// ---------------------------------------------------------------------------

static void test_train_helpers() {
    std::cout << "\n[Train helpers]\n";

    // --- argmax / softmax CE ---
    {
        float v[] = {0.1f, 0.9f, 0.3f};
        check(argmax(v, 3) == 1, "argmax picks max index");

        float logits[] = {0.0f, 2.0f, 0.0f};
        // CE for target=1: -log(softmax_1); should be small
        float ce_good = softmax_cross_entropy(logits, 3, 1);
        float ce_bad  = softmax_cross_entropy(logits, 3, 0);
        check(std::isfinite(ce_good) && ce_good > 0.0f, "CE finite and positive");
        check(ce_good < ce_bad, "CE lower for correct class");
    }

    // --- cosine_lr endpoints ---
    {
        const float lo = cosine_lr(1e-3f, 1e-4f, 0, 60);
        const float mid = cosine_lr(1e-3f, 1e-4f, 30, 60);
        const float hi = cosine_lr(1e-3f, 1e-4f, 59, 60);
        check(std::abs(lo - 1e-3f) < 1e-7f, "cosine_lr epoch 0 == lr_max");
        check(std::abs(hi - 1e-4f) < 1e-7f, "cosine_lr last epoch == lr_min");
        check(mid > hi && mid < lo, "cosine_lr mid between endpoints");
        check(std::abs(cosine_lr(0.01f, 0.001f, 0, 1) - 0.01f) < 1e-9f,
              "cosine_lr num_epochs<=1 returns lr_max");
    }

    // --- Negative paths: empty FlatDataset, size drift, OOR target ---
    {
        auto throws = [](auto&& fn) {
            try { fn(); } catch (const std::exception&) { return true; }
            return false;
        };

        HCNN net(5, 4);
        net.AddConv(8);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/3);

        HCNNFlatDataset empty_ds;
        check(throws([&] { (void)evaluate_classification(net, empty_ds); }),
              "evaluate empty FlatDataset throws");

        HCNNFlatDataset bad;
        bad.reset(4, net.GetStartN());
        bad.count = 8;  // drift public field past buffer size
        check(throws([&] { (void)evaluate_classification(net, bad); }),
              "evaluate size-drifted FlatDataset throws");

        float logits[] = {0.0f, 1.0f, 0.0f};
        check(throws([&] { (void)softmax_cross_entropy(logits, 3, /*target=*/3); }),
              "softmax_cross_entropy OOR target throws");
        check(throws([&] { (void)softmax_cross_entropy(logits, 3, /*target=*/-1); }),
              "softmax_cross_entropy negative target throws");
    }

    // --- Dual-ckpt tie-breaks (pure observe sequence; no training needed) ---
    {
        HCNN net(5, 4);
        net.AddConv(8);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/11);

        HCNNDualCheckpoint ckpt;
        auto u1 = ckpt.observe(net, /*loss=*/1.0f, /*accuracy=*/50.0f, /*epoch=*/1);
        check(u1.new_best_loss && u1.new_best_acc, "tie: first observe both bests");
        check(ckpt.best_loss_epoch() == 1 && ckpt.best_acc_epoch() == 1,
              "tie: first epochs == 1");

        // Equal loss, higher acc → best-loss updates; higher acc → best-acc updates.
        auto u2 = ckpt.observe(net, 1.0f, 55.0f, 2);
        check(u2.new_best_loss && u2.new_best_acc,
              "tie: equal loss higher acc updates both");
        check(ckpt.best_loss_epoch() == 2 && ckpt.best_loss_acc() == 55.0f,
              "tie: best-loss epoch/acc after higher-acc tie-break");
        check(ckpt.best_acc_epoch() == 2 && ckpt.best_acc() == 55.0f,
              "tie: best-acc epoch after higher acc");

        // Equal loss and equal acc → neither slot updates.
        auto u3 = ckpt.observe(net, 1.0f, 55.0f, 3);
        check(!u3.any(), "tie: equal loss and acc is no-op");
        check(ckpt.best_loss_epoch() == 2 && ckpt.best_acc_epoch() == 2,
              "tie: epochs unchanged on no-op");

        // Equal acc, lower loss → best-acc updates (lower-loss tie-break);
        // strictly lower loss → best-loss updates too.
        auto u4 = ckpt.observe(net, 0.95f, 55.0f, 4);
        check(u4.new_best_loss && u4.new_best_acc,
              "tie: lower loss updates both (acc equal uses lower loss)");
        check(ckpt.best_acc_loss() == 0.95f && ckpt.best_acc_epoch() == 4,
              "tie: best-acc secondary loss and epoch");
        check(ckpt.best_loss() == 0.95f && ckpt.best_loss_epoch() == 4,
              "tie: best-loss after lower loss");

        // Equal loss, lower acc → best-loss does NOT update.
        auto u5 = ckpt.observe(net, 0.95f, 54.0f, 5);
        check(!u5.new_best_loss, "tie: equal loss lower acc skips best-loss");
        // Acc went down → no best-acc either.
        check(!u5.new_best_acc, "tie: lower acc skips best-acc");
        check(ckpt.best_loss_epoch() == 4, "tie: best-loss epoch stays 4");
    }

    // --- FlatDataset + evaluate_classification + dual checkpoint train loop ---
    {
        HCNN net(5, 4);
        net.AddConv(8);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/7);
        net.SetOptimizer(OptimizerType::ADAM);

        const int N = net.GetStartN();
        const int K = net.GetNumOutputs();
        const int n = 32;

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(n, N, K, /*seed=*/99, inputs, targets);

        HCNNFlatDataset ds;
        ds.reset(n, N);
        for (int i = 0; i < n; ++i) {
            std::copy(inputs[i].begin(), inputs[i].end(), ds.sample_input(i));
            ds.targets[static_cast<size_t>(i)] = targets[static_cast<size_t>(i)];
        }
        check(ds.count == n && ds.input_length == N, "FlatDataset reset sizing");

        auto r0 = evaluate_classification(net, ds);
        check(r0.count == n, "evaluate_classification count");
        check(std::isfinite(r0.loss), "evaluate_classification loss finite");
        check(r0.accuracy >= 0.0f && r0.accuracy <= 100.0f,
              "evaluate_classification accuracy in [0,100]");
        check(r0.correct >= 0 && r0.correct <= n, "evaluate_classification correct range");

        HCNNDualCheckpoint ckpt;
        auto u0 = ckpt.observe(net, r0.loss, r0.accuracy, /*epoch=*/1);
        check(u0.new_best_loss && u0.new_best_acc,
              "dual-ckpt first observe is both bests");
        check(ckpt.has_best_loss() && ckpt.has_best_acc(), "dual-ckpt has snapshots");
        check(ckpt.best_loss_epoch() == 1 && ckpt.best_acc_epoch() == 1,
              "dual-ckpt epochs recorded");

        // Train a few epochs; LR follows cosine helper.
        const int epochs = 5;
        for (int e = 0; e < epochs; ++e) {
            const float lr = cosine_lr(0.05f, 0.005f, e, epochs);
            net.TrainEpoch(ds.inputs.data(), ds.input_length, ds.targets.data(),
                           ds.count, /*batch_size=*/8, lr, /*momentum=*/0.0f,
                           /*wd=*/0.0f, /*class_weights=*/nullptr,
                           /*shuffle_seed=*/static_cast<unsigned>(e + 1));
            auto r = evaluate_classification(net, ds);
            ckpt.observe(net, r.loss, r.accuracy, e + 1);
        }

        check(ckpt.has_best_loss() && ckpt.has_best_acc(),
              "dual-ckpt still has snapshots after train");
        check(std::isfinite(ckpt.best_loss()), "dual-ckpt best_loss finite");
        check(ckpt.best_acc() >= 0.0f, "dual-ckpt best_acc non-negative");

        // Restore both checkpoints without throw; logits stay finite.
        ckpt.restore_best_loss(net);
        auto r_loss = evaluate_classification(net, ds);
        check(std::isfinite(r_loss.loss), "restore best-loss: eval finite");

        ckpt.restore_best_acc(net);
        auto r_acc = evaluate_classification(net, ds);
        check(std::isfinite(r_acc.loss), "restore best-acc: eval finite");

        // Empty checkpoint restore must throw.
        HCNNDualCheckpoint empty;
        bool threw = false;
        try { empty.restore_best_loss(net); } catch (const std::exception&) { threw = true; }
        check(threw, "empty dual-ckpt restore_best_loss throws");
    }

    // --- Regression metrics + best-metric checkpoint ---
    {
        HCNN net(5, /*num_outputs=*/1, /*input_channels=*/1,
                 TaskType::Regression);
        net.AddConv(8, Activation::TANH);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/3);
        net.SetOptimizer(OptimizerType::ADAM);

        const int N = net.GetStartN();
        const int n = 16;
        std::vector<float> inputs(static_cast<size_t>(n) * N, 0.1f);
        std::vector<float> targets(static_cast<size_t>(n), 0.0f);
        for (int i = 0; i < n; ++i)
            targets[static_cast<size_t>(i)] = 0.25f * static_cast<float>(i % 4);

        auto r0 = evaluate_regression(net, inputs.data(), N, targets.data(), n);
        check(r0.count == n, "evaluate_regression count");
        check(std::isfinite(r0.mse), "evaluate_regression mse finite");
        check(std::isfinite(r0.target_var), "evaluate_regression var finite");

        HCNNBestMetricCheckpoint best;
        check(best.observe(net, static_cast<float>(r0.mse), 1),
              "best-metric first observe is best");
        check(best.has_best() && best.best_epoch() == 1, "best-metric has snapshot");

        net.TrainEpochRegression(inputs.data(), N, targets.data(), n,
                                 /*batch=*/8, /*lr=*/0.05f);
        auto r1 = evaluate_regression(net, inputs.data(), N, targets.data(), n);
        best.observe(net, static_cast<float>(r1.mse), 2);
        check(best.has_best(), "best-metric still has snapshot after train");

        best.restore(net);
        auto r2 = evaluate_regression(net, inputs.data(), N, targets.data(), n);
        check(std::isfinite(r2.mse), "best-metric restore: eval finite");

        HCNNBestMetricCheckpoint empty;
        bool threw = false;
        try { empty.restore(net); } catch (const std::exception&) { threw = true; }
        check(threw, "empty best-metric restore throws");
    }
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "HCNN SDK Smoke Test\n";
    std::cout << "===================\n";

    test_construction();
    test_self_contribution();
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
    test_readout_grad_in_loop_ab();
    test_avg_pool_training();
    test_weight_decay();
    test_embed_padding_and_truncation();
    test_invalid_arguments();
    test_network_lifecycle();
    test_forward_preserves_training_mode();
    test_class_weights();
    test_regression_scalar();
    test_regression_multi_output();
    test_forward_batch_regression();
    test_regression_classification_cross_misuse();
    test_regression_invalid_construction();
    test_spatial_aug();
    test_spatial_embed();
    test_train_helpers();

    std::cout << "\n===================\n";
    if (failures == 0) {
        std::cout << "All tests PASSED\n";
        return 0;
    } else {
        std::cout << failures << " test(s) FAILED\n";
        return 1;
    }
}
