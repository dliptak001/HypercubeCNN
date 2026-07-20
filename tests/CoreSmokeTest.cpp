// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// CoreSmokeTest — fast HCNN SDK smoke suite.
//
// Primary: public facade (HCNN / HypercubeCNN.h) — the only app-facing API.
// Secondary: private implementation contracts (in-tree only; headers not
// installed).  These are not a second SDK for applications:
//   - HCNNConv self-tap math and BN bn_save
//   - HCNNNetwork lifecycle (optimizer / prepare / pool floor)
//   - HCNNReadout FeatureOuter vs OutputOuter grad_in match
//   - ThreadPool / HCNNPool dim guards
//
// Goal: well under 1s on Release. Prefer short train drops (30–40 steps,
// 2–3 epochs, n_train≈16–20) over redundant “loss fell over 100 steps”
// variants.

#include "HypercubeCNN.h"   // public umbrella
#include "HCNNConv.h"       // private impl (in-tree tests only)
#include "HCNNNetwork.h"    // private impl (in-tree tests only)
#include "HCNNPool.h"
#include "HCNNReadout.h"
#include "ThreadPool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using hcnn::HCNN;
using hcnn::HCNNConfig;
using hcnn::HCNNNetwork;
using hcnn::HCNNPool;
using hcnn::HCNNReadout;
using hcnn::LayerSpec;
using hcnn::ReadoutGradInLoop;
using hcnn::ThreadPool;
using hcnn::TrainParams;
using hcnn::PoolType;
using hcnn::TaskType;
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
using hcnn::save_weights;
using hcnn::load_weights;
using hcnn::HCNNInputView;
using hcnn::HCNNInputBatch;
using hcnn::HCNNTrainer;

// ---------------------------------------------------------------------------
//  Reporting
// ---------------------------------------------------------------------------

static int g_passed = 0;
static int g_failed = 0;
static int g_section_passed = 0;
static int g_section_failed = 0;

static void begin_section(const char* name) {
    g_section_passed = 0;
    g_section_failed = 0;
    std::cout << "\n== " << name << " ==\n";
}

static void end_section() {
    std::cout << "  (" << g_section_passed << " passed";
    if (g_section_failed > 0)
        std::cout << ", " << g_section_failed << " failed";
    std::cout << ")\n";
}

static void check(bool condition, const std::string& name) {
    if (condition) {
        std::cout << "  PASS  " << name << "\n";
        ++g_passed;
        ++g_section_passed;
    } else {
        std::cout << "  FAIL  " << name << "\n";
        ++g_failed;
        ++g_section_failed;
    }
}

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

static bool all_finite(const float* v, int n) {
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(v[i])) return false;
    return true;
}

template <typename Fn>
static bool throws(Fn&& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

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
        float mean = static_cast<float>(s / N);
        targets_out[i] = std::tanh(2.0f * mean);
    }
}

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
//  1. ThreadPool (+ HCNNPool dim guards)
// ---------------------------------------------------------------------------

static void section_thread_pool() {
    begin_section("ThreadPool");

    {
        ThreadPool pool(2);
        int hits = 0;
        pool.ForEach(0, [&](size_t, size_t, size_t) { ++hits; });
        check(hits == 0, "ForEach(count=0) does not invoke func");
        check(pool.NumWorkers() == 2, "NumWorkers() == 2");
        check(pool.NumThreads() == 3, "NumThreads() == workers + caller");
    }

    {
        ThreadPool pool(3);
        constexpr size_t N = 1000;
        std::vector<int> hits(N, 0);
        pool.ForEach(N, [&](size_t /*tid*/, size_t b, size_t e) {
            for (size_t i = b; i < e; ++i) hits[i] += 1;
        });
        bool ok = true;
        for (size_t i = 0; i < N; ++i)
            if (hits[i] != 1) { ok = false; break; }
        check(ok, "ForEach visits each index exactly once");
    }

    {
        ThreadPool pool(2);
        check(throws([&] {
            pool.ForEach(64, [](size_t tid, size_t, size_t) {
                if (tid != 0) throw std::runtime_error("worker boom");
            });
        }), "worker exception rethrown after join");
        std::atomic<int> sum{0};
        pool.ForEach(32, [&](size_t, size_t b, size_t e) {
            sum.fetch_add(static_cast<int>(e - b));
        });
        check(sum.load() == 32, "pool usable after worker exception");
    }

    {
        ThreadPool pool(2);
        check(throws([&] {
            pool.ForEach(128, [](size_t tid, size_t, size_t) {
                if (tid == 0) throw std::runtime_error("caller boom");
            });
        }), "caller chunk exception rethrown after join");
        std::atomic<int> sum{0};
        pool.ForEach(40, [&](size_t, size_t b, size_t e) {
            sum.fetch_add(static_cast<int>(e - b));
        });
        check(sum.load() == 40, "pool usable after caller exception");
    }

    {
        ThreadPool auto_pool(0);
        check(auto_pool.NumThreads() >= 1, "auto pool has at least caller thread");
    }

    check(throws([&] { HCNNPool p(1); }), "HCNNPool(dim=1) throws");
    check(throws([&] { HCNNPool p(31); }), "HCNNPool(dim=31) throws");
    {
        HCNNPool p(5, PoolType::MAX);
        check(p.get_input_N() == 32 && p.get_output_N() == 16,
              "HCNNPool dim=5 sizes N=32->16");
    }

    end_section();
}

// ---------------------------------------------------------------------------
//  2. Construction, self-kernel, weight blob
// ---------------------------------------------------------------------------

static void section_construction() {
    begin_section("Construction & self-kernel");

    {
        HCNN net(5, 4);
        check(net.GetStartDim() == 5, "GetStartDim() == 5");
        check(net.GetStartN() == 32, "GetStartN() == 32");
        check(net.GetNumOutputs() == 4, "GetNumOutputs() == 4");
        check(net.GetInputChannels() == 1, "GetInputChannels() == 1");
        net.AddConv(8);
        net.AddPool(PoolType::MAX);
        net.AddConv(16);
        net.RandomizeWeights();
        check(net.GetStartDim() == 5, "GetStartDim() unchanged after build");
        check(net.GetNumOutputs() == 4, "GetNumOutputs() unchanged after build");
    }

    check(throws([&] {
        hcnn::HCNNConv bad(31, 1, 2, Activation::NONE, true, false);
    }), "HCNNConv(DIM=31) throws (max 30)");
    check(throws([&] {
        hcnn::HCNNConv bad(5, 0, 2, Activation::NONE, true, false);
    }), "HCNNConv(c_in=0) throws");
    check(throws([&] {
        hcnn::HCNNConv bad(5, 1, 0, Activation::NONE, true, false);
    }), "HCNNConv(c_out=0) throws");

    // Self/center kernel tap: K = DIM + 1, last index multiplies in[v].
    {
        const int dim = 5;
        const int N = 1 << dim;
        const int K = dim + 1;

        hcnn::HCNNConv conv(dim, /*c_in=*/1, /*c_out=*/2,
                            Activation::NONE, /*bias=*/true, /*bn=*/false);
        check(conv.get_K() == K, "get_K() == DIM + 1");
        check(conv.get_self_index() == dim, "get_self_index() == DIM");
        check(conv.get_kernel_size() == 2 * 1 * K, "kernel size = c_out*c_in*(DIM+1)");

        float* ker = conv.get_kernel_data();
        for (int i = 0; i < conv.get_kernel_size(); ++i) ker[i] = 0.0f;
        ker[0 * K + dim] = 2.0f;
        ker[1 * K + dim] = -0.5f;
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

        ker[0 * K + 0] = 1.0f;
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
    }

    // Weight blob includes self taps (K = DIM+1).
    {
        HCNN net(5, /*num_outputs=*/4);
        net.AddConv(8, Activation::RELU, /*bias=*/true, /*bn=*/false);
        check(!net.WeightsInitialized(), "WeightsInitialized false before Randomize");
        check(throws([&] { (void)net.GetWeightCount(); }),
              "GetWeightCount before RandomizeWeights throws");
        net.RandomizeWeights();
        check(net.WeightsInitialized(), "WeightsInitialized true after Randomize");
        // kernel: 1*8*6 + bias 8; readout FLATTEN 8*32 -> 4 + bias 4
        const size_t expected = static_cast<size_t>(1 * 8 * 6 + 8 + 8 * 32 * 4 + 4);
        check(net.GetWeightCount() == expected,
              "GetWeightCount includes self taps (K=DIM+1)");
        check(net.GetOptimizerType() == OptimizerType::ADAM,
              "GetOptimizerType default ADAM");
        check(net.GetNumConv() == 1 && net.GetNumPool() == 0,
              "GetNumConv/GetNumPool facade");
    }

    // Move transfers ownership; heap ThreadPool is not relocated.
    {
        HCNN a(5, 4);
        a.AddConv(8);
        a.RandomizeWeights(/*scale=*/0.0f, /*seed=*/3);
        const int N = a.GetStartN();
        const int K = a.GetNumOutputs();
        std::vector<float> x(static_cast<size_t>(N), 0.2f), out(static_cast<size_t>(K));
        a.Predict(x.data(), N, out.data());
        check(all_finite(out.data(), K), "pre-move Predict finite");

        HCNN b = std::move(a);
        check(b.WeightsInitialized(), "moved-to WeightsInitialized");
        check(b.GetNumConv() == 1, "moved-to layer count");
        std::fill(out.begin(), out.end(), 0.0f);
        b.Predict(x.data(), N, out.data());
        check(all_finite(out.data(), K), "post-move Predict finite");

        HCNN c(5, 2);
        c = std::move(b);
        check(c.GetNumOutputs() == 4, "move-assign keeps source num_outputs");
        c.Predict(x.data(), N, out.data());
        check(all_finite(out.data(), 4), "post move-assign Predict finite");
    }

    end_section();
}

// ---------------------------------------------------------------------------
//  3. Forward / train classification (consolidated)
// ---------------------------------------------------------------------------

static void section_forward_train() {
    begin_section("Forward / train classification");

    constexpr int DIM = 5;
    constexpr int K = 4;
    constexpr int n_train = 20;
    constexpr int steps = 40;

    // Embed + Forward finite
    {
        HCNN net(DIM, K);
        net.AddConv(8);
        net.AddPool(PoolType::MAX);
        net.AddConv(16);
        net.RandomizeWeights();

        const int N = net.GetStartN();
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> input(N);
        for (auto& v : input) v = dist(rng);

        std::vector<float> emb(N), logits(K);
        net.Embed(input.data(), N, emb.data());
        check(all_finite(emb.data(), N), "Embed produces finite values");
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), K), "Forward produces finite logits");

        // Predict == Embed + Forward; PredictClass == argmax
        std::vector<float> pred(K);
        net.Predict(input.data(), N, pred.data());
        check(all_finite(pred.data(), K), "Predict produces finite outputs");
        float max_diff = 0.0f;
        for (int i = 0; i < K; ++i)
            max_diff = std::max(max_diff, std::abs(pred[static_cast<size_t>(i)]
                                                   - logits[static_cast<size_t>(i)]));
        check(max_diff < 1e-6f, "Predict matches Embed+Forward");
        const int cls = net.PredictClass(input.data(), N);
        check(cls >= 0 && cls < K, "PredictClass in range");
        int argmax_i = 0;
        for (int i = 1; i < K; ++i)
            if (pred[static_cast<size_t>(i)] > pred[static_cast<size_t>(argmax_i)])
                argmax_i = i;
        check(cls == argmax_i, "PredictClass matches argmax of Predict");
    }

    // TrainParams overloads
    {
        HCNN net(DIM, K);
        net.AddConv(8);
        net.RandomizeWeights();
        const int N = net.GetStartN();
        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(8, N, K, 77, inputs, targets);
        auto flat = flatten_inputs(inputs, N);

        TrainParams p;
        p.learning_rate = 0.02f;
        p.shuffle_seed = 3u;
        net.TrainStep(inputs[0].data(), N, targets[0], p);
        net.TrainBatch(flat.data(), N, targets.data(), 8, p);
        net.TrainEpoch(flat.data(), N, targets.data(), 8, 4, p);

        std::vector<float> out(K);
        net.Predict(inputs[0].data(), N, out.data());
        check(all_finite(out.data(), K), "TrainParams path: Predict finite");
    }

    // TrainStep loss drop (~40 steps)
    {
        HCNN net(DIM, K);
        net.AddConv(16);
        net.RandomizeWeights();
        const int N = net.GetStartN();

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(n_train, N, K, 123, inputs, targets);

        std::vector<float> emb(N), logits(K);
        double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(std::isfinite(loss_before), "TrainStep: initial loss finite");

        for (int step = 0; step < steps; ++step) {
            int idx = step % n_train;
            net.TrainStep(inputs[idx].data(), N, targets[idx], 0.01f);
        }
        double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(std::isfinite(loss_after), "TrainStep: loss after finite");
        check(loss_after < loss_before,
              "TrainStep: loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
    }

    // TrainBatch finite
    {
        HCNN net(DIM, K);
        net.AddConv(16);
        net.RandomizeWeights();
        const int N = net.GetStartN();
        const int batch_size = 8;

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(batch_size, N, K, 456, inputs, targets);
        auto flat = flatten_inputs(inputs, N);
        net.TrainBatch(flat.data(), N, targets.data(), batch_size, 0.01f);

        std::vector<float> emb(N), logits(K);
        net.Embed(inputs[0].data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), K), "TrainBatch: logits finite");
    }

    // TrainEpoch shuffle + no-shuffle
    {
        HCNN net(DIM, K);
        net.AddConv(16);
        net.RandomizeWeights();
        const int N = net.GetStartN();

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(16, N, K, 999, inputs, targets);
        auto flat = flatten_inputs(inputs, N);

        std::vector<float> emb(N), logits(K);
        double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

        net.TrainEpoch(flat.data(), N, targets.data(), 16, /*batch_size=*/8,
                       /*lr=*/0.05f, /*momentum=*/0.0f, /*wd=*/0.0f,
                       /*class_weights=*/nullptr, /*shuffle_seed=*/1u);
        net.TrainEpoch(flat.data(), N, targets.data(), 16, 8, 0.05f, 0.0f, 0.0f,
                       nullptr, 2u);

        double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(loss_after < loss_before,
              "TrainEpoch (shuffled): loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");

        net.TrainEpoch(flat.data(), N, targets.data(), 16, 8, 0.01f, 0.0f, 0.0f,
                       nullptr, /*shuffle_seed=*/0u);
        net.Embed(inputs[0].data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), K), "TrainEpoch (no shuffle): logits finite");
    }

    // ForwardBatch matches single-sample
    {
        HCNN net(DIM, K);
        net.AddConv(16);
        net.AddPool(PoolType::MAX);
        net.RandomizeWeights();
        const int N = net.GetStartN();
        const int batch_size = 8;

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(batch_size, N, K, 789, inputs, targets);
        auto flat = flatten_inputs(inputs, N);

        std::vector<float> all_logits(static_cast<size_t>(batch_size) * K);
        net.ForwardBatch(flat.data(), N, batch_size, all_logits.data());
        check(all_finite(all_logits.data(), batch_size * K),
              "ForwardBatch: all logits finite");

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

    // MAX / AVG pool finite forward; short AVG train drop
    {
        for (PoolType type : {PoolType::MAX, PoolType::AVG}) {
            HCNN net(DIM, K);
            net.AddConv(8);
            net.AddPool(type);
            net.RandomizeWeights();
            const int N = net.GetStartN();
            std::vector<float> input(N, 0.2f), emb(N), logits(K);
            net.Embed(input.data(), N, emb.data());
            net.Forward(emb.data(), logits.data());
            const char* name = (type == PoolType::MAX) ? "MAX" : "AVG";
            check(all_finite(logits.data(), K),
                  std::string(name) + " pool produces finite logits");
        }

        HCNN net(DIM, K);
        net.AddConv(16);
        net.AddPool(PoolType::AVG);
        net.AddConv(16);
        net.RandomizeWeights();
        const int N = net.GetStartN();

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(n_train, N, K, 42, inputs, targets);
        std::vector<float> emb(N), logits(K);
        double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        for (int step = 0; step < steps; ++step) {
            int idx = step % n_train;
            net.TrainStep(inputs[idx].data(), N, targets[idx], 0.01f);
        }
        double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(loss_after < loss_before,
              "AVG pool: loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
    }

    // Adam short loss drop
    {
        HCNN net(DIM, K);
        net.AddConv(16);
        net.RandomizeWeights();
        net.SetOptimizer(OptimizerType::ADAM);
        const int N = net.GetStartN();

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(n_train, N, K, 42, inputs, targets);
        std::vector<float> emb(N), logits(K);
        double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        for (int step = 0; step < steps; ++step) {
            int idx = step % n_train;
            net.TrainStep(inputs[idx].data(), N, targets[idx], 0.001f);
        }
        double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(loss_after < loss_before,
              "Adam TrainStep: loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
    }

    // weight_decay: finite after steps (loss drop optional)
    {
        HCNN net(DIM, K);
        net.AddConv(16);
        net.RandomizeWeights();
        const int N = net.GetStartN();

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(n_train, N, K, 123, inputs, targets);
        for (int step = 0; step < steps; ++step) {
            int idx = step % n_train;
            net.TrainStep(inputs[idx].data(), N, targets[idx],
                          /*lr=*/0.01f, /*momentum=*/0.0f, /*weight_decay=*/0.01f);
        }
        std::vector<float> emb(N), logits(K);
        net.Embed(inputs[0].data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), K), "weight_decay: logits finite after steps");
    }

    // class weights: finite
    {
        HCNN net(DIM, K);
        net.AddConv(16);
        net.RandomizeWeights();
        const int N = net.GetStartN();
        std::vector<float> input(N, 0.15f);
        std::vector<float> class_weights = {10.0f, 1.0f, 1.0f, 1.0f};
        net.TrainStep(input.data(), N, 0, 0.01f, 0.0f, 0.0f, class_weights.data());

        const int batch_size = 4;
        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(batch_size, N, K, 42, inputs, targets);
        auto flat = flatten_inputs(inputs, N);
        net.TrainBatch(flat.data(), N, targets.data(), batch_size,
                       0.01f, 0.0f, 0.0f, class_weights.data());

        std::vector<float> emb(N), logits(K);
        net.Embed(inputs[0].data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), K), "class weights: logits finite");
    }

    end_section();
}

// ---------------------------------------------------------------------------
//  4. Batch norm
// ---------------------------------------------------------------------------

static void section_batchnorm() {
    begin_section("Batch normalization");

    // BN forward finite
    {
        HCNN net(5, 4);
        net.AddConv(16, Activation::RELU, true, /*use_batchnorm=*/true);
        net.AddPool(PoolType::MAX);
        net.AddConv(16, Activation::RELU, true, true);
        net.RandomizeWeights();
        const int N = net.GetStartN();
        const int K = net.GetNumOutputs();
        std::vector<float> input(N, 0.3f), emb(N), logits(K);
        net.Embed(input.data(), N, emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), K), "BN forward produces finite logits");
    }

    // Short BN train drop
    {
        HCNN net(5, 4);
        net.AddConv(16, Activation::RELU, true, true);
        net.RandomizeWeights();
        const int N = net.GetStartN();
        const int K = net.GetNumOutputs();

        std::vector<std::vector<float>> inputs;
        std::vector<int> targets;
        make_synth(20, N, K, 123, inputs, targets);

        std::vector<float> emb(N), logits(K);
        net.SetTraining(false);
        double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);

        for (int step = 0; step < 40; ++step) {
            int idx = step % 20;
            net.TrainStep(inputs[idx].data(), N, targets[idx], 0.01f);
        }
        net.SetTraining(false);
        double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(loss_after < loss_before,
              "BN TrainStep: loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
    }

    // bn_save required for BN backward
    {
        const int N = 32;
        hcnn::HCNNConv bn_layer(5, 1, 2, Activation::RELU, true, /*bn=*/true);
        std::mt19937 rng(1);
        bn_layer.randomize_weights(0.0f, rng);
        std::vector<float> in(static_cast<size_t>(N), 0.1f);
        std::vector<float> out(static_cast<size_t>(2 * N));
        std::vector<float> pre(static_cast<size_t>(2 * N));
        std::vector<float> save(static_cast<size_t>(bn_layer.get_bn_save_size()));
        bn_layer.forward(in.data(), out.data(), pre.data(), save.data());
        std::vector<float> gout(static_cast<size_t>(2 * N), 0.01f);
        std::vector<float> kg(static_cast<size_t>(bn_layer.get_kernel_size()));
        std::vector<float> bg(static_cast<size_t>(bn_layer.get_bias_size()));
        check(throws([&] {
            bn_layer.compute_gradients(gout.data(), in.data(), pre.data(),
                                       nullptr, kg.data(), bg.data(),
                                       nullptr, /*bn_save=*/nullptr);
        }), "BN compute_gradients without bn_save throws");
        bn_layer.compute_gradients(gout.data(), in.data(), pre.data(),
                                   nullptr, kg.data(), bg.data(),
                                   nullptr, save.data());
        check(all_finite(kg.data(), bn_layer.get_kernel_size()),
              "BN compute_gradients with bn_save produces finite kernel grads");
    }

    // Weight blob BN round-trip
    {
        HCNN net_bn(5, 4);
        net_bn.AddConv(8, Activation::RELU, true, /*bn=*/true);
        net_bn.RandomizeWeights();
        // non-BN: 1*8*6 + 8 + 8*32*4 + 4; BN adds 4 * c_out
        const size_t expected_bn =
            static_cast<size_t>(1 * 8 * 6 + 8 + 8 * 32 * 4 + 4 + 4 * 8);
        check(net_bn.GetWeightCount() == expected_bn,
              "GetWeightCount includes BN gamma/beta/running stats");

        auto w = net_bn.GetWeights();
        w[static_cast<size_t>(1 * 8 * 6 + 8)] = 2.5f;  // first float of gamma
        net_bn.SetWeights(w, /*reset_optimizer_moments=*/true);
        auto w2 = net_bn.GetWeights();
        check(std::fabs(w2[static_cast<size_t>(1 * 8 * 6 + 8)] - 2.5f) < 1e-6f,
              "BN gamma survives Get/SetWeights round-trip");
    }

    // Forward preserves training mode (BN footgun)
    {
        HCNN net(5, 4);
        net.AddConv(8, Activation::RELU, true, /*use_batchnorm=*/true);
        net.RandomizeWeights();
        net.SetTraining(true);

        std::vector<float> input(net.GetStartN(), 0.25f);
        std::vector<float> emb(net.GetStartN()), logits(net.GetNumOutputs());
        net.Embed(input.data(), net.GetStartN(), emb.data());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), net.GetNumOutputs()),
              "Forward in training mode: logits finite");
        net.TrainStep(input.data(), net.GetStartN(), 0, 0.01f);
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), net.GetNumOutputs()),
              "TrainStep + Forward after training-mode Forward: logits finite");
    }

    end_section();
}

// ---------------------------------------------------------------------------
//  5. Activations
// ---------------------------------------------------------------------------

static void section_activations() {
    begin_section("Activations");

    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    make_synth(20, 32, 4, 123, inputs, targets);

    // LEAKY + TANH finite forward
    for (Activation act : {Activation::LEAKY_RELU, Activation::TANH}) {
        HCNN net(5, 4);
        net.AddConv(16, act);
        net.RandomizeWeights();
        std::vector<float> emb(32), logits(4);
        net.Embed(inputs[0].data(), 32, emb.data());
        net.Forward(emb.data(), logits.data());
        const char* name = (act == Activation::LEAKY_RELU) ? "LeakyReLU" : "Tanh";
        check(all_finite(logits.data(), 4),
              std::string(name) + " forward produces finite logits");
    }

    // Short loss drop for TANH only
    {
        HCNN net(5, 4);
        net.AddConv(16, Activation::TANH);
        net.RandomizeWeights();
        std::vector<float> emb(32), logits(4);
        double loss_before = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        for (int step = 0; step < 40; ++step) {
            int idx = step % 20;
            net.TrainStep(inputs[idx].data(), 32, targets[idx], 0.01f);
        }
        double loss_after = cross_entropy_over_samples(net, inputs, targets, emb, logits);
        check(loss_after < loss_before,
              "Tanh loss decreased ("
              + std::to_string(loss_before) + " -> " + std::to_string(loss_after) + ")");
    }

    // Stacked TANH sanity
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

    end_section();
}

// ---------------------------------------------------------------------------
//  6. Contracts (invalid args, embed pad, lifecycle, readout grad_in)
// ---------------------------------------------------------------------------

static void section_contracts() {
    begin_section("Contracts");

    // Invalid args
    {
        HCNN net(5, 4);
        net.AddConv(8);
        net.RandomizeWeights();
        std::vector<float> dummy_input(net.GetStartN(), 0.0f);
        int N = net.GetStartN();
        int targets[1] = {0};
        std::vector<float> logits_out(net.GetNumOutputs());

        check(throws([&] { net.ForwardBatch(dummy_input.data(), N, 0, logits_out.data()); }),
              "ForwardBatch(batch_size=0) throws");
        check(throws([&] { net.TrainBatch(dummy_input.data(), N, targets, 0, 0.01f); }),
              "TrainBatch(batch_size=0) throws");
        check(throws([&] { net.TrainEpoch(dummy_input.data(), N, targets, 1, 0, 0.01f); }),
              "TrainEpoch(batch_size=0) throws");
        check(throws([&] { HCNN bad(31, 2); }), "HCNN(start_dim=31) throws (max 30)");
        check(throws([&] { HCNN bad(2, 2); }), "HCNN(start_dim=2) throws (min 3)");
    }

    // Embed pad / trunc
    {
        HCNN net(5, 4);
        net.AddConv(8);
        net.RandomizeWeights();
        const int N = net.GetStartN();

        std::vector<float> short_input(N - 5, 0.5f);
        std::vector<float> emb(N, -123.0f);
        net.Embed(short_input.data(), static_cast<int>(short_input.size()), emb.data());
        bool front_ok = true;
        for (int i = 0; i < static_cast<int>(short_input.size()); ++i)
            if (emb[i] != 0.5f) { front_ok = false; break; }
        bool tail_zeroed = true;
        for (int i = static_cast<int>(short_input.size()); i < N; ++i)
            if (emb[i] != 0.0f) { tail_zeroed = false; break; }
        check(front_ok, "Embed: front of short input copied verbatim");
        check(tail_zeroed, "Embed: tail of short input zero-padded");

        std::vector<float> logits(net.GetNumOutputs());
        net.Forward(emb.data(), logits.data());
        check(all_finite(logits.data(), net.GetNumOutputs()),
              "Forward on zero-padded embedding: logits finite");

        std::vector<float> oversized(N + 4, 0.0f);
        check(throws([&] {
            net.Embed(oversized.data(), static_cast<int>(oversized.size()), emb.data());
        }), "Embed: over-capacity input length throws");
    }

    // Network lifecycle (private HCNNNetwork — in-tree only)
    {
        {
            HCNNNetwork net(5, 4, 1, TaskType::Classification,
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

        {
            HCNNNetwork net(5, 4, 1, TaskType::Classification,
                            /*num_threads=*/1);
            net.add_conv(8);
            net.prepare_all_buffers();
            net.randomize_all_weights(0.0f, 11);
            check(net.get_readout().get_num_features() == 8 * 32,
                  "full head after prepare-then-randomize");
            const int N = net.get_start_N();
            std::vector<float> x(static_cast<size_t>(N), 0.1f);
            int target = 1;
            bool ok = true;
            try {
                net.train_step(x.data(), N, target, 0.01f);
                net.train_batch(x.data(), N, &target, 1, 0.01f);
            } catch (const std::exception&) {
                ok = false;
            }
            check(ok, "train_step/batch after prepare-then-randomize");
        }

        {
            HCNNNetwork net(5, 4, 1, TaskType::Classification,
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

        {
            HCNNNetwork net(5, 4, 1, TaskType::Classification,
                            /*num_threads=*/1);
            net.set_optimizer(OptimizerType::ADAM);
            net.add_conv(8);
            net.randomize_all_weights(0.0f, 1);
            check(net.get_readout().get_optimizer_type() == OptimizerType::ADAM,
                  "readout Adam when set_optimizer precedes add_conv");
        }

        {
            HCNNNetwork net(3, 2, 1, TaskType::Classification,
                            /*num_threads=*/1);
            net.add_conv(4);
            net.add_pool(PoolType::MAX);
            check(net.get_current_dim() == 2, "current_dim 2 after first pool");
            net.add_pool(PoolType::MAX);
            check(net.get_current_dim() == 1, "current_dim 1 after second pool");
            check(throws([&] { net.add_pool(PoolType::MAX); }),
                  "add_pool at current_dim=1 throws");
        }
    }

    // Readout grad_in A/B (advanced HCNNReadout only — not on HCNN facade)
    {
        HCNNNetwork net(5, 4, 1, TaskType::Classification, /*num_threads=*/1);
        net.add_conv(8);
        check(net.get_readout().get_grad_in_loop() == ReadoutGradInLoop::OutputOuter,
              "default grad_in loop is OutputOuter");
        net.get_readout().set_grad_in_loop(ReadoutGradInLoop::FeatureOuter);
        net.randomize_all_weights(0.0f, 99);
        check(net.get_readout().get_grad_in_loop() == ReadoutGradInLoop::FeatureOuter,
              "set_grad_in_loop survives randomize_all_weights");
        net.get_readout().set_grad_in_loop(ReadoutGradInLoop::OutputOuter);
        check(net.get_readout().get_grad_in_loop() == ReadoutGradInLoop::OutputOuter,
              "can switch back to OutputOuter");
    }

    {
        // Smaller head than the old 16*2048 microbench: still exercises both loops.
        const int O = 10;
        const int F = 16 * 64;
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
        for (int f = 0; f < F; ++f)
            max_abs = std::max(max_abs, std::abs(gin_a[f] - gin_b[f]));
        check(max_abs == 0.0f || max_abs < 1e-5f,
              "FeatureOuter vs OutputOuter grad_in match (max_abs="
              + std::to_string(max_abs) + ")");
    }

    end_section();
}

// ---------------------------------------------------------------------------
//  7. Regression
// ---------------------------------------------------------------------------

static void section_regression() {
    begin_section("Regression");

    // Predict works; PredictClass rejects regression nets
    {
        HCNN net(5, 1, 1, TaskType::Regression);
        net.AddConv(4);
        net.RandomizeWeights();
        const int N = net.GetStartN();
        std::vector<float> x(N, 0.1f), y(1);
        net.Predict(x.data(), N, y.data());
        check(std::isfinite(y[0]), "Regression Predict finite");
        check(throws([&] { (void)net.PredictClass(x.data(), N); }),
              "PredictClass throws on Regression");

        TrainParams p;
        p.learning_rate = 0.05f;
        float t = 0.25f;
        net.TrainStep(x.data(), N, &t, p);  // float* target → regression
        check(std::isfinite(y[0]), "TrainParams regression step ok");
    }

    // Scalar: step loss drop + batch/epoch finite
    {
        const int DIM = 6;
        const int num_outputs = 1;
        HCNN net(DIM, num_outputs, /*input_channels=*/1, TaskType::Regression);
        net.AddConv(16);
        net.AddPool(PoolType::MAX);
        net.AddConv(16);
        net.RandomizeWeights();

        check(net.GetNumOutputs() == 1, "GetNumOutputs() == 1");
        check(net.GetTaskType() == TaskType::Regression, "GetTaskType() == Regression");

        const int N = net.GetStartN();
        const int n_train = 20;

        std::vector<std::vector<float>> inputs;
        std::vector<float> targets;
        make_synth_regression_scalar(n_train, N, /*seed=*/7, inputs, targets);
        auto flat_inputs = flatten_inputs(inputs, N);

        std::vector<float> embedded(N), preds(num_outputs);
        double mse_before = mse_over_samples(net, inputs, targets, embedded, preds);
        check(std::isfinite(mse_before), "scalar: initial MSE finite");

        for (int step = 0; step < 40; ++step) {
            int i = step % n_train;
            net.TrainStep(inputs[i].data(), N, &targets[i],
                          /*lr=*/0.05f, /*momentum=*/0.9f);
        }
        double mse_after_step = mse_over_samples(net, inputs, targets, embedded, preds);
        check(mse_after_step < mse_before,
              "TrainStep(float*): MSE decreased ("
              + std::to_string(mse_before) + " -> "
              + std::to_string(mse_after_step) + ")");

        net.TrainBatch(flat_inputs.data(), N, targets.data(), n_train,
                       /*lr=*/0.05f, /*momentum=*/0.9f);
        double mse_batch = mse_over_samples(net, inputs, targets, embedded, preds);
        check(std::isfinite(mse_batch), "TrainBatch(float*): MSE finite");

        net.TrainEpoch(flat_inputs.data(), N, targets.data(),
                       n_train, /*batch_size=*/10,
                       /*lr=*/0.05f, /*momentum=*/0.9f,
                       /*weight_decay=*/1e-4f, /*shuffle_seed=*/1u);
        double mse_epoch = mse_over_samples(net, inputs, targets, embedded, preds);
        check(std::isfinite(mse_epoch), "TrainEpoch(float*): MSE finite");
        check(all_finite(preds.data(), num_outputs), "regression preds finite");
        net.TrainEpochRegression(flat_inputs.data(), N, targets.data(),
                                 n_train, 10, 0.01f);
        check(std::isfinite(mse_over_samples(net, inputs, targets, embedded, preds)),
              "TrainEpochRegression alias still works");
    }

    // Multi-output shorter
    {
        const int DIM = 6;
        const int num_outputs = 3;
        HCNN net(DIM, num_outputs, 1, TaskType::Regression);
        net.AddConv(16);
        net.AddPool(PoolType::MAX);
        net.RandomizeWeights();

        const int N = net.GetStartN();
        const int n_train = 16;

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
        std::vector<float> flat_targets(static_cast<size_t>(n_train) * num_outputs);
        for (int i = 0; i < n_train; ++i)
            std::copy(targets[i].begin(), targets[i].end(),
                      flat_targets.begin() + i * num_outputs);

        std::vector<float> embedded(N), preds(num_outputs);
        auto mean_mse = [&]() {
            double total = 0.0;
            for (int i = 0; i < n_train; ++i) {
                net.Embed(inputs[i].data(), N, embedded.data());
                net.Forward(embedded.data(), preds.data());
                for (int k = 0; k < num_outputs; ++k) {
                    double d = preds[k] - targets[i][k];
                    total += d * d;
                }
            }
            return total / (n_train * num_outputs);
        };

        double mse_before = mean_mse();
        for (int e = 0; e < 3; ++e)
            net.TrainEpoch(flat_inputs.data(), N, flat_targets.data(),
                           n_train, /*batch_size=*/8, /*lr=*/0.05f,
                           /*momentum=*/0.9f, /*wd=*/0.0f,
                           /*shuffle_seed=*/static_cast<unsigned>(e + 1));
        double mse_after = mean_mse();
        check(std::isfinite(mse_after), "multi-output MSE finite");
        check(mse_after < mse_before,
              "multi-output MSE decreased ("
              + std::to_string(mse_before) + " -> " + std::to_string(mse_after) + ")");
    }

    // API misuse
    {
        {
            HCNN net(5, /*num_outputs=*/2, 1, TaskType::Regression);
            net.AddConv(8);
            net.RandomizeWeights();
            const int N = net.GetStartN();
            std::vector<float> input(N, 0.1f);
            int target = 0;
            check(throws([&] { net.TrainStep(input.data(), N, 0, 0.01f); }),
                  "TrainStep on Regression net throws logic_error");
            check(throws([&] { net.TrainBatch(input.data(), N, &target, 1, 0.01f); }),
                  "TrainBatch on Regression net throws logic_error");
        }
        {
            HCNN net(5, /*num_outputs=*/2, 1, TaskType::Classification);
            net.AddConv(8);
            net.RandomizeWeights();
            const int N = net.GetStartN();
            std::vector<float> input(N, 0.1f);
            std::vector<float> target(2, 0.0f);
            check(throws([&] {
                net.TrainStep(input.data(), N, target.data(), 0.01f);
            }), "TrainStep(float*) on Classification net throws logic_error");
            check(throws([&] {
                net.TrainBatch(input.data(), N, target.data(), 1, 0.01f);
            }), "TrainBatch(float*) on Classification net throws logic_error");
            check(throws([&] {
                net.TrainStepRegression(input.data(), N, target.data(), 0.01f);
            }), "TrainStepRegression alias still throws on Classification");
        }
    }

    // Task construction (loss is fixed by task; no LossType on the API)
    {
        HCNN cls(5, 4, 1, TaskType::Classification);
        check(cls.GetTaskType() == TaskType::Classification,
              "Classification ctor sets task");
        HCNN reg(5, 1, 1, TaskType::Regression);
        check(reg.GetTaskType() == TaskType::Regression,
              "Regression ctor sets task");
        // Positional: num_threads is 5th arg (was 6th when LossType existed)
        HCNN thr(5, 4, 1, TaskType::Classification, /*num_threads=*/1);
        check(thr.GetTaskType() == TaskType::Classification,
              "5-arg ctor with num_threads");
    }

    end_section();
}

// ---------------------------------------------------------------------------
//  8. Spatial augmentation
// ---------------------------------------------------------------------------

static void section_spatial_aug() {
    begin_section("Spatial augmentation");

    // Identity / disabled
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

    {
        HCNNSpatialAugConfig cfg;
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

    // Determinism
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
        { std::mt19937 rng(12345); aug.apply(src.data(), a.data(), H, W, rng); }
        { std::mt19937 rng(12345); aug.apply(src.data(), b.data(), H, W, rng); }
        bool same = true;
        for (int i = 0; i < n; ++i) if (a[i] != b[i]) same = false;
        check(same, "fixed seed reproduces geometric aug");

        std::vector<float> c(n);
        { std::mt19937 rng(99999); aug.apply(src.data(), c.data(), H, W, rng); }
        bool differ = false;
        for (int i = 0; i < n; ++i) if (c[i] != a[i]) { differ = true; break; }
        check(differ, "different seed changes geometric aug");
    }

    // Noise clip
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

    // Shear: finite + change
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

        std::vector<float> a(n);
        std::mt19937 rng(4242);
        aug.apply(src.data(), a.data(), H, W, rng);
        bool differ = false;
        for (int i = 0; i < n; ++i) if (a[i] != src[i]) { differ = true; break; }
        check(differ, "shear aug changes a horizontal gradient");
        check(all_finite(a.data(), n), "shear aug produces finite values");
    }

    // Elastic: validation + determinism
    {
        HCNNSpatialAugConfig bad0;
        bad0.elastic_alpha = 1.0f;
        bad0.elastic_sigma = 0.0f;
        check(throws([&] { HCNNSpatialAugmenter aug(bad0); (void)aug; }),
              "elastic_alpha without elastic_sigma rejected");

        HCNNSpatialAugConfig bad_lo;
        bad_lo.elastic_alpha = 1.0f;
        bad_lo.elastic_sigma = 0.1f;
        check(throws([&] { HCNNSpatialAugmenter aug(bad_lo); (void)aug; }),
              "elastic_sigma below min rejected");

        HCNNSpatialAugConfig bad_hi;
        bad_hi.elastic_alpha = 1.0f;
        bad_hi.elastic_sigma = 100.0f;
        check(throws([&] { HCNNSpatialAugmenter aug(bad_hi); (void)aug; }),
              "elastic_sigma above max rejected");

        HCNNSpatialAugConfig bad_shear;
        bad_shear.shear_x_max = 1.0f;
        bad_shear.shear_y_max = 1.0f;
        check(throws([&] { HCNNSpatialAugmenter aug(bad_shear); (void)aug; }),
              "near-singular shear product rejected");

        HCNNSpatialAugConfig cfg;
        cfg.elastic_alpha = 1.5f;
        cfg.elastic_sigma = 4.0f;
        cfg.border_value = -1.0f;
        HCNNSpatialAugmenter aug(cfg);
        const int H = 16, W = 16, n = H * W;
        std::vector<float> src(n);
        for (int i = 0; i < n; ++i)
            src[i] = std::sin(0.2f * static_cast<float>(i));

        std::vector<float> a(n), b(n);
        { std::mt19937 rng(777); aug.apply(src.data(), a.data(), H, W, rng); }
        { std::mt19937 rng(777); aug.apply(src.data(), b.data(), H, W, rng); }
        bool same = true;
        for (int i = 0; i < n; ++i) if (a[i] != b[i]) same = false;
        check(same, "fixed seed reproduces elastic aug");

        bool differ = false;
        for (int i = 0; i < n; ++i)
            if (std::fabs(a[i] - src[i]) > 1e-6f) { differ = true; break; }
        check(differ, "elastic aug changes a sinusoidal field");
    }

    // Invalid configs / geometric in==out / batch
    {
        HCNNSpatialAugmenter aug;
        std::mt19937 rng(0);
        float x = 0.0f;
        check(throws([&] { aug.apply(&x, &x, 0, 4, rng); }),
              "apply rejects height < 1");
    }
    {
        HCNNSpatialAugConfig cfg;
        cfg.rot_deg_max = 10.0f;
        HCNNSpatialAugmenter aug(cfg);
        std::vector<float> buf(16, 1.0f);
        std::mt19937 rng(0);
        check(throws([&] { aug.apply(buf.data(), buf.data(), 4, 4, rng); }),
              "geometric aug rejects in == out");
    }
    {
        HCNNSpatialAugConfig cfg;
        cfg.value_min = 1.0f;
        cfg.value_max = -1.0f;
        check(throws([&] { HCNNSpatialAugmenter aug(cfg); (void)aug; }),
              "value_min > value_max rejected");
    }
    {
        HCNNSpatialAugConfig cfg;
        cfg.noise_sigma = -0.1f;
        check(throws([&] { HCNNSpatialAugmenter aug(cfg); (void)aug; }),
              "negative noise_sigma rejected");
    }
    {
        HCNNSpatialAugConfig cfg;
        cfg.rot_deg_max = 5.0f;
        cfg.border_value = 0.0f;
        HCNNSpatialAugmenter aug(cfg);
        const int B = 3, H = 5, W = 6, plane = H * W;
        std::vector<float> src(B * plane, 0.25f), dst(B * plane, 0.0f);
        std::mt19937 rng(11);
        aug.apply_batch(src.data(), dst.data(), B, H, W, rng);
        check(all_finite(dst.data(), B * plane), "apply_batch produces finite values");
    }

    end_section();
}

// ---------------------------------------------------------------------------
//  9. Spatial embed
// ---------------------------------------------------------------------------

static void section_spatial_embed() {
    begin_section("Spatial embed");

    // Capacity helpers
    check(HCNNSpatialEmbedder::max_dual_plane_side(2048) == 32,
          "max_dual_plane_side(2048) == 32");
    check(HCNNSpatialEmbedder::max_dual_plane_side(512) == 16,
          "max_dual_plane_side(512) == 16");
    check(HCNNSpatialEmbedder::max_square_side(512) == 22,
          "max_square_side(512) == 22");
    check(HCNNSpatialEmbedder::max_square_side(2048) *
              HCNNSpatialEmbedder::max_square_side(2048) <= 2048,
          "max_square_side(2048) is valid");

    // RowMajorPad
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 6;
        cfg.mode = HCNNSpatialEmbedMode::RowMajorPad;
        cfg.pad_value = -1.0f;
        HCNNSpatialEmbedder emb(cfg);
        check(emb.capacity() == 64, "capacity 2^6 == 64");

        const int H = 4, W = 5;
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

    // DualPlane full occupancy
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

        std::vector<float> src(28 * 28);
        for (int y = 0; y < 28; ++y)
            for (int x = 0; x < 28; ++x)
                src[y * 28 + x] = (x > 14) ? 1.0f : -1.0f;
        std::vector<float> out(emb.capacity());
        emb.embed(src.data(), 28, 28, out.data());
        check(all_finite(out.data(), emb.capacity()), "DualPlane produces finite values");

        bool grad_alive = false;
        for (int i = 1024; i < 2048; ++i)
            if (out[i] != cfg.pad_value) { grad_alive = true; break; }
        check(grad_alive, "DualPlane |grad| plane has structure");
    }

    // Reject oversize
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 5;
        cfg.mode = HCNNSpatialEmbedMode::RowMajorPad;
        HCNNSpatialEmbedder emb(cfg);
        std::vector<float> src(64, 0.0f), out(32);
        check(throws([&] { emb.embed(src.data(), 8, 8, out.data()); }),
              "RowMajorPad rejects H*W > N");
    }
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 8;
        cfg.mode = HCNNSpatialEmbedMode::DualPlaneResize;
        cfg.plane_side = 20;
        check(throws([&] { HCNNSpatialEmbedder emb(cfg); (void)emb; }),
              "oversized plane_side rejected at construct");
    }

    // Batch
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

    // Train chain (input_length = N)
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
        net.TrainStep(packed.data(), emb.capacity(), 0, 0.01f, 0.9f, 0.0f);
        net.ForwardBatch(packed.data(), emb.capacity(), 1, logits.data());
        check(std::isfinite(logits[0]) && std::isfinite(logits[1]),
              "embedded vector trains/infers on HCNN");
    }

    // HCNN Embed zero-pad contract vs spatial pad_value
    {
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 6;
        cfg.mode = HCNNSpatialEmbedMode::RowMajorPad;
        cfg.pad_value = -1.0f;
        HCNNSpatialEmbedder emb(cfg);
        std::vector<float> src(4 * 4, 0.5f), packed(emb.capacity());
        emb.embed(src.data(), 4, 4, packed.data());
        check(packed[16] == -1.0f, "spatial embed pad_value on unused verts");

        HCNN net(6, 2);
        std::vector<float> embedded(net.GetStartN());
        net.Embed(packed.data(), 16, embedded.data());
        check(embedded[16] == 0.0f,
              "HCNN::Embed zero-pads short length (overrides non-zero pad)");
    }

    end_section();
}

// ---------------------------------------------------------------------------
//  10. Train helpers
// ---------------------------------------------------------------------------

static void section_train_helpers() {
    begin_section("Train helpers");

    // argmax / CE
    {
        float v[] = {0.1f, 0.9f, 0.3f};
        check(argmax(v, 3) == 1, "argmax picks max index");
        float logits[] = {0.0f, 2.0f, 0.0f};
        float ce_good = softmax_cross_entropy(logits, 3, 1);
        float ce_bad  = softmax_cross_entropy(logits, 3, 0);
        check(std::isfinite(ce_good) && ce_good > 0.0f, "CE finite and positive");
        check(ce_good < ce_bad, "CE lower for correct class");
    }

    // cosine_lr
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

    // Dual-ckpt tie-breaks (compact)
    {
        HCNN net(5, 4);
        net.AddConv(8);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/11);

        HCNNDualCheckpoint ckpt;
        auto u1 = ckpt.observe(net, /*loss=*/1.0f, /*accuracy=*/50.0f, /*epoch=*/1);
        check(u1.new_best_loss && u1.new_best_acc, "dual-ckpt: first observe both bests");

        auto u2 = ckpt.observe(net, 1.0f, 55.0f, 2);
        check(u2.new_best_loss && u2.new_best_acc,
              "dual-ckpt: equal loss higher acc updates both");

        auto u3 = ckpt.observe(net, 1.0f, 55.0f, 3);
        check(!u3.any(), "dual-ckpt: equal loss and acc is no-op");

        auto u4 = ckpt.observe(net, 0.95f, 55.0f, 4);
        check(u4.new_best_loss && u4.new_best_acc,
              "dual-ckpt: lower loss updates both");

        auto u5 = ckpt.observe(net, 0.95f, 54.0f, 5);
        check(!u5.new_best_loss && !u5.new_best_acc,
              "dual-ckpt: equal loss lower acc skips both");
        check(ckpt.best_loss_epoch() == 4, "dual-ckpt: best-loss epoch stays 4");
    }

    // evaluate empty throws
    {
        HCNN net(5, 4);
        net.AddConv(8);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/3);

        HCNNFlatDataset empty_ds;
        check(throws([&] { (void)evaluate_classification(net, empty_ds); }),
              "evaluate empty FlatDataset throws");

        HCNNFlatDataset bad;
        bad.reset(4, net.GetStartN());
        bad.count = 8;
        check(throws([&] { (void)evaluate_classification(net, bad); }),
              "evaluate size-drifted FlatDataset throws");

        float logits[] = {0.0f, 1.0f, 0.0f};
        check(throws([&] { (void)softmax_cross_entropy(logits, 3, /*target=*/3); }),
              "softmax_cross_entropy OOR target throws");
    }

    // Checkpoint restore finite (short train + dual + best-metric)
    {
        HCNN net(5, 4);
        net.AddConv(8);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/7);
        net.SetOptimizer(OptimizerType::ADAM);

        const int N = net.GetStartN();
        const int K = net.GetNumOutputs();
        const int n = 16;

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
        check(r0.count == n && std::isfinite(r0.loss),
              "evaluate_classification loss finite");

        HCNNDualCheckpoint ckpt;
        ckpt.observe(net, r0.loss, r0.accuracy, /*epoch=*/1);

        for (int e = 0; e < 2; ++e) {
            const float lr = cosine_lr(0.05f, 0.005f, e, 2);
            net.TrainEpoch(ds.inputs.data(), ds.input_length, ds.targets.data(),
                           ds.count, /*batch_size=*/8, lr, 0.0f, 0.0f, nullptr,
                           static_cast<unsigned>(e + 1));
            auto r = evaluate_classification(net, ds);
            ckpt.observe(net, r.loss, r.accuracy, e + 1);
        }

        ckpt.restore_best_loss(net);
        auto r_loss = evaluate_classification(net, ds);
        check(std::isfinite(r_loss.loss), "restore best-loss: eval finite");

        ckpt.restore_best_acc(net);
        auto r_acc = evaluate_classification(net, ds);
        check(std::isfinite(r_acc.loss), "restore best-acc: eval finite");

        HCNNDualCheckpoint empty;
        check(throws([&] { empty.restore_best_loss(net); }),
              "empty dual-ckpt restore_best_loss throws");
    }

    {
        HCNN net(5, /*num_outputs=*/1, /*input_channels=*/1, TaskType::Regression);
        net.AddConv(8, Activation::TANH);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/3);

        const int N = net.GetStartN();
        const int n = 16;
        std::vector<float> inputs(static_cast<size_t>(n) * N, 0.1f);
        std::vector<float> targets(static_cast<size_t>(n), 0.0f);
        for (int i = 0; i < n; ++i)
            targets[static_cast<size_t>(i)] = 0.25f * static_cast<float>(i % 4);

        auto r0 = evaluate_regression(net, inputs.data(), N, targets.data(), n);
        check(r0.count == n && std::isfinite(r0.mse),
              "evaluate_regression mse finite");

        HCNNBestMetricCheckpoint best;
        check(best.observe(net, static_cast<float>(r0.mse), 1),
              "best-metric first observe is best");
        net.TrainEpoch(inputs.data(), N, targets.data(), n,
                       /*batch=*/8, /*lr=*/0.05f);
        auto r1 = evaluate_regression(net, inputs.data(), N, targets.data(), n);
        best.observe(net, static_cast<float>(r1.mse), 2);
        best.restore(net);
        auto r2 = evaluate_regression(net, inputs.data(), N, targets.data(), n);
        check(std::isfinite(r2.mse), "best-metric restore: eval finite");

        HCNNBestMetricCheckpoint empty;
        check(throws([&] { empty.restore(net); }),
              "empty best-metric restore throws");
    }

    // Unified FlatDataset regression path
    {
        HCNN net(5, /*num_outputs=*/2, 1, TaskType::Regression);
        net.AddConv(4);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/5);
        const int N = net.GetStartN();
        const int n = 8;

        HCNNFlatDataset ds;
        ds.reset_regression(n, N, /*num_outputs=*/2);
        check(ds.has_float_targets() && !ds.has_class_targets(),
              "reset_regression sizes float_targets only");
        for (int i = 0; i < n; ++i) {
            std::fill(ds.sample_input(i), ds.sample_input(i) + N, 0.05f * i);
            ds.sample_float_target(i)[0] = 0.1f * i;
            ds.sample_float_target(i)[1] = -0.05f * i;
        }
        auto r = evaluate_regression(net, ds);
        check(r.count == n && std::isfinite(r.mse),
              "evaluate_regression(FlatDataset) finite");

        TrainParams p;
        p.learning_rate = 0.05f;
        net.TrainEpoch(ds.inputs.data(), ds.input_length,
                       ds.float_targets.data(), ds.count, 4, p);
        check(std::isfinite(evaluate_regression(net, ds).mse),
              "TrainEpoch(float*) from FlatDataset buffers");
    }

    // Pointer Get/SetWeights + versioned save/load
    {
        HCNN net(5, 3);
        net.AddConv(4);
        net.AddPool(PoolType::MAX);
        net.AddConv(4);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/21);

        const size_t n = net.GetWeightCount();
        std::vector<float> buf(n, 0.0f);
        net.GetWeights(buf.data(), n);
        auto v = net.GetWeights();
        check(v.size() == n, "vector GetWeights size");
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
            max_diff = std::max(max_diff, std::abs(buf[i] - v[i]));
        check(max_diff == 0.0f, "pointer GetWeights matches vector form");

        check(throws([&] { net.GetWeights(buf.data(), n + 1); }),
              "GetWeights wrong n throws");
        check(throws([&] { net.SetWeights(static_cast<const float*>(nullptr), n); }),
              "SetWeights null throws");

        // Mutate then restore via pointer SetWeights
        std::vector<float> zeros(n, 0.0f);
        net.SetWeights(zeros.data(), n, /*reset_optimizer_moments=*/true);
        std::vector<float> after_zero(n);
        net.GetWeights(after_zero.data(), n);
        bool all_zero = true;
        for (float x : after_zero)
            if (x != 0.0f) { all_zero = false; break; }
        check(all_zero, "pointer SetWeights zeros blob");
        net.SetWeights(buf.data(), n, true);

        const char* path = "hcnn_smoke_weights_v1.bin";
        save_weights(net, path);

        // Corrupt live weights then reload
        net.SetWeights(zeros.data(), n, true);
        load_weights(net, path, /*reset_optimizer_moments=*/true);
        std::vector<float> restored(n);
        net.GetWeights(restored.data(), n);
        max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
            max_diff = std::max(max_diff, std::abs(restored[i] - buf[i]));
        check(max_diff == 0.0f, "save/load_weights round-trip");

        // Architecture mismatch
        HCNN other(5, 3);
        other.AddConv(8);  // different width
        other.RandomizeWeights();
        check(throws([&] { load_weights(other, path); }),
              "load_weights rejects arch mismatch");

        std::remove(path);
    }

    // Full-capacity HCNNInput (smell 6 — pad contract)
    {
        HCNN net(6, 2);
        net.AddConv(4);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/3);
        const int N = net.GetStartN();  // capacity for c_in=1

        // Short raw → explicit zero-pad factory
        std::vector<float> short_in(8, 0.5f);
        auto batch = HCNNInputBatch::from_short_zero_pad(
            short_in.data(), /*count=*/1, /*input_length=*/8, N);
        check(batch.capacity() == N && batch.count() == 1,
              "from_short_zero_pad sizes to capacity");
        check(batch.sample(0)[0] == 0.5f && batch.sample(0)[8] == 0.0f,
              "from_short_zero_pad zeros tail");

        std::vector<float> out(2);
        net.Predict(batch.view(), out.data());
        check(all_finite(out.data(), 2), "Predict(HCNNInputView) finite");

        // Capacity mismatch rejected
        auto wrong = HCNNInputView::from_full(batch.data(), 1, N / 2);
        check(throws([&] { net.Predict(wrong, out.data()); }),
              "Predict rejects capacity != network");

        // Spatial pad_value survives full-capacity path; short Embed wipes it
        HCNNSpatialEmbedConfig cfg;
        cfg.dim = 6;
        cfg.mode = HCNNSpatialEmbedMode::RowMajorPad;
        cfg.pad_value = -1.0f;
        HCNNSpatialEmbedder emb(cfg);
        std::vector<float> img(4 * 4, 0.25f);
        auto packed = hcnn::pack_spatial(emb, img.data(), 4, 4);
        check(packed.capacity() == emb.capacity(), "pack_spatial capacity == N");
        check(packed.sample(0)[16] == -1.0f, "pack_spatial keeps pad_value");

        // Typed train uses full N — pad intact through embed (copy of full buffer)
        TrainParams tp;
        tp.learning_rate = 0.01f;
        int lab = 0;
        bool step_ok = true;
        try {
            net.TrainStep(packed.view(), lab, tp);
        } catch (const std::exception&) {
            step_ok = false;
        }
        check(step_ok, "TrainStep(HCNNInputView) runs");

        // FlatDataset input_view
        HCNNFlatDataset ds;
        ds.reset(2, N);
        std::fill(ds.inputs.begin(), ds.inputs.end(), 0.1f);
        ds.targets[0] = 0;
        ds.targets[1] = 1;
        bool epoch_ok = true;
        try {
            net.TrainEpoch(ds.input_view(), ds.targets.data(), /*batch=*/2, tp);
        } catch (const std::exception&) {
            epoch_ok = false;
        }
        check(epoch_ok, "TrainEpoch(input_view) runs");

        check(throws([&] {
            (void)HCNNInputBatch::adopt(std::vector<float>(3), 1, N);
        }), "adopt rejects wrong size");
    }

    // Train defaults (A) + HCNNTrainer (C)
    {
        HCNN net(5, 3);
        net.AddConv(4);
        net.RandomizeWeights(/*scale=*/0.0f, /*seed=*/8);
        const int N = net.GetStartN();

        TrainParams d;
        d.learning_rate = 0.05f;
        d.weight_decay = 1e-4f;
        d.shuffle_seed = 7u;
        net.SetTrainDefaults(d);
        check(net.GetTrainDefaults().learning_rate == 0.05f,
              "GetTrainDefaults reflects SetTrainDefaults");

        std::vector<float> x(static_cast<size_t>(N), 0.1f);
        bool ok = true;
        try {
            net.TrainStep(x.data(), N, /*target=*/0);  // uses defaults
        } catch (const std::exception&) {
            ok = false;
        }
        check(ok, "TrainStep without params uses defaults");

        HCNNFlatDataset ds;
        ds.reset(8, N);
        std::fill(ds.inputs.begin(), ds.inputs.end(), 0.05f);
        for (int i = 0; i < 8; ++i)
            ds.targets[static_cast<size_t>(i)] = i % 3;

        HCNNTrainer tr(net);
        tr.params().weight_decay = 1e-3f;
        tr.set_cosine(0.05f, 0.005f, /*num_epochs=*/3);
        tr.train_epoch(ds, /*batch=*/4, /*epoch=*/0);
        check(tr.params().learning_rate == cosine_lr(0.05f, 0.005f, 0, 3)
              || std::abs(tr.params().learning_rate
                          - cosine_lr(0.05f, 0.005f, 0, 3)) < 1e-7f,
              "trainer cosine sets lr for epoch 0");
        check(tr.params().shuffle_seed == 1u, "trainer shuffle_seed = epoch+1");
        check(std::abs(net.GetTrainDefaults().learning_rate
                       - tr.params().learning_rate) < 1e-7f,
              "trainer syncs SetTrainDefaults");

        tr.train_epoch(ds.input_view(), ds.targets.data(), 4, /*epoch=*/2);
        check(std::abs(tr.params().learning_rate
                       - cosine_lr(0.05f, 0.005f, 2, 3)) < 1e-7f,
              "trainer cosine at last epoch");
    }

    end_section();
}

// ---------------------------------------------------------------------------
//  11. Architecture product (LayerSpec / HCNNConfig)
// ---------------------------------------------------------------------------

static void section_arch() {
    begin_section("Architecture (LayerSpec / HCNNConfig)");

    // summarize_arch matches GetWeightCount (no BN)
    {
        std::vector<LayerSpec> layers = {
            LayerSpec::Conv(8),
            LayerSpec::Pool(PoolType::MAX),
            LayerSpec::Conv(16, Activation::TANH),
        };
        auto sum = hcnn::summarize_arch(6, /*num_outputs=*/4, /*c_in=*/1, layers);
        check(sum.num_conv == 2 && sum.num_pool == 1, "summarize: conv/pool counts");
        check(sum.final_dim == 5, "summarize: pool drops DIM 6->5");
        check(sum.final_N == 32, "summarize: N after pool");
        check(sum.flatten_features == 16 * 32, "summarize: flatten features");

        HCNN net(6, 4);
        hcnn::apply_arch(net, layers);
        net.RandomizeWeights();
        check(static_cast<long long>(net.GetWeightCount()) == sum.total,
              "summarize total == GetWeightCount (no BN)");
        check(net.GetNumConv() == 2 && net.GetNumPool() == 1,
              "apply_arch layer counts on HCNN");
    }

    // BN blob floats included (4 * c_out)
    {
        std::vector<LayerSpec> layers = {
            LayerSpec::Conv(4, Activation::RELU, /*bias=*/true, /*bn=*/true),
        };
        auto sum = hcnn::summarize_arch(5, 2, 1, layers);
        // kernel 1*4*(5+1) + bias 4 + BN 4*4 + readout 4*32*2 + 2
        const long long expect =
            1LL * 4 * 6 + 4 + 4 * 4 + 4LL * 32 * 2 + 2;
        check(sum.total == expect, "summarize with BN includes 4*c_out stats");

        HCNN net(5, 2);
        hcnn::apply_arch(net, layers);
        net.RandomizeWeights();
        check(static_cast<long long>(net.GetWeightCount()) == sum.total,
              "BN summarize total == GetWeightCount");
    }

    // HCNNConfig::Build
    {
        HCNNConfig cfg;
        cfg.start_dim = 5;
        cfg.num_outputs = 3;
        cfg.layers = {LayerSpec::Conv(8), LayerSpec::Conv(4)};
        cfg.weight_seed = 99;
        auto net = cfg.Build();
        check(net != nullptr, "Build returns non-null");
        check(net->WeightsInitialized(), "Build randomizes by default");
        check(net->GetOptimizerType() == OptimizerType::ADAM,
              "Build sets Adam by default");
        check(net->GetNumConv() == 2, "Build applied two convs");
        check(static_cast<long long>(net->GetWeightCount()) == cfg.summarize().total,
              "Build weight count matches summarize");

        std::vector<float> x(static_cast<size_t>(net->GetStartN()), 0.1f);
        std::vector<float> out(static_cast<size_t>(net->GetNumOutputs()));
        net->Predict(x.data(), net->GetStartN(), out.data());
        check(all_finite(out.data(), net->GetNumOutputs()),
              "Build net Predict finite");
    }

    // Validation: too many pools / empty / dim floor
    {
        check(throws([&] {
            (void)hcnn::summarize_arch(3, 2, 1, {
                LayerSpec::Conv(4),
                LayerSpec::Pool(),
                LayerSpec::Pool(),
                LayerSpec::Pool(),  // would need dim>=2; after two pools dim=1
            });
        }), "summarize rejects pool at current_dim < 2");

        check(throws([&] {
            (void)hcnn::summarize_arch(5, 2, 1, {LayerSpec::Pool()});
        }), "summarize rejects pool-only stack (no conv)");

        check(throws([&] {
            (void)hcnn::summarize_arch(2, 2, 1, {LayerSpec::Conv(4)});
        }), "summarize rejects start_dim < 3");

        HCNN net(5, 2);
        net.AddConv(4);
        check(throws([&] {
            hcnn::apply_arch(net, 6, 2, 1, {LayerSpec::Conv(4)});
        }), "apply_arch throws on dim mismatch with net");
    }

    end_section();
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "HCNN SDK Core Smoke Test\n";
    std::cout << "========================\n";

    const auto t0 = std::chrono::steady_clock::now();

    section_thread_pool();
    section_construction();
    section_forward_train();
    section_batchnorm();
    section_activations();
    section_contracts();
    section_regression();
    section_spatial_aug();
    section_spatial_embed();
    section_train_helpers();
    section_arch();

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n========================\n";
    std::cout << "Summary: " << g_passed << " passed, " << g_failed
              << " failed (" << secs << " s)\n";
    return g_failed ? 1 : 0;
}
