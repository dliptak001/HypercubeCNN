// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// Time-series regression teaching demo for HypercubeCNN.
//
// Predicts sin(freq*(t+horizon)) from a length-N synthetic reservoir state
// (N independent leaky tanh integrators of a shared sine drive).
//
// Pipeline:
//   synthetic reservoir states  ->  flat float buffers
//   TrainEpochRegression + cosine_lr + evaluate_regression (MSE / R^2)
//   HCNNBestMetricCheckpoint (best test MSE)
//
// Developer knobs: DemoConfig below (same flavor as mnist_train.cpp).
// Architecture scaffolding: examples/demo_arch.h
//
// What this demo proves
//   - Regression API (TaskType::Regression, MSE, TrainEpochRegression)
//   - Mixed activations + full-N FLATTEN at DIM=10 (N=1024)
//   - Train-loop hygiene: cosine LR, target centering, best-MSE restore
//
// What this demo does NOT prove
//   - Real HypercubeRC / ESN dynamics (reservoir here is uncoupled; no
//     spectral radius, no recurrent mixing between vertices)
//   - Hard multi-step forecasting or chaotic attractor skill
//   - That near-perfect R^2 will transfer to production RC workloads
//     (synthetic next-step sine is an easy target once capacity is enough)

#include "HCNN.h"
#include "HCNNTrainHelpers.h"
#include "demo_arch.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using hcnn_demo::ArchLayer;
using hcnn_demo::ArchParamSummary;

// =============================================================================
// DEVELOPER CONFIG - edit knobs here; the rest of the file follows
// =============================================================================
//
// Documented default (seed 42): DIM=10 (N=1024), Conv16 RELU -> Conv16 TANH,
// no pool, Adam, cosine LR 0.002 -> 2e-4, 50 epochs. K=DIM+1 (self+neighbors).
// Best test MSE historically ~3e-8 (neighbor-only era); re-check after self tap.
// First-layer RELU beat dual-TANH on this seed; raise dim / depth to stress scale.
// =============================================================================

struct DemoConfig {
    // ----- Hypercube / task -----
    int dim = 10;                 // N = 2^dim vertices (state length)
    int num_outputs = 1;          // scalar regression
    int input_channels = 1;

    // Documented recipe: RELU head + TANH second conv, full N (no pool).
    std::vector<ArchLayer> layers = {
        ArchLayer::Conv(16, hcnn::Activation::RELU, /*bias=*/true, /*bn=*/false),
        ArchLayer::Conv(16, hcnn::Activation::TANH, /*bias=*/true, /*bn=*/false),
        // Examples:
        // ArchLayer::Pool(hcnn::PoolType::MAX),
        // ArchLayer::Conv(16, hcnn::Activation::TANH),
    };

    // ----- Synthetic data -----
    int n_warmup = 200;           // transient burn-in (discarded)
    int n_train = 4096;
    int n_test = 1024;
    int horizon = 1;              // predict this many steps ahead
    float input_freq = 0.1f;      // drive / target: sin(freq * t)
    unsigned reservoir_seed = 77;

    // ----- Init / optimizer -----
    unsigned weight_seed = 42;
    hcnn::OptimizerType optimizer = hcnn::OptimizerType::ADAM;

    // ----- Schedule -----
    int epochs = 50;
    float lr_max = 0.002f;
    float lr_min_ratio = 0.1f;    // lr_min = lr_max * ratio
    int batch_size = 32;
    float weight_decay = 0.0f;
    float momentum = 0.0f;        // Adam; kept for TrainEpochRegression API

    // ----- Logging -----
    int log_first_epochs = 5;
    int log_every = 10;
    int n_sample_preds = 8;

    float lr_min() const { return lr_max * lr_min_ratio; }
    int N() const { return 1 << dim; }
};

static ArchParamSummary summarize_demo(const DemoConfig& cfg) {
    if (cfg.epochs < 1 || cfg.batch_size < 1)
        throw std::runtime_error("DemoConfig: epochs and batch_size must be >= 1");
    if (cfg.n_train < 1 || cfg.n_test < 1)
        throw std::runtime_error("DemoConfig: n_train and n_test must be >= 1");
    if (cfg.lr_max <= 0.0f || cfg.lr_min_ratio < 0.0f || cfg.lr_min_ratio > 1.0f)
        throw std::runtime_error("DemoConfig: invalid lr_max / lr_min_ratio");
    return hcnn_demo::summarize_arch(cfg.dim, cfg.num_outputs, cfg.input_channels,
                                     cfg.layers);
}

// ---------------------------------------------------------------------------
// Synthetic reservoir (no HypercubeRC dependency)
// ---------------------------------------------------------------------------

struct ReservoirParams {
    std::vector<float> alpha;
    std::vector<float> w_in;
    std::vector<float> bias;
};

static ReservoirParams make_reservoir(int n_vertices, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> alpha_dist(0.05f, 0.45f);
    std::uniform_real_distribution<float> w_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> b_dist(-0.5f, 0.5f);

    ReservoirParams p;
    p.alpha.resize(static_cast<size_t>(n_vertices));
    p.w_in.resize(static_cast<size_t>(n_vertices));
    p.bias.resize(static_cast<size_t>(n_vertices));
    for (int i = 0; i < n_vertices; ++i) {
        p.alpha[static_cast<size_t>(i)] = alpha_dist(rng);
        p.w_in[static_cast<size_t>(i)]  = w_dist(rng);
        p.bias[static_cast<size_t>(i)]  = b_dist(rng);
    }
    return p;
}

struct TimeseriesSample {
    std::vector<float> state;
    float target = 0.0f;
};

static std::vector<TimeseriesSample>
drive_and_collect(const ReservoirParams& params,
                  int n_warmup, int n_collect,
                  float input_freq, int horizon) {
    const int n_vertices = static_cast<int>(params.alpha.size());
    std::vector<float> state(static_cast<size_t>(n_vertices), 0.0f);

    auto step = [&](int t) {
        const float u = std::sin(input_freq * static_cast<float>(t));
        for (int i = 0; i < n_vertices; ++i) {
            const float drive =
                std::tanh(u * params.w_in[static_cast<size_t>(i)]
                          + params.bias[static_cast<size_t>(i)]);
            state[static_cast<size_t>(i)] =
                (1.0f - params.alpha[static_cast<size_t>(i)])
                    * state[static_cast<size_t>(i)]
                + params.alpha[static_cast<size_t>(i)] * drive;
        }
    };

    for (int t = 0; t < n_warmup; ++t)
        step(t);

    std::vector<TimeseriesSample> out;
    out.reserve(static_cast<size_t>(n_collect));
    for (int t = n_warmup; t < n_warmup + n_collect; ++t) {
        step(t);
        TimeseriesSample s;
        s.state = state;
        s.target = std::sin(input_freq * static_cast<float>(t + horizon));
        out.push_back(std::move(s));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Flat regression buffers
// ---------------------------------------------------------------------------

struct FlatRegDataset {
    std::vector<float> inputs;
    std::vector<float> targets;
    int count = 0;
    int input_length = 0;

    void from_samples(const std::vector<TimeseriesSample>& ds, int N) {
        count = static_cast<int>(ds.size());
        input_length = N;
        inputs.resize(static_cast<size_t>(count) * static_cast<size_t>(N));
        targets.resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            if (static_cast<int>(ds[static_cast<size_t>(i)].state.size()) != N)
                throw std::runtime_error("FlatRegDataset: state length != N");
            std::copy(ds[static_cast<size_t>(i)].state.begin(),
                      ds[static_cast<size_t>(i)].state.end(),
                      inputs.begin() + static_cast<size_t>(i) * static_cast<size_t>(N));
            targets[static_cast<size_t>(i)] = ds[static_cast<size_t>(i)].target;
        }
    }
};

static void print_eval(const char* label, const hcnn::HCNNRegEval& r) {
    std::cout << label << ": mse=" << std::scientific << std::setprecision(4)
              << r.mse
              << "  target_var=" << r.target_var
              << "  R^2=" << std::fixed << std::setprecision(4) << r.r2()
              << "  (1-R^2=" << (1.0 - r.r2()) << ")\n";
}

static bool should_log_epoch(int epoch_0, int epochs, const DemoConfig& cfg) {
    const int e1 = epoch_0 + 1;
    if (e1 <= cfg.log_first_epochs) return true;
    if (e1 == epochs) return true;
    if (cfg.log_every > 0 && (e1 % cfg.log_every) == 0) return true;
    return false;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    const DemoConfig cfg{};
    const int N = cfg.N();
    const ArchParamSummary arch_sum = summarize_demo(cfg);

    std::cout << "HypercubeCNN Time-Series Regression\n";
    std::cout << "===================================\n";
    std::cout << "Task: predict sin(" << cfg.input_freq << "*(t+" << cfg.horizon
              << ")) from length-" << N << " reservoir state (DIM=" << cfg.dim
              << ")\n";
    std::cout << "      Synthetic uncoupled leaky-tanh integrators; no HypercubeRC.\n";
    std::cout << "\n";
    std::cout << "Proves:    regression API, RELU/TANH+FLATTEN, train hygiene\n";
    std::cout << "Does not:  real RC dynamics, hard forecasting, production RC skill\n";
    std::cout << "           (near-perfect R^2 on this sine task is expected smoke).\n";

    auto reservoir = make_reservoir(N, cfg.reservoir_seed);
    auto all_data = drive_and_collect(reservoir, cfg.n_warmup,
                                      cfg.n_train + cfg.n_test,
                                      cfg.input_freq, cfg.horizon);

    std::vector<TimeseriesSample> train_data(
        all_data.begin(), all_data.begin() + cfg.n_train);
    std::vector<TimeseriesSample> test_data(
        all_data.begin() + cfg.n_train, all_data.end());

    std::cout << "\nData:      train=" << train_data.size()
              << "  test=" << test_data.size()
              << "  (warmup=" << cfg.n_warmup << " discarded)\n";
    std::cout << "Reservoir: N=" << N
              << "  leak~U[0.05,0.45]  w_in~U[-1,1]  bias~U[-0.5,0.5]"
              << "  seed=" << cfg.reservoir_seed << "\n";
    std::cout << "Drive:     sin(" << cfg.input_freq << "*t)\n";
    std::cout << "Threads:   " << std::thread::hardware_concurrency() << "\n";

    FlatRegDataset train_flat;
    FlatRegDataset test_flat;
    train_flat.from_samples(train_data, N);
    test_flat.from_samples(test_data, N);

    double train_mean_d = 0.0;
    for (float t : train_flat.targets)
        train_mean_d += t;
    train_mean_d /= static_cast<double>(train_flat.count);
    const float train_mean = static_cast<float>(train_mean_d);
    for (float& t : train_flat.targets) t -= train_mean;
    for (float& t : test_flat.targets)  t -= train_mean;

    std::cout << "Centering: subtracted train target mean "
              << std::scientific << std::setprecision(3) << train_mean
              << std::defaultfloat << "\n";

    hcnn::HCNN net(cfg.dim, cfg.num_outputs, cfg.input_channels,
                   hcnn::TaskType::Regression);
    hcnn_demo::apply_arch(net, cfg.dim, cfg.num_outputs, cfg.input_channels,
                          cfg.layers);
    net.RandomizeWeights(/*scale=*/0.0f, cfg.weight_seed);
    net.SetOptimizer(cfg.optimizer);

    if (net.GetStartN() != N) {
        throw std::runtime_error("HCNN start N does not match DemoConfig::dim");
    }
    if (static_cast<long long>(net.GetWeightCount()) != arch_sum.total) {
        throw std::runtime_error(
            "param count mismatch: summary " + std::to_string(arch_sum.total)
            + " vs GetWeightCount " + std::to_string(net.GetWeightCount()));
    }

    std::cout << "Weight init seed: " << cfg.weight_seed << "\n";
    hcnn_demo::print_arch(std::cout, cfg.dim, cfg.num_outputs, cfg.input_channels,
                          cfg.layers, arch_sum);

    const float lr_max = cfg.lr_max;
    const float lr_min = cfg.lr_min();
    std::cout << "=== train (lr_max=" << lr_max
              << ", lr_min=" << lr_min
              << ", batch=" << cfg.batch_size
              << ", wd=" << cfg.weight_decay
              << ", epochs=" << cfg.epochs << ") ===\n";

    auto eval_ds = [&](const FlatRegDataset& ds) {
        return hcnn::evaluate_regression(net, ds.inputs.data(), ds.input_length,
                                         ds.targets.data(), ds.count,
                                         cfg.num_outputs);
    };

    hcnn::HCNNRegEval before = eval_ds(test_flat);
    print_eval("Initial test", before);
    std::cout << "\n";

    hcnn::HCNNBestMetricCheckpoint best_mse;
    auto t_run0 = std::chrono::steady_clock::now();

    for (int e = 0; e < cfg.epochs; ++e) {
        const float lr = hcnn::cosine_lr(lr_max, lr_min, e, cfg.epochs);

        auto t0 = std::chrono::steady_clock::now();
        net.TrainEpochRegression(train_flat.inputs.data(), train_flat.input_length,
                                 train_flat.targets.data(),
                                 train_flat.count, cfg.batch_size,
                                 lr, cfg.momentum, cfg.weight_decay,
                                 /*shuffle_seed=*/static_cast<unsigned>(e + 1));
        auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const double samples_per_s =
            (secs > 0.0) ? (static_cast<double>(train_flat.count) / secs) : 0.0;

        // Always score test for best-MSE; log train+test on selected epochs.
        hcnn::HCNNRegEval test_r = eval_ds(test_flat);
        const bool is_best = best_mse.observe(
            net, static_cast<float>(test_r.mse), e + 1);

        if (should_log_epoch(e, cfg.epochs, cfg)) {
            hcnn::HCNNRegEval train_r = eval_ds(train_flat);
            std::cout << "Epoch " << (e + 1) << "/" << cfg.epochs
                      << std::fixed << std::setprecision(6)
                      << "  lr=" << lr
                      << std::scientific << std::setprecision(4)
                      << "  train_mse=" << train_r.mse
                      << "  test_mse=" << test_r.mse
                      << std::fixed << std::setprecision(4)
                      << "  test_R^2=" << test_r.r2()
                      << std::setprecision(2)
                      << "  (" << secs << "s, " << samples_per_s << " samples/s)";
            if (is_best) std::cout << "  [best-mse]";
            std::cout << "\n";
        }
    }

    auto t_run1 = std::chrono::steady_clock::now();
    const double total_secs =
        std::chrono::duration<double>(t_run1 - t_run0).count();

    std::cout << "\n--- Checkpoints ---\n";
    if (best_mse.has_best()) {
        std::cout << "Best test MSE: epoch " << best_mse.best_epoch()
                  << "  mse=" << std::scientific << std::setprecision(4)
                  << best_mse.best_metric() << "\n";
        best_mse.restore(net);
        print_eval("Restored best-mse", eval_ds(test_flat));
    }

    hcnn::HCNNRegEval after = eval_ds(test_flat);
    const double reduction =
        (before.mse > 0.0) ? 100.0 * (1.0 - after.mse / before.mse) : 0.0;
    std::cout << "\n--- Final (best-mse weights) ---\n";
    print_eval("Test", after);
    std::cout << std::fixed << std::setprecision(2)
              << "MSE reduction vs initial: " << reduction << "%\n"
              << "Total train wall time: " << total_secs << "s\n";

    const int n_show = std::min(cfg.n_sample_preds, test_flat.count);
    const int stride = std::max(1, test_flat.count / std::max(1, n_show));
    std::cout << "\nSample predictions (test, original scale):\n";
    std::cout << "  step   target        pred          err\n";
    std::vector<float> embedded(static_cast<size_t>(N));
    std::vector<float> pred(static_cast<size_t>(cfg.num_outputs));
    for (int s = 0; s < n_show; ++s) {
        const int i = s * stride;
        if (i >= test_flat.count) break;
        net.Embed(test_flat.inputs.data()
                      + static_cast<size_t>(i) * static_cast<size_t>(N),
                  N, embedded.data());
        net.Forward(embedded.data(), pred.data());
        const float target_orig =
            test_flat.targets[static_cast<size_t>(i)] + train_mean;
        const float pred_orig = pred[0] + train_mean;
        const float err = pred_orig - target_orig;
        std::cout << "  " << std::setw(4) << i
                  << std::fixed << std::setprecision(6)
                  << "  " << std::showpos << std::setw(11) << target_orig
                  << "  " << std::setw(11) << pred_orig
                  << std::scientific << std::setprecision(3)
                  << "  " << std::setw(11) << err
                  << std::noshowpos << "\n";
    }

    return (after.r2() > 0.9) ? 0 : 1;
}
