// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// Time-series regression example for HypercubeCNN.
//
// This example tests HCNN's regression mode against a target shape that
// mimics HypercubeRC's actual readout workload: per-timestep prediction
// of a continuous-valued signal from a high-dimensional reservoir-like
// state vector.  Specifically, it learns to predict the next value of
// sin(0.1*t) from a length-N synthetic state vector that captures the
// input signal at N different leaky-integrator timescales.
//
// The target has variance that does NOT collapse with N: sin(0.1*(t+1))
// is bounded in [-1, 1] with variance ~0.5 regardless of reservoir size.
// This is what HypercubeRC's readout sees in BasicPrediction.cpp, so it
// is the right shape to validate HCNN-as-readout against.  The question
// is: can HCNN learn a non-trivially-shaped projection from reservoir
// state to target?
//
// What this example shows:
//
//   - Constructing an HCNN regression network with TANH activation
//     (the natural activation for time-series workloads, matching the
//     reservoir's own nonlinearity)
//   - Centering targets on the train mean -- standard regression
//     hygiene that HypercubeRC's existing pipeline does via its
//     feature_mean / feature_scale standardization
//   - Using ForwardBatch in the evaluate path so per-epoch eval cost
//     stays small as DIM grows
//   - A self-contained synthetic reservoir (no HypercubeRC dependency)
//     that produces state vectors with realistic structure
//
// How to scale this up:
//
//   The DIM constant below defaults to 12 for a good balance between
//   showcasing real scale and finishing in a reasonable time.  Bump it
//   to 14 or 16 to test how the architecture scales further.  At
//   DIM=16 (N=65536 vertices) per-epoch wall time should be ~1-3
//   seconds depending on machine; total runtime ~5-15 minutes.
//   If convergence stalls at higher DIM, the most likely fix is more
//   conv channels (capacity) or a second conv+pool pair (depth) --
//   the target is not exactly expressible, so the model needs enough
//   degrees of freedom to *learn* a useful projection, not just to
//   identify a sparse-recovery solution.

#include "HCNN.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Hypercube geometry
// ---------------------------------------------------------------------------
constexpr int DIM = 12;            // N = 4096 vertices.  Bump to 14, 16
constexpr int N   = 1 << DIM;      // to test scaling.

// ---------------------------------------------------------------------------
// Synthetic reservoir
//
// Each of the N vertices is an independent leaky tanh integrator of a
// shared input signal.  Per-vertex parameters (leak rate, input weight,
// bias) are drawn once at startup and then held fixed across all data
// generation.  This gives the state vector rich temporal structure --
// different vertices capture the input at different effective timescales
// and phase offsets -- without requiring the O(N^2) coupling that a real
// reservoir would have.
//
// The trade-off: a real reservoir has cross-vertex coupling that produces
// nonlinear interactions between input dimensions and richer dynamics.
// This synthetic version is "uncoupled" -- each vertex's state is a
// function of the (scalar) input history, not of other vertices.  For an
// HCNN-readout test that's actually a useful simplification: the model
// only needs to learn which timescale combinations predict the target,
// not how to undo cross-vertex coupling.
// ---------------------------------------------------------------------------
struct ReservoirParams {
    std::vector<float> alpha;  ///< Per-vertex leak rate, length N.
    std::vector<float> w_in;   ///< Per-vertex input weight, length N.
    std::vector<float> bias;   ///< Per-vertex bias, length N.
};

static ReservoirParams make_reservoir(int n_vertices, unsigned seed) {
    std::mt19937 rng(seed);
    // Leak rates spread across a wide range so different vertices capture
    // different timescales.  Slow vertices (small alpha) integrate over
    // long history; fast vertices (large alpha) track recent values.
    std::uniform_real_distribution<float> alpha_dist(0.05f, 0.45f);
    std::uniform_real_distribution<float> w_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> b_dist(-0.5f, 0.5f);

    ReservoirParams p;
    p.alpha.resize(n_vertices);
    p.w_in.resize(n_vertices);
    p.bias.resize(n_vertices);
    for (int i = 0; i < n_vertices; ++i) {
        p.alpha[i] = alpha_dist(rng);
        p.w_in[i]  = w_dist(rng);
        p.bias[i]  = b_dist(rng);
    }
    return p;
}

// ---------------------------------------------------------------------------
// Drive the reservoir for `n_warmup + n_collect` timesteps.  Discard the
// first n_warmup steps as transient burn-in, then capture (state, target)
// pairs for the remaining n_collect steps.  The target is the input value
// `horizon` steps in the future -- predicting it requires the model to
// extract phase information from the state vector.
// ---------------------------------------------------------------------------
struct TimeseriesSample {
    std::vector<float> state;   ///< Reservoir state at this timestep, length N.
    float target;               ///< Sine value `horizon` steps ahead.
};

static std::vector<TimeseriesSample>
drive_and_collect(const ReservoirParams& params,
                  int n_warmup, int n_collect,
                  float input_freq, int horizon) {
    const int n_vertices = static_cast<int>(params.alpha.size());
    std::vector<float> state(n_vertices, 0.0f);

    auto step = [&](int t) {
        const float u = std::sin(input_freq * static_cast<float>(t));
        for (int i = 0; i < n_vertices; ++i) {
            const float drive = std::tanh(u * params.w_in[i] + params.bias[i]);
            state[i] = (1.0f - params.alpha[i]) * state[i]
                     + params.alpha[i] * drive;
        }
    };

    // Warmup -- drive but don't capture.
    for (int t = 0; t < n_warmup; ++t) step(t);

    // Collect.
    std::vector<TimeseriesSample> out;
    out.reserve(n_collect);
    for (int t = n_warmup; t < n_warmup + n_collect; ++t) {
        step(t);
        TimeseriesSample s;
        s.state = state;  // copy current state vector
        s.target = std::sin(input_freq * static_cast<float>(t + horizon));
        out.push_back(std::move(s));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Contiguous flat-buffer view over a sample list, suitable for HCNN's flat
// batch/epoch APIs.  Built once and reused across epochs.
// ---------------------------------------------------------------------------
struct FlatDataset {
    std::vector<float> inputs;   // n * N contiguous floats
    std::vector<float> targets;  // n contiguous floats
    int count = 0;

    explicit FlatDataset(const std::vector<TimeseriesSample>& ds) {
        count = static_cast<int>(ds.size());
        inputs.resize(static_cast<size_t>(count) * N);
        targets.resize(count);
        for (int i = 0; i < count; ++i) {
            std::copy(ds[i].state.begin(), ds[i].state.end(),
                      inputs.begin() + i * N);
            targets[i] = ds[i].target;
        }
    }
};

// ---------------------------------------------------------------------------
// Dataset evaluation via HCNN's parallel ForwardBatch path.  Returns MSE
// and target variance so the caller can compute R^2.
// ---------------------------------------------------------------------------
struct EvalResult {
    double mse;
    double target_var;
    double r2() const {
        return target_var > 0.0 ? 1.0 - mse / target_var : 0.0;
    }
};

static EvalResult evaluate(hcnn::HCNN& net, const FlatDataset& ds) {
    const int n = ds.count;

    std::vector<float> preds(n);
    net.ForwardBatch(ds.inputs.data(), N, n, preds.data());

    double mse_sum = 0.0;
    double tgt_sum = 0.0;
    double tgt_sq  = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = preds[i] - ds.targets[i];
        mse_sum += d * d;
        tgt_sum += ds.targets[i];
        tgt_sq  += static_cast<double>(ds.targets[i]) * ds.targets[i];
    }
    const double dn = static_cast<double>(n);
    EvalResult r;
    r.mse        = mse_sum / dn;
    r.target_var = tgt_sq / dn - (tgt_sum / dn) * (tgt_sum / dn);
    return r;
}

int main() {
    std::cout << "HypercubeCNN Time-Series Regression Example\n";
    std::cout << "===========================================\n";
    std::cout << "Task: predict sin(0.1*(t+1)) from a length-" << N << " synthetic\n";
    std::cout << "      reservoir state vector.  Each of the " << N << " vertices is an\n";
    std::cout << "      independent leaky tanh integrator with its own leak rate,\n";
    std::cout << "      input weight, and bias -- so the state vector encodes the\n";
    std::cout << "      input signal at " << N << " different timescales.  The model\n";
    std::cout << "      learns which combination of those timescales predicts the\n";
    std::cout << "      next-step value.  No exact solution exists; this is a real\n";
    std::cout << "      regression problem, not a sparse-recovery demo.\n\n";

    // ----- Hyperparameters -----
    constexpr int    n_warmup    = 200;     // transient burn-in steps (discarded)
    constexpr int    n_train     = 4096;
    constexpr int    n_test      = 1024;
    constexpr int    horizon     = 1;       // predict 1 step ahead
    constexpr float  input_freq  = 0.1f;
    constexpr unsigned reservoir_seed = 77;

    // ----- Synthetic reservoir + data -----
    auto reservoir = make_reservoir(N, reservoir_seed);
    auto all_data  = drive_and_collect(reservoir, n_warmup, n_train + n_test,
                                       input_freq, horizon);

    // Split into train / test along the time axis (no shuffling -- this is
    // a time series; train sees earlier steps, test sees later ones).
    std::vector<TimeseriesSample> train_data(all_data.begin(),
                                             all_data.begin() + n_train);
    std::vector<TimeseriesSample> test_data(all_data.begin() + n_train,
                                            all_data.end());

    std::cout << "Reservoir: " << N << " independent leaky tanh integrators\n";
    std::cout << "           leak rate ~ U[0.05, 0.45], w_in ~ U[-1, 1], bias ~ U[-0.5, 0.5]\n";
    std::cout << "Input:     sin(" << input_freq << "*t)\n";
    std::cout << "Target:    sin(" << input_freq << "*(t+" << horizon << "))\n";
    std::cout << "Train:     " << train_data.size() << " timesteps (after "
              << n_warmup << " warmup steps)\n";
    std::cout << "Test:      " << test_data.size()  << " timesteps\n\n";

    // ----- Center the targets on the train mean -----
    //
    // Standard regression hygiene.  Sine targets are already centered to
    // zero in expectation, but the empirical train-set mean over a finite
    // window is small but nonzero, and Adam's moment estimates work much
    // better when there is no systematic offset to correct in the first
    // few epochs.
    // Flatten into contiguous buffers for HCNN's flat API.
    FlatDataset train_flat(train_data);
    FlatDataset test_flat(test_data);

    // Center targets on the train mean.
    double train_target_mean_d = 0.0;
    for (int i = 0; i < train_flat.count; ++i)
        train_target_mean_d += train_flat.targets[i];
    train_target_mean_d /= static_cast<double>(train_flat.count);
    const float train_target_mean = static_cast<float>(train_target_mean_d);
    for (auto& t : train_flat.targets) t -= train_target_mean;
    for (auto& t : test_flat.targets)  t -= train_target_mean;

    std::cout << "Target centering: subtracted train mean "
              << std::scientific << std::setprecision(3) << train_target_mean
              << std::fixed << "\n\n";

    // ----- Network -----
    //
    // Conv(16, TANH, bias) -> MaxPool -> GAP -> Linear(16 -> 1)
    //
    // TANH is the natural activation for time-series workloads -- smooth,
    // symmetric, bounded in (-1, 1).  Same nonlinearity used by the
    // synthetic reservoir, by HypercubeRC's actual reservoir, and by every
    // RNN-family model in the literature.
    //
    // 16 conv channels gives the model meaningful learning capacity for a
    // task that is NOT exactly expressible by a single channel.  The conv
    // kernel sees DIM single-bit-flip neighbors per vertex, so each
    // channel learns a different DIM-direction filter over the state.
    //
    // Antipodal max-pool pairs vertices on opposite sides of the hypercube
    // and collapses them into the larger of their two activations.  For a
    // tanh-bounded conv output this picks the more strongly responding
    // member of each pair, which is a useful inductive bias for "extract
    // the strongest signal across phase-related vertices."
    //
    // GAP averages the post-pool activations across the surviving
    // vertices and feeds the channel means into a linear projection
    // 16 -> 1.  At larger DIM you may need either more channels (more
    // capacity in the conv) or a second conv+pool pair (depth); both are
    // documented in the file header.
    hcnn::HCNN net(DIM, /*num_outputs=*/1, /*input_channels=*/1,
                   hcnn::ReadoutType::GAP,
                   hcnn::TaskType::Regression);
    net.AddConv(16, hcnn::Activation::TANH, /*use_bias=*/true);
    net.AddPool(hcnn::PoolType::MAX);
    net.RandomizeWeights();
    net.SetOptimizer(hcnn::OptimizerType::ADAM);

    const int conv_params = 1 * 16 * DIM + 16;   // kernel + bias
    const int readout_params = 16 + 1;           // weight + bias
    std::cout << "Architecture: Conv(16, TANH, bias)\n";
    std::cout << "              -> MaxPool (antipodal)\n";
    std::cout << "              -> GAP\n";
    std::cout << "              -> Linear(16 -> 1)\n";
    std::cout << "Parameters:   " << (conv_params + readout_params)
              << " (" << conv_params << " conv + " << readout_params << " readout)\n\n";

    // ----- Training loop -----
    constexpr int   epochs       = 200;
    constexpr int   batch_size   = 32;
    constexpr float lr_max       = 0.01f;
    constexpr float lr_min       = 1e-3f;    // 10% of lr_max -- keeps learning
    constexpr float momentum     = 0.0f;     // Adam handles adaptive scaling
    constexpr float weight_decay = 0.0f;

    EvalResult before = evaluate(net, test_flat);
    std::cout << std::scientific << std::setprecision(3)
              << "Initial test MSE: " << before.mse
              << "   target_var: " << before.target_var
              << "   1-R^2: " << (1.0 - before.r2()) << "\n\n";

    for (int e = 0; e < epochs; ++e) {
        const float progress = static_cast<float>(e) / static_cast<float>(epochs);
        const float lr = lr_min + 0.5f * (lr_max - lr_min) *
                         (1.0f + std::cos(static_cast<float>(std::numbers::pi) * progress));

        auto t0 = std::chrono::steady_clock::now();
        net.TrainEpochRegression(train_flat.inputs.data(), N,
                                 train_flat.targets.data(),
                                 train_flat.count, batch_size,
                                 lr, momentum, weight_decay,
                                 /*shuffle_seed=*/static_cast<unsigned>(e + 1));

        if (e < 5 || (e + 1) % 10 == 0 || e == epochs - 1) {
            EvalResult train_r = evaluate(net, train_flat);
            EvalResult test_r  = evaluate(net, test_flat);
            auto t1 = std::chrono::steady_clock::now();
            const double secs = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "Epoch " << std::setw(3) << (e + 1) << "/" << epochs
                      << std::fixed << std::setprecision(6)
                      << "  lr=" << std::setw(8) << lr
                      << std::scientific << std::setprecision(3)
                      << "  train_mse=" << std::setw(10) << train_r.mse
                      << "  test_mse="  << std::setw(10) << test_r.mse
                      << "  1-R^2=" << std::setw(10) << (1.0 - test_r.r2())
                      << std::fixed << std::setprecision(3)
                      << "  (" << secs << "s)\n";
        }
    }

    EvalResult after = evaluate(net, test_flat);
    const double reduction = 100.0 * (1.0 - after.mse / before.mse);
    std::cout << "\nFinal test MSE:    "
              << std::scientific << std::setprecision(3) << after.mse
              << std::fixed << std::setprecision(2)
              << "  (" << reduction << "% reduction)\n"
              << std::scientific << std::setprecision(3)
              << "Final test 1-R^2:  " << (1.0 - after.r2())
              << "  (0 = perfect fit)\n";

    // ----- Sample predictions (evenly spaced across the test set) -----
    //
    // Spread 8 samples across the full test window so the output covers
    // peaks, troughs, and zero crossings -- not just one phase of the sine.
    constexpr int n_samples = 8;
    const int stride = std::max(1, test_flat.count / n_samples);
    std::cout << "\nSample predictions (test set, original scale):\n";
    std::cout << "  step  target      pred       err\n";
    std::vector<float> embedded(N);
    std::vector<float> pred(1);
    for (int s = 0; s < n_samples && s * stride < test_flat.count; ++s) {
        const int i = s * stride;
        net.Embed(test_flat.inputs.data() + i * N, N, embedded.data());
        net.Forward(embedded.data(), pred.data());
        const float target_orig = test_flat.targets[i] + train_target_mean;
        const float pred_orig   = pred[0]              + train_target_mean;
        const float err         = pred_orig - target_orig;
        std::cout << "  " << std::setw(4) << i
                  << std::fixed << std::setprecision(6)
                  << "   " << std::showpos << std::setw(10) << target_orig
                  << "  " << std::setw(10) << pred_orig
                  << std::scientific << std::setprecision(3)
                  << "  " << std::setw(10) << err
                  << std::noshowpos << "\n";
    }

    // Defensive CI sanity check.  The target is not exactly expressible,
    // so a non-trivial R^2 is what counts, not perfect fit.
    return (after.r2() > 0.9) ? 0 : 1;
}
