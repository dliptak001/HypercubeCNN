// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNN.h"
#include "HCNNSpatialAug.h"
#include "HCNNSpatialEmbed.h"
#include "HCNNTrainHelpers.h"
#include "HCNNDataset.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// MNIST teaching demo — thin loop on core helpers
//
// Loader yields 28×28 in [-1, 1].  Pipeline:
//   (train only) HCNNSpatialAugmenter  — rot / scale / shift / noise
//   HCNNSpatialEmbedder                — DualPlaneResize → N = 2^11 = 2048
//   HCNN TrainEpoch + cosine_lr + dual checkpoint + evaluate_classification
// ---------------------------------------------------------------------------

static constexpr int kImgSide   = 28;
static constexpr int kImgPixels = kImgSide * kImgSide;  // 784
static constexpr int kDim       = 11;
static constexpr float kBackground = -1.0f;  // MNIST "ink off" after loader norm

/// Build MNIST train-time aug config (identity fields when disabled).
static hcnn::HCNNSpatialAugConfig make_mnist_aug_config(bool enabled) {
    if (!enabled)
        return hcnn::HCNNSpatialAugConfig::None();

    hcnn::HCNNSpatialAugConfig cfg;
    cfg.rot_deg_max  = 12.0f;       // uniform rotate ±12° about image center
    cfg.scale_min    = 0.9f;        // uniform scale low (size factor)
    cfg.scale_max    = 1.1f;        // uniform scale high
    cfg.shift_max    = 2;           // integer pixel shift dy,dx in {-2..+2}
    cfg.noise_sigma  = 0.03f;       // Gaussian N(0, σ²) after geometry
    cfg.value_min    = -1.0f;       // clip floor after noise (loader range)
    cfg.value_max    =  1.0f;       // clip ceiling after noise
    cfg.border_value = kBackground; // bilinear OOB = empty paper (-1)
    cfg.enabled      = true;        // master on (false → pure copy, no RNG)
    return cfg;
}

/// Dual-plane embed: 32×32 ink ‖ 32×32 |∇| into length N = 2048 (full occupancy).
static hcnn::HCNNSpatialEmbedConfig make_mnist_embed_config() {
    hcnn::HCNNSpatialEmbedConfig cfg;
    cfg.dim        = kDim;
    cfg.mode       = hcnn::HCNNSpatialEmbedMode::DualPlaneResize;
    cfg.pad_value  = kBackground;
    cfg.plane_side = 0;  // auto: floor(sqrt(N/2)) = 32 at DIM=11
    return cfg;
}

/// Optional aug at 28×28, then SpatialEmbed into HCNNFlatDataset (length N).
static void fill_spatial_dataset(const HCNNDataset& ds,
                                 hcnn::HCNNFlatDataset& out,
                                 const hcnn::HCNNSpatialEmbedder& emb,
                                 const hcnn::HCNNSpatialAugmenter& aug,
                                 unsigned seed) {
    const int n = static_cast<int>(ds.size());
    const int N = emb.capacity();
    out.reset(n, N);

    std::mt19937 rng(seed);
    std::vector<float> work(static_cast<size_t>(kImgPixels));
    const bool do_aug = aug.config().enabled && !aug.config().is_identity();

    for (int i = 0; i < n; ++i) {
        const auto& s = ds.get(static_cast<size_t>(i));
        if (static_cast<int>(s.input.size()) != kImgPixels) {
            throw std::runtime_error(
                "fill_spatial_dataset: expected 28x28 MNIST input");
        }

        const float* img = s.input.data();
        if (do_aug) {
            // Geometric warp requires in != out; work holds the warped plane.
            aug.apply(s.input.data(), work.data(), kImgSide, kImgSide, rng);
            img = work.data();
        }

        emb.embed(img, kImgSide, kImgSide, out.sample_input(i));
        out.targets[static_cast<size_t>(i)] = s.target_class;
    }
}

static void print_eval(const char* label, const hcnn::HCNNClassEval& r) {
    std::cout << label << ": loss=" << r.loss
              << " acc=" << r.correct << "/" << r.count
              << " (" << r.accuracy << "%)\n";
}

// Dual checkpoint (best test loss / best test accuracy).
// Each epoch rebuilds the train buffer with a fresh augment seed.
static void train_and_evaluate(const char* name, hcnn::HCNN& net,
                               const HCNNDataset& train_raw,
                               const hcnn::HCNNFlatDataset& test_ds,
                               const hcnn::HCNNSpatialEmbedder& emb,
                               const hcnn::HCNNSpatialAugmenter& train_aug,
                               float lr = 0.01f, int batch_size = 32,
                               float weight_decay = 0.0f) {
    const int epochs = 60;
    const float lr_max = lr;
    const float lr_min = lr_max * 0.1f;
    const float momentum = 0.9f;
    const auto& ac = train_aug.config();

    std::cout << "\n=== " << name << " (lr_max=" << lr_max
              << ", lr_min=" << lr_min
              << ", batch=" << batch_size
              << ", wd=" << weight_decay
              << ", epochs=" << epochs
              << ", aug=rot+/-" << ac.rot_deg_max
              << "+scale[" << ac.scale_min << "," << ac.scale_max << "]"
              << "+shift+/-" << ac.shift_max
              << "+N(0," << ac.noise_sigma << ")) ===\n";
    print_eval("Initial test", hcnn::evaluate_classification(net, test_ds));

    hcnn::HCNNFlatDataset train_ds;
    hcnn::HCNNDualCheckpoint ckpt;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const float current_lr = hcnn::cosine_lr(lr_max, lr_min, epoch, epochs);

        // Fresh train-time aug each epoch; seed mixes epoch so runs are reproducible.
        // Timer includes pack+aug rebuild (wall-clock cost of the epoch, not only
        // the optimizer step).
        auto t0 = std::chrono::steady_clock::now();
        fill_spatial_dataset(
            train_raw, train_ds, emb, train_aug,
            /*seed=*/static_cast<unsigned>(0xC0FFEEu + epoch * 9973u));
        net.TrainEpoch(train_ds.inputs.data(), train_ds.input_length,
                       train_ds.targets.data(), train_ds.count, batch_size,
                       current_lr, momentum, weight_decay,
                       /*class_weights=*/nullptr,
                       /*shuffle_seed=*/static_cast<unsigned>(epoch + 1));
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::string label = "Epoch " + std::to_string(epoch + 1) + "/"
                            + std::to_string(epochs);
        hcnn::HCNNClassEval r = hcnn::evaluate_classification(net, test_ds);
        print_eval(label.c_str(), r);

        auto upd = ckpt.observe(net, r.loss, r.accuracy, epoch + 1);

        std::cout << "  (lr=" << current_lr << ", " << secs << "s, "
                  << train_ds.count / secs << " samples/s)";
        if (upd.any()) {
            std::cout << "  [";
            if (upd.new_best_loss) std::cout << "best-loss";
            if (upd.new_best_loss && upd.new_best_acc) std::cout << " ";
            if (upd.new_best_acc) std::cout << "best-acc";
            std::cout << "]";
        }
        std::cout << "\n";
    }

    std::cout << "\n--- Checkpoints ---\n"
              << "Best loss: epoch " << ckpt.best_loss_epoch()
              << "  loss=" << ckpt.best_loss()
              << "  acc=" << ckpt.best_loss_acc() << "%\n"
              << "Best acc:  epoch " << ckpt.best_acc_epoch()
              << "  loss=" << ckpt.best_acc_loss()
              << "  acc=" << ckpt.best_acc() << "%\n";

    if (ckpt.has_best_loss()) {
        ckpt.restore_best_loss(net);
        print_eval("Restored best-loss",
                   hcnn::evaluate_classification(net, test_ds));
    }
    if (ckpt.has_best_acc()) {
        ckpt.restore_best_acc(net);
        print_eval("Restored best-acc",
                   hcnn::evaluate_classification(net, test_ds));
    }
}

int main() {
    auto src_dir = std::filesystem::path(__FILE__).parent_path().parent_path();
    auto data_dir = src_dir / "data";

    std::cout << "Loading MNIST from " << data_dir << "...\n";
    auto train_raw = load_mnist((data_dir / "train-images-idx3-ubyte").string(),
                                (data_dir / "train-labels-idx1-ubyte").string(), 60000);
    auto test_raw  = load_mnist((data_dir / "t10k-images-idx3-ubyte").string(),
                                (data_dir / "t10k-labels-idx1-ubyte").string(), 10000);
    std::cout << "Train: " << train_raw.size() << " samples, "
              << "Test: " << test_raw.size() << " samples\n";
    std::cout << "Threads: " << std::thread::hardware_concurrency() << "\n";

    hcnn::HCNNSpatialEmbedder emb(make_mnist_embed_config());
    hcnn::HCNNSpatialAugmenter train_aug(make_mnist_aug_config(/*enabled=*/true));
    hcnn::HCNNSpatialAugmenter test_aug(make_mnist_aug_config(/*enabled=*/false));

    const auto plan = emb.plan(kImgSide, kImgSide);
    const int N = emb.capacity();

    // Test: embed only (no aug).  Train: re-embedded with aug each epoch.
    hcnn::HCNNFlatDataset test_flat;
    fill_spatial_dataset(test_raw, test_flat, emb, test_aug, /*seed=*/0);

    std::cout << "Spatial pipeline: HCNNSpatialAug (train) -> "
              << "HCNNSpatialEmbed DualPlaneResize "
              << plan.plane_side << "x" << plan.plane_side
              << " ink || |grad|  (pattern_length=" << plan.pattern_length
              << ", N=" << N << ")\n";
    std::cout << "Train aug:  rot +/-12 deg, scale [0.9,1.1], shift +/-2 px, "
              << "Gaussian noise sigma=0.03 (train only, refreshed each epoch)\n";

    // Weight-init seed only (aug / shuffle seeds are fixed separately).
    // Documented default: ~99.23% mean best-acc over seeds; peak 99.27%
    // (seed 398479293) with rot±12 + scale[0.9,1.1] + shift±2 + noise,
    // 60 epochs, no pool.  Re-measure after Spatial* wire if claiming numbers.
    constexpr unsigned weight_seed = 398479293;

    // 2-layer no-pool: 16 -> 16 at full N=2048 (no antipodal pool).
    // FLATTEN head 32768->10. No BN (instance-over-N is a poor match here).
    constexpr int C1 = 16;
    constexpr int C2 = 16;
    constexpr bool kUseBN = false;

    hcnn::HCNN net(kDim, /*num_outputs=*/10, /*input_channels=*/1);
    net.AddConv(C1, hcnn::Activation::RELU, /*use_bias=*/true, kUseBN);
    net.AddConv(C2, hcnn::Activation::RELU, /*use_bias=*/true, kUseBN);
    net.RandomizeWeights(/*scale=*/0.0f, weight_seed);
    net.SetOptimizer(hcnn::OptimizerType::ADAM);

    std::cout << "Weight init seed: " << weight_seed << "\n";

    constexpr int N_final = 1 << kDim;
    static_assert(N_final == 2048, "DIM=11 N must be 2048");
    const int conv1_params   = 1 * C1 * kDim + C1;
    const int conv2_params   = C1 * C2 * kDim + C2;
    const int readout_params = C2 * N_final * 10 + 10;
    const int total_params   = conv1_params + conv2_params + readout_params;
    std::cout << "\nArchitecture: Conv(1->" << C1 << ", RELU, bias"
              << (kUseBN ? ", BN" : "") << ")  DIM=" << kDim
              << "  N=" << N_final << "\n"
              << "              -> Conv(" << C1 << "->" << C2 << ", RELU, bias"
              << (kUseBN ? ", BN" : "") << ") DIM=" << kDim << "\n"
              << "              -> FLATTEN\n"
              << "              -> Linear(" << (C2 * N_final) << " -> 10)\n"
              << "Parameters:   " << total_params
              << " (" << conv1_params << " conv1 + " << conv2_params
              << " conv2 + " << readout_params << " readout)\n\n";

    train_and_evaluate("HCNN", net, train_raw, test_flat, emb, train_aug,
                       /*lr=*/0.001f, 256, 1e-3f);

    return 0;
}
