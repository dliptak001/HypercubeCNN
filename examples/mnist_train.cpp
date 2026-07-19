// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNN.h"
#include "HCNNSpatialAug.h"
#include "HCNNSpatialEmbed.h"
#include "HCNNTrainHelpers.h"
#include "HCNNDataset.h"
#include "demo_arch.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using hcnn_demo::ArchLayer;
using hcnn_demo::ArchParamSummary;

// =============================================================================
// MNIST teaching demo - thin loop on core helpers
//
// Loader yields 28x28 in [-1, 1].  Pipeline:
//   (train only) HCNNSpatialAugmenter  - rot/scale/shift/shear/elastic + noise
//   HCNNSpatialEmbedder                - DualPlaneResize -> N = 2^dim
//   HCNN TrainEpoch + cosine_lr + dual checkpoint + evaluate_classification
//
// Why default dim=11 (not "because N=2048 > 784"):
//
// DualPlaneResize does NOT pack 28x28 raw pixels into the cube. It builds TWO
// SxS maps (ink || |grad|) with S = floor(sqrt(N/2)) so 2*S*S <= N, after a
// bilinear resize of the 28x28 digit to SxS. Capacity check is 2*S*S vs N, not
// 784 vs N. Comparing N > 784 is the wrong test for this mode.
//
//   dim | N    | DualPlane S | vs native 28x28
//   ----+------+-------------+---------------------------
//    8  |  256 |    11       | heavy downsample
//    9  |  512 |    16       | downsample
//   10  | 1024 |    22       | still downsample
//   11  | 2048 |    32       | upsample 28->32 (documented ~99% path)
//   12  | 4096 |    45       | stronger upsample; larger FLATTEN head
//
// Default dim=11 so DualPlane can use S=32 with full occupancy (2*32*32=2048):
// room for two SxS maps at a side large enough for digits. Smaller dim is legal
// (layout still P <= N) but downsamples; the runtime WARNING is for S < 28.
// dim=12 is legal and sharper spatially, but not the documented recipe (costlier).
// =============================================================================

// ---------------------------------------------------------------------------
// DEVELOPER CONFIG - edit knobs here; the rest of the file follows
// ---------------------------------------------------------------------------
//
// Documented default recipe (seed 398479293): DIM=11 DualPlane, three 16-wide
// RELU convs, no pool/BN, rot/scale/shift + shear_x (elastic off), Adam,
// cosine 1e-3->1e-4, 60 epochs, batch 256, wd 1e-3.
// Measured: 99.28% best-acc / 99.25% at best-loss (shear on). Pre-shear was
// 99.31% best-acc / 99.23% at best-loss — single-seed wash; multi-seed TBD.
// ---------------------------------------------------------------------------

static constexpr int   kImgSide     = 28;      // MNIST native side (loader)
static constexpr int   kImgPixels   = 28 * 28; // 784
static constexpr float kBackground  = -1.0f;   // "ink off" after loader norm

/**
 * All developer-facing knobs for this demo.
 * Change fields / layers / seeds here - main, train loop, aug, and embed read
 * this struct.  `dim` is shared by HCNN start capacity and SpatialEmbed.
 * Architecture helpers: shared `examples/demo_arch.h` (hcnn_demo::).
 */
struct DemoConfig {
    // ----- Data -----
    size_t max_train_samples = 60000;  // 0 = all in file
    size_t max_test_samples  = 10000;

    // ----- Architecture (dim also drives SpatialEmbed N = 2^dim) -----
    int dim            = 11;   // start DIM; DualPlane auto side = floor(sqrt(N/2))
    int num_outputs    = 10;   // MNIST classes
    int input_channels = 1;    // must stay 1 (Spatial* is single-channel)
    std::vector<ArchLayer> layers = {
        // Documented ~99.3% recipe: three 16-wide RELU convs, no pool
        ArchLayer::Conv(16, hcnn::Activation::RELU, /*bias=*/true, /*bn=*/false),
        ArchLayer::Conv(16, hcnn::Activation::RELU, /*bias=*/true, /*bn=*/false),
        ArchLayer::Conv(16, hcnn::Activation::RELU, /*bias=*/true, /*bn=*/false)
        // Examples:
        // ArchLayer::Conv(32),
        // ArchLayer::Pool(hcnn::PoolType::MAX),
        // ArchLayer::Conv(32),
    };

    // ----- Weight init / optimizer -----
    // Documented default seed: 99.28% best-acc / 99.25% best-loss (shear on).
    unsigned weight_seed = 398479293;
    hcnn::OptimizerType optimizer = hcnn::OptimizerType::ADAM;

    // ----- Schedule -----
    int   epochs          = 100;
    float lr_max          = 0.001f;   // cosine peak (epoch 0)
    float lr_min_ratio    = 0.1f;     // lr_min = lr_max * ratio (floor)
    int   batch_size      = 256;
    float weight_decay    = 1e-3f;
    float momentum        = 0.9f;     // passed to TrainEpoch (Adam ignores)

    // ----- Train-time spatial aug (test path uses None()) -----
    // Affine: rot/scale/shift/shear (one inverse warp). Optional mild elastic
    // (off by default — enable after shear A/B). Then Gaussian noise.
    // See HCNNSpatialAug.h. Elastic dominates aug cost when on.
    float aug_rot_deg_max = 12.0f;    // uniform +/-deg about center
    float aug_scale_min   = 0.9f;
    float aug_scale_max   = 1.1f;
    int   aug_shift_max   = 2;        // integer px dy,dx in {-s..+s}
    float aug_shear_x_max = 0.15f;    // horizontal shear ~ U[-m,m]; MNIST slant
    float aug_shear_y_max = 0.0f;     // vertical shear off by default
    float aug_elastic_alpha = 0.0f;   // 0=off; try 1.0 after shear A/B
    float aug_elastic_sigma = 5.0f;   // used when elastic_alpha > 0 (in [0.25,32])
    float aug_noise_sigma = 0.03f;    // Gaussian after geometry; then clip
    // Aug RNG: seed = aug_seed_base + epoch * aug_seed_stride (reproducible).
    unsigned aug_seed_base   = 0xC0FFEEu;
    unsigned aug_seed_stride = 9973u;
    // Shuffle stream (independent of weight_seed): shuffle_seed = epoch + 1.

    // ----- Embed -----
    hcnn::HCNNSpatialEmbedMode embed_mode = hcnn::HCNNSpatialEmbedMode::DualPlaneResize;
    float embed_pad_value = kBackground;  // OOB / blank |grad| / unused verts
    int   embed_plane_side = 0;           // 0 = auto from dim

    float lr_min() const { return lr_max * lr_min_ratio; }
};

static ArchParamSummary summarize_demo(const DemoConfig& cfg) {
    // Demo-specific contracts beyond shared arch validation.
    if (cfg.input_channels != 1)
        throw std::runtime_error(
            "DemoConfig: input_channels must be 1 (Spatial* path is single-channel)");
    if (cfg.epochs < 1 || cfg.batch_size < 1)
        throw std::runtime_error("DemoConfig: epochs and batch_size must be >= 1");
    if (cfg.lr_max <= 0.0f || cfg.lr_min_ratio < 0.0f || cfg.lr_min_ratio > 1.0f)
        throw std::runtime_error("DemoConfig: invalid lr_max / lr_min_ratio");
    return hcnn_demo::summarize_arch(cfg.dim, cfg.num_outputs, cfg.input_channels,
                                     cfg.layers);
}

// ---------------------------------------------------------------------------
// Spatial preprocess from DemoConfig
// ---------------------------------------------------------------------------

static hcnn::HCNNSpatialAugConfig make_aug_config(const DemoConfig& cfg,
                                                  bool enabled) {
    if (!enabled)
        return hcnn::HCNNSpatialAugConfig::None();

    hcnn::HCNNSpatialAugConfig ac;
    ac.rot_deg_max    = cfg.aug_rot_deg_max;
    ac.scale_min      = cfg.aug_scale_min;
    ac.scale_max      = cfg.aug_scale_max;
    ac.shift_max      = cfg.aug_shift_max;
    ac.shear_x_max    = cfg.aug_shear_x_max;
    ac.shear_y_max    = cfg.aug_shear_y_max;
    ac.elastic_alpha  = cfg.aug_elastic_alpha;
    ac.elastic_sigma  = cfg.aug_elastic_sigma;
    ac.noise_sigma    = cfg.aug_noise_sigma;
    ac.value_min      = -1.0f;           // clip after noise (loader range)
    ac.value_max      =  1.0f;
    ac.border_value   = kBackground;     // bilinear OOB = empty paper
    ac.enabled        = true;
    return ac;
}

static hcnn::HCNNSpatialEmbedConfig make_embed_config(const DemoConfig& cfg) {
    hcnn::HCNNSpatialEmbedConfig ec;
    ec.dim        = cfg.dim;           // N = 2^dim; matches HCNN start DIM
    ec.mode       = cfg.embed_mode;
    ec.pad_value  = cfg.embed_pad_value;
    ec.plane_side = cfg.embed_plane_side;
    return ec;
}

/// Optional aug at 28x28, then SpatialEmbed into HCNNFlatDataset (length N).
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
// Each epoch rebuilds the train buffer with a fresh augment seed from DemoConfig.
static void train_and_evaluate(const char* name, hcnn::HCNN& net,
                               const HCNNDataset& train_raw,
                               const hcnn::HCNNFlatDataset& test_ds,
                               const hcnn::HCNNSpatialEmbedder& emb,
                               const hcnn::HCNNSpatialAugmenter& train_aug,
                               const DemoConfig& cfg) {
    const float lr_max = cfg.lr_max;
    const float lr_min = cfg.lr_min();
    const auto& ac = train_aug.config();

    std::cout << "\n=== " << name << " (lr_max=" << lr_max
              << ", lr_min=" << lr_min
              << ", batch=" << cfg.batch_size
              << ", wd=" << cfg.weight_decay
              << ", epochs=" << cfg.epochs
              << ", aug=rot+/-" << ac.rot_deg_max
              << "+scale[" << ac.scale_min << "," << ac.scale_max << "]"
              << "+shift+/-" << ac.shift_max
              << "+shear_x+/-" << ac.shear_x_max
              << "+shear_y+/-" << ac.shear_y_max
              << "+elastic(a=" << ac.elastic_alpha
              << ",s=" << ac.elastic_sigma << ")"
              << "+N(0," << ac.noise_sigma << ")) ===\n";
    print_eval("Initial test", hcnn::evaluate_classification(net, test_ds));

    hcnn::HCNNFlatDataset train_ds;
    hcnn::HCNNDualCheckpoint ckpt;

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        const float current_lr =
            hcnn::cosine_lr(lr_max, lr_min, epoch, cfg.epochs);

        // Fresh train-time aug each epoch; seed from DemoConfig.
        // Timer includes pack+aug rebuild (wall-clock cost of the epoch).
        auto t0 = std::chrono::steady_clock::now();
        const unsigned aug_seed =
            cfg.aug_seed_base + static_cast<unsigned>(epoch) * cfg.aug_seed_stride;
        fill_spatial_dataset(train_raw, train_ds, emb, train_aug, aug_seed);
        net.TrainEpoch(train_ds.inputs.data(), train_ds.input_length,
                       train_ds.targets.data(), train_ds.count, cfg.batch_size,
                       current_lr, cfg.momentum, cfg.weight_decay,
                       /*class_weights=*/nullptr,
                       /*shuffle_seed=*/static_cast<unsigned>(epoch + 1));
        auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const double samples_per_s =
            (secs > 0.0) ? (static_cast<double>(train_ds.count) / secs) : 0.0;

        std::string label = "Epoch " + std::to_string(epoch + 1) + "/"
                            + std::to_string(cfg.epochs);
        hcnn::HCNNClassEval r = hcnn::evaluate_classification(net, test_ds);
        print_eval(label.c_str(), r);

        auto upd = ckpt.observe(net, r.loss, r.accuracy, epoch + 1);

        std::cout << "  (lr=" << current_lr << ", " << secs << "s, "
                  << samples_per_s << " samples/s)";
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
    // -------------------------------------------------------------------------
    // Edit DemoConfig fields at the top of this file (weight_seed, layers,
    // epochs, lr, aug, dim, ...).  No other knobs below.
    // -------------------------------------------------------------------------
    const DemoConfig cfg{};

    auto src_dir = std::filesystem::path(__FILE__).parent_path().parent_path();
    auto data_dir = src_dir / "data";
    // Prefer native separators for logs. operator<< on path quotes and can mix
    // '/' from __FILE__ with '\\' from path append on Windows (ugly mojibake-ish paths).
    const std::string data_dir_str =
        std::filesystem::absolute(data_dir).lexically_normal().make_preferred().string();

    std::cout << "Loading MNIST from " << data_dir_str << "...\n";
    auto train_raw = load_mnist((data_dir / "train-images-idx3-ubyte").string(),
                                (data_dir / "train-labels-idx1-ubyte").string(),
                                cfg.max_train_samples);
    auto test_raw  = load_mnist((data_dir / "t10k-images-idx3-ubyte").string(),
                                (data_dir / "t10k-labels-idx1-ubyte").string(),
                                cfg.max_test_samples);
    std::cout << "Train: " << train_raw.size() << " samples, "
              << "Test: " << test_raw.size() << " samples\n";
    std::cout << "Threads: " << std::thread::hardware_concurrency() << "\n";

    const ArchParamSummary arch_sum = summarize_demo(cfg);

    hcnn::HCNNSpatialEmbedder emb(make_embed_config(cfg));
    hcnn::HCNNSpatialAugmenter train_aug(make_aug_config(cfg, /*enabled=*/true));
    hcnn::HCNNSpatialAugmenter test_aug(make_aug_config(cfg, /*enabled=*/false));

    const auto plan = emb.plan(kImgSide, kImgSide);
    const int N = emb.capacity();
    if (N != (1 << cfg.dim)) {
        throw std::runtime_error(
            "embed capacity does not match DemoConfig::dim");
    }

    hcnn::HCNNFlatDataset test_flat;
    fill_spatial_dataset(test_raw, test_flat, emb, test_aug, /*seed=*/0);
    if (test_flat.input_length != N) {
        throw std::runtime_error(
            "test FlatDataset input_length != embed capacity");
    }

    std::cout << "Spatial pipeline: HCNNSpatialAug (train) -> "
              << "HCNNSpatialEmbed DualPlaneResize "
              << plan.plane_side << "x" << plan.plane_side
              << " ink || |grad|  (pattern_length=" << plan.pattern_length
              << ", N=" << N << ", dim=" << cfg.dim << ")\n";
    // DualPlane / ResizeToFit always fit N by changing S. That is valid capacity
    // math, but S < native 28 is a silent quality cliff (e.g. dim=8 -> S=11).
    if (plan.plane_side > 0 && plan.plane_side < kImgSide) {
        // ASCII only (no em-dash): Windows consoles often mis-decode UTF-8 as mojibake.
        std::cout
            << "\n"
            << "*** WARNING: embed plane side S=" << plan.plane_side
            << " < native " << kImgSide << "x" << kImgSide
            << " - input is DOWN-SAMPLED (dim=" << cfg.dim
            << ", N=" << N << ").\n"
            << "***          Layout is legal (P <= N); accuracy will not match "
               "the documented ~99% recipe\n"
            << "***          (default dim=11, DualPlane S=32). Raise dim or "
               "set embed_plane_side if that was unintentional.\n"
            << "\n";
    }
    std::cout << "Train aug:  rot +/-" << cfg.aug_rot_deg_max
              << " deg, scale [" << cfg.aug_scale_min << "," << cfg.aug_scale_max
              << "], shift +/-" << cfg.aug_shift_max << " px, "
              << "shear_x +/-" << cfg.aug_shear_x_max
              << ", shear_y +/-" << cfg.aug_shear_y_max
              << ", elastic alpha=" << cfg.aug_elastic_alpha
              << " sigma=" << cfg.aug_elastic_sigma
              << ", Gaussian noise sigma=" << cfg.aug_noise_sigma
              << " (train only, refreshed each epoch)\n";

    hcnn::HCNN net(cfg.dim, cfg.num_outputs, cfg.input_channels);
    hcnn_demo::apply_arch(net, cfg.dim, cfg.num_outputs, cfg.input_channels,
                          cfg.layers);
    net.RandomizeWeights(/*scale=*/0.0f, cfg.weight_seed);
    net.SetOptimizer(cfg.optimizer);

    if (net.GetStartDim() != cfg.dim || net.GetStartN() != N) {
        throw std::runtime_error(
            "HCNN start DIM/N does not match DemoConfig / SpatialEmbed");
    }

    std::cout << "Weight init seed: " << cfg.weight_seed << "\n";
    hcnn_demo::print_arch(std::cout, cfg.dim, cfg.num_outputs, cfg.input_channels,
                          cfg.layers, arch_sum);

    if (static_cast<long long>(net.GetWeightCount()) != arch_sum.total) {
        throw std::runtime_error(
            "DemoConfig param count " + std::to_string(arch_sum.total)
            + " != HCNN::GetWeightCount " + std::to_string(net.GetWeightCount()));
    }

    train_and_evaluate("HCNN", net, train_raw, test_flat, emb, train_aug, cfg);
    return 0;
}
