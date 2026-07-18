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
#include <string>
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

// ---------------------------------------------------------------------------
// Architecture config — edit the layer list; build + print follow automatically
// ---------------------------------------------------------------------------

/// One stack step: Hamming conv or antipodal pool.
struct ArchLayer {
    enum class Kind { Conv, Pool };

    Kind kind = Kind::Conv;

    // Conv
    int c_out = 16;
    hcnn::Activation activation = hcnn::Activation::RELU;
    bool use_bias = true;
    bool use_bn = false;

    // Pool
    hcnn::PoolType pool_type = hcnn::PoolType::MAX;

    static ArchLayer Conv(int c_out,
                          hcnn::Activation act = hcnn::Activation::RELU,
                          bool bias = true,
                          bool bn = false) {
        ArchLayer L;
        L.kind = Kind::Conv;
        L.c_out = c_out;
        L.activation = act;
        L.use_bias = bias;
        L.use_bn = bn;
        return L;
    }

    static ArchLayer Pool(hcnn::PoolType type = hcnn::PoolType::MAX) {
        ArchLayer L;
        L.kind = Kind::Pool;
        L.pool_type = type;
        return L;
    }
};

/// Network topology for the teaching demo (not part of the core SDK).
/// `dim` is the single source of truth for both HCNN start DIM and SpatialEmbed.
struct ArchConfig {
    int dim = kDim;              // start DIM (N = 2^dim); also drives SpatialEmbed
    int num_outputs = 10;        // class count
    int input_channels = 1;      // must be 1 for this demo (Spatial* is single-channel)
    std::vector<ArchLayer> layers;

    /// Default MNIST recipe: 16 -> 16, no pool, no BN.
    static ArchConfig MnistDefault() {
        ArchConfig a;
        a.dim = kDim;
        a.num_outputs = 10;
        a.input_channels = 1;
        a.layers = {
            ArchLayer::Conv(16, hcnn::Activation::RELU, /*bias=*/true, /*bn=*/false),
            ArchLayer::Conv(16, hcnn::Activation::RELU, /*bias=*/true, /*bn=*/false),
        };
        return a;
    }
};

static const char* activation_name(hcnn::Activation a) {
    switch (a) {
        case hcnn::Activation::NONE:       return "NONE";
        case hcnn::Activation::RELU:       return "RELU";
        case hcnn::Activation::LEAKY_RELU: return "LEAKY_RELU";
        case hcnn::Activation::TANH:       return "TANH";
    }
    return "?";
}

static const char* pool_name(hcnn::PoolType t) {
    switch (t) {
        case hcnn::PoolType::MAX: return "MAX";
        case hcnn::PoolType::AVG: return "AVG";
    }
    return "?";
}

/**
 * Walk ArchConfig: track DIM/N/channels, collect per-conv param counts, and
 * FLATTEN readout size. Matches GetWeightCount (kernel + bias; BN γ/β omitted).
 * Each pool reduces DIM by 1 and does not add parameters.
 *
 * Validates demo contracts: dim in range, input_channels == 1, >=1 conv,
 * c_out >= 1, pools do not drive DIM below 0.
 */
struct ArchParamSummary {
    long long total = 0;
    long long readout = 0;
    long long flatten_features = 0;  // last_c * N_final
    int final_dim = 0;
    int final_N = 0;
    int last_channels = 0;
    int num_conv = 0;
    std::vector<long long> conv_params;
};

static ArchParamSummary summarize_arch(const ArchConfig& arch) {
    if (arch.dim < 1 || arch.dim > 30)
        throw std::runtime_error("ArchConfig: dim must be in [1, 30]");
    if (arch.num_outputs < 1)
        throw std::runtime_error("ArchConfig: num_outputs must be >= 1");
    // Spatial preprocess in this demo is single-channel only.
    if (arch.input_channels != 1)
        throw std::runtime_error(
            "ArchConfig: input_channels must be 1 (Spatial* path is single-channel)");
    if (arch.layers.empty())
        throw std::runtime_error("ArchConfig: need at least one layer");

    ArchParamSummary s;
    int dim = arch.dim;
    int N = 1 << dim;
    int c_in = arch.input_channels;
    s.last_channels = c_in;

    for (const auto& L : arch.layers) {
        if (L.kind == ArchLayer::Kind::Conv) {
            if (L.c_out < 1)
                throw std::runtime_error("ArchConfig: conv c_out must be >= 1");
            // Hamming kernel size K = current DIM.
            const long long k_params =
                static_cast<long long>(c_in) * L.c_out * dim
                + (L.use_bias ? L.c_out : 0);
            s.conv_params.push_back(k_params);
            s.total += k_params;
            c_in = L.c_out;
            s.last_channels = L.c_out;
            ++s.num_conv;
        } else {
            dim -= 1;
            if (dim < 0)
                throw std::runtime_error("ArchConfig: too many pools (DIM < 0)");
            N = 1 << dim;
        }
    }

    if (s.num_conv < 1)
        throw std::runtime_error("ArchConfig: need at least one Conv layer");

    s.final_dim = dim;
    s.final_N = N;
    s.flatten_features = static_cast<long long>(s.last_channels) * N;
    s.readout = s.flatten_features * arch.num_outputs + arch.num_outputs;
    s.total += s.readout;
    return s;
}

/// Apply ArchConfig to an empty HCNN (AddConv / AddPool only). Validates first.
static void apply_arch(hcnn::HCNN& net, const ArchConfig& arch) {
    (void)summarize_arch(arch);  // validate contracts before mutating net
    for (const auto& L : arch.layers) {
        if (L.kind == ArchLayer::Kind::Conv)
            net.AddConv(L.c_out, L.activation, L.use_bias, L.use_bn);
        else
            net.AddPool(L.pool_type);
    }
}

/// Print stack and parameter breakdown (uses a precomputed summary).
static void print_arch(std::ostream& os, const ArchConfig& arch,
                       const ArchParamSummary& sum) {
    int dim = arch.dim;
    int N = 1 << dim;
    int c_in = arch.input_channels;

    os << "\nArchitecture: ";
    bool first_line = true;
    for (const auto& L : arch.layers) {
        if (!first_line)
            os << "              -> ";
        first_line = false;

        if (L.kind == ArchLayer::Kind::Conv) {
            os << "Conv(" << c_in << "->" << L.c_out
               << ", " << activation_name(L.activation);
            if (L.use_bias) os << ", bias";
            if (L.use_bn)   os << ", BN";
            os << ")  DIM=" << dim << "  N=" << N << "\n";
            c_in = L.c_out;
        } else {
            os << "Pool(" << pool_name(L.pool_type) << ")  DIM "
               << dim << "->" << (dim - 1)
               << "  N " << N << "->" << (N / 2) << "\n";
            dim -= 1;
            N = 1 << dim;
        }
    }

    os << "              -> FLATTEN\n"
       << "              -> Linear(" << sum.flatten_features
       << " -> " << arch.num_outputs << ")\n"
       << "Parameters:   " << sum.total << " (";
    for (size_t i = 0; i < sum.conv_params.size(); ++i) {
        if (i) os << " + ";
        os << sum.conv_params[i] << " conv" << (i + 1);
    }
    os << " + " << sum.readout << " readout)\n\n";
}

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

/// Dual-plane embed driven by ArchConfig::dim (same N as the network start).
static hcnn::HCNNSpatialEmbedConfig make_mnist_embed_config(int dim) {
    hcnn::HCNNSpatialEmbedConfig cfg;
    cfg.dim        = dim;  // N = 2^dim; must match HCNN start DIM
    cfg.mode       = hcnn::HCNNSpatialEmbedMode::DualPlaneResize;
    cfg.pad_value  = kBackground;
    cfg.plane_side = 0;  // auto: floor(sqrt(N/2)); 32 at DIM=11, 16 at DIM=9
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
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        // Guard zero/near-zero duration (tiny debug subsets or coarse timer).
        const double samples_per_s =
            (secs > 0.0) ? (static_cast<double>(train_ds.count) / secs) : 0.0;

        std::string label = "Epoch " + std::to_string(epoch + 1) + "/"
                            + std::to_string(epochs);
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

    // Edit layers / dim here — embed, net, print, and param counts follow.
    // Documented recipe: DIM=11, 16 -> 16, no pool, no BN.
    ArchConfig arch = ArchConfig::MnistDefault();
    // Examples:
    //   arch.layers = { ArchLayer::Conv(32), ArchLayer::Conv(32) };
    //   arch.layers = { ArchLayer::Conv(16), ArchLayer::Pool(),
    //                   ArchLayer::Conv(32), ArchLayer::Pool() };
    //   arch.dim = 9;  // also shrinks DualPlane embed (auto S=16)

    const ArchParamSummary arch_sum = summarize_arch(arch);

    // Embed dim tracks arch.dim so N_input == network start capacity.
    hcnn::HCNNSpatialEmbedder emb(make_mnist_embed_config(arch.dim));
    hcnn::HCNNSpatialAugmenter train_aug(make_mnist_aug_config(/*enabled=*/true));
    hcnn::HCNNSpatialAugmenter test_aug(make_mnist_aug_config(/*enabled=*/false));

    const auto plan = emb.plan(kImgSide, kImgSide);
    const int N = emb.capacity();
    if (N != (1 << arch.dim) || N != emb.config().capacity()) {
        throw std::runtime_error(
            "embed capacity does not match ArchConfig::dim");
    }

    // Test: embed only (no aug).  Train: re-embedded with aug each epoch.
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
              << ", N=" << N << ", dim=" << arch.dim << ")\n";
    std::cout << "Train aug:  rot +/-12 deg, scale [0.9,1.1], shift +/-2 px, "
              << "Gaussian noise sigma=0.03 (train only, refreshed each epoch)\n";

    // Weight-init seed only (aug / shuffle seeds are fixed separately).
    // Documented multi-seed ~99.23% mean was measured on this recipe (DIM=11
    // dual pack + geometric aug); re-run a seed if you change arch/embed.
    constexpr unsigned weight_seed = 398479293;

    hcnn::HCNN net(arch.dim, arch.num_outputs, arch.input_channels);
    apply_arch(net, arch);
    net.RandomizeWeights(/*scale=*/0.0f, weight_seed);
    net.SetOptimizer(hcnn::OptimizerType::ADAM);

    if (net.GetStartDim() != arch.dim || net.GetStartN() != N) {
        throw std::runtime_error(
            "HCNN start DIM/N does not match ArchConfig / SpatialEmbed");
    }

    std::cout << "Weight init seed: " << weight_seed << "\n";
    print_arch(std::cout, arch, arch_sum);

    // Printed total must match the SDK weight blob (kernel + bias; no BN γ/β).
    if (static_cast<long long>(net.GetWeightCount()) != arch_sum.total) {
        throw std::runtime_error(
            "ArchConfig param count " + std::to_string(arch_sum.total)
            + " != HCNN::GetWeightCount " + std::to_string(net.GetWeightCount()));
    }

    train_and_evaluate("HCNN", net, train_raw, test_flat, emb, train_aug,
                       /*lr=*/0.001f, 256, 1e-3f);

    return 0;
}
