// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include <cstdint>
#include <random>

namespace hcnn {

/**
 * @brief Configuration for optional 2D spatial augmentation.
 *
 * Operates on single-channel row-major images of any height x width.
 * Independent of hypercube DIM / vertex count -- callers map augmented
 * patterns onto the network input separately (see HCNNSpatialEmbed).
 *
 * Geometry (rotate, scale, shift) is applied as **one** inverse bilinear
 * warp when any geometric term is active. Gaussian noise is applied after
 * the warp; values are clipped to [value_min, value_max] when noise runs
 * (geometry-only paths do not clip).
 *
 * Defaults are identity (no augmentation). `None()` sets `enabled = false`
 * (master off; apply is a pure copy and draws no RNG).
 *
 * TODO(aug-next): planned extensions for the MNIST plateau (deferred; pull
 * other threads first). Keep aug-then-embed; do not aug on packed vertices.
 *   1) Shear in the existing single affine warp (start shear_x ~ U[-0.15,0.15],
 *      shear_y mild or off) — best cheap ROI vs FLATTEN slant memorization.
 *   2) Mild elastic (Simard-style smooth displacement, small amplitude so
 *      DualPlane |grad| stays sane) — higher variance; multi-seed to claim.
 * Expected buy: ~0.02-0.1 pp best-acc, not a free jump to 99.5%. Sequence:
 * shear A/B on seed 398479293, then elastic on top, then 3-seed mean.
 * See examples/mnist_train.md "Deferred TODO".
 */
struct HCNNSpatialAugConfig {
    /// Uniform rotation in degrees over [-|rot_deg_max|, +|rot_deg_max|]. 0 = off.
    float rot_deg_max = 0.0f;

    /// Uniform scale factor over [scale_min, scale_max] (order-independent).
    /// Both 1 = off. Non-positive bounds are clamped to a tiny positive floor.
    float scale_min = 1.0f;
    float scale_max = 1.0f;

    /// Integer pixel translation: dy, dx ~ U{-|s|,...,+|s|} with s = |shift_max|.
    /// 0 = off.
    int shift_max = 0;

    /// Additive Gaussian noise N(0, noise_sigma^2) after warp. 0 = off.
    float noise_sigma = 0.0f;

    /// Clip range after noise. Must satisfy value_min <= value_max (validated).
    float value_min = -1.0f;
    float value_max =  1.0f;

    /// Sampled value for bilinear out-of-bounds lookups.
    float border_value = 0.0f;

    /// Master switch. false => apply() copies in->out (no RNG draws).
    bool enabled = true;

    /// True when apply() is a pure copy under this config (no RNG).
    bool is_identity() const;

    /// Validate field ranges; throws std::runtime_error if invalid.
    void validate() const;

    /// Master off: enabled = false (apply is memcpy, no RNG).
    static HCNNSpatialAugConfig None();
};

/**
 * @class HCNNSpatialAugmenter
 * @brief Stateless (aside from config) 2D spatial augmenter.
 *
 * Thread-safe for concurrent apply() calls with distinct rng instances
 * when config is not mutated. Not safe to call set_config concurrently
 * with apply.
 *
 * @code
 * hcnn::HCNNSpatialAugConfig cfg;
 * cfg.rot_deg_max = 12.0f;
 * cfg.scale_min = 0.9f;
 * cfg.scale_max = 1.1f;
 * cfg.shift_max = 2;
 * cfg.noise_sigma = 0.03f;
 * cfg.border_value = -1.0f;
 *
 * hcnn::HCNNSpatialAugmenter aug(cfg);
 * std::mt19937 rng(seed);
 * aug.apply(src, dst, height, width, rng);
 * @endcode
 */
class HCNNSpatialAugmenter {
public:
    /// Constructs and validates `cfg` (throws if invalid).
    explicit HCNNSpatialAugmenter(HCNNSpatialAugConfig cfg = {});

    /// Replaces config after validate() (throws if invalid).
    void set_config(const HCNNSpatialAugConfig& cfg);
    const HCNNSpatialAugConfig& config() const { return cfg_; }

    /**
     * Augment one row-major image.
     *
     * @param in      Source, length height*width. May equal out only when
     *                the config is identity or noise-only (no geometry).
     *                Geometric warp requires in != out.
     * @param out     Destination, length height*width.
     * @param height  Image rows (>= 1).
     * @param width   Image cols (>= 1).
     * @param rng     Caller-owned RNG; advanced when not identity.
     */
    void apply(const float* in, float* out,
               int height, int width,
               std::mt19937& rng) const;

    /**
     * Augment `batch` images packed contiguously (sample stride = height*width).
     * Each sample draws independent geometry/noise from `rng`.
     */
    void apply_batch(const float* in, float* out,
                     int batch, int height, int width,
                     std::mt19937& rng) const;

private:
    HCNNSpatialAugConfig cfg_;
};

} // namespace hcnn
