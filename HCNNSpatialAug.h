// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include <cstdint>
#include <random>

namespace hcnn {

/**
 * @brief Configuration for optional 2D spatial augmentation.
 *
 * Operates on single-channel row-major images of any height × width.
 * Independent of hypercube DIM / vertex count — callers map augmented
 * patterns onto the network input separately (pad, pack, multi-view, etc.).
 *
 * Geometry (rotate, scale, shift) is applied as **one** inverse bilinear
 * warp when any geometric term is active. Gaussian noise is applied after
 * the warp and values are clipped to [value_min, value_max].
 *
 * Defaults are all identity (no augmentation). Use `enabled = false` as a
 * master off switch even if other fields are non-zero.
 */
struct HCNNSpatialAugConfig {
    /// Uniform rotation in degrees over [-rot_deg_max, +rot_deg_max]. 0 = off.
    float rot_deg_max = 0.0f;

    /// Uniform scale factor over [scale_min, scale_max]. Both 1 = off.
    float scale_min = 1.0f;
    float scale_max = 1.0f;

    /// Integer pixel translation: dy, dx ~ U{-shift_max,...,+shift_max}. 0 = off.
    int shift_max = 0;

    /// Additive Gaussian noise N(0, noise_sigma^2) after warp. 0 = off.
    float noise_sigma = 0.0f;

    /// Clip range after noise (and identity path does not clip unless noise runs).
    float value_min = -1.0f;
    float value_max =  1.0f;

    /// Sampled value for bilinear out-of-bounds lookups.
    float border_value = 0.0f;

    /// Master switch. false => apply() copies in→out (no RNG draws).
    bool enabled = true;

    /// True when apply() is a pure copy under this config.
    bool is_identity() const;

    /// All fields at identity; enabled remains true but ops are no-ops.
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
    explicit HCNNSpatialAugmenter(HCNNSpatialAugConfig cfg = {});

    void set_config(const HCNNSpatialAugConfig& cfg);
    const HCNNSpatialAugConfig& config() const { return cfg_; }

    /**
     * Augment one row-major image.
     *
     * @param in      Source buffer, length height*width (may equal out only
     *                when is_identity(); geometric warp needs distinct buffers
     *                or a temp — if in == out and geometry is active, behavior
     *                is undefined; use apply with separate buffers).
     * @param out     Destination, length height*width.
     * @param height  Image rows (> 0).
     * @param width   Image cols (> 0).
     * @param rng     Caller-owned RNG; advanced by this call when not identity.
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
