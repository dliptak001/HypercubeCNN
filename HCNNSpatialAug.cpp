// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNNSpatialAug.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>

namespace hcnn {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

bool HCNNSpatialAugConfig::is_identity() const {
    if (!enabled) return true;
    if (std::fabs(rot_deg_max) > 0.0f) return false;
    if (std::abs(shift_max) > 0) return false;
    if (noise_sigma > 0.0f) return false;
    const float s_lo = std::min(scale_min, scale_max);
    const float s_hi = std::max(scale_min, scale_max);
    return s_lo == 1.0f && s_hi == 1.0f;
}

void HCNNSpatialAugConfig::validate() const {
    if (!(value_min <= value_max)) {
        throw std::runtime_error(
            "HCNNSpatialAugConfig: value_min must be <= value_max");
    }
    if (noise_sigma < 0.0f) {
        throw std::runtime_error(
            "HCNNSpatialAugConfig: noise_sigma must be >= 0");
    }
    // rot_deg_max / shift_max: magnitude is used; negative is accepted as abs.
}

HCNNSpatialAugConfig HCNNSpatialAugConfig::None() {
    HCNNSpatialAugConfig c;
    c.enabled = false;
    return c;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

static float sample_bilinear(const float* img, int height, int width,
                             float y, float x, float border) {
    const int y0 = static_cast<int>(std::floor(y));
    const int x0 = static_cast<int>(std::floor(x));
    const int y1 = y0 + 1;
    const int x1 = x0 + 1;
    const float wy = y - static_cast<float>(y0);
    const float wx = x - static_cast<float>(x0);

    auto at = [img, height, width, border](int yy, int xx) -> float {
        if (yy < 0 || xx < 0 || yy >= height || xx >= width)
            return border;
        return img[yy * width + xx];
    };

    const float v00 = at(y0, x0);
    const float v01 = at(y0, x1);
    const float v10 = at(y1, x0);
    const float v11 = at(y1, x1);
    const float v0 = v00 * (1.0f - wx) + v01 * wx;
    const float v1 = v10 * (1.0f - wx) + v11 * wx;
    return v0 * (1.0f - wy) + v1 * wy;
}

// Inverse of: scale about center -> rotate about center -> integer shift.
static void warp_affine(const float* src, float* dst,
                        int height, int width,
                        float deg, float scale, int dy, int dx,
                        float border) {
    const float cy = 0.5f * static_cast<float>(height - 1);
    const float cx = 0.5f * static_cast<float>(width - 1);
    const float s = (scale > 1e-6f) ? scale : 1.0f;
    const float rad = deg * (static_cast<float>(std::numbers::pi) / 180.0f);
    const float c = std::cos(-rad);
    const float sn = std::sin(-rad);
    const float inv_s = 1.0f / s;
    const float fdy = static_cast<float>(dy);
    const float fdx = static_cast<float>(dx);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float yy = static_cast<float>(y) - fdy - cy;
            const float xx = static_cast<float>(x) - fdx - cx;
            const float xr = c * xx - sn * yy;
            const float yr = sn * xx + c * yy;
            const float sx = xr * inv_s + cx;
            const float sy = yr * inv_s + cy;
            dst[y * width + x] = sample_bilinear(src, height, width, sy, sx, border);
        }
    }
}

// ---------------------------------------------------------------------------
// HCNNSpatialAugmenter
// ---------------------------------------------------------------------------

HCNNSpatialAugmenter::HCNNSpatialAugmenter(HCNNSpatialAugConfig cfg)
    : cfg_(cfg) {
    cfg_.validate();
}

void HCNNSpatialAugmenter::set_config(const HCNNSpatialAugConfig& cfg) {
    cfg.validate();
    cfg_ = cfg;
}

void HCNNSpatialAugmenter::apply(const float* in, float* out,
                                 int height, int width,
                                 std::mt19937& rng) const {
    if (!in || !out) {
        throw std::runtime_error("HCNNSpatialAugmenter::apply: null buffer");
    }
    if (height < 1 || width < 1) {
        throw std::runtime_error(
            "HCNNSpatialAugmenter::apply: height and width must be >= 1");
    }

    const std::size_t n =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);

    if (!cfg_.enabled || cfg_.is_identity()) {
        if (in != out)
            std::memcpy(out, in, n * sizeof(float));
        return;
    }

    const float rot_span = std::fabs(cfg_.rot_deg_max);
    const int shift_span = std::abs(cfg_.shift_max);
    const float s_lo = std::min(cfg_.scale_min, cfg_.scale_max);
    const float s_hi = std::max(cfg_.scale_min, cfg_.scale_max);
    const bool do_rot = rot_span > 0.0f;
    const bool do_scale = !(s_lo == 1.0f && s_hi == 1.0f);
    const bool do_shift = shift_span > 0;
    const bool do_geom = do_rot || do_scale || do_shift;
    const bool do_noise = cfg_.noise_sigma > 0.0f;

    float deg = 0.0f;
    float scale = 1.0f;
    int dy = 0, dx = 0;

    if (do_rot) {
        std::uniform_real_distribution<float> dist(-rot_span, rot_span);
        deg = dist(rng);
    }
    if (do_scale) {
        const float lo = std::max(s_lo, 1e-6f);
        const float hi = std::max(s_hi, lo);
        std::uniform_real_distribution<float> dist(lo, hi);
        scale = dist(rng);
    }
    if (do_shift) {
        std::uniform_int_distribution<int> dist(-shift_span, shift_span);
        dy = dist(rng);
        dx = dist(rng);
    }

    if (do_geom) {
        if (in == out) {
            throw std::runtime_error(
                "HCNNSpatialAugmenter::apply: geometric aug requires in != out");
        }
        warp_affine(in, out, height, width, deg, scale, dy, dx, cfg_.border_value);
    } else if (in != out) {
        std::memcpy(out, in, n * sizeof(float));
    }

    if (do_noise) {
        std::normal_distribution<float> dist(0.0f, cfg_.noise_sigma);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = clampf(out[i] + dist(rng), cfg_.value_min, cfg_.value_max);
    }
}

void HCNNSpatialAugmenter::apply_batch(const float* in, float* out,
                                       int batch, int height, int width,
                                       std::mt19937& rng) const {
    if (batch < 0) {
        throw std::runtime_error(
            "HCNNSpatialAugmenter::apply_batch: batch must be >= 0");
    }
    if (!in || !out) {
        throw std::runtime_error("HCNNSpatialAugmenter::apply_batch: null buffer");
    }
    const std::size_t plane =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    for (int b = 0; b < batch; ++b) {
        apply(in + static_cast<std::size_t>(b) * plane,
              out + static_cast<std::size_t>(b) * plane,
              height, width, rng);
    }
}

} // namespace hcnn
