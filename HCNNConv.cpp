#include "HCNNConv.h"
#include "ThreadPool.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Minimum DIM at which per-layer threading kicks in.
// Below this, fork-join overhead exceeds the per-vertex work.
static constexpr int THREAD_DIM_THRESHOLD = 12;

// Vertex-loop tile size for cache locality.  Must be a power of 2 and a
// multiple of 8 (AVX-256 width).  At DIM=10 with T=64, 6 of 10 masks
// stay within-tile; the remaining 4 each touch exactly one other tile,
// keeping the working set in L1.
static constexpr size_t TILE = 64;

HCNNConv::HCNNConv(int dim, int c_in, int c_out, Activation activation,
                   bool use_bias, bool use_batchnorm)
    : DIM(dim), N(1 << dim), c_in(c_in), c_out(c_out),
      K(dim),
      activation(activation), use_bias(use_bias), use_batchnorm(use_batchnorm),
      kernel(c_out * c_in * K, 0.0f),
      bias(use_bias ? c_out : 0, 0.0f),
      kernel_m(c_out * c_in * K, 0.0f),
      bias_m(use_bias ? c_out : 0, 0.0f),
      bn_gamma(use_batchnorm ? c_out : 0, 1.0f),
      bn_beta(use_batchnorm ? c_out : 0, 0.0f),
      bn_running_mean(use_batchnorm ? c_out : 0, 0.0f),
      bn_running_var(use_batchnorm ? c_out : 0, 1.0f),
      bn_gamma_m(use_batchnorm ? c_out : 0, 0.0f),
      bn_beta_m(use_batchnorm ? c_out : 0, 0.0f) {
    if (DIM < 3) {
        throw std::runtime_error("HCNNConv requires DIM >= 3");
    }
}

void HCNNConv::randomize_weights(float scale, std::mt19937& rng) {
    // Auto-select initialization based on activation:
    //   ReLU/LeakyReLU: He/Kaiming uniform, scale = sqrt(6 / fan_in)
    //     (accounts for the variance-halving effect of ReLU)
    //   NONE (linear): Xavier/Glorot uniform, scale = sqrt(6 / (fan_in + fan_out))
    // fan_in = c_in * K, fan_out = c_out * K.
    if (scale <= 0.0f) {
        float fan_in  = static_cast<float>(c_in * K);
        float fan_out = static_cast<float>(c_out * K);
        // He/Kaiming for ReLU layers with c_in > 1 (intermediate layers).
        // First layer (c_in=1) uses Xavier — its input is raw data, not
        // post-ReLU activations, so the He variance assumption doesn't hold.
        if ((activation == Activation::RELU || activation == Activation::LEAKY_RELU)
            && c_in > 1) {
            scale = std::sqrt(6.0f / fan_in);
        } else {
            scale = std::sqrt(6.0f / (fan_in + fan_out));
        }
    }
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& w : kernel) w = dist(rng);
    if (use_bias) {
        for (auto& b : bias) b = 0.0f;
    }
    std::fill(kernel_m.begin(), kernel_m.end(), 0.0f);
    std::fill(bias_m.begin(), bias_m.end(), 0.0f);
    std::fill(kernel_m2.begin(), kernel_m2.end(), 0.0f);
    std::fill(bias_m2.begin(), bias_m2.end(), 0.0f);

    if (use_batchnorm) {
        std::fill(bn_gamma.begin(), bn_gamma.end(), 1.0f);
        std::fill(bn_beta.begin(), bn_beta.end(), 0.0f);
        std::fill(bn_running_mean.begin(), bn_running_mean.end(), 0.0f);
        std::fill(bn_running_var.begin(), bn_running_var.end(), 1.0f);
        std::fill(bn_gamma_m.begin(), bn_gamma_m.end(), 0.0f);
        std::fill(bn_beta_m.begin(), bn_beta_m.end(), 0.0f);
        std::fill(bn_gamma_m2.begin(), bn_gamma_m2.end(), 0.0f);
        std::fill(bn_beta_m2.begin(), bn_beta_m2.end(), 0.0f);
    }
}

void HCNNConv::set_optimizer(OptimizerType type, float beta1, float beta2, float eps) {
    optimizer_type_ = type;
    adam_beta1_ = beta1;
    adam_beta2_ = beta2;
    adam_eps_ = eps;
    if (type == OptimizerType::ADAM) {
        kernel_m2.assign(kernel.size(), 0.0f);
        bias_m2.assign(bias.size(), 0.0f);
        if (use_batchnorm) {
            bn_gamma_m2.assign(c_out, 0.0f);
            bn_beta_m2.assign(c_out, 0.0f);
        }
    } else {
        kernel_m2.clear(); kernel_m2.shrink_to_fit();
        bias_m2.clear(); bias_m2.shrink_to_fit();
        bn_gamma_m2.clear(); bn_gamma_m2.shrink_to_fit();
        bn_beta_m2.clear(); bn_beta_m2.shrink_to_fit();
    }
}

// ---------------------------------------------------------------------------
// Forward: vertex-level threading within each output channel.
// Each thread handles a contiguous vertex range — no write conflicts.
// Tiled: output tile stays in L1 for full ci*K accumulation + activation.
//
// When BN is enabled, the loop is split: tiled accumulation, then BN
// (needs global channel stats), then activation.  When BN is disabled,
// the original fused tiled accumulation+activation path is used.
// ---------------------------------------------------------------------------
void HCNNConv::forward(const float* in, float* out, float* pre_act,
                   float* bn_save) const {
    const bool use_threads = thread_pool && DIM >= THREAD_DIM_THRESHOLD;

    for (int co = 0; co < c_out; ++co) {
        float* out_co = out + co * N;
        float b = use_bias ? bias[co] : 0.0f;

        // Tiled accumulation lambda (shared by both BN and non-BN paths)
        auto do_accumulate = [&](size_t v_begin, size_t v_end) {
            for (size_t t = v_begin; t < v_end; t += TILE) {
                size_t t_end = std::min(t + TILE, v_end);
                for (size_t v = t; v < t_end; ++v)
                    out_co[v] = b;
                for (int ci = 0; ci < c_in; ++ci) {
                    const float* in_ci = in + ci * N;
                    for (int k = 0; k < K; ++k) {
                        float w = kernel[kernel_idx(co, ci, k)];
                        uint32_t m = 1u << k;
                        for (size_t v = t; v < t_end; ++v)
                            out_co[v] += w * in_ci[v ^ m];
                    }
                }
            }
        };

        if (use_batchnorm) {
            // Split path: accumulate → BN → activate

            // Phase 1: Tiled weighted sum (no activation yet)
            if (use_threads) {
                thread_pool->ForEach(static_cast<size_t>(N),
                    [&](size_t, size_t begin, size_t end) { do_accumulate(begin, end); });
            } else {
                do_accumulate(0, static_cast<size_t>(N));
            }

            // Phase 2: Batch normalization across all N vertices for this channel
            if (training_) {
                // Compute per-channel mean
                float mean = 0.0f;
                for (int v = 0; v < N; ++v) mean += out_co[v];
                mean /= static_cast<float>(N);

                // Compute per-channel variance
                float var = 0.0f;
                for (int v = 0; v < N; ++v) {
                    float d = out_co[v] - mean;
                    var += d * d;
                }
                var /= static_cast<float>(N);

                float inv_std = 1.0f / std::sqrt(var + bn_eps_);

                // Normalize, scale, shift
                for (int v = 0; v < N; ++v) {
                    float x_hat = (out_co[v] - mean) * inv_std;
                    out_co[v] = bn_gamma[co] * x_hat + bn_beta[co];
                }

                // Save inv_std, mean, var for backward and batch stats accumulation
                if (bn_save) {
                    bn_save[co] = inv_std;
                    bn_save[c_out + co] = mean;
                    bn_save[2 * c_out + co] = var;
                }

                // Update running stats (EMA) — skipped during batch-parallel mode
                if (!skip_running_stats_) {
                    float unbiased_var = var * static_cast<float>(N)
                                       / static_cast<float>(N - 1);
                    bn_running_mean[co] = (1.0f - bn_momentum_) * bn_running_mean[co]
                                        + bn_momentum_ * mean;
                    bn_running_var[co] = (1.0f - bn_momentum_) * bn_running_var[co]
                                       + bn_momentum_ * unbiased_var;
                }
            } else {
                // Eval mode: use running statistics
                float inv_std = 1.0f / std::sqrt(bn_running_var[co] + bn_eps_);
                float rm = bn_running_mean[co];
                for (int v = 0; v < N; ++v) {
                    float x_hat = (out_co[v] - rm) * inv_std;
                    out_co[v] = bn_gamma[co] * x_hat + bn_beta[co];
                }
            }

            // Phase 3: Activation
            if (pre_act) {
                float* pa = pre_act + co * N;
                for (int v = 0; v < N; ++v) {
                    pa[v] = out_co[v];
                    out_co[v] = activate(out_co[v]);
                }
            } else {
                for (int v = 0; v < N; ++v)
                    out_co[v] = activate(out_co[v]);
            }

        } else {
            // Fused path (no BN): accumulate + activate in one tiled pass
            auto do_vertices = [&](size_t v_begin, size_t v_end) {
                for (size_t t = v_begin; t < v_end; t += TILE) {
                    size_t t_end = std::min(t + TILE, v_end);
                    for (size_t v = t; v < t_end; ++v)
                        out_co[v] = b;
                    for (int ci = 0; ci < c_in; ++ci) {
                        const float* in_ci = in + ci * N;
                        for (int k = 0; k < K; ++k) {
                            float w = kernel[kernel_idx(co, ci, k)];
                            uint32_t m = 1u << k;
                            for (size_t v = t; v < t_end; ++v)
                                out_co[v] += w * in_ci[v ^ m];
                        }
                    }
                    if (pre_act) {
                        float* pa = pre_act + co * N;
                        for (size_t v = t; v < t_end; ++v) {
                            pa[v] = out_co[v];
                            out_co[v] = activate(out_co[v]);
                        }
                    } else {
                        for (size_t v = t; v < t_end; ++v)
                            out_co[v] = activate(out_co[v]);
                    }
                }
            };

            if (use_threads) {
                thread_pool->ForEach(static_cast<size_t>(N),
                    [&](size_t, size_t begin, size_t end) { do_vertices(begin, end); });
            } else {
                do_vertices(0, static_cast<size_t>(N));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Backward: vertex-level threading for input gradients (tiled).
// Channel-level threading for weight gradients (tiled reduction).
// ---------------------------------------------------------------------------
void HCNNConv::backward(const float* grad_out, const float* in, const float* pre_act,
                    float* grad_in, float learning_rate, float momentum,
                    float weight_decay, const float* bn_save, int timestep) {
    const bool use_adam = (optimizer_type_ == OptimizerType::ADAM && timestep > 0);
    const bool use_threads = thread_pool && DIM >= THREAD_DIM_THRESHOLD;

    // Pre-activation gradients (gradient through activation function)
    std::vector<float> grad_pre(c_out * N);
    for (int i = 0; i < c_out * N; ++i)
        grad_pre[i] = grad_out[i] * activate_derivative(pre_act[i]);

    // BN backward: transform grad from "w.r.t. BN output" to "w.r.t. raw sum"
    if (use_batchnorm && bn_save) {
        for (int co = 0; co < c_out; ++co) {
            float* gp = grad_pre.data() + co * N;
            const float* pa = pre_act + co * N;
            float inv_std = bn_save[co];
            float gamma_co = bn_gamma[co];
            float inv_gamma = (gamma_co != 0.0f) ? (1.0f / gamma_co) : 0.0f;
            float inv_N = 1.0f / static_cast<float>(N);

            // Pass 1: compute dgamma, dbeta, and intermediate sums
            float dgamma = 0.0f, dbeta = 0.0f;
            float sum_dx_hat = 0.0f, sum_dx_hat_xhat = 0.0f;
            for (int v = 0; v < N; ++v) {
                float x_hat = (pa[v] - bn_beta[co]) * inv_gamma;
                float dx_hat = gp[v] * gamma_co;
                dgamma += gp[v] * x_hat;
                dbeta += gp[v];
                sum_dx_hat += dx_hat;
                sum_dx_hat_xhat += dx_hat * x_hat;
            }

            float mean_dx = sum_dx_hat * inv_N;
            float mean_dx_xhat = sum_dx_hat_xhat * inv_N;

            // Pass 2: compute gradient w.r.t. raw weighted sum (replaces grad_pre)
            for (int v = 0; v < N; ++v) {
                float x_hat = (pa[v] - bn_beta[co]) * inv_gamma;
                float dx_hat = gp[v] * gamma_co;
                gp[v] = inv_std * (dx_hat - mean_dx - x_hat * mean_dx_xhat);
            }

            // Update BN parameters (skip if frozen)
            if (!frozen_) {
                if (use_adam) {
                    bn_gamma_m[co] = adam_beta1_ * bn_gamma_m[co] + (1.0f - adam_beta1_) * dgamma;
                    bn_gamma_m2[co] = adam_beta2_ * bn_gamma_m2[co] + (1.0f - adam_beta2_) * dgamma * dgamma;
                    float mh = bn_gamma_m[co] / (1.0f - std::pow(adam_beta1_, timestep));
                    float vh = bn_gamma_m2[co] / (1.0f - std::pow(adam_beta2_, timestep));
                    bn_gamma[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
                    bn_beta_m[co] = adam_beta1_ * bn_beta_m[co] + (1.0f - adam_beta1_) * dbeta;
                    bn_beta_m2[co] = adam_beta2_ * bn_beta_m2[co] + (1.0f - adam_beta2_) * dbeta * dbeta;
                    mh = bn_beta_m[co] / (1.0f - std::pow(adam_beta1_, timestep));
                    vh = bn_beta_m2[co] / (1.0f - std::pow(adam_beta2_, timestep));
                    bn_beta[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
                } else {
                    bn_gamma_m[co] = momentum * bn_gamma_m[co] + dgamma;
                    bn_gamma[co] -= learning_rate * bn_gamma_m[co];
                    bn_beta_m[co] = momentum * bn_beta_m[co] + dbeta;
                    bn_beta[co] -= learning_rate * bn_beta_m[co];
                }
            }
        }
    }

    // Input gradient: vertex-level parallelism, tiled
    if (grad_in) {
        for (int ci = 0; ci < c_in; ++ci) {
            float* gi = grad_in + ci * N;

            auto do_vertices = [&](size_t v_begin, size_t v_end) {
                for (size_t t = v_begin; t < v_end; t += TILE) {
                    size_t t_end = std::min(t + TILE, v_end);
                    for (size_t v = t; v < t_end; ++v) gi[v] = 0.0f;
                    for (int co = 0; co < c_out; ++co) {
                        const float* gp = grad_pre.data() + co * N;
                        for (int k = 0; k < K; ++k) {
                            float w = kernel[kernel_idx(co, ci, k)];
                            uint32_t m = 1u << k;
                            for (size_t v = t; v < t_end; ++v)
                                gi[v] += w * gp[v ^ m];
                        }
                    }
                }
            };

            if (use_threads) {
                thread_pool->ForEach(static_cast<size_t>(N),
                    [&](size_t, size_t b, size_t e) { do_vertices(b, e); });
            } else {
                do_vertices(0, static_cast<size_t>(N));
            }
        }
    }

    // Weight update: channel-level parallelism, tiled reduction
    // Skip entirely if frozen (grad_in already computed above).
    if (frozen_) return;

    auto do_weight_update = [&](int co) {
        const float* gp = grad_pre.data() + co * N;
        for (int ci = 0; ci < c_in; ++ci) {
            const float* in_ci = in + ci * N;
            // Accumulate per-mask gradients across tiles
            float grad_k[32] = {};  // K = DIM <= 32
            for (int t = 0; t < N; t += static_cast<int>(TILE)) {
                int t_end = std::min(t + static_cast<int>(TILE), N);
                for (int k = 0; k < K; ++k) {
                    uint32_t m = 1u << k;
                    for (int v = t; v < t_end; ++v)
                        grad_k[k] += gp[v] * in_ci[v ^ m];
                }
            }
            for (int k = 0; k < K; ++k) {
                int ki = kernel_idx(co, ci, k);
                float g = grad_k[k];
                if (use_adam) {
                    kernel_m[ki] = adam_beta1_ * kernel_m[ki] + (1.0f - adam_beta1_) * g;
                    kernel_m2[ki] = adam_beta2_ * kernel_m2[ki] + (1.0f - adam_beta2_) * g * g;
                    float mh = kernel_m[ki] / (1.0f - std::pow(adam_beta1_, timestep));
                    float vh = kernel_m2[ki] / (1.0f - std::pow(adam_beta2_, timestep));
                    kernel[ki] -= learning_rate * (mh / (std::sqrt(vh) + adam_eps_) + weight_decay * kernel[ki]);
                } else {
                    g += weight_decay * kernel[ki];
                    kernel_m[ki] = momentum * kernel_m[ki] + g;
                    kernel[ki] -= learning_rate * kernel_m[ki];
                }
            }
        }
        if (use_bias) {
            float grad_b = 0.0f;
            for (int v = 0; v < N; ++v) grad_b += gp[v];
            if (use_adam) {
                bias_m[co] = adam_beta1_ * bias_m[co] + (1.0f - adam_beta1_) * grad_b;
                bias_m2[co] = adam_beta2_ * bias_m2[co] + (1.0f - adam_beta2_) * grad_b * grad_b;
                float mh = bias_m[co] / (1.0f - std::pow(adam_beta1_, timestep));
                float vh = bias_m2[co] / (1.0f - std::pow(adam_beta2_, timestep));
                bias[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
            } else {
                bias_m[co] = momentum * bias_m[co] + grad_b;
                bias[co] -= learning_rate * bias_m[co];
            }
        }
    };

    if (use_threads) {
        thread_pool->ForEach(static_cast<size_t>(c_out),
            [&](size_t, size_t b, size_t e) {
                for (size_t co = b; co < e; ++co) do_weight_update(static_cast<int>(co));
            });
    } else {
        for (int co = 0; co < c_out; ++co) do_weight_update(co);
    }
}

// ---------------------------------------------------------------------------
// compute_gradients: same tiling strategy as backward, but writes raw
// gradients to caller buffers instead of updating weights.
// ---------------------------------------------------------------------------
void HCNNConv::compute_gradients(const float* grad_out, const float* in, const float* pre_act,
                             float* grad_in, float* kernel_grad, float* bias_grad,
                             float* work_buf, const float* bn_save,
                             float* bn_gamma_grad, float* bn_beta_grad) const {
    const bool use_threads = thread_pool && DIM >= THREAD_DIM_THRESHOLD;

    // work_buf must be at least c_out * N floats if provided.
    // Falls back to heap allocation if nullptr (backward compat).
    std::vector<float> grad_pre_storage;
    float* grad_pre;
    if (work_buf) {
        grad_pre = work_buf;
    } else {
        grad_pre_storage.resize(c_out * N);
        grad_pre = grad_pre_storage.data();
    }
    for (int i = 0; i < c_out * N; ++i)
        grad_pre[i] = grad_out[i] * activate_derivative(pre_act[i]);

    // BN backward: transform grad from "w.r.t. BN output" to "w.r.t. raw sum"
    if (use_batchnorm && bn_save) {
        for (int co = 0; co < c_out; ++co) {
            float* gp = grad_pre + co * N;
            const float* pa = pre_act + co * N;
            float inv_std = bn_save[co];
            float gamma_co = bn_gamma[co];
            float inv_gamma = (gamma_co != 0.0f) ? (1.0f / gamma_co) : 0.0f;
            float inv_N = 1.0f / static_cast<float>(N);

            float dgamma = 0.0f, dbeta = 0.0f;
            float sum_dx_hat = 0.0f, sum_dx_hat_xhat = 0.0f;
            for (int v = 0; v < N; ++v) {
                float x_hat = (pa[v] - bn_beta[co]) * inv_gamma;
                float dx_hat = gp[v] * gamma_co;
                dgamma += gp[v] * x_hat;
                dbeta += gp[v];
                sum_dx_hat += dx_hat;
                sum_dx_hat_xhat += dx_hat * x_hat;
            }

            float mean_dx = sum_dx_hat * inv_N;
            float mean_dx_xhat = sum_dx_hat_xhat * inv_N;

            for (int v = 0; v < N; ++v) {
                float x_hat = (pa[v] - bn_beta[co]) * inv_gamma;
                float dx_hat = gp[v] * gamma_co;
                gp[v] = inv_std * (dx_hat - mean_dx - x_hat * mean_dx_xhat);
            }

            if (bn_gamma_grad) bn_gamma_grad[co] = dgamma;
            if (bn_beta_grad) bn_beta_grad[co] = dbeta;
        }
    }

    // Input gradient: vertex-level, tiled
    if (grad_in) {
        for (int ci = 0; ci < c_in; ++ci) {
            float* gi = grad_in + ci * N;

            auto do_vertices = [&](size_t v_begin, size_t v_end) {
                for (size_t t = v_begin; t < v_end; t += TILE) {
                    size_t t_end = std::min(t + TILE, v_end);
                    for (size_t v = t; v < t_end; ++v) gi[v] = 0.0f;
                    for (int co = 0; co < c_out; ++co) {
                        const float* gp = grad_pre + co * N;
                        for (int k = 0; k < K; ++k) {
                            float w = kernel[kernel_idx(co, ci, k)];
                            uint32_t m = 1u << k;
                            for (size_t v = t; v < t_end; ++v)
                                gi[v] += w * gp[v ^ m];
                        }
                    }
                }
            };

            if (use_threads) {
                thread_pool->ForEach(static_cast<size_t>(N),
                    [&](size_t, size_t b, size_t e) { do_vertices(b, e); });
            } else {
                do_vertices(0, static_cast<size_t>(N));
            }
        }
    }

    // Kernel + bias gradient: channel-level, tiled reduction
    auto do_kernel_grad = [&](int co) {
        const float* gp = grad_pre + co * N;
        for (int ci = 0; ci < c_in; ++ci) {
            const float* in_ci = in + ci * N;
            float grad_k[32] = {};  // K = DIM <= 32
            for (int t = 0; t < N; t += static_cast<int>(TILE)) {
                int t_end = std::min(t + static_cast<int>(TILE), N);
                for (int k = 0; k < K; ++k) {
                    uint32_t m = 1u << k;
                    for (int v = t; v < t_end; ++v)
                        grad_k[k] += gp[v] * in_ci[v ^ m];
                }
            }
            for (int k = 0; k < K; ++k) {
                kernel_grad[kernel_idx(co, ci, k)] = grad_k[k];
            }
        }
        if (bias_grad && use_bias) {
            float grad_b = 0.0f;
            for (int v = 0; v < N; ++v) grad_b += gp[v];
            bias_grad[co] = grad_b;
        }
    };

    if (use_threads) {
        thread_pool->ForEach(static_cast<size_t>(c_out),
            [&](size_t, size_t b, size_t e) {
                for (size_t co = b; co < e; ++co) do_kernel_grad(static_cast<int>(co));
            });
    } else {
        for (int co = 0; co < c_out; ++co) do_kernel_grad(co);
    }
}

// ---------------------------------------------------------------------------
// apply_gradients: apply pre-computed (averaged) gradients with momentum SGD.
// ---------------------------------------------------------------------------
void HCNNConv::apply_gradients(const float* kernel_grad, const float* bias_grad,
                           float learning_rate, float momentum, float weight_decay,
                           const float* bn_gamma_grad_in, const float* bn_beta_grad_in,
                           int timestep) {
    if (frozen_) return;
    const bool use_adam = (optimizer_type_ == OptimizerType::ADAM && timestep > 0);
    int total_k = c_out * c_in * K;

    if (use_adam) {
        for (int i = 0; i < total_k; ++i) {
            float g = kernel_grad[i];
            kernel_m[i] = adam_beta1_ * kernel_m[i] + (1.0f - adam_beta1_) * g;
            kernel_m2[i] = adam_beta2_ * kernel_m2[i] + (1.0f - adam_beta2_) * g * g;
            float mh = kernel_m[i] / (1.0f - std::pow(adam_beta1_, timestep));
            float vh = kernel_m2[i] / (1.0f - std::pow(adam_beta2_, timestep));
            kernel[i] -= learning_rate * (mh / (std::sqrt(vh) + adam_eps_) + weight_decay * kernel[i]);
        }
    } else {
        for (int i = 0; i < total_k; ++i) {
            float g = kernel_grad[i] + weight_decay * kernel[i];
            kernel_m[i] = momentum * kernel_m[i] + g;
            kernel[i] -= learning_rate * kernel_m[i];
        }
    }

    if (use_bias && bias_grad) {
        for (int co = 0; co < c_out; ++co) {
            if (use_adam) {
                float g = bias_grad[co];
                bias_m[co] = adam_beta1_ * bias_m[co] + (1.0f - adam_beta1_) * g;
                bias_m2[co] = adam_beta2_ * bias_m2[co] + (1.0f - adam_beta2_) * g * g;
                float mh = bias_m[co] / (1.0f - std::pow(adam_beta1_, timestep));
                float vh = bias_m2[co] / (1.0f - std::pow(adam_beta2_, timestep));
                bias[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
            } else {
                bias_m[co] = momentum * bias_m[co] + bias_grad[co];
                bias[co] -= learning_rate * bias_m[co];
            }
        }
    }

    if (use_batchnorm && bn_gamma_grad_in && bn_beta_grad_in) {
        for (int co = 0; co < c_out; ++co) {
            if (use_adam) {
                float gg = bn_gamma_grad_in[co], bg = bn_beta_grad_in[co];
                bn_gamma_m[co] = adam_beta1_ * bn_gamma_m[co] + (1.0f - adam_beta1_) * gg;
                bn_gamma_m2[co] = adam_beta2_ * bn_gamma_m2[co] + (1.0f - adam_beta2_) * gg * gg;
                float mh = bn_gamma_m[co] / (1.0f - std::pow(adam_beta1_, timestep));
                float vh = bn_gamma_m2[co] / (1.0f - std::pow(adam_beta2_, timestep));
                bn_gamma[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
                bn_beta_m[co] = adam_beta1_ * bn_beta_m[co] + (1.0f - adam_beta1_) * bg;
                bn_beta_m2[co] = adam_beta2_ * bn_beta_m2[co] + (1.0f - adam_beta2_) * bg * bg;
                mh = bn_beta_m[co] / (1.0f - std::pow(adam_beta1_, timestep));
                vh = bn_beta_m2[co] / (1.0f - std::pow(adam_beta2_, timestep));
                bn_beta[co] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
            } else {
                bn_gamma_m[co] = momentum * bn_gamma_m[co] + bn_gamma_grad_in[co];
                bn_gamma[co] -= learning_rate * bn_gamma_m[co];
                bn_beta_m[co] = momentum * bn_beta_m[co] + bn_beta_grad_in[co];
                bn_beta[co] -= learning_rate * bn_beta_m[co];
            }
        }
    }
}

void HCNNConv::update_running_stats(const float* mean, const float* var) {
    for (int co = 0; co < c_out; ++co) {
        float unbiased_var = var[co] * static_cast<float>(N)
                           / static_cast<float>(N - 1);
        bn_running_mean[co] = (1.0f - bn_momentum_) * bn_running_mean[co]
                            + bn_momentum_ * mean[co];
        bn_running_var[co] = (1.0f - bn_momentum_) * bn_running_var[co]
                           + bn_momentum_ * unbiased_var;
    }
}

float HCNNConv::activate(float x) const {
    switch (activation) {
        case Activation::RELU:       return (x > 0.0f) ? x : 0.0f;
        case Activation::LEAKY_RELU: return (x > 0.0f) ? x : leaky_alpha_ * x;
        default:                     return x;
    }
}

float HCNNConv::activate_derivative(float x) const {
    switch (activation) {
        case Activation::RELU:       return (x > 0.0f) ? 1.0f : 0.0f;
        case Activation::LEAKY_RELU: return (x > 0.0f) ? 1.0f : leaky_alpha_;
        default:                     return 1.0f;
    }
}
