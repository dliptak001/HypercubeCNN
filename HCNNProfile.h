// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

/// @file HCNNProfile.h
/// @brief Opt-in per-phase timing instrumentation for HCNN training.
///
/// The core hot path (`HCNNNetwork::train_batch_impl`) is decorated with
/// `HCNN_PROFILE_SCOPE(...)` macros at each distinct phase (forward conv,
/// forward pool, forward readout, loss gradient, backward readout,
/// backward conv, backward pool, post-parallel gradient reduction, and
/// weight update).  When `HCNN_PROFILE` is defined to 0 (the default),
/// the macros expand to `((void)0)` and the ScopedTimer class is not
/// compiled — zero runtime cost in normal release builds.
///
/// To enable profiling, flip `HCNN_PROFILE` to 1 in this header and
/// rebuild `HypercubeCNNCore`.  After a training run, call
/// `hcnn::profile::Report(std::cout)` to print a cumulative per-phase
/// breakdown (wall-time and share of total instrumented time).
///
/// Reporting semantics: the counters accumulate **cumulative thread time**
/// across all threads that participated in training.  If training used
/// 4 threads and forward-conv took 10s wall-clock, the counter will read
/// ~40s because each thread contributed its own 10s.  The separate
/// `ns_wall_train_batch` counter records wall-clock time of the whole
/// `train_batch_impl` call for parallel-efficiency analysis.

#ifndef HCNN_PROFILE
#define HCNN_PROFILE 0
#endif

#if HCNN_PROFILE

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <ostream>

namespace hcnn::profile {

/// Atomic cumulative counters for each instrumented training phase.
/// Units are nanoseconds.  Incremented via ScopedTimer or direct fetch_add.
struct TrainProfile
{
    // --- Parallel (per-sample) phases inside train_batch_impl::process_sample ---
    std::atomic<int64_t> ns_embed{0};
    std::atomic<int64_t> ns_fwd_conv{0};
    std::atomic<int64_t> ns_fwd_pool{0};
    std::atomic<int64_t> ns_fwd_readout{0};
    std::atomic<int64_t> ns_loss_grad{0};
    std::atomic<int64_t> ns_bwd_readout{0};
    std::atomic<int64_t> ns_bwd_conv{0};
    std::atomic<int64_t> ns_bwd_pool{0};
    std::atomic<int64_t> ns_grad_accum{0};  // per-sample gradient accumulate into per-thread slot

    // --- Sequential post-parallel phases ---
    std::atomic<int64_t> ns_reduce{0};        // reduce per-thread accumulators
    std::atomic<int64_t> ns_apply_update{0};  // weight update (conv + readout)

    // --- Wall-time and bookkeeping ---
    std::atomic<int64_t> ns_wall_train_batch{0};  // total wall time of train_batch_impl
    std::atomic<int64_t> samples{0};
    std::atomic<int64_t> batches{0};

    void Reset()
    {
        ns_embed.store(0, std::memory_order_relaxed);
        ns_fwd_conv.store(0, std::memory_order_relaxed);
        ns_fwd_pool.store(0, std::memory_order_relaxed);
        ns_fwd_readout.store(0, std::memory_order_relaxed);
        ns_loss_grad.store(0, std::memory_order_relaxed);
        ns_bwd_readout.store(0, std::memory_order_relaxed);
        ns_bwd_conv.store(0, std::memory_order_relaxed);
        ns_bwd_pool.store(0, std::memory_order_relaxed);
        ns_grad_accum.store(0, std::memory_order_relaxed);
        ns_reduce.store(0, std::memory_order_relaxed);
        ns_apply_update.store(0, std::memory_order_relaxed);
        ns_wall_train_batch.store(0, std::memory_order_relaxed);
        samples.store(0, std::memory_order_relaxed);
        batches.store(0, std::memory_order_relaxed);
    }

    void Report(std::ostream& out) const;
};

/// Process-wide singleton.  All HCNNNetwork instances share this profile —
/// intentional for the typical "train one network, profile, report" flow.
inline TrainProfile& instance()
{
    static TrainProfile p;
    return p;
}

inline void Reset()                    { instance().Reset(); }
inline void Report(std::ostream& out)  { instance().Report(out); }

/// RAII helper: starts a steady_clock timer at construction, adds the
/// elapsed nanoseconds to `counter` on destruction via relaxed fetch_add.
/// Zero synchronization overhead in the happy path — relaxed atomics on
/// x86_64 compile down to plain LOCK XADD which is ~5-10 ns.
class ScopedTimer
{
public:
    explicit ScopedTimer(std::atomic<int64_t>& counter)
        : counter_(counter), t0_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer()
    {
        const auto t1 = std::chrono::steady_clock::now();
        const int64_t ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_).count();
        counter_.fetch_add(ns, std::memory_order_relaxed);
    }

    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::atomic<int64_t>&                              counter_;
    std::chrono::steady_clock::time_point              t0_;
};

}  // namespace hcnn::profile

// ---- Call-site macros ----------------------------------------------------
//
// HCNN_PROFILE_SCOPE(field) — stands up a ScopedTimer bound to the named
//                             TrainProfile counter for the rest of the
//                             enclosing scope.  Use like:
//                                 HCNN_PROFILE_SCOPE(ns_fwd_conv);
//
// HCNN_PROFILE_INCREMENT(field, value) — direct fetch_add without a timer,
//                                         for counters like `samples`.

#define HCNN_PROFILE_CONCAT_INNER(a, b) a ## b
#define HCNN_PROFILE_CONCAT(a, b)       HCNN_PROFILE_CONCAT_INNER(a, b)

#define HCNN_PROFILE_SCOPE(field)                                          \
    hcnn::profile::ScopedTimer HCNN_PROFILE_CONCAT(_hcnn_profile_timer_, __LINE__)( \
        hcnn::profile::instance().field)

#define HCNN_PROFILE_INCREMENT(field, value)                               \
    hcnn::profile::instance().field.fetch_add((value),                     \
                                              std::memory_order_relaxed)

#else  // HCNN_PROFILE == 0

#include <ostream>

namespace hcnn::profile {
inline void Reset()                  {}
inline void Report(std::ostream&)    {}
}  // namespace hcnn::profile

#define HCNN_PROFILE_SCOPE(field)              ((void)0)
#define HCNN_PROFILE_INCREMENT(field, value)   ((void)0)

#endif  // HCNN_PROFILE
