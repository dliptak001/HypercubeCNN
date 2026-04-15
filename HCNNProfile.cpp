// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNNProfile.h"

#if HCNN_PROFILE

#include <iomanip>
#include <ostream>

namespace hcnn::profile {

void TrainProfile::Report(std::ostream& out) const
{
    const auto to_ms = [](int64_t ns) {
        return static_cast<double>(ns) / 1.0e6;
    };

    const int64_t embed_ns       = ns_embed.load();
    const int64_t fwd_conv_ns    = ns_fwd_conv.load();
    const int64_t fwd_pool_ns    = ns_fwd_pool.load();
    const int64_t fwd_readout_ns = ns_fwd_readout.load();
    const int64_t loss_grad_ns   = ns_loss_grad.load();
    const int64_t bwd_readout_ns = ns_bwd_readout.load();
    const int64_t bwd_conv_ns    = ns_bwd_conv.load();
    const int64_t bwd_pool_ns    = ns_bwd_pool.load();
    const int64_t grad_accum_ns  = ns_grad_accum.load();
    const int64_t reduce_ns      = ns_reduce.load();
    const int64_t apply_ns       = ns_apply_update.load();
    const int64_t wall_ns        = ns_wall_train_batch.load();

    const int64_t parallel_cum_ns =
        embed_ns + fwd_conv_ns + fwd_pool_ns + fwd_readout_ns + loss_grad_ns +
        bwd_readout_ns + bwd_conv_ns + bwd_pool_ns + grad_accum_ns;
    const int64_t sequential_ns  = reduce_ns + apply_ns;

    const auto row = [&](const char* name, int64_t ns, int64_t denom) {
        const double ms  = to_ms(ns);
        const double pct = denom > 0 ? 100.0 * static_cast<double>(ns) /
                                          static_cast<double>(denom)
                                     : 0.0;
        out << "  " << std::left << std::setw(22) << name << std::right
            << std::fixed << std::setprecision(2) << std::setw(12) << ms << " ms  "
            << std::setw(6) << std::setprecision(2) << pct << "%\n";
    };

    out << "\n=== HCNN Train Profile ===\n";
    out << "  batches: " << batches.load()
        << "   samples: " << samples.load()
        << "   wall (train_batch_impl): "
        << std::fixed << std::setprecision(2) << to_ms(wall_ns) / 1000.0 << " s\n\n";

    out << "  Per-phase (cumulative across all threads):\n";
    out << "  phase                   cumulative      share\n";
    out << "  ----------------------+-------------+--------\n";
    row("embed",            embed_ns,       parallel_cum_ns);
    row("fwd conv",         fwd_conv_ns,    parallel_cum_ns);
    row("fwd pool",         fwd_pool_ns,    parallel_cum_ns);
    row("fwd readout",      fwd_readout_ns, parallel_cum_ns);
    row("loss grad",        loss_grad_ns,   parallel_cum_ns);
    row("bwd readout",      bwd_readout_ns, parallel_cum_ns);
    row("bwd conv",         bwd_conv_ns,    parallel_cum_ns);
    row("bwd pool",         bwd_pool_ns,    parallel_cum_ns);
    row("grad accum",       grad_accum_ns,  parallel_cum_ns);
    out << "  ----------------------+-------------+--------\n";
    row("  parallel CUM",   parallel_cum_ns, parallel_cum_ns);
    out << "\n";

    out << "  Sequential post-parallel (wall-time, single-threaded):\n";
    out << "  phase                   cumulative      share\n";
    out << "  ----------------------+-------------+--------\n";
    row("reduce",           reduce_ns, sequential_ns);
    row("apply update",     apply_ns,  sequential_ns);
    out << "  ----------------------+-------------+--------\n";
    row("  sequential SUM", sequential_ns, sequential_ns);
    out << "\n";

    // Parallel efficiency: if everything in the parallel region were
    // perfectly parallelized across T threads, parallel_cum_ns would
    // equal wall_ns * T.  The ratio tells us how close we are.
    const int64_t parallel_wall_ns = wall_ns - sequential_ns;
    if (parallel_wall_ns > 0 && parallel_cum_ns > 0) {
        const double implied_threads =
            static_cast<double>(parallel_cum_ns) /
            static_cast<double>(parallel_wall_ns);
        out << "  Parallel efficiency estimate:\n"
            << "    parallel wall time  : "
            << std::fixed << std::setprecision(2)
            << to_ms(parallel_wall_ns) << " ms\n"
            << "    parallel cumulative : "
            << to_ms(parallel_cum_ns) << " ms\n"
            << "    implied threads busy: "
            << std::setprecision(2) << implied_threads << "\n";
    }

    out << std::flush;
}

}  // namespace hcnn::profile

#endif  // HCNN_PROFILE
