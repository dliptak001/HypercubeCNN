// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// Shared architecture scaffolding for in-tree teaching demos (MNIST,
// regression_timeseries, ...). Not part of the public SDK install.
//
// Usage: #include "demo_arch.h" with examples/ on the include path.

#pragma once

#include "HCNN.h"

#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hcnn_demo {

/// One stack step: Hamming conv or antipodal pool.
struct ArchLayer {
    enum class Kind { Conv, Pool };

    Kind kind = Kind::Conv;

    int c_out = 16;
    hcnn::Activation activation = hcnn::Activation::RELU;
    bool use_bias = true;
    bool use_bn = false;

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

struct ArchParamSummary {
    long long total = 0;
    long long readout = 0;
    long long flatten_features = 0;
    int final_dim = 0;
    int final_N = 0;
    int last_channels = 0;
    int num_conv = 0;
    std::vector<long long> conv_params;
};

inline const char* activation_name(hcnn::Activation a) {
    switch (a) {
        case hcnn::Activation::NONE:       return "NONE";
        case hcnn::Activation::RELU:       return "RELU";
        case hcnn::Activation::LEAKY_RELU: return "LEAKY_RELU";
        case hcnn::Activation::TANH:       return "TANH";
    }
    return "?";
}

inline const char* pool_name(hcnn::PoolType t) {
    switch (t) {
        case hcnn::PoolType::MAX: return "MAX";
        case hcnn::PoolType::AVG: return "AVG";
    }
    return "?";
}

/**
 * Walk a layer list: track DIM/N/channels, per-conv params, FLATTEN readout.
 * Matches HCNN::GetWeightCount (kernel + bias; BN gamma/beta omitted).
 */
inline ArchParamSummary summarize_arch(int dim,
                                       int num_outputs,
                                       int input_channels,
                                       const std::vector<ArchLayer>& layers) {
    if (dim < 1 || dim > 30)
        throw std::runtime_error("demo_arch: dim must be in [1, 30]");
    if (num_outputs < 1)
        throw std::runtime_error("demo_arch: num_outputs must be >= 1");
    if (input_channels < 1)
        throw std::runtime_error("demo_arch: input_channels must be >= 1");
    if (layers.empty())
        throw std::runtime_error("demo_arch: need at least one layer");

    ArchParamSummary s;
    int d = dim;
    int N = 1 << d;
    int c_in = input_channels;
    s.last_channels = c_in;

    for (const auto& L : layers) {
        if (L.kind == ArchLayer::Kind::Conv) {
            if (L.c_out < 1)
                throw std::runtime_error("demo_arch: conv c_out must be >= 1");
            // K = DIM + 1 (DIM Hamming-1 neighbors + 1 self/center tap)
            const long long k_params =
                static_cast<long long>(c_in) * L.c_out * (d + 1)
                + (L.use_bias ? L.c_out : 0);
            s.conv_params.push_back(k_params);
            s.total += k_params;
            c_in = L.c_out;
            s.last_channels = L.c_out;
            ++s.num_conv;
        } else {
            d -= 1;
            if (d < 0)
                throw std::runtime_error("demo_arch: too many pools (DIM < 0)");
            N = 1 << d;
        }
    }

    if (s.num_conv < 1)
        throw std::runtime_error("demo_arch: need at least one Conv layer");

    s.final_dim = d;
    s.final_N = N;
    s.flatten_features = static_cast<long long>(s.last_channels) * N;
    s.readout = s.flatten_features * num_outputs + num_outputs;
    s.total += s.readout;
    return s;
}

inline void apply_arch(hcnn::HCNN& net,
                       int dim,
                       int num_outputs,
                       int input_channels,
                       const std::vector<ArchLayer>& layers) {
    (void)summarize_arch(dim, num_outputs, input_channels, layers);
    for (const auto& L : layers) {
        if (L.kind == ArchLayer::Kind::Conv)
            net.AddConv(L.c_out, L.activation, L.use_bias, L.use_bn);
        else
            net.AddPool(L.pool_type);
    }
}

inline void print_arch(std::ostream& os,
                       int dim,
                       int num_outputs,
                       int input_channels,
                       const std::vector<ArchLayer>& layers,
                       const ArchParamSummary& sum) {
    int d = dim;
    int N = 1 << d;
    int c_in = input_channels;

    os << "\nArchitecture: ";
    bool first_line = true;
    for (const auto& L : layers) {
        if (!first_line)
            os << "              -> ";
        first_line = false;

        if (L.kind == ArchLayer::Kind::Conv) {
            os << "Conv(" << c_in << "->" << L.c_out
               << ", " << activation_name(L.activation);
            if (L.use_bias) os << ", bias";
            if (L.use_bn)   os << ", BN";
            os << ")  DIM=" << d << "  N=" << N << "\n";
            c_in = L.c_out;
        } else {
            os << "Pool(" << pool_name(L.pool_type) << ")  DIM "
               << d << "->" << (d - 1)
               << "  N " << N << "->" << (N / 2) << "\n";
            d -= 1;
            N = 1 << d;
        }
    }

    os << "              -> FLATTEN\n"
       << "              -> Linear(" << sum.flatten_features
       << " -> " << num_outputs << ")\n"
       << "Parameters:   " << sum.total << " (";
    for (size_t i = 0; i < sum.conv_params.size(); ++i) {
        if (i) os << " + ";
        os << sum.conv_params[i] << " conv" << (i + 1);
    }
    os << " + " << sum.readout << " readout)\n\n";
}

} // namespace hcnn_demo
