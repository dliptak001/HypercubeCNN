// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// Compatibility shim for in-tree demos that historically used hcnn_demo::*.
// The real API is public: #include "HCNNArch.h" (or HypercubeCNN.h).
//
// New code should use hcnn::LayerSpec / hcnn::apply_arch / hcnn::HCNNConfig.

#pragma once

#include "HCNNArch.h"

#include <ostream>
#include <vector>

namespace hcnn_demo {

using ArchLayer = hcnn::LayerSpec;
using ArchParamSummary = hcnn::ArchParamSummary;

inline const char* activation_name(hcnn::Activation a) {
    return hcnn::activation_name(a);
}

inline const char* pool_name(hcnn::PoolType t) {
    return hcnn::pool_name(t);
}

inline ArchParamSummary summarize_arch(int dim,
                                       int num_outputs,
                                       int input_channels,
                                       const std::vector<ArchLayer>& layers) {
    return hcnn::summarize_arch(dim, num_outputs, input_channels, layers);
}

inline void apply_arch(hcnn::HCNN& net,
                       int dim,
                       int num_outputs,
                       int input_channels,
                       const std::vector<ArchLayer>& layers) {
    hcnn::apply_arch(net, dim, num_outputs, input_channels, layers);
}

inline void apply_arch(hcnn::HCNN& net, const std::vector<ArchLayer>& layers) {
    hcnn::apply_arch(net, layers);
}

inline void print_arch(std::ostream& os,
                       int dim,
                       int num_outputs,
                       int input_channels,
                       const std::vector<ArchLayer>& layers,
                       const ArchParamSummary& sum) {
    hcnn::print_arch(os, dim, num_outputs, input_channels, layers, sum);
}

} // namespace hcnn_demo
