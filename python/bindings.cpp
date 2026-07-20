// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// Phase 0: minimal extension that compiles the C++ core into the module and
// proves the wheel/install path. Public HCNN API lands in Phase 1.

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, m)
{
    m.doc() = "HypercubeCNN: dependency-free hypercube CNN core (Python bindings)";
    m.attr("__version__") = "1.0.0";
}
