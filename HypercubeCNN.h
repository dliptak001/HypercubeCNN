// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

/**
 * @file HypercubeCNN.h
 * @brief Umbrella include for the full teaching-oriented SDK surface.
 *
 * Pulls in:
 *   - `HCNN.h` / `HCNNInput.h` — core front door + full-capacity input types
 *   - `HCNNArch.h`          — LayerSpec, apply_arch, HCNNConfig::Build
 *   - `HCNNTrainHelpers.h`  — metrics, cosine LR, checkpoints, flat dataset
 *   - `HCNNSpatialAug.h`    — optional 2D augmentation
 *   - `HCNNSpatialEmbed.h`  — optional 2D → length-N packing
 *
 * Prefer this single include for demos and coursework.  Minimal apps that
 * only need the graph can `#include "HCNN.h"` alone.
 *
 * Private implementation headers (`HCNNNetwork`, layers, `ThreadPool`) are
 * **not** included and **not** installed — apps never need them.
 */

#include "HCNN.h"
#include "HCNNArch.h"
#include "HCNNTrainHelpers.h"
#include "HCNNSpatialAug.h"
#include "HCNNSpatialEmbed.h"
