// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

/**
 * @file HypercubeCNN.h
 * @brief Umbrella include for the full teaching-oriented SDK surface.
 *
 * Pulls in:
 *   - `HCNN.h`              — core train / infer front door
 *   - `HCNNArch.h`          — LayerSpec, apply_arch, HCNNConfig::Build
 *   - `HCNNTrainHelpers.h`  — metrics, cosine LR, checkpoints, flat dataset
 *   - `HCNNSpatialAug.h`    — optional 2D augmentation
 *   - `HCNNSpatialEmbed.h`  — optional 2D → length-N packing
 *
 * Prefer this single include for demos and coursework.  Minimal apps that
 * only need the graph can `#include "HCNN.h"` alone.
 *
 * Layer headers (`HCNNNetwork`, `HCNNConv`, `HCNNPool`, `HCNNReadout`,
 * `ThreadPool`) are **not** included here — they are advanced / internal.
 */

#include "HCNN.h"
#include "HCNNArch.h"
#include "HCNNTrainHelpers.h"
#include "HCNNSpatialAug.h"
#include "HCNNSpatialEmbed.h"
