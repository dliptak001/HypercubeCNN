#pragma once

#include <cstdint>
#include <string>
#include <vector>

/// A single molecule: 1024-bit fingerprint + binary label.
struct MolSample {
    std::vector<float> fingerprint;  // NUM_BITS floats, each 0.0 or 1.0
    int label;                       // 0 or 1 (-1 = missing)
};

/// Dataset split loaded from an .hcfp file (produced by featurize_bbbp.py).
struct MoleculeNetDataset {
    std::vector<MolSample> train;
    std::vector<MolSample> val;
    std::vector<MolSample> test;
    int num_bits{0};
    int num_tasks{0};
    std::string name;
};

/// Load an .hcfp binary file and split into train/val/test.
/// File format: 16-byte header + (num_bits + num_tasks + 1) bytes per sample.
MoleculeNetDataset load_hcfp(const std::string& path,
                              const std::string& name = "BBBP");
