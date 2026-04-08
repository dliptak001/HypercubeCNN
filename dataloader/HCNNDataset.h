#pragma once

#include "HCNNNetwork.h"
#include <random>
#include <vector>
#include <string>

struct HCNNDataset {
    struct Sample {
        std::vector<float> input;   // scalars in [-1.0, 1.0]
        int target_class;           // class index [0, num_classes)
    };

    std::vector<Sample> samples;
    std::mt19937 rng{42};           // shuffle RNG, per-dataset for thread safety

    size_t size() const { return samples.size(); }
    const Sample& get(size_t i) const { return samples[i]; }

    /// Train one epoch. batch_size=1 is pure SGD, batch_size>1 uses mini-batch
    /// parallelism via the network's ThreadPool.
    /// class_weights: optional per-class loss scaling (length num_classes), nullptr for uniform.
    void train_epoch(HCNNNetwork& net, float learning_rate, float momentum = 0.0f,
                     int batch_size = 1, float weight_decay = 0.0f,
                     const float* class_weights = nullptr);
};

// Factory: load real MNIST from IDX files.
// images_path: e.g. "data/train-images-idx3-ubyte"
// labels_path: e.g. "data/train-labels-idx1-ubyte"
// max_samples: 0 = load all, otherwise limit to first N samples
HCNNDataset load_mnist(const std::string& images_path,
                       const std::string& labels_path,
                       size_t max_samples = 0);
