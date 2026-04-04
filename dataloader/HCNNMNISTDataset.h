#pragma once

#include "HCNNNetwork.h"
#include <vector>
#include <string>

struct HCNNMNISTDataset {
    struct Sample {
        std::vector<float> input;   // scalars in [-1.0, 1.0]
        int target_class;           // class index [0, num_classes)
    };

    std::vector<Sample> samples;

    size_t size() const { return samples.size(); }
    const Sample& get(size_t i) const { return samples[i]; }

    void train_epoch(HCNNNetwork& net, float learning_rate);
};

// Factory: toy dataset (10 random samples, 16 features each)
HCNNMNISTDataset create_toy_mnist_like_dataset();

// Factory: load real MNIST from IDX files.
// images_path: e.g. "data/train-images-idx3-ubyte"
// labels_path: e.g. "data/train-labels-idx1-ubyte"
// max_samples: 0 = load all, otherwise limit to first N samples
HCNNMNISTDataset load_mnist(const std::string& images_path,
                            const std::string& labels_path,
                            size_t max_samples = 0);
