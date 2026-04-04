#pragma once

#include "HCNNNetwork.h"
#include <vector>

struct HCNNMNISTDataset {
    struct Sample {
        std::vector<float> input;   // 16 scalars in [-1.0, 1.0]
        int target_class;           // class index [0, num_classes)
    };

    std::vector<Sample> samples;

    size_t size() const { return samples.size(); }
    const Sample& get(size_t i) const { return samples[i]; }

    void train_epoch(HCNNNetwork& net, float learning_rate);
};

// Factory function — declared here so main.cpp can see it
HCNNMNISTDataset create_toy_mnist_like_dataset();