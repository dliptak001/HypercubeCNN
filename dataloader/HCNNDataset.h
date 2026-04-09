#pragma once

#include <string>
#include <vector>

/// In-tree example dataset container.  Pure data: holds samples and offers
/// accessors.  No coupling to HCNN/HCNNNetwork -- consumers gather raw
/// pointer arrays from `samples` and feed them to HCNN::TrainEpoch /
/// HCNN::ForwardBatch directly.
struct HCNNDataset {
    struct Sample {
        std::vector<float> input;   // scalars in [-1.0, 1.0]
        int target_class;           // class index [0, num_classes)
    };

    std::vector<Sample> samples;

    size_t size() const { return samples.size(); }
    const Sample& get(size_t i) const { return samples[i]; }
};

// Factory: load real MNIST from IDX files.
// images_path: e.g. "data/train-images-idx3-ubyte"
// labels_path: e.g. "data/train-labels-idx1-ubyte"
// max_samples: 0 = load all, otherwise limit to first N samples
HCNNDataset load_mnist(const std::string& images_path,
                       const std::string& labels_path,
                       size_t max_samples = 0);
