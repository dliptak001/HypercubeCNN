#include "HCNNDataset.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

// Read a 32-bit big-endian integer from a stream.
static uint32_t read_be32(std::ifstream& f) {
    uint8_t buf[4];
    f.read(reinterpret_cast<char*>(buf), 4);
    if (!f) throw std::runtime_error("Unexpected end of file reading IDX header");
    return (static_cast<uint32_t>(buf[0]) << 24) |
           (static_cast<uint32_t>(buf[1]) << 16) |
           (static_cast<uint32_t>(buf[2]) << 8)  |
           static_cast<uint32_t>(buf[3]);
}

HCNNDataset load_mnist(const std::string& images_path,
                       const std::string& labels_path,
                       size_t max_samples) {
    // --- Read images ---
    std::ifstream img_file(images_path, std::ios::binary);
    if (!img_file.is_open()) {
        throw std::runtime_error("Cannot open MNIST images: " + images_path);
    }

    uint32_t img_magic = read_be32(img_file);
    if (img_magic != 0x00000803) {
        throw std::runtime_error("Invalid MNIST image magic: " + std::to_string(img_magic));
    }

    uint32_t num_images = read_be32(img_file);
    uint32_t rows = read_be32(img_file);
    uint32_t cols = read_be32(img_file);
    int pixels = static_cast<int>(rows * cols); // 784 for MNIST

    // --- Read labels ---
    std::ifstream lbl_file(labels_path, std::ios::binary);
    if (!lbl_file.is_open()) {
        throw std::runtime_error("Cannot open MNIST labels: " + labels_path);
    }

    uint32_t lbl_magic = read_be32(lbl_file);
    if (lbl_magic != 0x00000801) {
        throw std::runtime_error("Invalid MNIST label magic: " + std::to_string(lbl_magic));
    }

    uint32_t num_labels = read_be32(lbl_file);
    if (num_labels != num_images) {
        throw std::runtime_error("MNIST image/label count mismatch");
    }

    size_t count = num_images;
    if (max_samples > 0 && max_samples < count) count = max_samples;

    HCNNDataset ds;
    ds.samples.resize(count);

    std::vector<uint8_t> pixel_buf(pixels);
    for (size_t i = 0; i < count; ++i) {
        auto& s = ds.samples[i];

        // Read pixels and normalize to [-1.0, 1.0]
        img_file.read(reinterpret_cast<char*>(pixel_buf.data()), pixels);
        if (!img_file) {
            throw std::runtime_error("Truncated MNIST image file at sample "
                                     + std::to_string(i));
        }
        s.input.resize(pixels);
        for (int p = 0; p < pixels; ++p) {
            float v = static_cast<float>(pixel_buf[p]) / 127.5f - 1.0f;
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            s.input[p] = v;
        }

        // Read label
        uint8_t label;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        if (!lbl_file) {
            throw std::runtime_error("Truncated MNIST label file at sample "
                                     + std::to_string(i));
        }
        s.target_class = static_cast<int>(label);
    }

    return ds;
}

void HCNNDataset::train_epoch(HCNNNetwork& net, float learning_rate,
                              float momentum, int batch_size,
                              float weight_decay, const float* class_weights) {
    std::vector<size_t> order(samples.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    if (batch_size <= 1) {
        // Pure SGD — one sample at a time
        for (size_t i : order) {
            const auto& s = samples[i];
            net.train_step(s.input.data(), static_cast<int>(s.input.size()),
                           s.target_class, learning_rate, momentum, weight_decay,
                           class_weights);
        }
    } else {
        // Mini-batch SGD — process batch_size samples in parallel
        int n = static_cast<int>(order.size());
        std::vector<const float*> batch_inputs(batch_size);
        std::vector<int> batch_lengths(batch_size);
        std::vector<int> batch_targets(batch_size);

        for (int start = 0; start < n; start += batch_size) {
            int actual = std::min(batch_size, n - start);
            for (int j = 0; j < actual; ++j) {
                const auto& s = samples[order[start + j]];
                batch_inputs[j] = s.input.data();
                batch_lengths[j] = static_cast<int>(s.input.size());
                batch_targets[j] = s.target_class;
            }
            net.train_batch(batch_inputs.data(), batch_lengths.data(),
                            batch_targets.data(), actual,
                            learning_rate, momentum, weight_decay, class_weights);
        }
    }
}
