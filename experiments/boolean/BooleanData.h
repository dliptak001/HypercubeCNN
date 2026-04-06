#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

/// A Boolean function dataset: each sample is a hypercube vertex (DIM bits)
/// with a binary label. Input to HCNN is one-hot: vertex v → N-float array
/// with 1.0 at position v, 0.0 elsewhere.
struct BooleanDataset {
    struct Sample {
        uint32_t vertex;    // vertex index (the DIM-bit input)
        int label;          // 0 or 1
    };

    std::string name;
    int dim;
    std::vector<Sample> train;
    std::vector<Sample> test;
};

/// Hamming weight (popcount) of a 32-bit integer.
inline int hamming_weight(uint32_t x) {
    int count = 0;
    while (x) { count += x & 1; x >>= 1; }
    return count;
}

// ---------------------------------------------------------------------------
// Boolean function generators — produce the full truth table, then split.
// ---------------------------------------------------------------------------

/// Full parity: XOR of all DIM bits. Output = 1 if popcount is odd.
inline std::vector<int> generate_parity(int dim) {
    int N = 1 << dim;
    std::vector<int> table(N);
    for (int v = 0; v < N; ++v)
        table[v] = hamming_weight(static_cast<uint32_t>(v)) % 2;
    return table;
}

/// k-bit parity: XOR of the first k bits only.
inline std::vector<int> generate_k_parity(int dim, int k) {
    int N = 1 << dim;
    uint32_t mask = (1u << k) - 1;  // lowest k bits
    std::vector<int> table(N);
    for (int v = 0; v < N; ++v)
        table[v] = hamming_weight(static_cast<uint32_t>(v) & mask) % 2;
    return table;
}

/// Majority: output 1 if more than DIM/2 bits are set.
inline std::vector<int> generate_majority(int dim) {
    int N = 1 << dim;
    int threshold = dim / 2 + 1;  // strict majority
    std::vector<int> table(N);
    for (int v = 0; v < N; ++v)
        table[v] = (hamming_weight(static_cast<uint32_t>(v)) >= threshold) ? 1 : 0;
    return table;
}

/// Threshold-k: output 1 if Hamming weight >= k.
inline std::vector<int> generate_threshold(int dim, int k) {
    int N = 1 << dim;
    std::vector<int> table(N);
    for (int v = 0; v < N; ++v)
        table[v] = (hamming_weight(static_cast<uint32_t>(v)) >= k) ? 1 : 0;
    return table;
}

/// Random monotone Boolean function: for each vertex, output 1 if any
/// "seed" vertex is a subset of its bits (bitwise: seed & v == seed).
/// Generates k random seeds of weight w.
inline std::vector<int> generate_random_dnf(int dim, int k_terms, int term_width,
                                            uint32_t seed = 99) {
    int N = 1 << dim;
    std::mt19937 rng(seed);
    std::vector<uint32_t> terms(k_terms);

    // Generate k random conjunctive terms, each with term_width bits set
    for (int t = 0; t < k_terms; ++t) {
        std::vector<int> bits(dim);
        std::iota(bits.begin(), bits.end(), 0);
        std::shuffle(bits.begin(), bits.end(), rng);
        uint32_t term = 0;
        for (int i = 0; i < term_width && i < dim; ++i)
            term |= 1u << bits[i];
        terms[t] = term;
    }

    std::vector<int> table(N, 0);
    for (int v = 0; v < N; ++v) {
        for (auto term : terms) {
            if ((static_cast<uint32_t>(v) & term) == term) {
                table[v] = 1;
                break;
            }
        }
    }
    return table;
}

// ---------------------------------------------------------------------------
// Split a truth table into train/test datasets.
// ---------------------------------------------------------------------------

/// Build a BooleanDataset from a truth table, with train_fraction of vertices
/// randomly assigned to training and the rest to test.
inline BooleanDataset make_dataset(const std::string& name, int dim,
                                   const std::vector<int>& truth_table,
                                   float train_fraction = 0.7f,
                                   uint32_t seed = 42) {
    int N = 1 << dim;
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    int n_train = static_cast<int>(N * train_fraction);

    BooleanDataset ds;
    ds.name = name;
    ds.dim = dim;
    ds.train.reserve(n_train);
    ds.test.reserve(N - n_train);

    for (int i = 0; i < N; ++i) {
        BooleanDataset::Sample s{static_cast<uint32_t>(indices[i]),
                                  truth_table[indices[i]]};
        if (i < n_train)
            ds.train.push_back(s);
        else
            ds.test.push_back(s);
    }
    return ds;
}
