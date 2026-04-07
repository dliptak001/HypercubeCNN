// Speck32/64 cipher distinguisher — trains HCNN to distinguish real ciphertext
// pairs from random data. Reproduces the setup from Gohr (CRYPTO 2019).
//
// Input: 64 bits = two 32-bit ciphertexts (C0 || C1) from encrypting a
//        plaintext pair (P, P ^ delta) under the same random key.
// Label 1: real ciphertext pair.  Label 0: random 64-bit string.
// Maps to DIM=6 hypercube (2^6 = 64 vertices), one vertex per bit.

#include "HCNNNetwork.h"
#include "speck32.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

// Gohr's input difference: delta = (0x0040, 0x0000)
static constexpr uint32_t DELTA = 0x00400000u;

struct Sample {
    float bits[64]; // bipolar: -1.0 or +1.0
    int label;      // 1 = real ciphertext pair, 0 = random
};

// Pack a 64-bit value into bipolar floats.
static inline void to_bipolar(uint64_t val, float* out) {
    for (int i = 0; i < 64; ++i)
        out[i] = ((val >> (63 - i)) & 1) ? 1.0f : -1.0f;
}

// Fill a buffer with samples (50/50 split). Thread-safe with per-call RNG.
static void generate_into(Sample* buf, int n, int rounds, std::mt19937_64& rng) {
    int half = n / 2;
    for (int i = 0; i < n; ++i) {
        if (i < half) {
            // Positive: real ciphertext pair
            uint16_t master[4];
            uint64_t key_bits = rng();
            master[0] = static_cast<uint16_t>(key_bits);
            master[1] = static_cast<uint16_t>(key_bits >> 16);
            master[2] = static_cast<uint16_t>(key_bits >> 32);
            master[3] = static_cast<uint16_t>(key_bits >> 48);

            uint16_t rk[speck::FULL_ROUNDS];
            speck::key_schedule(master, rk);

            auto p0 = static_cast<uint32_t>(rng());
            uint32_t p1 = p0 ^ DELTA;
            uint32_t c0 = speck::encrypt_block(p0, rk, rounds);
            uint32_t c1 = speck::encrypt_block(p1, rk, rounds);

            uint64_t combined = (static_cast<uint64_t>(c0) << 32) | c1;
            to_bipolar(combined, buf[i].bits);
            buf[i].label = 1;
        } else {
            // Negative: random 64 bits
            to_bipolar(rng(), buf[i].bits);
            buf[i].label = 0;
        }
    }
}

// Parallel data generation across multiple threads.
static void generate_parallel(Sample* buf, int n, int rounds, uint64_t seed) {
    int nt = static_cast<int>(std::thread::hardware_concurrency());
    if (nt < 1) nt = 1;
    if (n < nt * 100) nt = 1; // don't bother threading for small n

    int chunk = n / nt;
    std::vector<std::thread> threads;
    for (int t = 0; t < nt; ++t) {
        int begin = t * chunk;
        int end = (t == nt - 1) ? n : begin + chunk;
        int count = end - begin;
        // Each thread gets a unique seed derived from base seed + thread id
        threads.emplace_back([=]() {
            std::mt19937_64 rng(seed + t * 0x9E3779B97F4A7C15ULL);
            generate_into(buf + begin, count, rounds, rng);
        });
    }
    for (auto& t : threads) t.join();

    // Shuffle across thread boundaries
    std::mt19937_64 shuffle_rng(seed ^ 0xDEADBEEF);
    std::shuffle(buf, buf + n, shuffle_rng);
}

// Evaluate accuracy and loss on a dataset. Pre-allocated buffers.
struct EvalResult { float accuracy; float loss; };

static EvalResult evaluate(HCNNNetwork& net, const std::vector<Sample>& data) {
    int correct = 0;
    float total_loss = 0.0f;
    int N = net.get_start_N();

    // Pre-allocate once
    std::vector<float> logits(2);
    std::vector<float> embedded(N);

    for (auto& s : data) {
        std::fill(logits.begin(), logits.end(), 0.0f);
        std::fill(embedded.begin(), embedded.end(), 0.0f);
        net.embed_input(s.bits, 64, embedded.data());
        net.forward(embedded.data(), logits.data());

        int pred = (logits[1] > logits[0]) ? 1 : 0;
        if (pred == s.label) ++correct;

        float max_l = std::max(logits[0], logits[1]);
        float sum_exp = std::exp(logits[0] - max_l) + std::exp(logits[1] - max_l);
        float log_prob = (logits[s.label] - max_l) - std::log(sum_exp);
        total_loss -= log_prob;
    }

    return {
        100.0f * correct / static_cast<float>(data.size()),
        total_loss / static_cast<float>(data.size())
    };
}

int main(int argc, char** argv) {
    int rounds = 5;
    int train_n = 2000000;
    int val_n = 50000;
    int test_n = 50000;
    int epochs = 80;

    if (argc >= 2) rounds = std::atoi(argv[1]);
    if (argc >= 3) train_n = std::atoi(argv[2]);

    printf("Speck32/64 cipher distinguisher\n");
    printf("Rounds: %d  Train: %d  Val: %d  Test: %d\n\n",
           rounds, train_n, val_n, test_n);

    // Val/test are fixed; train is regenerated each epoch
    printf("Generating val/test data...\n");
    std::vector<Sample> val_data(val_n);
    std::vector<Sample> test_data(test_n);
    generate_parallel(val_data.data(), val_n, rounds, 123);
    generate_parallel(test_data.data(), test_n, rounds, 456);

    // DIM=6 → 64 vertices, 2 classes
    const int DIM = 6;
    const int num_classes = 2;
    const float momentum = 0.9f;
    const float weight_decay = 1e-5f;
    const int batch_size = 512;
    const float lr_max = 0.02f;
    const float lr_min = 1e-5f;

    HCNNNetwork net(DIM, num_classes);
    // Delayed pooling: conv-conv-pool-conv-conv-pool
    // Keeps DIM=6 for two conv layers before first pool.
    // Wider first layer (64ch) for more initial feature capacity.
    // DIM: 6 → 6 → 5 → 5 → 4
    net.add_conv(64, true, true);   // 1*64*6 + 64 = 448 params
    net.add_conv(64, true, true);   // 64*64*6 + 64 = 24,640 params
    net.add_pool(PoolType::MAX);    // DIM 6→5
    net.add_conv(128, true, true);  // 64*128*5 + 128 = 41,088 params
    net.add_conv(128, true, true);  // 128*128*5 + 128 = 82,048 params
    net.add_pool(PoolType::MAX);    // DIM 5→4
    net.randomize_all_weights();
    // Total: ~148K params. Final: 128ch × 16 vertices → GAP → 128 → 2 classes

    printf("Architecture: 4 conv, 2 pool (64->64->128->128), DIM %d->%d\n",
           DIM, DIM - 2);
    printf("Optimizer: SGD, momentum=%.1f wd=%.1e lr=%.4f->cosine->%.1e batch=%d epochs=%d\n\n",
           momentum, weight_decay, lr_max, lr_min, batch_size, epochs);

    int N = net.get_start_N(); // 64

    // Pre-allocate training data buffer and batch buffers (reused every epoch)
    std::vector<Sample> train_data(train_n);
    std::vector<const float*> input_ptrs(train_n);
    std::vector<int> input_lens(train_n, N);
    std::vector<int> targets(train_n);
    std::vector<size_t> order(train_n);
    std::iota(order.begin(), order.end(), 0);

    // Pre-allocate batch assembly buffers at max batch size
    std::vector<const float*> batch_ptrs(batch_size);
    std::vector<int> batch_lens(batch_size, N);
    std::vector<int> batch_targets(batch_size);

    float best_val_acc = 0.0f;
    int best_epoch = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float progress = static_cast<float>(epoch) / static_cast<float>(epochs);
        float current_lr = lr_min + 0.5f * (lr_max - lr_min)
                           * (1.0f + std::cos(static_cast<float>(std::numbers::pi) * progress));

        // Fresh training data each epoch (key advantage of synthetic data —
        // infinite effective dataset, no overfitting to specific samples)
        generate_parallel(train_data.data(), train_n, rounds,
                          42 + static_cast<uint64_t>(epoch) * 1000003ULL);

        for (int i = 0; i < train_n; ++i) {
            input_ptrs[i] = train_data[i].bits;
            targets[i] = train_data[i].label;
        }

        // Shuffle order
        std::mt19937_64 shuffle_rng(99 + epoch);
        std::shuffle(order.begin(), order.end(), shuffle_rng);

        auto t0 = std::chrono::steady_clock::now();

        for (int start = 0; start < train_n; start += batch_size) {
            int actual = std::min(batch_size, train_n - start);
            for (int j = 0; j < actual; ++j) {
                size_t idx = order[start + j];
                batch_ptrs[j] = input_ptrs[idx];
                batch_targets[j] = targets[idx];
            }
            net.train_batch(batch_ptrs.data(), batch_lens.data(),
                            batch_targets.data(), actual,
                            current_lr, momentum, weight_decay, nullptr);
        }

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        auto val = evaluate(net, val_data);
        if (val.accuracy > best_val_acc) {
            best_val_acc = val.accuracy;
            best_epoch = epoch + 1;
        }

        printf("ep %2d  val_acc=%5.1f%%  val_loss=%.4f  lr=%.5f  %.1fs\n",
               epoch + 1, val.accuracy, val.loss, current_lr, secs);
    }

    printf("\nBest val accuracy: %.1f%% (epoch %d)\n\n", best_val_acc, best_epoch);

    auto test = evaluate(net, test_data);
    printf("Test accuracy: %.1f%%  Test loss: %.4f\n", test.accuracy, test.loss);
    printf("(Random baseline: 50.0%%)\n");

    return 0;
}
