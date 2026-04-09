#pragma once

#include "HCNNNetwork.h"   // re-exports HCNNConv, HCNNPool, HCNNReadout, all enums
#include <memory>
#include <vector>

namespace hcnn {

/// @class HCNN
/// @brief Top-level HypercubeCNN SDK front door.  One class wraps the entire
///        pipeline: input embedding -> conv/pool stack -> readout, plus
///        single-sample inference, batch inference, single-sample training,
///        and mini-batch / full-epoch training.
///
/// HCNN owns a `HCNNNetwork` (the internal orchestrator) and forwards every
/// public call to it through a thin PIMPL-style wrapper.  Use this class for
/// virtually all SDK consumption -- the underlying layer classes
/// (`HCNNNetwork`, `HCNNConv`, `HCNNPool`, `HCNNReadout`, `ThreadPool`) are
/// re-exported transitively via this header for power users who need direct
/// weight access (serialization, gradient checking, custom training loops),
/// but ordinary code should never need to reach for them.
///
/// Build the architecture incrementally with AddConv()/AddPool(), then call
/// RandomizeWeights() before training:
///
///     hcnn::HCNN net(10, /*num_classes=*/10);   // DIM=10, N=1024
///     net.AddConv(32);
///     net.AddPool(hcnn::PoolType::MAX);
///     net.AddConv(64);
///     net.AddPool(hcnn::PoolType::MAX);
///     net.RandomizeWeights();
///
///     // Single-sample inference: caller owns and reuses both buffers.
///     std::vector<float> embedded(net.GetStartN());
///     std::vector<float> logits(net.GetNumClasses());
///     net.Embed(raw, raw_len, embedded.data());
///     net.Forward(embedded.data(), logits.data());
///
/// All methods that take raw inputs avoid hidden per-call allocations:
/// single-sample inference reuses persistent ping-pong scratch on the
/// network; batch inference and batch training reuse lazily-allocated
/// per-thread buffers.
///
/// **Enums** consumed by this API are defined alongside their owning
/// internal headers (all re-exported transitively via HCNN.h):
///   - `hcnn::PoolType`      (HCNNPool.h)     — MAX, AVG
///   - `hcnn::ReadoutType`   (HCNNNetwork.h)  — GAP, FLATTEN
///   - `hcnn::Activation`    (HCNNConv.h)     — NONE, RELU, LEAKY_RELU
///   - `hcnn::OptimizerType` (HCNNConv.h)     — SGD, ADAM
///
/// **Non-copyable, non-movable.**  HCNN owns a HCNNNetwork (which in turn
/// owns a ThreadPool with live worker threads) and persistent scratch
/// vectors used by inference and training.  Move semantics would require
/// either teaching the worker threads to follow the moved-from object or
/// rebuilding the pool on the destination — both add complexity for no
/// real-world win, so move is deleted entirely.  Wrap in
/// `std::unique_ptr<HCNN>` if you need transfer-of-ownership semantics.
class HCNN {
public:
    explicit HCNN(int start_dim, int num_classes = 10,
                  int input_channels = 1,
                  ReadoutType readout_type = ReadoutType::GAP,
                  size_t num_threads = 0);
    ~HCNN();

    HCNN(const HCNN&) = delete;
    HCNN& operator=(const HCNN&) = delete;
    HCNN(HCNN&&) = delete;
    HCNN& operator=(HCNN&&) = delete;

    // -----------------------------------------------------------------
    //  Architecture (incremental builder)
    // -----------------------------------------------------------------

    /// Append a convolutional layer with `c_out` output channels.
    void AddConv(int c_out, Activation activation = Activation::RELU,
                 bool use_bias = true, bool use_batchnorm = false);

    /// Append an antipodal pooling layer.  Reduces DIM by 1.
    void AddPool(PoolType type = PoolType::MAX);

    /// Initialize all weights.  scale > 0: uniform [-scale, +scale].
    /// scale <= 0 (default): per-layer Xavier/He init based on activation.
    void RandomizeWeights(float scale = 0.0f, unsigned seed = 42);

    // -----------------------------------------------------------------
    //  Mode / optimizer
    // -----------------------------------------------------------------

    /// Switch all batch-norm layers between training and eval mode.
    void SetTraining(bool training);

    /// Configure the optimizer for all layers.  Resets the timestep.
    void SetOptimizer(OptimizerType type, float beta1 = 0.9f,
                      float beta2 = 0.999f, float eps = 1e-8f);

    // -----------------------------------------------------------------
    //  Inference
    // -----------------------------------------------------------------

    /// Map a raw scalar array onto N = 2^start_dim hypercube vertices via
    /// Direct Linear Assignment.  Values must be in [-1.0, 1.0].
    /// `embedded_out` must hold GetStartN() floats.  Caller-owned buffer
    /// (designed for reuse across calls -- no hidden allocation).
    void Embed(const float* raw_input, int input_length,
               float* embedded_out) const;

    /// Run conv/pool/readout from already-embedded activations.
    /// `embedded` is GetStartN() floats; `logits` is GetNumClasses() floats.
    /// No allocation.
    void Forward(const float* embedded, float* logits) const;

    /// Batch inference (parallel via internal thread pool).  Embeds and
    /// forwards multiple samples.  Per-thread buffers are lazily allocated
    /// on first call and reused thereafter.
    /// `logits_out` must hold batch_size * GetNumClasses() floats.
    void ForwardBatch(const float* const* raw_inputs, const int* input_lengths,
                      int batch_size, float* logits_out);

    // -----------------------------------------------------------------
    //  Training
    // -----------------------------------------------------------------

    /// Single-sample SGD step (forward + backward + weight update).
    /// `class_weights` (optional, length GetNumClasses()) scales the
    /// per-class loss; pass nullptr for uniform weighting.
    void TrainStep(const float* raw_input, int input_length, int target_class,
                   float learning_rate, float momentum = 0.0f,
                   float weight_decay = 0.0f,
                   const float* class_weights = nullptr);

    /// Mini-batch parallel SGD step.  Forward+backward run in parallel for
    /// each sample, gradients are reduced (averaged), then a single weight
    /// update is applied.  Per-thread buffers are lazily allocated and reused.
    void TrainBatch(const float* const* inputs, const int* input_lengths,
                    const int* targets, int batch_size,
                    float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f,
                    const float* class_weights = nullptr);

    /// Iterate `sample_count` samples and dispatch TrainBatch in chunks of
    /// `batch_size` (the final chunk may be smaller).  Throws
    /// `std::invalid_argument` if `batch_size <= 0` or `sample_count < 0`.
    ///
    /// `shuffle_seed`:
    ///   - 0 (default): no shuffle, samples are processed in input order.
    ///   - nonzero: deterministic shuffle for this call, seeded by this value.
    ///     Pass a different seed each epoch (e.g. epoch index) for a fresh
    ///     reproducible permutation.  HCNN owns persistent gather buffers
    ///     used by the shuffle path -- after the first shuffled epoch, no
    ///     further allocations occur as long as `sample_count` does not grow.
    ///     Note: the gather buffers grow on demand but never shrink; their
    ///     steady-state size is the largest `sample_count` ever passed.
    void TrainEpoch(const float* const* inputs, const int* input_lengths,
                    const int* targets, int sample_count, int batch_size,
                    float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f,
                    const float* class_weights = nullptr,
                    unsigned shuffle_seed = 0);

    // -----------------------------------------------------------------
    //  Sizing accessors (everything a consumer needs to size buffers)
    // -----------------------------------------------------------------
    int GetStartDim() const;
    int GetStartN() const;
    int GetInputChannels() const;
    int GetNumClasses() const;

private:
    std::unique_ptr<HCNNNetwork> net_;

    // Persistent gather buffers used by TrainEpoch's shuffle path.
    // Allocated lazily and grown as `sample_count` increases; reused
    // across epochs so the steady-state shuffle is allocation-free.
    // Buffers grow on demand but never shrink — steady-state size is the
    // largest sample_count ever passed.
    std::vector<const float*> shuffle_inputs_;
    std::vector<int>          shuffle_lengths_;
    std::vector<int>          shuffle_targets_;
    std::vector<int>          shuffle_idx_;
};

} // namespace hcnn
