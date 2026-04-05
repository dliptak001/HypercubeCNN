/**
 * @file HCNN.h
 * @brief Hypercube convolutional layer — sparse-vertex convolution on a
 *        binary hypercube using fixed XOR masks instead of spatial grids.
 *
 * An HCNN layer maps c_in input channels defined on the vertices of a
 * DIM-dimensional binary hypercube (N = 2^DIM vertices) to c_out output
 * channels on the same hypercube.  For each output vertex v, the layer
 * computes:
 *
 *   out_co(v) = b_co + sum over (ci, k) of w[co,ci,k] * in[ci, v ^ (1 << k)]
 *
 * where k ranges over [0, DIM), so each mask is a single-bit flip
 * selecting the nearest neighbor at Hamming distance 1 along bit k.
 *
 * Each mask selects exactly one neighbor per vertex; each gets its own learned
 * weight, shared across all vertices (CNN-style weight sharing).
 *
 * All geometry is bitwise — neighbor lookup uses XOR with the mask array;
 * there are no adjacency lists or spatial padding.
 *
 * Memory layout is **channel-major**: element [c*N + v] stores channel c,
 * vertex v.
 */

#pragma once

#include <vector>
#include <random>

class ThreadPool;

/**
 * @class HCNN
 * @brief A single hypercube convolutional layer with optional ReLU and bias.
 *
 * Supports forward inference, backpropagation with SGD+momentum weight
 * updates, and a separated gradient-computation path for numerical
 * gradient checking.
 */
class HCNN {
public:
    /**
     * @brief Construct a hypercube convolutional layer.
     *
     * Builds the K = DIM nearest-neighbor XOR mask table.  Kernel and bias
     * weights are initialized to zero; call randomize_weights() before training.
     *
     * Requires dim >= 3 so that K >= 3.
     *
     * @param dim      Hypercube dimension.  The layer operates on N = 2^dim vertices.
     * @param c_in     Number of input channels.
     * @param c_out    Number of output channels (filters).
     * @param use_relu If true, apply ReLU activation after convolution (default: true).
     * @param use_bias If true, add a learnable per-output-channel bias (default: true).
     */
    HCNN(int dim, int c_in, int c_out, bool use_relu = true, bool use_bias = true);

    /**
     * @brief Initialize kernel weights.
     *
     * When scale > 0, uses uniform random values in [-scale, +scale].
     * When scale <= 0, uses Xavier/Glorot uniform initialization:
     * [-s, +s] where s = sqrt(6 / (fan_in + fan_out)),
     * fan_in = c_in * K, fan_out = c_out * K.
     *
     * Biases are reset to zero.  Momentum velocity buffers are cleared.
     *
     * @param scale  Half-width of the uniform range, or <= 0 for Xavier init.
     * @param rng    Mersenne Twister PRNG instance (caller-owned).
     */
    void randomize_weights(float scale, std::mt19937& rng);

    /**
     * @brief Execute the forward pass over all output channels.
     *
     * For each output channel and each vertex, looks up K specific neighbors
     * via XOR masks, multiplies by the corresponding kernel weight, sums,
     * adds bias, and applies the activation function.
     *
     * @param[in]  in       Input activations, channel-major [c_in * N].
     * @param[out] out      Output activations, channel-major [c_out * N].
     * @param[out] pre_act  If non-null, receives the pre-activation (pre-ReLU)
     *                      values [c_out * N].  Required by backward().
     */
    void forward(const float* in, float* out, float* pre_act = nullptr) const;

    /**
     * @brief Backward pass: compute input gradients and update weights via SGD.
     *
     * Applies the chain rule through the activation function, then:
     *   -# Computes grad_in (if non-null) using the same XOR-lookup structure
     *      as forward (XOR is self-inverse, so the transpose is itself).
     *   -# Updates kernel weights using momentum SGD:
     *      v <- mu*v + g,  w <- w - eta*v
     *   -# Updates bias weights similarly (if bias is enabled).
     *
     * @param[in]  grad_out      Gradient of loss w.r.t. output activations [c_out * N].
     * @param[in]  in            Input activations from the forward pass [c_in * N].
     * @param[in]  pre_act       Pre-activation values from the forward pass [c_out * N].
     * @param[out] grad_in       Gradient of loss w.r.t. input activations [c_in * N],
     *                           or nullptr if not needed (e.g. first layer).
     * @param      learning_rate SGD learning rate (eta).
     * @param      momentum      SGD momentum coefficient (mu); default 0 (no momentum).
     */
    void backward(const float* grad_out, const float* in, const float* pre_act,
                  float* grad_in, float learning_rate, float momentum = 0.0f);

    /**
     * @brief Compute gradients without applying an SGD update.
     *
     * Identical to the gradient-computation portion of backward(), but writes
     * raw gradients into caller-provided buffers instead of updating internal
     * weights.  Used for numerical gradient checking.
     *
     * @param[in]  grad_out    Gradient of loss w.r.t. output activations [c_out * N].
     * @param[in]  in          Input activations from the forward pass [c_in * N].
     * @param[in]  pre_act     Pre-activation values from the forward pass [c_out * N].
     * @param[out] grad_in     Gradient of loss w.r.t. input activations [c_in * N],
     *                         or nullptr if not needed.
     * @param[out] kernel_grad Gradient of loss w.r.t. kernel weights [c_out * c_in * K].
     * @param[out] bias_grad   Gradient of loss w.r.t. bias [c_out],
     *                         or nullptr if bias is disabled.
     */
    void compute_gradients(const float* grad_out, const float* in, const float* pre_act,
                           float* grad_in, float* kernel_grad, float* bias_grad) const;

    /**
     * @brief Apply externally computed gradients via momentum SGD.
     *
     * Used by mini-batch training: gradients are computed per-sample via
     * compute_gradients(), averaged across the batch, then applied here.
     *
     * @param kernel_grad  Averaged kernel gradients [c_out * c_in * K].
     * @param bias_grad    Averaged bias gradients [c_out], or nullptr if no bias.
     * @param learning_rate SGD learning rate.
     * @param momentum      SGD momentum coefficient.
     */
    void apply_gradients(const float* kernel_grad, const float* bias_grad,
                         float learning_rate, float momentum);

    /** @name Accessors */
    ///@{
    int get_dim() const { return DIM; }       ///< Hypercube dimension.
    int get_N() const { return N; }           ///< Vertex count (2^DIM).
    int get_c_in() const { return c_in; }     ///< Number of input channels.
    int get_c_out() const { return c_out; }   ///< Number of output channels.
    int get_K() const { return K; }           ///< Number of connection masks (= DIM).
    ///@}

    /// Set the thread pool for parallel execution (nullptr = single-threaded).
    void set_thread_pool(ThreadPool* pool) { thread_pool = pool; }

    /** @name Raw weight access (for serialization and gradient checking) */
    ///@{
    float* get_kernel_data() { return kernel.data(); }                           ///< Pointer to kernel weight array.
    int get_kernel_size() const { return static_cast<int>(kernel.size()); }      ///< Total kernel weight count.
    float* get_bias_data() { return bias.data(); }                               ///< Pointer to bias array.
    int get_bias_size() const { return static_cast<int>(bias.size()); }          ///< Bias element count (0 if bias disabled).
    ///@}

private:
    int DIM;          ///< Hypercube dimension.
    int N;            ///< Number of vertices, always 2^DIM.
    int c_in;         ///< Input channel count.
    int c_out;        ///< Output channel count (number of filters).
    int K;            ///< Number of connection masks (= DIM).
    bool use_relu;    ///< Whether ReLU activation is applied after convolution.
    bool use_bias;    ///< Whether a learnable bias term is added per output channel.

    std::vector<float> kernel;          ///< Kernel weights, layout [c_out * c_in * K].
    std::vector<float> bias;            ///< Per-output-channel bias, size c_out (empty if bias disabled).
    std::vector<float> kernel_vel;      ///< Momentum velocity for kernel weights (same layout as kernel).
    std::vector<float> bias_vel;        ///< Momentum velocity for bias (same layout as bias).

    ThreadPool* thread_pool = nullptr;  ///< Optional thread pool for parallel execution.

    /**
     * @brief Compute the flat index into the kernel array.
     * @param co Output channel index.
     * @param ci Input channel index.
     * @param k  Mask index (0 .. K-1).
     * @return   Index into the kernel vector.
     */
    int kernel_idx(int co, int ci, int k) const {
        return (co * c_in + ci) * K + k;
    }

    /**
     * @brief Apply the activation function (ReLU or identity).
     * @param x Pre-activation value.
     * @return  Activated value.
     */
    float activate(float x) const;

    /**
     * @brief Derivative of the activation function evaluated at @p x.
     * @param x Pre-activation value.
     * @return  1.0 if identity or x > 0; 0.0 otherwise.
     */
    float activate_derivative(float x) const;
};
