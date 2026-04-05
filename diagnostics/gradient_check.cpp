#include "HCNNNetwork.h"
#include "dataloader/HCNNMNISTDataset.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
#include <iomanip>

// Compute cross-entropy loss for a single sample through the full network.
// Uses double precision for the loss computation to avoid float quantization
// issues in finite-difference gradient checking (loss ~2.3, float eps ~2.7e-7).
static double compute_loss(HCNNNetwork& net, const float* input, int input_len,
                           int target_class) {
    int N = net.get_start_N();
    int K = net.get_num_classes();
    std::vector<float> embedded(N, 0.0f);
    std::vector<float> logits(K, 0.0f);

    net.embed_input(input, input_len, embedded.data());
    net.forward(embedded.data(), logits.data());

    double max_logit = logits[0];
    for (int i = 1; i < K; ++i)
        if (logits[i] > max_logit) max_logit = logits[i];
    double sum_exp = 0.0;
    for (int i = 0; i < K; ++i)
        sum_exp += std::exp(static_cast<double>(logits[i]) - max_logit);
    return -(static_cast<double>(logits[target_class]) - max_logit) + std::log(sum_exp);
}

struct GradCheckResult {
    int total = 0;
    int passed = 0;
    int failed = 0;
    float max_rel_error = 0.0f;
};

// Check gradients for a flat weight array.
// perturb_fn(i, delta) should set weight[i] += delta.
// get_analytical(i) should return the analytical gradient for weight[i].
static GradCheckResult check_weights(
    const char* name, int num_weights,
    std::function<void(int, float)> perturb_fn,
    std::function<float(int)> get_analytical,
    std::function<double()> loss_fn,
    float eps = 1e-4f, float tol = 1e-2f)
{
    GradCheckResult result;
    result.total = num_weights;

    std::cout << "  " << name << " (" << num_weights << " weights):\n";

    for (int i = 0; i < num_weights; ++i) {
        perturb_fn(i, eps);
        double loss_plus = loss_fn();
        perturb_fn(i, -2.0f * eps);
        double loss_minus = loss_fn();
        perturb_fn(i, eps); // restore

        float numerical = static_cast<float>((loss_plus - loss_minus) / (2.0 * eps));
        float analytical = get_analytical(i);

        float abs_diff = std::fabs(numerical - analytical);
        // Skip relative error check when both values are near zero —
        // relative error inflates with tiny denominators, especially under
        // -ffast-math where rounding order varies between passes.
        if (std::fabs(numerical) < 1e-3f && std::fabs(analytical) < 1e-3f) {
            result.passed++;
            continue;
        }
        float denom = std::max(std::fabs(numerical) + std::fabs(analytical), 1e-8f);
        float rel_error = abs_diff / denom;

        if (rel_error > result.max_rel_error) result.max_rel_error = rel_error;

        if (rel_error > tol) {
            result.failed++;
            std::cout << "    FAIL w[" << i << "]: analytical=" << analytical
                      << " numerical=" << numerical << " rel_err=" << rel_error << "\n";
        } else {
            result.passed++;
        }
    }

    if (result.failed == 0) {
        std::cout << "    PASS all " << result.total << " weights (max rel_err="
                  << result.max_rel_error << ")\n";
    } else {
        std::cout << "    " << result.failed << "/" << result.total << " FAILED\n";
    }
    return result;
}

int main() {
    // Small network for fast gradient checking — no ReLU so gradients flow cleanly.
    // ReLU introduces non-differentiable points at zero which cause spurious failures.
    HCNNNetwork net(4); // DIM=4, N=16
    net.add_conv(4, false, true);  // no ReLU
    net.add_conv(4, false, true);  // no ReLU
    net.randomize_all_weights(0.3f);

    // Single sample
    const int input_len = 16;
    const int target_class = 3;
    std::vector<float> input(input_len);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : input) v = dist(rng);

    auto loss_fn = [&]() {
        return compute_loss(net, input.data(), input_len, target_class);
    };

    // --- Full forward/backward to get analytical gradients ---
    int N = net.get_start_N();
    int K = net.get_num_classes();
    std::vector<float> embedded(N);
    net.embed_input(input.data(), input_len, embedded.data());

    // Forward through all layers, caching activations
    const auto& layer_types = net.get_layer_types();
    int num_layers = static_cast<int>(layer_types.size());

    struct LayerCache {
        std::vector<float> activation;
        std::vector<float> pre_act;
        int N;
        int channels;
    };

    std::vector<LayerCache> cache(num_layers + 1);
    cache[0].N = N;
    cache[0].channels = 1;
    cache[0].activation.assign(embedded.begin(), embedded.end());

    int cur_N = N;
    size_t ci = 0;
    for (int i = 0; i < num_layers; ++i) {
        auto& c = cache[i + 1];
        // All layers are conv in this test (no pool)
        c.N = cur_N;
        c.channels = net.get_conv(ci).get_c_out();
        c.activation.resize(c.channels * cur_N);
        c.pre_act.resize(c.channels * cur_N);
        net.get_conv(ci).forward(cache[i].activation.data(),
                                 c.activation.data(), c.pre_act.data());
        ++ci;
    }

    // Readout forward
    auto& final_c = cache[num_layers];
    std::vector<float> logits(K, 0.0f);
    net.get_readout().forward(final_c.activation.data(), logits.data(), final_c.N);

    // Softmax + cross-entropy gradient
    float max_logit = logits[0];
    for (int i = 1; i < K; ++i)
        if (logits[i] > max_logit) max_logit = logits[i];
    std::vector<float> probs(K);
    float sum_exp = 0.0f;
    for (int i = 0; i < K; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < K; ++i) probs[i] /= sum_exp;

    std::vector<float> grad_logits(K);
    for (int i = 0; i < K; ++i)
        grad_logits[i] = probs[i] - (i == target_class ? 1.0f : 0.0f);

    // Backward through readout
    std::vector<float> readout_weight_grad(net.get_readout().get_weight_size());
    std::vector<float> readout_bias_grad(net.get_readout().get_bias_size());
    std::vector<float> grad_current(final_c.channels * final_c.N);
    net.get_readout().compute_gradients(grad_logits.data(), final_c.activation.data(),
                                       final_c.N, grad_current.data(),
                                       readout_weight_grad.data(),
                                       readout_bias_grad.data());

    // Backward through conv layers
    struct ConvGrads {
        std::vector<float> kernel_grad;
        std::vector<float> bias_grad;
    };
    std::vector<ConvGrads> conv_grads(net.get_num_conv());

    ci = net.get_num_conv();
    for (int i = num_layers - 1; i >= 0; --i) {
        --ci;
        auto& cg = conv_grads[ci];
        cg.kernel_grad.resize(net.get_conv(ci).get_kernel_size());
        cg.bias_grad.resize(net.get_conv(ci).get_bias_size());
        std::vector<float> grad_prev(cache[i].channels * cache[i].N, 0.0f);

        net.get_conv(ci).compute_gradients(
            grad_current.data(), cache[i].activation.data(),
            cache[i + 1].pre_act.data(),
            (i > 0) ? grad_prev.data() : nullptr,
            cg.kernel_grad.data(), cg.bias_grad.data());

        grad_current = std::move(grad_prev);
    }

    // --- Numerical gradient checks ---
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Gradient check (eps=1e-4, tol=1e-2)\n";
    std::cout << "Initial loss: " << loss_fn() << "\n\n";

    int total_pass = 0, total_fail = 0;

    // Check readout weights
    {
        auto r = check_weights("Readout weights",
            net.get_readout().get_weight_size(),
            [&](int i, float delta) { net.get_readout().get_weight_data()[i] += delta; },
            [&](int i) { return readout_weight_grad[i]; },
            loss_fn);
        total_pass += r.passed;
        total_fail += r.failed;
    }

    // Check readout bias
    {
        auto r = check_weights("Readout bias",
            net.get_readout().get_bias_size(),
            [&](int i, float delta) { net.get_readout().get_bias_data()[i] += delta; },
            [&](int i) { return readout_bias_grad[i]; },
            loss_fn);
        total_pass += r.passed;
        total_fail += r.failed;
    }

    // Check conv layer weights
    for (size_t li = 0; li < net.get_num_conv(); ++li) {
        auto& conv = net.get_conv(li);
        std::string name = "Conv[" + std::to_string(li) + "] kernel";
        auto r = check_weights(name.c_str(),
            conv.get_kernel_size(),
            [&](int i, float delta) { conv.get_kernel_data()[i] += delta; },
            [&](int i) { return conv_grads[li].kernel_grad[i]; },
            loss_fn);
        total_pass += r.passed;
        total_fail += r.failed;

        if (conv.get_bias_size() > 0) {
            name = "Conv[" + std::to_string(li) + "] bias";
            r = check_weights(name.c_str(),
                conv.get_bias_size(),
                [&](int i, float delta) { conv.get_bias_data()[i] += delta; },
                [&](int i) { return conv_grads[li].bias_grad[i]; },
                loss_fn);
            total_pass += r.passed;
            total_fail += r.failed;
        }
    }

    std::cout << "\n=== SUMMARY: " << total_pass << " passed, "
              << total_fail << " failed out of "
              << (total_pass + total_fail) << " total ===\n";

    return total_fail > 0 ? 1 : 0;
}
