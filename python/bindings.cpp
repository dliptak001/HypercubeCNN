// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// Thin pybind11 surface for HypercubeCNN. Ergonomics (shape checks, TrainParams
// dataclass, docs) live in hypercube_cnn/__init__.py.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "HCNN.h"
#include "HCNNSpatialAug.h"
#include "HCNNSpatialEmbed.h"
#include "HCNNTrainHelpers.h"

namespace py = pybind11;
using namespace hcnn;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

namespace {

TrainParams make_params(float learning_rate, float momentum, float weight_decay,
                        unsigned shuffle_seed,
                        const float* class_weights) {
    TrainParams p;
    p.learning_rate = learning_rate;
    p.momentum = momentum;
    p.weight_decay = weight_decay;
    p.shuffle_seed = shuffle_seed;
    p.class_weights = class_weights;
    return p;
}

// contiguous float buffer view (size in elements)
struct FBuf {
    const float* ptr = nullptr;
    size_t size = 0;
};

FBuf req_float(FloatArray& a) {
    auto buf = a.request();
    return {static_cast<const float*>(buf.ptr), static_cast<size_t>(buf.size)};
}

struct IBuf {
    const int* ptr = nullptr;
    size_t size = 0;
};

IBuf req_int(IntArray& a) {
    auto buf = a.request();
    return {static_cast<const int*>(buf.ptr), static_cast<size_t>(buf.size)};
}

} // namespace

PYBIND11_MODULE(_core, m)
{
    m.doc() = "HypercubeCNN: dependency-free hypercube CNN core (Python bindings)";
    m.attr("__version__") = "1.0.0";

    // ── Enums ──
    py::enum_<Activation>(m, "Activation")
        .value("NONE", Activation::NONE)
        .value("RELU", Activation::RELU)
        .value("LEAKY_RELU", Activation::LEAKY_RELU)
        .value("TANH", Activation::TANH);

    py::enum_<PoolType>(m, "PoolType")
        .value("MAX", PoolType::MAX)
        .value("AVG", PoolType::AVG);

    py::enum_<TaskType>(m, "TaskType")
        .value("Classification", TaskType::Classification)
        .value("Regression", TaskType::Regression);

    py::enum_<OptimizerType>(m, "OptimizerType")
        .value("SGD", OptimizerType::SGD)
        .value("ADAM", OptimizerType::ADAM);

    // ── HCNN ──
    py::class_<HCNN>(m, "_HCNN")
        .def(py::init([](int start_dim, int num_outputs, int input_channels,
                         TaskType task_type, size_t num_threads) {
                 return std::make_unique<HCNN>(
                     start_dim, num_outputs, input_channels, task_type, num_threads);
             }),
             py::arg("start_dim"),
             py::arg("num_outputs") = 10,
             py::arg("input_channels") = 1,
             py::arg("task_type") = TaskType::Classification,
             py::arg("num_threads") = 0ULL)

        // Architecture
        .def("add_conv",
             &HCNN::AddConv,
             py::arg("c_out"),
             py::arg("activation") = Activation::RELU,
             py::arg("use_bias") = true,
             py::arg("use_batchnorm") = false)

        .def("add_pool",
             &HCNN::AddPool,
             py::arg("type") = PoolType::MAX)

        .def("randomize_weights",
             &HCNN::RandomizeWeights,
             py::arg("scale") = 0.0f,
             py::arg("seed") = 42u)

        // Mode / optimizer
        .def("set_training", &HCNN::SetTraining, py::arg("training"))

        .def("set_optimizer",
             &HCNN::SetOptimizer,
             py::arg("type"),
             py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f,
             py::arg("eps") = 1e-8f)

        .def("set_train_defaults",
             [](HCNN& self, float learning_rate, float momentum, float weight_decay,
                unsigned shuffle_seed) {
                 self.SetTrainDefaults(
                     make_params(learning_rate, momentum, weight_decay, shuffle_seed,
                                 nullptr));
             },
             py::arg("learning_rate") = 1e-3f,
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f,
             py::arg("shuffle_seed") = 0u)

        .def("prepare_buffers", &HCNN::PrepareBuffers)

        // ── Inference ──
        .def("predict",
             [](HCNN& self, FloatArray input) {
                 auto in = req_float(input);
                 const int K = self.GetNumOutputs();
                 py::array_t<float> out(K);
                 {
                     py::gil_scoped_release release;
                     self.Predict(in.ptr, static_cast<int>(in.size), out.mutable_data());
                 }
                 return out;
             },
             py::arg("input"),
             "Embed + forward. Returns (num_outputs,) float32.")

        .def("predict_class",
             [](HCNN& self, FloatArray input) {
                 auto in = req_float(input);
                 int cls = 0;
                 {
                     py::gil_scoped_release release;
                     cls = self.PredictClass(in.ptr, static_cast<int>(in.size));
                 }
                 return cls;
             },
             py::arg("input"),
             "Classification only: embed + forward + argmax.")

        .def("forward",
             [](HCNN& self, FloatArray embedded) {
                 auto in = req_float(embedded);
                 const size_t cap = static_cast<size_t>(self.GetInputChannels())
                                    * static_cast<size_t>(self.GetStartN());
                 if (in.size != cap)
                     throw std::invalid_argument(
                         "forward: embedded size (" + std::to_string(in.size)
                         + ") must equal capacity input_channels*N ("
                         + std::to_string(cap) + ")");
                 const int K = self.GetNumOutputs();
                 py::array_t<float> out(K);
                 {
                     py::gil_scoped_release release;
                     self.Forward(in.ptr, out.mutable_data());
                 }
                 return out;
             },
             py::arg("embedded"),
             "Forward from already-embedded activations of length C*N.")

        .def("forward_batch",
             [](HCNN& self, FloatArray flat_inputs, int input_length, int batch_size) {
                 auto in = req_float(flat_inputs);
                 const size_t need =
                     static_cast<size_t>(batch_size) * static_cast<size_t>(input_length);
                 if (in.size != need)
                     throw std::invalid_argument(
                         "forward_batch: flat_inputs size (" + std::to_string(in.size)
                         + ") must equal batch_size * input_length ("
                         + std::to_string(need) + ")");
                 const int K = self.GetNumOutputs();
                 py::array_t<float> out({batch_size, K});
                 {
                     py::gil_scoped_release release;
                     self.ForwardBatch(in.ptr, input_length, batch_size, out.mutable_data());
                 }
                 return out;
             },
             py::arg("flat_inputs"), py::arg("input_length"), py::arg("batch_size"),
             "Batch inference. flat_inputs: (batch_size * input_length,). "
             "Returns (batch_size, num_outputs).")

        // ── Training (TrainParams fields; class_weights optional array) ──
        .def("train_step_class",
             [](HCNN& self, FloatArray input, int target_class,
                float learning_rate, float momentum, float weight_decay,
                py::object class_weights_obj) {
                 auto in = req_float(input);
                 const float* cw = nullptr;
                 FloatArray cw_keep; // keep buffer alive if provided
                 if (!class_weights_obj.is_none()) {
                     cw_keep = class_weights_obj.cast<FloatArray>();
                     auto cwb = req_float(cw_keep);
                     if (static_cast<int>(cwb.size) != self.GetNumOutputs())
                         throw std::invalid_argument(
                             "class_weights length must equal num_outputs");
                     cw = cwb.ptr;
                 }
                 auto p = make_params(learning_rate, momentum, weight_decay, 0u, cw);
                 {
                     py::gil_scoped_release release;
                     self.TrainStep(in.ptr, static_cast<int>(in.size), target_class, p);
                 }
             },
             py::arg("input"), py::arg("target_class"),
             py::arg("learning_rate") = 1e-3f,
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f,
             py::arg("class_weights") = py::none())

        .def("train_step_reg",
             [](HCNN& self, FloatArray input, FloatArray target,
                float learning_rate, float momentum, float weight_decay) {
                 auto in = req_float(input);
                 auto t = req_float(target);
                 if (static_cast<int>(t.size) != self.GetNumOutputs())
                     throw std::invalid_argument(
                         "target size (" + std::to_string(t.size)
                         + ") must equal num_outputs ("
                         + std::to_string(self.GetNumOutputs()) + ")");
                 auto p = make_params(learning_rate, momentum, weight_decay, 0u, nullptr);
                 {
                     py::gil_scoped_release release;
                     self.TrainStep(in.ptr, static_cast<int>(in.size), t.ptr, p);
                 }
             },
             py::arg("input"), py::arg("target"),
             py::arg("learning_rate") = 1e-3f,
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f)

        .def("train_batch_class",
             [](HCNN& self, FloatArray flat_inputs, int input_length,
                IntArray targets, int batch_size,
                float learning_rate, float momentum, float weight_decay,
                py::object class_weights_obj) {
                 auto in = req_float(flat_inputs);
                 auto y = req_int(targets);
                 const size_t need =
                     static_cast<size_t>(batch_size) * static_cast<size_t>(input_length);
                 if (in.size != need)
                     throw std::invalid_argument(
                         "train_batch: flat_inputs size mismatch");
                 if (static_cast<int>(y.size) != batch_size)
                     throw std::invalid_argument(
                         "train_batch: targets length must equal batch_size");
                 const float* cw = nullptr;
                 FloatArray cw_keep;
                 if (!class_weights_obj.is_none()) {
                     cw_keep = class_weights_obj.cast<FloatArray>();
                     auto cwb = req_float(cw_keep);
                     if (static_cast<int>(cwb.size) != self.GetNumOutputs())
                         throw std::invalid_argument(
                             "class_weights length must equal num_outputs");
                     cw = cwb.ptr;
                 }
                 auto p = make_params(learning_rate, momentum, weight_decay, 0u, cw);
                 {
                     py::gil_scoped_release release;
                     self.TrainBatch(in.ptr, input_length, y.ptr, batch_size, p);
                 }
             },
             py::arg("flat_inputs"), py::arg("input_length"),
             py::arg("targets"), py::arg("batch_size"),
             py::arg("learning_rate") = 1e-3f,
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f,
             py::arg("class_weights") = py::none())

        .def("train_batch_reg",
             [](HCNN& self, FloatArray flat_inputs, int input_length,
                FloatArray flat_targets, int batch_size,
                float learning_rate, float momentum, float weight_decay) {
                 auto in = req_float(flat_inputs);
                 auto t = req_float(flat_targets);
                 const size_t need_in =
                     static_cast<size_t>(batch_size) * static_cast<size_t>(input_length);
                 const size_t need_t =
                     static_cast<size_t>(batch_size)
                     * static_cast<size_t>(self.GetNumOutputs());
                 if (in.size != need_in)
                     throw std::invalid_argument(
                         "train_batch: flat_inputs size mismatch");
                 if (t.size != need_t)
                     throw std::invalid_argument(
                         "train_batch: flat_targets size must be batch_size * num_outputs");
                 auto p = make_params(learning_rate, momentum, weight_decay, 0u, nullptr);
                 {
                     py::gil_scoped_release release;
                     self.TrainBatch(in.ptr, input_length, t.ptr, batch_size, p);
                 }
             },
             py::arg("flat_inputs"), py::arg("input_length"),
             py::arg("flat_targets"), py::arg("batch_size"),
             py::arg("learning_rate") = 1e-3f,
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f)

        .def("train_epoch_class",
             [](HCNN& self, FloatArray flat_inputs, int input_length,
                IntArray targets, int sample_count, int batch_size,
                float learning_rate, float momentum, float weight_decay,
                unsigned shuffle_seed, py::object class_weights_obj) {
                 auto in = req_float(flat_inputs);
                 auto y = req_int(targets);
                 const size_t need_in =
                     static_cast<size_t>(sample_count) * static_cast<size_t>(input_length);
                 if (in.size != need_in)
                     throw std::invalid_argument(
                         "train_epoch: flat_inputs size mismatch");
                 if (static_cast<int>(y.size) != sample_count)
                     throw std::invalid_argument(
                         "train_epoch: targets length must equal sample_count");
                 const float* cw = nullptr;
                 FloatArray cw_keep;
                 if (!class_weights_obj.is_none()) {
                     cw_keep = class_weights_obj.cast<FloatArray>();
                     auto cwb = req_float(cw_keep);
                     if (static_cast<int>(cwb.size) != self.GetNumOutputs())
                         throw std::invalid_argument(
                             "class_weights length must equal num_outputs");
                     cw = cwb.ptr;
                 }
                 auto p = make_params(learning_rate, momentum, weight_decay,
                                      shuffle_seed, cw);
                 {
                     py::gil_scoped_release release;
                     self.TrainEpoch(in.ptr, input_length, y.ptr, sample_count,
                                     batch_size, p);
                 }
             },
             py::arg("flat_inputs"), py::arg("input_length"),
             py::arg("targets"), py::arg("sample_count"), py::arg("batch_size"),
             py::arg("learning_rate") = 1e-3f,
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f,
             py::arg("shuffle_seed") = 0u,
             py::arg("class_weights") = py::none())

        .def("train_epoch_reg",
             [](HCNN& self, FloatArray flat_inputs, int input_length,
                FloatArray flat_targets, int sample_count, int batch_size,
                float learning_rate, float momentum, float weight_decay,
                unsigned shuffle_seed) {
                 auto in = req_float(flat_inputs);
                 auto t = req_float(flat_targets);
                 const size_t need_in =
                     static_cast<size_t>(sample_count) * static_cast<size_t>(input_length);
                 const size_t need_t =
                     static_cast<size_t>(sample_count)
                     * static_cast<size_t>(self.GetNumOutputs());
                 if (in.size != need_in)
                     throw std::invalid_argument(
                         "train_epoch: flat_inputs size mismatch");
                 if (t.size != need_t)
                     throw std::invalid_argument(
                         "train_epoch: flat_targets size must be sample_count * num_outputs");
                 auto p = make_params(learning_rate, momentum, weight_decay,
                                      shuffle_seed, nullptr);
                 {
                     py::gil_scoped_release release;
                     self.TrainEpoch(in.ptr, input_length, t.ptr, sample_count,
                                     batch_size, p);
                 }
             },
             py::arg("flat_inputs"), py::arg("input_length"),
             py::arg("flat_targets"), py::arg("sample_count"), py::arg("batch_size"),
             py::arg("learning_rate") = 1e-3f,
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f,
             py::arg("shuffle_seed") = 0u)

        // ── Weights ──
        .def("weight_count", &HCNN::GetWeightCount)

        .def("get_weights",
             [](const HCNN& self) {
                 auto v = self.GetWeights();
                 py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
                 if (!v.empty())
                     std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
                 return arr;
             },
             "Parameter blob (float32); excludes optimizer moments.")

        .def("set_weights",
             [](HCNN& self, FloatArray data, bool reset_optimizer_moments) {
                 auto b = req_float(data);
                 self.SetWeights(b.ptr, b.size, reset_optimizer_moments);
             },
             py::arg("data"),
             py::arg("reset_optimizer_moments") = false)

        // ── HCNW file I/O (params only; keep arch JSON beside the file) ──
        .def("save_weights",
             [](const HCNN& self, const std::string& path) {
                 py::gil_scoped_release release;
                 save_weights(self, path);
             },
             py::arg("path"),
             "Write HCNW weight file (parameters + coarse arch checks).")

        .def("load_weights",
             [](HCNN& self, const std::string& path, bool reset_optimizer_moments) {
                 py::gil_scoped_release release;
                 load_weights(self, path, reset_optimizer_moments);
             },
             py::arg("path"),
             py::arg("reset_optimizer_moments") = false,
             "Load HCNW into an already-built net with matching architecture.")

        // ── Sizing ──
        .def_property_readonly("start_dim", &HCNN::GetStartDim)
        .def_property_readonly("start_n", &HCNN::GetStartN)
        .def_property_readonly("current_dim", &HCNN::GetCurrentDim)
        .def_property_readonly("input_channels", &HCNN::GetInputChannels)
        .def_property_readonly("num_outputs", &HCNN::GetNumOutputs)
        .def_property_readonly("num_conv", &HCNN::GetNumConv)
        .def_property_readonly("num_pool", &HCNN::GetNumPool)
        .def_property_readonly("task_type", &HCNN::GetTaskType)
        .def_property_readonly("optimizer_type", &HCNN::GetOptimizerType)
        .def_property_readonly("weights_initialized", &HCNN::WeightsInitialized)
        ;

    // ── Train helpers: metrics + cosine LR (optional product) ──
    m.def(
        "cosine_lr",
        &cosine_lr,
        py::arg("lr_max"), py::arg("lr_min"), py::arg("epoch"), py::arg("num_epochs"),
        "Cosine anneal: epoch 0 -> lr_max, last epoch -> lr_min.");

    py::class_<HCNNClassEval>(m, "ClassEval")
        .def_readonly("loss", &HCNNClassEval::loss)
        .def_readonly("accuracy", &HCNNClassEval::accuracy)
        .def_readonly("correct", &HCNNClassEval::correct)
        .def_readonly("count", &HCNNClassEval::count)
        .def("__repr__", [](const HCNNClassEval& r) {
            return "ClassEval(loss=" + std::to_string(r.loss)
                + ", accuracy=" + std::to_string(r.accuracy)
                + ", correct=" + std::to_string(r.correct)
                + "/" + std::to_string(r.count) + ")";
        });

    py::class_<HCNNRegEval>(m, "RegEval")
        .def_readonly("mse", &HCNNRegEval::mse)
        .def_readonly("target_var", &HCNNRegEval::target_var)
        .def_readonly("count", &HCNNRegEval::count)
        .def_property_readonly("r2", &HCNNRegEval::r2)
        .def("__repr__", [](const HCNNRegEval& r) {
            return "RegEval(mse=" + std::to_string(r.mse)
                + ", r2=" + std::to_string(r.r2())
                + ", count=" + std::to_string(r.count) + ")";
        });

    m.def(
        "evaluate_classification",
        [](HCNN& net, FloatArray flat_inputs, int input_length, IntArray targets,
           int count) {
            auto in = req_float(flat_inputs);
            auto y = req_int(targets);
            const size_t need =
                static_cast<size_t>(count) * static_cast<size_t>(input_length);
            if (in.size != need)
                throw std::invalid_argument(
                    "evaluate_classification: flat_inputs size mismatch");
            if (static_cast<int>(y.size) != count)
                throw std::invalid_argument(
                    "evaluate_classification: targets length must equal count");
            HCNNClassEval r;
            {
                py::gil_scoped_release release;
                r = evaluate_classification(net, in.ptr, input_length, y.ptr,
                                            count);
            }
            return r;
        },
        py::arg("net"), py::arg("flat_inputs"), py::arg("input_length"),
        py::arg("targets"), py::arg("count"),
        "Mean softmax CE + accuracy percent over a flat batch.");

    m.def(
        "evaluate_regression",
        [](HCNN& net, FloatArray flat_inputs, int input_length,
           FloatArray flat_targets, int count, int num_outputs) {
            auto in = req_float(flat_inputs);
            auto t = req_float(flat_targets);
            const size_t need_in =
                static_cast<size_t>(count) * static_cast<size_t>(input_length);
            if (in.size != need_in)
                throw std::invalid_argument(
                    "evaluate_regression: flat_inputs size mismatch");
            const int K = num_outputs > 0 ? num_outputs : net.GetNumOutputs();
            const size_t need_t =
                static_cast<size_t>(count) * static_cast<size_t>(K);
            if (t.size != need_t)
                throw std::invalid_argument(
                    "evaluate_regression: flat_targets size mismatch");
            HCNNRegEval r;
            {
                py::gil_scoped_release release;
                r = evaluate_regression(net, in.ptr, input_length, t.ptr, count,
                                        num_outputs);
            }
            return r;
        },
        py::arg("net"), py::arg("flat_inputs"), py::arg("input_length"),
        py::arg("flat_targets"), py::arg("count"),
        py::arg("num_outputs") = 0,
        "Mean MSE (+ target variance / R^2) over a flat batch.");

    // ── Spatial embed (optional product) ──
    py::enum_<HCNNSpatialEmbedMode>(m, "SpatialEmbedMode")
        .value("RowMajorPad", HCNNSpatialEmbedMode::RowMajorPad)
        .value("ResizeToFit", HCNNSpatialEmbedMode::ResizeToFit)
        .value("DualPlaneResize", HCNNSpatialEmbedMode::DualPlaneResize);

    py::class_<HCNNSpatialEmbedPlan>(m, "SpatialEmbedPlan")
        .def_readonly("dim", &HCNNSpatialEmbedPlan::dim)
        .def_readonly("N", &HCNNSpatialEmbedPlan::N)
        .def_readonly("height_in", &HCNNSpatialEmbedPlan::height_in)
        .def_readonly("width_in", &HCNNSpatialEmbedPlan::width_in)
        .def_readonly("plane_side", &HCNNSpatialEmbedPlan::plane_side)
        .def_readonly("pattern_length", &HCNNSpatialEmbedPlan::pattern_length)
        .def_readonly("mode", &HCNNSpatialEmbedPlan::mode)
        .def("__repr__", [](const HCNNSpatialEmbedPlan& p) {
            return "SpatialEmbedPlan(N=" + std::to_string(p.N)
                + ", pattern_length=" + std::to_string(p.pattern_length)
                + ", plane_side=" + std::to_string(p.plane_side) + ")";
        });

    py::class_<HCNNSpatialEmbedder>(m, "_SpatialEmbedder")
        .def(py::init([](int dim, HCNNSpatialEmbedMode mode, float pad_value,
                         int plane_side) {
                 HCNNSpatialEmbedConfig cfg;
                 cfg.dim = dim;
                 cfg.mode = mode;
                 cfg.pad_value = pad_value;
                 cfg.plane_side = plane_side;
                 return std::make_unique<HCNNSpatialEmbedder>(cfg);
             }),
             py::arg("dim") = 10,
             py::arg("mode") = HCNNSpatialEmbedMode::RowMajorPad,
             py::arg("pad_value") = 0.0f,
             py::arg("plane_side") = 0)

        .def_property_readonly("capacity", &HCNNSpatialEmbedder::capacity)
        .def_property_readonly(
            "dim",
            [](const HCNNSpatialEmbedder& self) { return self.config().dim; })
        .def_property_readonly(
            "mode",
            [](const HCNNSpatialEmbedder& self) { return self.config().mode; })
        .def_property_readonly(
            "pad_value",
            [](const HCNNSpatialEmbedder& self) {
                return self.config().pad_value;
            })
        .def_property_readonly(
            "plane_side",
            [](const HCNNSpatialEmbedder& self) {
                return self.config().plane_side;
            })

        .def(
            "plan",
            [](const HCNNSpatialEmbedder& self, int height, int width) {
                return self.plan(height, width);
            },
            py::arg("height"), py::arg("width"))

        .def(
            "embed",
            [](const HCNNSpatialEmbedder& self, FloatArray image, int height,
               int width) {
                auto in = req_float(image);
                const size_t need =
                    static_cast<size_t>(height) * static_cast<size_t>(width);
                if (in.size != need)
                    throw std::invalid_argument(
                        "embed: image size must equal height * width");
                const int N = self.capacity();
                py::array_t<float> out(N);
                {
                    py::gil_scoped_release release;
                    self.embed(in.ptr, height, width, out.mutable_data());
                }
                return out;
            },
            py::arg("image"), py::arg("height"), py::arg("width"),
            "Embed one HxW image to length-N (full capacity).")

        .def(
            "embed_batch",
            [](const HCNNSpatialEmbedder& self, FloatArray images, int batch,
               int height, int width) {
                auto in = req_float(images);
                const size_t need = static_cast<size_t>(batch)
                                    * static_cast<size_t>(height)
                                    * static_cast<size_t>(width);
                if (in.size != need)
                    throw std::invalid_argument(
                        "embed_batch: images size must equal batch*height*width");
                const int N = self.capacity();
                py::array_t<float> out({batch, N});
                {
                    py::gil_scoped_release release;
                    self.embed_batch(in.ptr, batch, height, width,
                                     out.mutable_data());
                }
                return out;
            },
            py::arg("images"), py::arg("batch"), py::arg("height"),
            py::arg("width"),
            "Embed batch of HxW images -> (batch, N).")

        .def_static("max_square_side", &HCNNSpatialEmbedder::max_square_side,
                    py::arg("N"))
        .def_static("max_dual_plane_side",
                    &HCNNSpatialEmbedder::max_dual_plane_side, py::arg("N"));

    // ── Spatial aug (optional product) ──
    py::class_<HCNNSpatialAugmenter>(m, "_SpatialAugmenter")
        .def(py::init([](float rot_deg_max, float scale_min, float scale_max,
                         int shift_max, float shear_x_max, float shear_y_max,
                         float elastic_alpha, float elastic_sigma,
                         float noise_sigma, float value_min, float value_max,
                         float border_value, bool enabled) {
                 HCNNSpatialAugConfig cfg;
                 cfg.rot_deg_max = rot_deg_max;
                 cfg.scale_min = scale_min;
                 cfg.scale_max = scale_max;
                 cfg.shift_max = shift_max;
                 cfg.shear_x_max = shear_x_max;
                 cfg.shear_y_max = shear_y_max;
                 cfg.elastic_alpha = elastic_alpha;
                 cfg.elastic_sigma = elastic_sigma;
                 cfg.noise_sigma = noise_sigma;
                 cfg.value_min = value_min;
                 cfg.value_max = value_max;
                 cfg.border_value = border_value;
                 cfg.enabled = enabled;
                 return std::make_unique<HCNNSpatialAugmenter>(cfg);
             }),
             py::arg("rot_deg_max") = 0.0f,
             py::arg("scale_min") = 1.0f,
             py::arg("scale_max") = 1.0f,
             py::arg("shift_max") = 0,
             py::arg("shear_x_max") = 0.0f,
             py::arg("shear_y_max") = 0.0f,
             py::arg("elastic_alpha") = 0.0f,
             py::arg("elastic_sigma") = 0.0f,
             py::arg("noise_sigma") = 0.0f,
             py::arg("value_min") = -1.0f,
             py::arg("value_max") = 1.0f,
             py::arg("border_value") = 0.0f,
             py::arg("enabled") = true)

        .def(
            "apply",
            [](const HCNNSpatialAugmenter& self, FloatArray image, int height,
               int width, unsigned seed) {
                auto in = req_float(image);
                const size_t need =
                    static_cast<size_t>(height) * static_cast<size_t>(width);
                if (in.size != need)
                    throw std::invalid_argument(
                        "apply: image size must equal height * width");
                // Always write a separate out buffer (safe for geometry warps).
                py::array_t<float> out(static_cast<py::ssize_t>(need));
                std::mt19937 rng(seed);
                {
                    py::gil_scoped_release release;
                    self.apply(in.ptr, out.mutable_data(), height, width, rng);
                }
                return out;
            },
            py::arg("image"), py::arg("height"), py::arg("width"),
            py::arg("seed") = 0u,
            "Augment one HxW image; seed drives std::mt19937.")

        .def(
            "apply_batch",
            [](const HCNNSpatialAugmenter& self, FloatArray images, int batch,
               int height, int width, unsigned seed) {
                auto in = req_float(images);
                const size_t plane =
                    static_cast<size_t>(height) * static_cast<size_t>(width);
                const size_t need = static_cast<size_t>(batch) * plane;
                if (in.size != need)
                    throw std::invalid_argument(
                        "apply_batch: images size must equal batch*height*width");
                py::array_t<float> out({batch, height, width});
                std::mt19937 rng(seed);
                {
                    py::gil_scoped_release release;
                    self.apply_batch(in.ptr, out.mutable_data(), batch, height,
                                     width, rng);
                }
                return out;
            },
            py::arg("images"), py::arg("batch"), py::arg("height"),
            py::arg("width"), py::arg("seed") = 0u);
}
