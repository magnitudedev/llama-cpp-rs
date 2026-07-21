#include "wrapper_common_fit.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <stdint.h>
#include <tuple>
#include <utility>
#include <vector>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/fit.h"
#include "llama.cpp/include/llama.h"
#include "llama.cpp/src/llama-ext.h"
#include "llama.cpp/src/llama-model.h"
#include "wrapper_utils.h"

// The legacy llama_rs_fit_params ABI returns the upstream numeric status and
// the structured report stores it in the stable bridge enum. Fail the pinned
// native build if those deliberately mirrored values ever diverge.
static_assert(
    static_cast<int>(LLAMA_RS_FIT_STATUS_SUCCESS) ==
    static_cast<int>(COMMON_PARAMS_FIT_STATUS_SUCCESS));
static_assert(
    static_cast<int>(LLAMA_RS_FIT_STATUS_FAILURE) ==
    static_cast<int>(COMMON_PARAMS_FIT_STATUS_FAILURE));
static_assert(
    static_cast<int>(LLAMA_RS_FIT_STATUS_ERROR) ==
    static_cast<int>(COMMON_PARAMS_FIT_STATUS_ERROR));

static std::string llama_rs_fit_string(const char * value);
static std::string llama_rs_fit_backend_name(ggml_backend_dev_t device);
static std::string llama_rs_fit_device_id(ggml_backend_dev_t device);

extern "C" int32_t llama_rs_common_default_math_threads(void) {
    return common_cpu_get_num_math();
}

static constexpr const char * LLAMA_RS_FIT_CALIBRATION_METHOD =
    "llama-native-ggml-decode-calibration-v1";
static constexpr const char * LLAMA_RS_FIT_DECODE_WORKLOAD_METHOD =
    "llama-native-decode-workload-v1";

struct llama_rs_fit_calibration_metric_storage {
    int32_t backend_type = 0;
    std::string backend;
    std::string device_id;
    int32_t tensor_type = 0;
    bool routed = false;
    double bytes_per_second = 0.0;
    double launch_microseconds = 0.0;
    double relative_spread = 0.0;
};

struct llama_rs_fit_calibration {
    int64_t elapsed_microseconds = 0;
    std::vector<llama_rs_fit_calibration_metric_storage> metrics;
};

struct llama_rs_fit_tensor_workload_storage {
    std::string name;
    int32_t backend_type = 0;
    std::string backend;
    std::string device_id;
    int32_t tensor_type = 0;
    enum llama_rs_fit_tensor_workload_kind kind = LLAMA_RS_FIT_TENSOR_ALWAYS_ACTIVE;
    uint64_t stored_bytes = 0;
    uint64_t operation_bytes = 0;
};

struct llama_rs_fit_kv_layer_workload_storage {
    uint32_t layer = 0;
    int32_t backend_type = 0;
    std::string backend;
    std::string device_id;
    int32_t key_type = 0;
    int32_t value_type = 0;
    uint64_t key_bytes_per_token = 0;
    uint64_t value_bytes_per_token = 0;
    uint32_t sliding_window_tokens = 0;
    bool recurrent = false;
};

struct llama_rs_fit_decode_workload_storage {
    bool available = false;
    std::string method = LLAMA_RS_FIT_DECODE_WORKLOAD_METHOD;
    std::string unavailable_reason = "decode workload was not requested";
    uint32_t expert_count = 0;
    uint32_t expert_used_count = 0;
    bool hybrid_model = false;
    bool recurrent_model = false;
    std::vector<llama_rs_fit_tensor_workload_storage> tensors;
    std::vector<llama_rs_fit_kv_layer_workload_storage> kv_layers;
};

struct llama_rs_fit_measurement_device {
    enum llama_rs_fit_device_kind kind = LLAMA_RS_FIT_DEVICE_ACCELERATOR;
    ggml_backend_dev_t device = nullptr;
    int32_t backend_type = 0;
    std::string backend;
    std::string device_id;
    std::string name;
    std::string description;
    struct llama_rs_fit_memory memory = {};
};

struct llama_rs_fit_measurement {
    bool available = false;
    std::string error;
    uint32_t model_layer_count = 0;
    uint32_t model_context_tokens = 0;
    uint32_t model_expert_count = 0;
    uint64_t model_tensor_bytes = 0;
    std::vector<llama_rs_fit_measurement_device> devices;
    struct llama_rs_fit_decode_workload_storage decode_workload;
};

static bool llama_rs_fit_valid_rate(double value) {
    return std::isfinite(value) && value > 0.0;
}

static bool llama_rs_fit_is_routed_expert_tensor(const ggml_tensor * tensor) {
    const char * name = tensor ? ggml_get_name(tensor) : nullptr;
    return name && std::strstr(name, "_exps") != nullptr;
}

static ggml_backend_dev_t llama_rs_fit_tensor_device(const ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) {
        return nullptr;
    }
    const auto buft = ggml_backend_buffer_get_type(tensor->buffer);
    if (ggml_backend_buft_is_host(buft)) {
        return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    return ggml_backend_buft_get_device(buft);
}

static struct llama_rs_fit_decode_workload_storage llama_rs_fit_extract_decode_workload(
    const llama_model * model,
    const llama_context_params * cparams) {
    llama_rs_fit_decode_workload_storage result;
    result.expert_count = model->hparams.n_expert;
    result.expert_used_count = model->hparams.n_expert_used;
    result.hybrid_model = llama_model_is_hybrid(model);
    result.recurrent_model = llama_model_is_recurrent(model);

    const auto & tensor_map = llama_internal_get_tensor_map(model);
    result.tensors.reserve(tensor_map.size());
    for (const auto & entry : tensor_map) {
        const ggml_tensor * tensor = entry.second;
        if (!tensor || ggml_nbytes(tensor) == 0) {
            continue;
        }
        const auto device = llama_rs_fit_tensor_device(tensor);
        if (!device) {
            result.unavailable_reason = "a model tensor has no native buffer device";
            return result;
        }
        llama_rs_fit_tensor_workload_storage output;
        output.name = entry.first;
        output.backend_type = static_cast<int32_t>(ggml_backend_dev_type(device));
        output.backend = llama_rs_fit_backend_name(device);
        output.device_id = llama_rs_fit_device_id(device);
        output.tensor_type = static_cast<int32_t>(tensor->type);
        output.stored_bytes = ggml_nbytes(tensor);
        output.operation_bytes = output.stored_bytes;
        if (llama_rs_fit_is_routed_expert_tensor(tensor)) {
            output.kind = LLAMA_RS_FIT_TENSOR_ROUTED_EXPERT;
        } else if ((tensor == model->tok_embd && tensor != model->output) ||
                   tensor == model->per_layer_tok_embd || tensor == model->pos_embd ||
                   tensor == model->type_embd) {
            output.kind = LLAMA_RS_FIT_TENSOR_ROW_LOOKUP;
            output.operation_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
        }
        result.tensors.push_back(std::move(output));
    }
    if (result.tensors.empty()) {
        result.unavailable_reason = "native tensor workload is empty";
        return result;
    }

    const auto & hparams = model->hparams;
    result.kv_layers.reserve(hparams.n_layer());
    for (uint32_t layer = 0; layer < hparams.n_layer(); ++layer) {
        const auto device = cparams->offload_kqv
            ? model->dev_layer(layer)
            : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!device) {
            result.unavailable_reason = "a KV layer has no native buffer device";
            result.tensors.clear();
            return result;
        }
        llama_rs_fit_kv_layer_workload_storage output;
        output.layer = layer;
        output.backend_type = static_cast<int32_t>(ggml_backend_dev_type(device));
        output.backend = llama_rs_fit_backend_name(device);
        output.device_id = llama_rs_fit_device_id(device);
        output.key_type = static_cast<int32_t>(cparams->type_k);
        output.value_type = static_cast<int32_t>(cparams->type_v);
        output.key_bytes_per_token = ggml_row_size(
            cparams->type_k, hparams.n_embd_k_gqa(layer));
        output.value_bytes_per_token = ggml_row_size(
            cparams->type_v, hparams.n_embd_v_gqa(layer));
        output.sliding_window_tokens = hparams.n_swa > 0 && hparams.is_swa(layer)
            ? hparams.n_swa
            : 0;
        output.recurrent = hparams.is_recr(layer);
        result.kv_layers.push_back(std::move(output));
    }

    result.available = true;
    result.unavailable_reason.clear();
    return result;
}

static bool llama_rs_fit_calibrate_metric(
    ggml_backend_t backend,
    ggml_backend_dev_t device,
    ggml_type tensor_type,
    bool routed,
    llama_rs_fit_calibration_metric_storage * out) {
    constexpr int64_t k = 4096;
    constexpr int64_t expert_count = 4;
    constexpr int64_t expert_used_count = 2;
    // Larger than the shared caches on the consumer systems this estimator targets, avoiding a
    // cache-resident rate that would not represent streaming real model weights.
    constexpr size_t target_weight_bytes = 128ULL * 1024 * 1024;
    constexpr int graph_repetitions = 1;
    constexpr int samples = 5;

    const int64_t block = ggml_blck_size(tensor_type);
    if (block <= 0 || k % block != 0) {
        return false;
    }
    const size_t row_bytes = ggml_row_size(tensor_type, k);
    if (row_bytes == 0) {
        return false;
    }
    const int64_t matrices = routed ? expert_count : 1;
    const int64_t rows = std::max<int64_t>(256,
        target_weight_bytes / row_bytes / matrices);

    if (!backend) {
        return false;
    }
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(64, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr context(ggml_init(params));
    if (!context) {
        return false;
    }

    ggml_tensor * weights = routed
        ? ggml_new_tensor_3d(context.get(), tensor_type, k, rows, expert_count)
        : ggml_new_tensor_2d(context.get(), tensor_type, k, rows);
    ggml_tensor * output = nullptr;
    ggml_tensor * ids_storage = nullptr;
    if (routed) {
        ids_storage = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, expert_count, 1);
        ggml_tensor * ids = ggml_view_2d(
            context.get(), ids_storage, expert_used_count, 1, ids_storage->nb[1], 0);
        ggml_tensor * activation = ggml_new_tensor_3d(
            context.get(), GGML_TYPE_F32, k, expert_used_count, 1);
        output = ggml_mul_mat_id(context.get(), weights, activation, ids);
    } else {
        ggml_tensor * activation = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, k, 1);
        output = ggml_mul_mat(context.get(), weights, activation);
    }
    if (!output || !ggml_backend_supports_op(backend, output)) {
        return false;
    }

    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) {
        return false;
    }
    if (weights->buffer) {
        ggml_backend_buffer_set_usage(weights->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    if (ids_storage) {
        const int32_t ids[expert_count] = {0, 1, 0, 0};
        ggml_backend_tensor_set(ids_storage, ids, 0, sizeof(ids));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    for (int index = 1; index < graph_repetitions; ++index) {
        ggml_graph_add_node(graph, output);
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    ggml_backend_synchronize(backend);

    std::vector<double> rates;
    rates.reserve(samples);
    const uint64_t active_weight_bytes = routed
        ? ggml_nbytes(weights) * expert_used_count / expert_count
        : ggml_nbytes(weights);
    for (int sample = 0; sample < samples; ++sample) {
        const int64_t started_at = llama_time_us();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_synchronize(backend);
        const int64_t elapsed = llama_time_us() - started_at;
        if (elapsed <= 0) {
            return false;
        }
        rates.push_back(
            static_cast<double>(active_weight_bytes) * graph_repetitions * 1'000'000.0 /
            elapsed);
    }
    std::sort(rates.begin(), rates.end());
    const double median = rates[rates.size() / 2];
    if (!llama_rs_fit_valid_rate(median)) {
        return false;
    }

    out->backend_type = static_cast<int32_t>(ggml_backend_dev_type(device));
    out->backend = llama_rs_fit_backend_name(device);
    out->device_id = llama_rs_fit_device_id(device);
    out->tensor_type = static_cast<int32_t>(tensor_type);
    out->routed = routed;
    out->bytes_per_second = median;
    out->launch_microseconds = 0.0;
    out->relative_spread = (rates.back() - rates.front()) / median;
    return true;
}

extern "C" llama_rs_status llama_rs_fit_calibration_create(
    struct llama_rs_fit_calibration ** out_calibration,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_calibration) {
        *out_calibration = nullptr;
    }
    if (!out_calibration) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "fit calibration output must not be null");
    }
    try {
        const int64_t started_at = llama_time_us();
        auto calibration = std::make_unique<llama_rs_fit_calibration>();
        const ggml_type types[] = {
            GGML_TYPE_F32,
            GGML_TYPE_F16,
            GGML_TYPE_BF16,
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q5_1,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_Q8_0,
            GGML_TYPE_IQ4_NL,
            GGML_TYPE_MXFP4,
        };
        for (size_t device_index = 0; device_index < ggml_backend_dev_count(); ++device_index) {
            const auto device = ggml_backend_dev_get(device_index);
            if (!device) {
                continue;
            }
            ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
            if (!backend) {
                continue;
            }
            for (const ggml_type type : types) {
                for (const bool routed : {false, true}) {
                    llama_rs_fit_calibration_metric_storage metric;
                    if (llama_rs_fit_calibrate_metric(
                            backend.get(), device, type, routed, &metric)) {
                        calibration->metrics.push_back(std::move(metric));
                    }
                }
            }
        }
        if (calibration->metrics.empty()) {
            throw std::runtime_error("no enabled backend accepted a calibration operation");
        }
        calibration->elapsed_microseconds = llama_time_us() - started_at;
        *out_calibration = calibration.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" void llama_rs_fit_calibration_free(
    struct llama_rs_fit_calibration * calibration) {
    delete calibration;
}

extern "C" int64_t llama_rs_fit_calibration_elapsed_microseconds(
    const struct llama_rs_fit_calibration * calibration) {
    return calibration ? calibration->elapsed_microseconds : 0;
}

extern "C" const char * llama_rs_fit_calibration_method(
    const struct llama_rs_fit_calibration * calibration) {
    return calibration ? LLAMA_RS_FIT_CALIBRATION_METHOD : nullptr;
}

extern "C" size_t llama_rs_fit_calibration_metric_count(
    const struct llama_rs_fit_calibration * calibration) {
    return calibration ? calibration->metrics.size() : 0;
}

extern "C" bool llama_rs_fit_calibration_get_metric(
    const struct llama_rs_fit_calibration * calibration,
    size_t index,
    struct llama_rs_fit_calibration_metric * out_metric) {
    if (!calibration || !out_metric || index >= calibration->metrics.size()) {
        return false;
    }
    const auto & source = calibration->metrics[index];
    *out_metric = {};
    out_metric->backend_type = source.backend_type;
    out_metric->backend = source.backend.c_str();
    out_metric->device_id = source.device_id.empty() ? nullptr : source.device_id.c_str();
    out_metric->tensor_type = source.tensor_type;
    out_metric->routed = source.routed;
    out_metric->bytes_per_second = source.bytes_per_second;
    out_metric->launch_microseconds = source.launch_microseconds;
    out_metric->relative_spread = source.relative_spread;
    return true;
}

static struct llama_rs_fit_measurement llama_rs_fit_measure_loaded(
    const struct llama_model * model,
    const struct llama_context_params * cparams,
    bool capture_decode_workload) {
    struct llama_rs_fit_measurement result;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(const_cast<llama_model *>(model), *cparams), llama_free);
    if (!context) {
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t device_count = llama_model_n_devices(model);
    const auto memory = llama_get_memory_breakdown(context.get());
    result.devices.reserve(device_count + 1);
    for (size_t index = 0; index < device_count; ++index) {
        const auto device = llama_model_get_device(model, index);
        if (!device) {
            throw std::runtime_error("llama.cpp returned a null model device");
        }
        llama_rs_fit_measurement_device value;
        value.kind = LLAMA_RS_FIT_DEVICE_ACCELERATOR;
        value.device = device;
        value.backend_type = static_cast<int32_t>(ggml_backend_dev_type(device));
        value.backend = llama_rs_fit_backend_name(device);
        value.device_id = llama_rs_fit_device_id(device);
        value.name = llama_rs_fit_string(ggml_backend_dev_name(device));
        value.description = llama_rs_fit_string(ggml_backend_dev_description(device));
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        value.memory.free_bytes = static_cast<int64_t>(free_bytes);
        value.memory.total_bytes = static_cast<int64_t>(total_bytes);
        result.devices.push_back(std::move(value));
    }

    llama_rs_fit_measurement_device host;
    host.kind = LLAMA_RS_FIT_DEVICE_HOST;
    const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    host.device = cpu;
    host.backend_type = static_cast<int32_t>(GGML_BACKEND_DEVICE_TYPE_CPU);
    host.backend = llama_rs_fit_backend_name(cpu);
    host.device_id = llama_rs_fit_device_id(cpu);
    host.name = cpu ? llama_rs_fit_string(ggml_backend_dev_name(cpu)) : "CPU";
    host.description = cpu ? llama_rs_fit_string(ggml_backend_dev_description(cpu)) : "Host memory";
    if (!cpu) {
        throw std::runtime_error("no CPU backend found");
    }
    size_t host_free_bytes = 0;
    size_t host_total_bytes = 0;
    ggml_backend_dev_memory(cpu, &host_free_bytes, &host_total_bytes);
    host.memory.free_bytes = static_cast<int64_t>(host_free_bytes);
    host.memory.total_bytes = static_cast<int64_t>(host_total_bytes);
    result.devices.push_back(std::move(host));

    for (const auto & [buffer_type, breakdown] : memory) {
        llama_rs_fit_measurement_device * target = nullptr;
        if (ggml_backend_buft_is_host(buffer_type)) {
            target = &result.devices.back();
        } else if (const auto device = ggml_backend_buft_get_device(buffer_type)) {
            const auto found = std::find_if(result.devices.begin(), result.devices.end(),
                [&](const auto & candidate) { return candidate.device == device; });
            if (found != result.devices.end()) {
                target = &*found;
            }
        }
        if (target) {
            target->memory.model_bytes += breakdown.model;
            target->memory.context_bytes += breakdown.context;
            target->memory.compute_bytes += breakdown.compute;
        }
    }

    result.model_layer_count = llama_model_n_layer(model);
    result.model_context_tokens = llama_model_n_ctx_train(model);
    result.model_expert_count = llama_model_n_expert(model);
    result.model_tensor_bytes = llama_model_size(model);
    if (capture_decode_workload) {
        result.decode_workload = llama_rs_fit_extract_decode_workload(model, cparams);
    }
    result.available = true;
    return result;
}

struct llama_rs_fit_device_storage {
    enum llama_rs_fit_device_kind kind = LLAMA_RS_FIT_DEVICE_ACCELERATOR;
    ggml_backend_dev_t device = nullptr;
    int32_t backend_type = 0;
    std::string backend;
    std::string device_id;
    std::string name;
    std::string description;
    bool initial_available = false;
    struct llama_rs_fit_memory initial = {};
    bool fitted_available = false;
    struct llama_rs_fit_memory fitted = {};
    bool margin_applies = false;
    uint64_t margin_bytes = 0;
};

struct llama_rs_fit_placement_storage {
    std::string pattern;
    std::string buffer_type;
    enum llama_rs_fit_placement_kind kind = LLAMA_RS_FIT_PLACEMENT_OTHER;
    int32_t device_index = -1;
    std::string device_name;
    std::string device_description;
};

struct llama_rs_fit_report {
    struct llama_rs_fit_summary summary = {};
    std::string initial_error;
    std::string fitted_error;
    std::vector<llama_rs_fit_device_storage> devices;
    std::vector<float> tensor_split;
    std::vector<llama_rs_fit_placement_storage> placements;
    struct llama_rs_fit_decode_workload_storage decode_workload;
};

struct llama_rs_context_other_guard {
    llama_context_params * params;
    llama_context * original;

    llama_rs_context_other_guard(llama_context_params * params, llama_context * original) :
        params(params), original(original) {}

    llama_rs_context_other_guard(const llama_rs_context_other_guard &) = delete;
    llama_rs_context_other_guard & operator=(const llama_rs_context_other_guard &) = delete;

    ~llama_rs_context_other_guard() {
        params->ctx_other = original;
    }
};

static std::string llama_rs_fit_string(const char * value) {
    return value ? value : "";
}

static std::string llama_rs_fit_backend_name(ggml_backend_dev_t device) {
    if (!device) {
        return "";
    }
    const auto registration = ggml_backend_dev_backend_reg(device);
    return registration ? llama_rs_fit_string(ggml_backend_reg_name(registration)) : "";
}

static std::string llama_rs_fit_device_id(ggml_backend_dev_t device) {
    if (!device) {
        return "";
    }
    struct ggml_backend_dev_props properties = {};
    ggml_backend_dev_get_props(device, &properties);
    return llama_rs_fit_string(properties.device_id);
}

static struct llama_rs_fit_measurement llama_rs_fit_measure(
    const char * path_model,
    const struct llama_model_params * mparams,
    const struct llama_context_params * cparams,
    enum ggml_log_level log_level) {
    struct llama_rs_fit_measurement result;
    try {
        std::vector<ggml_backend_dev_t> devices;
        const auto memory = common_get_device_memory_data(
            path_model,
            mparams,
            cparams,
            devices,
            result.model_layer_count,
            result.model_context_tokens,
        result.model_expert_count,
        log_level);
        if (memory.size() != devices.size() + 1) {
            throw std::runtime_error("llama.cpp returned an inconsistent device memory report");
        }
        result.devices.reserve(memory.size());

        for (size_t index = 0; index < devices.size(); ++index) {
            const auto device = devices[index];
            if (!device) {
                throw std::runtime_error("llama.cpp returned a null model device");
            }
            llama_rs_fit_measurement_device value;
            value.kind = LLAMA_RS_FIT_DEVICE_ACCELERATOR;
            value.device = device;
            value.backend_type = static_cast<int32_t>(ggml_backend_dev_type(device));
            value.backend = llama_rs_fit_backend_name(device);
            value.device_id = llama_rs_fit_device_id(device);
            value.name = llama_rs_fit_string(ggml_backend_dev_name(device));
            value.description = llama_rs_fit_string(ggml_backend_dev_description(device));
            value.memory.total_bytes = memory[index].total;
            value.memory.free_bytes = memory[index].free;
            value.memory.model_bytes = memory[index].model;
            value.memory.context_bytes = memory[index].context;
            value.memory.compute_bytes = memory[index].compute;
            result.devices.push_back(std::move(value));
        }

        const auto & host_memory = memory.back();
        llama_rs_fit_measurement_device host;
        host.kind = LLAMA_RS_FIT_DEVICE_HOST;
        const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        host.device = cpu;
        host.backend_type = static_cast<int32_t>(GGML_BACKEND_DEVICE_TYPE_CPU);
        host.backend = llama_rs_fit_backend_name(cpu);
        host.device_id = llama_rs_fit_device_id(cpu);
        host.name = cpu ? llama_rs_fit_string(ggml_backend_dev_name(cpu)) : "CPU";
        host.description = cpu
            ? llama_rs_fit_string(ggml_backend_dev_description(cpu))
            : "Host memory";
        host.memory.total_bytes = host_memory.total;
        host.memory.free_bytes = host_memory.free;
        host.memory.model_bytes = host_memory.model;
        host.memory.context_bytes = host_memory.context;
        host.memory.compute_bytes = host_memory.compute;
        result.devices.push_back(std::move(host));
        result.available = true;
    } catch (const std::exception & error) {
        result.error = error.what();
    } catch (...) {
        result.error = "unknown error while measuring llama.cpp memory";
    }

    return result;
}

static bool llama_rs_fit_same_device(
    const struct llama_rs_fit_measurement_device & left,
    const struct llama_rs_fit_measurement_device & right) {
    return left.kind == right.kind &&
        (left.kind == LLAMA_RS_FIT_DEVICE_HOST || left.device == right.device);
}

static const struct llama_rs_fit_measurement_device * llama_rs_fit_find_device(
    const struct llama_rs_fit_measurement & measurement,
    const struct llama_rs_fit_measurement_device & target) {
    const auto found = std::find_if(
        measurement.devices.begin(),
        measurement.devices.end(),
        [&](const auto & candidate) { return llama_rs_fit_same_device(candidate, target); });
    return found == measurement.devices.end() ? nullptr : &*found;
}

static uint32_t llama_rs_fit_resolve_context(uint32_t configured, uint32_t trained) {
    return configured == 0 ? trained : configured;
}

static uint32_t llama_rs_fit_resolve_gpu_layers(
    int32_t configured,
    uint32_t model_layer_count,
    size_t accelerator_count) {
    if (accelerator_count == 0) {
        return 0;
    }
    if (model_layer_count == std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("model offloadable layer count overflowed u32");
    }
    const uint32_t offloadable = model_layer_count + 1;
    if (configured < 0) {
        return offloadable;
    }
    return std::min(static_cast<uint32_t>(configured), offloadable);
}

static void llama_rs_fit_merge_devices(
    struct llama_rs_fit_report & report,
    const struct llama_rs_fit_measurement & initial,
    const struct llama_rs_fit_measurement & fitted,
    const size_t * margins) {
    std::vector<llama_rs_fit_measurement_device> identities;
    const auto append_accelerators = [&](const auto & measurement) {
        for (const auto & device : measurement.devices) {
            if (device.kind != LLAMA_RS_FIT_DEVICE_ACCELERATOR) {
                continue;
            }
            const bool present = std::any_of(
                identities.begin(),
                identities.end(),
                [&](const auto & current) { return llama_rs_fit_same_device(current, device); });
            if (!present) {
                identities.push_back(device);
            }
        }
    };
    // The final measurement order is the order used by the fitted tensor split
    // and buffer overrides. Seed identities from it so report indices preserve
    // that relationship; append only initial-only devices after it.
    append_accelerators(fitted);
    append_accelerators(initial);

    llama_rs_fit_measurement_device host;
    host.kind = LLAMA_RS_FIT_DEVICE_HOST;
    const auto initial_host = std::find_if(
        initial.devices.begin(), initial.devices.end(),
        [](const auto & device) { return device.kind == LLAMA_RS_FIT_DEVICE_HOST; });
    const auto fitted_host = std::find_if(
        fitted.devices.begin(), fitted.devices.end(),
        [](const auto & device) { return device.kind == LLAMA_RS_FIT_DEVICE_HOST; });
    if (fitted_host != fitted.devices.end()) {
        host = *fitted_host;
    } else if (initial_host != initial.devices.end()) {
        host = *initial_host;
    } else {
        host.name = "CPU";
        host.description = "Host memory";
        host.backend_type = static_cast<int32_t>(GGML_BACKEND_DEVICE_TYPE_CPU);
        host.backend = "CPU";
    }
    identities.push_back(std::move(host));

    const size_t accelerator_count = identities.size() - 1;
    report.devices.reserve(identities.size());
    for (size_t index = 0; index < identities.size(); ++index) {
        const auto & identity = identities[index];
        llama_rs_fit_device_storage output;
        output.kind = identity.kind;
        output.device = identity.device;
        output.backend_type = identity.backend_type;
        output.backend = identity.backend;
        output.device_id = identity.device_id;
        output.name = identity.name;
        output.description = identity.description;
        if (const auto value = llama_rs_fit_find_device(initial, identity)) {
            output.initial_available = true;
            output.initial = value->memory;
        }
        if (const auto value = llama_rs_fit_find_device(fitted, identity)) {
            output.fitted_available = true;
            output.fitted = value->memory;
        }
        output.margin_applies =
            identity.kind == LLAMA_RS_FIT_DEVICE_ACCELERATOR || accelerator_count == 0;
        if (output.margin_applies) {
            output.margin_bytes = margins[identity.kind == LLAMA_RS_FIT_DEVICE_HOST ? 0 : index];
        }
        report.devices.push_back(std::move(output));
    }
}

static void llama_rs_fit_capture_placement(
    struct llama_rs_fit_report & report,
    const struct llama_model_tensor_buft_override & source) {
    llama_rs_fit_placement_storage output;
    output.pattern = llama_rs_fit_string(source.pattern);
    output.buffer_type = source.buft
        ? llama_rs_fit_string(ggml_backend_buft_name(source.buft))
        : "";
    if (source.buft && ggml_backend_buft_is_host(source.buft)) {
        output.kind = LLAMA_RS_FIT_PLACEMENT_HOST;
    } else if (source.buft) {
        const auto device = ggml_backend_buft_get_device(source.buft);
        if (device) {
            output.kind = LLAMA_RS_FIT_PLACEMENT_DEVICE;
            output.device_name = llama_rs_fit_string(ggml_backend_dev_name(device));
            output.device_description = llama_rs_fit_string(ggml_backend_dev_description(device));
            for (size_t index = 0; index < report.devices.size(); ++index) {
                const auto & candidate = report.devices[index];
                if (candidate.kind == LLAMA_RS_FIT_DEVICE_ACCELERATOR &&
                    candidate.device == device) {
                    output.device_index = static_cast<int32_t>(index);
                    break;
                }
            }
        }
    }
    report.placements.push_back(std::move(output));
}

static llama_rs_status llama_rs_fit_report_create_impl(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    const char * target_path,
    const struct llama_model_params * target_mparams,
    const struct llama_context_params * target_cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    size_t margins_count,
    bool capture_decode_workload,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level,
    struct llama_rs_fit_report ** out_report,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_report) {
        *out_report = nullptr;
    }
    if (!path_model || !mparams || !cparams || !tensor_split ||
        !tensor_buft_overrides || !margins || !out_report) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "fit report arguments must not be null");
    }
    const bool has_linked_target = target_path || target_mparams || target_cparams;
    if (has_linked_target && (!target_path || !target_mparams || !target_cparams)) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "linked fit target arguments must be supplied together");
    }
    if (margins_count < llama_max_devices()) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "fit margins must contain at least llama_max_devices entries");
    }

    try {
        std::unique_ptr<llama_model, decltype(&llama_model_free)> linked_model(nullptr, llama_model_free);
        std::unique_ptr<llama_context, decltype(&llama_free)> linked_context(nullptr, llama_free);
        std::unique_ptr<llama_rs_context_other_guard> context_guard;
        if (has_linked_target) {
            llama_model_params linked_mparams = *target_mparams;
            linked_mparams.no_alloc = true;
            linked_mparams.use_mmap = false;
            linked_mparams.use_mlock = false;
            linked_model.reset(llama_model_load_from_file(target_path, linked_mparams));
            if (!linked_model) {
                throw std::runtime_error("failed to inspect linked fit target model");
            }
            llama_context_params linked_cparams = *target_cparams;
            linked_cparams.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
            linked_cparams.ctx_other = nullptr;
            linked_context.reset(llama_init_from_model(linked_model.get(), linked_cparams));
            if (!linked_context) {
                throw std::runtime_error("failed to construct linked fit target context");
            }
            context_guard = std::make_unique<llama_rs_context_other_guard>(
                cparams, cparams->ctx_other);
            cparams->ctx_other = linked_context.get();
        }

        const int64_t started_at = llama_time_us();
        auto report = std::make_unique<llama_rs_fit_report>();
        const uint32_t requested_context_tokens = cparams->n_ctx;
        const int32_t requested_gpu_layers = mparams->n_gpu_layers;

        const auto initial = llama_rs_fit_measure(path_model, mparams, cparams, log_level);
        const auto fit_status = common_fit_params(
            path_model,
            mparams,
            cparams,
            tensor_split,
            tensor_buft_overrides,
            margins,
            n_ctx_min,
            log_level);
        const auto fitted = llama_rs_fit_measure(path_model, mparams, cparams, log_level);

        report->initial_error = initial.error;
        report->fitted_error = fitted.error;
        llama_rs_fit_merge_devices(*report, initial, fitted, margins);

        const auto & hparams = fitted.available ? fitted : initial;
        const size_t accelerator_count = std::count_if(
            report->devices.begin(),
            report->devices.end(),
            [](const auto & device) { return device.kind == LLAMA_RS_FIT_DEVICE_ACCELERATOR; });

        report->summary.status = static_cast<llama_rs_fit_status>(fit_status);
        report->summary.requested_context_tokens = requested_context_tokens;
        report->summary.fitted_context_tokens = cparams->n_ctx;
        report->summary.resolved_requested_context_tokens = llama_rs_fit_resolve_context(
            requested_context_tokens, hparams.model_context_tokens);
        report->summary.resolved_fitted_context_tokens = llama_rs_fit_resolve_context(
            cparams->n_ctx, hparams.model_context_tokens);
        report->summary.requested_gpu_layers = requested_gpu_layers;
        report->summary.fitted_gpu_layers = mparams->n_gpu_layers;
        report->summary.resolved_requested_gpu_layers = llama_rs_fit_resolve_gpu_layers(
            requested_gpu_layers, hparams.model_layer_count, accelerator_count);
        report->summary.resolved_fitted_gpu_layers = llama_rs_fit_resolve_gpu_layers(
            mparams->n_gpu_layers, hparams.model_layer_count, accelerator_count);
        report->summary.model_layer_count = hparams.model_layer_count;
        report->summary.model_context_tokens = hparams.model_context_tokens;
        report->summary.model_expert_count = hparams.model_expert_count;
        report->summary.model_tensor_bytes = hparams.model_tensor_bytes;
        report->summary.accelerator_count = accelerator_count;
        report->summary.initial_measurement_available = initial.available;
        report->summary.fitted_measurement_available = fitted.available;
        report->summary.elapsed_microseconds = llama_time_us() - started_at;

        if (fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS && capture_decode_workload) {
            llama_model_params workload_mparams = *mparams;
            workload_mparams.no_alloc = true;
            workload_mparams.use_mmap = false;
            workload_mparams.use_mlock = false;
            std::unique_ptr<llama_model, decltype(&llama_model_free)> workload_model(
                llama_model_load_from_file(path_model, workload_mparams), llama_model_free);
            if (workload_model) {
                report->decode_workload = llama_rs_fit_extract_decode_workload(
                    workload_model.get(), cparams);
            } else {
                report->decode_workload.unavailable_reason =
                    "failed to reconstruct the fitted no-allocation model";
            }
        }

        // For an explicit caller split common_fit_params leaves mparams intact;
        // for an automatic split it points mparams at the supplied output
        // buffer. Report the effective values in either case.
        const float * effective_tensor_split = mparams->tensor_split
            ? mparams->tensor_split
            : tensor_split;
        report->tensor_split.assign(
            effective_tensor_split,
            effective_tensor_split + accelerator_count);
        const size_t max_overrides = llama_max_tensor_buft_overrides();
        for (size_t index = 0;
             index < max_overrides && tensor_buft_overrides[index].pattern;
             ++index) {
            llama_rs_fit_capture_placement(*report, tensor_buft_overrides[index]);
        }

        *out_report = report.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_fit_measure_reports_create(
    const char * path_model,
    const struct llama_model_params * mparams,
    const struct llama_context_params * cparams,
    size_t profile_count,
    const size_t * margins,
    size_t margins_count,
    bool capture_decode_workload,
    enum ggml_log_level,
    struct llama_rs_fit_report ** out_reports,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!path_model || !mparams || (!cparams && profile_count != 0) ||
        !margins || !out_reports || margins_count < llama_max_devices()) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "batch fit measurement arguments are invalid");
    }
    std::fill(out_reports, out_reports + profile_count, nullptr);
    try {
        llama_model_params loading = *mparams;
        loading.no_alloc = true;
        loading.use_mmap = false;
        loading.use_mlock = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(path_model, loading), llama_model_free);
        if (!model) {
            throw std::runtime_error("failed to load model");
        }

        std::vector<std::unique_ptr<llama_rs_fit_report>> reports;
        reports.reserve(profile_count);
        for (size_t profile = 0; profile < profile_count; ++profile) {
            const int64_t started_at = llama_time_us();
            const auto measurement = llama_rs_fit_measure_loaded(
                model.get(),
                &cparams[profile],
                capture_decode_workload);
            auto report = std::make_unique<llama_rs_fit_report>();
            const llama_rs_fit_measurement empty;
            llama_rs_fit_merge_devices(*report, measurement, empty, margins);
            const size_t accelerator_count = std::count_if(
                report->devices.begin(), report->devices.end(),
                [](const auto & device) { return device.kind == LLAMA_RS_FIT_DEVICE_ACCELERATOR; });
            report->summary.status = LLAMA_RS_FIT_STATUS_SUCCESS;
            report->summary.requested_context_tokens = cparams[profile].n_ctx;
            report->summary.fitted_context_tokens = cparams[profile].n_ctx;
            report->summary.resolved_requested_context_tokens = llama_rs_fit_resolve_context(
                cparams[profile].n_ctx, measurement.model_context_tokens);
            report->summary.resolved_fitted_context_tokens = report->summary.resolved_requested_context_tokens;
            report->summary.requested_gpu_layers = mparams->n_gpu_layers;
            report->summary.fitted_gpu_layers = mparams->n_gpu_layers;
            report->summary.resolved_requested_gpu_layers = llama_rs_fit_resolve_gpu_layers(
                mparams->n_gpu_layers, measurement.model_layer_count, accelerator_count);
            report->summary.resolved_fitted_gpu_layers = report->summary.resolved_requested_gpu_layers;
            report->summary.model_layer_count = measurement.model_layer_count;
            report->summary.model_context_tokens = measurement.model_context_tokens;
            report->summary.model_expert_count = measurement.model_expert_count;
            report->summary.model_tensor_bytes = measurement.model_tensor_bytes;
            report->summary.accelerator_count = accelerator_count;
            report->summary.initial_measurement_available = true;
            report->summary.fitted_measurement_available = false;
            report->summary.elapsed_microseconds = llama_time_us() - started_at;
            report->decode_workload = measurement.decode_workload;
            reports.push_back(std::move(report));
        }
        for (size_t profile = 0; profile < profile_count; ++profile) {
            out_reports[profile] = reports[profile].release();
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_fit_report_create(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    size_t margins_count,
    bool capture_decode_workload,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level,
    struct llama_rs_fit_report ** out_report,
    char ** out_error) {
    return llama_rs_fit_report_create_impl(
        path_model, mparams, cparams,
        nullptr, nullptr, nullptr,
        tensor_split, tensor_buft_overrides, margins, margins_count,
        capture_decode_workload,
        n_ctx_min, log_level, out_report, out_error);
}

extern "C" llama_rs_status llama_rs_fit_report_create_linked(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    const char * target_path,
    const struct llama_model_params * target_mparams,
    const struct llama_context_params * target_cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    size_t margins_count,
    bool capture_decode_workload,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level,
    struct llama_rs_fit_report ** out_report,
    char ** out_error) {
    return llama_rs_fit_report_create_impl(
        path_model, mparams, cparams,
        target_path, target_mparams, target_cparams,
        tensor_split, tensor_buft_overrides, margins, margins_count,
        capture_decode_workload,
        n_ctx_min, log_level, out_report, out_error);
}

extern "C" void llama_rs_fit_report_free(struct llama_rs_fit_report * report) {
    delete report;
}

extern "C" bool llama_rs_fit_report_get_summary(
    const struct llama_rs_fit_report * report,
    struct llama_rs_fit_summary * out_summary) {
    if (!report || !out_summary) {
        return false;
    }
    *out_summary = report->summary;
    return true;
}

extern "C" const char * llama_rs_fit_report_initial_error(
    const struct llama_rs_fit_report * report) {
    return report && !report->initial_error.empty() ? report->initial_error.c_str() : nullptr;
}

extern "C" const char * llama_rs_fit_report_fitted_error(
    const struct llama_rs_fit_report * report) {
    return report && !report->fitted_error.empty() ? report->fitted_error.c_str() : nullptr;
}

extern "C" size_t llama_rs_fit_report_device_count(
    const struct llama_rs_fit_report * report) {
    return report ? report->devices.size() : 0;
}

extern "C" bool llama_rs_fit_report_get_device(
    const struct llama_rs_fit_report * report,
    size_t index,
    struct llama_rs_fit_device * out_device) {
    if (!report || !out_device || index >= report->devices.size()) {
        return false;
    }
    const auto & source = report->devices[index];
    *out_device = {};
    out_device->index = index;
    out_device->kind = source.kind;
    out_device->backend_type = source.backend_type;
    out_device->backend = source.backend.c_str();
    out_device->device_id =
        source.device_id.empty() ? nullptr : source.device_id.c_str();
    out_device->name = source.name.c_str();
    out_device->description = source.description.c_str();
    out_device->initial_available = source.initial_available;
    out_device->initial = source.initial;
    out_device->fitted_available = source.fitted_available;
    out_device->fitted = source.fitted;
    out_device->margin_applies = source.margin_applies;
    out_device->margin_bytes = source.margin_bytes;
    return true;
}

extern "C" size_t llama_rs_fit_report_tensor_split_count(
    const struct llama_rs_fit_report * report) {
    return report ? report->tensor_split.size() : 0;
}

extern "C" bool llama_rs_fit_report_get_tensor_split(
    const struct llama_rs_fit_report * report,
    size_t index,
    float * out_value) {
    if (!report || !out_value || index >= report->tensor_split.size()) {
        return false;
    }
    *out_value = report->tensor_split[index];
    return true;
}

extern "C" size_t llama_rs_fit_report_placement_count(
    const struct llama_rs_fit_report * report) {
    return report ? report->placements.size() : 0;
}

extern "C" bool llama_rs_fit_report_get_placement(
    const struct llama_rs_fit_report * report,
    size_t index,
    struct llama_rs_fit_placement * out_placement) {
    if (!report || !out_placement || index >= report->placements.size()) {
        return false;
    }
    const auto & source = report->placements[index];
    *out_placement = {};
    out_placement->pattern = source.pattern.c_str();
    out_placement->buffer_type = source.buffer_type.c_str();
    out_placement->kind = source.kind;
    out_placement->device_index = source.device_index;
    out_placement->device_name = source.device_name.empty() ? nullptr : source.device_name.c_str();
    out_placement->device_description = source.device_description.empty()
        ? nullptr
        : source.device_description.c_str();
    return true;
}

extern "C" bool llama_rs_fit_report_get_decode_workload_summary(
    const struct llama_rs_fit_report * report,
    struct llama_rs_fit_decode_workload_summary * out_summary) {
    if (!report || !out_summary) {
        return false;
    }
    const auto & source = report->decode_workload;
    *out_summary = {};
    out_summary->available = source.available;
    out_summary->method = source.method.c_str();
    out_summary->unavailable_reason = source.unavailable_reason.empty()
        ? nullptr
        : source.unavailable_reason.c_str();
    out_summary->expert_count = source.expert_count;
    out_summary->expert_used_count = source.expert_used_count;
    out_summary->hybrid_model = source.hybrid_model;
    out_summary->recurrent_model = source.recurrent_model;
    return true;
}

extern "C" size_t llama_rs_fit_report_tensor_workload_count(
    const struct llama_rs_fit_report * report) {
    return report ? report->decode_workload.tensors.size() : 0;
}

extern "C" bool llama_rs_fit_report_get_tensor_workload(
    const struct llama_rs_fit_report * report,
    size_t index,
    struct llama_rs_fit_tensor_workload * out_tensor) {
    if (!report || !out_tensor || index >= report->decode_workload.tensors.size()) {
        return false;
    }
    const auto & source = report->decode_workload.tensors[index];
    *out_tensor = {};
    out_tensor->name = source.name.c_str();
    out_tensor->backend_type = source.backend_type;
    out_tensor->backend = source.backend.c_str();
    out_tensor->device_id = source.device_id.empty() ? nullptr : source.device_id.c_str();
    out_tensor->tensor_type = source.tensor_type;
    out_tensor->kind = source.kind;
    out_tensor->stored_bytes = source.stored_bytes;
    out_tensor->operation_bytes = source.operation_bytes;
    return true;
}

extern "C" size_t llama_rs_fit_report_kv_layer_workload_count(
    const struct llama_rs_fit_report * report) {
    return report ? report->decode_workload.kv_layers.size() : 0;
}

extern "C" bool llama_rs_fit_report_get_kv_layer_workload(
    const struct llama_rs_fit_report * report,
    size_t index,
    struct llama_rs_fit_kv_layer_workload * out_layer) {
    if (!report || !out_layer || index >= report->decode_workload.kv_layers.size()) {
        return false;
    }
    const auto & source = report->decode_workload.kv_layers[index];
    *out_layer = {};
    out_layer->layer = source.layer;
    out_layer->backend_type = source.backend_type;
    out_layer->backend = source.backend.c_str();
    out_layer->device_id = source.device_id.empty() ? nullptr : source.device_id.c_str();
    out_layer->key_type = source.key_type;
    out_layer->value_type = source.value_type;
    out_layer->key_bytes_per_token = source.key_bytes_per_token;
    out_layer->value_bytes_per_token = source.value_bytes_per_token;
    out_layer->sliding_window_tokens = source.sliding_window_tokens;
    out_layer->recurrent = source.recurrent;
    return true;
}

// Thin pass-through to llama.cpp's common_fit_params (a C++ symbol in libcommon).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
extern "C" int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level) {
    // Preserve the legacy three-value ABI while ensuring that neither a
    // llama-common exception nor an allocation failure can unwind through C
    // and Rust frames. Callers that need the diagnostic message use the
    // structured llama_rs_fit_report_create API above.
    try {
        return static_cast<int>(common_fit_params(
            path_model,
            mparams,
            cparams,
            tensor_split,
            tensor_buft_overrides,
            margins,
            n_ctx_min,
            log_level));
    } catch (...) {
        return static_cast<int>(COMMON_PARAMS_FIT_STATUS_ERROR);
    }
}

extern "C" void llama_rs_memory_breakdown_print(const struct llama_context * ctx) {
    try {
        common_memory_breakdown_print(ctx);
    } catch (...) {
        // Legacy diagnostic ABI has no error channel; never unwind through C.
    }
}
