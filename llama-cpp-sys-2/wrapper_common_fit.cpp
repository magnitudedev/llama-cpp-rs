#include "wrapper_common_fit.h"

#include <algorithm>
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

#include "llama.cpp/common/fit.h"
#include "llama.cpp/include/llama.h"
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

struct llama_rs_fit_measurement_device {
    enum llama_rs_fit_device_kind kind = LLAMA_RS_FIT_DEVICE_ACCELERATOR;
    ggml_backend_dev_t device = nullptr;
    int32_t backend_type = 0;
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
    std::vector<llama_rs_fit_measurement_device> devices;
};

struct llama_rs_fit_device_storage {
    enum llama_rs_fit_device_kind kind = LLAMA_RS_FIT_DEVICE_ACCELERATOR;
    ggml_backend_dev_t device = nullptr;
    int32_t backend_type = 0;
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
};

struct llama_rs_context_device_memory_storage {
    ggml_backend_dev_t device = nullptr;
    bool has_device_index = false;
    size_t device_index = 0;
    int32_t backend_type = 0;
    std::string name;
    std::string description;
    bool has_total_bytes = false;
    uint64_t total_bytes = 0;
    bool has_free_bytes = false;
    uint64_t free_bytes = 0;
    struct llama_rs_context_memory_values allocations = {};
    size_t registry_order = std::numeric_limits<size_t>::max();
};

struct llama_rs_buffer_type_memory_storage {
    std::string name;
    struct llama_rs_context_memory_values allocations = {};
};

struct llama_rs_context_memory_report {
    std::vector<llama_rs_context_device_memory_storage> devices;
    std::vector<llama_rs_buffer_type_memory_storage> other_buffer_types;
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

static struct llama_rs_bytes_view llama_rs_context_memory_view(const std::string & value) {
    return {
        reinterpret_cast<const uint8_t *>(value.data()),
        value.size(),
    };
}

static void llama_rs_context_memory_add(
    uint64_t & target,
    size_t value,
    const char * category) {
    static_assert(sizeof(size_t) <= sizeof(uint64_t), "size_t must fit into uint64_t");
    const uint64_t converted = static_cast<uint64_t>(value);
    if (converted > std::numeric_limits<uint64_t>::max() - target) {
        throw std::overflow_error(std::string("context memory ") + category + " total overflowed u64");
    }
    target += converted;
}

static void llama_rs_context_memory_add(
    struct llama_rs_context_memory_values & target,
    const struct llama_memory_breakdown_data & source) {
    llama_rs_context_memory_add(target.model_bytes, source.model, "model");
    llama_rs_context_memory_add(target.context_bytes, source.context, "context");
    llama_rs_context_memory_add(target.compute_bytes, source.compute, "compute");
}

static size_t llama_rs_context_memory_registry_order(ggml_backend_dev_t device) {
    const size_t count = ggml_backend_dev_count();
    for (size_t index = 0; index < count; ++index) {
        if (ggml_backend_dev_get(index) == device) {
            return index;
        }
    }
    return std::numeric_limits<size_t>::max();
}

static struct llama_rs_context_device_memory_storage llama_rs_context_memory_device(
    ggml_backend_dev_t device,
    bool has_device_index,
    size_t device_index) {
    if (!device) {
        throw std::runtime_error("llama.cpp returned a null backend device");
    }

    llama_rs_context_device_memory_storage result;
    result.device = device;
    result.has_device_index = has_device_index;
    result.device_index = device_index;
    result.backend_type = static_cast<int32_t>(ggml_backend_dev_type(device));
    result.name = llama_rs_fit_string(ggml_backend_dev_name(device));
    result.description = llama_rs_fit_string(ggml_backend_dev_description(device));
    result.registry_order = llama_rs_context_memory_registry_order(device);
    return result;
}

static void llama_rs_context_memory_read_totals(
    struct llama_rs_context_device_memory_storage & device) {
    if (!device.device) {
        return;
    }
    size_t free = 0;
    size_t total = 0;
    ggml_backend_dev_memory(device.device, &free, &total);
    // llama.cpp treats 0/0 as an unavailable memory budget for a backend.
    if (free == 0 && total == 0) {
        return;
    }
    device.has_total_bytes = true;
    device.total_bytes = static_cast<uint64_t>(total);
    device.has_free_bytes = true;
    device.free_bytes = static_cast<uint64_t>(free);
}

static std::unique_ptr<llama_rs_context_memory_report> llama_rs_context_memory_build(
    const struct llama_context * context) {
    const auto * model = llama_get_model(context);
    if (!model) {
        throw std::runtime_error("llama.cpp context has no model");
    }

    auto report = std::make_unique<llama_rs_context_memory_report>();
    const int32_t model_device_count = llama_model_n_devices(model);
    if (model_device_count < 0) {
        throw std::runtime_error("llama.cpp returned a negative model device count");
    }
    report->devices.reserve(static_cast<size_t>(model_device_count) + 1);
    for (int32_t index = 0; index < model_device_count; ++index) {
        report->devices.push_back(llama_rs_context_memory_device(
            llama_model_get_device(model, index),
            true,
            static_cast<size_t>(index)));
    }

    llama_rs_context_device_memory_storage host;
    host.backend_type = static_cast<int32_t>(GGML_BACKEND_DEVICE_TYPE_CPU);
    host.name = "CPU";
    host.description = "Host memory";
    if (const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        host = llama_rs_context_memory_device(cpu, false, 0);
    }

    std::vector<llama_rs_context_device_memory_storage> unmatched_devices;
    const auto memory_breakdown = llama_get_memory_breakdown(context);
    for (const auto & [buffer_type, memory] : memory_breakdown) {
        if (!buffer_type) {
            throw std::runtime_error("llama.cpp memory breakdown contains a null buffer type");
        }
        if (ggml_backend_buft_is_host(buffer_type)) {
            llama_rs_context_memory_add(host.allocations, memory);
            continue;
        }

        const auto device = ggml_backend_buft_get_device(buffer_type);
        if (device) {
            auto registered = std::find_if(
                report->devices.begin(),
                report->devices.end(),
                [device](const auto & candidate) { return candidate.device == device; });
            if (registered != report->devices.end()) {
                llama_rs_context_memory_add(registered->allocations, memory);
                continue;
            }

            auto unmatched = std::find_if(
                unmatched_devices.begin(),
                unmatched_devices.end(),
                [device](const auto & candidate) { return candidate.device == device; });
            if (unmatched == unmatched_devices.end()) {
                unmatched_devices.push_back(llama_rs_context_memory_device(device, false, 0));
                unmatched = std::prev(unmatched_devices.end());
            }
            llama_rs_context_memory_add(unmatched->allocations, memory);
            continue;
        }

        llama_rs_buffer_type_memory_storage other;
        other.name = llama_rs_fit_string(ggml_backend_buft_name(buffer_type));
        llama_rs_context_memory_add(other.allocations, memory);
        report->other_buffer_types.push_back(std::move(other));
    }

    std::sort(
        unmatched_devices.begin(),
        unmatched_devices.end(),
        [](const auto & left, const auto & right) {
            return std::tie(
                       left.registry_order,
                       left.backend_type,
                       left.name,
                       left.description) <
                std::tie(
                       right.registry_order,
                       right.backend_type,
                       right.name,
                       right.description);
        });
    report->devices.insert(
        report->devices.end(),
        std::make_move_iterator(unmatched_devices.begin()),
        std::make_move_iterator(unmatched_devices.end()));
    report->devices.push_back(std::move(host));

    for (auto & device : report->devices) {
        llama_rs_context_memory_read_totals(device);
    }
    std::sort(
        report->other_buffer_types.begin(),
        report->other_buffer_types.end(),
        [](const auto & left, const auto & right) {
            return std::tie(
                       left.name,
                       left.allocations.model_bytes,
                       left.allocations.context_bytes,
                       left.allocations.compute_bytes) <
                std::tie(
                       right.name,
                       right.allocations.model_bytes,
                       right.allocations.context_bytes,
                       right.allocations.compute_bytes);
        });
    return report;
}

static std::string llama_rs_fit_string(const char * value) {
    return value ? value : "";
}

static struct llama_rs_fit_memory llama_rs_fit_copy_memory(
    const struct llama_device_memory_data & source) {
    struct llama_rs_fit_memory result = {};
    result.total_bytes = source.total;
    result.free_bytes = source.free;
    result.model_bytes = source.mb.model;
    result.context_bytes = source.mb.context;
    result.compute_bytes = source.mb.compute;
    return result;
}

// Pinned upstream common/fit temporarily installs a process-global logger backed by call-local
// state. The safe Rust API documents that fit must not run concurrently with native work that logs.
static struct llama_rs_fit_measurement llama_rs_fit_measure(
    const char * path_model,
    const struct llama_model_params * mparams,
    const struct llama_context_params * cparams,
    enum ggml_log_level log_level) {
    struct llama_rs_fit_measurement result;
    try {
        std::vector<ggml_backend_dev_t> devices;
        uint32_t model_layer_count = 0;
        uint32_t model_context_tokens = 0;
        uint32_t model_expert_count = 0;
        const auto memory = common_get_device_memory_data(
            path_model,
            mparams,
            cparams,
            devices,
            model_layer_count,
            model_context_tokens,
            model_expert_count,
            log_level);
        if (memory.size() != devices.size() + 1) {
            throw std::runtime_error("common/fit returned an invalid device memory result");
        }

        result.model_layer_count = model_layer_count;
        result.model_context_tokens = model_context_tokens;
        result.model_expert_count = model_expert_count;
        result.devices.reserve(memory.size());

        for (size_t index = 0; index < devices.size(); ++index) {
            const auto device = devices[index];
            if (!device) {
                throw std::runtime_error("common/fit returned a null accelerator device");
            }
            llama_rs_fit_measurement_device value;
            value.kind = LLAMA_RS_FIT_DEVICE_ACCELERATOR;
            value.device = device;
            value.backend_type = static_cast<int32_t>(ggml_backend_dev_type(device));
            value.name = llama_rs_fit_string(ggml_backend_dev_name(device));
            value.description = llama_rs_fit_string(ggml_backend_dev_description(device));
            value.memory = llama_rs_fit_copy_memory(memory[index]);
            result.devices.push_back(std::move(value));
        }

        llama_rs_fit_measurement_device host;
        host.kind = LLAMA_RS_FIT_DEVICE_HOST;
        const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        host.device = cpu;
        host.backend_type = static_cast<int32_t>(GGML_BACKEND_DEVICE_TYPE_CPU);
        host.name = cpu ? llama_rs_fit_string(ggml_backend_dev_name(cpu)) : "CPU";
        host.description = cpu
            ? llama_rs_fit_string(ggml_backend_dev_description(cpu))
            : "Host memory";
        host.memory = llama_rs_fit_copy_memory(memory.back());
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
        report->summary.accelerator_count = accelerator_count;
        report->summary.initial_measurement_available = initial.available;
        report->summary.fitted_measurement_available = fitted.available;
        report->summary.elapsed_microseconds = llama_time_us() - started_at;

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

extern "C" llama_rs_status llama_rs_fit_report_create(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    size_t margins_count,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level,
    struct llama_rs_fit_report ** out_report,
    char ** out_error) {
    return llama_rs_fit_report_create_impl(
        path_model, mparams, cparams,
        nullptr, nullptr, nullptr,
        tensor_split, tensor_buft_overrides, margins, margins_count,
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
    uint32_t n_ctx_min,
    enum ggml_log_level log_level,
    struct llama_rs_fit_report ** out_report,
    char ** out_error) {
    return llama_rs_fit_report_create_impl(
        path_model, mparams, cparams,
        target_path, target_mparams, target_cparams,
        tensor_split, tensor_buft_overrides, margins, margins_count,
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

extern "C" llama_rs_status llama_rs_context_memory_report_create(
    const struct llama_context * context,
    struct llama_rs_context_memory_report ** out_report,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_report) {
        *out_report = nullptr;
    }
    if (!context || !out_report) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "context memory report arguments must not be null");
    }

    try {
        auto report = llama_rs_context_memory_build(context);
        *out_report = report.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" void llama_rs_context_memory_report_free(
    struct llama_rs_context_memory_report * report) {
    delete report;
}

extern "C" size_t llama_rs_context_memory_report_device_count(
    const struct llama_rs_context_memory_report * report) {
    return report ? report->devices.size() : 0;
}

extern "C" llama_rs_status llama_rs_context_memory_report_device_get(
    const struct llama_rs_context_memory_report * report,
    size_t index,
    struct llama_rs_context_device_memory * out_device,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!report || !out_device || index >= report->devices.size()) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "context memory device index is out of range");
    }

    const auto & source = report->devices[index];
    struct llama_rs_context_device_memory result = {};
    result.has_device_index = source.has_device_index;
    result.device_index = source.device_index;
    result.backend_type = source.backend_type;
    result.name = llama_rs_context_memory_view(source.name);
    result.description = llama_rs_context_memory_view(source.description);
    result.has_total_bytes = source.has_total_bytes;
    result.total_bytes = source.total_bytes;
    result.has_free_bytes = source.has_free_bytes;
    result.free_bytes = source.free_bytes;
    result.allocations = source.allocations;
    *out_device = result;
    return LLAMA_RS_STATUS_OK;
}

extern "C" size_t llama_rs_context_memory_report_other_count(
    const struct llama_rs_context_memory_report * report) {
    return report ? report->other_buffer_types.size() : 0;
}

extern "C" llama_rs_status llama_rs_context_memory_report_other_get(
    const struct llama_rs_context_memory_report * report,
    size_t index,
    struct llama_rs_buffer_type_memory * out_buffer,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!report || !out_buffer || index >= report->other_buffer_types.size()) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "context memory buffer type index is out of range");
    }

    const auto & source = report->other_buffer_types[index];
    struct llama_rs_buffer_type_memory result = {};
    result.name = llama_rs_context_memory_view(source.name);
    result.allocations = source.allocations;
    *out_buffer = result;
    return LLAMA_RS_STATUS_OK;
}

extern "C" void llama_rs_memory_breakdown_print(const struct llama_context * ctx) {
    try {
        common_memory_breakdown_print(ctx);
    } catch (...) {
        // Legacy diagnostic ABI has no error channel; never unwind through C.
    }
}
