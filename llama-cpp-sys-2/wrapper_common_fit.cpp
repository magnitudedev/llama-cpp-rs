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

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/fit.h"
#include "llama.cpp/include/llama.h"
#include "llama.cpp/src/llama-ext.h"
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

extern "C" int32_t llama_rs_common_default_math_threads(void) {
    return common_cpu_get_num_math();
}

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
    uint64_t model_tensor_bytes = 0;
    std::vector<llama_rs_fit_measurement_device> devices;
};

static struct llama_rs_fit_measurement llama_rs_fit_measure_loaded(
    const struct llama_model * model,
    const struct llama_context_params * cparams) {
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
    result.available = true;
    return result;
}

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
        report->summary.model_tensor_bytes = hparams.model_tensor_bytes;
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

extern "C" llama_rs_status llama_rs_fit_measure_reports_create(
    const char * path_model,
    const struct llama_model_params * mparams,
    const struct llama_context_params * cparams,
    size_t profile_count,
    const size_t * margins,
    size_t margins_count,
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
            const auto measurement = llama_rs_fit_measure_loaded(model.get(), &cparams[profile]);
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

extern "C" void llama_rs_memory_breakdown_print(const struct llama_context * ctx) {
    try {
        common_memory_breakdown_print(ctx);
    } catch (...) {
        // Legacy diagnostic ABI has no error channel; never unwind through C.
    }
}
