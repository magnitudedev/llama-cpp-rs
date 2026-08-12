#include "wrapper_common_speculative.h"
#include "wrapper_common_misc.h"
#include "llama.cpp/src/llama-ext.h"
#include "llama.cpp/src/llama-model.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace {

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using speculative_ptr = std::unique_ptr<llama_rs_speculative, decltype(&llama_rs_speculative_free)>;

model_ptr load_model_no_alloc(
    const char * path,
    const llama_model_params & source) {
    llama_model_params params = source;
    params.no_alloc = true;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    model_ptr model(llama_model_load_from_file(path, params), llama_model_free);
    if (!model) {
        throw std::runtime_error("failed to inspect GGUF model");
    }
    return model;
}

void set_result(llama_rs_speculative_preflight_result & result, llama_rs_speculative_preflight_code code) {
    result.code = code;
}

void set_effective_bounds(
    llama_rs_speculative_preflight_result & result,
    llama_rs_speculative_method method,
    const llama_model * draft_model,
    int32_t requested_n_max,
    int32_t requested_n_min) {
    int32_t artifact_n_max = requested_n_max;
    if (method == LLAMA_RS_SPECULATIVE_METHOD_DFLASH ||
        method == LLAMA_RS_SPECULATIVE_METHOD_DSPARK) {
        char value[32] = {};
        int32_t block_size = 16;
        if (llama_model_meta_val_str(draft_model, "dflash.block_size", value, sizeof(value)) >= 0) {
            block_size = std::atoi(value);
        }
        artifact_n_max = method == LLAMA_RS_SPECULATIVE_METHOD_DSPARK
            ? block_size
            : block_size - 1;
        if (artifact_n_max <= 0) {
            throw std::runtime_error("speculative draft artifact has an invalid dflash.block_size");
        }
    }
    result.effective_n_max = std::min(requested_n_max, artifact_n_max);
    result.effective_n_min = std::min(requested_n_min, result.effective_n_max);
}

bool method_matches_artifact(llama_rs_speculative_method method, const llama_model * model) {
    const bool has_dspark_head = model->dspark_markov_w1 && model->dspark_markov_w2 &&
        model->dspark_conf_proj;
    switch (method) {
        case LLAMA_RS_SPECULATIVE_METHOD_MTP:
            return llama_model_n_layer_nextn(model) > 0;
        case LLAMA_RS_SPECULATIVE_METHOD_DFLASH:
            return llama_model_target_layer_ids_n(model) > 0 && !has_dspark_head;
        case LLAMA_RS_SPECULATIVE_METHOD_DSPARK:
            return llama_model_target_layer_ids_n(model) > 0 && has_dspark_head;
        default:
            return false;
    }
}

} // namespace

extern "C" llama_rs_status llama_rs_speculative_preflight(
    const char * target_path,
    const char * draft_path,
    enum llama_rs_speculative_method method,
    int32_t requested_n_max,
    int32_t requested_n_min,
    float threshold,
    const llama_model_params * target_model_params,
    const llama_context_params * target_context_params,
    const llama_model_params * draft_model_params,
    const llama_context_params * draft_context_params,
    llama_rs_speculative_preflight_result * out_result,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!target_path || requested_n_max <= 0 || requested_n_min < 0 ||
        requested_n_min > requested_n_max || !target_model_params || !target_context_params || !out_result ||
        (draft_path && (!draft_model_params || !draft_context_params))) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid speculative preflight arguments");
    }

    *out_result = {};
    try {
        auto target = load_model_no_alloc(target_path, *target_model_params);

        llama_context_params target_params = *target_context_params;
        target_params.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
        target_params.ctx_other = nullptr;
        context_ptr target_context(llama_init_from_model(target.get(), target_params), llama_free);
        if (!target_context) {
            set_result(*out_result, LLAMA_RS_SPECULATIVE_PREFLIGHT_CONTEXT_UNSUPPORTED);
            return LLAMA_RS_STATUS_OK;
        }

        model_ptr separate_draft(nullptr, llama_model_free);
        llama_model * draft_model = target.get();
        llama_context_params speculative_params = *target_context_params;

        if (draft_path) {
            separate_draft = load_model_no_alloc(draft_path, *draft_model_params);
            draft_model = separate_draft.get();
            speculative_params = *draft_context_params;
        }
        if (!method_matches_artifact(method, draft_model)) {
            set_result(*out_result, LLAMA_RS_SPECULATIVE_PREFLIGHT_METHOD_UNSUPPORTED);
            return LLAMA_RS_STATUS_OK;
        }

        speculative_params.ctx_type = method == LLAMA_RS_SPECULATIVE_METHOD_MTP
            ? LLAMA_CONTEXT_TYPE_MTP
            : LLAMA_CONTEXT_TYPE_DEFAULT;
        speculative_params.ctx_other = target_context.get();
        speculative_params.n_rs_seq = 0;
        context_ptr speculative_context(
            llama_init_from_model(draft_model, speculative_params), llama_free);
        if (!speculative_context) {
            set_result(*out_result, LLAMA_RS_SPECULATIVE_PREFLIGHT_CONTEXT_UNSUPPORTED);
            return LLAMA_RS_STATUS_OK;
        }

        set_effective_bounds(
            *out_result, method, draft_model, requested_n_max, requested_n_min);

        speculative_ptr speculative(
            llama_rs_speculative_init(
                target_context.get(), speculative_context.get(), method,
                out_result->effective_n_max, out_result->effective_n_min, threshold, 1, false),
            llama_rs_speculative_free);
        if (!speculative) {
            set_result(*out_result, LLAMA_RS_SPECULATIVE_PREFLIGHT_CONTEXT_UNSUPPORTED);
            return LLAMA_RS_STATUS_OK;
        }

        set_result(*out_result, LLAMA_RS_SPECULATIVE_PREFLIGHT_SUPPORTED);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}
