#include "wrapper_common_mtp.h"
#include "llama.cpp/common/speculative.h"

#include <memory>
#include <stdexcept>

namespace {

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

model_ptr load_model_no_alloc(const char * path, const llama_model_params & source) {
    llama_model_params params = source;
    params.no_alloc = true;
    params.use_mmap = false;
    params.use_mlock = false;
    model_ptr model(llama_model_load_from_file(path, params), llama_model_free);
    if (!model) {
        throw std::runtime_error("failed to inspect GGUF model");
    }
    return model;
}

void set_result(
    llama_rs_mtp_preflight_result & result,
    llama_rs_mtp_preflight_code code,
    llama_mtp_model_info info = { LLAMA_MTP_MODEL_NONE, 0, false }) {
    result.kind = info.kind;
    result.prediction_layers = info.prediction_layers;
    result.requires_target_context = info.requires_target_context;
    result.code = code;
}

} // namespace

extern "C" llama_rs_status llama_rs_mtp_model_info_from_file(
    const char * path,
    const llama_model_params * model_params,
    llama_rs_mtp_model_info_result * out_result,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!path || !model_params || !out_result) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid MTP inspection arguments");
    }

    *out_result = {};
    try {
        const auto model = load_model_no_alloc(path, *model_params);
        const llama_mtp_model_info info = llama_model_mtp_info(model.get());
        out_result->kind = info.kind;
        out_result->prediction_layers = info.prediction_layers;
        out_result->requires_target_context = info.requires_target_context;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtp_preflight(
    const char * target_path,
    const char * draft_path,
    const llama_model_params * target_model_params,
    const llama_context_params * target_context_params,
    const llama_model_params * draft_model_params,
    const llama_context_params * draft_context_params,
    llama_rs_mtp_preflight_result * out_result,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!target_path || !target_model_params || !target_context_params || !out_result ||
        (draft_path && (!draft_model_params || !draft_context_params))) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid MTP preflight arguments");
    }

    *out_result = {};
    try {
        auto target = load_model_no_alloc(target_path, *target_model_params);
        const llama_mtp_model_info target_info = llama_model_mtp_info(target.get());
        if (!draft_path && target_info.kind == LLAMA_MTP_MODEL_NONE) {
            set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_NO_MTP);
            return LLAMA_RS_STATUS_OK;
        }
        if (!draft_path && target_info.kind == LLAMA_MTP_MODEL_DRAFT) {
            set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_TARGET_IS_DRAFT, target_info);
            return LLAMA_RS_STATUS_OK;
        }

        llama_context_params target_params = *target_context_params;
        target_params.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
        target_params.ctx_other = nullptr;
        context_ptr target_context(llama_init_from_model(target.get(), target_params), llama_free);
        if (!target_context) {
            set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_CONTEXT_UNSUPPORTED, target_info);
            return LLAMA_RS_STATUS_OK;
        }

        model_ptr separate_draft(nullptr, llama_model_free);
        llama_model * draft_model = target.get();
        llama_mtp_model_info draft_info = target_info;
        llama_context_params mtp_params = *target_context_params;

        if (draft_path) {
            separate_draft = load_model_no_alloc(draft_path, *draft_model_params);
            draft_model = separate_draft.get();
            draft_info = llama_model_mtp_info(draft_model);
            if (draft_info.kind != LLAMA_MTP_MODEL_DRAFT) {
                set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_DRAFT_NOT_SUPPORTED, draft_info);
                return LLAMA_RS_STATUS_OK;
            }
            if (!common_speculative_are_compatible(target.get(), draft_model)) {
                set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_VOCABULARY_MISMATCH, draft_info);
                return LLAMA_RS_STATUS_OK;
            }
            if (llama_model_n_embd(target.get()) != llama_model_n_embd_inp(draft_model)) {
                set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_EMBEDDING_MISMATCH, draft_info);
                return LLAMA_RS_STATUS_OK;
            }
            mtp_params = *draft_context_params;
        }

        mtp_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        mtp_params.ctx_other = target_context.get();
        mtp_params.n_rs_seq = 0;
        context_ptr mtp_context(llama_init_from_model(draft_model, mtp_params), llama_free);
        if (!mtp_context) {
            set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_CONTEXT_UNSUPPORTED, draft_info);
            return LLAMA_RS_STATUS_OK;
        }

        set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_SUPPORTED, draft_info);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}
