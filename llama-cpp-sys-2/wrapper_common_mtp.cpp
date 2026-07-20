#include "wrapper_common_mtp.h"
#include "wrapper_common_misc.h"

#include <memory>
#include <stdexcept>

namespace {

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using speculative_ptr = std::unique_ptr<llama_rs_mtp_speculative, decltype(&llama_rs_mtp_speculative_free)>;

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

void set_result(llama_rs_mtp_preflight_result & result, llama_rs_mtp_preflight_code code) {
    result.code = code;
}

} // namespace

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

        llama_context_params target_params = *target_context_params;
        target_params.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
        target_params.ctx_other = nullptr;
        context_ptr target_context(llama_init_from_model(target.get(), target_params), llama_free);
        if (!target_context) {
            set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_CONTEXT_UNSUPPORTED);
            return LLAMA_RS_STATUS_OK;
        }

        model_ptr separate_draft(nullptr, llama_model_free);
        llama_model * draft_model = target.get();
        llama_context_params mtp_params = *target_context_params;

        if (draft_path) {
            separate_draft = load_model_no_alloc(draft_path, *draft_model_params);
            draft_model = separate_draft.get();
            mtp_params = *draft_context_params;
        }

        mtp_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        mtp_params.ctx_other = target_context.get();
        mtp_params.n_rs_seq = 0;
        context_ptr mtp_context(llama_init_from_model(draft_model, mtp_params), llama_free);
        if (!mtp_context) {
            set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_CONTEXT_UNSUPPORTED);
            return LLAMA_RS_STATUS_OK;
        }

        speculative_ptr speculative(
            llama_rs_mtp_speculative_init(
                target_context.get(), mtp_context.get(), 3, 0, 0.0f, 1),
            llama_rs_mtp_speculative_free);
        if (!speculative) {
            set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_CONTEXT_UNSUPPORTED);
            return LLAMA_RS_STATUS_OK;
        }

        set_result(*out_result, LLAMA_RS_MTP_PREFLIGHT_SUPPORTED);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}
