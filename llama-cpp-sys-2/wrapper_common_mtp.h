#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stdint.h>

typedef enum llama_rs_mtp_preflight_code {
    LLAMA_RS_MTP_PREFLIGHT_SUPPORTED = 0,
    LLAMA_RS_MTP_PREFLIGHT_CONTEXT_UNSUPPORTED = 1,
} llama_rs_mtp_preflight_code;

typedef struct llama_rs_mtp_preflight_result {
    enum llama_rs_mtp_preflight_code code;
} llama_rs_mtp_preflight_result;

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_mtp_preflight(
    const char * target_path,
    const char * draft_path,
    const struct llama_model_params * target_model_params,
    const struct llama_context_params * target_context_params,
    const struct llama_model_params * draft_model_params,
    const struct llama_context_params * draft_context_params,
    struct llama_rs_mtp_preflight_result * out_result,
    char ** out_error);

#ifdef __cplusplus
}
#endif
