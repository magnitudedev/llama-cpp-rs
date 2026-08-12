#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_common_misc.h"
#include "wrapper_utils.h"

#include <stdint.h>

typedef enum llama_rs_speculative_preflight_code {
    LLAMA_RS_SPECULATIVE_PREFLIGHT_SUPPORTED = 0,
    LLAMA_RS_SPECULATIVE_PREFLIGHT_CONTEXT_UNSUPPORTED = 1,
    LLAMA_RS_SPECULATIVE_PREFLIGHT_METHOD_UNSUPPORTED = 2,
} llama_rs_speculative_preflight_code;

typedef struct llama_rs_speculative_preflight_result {
    enum llama_rs_speculative_preflight_code code;
    int32_t effective_n_max;
    int32_t effective_n_min;
} llama_rs_speculative_preflight_result;

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_speculative_preflight(
    const char * target_path,
    const char * draft_path,
    enum llama_rs_speculative_method method,
    int32_t requested_n_max,
    int32_t requested_n_min,
    float threshold,
    const struct llama_model_params * target_model_params,
    const struct llama_context_params * target_context_params,
    const struct llama_model_params * draft_model_params,
    const struct llama_context_params * draft_context_params,
    struct llama_rs_speculative_preflight_result * out_result,
    char ** out_error);

#ifdef __cplusplus
}
#endif
