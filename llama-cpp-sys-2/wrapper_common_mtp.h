#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stdint.h>

typedef enum llama_rs_mtp_preflight_code {
    LLAMA_RS_MTP_PREFLIGHT_SUPPORTED = 0,
    LLAMA_RS_MTP_PREFLIGHT_NO_MTP = 1,
    LLAMA_RS_MTP_PREFLIGHT_TARGET_IS_DRAFT = 2,
    LLAMA_RS_MTP_PREFLIGHT_DRAFT_NOT_SUPPORTED = 3,
    LLAMA_RS_MTP_PREFLIGHT_VOCABULARY_MISMATCH = 4,
    LLAMA_RS_MTP_PREFLIGHT_EMBEDDING_MISMATCH = 5,
    LLAMA_RS_MTP_PREFLIGHT_CONTEXT_UNSUPPORTED = 6,
} llama_rs_mtp_preflight_code;

typedef struct llama_rs_mtp_preflight_result {
    enum llama_mtp_model_kind kind;
    uint32_t prediction_layers;
    bool requires_target_context;
    enum llama_rs_mtp_preflight_code code;
} llama_rs_mtp_preflight_result;

typedef struct llama_rs_mtp_model_info_result {
    enum llama_mtp_model_kind kind;
    uint32_t prediction_layers;
    bool requires_target_context;
} llama_rs_mtp_model_info_result;

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

// Inspect the native MTP role implemented by one GGUF without allocating its tensors.
llama_rs_status llama_rs_mtp_model_info_from_file(
    const char * path,
    const struct llama_model_params * model_params,
    struct llama_rs_mtp_model_info_result * out_result,
    char ** out_error);

#ifdef __cplusplus
}
#endif
