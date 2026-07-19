#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct llama_rs_fit_report;
struct llama_rs_context_memory_report;

#ifdef __cplusplus
extern "C" {
#endif

// Fit model/context params to device memory (wraps llama.cpp's common_fit_params).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level);

// Stable C projection of common/fit diagnostics. These structures deliberately
// copy values out of llama.cpp's private C++ llama_device_memory_data and
// llama_memory_breakdown types instead of exposing their layouts.
typedef enum llama_rs_fit_status {
    LLAMA_RS_FIT_STATUS_SUCCESS = 0,
    LLAMA_RS_FIT_STATUS_FAILURE = 1,
    LLAMA_RS_FIT_STATUS_ERROR = 2,
} llama_rs_fit_status;

typedef enum llama_rs_fit_device_kind {
    LLAMA_RS_FIT_DEVICE_ACCELERATOR = 0,
    LLAMA_RS_FIT_DEVICE_HOST = 1,
} llama_rs_fit_device_kind;

typedef enum llama_rs_fit_placement_kind {
    LLAMA_RS_FIT_PLACEMENT_HOST = 0,
    LLAMA_RS_FIT_PLACEMENT_DEVICE = 1,
    LLAMA_RS_FIT_PLACEMENT_OTHER = 2,
} llama_rs_fit_placement_kind;

typedef struct llama_rs_fit_memory {
    int64_t total_bytes;
    int64_t free_bytes;
    uint64_t model_bytes;
    uint64_t context_bytes;
    uint64_t compute_bytes;
} llama_rs_fit_memory;

typedef struct llama_rs_fit_summary {
    enum llama_rs_fit_status status;
    uint32_t requested_context_tokens;
    uint32_t fitted_context_tokens;
    uint32_t resolved_requested_context_tokens;
    uint32_t resolved_fitted_context_tokens;
    int32_t requested_gpu_layers;
    int32_t fitted_gpu_layers;
    uint32_t resolved_requested_gpu_layers;
    uint32_t resolved_fitted_gpu_layers;
    uint32_t model_layer_count;
    uint32_t model_context_tokens;
    uint32_t model_expert_count;
    size_t accelerator_count;
    bool initial_measurement_available;
    bool fitted_measurement_available;
    int64_t elapsed_microseconds;
} llama_rs_fit_summary;

typedef struct llama_rs_fit_device {
    size_t index;
    enum llama_rs_fit_device_kind kind;
    int32_t backend_type;
    const char * name;
    const char * description;
    bool initial_available;
    struct llama_rs_fit_memory initial;
    bool fitted_available;
    struct llama_rs_fit_memory fitted;
    bool margin_applies;
    uint64_t margin_bytes;
} llama_rs_fit_device;

typedef struct llama_rs_fit_placement {
    const char * pattern;
    const char * buffer_type;
    enum llama_rs_fit_placement_kind kind;
    int32_t device_index;
    const char * device_name;
    const char * device_description;
} llama_rs_fit_placement;

// Runs the exact pinned common_get_device_memory_data/common_fit_params path.
// The report remains useful when common_fit_params returns FAILURE or ERROR;
// the function's own llama_rs_status only reports bridge/argument failures.
llama_rs_status llama_rs_fit_report_create(
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
    char ** out_error);

// Runs the same fit path while keeping a no-allocation target context alive as `ctx_other`.
// This is required for MTP graph planning and does not include the target allocations in the
// returned linked-model report; callers compose the separately measured target report.
llama_rs_status llama_rs_fit_report_create_linked(
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
    char ** out_error);

void llama_rs_fit_report_free(struct llama_rs_fit_report * report);

bool llama_rs_fit_report_get_summary(
    const struct llama_rs_fit_report * report,
    struct llama_rs_fit_summary * out_summary);

const char * llama_rs_fit_report_initial_error(const struct llama_rs_fit_report * report);
const char * llama_rs_fit_report_fitted_error(const struct llama_rs_fit_report * report);

size_t llama_rs_fit_report_device_count(const struct llama_rs_fit_report * report);
bool llama_rs_fit_report_get_device(
    const struct llama_rs_fit_report * report,
    size_t index,
    struct llama_rs_fit_device * out_device);

size_t llama_rs_fit_report_tensor_split_count(const struct llama_rs_fit_report * report);
bool llama_rs_fit_report_get_tensor_split(
    const struct llama_rs_fit_report * report,
    size_t index,
    float * out_value);

size_t llama_rs_fit_report_placement_count(const struct llama_rs_fit_report * report);
bool llama_rs_fit_report_get_placement(
    const struct llama_rs_fit_report * report,
    size_t index,
    struct llama_rs_fit_placement * out_placement);

// Stable, owner-borrowed projection of llama_get_memory_breakdown(). The report owns every string
// referenced by the byte views below. Views remain valid until the report is freed. The pinned
// upstream breakdown does not include llama_context's private logits/embeddings output buffer.
typedef struct llama_rs_context_memory_values {
    uint64_t model_bytes;
    uint64_t context_bytes;
    uint64_t compute_bytes;
} llama_rs_context_memory_values;

typedef struct llama_rs_context_device_memory {
    bool has_device_index;
    size_t device_index;
    int32_t backend_type;
    struct llama_rs_bytes_view name;
    struct llama_rs_bytes_view description;
    bool has_total_bytes;
    uint64_t total_bytes;
    bool has_free_bytes;
    uint64_t free_bytes;
    struct llama_rs_context_memory_values allocations;
} llama_rs_context_device_memory;

typedef struct llama_rs_buffer_type_memory {
    struct llama_rs_bytes_view name;
    struct llama_rs_context_memory_values allocations;
} llama_rs_buffer_type_memory;

llama_rs_status llama_rs_context_memory_report_create(
    const struct llama_context * context,
    struct llama_rs_context_memory_report ** out_report,
    char ** out_error);

void llama_rs_context_memory_report_free(struct llama_rs_context_memory_report * report);

size_t llama_rs_context_memory_report_device_count(
    const struct llama_rs_context_memory_report * report);

llama_rs_status llama_rs_context_memory_report_device_get(
    const struct llama_rs_context_memory_report * report,
    size_t index,
    struct llama_rs_context_device_memory * out_device,
    char ** out_error);

size_t llama_rs_context_memory_report_other_count(
    const struct llama_rs_context_memory_report * report);

llama_rs_status llama_rs_context_memory_report_other_get(
    const struct llama_rs_context_memory_report * report,
    size_t index,
    struct llama_rs_buffer_type_memory * out_buffer,
    char ** out_error);

void llama_rs_memory_breakdown_print(const struct llama_context * ctx);

#ifdef __cplusplus
}
#endif
