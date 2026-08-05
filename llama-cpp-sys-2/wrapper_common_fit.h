#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct llama_rs_fit_report;
struct llama_rs_memory_breakdown_report;
struct llama_rs_fit_calibration;

#ifdef __cplusplus
extern "C" {
#endif

// Resolve the pinned common runtime's math-thread default. Callers must use
// this instead of mirroring platform-specific common code.
int32_t llama_rs_common_default_math_threads(void);

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

typedef enum llama_rs_fit_tensor_workload_kind {
    LLAMA_RS_FIT_TENSOR_ALWAYS_ACTIVE = 0,
    LLAMA_RS_FIT_TENSOR_ROUTED_EXPERT = 1,
    LLAMA_RS_FIT_TENSOR_ROW_LOOKUP = 2,
} llama_rs_fit_tensor_workload_kind;

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
    uint64_t model_tensor_bytes;
    size_t accelerator_count;
    bool initial_measurement_available;
    bool fitted_measurement_available;
    int64_t elapsed_microseconds;
} llama_rs_fit_summary;

typedef struct llama_rs_fit_calibration_metric {
    int32_t backend_type;
    const char * backend;
    const char * device_id;
    int32_t tensor_type;
    bool routed;
    double bytes_per_second;
    double launch_microseconds;
    double relative_spread;
    uint32_t sample_count;
    uint64_t measured_microseconds;
    bool stable;
} llama_rs_fit_calibration_metric;

typedef struct llama_rs_fit_decode_workload_summary {
    bool available;
    const char * method;
    const char * unavailable_reason;
    const char * architecture;
    uint32_t expert_count;
    uint32_t expert_used_count;
    uint32_t nextn_layer_count;
    uint32_t kv_lora_rank;
    uint32_t indexer_head_count;
    uint32_t indexer_head_size;
    uint32_t indexer_top_k;
    bool mla;
    bool hybrid_model;
    bool recurrent_model;
} llama_rs_fit_decode_workload_summary;

typedef struct llama_rs_fit_tensor_workload {
    const char * name;
    int32_t backend_type;
    const char * backend;
    const char * device_id;
    int32_t tensor_type;
    enum llama_rs_fit_tensor_workload_kind kind;
    bool baseline_executed;
    uint64_t stored_bytes;
    uint64_t operation_bytes;
} llama_rs_fit_tensor_workload;

typedef struct llama_rs_fit_kv_layer_workload {
    uint32_t layer;
    int32_t backend_type;
    const char * backend;
    const char * device_id;
    int32_t key_type;
    int32_t value_type;
    uint64_t key_bytes_per_token;
    uint64_t value_bytes_per_token;
    uint32_t attention_head_size;
    int32_t attention_state_type;
    uint32_t sliding_window_tokens;
    uint32_t compression_ratio;
    bool sparse_index;
    uint64_t indexer_bytes_per_token;
    bool recurrent;
    int32_t recurrent_type;
    uint64_t recurrent_conv_bytes;
    uint64_t recurrent_state_bytes;
} llama_rs_fit_kv_layer_workload;

// Runs bounded, model-free ggml backend calibration. No model is loaded and no
// token decode is performed. The returned values can be serialized and passed
// to isolated planner processes.
llama_rs_status llama_rs_fit_calibration_create(
    struct llama_rs_fit_calibration ** out_calibration,
    char ** out_error);

void llama_rs_fit_calibration_free(struct llama_rs_fit_calibration * calibration);

int64_t llama_rs_fit_calibration_elapsed_microseconds(
    const struct llama_rs_fit_calibration * calibration);

const char * llama_rs_fit_calibration_method(
    const struct llama_rs_fit_calibration * calibration);

size_t llama_rs_fit_calibration_metric_count(
    const struct llama_rs_fit_calibration * calibration);

bool llama_rs_fit_calibration_get_metric(
    const struct llama_rs_fit_calibration * calibration,
    size_t index,
    struct llama_rs_fit_calibration_metric * out_metric);

// Measures several contexts against one no-allocation model construction. Each
// returned report contains an initial measurement only; no fitting is run.
llama_rs_status llama_rs_fit_measure_reports_create(
    const char * path_model,
    const struct llama_model_params * mparams,
    const struct llama_context_params * cparams,
    size_t profile_count,
    const size_t * margins,
    size_t margins_count,
    bool capture_decode_workload,
    enum ggml_log_level log_level,
    struct llama_rs_fit_report ** out_reports,
    char ** out_error);

typedef struct llama_rs_fit_device {
    size_t index;
    enum llama_rs_fit_device_kind kind;
    int32_t backend_type;
    const char * backend;
    const char * device_id;
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
    bool capture_decode_workload,
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
    bool capture_decode_workload,
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

bool llama_rs_fit_report_get_decode_workload_summary(
    const struct llama_rs_fit_report * report,
    struct llama_rs_fit_decode_workload_summary * out_summary);

size_t llama_rs_fit_report_tensor_workload_count(
    const struct llama_rs_fit_report * report);

bool llama_rs_fit_report_get_tensor_workload(
    const struct llama_rs_fit_report * report,
    size_t index,
    struct llama_rs_fit_tensor_workload * out_tensor);

size_t llama_rs_fit_report_kv_layer_workload_count(
    const struct llama_rs_fit_report * report);

bool llama_rs_fit_report_get_kv_layer_workload(
    const struct llama_rs_fit_report * report,
    size_t index,
    struct llama_rs_fit_kv_layer_workload * out_layer);

typedef enum llama_rs_memory_location_kind {
    LLAMA_RS_MEMORY_LOCATION_HOST = 0,
    LLAMA_RS_MEMORY_LOCATION_DEVICE = 1,
} llama_rs_memory_location_kind;

struct llama_rs_memory_breakdown_entry {
    enum llama_rs_memory_location_kind location;
    size_t native_index;
    const char * backend;
    const char * device_id;
    uint64_t model_bytes;
    uint64_t context_bytes;
    uint64_t compute_bytes;
};

llama_rs_status llama_rs_memory_breakdown_create(
    const struct llama_context * ctx,
    struct llama_rs_memory_breakdown_report ** out_report,
    char ** out_error);
void llama_rs_memory_breakdown_free(struct llama_rs_memory_breakdown_report * report);
size_t llama_rs_memory_breakdown_count(const struct llama_rs_memory_breakdown_report * report);
bool llama_rs_memory_breakdown_get(
    const struct llama_rs_memory_breakdown_report * report,
    size_t index,
    struct llama_rs_memory_breakdown_entry * out_entry);

void llama_rs_memory_breakdown_print(const struct llama_context * ctx);

#ifdef __cplusplus
}
#endif
