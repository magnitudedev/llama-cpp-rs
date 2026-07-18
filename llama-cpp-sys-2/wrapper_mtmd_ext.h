#pragma once

#include "wrapper_utils.h"

#include "llama.cpp/tools/mtmd/mtmd.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct llama_rs_mtmd_memory_report;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llama_rs_mtmd_capabilities {
    bool vision;
    bool audio;
} llama_rs_mtmd_capabilities;

typedef struct llama_rs_mtmd_device_memory {
    bool has_device_index;
    size_t device_index;
    int32_t backend_type;
    llama_rs_bytes_view device_name;
    llama_rs_bytes_view device_description;
    uint64_t bytes;
} llama_rs_mtmd_device_memory;

// Exception-safe entry points for the mtmd operations used by the safe Rust crate. Upstream's
// public C surface is implemented in C++ and several of these operations allocate without catching
// exceptions. No safe Rust operation calls those allocating functions directly.
llama_rs_status llama_rs_mtmd_context_init_from_file(
    const char * path,
    const struct llama_model * model,
    const struct mtmd_context_params * params,
    struct mtmd_context ** out_context,
    char ** out_error);

llama_rs_status llama_rs_mtmd_bitmap_init_image(
    uint32_t width,
    uint32_t height,
    const uint8_t * data,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error);

llama_rs_status llama_rs_mtmd_bitmap_init_audio(
    size_t sample_count,
    const float * data,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error);

llama_rs_status llama_rs_mtmd_bitmap_init_from_file(
    struct mtmd_context * context,
    const char * path,
    bool placeholder,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error);

llama_rs_status llama_rs_mtmd_bitmap_init_from_buffer(
    struct mtmd_context * context,
    const uint8_t * data,
    size_t data_len,
    bool placeholder,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error);

llama_rs_status llama_rs_mtmd_bitmap_set_id(
    struct mtmd_bitmap * bitmap,
    const char * id,
    char ** out_error);

llama_rs_status llama_rs_mtmd_input_chunks_init(
    struct mtmd_input_chunks ** out_chunks,
    char ** out_error);

llama_rs_status llama_rs_mtmd_tokenize(
    struct mtmd_context * context,
    struct mtmd_input_chunks * output,
    const struct mtmd_input_text * text,
    const struct mtmd_bitmap ** bitmaps,
    size_t bitmap_count,
    int32_t * out_result,
    char ** out_error);

llama_rs_status llama_rs_mtmd_encode_chunk(
    struct mtmd_context * context,
    const struct mtmd_input_chunk * chunk,
    int32_t * out_result,
    char ** out_error);

llama_rs_status llama_rs_mtmd_eval_chunks(
    struct mtmd_context * context,
    struct llama_context * llama_context,
    const struct mtmd_input_chunks * chunks,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * out_new_n_past,
    int32_t * out_result,
    char ** out_error);

// Unlike mtmd_get_cap_from_file, this adapter preserves clip_get_cap failures.
llama_rs_status llama_rs_mtmd_capabilities_from_file(
    const char * path,
    struct llama_rs_mtmd_capabilities * out_capabilities,
    char ** out_error);

llama_rs_status llama_rs_mtmd_memory_report_create(
    const char * path,
    const struct mtmd_context_params * params,
    struct llama_rs_mtmd_memory_report ** out_report,
    char ** out_error);

void llama_rs_mtmd_memory_report_free(struct llama_rs_mtmd_memory_report * report);

size_t llama_rs_mtmd_memory_report_count(
    const struct llama_rs_mtmd_memory_report * report);

llama_rs_status llama_rs_mtmd_memory_report_get(
    const struct llama_rs_mtmd_memory_report * report,
    size_t index,
    struct llama_rs_mtmd_device_memory * out_device,
    char ** out_error);

#ifdef __cplusplus
} // extern "C"
#endif
