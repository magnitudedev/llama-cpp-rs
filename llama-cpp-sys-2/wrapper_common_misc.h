#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct llama_rs_speculative;
struct llama_vocab;

typedef enum llama_rs_speculative_method {
    LLAMA_RS_SPECULATIVE_METHOD_MTP = 0,
    LLAMA_RS_SPECULATIVE_METHOD_DFLASH = 1,
    LLAMA_RS_SPECULATIVE_METHOD_DSPARK = 2,
} llama_rs_speculative_method;

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar);

struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token);

struct llama_rs_speculative * llama_rs_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    enum llama_rs_speculative_method method,
    int32_t n_max,
    int32_t n_min,
    float p_min,
    uint32_t n_seq,
    bool probe_rollback);

void llama_rs_speculative_free(struct llama_rs_speculative * spec);

llama_rs_status llama_rs_speculative_begin(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    char ** out_error);

llama_rs_status llama_rs_speculative_process(
    struct llama_rs_speculative * spec,
    const struct llama_batch * batch,
    char ** out_error);

llama_rs_status llama_rs_speculative_prepare_draft(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    int32_t n_max,
    char ** out_error);

llama_rs_status llama_rs_speculative_draft(
    struct llama_rs_speculative * spec,
    char ** out_error);

llama_rs_status llama_rs_speculative_get_draft(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count,
    char ** out_error);

llama_rs_status llama_rs_speculative_resolve(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    size_t proposed_count,
    uint16_t accepted_count,
    llama_pos next_position,
    bool * out_replay,
    char ** out_error);

llama_rs_status llama_rs_speculative_seq_rm(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1,
    char ** out_error);

#ifdef __cplusplus
}
#endif
