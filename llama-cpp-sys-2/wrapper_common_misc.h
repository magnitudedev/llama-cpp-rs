#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct llama_rs_mtp_speculative;
struct llama_vocab;

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

struct llama_rs_mtp_speculative * llama_rs_mtp_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min,
    uint32_t n_seq);

void llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative * spec);

llama_rs_status llama_rs_mtp_speculative_begin(
    struct llama_rs_mtp_speculative * spec,
    llama_seq_id seq_id,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count);

llama_rs_status llama_rs_mtp_speculative_process(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_batch * batch);

llama_rs_status llama_rs_mtp_speculative_prepare_draft(
    struct llama_rs_mtp_speculative * spec,
    llama_seq_id seq_id,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    int32_t n_max);

llama_rs_status llama_rs_mtp_speculative_draft(struct llama_rs_mtp_speculative * spec);

llama_rs_status llama_rs_mtp_speculative_get_draft(
    struct llama_rs_mtp_speculative * spec,
    llama_seq_id seq_id,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count);

llama_rs_status llama_rs_mtp_speculative_accept(
    struct llama_rs_mtp_speculative * spec,
    llama_seq_id seq_id,
    uint16_t n_accepted);

llama_rs_status llama_rs_mtp_speculative_seq_rm(
    struct llama_rs_mtp_speculative * spec,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1);

#ifdef __cplusplus
}
#endif
