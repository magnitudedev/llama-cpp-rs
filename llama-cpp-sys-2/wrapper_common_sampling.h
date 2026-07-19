#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct llama_model;
struct llama_rs_common_sampler;

#ifdef __cplusplus
extern "C" {
#endif

// Coarse wrapper around llama.cpp's common/sampling policy. The configuration is
// deliberately a C POD, while construction and ownership of common_params_sampling
// remain on the C++ side. A `*_set` field distinguishes an explicit override from
// the pinned llama.cpp/model-metadata default.
typedef enum llama_rs_common_sampler_type {
    LLAMA_RS_COMMON_SAMPLER_DRY = 0,
    LLAMA_RS_COMMON_SAMPLER_TOP_K = 1,
    LLAMA_RS_COMMON_SAMPLER_TOP_P = 2,
    LLAMA_RS_COMMON_SAMPLER_MIN_P = 3,
    LLAMA_RS_COMMON_SAMPLER_TYPICAL_P = 4,
    LLAMA_RS_COMMON_SAMPLER_TEMPERATURE = 5,
    LLAMA_RS_COMMON_SAMPLER_XTC = 6,
    LLAMA_RS_COMMON_SAMPLER_INFILL = 7,
    LLAMA_RS_COMMON_SAMPLER_PENALTIES = 8,
    LLAMA_RS_COMMON_SAMPLER_TOP_N_SIGMA = 9,
    LLAMA_RS_COMMON_SAMPLER_ADAPTIVE_P = 10,
} llama_rs_common_sampler_type;

typedef enum llama_rs_common_grammar_type {
    LLAMA_RS_COMMON_GRAMMAR_USER = 0,
    LLAMA_RS_COMMON_GRAMMAR_OUTPUT_FORMAT = 1,
    LLAMA_RS_COMMON_GRAMMAR_TOOL_CALLS = 2,
} llama_rs_common_grammar_type;

typedef enum llama_rs_common_grammar_trigger_type {
    LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TOKEN = 0,
    LLAMA_RS_COMMON_GRAMMAR_TRIGGER_WORD = 1,
    LLAMA_RS_COMMON_GRAMMAR_TRIGGER_PATTERN = 2,
    LLAMA_RS_COMMON_GRAMMAR_TRIGGER_PATTERN_FULL = 3,
} llama_rs_common_grammar_trigger_type;

typedef struct llama_rs_common_grammar_trigger_input {
    enum llama_rs_common_grammar_trigger_type type;
    const char * value;
    llama_token token;
} llama_rs_common_grammar_trigger_input;

typedef struct llama_rs_common_logit_bias_input {
    llama_token token;
    float bias;
} llama_rs_common_logit_bias_input;

typedef struct llama_rs_common_sampler_config {
    bool seed_set;
    uint32_t seed;
    bool n_prev_set;
    int32_t n_prev;
    bool n_probs_set;
    int32_t n_probs;
    bool min_keep_set;
    int32_t min_keep;
    bool top_k_set;
    int32_t top_k;
    bool top_p_set;
    float top_p;
    bool min_p_set;
    float min_p;
    bool xtc_probability_set;
    float xtc_probability;
    bool xtc_threshold_set;
    float xtc_threshold;
    bool typical_p_set;
    float typical_p;
    bool temperature_set;
    float temperature;
    bool dynatemp_range_set;
    float dynatemp_range;
    bool dynatemp_exponent_set;
    float dynatemp_exponent;
    bool repeat_last_n_set;
    int32_t repeat_last_n;
    bool repeat_penalty_set;
    float repeat_penalty;
    bool frequency_penalty_set;
    float frequency_penalty;
    bool presence_penalty_set;
    float presence_penalty;
    bool dry_multiplier_set;
    float dry_multiplier;
    bool dry_base_set;
    float dry_base;
    bool dry_allowed_length_set;
    int32_t dry_allowed_length;
    bool dry_penalty_last_n_set;
    int32_t dry_penalty_last_n;
    bool adaptive_target_set;
    float adaptive_target;
    bool adaptive_decay_set;
    float adaptive_decay;
    bool mirostat_set;
    int32_t mirostat;
    bool top_n_sigma_set;
    float top_n_sigma;
    bool mirostat_tau_set;
    float mirostat_tau;
    bool mirostat_eta_set;
    float mirostat_eta;
    bool ignore_eos_set;
    bool ignore_eos;
    bool no_perf_set;
    bool no_perf;
    bool timing_per_token_set;
    bool timing_per_token;

    bool dry_sequence_breakers_set;
    const char * const * dry_sequence_breakers;
    size_t dry_sequence_breakers_count;
    bool samplers_set;
    const enum llama_rs_common_sampler_type * samplers;
    size_t samplers_count;

    bool grammar_set;
    enum llama_rs_common_grammar_type grammar_type;
    const char * grammar;
    bool grammar_lazy_set;
    bool grammar_lazy;
    bool grammar_triggers_set;
    const struct llama_rs_common_grammar_trigger_input * grammar_triggers;
    size_t grammar_triggers_count;
    bool preserved_tokens_set;
    const char * const * preserved_tokens;
    size_t preserved_tokens_count;

    bool logit_bias_set;
    const struct llama_rs_common_logit_bias_input * logit_bias;
    size_t logit_bias_count;
    bool generation_prompt_set;
    const char * generation_prompt;

    bool reasoning_budget_set;
    int32_t reasoning_budget_tokens; // -1 means unlimited
    const char * reasoning_budget_start_tag;
    const char * reasoning_budget_end_tag;
    const char * reasoning_budget_message;
    bool reasoning_control;
} llama_rs_common_sampler_config;

llama_rs_status llama_rs_common_sampler_init(
    const struct llama_model * model,
    const struct llama_rs_common_sampler_config * config,
    struct llama_rs_common_sampler ** out_sampler,
    char ** out_error);

void llama_rs_common_sampler_free(struct llama_rs_common_sampler * sampler);

llama_rs_status llama_rs_common_sampler_accept(
    struct llama_rs_common_sampler * sampler,
    llama_token token,
    bool is_generated,
    char ** out_error);

llama_rs_status llama_rs_common_sampler_sample(
    struct llama_rs_common_sampler * sampler,
    struct llama_context * context,
    int32_t index,
    bool grammar_first,
    llama_token * out_token,
    char ** out_error);

llama_rs_status llama_rs_common_sampler_sample_and_accept_n(
    struct llama_rs_common_sampler * sampler,
    struct llama_context * context,
    const int32_t * indices,
    size_t indices_count,
    const llama_token * draft,
    size_t draft_count,
    bool grammar_first,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count,
    char ** out_error);

llama_rs_status llama_rs_common_sampler_reset(
    struct llama_rs_common_sampler * sampler,
    char ** out_error);

uint32_t llama_rs_common_sampler_seed(const struct llama_rs_common_sampler * sampler);

bool llama_rs_common_sampler_reasoning_budget_force(struct llama_rs_common_sampler * sampler);

llama_token llama_rs_common_sampler_last(const struct llama_rs_common_sampler * sampler);

llama_rs_status llama_rs_common_sampler_description(
    const struct llama_rs_common_sampler * sampler,
    char ** out_description,
    char ** out_error);

struct llama_perf_sampler_data llama_rs_common_sampler_perf(
    const struct llama_rs_common_sampler * sampler);

void llama_rs_common_sampler_perf_reset(struct llama_rs_common_sampler * sampler);

#ifdef __cplusplus
}
#endif
