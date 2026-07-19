#include "wrapper_common_sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <stdint.h>
#include <utility>
#include <vector>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/sampling.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

struct llama_rs_common_sampler {
    common_sampler * value;
    int32_t n_vocab;
    bool has_last = false;
    llama_token last = LLAMA_TOKEN_NULL;

    llama_rs_common_sampler(common_sampler * sampler, int32_t vocabulary_size)
        : value(sampler), n_vocab(vocabulary_size) {
    }

    ~llama_rs_common_sampler() {
        common_sampler_free(value);
    }
};

static void llama_rs_common_sampler_apply_model_metadata(
    const llama_model * model,
    common_params_sampling & params) {
    auto get_int32 = [&](llama_model_meta_key key, int32_t & output) {
        char buffer[64] = {};
        if (llama_model_meta_val_str(model, llama_model_meta_key_str(key), buffer, sizeof(buffer)) <= 0) {
            return;
        }
        char * end = nullptr;
        const long parsed = std::strtol(buffer, &end, 10);
        if (end != buffer && end && parsed >= INT32_MIN && parsed <= INT32_MAX) {
            output = static_cast<int32_t>(parsed);
        }
    };

    auto get_float = [&](llama_model_meta_key key, float & output) {
        char buffer[128] = {};
        if (llama_model_meta_val_str(model, llama_model_meta_key_str(key), buffer, sizeof(buffer)) <= 0) {
            return;
        }
        char * end = nullptr;
        const float parsed = std::strtof(buffer, &end);
        if (end != buffer && end) {
            output = parsed;
        }
    };

    char sequence[512] = {};
    if (llama_model_meta_val_str(
            model,
            llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE),
            sequence,
            sizeof(sequence)) > 0) {
        std::vector<std::string> names;
        std::string current;
        for (const char character : std::string(sequence)) {
            if (character == ';') {
                if (!current.empty()) {
                    names.push_back(std::move(current));
                    current.clear();
                }
            } else {
                current.push_back(character);
            }
        }
        if (!current.empty()) {
            names.push_back(std::move(current));
        }
        if (!names.empty()) {
            params.samplers = common_sampler_types_from_names(names);
        }
    }

    get_int32(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K, params.top_k);
    get_float(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P, params.top_p);
    get_float(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P, params.min_p);
    get_float(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY, params.xtc_probability);
    get_float(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD, params.xtc_threshold);
    get_float(LLAMA_MODEL_META_KEY_SAMPLING_TEMP, params.temp);
    get_int32(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N, params.penalty_last_n);
    get_float(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT, params.penalty_repeat);
    get_int32(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT, params.mirostat);
    get_float(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU, params.mirostat_tau);
    get_float(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA, params.mirostat_eta);
}

static common_sampler_type llama_rs_common_sampler_type_from_raw(
    llama_rs_common_sampler_type type) {
    switch (type) {
        case LLAMA_RS_COMMON_SAMPLER_DRY:
            return COMMON_SAMPLER_TYPE_DRY;
        case LLAMA_RS_COMMON_SAMPLER_TOP_K:
            return COMMON_SAMPLER_TYPE_TOP_K;
        case LLAMA_RS_COMMON_SAMPLER_TOP_P:
            return COMMON_SAMPLER_TYPE_TOP_P;
        case LLAMA_RS_COMMON_SAMPLER_MIN_P:
            return COMMON_SAMPLER_TYPE_MIN_P;
        case LLAMA_RS_COMMON_SAMPLER_TYPICAL_P:
            return COMMON_SAMPLER_TYPE_TYPICAL_P;
        case LLAMA_RS_COMMON_SAMPLER_TEMPERATURE:
            return COMMON_SAMPLER_TYPE_TEMPERATURE;
        case LLAMA_RS_COMMON_SAMPLER_XTC:
            return COMMON_SAMPLER_TYPE_XTC;
        case LLAMA_RS_COMMON_SAMPLER_INFILL:
            return COMMON_SAMPLER_TYPE_INFILL;
        case LLAMA_RS_COMMON_SAMPLER_PENALTIES:
            return COMMON_SAMPLER_TYPE_PENALTIES;
        case LLAMA_RS_COMMON_SAMPLER_TOP_N_SIGMA:
            return COMMON_SAMPLER_TYPE_TOP_N_SIGMA;
        case LLAMA_RS_COMMON_SAMPLER_ADAPTIVE_P:
            return COMMON_SAMPLER_TYPE_ADAPTIVE_P;
    }
    throw std::invalid_argument("unknown common sampler type");
}

static common_grammar_type llama_rs_common_grammar_type_from_raw(
    llama_rs_common_grammar_type type) {
    switch (type) {
        case LLAMA_RS_COMMON_GRAMMAR_USER:
            return COMMON_GRAMMAR_TYPE_USER;
        case LLAMA_RS_COMMON_GRAMMAR_OUTPUT_FORMAT:
            return COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT;
        case LLAMA_RS_COMMON_GRAMMAR_TOOL_CALLS:
            return COMMON_GRAMMAR_TYPE_TOOL_CALLS;
    }
    throw std::invalid_argument("unknown common grammar type");
}

static void llama_rs_common_sampler_validate_token(
    const llama_vocab * vocab,
    llama_token token,
    const char * field) {
    if (token < 0 || token >= llama_vocab_n_tokens(vocab)) {
        throw std::invalid_argument(std::string(field) + " contains an invalid token identifier");
    }
}

static void llama_rs_common_sampler_apply_config(
    const llama_model * model,
    const llama_rs_common_sampler_config & config,
    common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
        throw std::invalid_argument("model has no vocabulary");
    }

#define LLAMA_RS_APPLY_OPTION(field, target) \
    do {                                      \
        if (config.field##_set) {             \
            params.target = config.field;     \
        }                                     \
    } while (false)

    LLAMA_RS_APPLY_OPTION(seed, seed);
    LLAMA_RS_APPLY_OPTION(n_prev, n_prev);
    LLAMA_RS_APPLY_OPTION(n_probs, n_probs);
    LLAMA_RS_APPLY_OPTION(min_keep, min_keep);
    LLAMA_RS_APPLY_OPTION(top_k, top_k);
    LLAMA_RS_APPLY_OPTION(top_p, top_p);
    LLAMA_RS_APPLY_OPTION(min_p, min_p);
    LLAMA_RS_APPLY_OPTION(xtc_probability, xtc_probability);
    LLAMA_RS_APPLY_OPTION(xtc_threshold, xtc_threshold);
    LLAMA_RS_APPLY_OPTION(typical_p, typ_p);
    LLAMA_RS_APPLY_OPTION(temperature, temp);
    LLAMA_RS_APPLY_OPTION(dynatemp_range, dynatemp_range);
    LLAMA_RS_APPLY_OPTION(dynatemp_exponent, dynatemp_exponent);
    LLAMA_RS_APPLY_OPTION(repeat_last_n, penalty_last_n);
    LLAMA_RS_APPLY_OPTION(repeat_penalty, penalty_repeat);
    LLAMA_RS_APPLY_OPTION(frequency_penalty, penalty_freq);
    LLAMA_RS_APPLY_OPTION(presence_penalty, penalty_present);
    LLAMA_RS_APPLY_OPTION(dry_multiplier, dry_multiplier);
    LLAMA_RS_APPLY_OPTION(dry_base, dry_base);
    LLAMA_RS_APPLY_OPTION(dry_allowed_length, dry_allowed_length);
    LLAMA_RS_APPLY_OPTION(dry_penalty_last_n, dry_penalty_last_n);
    LLAMA_RS_APPLY_OPTION(adaptive_target, adaptive_target);
    LLAMA_RS_APPLY_OPTION(adaptive_decay, adaptive_decay);
    LLAMA_RS_APPLY_OPTION(mirostat, mirostat);
    LLAMA_RS_APPLY_OPTION(top_n_sigma, top_n_sigma);
    LLAMA_RS_APPLY_OPTION(mirostat_tau, mirostat_tau);
    LLAMA_RS_APPLY_OPTION(mirostat_eta, mirostat_eta);
    LLAMA_RS_APPLY_OPTION(ignore_eos, ignore_eos);
    LLAMA_RS_APPLY_OPTION(no_perf, no_perf);
    LLAMA_RS_APPLY_OPTION(timing_per_token, timing_per_token);

#undef LLAMA_RS_APPLY_OPTION

    if (params.mirostat < 0 || params.mirostat > 2) {
        throw std::invalid_argument("mirostat must be disabled, version 1, or version 2");
    }

    if (config.dry_sequence_breakers_set) {
        llama_rs_chat_validate_array(
            config.dry_sequence_breakers,
            config.dry_sequence_breakers_count,
            "dry_sequence_breakers");
        params.dry_sequence_breakers.clear();
        params.dry_sequence_breakers.reserve(config.dry_sequence_breakers_count);
        for (size_t index = 0; index < config.dry_sequence_breakers_count; ++index) {
            const char * value = config.dry_sequence_breakers[index];
            if (!value) {
                throw std::invalid_argument("dry_sequence_breakers contains null");
            }
            params.dry_sequence_breakers.emplace_back(value);
        }
    }

    if (config.samplers_set) {
        llama_rs_chat_validate_array(config.samplers, config.samplers_count, "samplers");
        params.samplers.clear();
        params.samplers.reserve(config.samplers_count);
        for (size_t index = 0; index < config.samplers_count; ++index) {
            params.samplers.push_back(llama_rs_common_sampler_type_from_raw(config.samplers[index]));
        }
    }

    if (config.grammar_set) {
        if (!config.grammar || config.grammar[0] == '\0') {
            throw std::invalid_argument("grammar must not be null or empty when set");
        }
        params.grammar = {
            llama_rs_common_grammar_type_from_raw(config.grammar_type),
            config.grammar,
        };
    }
    if (config.grammar_lazy_set) {
        params.grammar_lazy = config.grammar_lazy;
    }

    if (config.preserved_tokens_set) {
        llama_rs_chat_validate_array(
            config.preserved_tokens,
            config.preserved_tokens_count,
            "preserved_tokens");
        params.preserved_tokens.clear();
        for (size_t index = 0; index < config.preserved_tokens_count; ++index) {
            const char * value = config.preserved_tokens[index];
            if (!value) {
                throw std::invalid_argument("preserved_tokens contains null");
            }
            const auto tokens = common_tokenize(vocab, value, false, true);
            if (tokens.size() == 1) {
                params.preserved_tokens.insert(tokens[0]);
            }
        }
    }

    if (config.grammar_triggers_set) {
        llama_rs_chat_validate_array(
            config.grammar_triggers,
            config.grammar_triggers_count,
            "grammar_triggers");
        params.grammar_triggers.clear();
        params.grammar_triggers.reserve(config.grammar_triggers_count);
        for (size_t index = 0; index < config.grammar_triggers_count; ++index) {
            const auto & input = config.grammar_triggers[index];
            switch (input.type) {
                case LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TOKEN:
                    llama_rs_common_sampler_validate_token(vocab, input.token, "grammar_triggers");
                    params.grammar_triggers.push_back({
                        COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
                        input.value ? input.value : "",
                        input.token,
                    });
                    break;
                case LLAMA_RS_COMMON_GRAMMAR_TRIGGER_WORD: {
                    if (!input.value) {
                        throw std::invalid_argument("word grammar trigger has no value");
                    }
                    const auto tokens = common_tokenize(vocab, input.value, false, true);
                    if (tokens.size() == 1) {
                        if (params.preserved_tokens.find(tokens[0]) == params.preserved_tokens.end()) {
                            throw std::invalid_argument(
                                std::string("single-token grammar trigger is not preserved: ") + input.value);
                        }
                        params.grammar_triggers.push_back({
                            COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
                            input.value,
                            tokens[0],
                        });
                    } else {
                        params.grammar_triggers.push_back({
                            COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
                            input.value,
                        });
                    }
                    break;
                }
                case LLAMA_RS_COMMON_GRAMMAR_TRIGGER_PATTERN:
                case LLAMA_RS_COMMON_GRAMMAR_TRIGGER_PATTERN_FULL:
                    if (!input.value) {
                        throw std::invalid_argument("pattern grammar trigger has no value");
                    }
                    params.grammar_triggers.push_back({
                        input.type == LLAMA_RS_COMMON_GRAMMAR_TRIGGER_PATTERN
                            ? COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN
                            : COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                        input.value,
                    });
                    break;
                default:
                    throw std::invalid_argument("unknown common grammar trigger type");
            }
        }
    }

    if (params.grammar_lazy && params.grammar_triggers.empty()) {
        throw std::invalid_argument("lazy grammar requires at least one activation trigger");
    }

    if (config.logit_bias_set) {
        llama_rs_chat_validate_array(config.logit_bias, config.logit_bias_count, "logit_bias");
        params.logit_bias.clear();
        params.logit_bias.reserve(config.logit_bias_count);
        for (size_t index = 0; index < config.logit_bias_count; ++index) {
            const auto & input = config.logit_bias[index];
            llama_rs_common_sampler_validate_token(vocab, input.token, "logit_bias");
            params.logit_bias.push_back({input.token, input.bias});
        }
    }

    if (config.generation_prompt_set) {
        if (!config.generation_prompt) {
            throw std::invalid_argument("generation_prompt is null while set");
        }
        params.generation_prompt = config.generation_prompt;
    }

    if (config.reasoning_budget_set) {
        if (config.reasoning_budget_tokens < -1) {
            throw std::invalid_argument("reasoning budget must be -1 (unlimited) or non-negative");
        }
        const char * start_tag = config.reasoning_budget_start_tag
            ? config.reasoning_budget_start_tag
            : "";
        const char * end_tag = config.reasoning_budget_end_tag
            ? config.reasoning_budget_end_tag
            : "";
        const char * message = config.reasoning_budget_message
            ? config.reasoning_budget_message
            : "";
        params.reasoning_budget_tokens = config.reasoning_budget_tokens;
        params.reasoning_control = config.reasoning_control;
        params.reasoning_budget_message = message;
        if (start_tag[0] != '\0') {
            params.reasoning_budget_start = common_tokenize(vocab, start_tag, false, true);
        }
        if (end_tag[0] != '\0') {
            params.reasoning_budget_end = common_tokenize(vocab, end_tag, false, true);
            params.reasoning_budget_forced = common_tokenize(
                vocab,
                std::string(message) + end_tag,
                false,
                true);
        }
    }

    params.logit_bias_eog.clear();
    for (llama_token token = 0; token < llama_vocab_n_tokens(vocab); ++token) {
        if (llama_vocab_is_eog(vocab, token)) {
            params.logit_bias_eog.push_back({token, -INFINITY});
        }
    }
    if (params.ignore_eos) {
        params.logit_bias.insert(
            params.logit_bias.end(),
            params.logit_bias_eog.begin(),
            params.logit_bias_eog.end());
    }

    // ICN calls common_sampler_sample itself. Backend sampling requires wiring the
    // chain into llama_context at context construction and is intentionally a
    // separate future binding surface.
    params.backend_sampling = false;
}

extern "C" llama_rs_status llama_rs_common_sampler_init(
    const struct llama_model * model,
    const struct llama_rs_common_sampler_config * config,
    struct llama_rs_common_sampler ** out_sampler,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!model || !config || !out_sampler) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "model, config, and out_sampler must not be null");
    }
    *out_sampler = nullptr;
    try {
        common_params_sampling params;
        llama_rs_common_sampler_apply_model_metadata(model, params);
        llama_rs_common_sampler_apply_config(model, *config, params);
        common_sampler_ptr native(common_sampler_init(model, params));
        if (!native) {
            return llama_rs_chat_set_error(
                out_error,
                LLAMA_RS_STATUS_EXCEPTION,
                "common_sampler_init returned null");
        }
        auto wrapper = std::make_unique<llama_rs_common_sampler>(
            native.get(),
            llama_vocab_n_tokens(llama_model_get_vocab(model)));
        native.release();
        *out_sampler = wrapper.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" void llama_rs_common_sampler_free(struct llama_rs_common_sampler * sampler) {
    delete sampler;
}

extern "C" llama_rs_status llama_rs_common_sampler_accept(
    struct llama_rs_common_sampler * sampler,
    llama_token token,
    bool is_generated,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!sampler || !sampler->value) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "sampler must not be null");
    }
    if (token < 0 || token >= sampler->n_vocab) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "token is outside the model vocabulary");
    }
    try {
        common_sampler_accept(sampler->value, token, is_generated);
        sampler->has_last = true;
        sampler->last = token;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_common_sampler_sample(
    struct llama_rs_common_sampler * sampler,
    struct llama_context * context,
    int32_t index,
    bool grammar_first,
    llama_token * out_token,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!sampler || !sampler->value || !context || !out_token) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "sampler, context, and out_token must not be null");
    }
    try {
        *out_token = common_sampler_sample(sampler->value, context, index, grammar_first);
        if (*out_token == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("common_sampler_sample returned a null token");
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_common_sampler_sample_and_accept_n(
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
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!sampler || !sampler->value || !context || !indices ||
        indices_count != draft_count + 1 || (!draft && draft_count > 0) ||
        !out_tokens_count || out_tokens_capacity < indices_count || !out_tokens) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "invalid speculative sampler arguments");
    }
    try {
        const std::vector<int32_t> index_values(indices, indices + indices_count);
        llama_tokens draft_values;
        if (draft_count > 0) {
            draft_values.assign(draft, draft + draft_count);
        }
        const auto accepted = common_sampler_sample_and_accept_n(
            sampler->value,
            context,
            index_values,
            draft_values,
            grammar_first);
        *out_tokens_count = accepted.size();
        if (accepted.size() > out_tokens_capacity) {
            return llama_rs_chat_set_error(
                out_error,
                LLAMA_RS_STATUS_ALLOCATION_FAILED,
                "accepted speculative output exceeded capacity");
        }
        std::copy(accepted.begin(), accepted.end(), out_tokens);
        if (!accepted.empty()) {
            sampler->last = accepted.back();
            sampler->has_last = true;
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_common_sampler_reset(
    struct llama_rs_common_sampler * sampler,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!sampler || !sampler->value) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "sampler must not be null");
    }
    try {
        common_sampler_reset(sampler->value);
        sampler->has_last = false;
        sampler->last = LLAMA_TOKEN_NULL;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" uint32_t llama_rs_common_sampler_seed(
    const struct llama_rs_common_sampler * sampler) {
    if (!sampler || !sampler->value) {
        return LLAMA_DEFAULT_SEED;
    }
    try {
        return common_sampler_get_seed(sampler->value);
    } catch (...) {
        return LLAMA_DEFAULT_SEED;
    }
}

extern "C" bool llama_rs_common_sampler_reasoning_budget_force(
    struct llama_rs_common_sampler * sampler) {
    if (!sampler || !sampler->value) {
        return false;
    }
    try {
        return common_sampler_reasoning_budget_force(sampler->value);
    } catch (...) {
        return false;
    }
}

extern "C" llama_token llama_rs_common_sampler_last(
    const struct llama_rs_common_sampler * sampler) {
    return sampler && sampler->value && sampler->has_last
        ? sampler->last
        : LLAMA_TOKEN_NULL;
}

extern "C" llama_rs_status llama_rs_common_sampler_description(
    const struct llama_rs_common_sampler * sampler,
    char ** out_description,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!sampler || !sampler->value || !out_description) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "sampler and out_description must not be null");
    }
    *out_description = nullptr;
    try {
        *out_description = llama_rs_dup_string(common_sampler_print(sampler->value));
        return *out_description ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" struct llama_perf_sampler_data llama_rs_common_sampler_perf(
    const struct llama_rs_common_sampler * sampler) {
    if (!sampler || !sampler->value) {
        return {0.0, 0};
    }
    try {
        return llama_perf_sampler(common_sampler_get(sampler->value));
    } catch (...) {
        return {0.0, 0};
    }
}

extern "C" void llama_rs_common_sampler_perf_reset(struct llama_rs_common_sampler * sampler) {
    if (sampler && sampler->value) {
        try {
            llama_perf_sampler_reset(common_sampler_get(sampler->value));
        } catch (...) {
            // Performance counters are optional diagnostics. The no-error ABI
            // cannot report failure, but it must never unwind through Rust.
        }
    }
}
