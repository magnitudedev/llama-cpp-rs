#include "wrapper_common_misc.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <stdint.h>
#include <utility>
#include <vector>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/common/speculative.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = common_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        return *out_grammar ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root) {
    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        std::vector<std::string> trigger_patterns;
        trigger_patterns.reserve(num_trigger_words);
        for (size_t i = 0; i < num_trigger_words; ++i) {
            const char * word = trigger_words ? trigger_words[i] : nullptr;
            if (word && word[0] != '\0') {
                trigger_patterns.push_back(regex_escape(word));
            }
        }
        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & pattern : trigger_patterns) {
            trigger_patterns_c.push_back(pattern.c_str());
        }
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns_c.data(),
            trigger_patterns_c.size(),
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token) {
    if (!sampler) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_sampler_accept(sampler, token);
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}


struct llama_rs_speculative {
    struct sequence_state {
        std::vector<llama_token> prompt;
        std::vector<llama_token> draft;
        common_prompt_checkpoint target_checkpoint;
        common_prompt_checkpoint draft_checkpoint;
        bool target_checkpointed = false;
        bool draft_checkpointed = false;
        size_t last_draft_len = 0;
        bool draft_pending = false;
        bool prepared = false;
    };
    common_params_speculative params;
    common_speculative * spec = nullptr;
    common_context_seq_rm_type target_remove_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type draft_remove_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    std::vector<sequence_state> sequences;
};

static bool llama_rs_speculative_batch_compatible(
    const struct llama_batch & batch,
    size_t n_seq) {
    const bool has_tokens = batch.token != nullptr;
    const bool has_embeddings = batch.embd != nullptr;
    if (batch.n_tokens <= 0 || has_tokens == has_embeddings || !batch.pos || !batch.n_seq_id ||
        !batch.seq_id) {
        return false;
    }
    for (int32_t k = 0; k < batch.n_tokens; ++k) {
        if (batch.n_seq_id[k] != 1 || !batch.seq_id[k] ||
            batch.seq_id[k][0] < 0 ||
            static_cast<size_t>(batch.seq_id[k][0]) >= n_seq) {
                return false;
        }
    }
    return true;
}

static void llama_rs_assign_tokens(
    std::vector<llama_token> & dst,
    const llama_token * tokens,
    size_t count) {
    if (count == 0) {
        dst.clear();
        return;
    }
    dst.assign(tokens, tokens + count);
}

static llama_rs_status llama_rs_speculative_remove_memories(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    llama_pos target_p0,
    llama_pos target_p1,
    llama_pos draft_p0,
    llama_pos draft_p1,
    char ** out_error) {
    const bool target = llama_memory_seq_rm(
        llama_get_memory(spec->params.draft.ctx_tgt), seq_id, target_p0, target_p1);
    const bool draft = llama_memory_seq_rm(
        llama_get_memory(spec->params.draft.ctx_dft), seq_id, draft_p0, draft_p1);
    if (!target && !draft) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_EXCEPTION, "target and draft memories rejected the speculative sequence range");
    }
    if (!target) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_EXCEPTION, "target memory rejected the speculative sequence range");
    }
    if (!draft) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_EXCEPTION, "draft memory rejected the speculative sequence range");
    }
    return LLAMA_RS_STATUS_OK;
}

extern "C" struct llama_rs_speculative * llama_rs_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    enum llama_rs_speculative_method method,
    int32_t n_max,
    int32_t n_min,
    float p_min,
    uint32_t n_seq,
    bool probe_rollback) {
    if (!ctx_tgt || !ctx_dft || n_max <= 0 || n_min < 0 || n_min > n_max || n_seq == 0) {
        return nullptr;
    }

    try {
        auto wrapper = std::make_unique<llama_rs_speculative>();
        switch (method) {
            case LLAMA_RS_SPECULATIVE_METHOD_MTP:
                wrapper->params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
                break;
            case LLAMA_RS_SPECULATIVE_METHOD_DFLASH:
                wrapper->params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
                break;
            case LLAMA_RS_SPECULATIVE_METHOD_DSPARK:
                wrapper->params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
                break;
            default:
                return nullptr;
        }
        wrapper->params.draft.ctx_tgt = ctx_tgt;
        wrapper->params.draft.ctx_dft = ctx_dft;
        wrapper->params.draft.n_max = n_max;
        wrapper->params.draft.n_min = n_min;
        wrapper->params.draft.p_min = p_min;

        if (probe_rollback) {
            wrapper->target_remove_type = common_context_can_seq_rm(ctx_tgt);
            if (wrapper->target_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
                return nullptr;
            }
        }

        wrapper->sequences.resize(n_seq);
        wrapper->spec = common_speculative_init(wrapper->params, n_seq);
        if (!wrapper->spec) {
            return nullptr;
        }
        if (probe_rollback) {
            wrapper->draft_remove_type = common_context_can_seq_rm(ctx_dft);
            if (wrapper->draft_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
                common_speculative_free(wrapper->spec);
                wrapper->spec = nullptr;
                return nullptr;
            }
        }

        return wrapper.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void llama_rs_speculative_free(struct llama_rs_speculative * spec) {
    if (!spec) {
        return;
    }
    if (spec->spec) {
        common_speculative_free(spec->spec);
        spec->spec = nullptr;
    }
    delete spec;
}

extern "C" llama_rs_status llama_rs_speculative_begin(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || seq_id < 0 ||
        static_cast<size_t>(seq_id) >= spec->sequences.size() ||
        (!prompt_tokens && prompt_tokens_count > 0)) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid speculative begin arguments");
    }

    try {
        auto & sequence = spec->sequences[seq_id];
        llama_rs_assign_tokens(sequence.prompt, prompt_tokens, prompt_tokens_count);
        sequence.last_draft_len = 0;
        sequence.draft_pending = false;
        sequence.prepared = false;
        sequence.target_checkpoint.clear();
        sequence.draft_checkpoint.clear();
        sequence.target_checkpointed = false;
        sequence.draft_checkpointed = false;
        common_speculative_begin(spec->spec, seq_id, sequence.prompt);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_speculative_process(
    struct llama_rs_speculative * spec,
    const struct llama_batch * batch,
    const llama_pos * draft_positions,
    size_t draft_positions_count,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || !batch || !draft_positions ||
        draft_positions_count != static_cast<size_t>(batch->n_tokens)) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid speculative process arguments");
    }
    if (!llama_rs_speculative_batch_compatible(*batch, spec->sequences.size())) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "speculative process batch is incompatible with the configured sequences");
    }

    try {
        llama_batch draft_batch = *batch;
        draft_batch.pos = const_cast<llama_pos *>(draft_positions);
        if (!common_speculative_process(spec->spec, draft_batch)) {
            return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_EXCEPTION, "common_speculative_process returned false");
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_speculative_prepare_draft(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    llama_pos target_n_past,
    llama_pos draft_n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    int32_t n_max,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || (!prompt_tokens && prompt_tokens_count > 0) ||
        seq_id < 0 || static_cast<size_t>(seq_id) >= spec->sequences.size() ||
        target_n_past < 0 || draft_n_past < 0 || n_max <= 0 || n_max > spec->params.draft.n_max) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid speculative prepare-draft arguments");
    }

    try {
        auto & sequence = spec->sequences[seq_id];
        if (sequence.draft_pending || sequence.prepared) {
            return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_STATE, "speculative sequence already has a pending or prepared draft");
        }
        llama_rs_assign_tokens(sequence.prompt, prompt_tokens, prompt_tokens_count);
        sequence.draft.clear();
        sequence.last_draft_len = 0;
        sequence.target_checkpoint.clear();
        sequence.draft_checkpoint.clear();
        sequence.target_checkpointed = false;
        sequence.draft_checkpointed = false;
        sequence.target_checkpoint.update_pos(
            target_n_past,
            llama_memory_seq_pos_min(llama_get_memory(spec->params.draft.ctx_tgt), seq_id),
            llama_memory_seq_pos_max(llama_get_memory(spec->params.draft.ctx_tgt), seq_id));
        sequence.draft_checkpoint.update_pos(
            draft_n_past,
            llama_memory_seq_pos_min(llama_get_memory(spec->params.draft.ctx_dft), seq_id),
            llama_memory_seq_pos_max(llama_get_memory(spec->params.draft.ctx_dft), seq_id));
        if (spec->draft_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            sequence.draft_checkpoint.update_dft(
                spec->params.draft.ctx_dft,
                seq_id,
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            sequence.draft_checkpointed = true;
        }

        auto & params = common_speculative_get_draft_params(spec->spec, seq_id);
        params = {
            true,
            n_max,
            draft_n_past,
            id_last,
            &sequence.prompt,
            &sequence.draft,
        };
        sequence.prepared = true;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_speculative_draft(
    struct llama_rs_speculative * spec,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid speculative draft arguments");
    }
    try {
        common_speculative_draft(spec->spec);
        for (size_t seq_id = 0; seq_id < spec->sequences.size(); ++seq_id) {
            auto & sequence = spec->sequences[seq_id];
            if (sequence.prepared) {
                sequence.last_draft_len = sequence.draft.size();
                sequence.draft_pending = !sequence.draft.empty();
                sequence.prepared = false;

                if (spec->draft_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                    sequence.draft_checkpoint.load_dft(
                        spec->params.draft.ctx_dft,
                        static_cast<llama_seq_id>(seq_id),
                        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                if (!llama_memory_seq_rm(
                        llama_get_memory(spec->params.draft.ctx_dft),
                        static_cast<llama_seq_id>(seq_id),
                        sequence.draft_checkpoint.pos_max + 1,
                        -1)) {
                    return llama_rs_chat_set_error(
                        out_error,
                        LLAMA_RS_STATUS_EXCEPTION,
                        "draft memory rejected cleanup after speculative drafting");
                }

                if (sequence.draft_pending) {
                    const bool checkpoint_target =
                        spec->target_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                        (spec->target_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
                         sequence.draft.size() > llama_n_rs_seq(spec->params.draft.ctx_tgt));
                    const bool checkpoint_draft =
                        spec->draft_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
                        sequence.draft.size() > llama_n_rs_seq(spec->params.draft.ctx_dft);
                    if (checkpoint_target) {
                        sequence.target_checkpoint.update_tgt(
                            spec->params.draft.ctx_tgt,
                            static_cast<llama_seq_id>(seq_id),
                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        sequence.target_checkpointed = true;
                    }
                    if (checkpoint_draft) {
                        sequence.draft_checkpoint.update_dft(
                            spec->params.draft.ctx_dft,
                            static_cast<llama_seq_id>(seq_id),
                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        sequence.draft_checkpointed = true;
                    }
                } else {
                    sequence.target_checkpoint.clear();
                    sequence.draft_checkpoint.clear();
                    sequence.target_checkpointed = false;
                    sequence.draft_checkpointed = false;
                }
            }
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_speculative_get_draft(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || !out_tokens_count || seq_id < 0 ||
        static_cast<size_t>(seq_id) >= spec->sequences.size()) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid speculative get-draft arguments");
    }
    try {
        auto & sequence = spec->sequences[seq_id];
        *out_tokens_count = sequence.draft.size();
        if (sequence.draft.size() > out_tokens_capacity) {
            return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_ALLOCATION_FAILED, "speculative draft exceeds the output capacity");
        }
        if (!sequence.draft.empty() && !out_tokens) {
            return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "speculative draft output is null");
        }
        if (!sequence.draft.empty()) {
            std::memcpy(out_tokens, sequence.draft.data(), sequence.draft.size() * sizeof(llama_token));
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_speculative_resolve(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    size_t proposed_count,
    uint16_t accepted_count,
    llama_pos next_target_position,
    llama_pos next_draft_position,
    bool * out_replay,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || !out_replay || next_target_position < 0 ||
        next_draft_position < 0 || seq_id < 0 ||
        static_cast<size_t>(seq_id) >= spec->sequences.size()) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid speculative resolve arguments");
    }
    *out_replay = false;
    auto & sequence = spec->sequences[seq_id];
    if (!sequence.draft_pending || proposed_count == 0 ||
        proposed_count > sequence.last_draft_len || accepted_count > proposed_count) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_STATE, "speculative resolution does not match the pending draft");
    }

    try {
        const size_t rollback = proposed_count - accepted_count;
        const bool restore_target =
            spec->target_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
            (spec->target_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
             rollback > llama_n_rs_seq(spec->params.draft.ctx_tgt));

        if (rollback > 0 && restore_target) {
            if (!sequence.target_checkpointed) {
                return llama_rs_chat_set_error(
                    out_error,
                    LLAMA_RS_STATUS_INVALID_STATE,
                    "speculative target rollback requires a missing checkpoint");
            }
            sequence.target_checkpoint.load_tgt(
                spec->params.draft.ctx_tgt,
                seq_id,
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (sequence.draft_checkpointed) {
                sequence.draft_checkpoint.load_dft(
                    spec->params.draft.ctx_dft,
                    seq_id,
                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            const auto status = llama_rs_speculative_remove_memories(
                spec,
                seq_id,
                sequence.target_checkpoint.pos_max + 1,
                -1,
                sequence.draft_checkpoint.pos_max + 1,
                -1,
                out_error);
            if (status != LLAMA_RS_STATUS_OK) {
                return status;
            }
            *out_replay = true;
            return LLAMA_RS_STATUS_OK;
        }

        common_speculative_accept(spec->spec, seq_id, accepted_count);
        const auto status = llama_rs_speculative_remove_memories(
            spec, seq_id, next_target_position, -1, next_draft_position, -1, out_error);
        if (status != LLAMA_RS_STATUS_OK) {
            return status;
        }
        sequence.last_draft_len = 0;
        sequence.draft_pending = false;
        sequence.draft.clear();
        sequence.target_checkpoint.clear();
        sequence.draft_checkpoint.clear();
        sequence.target_checkpointed = false;
        sequence.draft_checkpointed = false;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_speculative_seq_rm(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    llama_pos target_p0,
    llama_pos target_p1,
    llama_pos draft_p0,
    llama_pos draft_p1,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || seq_id < 0 ||
        static_cast<size_t>(seq_id) >= spec->sequences.size() ||
        !spec->params.draft.ctx_tgt || !spec->params.draft.ctx_dft) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid speculative sequence-remove arguments");
    }
    try {
        return llama_rs_speculative_remove_memories(
            spec, seq_id, target_p0, target_p1, draft_p0, draft_p1, out_error);
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_speculative_state_size(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    size_t * out_size,
    bool * out_has_state,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || !out_size || !out_has_state || seq_id < 0 ||
        static_cast<size_t>(seq_id) >= spec->sequences.size()) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "invalid speculative state-size arguments");
    }
    if (spec->sequences[seq_id].draft_pending || spec->sequences[seq_id].prepared) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_STATE,
            "speculative state can only be captured at a stable prompt boundary");
    }
    try {
        std::vector<uint8_t> data;
        *out_has_state = common_speculative_get_state(spec->spec, seq_id, data);
        *out_size = data.size();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_speculative_state_get(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    uint8_t * out_data,
    size_t out_capacity,
    size_t * out_size,
    bool * out_has_state,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || !out_size || !out_has_state || seq_id < 0 ||
        static_cast<size_t>(seq_id) >= spec->sequences.size()) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "invalid speculative state-get arguments");
    }
    if (spec->sequences[seq_id].draft_pending || spec->sequences[seq_id].prepared) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_STATE,
            "speculative state can only be captured at a stable prompt boundary");
    }
    try {
        std::vector<uint8_t> data;
        *out_has_state = common_speculative_get_state(spec->spec, seq_id, data);
        *out_size = data.size();
        if (data.size() > out_capacity) {
            return llama_rs_chat_set_error(
                out_error,
                LLAMA_RS_STATUS_ALLOCATION_FAILED,
                "speculative state exceeds output capacity");
        }
        if (!data.empty() && !out_data) {
            return llama_rs_chat_set_error(
                out_error,
                LLAMA_RS_STATUS_INVALID_ARGUMENT,
                "speculative state output is null");
        }
        if (!data.empty()) {
            std::memcpy(out_data, data.data(), data.size());
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_speculative_state_set(
    struct llama_rs_speculative * spec,
    llama_seq_id seq_id,
    const uint8_t * data,
    size_t data_size,
    bool has_state,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || (data_size > 0 && !data) || seq_id < 0 ||
        static_cast<size_t>(seq_id) >= spec->sequences.size()) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "invalid speculative state-set arguments");
    }
    if (spec->sequences[seq_id].draft_pending || spec->sequences[seq_id].prepared) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_STATE,
            "speculative state can only be restored at a stable prompt boundary");
    }
    try {
        if (has_state) {
            std::vector<uint8_t> state;
            if (data_size > 0) {
                state.assign(data, data + data_size);
            }
            common_speculative_set_state(spec->spec, seq_id, state);
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}
