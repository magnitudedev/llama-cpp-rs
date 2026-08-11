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

#include <nlohmann/json.hpp>

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
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


struct llama_rs_mtp_speculative {
    struct sequence_state {
        std::vector<llama_token> prompt;
        std::vector<llama_token> draft;
        common_prompt_checkpoint checkpoint;
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

static bool llama_rs_mtp_batch_compatible(
    const struct llama_batch & batch,
    size_t n_seq) {
    if (batch.n_tokens <= 0 || !batch.token || batch.embd || !batch.pos || !batch.n_seq_id ||
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

static llama_rs_status llama_rs_mtp_remove_memories(
    struct llama_rs_mtp_speculative * spec,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1,
    char ** out_error) {
    const bool target = llama_memory_seq_rm(
        llama_get_memory(spec->params.draft.ctx_tgt), seq_id, p0, p1);
    const bool draft = llama_memory_seq_rm(
        llama_get_memory(spec->params.draft.ctx_dft), seq_id, p0, p1);
    if (!target && !draft) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_EXCEPTION, "target and draft memories rejected the MTP sequence range");
    }
    if (!target) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_EXCEPTION, "target memory rejected the MTP sequence range");
    }
    if (!draft) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_EXCEPTION, "draft memory rejected the MTP sequence range");
    }
    return LLAMA_RS_STATUS_OK;
}

extern "C" struct llama_rs_mtp_speculative * llama_rs_mtp_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min,
    uint32_t n_seq,
    bool probe_rollback) {
    if (!ctx_tgt || !ctx_dft || n_max <= 0 || n_min < 0 || n_min > n_max || n_seq == 0) {
        return nullptr;
    }

    try {
        auto wrapper = std::make_unique<llama_rs_mtp_speculative>();
        wrapper->params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
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

extern "C" void llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative * spec) {
    if (!spec) {
        return;
    }
    if (spec->spec) {
        common_speculative_free(spec->spec);
        spec->spec = nullptr;
    }
    delete spec;
}

extern "C" llama_rs_status llama_rs_mtp_speculative_begin(
    struct llama_rs_mtp_speculative * spec,
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
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid MTP begin arguments");
    }

    try {
        auto & sequence = spec->sequences[seq_id];
        llama_rs_assign_tokens(sequence.prompt, prompt_tokens, prompt_tokens_count);
        sequence.last_draft_len = 0;
        sequence.draft_pending = false;
        sequence.prepared = false;
        sequence.checkpoint.clear();
        common_speculative_begin(spec->spec, seq_id, sequence.prompt);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_process(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_batch * batch,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || !batch) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid MTP process arguments");
    }
    if (!llama_rs_mtp_batch_compatible(*batch, spec->sequences.size())) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "MTP process batch is incompatible with the configured sequences");
    }

    try {
        if (!common_speculative_process(spec->spec, *batch)) {
            return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_EXCEPTION, "common_speculative_process returned false");
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_prepare_draft(
    struct llama_rs_mtp_speculative * spec,
    llama_seq_id seq_id,
    llama_pos n_past,
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
        n_past < 0 || n_max <= 0 || n_max > spec->params.draft.n_max) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid MTP prepare-draft arguments");
    }

    try {
        auto & sequence = spec->sequences[seq_id];
        if (sequence.draft_pending || sequence.prepared) {
            return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_STATE, "MTP sequence already has a pending or prepared draft");
        }
        llama_rs_assign_tokens(sequence.prompt, prompt_tokens, prompt_tokens_count);
        sequence.draft.clear();
        sequence.last_draft_len = 0;
        sequence.checkpoint.clear();
        sequence.checkpoint.update_pos(
            n_past,
            llama_memory_seq_pos_min(llama_get_memory(spec->params.draft.ctx_tgt), seq_id),
            llama_memory_seq_pos_max(llama_get_memory(spec->params.draft.ctx_tgt), seq_id));
        if (spec->draft_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            sequence.checkpoint.update_dft(
                spec->params.draft.ctx_dft,
                seq_id,
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }

        auto & params = common_speculative_get_draft_params(spec->spec, seq_id);
        params = {
            true,
            n_max,
            n_past,
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

extern "C" llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative * spec,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid MTP draft arguments");
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
                    sequence.checkpoint.load_dft(
                        spec->params.draft.ctx_dft,
                        static_cast<llama_seq_id>(seq_id),
                        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                if (!llama_memory_seq_rm(
                        llama_get_memory(spec->params.draft.ctx_dft),
                        static_cast<llama_seq_id>(seq_id),
                        sequence.checkpoint.pos_max + 1,
                        -1)) {
                    return llama_rs_chat_set_error(
                        out_error,
                        LLAMA_RS_STATUS_EXCEPTION,
                        "draft memory rejected cleanup after MTP drafting");
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
                        sequence.checkpoint.update_tgt(
                            spec->params.draft.ctx_tgt,
                            static_cast<llama_seq_id>(seq_id),
                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    }
                    if (checkpoint_draft) {
                        sequence.checkpoint.update_dft(
                            spec->params.draft.ctx_dft,
                            static_cast<llama_seq_id>(seq_id),
                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    }
                } else {
                    sequence.checkpoint.clear();
                }
            }
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_get_draft(
    struct llama_rs_mtp_speculative * spec,
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
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid MTP get-draft arguments");
    }
    try {
        auto & sequence = spec->sequences[seq_id];
        *out_tokens_count = sequence.draft.size();
        if (sequence.draft.size() > out_tokens_capacity) {
            return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_ALLOCATION_FAILED, "MTP draft exceeds the output capacity");
        }
        if (!sequence.draft.empty() && !out_tokens) {
            return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "MTP draft output is null");
        }
        if (!sequence.draft.empty()) {
            std::memcpy(out_tokens, sequence.draft.data(), sequence.draft.size() * sizeof(llama_token));
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_resolve(
    struct llama_rs_mtp_speculative * spec,
    llama_seq_id seq_id,
    size_t proposed_count,
    uint16_t accepted_count,
    llama_pos next_position,
    bool * out_replay,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || !out_replay || next_position < 0 || seq_id < 0 ||
        static_cast<size_t>(seq_id) >= spec->sequences.size()) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid MTP resolve arguments");
    }
    *out_replay = false;
    auto & sequence = spec->sequences[seq_id];
    if (!sequence.draft_pending || proposed_count == 0 ||
        proposed_count > sequence.last_draft_len || accepted_count > proposed_count) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_STATE, "MTP resolution does not match the pending draft");
    }

    try {
        const size_t rollback = proposed_count - accepted_count;
        const bool restore_target =
            spec->target_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
            (spec->target_remove_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
             rollback > llama_n_rs_seq(spec->params.draft.ctx_tgt));

        if (rollback > 0 && restore_target) {
            if (sequence.checkpoint.empty()) {
                return llama_rs_chat_set_error(
                    out_error,
                    LLAMA_RS_STATUS_INVALID_STATE,
                    "MTP target rollback requires a missing checkpoint");
            }
            sequence.checkpoint.load_tgt(
                spec->params.draft.ctx_tgt,
                seq_id,
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            sequence.checkpoint.load_dft(
                spec->params.draft.ctx_dft,
                seq_id,
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            const auto status = llama_rs_mtp_remove_memories(
                spec, seq_id, sequence.checkpoint.pos_max + 1, -1, out_error);
            if (status != LLAMA_RS_STATUS_OK) {
                return status;
            }
            *out_replay = true;
            return LLAMA_RS_STATUS_OK;
        }

        common_speculative_accept(spec->spec, seq_id, accepted_count);
        const auto status = llama_rs_mtp_remove_memories(
            spec, seq_id, next_position, -1, out_error);
        if (status != LLAMA_RS_STATUS_OK) {
            return status;
        }
        sequence.last_draft_len = 0;
        sequence.draft_pending = false;
        sequence.draft.clear();
        sequence.checkpoint.clear();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_seq_rm(
    struct llama_rs_mtp_speculative * spec,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!spec || !spec->spec || seq_id < 0 ||
        static_cast<size_t>(seq_id) >= spec->sequences.size() ||
        !spec->params.draft.ctx_tgt || !spec->params.draft.ctx_dft) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "invalid MTP sequence-remove arguments");
    }
    try {
        return llama_rs_mtp_remove_memories(spec, seq_id, p0, p1, out_error);
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}
