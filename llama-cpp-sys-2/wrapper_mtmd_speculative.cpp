#include "wrapper_mtmd_speculative.h"

#include "llama.cpp/tools/mtmd/mtmd-helper.h"

#include <vector>

namespace {

struct llama_batch_owner {
    llama_batch value;

    explicit llama_batch_owner(int32_t n_batch) : value(llama_batch_init(n_batch, 0, 1)) {}
    ~llama_batch_owner() { llama_batch_free(value); }
};

struct speculative_callback_data {
    llama_rs_speculative * speculative;
    llama_pos draft_n_past;
    char ** out_error;
    llama_rs_status status = LLAMA_RS_STATUS_OK;
};

int32_t process_speculative_batch(llama_batch batch, void * user_data) {
    auto & data = *static_cast<speculative_callback_data *>(user_data);
    std::vector<llama_pos> draft_positions(static_cast<size_t>(batch.n_tokens));
    for (int32_t row = 0; row < batch.n_tokens; ++row) {
        draft_positions[static_cast<size_t>(row)] = data.draft_n_past++;
    }
    data.status = llama_rs_speculative_process(
        data.speculative,
        &batch,
        draft_positions.data(),
        draft_positions.size(),
        data.out_error);
    return data.status == LLAMA_RS_STATUS_OK ? 0 : 1;
}

int32_t eval_chunk_speculative(
    mtmd_context * context,
    llama_context * llama_context,
    const mtmd_input_chunk * chunk,
    llama_pos target_n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * new_target_n_past,
    speculative_callback_data & callback) {
    llama_batch_owner text(n_batch);

    const mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);
    if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens = 0;
        const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        for (size_t offset = 0; offset < n_tokens;) {
            text.value.n_tokens = 0;
            while (offset < n_tokens && text.value.n_tokens < n_batch) {
                const int32_t row = text.value.n_tokens++;
                text.value.token[row] = tokens[offset++];
                text.value.pos[row] = target_n_past++;
                text.value.n_seq_id[row] = 1;
                text.value.seq_id[row][0] = seq_id;
                text.value.logits[row] = false;
            }
            if (logits_last && offset == n_tokens) {
                text.value.logits[text.value.n_tokens - 1] = true;
            }
            int32_t result = llama_decode(llama_context, text.value);
            if (result != 0) {
                return result;
            }
            result = process_speculative_batch(text.value, &callback);
            if (result != 0) {
                return result;
            }
            *new_target_n_past = target_n_past;
        }
    } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        int32_t result = mtmd_encode_chunk(context, chunk);
        if (result != 0) {
            return result;
        }
        result = mtmd_helper_decode_image_chunk(
            context,
            llama_context,
            chunk,
            mtmd_get_output_embd(context),
            target_n_past,
            seq_id,
            n_batch,
            &target_n_past,
            process_speculative_batch,
            &callback);
        if (result != 0) {
            return result;
        }
        *new_target_n_past = target_n_past;
    } else {
        return -1;
    }
    return 0;
}

int32_t eval_chunks_speculative(
    mtmd_context * context,
    llama_context * llama_context,
    const mtmd_input_chunks * chunks,
    llama_pos target_n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * new_target_n_past,
    speculative_callback_data & callback) {
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t chunk_index = 0; chunk_index < n_chunks; ++chunk_index) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, chunk_index);
        const bool chunk_logits_last = logits_last && chunk_index + 1 == n_chunks;
        const int32_t result = eval_chunk_speculative(
            context,
            llama_context,
            chunk,
            target_n_past,
            seq_id,
            n_batch,
            chunk_logits_last,
            &target_n_past,
            callback);
        if (result != 0) {
            return result;
        }
        *new_target_n_past = target_n_past;
    }
    return 0;
}

} // namespace

extern "C" llama_rs_status llama_rs_mtmd_eval_chunks_speculative(
    struct mtmd_context * context,
    struct llama_context * llama_context,
    struct llama_rs_speculative * speculative,
    const struct mtmd_input_chunks * chunks,
    llama_pos target_n_past,
    llama_pos draft_n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * out_new_target_n_past,
    llama_pos * out_new_draft_n_past,
    int32_t * out_result,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!context || !llama_context || !speculative || !chunks ||
        !out_new_target_n_past || !out_new_draft_n_past || !out_result || n_batch <= 0 ||
        target_n_past < 0 || draft_n_past < 0) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "multimodal speculative evaluation arguments are invalid");
    }

    try {
        llama_pos new_target_n_past = target_n_past;
        speculative_callback_data callback { speculative, draft_n_past, out_error };
        *out_result = eval_chunks_speculative(
            context,
            llama_context,
            chunks,
            target_n_past,
            seq_id,
            n_batch,
            logits_last,
            &new_target_n_past,
            callback);
        *out_new_target_n_past = new_target_n_past;
        *out_new_draft_n_past = callback.draft_n_past;
        return callback.status;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_eval_chunk_speculative(
    struct mtmd_context * context,
    struct llama_context * llama_context,
    struct llama_rs_speculative * speculative,
    const struct mtmd_input_chunk * chunk,
    llama_pos target_n_past,
    llama_pos draft_n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * out_new_target_n_past,
    llama_pos * out_new_draft_n_past,
    int32_t * out_result,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!context || !llama_context || !speculative || !chunk ||
        !out_new_target_n_past || !out_new_draft_n_past || !out_result || n_batch <= 0 ||
        target_n_past < 0 || draft_n_past < 0) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "multimodal speculative chunk evaluation arguments are invalid");
    }

    try {
        llama_pos new_target_n_past = target_n_past;
        speculative_callback_data callback { speculative, draft_n_past, out_error };
        *out_result = eval_chunk_speculative(
            context,
            llama_context,
            chunk,
            target_n_past,
            seq_id,
            n_batch,
            logits_last,
            &new_target_n_past,
            callback);
        *out_new_target_n_past = new_target_n_past;
        *out_new_draft_n_past = callback.draft_n_past;
        return callback.status;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}
