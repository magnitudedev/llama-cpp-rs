#pragma once

#include "llama.cpp/include/llama.h"
#include "llama.cpp/tools/mtmd/mtmd.h"
#include "wrapper_common_misc.h"
#include "wrapper_utils.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Evaluate every text or media subbatch on the target and immediately mirror the exact batch into
// the linked speculative context before advancing to the next subbatch.
llama_rs_status llama_rs_mtmd_eval_chunks_speculative(
    struct mtmd_context * context,
    struct llama_context * llama_context,
    struct llama_rs_speculative * speculative,
    const struct mtmd_input_chunks * chunks,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * out_new_n_past,
    int32_t * out_result,
    char ** out_error);

#ifdef __cplusplus
}
#endif
