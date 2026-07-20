#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct llama_model;
struct llama_rs_chat_delta_batch;
struct llama_rs_chat_finish;
struct llama_rs_chat_parse_result;
struct llama_rs_chat_parser;
struct llama_rs_chat_prepared;
struct llama_rs_chat_stream;
struct llama_rs_chat_templates;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_chat_tool_choice {
    LLAMA_RS_CHAT_TOOL_CHOICE_AUTO = 0,
    LLAMA_RS_CHAT_TOOL_CHOICE_REQUIRED = 1,
    LLAMA_RS_CHAT_TOOL_CHOICE_NONE = 2,
} llama_rs_chat_tool_choice;

typedef enum llama_rs_chat_reasoning_format {
    LLAMA_RS_CHAT_REASONING_FORMAT_NONE = 0,
    LLAMA_RS_CHAT_REASONING_FORMAT_AUTO = 1,
    LLAMA_RS_CHAT_REASONING_FORMAT_DEEPSEEK_LEGACY = 2,
    LLAMA_RS_CHAT_REASONING_FORMAT_DEEPSEEK = 3,
} llama_rs_chat_reasoning_format;

typedef enum llama_rs_chat_continuation {
    LLAMA_RS_CHAT_CONTINUATION_NONE = 0,
    LLAMA_RS_CHAT_CONTINUATION_AUTO = 1,
    LLAMA_RS_CHAT_CONTINUATION_REASONING = 2,
    LLAMA_RS_CHAT_CONTINUATION_CONTENT = 3,
} llama_rs_chat_continuation;

typedef enum llama_rs_chat_format {
    LLAMA_RS_CHAT_FORMAT_CONTENT_ONLY = 0,
    LLAMA_RS_CHAT_FORMAT_PEG_SIMPLE = 1,
    LLAMA_RS_CHAT_FORMAT_PEG_NATIVE = 2,
    LLAMA_RS_CHAT_FORMAT_PEG_GEMMA4 = 3,
} llama_rs_chat_format;

typedef enum llama_rs_chat_grammar_trigger_type {
    LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TOKEN = 0,
    LLAMA_RS_CHAT_GRAMMAR_TRIGGER_WORD = 1,
    LLAMA_RS_CHAT_GRAMMAR_TRIGGER_PATTERN = 2,
    LLAMA_RS_CHAT_GRAMMAR_TRIGGER_PATTERN_FULL = 3,
} llama_rs_chat_grammar_trigger_type;

typedef enum llama_rs_chat_prepared_string {
    LLAMA_RS_CHAT_PREPARED_PROMPT = 0,
    LLAMA_RS_CHAT_PREPARED_GRAMMAR = 1,
    LLAMA_RS_CHAT_PREPARED_GENERATION_PROMPT = 2,
    LLAMA_RS_CHAT_PREPARED_THINKING_START_TAG = 3,
    LLAMA_RS_CHAT_PREPARED_THINKING_END_TAG = 4,
    LLAMA_RS_CHAT_PREPARED_PARSER = 5,
} llama_rs_chat_prepared_string;

typedef enum llama_rs_chat_prepared_string_list {
    LLAMA_RS_CHAT_PREPARED_PRESERVED_TOKENS = 0,
    LLAMA_RS_CHAT_PREPARED_ADDITIONAL_STOPS = 1,
} llama_rs_chat_prepared_string_list;

typedef enum llama_rs_chat_result_string {
    LLAMA_RS_CHAT_RESULT_ROLE = 0,
    LLAMA_RS_CHAT_RESULT_CONTENT = 1,
    LLAMA_RS_CHAT_RESULT_REASONING_CONTENT = 2,
    LLAMA_RS_CHAT_RESULT_TOOL_NAME = 3,
    LLAMA_RS_CHAT_RESULT_TOOL_CALL_ID = 4,
} llama_rs_chat_result_string;

typedef struct llama_rs_chat_content_part_input {
    const char * type;
    const char * text;
} llama_rs_chat_content_part_input;

typedef struct llama_rs_chat_tool_call_input {
    const char * name;
    const char * arguments;
    const char * id;
} llama_rs_chat_tool_call_input;

typedef struct llama_rs_chat_message_input {
    const char * role;
    const char * content;
    const struct llama_rs_chat_content_part_input * content_parts;
    size_t content_parts_count;
    const struct llama_rs_chat_tool_call_input * tool_calls;
    size_t tool_calls_count;
    const char * reasoning_content;
    const char * tool_name;
    const char * tool_call_id;
} llama_rs_chat_message_input;

typedef struct llama_rs_chat_tool_input {
    const char * name;
    const char * description;
    const char * parameters_json;
} llama_rs_chat_tool_input;

typedef struct llama_rs_chat_template_kwarg_input {
    const char * key;
    const char * value_json;
} llama_rs_chat_template_kwarg_input;

typedef struct llama_rs_chat_prepare_options {
    const struct llama_rs_chat_message_input * messages;
    size_t messages_count;
    const char * grammar;
    const char * json_schema;
    bool add_generation_prompt;
    enum llama_rs_chat_continuation continuation;
    bool use_jinja;
    const struct llama_rs_chat_tool_input * tools;
    size_t tools_count;
    enum llama_rs_chat_tool_choice tool_choice;
    bool parallel_tool_calls_set;
    bool parallel_tool_calls;
    enum llama_rs_chat_reasoning_format reasoning_format;
    bool enable_thinking;
    const struct llama_rs_chat_template_kwarg_input * template_kwargs;
    size_t template_kwargs_count;
    bool force_pure_content;
} llama_rs_chat_prepare_options;

typedef struct llama_rs_chat_parser_options {
    bool reasoning_in_content;
    bool parse_tool_calls;
    bool is_continuation;
    bool echo;
    bool debug;
} llama_rs_chat_parser_options;

typedef struct llama_rs_chat_capabilities {
    bool supports_string_content;
    bool supports_typed_content;
    bool supports_tools;
    bool supports_tool_calls;
    bool supports_parallel_tool_calls;
    bool supports_system_role;
    bool supports_preserve_reasoning;
    bool supports_object_arguments;
    bool supports_enable_thinking;
} llama_rs_chat_capabilities;

typedef struct llama_rs_chat_grammar_trigger {
    enum llama_rs_chat_grammar_trigger_type type;
    char * value;
    llama_token token;
} llama_rs_chat_grammar_trigger;

// Borrowed projection into an owning llama_rs_chat_parse_result. Every byte
// view remains valid only until that parse-result owner is freed.
typedef struct llama_rs_chat_tool_call {
    struct llama_rs_bytes_view name;
    struct llama_rs_bytes_view arguments;
    struct llama_rs_bytes_view id;
} llama_rs_chat_tool_call;

typedef enum llama_rs_chat_delta_kind {
    LLAMA_RS_CHAT_DELTA_REASONING = 0,
    LLAMA_RS_CHAT_DELTA_CONTENT = 1,
    LLAMA_RS_CHAT_DELTA_TOOL_CALL = 2,
} llama_rs_chat_delta_kind;

// Borrowed projection into an owning llama_rs_chat_delta_batch. The byte views
// remain valid only until that batch is freed.
typedef struct llama_rs_chat_delta {
    enum llama_rs_chat_delta_kind kind;
    size_t tool_call_index;
    bool has_tool_call_id;
    bool has_tool_name;
    struct llama_rs_bytes_view tool_call_id;
    struct llama_rs_bytes_view tool_name;
    struct llama_rs_bytes_view text;
} llama_rs_chat_delta;

llama_rs_status llama_rs_chat_templates_init(
    const struct llama_model * model,
    const char * template_override,
    const char * bos_token_override,
    const char * eos_token_override,
    struct llama_rs_chat_templates ** out_templates,
    char ** out_error);

void llama_rs_chat_templates_free(struct llama_rs_chat_templates * templates);

llama_rs_status llama_rs_chat_templates_source(
    const struct llama_rs_chat_templates * templates,
    const char * variant,
    char ** out_source,
    char ** out_error);

llama_rs_status llama_rs_chat_templates_capabilities(
    const struct llama_rs_chat_templates * templates,
    struct llama_rs_chat_capabilities * out_capabilities,
    char ** out_error);

bool llama_rs_chat_templates_was_explicit(const struct llama_rs_chat_templates * templates);

llama_rs_status llama_rs_chat_templates_prepare(
    const struct llama_rs_chat_templates * templates,
    const struct llama_rs_chat_prepare_options * options,
    struct llama_rs_chat_prepared ** out_prepared,
    char ** out_error);

void llama_rs_chat_prepared_free(struct llama_rs_chat_prepared * prepared);

enum llama_rs_chat_format llama_rs_chat_prepared_format(const struct llama_rs_chat_prepared * prepared);

enum llama_rs_chat_reasoning_format llama_rs_chat_prepared_reasoning_format(
    const struct llama_rs_chat_prepared * prepared);

bool llama_rs_chat_prepared_grammar_lazy(const struct llama_rs_chat_prepared * prepared);

bool llama_rs_chat_prepared_supports_thinking(const struct llama_rs_chat_prepared * prepared);

llama_rs_status llama_rs_chat_prepared_get_string(
    const struct llama_rs_chat_prepared * prepared,
    enum llama_rs_chat_prepared_string field,
    char ** out_value);

size_t llama_rs_chat_prepared_string_list_count(
    const struct llama_rs_chat_prepared * prepared,
    enum llama_rs_chat_prepared_string_list list);

llama_rs_status llama_rs_chat_prepared_string_list_get(
    const struct llama_rs_chat_prepared * prepared,
    enum llama_rs_chat_prepared_string_list list,
    size_t index,
    char ** out_value);

size_t llama_rs_chat_prepared_grammar_trigger_count(const struct llama_rs_chat_prepared * prepared);

llama_rs_status llama_rs_chat_prepared_grammar_trigger_get(
    const struct llama_rs_chat_prepared * prepared,
    size_t index,
    struct llama_rs_chat_grammar_trigger * out_trigger);

llama_rs_status llama_rs_chat_parser_init(
    const struct llama_rs_chat_prepared * prepared,
    const struct llama_rs_chat_parser_options * options,
    struct llama_rs_chat_parser ** out_parser,
    char ** out_error);

void llama_rs_chat_parser_free(struct llama_rs_chat_parser * parser);

llama_rs_status llama_rs_chat_parser_parse(
    const struct llama_rs_chat_parser * parser,
    const uint8_t * generated_utf8,
    size_t generated_utf8_len,
    bool is_partial,
    struct llama_rs_chat_parse_result ** out_result,
    char ** out_error);

llama_rs_status llama_rs_chat_stream_init(
    const struct llama_rs_chat_prepared * prepared,
    const struct llama_rs_chat_parser_options * options,
    struct llama_rs_chat_stream ** out_stream,
    char ** out_error);

llama_rs_status llama_rs_chat_stream_push(
    struct llama_rs_chat_stream * stream,
    const uint8_t * utf8,
    size_t utf8_len,
    struct llama_rs_chat_delta_batch ** out_batch,
    char ** out_error);

llama_rs_status llama_rs_chat_stream_finish(
    struct llama_rs_chat_stream * stream,
    struct llama_rs_chat_delta_batch ** out_batch,
    struct llama_rs_chat_parse_result ** out_final,
    char ** out_error);

// Two-phase finalization used by the safe Rust wrapper. Preparing does not
// mutate the stream. The borrowed batch and parse-result projections remain
// valid until the finish owner is freed. Committing succeeds only for the same
// stream revision that was prepared, and consumes that stream's terminal
// transition without consuming the finish owner.
llama_rs_status llama_rs_chat_stream_prepare_finish(
    struct llama_rs_chat_stream * stream,
    struct llama_rs_chat_finish ** out_finish,
    char ** out_error);

const struct llama_rs_chat_delta_batch * llama_rs_chat_finish_delta_batch(
    const struct llama_rs_chat_finish * finish);

const struct llama_rs_chat_parse_result * llama_rs_chat_finish_parse_result(
    const struct llama_rs_chat_finish * finish);

llama_rs_status llama_rs_chat_stream_commit_finish(
    struct llama_rs_chat_stream * stream,
    struct llama_rs_chat_finish * finish,
    char ** out_error);

void llama_rs_chat_finish_free(struct llama_rs_chat_finish * finish);

void llama_rs_chat_stream_free(struct llama_rs_chat_stream * stream);

void llama_rs_chat_delta_batch_free(struct llama_rs_chat_delta_batch * batch);

size_t llama_rs_chat_delta_batch_count(const struct llama_rs_chat_delta_batch * batch);

llama_rs_status llama_rs_chat_delta_batch_get(
    const struct llama_rs_chat_delta_batch * batch,
    size_t index,
    struct llama_rs_chat_delta * out_delta,
    char ** out_error);

void llama_rs_chat_parse_result_free(struct llama_rs_chat_parse_result * result);

llama_rs_status llama_rs_chat_parse_result_get_string(
    const struct llama_rs_chat_parse_result * result,
    enum llama_rs_chat_result_string field,
    struct llama_rs_bytes_view * out_value);

size_t llama_rs_chat_parse_result_tool_call_count(const struct llama_rs_chat_parse_result * result);

llama_rs_status llama_rs_chat_parse_result_tool_call_get(
    const struct llama_rs_chat_parse_result * result,
    size_t index,
    struct llama_rs_chat_tool_call * out_tool_call);

void llama_rs_chat_grammar_trigger_clear(struct llama_rs_chat_grammar_trigger * trigger);

#ifdef __cplusplus
}
#endif
