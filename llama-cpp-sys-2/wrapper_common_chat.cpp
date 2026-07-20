#include "wrapper_common_chat.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <stdint.h>
#include <utility>
#include <vector>

#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

static_assert(
    static_cast<int>(LLAMA_RS_CHAT_FORMAT_CONTENT_ONLY) ==
    static_cast<int>(COMMON_CHAT_FORMAT_CONTENT_ONLY));
static_assert(
    static_cast<int>(LLAMA_RS_CHAT_FORMAT_PEG_SIMPLE) ==
    static_cast<int>(COMMON_CHAT_FORMAT_PEG_SIMPLE));
static_assert(
    static_cast<int>(LLAMA_RS_CHAT_FORMAT_PEG_NATIVE) ==
    static_cast<int>(COMMON_CHAT_FORMAT_PEG_NATIVE));
static_assert(
    static_cast<int>(LLAMA_RS_CHAT_FORMAT_PEG_GEMMA4) ==
    static_cast<int>(COMMON_CHAT_FORMAT_PEG_GEMMA4));

static_assert(
    static_cast<int>(LLAMA_RS_CHAT_REASONING_FORMAT_NONE) ==
    static_cast<int>(COMMON_REASONING_FORMAT_NONE));
static_assert(
    static_cast<int>(LLAMA_RS_CHAT_REASONING_FORMAT_AUTO) ==
    static_cast<int>(COMMON_REASONING_FORMAT_AUTO));
static_assert(
    static_cast<int>(LLAMA_RS_CHAT_REASONING_FORMAT_DEEPSEEK_LEGACY) ==
    static_cast<int>(COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY));
static_assert(
    static_cast<int>(LLAMA_RS_CHAT_REASONING_FORMAT_DEEPSEEK) ==
    static_cast<int>(COMMON_REASONING_FORMAT_DEEPSEEK));

static_assert(
    static_cast<int>(LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TOKEN) ==
    static_cast<int>(COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN));
static_assert(
    static_cast<int>(LLAMA_RS_CHAT_GRAMMAR_TRIGGER_WORD) ==
    static_cast<int>(COMMON_GRAMMAR_TRIGGER_TYPE_WORD));
static_assert(
    static_cast<int>(LLAMA_RS_CHAT_GRAMMAR_TRIGGER_PATTERN) ==
    static_cast<int>(COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN));
static_assert(
    static_cast<int>(LLAMA_RS_CHAT_GRAMMAR_TRIGGER_PATTERN_FULL) ==
    static_cast<int>(COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL));

struct llama_rs_chat_templates {
    common_chat_templates_ptr value;

    explicit llama_rs_chat_templates(common_chat_templates_ptr templates)
        : value(std::move(templates)) {
    }
};

struct llama_rs_chat_prepared {
    common_chat_params value;
    common_reasoning_format reasoning_format;

    llama_rs_chat_prepared(common_chat_params params, common_reasoning_format format)
        : value(std::move(params)), reasoning_format(format) {
    }
};

struct llama_rs_chat_parser {
    common_chat_parser_params value;

    explicit llama_rs_chat_parser(const common_chat_params & prepared)
        : value(prepared) {
    }
};

struct llama_rs_chat_parse_result {
    common_chat_msg value;

    explicit llama_rs_chat_parse_result(common_chat_msg message)
        : value(std::move(message)) {
    }
};

struct llama_rs_chat_delta_storage {
    llama_rs_chat_delta_kind kind = LLAMA_RS_CHAT_DELTA_CONTENT;
    size_t tool_call_index = 0;
    bool has_tool_call_id = false;
    bool has_tool_name = false;
    std::string tool_call_id;
    std::string tool_name;
    std::string text;
};

struct llama_rs_chat_delta_batch {
    std::vector<llama_rs_chat_delta_storage> values;
};

struct llama_rs_chat_stream_state {
    std::string generated_text;
    common_chat_msg previous_message;
};

struct llama_rs_chat_stream {
    common_chat_parser_params parser;
    std::unique_ptr<llama_rs_chat_stream_state> current;
    size_t revision = 0;
    bool finished = false;

    explicit llama_rs_chat_stream(const common_chat_params & prepared)
        : parser(prepared), current(std::make_unique<llama_rs_chat_stream_state>()) {
    }
};

struct llama_rs_chat_finish {
    llama_rs_chat_stream * source;
    size_t source_revision;
    std::unique_ptr<llama_rs_chat_stream_state> next;
    std::unique_ptr<llama_rs_chat_delta_batch> batch;
    std::unique_ptr<llama_rs_chat_parse_result> final_result;
    bool committed = false;

    llama_rs_chat_finish(
        llama_rs_chat_stream * stream,
        std::unique_ptr<llama_rs_chat_stream_state> next_state,
        std::unique_ptr<llama_rs_chat_delta_batch> delta_batch,
        std::unique_ptr<llama_rs_chat_parse_result> parsed_result)
        : source(stream),
          source_revision(stream->revision),
          next(std::move(next_state)),
          batch(std::move(delta_batch)),
          final_result(std::move(parsed_result)) {
    }
};

static bool llama_rs_chat_is_valid_utf8(const uint8_t * data, size_t len) {
    size_t index = 0;
    while (index < len) {
        const uint8_t first = data[index++];
        if (first <= 0x7f) {
            continue;
        }

        if (first >= 0xc2 && first <= 0xdf) {
            if (index >= len || data[index] < 0x80 || data[index] > 0xbf) {
                return false;
            }
            ++index;
            continue;
        }

        if (first >= 0xe0 && first <= 0xef) {
            if (index + 1 >= len) {
                return false;
            }
            const uint8_t second = data[index];
            const uint8_t third = data[index + 1];
            const bool second_valid = first == 0xe0
                ? second >= 0xa0 && second <= 0xbf
                : first == 0xed
                    ? second >= 0x80 && second <= 0x9f
                    : second >= 0x80 && second <= 0xbf;
            if (!second_valid || third < 0x80 || third > 0xbf) {
                return false;
            }
            index += 2;
            continue;
        }

        if (first >= 0xf0 && first <= 0xf4) {
            if (index + 2 >= len) {
                return false;
            }
            const uint8_t second = data[index];
            const uint8_t third = data[index + 1];
            const uint8_t fourth = data[index + 2];
            const bool second_valid = first == 0xf0
                ? second >= 0x90 && second <= 0xbf
                : first == 0xf4
                    ? second >= 0x80 && second <= 0x8f
                    : second >= 0x80 && second <= 0xbf;
            if (!second_valid ||
                third < 0x80 || third > 0xbf ||
                fourth < 0x80 || fourth > 0xbf) {
                return false;
            }
            index += 3;
            continue;
        }

        return false;
    }
    return true;
}

static bool llama_rs_chat_is_valid_utf8(const std::string & value) {
    return llama_rs_chat_is_valid_utf8(
        reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

static void llama_rs_chat_validate_message_utf8(const common_chat_msg & message) {
    auto validate = [](const std::string & value, const char * field) {
        if (!llama_rs_chat_is_valid_utf8(value)) {
            throw llama_rs_invalid_utf8_error(
                std::string("native chat parser returned invalid UTF-8 in ") + field);
        }
    };

    validate(message.role, "role");
    validate(message.content, "content");
    validate(message.reasoning_content, "reasoning_content");
    validate(message.tool_name, "tool_name");
    validate(message.tool_call_id, "tool_call_id");
    for (const auto & part : message.content_parts) {
        validate(part.type, "content part type");
        validate(part.text, "content part text");
    }
    for (const auto & call : message.tool_calls) {
        validate(call.id, "tool call id");
        validate(call.name, "tool call name");
        validate(call.arguments, "tool call arguments");
    }
}

static llama_rs_bytes_view llama_rs_chat_bytes_view(const std::string & value) {
    return {
        value.empty() ? nullptr : reinterpret_cast<const uint8_t *>(value.data()),
        value.size(),
    };
}

static std::unique_ptr<llama_rs_chat_delta_batch> llama_rs_chat_compute_delta_batch(
    const common_chat_msg & previous,
    const common_chat_msg & current) {
    const auto native_diffs = common_chat_msg_diff::compute_diffs(previous, current);
    auto result = std::make_unique<llama_rs_chat_delta_batch>();
    result->values.reserve(native_diffs.size());

    size_t diff_index = 0;
    if (previous.reasoning_content != current.reasoning_content) {
        if (diff_index >= native_diffs.size() ||
            native_diffs[diff_index].tool_call_index != std::string::npos) {
            throw std::runtime_error("unexpected native reasoning diff projection");
        }
        llama_rs_chat_delta_storage delta;
        delta.kind = LLAMA_RS_CHAT_DELTA_REASONING;
        delta.text = native_diffs[diff_index++].reasoning_content_delta;
        result->values.push_back(std::move(delta));
    }

    if (previous.content != current.content) {
        if (diff_index >= native_diffs.size() ||
            native_diffs[diff_index].tool_call_index != std::string::npos) {
            throw std::runtime_error("unexpected native content diff projection");
        }
        llama_rs_chat_delta_storage delta;
        delta.kind = LLAMA_RS_CHAT_DELTA_CONTENT;
        delta.text = native_diffs[diff_index++].content_delta;
        result->values.push_back(std::move(delta));
    }

    for (; diff_index < native_diffs.size(); ++diff_index) {
        const auto & native = native_diffs[diff_index];
        if (native.tool_call_index == std::string::npos) {
            throw std::runtime_error("unexpected native chat diff without a semantic field");
        }
        llama_rs_chat_delta_storage delta;
        delta.kind = LLAMA_RS_CHAT_DELTA_TOOL_CALL;
        delta.tool_call_index = native.tool_call_index;
        if (native.tool_call_index < previous.tool_calls.size()) {
            if (native.tool_call_index >= current.tool_calls.size()) {
                throw std::runtime_error("native tool-call diff index is out of range");
            }
            const auto & old_call = previous.tool_calls[native.tool_call_index];
            const auto & new_call = current.tool_calls[native.tool_call_index];
            delta.has_tool_call_id = old_call.id != new_call.id;
            delta.has_tool_name = old_call.name != new_call.name;
            if (delta.has_tool_call_id) {
                delta.tool_call_id = new_call.id;
            }
            if (delta.has_tool_name) {
                delta.tool_name = new_call.name;
            }
        } else {
            // A newly introduced common_chat_tool_call always has a semantic
            // name field, even while the incremental parser can only expose an
            // empty partial value. Keep that distinct from later deltas that do
            // not carry a name update. Upstream represents an absent optional
            // ID and an explicitly empty ID with the same empty std::string, so
            // only non-empty IDs can truthfully be marked present here.
            delta.has_tool_call_id = !native.tool_call_delta.id.empty();
            delta.has_tool_name = true;
            delta.tool_call_id = native.tool_call_delta.id;
            delta.tool_name = native.tool_call_delta.name;
        }
        delta.text = native.tool_call_delta.arguments;
        result->values.push_back(std::move(delta));
    }

    for (const auto & delta : result->values) {
        if (!llama_rs_chat_is_valid_utf8(delta.tool_call_id) ||
            !llama_rs_chat_is_valid_utf8(delta.tool_name) ||
            !llama_rs_chat_is_valid_utf8(delta.text)) {
            throw llama_rs_invalid_utf8_error("native chat diff returned invalid UTF-8");
        }
    }
    return result;
}

static void llama_rs_chat_configure_parser(
    common_chat_parser_params & parser,
    const llama_rs_chat_prepared & prepared,
    const llama_rs_chat_parser_options & options) {
    parser.reasoning_format = prepared.reasoning_format;
    parser.reasoning_in_content = options.reasoning_in_content;
    parser.parse_tool_calls = options.parse_tool_calls;
    parser.is_continuation = options.is_continuation;
    parser.echo = options.echo;
    parser.debug = options.debug;
    if (!prepared.value.parser.empty()) {
        parser.parser.load(prepared.value.parser);
    }
}

static std::unique_ptr<llama_rs_chat_finish> llama_rs_chat_prepare_finish_impl(
    llama_rs_chat_stream & stream) {
    auto next = std::make_unique<llama_rs_chat_stream_state>(*stream.current);
    auto parsed = common_chat_parse(next->generated_text, false, stream.parser);
    llama_rs_chat_validate_message_utf8(parsed);

    std::unique_ptr<llama_rs_chat_delta_batch> batch;
    if (parsed.empty()) {
        batch = std::make_unique<llama_rs_chat_delta_batch>();
    } else {
        batch = llama_rs_chat_compute_delta_batch(next->previous_message, parsed);
        next->previous_message = std::move(parsed);
    }

    llama_rs_chat_validate_message_utf8(next->previous_message);
    auto final_result =
        std::make_unique<llama_rs_chat_parse_result>(next->previous_message);
    return std::make_unique<llama_rs_chat_finish>(
        &stream, std::move(next), std::move(batch), std::move(final_result));
}


static common_chat_tool_choice llama_rs_chat_convert_tool_choice(enum llama_rs_chat_tool_choice choice) {
    switch (choice) {
        case LLAMA_RS_CHAT_TOOL_CHOICE_AUTO:
            return COMMON_CHAT_TOOL_CHOICE_AUTO;
        case LLAMA_RS_CHAT_TOOL_CHOICE_REQUIRED:
            return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        case LLAMA_RS_CHAT_TOOL_CHOICE_NONE:
            return COMMON_CHAT_TOOL_CHOICE_NONE;
    }
    throw std::invalid_argument("invalid chat tool choice");
}

static common_reasoning_format llama_rs_chat_convert_reasoning_format(
    enum llama_rs_chat_reasoning_format format) {
    switch (format) {
        case LLAMA_RS_CHAT_REASONING_FORMAT_NONE:
            return COMMON_REASONING_FORMAT_NONE;
        case LLAMA_RS_CHAT_REASONING_FORMAT_AUTO:
            return COMMON_REASONING_FORMAT_AUTO;
        case LLAMA_RS_CHAT_REASONING_FORMAT_DEEPSEEK_LEGACY:
            return COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
        case LLAMA_RS_CHAT_REASONING_FORMAT_DEEPSEEK:
            return COMMON_REASONING_FORMAT_DEEPSEEK;
    }
    throw std::invalid_argument("invalid chat reasoning format");
}

static common_chat_continuation llama_rs_chat_convert_continuation(enum llama_rs_chat_continuation continuation) {
    switch (continuation) {
        case LLAMA_RS_CHAT_CONTINUATION_NONE:
            return COMMON_CHAT_CONTINUATION_NONE;
        case LLAMA_RS_CHAT_CONTINUATION_AUTO:
            return COMMON_CHAT_CONTINUATION_AUTO;
        case LLAMA_RS_CHAT_CONTINUATION_REASONING:
            return COMMON_CHAT_CONTINUATION_REASONING;
        case LLAMA_RS_CHAT_CONTINUATION_CONTENT:
            return COMMON_CHAT_CONTINUATION_CONTENT;
    }
    throw std::invalid_argument("invalid chat continuation mode");
}

extern "C" llama_rs_status llama_rs_chat_templates_init(
    const struct llama_model * model,
    const char * template_override,
    const char * bos_token_override,
    const char * eos_token_override,
    struct llama_rs_chat_templates ** out_templates,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!out_templates) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "out_templates must not be null");
    }
    *out_templates = nullptr;
    try {
        auto templates = common_chat_templates_init(
            model,
            llama_rs_chat_optional_string(template_override),
            llama_rs_chat_optional_string(bos_token_override),
            llama_rs_chat_optional_string(eos_token_override));
        auto wrapper = std::make_unique<llama_rs_chat_templates>(std::move(templates));
        *out_templates = wrapper.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" void llama_rs_chat_templates_free(struct llama_rs_chat_templates * templates) {
    delete templates;
}

extern "C" llama_rs_status llama_rs_chat_templates_source(
    const struct llama_rs_chat_templates * templates,
    const char * variant,
    char ** out_source,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!templates || !out_source) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, "templates and out_source must not be null");
    }
    *out_source = nullptr;
    try {
        const auto source = common_chat_templates_source(
            templates->value.get(), llama_rs_chat_optional_string(variant));
        *out_source = llama_rs_dup_string(source);
        return *out_source ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_chat_templates_capabilities(
    const struct llama_rs_chat_templates * templates,
    struct llama_rs_chat_capabilities * out_capabilities,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!templates || !out_capabilities) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "templates and out_capabilities must not be null");
    }

    try {
        const auto capabilities = common_chat_templates_get_caps(templates->value.get());
        *out_capabilities = {
            capabilities.at("supports_string_content"),
            capabilities.at("supports_typed_content"),
            capabilities.at("supports_tools"),
            capabilities.at("supports_tool_calls"),
            capabilities.at("supports_parallel_tool_calls"),
            capabilities.at("supports_system_role"),
            capabilities.at("supports_preserve_reasoning"),
            capabilities.at("supports_object_arguments"),
            common_chat_templates_support_enable_thinking(templates->value.get()),
        };
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" bool llama_rs_chat_templates_was_explicit(const struct llama_rs_chat_templates * templates) {
    return templates && common_chat_templates_was_explicit(templates->value.get());
}

extern "C" llama_rs_status llama_rs_chat_templates_prepare(
    const struct llama_rs_chat_templates * templates,
    const struct llama_rs_chat_prepare_options * options,
    struct llama_rs_chat_prepared ** out_prepared,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!templates || !options || !out_prepared) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "templates, options, and out_prepared must not be null");
    }
    *out_prepared = nullptr;

    try {
        llama_rs_chat_validate_array(options->messages, options->messages_count, "messages");
        llama_rs_chat_validate_array(options->tools, options->tools_count, "tools");
        llama_rs_chat_validate_array(
            options->template_kwargs, options->template_kwargs_count, "template_kwargs");

        common_chat_templates_inputs inputs;
        inputs.messages.reserve(options->messages_count);
        for (size_t i = 0; i < options->messages_count; ++i) {
            const auto & source = options->messages[i];
            if (!source.role) {
                throw std::invalid_argument("message role must not be null");
            }
            llama_rs_chat_validate_array(source.content_parts, source.content_parts_count, "content_parts");
            llama_rs_chat_validate_array(source.tool_calls, source.tool_calls_count, "tool_calls");
            if (source.content && source.content_parts_count > 0) {
                throw std::invalid_argument("message cannot contain both text content and content parts");
            }

            common_chat_msg message;
            message.role = source.role;
            if (source.content) {
                message.content = source.content;
            }
            message.content_parts.reserve(source.content_parts_count);
            for (size_t part_index = 0; part_index < source.content_parts_count; ++part_index) {
                const auto & part = source.content_parts[part_index];
                if (!part.type || !part.text) {
                    throw std::invalid_argument("content part type and text must not be null");
                }
                message.content_parts.push_back({part.type, part.text});
            }
            message.tool_calls.reserve(source.tool_calls_count);
            for (size_t call_index = 0; call_index < source.tool_calls_count; ++call_index) {
                const auto & call = source.tool_calls[call_index];
                if (!call.name || !call.arguments) {
                    throw std::invalid_argument("tool call name and arguments must not be null");
                }
                message.tool_calls.push_back({
                    call.name,
                    call.arguments,
                    llama_rs_chat_optional_string(call.id),
                });
            }
            message.reasoning_content = llama_rs_chat_optional_string(source.reasoning_content);
            message.tool_name = llama_rs_chat_optional_string(source.tool_name);
            message.tool_call_id = llama_rs_chat_optional_string(source.tool_call_id);
            inputs.messages.push_back(std::move(message));
        }

        inputs.grammar = llama_rs_chat_optional_string(options->grammar);
        inputs.json_schema = llama_rs_chat_optional_string(options->json_schema);
        inputs.add_generation_prompt = options->add_generation_prompt;
        inputs.continue_final_message = llama_rs_chat_convert_continuation(options->continuation);
        inputs.use_jinja = options->use_jinja;
        inputs.tools.reserve(options->tools_count);
        for (size_t i = 0; i < options->tools_count; ++i) {
            const auto & source = options->tools[i];
            if (!source.name || !source.parameters_json) {
                throw std::invalid_argument("tool name and parameters_json must not be null");
            }
            inputs.tools.push_back({
                source.name,
                llama_rs_chat_optional_string(source.description),
                source.parameters_json,
            });
        }
        inputs.tool_choice = llama_rs_chat_convert_tool_choice(options->tool_choice);
        inputs.parallel_tool_calls = options->parallel_tool_calls_set
            ? options->parallel_tool_calls
            : common_chat_templates_get_caps(templates->value.get()).at("supports_parallel_tool_calls");
        inputs.reasoning_format = llama_rs_chat_convert_reasoning_format(options->reasoning_format);
        inputs.enable_thinking = options->enable_thinking;
        for (size_t i = 0; i < options->template_kwargs_count; ++i) {
            const auto & kwarg = options->template_kwargs[i];
            if (!kwarg.key || !kwarg.value_json) {
                throw std::invalid_argument("template kwarg key and value_json must not be null");
            }
            inputs.chat_template_kwargs[kwarg.key] = kwarg.value_json;
        }
        inputs.force_pure_content = options->force_pure_content;

        auto reasoning_format = inputs.reasoning_format;
        auto prepared = common_chat_templates_apply(templates->value.get(), inputs);
        auto wrapper = std::make_unique<llama_rs_chat_prepared>(std::move(prepared), reasoning_format);
        *out_prepared = wrapper.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" void llama_rs_chat_prepared_free(struct llama_rs_chat_prepared * prepared) {
    delete prepared;
}

extern "C" enum llama_rs_chat_format llama_rs_chat_prepared_format(
    const struct llama_rs_chat_prepared * prepared) {
    return prepared
        ? static_cast<llama_rs_chat_format>(prepared->value.format)
        : LLAMA_RS_CHAT_FORMAT_CONTENT_ONLY;
}

extern "C" enum llama_rs_chat_reasoning_format llama_rs_chat_prepared_reasoning_format(
    const struct llama_rs_chat_prepared * prepared) {
    return prepared
        ? static_cast<llama_rs_chat_reasoning_format>(prepared->reasoning_format)
        : LLAMA_RS_CHAT_REASONING_FORMAT_NONE;
}

extern "C" bool llama_rs_chat_prepared_grammar_lazy(const struct llama_rs_chat_prepared * prepared) {
    return prepared && prepared->value.grammar_lazy;
}

extern "C" bool llama_rs_chat_prepared_supports_thinking(const struct llama_rs_chat_prepared * prepared) {
    return prepared && prepared->value.supports_thinking;
}

extern "C" llama_rs_status llama_rs_chat_prepared_get_string(
    const struct llama_rs_chat_prepared * prepared,
    enum llama_rs_chat_prepared_string field,
    char ** out_value) {
    if (!prepared || !out_value) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_value = nullptr;
    const std::string * value = nullptr;
    switch (field) {
        case LLAMA_RS_CHAT_PREPARED_PROMPT:
            value = &prepared->value.prompt;
            break;
        case LLAMA_RS_CHAT_PREPARED_GRAMMAR:
            value = &prepared->value.grammar;
            break;
        case LLAMA_RS_CHAT_PREPARED_GENERATION_PROMPT:
            value = &prepared->value.generation_prompt;
            break;
        case LLAMA_RS_CHAT_PREPARED_THINKING_START_TAG:
            value = &prepared->value.thinking_start_tag;
            break;
        case LLAMA_RS_CHAT_PREPARED_THINKING_END_TAG:
            value = &prepared->value.thinking_end_tag;
            break;
        case LLAMA_RS_CHAT_PREPARED_PARSER:
            value = &prepared->value.parser;
            break;
        default:
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_value = llama_rs_dup_string(*value);
    return *out_value ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
}

static const std::vector<std::string> * llama_rs_chat_prepared_string_list_value(
    const struct llama_rs_chat_prepared * prepared,
    enum llama_rs_chat_prepared_string_list list) {
    if (!prepared) {
        return nullptr;
    }
    switch (list) {
        case LLAMA_RS_CHAT_PREPARED_PRESERVED_TOKENS:
            return &prepared->value.preserved_tokens;
        case LLAMA_RS_CHAT_PREPARED_ADDITIONAL_STOPS:
            return &prepared->value.additional_stops;
    }
    return nullptr;
}

extern "C" size_t llama_rs_chat_prepared_string_list_count(
    const struct llama_rs_chat_prepared * prepared,
    enum llama_rs_chat_prepared_string_list list) {
    const auto * value = llama_rs_chat_prepared_string_list_value(prepared, list);
    return value ? value->size() : 0;
}

extern "C" llama_rs_status llama_rs_chat_prepared_string_list_get(
    const struct llama_rs_chat_prepared * prepared,
    enum llama_rs_chat_prepared_string_list list,
    size_t index,
    char ** out_value) {
    if (!out_value) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_value = nullptr;
    const auto * values = llama_rs_chat_prepared_string_list_value(prepared, list);
    if (!values || index >= values->size()) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_value = llama_rs_dup_string((*values)[index]);
    return *out_value ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
}

extern "C" size_t llama_rs_chat_prepared_grammar_trigger_count(
    const struct llama_rs_chat_prepared * prepared) {
    return prepared ? prepared->value.grammar_triggers.size() : 0;
}

extern "C" llama_rs_status llama_rs_chat_prepared_grammar_trigger_get(
    const struct llama_rs_chat_prepared * prepared,
    size_t index,
    struct llama_rs_chat_grammar_trigger * out_trigger) {
    if (!prepared || !out_trigger || index >= prepared->value.grammar_triggers.size()) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    out_trigger->value = nullptr;
    const auto & source = prepared->value.grammar_triggers[index];
    out_trigger->type = static_cast<llama_rs_chat_grammar_trigger_type>(source.type);
    out_trigger->token = source.token;
    out_trigger->value = llama_rs_dup_string(source.value);
    return out_trigger->value ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
}

extern "C" llama_rs_status llama_rs_chat_parser_init(
    const struct llama_rs_chat_prepared * prepared,
    const struct llama_rs_chat_parser_options * options,
    struct llama_rs_chat_parser ** out_parser,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!prepared || !options || !out_parser) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "prepared, options, and out_parser must not be null");
    }
    *out_parser = nullptr;
    try {
        auto parser = std::make_unique<llama_rs_chat_parser>(prepared->value);
        llama_rs_chat_configure_parser(parser->value, *prepared, *options);
        *out_parser = parser.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" void llama_rs_chat_parser_free(struct llama_rs_chat_parser * parser) {
    delete parser;
}

extern "C" llama_rs_status llama_rs_chat_parser_parse(
    const struct llama_rs_chat_parser * parser,
    const uint8_t * generated_utf8,
    size_t generated_utf8_len,
    bool is_partial,
    struct llama_rs_chat_parse_result ** out_result,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!parser || !out_result || (generated_utf8_len > 0 && !generated_utf8)) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "parser and out_result must not be null, and generated_utf8 must be non-null when generated_utf8_len is non-zero");
    }
    *out_result = nullptr;
    if (!llama_rs_chat_is_valid_utf8(generated_utf8, generated_utf8_len)) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_UTF8, "chat parser input is not valid UTF-8");
    }
    try {
        const auto generated_text = generated_utf8_len == 0
            ? std::string()
            : std::string(
                reinterpret_cast<const char *>(generated_utf8), generated_utf8_len);
        auto result = common_chat_parse(generated_text, is_partial, parser->value);
        llama_rs_chat_validate_message_utf8(result);
        auto wrapper = std::make_unique<llama_rs_chat_parse_result>(std::move(result));
        *out_result = wrapper.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_chat_stream_init(
    const struct llama_rs_chat_prepared * prepared,
    const struct llama_rs_chat_parser_options * options,
    struct llama_rs_chat_stream ** out_stream,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_stream) {
        *out_stream = nullptr;
    }
    if (!prepared || !options || !out_stream) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "prepared, options, and out_stream must not be null");
    }
    try {
        auto stream = std::make_unique<llama_rs_chat_stream>(prepared->value);
        llama_rs_chat_configure_parser(stream->parser, *prepared, *options);
        if (stream->parser.is_continuation && !stream->parser.echo) {
            auto initial = common_chat_parse("", true, stream->parser);
            llama_rs_chat_validate_message_utf8(initial);
            stream->current->previous_message = std::move(initial);
        }
        *out_stream = stream.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_chat_stream_push(
    struct llama_rs_chat_stream * stream,
    const uint8_t * utf8,
    size_t utf8_len,
    struct llama_rs_chat_delta_batch ** out_batch,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_batch) {
        *out_batch = nullptr;
    }
    if (!stream || !out_batch || (utf8_len > 0 && !utf8)) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "stream and out_batch must not be null, and utf8 must be non-null when utf8_len is non-zero");
    }
    if (stream->finished) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_STATE, "chat stream is already finished");
    }
    if (!llama_rs_chat_is_valid_utf8(utf8, utf8_len)) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_UTF8, "chat stream input is not valid UTF-8");
    }

    try {
        auto next = std::make_unique<llama_rs_chat_stream_state>(*stream->current);
        if (utf8_len > 0) {
            next->generated_text.append(reinterpret_cast<const char *>(utf8), utf8_len);
        }

        auto parsed = common_chat_parse(next->generated_text, true, stream->parser);
        std::unique_ptr<llama_rs_chat_delta_batch> batch;
        if (parsed.empty()) {
            batch = std::make_unique<llama_rs_chat_delta_batch>();
        } else {
            llama_rs_chat_validate_message_utf8(parsed);
            batch = llama_rs_chat_compute_delta_batch(next->previous_message, parsed);
            next->previous_message = std::move(parsed);
        }

        stream->current = std::move(next);
        ++stream->revision;
        *out_batch = batch.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_chat_stream_finish(
    struct llama_rs_chat_stream * stream,
    struct llama_rs_chat_delta_batch ** out_batch,
    struct llama_rs_chat_parse_result ** out_final,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_batch) {
        *out_batch = nullptr;
    }
    if (out_final) {
        *out_final = nullptr;
    }
    if (!stream || !out_batch || !out_final) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "stream, out_batch, and out_final must not be null");
    }
    if (stream->finished) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_STATE, "chat stream is already finished");
    }

    try {
        auto finish = llama_rs_chat_prepare_finish_impl(*stream);
        stream->current = std::move(finish->next);
        stream->finished = true;
        ++stream->revision;
        *out_batch = finish->batch.release();
        *out_final = finish->final_result.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_chat_stream_prepare_finish(
    struct llama_rs_chat_stream * stream,
    struct llama_rs_chat_finish ** out_finish,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_finish) {
        *out_finish = nullptr;
    }
    if (!stream || !out_finish) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "stream and out_finish must not be null");
    }
    if (stream->finished) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_STATE, "chat stream is already finished");
    }

    try {
        *out_finish = llama_rs_chat_prepare_finish_impl(*stream).release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" const struct llama_rs_chat_delta_batch * llama_rs_chat_finish_delta_batch(
    const struct llama_rs_chat_finish * finish) {
    return finish ? finish->batch.get() : nullptr;
}

extern "C" const struct llama_rs_chat_parse_result * llama_rs_chat_finish_parse_result(
    const struct llama_rs_chat_finish * finish) {
    return finish ? finish->final_result.get() : nullptr;
}

extern "C" llama_rs_status llama_rs_chat_stream_commit_finish(
    struct llama_rs_chat_stream * stream,
    struct llama_rs_chat_finish * finish,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!stream || !finish) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "stream and finish must not be null");
    }
    if (stream->finished) {
        return llama_rs_chat_set_error(
            out_error, LLAMA_RS_STATUS_INVALID_STATE, "chat stream is already finished");
    }
    if (finish->committed || finish->source != stream || !finish->next ||
        finish->source_revision != stream->revision) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_STATE,
            "prepared chat finish does not match the current stream revision");
    }

    stream->current = std::move(finish->next);
    stream->finished = true;
    ++stream->revision;
    finish->committed = true;
    return LLAMA_RS_STATUS_OK;
}

extern "C" void llama_rs_chat_finish_free(struct llama_rs_chat_finish * finish) {
    delete finish;
}

extern "C" void llama_rs_chat_stream_free(struct llama_rs_chat_stream * stream) {
    delete stream;
}

extern "C" void llama_rs_chat_delta_batch_free(struct llama_rs_chat_delta_batch * batch) {
    delete batch;
}

extern "C" size_t llama_rs_chat_delta_batch_count(
    const struct llama_rs_chat_delta_batch * batch) {
    return batch ? batch->values.size() : 0;
}

extern "C" llama_rs_status llama_rs_chat_delta_batch_get(
    const struct llama_rs_chat_delta_batch * batch,
    size_t index,
    struct llama_rs_chat_delta * out_delta,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!batch || !out_delta || index >= batch->values.size()) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "batch and out_delta must not be null and index must be in range");
    }

    const auto & source = batch->values[index];
    llama_rs_chat_delta projection = {
        source.kind,
        source.tool_call_index,
        source.has_tool_call_id,
        source.has_tool_name,
        llama_rs_chat_bytes_view(source.tool_call_id),
        llama_rs_chat_bytes_view(source.tool_name),
        llama_rs_chat_bytes_view(source.text),
    };
    *out_delta = projection;
    return LLAMA_RS_STATUS_OK;
}

extern "C" void llama_rs_chat_parse_result_free(struct llama_rs_chat_parse_result * result) {
    delete result;
}

extern "C" llama_rs_status llama_rs_chat_parse_result_get_string(
    const struct llama_rs_chat_parse_result * result,
    enum llama_rs_chat_result_string field,
    struct llama_rs_bytes_view * out_value) {
    if (!result || !out_value) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_value = {nullptr, 0};
    const std::string * value = nullptr;
    switch (field) {
        case LLAMA_RS_CHAT_RESULT_ROLE:
            value = &result->value.role;
            break;
        case LLAMA_RS_CHAT_RESULT_CONTENT:
            value = &result->value.content;
            break;
        case LLAMA_RS_CHAT_RESULT_REASONING_CONTENT:
            value = &result->value.reasoning_content;
            break;
        case LLAMA_RS_CHAT_RESULT_TOOL_NAME:
            value = &result->value.tool_name;
            break;
        case LLAMA_RS_CHAT_RESULT_TOOL_CALL_ID:
            value = &result->value.tool_call_id;
            break;
        default:
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_value = llama_rs_chat_bytes_view(*value);
    return LLAMA_RS_STATUS_OK;
}

extern "C" size_t llama_rs_chat_parse_result_tool_call_count(
    const struct llama_rs_chat_parse_result * result) {
    return result ? result->value.tool_calls.size() : 0;
}

extern "C" llama_rs_status llama_rs_chat_parse_result_tool_call_get(
    const struct llama_rs_chat_parse_result * result,
    size_t index,
    struct llama_rs_chat_tool_call * out_tool_call) {
    if (!result || !out_tool_call || index >= result->value.tool_calls.size()) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_tool_call = {
        {nullptr, 0},
        {nullptr, 0},
        {nullptr, 0},
    };
    const auto & source = result->value.tool_calls[index];
    out_tool_call->name = llama_rs_chat_bytes_view(source.name);
    out_tool_call->arguments = llama_rs_chat_bytes_view(source.arguments);
    out_tool_call->id = llama_rs_chat_bytes_view(source.id);
    return LLAMA_RS_STATUS_OK;
}

extern "C" void llama_rs_chat_grammar_trigger_clear(struct llama_rs_chat_grammar_trigger * trigger) {
    if (trigger) {
        llama_rs_string_free(trigger->value);
        trigger->value = nullptr;
    }
}
