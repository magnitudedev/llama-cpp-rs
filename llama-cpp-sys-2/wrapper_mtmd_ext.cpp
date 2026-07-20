#include "wrapper_mtmd_ext.h"

#include "llama.cpp/ggml/include/ggml-backend.h"
#include "llama.cpp/tools/mtmd/clip.h"
#include "llama.cpp/tools/mtmd/mtmd-helper.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

// Pinned `mtmd_get_memory_usage` temporarily swaps the process-global mtmd logger while loading a
// projector. Serialize every bridge entry to that unstable helper so concurrent callers cannot
// restore each other's callback/user-data pair out of order.
static std::mutex g_mtmd_memory_usage_mutex;

struct mtmd_memory_entry {
    std::optional<size_t> device_index;
    int32_t backend_type;
    std::string name;
    std::string description;
    uint64_t bytes;
};

static std::string nullable_string(const char * value) {
    return value ? std::string(value) : std::string();
}

static llama_rs_bytes_view bytes_view(const std::string & value) {
    return {
        value.empty() ? nullptr : reinterpret_cast<const uint8_t *>(value.data()),
        value.size(),
    };
}

static std::optional<size_t> registered_device_index(ggml_backend_dev_t device) {
    const size_t count = ggml_backend_dev_count();
    for (size_t index = 0; index < count; ++index) {
        if (ggml_backend_dev_get(index) == device) {
            return index;
        }
    }
    return std::nullopt;
}

static mtmd_memory_entry project_memory_entry(ggml_backend_dev_t device, size_t bytes) {
    if (!device) {
        throw std::runtime_error("mtmd memory report contained a null backend device");
    }
    if constexpr (sizeof(size_t) > sizeof(uint64_t)) {
        if (bytes > static_cast<size_t>(std::numeric_limits<uint64_t>::max())) {
            throw std::overflow_error("mtmd memory estimate does not fit in uint64_t");
        }
    }
    return {
        registered_device_index(device),
        static_cast<int32_t>(ggml_backend_dev_type(device)),
        nullable_string(ggml_backend_dev_name(device)),
        nullable_string(ggml_backend_dev_description(device)),
        static_cast<uint64_t>(bytes),
    };
}

} // namespace

struct llama_rs_mtmd_memory_report {
    std::vector<mtmd_memory_entry> devices;
};

extern "C" llama_rs_status llama_rs_mtmd_context_init_from_file(
    const char * path,
    const struct llama_model * model,
    const struct mtmd_context_params * params,
    struct mtmd_context ** out_context,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_context) {
        *out_context = nullptr;
    }
    if (!path || !model || !params || !out_context) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "path, model, params, and out_context must not be null");
    }

    try {
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> context(
            mtmd_init_from_file(path, model, *params),
            &mtmd_free);
        *out_context = context.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_bitmap_init_image(
    uint32_t width,
    uint32_t height,
    const uint8_t * data,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_bitmap) {
        *out_bitmap = nullptr;
    }
    if (!out_bitmap) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "out_bitmap must not be null");
    }

    try {
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
            mtmd_bitmap_init(width, height, data),
            &mtmd_bitmap_free);
        *out_bitmap = bitmap.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_bitmap_init_audio(
    size_t sample_count,
    const float * data,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_bitmap) {
        *out_bitmap = nullptr;
    }
    if (!out_bitmap) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "out_bitmap must not be null");
    }
    if (sample_count > std::numeric_limits<uint32_t>::max()) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "audio sample count exceeds mtmd's uint32_t representation");
    }

    try {
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
            mtmd_bitmap_init_from_audio(sample_count, data),
            &mtmd_bitmap_free);
        *out_bitmap = bitmap.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_bitmap_init_from_file(
    struct mtmd_context * context,
    const char * path,
    bool placeholder,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_bitmap) {
        *out_bitmap = nullptr;
    }
    if (!context || !path || !out_bitmap) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "context, path, and out_bitmap must not be null");
    }

    try {
        const auto wrapped = mtmd_helper_bitmap_init_from_file(context, path, placeholder);
        if (wrapped.video_ctx) {
            mtmd_bitmap_free(wrapped.bitmap);
            mtmd_helper_video_free(wrapped.video_ctx);
            throw std::runtime_error("video bitmaps are not enabled by this binding");
        }
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
            wrapped.bitmap,
            &mtmd_bitmap_free);
        *out_bitmap = bitmap.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_bitmap_init_from_buffer(
    struct mtmd_context * context,
    const uint8_t * data,
    size_t data_len,
    bool placeholder,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_bitmap) {
        *out_bitmap = nullptr;
    }
    if (!context || !out_bitmap || (data_len > 0 && !data)) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "context and out_bitmap must not be null; data may only be null when empty");
    }

    try {
        const auto wrapped = mtmd_helper_bitmap_init_from_buf(
            context, data, data_len, placeholder);
        if (wrapped.video_ctx) {
            mtmd_bitmap_free(wrapped.bitmap);
            mtmd_helper_video_free(wrapped.video_ctx);
            throw std::runtime_error("video bitmaps are not enabled by this binding");
        }
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
            wrapped.bitmap,
            &mtmd_bitmap_free);
        *out_bitmap = bitmap.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_bitmap_set_id(
    struct mtmd_bitmap * bitmap,
    const char * id,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!bitmap) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "bitmap must not be null");
    }

    try {
        mtmd_bitmap_set_id(bitmap, id);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_input_chunks_init(
    struct mtmd_input_chunks ** out_chunks,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_chunks) {
        *out_chunks = nullptr;
    }
    if (!out_chunks) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "out_chunks must not be null");
    }

    try {
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
            mtmd_input_chunks_init(),
            &mtmd_input_chunks_free);
        *out_chunks = chunks.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_tokenize(
    struct mtmd_context * context,
    struct mtmd_input_chunks * output,
    const struct mtmd_input_text * text,
    const struct mtmd_bitmap ** bitmaps,
    size_t bitmap_count,
    int32_t * out_result,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!context || !output || !text || !text->text || !out_result ||
        (bitmap_count > 0 && !bitmaps)) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "context, output, text, text bytes, and out_result must not be null; bitmap array may only be null when empty");
    }
    for (size_t index = 0; index < bitmap_count; ++index) {
        if (!bitmaps[index]) {
            return llama_rs_chat_set_error(
                out_error,
                LLAMA_RS_STATUS_INVALID_ARGUMENT,
                "bitmap array contains a null element");
        }
    }

    try {
        *out_result = mtmd_tokenize(context, output, text, bitmaps, bitmap_count);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_encode_chunk(
    struct mtmd_context * context,
    const struct mtmd_input_chunk * chunk,
    int32_t * out_result,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!context || !chunk || !out_result) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "context, chunk, and out_result must not be null");
    }

    try {
        *out_result = mtmd_encode_chunk(context, chunk);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_eval_chunks(
    struct mtmd_context * context,
    struct llama_context * llama_context,
    const struct mtmd_input_chunks * chunks,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * out_new_n_past,
    int32_t * out_result,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!context || !llama_context || !chunks || !out_new_n_past || !out_result) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "context, llama_context, chunks, out_new_n_past, and out_result must not be null");
    }

    try {
        llama_pos new_n_past = n_past;
        *out_result = mtmd_helper_eval_chunks(
            context,
            llama_context,
            chunks,
            n_past,
            seq_id,
            n_batch,
            logits_last,
            &new_n_past);
        *out_new_n_past = new_n_past;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_capabilities_from_file(
    const char * path,
    struct llama_rs_mtmd_capabilities * out_capabilities,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!path || !out_capabilities) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "path and out_capabilities must not be null");
    }

    try {
        const clip_cap capabilities = clip_get_cap(path);
        const llama_rs_mtmd_capabilities projected = {
            capabilities.has_vision,
            capabilities.has_audio,
        };
        *out_capabilities = projected;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" llama_rs_status llama_rs_mtmd_memory_report_create(
    const char * path,
    const struct mtmd_context_params * params,
    struct llama_rs_mtmd_memory_report ** out_report,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_report) {
        *out_report = nullptr;
    }
    if (!path || !params || !out_report) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "path, params, and out_report must not be null");
    }

    try {
        const std::lock_guard<std::mutex> memory_usage_guard(g_mtmd_memory_usage_mutex);
        const auto memory = mtmd_get_memory_usage(path, *params);
        auto report = std::make_unique<llama_rs_mtmd_memory_report>();
        report->devices.reserve(memory.size());

        // Registered devices are emitted in registry order rather than std::map's
        // pointer order. Any future unregistered device is emitted afterwards.
        const size_t registered_count = ggml_backend_dev_count();
        for (size_t index = 0; index < registered_count; ++index) {
            ggml_backend_dev_t device = ggml_backend_dev_get(index);
            const auto found = memory.find(device);
            if (found != memory.end()) {
                report->devices.push_back(project_memory_entry(found->first, found->second));
            }
        }

        std::vector<mtmd_memory_entry> unmatched;
        for (const auto & [device, bytes] : memory) {
            if (!registered_device_index(device)) {
                unmatched.push_back(project_memory_entry(device, bytes));
            }
        }
        std::sort(
            unmatched.begin(),
            unmatched.end(),
            [](const mtmd_memory_entry & lhs, const mtmd_memory_entry & rhs) {
                return std::tie(lhs.name, lhs.backend_type, lhs.description, lhs.bytes) <
                       std::tie(rhs.name, rhs.backend_type, rhs.description, rhs.bytes);
            });
        report->devices.insert(
            report->devices.end(),
            std::make_move_iterator(unmatched.begin()),
            std::make_move_iterator(unmatched.end()));

        *out_report = report.release();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return llama_rs_chat_current_exception(out_error);
    }
}

extern "C" void llama_rs_mtmd_memory_report_free(
    struct llama_rs_mtmd_memory_report * report) {
    delete report;
}

extern "C" size_t llama_rs_mtmd_memory_report_count(
    const struct llama_rs_mtmd_memory_report * report) {
    return report ? report->devices.size() : 0;
}

extern "C" llama_rs_status llama_rs_mtmd_memory_report_get(
    const struct llama_rs_mtmd_memory_report * report,
    size_t index,
    struct llama_rs_mtmd_device_memory * out_device,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!report || !out_device) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "report and out_device must not be null");
    }
    if (index >= report->devices.size()) {
        return llama_rs_chat_set_error(
            out_error,
            LLAMA_RS_STATUS_INVALID_ARGUMENT,
            "mtmd memory report index is out of range");
    }

    const mtmd_memory_entry & entry = report->devices[index];
    const llama_rs_mtmd_device_memory projected = {
        entry.device_index.has_value(),
        entry.device_index.value_or(0),
        entry.backend_type,
        bytes_view(entry.name),
        bytes_view(entry.description),
        entry.bytes,
    };
    *out_device = projected;
    return LLAMA_RS_STATUS_OK;
}
