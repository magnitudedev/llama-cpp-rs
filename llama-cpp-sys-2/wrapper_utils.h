#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum llama_rs_status {
    LLAMA_RS_STATUS_OK = 0,
    LLAMA_RS_STATUS_INVALID_ARGUMENT = -1,
    LLAMA_RS_STATUS_ALLOCATION_FAILED = -2,
    LLAMA_RS_STATUS_EXCEPTION = -3,
    LLAMA_RS_STATUS_INVALID_UTF8 = -4,
    LLAMA_RS_STATUS_INVALID_STATE = -5
} llama_rs_status;

typedef struct llama_rs_bytes_view {
    const uint8_t * data;
    size_t len;
} llama_rs_bytes_view;

#ifdef __cplusplus
extern "C" {
#endif

void llama_rs_string_free(char * ptr);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

static inline char * llama_rs_dup_bytes(const char * data, size_t len) noexcept {
    if ((!data && len > 0) || len == SIZE_MAX) {
        return nullptr;
    }
    char * buffer = static_cast<char *>(std::malloc(len + 1));
    if (!buffer) {
        return nullptr;
    }
    if (len > 0) {
        std::memcpy(buffer, data, len);
    }
    buffer[len] = '\0';
    return buffer;
}

static inline char * llama_rs_dup_string(const std::string & value) noexcept {
    return llama_rs_dup_bytes(value.data(), value.size());
}

static inline char * llama_rs_dup_string(const char * value) noexcept {
    return value ? llama_rs_dup_bytes(value, std::strlen(value)) : nullptr;
}

class llama_rs_invalid_utf8_error : public std::runtime_error {
public:
    explicit llama_rs_invalid_utf8_error(const std::string & message)
        : std::runtime_error(message) {
    }
};

static inline llama_rs_status llama_rs_chat_set_error(
    char ** out_error,
    llama_rs_status status,
    const char * message) noexcept {
    if (out_error) {
        *out_error = llama_rs_dup_string(message);
        if (!*out_error) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    return status;
}

// Must only be invoked from within a catch block.
static inline llama_rs_status llama_rs_chat_current_exception(char ** out_error) {
    try {
        throw;
    } catch (const std::bad_alloc &) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_ALLOCATION_FAILED, "allocation failed");
    } catch (const llama_rs_invalid_utf8_error & error) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_UTF8, error.what());
    } catch (const std::invalid_argument & error) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::exception & error) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_EXCEPTION, error.what());
    } catch (...) {
        return llama_rs_chat_set_error(out_error, LLAMA_RS_STATUS_EXCEPTION, "unknown llama.cpp exception");
    }
}

static inline std::string llama_rs_chat_optional_string(const char * value) {
    return value ? value : "";
}

static inline void llama_rs_chat_validate_array(const void * values, size_t count, const char * name) {
    if (count > 0 && !values) {
        throw std::invalid_argument(std::string(name) + " is null while its count is non-zero");
    }
}

#endif
