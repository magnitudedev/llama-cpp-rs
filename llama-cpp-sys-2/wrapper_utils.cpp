#include "wrapper_utils.h"

#include <cstdlib>

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}
