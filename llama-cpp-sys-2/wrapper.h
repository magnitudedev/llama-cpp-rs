#include "llama.cpp/include/llama.h"
#include "llama.cpp/ggml/include/gguf.h"

#ifdef LLAMA_RS_BUILD_COMMON
#include "wrapper_common.h"
#endif

#ifdef LLAMA_RS_BUILD_MTMD_EXT
#include "wrapper_mtmd_ext.h"
#endif

#if defined(LLAMA_RS_BUILD_COMMON) && defined(LLAMA_RS_BUILD_MTMD_EXT)
#include "wrapper_mtmd_speculative.h"
#endif
