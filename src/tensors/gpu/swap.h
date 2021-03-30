#pragma once
#include <stdlib.h>
#include "common/logging.h"
namespace marian {
    namespace swapper {
#ifdef CUDA_FOUND
        void copyCpuToGpu(char * gpuOut, const char * in, size_t count);
#else
        inline void copyCpuToGpu(char * gpuOut, const char * in, size_t count) {
            ABORT("Copy from CPU to GPU memory is only available with CUDA.");
        }
#endif
    }
}
