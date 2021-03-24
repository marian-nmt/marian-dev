#include "cuda_helpers.h"
void copyCpuToGpu(const char * in, char * gpuOut);

namespace marian {
    namespace swapper {
        void copyCpuToGpu(char * gpuOut, const char * in, size_t count) {
            CUDA_CHECK(cudaMemcpy(gpuOut, in, count, cudaMemcpyHostToDevice));
        }
    }
}
