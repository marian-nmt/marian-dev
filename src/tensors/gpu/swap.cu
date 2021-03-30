#include "cuda_helpers.h"
#include "swap.h"
void copyCpuToGpu(const char * in, char * gpuOut);

namespace marian {
    namespace swapper {
        void copyCpuToGpu(char * gpuOut, const char * in, size_t count, const marian::DeviceId& deviceId) {
            CUDA_CHECK(cudaSetDevice(deviceId.no));
            CUDA_CHECK(cudaMemcpy(gpuOut, in, count, cudaMemcpyHostToDevice));
        }
    }
}
