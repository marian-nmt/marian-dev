#include "common/definitions.h"

#if CUDA_FOUND
#include "tensors/gpu/cuda_helpers.h"
#endif

namespace marian {
namespace gpu {
  size_t availableDevices() {
#if CUDA_FOUND
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    return (size_t)deviceCount;
#else
    return 0;
#endif
  }
}
}