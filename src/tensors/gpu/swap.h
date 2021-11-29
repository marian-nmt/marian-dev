#pragma once
#include <stdlib.h>
#include "common/definitions.h"
#include "common/logging.h"

namespace marian {
namespace swapper {

#ifdef CUDA_FOUND
void copyCpuToGpu(char * gpuOut, const char * in, size_t count, const marian::DeviceId& deviceId);
#else
inline void copyCpuToGpu(char * gpuOut, const char * in, size_t count, const marian::DeviceId& deviceId) {
  ABORT("Copy from CPU to GPU memory is only available with CUDA.");
}
#endif

}
}
