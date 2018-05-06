#pragma once
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
  
inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort = true) {
  if(code != cudaSuccess) {
    LOG(critical, "Error: {} - {}:{}", cudaGetErrorString(code), file, line);
    std::abort();
  }
}
