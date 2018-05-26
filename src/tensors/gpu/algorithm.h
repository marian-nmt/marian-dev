#pragma once

#include "tensors/backend.h"
#include "cuda_runtime.h"

namespace marian {
namespace gpu {
template <typename T>
void copy(Ptr<Backend> backend, const T* begin, const T* end, T* dest);

template <typename T>
  void copyStream(Ptr<Backend> backend, const T* begin, const T* end, T* dest,  cudaStream_t stream);

template <typename T>
void fill(Ptr<Backend> backend, T* begin, T* end, T value);

void setSparse(Ptr<Backend> backend,
               const std::vector<size_t>&,
               const std::vector<float>&,
               float*);
}
}
