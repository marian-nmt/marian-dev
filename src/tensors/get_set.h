#pragma once

namespace marian {

namespace cpu {
  template <typename T>
  void Set(marian::Tensor out, const T* beg, const T* end);
}

namespace gpu {
  template <typename T>
  void Set(marian::Tensor out, const T* beg, const T* end);
}

template <typename T>
void Set(marian::Tensor out, const T* beg, const T* end) {
#ifdef CUDA_FOUND
  if(out->getBackend()->getDeviceId().type == DeviceType::gpu)
    gpu::Set(out, beg, end);
  else
#endif
    cpu::Set(out, beg, end);
}

namespace cpu {
  template <typename T>
  void Get(const marian::Tensor out, T* beg, T* end);
}

namespace gpu {
  template <typename T>
  void Get(const marian::Tensor out, T* beg, T* end);
}

template <typename T>
void Get(const marian::Tensor out, T* beg, T* end) {
#ifdef CUDA_FOUND
  if(out->getBackend()->getDeviceId().type == DeviceType::gpu)
    gpu::Get(out, beg, end);
  else
#endif
    cpu::Get(out, beg, end);
}

}