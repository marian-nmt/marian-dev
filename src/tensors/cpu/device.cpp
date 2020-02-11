#include "tensors/device.h"
#include <iostream>

#ifdef _WIN32
#include <malloc.h>
#endif
#include <stdlib.h>

namespace marian {
namespace cpu {

// allocate function for tensor reserve() below. 
// Alignment is needed because we use AVX512 and AVX2 vectors. We should fail if we can't allocate aligned memory.
#ifdef _WIN32
#define MALLOC(size) _aligned_malloc(size, alignment_)
#else
// On macos, aligned_alloc is available only on c++17
// Furthermore, it requires that the memory requested is an exact multiple of the alignment, otherwise it fails.
// posix_memalign is available both Mac (Since 2016) and Linux and in both gcc and clang
void * posix_aligned_alloc(size_t alignment_, size_t size) {
    void * mem_;
    posix_memalign(&mem_, alignment_, size);
    return mem_;
}
#define MALLOC(size) posix_aligned_alloc(alignment_, size)
#endif

#ifdef _WIN32
#define FREE(ptr) _aligned_free(ptr)
#else
#define FREE(ptr) free(ptr)
#endif

Device::~Device() {
  FREE(data_);
}

void Device::reserve(size_t size) {
  size = align(size);
  ABORT_IF(size < size_ || size == 0,
           "New size must be larger than old size and larger than 0");

  uint8_t *temp = static_cast<uint8_t*>(MALLOC(size));
  if(data_) {
    std::copy(data_, data_ + size_, temp);
    FREE(data_);
  }
  data_ = temp;
  size_ = size;
}
}  // namespace cpu
}  // namespace marian
