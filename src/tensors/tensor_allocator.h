#pragma once

#include <deque>
#include <set>

#include "common/definitions.h"
#include "common/utils.h"
#include "tensors/tensor.h"
#include "tensors/allocator.h"
#include "tensors/device_cpu.h"

#if CUDA_FOUND
#include "tensors/device_gpu.h"
#endif

namespace marian {

struct TensorAllocator {
  const ResidentDevice residency;
  TensorAllocator(ResidentDevice residency) : residency(residency) {}

  virtual void throwAtReallocation(bool throwRealloc) = 0;
  virtual void reserve(size_t bytes = 0) = 0;
  virtual void reserveExact(size_t bytes = 0) = 0;
  virtual void clear() = 0;
  virtual size_t capacity(Shape shape) = 0;
  virtual void allocate(Tensor& t, Shape shape) = 0;
  virtual void free(Tensor& t) = 0;
  virtual Tensor asTensor() = 0;
  virtual size_t size() = 0;
};

template <typename Device, typename DeviceTensor, size_t ALIGN>
class TensorAllocatorConcrete : public TensorAllocator {
private:
  const size_t CHUNK = 512;
  const size_t MBYTE = 1024 * 1024;
  const size_t GROW = CHUNK * MBYTE;

  Ptr<Allocator<Device>> allocator_;


public:
  TensorAllocatorConcrete(size_t device)
    : TensorAllocator(residency_trait<DeviceTensor>::residency),
      allocator_(New<Allocator<Device>>(device, 0, GROW, ALIGN))
  {}

  ~TensorAllocatorConcrete() { clear(); }

  void throwAtReallocation(bool throwRealloc) {
    allocator_->throwAtReallocation(throwRealloc);
  }

  void reserve(size_t bytes = 0) {
    float mult = bytes / GROW + 1;
    LOG(memory)->info(
        "Extending reserved space to {} MB (device {})",
        mult * CHUNK,
        allocator_->getDevice());

    allocator_->reserve(mult * GROW);
  }

  void reserveExact(size_t bytes = 0) {
    size_t mbytes = bytes / MBYTE;
    LOG(memory)->info(
        "Reserving {} MB, device {}",
        mbytes,
        allocator_->getDevice());

    allocator_->reserve(bytes);
  }

  void clear() {
    allocator_->clear();
  }

  size_t capacity(Shape shape) {
    return allocator_->template capacity<float>(shape.elements());
  }

  void allocate(Tensor& t, Shape shape) {
    if(!t || t->shape() != shape) {
      int size = shape.elements();
      auto mem = allocator_->template alloc<float>(size);
      Poison(reinterpret_cast<float*>(mem->data()), size);
      t = Tensor(new DeviceTensor(mem, shape, allocator_->getDevice()));
    }
  }

  void free(Tensor& t) {
    allocator_->free(t->memory());
  }

  Tensor asTensor() {
    auto mem = allocator_->memory();
    int size = mem->size() / sizeof(float);
    return Tensor(new DeviceTensor(mem, {1, size}, allocator_->getDevice()));
  }

  size_t size() { return allocator_->size() / sizeof(float); }

};

typedef TensorAllocatorConcrete<DeviceCPU, TensorCPU, 64> TensorAllocatorCPU;

#if CUDA_FOUND
typedef TensorAllocatorConcrete<DeviceGPU, TensorGPU, 256> TensorAllocatorGPU;
#endif

}
