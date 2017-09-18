#pragma once

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "3rd_party/exception.h"
#include "common/definitions.h"
#include "common/shape.h"
#include "tensors/memory_piece.h"
#include "tensors/residency.h"

namespace marian {

class TensorBase : public std::enable_shared_from_this<TensorBase> {
protected:
  Ptr<MemoryPiece> memory_;
  Shape shape_;
  size_t device_;

public:
  const ResidentDevice residency;

  TensorBase(Ptr<MemoryPiece> memory, Shape shape, size_t device, ResidentDevice residency)
      : memory_(memory), shape_(shape), device_(device), residency(residency) {}

  virtual ~TensorBase() {}

  void reset(Ptr<MemoryPiece> memory) { memory_ = memory; }

  Ptr<MemoryPiece> memory() { return memory_; }

  Shape& shape() { return shape_; }

  float* data() { return (float*)memory_->data(); }

  size_t size() { return shape_.elements(); }

  float scalar() {
    UTIL_THROW_IF2(size() != 1, "Tensor is not a scalar");
    return get(0);
  }

  size_t getDevice() { return device_; }

  virtual float get(size_t i) = 0;

  virtual void set(size_t i, float value) = 0;

  virtual void get(std::vector<float>& v) = 0;

  virtual void set(float value) = 0;

  virtual void set(const std::vector<float>& v) = 0;

  virtual void setSparse(const std::vector<size_t>& k, const std::vector<float>& v) = 0;
  
  virtual void copyFrom(Tensor) = 0;

  virtual Tensor view(const Shape& shape, ptrdiff_t offset = 0) = 0;

  Tensor subtensor(int offset, int size) {
    return view({ 1, size }, offset);
  }

  virtual std::string debug() = 0;
};

typedef std::shared_ptr<TensorBase> Tensor;

static inline Tensor operator<<(Tensor t, const std::vector<float> &v) {
  t->set(v);
  return t;
}

static inline Tensor operator>>(Tensor t, std::vector<float> &v) {
  t->get(v);
  return t;
}

struct TensorCPU : TensorBase {
  TensorCPU(Ptr<MemoryPiece> data, Shape shape, size_t device)
    : TensorBase(data, shape, device, DEVICE_CPU) {}

  float get(size_t i);
  
  void set(size_t i, float value);

  void get(std::vector<float>& v);

  void set(float value);

  void set(const std::vector<float>& v);

  void setSparse(const std::vector<size_t>& k, const std::vector<float>& v);

  void copyFrom(Tensor);

  Tensor view(const Shape& shape, ptrdiff_t offset);

  std::string debug();
};

template <> struct residency_trait<TensorCPU> {
  static constexpr ResidentDevice residency = DEVICE_CPU;
};

#if CUDA_FOUND
struct TensorGPU : TensorBase {
  TensorGPU(Ptr<MemoryPiece> data, Shape shape, size_t device)
    : TensorBase(data, shape, device, DEVICE_GPU) {}

  float get(size_t i);
  
  void set(size_t i, float value);

  void get(std::vector<float>& v);

  void set(float value);

  void set(const std::vector<float>& v);

  void setSparse(const std::vector<size_t>& k, const std::vector<float>& v);

  void copyFrom(Tensor);

  Tensor view(const Shape& shape, ptrdiff_t offset);

  std::string debug();
};

template <> struct residency_trait<TensorGPU> {
  static constexpr ResidentDevice residency = DEVICE_GPU;
};
#endif

}
