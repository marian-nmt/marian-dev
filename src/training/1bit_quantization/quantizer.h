#pragma once

#include "training/gradient_dropping/sparse_tensor.h"
#include "tensors/tensor_operators.h"
#include "tensors/tensor_allocator.h"
#include "functional/functional.h"

namespace marian {

class QuantizerBase {
protected:
  Tensor residual;
  Tensor tmp;

  std::vector<Ptr<TensorAllocator>> allocators;

  Tensor newTensor(int size, Ptr<Backend> backend) {
    Tensor t;
    Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(backend);
    allocator_->reserveExact(size * sizeof(float));
    allocator_->allocate(t, {1, size});
    allocators.push_back(allocator_);

    return t;
  }

  void quantize_do(Tensor t, Tensor residual, Tensor quantized, float avg);
  void dequantize_do(Tensor t, Tensor quantized, float avg);

public:
  QuantizerBase() {}
  ~QuantizerBase() {}

  virtual float quantize(Tensor t, Tensor quantized) {
    if (!tmp) {
      tmp = newTensor(1, t->getBackend());
    }

    if (!residual) {
      residual = newTensor(t->size(), t->getBackend());
    }

    using namespace functional;
    // add residual
    Element(_1 = _1 + _2, t, residual);

    // get average
    float scale = 1.f / t->size();
    Reduce(abs(_1), scale, tmp, t);
    float avg = tmp->get(0);

    // quantize
    // t[i] = avg if t[i] > 0, 
    // t[i] = -avg otherwise
    // quantized's i-th bit will be 1 if t[i] > 0, 0 otherwise
    quantize_do(t, residual, quantized, avg);
    dequantize_do(t, quantized, avg);
    
    return avg;
  }

  virtual void dequantize(Tensor t, Tensor quantized, float avg) {
    dequantize_do(t, quantized, avg);
  }
};

typedef Ptr<QuantizerBase> Quantizer;

}
