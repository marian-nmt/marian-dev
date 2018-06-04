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

  float quantize_do(Tensor t, Tensor quantized, int quantize_bit);
  void dequantize_do(Tensor t, Tensor quantized, float avg, int quantize_bit);

public:
  QuantizerBase() {}
  ~QuantizerBase() {}

  virtual void test(Ptr<Backend> backend){
    LOG(info, " Quantization testing");
    // only needs the backend information
    int size = 32;

    Tensor t = newTensor(size, backend);
    
    int bits[3] = {1, 2, 4};
    for (int bit: bits){
      LOG(info, "Quantize to {}-bits", bit);

      // init
      Tensor quantized = newTensor(size * bit / 32, backend);
      std::vector<float> ori(size), quant(size);

      LOG(info, "  original bits          : {}", t->size() * 32);
      LOG(info, "  quantized bits          : {}", quantized->size() * 32);
      

      // random [-5,5]
      for (int i=0;i<size;i++){
        LOG(info,"set {}", i);
        t->set(i, (float) (((rand() % 1000) / 100.0) - 5.0));
      }
      t->get(ori);
      LOG(info, "done fill");

      float step = quantize_do(t, quantized, bit);

      // validate
      LOG(info, "  step size               : {}", step);

      // revert back
      dequantize_do(t, quantized, step, bit);
      t->get(quant);

      LOG(info, "  quantized values : ");
      for (int j=0;j<=10;j++)
        LOG(info, "   {} -> {}", ori[j], quant[j]);

    }
    LOG(info, "Done testing");
  }

  virtual float quantize(Tensor t, Tensor quantized, int quantize_bit = 1) {
    if (!tmp) {
      tmp = newTensor(1, t->getBackend());
    }

    if (!residual) {
      residual = newTensor(t->size(), t->getBackend());
    }

    using namespace functional;
    // add residual
    Element(_1 = _1 + _2, t, residual);

    // quantize
    float step = quantize_do(t, quantized, quantize_bit);

    // update residual
    Element(_1 = _1 - _2, residual, t);

    return step;
  }

  virtual void dequantize(Tensor t, Tensor quantized, float avg, int quantize_bit = 1) {
    dequantize_do(t, quantized, avg, quantize_bit);
  }
};

typedef Ptr<QuantizerBase> Quantizer;

}
