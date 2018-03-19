#include <memory>

#include "tensors/tensor_operators.h"
#include "training/gradient_dropping/dropper.h"
#include "training/gradient_dropping/sparse_tensor.h"

namespace marian {

namespace cpu {

float GradientDropBase::find_threshold(Tensor grads, float rate) { 
  int size = grads->size();
  int sortSize = std::min(100000, size);

  if (!tmp) {
    tmp = newTensor(sortSize, grads->getBackend());
  }

//sampling
#pragma omp parallel for simd
  for (int i = 0; i < sortSize; i++) {
    tmp->data()[i++] = grads->data()[rand() % size];
  }

  std::sort(tmp->data(), tmp->data() + sortSize);
  int cut_index = std::max(0, (int)(sortSize * rate) - 1);

  return tmp->data()[cut_index];
}

void GradientDropBase::dropGraph(Tensor grads,
                                 SparseTensor destination,
                                 float rate,
                                 float momentum) {
  // init
  if(!residual) {
    residual = newTensor(grads->size(), grads->getBackend());
    step = 0;
  }

  if(!velocity && momentum > 0.0) {
    velocity = newTensor(grads->size(), grads->getBackend());
  }

  // Step 1: add residual to the current gradient
  {
    using namespace functional;
    marian::cpu::Element(_1 = _1 + _2, grads, residual);
  }

  // step 2: find threshold 
  float t = find_threshold(grads, rate);

  // step 3: drop gradients lower than threshold
  //         store gradients lower than threshold into the residual
  {
    using namespace functional;
    marian::cpu::Element(_1 = if_then_else(abs(_2) > t, 0, _2), residual, grads);
    marian::cpu::Element(_1 = if_then_else(abs(_1) <= t, 0, _1), grads);
  }

  destination->fromDense(grads);

  step++;
}

}
}
