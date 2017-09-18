#pragma once

#include "tensors/tensor.h"
#include "training/sparse_tensor.h"

namespace marian {

struct GradientDropBase {
  virtual void dropGraph(Tensor t, SparseTensor destination, double rate = 0.99) = 0;
};

typedef Ptr<GradientDropBase> GradientDrop;
}

#include "training/dropper_cpu.h"

#if CUDA_FOUND
#include "training/dropper_gpu.h"
#endif
