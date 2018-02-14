/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include <vector>
#include "tensors/tensor.h"
#include "tensors/residency.h"

namespace marian {

namespace sparse {

class CSR {
protected:
  int nnz_{0};
  int rows_{0};
  int cols_{0};
  size_t device_{0};

public:
  const ResidentDevice residency;

  CSR(int rows, int cols, size_t device, ResidentDevice residency)
      : rows_(rows), cols_(cols), device_(device), residency(residency) {}

  CSR(int rows,
      int cols,
      const std::vector<float>& values,
      const std::vector<int>& rowIndices,
      const std::vector<int>& colIndices,
      size_t device,
      ResidentDevice residency)
      : nnz_(values.size()), rows_(rows), cols_(cols), device_(device), residency(residency) {}

  CSR(Tensor dense, ResidentDevice residency)
    : rows_(dense->shape()[0] * dense->shape()[2] * dense->shape()[3]),
      cols_(dense->shape()[1]), device_(dense->getDevice()), residency(residency) {}

  virtual ~CSR() {}

  virtual void toTensor(Tensor dense) = 0;

  int nnz() { return nnz_; }
  int rows() { return rows_; }
  int cols() { return cols_; }

  virtual float* values() = 0;
  virtual int* rowIndices() = 0;
  virtual int* colIndices() = 0;

  size_t getDevice() { return device_; }

  virtual std::string debug() = 0;
};

}

}

#if MKL_FOUND
#include "kernels/sparse_cpu.h"
#endif

#if CUDA_FOUND
#include "kernels/sparse_gpu.h"
#endif

#if MKL_FOUND || CUDA_FOUND
namespace marian {

namespace sparse {

void multiply(Ptr<CSR> C, const Ptr<CSR> A, const Ptr<CSR> B, bool transA, bool transB);

void LfaForward(Tensor out, Tensor logits, Tensor att, Ptr<CSR> sparseLf);

void LfaBackward(Tensor grad, Tensor adj, Ptr<CSR> sparseLf);

}

}
#endif
