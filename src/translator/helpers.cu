/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include <cuda.h>
#include <limits>

#include "data/types.h"
#include "tensors/tensor.h"
#include "translator/helpers.h"
#include "tensors/gpu/cuda_helpers.h"

namespace marian {

namespace gpu {

template <typename T>
__global__ void gSetColumn(T* d_in,
                           size_t n_columns,
                           size_t n_rows,
                           size_t noColumn,
                           float value) {
  size_t rowNumber = threadIdx.x + blockDim.x * blockIdx.x;
  size_t index = noColumn + rowNumber * n_columns;

  if(index < n_columns * n_rows) {
    d_in[index] = (T)value;
  }
}

void suppressWord(Expr probs, Word id) {
  Tensor p = probs->val();

  int nRows = p->shape().elements() / p->shape()[-1];
  int nColumns = p->shape()[-1];

  int nBlocks = nRows / 512 + ((nRows % 512 == 0) ? 0 : 1);
  int nThreads = std::min(512, nRows);

  if(p->type() == Type::float32) {
    gSetColumn<<<nBlocks, nThreads>>>(p->data<float>(), nColumns, nRows, id, std::numeric_limits<float>::lowest());
  } else if (p->type() == Type::float16) {
    gSetColumn<<<nBlocks, nThreads>>>(p->data<half>(), nColumns, nRows, id, std::numeric_limits<float16>::lowest());
  } else {
    ABORT("suppressWord not implemented for type {}", p->type());
  }
}
}  // namespace gpu
}  // namespace marian
