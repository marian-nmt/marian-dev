/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include <limits>

#include "data/types.h"
#include "tensors/tensor.h"
#include "translator/helpers.h"

namespace marian {

namespace cpu {

void SetColumn(Tensor in_, size_t col, float value) {
  int nRows = in_->shape()[0] * in_->shape()[2] * in_->shape()[3];
  int nColumns = in_->shape()[1];

  float* in = in_->data();
  for (int rowNumber = 0; rowNumber < nRows; ++rowNumber) {
    int index = col + rowNumber * nColumns;
    in[index] = value;
  }
}

void suppressUnk(Expr probs) {
  SetColumn(probs->val(), UNK_ID, std::numeric_limits<float>::lowest());
}

void suppressWord(Expr probs, Word id) {
  SetColumn(probs->val(), id, std::numeric_limits<float>::lowest());
}

}

void suppressUnk(Expr probs) {
  if (probs->val()->residency == DEVICE_CPU) {
    cpu::suppressUnk(probs);
  }
  #if CUDA_FOUND
  else {
    gpu::suppressUnk(probs);
  }
  #endif
}

void suppressWord(Expr probs, Word id) {
  if (probs->val()->residency == DEVICE_GPU) {
    cpu::suppressWord(probs, id);
  }
  #if CUDA_FOUND
  else {
    gpu::suppressWord(probs, id);
  }
  #endif
}

}
