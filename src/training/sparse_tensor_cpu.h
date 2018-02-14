/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "training/sparse_tensor.h" 

namespace marian {

struct SparseTensorCPU : SparseTensorBase {
  SparseTensorCPU(int capacity, size_t device)
    : SparseTensorBase(capacity, device, DEVICE_CPU) {
  }

  SparseTensorCPU(float* data, int* indices, int size, size_t device)
    : SparseTensorBase(data, indices, size, device, DEVICE_CPU) {
  }

  void copyFrom(float* data, int* indices, int size, bool data_only) {
    if (capacity_ < size) {
      return;
    }

    size_ = size;
    if (size == 0) {
      return;
    }

    std::copy(data, data + size, data_);
    if (!data_only) {
      std::copy(indices, indices + size, indices_);
    }
  }

  void scatterAdd(Tensor t, int offset) {
    int denseSize = t->size();
    float* denseData = t->data();
    for (int idx = 0; idx < size_; ++idx) {
      int denseIdx = indices_[idx] + offset;
      if (denseIdx >= 0 && denseIdx < denseSize) {
        denseData[denseIdx] += data_[idx];
      }
    }
  }

  std::shared_ptr<SparseTensorBase> subtensor(int pos, int size, int idx) {
    int* start = std::find(indices_, indices_ + size_, pos);
    int* end = std::find(start, indices_ + size, pos + size);

    int startOffset = start - indices_;
    int endOffset = end - indices_;
    int sparseSize = std::max(0, endOffset - startOffset); // FIXME: + 1 in GPU version?

    return std::shared_ptr<SparseTensorBase>(new SparseTensorCPU(
          data_ + startOffset, indices_ + startOffset, sparseSize, device_));
  }
};

}
