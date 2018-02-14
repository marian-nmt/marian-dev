/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include <memory>

#include "kernels/tensor_operators.h"

namespace marian {

class SparseTensorBase : public std::enable_shared_from_this<SparseTensorBase> {
protected:
  float* data_;
  int* indices_;
  int size_;
  int capacity_;
  size_t device_;

public:
  const ResidentDevice residency;

  SparseTensorBase(int capacity, size_t device, ResidentDevice residency)
    : capacity_(capacity), device_(device), residency(residency) {
  }

  SparseTensorBase(float* data, int* indices, int size, size_t device, ResidentDevice residency)
    : data_(data), indices_(indices), size_(size), capacity_(size), device_(device), residency(residency) {
  }

  virtual ~SparseTensorBase() {}

  int capacity() { return capacity_; }

  int size() { return size_; }

  float* data() { return data_; }

  int* indices() { return indices_; }

  virtual void copyFrom(float* data, int* indices, int size, bool data_only) = 0;

  // copy from another sparse tensor
  void copyFrom(std::shared_ptr<SparseTensorBase> t, bool data_only = false) {
    copyFrom(t->data(), t->indices(), t->size(), data_only);
  }

  size_t getDevice() { return device_; }

  void setSize(int size) { size_ = size; }

  // return the dense representation of this tensor
  void toDense(Tensor t, int offset) {
    t->set(0);
    scatterAdd(t, offset);
  }

  virtual void scatterAdd(Tensor t, int offset = 0) = 0;

  virtual std::shared_ptr<SparseTensorBase> subtensor(int pos, int size, int idx) = 0;
};

typedef std::shared_ptr<SparseTensorBase> SparseTensor;
}

#include "training/sparse_tensor_cpu.h"

#if CUDA_FOUND
#include "training/sparse_tensor_gpu.h"
#endif
