#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <memory>

#include "kernels/cuda_helpers.h"
#include "kernels/tensor_operators.h"
#include "training/sparse_tensor.h"

namespace marian {

// TODO:  create actual sparse tensor class. This one is just minimal
__global__ void gScatterAdd(float* denseData,
                            float* sparseData,
                            int* sparseIndices,
                            int denseSize,
                            int sparseSize,
                            int offset) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= sparseSize)
    return;
  if(sparseIndices[idx] + offset >= 0
     && sparseIndices[idx] + offset < denseSize)
    denseData[sparseIndices[idx] + offset] += sparseData[idx];
}

__global__ void gFindSubtensor(int* indices,
                               int size,
                               int targetStart,
                               int targetEnd,
                               int* resultStart,
                               int* resultEnd) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;

  if(indices[idx] >= targetStart
     && (idx == 0 || indices[idx - 1] < targetStart)) {
    resultStart[0] = idx;
  }

  if(indices[idx] < targetEnd
     && (idx == size - 1 || indices[idx + 1] >= targetEnd))
    resultEnd[0] = idx;
}

class SparseTensorGPU : public SparseTensorBase {
  int* d_is_unsorted;
  int* gstart_;
  int* gend_;

public:
  SparseTensorGPU(int capacity, size_t device) 
    : SparseTensorBase(capacity, device) {
    cudaSetDevice(device_);
    CUDA_CHECK(cudaMalloc(&data_, sizeof(float) * capacity));
    CUDA_CHECK(cudaMalloc(&indices_, sizeof(int) * capacity));

    CUDA_CHECK(cudaMalloc(&gstart_, sizeof(int) * 100));
    CUDA_CHECK(cudaMalloc(&gend_, sizeof(int) * 100));
  }

  SparseTensorGPU(float* data, int* indices, int size, size_t device)
    : SparseTensorBase(data, indices, size, device) {
  }

  void copyFrom(float* data, int* indices, int size, bool data_only) {
    if(capacity_ < size) {
      return;
      // NO enough capacity
    }
    size_ = size;
    if(size == 0)
      return;
    cudaSetDevice(device_);

    cudaMemcpy(data_, data, size * sizeof(float), cudaMemcpyDefault);
    if(!data_only)
      cudaMemcpy(indices_, indices, size * sizeof(int), cudaMemcpyDefault);
    cudaStreamSynchronize(0);
  }

  void scatterAdd(Tensor t, int offset) {
    cudaSetDevice(device_);
    cudaStreamSynchronize(0);
    int threads = 512;
    int blocks = 1 + size_ / threads;
    gScatterAdd<<<blocks, threads>>>(
        t->data(), data_, indices_, t->size(), size_, offset);
    cudaStreamSynchronize(0);
  }

  std::shared_ptr<SparseTensorBase> subtensor(int pos, int size, int idx) {
    cudaSetDevice(device_);
    cudaStreamSynchronize(0);
    int* start = gstart_ + idx;
    int* end = gend_ + idx;

    int threads = 512;
    int blocks = 1 + size_ / threads;
    cudaMemset(start, -1, sizeof(int));
    cudaMemset(end, 0, sizeof(int));

    gFindSubtensor<<<blocks, threads>>>(
        indices_, size_, pos, pos + size, start, end);

    int startOffset;
    int endOffset;
    int tmp_dt;
    cudaMemcpy(&startOffset, start, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&endOffset, end, sizeof(int), cudaMemcpyDeviceToHost);

    if(startOffset != -1 && startOffset < size_)
      cudaMemcpy(
          &tmp_dt, indices_ + startOffset, sizeof(int), cudaMemcpyDeviceToHost);

    int subtensorSize = max(0, endOffset - startOffset + 1);
    cudaStreamSynchronize(0);
    return std::shared_ptr<SparseTensorBase>(new SparseTensorGPU(
        data_ + startOffset, indices_ + startOffset, subtensorSize, device_));
  }
};

}
