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

__global__ void gradDrop(
    float* data, float* tmpData, float* errors, float cutOffValue, int maxSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= maxSize)
    return;
  if(std::abs(data[idx]) <= cutOffValue) {
    errors[idx] = data[idx];
    data[idx] = 0;
    tmpData[idx] = 0;
  } else {
    errors[idx] = 0;
    tmpData[idx] = 1;
  }
}

__global__ void gradAddError(float* data, float* errors, int maxSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= maxSize)
    return;
  data[idx] += errors[idx];
}

__global__ void buildIndices(float* denseData,
                             float* denseSum,
                             float* sparseData,
                             int* sparseIndices,
                             int denseSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= denseSize)
    return;
  int tId = round(denseSum[idx]);
  if(tId <= 0) {
    return;
  }

  if(idx == 0 && tId > 0) {
    sparseIndices[tId - 1] = idx;
    sparseData[tId - 1] = denseData[idx];
  } else if(idx > 0 && tId > round(denseSum[idx - 1])) {
    sparseIndices[tId - 1] = idx;
    sparseData[tId - 1] = denseData[idx];
  }
}

__global__ void randomSampling(
    float* originalData, float* data, int size, int scale, int fullSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
  data[idx] = abs(originalData[idx * scale]);
}

class GradientDropBase {
private:
  float* feedback;
  float* tmpData;
  int step;
  int device_;

  // A helper, returns i-th element from a GPU stored array.
  float get(float* data, int i) {
    float res;
    cudaMemcpy(&res, data + i, sizeof(float), cudaMemcpyDeviceToHost);
    return res;
  }

  void gradDropDo(
      float* data, float* errors, float* tmpData, int totalSize, float rate) {
    int threads = 512;
    int blocks = 1 + totalSize / threads;
    cudaSetDevice(device_);

    gradAddError<<<blocks, threads>>>(data, errors, totalSize);
    // full sort
    int sortSize = min(100000, totalSize);
    int blocksSample = 1 + sortSize / threads;
    randomSampling<<<blocksSample, threads>>>(
        data, tmpData, sortSize, totalSize / sortSize, totalSize);
    thrust::device_ptr<float> tmpDataPtr(tmpData);
    thrust::sort(tmpDataPtr, tmpDataPtr + sortSize);

    int cutOffIndex = std::max(0, (int)(sortSize * rate) - 1);
    float cutOffValue = get(tmpData, cutOffIndex);

    gradDrop<<<blocks, threads>>>(data, tmpData, errors, cutOffValue, totalSize);
  }

public:
  void dropGraph(Tensor sourceTensor, SparseTensor destinationTensor, double rate = 0.99) {
    cudaSetDevice(sourceTensor->getDevice());    
    if(!feedback) {
      device_ = sourceTensor->getDevice();
      CUDA_CHECK(cudaMalloc(&feedback, sizeof(float) * sourceTensor->size()));
      CUDA_CHECK(cudaMalloc(&tmpData, sizeof(float) * sourceTensor->size()));
      cudaMemset(feedback, 0, sizeof(float) * sourceTensor->size());
      cudaMemset(tmpData, 0, sizeof(float) * sourceTensor->size());

      step = 0;
    }

    gradDropDo(sourceTensor->data(), feedback, tmpData, sourceTensor->size(), rate);

    thrust::device_ptr<float> maskPtr(tmpData);
    int denseSize = sourceTensor->size();
    thrust::inclusive_scan(maskPtr, maskPtr + denseSize, maskPtr);
    float sparseSize;

    cudaMemcpy(&sparseSize,
               tmpData + denseSize - 1,
               sizeof(float),
               cudaMemcpyDeviceToHost);

    // Convert result of inclusive scan to indices.
    int threads = 512;
    int blocks = 1 + denseSize / threads;
    cudaSetDevice(sourceTensor->getDevice());
    buildIndices<<<blocks, threads>>>(sourceTensor->data(),
                                      tmpData,
                                      destinationTensor->data(),
                                      destinationTensor->indices(),
                                      denseSize);
    destinationTensor->setSize(sparseSize);

    cudaStreamSynchronize(0);

    step++;
  }
};

typedef Ptr<GradientDropBase> GradientDrop;
}
