#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <memory>
#include <cfloat>

#include "common/definitions.h"
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

__global__ void columnWiseQuantize(float* input, float* tmpData, float* error,
    const float minVal, const size_t rowSize, const size_t totalSize, const bool useMinDrop) {
  __shared__ float sdata[600];
  __shared__ float smin[600];
  __shared__ int scount[600];

  float perBlockResults;
  float sum = 0.0; // sum of column elements greater that the cut off value
  int counter = 0; // number of column elements greater that the cut off value
  float minimum = FLT_MAX;

  float* columnData = &input[blockIdx.x * rowSize];
  float* columnErr = &error[blockIdx.x * rowSize];
  float* tmpColumnData = &tmpData[blockIdx.x * rowSize];
  // Accumulate per thread partial sum.
  for(int i = threadIdx.x; i < rowSize; i += blockDim.x) {
    // if(blockIdx.x * rowSize + i >= totalSize)
    //   printf("%d vs %d\totalSize", blockIdx.x * rowSize + i, totalSize);
    if(std::abs(columnData[i]) > minVal) {
      sum += std::abs(columnData[i]);
      if(minimum > std::abs(columnData[i]))
          minimum = std::abs(columnData[i]);
      counter++;
    }
    tmpColumnData[i] = 0;
  }

  sdata[threadIdx.x] = sum;
  scount[threadIdx.x] = counter;
  smin[threadIdx.x] = minimum;

  __syncthreads();

  // Accumulate sum and count from all the threads.
  for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if(threadIdx.x < offset) {
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
      scount[threadIdx.x] += scount[threadIdx.x + offset];
      
      if(smin[threadIdx.x] > smin[threadIdx.x + offset])
        smin[threadIdx.x] = smin[threadIdx.x + offset];
    }
    __syncthreads();
  }
  
  if(scount[0] == 0)
    return;
  
  if(useMinDrop)
    perBlockResults = smin[0];
  else
    perBlockResults = sdata[0] / (float) scount[0];

  __syncthreads();
  // Min/average is obtained. Now replace all:
  for(int i = threadIdx.x; i < rowSize; i += blockDim.x) {
    if(std::abs(columnData[i]) <= minVal) {
      columnErr[i] = columnData[i];
      columnData[i] = 0;
    } else {
      int sign = (columnData[i] > 0) ? 1 : -1;
      float replaceTo = perBlockResults * sign;
      columnErr[i] = columnData[i] - replaceTo;
      columnData[i] = replaceTo;
      tmpColumnData[i] = 1;
    }
  }
}

__global__ void gradDropQuantize(float* data, float* tmpData, float* errors,
    float minVal, float bucketVal, int maxBucketId, int maxSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= maxSize)
    return;
  if(std::abs(data[idx]) <= minVal) {
    errors[idx] = data[idx];
    data[idx] = 0;
    tmpData[idx] = 0;
  } else {
    int sign = (data[idx] < 0) ? -1 : 1; 
    int bucketId = (int)((std::abs(data[idx]) - minVal) / bucketVal);
    if(bucketId > maxBucketId)
      bucketId = maxBucketId;
    float replaceTo = (bucketId * bucketVal + minVal) * sign;

    errors[idx] = data[idx] - replaceTo;
    data[idx] = replaceTo;
    tmpData[idx] = 1;
  }
}

__global__ void gradDropQuantizeMean(float* data, float* tmpData, float* errors,
    float minVal, float bucketVal, int maxBucketId, float mean1, float mean2, int maxSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= maxSize)
    return;
  if(std::abs(data[idx]) <= minVal) {
    errors[idx] = data[idx];
    data[idx] = 0;
    tmpData[idx] = 0;
  } else {
    int sign = (data[idx] < 0) ? -1 : 1; 
    int bucketId = (int)((std::abs(data[idx]) - minVal) / bucketVal);
    if(bucketId > maxBucketId)
      bucketId = maxBucketId;
    float replaceTo = mean1;
    if(bucketId == 1)
      replaceTo = mean2;
    replaceTo *= sign;

    errors[idx] = data[idx] - replaceTo;
    data[idx] = replaceTo;
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

__global__ void locate(float* data, float toLocate, int size, int* result) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
  if(data[idx] <= toLocate && (idx == size - 1 || data[idx+1] > toLocate))
    *result = idx;
}

class GradientDropBase {
private:
  Ptr<Config> options_;
  int device_;

  // A helper, returns i-th element from a GPU stored array.
  float get(float* data, int i) {
    float res;
    cudaMemcpy(&res, data + i, sizeof(float), cudaMemcpyDeviceToHost);
    return res;
  }

  void gradDropDo(
      float* data, float* errors, float* tmpData, int rowSize, int colSize, float dropRate) {
    int totalSize = rowSize * colSize;
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

    // dont update the cut threshold every step
    int cutOffIndex = std::max(0, (int)(sortSize * dropRate) - 1);
    float cutOffValue = get(tmpData, cutOffIndex);

    int bits = options_->get<int>("quantize-bits");
    if(bits == 32) {
      gradDrop<<<blocks, threads>>>(data, tmpData, errors, cutOffValue, totalSize);
      return;
    }

    // bits--;
    long long bucketCount = (1<<bits);
    float minVal = cutOffValue, maxVal = get(tmpData, sortSize - 1);
    float range = maxVal - minVal;
    float bucketVal = range / bucketCount;
    float mean1;
    float mean2;

    if(options_->get<bool>("quantize-column-wise") && colSize != 1) {
      if (step == 0)
        LOG(info)->info("COLUMN WISE...");
      // Each column gets assigned to one block.  
      columnWiseQuantize<<<colSize, threads>>>(data, tmpData, errors, minVal,
          rowSize, totalSize, options_->get<bool>("quantize-min-drop"));
      return;
    }

    if(options_->get<bool>("quantize-min-drop")) {
      if (step == 0)
        LOG(info)->info("MIN DROP...");
      gradDropQuantize<<<blocks, threads>>>(data, tmpData, errors, minVal,
          bucketVal, bucketCount - 1, totalSize);
      return;
    }

    LOG(info)->info("AVG DROP...");

    int* result;
    int idx;
    cudaMalloc(&result, sizeof(int));
    locate<<<blocks, threads>>>(tmpData, minVal + bucketVal, sortSize, result);
    cudaMemcpy(&idx, result, sizeof(int), cudaMemcpyDeviceToHost); 
    idx++;
    if(idx > cutOffIndex && idx <= sortSize)
      mean1 = thrust::reduce(tmpDataPtr + cutOffIndex, tmpDataPtr + idx)
          / (idx - cutOffIndex);
    if(idx > cutOffIndex && idx < sortSize)
      mean2 = thrust::reduce(tmpDataPtr + idx, tmpDataPtr + sortSize)
          / (sortSize - idx);
    
    cudaFree(result);
    
    gradDropQuantizeMean<<<blocks, threads>>>(data, tmpData, errors, minVal,
        bucketVal, bucketCount - 1, mean1, mean2, totalSize);
  }

public:
  GradientDropBase(Ptr<Config> options) : options_(options) {}

  void dropGraph(Tensor sourceTensor, SparseTensor destinationTensor,
      double dropRate = 0.99, std::vector<std::pair<int,int> > const &layerShapes = {}) {
    float* feedback = NULL;
    float* tmpData;
    int step;
    std::vector<std::pair<std::pair<int,int>, int > > layerShapesSizes;

    cudaSetDevice(sourceTensor->getDevice());    
    if(!feedback) {
      device_ = sourceTensor->getDevice();
      CUDA_CHECK(cudaMalloc(&feedback, sizeof(float) * sourceTensor->size()));
      CUDA_CHECK(cudaMalloc(&tmpData, sizeof(float) * sourceTensor->size()));
      cudaMemset(feedback, 0, sizeof(float) * sourceTensor->size());
      cudaMemset(tmpData, 0, sizeof(float) * sourceTensor->size());

      step = 0;

      // Add layer dimentions along with incremental layer sizes to a vector.
      int totalSize = 0;
      for(auto& shape: layerShapes) {
        std::pair<std::pair<int,int>, int > tmpColumnData;
        tmpColumnData.first = shape;
        tmpColumnData.second = totalSize;
        layerShapesSizes.push_back(tmpColumnData);
        totalSize += shape.second * shape.first;    
      }
    }

    // If col-wise drop is disabled OR layer shapes info not provided, drop globally.
    if(!options_->get<bool>("quantize-column-wise") || layerShapes.size() == 0) {
      LOG(info)->info("NOT COLUMN WISE");
      gradDropDo(sourceTensor->data(), feedback, tmpData, sourceTensor->size(), 1, dropRate);
    } else {
      LOG(info)->info("COLUMN WISE");
      for(auto &shape: layerShapesSizes) {
        int offset = shape.second;
        gradDropDo(sourceTensor->data() + offset, feedback + offset, tmpData + offset,
            shape.first.first, shape.first.second, dropRate);
      }
    }

    // if(dropRate < 0.9)
        // return;

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
