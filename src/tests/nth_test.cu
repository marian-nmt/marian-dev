#include <cuda.h>
#include <thrust/device_vector.h>

#include "marian.h"

using namespace marian;

void HandleError(cudaError_t err, const char* file, int line) {
  if(err != cudaSuccess) {
    UTIL_THROW2("ERROR: " << cudaGetErrorString(err) << " in " << file
                          << " at line "
                          << line);
  }
}

#define UNROLL_MAXARG_LOOP(n, max)       \
  if(tid < (n) && tid + (n) < (max)) {   \
    if(sdata[tid + (n)] > sdata[tid]) {  \
      sdata[tid] = sdata[tid + (n)];     \
      indices[tid] = indices[tid + (n)]; \
    }                                    \
  }

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void gNthElementReduce(
    float* outCosts,
    size_t* outIndices,
    float* inCosts,
    const size_t* beamSizes,
    int batchSize,
    int batchStride,
    int vocabSize)
{
  extern __shared__ float sdata[];
  int* indices = (int*)(sdata + blockDim.x);


  int tid = threadIdx.x;

  for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
    int begin = batchIdx * batchStride * vocabSize;
    int end = begin + batchStride * vocabSize;

    int i = begin + blockIdx.x * (blockDim.x * 2) + tid;

    if (i < end) {
      sdata[tid] = -3.40282e+38f;
    }


    if (i < end) {
      sdata[tid] = inCosts[i];
      indices[tid] = i;
    }

    if(i + blockDim.x < end) {
      float a = inCosts[i];
      float b = inCosts[i + blockDim.x];

      if(a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while(i + 2 * gridDim.x * blockDim.x < end) {
      i += 2 * gridDim.x * blockDim.x;

      float a = inCosts[i];
      if(a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if(i + blockDim.x < end) {
        float b = inCosts[i + blockDim.x];
        if(b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for(int s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if(tid < s && tid + s < end) {
        if(sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP(32, end);
    UNROLL_MAXARG_LOOP(16, end);
    UNROLL_MAXARG_LOOP(8, end);
    UNROLL_MAXARG_LOOP(4, end);
    UNROLL_MAXARG_LOOP(2, end);
    UNROLL_MAXARG_LOOP(1, end);

    if(tid == 0) {
      outCosts[blockIdx.x + batchIdx * gridDim.x] = sdata[0];
      outIndices[blockIdx.x + batchIdx * gridDim.x] = indices[0];
    }
    __syncthreads();
  }
}

__global__ void gNthElementUpdate(
    float* outCosts,
    size_t* outIndices,
    float* bucketCosts,
    size_t* bucketIndices,
    float* inProbs,
    const size_t* beamSizes,
    int numBuckets,
    int batchStride,
    int vocabSize,
    int maxBeamSize)
{
  extern __shared__ float sdata[];
  int* indices = (int*)(sdata + blockDim.x);
  __shared__ float bestBinCost;
  __shared__ int bestBinCostIdx;

  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;

  for (int nth = 0; nth < beamSizes[batchIdx]; ++nth) {
    int i = tid;

    sdata[tid] = -3.40282e+38f;

    if(i < numBuckets) {
      sdata[tid] = bucketCosts[batchIdx * numBuckets + i];
      indices[tid] = i;
    }

    if(i + blockDim.x < numBuckets) {
      float a = bucketCosts[batchIdx * numBuckets + i];
      float b = bucketCosts[batchIdx * numBuckets + i + blockDim.x];
      if(a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while(i + 2 * blockDim.x < numBuckets) {
      i += 2 * blockDim.x;

      float a = bucketCosts[batchIdx * numBuckets + i];
      if(a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if(i + blockDim.x < numBuckets) {
        float b = bucketCosts[batchIdx * numBuckets + i + blockDim.x];
        if(b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for(int s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if(tid < s && tid + s < numBuckets) {
        if(sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP(32, numBuckets);
    UNROLL_MAXARG_LOOP(16, numBuckets);
    UNROLL_MAXARG_LOOP(8, numBuckets);
    UNROLL_MAXARG_LOOP(4, numBuckets);
    UNROLL_MAXARG_LOOP(2, numBuckets);
    UNROLL_MAXARG_LOOP(1, numBuckets);

    if (tid == 0) {
      bestBinCost = sdata[0];
      bestBinCostIdx = batchIdx * numBuckets + indices[0];

      inProbs[bucketIndices[bestBinCostIdx]] = -3.40282e+38f;

      outIndices[maxBeamSize * batchIdx + nth] = bucketIndices[bestBinCostIdx];
      outCosts[maxBeamSize * batchIdx + nth] = bestBinCost;
    }

    __syncthreads();

    i = (bestBinCostIdx - batchIdx * numBuckets) * (blockDim.x * 2) //go to bucket
        + tid;
    const int dist = numBuckets * 2 * blockDim.x;
    float* batchProbs = inProbs + batchIdx * batchStride * vocabSize;

    sdata[tid] = -3.40282e+38f;

    if(i < batchStride * vocabSize) {
      sdata[tid] = batchProbs[i];
      indices[tid] = i;
    }

    if(i + blockDim.x < batchStride * vocabSize) {
      float a = batchProbs[i];
      float b = batchProbs[i + blockDim.x];
      if(a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while(i + dist < batchStride * vocabSize) {
      i += dist;

      float a = batchProbs[i];
      if(a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if(i + blockDim.x < batchStride * vocabSize) {
        float b = batchProbs[i + blockDim.x];
        if(b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for(int s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if(tid < s && tid + s < batchStride * vocabSize) {
        if(sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP(32, batchStride * vocabSize);
    UNROLL_MAXARG_LOOP(16, batchStride * vocabSize);
    UNROLL_MAXARG_LOOP(8, batchStride * vocabSize);
    UNROLL_MAXARG_LOOP(4, batchStride * vocabSize);
    UNROLL_MAXARG_LOOP(2, batchStride * vocabSize);
    UNROLL_MAXARG_LOOP(1, batchStride * vocabSize);

    if(tid == 0) {
      indices[0] += batchIdx * batchStride * vocabSize;
      bucketCosts[bestBinCostIdx] = sdata[0];
      bucketIndices[bestBinCostIdx] = indices[0];
    }
    __syncthreads();
  }
}

void NthElement(
    Tensor outCosts,
    thrust::device_vector<size_t>& outKeys,
    Tensor Probs,
    const thrust::device_vector<size_t>& beamSizes)
{
  int batchSize = Probs->shape()[0];
  int vocabSize = Probs->shape()[1];
  int batchStride = Probs->shape()[3];
  int maxBeamSize = outCosts->shape()[3];

  int blockSize = 512;
  int numBuckets = ((batchStride * vocabSize) % (2 * blockSize) != 0)
                   + int(batchStride * vocabSize / (2 * blockSize));

  thrust::device_vector<float> bucketCosts(batchSize * numBuckets);
  thrust::device_vector<size_t> bucketIndices(batchSize * numBuckets);

  float* bucketCostsPtr = thrust::raw_pointer_cast(bucketCosts.data());
  size_t* bucketIndicesPtr = thrust::raw_pointer_cast(bucketIndices.data());
  const size_t* beamSizesPtr = thrust::raw_pointer_cast(beamSizes.data());
  size_t* outKeysPtr = thrust::raw_pointer_cast(outKeys.data());

  unsigned shared = blockSize * (sizeof(float) + sizeof(int));

  gNthElementReduce
    <<<numBuckets, blockSize, shared, 0>>>
    (bucketCostsPtr, bucketIndicesPtr, Probs->data(), beamSizesPtr,
     batchSize, batchStride, vocabSize);

  gNthElementUpdate
    <<<batchSize, blockSize, shared, 0>>>
    (outCosts->data(), outKeysPtr, bucketCostsPtr, bucketIndicesPtr,
     Probs->data(), beamSizesPtr, numBuckets, batchStride, vocabSize, maxBeamSize);
}

int main(int argc, char** argv) {
  auto options = New<Config>(argc, argv, false);
  auto graph = New<ExpressionGraph>(false);

  graph->setDevice(0);
  graph->reserveWorkspaceMB(128);

  int dimBatch = 2;
  int dimWord = 1;
  int batchLength = 50000;
  int numLayers = 1;

  int elemNum = dimBatch * dimWord * batchLength * numLayers;

  std::vector<float> embData(elemNum);
  for (size_t i = 0; i < embData.size(); ++i) {
    embData[i] = i;
  }

  auto x = graph->param("x", {dimBatch, batchLength, dimWord, numLayers},
                        keywords::init=inits::from_vector(embData));

  auto y = graph->param("y", {dimBatch, 1, 1, 12},
                        keywords::init=inits::zeros);
  graph->forward();

  const thrust::device_vector<size_t> beamSizes({12, 12});
  thrust::device_vector<float> outCosts(24);
  thrust::device_vector<size_t> outIdx(24);

  NthElement(y->val(), outIdx, x->val(), beamSizes);

  std::vector<float> yy;
  y->val() >> yy;
  for (auto f : yy) {
    std::cerr << f << " ";
  }
  std::cerr << std::endl;
  return 0;
}
