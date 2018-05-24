#include <curand.h>
#include <curand_kernel.h>

#include <memory>

#include "tensors/gpu/cuda_helpers.h"
#include "tensors/tensor_operators.h"
#include "training/1bit_quantization/quantizer.h"


namespace marian {

__global__ void gQuantize(float* original,
                          float* residual, 
                          uint32_t* quantized,
                          float avg,
                          int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
  // set bit to 1
  if (original[idx] > 0) {
    atomicAdd(&quantized[idx / 32], (1 << (idx % 32)));
    residual[idx] = (original[idx] - avg);
    original[idx] = avg;
  } 
  // set bit to 0
  else {
    residual[idx] = (original[idx] + avg);
    original[idx] = -avg;
  }
}

__global__ void gDequantize(float* original,
                            uint32_t* quantized,
                            float avg,
                            int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
  // set bit to 1
  bool bit = (quantized[idx / 32] >> (idx % 32)) & 1;
  if (bit) {
    original[idx] = avg;
  } 
  // set bit to 0
  else {
    original[idx] = -avg;
  }
}

// t[i] = avg if t[i] > 0, 
// t[i] = -avg otherwise
// quantized's i-th bit will be 1 if t[i] > 0, 0 otherwise
void QuantizerBase::quantize_do(Tensor t, Tensor residual, Tensor quantized, float avg) { 
  cudaSetDevice(t->getDevice().no);

  int size = t->size();

  int threads = 512;
  int blocksSample = size / threads;

  // convert to int32 as you can't do bit manipulation in float
  quantized->set(0);
  gQuantize<<<blocksSample, threads>>>(t->data(), 
                                         residual->data(),
                                         (uint32_t*) quantized->data(), 
                                         avg,
                                         size);
}

// revert quantized bit into a full tensor
void QuantizerBase::dequantize_do(Tensor t, Tensor quantized, float avg) { 
  cudaSetDevice(t->getDevice().no);

  int size = t->size();

  int threads = 512;
  int blocksSample = size / threads;

  // convert to int32 as you can't do bit manipulation in float
  gDequantize<<<blocksSample, threads>>>(t->data(),
                                         (uint32_t*) quantized->data(), 
                                         avg,
                                         size);
}

}


