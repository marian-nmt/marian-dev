#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

#include <memory>

#include "tensors/gpu/cuda_helpers.h"
#include "tensors/tensor_operators.h"
#include "training/1bit_quantization/quantizer.h"
#include "training/1bit_quantization/quantized_float.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>


namespace marian {

__global__ void gQuantize8bit(float* data, float8_s* q, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return; 
  q[idx].fromFloat(data[idx]);
}


__global__ void gDequantize8bit(float* data, float8_s* q, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return; 
  q[idx].toFloat(data + idx);
}

__global__ void gQuantize16bit(float * data, __half * q, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size)
      return;
    q[idx] = __float2half(data[idx]);
}

__global__ void gDequantize16bit(float * data, __half * q, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size)
      return;
    data[idx] = __half2float(q[idx]);
}


template<typename T>
__global__ void gQuantize(float* original,
                          T q,
                          float step,
                          int size,
                          uint8_t bit) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
 
  uint8_t bucket_size = (1<<bit)>>1;
  q[idx].fromFloat(original + (idx * 8 / bit),
              step,
              bucket_size);
}


template<typename T>
__global__ void gDequantize(float* original,
                            T q,
                            float step,
                            int size,
                            uint8_t bit) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
  q[idx].toFloat(original + (idx * 8 / bit), step);
}

// Equal-width discretization of 2^quantize_bit number of bins
float QuantizerBase::quantize_do(Tensor t, Tensor quantized, int quantize_bit) { 
  cudaSetDevice(t->getDevice().no);
  if (quantize_bit <= 4) {
    // get average
    if (!tmp) {
        tmp = newTensor(1, t->getBackend());
      }

    using namespace functional;
    
    float scale = 1.f / t->size();
    Reduce(abs(_1), scale, tmp, t);
    float step = 2.0 * tmp->get(0) / (1<<(quantize_bit-1));

    // when doing quantization, we work in 8-bit int
    // each int8 holds information of (8/quantize_bit) gradients
    // Hence in total, there will be t->size() * quantize_bit / 8 8-bit ints
    int size = t->size() * quantize_bit / 8;
    int threads = 512;
    int blocksSample = (size + threads - 1) / threads;
    if (quantize_bit == 1)
      gQuantize<<<blocksSample, threads>>>(t->data(),
                                         (float1_s*) quantized->data(), 
                                         step,
                                         size,
                                         quantize_bit);
 
    if (quantize_bit == 2)
      gQuantize<<<blocksSample, threads>>>(t->data(),
                                         (float2_s*) quantized->data(),
                                         step,
                                         size,
                                         quantize_bit);
    if (quantize_bit == 4)
      gQuantize<<<blocksSample, threads>>>(t->data(),
                                         (float4_s*) quantized->data(),
                                         step,
                                         size,
                                         quantize_bit);

    return step;
  } else {
    int size = t->size();

    int threads = 512;
    int blocksSample = (size + threads - 1) / threads;
    if (quantize_bit == 8) {
      gQuantize8bit<<<blocksSample, threads>>>(t->data(), (float8_s*) quantized->data(), size);
      // 8 and 16 bit quantization does not need step information
      return 0;
    } else if (quantize_bit == 16) {
      gQuantize16bit<<<blocksSample, threads>>>(t->data(), (__half *) quantized->data(), size);
      // 8 and 16 bit quantization does not need step information
      return 0;
    } else {
      LOG(critical, " Unsupported quantization value: {}.", quantize_bit);
      std::abort();
    }
  }
}

// revert quantized bit into a full tensor
void QuantizerBase::dequantize_do(Tensor t, Tensor quantized, float step, int quantize_bit) { 
  if (quantize_bit <= 4) {
    cudaSetDevice(t->getDevice().no);

    // when doing quantization, we work in 8-bit int
    // each int8 holds information of (8/quantize_bit) gradients
    // Hence in total, there will be t->size() * quantize_bit / 8 8-bit ints
    int size = t->size() * quantize_bit / 8;;

    int threads = 512;
    int blocksSample = (size + threads - 1) / threads;
    if (quantize_bit == 1)
      gDequantize<<<blocksSample, threads>>>(t->data(),
                                           (float1_s*) quantized->data(), 
                                           step,
                                           size,
                                           quantize_bit);
    if (quantize_bit == 2)
      gDequantize<<<blocksSample, threads>>>(t->data(),
                                           (float2_s*) quantized->data(),
                                           step,
                                           size,
                                           quantize_bit);
    if (quantize_bit == 4)
      gDequantize<<<blocksSample, threads>>>(t->data(),
                                           (float4_s*) quantized->data(),
                                           step,
                                           size,
                                           quantize_bit);
  } else {
    int size = t->size();

    int threads = 512;
    int blocksSample = (size + threads - 1) / threads;
    if (quantize_bit == 8) {
      gDequantize8bit<<<blocksSample, threads>>>(t->data(), (float8_s*) quantized->data(), size);
    } else if (quantize_bit == 16) {
      gDequantize16bit<<<blocksSample, threads>>>(t->data(), (__half *) quantized->data(), size);
    } else {
      LOG(critical, " Unsupported quantization value: {}.", quantize_bit);
      std::abort();
    }
  }
}

}


