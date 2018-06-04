#include <curand.h>
#include <curand_kernel.h>

#include <memory>

#include "tensors/gpu/cuda_helpers.h"
#include "tensors/tensor_operators.h"
#include "training/1bit_quantization/quantizer.h"

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>


namespace marian {
__global__ void gQuantize(float* original,
                          uint8_t* quantized,
                          float step,
                          int size,
                          int bucket_size,
                          int quantize_bit) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;

  printf("idx = %d\n", idx);

  // for each K bits
  for (int i=0;i < 8 / quantize_bit;i++) {
    int original_idx = idx * 8 / quantize_bit + i;

    // get the sign
    bool sign = (original[original_idx] > 0);

    int bucket = min(bucket_size - 1, 
                    (int) (abs(original[original_idx]) / bucket_size));
    
    if (idx == 0) 
      printf("original = %d | bucket %d | sign %d | wut %d\n", original_idx, bucket, sign, ((bucket * 2 + sign) << (i * quantize_bit)));
    // include sign into the quantization ID.
    quantized[idx] += ((bucket * 2 + sign) << (i * quantize_bit));
    if (sign)
      original[original_idx] = step * (0.5 + bucket);
    else
      original[original_idx] = -step * (0.5 + bucket);
  }
  if (idx == 0)
    printf("jadi %d\n", quantized[idx]);
}

__global__ void gDequantize(float* original,
                            uint8_t* quantized,
                            float step,
                            int size,
                            int quantize_bit) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;

  // obtain 'quantize_bit' amount of bits from the quantized container,
  // offset the corresponding location
  int K = 8 / quantize_bit;
  int bucket = (quantized[idx / K] >> (quantize_bit * (idx % K))) 
               % (1 << quantize_bit);
  if (idx < 8)
    printf("idx = %d | bucket = %d\n", idx, bucket);
  // get the sign
  bool sign = bucket % 2;
  bucket /= 2;
  if (sign)
    original[idx] = step * (0.5 + bucket);
  else {
    original[idx] = -step * (0.5 + bucket);
  }
}

// t[i] = avg if t[i] > 0, 
// t[i] = -avg otherwise
// quantized's i-th bit will be 1 if t[i] > 0, 0 otherwise
float QuantizerBase::quantize_do(Tensor t, Tensor quantized, int quantize_bit) { 
  cudaSetDevice(t->getDevice().no);
  LOG(info, "AA");

  // get step size: maximum value / 2^(quantize_bit - 1)
  // TODO: Remove thrust dependency
  thrust::device_ptr<float> d_vec(t->data());
  float max_val = *thrust::max_element(d_vec, d_vec + t->size());


  LOG(info, "maxval {}", max_val);

  float step = max_val / (1 << (quantize_bit - 1));

  LOG(info, "STEP {}", step);

  // when doing quantization, we work in 8-bit int
  // each int8 holds information of (8/quantize_bit) gradients
  // Hence in total, there will be t->size() * quantize_bit / 8 8-bit ints
  int size = t->size() * quantize_bit / 8;

  int threads = 512;
  int blocksSample = (size + threads - 1) / threads;
  LOG(info, "BB");
  // convert to int8 as you can't do bit manipulation in float
  quantized->set(0);
  LOG(info, "CC");
  gQuantize<<<blocksSample, threads>>>(t->data(),
                                       (uint8_t*) quantized->data(), 
                                       step,
                                       size,
                                       (1<<(quantize_bit-1)),
                                       quantize_bit);
  return step;
}

// revert quantized bit into a full tensor
void QuantizerBase::dequantize_do(Tensor t, Tensor quantized, float step, int quantize_bit) { 
  cudaSetDevice(t->getDevice().no);

  int size = t->size();

  int threads = 512;
  int blocksSample = (size + threads - 1) / threads;

  // convert to int8 as you can't do bit manipulation in float
  gDequantize<<<blocksSample, threads>>>(t->data(),
                                         (uint8_t*) quantized->data(), 
                                         step,
                                         size,
                                         quantize_bit);
}

}


