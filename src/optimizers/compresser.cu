#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>


#include "optimizers/compresser.h"
#include "tensors/tensor_operators.h"
#include "tensors/tensor_allocator.h"



namespace marian {

  __global__ void gClip(float* data,
                            int size,
                            float range) {
      int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if(idx >= size)
        return;                       
    
      if (data[idx] < -range) data[idx] = -range;
      if (data[idx] > range) data[idx] = range;
    }

  __global__ void gQuantize(float* data,
                            int size,
                            int num_centers,
                            float base,
                            float max) {
  
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)
      return;

    // get sign flag
    bool isNeg = false;
      if (data[idx] < 0) {
      isNeg = true;
      data[idx] *= -1;
    }

    // compute the log of the parameter
    data[idx] /= max;
    int center = round(log(data[idx]) / log(base));
    
    // clip the center to [0, 2^(bit-1)-1]    
    if (center < -num_centers)
      center = -num_centers;
    if (center > 0)
      center = 0;

    // revert back to floating point representation
    data[idx] = std::pow(base, center) * max;
    if (isNeg)
      data[idx] *= -1;
  }

  void Compresser::compressImpl(Tensor t, int bit, float base, float clipRange){
    cudaSetDevice(t->getDeviceId().no);
    int threads = 512;
    int blocksSample = 1 + t->size() / threads;
    
    // clip first
    if (clipRange > 0.0)
      gClip<<<blocksSample, threads>>>(t->data(), t->size(), clipRange);

    // get max element in Tensor
    thrust::device_ptr<float> d_ptr(t->data());
    float max = *(thrust::max_element(d_ptr, d_ptr + t->size()));
    float min = *(thrust::min_element(d_ptr, d_ptr + t->size())) * -1;
    max = std::max(max, min);

    // get maximum center
    max = std::pow(base, std::round(std::log(max) / std::log(base)));
    
    // compress by log quantization
    gQuantize<<<blocksSample, threads>>>(t->data(), t->size(), (1<<(bit-1)) - 1, base, max);
  }	
}
