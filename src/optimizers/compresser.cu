#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>


#include "optimizers/compresser.h"
#include "tensors/tensor_operators.h"
#include "tensors/tensor_allocator.h"



namespace marian {
  __global__ void gQuantize_fixed(float* data,
                            float* delta,
                            int size,
                            int num_centers,
                            float max) {
  
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)
      return;

    // get sign flag
    float dataTemp = data[idx];

    // helper
    // will be 127 / max if we set the bit to be 8
    float multiplier = num_centers / max;
    
    // clip
    if (data[idx] < -max)
      data[idx] = -max;
    if (data[idx] > max)
      data[idx] = max;

    // quantize
    int tmp = round(data[idx] * multiplier);

    // reverse-back
    data[idx] = tmp / multiplier;
    
    
    if (delta != NULL) {  
      // normal delta
      delta[idx] = data[idx] / max;
      
      // scaled delta
      data[idx] = dataTemp;
    }
  }

  __global__ void gQuantize(float* data,
                            float* delta,
                            int size,
                            int num_centers,
                            float base,
                            float max) {
  
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)
      return;

    // get sign flag
    float dataTemp = data[idx];

    bool isNeg = false;
      if (data[idx] < 0) {
      isNeg = true;
      data[idx] *= -1;
    }

    // compute the log of the parameter
    data[idx] /= max;
    int center = floor(log(data[idx] * (2.0 * base)/(1.0 + base)) / log(base));
    
    // clip the center to [0, 2^(bit-1)-1]    
    if (center < -num_centers)
      center = -num_centers;
    if (center > 0)
      center = 0;

    // revert back to floating point representation
    data[idx] = std::pow(base, center) * max;
    if (isNeg)
      data[idx] *= -1;
   
    if (delta != NULL) {
      // normal delta
      delta[idx] = data[idx] / max;
      
      // scaled delta
      data[idx] = dataTemp;
    }
  }

  void compressImpl(Tensor t, int bit, int kMeanStep){
    cudaSetDevice(t->getDeviceId().no);

    // Lazy init delta variable
    int id = t->getDeviceId().no;
    static Tensor delta[8];
    static Ptr<TensorAllocator> alloc_[8];
    if (!delta[id] && kMeanStep > 0) {
      int msize = t->size();
      alloc_[id] = New<TensorAllocator>(t->getBackend());

      int elements = (int)msize;
      alloc_[id]->reserveExact(msize *sizeof(float));
      alloc_[id]->allocate(delta[id], {1, elements});
  
    }

    int threads = 512;
    int blocksSample = 1 + t->size() / threads;
 
    // get intial scale (S) based on max element in Tensor
    thrust::device_ptr<float> d_ptr(t->data());
    float max = *(thrust::max_element(d_ptr, d_ptr + t->size()));
    float min = *(thrust::min_element(d_ptr, d_ptr + t->size())) * -1;
    float S = std::max(max, min);

    // optimze scale 
    for (int i=0;i< kMeanStep;i++) {
      // gQuantize<<<blocksSample, threads>>>(t->data(), delta[id]->data(), t->size(), (1<<(bit-1)) - 1, base, S);
      gQuantize_fixed<<<blocksSample, threads>>>(t->data(), delta[id]->data(), t->size(), (1<<(bit-1)) - 1, S);
      
      thrust::device_ptr<float> delta_ptr(delta[id]->data());
      float delta_top = thrust::inner_product(delta_ptr, delta_ptr + t->size(), d_ptr, 0.0f);
      float delta_btm = thrust::inner_product(delta_ptr, delta_ptr + t->size(), delta_ptr, 0.0f);
      S = delta_top / delta_btm;
    }

    // compress
    // gQuantize<<<blocksSample, threads>>>(t->data(), NULL, t->size(), (1<<(bit-1)) - 1, base, S);
    gQuantize_fixed<<<blocksSample, threads>>>(t->data(), NULL, t->size(), (1<<(bit-1)) - 1, S);
 
  }
}
