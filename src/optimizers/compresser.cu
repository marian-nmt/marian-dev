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

#include "functional/functional.h"


namespace marian {

   /* simulate a fixed quantization for values in data.
   * For example:
   * data  = [0.96, 0.73, 0.82, 0.84, 0.42, 0.29, 0.65]
   * quant = [1   , 0.6,  0.8 , 0.8 , 0.4,  0.2 , 0.6 ]
   *
   * @param data contains the original data
   * @param quant will contain the resulting quantized data. set data = quant for in-place operation
   * @param size the data size
   * @param num_centers the number of quantized centers in absolute. It should be 2^(bit-1)
   * @param max stores the scaling factor.
   */
  __global__ void gQuantize_fixed(float* data,
                            float* quant,
                            int size,
                            int num_centers,
                            float max) {
  
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)
      return;

    quant[idx] = data[idx];

    // helper
    // will be 127 / max if we set the bit to be 8
    float multiplier = num_centers / max;
    
    // clip
    if (quant[idx] < -max)
      quant[idx] = -max;
    if (quant[idx] > max)
      quant[idx] = max;

    // quantize
    int tmp = round(quant[idx] * multiplier);

    // reverse-back
    quant[idx] = tmp / multiplier;
  }

  
  /* simulate a log-based quantization for values in data. The quantized value will be in the form of S*2^q
   * For example:
   * data  = [0.9, 0.7, 0.5, 0.2 , 1.1]
   * quant = [1,   0.5, 0.5, 0.25, 1  ]
   *
   * @param data contains the original data
   * @param quant will contain the resulting quantized data. set data = quant for in-place operation
   * @param size the data size
   * @param num_centers the number of quantized centers in absolute. It should be 2^(bit-1)
   * @param max stores the scaling factor.
   * @param base for log quantized center. Default of 2
   */
  __global__ void gQuantize(float* data,
                            float* quant,
                            int size,
                            int num_centers,
                            float base,
                            float max) {
  
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)
      return;

    quant[idx] = data[idx];

    bool isNeg = false;
      if (quant[idx] < 0) {
      isNeg = true;
      quant[idx] *= -1;
    }

    // compute the log of the parameter
    quant[idx] /= max;
    int center = floor(log(quant[idx] * (2.0 * base)/(1.0 + base)) / log(base));
    
    // clip the center to [0, 2^(bit-1)-1]    
    if (center < -num_centers)
      center = -num_centers;
    if (center > 0)
      center = 0;

    // revert back to floating point representation
    quant[idx] = std::pow(base, center) * max;
    if (isNeg)
      quant[idx] *= -1;
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
 
    // get intial scaling factor (S) based on max element in Tensor
    thrust::device_ptr<float> d_ptr(t->data());
    float max = *(thrust::max_element(d_ptr, d_ptr + t->size()));
    float min = *(thrust::min_element(d_ptr, d_ptr + t->size())) * -1;
    float S = std::max(max, min);

    // optimze the scaling factor S
    for (int i=0;i< kMeanStep;i++) {
      // let t be the original tensor, and q be the quantised tensor, and q = S*a where S is the scaling factor.
      // we want to optimize S to minimize MSE(S*a - t)
      // therefore, S = sum(a*t)/sum(a*a)
      // see https://www.aclweb.org/anthology/2020.ngt-1.4.pdf for more details.
      
      // gQuantize<<<blocksSample, threads>>>(t->data(), delta[id]->data(), t->size(), (1<<(bit-1)) - 1, base, S);
      gQuantize_fixed<<<blocksSample, threads>>>(t->data(), delta[id]->data(), t->size(), (1<<(bit-1)) - 1, S);
      
      {
	// obtains a by applying q/=S
        using namespace functional;
        Element(_1 /= S, delta[id]);
      }

      thrust::device_ptr<float> delta_ptr(delta[id]->data());
      float delta_top = thrust::inner_product(delta_ptr, delta_ptr + (int)t->size(), d_ptr, 0.0f); // computes (a*t)
      float delta_btm = thrust::inner_product(delta_ptr, delta_ptr + (int)t->size(), delta_ptr, 0.0f); // computes (a*a)
      S = delta_top / delta_btm; // S = (a*t)/(a*a)
    }

    // compress
    // gQuantize<<<blocksSample, threads>>>(t->data(), t->data(), t->size(), (1<<(bit-1)) - 1, base, S);
    gQuantize_fixed<<<blocksSample, threads>>>(t->data(), t->data(), t->size(), (1<<(bit-1)) - 1, S);
 
  }
}
