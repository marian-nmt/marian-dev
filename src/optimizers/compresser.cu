#include <cmath>

#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "optimizers/compresser.h"
#include "tensors/tensor_operators.h"
#include "tensors/tensor_allocator.h"

#include "functional/functional.h"
#include "functional/floats.h"

namespace marian {

   /* simulate a fixed quantization for values in data.
   * For example:
   * data  = [0.96, 0.73, 0.82, 0.84, 0.42, 0.29, 0.65]
   * quant = [1   , 0.6,  0.8 , 0.8 , 0.4,  0.2 , 0.6 ]
   *
   * @param data contains the original data
   * @param quant will contain the resulting quantized data. set data = quant for in-place operation
   * @param num_centers the number of quantized centers in absolute. It should be 2^(bit-1)
   * @param S stores the scaling factor.
   */
   void quantize_fixed(Tensor data, Tensor res, int num_centers, float S) {
     using namespace functional;
     float multiplier = num_centers / S;
     
     // clip based on the scale
     Element(_1 = clip(_2, S), res, data);

     // get the quantization center
     Element(_1 = round(_1 * multiplier), res); 
     
     // revert back to floating point representation
     Element(_1 /= multiplier, res);
 
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
   void quantize_log(Tensor data, Tensor res, int num_centers, float S, int base = 2) {
     using namespace functional;     

     // multiplier such that the quantization is rounded in normal-space instead of log space.
     // 4/3 for base = 2. example: 11.8 should be quantized to 8, instead of 16. 
     float _mult = (2.0 * base) / (1.0 + base);
     
     // get the quantization center
     Element(_1 = floor(log(abs(_2 / S) * _mult) / log(base)), res, data);

     // clip the center to [0, 2^(bit-1)-1]
     Element(_1 = clip(_1, num_centers), res);

     // revert back to floating point representation
     Element(_1 = pow(base, _1) * S * sgn(_2), res, data);

   }

  void Compresser::compressImpl(Tensor t, int bit, int kMeanStep, bool logQuant){
    cudaSetDevice(t->getDeviceId().no);

    // Lazy init delta variable
    int id = t->getDeviceId().no;
    if (!delta && kMeanStep > 0) {
      int msize = t->size();
      alloc_ = New<TensorAllocator>(t->getBackend());

      int elements = (int)msize;
      alloc_->reserveExact(msize *sizeof(float));
      alloc_->allocate(delta, {1, elements});
  
    }

    float S = 0;
    // get intial scaling factor (S) based on max element in Tensor
    thrust::device_ptr<float> d_ptr(t->data());
    float max = *(thrust::max_element(d_ptr, d_ptr + t->size()));
    float min = *(thrust::min_element(d_ptr, d_ptr + t->size())) * -1;
    S = std::max(max, min);

    // optimze the scaling factor S
    for (int i=0;i< kMeanStep;i++) {
      // let t be the original tensor, and q be the quantised tensor, and q = S*a where S is the scaling factor.
      // we want to optimize S to minimize MSE(S*a - t)
      // therefore, S = sum(a*t)/sum(a*a)
      // see https://www.aclweb.org/anthology/2020.ngt-1.4.pdf for more details.
      if (logQuant)
        quantize_log(t, delta->subtensor(0, t->size()), (1<<(bit-1)) - 1, S);
      else      
        quantize_fixed(t, delta->subtensor(0, t->size()) , (1<<(bit-1)) - 1, S);

      {
	// obtains a by applying q/=S
        using namespace functional;
        Element(_1 /= S, delta);
      }

      thrust::device_ptr<float> delta_ptr(delta->data());
      float delta_top = thrust::inner_product(delta_ptr, delta_ptr + (int)t->size(), d_ptr, 0.0f); // computes (a*t)
      float delta_btm = thrust::inner_product(delta_ptr, delta_ptr + (int)t->size(), delta_ptr, 0.0f); // computes (a*a)
      
      S = delta_top / delta_btm; // S = (a*t)/(a*a)
    }
    // final compress
    if (logQuant)
      quantize_log(t, t, (1<<(bit-1)) - 1, S);
    else
      quantize_fixed(t, t, (1<<(bit-1)) - 1, S); 
  }
}
