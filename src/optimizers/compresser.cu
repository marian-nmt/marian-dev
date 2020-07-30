#include <cmath>

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

     // clip based on the scaling factor
     Element(_1 = clip(_2, S), res, data);
     
     // multiplier such that the quantization is rounded in normal-space instead of log space.
     // 4/3 for base = 2. example: 11.8 should be quantized to 8, instead of 16. 
     float _mult = (2.0 * base) / (1.0 + base);
     
   
     // log-quantization works as the following:
     // 1. capture the sign:
     // sign = sgn(v)
     // 2. get the quantization center:
     // qc = floor(log2(abs(v/S) * _mult))
     // 3. clip the center to make sure we have no more than 2^(bit-1) centers.
     // qc = clip(qc, num_centers)
     // 4. revert back to floating point space:
     // q = 2^qc * S * sign
     //
     // The above steps are writen in 1 call as below, to avoid reserving extra Tensors:
  
     Element(_1 = sgn(_1) // revert the sign back
		  * S     // revert the scaling function
		  * pow(base, // revert from log space to normal FP represtation
		        clip(floor(log(abs(_1 / S) * _mult) / log(base)), // get the quantization center
			     num_centers)), //clip
	     res);
  }


  // helper Tensor init function
  void Compresser::init(Tensor t) {
    // init the swap tensor
    temp_alloc = New<TensorAllocator>(t->getBackend());
    temp_alloc->reserveExact(sizeof(float));
    temp_alloc->allocate(temp_var, {1,1});

    // Lazy init delta variable
    if (!delta && opt_step_ > 0) {
      int msize = t->size();
      alloc_ = New<TensorAllocator>(t->getBackend());
      alloc_->reserveExact(msize * sizeof(float));
      alloc_->allocate(delta, {1, msize});
    }
  }


  /* Tensor compression implementation.
   * @param t is the tensor to be compressed
   * @param bit is the bit size
   * @param kMeanStep is the steps for optimizing the scaling factor S
   * @param logQuant is true when using log-based quantization. Otherwise will use a fixed-point quantization
   */
  void Compresser::compressImpl(Tensor t, int bit, int opt_step, bool log_quant){
    if (!temp_var) init(t);

    Tensor q = delta->subtensor(0, t->size());  // to store the quantized t
    Tensor t_flat = t->subtensor(0, t->size()); // flatten t for reduce    

    float S = 0; 
    // get intial scaling factor (S) based on max element in Tensor
    {
      using namespace functional;
      Reduce(abs(_1), max(_1,_2), 0.0f  ,temp_var , t_flat);
      S = temp_var->get(0);
    }
    
    // optimze the scaling factor S
    for (int i = 0; i < opt_step; i++) {
      // let t be the original tensor, and q be the quantised tensor, and q = S*a where S is the scaling factor.
      // we want to optimize S to minimize MSE(S*a - t)
      // therefore, S = sum(a*t)/sum(a*a)
      // see https://www.aclweb.org/anthology/2020.ngt-1.4.pdf for more details.
      if (log_quant)
        quantize_log(t, q, (1<<(bit-1)) - 1, S);
      else      
        quantize_fixed(t, q, (1<<(bit-1)) - 1, S);

      // obtains a by applying q/=S
      using namespace functional;
      Element(_1 /= S, delta);
      
      // get sum(a*t)
      Reduce(_1 * _2, temp_var, t_flat, q);
      float delta_top = temp_var->get(0);
      
      // get sum(a*a)
      Reduce(_1 * _1, temp_var, q); 
      float delta_btm = temp_var->get(0);
      
      S = delta_top / delta_btm; // S = sum(a*t)/sum(a*a)
    }

    // final compress
    if (log_quant) {
      quantize_log(t, t, (1<<(bit-1)) - 1, S);
    }
    else
      quantize_fixed(t, t, (1<<(bit-1)) - 1, S); 
  }
}
