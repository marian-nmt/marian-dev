#include <stdio.h>
#include <stdlib.h>
#include <boost/variant.hpp>

#include "tensors/tensor.h"

#if MKL_FOUND
#include <mkl_vsl.h>

namespace marian {

namespace cpu {

void Dropout(Tensor tensor, float h, VSLStreamStatePtr gen);

void Gaussian(Tensor tensor, float mean, float stddev, VSLStreamStatePtr gen);

}

}
#endif

#if CUDA_FOUND
#include <cuda.h>
#include <curand.h>

namespace marian {

namespace gpu {

void Dropout(Tensor tensor, float h, curandGenerator_t gen);

void Gaussian(Tensor tensor, float mean, float stddev, curandGenerator_t gen);

}

}
#endif

#if !MKL_FOUND
typedef void* VSLStreamStatePtr;
#endif

#if !CUDA_FOUND
typedef void* curandGenerator_t;
#endif

#if MKL_FOUND || CUDA_FOUND
namespace marian {

#if MKL_FOUND && CUDA_FOUND
typedef boost::variant<VSLStreamStatePtr, curandGenerator_t> RNG;
#elif MKL_FOUND
typedef boost::variant<VSLStreamStatePtr> RNG;
#elif CUDA_FOUND
typedef boost::variant<curandGenerator_t> RNG;
#endif

void Dropout(Tensor tensor, float h, RNG gen);

void Gaussian(Tensor tensor, float mean, float stddev, RNG gen);

}
#endif
