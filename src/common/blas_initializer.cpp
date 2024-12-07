#include "blas_initializer.h"
#if MKL_FOUND
  #include <mkl.h>
#endif
#if OPENBLAS_FOUND
  #include <cblas.h>
#endif

namespace marian {

BLASInitializer::BLASInitializer() {
#if MKL_FOUND
  mkl_set_num_threads(1);
#endif
#if OPENBLAS_FOUND
  openblas_set_num_threads(1);
#endif
}

// Define the global instance
BLASInitializer blasInitializer;

}  // namespace marian
