#if DNNL_FOUND
#include <oneapi/dnnl/dnnl.hpp>
#include "tensors/cpu/bf16/bf16.h"
#endif

#if MKL_FOUND
#include <mkl.h>
#elif BLAS_FOUND
#include <cblas.h>
#endif

inline void sgemm(bool transA,
                  bool transB,
                  int rows_a,
                  int rows_b,
                  int width,
                  float alpha,
                  float* a,
                  int lda,
                  float* b,
                  int ldb,
                  float beta,
                  float* c,
                  int ldc,
                  bool cache=false) {
// MKL_FOUND also implies BLAS_FOUND so use DNNL only if MKL is not found
#if defined(DNNL_FOUND) //&& !defined(MKL_FOUND)
#if 1
  /*sgemm_dnnl*/gemm_f32f32bf16f32(transA,
             transB,
             rows_a,
             rows_b,
             width,
             alpha,
             a,
             lda,
             b,
             ldb,
             beta,
             c,
             ldc,
             cache);
#else
  dnnl::sgemm(transA ? 't' : 'n',
              transB ? 't' : 'n',
              rows_a,
              rows_b,
              width,
              alpha,
              a,
              lda,
              b,
              ldb,
              beta,
              c,
              ldc);
#endif
#elif defined(BLAS_FOUND)
  cblas_sgemm(CblasRowMajor,
              transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans,
              rows_a,
              rows_b,
              width,
              alpha,
              a,
              lda,
              b,
              ldb,
              beta,
              c,
              ldc);
#else
    transA; transB; rows_a; rows_b; width; alpha; a; lda; b; ldb; beta; c; ldc; // make compiler happy
    ABORT("Marian must be compiled with a BLAS library");
#endif
}
