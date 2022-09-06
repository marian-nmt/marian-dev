#if DNNL_FOUND
#include <oneapi/dnnl/dnnl.hpp>
#else
#if BLAS_FOUND
#include <cblas.h>
#endif
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
                  int ldc) {
#if DNNL_FOUND
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
#else
#if BLAS_FOUND
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
#endif
}
