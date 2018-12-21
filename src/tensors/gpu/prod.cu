
#include <cublas_v2.h>

// clang-format off
#include "tensors/gpu/prod.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/cuda_helpers.h"
// clang-format on

namespace marian {

namespace gpu {

static void setTensorMode(cublasHandle_t cublasHandle) {
  static int mode = 0;  // 1: use TC; -1: do not use TC; 0: not set yet
  if (mode == 0) { // multi-thread note: this is sort-of thread-safe, since multiple threads would determine the same value
    const char* var = getenv("ENABLE_CUBLAS_TENSOR_OP_MATH_FP32");
    if (!var)
      var = "1";
    switch(var[0]) {
    case '0': mode = -1; break;
    case '1': mode =  1; break;
    default: ABORT("Invalid ENABLE_CUBLAS_TENSOR_OP_MATH_FP32={}", var);
    }
    if (mode > 0) { // try whether it can be set   --@TODO: check whether this actually works
      cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
      cublasMath_t actual = CUBLAS_DEFAULT_MATH;
      cublasGetMathMode(cublasHandle, &actual);
      if (actual != CUBLAS_TENSOR_OP_MATH) {
        LOG(info, "WARNING: TensorCores requested but not available");
        mode = -1;
      }
    }
    if (mode > 0)
      LOG(info, "16-bit TensorCores enabled for float32 matrix operations");
  }
  cublasSetMathMode(cublasHandle, mode > 0 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
}


cublasStatus_t cublasGemmTyped(cublasHandle_t handle,
                               cublasOperation_t transa, 
                               cublasOperation_t transb,
                               int m, int n, int k,
                               const float* alpha,
                               const float* A, int lda,
                               const float* B, int ldb,
                               const float* beta,
                               float* C, int ldc) {
  return cublasSgemm(handle, transa, transb, 
                     m, n, k, alpha, 
                     A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasGemmTyped(cublasHandle_t handle,
                               cublasOperation_t transa, 
                               cublasOperation_t transb,
                               int m, int n, int k,
                               const half* alpha,
                               const half* A, int lda,
                               const half* B, int ldb,
                               const half* beta,
                               half* C, int ldc) {
  return cublasHgemm(handle, transa, transb, 
                     m, n, k, alpha, 
                     A, lda, B, ldb, beta, C, ldc);
}

template <typename T>
void ProdTyped(marian::Tensor C,
               const marian::Tensor& A,
               const marian::Tensor& B,
               bool transA,
               bool transB,
               T beta,
               T scalar) {
  cudaSetDevice(C->getDeviceId().no);
  T alpha = scalar;

  size_t m = A->shape().elements() / A->shape().back();
  size_t k = A->shape().back();
  if(transA)
    std::swap(m, k);

  size_t l = B->shape().elements() / B->shape().back();
  size_t n = B->shape().back();
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape().back();
  size_t ldb = B->shape().back();
  size_t ldc = B->shape().back();

  if(transB)
    ldc = B->shape().elements() / B->shape().back();

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto cublasHandle = std::static_pointer_cast<gpu::Backend>(C->getBackend())
                          ->getCublasHandle();

#if CUDA_VERSION >= 9000
  setTensorMode(cublasHandle);
  //cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
#endif
  cublasGemmTyped(cublasHandle,
                  opB,
                  opA,
                  n,
                  m,
                  k,
                  &alpha,
                  B->data<T>(),
                  ldb,
                  A->data<T>(),
                  lda,
                  &beta,
                  C->data<T>(),
                  ldc);

#if CUDA_VERSION >= 9000
  cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
#endif
}

void Prod(marian::Tensor C,
          const marian::Tensor& A,
          const marian::Tensor& B,
          bool transA,
          bool transB,
          float beta,
          float scalar) {
  if(C->type() == Type::float32) {
    ProdTyped<float>(C, A, B, transA, transB, beta, scalar);
  } else if(C->type() == Type::float16) {
    ProdTyped<half>(C, A, B, transA, transB, __float2half(beta), __float2half(scalar));
  } else {
    ABORT("Prod not implemented for type {}", C->type());
  }
}

cublasStatus_t cublasGemmBatchedTyped(cublasHandle_t handle,
                                      cublasOperation_t transa, 
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const float *alpha,
                                      const float *Aarray[], int lda,
                                      const float *Barray[], int ldb,
                                      const float *beta,
                                      float *Carray[], int ldc, 
                                      int batchCount) {
  return
  cublasSgemmBatched(handle, transa, transb, 
                     m, n, k, alpha, 
                     Aarray, lda, Barray, ldb, beta,
                     Carray, ldc, batchCount);
}

cublasStatus_t cublasGemmBatchedTyped(cublasHandle_t handle,
                                      cublasOperation_t transa, 
                                      cublasOperation_t transb,
                                      int m, int n, int k,
                                      const half *alpha,
                                      const half *Aarray[], int lda,
                                      const half *Barray[], int ldb,
                                      const half *beta,
                                      half *Carray[], int ldc, 
                                      int batchCount) {
  return
  cublasHgemmBatched(handle, transa, transb, 
                     m, n, k, alpha, 
                     Aarray, lda, Barray, ldb, beta,
                     Carray, ldc, batchCount);
}

template <typename T>
void ProdBatchedTyped(marian::Tensor C,
                      Ptr<Allocator> allocator,
                      const marian::Tensor A,
                      const marian::Tensor B,
                      bool transA,
                      bool transB,
                      T beta,
                      T scalar) {
  cudaSetDevice(C->getDeviceId().no);
  T alpha = scalar;

  size_t batchA = A->shape().elements() / (A->shape()[-1] * A->shape()[-2]);
  size_t batchB = B->shape().elements() / (B->shape()[-1] * B->shape()[-2]);

  size_t m = A->shape()[-2];
  size_t k = A->shape()[-1];
  if(transA)
    std::swap(m, k);

  size_t l = B->shape()[-2];
  size_t n = B->shape()[-1];
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape()[-1];
  size_t ldb = B->shape()[-1];
  size_t ldc = B->shape()[-1];

  if(transB)
    ldc = B->shape()[-2];

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto cublasHandle = std::static_pointer_cast<gpu::Backend>(C->getBackend())
                          ->getCublasHandle();

  int strideA = batchA == 1 ? 0 : m * k;
  int strideB = batchB == 1 ? 0 : n * k;
  int strideC = n * m;
  int batchC = std::max(batchA, batchB);

  std::vector<const T*> aptr;
  std::vector<const T*> bptr;
  std::vector<T*> cptr;

  for(int i = 0; i < batchC; i++) {
    aptr.push_back(A->data<T>() + (i % batchA) * strideA);
    bptr.push_back(B->data<T>() + (i % batchB) * strideB);
    cptr.push_back(C->data<T>() + i * strideC);
  }

  IPtr<MemoryPiece> mp_aptr = allocator->alloc<const T*>(aptr.size());
  CudaCopy(
      aptr.data(), aptr.data() + aptr.size(), mp_aptr->data<const T*>());

  IPtr<MemoryPiece> mp_bptr = allocator->alloc<const T*>(bptr.size());
  CudaCopy(
      bptr.data(), bptr.data() + bptr.size(), mp_bptr->data<const T*>());

  IPtr<MemoryPiece> mp_cptr = allocator->alloc<T*>(cptr.size());
  CudaCopy(cptr.data(), cptr.data() + cptr.size(), mp_cptr->data<T*>());

#if CUDA_VERSION >= 9000
  setTensorMode(cublasHandle);
  //cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
#endif
  cublasGemmBatchedTyped(cublasHandle,
                         opB,
                         opA,
                         n,
                         m,
                         k,
                         &alpha,
                         mp_bptr->data<const T*>(),
                         ldb,
                         mp_aptr->data<const T*>(),
                         lda,
                         &beta,
                         mp_cptr->data<T*>(),
                         ldc,
                         batchC);
#if CUDA_VERSION >= 9000
  cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
#endif

  allocator->free(mp_aptr);
  allocator->free(mp_bptr);
  allocator->free(mp_cptr);
}

void ProdBatched(marian::Tensor C,
                 Ptr<Allocator> allocator,
                 const marian::Tensor A,
                 const marian::Tensor B,
                 bool transA,
                 bool transB,
                 float beta,
                 float scalar) {
  if(C->type() == Type::float32) {
    ProdBatchedTyped<float>(C, allocator, A, B, transA, transB, beta, scalar);
  } else if(C->type() == Type::float16) {
    ProdBatchedTyped<half>(C, allocator, A, B, transA, transB, __float2half(beta), __float2half(scalar));
  } else {
    ABORT("ProdBatched not implemented for type {}", C->type());
  }
}

}  // namespace gpu
}  // namespace marian
