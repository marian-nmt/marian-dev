#ifdef _MSC_VER
#pragma warning(disable: 4505) // warning C4505: '__float2half_rz': unreferenced local function has been removed (missing 'static inline')
#endif

#include <cublas_v2.h>

// clang-format off
#include "tensors/gpu/prod.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/cuda_helpers.h"
// clang-format on

namespace marian {

namespace gpu {

// The explicit version of matmult like cublasGemmEx choose their math mode based on the algorithm that
// has been passed into the function call and seem to ignore setMathMode. Here we query the used math mode
// to choose the algorithm.
static bool tensorOpsEnabled(cublasHandle_t cublasHandle) {
#if CUDA_VERSION >= 9000
  cublasMath_t actual = CUBLAS_DEFAULT_MATH;
  cublasGetMathMode(cublasHandle, &actual);
  return actual == CUBLAS_TENSOR_OP_MATH;
#else
  return false;
#endif
}

static void setTensorMode(cublasHandle_t cublasHandle) {
  cublasHandle; // fool warnings
#if CUDA_VERSION >= 9000
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
      CUBLAS_CHECK(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
      cublasMath_t actual = CUBLAS_DEFAULT_MATH;
      cublasGetMathMode(cublasHandle, &actual);
      if (actual != CUBLAS_TENSOR_OP_MATH) {
        LOG(warn, "[gpu] TensorCores requested but not available");
        mode = -1;
      }
    }
    if (mode > 0)
      LOG(info, "[gpu] 16-bit TensorCores enabled for float32 matrix operations");
  }
  CUBLAS_CHECK(cublasSetMathMode(cublasHandle, mode > 0 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
#endif
}

static void unsetTensorMode(cublasHandle_t cublasHandle) {
  cublasHandle; // fool warnings
#if CUDA_VERSION >= 9000
  CUBLAS_CHECK(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));
#endif
}

// primary template for specialization with different element and compute types
template <typename ElementType, typename ComputeType>
struct TypedGemm { };

template <>
struct TypedGemm</*ElementType=*/float, /*ComputeType=*/float> { // specialization for element type float32 and compute type float32
  static void gemm(cublasHandle_t handle,
                   CudaCompute computeCapability,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m, int n, int k,
                   const float* alpha, // has to match compute type!
                   const float* A, int lda,
                   const float* B, int ldb,
                   const float* beta,  // has to match compute type!
                   float* C, int ldc) {
  // double #if and if unfortunately required to safeguard against compilation error 
  // with CUDA 8.0 and runtime error with CUDA >9.0 on GPUs with compute capability under 5
  #if CUDA_VERSION > 9000
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    if(computeCapability.major >= 5)
      CUBLAS_CHECK(cublasGemmEx(handle, transa, transb,
                                m, n, k, alpha,
                                A, CUDA_R_32F, lda,
                                B, CUDA_R_32F, ldb, beta,
                                C, CUDA_R_32F, ldc,
                                CUDA_R_32F, algorithm));
    else // don't lose the "else"
  #endif
      CUBLAS_CHECK(cublasSgemm(handle, transa, transb,
                               m, n, k, alpha,
                               A, lda,
                               B, ldb, beta,
                               C, ldc));
  
  }

  static void batchedGemm(cublasHandle_t handle,
                          CudaCompute computeCapability,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m, int n, int k,
                          const float *alpha, // has to match compute type!
                          const float *Aarray[], int lda,
                          const float *Barray[], int ldb,
                          const float *beta,  // has to match compute type!
                          float *Carray[], int ldc,
                          int batchCount) {
  // double #if and if unfortunately required to safeguard against compilation error
  // with CUDA 8.0 and runtime error with CUDA >9.0 on GPUs with compute capability under 5
  #if CUDA_VERSION > 9000
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    if(computeCapability.major >= 5)
      CUBLAS_CHECK(cublasGemmBatchedEx(handle, transa, transb,
                                       m, n, k, alpha,
                                       (void* const*)Aarray, CUDA_R_32F, lda,
                                       (void* const*)Barray, CUDA_R_32F, ldb, beta,
                                       (void**)Carray, CUDA_R_32F, ldc, batchCount,
                                       CUDA_R_32F, algorithm));
    else // don't lose the "else"
  #endif
      CUBLAS_CHECK(cublasSgemmBatched(handle, transa, transb,
                                      m, n, k, alpha,
                                      Aarray, lda,
                                      Barray, ldb, beta,
                                      Carray, ldc, batchCount));
  }

  static void stridedBatchedGemm(cublasHandle_t handle,
                                 CudaCompute computeCapability,
                                 cublasOperation_t transa, 
                                 cublasOperation_t transb,
                                 int m, int n, int k,
                                 const float *alpha,
                                 const float *A, int lda, int strideA,
                                 const float *B, int ldb, int strideB,
                                 const float *beta,
                                 float *C, int ldc, int strideC,
                                 int batchCount) {
  // double #if and if unfortunately required to safeguard against compilation error 
  // with CUDA 8.0 and runtime error with CUDA >9.0 on GPUs with compute capability under 5
  #if CUDA_VERSION > 9000
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    if(computeCapability.major >= 5)
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, transa, transb, 
                                              m, n, k, alpha, 
                                              (void const*)A, CUDA_R_32F, lda, strideA,
                                              (void const*)B, CUDA_R_32F, ldb, strideB, beta,
                                              (void*)C, CUDA_R_32F, ldc, strideC, batchCount,
                                              CUDA_R_32F, algorithm));
  #endif
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, transa, transb, 
                                           m, n, k, alpha, 
                                           A, lda, strideA,
                                           B, ldb, strideB, 
                                           beta,
                                           C, ldc, strideC, 
                                           batchCount));
  }  
};

#if COMPILE_FP16
template <>
struct TypedGemm</*ElementType=*/half, /*ComputeType=*/half> { // specialization for element type float16 and compute type float16
  // overload for half, contains configuration settings for float16
  static void gemm(cublasHandle_t handle,
                   CudaCompute computeCapability,
                   cublasOperation_t transa, 
                   cublasOperation_t transb,
                   int m, int n, int k,
                   const half* alpha,  // has to match compute type!
                   const half* A, int lda,
                   const half* B, int ldb,
                   const half* beta,  // has to match compute type!
                   half* C, int ldc) {
    ABORT_IF(computeCapability.major < 6, "Compute capability {} below 6 should not happen for FP16", computeCapability.major);
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK(cublasGemmEx(handle, transa, transb,
                              m, n, k, alpha,
                              A, CUDA_R_16F, lda,
                              B, CUDA_R_16F, ldb, beta,
                              C, CUDA_R_16F, ldc,
                              CUDA_R_16F, algorithm)); // @TODO: review algorithm
  }

  static void batchedGemm(cublasHandle_t handle,
                          CudaCompute computeCapability,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m, int n, int k,
                          const half *alpha,  // has to match compute type!
                          const half *Aarray[], int lda,
                          const half *Barray[], int ldb,
                          const half *beta,   // has to match compute type!
                          half *Carray[], int ldc,
                          int batchCount) {
    ABORT_IF(computeCapability.major < 6, "Compute capability {} below 6 should not happen for FP16", computeCapability.major);
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK(cublasGemmBatchedEx(handle, transa, transb,
                                     m, n, k, alpha,
                                     (void* const*)Aarray, CUDA_R_16F, lda,
                                     (void* const*)Barray, CUDA_R_16F, ldb, beta,
                                     (void**)Carray, CUDA_R_16F, ldc, batchCount,
                                     CUDA_R_16F, algorithm));
  }

  static void stridedBatchedGemm(cublasHandle_t handle,
                                 CudaCompute computeCapability,
                                 cublasOperation_t transa, 
                                 cublasOperation_t transb,
                                 int m, int n, int k,
                                 const half *alpha,
                                 const half *A, int lda, int strideA,
                                 const half *B, int ldb, int strideB,
                                 const half *beta,
                                 half *C, int ldc, int strideC,
                                 int batchCount) {
    ABORT_IF(computeCapability.major < 6, "Compute capability {} below 6 should not happen for FP16", computeCapability.major);
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, transa, transb, 
                                            m, n, k, alpha, 
                                            (void const*)A, CUDA_R_16F, lda, strideA,
                                            (void const*)B, CUDA_R_16F, ldb, strideB, beta,
                                            (void*)C, CUDA_R_16F, ldc, strideC, batchCount,
                                            CUDA_R_16F, algorithm));
  }
  
};

template <>
struct TypedGemm</*ElementType=*/half, /*ComputeType=*/float> { // specialization for element type float16 and compute type float32
// overload for half, contains configuration settings for float16 and accumulation in float32
  static void gemm(cublasHandle_t handle,
                   CudaCompute computeCapability,
                   cublasOperation_t transa, 
                   cublasOperation_t transb,
                   int m, int n, int k,
                   const float* alpha, // has to match compute type!
                   const half* A, int lda,
                   const half* B, int ldb,
                   const float* beta, // has to match compute type!
                   half* C, int ldc) {
    ABORT_IF(computeCapability.major < 6, "Compute capability {} below 6 should not happen for FP16", computeCapability.major);
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK(cublasGemmEx(handle, transa, transb, 
                              m, n, k, alpha,
                              A, CUDA_R_16F, lda,
                              B, CUDA_R_16F, ldb, beta,
                              C, CUDA_R_16F, ldc,
                              CUDA_R_32F, algorithm)); // use 32-bit compute type for accumulation
  }

  static void batchedGemm(cublasHandle_t handle,
                          CudaCompute computeCapability,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m, int n, int k,
                          const float *alpha, // has to match compute type!
                          const half *Aarray[], int lda,
                          const half *Barray[], int ldb,
                          const float *beta,  // has to match compute type!
                          half *Carray[], int ldc,
                          int batchCount) {
    ABORT_IF(computeCapability.major < 6, "Compute capability {} below 6 should not happen for FP16", computeCapability.major);
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK(cublasGemmBatchedEx(handle, transa, transb,
                                     m, n, k, alpha,
                                     (void* const*)Aarray, CUDA_R_16F, lda,
                                     (void* const*)Barray, CUDA_R_16F, ldb, beta,
                                     (void**)Carray, CUDA_R_16F, ldc, batchCount,
                                     CUDA_R_32F, algorithm)); // use 32-bit compute type for accumulation
  }

  static void stridedBatchedGemm(cublasHandle_t handle,
                                 CudaCompute computeCapability,
                                 cublasOperation_t transa, 
                                 cublasOperation_t transb,
                                 int m, int n, int k,
                                 const float *alpha,
                                 const half *A, int lda, int strideA,
                                 const half *B, int ldb, int strideB,
                                 const float *beta,
                                 half *C, int ldc, int strideC,
                                 int batchCount) {
    ABORT_IF(computeCapability.major < 6, "Compute capability {} below 6 should not happen for FP16", computeCapability.major);
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, transa, transb, 
                                            m, n, k, alpha, 
                                            (void const*)A, CUDA_R_16F, lda, strideA,
                                            (void const*)B, CUDA_R_16F, ldb, strideB, beta,
                                            (void*)C, CUDA_R_16F, ldc, strideC, batchCount,
                                            CUDA_R_32F, algorithm));  
  }
};
#endif


// overload for float, contains configuration settings for float32
template <typename ElementType, typename ComputeType>
void ProdTyped(marian::Tensor C,
               const marian::Tensor& A,
               const marian::Tensor& B,
               bool transA,
               bool transB,
               ComputeType beta,
               ComputeType scalar) {
  CUDA_CHECK(cudaSetDevice((int)C->getDeviceId().no));
  ComputeType alpha = scalar;

  int m = A->shape().elements() / A->shape().back();
  int k = A->shape().back();
  if(transA)
    std::swap(m, k);

  int l = B->shape().elements() / B->shape().back();
  int n = B->shape().back();
  if(transB)
    std::swap(l, n);

  int lda = A->shape().back();
  int ldb = B->shape().back();
  int ldc = B->shape().back();

  if(transB)
    ldc = B->shape().elements() / B->shape().back();

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto backend = std::static_pointer_cast<gpu::Backend>(C->getBackend());
  auto cublasHandle = backend->getCublasHandle();
  auto computeCapability = backend->getCudaComputeCapability();

  setTensorMode(cublasHandle);
  TypedGemm<ElementType, ComputeType>::gemm(cublasHandle, computeCapability,
                                            opB, opA,
                                            n, m, k,
                                            &alpha,
                                            B->data<ElementType>(), ldb,
                                            A->data<ElementType>(), lda,
                                            &beta,
                                            C->data<ElementType>(), ldc);
  unsetTensorMode(cublasHandle);
}

void Prod(marian::Tensor C,
          const marian::Tensor& A,
          const marian::Tensor& B,
          bool transA,
          bool transB,
          float beta,
          float scalar) {
  gpu::Prod(C, A, B, transA, transB, beta, scalar, C->type());
}

void Prod(marian::Tensor C,
          const marian::Tensor& A,
          const marian::Tensor& B,
          bool transA,
          bool transB,
          float beta,
          float scalar,
          Type computeType) {
  if(C->type() == Type::float32 && computeType == Type::float32) {
    ProdTyped</*ElementType=*/float, /*ComputeType=*/float>(C, A, B, transA, transB, beta, scalar);
#if COMPILE_FP16
  } else if(C->type() == Type::float16 && computeType == Type::float16) {
    ProdTyped</*ElementType=*/half, /*ComputeType=*/half>(C, A, B, transA, transB, __float2half(beta), __float2half(scalar));
  } else if(C->type() == Type::float16 && computeType == Type::float32) {
    ProdTyped</*ElementType=*/half, /*ComputeType=*/float>(C, A, B, transA, transB, beta, scalar);
#endif
  } else {
    ABORT("Prod not implemented for element type {} and compute type {}", C->type(), computeType);
  }
}

template <typename ElementType, typename ComputeType>
void ProdBatchedTyped(marian::Tensor C,                 
                      Ptr<Allocator> allocator,
                      const marian::Tensor A,
                      const marian::Tensor B,
                      bool transA,
                      bool transB,
                      ComputeType beta,
                      ComputeType scalar) {
  CUDA_CHECK(cudaSetDevice((int)C->getDeviceId().no));
  ComputeType alpha = scalar;

  int batchDimA = A->shape().elements() / (A->shape()[-1] * A->shape()[-2]);
  int batchDimB = B->shape().elements() / (B->shape()[-1] * B->shape()[-2]);

  int m = A->shape()[-2];
  int k = A->shape()[-1];
  if(transA)
    std::swap(m, k);

  int l = B->shape()[-2];
  int n = B->shape()[-1];
  if(transB)
    std::swap(l, n);

  int lda = A->shape()[-1];
  int ldb = B->shape()[-1];
  int ldc = B->shape()[-1];

  if(transB)
    ldc = B->shape()[-2];

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto backend = std::static_pointer_cast<gpu::Backend>(C->getBackend());
  auto cublasHandle = backend->getCublasHandle();
  auto compute = backend->getCudaComputeCapability();

  auto strideA = batchDimA == 1 ? 0 : m * k;
  auto strideB = batchDimB == 1 ? 0 : n * k;
  auto strideC = n * m;

  if(batchDimA == batchDimB) {
    setTensorMode(cublasHandle);
    TypedGemm<ElementType, ComputeType>::stridedBatchedGemm(cublasHandle, compute, 
                                                            opB, opA, 
                                                            n, m, k,
                                                            &alpha,
                                                            B->data<const ElementType>(), ldb, strideB, 
                                                            A->data<const ElementType>(), lda, strideA, 
                                                            &beta, 
                                                            C->data<ElementType>(), ldc, strideC, 
                                                            batchDimA);
    unsetTensorMode(cublasHandle);
  } else {
    auto batchDimC = std::max(batchDimA, batchDimB);
    size_t size = 3*batchDimC;
    std::vector<ElementType*> ptrs(size);
    auto aStart = 0;
    auto bStart = batchDimC;
    auto cStart = bStart + batchDimC;

    for(int i = 0; i < batchDimC; i++) {
      ptrs[aStart + i] = A->data<ElementType>() + (i % batchDimA) * strideA;
      ptrs[bStart + i] = B->data<ElementType>() + (i % batchDimB) * strideB;
      ptrs[cStart + i] = C->data<ElementType>() + i * strideC;
    }

    // auto fails here from weird reason
    IPtr<MemoryPiece> mp_ptrs = allocator->alloc<ElementType*>(size); 
    ElementType** dest = mp_ptrs->data<ElementType*>();
    cudaStream_t cublasStream = 0;
    CUBLAS_CHECK(cublasGetStream(cublasHandle, &cublasStream));
    CUDA_CHECK(cudaMemcpyAsync(dest, ptrs.data(), size * sizeof(ElementType*), cudaMemcpyHostToDevice, cublasStream));

    setTensorMode(cublasHandle);
    TypedGemm<ElementType, ComputeType>::batchedGemm(cublasHandle, compute,
                                                     opB, opA,
                                                     n, m, k,
                                                     &alpha,
                                                     mp_ptrs->data<const ElementType*>() + bStart, ldb,
                                                     mp_ptrs->data<const ElementType*>() + aStart, lda,
                                                     &beta,
                                                     mp_ptrs->data<ElementType*>() + cStart, ldc,
                                                     batchDimC);
    unsetTensorMode(cublasHandle);

    allocator->free(mp_ptrs);
  }
}

// @TODO: add version with compute type for completeness
void ProdBatched(marian::Tensor C,
                 Ptr<Allocator> allocator,
                 const marian::Tensor A,
                 const marian::Tensor B,
                 bool transA,
                 bool transB,
                 float beta,
                 float scalar) {
  if(C->type() == Type::float32) {
    ProdBatchedTyped<float, float>(C, allocator, A, B, transA, transB, beta, scalar);
#if COMPILE_FP16
  } else if(C->type() == Type::float16) { // not a *.cu file
    ProdBatchedTyped<half, half>(C, allocator, A, B, transA, transB, __float2half(beta), __float2half(scalar));
#endif
  } else {
    ABORT("ProdBatched not implemented for element type {}", C->type());
  }
}

}  // namespace gpu
}  // namespace marian
