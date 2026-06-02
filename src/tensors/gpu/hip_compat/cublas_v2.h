#pragma once
// MOAT HIP shim: <cublas_v2.h> -> hipBLAS. Aliases the cuBLAS symbols Marian
// uses (prod.cpp, prod_sparse.cpp, backend.h) to their hipBLAS equivalents.
#include <hipblas/hipblas.h>

using cublasHandle_t    = hipblasHandle_t;
using cublasStatus_t    = hipblasStatus_t;
using cublasOperation_t = hipblasOperation_t;
using cublasMath_t      = hipblasMath_t;
using cublasComputeType_t = hipblasComputeType_t;

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N           HIPBLAS_OP_N
#define CUBLAS_OP_T           HIPBLAS_OP_T

#define CUBLAS_DEFAULT_MATH         HIPBLAS_DEFAULT_MATH
#define CUBLAS_TENSOR_OP_MATH       HIPBLAS_TENSOR_OP_MATH
// hipBLAS has no separate FP32 tensor-op math mode; the plain tensor-op mode is
// the closest analog.
#define CUBLAS_TENSOR_OP_MATH_FP32  HIPBLAS_TENSOR_OP_MATH

#define CUBLAS_GEMM_DEFAULT            HIPBLAS_GEMM_DEFAULT
// hipBLAS has no separate tensor-op gemm algorithm selector.
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT

#define CUBLAS_COMPUTE_16F          HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F          HIPBLAS_COMPUTE_32F
// hipBLAS has no 32F_FAST_16F (TF32-like) compute type; fall back to plain 32F.
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F

#define cublasCreate            hipblasCreate
#define cublasDestroy           hipblasDestroy
#define cublasGetStream         hipblasGetStream
#define cublasSetStream         hipblasSetStream
#define cublasGetMathMode       hipblasGetMathMode
#define cublasSetMathMode       hipblasSetMathMode
#define cublasSgemm             hipblasSgemm
#define cublasSgemmBatched      hipblasSgemmBatched
#define cublasGemmEx            hipblasGemmEx
#define cublasGemmBatchedEx     hipblasGemmBatchedEx
