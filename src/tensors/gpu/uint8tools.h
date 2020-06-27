#pragma once

#include <cublas_v2.h>
#include "tensors/gpu/backend.h"
#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"
#include "tensors/tensor_allocator.h"

namespace marian {

namespace hacky8bit {

float maxAbs(cublasHandle_t& handle, float * input_gpu, size_t items, float * scratchMem);

cublasStatus_t cublas8bitGemmm(marian::Tensor& C,
               const marian::Tensor& A,
               const marian::Tensor& B,
               bool transA,
               bool transB,
               float beta,
               float scalar);


cublasStatus_t cublas8bitGemmmEx(cublasHandle_t handle,
        cublasOperation_t transa, 
        cublasOperation_t transb,
        int m, int n, int k,
        const float* alpha,
        const float* A, int lda,
        const float* B, int ldb,
        const float* beta,
        float* C, int ldc);
}
}