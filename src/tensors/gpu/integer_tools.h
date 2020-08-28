#pragma once

#include <cublas_v2.h>
#include "tensors/gpu/backend.h"
#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"
#include "tensors/tensor_allocator.h"

namespace marian {
namespace gpu {
namespace integer {

//Convenient function to get rows and columns of a tensor, shadowed by namespace.
inline int cols(Tensor& tensor) { return tensor->shape()[-1]; }
inline int rows(Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }

#ifdef CUDA_FOUND
    void maxAbsQuantMult(cublasHandle_t& handle, const float * input_gpu, size_t items, float * output_gpu);
    void quantize(const float * input, int8_t * output, size_t rows, size_t cols, const float * quantMultAddr);
    void quantizeToRowMajorWrapper(const float * input, int8_t * output, size_t rows, size_t cols, const float * quantMultAddr);
    void dequantize(const int32_t * input, float * output, size_t rows, size_t cols, const float * quantMultAaddr, const float * quantMultBaddr);
    void cutlass_igemm_dispatcher(bool transA, bool transB,
        int M,
        int N,
        int K,
        float alpha,
        int8_t const *A,
        int lda,
        int8_t const *B,
        int ldb,
        float beta,
        int32_t *C,
        int ldc,
        bool tensorCore = false);
    void gpuPrinterDispatch(float * mem, size_t idx);
    void gpuPrinterDispatch(int32_t * mem, size_t idx);
    void gpuPrinterDispatch(int8_t * mem, size_t idx);
#else
    void maxAbsQuantMult(cublasHandle_t& handle, const float * input_gpu, size_t items, float * output_gpu) {
        handle;
        input_gpu;
        items;
        output_gpu;
        return;
    }
    void quantize(const float * intput, int8_t * output, size_t rows, size_t cols, const float * quantMultAddr) {
        input;
        output;
        rows;
        cols;
        quantMult;
        return;
    }
    void quantizeToRowMajorWrapper(const float * input, int8_t * output, size_t rows, size_t cols, const float * quantMultAddr) {
        input;
        output;
        rows;
        cols;
        quantMult;
        return;
    }
    void dequantize(const int32_t * input, float * output, size_t rows, size_t cols, const float * quantMultAaddr, const float * quantMultBaddr) {
        input;
        output;
        rows;
        cols;
        quantMultAaddr;
        quantMultBaddr;
        return;
    }
    void cutlass_igemm_dispatcher(bool transA, bool transB,
        int M,
        int N,
        int K,
        float alpha,
        int8_t const *A,
        int lda,
        int8_t const *B,
        int ldb,
        float beta,
        int32_t *C,
        int ldc) {
            M;
            N;
            K;
            alpha;
            A;
            lda;
            B;
            ldb;
            beta;
            C;
            ldc;
        }
        //void gpuPrinterDispatch(float * mem, size_t idx) {
        //    mem;
        //    idx;
        //}
#endif
} // namespace integer
} // namespace gpu
} // namespace marian