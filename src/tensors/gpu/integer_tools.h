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
    void maxAbsQuantMult(cublasHandle_t& handle, float * input_gpu, size_t items, float * output_gpu);
    void quantize(const float * input, int8_t * output, size_t rows, size_t cols, const float * quantMultAddr);
    void dequantize(const int8_t * input, float * output, size_t rows, size_t cols, const float * dequantMultAddr);
#else
    void maxAbsQuantMult(cublasHandle_t& handle, float * input_gpu, size_t items, float * output_gpu) {
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
    void dequantize(const int8_t * input, float * output, size_t rows, size_t cols, const float * dequantMultAddr) {
        input;
        output;
        rows;
        cols;
        dequantMultAddr;
        return;
    }
#endif
} // namespace integer
} // namespace gpu
} // namespace marian