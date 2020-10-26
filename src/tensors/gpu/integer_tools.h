#pragma once

#ifdef CUDA_FOUND
#include <cublas_v2.h>
#else
struct cublasHandle_t;
#endif
#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"
#include "tensors/tensor_allocator.h"

namespace marian {
namespace gpu {
namespace integer {

/*Kept in CPU space so that we can easily format and print*/
struct MeanStd {
    float mean;
    float stddev;
    float absMean;
    float absStddev;
};

//Convenient function to get rows and columns of a tensor, shadowed by namespace.
inline int cols(Tensor& tensor) { return tensor->shape()[-1]; }
inline int rows(Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }

#ifdef CUDA_FOUND
    float getMaxAbs(cublasHandle_t& handle, const float * input_gpu, size_t items);
    void maxAbsQuantMult(cublasHandle_t& handle, const float * input_gpu, size_t items, float * output_gpu);
    void quantize(const float * input, int8_t * output, size_t rows, size_t cols, const float * quantMultAddr);
    void quantizeToRowMajorWrapper(const float * input, int8_t * output, size_t rows, size_t cols, const float * quantMultAddr);
    void dequantize(const int32_t * input, float * output, size_t rows, size_t cols, const float * quantMultAaddr, const float * quantMultBaddr);
    void dequantize(const int32_t * input, float * output, size_t rows, size_t cols, const float * dequantMultAddr);
    void cutlass_igemm_dispatcher(bool transA, bool transB,
        int M,
        int N,
        int K,
        float * alpha,
        int8_t const *A,
        int lda,
        int8_t const *B,
        int ldb,
        float * beta,
        int32_t *C,
        int ldc,
        bool tensorCore,
        bool fused,
        float * bias,
        bool doRelu);
    void gpuPrinterDispatch(float * mem, size_t idx);
    void gpuPrinterDispatch(int32_t * mem, size_t idx);
    void gpuPrinterDispatch(int8_t * mem, size_t idx);
    void memCpyDevice(float * dest, float * source, size_t elems);
    void memCpyDevice(int8_t * dest, int8_t * source, size_t elems);
    void getDequantMultWrapper(float * output, float * quantMultAaddr, float * quantMultBaddr);
    MeanStd getMeanStd(float * input, size_t elems);
    void fieldSetGPU(float * gpuMem, float value);
    void memCpyHost(float * dest, float * source, size_t elems);
    void memCpyHost(int8_t * dest, int8_t * source, size_t elems);
    /*
    float * unmanagedGPUAlloc(size_t num);
    void unmanagedFree(float * in);*/

#else
    inline float getMaxAbs(cublasHandle_t& /*handle*/, const float * /*input_gpu*/, size_t /*items*/) {return 0.0f;}
    inline void maxAbsQuantMult(cublasHandle_t& /*handle*/, const float * /*input_gpu*/, size_t /*items*/, float * /*output_gpu*/) {}
    inline void quantize(const float * /*input*/, int8_t * /*output*/, size_t /*rows*/, size_t /*cols*/, const float * /*quantMultAddr*/) {}
    inline void quantizeToRowMajorWrapper(const float * /*input*/, int8_t * /*output*/, size_t /*rows*/, size_t /*cols*/, const float * /*quantMultAddr*/) {}
    inline void dequantize(const int32_t * /*input*/, float * /*output*/, size_t /*rows*/, size_t /*cols*/, const float * /*quantMultAaddr*/, const float * /*quantMultBaddr*/) {}
    inline void dequantize(const int32_t * /*input*/, float * /*output*/, size_t /*rows*/, size_t /*cols*/, const float * /*dequantMultAddr*/) {}
    inline void cutlass_igemm_dispatcher(bool /*transA*/, bool /*transB*/,
        int /*M*/,
        int /*N*/,
        int /*K*/,
        float * /*alpha*/,
        int8_t const */*A*/,
        int /*lda*/,
        int8_t const */*B*/,
        int /*ldb*/,
        float * /*beta*/,
        int32_t */*C*/,
        int /*ldc*/,
        bool /*tensorCore*/,
        bool /*fused*/,
        float * /*bias*/,
        bool /*doRelu*/) {}
    inline void memCpyDevice(float * /*dest*/, float * /*source*/, size_t /*elems*/) {}
    inline void memCpyDevice(int8_t * /*dest*/, int8_t * /*source*/, size_t /*elems*/) {}
    inline void getDequantMultWrapper(float * /*output*/, float * /*quantMultAaddr*/, float * /*quantMultBaddr*/) {}
    inline MeanStd getMeanStd(float * /*input*/, size_t /*elems*/) {MeanStd ret; return ret;}
    inline void fieldSetGPU(float * /*gpuMem*/, float /*value*/) {}
    //void gpuPrinterDispatch(float * /*mem*/, size_t /*idx*/) {}
#endif
} // namespace integer
} // namespace gpu
} // namespace marian
