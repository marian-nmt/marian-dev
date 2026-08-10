#pragma once

#include "common/config.h"
#include "tensors/backend.h"  // note: this is one folder up
#include "tensors/gpu/cuda_helpers.h"
#include "tensors/gpu/cusparse_include.h"
#include "common/logging.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>

#if defined(USE_HIP) && CUDA_VERSION >= 11000
#include <cublasLt.h> // for cublasLtHandle_t (hipblasLtHandle_t); see getCublasLtHandle()
#endif


namespace marian {
namespace gpu {

// @TODO: in the future this should pobably become a fully fledged CudaInfo class with many attributes
struct CudaCompute {
  int major;
  int minor;
};

class Backend : public marian::Backend {
private:
  void setCudaComputeCapability() {
    CUDA_CHECK(cudaDeviceGetAttribute(&compute_.major, cudaDevAttrComputeCapabilityMajor, (int)deviceId_.no));
    CUDA_CHECK(cudaDeviceGetAttribute(&compute_.minor, cudaDevAttrComputeCapabilityMinor, (int)deviceId_.no));
  }

public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {
    setDevice();
    setCudaComputeCapability();
  }

  ~Backend() {
    setDevice();
    if(cusparseHandle_) {
      cusparseDestroy(cusparseHandle_);
      cusparseHandle_ = 0;
    }
    if(cublasHandle_) {
      cublasDestroy(cublasHandle_);
      cublasHandle_ = 0;
    }
#if defined(USE_HIP) && CUDA_VERSION >= 11000
    if(cublasLtHandle_) {
      hipblasLtDestroy(cublasLtHandle_);
      cublasLtHandle_ = 0;
    }
#endif
  }

  void setDevice() override { CUDA_CHECK(cudaSetDevice((int)deviceId_.no)); }

  void synchronize() override { CUDA_CHECK(cudaStreamSynchronize(0)); }

  cublasHandle_t getCublasHandle() {
    if(!cublasHandle_) { // lazy initialization here to avoid memory usage when unused
      setDevice();
      cublasCreate(&cublasHandle_);
    }
    return cublasHandle_;
  }

  cusparseHandle_t getCusparseHandle() {
    if(!cusparseHandle_) { // lazy initialization here to avoid memory usage when unused
      setDevice();
      cusparseCreate(&cusparseHandle_);
    }
    return cusparseHandle_;
  }

#if defined(USE_HIP) && CUDA_VERSION >= 11000
  // On CUDA a cublasLt handle is encapsulated inside the cublas handle, so
  // prod.cpp casts getCublasHandle() to cublasLtHandle_t. hipBLASLt is a
  // separate library with a distinct handle, so we maintain a dedicated one.
  cublasLtHandle_t getCublasLtHandle() {
    if(!cublasLtHandle_) {
      setDevice();
      hipblasLtCreate(&cublasLtHandle_);
    }
    return cublasLtHandle_;
  }
#endif

  // for CPU, sets to use optimized code for inference.
  // for GPU, this is invalid. for gpu, isOptimized() function always returns false.
  void setOptimized(bool optimize) override {
    LOG_ONCE(info, "setOptimized() not supported for GPU_{}", optimize);
  }
  bool isOptimized() override {
    LOG_ONCE(info, "isOptimized() not supported for GPU");
    return false;
  };

  // for CPU, selects different GEMM types for the inference.
  // for GPU, there's no gemm type. so, it does nothing.
  void setGemmType(std::string gemmType) override {
    LOG_ONCE(info, "setGemmType() not supported for GPU_{}", gemmType);
  }
  GemmType getGemmType() override {
    LOG_ONCE(info, "getGemmType() not supported for GPU");
    return GemmType::Float32;
  }

  // for CPU, sets quantization range of weight matrices for the inference.
  // for GPU, there's no quantization. so, it does nothing.
  void setQuantizeRange(float range) override {
    LOG_ONCE(info, "setQuantizeRange() not supported for GPU_{}", range);
  }
  float getQuantizeRange() override {
    LOG_ONCE(info, "getQuantizeRange() not supported for GPU");
    return 0.f;
  }

  CudaCompute getCudaComputeCapability() { return compute_; }

  size_t getGlobalMemorySize() override {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, (int)deviceId_.no));
    return prop.totalGlobalMem;
  }

private:
  cublasHandle_t cublasHandle_{0};     // make sure it's 0, so it can be initalized lazily
  cusparseHandle_t cusparseHandle_{0}; // as above
#if defined(USE_HIP) && CUDA_VERSION >= 11000
  cublasLtHandle_t cublasLtHandle_{0};
#endif
  CudaCompute compute_;
};
}  // namespace gpu
}  // namespace marian
