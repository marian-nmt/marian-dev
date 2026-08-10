#pragma once

#include <iostream>

#include "common/shape.h"
#include "tensors/tensor.h"

#ifdef CUDNN

#if defined(USE_HIP)
#include <miopen/miopen.h>
#include <memory>
#else
#include <cudnn.h>
#endif

namespace marian {

#if defined(USE_HIP)
// MIOpen is the ROCm counterpart of cuDNN and carries the same concepts under
// different names, so the wrappers below keep one set of declarations and spell
// the handle and descriptor types through these aliases. MIOpen describes
// convolution weights with an ordinary tensor descriptor rather than with a
// separate filter type.
using cudnnHandle_t                = miopenHandle_t;
using cudnnTensorDescriptor_t      = miopenTensorDescriptor_t;
using cudnnFilterDescriptor_t      = miopenTensorDescriptor_t;
using cudnnConvolutionDescriptor_t = miopenConvolutionDescriptor_t;
using cudnnPoolingDescriptor_t     = miopenPoolingDescriptor_t;
using cudnnPoolingMode_t           = miopenPoolingMode_t;

// Device scratch memory living as long as the wrapper that needs it. MIOpen
// takes an explicit workspace for the convolution algorithms and for pooling
// with a backward pass, and the convolution gradients need a staging buffer
// (see cudnn_wrappers.cu). Holding these in a shared_ptr with a deleter leaves
// the wrappers as copyable as they were, which a raw pointer would not.
using DeviceBuffer = std::shared_ptr<void>;
DeviceBuffer allocateDeviceBuffer(size_t bytes);
#endif

class CUDNNWrapper {
public:
  CUDNNWrapper();
  virtual ~CUDNNWrapper();

protected:
  void setCudnnTensor(cudnnTensorDescriptor_t& desc, Tensor x);

  void setCudnnTensor(cudnnTensorDescriptor_t& desc, const Shape& shape);

protected:
  cudnnHandle_t cudnnHandle_;
};

class ConvolutionWrapper : public CUDNNWrapper {
public:
  ConvolutionWrapper(const Shape& kernelShape,
                     const Shape& biasShape,
                     int hPad = 0,
                     int wPad = 0,
                     int hStride = 1,
                     int wStride = 1);

  void getOutputShape(const Shape& xShape, Shape& shape);

  virtual ~ConvolutionWrapper();

  void forward(Tensor x, Tensor Kernels, Tensor bias, Tensor y);

  void backward(Tensor x,
                Tensor xGrad,
                Tensor kernels,
                Tensor kernelGrad,
                Tensor biasGrad,
                Tensor yGrad);

protected:
  void setConvDescriptor(int hPad, int wPad, int hStride, int wStride);

  void setKernelDescriptor(const Shape& shape);

protected:
  cudnnConvolutionDescriptor_t convDesc_;
  cudnnFilterDescriptor_t kernelDesc_;
  cudnnTensorDescriptor_t biasDesc_;

#if defined(USE_HIP)
  // MIOpen picks a convolution algorithm by benchmarking, which is worth doing
  // once per input shape rather than once per call. The benchmark WRITES to the
  // tensor it is told is the output, so each of these is handed a destination
  // that is about to be overwritten anyway.
  void selectForwardAlgorithm(cudnnTensorDescriptor_t xDesc,
                              const void* x,
                              const void* kernels,
                              cudnnTensorDescriptor_t yDesc,
                              void* y,
                              const Shape& xShape);

  void selectBackwardAlgorithms(cudnnTensorDescriptor_t xDesc,
                                const void* x,
                                const void* kernels,
                                cudnnTensorDescriptor_t yDesc,
                                const void* yGrad,
                                void* scratch,
                                const Shape& xShape);

  miopenConvFwdAlgorithm_t fwdAlgo_{};
  miopenConvBwdDataAlgorithm_t bwdDataAlgo_{};
  miopenConvBwdWeightsAlgorithm_t bwdWeightsAlgo_{};
  Shape fwdAlgoShape_;
  Shape bwdAlgoShape_;
  DeviceBuffer workspace_;
  size_t workspaceSize_{0};
  DeviceBuffer gradStaging_;
  size_t gradStagingSize_{0};
#endif
};

class PoolingWrapper : public CUDNNWrapper {
public:
  PoolingWrapper(int height,
                 int width,
                 int padHeight,
                 int padWidth,
                 int strideHeight,
                 int strideWidth,
                 std::string mode);

  void getOutputShape(const Shape& xShape, Shape& shape);

  void forward(Tensor x, Tensor y);

  void backward(Tensor x, Tensor xGrad, Tensor y, Tensor yGrad);

  virtual ~PoolingWrapper();

protected:
  void setPoolingDescriptor(int height,
                            int width,
                            int padHeight,
                            int padWidth,
                            int strideHeight,
                            int strideWidth);

protected:
  cudnnPoolingDescriptor_t poolingDesc_;
  cudnnPoolingMode_t poolingMode_;

#if defined(USE_HIP)
  // MIOpen's pooling backward reads the indices its forward pass records, so
  // the workspace has to outlive the forward call.
  DeviceBuffer workspace_;
  size_t workspaceSize_{0};
  DeviceBuffer gradStaging_;
  size_t gradStagingSize_{0};
#endif
};
}  // namespace marian

#else

namespace marian {

class CUDNNWrapper {
public:
  CUDNNWrapper();
  virtual ~CUDNNWrapper();
};

class ConvolutionWrapper : public CUDNNWrapper {
public:
  ConvolutionWrapper(const Shape& kernelShape,
                     const Shape& biasShape,
                     int hPad = 1,
                     int wPad = 1,
                     int hStride = 1,
                     int wStride = 1);

  void getOutputShape(const Shape& xShape, Shape& shape);

  virtual ~ConvolutionWrapper();

  void forward(Tensor x, Tensor Kernels, Tensor bias, Tensor y);

  void backward(Tensor x,
                Tensor xGrad,
                Tensor kernels,
                Tensor kernelGrad,
                Tensor biasGrad,
                Tensor yGrad);
};

class PoolingWrapper : public CUDNNWrapper {
public:
  PoolingWrapper(int height,
                 int width,
                 int padHeight,
                 int padWidth,
                 int strideHeight,
                 int strideWidth,
                 std::string mode);

  void getOutputShape(const Shape& xShape, Shape& shape);

  void forward(Tensor x, Tensor y);

  void backward(Tensor x, Tensor xGrad, Tensor y, Tensor yGrad);

  virtual ~PoolingWrapper();
};
}  // namespace marian

#endif
