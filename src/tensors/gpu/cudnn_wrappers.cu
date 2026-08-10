#include "tensors/gpu/cudnn_wrappers.h"

#include <algorithm>

namespace marian {

#if defined(CUDNN) && defined(USE_HIP)

/******************************************************************************
 * MIOpen implementation.
 *
 * MIOpen offers the same operations as cuDNN but differs in three ways that
 * shape the code below rather than being spelled out at each call site.
 *
 * It takes an explicit workspace and an algorithm chosen by benchmarking, so
 * every convolution first runs a find step; that step is cached per input shape
 * because a graph replays the same convolution once per batch.
 *
 * Its convolutions and its bias addition only honour a unit alpha and a zero
 * beta, so the accumulating form the backward pass wants (beta = 1) is built
 * out of a convolution into a staging buffer followed by a tensor add.
 *
 * Its pooling backward pass reads the indices recorded by the forward pass, so
 * the pooling workspace is a member of the wrapper rather than a local.
 *
 * The cuDNN implementation follows unchanged below.
 *****************************************************************************/

#define MIOPEN_CALL(x)                     \
  do {                                     \
    miopenStatus_t _s = (x);               \
    if(_s != miopenStatusSuccess) {        \
      printf("Error (%s) at %s:%d\n",      \
             miopenGetErrorString(_s),     \
             __FILE__,                     \
             __LINE__);                    \
    }                                      \
  } while(0)

DeviceBuffer allocateDeviceBuffer(size_t bytes) {
  void* p = nullptr;
  if(bytes > 0) {
    hipError_t err = hipMalloc(&p, bytes);
    ABORT_IF(err != hipSuccess,
             "Cannot allocate {} bytes of scratch memory: {}",
             bytes,
             hipGetErrorString(err));
  }
  return DeviceBuffer(p, [](void* q) {
    if(q)
      (void)hipFree(q);
  });
}

namespace {
// Grows a buffer to at least the requested size, reusing it otherwise.
void ensureBuffer(DeviceBuffer& buffer, size_t& capacity, size_t bytes) {
  if(bytes > capacity) {
    buffer = allocateDeviceBuffer(bytes);
    capacity = bytes;
  }
}

// C = A + B for identically shaped tensors, in place on A.
void addInPlace(miopenHandle_t handle,
                miopenTensorDescriptor_t desc,
                void* accumulator,
                const void* addend) {
  const float one = 1.0f;
  const float zero = 0.0f;
  MIOPEN_CALL(miopenOpTensor(handle,
                             miopenTensorOpAdd,
                             &one,
                             desc,
                             accumulator,
                             &one,
                             desc,
                             addend,
                             &zero,
                             desc,
                             accumulator));
}
}  // namespace

CUDNNWrapper::CUDNNWrapper() {
  MIOPEN_CALL(miopenCreate(&cudnnHandle_));
}

CUDNNWrapper::~CUDNNWrapper() {
  MIOPEN_CALL(miopenDestroy(cudnnHandle_));
}

void CUDNNWrapper::setCudnnTensor(cudnnTensorDescriptor_t& desc, Tensor x) {
  setCudnnTensor(desc, x->shape());
}

void CUDNNWrapper::setCudnnTensor(cudnnTensorDescriptor_t& desc,
                                  const Shape& shape) {
  MIOPEN_CALL(miopenCreateTensorDescriptor(&desc));
  MIOPEN_CALL(miopenSet4dTensorDescriptor(
      desc, miopenFloat, shape[0], shape[1], shape[2], shape[3]));
}

/******************************************************************************
 * ConvolutionWrapper
 *****************************************************************************/

ConvolutionWrapper::ConvolutionWrapper(const Shape& kernelShape,
                                       const Shape& biasShape,
                                       int hPad,
                                       int wPad,
                                       int hStride,
                                       int wStride) {
  setKernelDescriptor(kernelShape);
  setConvDescriptor(hPad, wPad, hStride, wStride);
  setCudnnTensor(biasDesc_, biasShape);
}

void ConvolutionWrapper::getOutputShape(const Shape& xShape, Shape& shape) {
  cudnnTensorDescriptor_t xDesc;
  setCudnnTensor(xDesc, xShape);
  shape.resize(4);
  MIOPEN_CALL(miopenGetConvolutionForwardOutputDim(convDesc_,
                                                   xDesc,
                                                   kernelDesc_,
                                                   shape.data(),
                                                   shape.data() + 1,
                                                   shape.data() + 2,
                                                   shape.data() + 3));
  MIOPEN_CALL(miopenDestroyTensorDescriptor(xDesc));
}

void ConvolutionWrapper::selectForwardAlgorithm(cudnnTensorDescriptor_t xDesc,
                                                const void* x,
                                                const void* kernels,
                                                cudnnTensorDescriptor_t yDesc,
                                                void* y,
                                                const Shape& xShape) {
  if(fwdAlgoShape_ == xShape)
    return;

  size_t needed = 0;
  MIOPEN_CALL(miopenConvolutionForwardGetWorkSpaceSize(
      cudnnHandle_, kernelDesc_, xDesc, convDesc_, yDesc, &needed));
  ensureBuffer(workspace_, workspaceSize_, needed);

  int found = 0;
  miopenConvAlgoPerf_t perf;
  MIOPEN_CALL(miopenFindConvolutionForwardAlgorithm(cudnnHandle_,
                                                    xDesc,
                                                    x,
                                                    kernelDesc_,
                                                    kernels,
                                                    convDesc_,
                                                    yDesc,
                                                    y,
                                                    1,
                                                    &found,
                                                    &perf,
                                                    workspace_.get(),
                                                    workspaceSize_,
                                                    false));
  ABORT_IF(found < 1, "MIOpen found no forward convolution algorithm");
  fwdAlgo_ = perf.fwd_algo;
  fwdAlgoShape_ = xShape;
}

void ConvolutionWrapper::selectBackwardAlgorithms(
    cudnnTensorDescriptor_t xDesc,
    const void* x,
    const void* kernels,
    cudnnTensorDescriptor_t yDesc,
    const void* yGrad,
    void* scratch,
    const Shape& xShape) {
  if(bwdAlgoShape_ == xShape)
    return;

  size_t bwdDataSize = 0, bwdWeightsSize = 0;
  MIOPEN_CALL(miopenConvolutionBackwardDataGetWorkSpaceSize(
      cudnnHandle_, yDesc, kernelDesc_, convDesc_, xDesc, &bwdDataSize));
  MIOPEN_CALL(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
      cudnnHandle_, yDesc, xDesc, convDesc_, kernelDesc_, &bwdWeightsSize));
  ensureBuffer(workspace_, workspaceSize_, std::max(bwdDataSize, bwdWeightsSize));

  int found = 0;
  miopenConvAlgoPerf_t perf;
  MIOPEN_CALL(miopenFindConvolutionBackwardDataAlgorithm(cudnnHandle_,
                                                         yDesc,
                                                         yGrad,
                                                         kernelDesc_,
                                                         kernels,
                                                         convDesc_,
                                                         xDesc,
                                                         scratch,
                                                         1,
                                                         &found,
                                                         &perf,
                                                         workspace_.get(),
                                                         workspaceSize_,
                                                         false));
  ABORT_IF(found < 1, "MIOpen found no backward data convolution algorithm");
  bwdDataAlgo_ = perf.bwd_data_algo;

  MIOPEN_CALL(miopenFindConvolutionBackwardWeightsAlgorithm(cudnnHandle_,
                                                            yDesc,
                                                            yGrad,
                                                            xDesc,
                                                            x,
                                                            convDesc_,
                                                            kernelDesc_,
                                                            scratch,
                                                            1,
                                                            &found,
                                                            &perf,
                                                            workspace_.get(),
                                                            workspaceSize_,
                                                            false));
  ABORT_IF(found < 1, "MIOpen found no backward weights convolution algorithm");
  bwdWeightsAlgo_ = perf.bwd_weights_algo;
  bwdAlgoShape_ = xShape;
}

void ConvolutionWrapper::forward(Tensor x,
                                 Tensor kernels,
                                 Tensor bias,
                                 Tensor y) {
  cudaSetDevice(x->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, x);
  setCudnnTensor(yDesc, y);

  selectForwardAlgorithm(
      xDesc, x->data(), kernels->data(), yDesc, y->data(), x->shape());

  const float alpha = 1.0f;
  const float beta = 0.0f;

  MIOPEN_CALL(miopenConvolutionForward(cudnnHandle_,
                                       &alpha,
                                       xDesc,
                                       x->data(),
                                       kernelDesc_,
                                       kernels->data(),
                                       convDesc_,
                                       fwdAlgo_,
                                       &beta,
                                       yDesc,
                                       y->data(),
                                       workspace_.get(),
                                       workspaceSize_));
  // miopenConvolutionForwardBias overwrites rather than accumulates, so add the
  // broadcast bias with a tensor op instead.
  MIOPEN_CALL(miopenOpTensor(cudnnHandle_,
                             miopenTensorOpAdd,
                             &alpha,
                             yDesc,
                             y->data(),
                             &alpha,
                             biasDesc_,
                             bias->data(),
                             &beta,
                             yDesc,
                             y->data()));

  MIOPEN_CALL(miopenDestroyTensorDescriptor(xDesc));
  MIOPEN_CALL(miopenDestroyTensorDescriptor(yDesc));
}

void ConvolutionWrapper::backward(Tensor x,
                                  Tensor xGrad,
                                  Tensor kernels,
                                  Tensor kernelGrad,
                                  Tensor biasGrad,
                                  Tensor yGrad) {
  cudaSetDevice(xGrad->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, xGrad);
  setCudnnTensor(yDesc, yGrad);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Each gradient accumulates onto what is already there, which MIOpen cannot
  // express directly, so every result lands in the staging buffer first and is
  // added on. One buffer serves all three because they run in sequence on the
  // same stream, and it doubles as the scratch destination for the benchmark.
  size_t staging = std::max(xGrad->shape().elements(),
                            std::max(kernelGrad->shape().elements(),
                                     biasGrad->shape().elements()))
                   * sizeof(float);
  ensureBuffer(gradStaging_, gradStagingSize_, staging);
  void* scratch = gradStaging_.get();

  selectBackwardAlgorithms(xDesc,
                           x->data(),
                           kernels->data(),
                           yDesc,
                           yGrad->data(),
                           scratch,
                           xGrad->shape());

  MIOPEN_CALL(miopenConvolutionBackwardData(cudnnHandle_,
                                            &alpha,
                                            yDesc,
                                            yGrad->data(),
                                            kernelDesc_,
                                            kernels->data(),
                                            convDesc_,
                                            bwdDataAlgo_,
                                            &beta,
                                            xDesc,
                                            scratch,
                                            workspace_.get(),
                                            workspaceSize_));
  addInPlace(cudnnHandle_, xDesc, xGrad->data(), scratch);

  MIOPEN_CALL(miopenConvolutionBackwardWeights(cudnnHandle_,
                                               &alpha,
                                               yDesc,
                                               yGrad->data(),
                                               xDesc,
                                               x->data(),
                                               convDesc_,
                                               bwdWeightsAlgo_,
                                               &beta,
                                               kernelDesc_,
                                               scratch,
                                               workspace_.get(),
                                               workspaceSize_));
  addInPlace(cudnnHandle_, kernelDesc_, kernelGrad->data(), scratch);

  MIOPEN_CALL(miopenConvolutionBackwardBias(
      cudnnHandle_, &alpha, yDesc, yGrad->data(), &beta, biasDesc_, scratch));
  addInPlace(cudnnHandle_, biasDesc_, biasGrad->data(), scratch);

  MIOPEN_CALL(miopenDestroyTensorDescriptor(xDesc));
  MIOPEN_CALL(miopenDestroyTensorDescriptor(yDesc));
}

ConvolutionWrapper::~ConvolutionWrapper() {
  MIOPEN_CALL(miopenDestroyConvolutionDescriptor(convDesc_));
  MIOPEN_CALL(miopenDestroyTensorDescriptor(kernelDesc_));
  MIOPEN_CALL(miopenDestroyTensorDescriptor(biasDesc_));
}

void ConvolutionWrapper::setConvDescriptor(int hPad,
                                           int wPad,
                                           int hStride,
                                           int wStride) {
  MIOPEN_CALL(miopenCreateConvolutionDescriptor(&convDesc_));
  // miopenConvolution is cross-correlation, matching what the cuDNN path asks
  // for; MIOpen has no separate true-convolution mode to choose between.
  MIOPEN_CALL(miopenInitConvolutionDescriptor(convDesc_,
                                              miopenConvolution,
                                              hPad,
                                              wPad,
                                              hStride,
                                              wStride,
                                              1,
                                              1));
}

void ConvolutionWrapper::setKernelDescriptor(const Shape& shape) {
  ABORT_IF(shape.size() != 4,
           "Convolution requires tensors 4D. Provided {}",
           shape.toString());
  MIOPEN_CALL(miopenCreateTensorDescriptor(&kernelDesc_));

  int layerIn = shape[0];
  int layerOut = shape[1];
  int kernelH = shape[2];
  int kernelW = shape[3];

  MIOPEN_CALL(miopenSet4dTensorDescriptor(
      kernelDesc_, miopenFloat, layerOut, layerIn, kernelH, kernelW));
}

/******************************************************************************
 * PoolingWrapper
 *****************************************************************************/

PoolingWrapper::PoolingWrapper(int height,
                               int width,
                               int padHeight,
                               int padWidth,
                               int strideHeight,
                               int strideWidth,
                               std::string mode) {
  if(mode == "max") {
    poolingMode_ = miopenPoolingMax;
  } else if(mode == "avg") {
    poolingMode_ = miopenPoolingAverageInclusive;
  } else {
    ABORT("Unknown pooling mode.");
  }

  setPoolingDescriptor(
      height, width, padHeight, padWidth, strideHeight, strideWidth);
}

void PoolingWrapper::getOutputShape(const Shape& xShape, Shape& shape) {
  cudnnTensorDescriptor_t xDesc;
  setCudnnTensor(xDesc, xShape);
  shape.resize(4);
  MIOPEN_CALL(miopenGetPoolingForwardOutputDim(poolingDesc_,
                                               xDesc,
                                               shape.data(),
                                               shape.data() + 1,
                                               shape.data() + 2,
                                               shape.data() + 3));
  MIOPEN_CALL(miopenDestroyTensorDescriptor(xDesc));
}

void PoolingWrapper::forward(Tensor x, Tensor y) {
  cudaSetDevice(x->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, x);
  setCudnnTensor(yDesc, y);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  size_t needed = 0;
  MIOPEN_CALL(miopenPoolingGetWorkSpaceSizeV2(poolingDesc_, yDesc, &needed));
  ensureBuffer(workspace_, workspaceSize_, needed);

  MIOPEN_CALL(miopenPoolingForward(cudnnHandle_,
                                   poolingDesc_,
                                   &alpha,
                                   xDesc,
                                   x->data(),
                                   &beta,
                                   yDesc,
                                   y->data(),
                                   /*do_backward=*/true,
                                   workspace_.get(),
                                   workspaceSize_));

  MIOPEN_CALL(miopenDestroyTensorDescriptor(xDesc));
  MIOPEN_CALL(miopenDestroyTensorDescriptor(yDesc));
}

void PoolingWrapper::backward(Tensor x, Tensor xGrad, Tensor y, Tensor yGrad) {
  cudaSetDevice(x->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, x);
  setCudnnTensor(yDesc, y);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // As with the convolution gradients, the accumulating form is a pooling
  // backward into staging plus a tensor add. The buffer is a cached member for
  // consistency with that path, but the wrapper lives on its node and the
  // graph drops it for the next batch, so per batch nothing is saved: the
  // allocation still happens once per pooling node, and the free moves from
  // the end of backward to node teardown.
  ensureBuffer(gradStaging_,
               gradStagingSize_,
               xGrad->shape().elements() * sizeof(float));
  void* staging = gradStaging_.get();

  MIOPEN_CALL(miopenPoolingBackward(cudnnHandle_,
                                    poolingDesc_,
                                    &alpha,
                                    yDesc,
                                    y->data(),
                                    yDesc,
                                    yGrad->data(),
                                    xDesc,
                                    x->data(),
                                    &beta,
                                    xDesc,
                                    staging,
                                    workspace_.get()));
  addInPlace(cudnnHandle_, xDesc, xGrad->data(), staging);

  MIOPEN_CALL(miopenDestroyTensorDescriptor(xDesc));
  MIOPEN_CALL(miopenDestroyTensorDescriptor(yDesc));
}

void PoolingWrapper::setPoolingDescriptor(int height,
                                          int width,
                                          int padHeight,
                                          int padWidth,
                                          int strideHeight,
                                          int strideWidth) {
  MIOPEN_CALL(miopenCreatePoolingDescriptor(&poolingDesc_));
  MIOPEN_CALL(miopenSet2dPoolingDescriptor(poolingDesc_,
                                           poolingMode_,
                                           height,
                                           width,
                                           padHeight,
                                           padWidth,
                                           strideHeight,
                                           strideWidth));
}

PoolingWrapper::~PoolingWrapper() {
  MIOPEN_CALL(miopenDestroyPoolingDescriptor(poolingDesc_));
}

#elif defined(CUDNN)

#include <cudnn.h>

#define CUDNN_CALL(x)                 \
  do {                                \
    if((x) != CUDNN_STATUS_SUCCESS) { \
      printf("Error (%s) at %s:%d\n", \
             cudnnGetErrorString(x),  \
             __FILE__,                \
             __LINE__);               \
    }                                 \
  } while(0)

CUDNNWrapper::CUDNNWrapper() {
  CUDNN_CALL(cudnnCreate(&cudnnHandle_));
}

CUDNNWrapper::~CUDNNWrapper() {
  // std::cerr << "destroy wrapper" << std::endl;
  CUDNN_CALL(cudnnDestroy(cudnnHandle_));
}

void CUDNNWrapper::setCudnnTensor(cudnnTensorDescriptor_t& desc, Tensor x) {
  setCudnnTensor(desc, x->shape());
}

void CUDNNWrapper::setCudnnTensor(cudnnTensorDescriptor_t& desc,
                                  const Shape& shape) {
  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        shape[0],
                                        shape[1],
                                        shape[2],
                                        shape[3]));
}

/******************************************************************************
 * ConvolutionWrapper
 *****************************************************************************/

ConvolutionWrapper::ConvolutionWrapper(const Shape& kernelShape,
                                       const Shape& biasShape,
                                       int hPad,
                                       int wPad,
                                       int hStride,
                                       int wStride) {
  setKernelDescriptor(kernelShape);
  setConvDescriptor(hPad, wPad, hStride, wStride);
  setCudnnTensor(biasDesc_, biasShape);
}

void ConvolutionWrapper::getOutputShape(const Shape& xShape, Shape& shape) {
  cudnnTensorDescriptor_t xDesc;
  setCudnnTensor(xDesc, xShape);
  shape.resize(4);
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(convDesc_,
                                                   xDesc,
                                                   kernelDesc_,
                                                   shape.data(),
                                                   shape.data() + 1,
                                                   shape.data() + 2,
                                                   shape.data() + 3));
  cudnnDestroyTensorDescriptor(xDesc);
}

void ConvolutionWrapper::forward(Tensor x,
                                 Tensor kernels,
                                 Tensor bias,
                                 Tensor y) {
  cudaSetDevice(x->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, x);
  setCudnnTensor(yDesc, y);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CUDNN_CALL(cudnnConvolutionForward(cudnnHandle_,
                                     &alpha,
                                     xDesc,
                                     x->data(),
                                     kernelDesc_,
                                     kernels->data(),
                                     convDesc_,
                                     CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                     nullptr,
                                     0,
                                     &beta,
                                     yDesc,
                                     y->data()));
  CUDNN_CALL(cudnnAddTensor(
      cudnnHandle_, &alpha, biasDesc_, bias->data(), &alpha, yDesc, y->data()));
  cudnnDestroyTensorDescriptor(xDesc);
  cudnnDestroyTensorDescriptor(yDesc);
}

void ConvolutionWrapper::backward(Tensor x,
                                  Tensor xGrad,
                                  Tensor kernels,
                                  Tensor kernelGrad,
                                  Tensor biasGrad,
                                  Tensor yGrad) {
  cudaSetDevice(xGrad->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, xGrad);
  setCudnnTensor(yDesc, yGrad);

  const float alpha = 1.0f;
  const float beta = 1.0f;

  CUDNN_CALL(cudnnConvolutionBackwardData(cudnnHandle_,
                                          &alpha,
                                          kernelDesc_,
                                          kernels->data(),
                                          yDesc,
                                          yGrad->data(),
                                          convDesc_,
                                          CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                          nullptr,
                                          0,
                                          &beta,
                                          xDesc,
                                          xGrad->data()));

  CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnnHandle_,
                                            &alpha,
                                            xDesc,
                                            x->data(),
                                            yDesc,
                                            yGrad->data(),
                                            convDesc_,
                                            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                            nullptr,
                                            0,
                                            &beta,
                                            kernelDesc_,
                                            kernelGrad->data()));

  CUDNN_CALL(cudnnConvolutionBackwardBias(cudnnHandle_,
                                          &alpha,
                                          yDesc,
                                          yGrad->data(),
                                          &beta,
                                          biasDesc_,
                                          biasGrad->data()));

  cudnnDestroyTensorDescriptor(xDesc);
  cudnnDestroyTensorDescriptor(yDesc);
}

ConvolutionWrapper::~ConvolutionWrapper() {
  // std::cerr << "destroy conv-wrapper" << std::endl;
  cudnnDestroyConvolutionDescriptor(convDesc_);
  cudnnDestroyFilterDescriptor(kernelDesc_);
  cudnnDestroyTensorDescriptor(biasDesc_);
}

void ConvolutionWrapper::setConvDescriptor(int hPad,
                                           int wPad,
                                           int hStride,
                                           int wStride) {
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc_));

#if CUDNN_MAJOR > 5
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(convDesc_,
                                             hPad,
                                             wPad,
                                             hStride,
                                             wStride,
                                             1,
                                             1,  // upscales
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));
#else
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(convDesc_,
                                             hPad,
                                             wPad,
                                             hStride,
                                             wStride,
                                             1,
                                             1,  // upscales
                                             CUDNN_CROSS_CORRELATION));
#endif
}

void ConvolutionWrapper::setKernelDescriptor(const Shape& shape) {
  ABORT_IF(shape.size() != 4,
           "CUDN requires tensors 4D. Provided {}",
           shape.toString());
  CUDNN_CALL(cudnnCreateFilterDescriptor(&kernelDesc_));

  int layerIn = shape[0];
  int layerOut = shape[1];
  int kernelH = shape[2];
  int kernelW = shape[3];

  CUDNN_CALL(cudnnSetFilter4dDescriptor(kernelDesc_,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        layerOut,
                                        layerIn,
                                        kernelH,
                                        kernelW));
}

/******************************************************************************
 * PoolingWrapper
 *****************************************************************************/

PoolingWrapper::PoolingWrapper(int height,
                               int width,
                               int padHeight,
                               int padWidth,
                               int strideHeight,
                               int strideWidth,
                               std::string mode) {
  if(mode == "max") {
    poolingMode_ = CUDNN_POOLING_MAX;
  } else if(mode == "avg") {
    poolingMode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else {
    ABORT("Unknown pooling mode.");
  }

  setPoolingDescriptor(
      height, width, padHeight, padWidth, strideHeight, strideWidth);
}

void PoolingWrapper::getOutputShape(const Shape& xShape, Shape& shape) {
  cudnnTensorDescriptor_t xDesc;
  setCudnnTensor(xDesc, xShape);
  shape.resize(4);
  CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(poolingDesc_,
                                               xDesc,
                                               shape.data(),
                                               shape.data() + 1,
                                               shape.data() + 2,
                                               shape.data() + 3));
  cudnnDestroyTensorDescriptor(xDesc);
}

void PoolingWrapper::forward(Tensor x, Tensor y) {
  cudaSetDevice(x->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, x);
  setCudnnTensor(yDesc, y);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CUDNN_CALL(cudnnPoolingForward(cudnnHandle_,
                                 poolingDesc_,
                                 &alpha,
                                 xDesc,
                                 x->data(),
                                 &beta,
                                 yDesc,
                                 y->data()));
  cudnnDestroyTensorDescriptor(xDesc);
  cudnnDestroyTensorDescriptor(yDesc);
}

void PoolingWrapper::backward(Tensor x, Tensor xGrad, Tensor y, Tensor yGrad) {
  cudaSetDevice(x->getDeviceId().no);

  cudnnTensorDescriptor_t xDesc, yDesc;
  setCudnnTensor(xDesc, x);
  setCudnnTensor(yDesc, y);

  const float alpha = 1.0f;
  const float beta = 1.0f;

  CUDNN_CALL(cudnnPoolingBackward(cudnnHandle_,
                                  poolingDesc_,
                                  &alpha,
                                  yDesc,
                                  y->data(),
                                  yDesc,
                                  yGrad->data(),
                                  xDesc,
                                  x->data(),
                                  &beta,
                                  xDesc,
                                  xGrad->data()));
  cudnnDestroyTensorDescriptor(xDesc);
  cudnnDestroyTensorDescriptor(yDesc);
}

void PoolingWrapper::setPoolingDescriptor(int height,
                                          int width,
                                          int padHeight,
                                          int padWidth,
                                          int strideHeight,
                                          int strideWidth) {
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc_));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(poolingDesc_,
                                         poolingMode_,
                                         CUDNN_NOT_PROPAGATE_NAN,
                                         height,
                                         width,
                                         padHeight,
                                         padWidth,
                                         strideHeight,
                                         strideWidth));
}

PoolingWrapper::~PoolingWrapper() {
  // std::cerr << "destroy pool-wrapper" << std::endl;
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc_));
}

#else

CUDNNWrapper::CUDNNWrapper() {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

CUDNNWrapper::~CUDNNWrapper() {}

ConvolutionWrapper::ConvolutionWrapper(const Shape&,
                                       const Shape&,
                                       int,
                                       int,
                                       int,
                                       int) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void ConvolutionWrapper::getOutputShape(const Shape&, Shape&) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void ConvolutionWrapper::forward(Tensor, Tensor, Tensor, Tensor) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void ConvolutionWrapper::backward(Tensor,
                                  Tensor,
                                  Tensor,
                                  Tensor,
                                  Tensor,
                                  Tensor) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

ConvolutionWrapper::~ConvolutionWrapper() {}

PoolingWrapper::PoolingWrapper(int, int, int, int, int, int, std::string) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void PoolingWrapper::getOutputShape(const Shape&, Shape&) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void PoolingWrapper::forward(Tensor x, Tensor y) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

void PoolingWrapper::backward(Tensor, Tensor, Tensor, Tensor) {
  ABORT(
      "To use convolution and pooling, recompile with CUDNN (cmake flag "
      "-DUSE_CUDNN=on)");
}

PoolingWrapper::~PoolingWrapper() {}

#endif
}  // namespace marian
