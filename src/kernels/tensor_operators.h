#pragma once

#include <cublas_v2.h>

#include <thrust/pair.h>

#include "tensors/tensor.h"

#include "tensors/allocator.h"
#include "tensors/device_gpu.h"

#include "gpu/shape.h"
#include "gpu/tmp.h"
#include "gpu/tensor.h"
#include "functional/functional.h"

namespace marian {

bool IsNan(Tensor in);

const int MAX_THREADS = 512;
const int MAX_BLOCKS = 65535;

cublasHandle_t create_handle(size_t);

namespace gpu {

  template <class Lambda>
  __DI__ void foreach(size_t length, Lambda& lambda) {
    for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
      int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
      if(index < length)
        lambda(index);
    }
  }

  template <size_t K, typename T>
  __DI__ gpu::Array<gpu::Tensor<T>, K> rows(gpu::Array<gpu::Tensor<T>, K>& args,
                                            size_t i) {
    gpu::Array<gpu::Tensor<T>, K> rowArgs;
    for(int j = 0; j < K; ++j)
      rowArgs[j] = args[j].row(i);
    return rowArgs;
  }

  template <size_t K, typename T>
  __DI__ gpu::Array<gpu::Tensor<T>, K> rows(gpu::Array<gpu::Tensor<T>, K>& args,
                                            gpu::Array<int, K>& indices) {
    gpu::Array<gpu::Tensor<T>, K> rowArgs;
    for(int j = 0; j < K; ++j)
      rowArgs[j] = args[j].row(indices[j]);
    return rowArgs;
  }

  template <bool broadcast = false, class Functor, typename T>
  __DI__ void transform_row(gpu::Tensor<T> rowOut,
                            gpu::Tensor<T> rowArg,
                            Functor functor) {
    Array<Tensor<T>, 1> rowArgs = { rowArg };
    transform_row<broadcast>(rowOut, rowArgs, functor);
  }


  template <bool broadcast = false, size_t K, class Functor, typename T>
  __DI__ void transform_row(gpu::Tensor<T> rowOut,
                            gpu::Array<gpu::Tensor<T>, K> rowArgs,
                            Functor functor) {
    int cols = rowOut.shape().elements();
    for(int tid = 0; tid < cols; tid += blockDim.x) {
      int index = tid + threadIdx.x;
      if(index < cols) {
        if(broadcast) {
          gpu::Array<int, gpu::Shape::size()> dims;
          rowOut.shape().dims(index, dims);

          gpu::Array<int, K> indices;
          for(int i = 0; i < K; ++i)
            indices[i] = rowArgs[i].shape().bindex(dims);
          rowOut[index] = gpu::apply(functor, rowArgs, indices);
        }
        else {
          rowOut[index] = gpu::apply(functor, rowArgs, index);
        }
      }
    }
  }

  template <size_t K, class Functor, typename T, bool broadcast=false>
  __DI__ void transform_rows(gpu::Tensor<T>& out,
                             Functor& functor,
                             gpu::Array<gpu::Tensor<T>, K>& args) {

    int rows = out.shape().elements() / out.shape().back();

    for(int bid = 0; bid < rows; bid += gridDim.x) {
      int j = bid + blockIdx.x;
      if(j < rows) {

        //*********************************************************************/

        auto rowArgs = gpu::rows(args, j);
        transform_row<broadcast>(out.row(j),
                                 functor,
                                 rowArgs);

        //*********************************************************************/

      }
    }
  }

  template <class Lambda>
  __device__ inline void foreach_row(size_t rows, Lambda& lambda) {

    for(int bid = 0; bid < rows; bid += gridDim.x) {
      int row_index = bid + blockIdx.x;
      if(row_index < rows)
        lambda(row_index);
    }
  }

  template <bool broadcast=false,
            typename T,
            class Functor = decltype(functional::_1),
            class Accumulator = decltype(functional::_1 += functional::_2),
            class AccumulatorZero = decltype(functional::_0c)>
  __device__ inline T reduce_row(int cols,
                                 gpu::Tensor<T>& rowArg,
                                 Functor functor = functional::_1,
                                 Accumulator acc = functional::_1 += functional::_2,
                                 AccumulatorZero accZero = functional::_0c) {
    gpu::Array<Tensor<T>, 1> rowArgs = { rowArg };
    return reduce_row<broadcast>(cols, rowArgs, functor, acc, accZero);
  }

  template <bool broadcast=false,
            size_t K,
            typename T,
            class Functor = decltype(functional::_1),
            class Accumulator = decltype(functional::_1 += functional::_2),
            class AccumulatorZero = decltype(functional::_0c)>
  __device__ inline T reduce_row(int cols,
                                 gpu::Array<gpu::Tensor<T>, K>& rowArgs,
                                 Functor functor = functional::_1,
                                 Accumulator acc = functional::_1 += functional::_2,
                                 AccumulatorZero accZero = functional::_0c) {

    __syncthreads();
    extern __shared__ T _share[];
    T* _reduce = _share + blockDim.x;

    _reduce[threadIdx.x] = accZero(gpu::apply(functor, rowArgs, 0));
    for(int tid = 0; tid < cols; tid += blockDim.x) {
      int index = tid + threadIdx.x;
      if(index < cols) {

        //*********************************************************************/
        if(broadcast) {
          gpu::Array<int, K> indices;
          for(int i = 0; i < K; ++i)
            indices[i] = cols == rowArgs[i].shape().back() ? index : 0;
          acc(_reduce[threadIdx.x], gpu::apply(functor, rowArgs, indices));
        }
        else {
          acc(_reduce[threadIdx.x], gpu::apply(functor, rowArgs, index));
        }
        //*********************************************************************/

      }
    }

    __syncthreads();
    int len = blockDim.x;
    while(len != 1) {
      __syncthreads();
      int skip = (len + 1) >> 1;
      if(threadIdx.x < (len >> 1)) {

        //*********************************************************************/
        acc(_reduce[threadIdx.x], _reduce[threadIdx.x + skip]);
        //*********************************************************************/

      }
      len = (len + 1) >> 1;
    }
    __syncthreads();
    return _reduce[0];
  }

  namespace kernels {
    using namespace functional;

    template <bool broadcast = false,
              size_t K = 1,
              class PreReduceFunctor,
              class Accumulator,
              class AccumulatorZero,
              class AssignFunctor,
              typename T = float>
    __global__ void gReduceRows(gpu::Tensor<T>& out,
                                gpu::Array<gpu::Tensor<T>, K>& args,
                                gpu::Shape& full,
                                PreReduceFunctor& preFunc = _1,
                                Accumulator& accFunc = _1 += _2,
                                AccumulatorZero& accZero = _0c,
                                AssignFunctor& assignFunc = _1 = _2) {

      auto lambda = [&](size_t row_index) {

        int cols = full.back();
        gpu::Array<gpu::Tensor<T>, K> rowArgs;
        if(broadcast) {
          gpu::Array<int, K> indices;
          for(int i = 0; i < K; ++i) {
            int argRows = args[i].shape().elements() / args[i].shape().back();
            indices[i] = argRows % row_index;
          }
          rowArgs = gpu::rows(args, indices);
        }
        else {
          rowArgs = gpu::rows(args, row_index);
        }

        T result = gpu::reduce_row<broadcast>(cols,
                                              rowArgs,
                                              preFunc,
                                              accFunc,
                                              accZero);

        assignFunc(out[row_index], result);
      };

      size_t rows = full.elements() / full.back();
      gpu::foreach_row(rows, lambda);
    }

    template <bool broadcast = false,
              size_t K, class Functor, typename T>
    __global__ void gForeach(gpu::Tensor<T> out,
                             gpu::Array<gpu::Tensor<T>, K> args,
                             Functor functor) {

      auto lambda = [&](size_t index) {
        if(broadcast) {
          gpu::Array<int, gpu::Shape::size()> dims;
          out.shape().dims(index, dims);

          gpu::Array<int, K> indices;
          for(int i = 0; i < K; ++i)
            indices[i] = args[i].shape().bindex(dims);
          out[index] = gpu::apply(functor, args, indices);
        }
        else {
          out[index] = gpu::apply(functor, args, index);
        }
      };

      gpu::foreach(out.size(), lambda);
    }

  }
}

template <class Functor, class ...Tensors>
void Element(Functor gFunctor, Tensor out, Tensors ...tensors) {
  cudaSetDevice(out->getDevice());

  constexpr size_t K = sizeof...(tensors) + 1;

  gpu::Tensor<float> gOut = out;
  gpu::Array<gpu::Tensor<float>, K> gArgs = {out, tensors...};

  int length = gOut.shape().elements();
  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  bool broadcast = false;
  for(int i = 1; i < K; ++i)
    broadcast = broadcast || gOut.shape() != gArgs[i].shape();

  if(broadcast)
    gpu::kernels::gForeach<true><<<blocks, threads>>>(gOut, gArgs, gFunctor);
  else
    gpu::kernels::gForeach<false><<<blocks, threads>>>(gOut, gArgs, gFunctor);
}

void TransposeND(Tensor out, Tensor in, const std::vector<int>& vAxis);

void Select(Ptr<Allocator<DeviceGPU>> allocator,
            Tensor out,
            Tensor in,
            int axis,
            const std::vector<size_t>&);

void Insert(Ptr<Allocator<DeviceGPU>> allocator,
            Tensor out,
            Tensor in,
            int axis,
            const std::vector<size_t>&);

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax);

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax);

template <size_t K, class Functor>
__global__ void gAddGeneric(Functor functor,
                            const gpu::Shape full,
                            gpu::Tensor<float> out,
                            gpu::Array<gpu::Tensor<float>, K> ins,
                            float scale = 1.0) {

  int outLength = out.shape().elements();
  bool same = outLength == full.elements();
  for(int i = 0; i < K; ++i)
    same = same && outLength == ins[i].shape().elements();

  constexpr size_t N = gpu::Shape::size();
  gpu::Array<int, N> len;
  for(int i = 0; i < N; ++i)
    len[i] = full[i] / out.shape()[i];

  gpu::Array<int, N> dims;
  for(int bid = 0; bid < outLength; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < outLength) {

      if(same) {
        out[index] += gpu::apply(functor, ins, index) * scale;
      } else {
        out.shape().dims(index, dims);
        out[index] += gpu::loops(functor, ins, len, dims) * scale;
      }

    }
  }
}

template <size_t K, class Functor>
__global__ void gAddEqual(Functor functor,
                          gpu::Tensor<float> out,
                          gpu::Array<gpu::Tensor<float>, K> ins,
                          float scale,
                          bool broadcast) {
  int length = out.shape().elements();
  gpu::Array<int, gpu::Shape::size()> dims;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      gpu::Array<int, K> indices;
      indices.fill(index);

      if(broadcast) {
        out.shape().dims(index, dims);
        for(size_t i = 0; i < K; ++i)
          indices[i] = ins[i].shape().bindex(dims);
      }

      out[index] += gpu::apply(functor, ins, indices) * scale;
    }
  }
}

template <class Functor, class ...Tensors>
void Add(Functor functor,
         float scale,
         Tensor out,
         Tensors... tensors) {
  cudaSetDevice(out->getDevice());

  Shape full = Shape::broadcast({out, tensors...});

  int length = out->shape().elements();

  constexpr size_t K = sizeof...(Tensors);

  gpu::Tensor<float> gOut = out;
  gpu::Array<gpu::Tensor<float>, K> gIns = {tensors ...};

  if(full.back() != 1 && out->shape().back() == 1) {
    size_t m = full.elements() / length;
    size_t k = full.back();

    int blocks = std::min(MAX_BLOCKS, (int)m);
    int threads = std::min(MAX_THREADS, (int)k);
    int shared = sizeof(float) * threads * 2;

    gpu::Shape gFull = full;
    bool broadcast = false;
    for(int i = 0; i < K; ++i)
      broadcast = broadcast || gFull != gIns[i].shape();

    namespace f = functional;
    auto addScale = f::_1 += f::_2 * scale;
    auto acc = f::_1 += f::_2;
    auto accZero = f::_1c;

    if(broadcast)
      gpu::kernels::gReduceRows<true><<<blocks, threads, shared>>>(
        gOut, gIns, gFull, functor, acc, accZero, addScale);
    else
      gpu::kernels::gReduceRows<false><<<blocks, threads, shared>>>(
        gOut, gIns, gFull, functor, acc, accZero, addScale);

  } else if(out->shape() == full) {
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    bool broadcast = false;
    for(int i = 0; i < K; ++i)
      broadcast = broadcast || gOut.shape() != gIns[i].shape();

    gAddEqual<<<blocks, threads>>>(functor, gOut, gIns, scale, broadcast);
  } else {
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    gAddGeneric<<<blocks, threads>>>(functor, full, gOut, gIns, scale);
  }
}

template <class Functor, class ...Tensors>
void Add(Functor functor,
         Tensor out,
         Tensors... tensors) {
  Add(functor, 1, out, tensors...);
}

template <class Functor, class ...Tensors>
void Reduce(Functor functor,
            float scale,
            Tensor out,
            Tensors... tensors) {
  out->set(0);
  Add(functor, scale, out, tensors...);
}

template <class Functor, class ...Tensors>
void Reduce(Functor functor,
            Tensor out,
            Tensors... tensors) {
  out->set(0);
  Add(functor, out, tensors...);
}

float L2Norm(Tensor in);

void Softmax(Tensor out, Tensor in, Tensor mask = nullptr);
void LogSoftmax(Tensor out, Tensor in);

void SoftmaxGrad(Tensor grad, Tensor adj, Tensor val);
void LogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CudnnSoftmax(Tensor out, Tensor in);
void CudnnSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CudnnLogSoftmax(Tensor out, Tensor in);
void CudnnLogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CrossEntropyPick(Tensor out, Tensor in, Tensor pick);
void CrossEntropyPickBackward(Tensor out, Tensor adj, Tensor a, Tensor pick);

void Argmax(Tensor Out, const Tensor In);

void Prod(cublasHandle_t handle,
          Tensor C,
          const Tensor A,
          const Tensor B,
          bool transA,
          bool transB,
          float beta = 0,
          float scalar = 1);

void ProdBatched(cublasHandle_t handle,
                 Tensor C,
                 const Tensor A,
                 const Tensor B,
                 bool transA,
                 bool transB,
                 float beta = 0,
                 float scalar = 1);

void CopyRowsByIndex(Tensor out,
                     const Tensor in,
                     thrust::pair<size_t, size_t>* ipair,
                     size_t length);

void CopyRows(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void PasteRows(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void CopyCols(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void PasteCols(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void LSTMCellForward(Tensor out, std::vector<Tensor> inputs);
void LSTMOutputForward(Tensor out, std::vector<Tensor> inputs);
void LSTMCellBackward(std::vector<Tensor> outputs,
                      std::vector<Tensor> inputs,
                      Tensor adj);
void LSTMOutputBackward(std::vector<Tensor> outputs,
                        std::vector<Tensor> inputs,
                        Tensor adj);

void GRUFastForward(Tensor out, std::vector<Tensor> inputs, bool final = false);

void GRUFastBackward(std::vector<Tensor> outputs,
                     std::vector<Tensor> inputs,
                     Tensor adj,
                     bool final = false);

void Att(Tensor out, Tensor va, Tensor context, Tensor state, Tensor coverage);
void AttBack(Tensor gva,
             Tensor gContext,
             Tensor gState,
             Tensor gCoverage,
             Tensor va,
             Tensor context,
             Tensor state,
             Tensor coverage,
             Tensor adj);

void LayerNormalization(Tensor out,
                        Tensor in,
                        Tensor gamma,
                        Tensor beta,
                        float eps = 1e-9);
void LayerNormalizationGrad(Tensor gradX,
                            Tensor gradGamma,
                            Tensor gradBeta,
                            Tensor adj,
                            Tensor y,
                            Tensor x,
                            Tensor gamma,
                            Tensor beta,
                            float eps = 1e-9);

void Shift(Tensor out, Tensor in, Shape shift, bool invert = false);

void SetSparse(float*,
               const std::vector<size_t>& indeces,
               const std::vector<float>& values);

void HighwayForward(Tensor out,
                    const Tensor in1,
                    const Tensor in2,
                    const Tensor t);

void HighwayBackward(Tensor out1,
                     Tensor out2,
                     Tensor outt,
                     const Tensor in1,
                     const Tensor in2,
                     const Tensor t,
                     const Tensor adj);
}
