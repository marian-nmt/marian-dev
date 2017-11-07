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
#include "gpu/primitives.h"

namespace marian {

bool IsNan(Tensor in);

const int MAX_THREADS = 512;
const int MAX_BLOCKS = 65535;

cublasHandle_t create_handle(size_t);

namespace gpu {
  namespace kernels {
    using namespace functional;

    template <bool broadcast = false,
              size_t K,
              class PreReduceFunctor,
              class Accumulator,
              class AccumulatorZero,
              class AssignFunctor,
              typename T = float>
    __global__ void gReduceRows(gpu::Tensor<T> out,
                                gpu::Array<gpu::Tensor<T>, K> args,
                                gpu::Shape full,
                                PreReduceFunctor preFunc = same,
                                Accumulator accFunc = accumulator::plus,
                                AccumulatorZero accZero = zero,
                                AssignFunctor assignFunc = _1 = _2) {

      auto lambda = [&](size_t row_index) {

        int cols = full.back();
        gpu::Array<gpu::Tensor<T>, K> rowArgs;
        if(broadcast) {
          //gpu::Array<int, K> indices;
          //for(int i = 0; i < K; ++i) {
          //  int argRows = args[i].shape().elements() / args[i].shape().back();
          //  indices[i] = argRows % row_index;
          //}
          //rowArgs = gpu::rows(args, indices);
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
    int rows = full.elements() / length;
    int cols = full.back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols);
    int shared = sizeof(float) * threads * 2;

    gpu::Shape gFull = full;
    bool broadcast = false;
    for(int i = 0; i < K; ++i)
      broadcast = broadcast || gFull != gIns[i].shape();

    using namespace functional;
    auto addScale = _1 += _2 * scale;

    //std::cerr << broadcast << std::endl;

    if(broadcast)
      gpu::kernels::gReduceRows<true><<<blocks, threads, shared>>>(
        gOut, gIns, gFull, functor, accumulator::plus, zero, addScale);
    else
      gpu::kernels::gReduceRows<false><<<blocks, threads, shared>>>(
        gOut, gIns, gFull, functor, accumulator::plus, zero, addScale);

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
