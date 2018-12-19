/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "tensors/gpu/add.h"

#include "tensors/gpu/cuda_helpers.h"

#include "functional/functional.h"
#include "functional/shape.h"
#include "functional/tensor.h"
#include "functional/tmp.h"

namespace marian {

namespace gpu {

template <size_t K, class Functor, typename T>
__global__ void gAddGeneric(Functor functor,
                            const functional::Shape full,
                            functional::Tensor<T> out,
                            functional::Array<functional::Tensor<T>, K> ins,
                            T scale = 1.0) {
  int outLength = out.shape().elements();
  bool same = outLength == full.elements();
  for(int i = 0; i < K; ++i)
    same = same && outLength == ins[i].shape().elements();

  constexpr size_t N = functional::Shape::size();
  functional::Array<int, N> len;
  for(int i = 0; i < N; ++i)
    len[i] = full[i] / out.shape()[i];

  functional::Array<int, N> dims;
  for(int bid = 0; bid < outLength; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < outLength) {
      if(same) {
        out.data()[index] += functional::apply(functor, ins, index) * scale;
      } else {
        out.shape().dims(index, dims);
        out.data()[index] += functional::loops(functor, ins, len, dims) * scale;
      }
    }
  }
}

template <size_t K, class Functor, typename T>
__global__ void gAddEqual(Functor functor,
                          functional::Tensor<T> out,
                          functional::Array<functional::Tensor<T>, K> ins,
                          T scale,
                          bool broadcast) {
  int length = out.shape().elements();
  functional::Array<int, functional::Shape::size()> dims;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      functional::Array<int, K> indices;
      indices.fill(index);

      if(broadcast) {
        out.shape().dims(index, dims);
        for(size_t i = 0; i < K; ++i)
          indices[i] = ins[i].shape().bindex(dims);
      }

      out.data()[index] += functional::apply(functor, ins, indices) * scale;
    }
  }
}

template <size_t K, class Functor, typename T, typename AccType = float>
__global__ void gAddReduce(Functor functor,
                           const functional::Shape full,
                           functional::Tensor<T> out,
                           functional::Array<functional::Tensor<T>, K> ins,
                           AccType scale = 1.0) {
  int rows = full.elements() / full.back();
  int cols = full.back();

  bool same = true;
  for(int i = 0; i < K; ++i)
    same = same && ins[i].shape().elements() == full.elements();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      // make sure shared memory is the same for different types
      // by using bytes instead of typeS T
      extern __shared__ uint8_t _sharedBytes[];
      AccType* _share = (AccType*)_sharedBytes;

      AccType* _sum = _share + blockDim.x;
      if(same) {
        _sum[threadIdx.x] = (AccType)0.f;
        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols)
            _sum[threadIdx.x] += (AccType)functional::apply(functor, ins, j * cols + id);
        }
      } else {
        functional::Array<int, functional::Shape::size()> dims;
        _sum[threadIdx.x] = (AccType)0.f;

        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols) {
            full.dims(j * cols + id, dims);
            functional::Array<int, K> indices;
            for(int i = 0; i < K; ++i)
              indices[i] = ins[i].shape().bindex(dims);
            _sum[threadIdx.x] += (AccType)functional::apply(functor, ins, indices);
          }
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      out.data()[j] += _sum[0] * scale;
    }
  }
}

template <typename T, class Functor, class... Tensors>
void AddTyped(Functor functor, T scale, marian::Tensor out, Tensors... tensors) {
  cudaSetDevice(out->getDeviceId().no);

  auto full = marian::Shape::broadcast({out, tensors...});

  int length = out->shape().elements();

  constexpr size_t K = sizeof...(Tensors);

  functional::Tensor<T> gOut = out;
  functional::Array<functional::Tensor<T>, K> gIns = {tensors...};

  if(full.back() != 1 && out->shape().back() == 1) {
    size_t m = full.elements() / length;
    size_t k = full.back();

    int blocks = std::min(MAX_BLOCKS, (int)m);
    int threads = std::min(MAX_THREADS, (int)k);
    int shared = sizeof(float) * threads * 2;

    gAddReduce<K, Functor, T, float><<<blocks, threads, shared>>>(functor, full, gOut, gIns, scale);

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

template <class Functor, class... Tensors>
void Add(Functor functor, float scale, marian::Tensor out, Tensors... tensors) {

  if(out->type() == Type::float32) {
    AddTyped<float>(functor, scale, out, tensors...);
  } else if(out->type() == Type::float16) {
    AddTyped<half>(functor, __float2half(scale), out, tensors...);
  } else {
    ABORT("Type {} not yet supported", out->type());
  }
}

#include "tensors/gpu/add.inc"
}  // namespace gpu
}  // namespace marian
