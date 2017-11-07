#pragma once

#include "gpu/shape.h"
#include "gpu/tmp.h"
#include "gpu/tensor.h"

#include "functional/functional.h"

namespace marian {
  namespace gpu {
    namespace f = functional;
    namespace facc = functional::accumulator;

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

    template <bool broadcast=false,
              typename T,
              class Functor = decltype(f::same),
              class Accumulator = decltype(facc::plus),
              class AccumulatorZero = decltype(f::zero)>
    __device__ inline T reduce_row(int cols,
                                   gpu::Tensor<T>& rowArg,
                                   Functor functor = f::same,
                                   Accumulator acc = facc::plus,
                                   AccumulatorZero accZero = f::zero) {
      gpu::Array<Tensor<T>, 1> rowArgs = { rowArg };
      return reduce_row<broadcast>(cols, rowArgs, functor, acc, accZero);
    }

    template <bool broadcast=false,
              size_t K,
              typename T,
              class Functor = decltype(f::same),
              class Accumulator = decltype(facc::plus),
              class AccumulatorZero = decltype(f::zero)>
    __device__ inline T reduce_row(int cols,
                                   gpu::Array<gpu::Tensor<T>, K>& rowArgs,
                                   Functor functor = f::same,
                                   Accumulator acc = facc::plus,
                                   AccumulatorZero accZero = f::zero) {

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

    template <class Lambda>
    __DI__ void foreach(size_t length, Lambda& lambda) {
      for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
        int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
        if(index < length)
          lambda(index);
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
          auto rowArgs = gpu::rows(args, j);
          transform_row<broadcast>(out.row(j), rowArgs, functor);
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

    struct ReduceRow {
      template <typename ...Args>
      __device__ inline static float apply(Args ...args) {
        return gpu::reduce_row(args...);
      }
    };

    template <class Functor, typename T>
    __device__ inline void foreach_row(gpu::Tensor<T>& out,
                                       gpu::Tensor<T>& in,
                                       Functor functor) {
      int rows = out.shape().elements() / out.shape().back();
      for(int bid = 0; bid < rows; bid += gridDim.x) {
        int row_index = bid + blockIdx.x;
        if(row_index < rows) {
          auto outRow = out.row(row_index);
          auto inRow = in.row(row_index);

          gpu::transform_row(outRow, inRow,
                             f::reduce<ReduceRow>(functor, inRow));
        }
      }
    }

  }
}