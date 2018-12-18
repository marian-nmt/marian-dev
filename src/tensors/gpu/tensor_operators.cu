//#include <thrust/transform_reduce.h>

#include "tensors/tensor_operators.h"

#include "functional/functional.h"
#include "functional/tensor.h"
#include "tensors/allocator.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/cuda_helpers.h"

#include "3rd_party/reduce_all.h"

namespace marian {

namespace gpu {

namespace atomics {

static inline  __device__ void atomicAdd(float *address, float val) {
  ::atomicAdd(address, val);
}

// @TODO: copied from CuTorch, adapt this better, give credit.
static inline  __device__ void atomicAdd(half *address, half val) {
  unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
#if CUDA_VERSION < 9000
    half hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
#else
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = hsum + val;
    hsum = __half_raw(tmpres);
#endif
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
   } while (assumed != old);
}
}

struct isnan_test {
  __host__ __device__ bool operator()(const float a) const { return isnan(a); }
};

__device__ inline float stableSigmoid(float x) {
  if(x >= 0) {
    float z = expf(-x);
    return 1.0 / (1.0 + z);
  } else {
    float z = expf(x);
    return z / (1.0 + z);
  }
}

bool IsNan(Tensor in) {
  // cudaSetDevice(in->getDeviceId().no);
  // thrust::device_ptr<float> begin = thrust::device_pointer_cast(in->data());
  // thrust::device_ptr<float> end
  //    = thrust::device_pointer_cast(in->data() + in->size());
  // return thrust::transform_reduce(
  //    begin, end, isnan_test(), 0, thrust::plus<bool>());
  return false;
}

template <typename To, typename From>
__global__ void gCopyCastTo(To* out, const From* in, int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out[index] = in[index];
    }
  }
}

template <typename To, typename From>
void CopyCastTo(To* out, const From* in, int length) {
  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
  gCopyCastTo<<<blocks, threads>>>(out, in, length);
}

template <typename T>
void CopyCastFrom(Tensor out, const T* in, int length) {
  if(out->type() == Type::float32) {
    CopyCastTo(out->data<half>(), in, length);
  } else if(out->type() == Type::float16) {
    CopyCastTo(out->data<half>(), in, length);
  } else {
    ABORT("CopyCastTo to type {} not implemented", out->type());
  }
}

void CopyCast(Tensor out, Tensor in) {
  cudaSetDevice(out->getDeviceId().no);

  if(in->type() == Type::float32) {
    CopyCastFrom(out, in->data<float>(), (int)in->size());
  } else if(in->type() == Type::float16) {
    CopyCastFrom(out, in->data<half>(), (int)in->size());
  } else {
    ABORT("CopyCastFrom from type {} not implemented", in->type());
  }
}

void ConcatCont(Tensor out, const std::vector<Tensor>& inputs, int axis) {
  cudaSetDevice(out->getDeviceId().no);
  int step = 1;
  for(int i = 0; i < axis; ++i)
    step *= out->shape()[i];

  size_t offset1 = 0;
  for(int i = 0; i < step; ++i) {
    for(auto in : inputs) {
      size_t size = (in->shape().elements() / step) * sizeOf(out->type());
      size_t offset2 = i * size;

      cudaMemcpy(out->data<uint8_t>() + offset1,
                 in->data<uint8_t>() + offset2,
                 size,
                 cudaMemcpyDeviceToDevice);

      offset1 += size;
    }
  }
  cudaStreamSynchronize(0);
}

template <bool add, typename T>
__global__ void gInsertCols(T* out,
                            const T* in,
                            size_t rows,
                            size_t cols,
                            size_t cols_out,
                            size_t cols_in,
                            size_t offset_out,
                            size_t offset_in) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* rowOut = out + j * cols_out + offset_out;
      const T* rowIn = in + j * cols_in + offset_in;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          if(add)
            rowOut[i] += rowIn[i];
          else
            rowOut[i] = rowIn[i];
      }
    }
  }
}

void Concatenate1(Tensor out, const std::vector<Tensor>& inputs) {
  cudaSetDevice(out->getDeviceId().no);

  int rows = out->shape().elements() / out->shape().back();

  size_t offset = 0;
  int cols_out = out->shape().back();

  for(auto in : inputs) {
    ABORT_IF(rows != in->shape().elements() / in->shape().back(),
             "First dimension must be equal");
    int cols_in = in->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols_in);

    if(out->type() == Type::float32) {
      gInsertCols<false><<<blocks, threads>>>(
          out->data<float>(), in->data<float>(), rows, cols_in, cols_out, cols_in, offset, 0);
    } else if(out->type() == Type::float16) {
      gInsertCols<false><<<blocks, threads>>>(
          out->data<half>(), in->data<half>(), rows, cols_in, cols_out, cols_in, offset, 0);
    } else {
      ABORT("Concatenate1 not implemented for type {}", out->type());
    }
    offset += cols_in;
  }
  cudaStreamSynchronize(0);
}

template <typename T>
__global__ void gJoin2(T* out,
                       size_t rowBatch,
                       size_t cols,
                       const T* in1,
                       size_t inStride1,
                       const T* in2,
                       size_t inStride2) {
  int outStride = inStride1 + inStride2;
  int rows = rowBatch * outStride;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* rowOut = out + j * cols;

      int curBatch = j / outStride;
      int curPos = j % outStride;

      int jIn1 = (curBatch * inStride1) + curPos;
      int jIn2 = (curBatch * inStride2) + curPos - inStride1;

      const T* rowIn1 = in1 + jIn1 * cols;
      const T* rowIn2 = in2 + jIn2 * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          if(curPos < inStride1)
            rowOut[i] = rowIn1[i];
          else
            rowOut[i] = rowIn2[i];
        }
      }
    }
  }
}

void Concatenate2(Tensor out, Tensor in1, Tensor in2) {
  cudaSetDevice(out->getDeviceId().no);

  size_t rows = out->shape().elements() / out->shape().back();
  size_t cols = out->shape().back();

  size_t rowStride1 = in1->shape()[-2];
  size_t rowStride2 = in2->shape()[-2];

  size_t rowBatch = rows / out->shape()[-2];

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);


  if(out->type() == Type::float32) {
     gJoin2<<<blocks, threads>>>(out->data<float>(),
                                 rowBatch,
                                 cols,
                                 in1->data<float>(),
                                 rowStride1,
                                 in2->data<float>(),
                                 rowStride2);
  } else if(out->type() == Type::float16) {
     gJoin2<<<blocks, threads>>>(out->data<half>(),
                                 rowBatch,
                                 cols,
                                 in1->data<half>(),
                                 rowStride1,
                                 in2->data<half>(),
                                 rowStride2);
  } else {
    ABORT("Concatenate2 not implemented for type {}", out->type());
  }

  cudaStreamSynchronize(0);
}

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax) {
  if(ax == out->shape().size() - 1)
    Concatenate1(out, inputs);
  else if(ax == out->shape().size() - 2 && inputs.size() == 2)
    Concatenate2(out, inputs[0], inputs[1]);
  else
    ConcatCont(out, inputs, ax);
}

void Split1(std::vector<Tensor>& outputs, const Tensor in) {
  cudaSetDevice(in->getDeviceId().no);

  size_t offset = 0;
  int rows = in->shape().elements() / in->shape().back();
  int cols_in = in->shape().back();
  for(auto out : outputs) {
    ABORT_IF(rows != out->shape().elements() / out->shape().back(),
             "First dimension must be equal");
    int cols_out = out->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols_out);

    if(out->type() == Type::float32) {
      gInsertCols<true><<<blocks, threads>>>(
          out->data<float>(), in->data<float>(), rows, cols_out, cols_out, cols_in, 0, offset);
    } else if(out->type() == Type::float16) {
      gInsertCols<true><<<blocks, threads>>>(
          out->data<half>(), in->data<half>(), rows, cols_out, cols_out, cols_in, 0, offset);
    } else {
      ABORT("Split1 not implemented for type {}", out->type());
    }

    offset += cols_out;
  }
  cudaStreamSynchronize(0);
}

// @TODO: this function is just a temporary fix until I come up with
// something better for the situation below.
template <typename T>
__global__ void gAddRow(T* out, const T* in, int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out[index] = in[index] + out[index];
    }
  }
}

void SplitCont(std::vector<Tensor>& outputs, const Tensor in, int axis) {
  cudaSetDevice(in->getDeviceId().no);

  int step = 1;
  for(int i = 0; i < axis; ++i)
    step *= in->shape()[i];

  int offset1 = 0;
  for(int i = 0; i < step; ++i) {
    for(auto out : outputs) {
      int size = out->shape().elements() / step;
      int offset2 = i * size;

      // BUG: this is does not add gradients
      // cudaMemcpyAsync(out->data() + offset2,
      //                in->data() + offset1,
      //                size * sizeof(float),
      //                cudaMemcpyDeviceToDevice);

      // @TODO: this is a quick but bad fix for the above bug
      int threads = std::min(MAX_THREADS, size);
      int blocks = std::min(MAX_BLOCKS, size / threads + (size % threads != 0));

      if(out->type() == Type::float32) {
        gAddRow<<<blocks, threads>>>(
            out->data<float>() + offset2, in->data<float>() + offset1, size);
      } else if(out->type() == Type::float16) {
        gAddRow<<<blocks, threads>>>(
            out->data<half>() + offset2, in->data<half>() + offset1, size);
      } else {
        ABORT("SplitCont not implemented for type {}", out->type());
      }
      offset1 += size;
    }
  }
  cudaStreamSynchronize(0);
}

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax) {
  if(ax == in->shape().size() - 1)
    Split1(outputs, in);
  else
    SplitCont(outputs, in, ax);
}

template <bool add, typename T>
__global__ void gTransposeND(
    functional::Tensor<T> out,
    const functional::Tensor<T> in,
    const functional::Array<int, functional::Shape::size()> permute) {
  constexpr size_t N = functional::Shape::size();
  functional::Array<int, N> oDims;
  functional::Array<int, N> pDims;

  int length = out.shape().elements();
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out.shape().dims(index, oDims);
      for(int i = 0; i < N; ++i)
        pDims[permute[i]] = oDims[i];

      int inIndex = in.shape().index(pDims);

      // TODO: operates on raw indices, change to
      // converting Tensor::operator[]
      if(add)
        out.data()[index] += in.data()[inIndex];
      else
        out.data()[index] = in.data()[inIndex];
    }
  }
}

template <bool add, typename T>
__global__ void gTranspose0213(T* out,
                               const T* in,
                               int rows,
                               int cols,
                               int stride1,
                               int stride2) {
  int stride = stride1 * stride2;
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* rowOut = out + j * cols;

      int z = j / stride;
      int y = (j % stride) / stride1;
      int x = (j % stride) % stride1;
      int j2 = z * stride + x * stride2 + y;

      const T* rowIn = in + j2 * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          if(add)
            rowOut[i] += rowIn[i];
          else
            rowOut[i] = rowIn[i];
        }
      }
    }
  }
}

void TransposeND(Tensor out, Tensor in, const std::vector<int>& vAxis) {
  cudaSetDevice(out->getDeviceId().no);

  if(vAxis == std::vector<int>({0, 2, 1, 3})) {
    int rows = out->shape().elements() / out->shape().back();
    int cols = out->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols);

    int stride1 = out->shape()[-2];
    int stride2 = out->shape()[-3];

    if(in->type() == Type::float32) {
      gTranspose0213<false><<<blocks, threads>>>(out->data<float>(), in->data<float>(), rows, cols, stride1, stride2);
    } else if(in->type() == Type::float16) {
      gTranspose0213<false><<<blocks, threads>>>(out->data<__half>(), in->data<__half>(), rows, cols, stride1, stride2);
    } else {
      ABORT("Transpose for type {} not implemented", in->type());
    }
  } else {
    functional::Array<int, functional::Shape::size()> axes;
    int diff = functional::Shape::size() - vAxis.size();
    for(int i = 0; i < axes.size(); ++i)
      if(i < diff)
        axes[i] = i;
      else
        axes[i] = vAxis[i - diff] + diff;

    int length = out->shape().elements();
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    if(in->type() == Type::float32) {
      gTransposeND<false, float><<<blocks, threads>>>(out, in, axes);
    } else if(in->type() == Type::float16) {
      gTransposeND<false, __half><<<blocks, threads>>>(out, in, axes);
    } else {
      ABORT("Transpose for type {} not implemented", in->type());
    }
  }
}

//@TODO: code duplication?
void TransposeNDGrad(Tensor out, Tensor in, const std::vector<int>& vAxis) {
  cudaSetDevice(out->getDeviceId().no);
  if(vAxis == std::vector<int>({0, 2, 1, 3})) {
    int rows = out->shape().elements() / out->shape().back();
    int cols = out->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols);

    int stride1 = out->shape()[-2];
    int stride2 = out->shape()[-3];

    if(in->type() == Type::float32) {
      gTranspose0213<true><<<blocks, threads>>>(out->data<float>(), in->data<float>(), rows, cols, stride1, stride2);
    } else if(in->type() == Type::float16) {
      gTranspose0213<true><<<blocks, threads>>>(out->data<__half>(), in->data<__half>(), rows, cols, stride1, stride2);
    } else {
      ABORT("Transpose for type {} not implemented", in->type());
    }
  } else {
    functional::Array<int, functional::Shape::size()> axes;
    int diff = functional::Shape::size() - vAxis.size();
    for(int i = 0; i < axes.size(); ++i)
      if(i < diff)
        axes[i] = i;
      else
        axes[i] = vAxis[i - diff] + diff;

    int length = out->shape().elements();
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    if(in->type() == Type::float32) {
      gTransposeND<true, float><<<blocks, threads>>>(out, in, axes);
    } else if(in->type() == Type::float16) {
      gTransposeND<true, __half><<<blocks, threads>>>(out, in, axes);
    } else {
      ABORT("Transpose for type {} not implemented", in->type());
    }
  }
}

// @TODO: handle __half2
template <typename T, typename AccType = float>
__global__ void gSoftmax(T* out,
                         functional::Shape outShape,
                         const T* in) {
  using namespace functional;

  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* so = out + j * cols;
      const T* sp = in + j * cols;

      // CUDA complains if type or size of shared memory changes, keep size constant.
      extern __shared__ uint8_t _sharedBytes[];
      T* _share = (T*)_sharedBytes;
      AccType* _shareAccType = (AccType*)_sharedBytes;

      T* _max = _share + blockDim.x;
      _max[threadIdx.x] = -CUDA_FLT_MAX;  // mask
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      T max = _max[0];
      __syncthreads();

      AccType* _sum = _shareAccType + blockDim.x; // accumulate into AccType

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          T ex = Ops<T>::exp(sp[id] - max);
          so[id] = (T)ex;
          _sum[threadIdx.x] += (AccType)ex; // accumulate into AccType
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          so[id] = (T)((AccType)so[id] / _sum[0]); // divide as AccType then convert
        }
      }
    }
  }
}

void Softmax(Tensor out, Tensor in) {
  cudaSetDevice(out->getDeviceId().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t k = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);

  if(in->type() == Type::float32) {
    int shared = sizeof(float) * threads * 2;
    gSoftmax<float, float><<<blocks, threads, shared>>>(out->data<float>(), out->shape(), in->data<float>());
  } else if (in->type() == Type::float16) {
    int shared = sizeof(float) * threads * 2; // keep size of shared memory
    // accumulate into float
    gSoftmax<half, float><<<blocks, threads, shared>>>(out->data<__half>(), out->shape(), in->data<__half>());
  } else {
    ABORT("Softmax not implemented for type {}", in->type());
  }
}

template <typename T, typename AccType = float>
__global__ void gLogSoftmax(T* out,
                            const functional::Shape outShape,
                            const T* in) {

  using namespace functional;

  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* so = out + j * cols;
      const T* sp = in + j * cols;

      // CUDA complains if type or size of shared memory changes, keep size constant.
      extern __shared__ uint8_t _sharedBytes[];
      T* _share = (T*)_sharedBytes;
      AccType* _shareAccType = (AccType*)_sharedBytes;

      T* _max = _share + blockDim.x;
      _max[threadIdx.x] = sp[threadIdx.x];
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      T max = _max[0];
      __syncthreads();

      AccType* _sum = _shareAccType + blockDim.x; // keep AccType for accumulation

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          T sm = sp[id] - max;
          AccType ex = Ops<AccType>::exp(sm); // sum with AccType
          so[id] = sm;
          _sum[threadIdx.x] += ex; // sum with AccType
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          so[id] -= (T)Ops<AccType>::log(_sum[0]); // take log at the end and convert
        }
    }
  }
}

void LogSoftmax(Tensor out, Tensor in) {
  cudaSetDevice(out->getDeviceId().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t k = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);

  if(in->type() == Type::float32) {
    int shared = sizeof(float) * threads * 2;
    gLogSoftmax<float, float><<<blocks, threads, shared>>>(out->data<float>(), out->shape(), in->data<float>());
  } else if (in->type() == Type::float16) {
    int shared = sizeof(float) * threads * 2; // keep size of shared memory
    // accumulate in float
    gLogSoftmax<half, float><<<blocks, threads, shared>>>(out->data<__half>(), out->shape(), in->data<__half>());
  } else {
    ABORT("LogSoftmax not implemented for type {}", in->type());
  }
}

///////////////////////////////////////////////////////

template <typename T, typename AccType = float>
__global__ void gSoftmaxGrad(T* grad,
                             const T* adj,
                             const T* val,
                             const int rows,
                             const int cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ uint8_t _sharedBytes[];
      AccType* _shareAccType = (AccType*)_sharedBytes;

      AccType* _sum = _shareAccType + blockDim.x;

      T* gradRow = grad + j * cols;
      const T* adjRow = adj + j * cols;
      const T* valRow = val + j * cols;
      _sum[threadIdx.x] = (AccType)0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += (AccType)valRow[id] * (AccType)adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip]; // accumulates in AccType
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType val = (AccType)valRow[id] * ((AccType)adjRow[id] - _sum[0]);
          if(val)
            gradRow[id] += (T)val;
        }
      }
    }
  }
}

void SoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  cudaSetDevice(adj->getDeviceId().no);
  // grad and val are both m-by-k matrices, passed as input.
  // A weighted average of each row of grad (according to the weights
  // specified in val) is computed and subtracted from Out.
  // adj is multiplied for each element to get backward step in autodiff
  int m = grad->shape().elements() / grad->shape().back();
  int k = grad->shape().back();

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads * 2;

  if(grad->type() == Type::float32) {
    gSoftmaxGrad<float, float><<<blocks, threads, shared>>>(
      grad->data<float>(), adj->data<float>(), val->data<float>(), m, k);
  } else if (grad->type() == Type::float16) {
    // Accumulate into float
    gSoftmaxGrad<half, float><<<blocks, threads, shared>>>(
      grad->data<half>(), adj->data<half>(), val->data<half>(), m, k);
  } else {
    ABORT("SoftmaxGrad not implemented for type {}", grad->type());
  }
}

template <typename T, typename AccType = float>
__global__ void gLogSoftmaxGrad(T* grad,
                                const T* adj,
                                const T* val,
                                const int rows,
                                const int cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ uint8_t _sharedBytes[];
      AccType* _share = (AccType*)_sharedBytes;

      AccType* _sum = _share + blockDim.x;

      T* gradRow = grad + j * cols;
      const T* adjRow = adj + j * cols;
      const T* valRow = val + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += (AccType)adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip]; // AccType
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          gradRow[id] += (T)((AccType)adjRow[id] - (functional::Ops<AccType>::exp((AccType)valRow[id]) * _sum[0]));
      }
    }
  }
}

void LogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  cudaSetDevice(adj->getDeviceId().no);

  // grad and val are both m-by-k matrices, passed as input.
  // A weighted average of each row of grad (according to the weights
  // specified in val) is computed and subtracted from Out.
  // adj is multiplied for each element to get backward step in autodiff
  int m = grad->shape().elements() / grad->shape().back();
  int k = grad->shape().back();

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads * 2;

  if(grad->type() == Type::float32) {
    gLogSoftmaxGrad<float, float><<<blocks, threads, shared>>>(
      grad->data<float>(), adj->data<float>(), val->data<float>(), m, k);
  } else if (grad->type() == Type::float16) {
    // accumulate into float
    gLogSoftmaxGrad<half, float><<<blocks, threads, shared>>>(
      grad->data<half>(), adj->data<half>(), val->data<half>(), m, k);
  } else {
    ABORT("LogSoftmaxGrad not implemented for type {}", grad->type());
  }  
}

///////////////////////////////////////////////////////

template <typename T>
__global__ void gCopyRows(T* out,
                          const T* in,
                          size_t cols,
                          const IndexType* sourceRowIdx,
                          size_t rows) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      size_t dstId = j;
      size_t srcId = sourceRowIdx[j];

      T* rowOut = out + dstId * cols;
      const T* rowIn = in + srcId * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i];
      }
    }
  }
}

void CopyRows(Tensor out,
              const Tensor in,
              const Tensor indices) {

  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  size_t cols = in->shape().back();
  size_t rowsToCopy = indices->size();

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);

  if(out->type() == Type::float32) {
    gCopyRows<<<blocks, threads>>>(
      out->data<float>(), in->data<float>(), cols, indices->data<IndexType>(), rowsToCopy);
  } else if (out->type() == Type::float16) {
    gCopyRows<<<blocks, threads>>>(
      out->data<half>(), in->data<half>(), cols, indices->data<IndexType>(), rowsToCopy);
  } else {
    ABORT("CopyRows not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gPasteRows(T* out,
                           const T* in,
                           size_t cols,
                           const IndexType* targetRowIdx,
                           size_t rows) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      size_t dstId = targetRowIdx[j];
      size_t srcId = j;

      T* rowOut = out + dstId * cols;
      const T* rowIn = in + srcId * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          atomics::atomicAdd(rowOut + i, rowIn[i]);
      }
    }
  }
}

void PasteRows(Tensor out,
               const Tensor in,
               const Tensor indices) {

  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  size_t cols = in->shape().back();
  size_t rowsToCopy = indices->size();

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);

  if(out->type() == Type::float32) {
    gPasteRows<<<blocks, threads>>>(
      out->data<float>(), in->data<float>(), cols, indices->data<IndexType>(), rowsToCopy);
  } else if (out->type() == Type::float16) {
    gPasteRows<<<blocks, threads>>>(
      out->data<half>(), in->data<half>(), cols, indices->data<IndexType>(), rowsToCopy);
  } else {
    ABORT("CopyRows not implemented for type {}", out->type());
  }
}

/////////////

template <typename T>
__global__ void gCopyCols(T* out,
                          const T* in,
                          size_t rows,
                          size_t colsIn,
                          const IndexType* sourceColIdx,
                          size_t colsOut) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const T* rowIn = in + j * colsIn;
      T* rowOut = out + j * colsOut;

      for(int tid = 0; tid < colsOut; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < colsOut)
          rowOut[i] = rowIn[sourceColIdx[i]];
      }
    }
  }
}

void CopyCols(Tensor out, const Tensor in, const Tensor indices) {
  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  size_t rows = in->shape().elements() / in->shape().back();
  size_t cols = in->shape().back();

  size_t colsToCopy = indices->size();

  int threads = std::min(MAX_THREADS, (int)colsToCopy);
  int blocks = std::min(MAX_BLOCKS, (int)rows);

  if(out->type() == Type::float32) {
    gCopyCols<<<blocks, threads>>>(
      out->data<float>(), in->data<float>(), rows, cols, indices->data<IndexType>(), colsToCopy);
  } else if (out->type() == Type::float16) {
    gCopyCols<<<blocks, threads>>>(
      out->data<half>(), in->data<half>(), rows, cols, indices->data<IndexType>(), colsToCopy);
  } else {
    ABORT("CopyCols not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gPasteCols(T* out,
                           const T* in,
                           size_t rows,
                           size_t colsOut,
                           const IndexType* targetColIdx,
                           size_t colsIn) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const T* rowIn = in + j * colsIn;
      T* rowOut = out + j * colsOut;

      for(int tid = 0; tid < colsIn; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < colsIn)
          rowOut[targetColIdx[i]] += rowIn[i]; // @TODO: atomicAdd?
      }
    }
  }
}

void PasteCols(Tensor out,
               const Tensor in,
               const Tensor indices) {
  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  size_t rows = in->shape().elements() / in->shape().back();
  size_t cols = in->shape().back();

  size_t colsToCopy = indices->size();

  int threads = std::min(MAX_THREADS, (int)colsToCopy);
  int blocks = std::min(MAX_BLOCKS, (int)rows);

  if(out->type() == Type::float32) {
    gPasteCols<<<blocks, threads>>>(
      out->data<float>(), in->data<float>(), rows, cols, indices->data<IndexType>(), colsToCopy);
  } else if (out->type() == Type::float16) {
    gPasteCols<<<blocks, threads>>>(
      out->data<half>(), in->data<half>(), rows, cols, indices->data<IndexType>(), colsToCopy);
  } else {
    ABORT("PasteCols not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gSelect(T* out,
                        functional::Shape outShape,
                        const T* in,
                        const functional::Shape inShape,
                        int axis,
                        IndexType* d_indices) {
  int length = outShape.elements();
  functional::Array<int, functional::Shape::size()> dims;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      outShape.dims(index, dims);
      dims[axis] = d_indices[dims[axis]];
      int inIndex = inShape.index(dims);
      out[index] = in[inIndex];
    }
  }
}

template <typename T>
__global__ void gInsert(T* out,
                        functional::Shape outShape,
                        const T* in,
                        const functional::Shape inShape,
                        int axis,
                        IndexType* d_indices) {
  int length = inShape.elements();
  functional::Array<int, functional::Shape::size()> dims;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      inShape.dims(index, dims);
      dims[axis] = d_indices[dims[axis]];
      int outIndex = outShape.index(dims);
      out[outIndex] += in[index];
    }
  }
}

void Select(Tensor out,
            const Tensor in,
            const Tensor indices,
            int axis) {
  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  int axisGPU = axis + functional::Shape::size() - out->shape().size();

  if(out->type() == Type::float32) {
    gSelect<<<blocks, threads>>>(out->data<float>(),
                                out->shape(),
                                in->data<float>(),
                                in->shape(),
                                axisGPU,
                                indices->data<IndexType>());  
  } else if (out->type() == Type::float16) {
    gSelect<<<blocks, threads>>>(out->data<half>(),
                                out->shape(),
                                in->data<half>(),
                                in->shape(),
                                axisGPU,
                                indices->data<IndexType>());  
  } else {
    ABORT("Select not implemented for type {}", out->type());
  }
}

void Insert(Tensor out,
            const Tensor in,
            const Tensor indices,
            int axis) {
  matchOrAbort<IndexType>(indices->type());
  cudaSetDevice(in->getDeviceId().no);

  int length = in->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  int axisGPU = axis + functional::Shape::size() - out->shape().size();

  if(out->type() == Type::float32) {
    gInsert<<<blocks, threads>>>(out->data<float>(),
                                out->shape(),
                                in->data<float>(),
                                in->shape(),
                                axisGPU,
                                indices->data<IndexType>());  
  } else if (out->type() == Type::float16) {
    gInsert<<<blocks, threads>>>(out->data<half>(),
                                out->shape(),
                                in->data<half>(),
                                in->shape(),
                                axisGPU,
                                indices->data<IndexType>());  
  } else {
    ABORT("Insert not implemented for type {}", out->type());
  }
}

__global__ void gGRUFastForward(float* out,
                                const float* state,
                                const float* xW,
                                const float* sU,
                                const float* b,
                                const float* mask,
                                size_t rows,
                                size_t cols,
                                bool final) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];
      float* rowOut = out + j * cols;
      const float* rowState = state + j * cols;

      const float* xWrow = xW + j * cols * 3;
      const float* sUrow = sU + j * cols * 3;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float r = stableSigmoid(xWrow[i] + sUrow[i] + b[i]);

          int k = i + cols;

          float z = stableSigmoid(xWrow[k] + sUrow[k] + b[k]);

          int l = i + 2 * cols;
          float h;
          if(final)
            h = tanhf(xWrow[l] + (sUrow[l] + b[l]) * r);
          else
            h = tanhf(xWrow[l] + sUrow[l] * r + b[l]);

          float out = (1.0f - z) * h + z * rowState[i];
          rowOut[i] = m * out + (1 - m) * rowState[i];
        }
      }
    }
  }
}

void GRUFastForward(Tensor out, std::vector<Tensor> inputs, bool final) {
  matchOrAbort<float>(out->type());
  cudaSetDevice(out->getDeviceId().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gGRUFastForward<<<blocks, threads>>>(
      out->data(),                                // output
      inputs[0]->data(),                          // state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      rows,
      cols,
      final);
}

__global__ void gGRUFastBackward(float* outState,
                                 float* outXW,
                                 float* outSU,
                                 float* outB,
                                 const float* state,
                                 const float* xW,
                                 const float* sU,
                                 const float* b,
                                 const float* mask,
                                 const float* adj,
                                 size_t rows,
                                 size_t cols,
                                 bool final) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];

      float* rowOutState = outState + j * cols;
      float* rowOutXW = outXW + j * cols * 3;
      float* rowOutSU = outSU + j * cols * 3;

      const float* rowState = state + j * cols;
      const float* rowXW = xW + j * cols * 3;
      const float* rowSU = sU + j * cols * 3;
      const float* rowAdj = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          int k = i + cols;
          int l = i + 2 * cols;

          float r = stableSigmoid(rowXW[i] + rowSU[i] + b[i]);
          float z = stableSigmoid(rowXW[k] + rowSU[k] + b[k]);

          float h;
          if(final)
            h = tanhf(rowXW[l] + (rowSU[l] + b[l]) * r);
          else
            h = tanhf(rowXW[l] + rowSU[l] * r + b[l]);

          float adj = rowAdj[i];

          float t = (1 - z) * (1 - h * h);

          // df/ds
          if(outState)
            rowOutState[i] += (m * z - m + 1) * adj;

          // df/d(xW_r) ...
          float dfdxW_r = m * r * (1 - r) * t * adj;
          if(final)
            dfdxW_r *= rowSU[l] + b[l];
          else
            dfdxW_r *= rowSU[l];
          if(outXW)
            rowOutXW[i] += dfdxW_r;
          if(outSU)
            rowOutSU[i] += dfdxW_r;
          if(outB)
            atomicAdd(outB + i, dfdxW_r);

          // df/d(xW_z) ...
          float dfdxW_z = m * (1 - z) * z * (rowState[i] - h) * adj;
          if(outXW)
            rowOutXW[k] += dfdxW_z;
          if(outSU)
            rowOutSU[k] += dfdxW_z;
          if(outB)
            atomicAdd(outB + k, dfdxW_z);

          // df/d(xW_x) ...
          float dfdxW_x = m * t * adj;
          if(outXW)
            rowOutXW[l] += dfdxW_x;
          if(outSU)
            rowOutSU[l] += dfdxW_x * r;
          if(outB)
            if(final)
              atomicAdd(outB + l, dfdxW_x * r);
            else
              atomicAdd(outB + l, dfdxW_x);
        }
      }
    }
  }
}

void GRUFastBackward(std::vector<Tensor> outputs,
                     std::vector<Tensor> inputs,
                     Tensor adj,
                     bool final) {
  matchOrAbort<float>(outputs[0]->type());
  cudaSetDevice(adj->getDeviceId().no);

  int rows = adj->shape().elements() / adj->shape().back();
  int cols = adj->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gGRUFastBackward<<<blocks, threads>>>(
      outputs[0] ? outputs[0]->data() : 0,        // state - adj
      outputs[1] ? outputs[1]->data() : 0,        // xW - adj
      outputs[2] ? outputs[2]->data() : 0,        // sU - adj
      outputs[3] ? outputs[3]->data() : 0,        // b - adj
      inputs[0]->data(),                          // state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      adj->data(),
      rows,
      cols,
      final);
}

__global__ void gCrossEntropyPick(float* out,
                                  const functional::Shape outShape,
                                  const float* in,
                                  const functional::Shape inShape,
                                  const IndexType* pick) {
  int rows = inShape.elements() / inShape.back();
  int cols = inShape.back();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* sp = in + j * cols;

      extern __shared__ float _share[];
      float* _max = _share + blockDim.x;

      _max[threadIdx.x] = sp[threadIdx.x];
      for(int tid = 1; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += __expf(sp[id] - max);
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();

      // cross-entropy
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id == (int)pick[j]) {
          out[j] = __logf(_sum[0]) - sp[id] + max;
        }
      }
    }
  }
}

// In each j-th row, take the corresponding j-th label index i from indices and compute:
// For each vocabulary item v, the only non-zero element in a row in the sum is the item
// that matches the label indexed by i (the picked element).
// C = sum_{v in V}(-logsoftmax(A) * delta(v, i) = -logsoftmax(A)[i]
void CrossEntropyPick(Tensor out, Tensor in, Tensor indices) {
  matchOrAbort<float>(out->type());
  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  int rows = in->shape().elements() / in->shape().back();
  int cols = in->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  int shared = sizeof(float) * threads * 2;

  gCrossEntropyPick<<<blocks, threads, shared>>>(
      out->data(), out->shape(), in->data(), in->shape(), indices->data<IndexType>());
}

__global__ void gCrossEntropyPickBackward(float* out,
                                          const functional::Shape outShape,
                                          const float* adj,
                                          const float* in,
                                          const IndexType* pick) {
  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* sp = in + j * cols;
      float* so = out + j * cols;

      extern __shared__ float _share[];
      float* _max = _share + blockDim.x;

      _max[threadIdx.x] = sp[threadIdx.x];
      for(int tid = 1; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = __expf(sp[id] - max);
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();

      // cross-entropy
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float sub = (float)(id == (int)pick[j]);
          so[id] += adj[j] * (__expf(sp[id] - max) / _sum[0] - sub);
        }
      }
    }
  }
}

void CrossEntropyPickBackward(Tensor out, Tensor adj, Tensor a, Tensor indices) {
  matchOrAbort<float>(out->type());
  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  int shared = sizeof(float) * threads * 2;

  gCrossEntropyPickBackward<<<blocks, threads, shared>>>(
      out->data(), out->shape(), adj->data(), a->data(), indices->data<IndexType>());
}

float L2Norm(Tensor in) {
  cudaSetDevice(in->getDeviceId().no);

  int size = in->shape().elements();
  int threads = std::min(MAX_THREADS, size);
  int blocks = std::min(MAX_BLOCKS, size / threads + (size % threads != 0));

  uint8_t* data;
  cudaMalloc(&data, blocks * sizeof(float));
  auto out = TensorBase::New(MemoryPiece::New(data, blocks * sizeof(float)),
                             Shape({1, blocks}),
                             in->getBackend());

  using namespace functional;
  ReduceAll(_1 * _1, out, in);
  float dataCpu = sqrtf(out->get(0));
  out.reset();
  cudaFree(data);
  return dataCpu;
}

__global__ void gAtt(float* out,
                     const float* va,
                     const float* ctx,
                     const float* state,
                     int m,  // total rows (batch x time x beam)
                     int k,  // depth
                     int b,  // batch size
                     int t   // time of ctx
) {
  int rows = m;
  int cols = k;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* vaRow = va;
      const float* ctxRow = ctx + (j % (b * t)) * cols;
      const float* stateRow = state + ((j / (b * t)) * b + j % b) * cols;

      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float z = ctxRow[id] + stateRow[id];
          float ex = tanhf(z) * vaRow[id];
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      out[j] = _sum[0];
      __syncthreads();
    }
  }
}

void Att(Tensor out, Tensor va, Tensor context, Tensor state) {
  matchOrAbort<float>(out->type());
  cudaSetDevice(out->getDeviceId().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t k = context->shape()[-1];
  size_t b = context->shape()[-2];
  size_t t = context->shape()[-3];

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads * 2;

  gAtt<<<blocks, threads, shared>>>(
      out->data(), va->data(), context->data(), state->data(), m, k, b, t);
}

__global__ void gAttBack(float* gVa,
                         float* gContext,
                         float* gState,
                         const float* va,
                         const float* context,
                         const float* state,
                         const float* adj,
                         int m,  // rows
                         int k,  // cols
                         int n   // batch size
) {
  int rows = m;
  int cols = k;
  for(int bid = 0; bid < m; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* gcRow = gContext + j * cols;
      float* gsRow = gState + (j % n) * cols;

      const float* cRow = context + j * cols;
      const float* sRow = state + (j % n) * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float z = cRow[id] + sRow[id];

          float t = tanhf(z);
          float r = va[id] * (1.f - t * t);

          gcRow[id] += r * adj[j];
          gsRow[id] += r * adj[j];
          atomicAdd(gVa + id, t * adj[j]);
        }
      }
    }
  }
}

void AttBack(Tensor gVa,
             Tensor gContext,
             Tensor gState,
             Tensor va,
             Tensor context,
             Tensor state,
             Tensor adj) {
  matchOrAbort<float>(gVa->type());
  cudaSetDevice(adj->getDeviceId().no);

  size_t m = adj->shape().elements() / adj->shape()[-1];
  size_t k = context->shape()[-1];
  size_t n = context->shape()[-2];

  int blocks = std::min(MAX_BLOCKS, (int)n);
  int threads = std::min(MAX_THREADS, (int)k);

  gAttBack<<<blocks, threads>>>(gVa->data(),
                                gContext->data(),
                                gState->data(),

                                va->data(),
                                context->data(),
                                state->data(),

                                adj->data(),
                                m,
                                k,
                                n);
}

template <typename T, typename AccType = float>
__global__ void gLNormalization(T* out,
                                const T* in,
                                const T* alpha,
                                const T* beta,
                                int rows,
                                int cols,
                                AccType eps = 1e-9) {
  extern __shared__ uint8_t _sharedBytes[];
  AccType* _shareFloat = (AccType*)_sharedBytes;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* so = out + j * cols;
      const T* sp = in + j * cols;

      AccType* _sum = _shareFloat + blockDim.x; // accumulate into floats
      _sum[threadIdx.x] = 0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += (AccType)sp[id];
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
      AccType mean = (T)(_sum[0] / (AccType)cols);
      __syncthreads();

      AccType* _sqSum = _shareFloat + blockDim.x;

      _sqSum[threadIdx.x] = 0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType ex = ((AccType)sp[id]) - mean;
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      AccType sigma = functional::Ops<AccType>::sqrt((AccType)eps + (_sqSum[0] / (AccType)cols)); // all AccType
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          T t = alpha[id] * ((sp[id] - (T)mean) / (T)sigma);
          if(beta != nullptr)
            t += beta[id];
          so[id] = t;
        }
      }
    }
  }
}

void LayerNormalization(Tensor out,
                        Tensor in,
                        Tensor gamma,
                        Tensor beta,
                        float eps) {
  cudaSetDevice(out->getDeviceId().no);

  int rows = in->shape().elements() / in->shape().back();
  int cols = in->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  int shared = 2 * threads * sizeof(float);

  if(out->type() == Type::float32) {
    gLNormalization<float, float><<<blocks, threads, shared>>>(out->data<float>(),
                                                 in->data<float>(),
                                                 gamma->data<float>(),
                                                 beta ? beta->data<float>() : nullptr,
                                                 rows,
                                                 cols,
                                                 eps);
  } else if (out->type() == Type::float16) {
    gLNormalization<half, float><<<blocks, threads, shared>>>(out->data<half>(),
                                                 in->data<half>(),
                                                 gamma->data<half>(),
                                                 beta ? beta->data<half>() : nullptr,
                                                 rows,
                                                 cols,
                                                 eps);

  } else {
    ABORT("LayerNormaliztion not implemented for type {}", out->type());
  }
}

template <typename T, typename AccType = float>
__global__ void gLayerNormalizationGrad(T* gradX,
                                        T* gradGamma,
                                        T* gradBeta,
                                        T* adj,
                                        T* y,
                                        T* x,
                                        T* gamma,
                                        T* beta,
                                        int rows,
                                        int cols,
                                        AccType eps = 1e-9) {
  extern __shared__ uint8_t sharedBytes[];
  AccType* shared = (AccType*)sharedBytes;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      AccType* sum_adj = shared;
      AccType* sum_adj_x = shared + blockDim.x;
      AccType* sum_x = shared + 2 * blockDim.x;
      AccType* sum_sqr = shared + 3 * blockDim.x;

      const T* xRow = x + j * cols;
      const T* yRow = y + j * cols;
      const T* adjRow = adj + j * cols;
      T* gradXRow = gradX + j * cols;

      sum_x[threadIdx.x] = (AccType)0.0f;
      sum_adj[threadIdx.x] = (AccType)0.0f;
      sum_adj_x[threadIdx.x] = (AccType)0.0f;
      sum_sqr[threadIdx.x] = (AccType)0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          sum_x[threadIdx.x] += (AccType)xRow[id];
          sum_adj_x[threadIdx.x]
              += (AccType)(adjRow[id] * (yRow[id] - ((beta) ? beta[id] : (T)0.f)) / gamma[id]);
          sum_adj[threadIdx.x] += (AccType)adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          sum_x[threadIdx.x] += sum_x[threadIdx.x + skip]; // Accumulates in AccType
          sum_adj[threadIdx.x] += sum_adj[threadIdx.x + skip]; // Accumulates in AccType
          sum_adj_x[threadIdx.x] += sum_adj_x[threadIdx.x + skip]; // Accumulates in AccType
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      AccType mean = sum_x[0] / (AccType)cols;
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType ex = (AccType)xRow[id] - mean;
          sum_sqr[threadIdx.x] += ex * ex;
        }
      }

      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          sum_sqr[threadIdx.x] += sum_sqr[threadIdx.x + skip]; // Accumulates in AccType
        len = (len + 1) >> 1;
      }
      __syncthreads();
      AccType sigma = functional::Ops<AccType>::sqrt(eps + (sum_sqr[0] / (AccType)cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType grad_x = (AccType)0.0f;
          AccType x_hat = (yRow[id] - ((beta) ? beta[id] : (T)0.f)) / gamma[id];
          grad_x += (AccType)cols * (AccType)adjRow[id];
          grad_x -= sum_adj[0];
          grad_x -= sum_adj_x[0] * x_hat;
          grad_x /= ((AccType)cols * sigma);

          AccType valX = (AccType)gamma[id] * grad_x;

          // @TODO: What is this? Some kind of clipping? Did I add this?
          AccType sign = functional::Ops<AccType>::sgn(valX);
          valX = functional::Ops<AccType>::abs(valX) > (AccType)1000.f ? sign * (AccType)1000.f : valX;

          gradXRow[id] += (T)valX;
          atomics::atomicAdd(gradGamma + id, adjRow[id] * (T)x_hat);
          if(beta) {
            atomics::atomicAdd(gradBeta + id, adjRow[id]);
          }
        }
      }
    }
  }
}

void LayerNormalizationGrad(Tensor gradX,
                            Tensor gradGamma,
                            Tensor gradBeta,
                            Tensor adj,
                            Tensor y,
                            Tensor x,
                            Tensor gamma,
                            Tensor beta,
                            float eps) {
  cudaSetDevice(adj->getDeviceId().no);
  int rows = y->shape().elements() / y->shape()[-1];
  int cols = y->shape()[-1];

  int threads = std::min(MAX_THREADS, cols);
  int blocks = std::min(MAX_BLOCKS, rows);
  int shared = sizeof(float) * threads * 4;

  if(gradX->type() == Type::float32) {
    gLayerNormalizationGrad<float, float><<<blocks, threads, shared>>>(
      gradX->data<float>(),
      gradGamma->data<float>(),
      (gradBeta) ? gradBeta->data<float>() : nullptr,
      adj->data<float>(),
      y->data<float>(),
      x->data<float>(),
      gamma->data<float>(),
      (beta) ? beta->data<float>() : nullptr,
      rows,
      cols,
      eps);
  } else if (gradX->type() == Type::float16) {
    // accumulate in float
    gLayerNormalizationGrad<half, float><<<blocks, threads, shared>>>(
      gradX->data<half>(),
      gradGamma->data<half>(),
      (gradBeta) ? gradBeta->data<half>() : nullptr,
      adj->data<half>(),
      y->data<half>(),
      x->data<half>(),
      gamma->data<half>(),
      (beta) ? beta->data<half>() : nullptr,
      rows,
      cols,
      eps);
  } else {
    ABORT("LayerNormaliztionGrad not implemented for type {}", gradX->type());
  }
}

template <bool add, typename T>
__global__ void gShift(T* out,
                       const T* in,
                       int length,
                       int offset,
                       float padValue) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      if(add) {
        if(index - offset >= 0 && index - offset < length)
          out[index] += in[index - offset];
      } else {
        if(index - offset < 0 || index - offset >= length)
          out[index] = (T)padValue;
        else
          out[index] = in[index - offset];
      }
    }
  }
}

void Shift(Tensor out,
           Tensor in,
           marian::Shape shift,
           float padValue,
           bool invert) {
  matchOrAbort<float>(out->type());
  ABORT_IF(in->shape().size() != shift.size(), "bad dimensions");

  // BUGBUG: This can only shift along the first axis. Shifting, e.g., along the
  // last axis cannot be implemented this way.
  int offset = 0;
  for(int i = 0; i < shift.size(); ++i)
    offset += in->shape().stride(i) * shift[i];

  if(invert)
    offset = -offset;

  cudaSetDevice(out->getDeviceId().no);

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  if(out->type() == Type::float32) {
    gShift<false>
        <<<blocks, threads>>>(out->data<float>(), in->data<float>(), length, offset, padValue);
  } else if() {
    gShift<false>
        <<<blocks, threads>>>(out->data<half>(), in->data<half>(), length, offset, padValue);
  } else {
    ABORT("Shift not implemented for type {}", out->type());
  }
}

void ShiftGrad(Tensor out, Tensor in, marian::Shape shift, bool invert) {
  matchOrAbort<float>(out->type());
  ABORT_IF(in->shape().size() != shift.size(), "bad dimensions");

  // BUGBUG: This can only shift along the first axis. Shifting, e.g., along the
  // last axis cannot be implemented this way.
  int offset = 0;
  for(int i = 0; i < shift.size(); ++i)
    offset += in->shape().stride(i) * shift[i];

  if(invert)
    offset = -offset;

  cudaSetDevice(out->getDeviceId().no);

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  if(out->type() == Type::float32) {
    gShift<true>
        <<<blocks, threads>>>(out->data<float>(), in->data<float>(), length, offset, 0.f); // @TODO: What about padValue?
  } else if() {
    gShift<true>
        <<<blocks, threads>>>(out->data<half>(), in->data<half>(), length, offset, 0.f);
  } else {
    ABORT("Shift not implemented for type {}", out->type());
  }
}

__global__ void gSetSparse(float* out,
                           const size_t* indices,
                           const float* values,
                           int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out[indices[index]] = values[index];
    }
  }
}

void SetSparse(float* out,
               const std::vector<size_t>& indices,
               const std::vector<float>& values) {
  int length = indices.size();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  size_t* d_indices;
  CUDA_CHECK(cudaMalloc(&d_indices, length * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indices,
                        indices.data(),
                        length * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  float* d_values;
  CUDA_CHECK(cudaMalloc(&d_values, length * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(
      d_values, values.data(), length * sizeof(float), cudaMemcpyHostToDevice));

  gSetSparse<<<blocks, threads>>>(out, d_indices, d_values, length);

  cudaFree(d_indices);
  cudaFree(d_values);
}

/******************************************************************************/

__global__ void gLSTMCellForward(float* out,
                                 const float* cell,
                                 const float* xW,
                                 const float* sU,
                                 const float* b,
                                 const float* mask,
                                 size_t rows,
                                 size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];

      float* rowOut = out + j * cols;
      const float* rowCell = cell + j * cols;

      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float gf = stableSigmoid(xWrow[i] + sUrow[i] + b[i]);

          int k = i + cols;
          float gi = stableSigmoid(xWrow[k] + sUrow[k] + b[k]);

          int l = i + 2 * cols;
          float gc = tanhf(xWrow[l] + sUrow[l] + b[l]);

          float cout = gf * rowCell[i] + gi * gc;
          rowOut[i] = m * cout + (1 - m) * rowCell[i];
        }
      }
    }
  }
}

void LSTMCellForward(Tensor out, std::vector<Tensor> inputs) {
  matchOrAbort<float>(out->type());
  cudaSetDevice(out->getDeviceId().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMCellForward<<<blocks, threads>>>(
      out->data(),                                // output
      inputs[0]->data(),                          // cell state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      rows,
      cols);
}

__global__ void gLSTMOutputForward(float* out,
                                   const float* cell,
                                   const float* xW,
                                   const float* sU,
                                   const float* b,
                                   size_t rows,
                                   size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowCell = cell + j * cols;

      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          int k = i + 3 * cols;
          float go = stableSigmoid(xWrow[k] + sUrow[k] + b[k]);

          rowOut[i] = go * tanhf(rowCell[i]);
        }
      }
    }
  }
}

void LSTMOutputForward(Tensor out, std::vector<Tensor> inputs) {
  matchOrAbort<float>(out->type());
  cudaSetDevice(out->getDeviceId().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMOutputForward<<<blocks, threads>>>(out->data(),        // output
                                          inputs[0]->data(),  // cell state
                                          inputs[1]->data(),  // xW
                                          inputs[2]->data(),  // sU
                                          inputs[3]->data(),  // b
                                          rows,
                                          cols);
}

__global__ void gLSTMCellBackward(float* outCell,
                                  float* outXW,
                                  float* outSU,
                                  float* outB,
                                  const float* cell,
                                  const float* xW,
                                  const float* sU,
                                  const float* b,
                                  const float* mask,
                                  const float* adj,
                                  size_t rows,
                                  size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];

      float* rowOutCell = outCell + j * cols;
      float* rowOutXW = outXW + j * cols * 4;
      float* rowOutSU = outSU + j * cols * 4;

      const float* rowCell = cell + j * cols;
      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      const float* rowAdj = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float gf = stableSigmoid(xWrow[i] + sUrow[i] + b[i]);

          int k = i + cols;
          float gi = stableSigmoid(xWrow[k] + sUrow[k] + b[k]);

          int l = i + 2 * cols;
          float gc = tanhf(xWrow[l] + sUrow[l] + b[l]);

          float adj = rowAdj[i];

          // dc/dc_{t-1}
          if(outCell)
            rowOutCell[i] += (m * gf - m + 1) * adj;

          // dc/d(b_f) = dc/d(xW_f) ...
          float dcdxf = m * rowCell[i] * gf * (1 - gf) * adj;
          if(outXW)
            rowOutXW[i] += dcdxf;
          if(outSU)
            rowOutSU[i] += dcdxf;
          if(outB)
            atomicAdd(outB + i, dcdxf);

          // dc/d(b_i) ...
          float dcdb_i = m * gc * gi * (1 - gi) * adj;
          if(outXW)
            rowOutXW[k] += dcdb_i;
          if(outSU)
            rowOutSU[k] += dcdb_i;
          if(outB)
            atomicAdd(outB + k, dcdb_i);

          // dc/d(b_c) ...
          float dcdxc = m * gi * (1 - gc * gc) * adj;
          if(outXW)
            rowOutXW[l] += dcdxc;
          if(outSU)
            rowOutSU[l] += dcdxc;
          if(outB)
            atomicAdd(outB + l, dcdxc);
        }
      }
    }
  }
}

void LSTMCellBackward(std::vector<Tensor> outputs,
                      std::vector<Tensor> inputs,
                      Tensor adj) {
  matchOrAbort<float>(outputs[0]->type());
  cudaSetDevice(adj->getDeviceId().no);

  int rows = adj->shape().elements() / adj->shape().back();
  int cols = adj->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMCellBackward<<<blocks, threads>>>(
      outputs[0] ? outputs[0]->data() : 0,        // state - adj
      outputs[1] ? outputs[1]->data() : 0,        // xW - adj
      outputs[2] ? outputs[2]->data() : 0,        // sU - adj
      outputs[3] ? outputs[3]->data() : 0,        // b - adj
      inputs[0]->data(),                          // state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      adj->data(),
      rows,
      cols);
}

__global__ void gLSTMOutputBackward(float* outCell,
                                    float* outXW,
                                    float* outSU,
                                    float* outB,
                                    const float* cell,
                                    const float* xW,
                                    const float* sU,
                                    const float* b,
                                    const float* adj,
                                    size_t rows,
                                    size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOutCell = outCell + j * cols;
      float* rowOutXW = outXW + j * cols * 4;
      float* rowOutSU = outSU + j * cols * 4;

      const float* rowCell = cell + j * cols;
      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      const float* rowAdj = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          int k = i + 3 * cols;
          float go = stableSigmoid(xWrow[k] + sUrow[k] + b[k]);

          float t = tanhf(rowCell[i]);

          float adj = rowAdj[i];

          // dc/dc_{t-1}
          if(outCell)
            rowOutCell[i] += go * (1 - t * t) * adj;

          // dc/d(b_o) = dc/d(xW_f) ...
          float dcdxo = t * go * (1 - go) * adj;
          if(outXW)
            rowOutXW[k] += dcdxo;
          if(outSU)
            rowOutSU[k] += dcdxo;
          if(outB)
            atomicAdd(outB + k, dcdxo);
        }
      }
    }
  }
}

void LSTMOutputBackward(std::vector<Tensor> outputs,
                        std::vector<Tensor> inputs,
                        Tensor adj) {
  matchOrAbort<float>(outputs[0]->type());
  cudaSetDevice(adj->getDeviceId().no);

  int rows = adj->shape().elements() / adj->shape().back();
  int cols = adj->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMOutputBackward<<<blocks, threads>>>(
      outputs[0] ? outputs[0]->data() : 0,  // state - adj
      outputs[1] ? outputs[1]->data() : 0,  // xW - adj
      outputs[2] ? outputs[2]->data() : 0,  // sU - adj
      outputs[3] ? outputs[3]->data() : 0,  // b - adj
      inputs[0]->data(),                    // state
      inputs[1]->data(),                    // xW
      inputs[2]->data(),                    // sU
      inputs[3]->data(),                    // b
      adj->data(),
      rows,
      cols);
}

template <typename T>
__global__ void gHighwayForward(T* out,
                                const T* in1,
                                const T* in2,
                                const T* t,
                                size_t length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      T sigma = functional::Ops<T>::sigmoid(t[index]);
      out[index] = in1[index] * sigma + in2[index] * ((T)1.f - sigma);
    }
  }
}

void HighwayForward(Tensor out,
                    const Tensor in1,
                    const Tensor in2,
                    const Tensor t) {
  cudaSetDevice(out->getDeviceId().no);

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  if(out->type() == Type::float32) {
    gHighwayForward<<<blocks, threads>>>(
        out->data<float>(), in1->data<float>(), in2->data<float>(), t->data<float>(), length);
  } else if(out->type() == Type::float16) {
    gHighwayForward<<<blocks, threads>>>(
        out->data<half>(), in1->data<half>(), in2->data<half>(), t->data<half>(), length);
  } else {
    ABORT("HighwayForward not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gHighwayBackward(T* out1,
                                 T* out2,
                                 T* outt,
                                 const T* in1,
                                 const T* in2,
                                 const T* t,
                                 const T* adj,
                                 size_t length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      T sigma = functional::Ops<T>::sigmoid(t[index]);
      out1[index] = sigma * adj[index];
      out2[index] = ((T)1.f - sigma) * adj[index];
      outt[index]
          = sigma * ((T)1.f - sigma) * (in1[index] - in2[index]) * adj[index];
    }
  }
}

void HighwayBackward(Tensor out1,
                     Tensor out2,
                     Tensor outt,
                     const Tensor in1,
                     const Tensor in2,
                     const Tensor t,
                     const Tensor adj) {
  cudaSetDevice(out1->getDeviceId().no);

  int length = out1->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  if(out1->type() == Type::float32) {
    gHighwayBackward<<<blocks, threads>>>(out1->data<float>(),
                                          out2->data<float>(),
                                          outt->data<float>(),
                                          in1->data<float>(),
                                          in2->data<float>(),
                                          t->data<float>(),
                                          adj->data<float>(),
                                          length); 
  } else if(out1->type() == Type::float16) {
    gHighwayBackward<<<blocks, threads>>>(out1->data<half>(),
                                          out2->data<half>(),
                                          outt->data<half>(),
                                          in1->data<half>(),
                                          in2->data<half>(),
                                          t->data<half>(),
                                          adj->data<half>(),
                                          length); 
  } else {
    ABORT("HighwayForward not implemented for type {}", out1->type());
  }
}

__global__ void gMaxPoolingForward(float* out,
                                   int outRows,
                                   int outCols,
                                   float* in,
                                   int inRows,
                                   int inCols,
                                   float* mask,
                                   int numKernels,
                                   int maskCols,
                                   int width,
                                   int lastWidth) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid >= outRows * outCols)
    return;

  int rowId = tid / outRows;
  int colId = tid % outRows;

  float* b = in + (rowId * inCols) + (colId * width);
  float* localMask = mask + (rowId / numKernels) * maskCols + colId * width;

  if(colId == outRows - 1) {
    width = lastWidth;
  }

  float currentMax = b[0] * localMask[0];
  for(int i = 1; i < width; ++i) {
    if(b[i] * localMask[i] > currentMax) {
      currentMax = b[i] * localMask[i];
    }
  }

  out[rowId + (colId * outCols)] = currentMax;
}

void PoolingWithMaskingForward(Tensor out,
                               Tensor in,
                               Tensor mask,
                               int width,
                               bool isEven) {
  matchOrAbort<float>(out->type());
  int n = out->shape().elements();
  int threads = std::min(n, MAX_THREADS);
  int blocks = n / threads + (n % threads != 0);

  auto& inShape = in->shape();
  int inRows = inShape[0] * inShape[1];
  int inCols = inShape[2];

  auto& outShape = out->shape();
  int outRows = outShape[2];
  int outCols = outShape[0] * outShape[1];

  int lastWidth
      = ((inCols - isEven) % width == 0) ? width : (inCols - isEven) % width;

  gMaxPoolingForward<<<blocks, threads>>>(out->data(),
                                          outRows,
                                          outCols,
                                          in->data(),
                                          inRows,
                                          inCols,
                                          mask->data(),
                                          outShape[1],
                                          mask->shape()[2],
                                          width,
                                          lastWidth);
}

__global__ void gMaxPoolingBackward(float* adj,
                                    int adjRows,
                                    int adjCols,
                                    float* in,
                                    float* adjIn,
                                    int inRows,
                                    int inCols,
                                    float* mask,
                                    int numKernels,
                                    int maskCols,
                                    int width,
                                    int lastWidth) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid >= adjRows * adjCols)
    return;

  int rowId = tid / adjRows;
  int colId = tid % adjRows;

  float* b = in + (rowId * inCols) + (colId * width);

  if(colId == adjRows - 1) {
    width = lastWidth;
  }

  float* localMask = mask + (rowId / numKernels) * maskCols + colId * width;
  size_t currentMaxIdx = 0;
  for(int i = 1; i < width; ++i) {
    if(b[i] * localMask[i] > b[currentMaxIdx] * localMask[currentMaxIdx]) {
      currentMaxIdx = i;
    }
  }

  adjIn[(rowId * inCols) + (colId * width) + currentMaxIdx]
      += adj[rowId + (colId * adjCols)];
}

void PoolingWithMaskingBackward(Tensor adj,
                                Tensor adjIn,
                                Tensor in,
                                Tensor mask,
                                int width,
                                bool isEven) {
  matchOrAbort<float>(adj->type());
  int n = adj->shape().elements();
  int threads = std::min(n, 512);
  int blocks = n / threads + (n % threads != 0);

  auto& inShape = in->shape();
  int inRows = inShape[0] * inShape[1];
  int inCols = inShape[2];

  auto& adjShape = adj->shape();
  int adjRows = adjShape[2];
  int adjCols = adjShape[0] * adjShape[1];

  int lastWidth
      = ((inCols - isEven) % width == 0) ? width : (inCols - isEven) % width;

  gMaxPoolingBackward<<<blocks, threads>>>(adj->data(),
                                           adjRows,
                                           adjCols,
                                           in->data(),
                                           adjIn->data(),
                                           inRows,
                                           inCols,
                                           mask->data(),
                                           adjShape[1],
                                           mask->shape()[2],
                                           width,
                                           lastWidth);
}
}  // namespace gpu
}  // namespace marian
