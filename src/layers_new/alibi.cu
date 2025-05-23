#include "common/types.h"
#include "functional/functional.h"
#include "functional/tensor.h"
#include "tensors/gpu/cuda_helpers.h"

#include <thrust/tuple.h>

namespace marian {

namespace gpu {



template <typename T>
__global__ void gAlibi(
  functional::Tensor<T> out,
  functional::Array<functional::Tensor<T>, 4> inputs,
  int numHeads,
  int start,
  float maskFactor,
  bool addCausalMask,
  int window  // new parameter for the window limit
) {

  constexpr size_t N = functional::Shape::size();
  functional::Array<int, N> oDims;
  int length = out.shape().elements();

  const auto& mask   = inputs[0];
  const auto& slopes = inputs[1];
  const auto& biases = inputs[2];
  const auto& shift  = inputs[3];

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out.shape().dims(index, oDims);

      int beamIdx      = oDims[0];
      int batchHeadIdx = oDims[1];
      int queryIdx     = oDims[2];
      int keyIdx       = oDims[3];

      // [[maybe_unused]] because NVCC seems to have a bug telling me the variable is not referenced when it appears in an intializer; this surpresses the warning.
      [[maybe_unused]] int batchIdx = batchHeadIdx / numHeads;
      [[maybe_unused]] int headIdx  = batchHeadIdx % numHeads;

      int keyPos       = keyIdx;
      int queryPos     = queryIdx + start;

      float relPos   = (float)keyPos - (float)queryPos;

      if(shift.data() != nullptr)
        relPos -= (float)shift[{beamIdx, batchIdx, queryIdx, 0}];

      float slope = (float)slopes[{0, headIdx, 0, 0}];
      float bias  = (float)biases[{0, headIdx, 0, 0}];
      float signedAlibi = relPos + bias;
      float alibi = -abs(slope) * abs(signedAlibi); // range (-inf, 0)

      float binMask = (float)mask[{0, batchIdx, keyIdx, 0}];
      // Only apply the window check if window != 0.
      if(window > 0 && abs(relPos) > window)
        binMask = 0.f; // set binMask to 0 if outside the window

      float logMask = binMask == 0 ? -maskFactor : maskFactor; // range (-maskFactor, maskFactor)

      if(addCausalMask) {
        float causalMask = keyPos > queryPos ? -maskFactor : maskFactor; // range (-maskFactor, maskFactor)
        logMask          = min(logMask, causalMask); // range (-maskFactor, maskFactor) if any mask is set to -maskFactor then the result is -maskFactor
      }

      out[index] = (T)min(logMask, alibi);
    }
  }
}

template <class... Tensors>
void Alibi(int numHeads, int start, bool addCausalMask, Tensor out, Tensors... tensors) {
  cudaSetDevice(out->getDeviceId().no);
  int length = out->size();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  float maskFactor = std::numeric_limits<float>::infinity();

  // Get the ALIBI_WINDOW environment variable (default is 512).
  int window = 0;
  char* envWindow = std::getenv("ALIBI_WINDOW");
  if(envWindow)
    window = std::atoi(envWindow);

  constexpr size_t K = sizeof...(tensors);
  if(out->type() == Type::float32) {
    functional::Array<functional::Tensor<float>, K> inputs = {tensors...};
    gAlibi<float><<<blocks, threads>>>(out, inputs, numHeads, start, maskFactor, addCausalMask, window);
#if COMPILE_FP16
  } else if(out->type() == Type::float16) {
    functional::Array<functional::Tensor<half>, K> inputs = {tensors...};
    gAlibi<half><<<blocks, threads>>>(out, inputs, numHeads, start, maskFactor, addCausalMask, window);
#endif
  } else {
    ABORT("Alibi for type {} not implemented", out->type());
  }
}

// template specialization for h/cpp separation
template void Alibi<marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor >(int, int, bool, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor);

template <typename T>
__global__ void gAlibiGrad(
  functional::Tensor<T> slopesGrad,
  functional::Tensor<T> biasesGrad,
  functional::Array<functional::Tensor<T>, 5> inputs,
  int numHeads,
  int start,
  bool addCausalMask,
  int window            // new parameter for the window limit
) {
  const auto& mask   = inputs[0];
  const auto& slopes = inputs[1];
  const auto& biases = inputs[2];
  const auto& shift  = inputs[3];
  const auto& adj    = inputs[4];

  int cols = adj.size() / numHeads;

  functional::Shape fullShape = adj.shape();
  int dimBeam      = fullShape[0];
  int dimBatchHead = fullShape[1];
  [[maybe_unused]] int dimBatch = dimBatchHead / numHeads;
  int dimQuery     = fullShape[2];
  int dimKeys      = fullShape[3];

  using A5 = functional::Array<int, 5>;
  using S5 = functional::ConstantShape<5>;
  S5 fullShape5(A5({dimBeam, dimBatch, numHeads, dimQuery, dimKeys}));
  S5 headShape5(A5({dimBeam, dimBatch, 1, dimQuery, dimKeys}));

  A5 dims5;
  const int HEAD_DIM = 2;

  // compute single element derivative for slopes and biases
  auto dJ_dxy = [&](int headIdx, int colIdx) -> thrust::tuple<float, float> {
    // get the location for one head
    headShape5.dims(colIdx, dims5);
    dims5[HEAD_DIM] = headIdx;
    int index = fullShape5.index(dims5);
    float vadj = (float)adj[index];

    int beamIdx  = dims5[0];
    int batchIdx = dims5[1];
    int queryIdx = dims5[3];
    int keyIdx   = dims5[4];

    int keyPos   = keyIdx;
    int queryPos = queryIdx + start;

    float relPos = (float)keyPos - (float)queryPos;
    if(shift.data() != nullptr)
      relPos -= (float)shift[{beamIdx, batchIdx, queryIdx, 0}];

    float slope = (float)slopes[{0, headIdx, 0, 0}];
    float bias  = (float)biases[{0, headIdx, 0, 0}];
    float binMask = (float)mask[{0, batchIdx, keyIdx, 0}];

    float signedAlibi = relPos + bias;

    // Incorporate the window: if window is active and the absolute value exceeds window, then gradients are zero.
    if(window > 0 && abs(relPos) > window) {
      binMask = 0.f;
    }

    // If causal masking is enabled.
    if(addCausalMask) {
      float causalMask = keyPos > queryPos ? 0.f : 1.f;
      binMask *= causalMask;
    }

    // compute derivative of slope
    float dslope = binMask * - 1.f * abs(signedAlibi) * vadj;
    if(slope < 0)
      dslope = -dslope;

    // compute derivative of bias
    float dbias  = binMask * -abs(slope) * 1.f * vadj;
    if(signedAlibi < 0)
      dbias = -dbias;

    return { dslope, dbias };
  };

  for(int bid = 0; bid < numHeads; bid += gridDim.x) {
    int headIdx = bid + blockIdx.x;
    if(headIdx < numHeads) {
      // get and assign shared memory
      extern __shared__ uint8_t _sharedBytes[];
      float* _sum = (float*)(_sharedBytes);
      auto sharedSlopes = [_sum](int idx) -> float& { return _sum[2 * idx + 0]; }; // use even indices for slopes
      auto sharedBiases = [_sum](int idx) -> float& { return _sum[2 * idx + 1]; }; // use odd indices for biases

      sharedSlopes(threadIdx.x) = 0.0;
      sharedBiases(threadIdx.x) = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int colIdx = tid + threadIdx.x;
        if(colIdx < cols) {
          float dslopes = 0, dbiases = 0;
          // get the element-wise derivative
          thrust::tie(dslopes, dbiases) = dJ_dxy(headIdx, colIdx);
          // accumulate by thread id
          sharedSlopes(threadIdx.x) += dslopes;
          sharedBiases(threadIdx.x) += dbiases;
        }
      }
      __syncthreads();

      // accumulate here over matrix dimensions, tree reduction
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          sharedSlopes(threadIdx.x) += sharedSlopes(threadIdx.x + skip); // float
          sharedBiases(threadIdx.x) += sharedBiases(threadIdx.x + skip); // float
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();

      // assign accumulated gradients here (preserving existing gradients)
      slopesGrad[headIdx] += (T)sharedSlopes(0);
      biasesGrad[headIdx] += (T)sharedBiases(0);
    }
    __syncthreads();
  }
}

template <typename T, class... Tensors>
void TypedAlibiGrad(int numHeads, int start, bool addCausalMask,
                    Tensor slopesGrad, Tensor biasesGrad, Tensors... tensors) {
  cudaSetDevice(slopesGrad->getDeviceId().no);

  constexpr size_t K = sizeof...(tensors);
  functional::Array<functional::Tensor<T>, K> inputs = {tensors...};

  const auto& adj = inputs[K - 1]; // last one is adjoint and full broadcast shape
  int total = adj.size();

  // we will reduce over each head
  int blocks  = std::min(MAX_BLOCKS,  numHeads);
  int threads = std::min(MAX_THREADS, total / numHeads);
  int shared  = sizeof(float) * threads * 2; // Use float32 as accumulation type, we accumulate slopes and biases

  // Get the ALIBI_WINDOW environment variable (default to 0 if not set)
  int window = 0;
  char* envWindow = std::getenv("ALIBI_WINDOW");
  if(envWindow)
    window = std::atoi(envWindow);

  gAlibiGrad<T><<<blocks, threads, shared>>>(slopesGrad, biasesGrad, inputs, numHeads, start, addCausalMask, window);
}

template <class... Tensors>
void AlibiGrad(int numHeads, int start, bool addCausalMask, Tensor slopesGrad, Tensor biasesGrad, Tensors... tensors) {
  if(slopesGrad->type() == Type::float32) {
    TypedAlibiGrad<float>(numHeads, start, addCausalMask, slopesGrad, biasesGrad, tensors...);
#if COMPILE_FP16
  } else if(slopesGrad->type() == Type::float16) {
    TypedAlibiGrad<half>(numHeads, start, addCausalMask, slopesGrad, biasesGrad, tensors...);
#endif
  } else {
    ABORT("AlibiGrad for type {} not implemented", slopesGrad->type());
  }
}

// template specialization for h/cpp separation
template void AlibiGrad<marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor>(int, int, bool, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor);
}
}
