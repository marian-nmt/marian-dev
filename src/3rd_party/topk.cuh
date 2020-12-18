/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* This code is modified from the topk implementation in NVIDIA's faster
* transformer repository. The original source code files can be found here:
*
* https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v3.0/fastertransformer/cuda/topk_kernels.cu
* https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v3.0/fastertransformer/cuda/topk_kernels.cuh
*/

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#if CUDA_VERSION >= 11000
#include <cub/cub.cuh>
#else
#include "cub/cub/cub.cuh"
#endif

#define MAX_BLOCKS_PER_BEAM 8

/// A struct to access the infinity constant on device based on type. 
template<typename T> 
struct FpInfinity;

/// Specialization of FpInfinity for float
template <>
struct FpInfinity<float> {
  static __host__ __device__ __forceinline__ float infinity() {
      return INFINITY;
  }
};

/// Specialization of FpInfinity for half
template <>
struct FpInfinity<__half> {
  static __host__ __device__ __forceinline__ __half infinity() {
      return __float2half(INFINITY);
  }
};

/// A struct used to track the largest value along with the index at which 
/// it occurs at when performing the topk reduction.

/// It is assumed that IndexType is an integral type and T is a floating point
/// type.
template<typename IndexType, typename T>
struct TopK {
  // The index of the largest/smallest value in the large
  IndexType index = 0;
  // The largest/smallest value encountered
  T value = -FpInfinity<T>::infinity();

  /// Updates the value and index in the topk struct if elem is larger than the current
  /// value field of the struct. This is intended to be used during the initial reduction 
  /// before we reduce across a block to ensure all threads in the block have the largest
  /// values within the block's range.
  __device__ __forceinline__ void updateIfLarger(T elem, IndexType elem_id) {
    if (elem > value) {
      value = elem;
      index = elem_id;
    }
  }

  /// Updates the value and index in the topk struct if elem is smaller than the current
  /// value field of the struct. This is intended to be used during the initial reduction 
  /// before we reduce across a block to ensure all threads in the block have the smallest
  /// values within the block's range.
  __device__ __forceinline__ void updateIfSmaller(T elem, IndexType elem_id) {
    if (elem < value) {
      value = elem;
      index = elem_id;
    }
  }

  /// Initializes the value and index fields of the topk struct before starting a reduction.
  /// If the descendingOrder flag is true, the value starts at negative infinity so that we 
  /// store the max values. We do the opposite if descendingOrder is false.
  __device__ __forceinline__ void init(bool descendingOrder) {
    value = descendingOrder ? -FpInfinity<T>::infinity() : FpInfinity<T>::infinity();
    index = 0;
  }
};

template<typename IndexType, typename T>
__device__ __forceinline__ TopK<IndexType, T> reduce_topk_max(const TopK<IndexType, T>& a, const TopK<IndexType, T>& b) {
  return a.value > b.value ? a : b;
}

template<typename IndexType, typename T>
__device__ __forceinline__ TopK<IndexType, T> reduce_topk_min(const TopK<IndexType, T>& a, const TopK<IndexType, T>& b) {
  return a.value < b.value ? a : b;
}

template<typename IndexType, typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_, bool getRowOffsets>
__global__ void topk_stage_1(T* log_probs, 
                             IndexType* topk_tmp_id_buf,
                             T* topk_tmp_val_buf, 
                             const int k, 
                             const int vocab_size,
                             const int descendingOrder) {

  typedef cub::BlockReduce<TopK<IndexType, T>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const IndexType tid = threadIdx.x;
  const IndexType bid = blockIdx.x;
  const IndexType row_id = bid / BLOCKS_PER_BEAM_; // row id for log_probs
  const IndexType block_lane = bid % BLOCKS_PER_BEAM_; // block id for a beam 
  const IndexType tmp_log_buf_index = row_id * vocab_size; 
  const IndexType tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
  TopK<IndexType, T> partial;
  const T minimal = descendingOrder? -FpInfinity<T>::infinity() : FpInfinity<T>::infinity();;

  for(int ite = 0; ite < k; ite++) {
    partial.init(descendingOrder);
    const IndexType threadStart = tid + block_lane * BLOCK_SIZE_;

    // This is needed to ensure the indices for the threads in each valid block starts in a valid range for that block.
    if(threadStart < vocab_size) partial.index = threadStart;
    #pragma unroll
    for(IndexType elem_id = threadStart; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
      IndexType index = elem_id + tmp_log_buf_index;
      descendingOrder? partial.updateIfLarger(log_probs[index], index) : partial.updateIfSmaller(log_probs[index], index);
    }

    TopK<IndexType, T> total = BlockReduce(temp_storage).Reduce(partial, descendingOrder? reduce_topk_max<IndexType, T>: reduce_topk_min<IndexType, T>);

    if (tid == 0) {
      const int index = tmp_topk_buf_index + ite;
      topk_tmp_id_buf[index] = getRowOffsets? total.index - tmp_log_buf_index : total.index;
      topk_tmp_val_buf[index] = total.value;
      // If we found a max, blank out the value in the log prob array before starting the next iteration.
      // Otherwise, we don't need to issue a write since all prob values must have been T::min()
      if(total.value != minimal) log_probs[total.index] = minimal;
    }
    __syncthreads();
  }

  // Update prob array with original values.
  for(int beam = tid; beam < k; beam += BLOCK_SIZE_) {
    const IndexType index = tmp_topk_buf_index + beam;
    T val = topk_tmp_val_buf[index];
    // We only want to replace the value in the log prob array if a value was blanked out (we found a max).
    // When a max isn't found, topk_tmp_val_buf[index] will be T::min()
    if(val != minimal) {
      IndexType k_idx = getRowOffsets? topk_tmp_id_buf[index] + tmp_log_buf_index : topk_tmp_id_buf[index];
      log_probs[k_idx] = (T)topk_tmp_val_buf[index];
    } 
  }
}

template<typename IndexType, typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_2(const IndexType* __restrict topk_tmp_id_buf,
                             T* topk_tmp_val_buf,
                             TopK<IndexType, T>* top,
                             IndexType* outIndices,
                             T* outVals,
                             const int beams_per_batch,
                             const int k,
                             bool descendingOrder) {

  const int size = beams_per_batch * k * BLOCKS_PER_BEAM_; 
  const int tid = threadIdx.x;
  const int batch_id = blockIdx.x;
  const T minimal = descendingOrder? -FpInfinity<T>::infinity() : FpInfinity<T>::infinity();;

  typedef cub::BlockReduce<TopK<IndexType, T>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  extern __shared__ char array[];
  T *s_val = topk_tmp_val_buf + batch_id * size;
  TopK<IndexType, T> *topks = (TopK<IndexType, T>*)(array);
  
  TopK<IndexType, T> partial;

  for(int ite = 0; ite < k; ite++) {
    partial.init(descendingOrder);
    #pragma unroll
    for(IndexType i = tid; i < size; i+= BLOCK_SIZE_) {
      descendingOrder? partial.updateIfLarger(s_val[i], i) : partial.updateIfSmaller(s_val[i], i);
    }

    TopK<IndexType, T> total = BlockReduce(temp_storage).Reduce(partial, descendingOrder? reduce_topk_max<IndexType, T>: reduce_topk_min<IndexType, T>);

    if(tid == 0) {
      topks[ite] = total;
      s_val[total.index] = minimal;
    }
    __syncthreads();
  }

  for(int beam = tid; beam < k; beam += BLOCK_SIZE_) {
    TopK<IndexType, T> beamOut; 
    IndexType indexInTmpValRow = topks[beam].index;
    beamOut.index = topk_tmp_id_buf[batch_id * size + indexInTmpValRow];
    beamOut.value = topks[beam].value;
    if(top) top[batch_id * k + beam] = beamOut;
    if(outIndices) outIndices[batch_id * k + beam] = beamOut.index;
    if(outVals) outVals[batch_id * k + beam] = beamOut.value;
  }
}

#define CASE_K(K,BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_) \
  case K: \
    topk_stage_1<IndexType, T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_, getRowOffsets><<<batch_size * beams_per_batch * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>( \
        log_probs, \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        k, vocab_size, descendingOrder); \
    topk_stage_2<IndexType, T, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_><<<batch_size, BLOCK_SIZE_2_, K * sizeof(TopK<IndexType, T>), stream>>>( \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        tops, \
        outIndices, \
        outVals, \
        beams_per_batch, \
        k, descendingOrder); \
  break; \

// The getRowOffsets template parameter is added so the topk implementation works with both nth_element.cu and the topk operator.
// It is a template parameter since we know at compile time which version of topk we want to call. This flag can be removed whenever nth
// element.cu is removed. When this flag is true, the indices returns are the offsets within the row. When the flag is false, the indices
// returned are offset from the base pointer.
template <typename IndexType, typename T, bool getRowOffsets=false>
void topK_kernelLauncher(T* log_probs,
                         IndexType* topk_tmp_id_buf,
                         T* topk_tmp_val_buf,
                         TopK<IndexType, T>* tops,
                         const int batch_size,
                         const int beams_per_batch,
                         const int k,
                         const int vocab_size,
                         bool descendingOrder,
                         cudaStream_t stream) {
  
  IndexType* outIndices = nullptr;
  T* outVals = nullptr;                        
  switch(k) {
    CASE_K(1,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(2,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(4,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(6,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(8,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(10,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(16,128,128,5);
    CASE_K(32,256,128,1);
    CASE_K(64,256,256,1);
    default:
      topk_stage_1<IndexType, T, 128, 1, getRowOffsets><<<batch_size * beams_per_batch * 1, 128, 0, stream>>>(log_probs, 
                                                                                               topk_tmp_id_buf, 
                                                                                               topk_tmp_val_buf,
                                                                                               k, 
                                                                                               vocab_size, 
                                                                                               descendingOrder);

      topk_stage_2<IndexType, T, 128, 1><<<batch_size, 128, k * sizeof(TopK<IndexType, T>), stream>>>(topk_tmp_id_buf,
                                                                                                      topk_tmp_val_buf,
                                                                                                      tops,
                                                                                                      outIndices,
                                                                                                      outVals,
                                                                                                      beams_per_batch,
                                                                                                      k,
                                                                                                      descendingOrder);
    break;
  }
}

template <typename IndexType, typename T, bool getRowOffsets=false>
void topK_kernelLauncher(T* log_probs,
                         IndexType* topk_tmp_id_buf,
                         T* topk_tmp_val_buf,
                         IndexType* outIndices,
                         T* outVals,
                         const int batch_size,
                         const int beams_per_batch,
                         const int k,
                         const int vocab_size,
                         bool descendingOrder,
                         cudaStream_t stream) {
  
  TopK<IndexType, T>* tops = nullptr;
  switch(k) {
    CASE_K(1,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(2,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(4,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(6,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(8,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(10,128,128,MAX_BLOCKS_PER_BEAM);
    CASE_K(16,128,128,5);
    CASE_K(32,256,128,1);
    CASE_K(64,256,256,1);
    default:
      topk_stage_1<IndexType, T, 128, 1, getRowOffsets><<<batch_size * beams_per_batch * 1, 128, 0, stream>>>(log_probs, 
                                                                                               topk_tmp_id_buf, 
                                                                                               topk_tmp_val_buf,
                                                                                               k, 
                                                                                               vocab_size, 
                                                                                               descendingOrder);

      topk_stage_2<IndexType, T, 128, 1><<<batch_size, 128, k * sizeof(TopK<IndexType, T>), stream>>>(topk_tmp_id_buf,
                                                                                                      topk_tmp_val_buf,
                                                                                                      tops,
                                                                                                      outIndices,
                                                                                                      outVals,
                                                                                                      beams_per_batch,
                                                                                                      k,
                                                                                                      descendingOrder);
    break;
  }
}