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
#include "cub/cub/cub.cuh"
#include "cub/cub/util_type.cuh"

#define MAX_BLOCKS_PER_BEAM 8

struct TopK {
  int p = 0;
  float u = cub::FpLimits<float>::Lowest();

  __device__ __forceinline__ void insert(float elem, int elem_id) {
    if(elem > u) {
      u = elem;
      p = elem_id;
    }
  }

  __device__ __forceinline__ void init() {
    u = cub::FpLimits<float>::Lowest();
    p = 0;
  }
};

__device__ __forceinline__ TopK reduce_topk_op(const TopK& a, const TopK& b) {
  return a.u > b.u ? a : b;
}

template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_1(T* log_probs, 
                             int* topk_tmp_id_buf,
                             float* topk_tmp_val_buf, 
                             const int k, 
                             const int vocab_size) {

  typedef cub::BlockReduce<TopK, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int row_id = bid / BLOCKS_PER_BEAM_; // row id for log_probs
  const int block_lane = bid % BLOCKS_PER_BEAM_; // block id for a beam 
  const int tmp_log_buf_index = row_id * vocab_size; 
  const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
  TopK partial;

  for(int ite = 0; ite < k; ite++) {
    partial.init();
    #pragma unroll
    for(int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
      int index = elem_id + tmp_log_buf_index;
      partial.insert(log_probs[index], index);
    }

    TopK total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op);

    if (tid == 0) {
      const int index = tmp_topk_buf_index + ite;
      topk_tmp_id_buf[index] = total.p;
      topk_tmp_val_buf[index] = total.u;
      log_probs[total.p] = cub::FpLimits<T>::Lowest();
    }
    __syncthreads();
  }

  // Update prob array with original values.
  for(int beam = tid; beam < k; beam += BLOCK_SIZE_) {
    const int index = tmp_topk_buf_index + beam;
    log_probs[topk_tmp_id_buf[index]] = (T)topk_tmp_val_buf[index];
  }
}

template<int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_2(const int* __restrict topk_tmp_id_buf,
                             float* topk_tmp_val_buf,
                             TopK* top,
                             const int beams_per_batch,
                             const int k) {

  const int size = beams_per_batch * k * BLOCKS_PER_BEAM_; 
  const int tid = threadIdx.x;
  const int batch_id = blockIdx.x;

  typedef cub::BlockReduce<TopK, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  extern __shared__ char array[];
  float *s_val = topk_tmp_val_buf + batch_id * size;
  TopK *topks = (TopK*)(array);
  
  TopK partial;

  for(int ite = 0; ite < k; ite++) {
    partial.init();
    #pragma unroll
    for(int i = tid; i < size; i+= BLOCK_SIZE_) {
      partial.insert(s_val[i], i); 
    }

    TopK total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op);

    if(tid == 0) {
      topks[ite] = total;
      s_val[total.p] = cub::FpLimits<float>::Lowest();
    }
    __syncthreads();
  }

  for(int beam = tid; beam < k; beam += BLOCK_SIZE_) {
    TopK beamOut; 
    beamOut.p = topk_tmp_id_buf[batch_id * size + topks[beam].p];
    beamOut.u = topks[beam].u;
    top[batch_id * k + beam] = beamOut;
  }
}

#define CASE_K(K,BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_) \
  case K: \
    topk_stage_1<T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_><<<batch_size * beams_per_batch * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>( \
        log_probs, \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        k, vocab_size); \
    topk_stage_2<BLOCK_SIZE_2_, BLOCKS_PER_BEAM_><<<batch_size, BLOCK_SIZE_2_, K * sizeof(TopK), stream>>>( \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        tops, \
        beams_per_batch, \
        k); \
  break; \

template <typename T>
void topK_kernelLauncher(T* log_probs,
                         int* topk_tmp_id_buf,
                         float* topk_tmp_val_buf,
                         TopK* tops,
                         const int batch_size,
                         const int beams_per_batch,
                         const int k,
                         const int vocab_size,
                         cudaStream_t stream) {
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
      topk_stage_1<T, 128, 1><<<batch_size * beams_per_batch * 1, 128, 0, stream>>>(log_probs, 
                                                                                    topk_tmp_id_buf, 
                                                                                    topk_tmp_val_buf,
                                                                                    k, 
                                                                                    vocab_size);

      topk_stage_2<128, 1><<<batch_size, 128, k * sizeof(TopK), stream>>>(topk_tmp_id_buf,
                                                                                           topk_tmp_val_buf,
                                                                                           tops,
                                                                                           beams_per_batch,
                                                                                           k);
    break;
  }
}