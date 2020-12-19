 /* All or part of this file was contributed by NVIDIA under license:
 *   Copyright (C) 2020 NVIDIA Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "common/definitions.h"
#include "tensors/tensor_operators.h"
#include "tensors/gpu/cuda_helpers.h"
#include "tensors/allocator.h"

#include <cuda.h>
#include "3rd_party/topk.cuh"

// GPU implementation of proper Marian top-k operator for TopkNodeOp
// This file contains a lot of code-duplicaton with src/translator/nth_element.cu
// the goal is to replace the beam-search specific topk search with this code. 
// Currently this is only used in the unit tests, but we will move forward and 
// make the beam-search more graph and operator-based.

namespace marian {
namespace gpu {  

void TopK(Tensor outVal, Tensor outInd, Ptr<Allocator> allocator, const Tensor in, int k, int axis, bool descending) {
  
  ABORT_IF(axis != in->shape().size() - 1, "Currently only works for last axis");
  ABORT_IF(!isFloat(in->type()), "Input should be float type and not {}", in->type());
  ABORT_IF(outInd->type() != Type::uint32, "Output should be have type {}", Type::uint32);
  ABORT_IF(outVal->type() != in->type(), "Output should be have type {}", in->type());

  cudaSetDevice(outInd->getDeviceId().no);

  int cols = in->shape()[-1];               // e.g. in beam search that would be [beam * dimVoc]
  int rows = in->shape().elements() / cols; // e.g. in beam search that would be [time * batch]

  ABORT_IF(k > cols, "Cannot select more than {} elements for axis {}", cols, axis);

  const int beams = 1;
  const int tempElts = rows * beams * beams * MAX_BLOCKS_PER_ITEM;

  auto tempMemInd = allocator->alloc<IndexType>(tempElts);

  MemoryPiece::PtrType tempMemVal;
  if(in->type() == Type::float32) {
    tempMemVal = allocator->alloc<float>(tempElts);
    topK_kernelLauncher<IndexType, float, true /*get indices relative to row */>(in->data<float>(), 
                        tempMemInd->data<IndexType>(),
                        tempMemVal->data<float>(),
                        outInd->data<IndexType>(),
                        outVal->data<float>(),
                        rows,
                        1, // This is the beam size. This is set to 1 since we "trick" the existing implementation to treat a row as a beam
                        k,
                        cols,
                        descending,
                        cudaStreamPerThread);
#if COMPILE_FP16
  } else if(in->type() == Type::float16) {
    tempMemVal = allocator->alloc<__half>(tempElts);
    topK_kernelLauncher<IndexType, __half, true /*get indices relative to row */>(in->data<__half>(), 
                        tempMemInd->data<IndexType>(),
                        tempMemVal->data<__half>(),
                        outInd->data<IndexType>(),
                        outVal->data<__half>(),
                        rows,
                        1, // This is the beam size. This is set to 1 since we "trick" the existing implementation to treat a row as a beam
                        k,
                        cols,
                        descending,
                        cudaStreamPerThread);
#endif
  } else {
    ABORT("Topk not implemented for type {}", in->type());
  }

  allocator->free(tempMemInd);
  allocator->free(tempMemVal);
}

}
}
