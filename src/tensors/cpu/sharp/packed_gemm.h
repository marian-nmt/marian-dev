#pragma once

#include "tensors/tensor.h"

namespace marian {
namespace cpu {
namespace variant { // Variants of GEMM implementations

const int PACK16_PADDING = 1024;  // required by sw pipelined kernels
const int PACK16_SPECIALMEM = 256;

void PackInfoFp16(const marian::Shape& shape,
                  const bool transpose,
                  int& nrow,
                  int& ncol,
                  int& kernel_ncol_blocks,
                  int& brow,
                  int& bcol,
                  int& last_brow,
                  int& nbrow,
                  int& nbcol,
                  uint64_t& packsize);

void PackInfoInt8(const marian::Shape& shape,
                  const bool transpose,
                  int& nrow,
                  int& ncol,
                  uint64_t& packsize);

// Pack a matrix into cache utilization efficient way (block format) into fp16
// out: output tensor - packed format
// in: input tensor - normal format
// transpose: the matrix is transposed
// nrow: the number of rows
// ncol: the number of columns
// kernel_ncol_blocks: the number of column blocks
// brow: the number of rows in a block
// bcol: the number of columns in a block
// last_brow: the number of rows in the last block
// nbrow: row index in a block
// nbcol: column index in a block
// packsize: the size of the packed matrix
//          (the number of fp16 elements + padding (1024) + extra temporary memory (256))
void PackFp16(marian::Tensor out,
              const marian::Tensor in,
              const bool transpose,
              const int nrow,
              const int ncol,
              const int kernel_ncol_blocks,
              const int brow,
              const int bcol,
              const int last_brow,
              const int nbrow,
              const int nbcol,
              const uint64_t packsize);

// GEMM operation on the packed B matrix
// C: output matrix
// A: A matrix
// B: B matrix (packed)
// m: the number of rows in A and C
// n: the number of columns in B and C
// transA: transpose of A matrix
// transB: transpose of B matrix
void GemmPackFp16(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const marian::Tensor bias,
                  const size_t m,
                  const size_t n,
                  const size_t k,
                  const int transA = 0,
                  const int transB = 0);

// Pack a matrix into cache utilization efficient way (block format) together with quantization
// 8 bit integers.
// out: output tensor - packed format
// in: input tensor - normal format
// transpose: the matrix is transposed
// nrow: the number of rows
// ncol: the number of columns
// kernel_ncol_blocks: the number of column blocks
// brow: the number of rows in a block
// bcol: the number of columns in a block
// last_brow: the number of rows in the last block
// nbrow: row index in a block
// nbcol: column index in a block
// packsize: the size of the packed matrix
//          (the number of fp16 elements + padding (1024) + extra temporary memory (256))
void PackInt8(marian::Tensor out,
              const marian::Tensor in,
              const bool transpose,
              const int nrow,
              const int ncol,
              const uint64_t packsize);

// GEMM operation on the packed B matrix in 8 bit integers
// C: output matrix
// A: A matrix
// B: B matrix (packed)
// m: the number of rows in A and C
// n: the number of columns in B and C
// transA: transpose of A matrix
// transB: transpose of B matrix
void GemmPackInt8(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const size_t m,
                  const size_t n,
                  const size_t k,
                  const int transA = 0,
                  const int transB = 0);

void AddBias(marian::Tensor C, const marian::Tensor Bias);

}  // namespace variant
}  // namespace cpu
}  // namespace marian
