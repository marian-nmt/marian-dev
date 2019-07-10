#include "packed_gemm.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <cassert>
#include <cstddef>
#include <unordered_map>
//#include <chrono>
#include <limits>

#ifdef _MSC_VER
#pragma warning(disable: 4505) // warning C4505: 'fbgemmAlignedAlloc' in fbgemm.h: unreferenced local function has been removed (missing 'static inline')
#endif

#if USE_FBGEMM
#include "3rd_party/fbgemm/include/fbgemm/FbgemmFP16.h"
#include "3rd_party/fbgemm/include/fbgemm/QuantUtils.h"
#include "3rd_party/fbgemm/include/fbgemm/Fbgemm.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if MKL_FOUND
#include <mkl.h>
#include <mkl_types.h>
#endif

using namespace fbgemm;
#endif // USE_FBGEMM

namespace marian {
namespace cpu {
namespace variant { // Variants of GEMM implementations

#if USE_FBGEMM
// initialize with a dummy
// When this class is instantiated,
// the actual packing operation is happening. If we create this instance every time we call GEMM,
// we are doing packing every time and very slow.
// In Caffe2, the operator is stateful and hold an instance of this.
// But, we don't have any logic for this in marian. We can only cache a tensor (which means a memory chunk).
// So, for now, we keep the packed memory on our own 1D tensor, then when we call GEMM,
// we just reuse this instance again and again by replacing the class members (including memory pointer). Eventually,
// I will add a new constructor to the class in FBGEMM which accepts
// pre - allocated and pre - packed memory as a parameter.After it's done,
// this temporary buffer will be removed.
// When constructing this dummy buffer, ones are used for all the parameters to allocate minimum amount of memory.
//
// In a multi marian instance setting (as a dynamic library),
// different marian instances should not share this variable.
static thread_local PackedGemmMatrixFP16 packedPlaceholder(1, 1, 1, 1, 1, 1, 1, 1);

// This is copied from FBGEMM code
// A better way?
// will be removed, when FBGEMM api is changed
// blocked row-major format address arithmetic
/**
 * Returns the memory address in the packed (block formatted) matrix array of a specific element 
 * indexed by the original non-packed array.
 *
 * @param r_ row index in the original matrix
 * @param c_ column index in the original matrix
 * @param brow_ row wide block index
 * @param bcol_ column wide block index
 * @param nbrow_ number of blocks in row
 * @param nbcol_ number of blocks in column
 * @param last_brow_ row number of the last block
 */
inline uint64_t addr(const int r_,
                     const int c_,
                     const int brow_,
                     const int bcol_,
                     const int nbrow_,
                     const int nbcol_,
                     const int last_brow_) {
  uint64_t r = (uint64_t)r_;
  uint64_t c = (uint64_t)c_;

  uint64_t block_row_id = r / brow_;
  uint64_t brow_offset = (block_row_id * nbcol_) * (brow_ * bcol_);
  uint64_t block_col_id = c / bcol_;
  uint64_t bcol_offset
      = block_col_id * ((block_row_id != nbrow_ - 1) ? (brow_ * bcol_) : (last_brow_ * bcol_));
  uint64_t block_offset = brow_offset + bcol_offset;
  uint64_t inblock_offset = r % brow_ * bcol_ + c % bcol_;

  uint64_t index = block_offset + inblock_offset;
  return index;
}

inline void col_offsets_with_zero_pt_s8acc32_ref(
    bool transpose,
    int K,
    int N,
    const int8_t* Bint8,
    const int32_t* B_zero_point,
    int32_t* col_offsets,
    int ncols_per_quant_group) {
  for (int n = 0; n < N; ++n) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += transpose ? Bint8[k + n * K] : Bint8[k * N + n];
    }
    col_offsets[n] = sum - B_zero_point[n / ncols_per_quant_group] * K;
  }
}

void PackFp32(marian::Tensor out,
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
              const uint64_t packsize) {
  if(true) {
  //  if(!transpose) {
  //if (in->shape().size() == 2 && (in->shape()[0] < 3200 && in->shape()[1] < 3200)) {
    // initialize memory
    uint8_t* outmemorg = out->data<uint8_t>();
    for(auto i = 0; i < packsize; i++) {
      outmemorg[i] = 0;
    }
    // save the other auxiliary variables
    uint64_t* auxmemsize = (uint64_t*)outmemorg;
    auxmemsize[0] = packsize;
    // save FBGEMM related parameters into the header of the allocated memory by marian
    int32_t header[8];
    header[0] = nrow;
    header[1] = ncol;
    header[2] = kernel_ncol_blocks;
    header[3] = brow;
    header[4] = bcol;
    header[5] = last_brow;
    header[6] = nbrow;
    header[7] = nbcol;
    memcpy(auxmemsize + 1, header, sizeof(header));
    // cast to float16
    fbgemm::float16* outmem = (fbgemm::float16*)(outmemorg + 256);
    fbgemm::float16* dummy = new fbgemm::float16;
    // pack the matrix
    float* inmem = in->data();
    for(int i = 0; i < nrow; i++) {
      for(int j = 0; j < ncol; j++) {
        outmem[addr(i, j, brow, bcol, nbrow, nbcol, last_brow)]
            = tconv(!transpose ? inmem[i * ncol + j] : inmem[i + nrow * j], *dummy);
      }
    }
    delete dummy;
  } else {
    // Quantize to int8
    int k = transpose ? in->shape()[1] : in->shape()[0];
    int n = transpose ? in->shape()[0] : in->shape()[1];
    // std::cout << "transpose: " << transpose << ", k: " << k << ", n: " << n << std::endl;
    // std::cout << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
    // two steps
    // 0. quantize --> this should be done outside
    int len = in->shape()[0]*in->shape()[1];

    // 0-1. collect stats for each class
    float* bqScale = new float[n];
    int32_t* bqZeropoint = new int32_t[n];

    // int numBin = 20;
    // float denum = 2/(float)numBin;

    // int hist[numBin] = { 0, };

    float* data = in->data();
    float val = 0;
    for (int jj = 0; jj < n; jj++) {
      float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
      for (int ii = 0; ii < k; ii++) {
        if (transpose)
          val = data[jj*k + ii];
        else
          val = data[jj + ii*n];
        if (val < min) min = val;
        if (val > max) max = val;
        // hist[(int)((val + 1)/denum)]++;
      }
      bqScale[jj] = (max - min)/255;
      bqZeropoint[jj] = (int32_t)(127 - max / bqScale[jj]);
      // bqScale[jj] = (0.3 + 0.4)/255;
      // bqZeropoint[jj] = (int32_t)(127 - 0.3 / bqScale[jj]);
    }

    // std::cout << "hist: ";
    // for (int ii = 0; ii < numBin; ii++) {
    //   std::cout << hist[ii] << ", ";
    // }
    // std::cout << std::endl;
    // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
    //int8_t quantized[len]; // aligned malloc?
    int8_t* quantized;
    int result = posix_memalign((void**)&quantized, 256, len);
    assert(result == 0);
    //int8_t* quantized = (int8_t*)aligned_alloc(256, len);
    // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
    for (int jj = 0; jj < n; jj++) {
      TensorQuantizationParams bQuantParam;
      // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
      bQuantParam.scale = bqScale[jj];
      bQuantParam.zero_point = bqZeropoint[jj];
      bQuantParam.precision = 8;
      // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;

      if (transpose)
        fbgemm::Quantize<int8_t>(data + jj * k, quantized + jj * k, k, bQuantParam);
      else {
        for (int ii = 0; ii < k; ii++) {
          quantized[ii*n + jj] = fbgemm::Quantize<int8_t>(data[ii*n + jj], bQuantParam);
        }
      }
    }
    // std::cout << "original" << std::endl;
    // for (int ii = 0; ii < n; ii++) {
    //   for (int jj = 0; jj < 1; jj++) {
    //     std::cout << in->data()[ii * k + jj] << ","; 
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "quantized" << std::endl;
    // for (int ii = 0; ii < 1; ii++) {
    //   for (int jj = 0; jj < k; jj++) {
    //     std::cout << (int32_t)quantized[ii * k + jj] << ","; 
    //   }
    //   std::cout << std::endl;
    // }
    // 1. compute column offsets
    int32_t* col_offsets = new int32_t[n];
    col_offsets_with_zero_pt_s8acc32_ref(transpose, k, n, quantized, bqZeropoint, col_offsets, 1);
    // for (int ii = 0; ii < n; ii++) {
    //   std::cout << (int32_t)col_offsets->data()[ii] << ","; 
    // }
    // std::cout << std::endl;
    // std::cout << "calc offset done" << std::endl;
    // 2. packing
    // uint8_t* packedmem = aligned_alloc(256, len);
    // packedBint8 = new PackBMatrix<int8_t>(transpose ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
    //                                       k, // in->shape()[0],
    //                                       n, // in->shape()[1],
    //                                       quantized,
    //                                       in->shape()[1]);


    int8_t* packedbuf = out->data<int8_t>();
    for(auto i = 0; i < packsize; i++) {
      packedbuf[i] = 0;
    }
    // packing
    PackBMatrix<int8_t> packedBN(
        transpose ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
        nrow, ncol, quantized, in->shape()[1], packedbuf, 1);

    // copy quantization scale
    memcpy(packedbuf + (packsize - n * (sizeof(float) + sizeof(int32_t) + sizeof(int32_t))), bqScale, n * sizeof(float));
    //std::cout << "bqScale original: "
    //          << (void*)(packedbuf + (packsize - n * (sizeof(float) + sizeof(int32_t) + sizeof(int32_t))))
    //          << std::endl;
    // copy quantization offset
    memcpy(packedbuf + (packsize - n * (sizeof(int32_t) + sizeof(int32_t))), bqZeropoint, n * sizeof(int32_t));
    //std::cout << "bqZeropoint original: "
    //          << (void*)(packedbuf + (packsize - n * (sizeof(int32_t) + sizeof(int32_t))))
    //          << std::endl;
    // copy column offsets to the memory
    memcpy(packedbuf + (packsize - n * sizeof(int32_t)), col_offsets, n * sizeof(int32_t));
    //std::cout << "col_offsets original: " << (void*)(packedbuf + (packsize - n * sizeof(int32_t)))
    //          << std::endl;

    free(quantized);
    delete[] col_offsets;
    delete[] bqScale;
    delete[] bqZeropoint;
  }
}

// GEMM operation on the packed B matrix
// C: output matrix
// A: A matrix
// B: B matrix (packed)
// m: the number of rows in A and C
// n: the number of columns in B and C
// transA: transpose of A matrix
// transB: transpose of B matrix
void GemmPackFp32(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const marian::Tensor bias,
                  const size_t m,
                  const size_t n,
                  const size_t k,
                  const int transA,
                  const int transB) {
  if(true) {
  //  if(!transB) {
  //std::cout << "packed gemm: " << m << ", " << n << ", " << k << std::endl;
  //if (n < 3200) {
    // row major
    // keep the original mem
    fbgemm::float16* pmat = packedPlaceholder.pmat_;
    // retreive aux fields from the memory
    uint64_t* packedmemSize = (uint64_t*)B->data();
    packedPlaceholder.size_ = packedmemSize[0];
    int32_t header[8];
    memcpy(header, packedmemSize + 1, sizeof(header));
    packedPlaceholder.nrow_ = header[0];
    packedPlaceholder.ncol_ = header[1];
    packedPlaceholder.kernel_ncol_blocks_ = header[2];
    packedPlaceholder.brow_ = header[3];
    packedPlaceholder.bcol_ = header[4];
    packedPlaceholder.last_brow_ = header[5];
    packedPlaceholder.nbrow_ = header[6];
    packedPlaceholder.nbcol_ = header[7];

    // packed matrix
    packedPlaceholder.pmat_ = (fbgemm::float16*)(B->data<uint8_t>() + 256);

#if MKL_FOUND
    for(int i = 0; i < m; ++i) {
      mkl_somatcopy('R', 'N', 1, n, 1, bias->data(), n, C->data() + n * i, n);
    }
#else
    for(int i = 0; i < m; ++i) {
      std::copy(bias->data(), bias->data() + n, C->data() + n * i);
    }
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      int num_threads = omp_get_num_threads();
      int tid = omp_get_thread_num();
#else
      int num_threads = 1;
      int tid = 0;
#endif
      fbgemm::cblas_gemm_compute(transA ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
                        (int)m,
                        A->data(),
                        packedPlaceholder,
                        1,
                        C->data(),
                        tid,
                        num_threads);
    }

    // return back the original mem
    packedPlaceholder.pmat_ = pmat;
  } else {
    //std::cout << "int8 gemm: " << m << ", " << n << ", " << k << std::endl;
    // quantize & pack A
    // transformer base wmt 2017
    // float ascale = 7.8/104;
    // int32_t azeropoint = 151;
    // old student de-en
    // float ascale = 14.85/117;
    // int32_t azeropoint = 138;

    // compute range
    float min_est = std::numeric_limits<float>::max(), max_est = std::numeric_limits<float>::min();
    // VSLSSTaskPtr task;
    // MKL_INT task_p, task_n, xstorage;

    // /* Parameters of the task and initialization */
    // task_p = 1;
    // task_n = A->shape().elements();
    // xstorage = VSL_SS_MATRIX_STORAGE_ROWS;
    // min_est = max_est = A->data()[0];
    // /* Create a task */
    // vslsSSNewTask( &task, &task_p, &task_n, &xstorage, (float*)A->data(), 0, 0 );
    // /* Initialize the task parameters */
    // vslsSSEditTask( task, VSL_SS_ED_MIN, &min_est );
    // vslsSSEditTask( task, VSL_SS_ED_MAX, &max_est );
    // /* Compute the minimum and maximum values in observations */
    // vslsSSCompute( task, VSL_SS_MIN|VSL_SS_MAX, VSL_SS_METHOD_FAST );
    // /* Deallocate the task resources */

    // vslSSDeleteTask( &task );

    int elem = A->shape().elements();
    float* data = A->data();
    //for(int ii = 0; ii < elem; ii++) {
    //  if(data[ii] < min_est)
    //    min_est = data[ii];
    //  if(data[ii] > max_est)
    //    max_est = data[ii];
    //}
    FindMinMax(data, &min_est, &max_est, elem);

    std::vector<int32_t> row_offset_buf(PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize());

    float ascale = (max_est - min_est) / 255;
    //std::cout << "ascale: " << ascale << std::endl;
    int32_t azeropoint = (int32_t)(255 - max_est / ascale);
    //std::cout << "azeropoint: " << azeropoint << std::endl;
    PackAWithQuantRowOffset<uint8_t> packAN(
        transA ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
        transA ? k : m,
        transA ? m : k,
        A->data(),
        transA ? m : k,
        nullptr, /*buffer for packed matrix*/
        ascale,
        azeropoint,
        1, /*groups*/
        row_offset_buf.data());

    // packed matrix size
    //auto shapeMat = B->shape();
    int bPackSize = PackMatrix<PackBMatrix<int8_t>, int8_t>::packedBufferSize(k, n);

    // retrieve B matrix
    int8_t* bdata = B->data<int8_t>();
    float* bqScale = new float[n];
    memcpy(bqScale, bdata + bPackSize, n * sizeof(float));
    //std::cout << "bqScale ret: " << (void*)(bdata + bPackSize)
    //          << std::endl;
    //std::cout << "bqScale: " << bqScale[0] << ", " << bqScale[1] << ", " << bqScale[2] << std::endl;

    int32_t* bqZeropoint = new int32_t[n];
    memcpy(bqZeropoint, bdata + bPackSize + n * sizeof(float), n * sizeof(int32_t));
    //std::cout << "bqZeropoint ret: " << (void*)(bdata + bPackSize + n * sizeof(float)) << std::endl;
    //std::cout << "bqZeropoint: " << bqZeropoint[0] << ", " << bqZeropoint[1] << ", "
    //          << bqZeropoint[2]
    //          << std::endl;

    int32_t* col_offsets = new int32_t[n];
    memcpy(col_offsets, bdata + bPackSize + n * (sizeof(float) + sizeof(int32_t)), n * sizeof(int32_t));
    //std::cout << "col_offsets ret: " << (void*)(bdata + bPackSize + n * (sizeof(float) + sizeof(int32_t)))
    //          << std::endl;
    //std::cout << "col_offsets: " << col_offsets[0] << ", " << col_offsets[1] << ", "
    //    << col_offsets[2] << std::endl;

    DoNothing<float, float> doNothingObj{};
    ReQuantizeForFloat<false, QuantizationGranularity::OUT_CHANNEL> outputProcObj(
        doNothingObj,
        ascale,
        bqScale,
        azeropoint,
        bqZeropoint,
        packAN.getRowOffsetBuffer(),
        col_offsets,
        nullptr,
        n);

    PackBMatrix<int8_t> repackedBN(
      transB ? matrix_op_t::Transpose : matrix_op_t::NoTranspose, k, n, bdata, transB ? k : n, 1);

    // gemm
    fbgemmPacked(packAN, repackedBN, C->data(), (int32_t*)C->data(), n, outputProcObj, 0, 1);

    delete[] col_offsets;
    delete[] bqZeropoint;
    delete[] bqScale;
    // std::cout << "lowp gemm: " << std::endl;
    // for (int ii = 0; ii < n; ii++) {
    //   std::cout << C->data()[ii] << std::endl;
    // }
    // std::cout << std::endl;
  }
}
#else // USE_FBGEMM
void PackFp32(marian::Tensor out,
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
              const uint64_t packsize) {
  // does nothing. supports only FBGEMM based packed gemm at this moment.
  ABORT("FBGEMM is needed to use packed GEMM.");
}
void GemmPackFp32(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const marian::Tensor bias,
                  const size_t m,
                  const size_t n,
                  const int transA,
                  const int transB) {
  // does nothing. supports only FBGEMM based packed gemm at this moment.
  ABORT("FBGEMM is needed to use packed GEMM.");
}
#endif // USE_FBGEMM

// This operates on floats after processing so doesn't care about int8_t vs
// int16_t.
void AddBias(marian::Tensor C, const marian::Tensor Bias) {
  float* y = C->data();
  const float* x = C->data();
  const float* bias = Bias->data();

  int m = C->shape().elements() / C->shape()[-1];
  int n = C->shape()[-1];
#ifdef __AVX512F__
  int n16 = n & ~15;
#else
  int n4 = (n / 4) * 4;
#endif

  for(int j = 0; j < m; ++j) {
    int i = 0;
#ifdef __AVX512F__
    for(; i < n16; i += 16) {
      __m512 ai = _mm512_loadu_ps(x + j * n + i);
      __m512 bi = _mm512_loadu_ps(bias + i);
      __m512 yi = _mm512_add_ps(ai, bi);
      _mm512_storeu_ps(y + j * n + i, yi);
    }
#else
    for(; i < n4; i += 4) {
      __m128 ai = _mm_loadu_ps(x + j * n + i);
      __m128 bi = _mm_loadu_ps(bias + i);
      __m128 yi = _mm_add_ps(ai, bi);
      _mm_storeu_ps(y + j * n + i, yi);
    }
#endif
    for(; i < n; i++) {
      y[j * n + i] = x[j * n + i] + bias[i];
    }
  }

  // std::cout << "Output: " << std::endl;
  // for (int ii = 0; ii < n; ii++) {
  //   std::cout << y[ii] << ",";
  // }
  // std::cout << std::endl;
}

}  // namespace variant
}  // namespace cpu
}  // namespace marian
