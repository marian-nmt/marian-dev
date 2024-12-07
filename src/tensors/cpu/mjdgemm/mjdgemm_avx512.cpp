#include "mjdgemm.h"
#include "mjdgemm_utils.h"

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Computes a block of matrix multiplication using AVX-512 instructions.
 *
 * @tparam IS Instruction set, should be AVX512.
 * @tparam N Number of columns in the block.
 * @tparam K Number of rows in the block.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 * @param MCBActual Actual number of rows in the block.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::AVX512>::type
computeBlock(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC, int MCBActual, bool initBlockWithZero = false) {
  constexpr int NCB = BlockingFactors<IS>::NCB;
  constexpr int KCB = BlockingFactors<IS>::KCB;
  constexpr int RI  = BlockingFactors<IS>::RI;
  constexpr int NR  = BlockingFactors<IS>::NR;

  constexpr int MCB = BlockingFactors<IS>::MCB;
  constexpr int MR  = BlockingFactors<IS>::MR;

  static_assert(MCB % MR == 0);
  static_assert(KCB % RI == 0);
  static_assert(K % KCB == 0);
  static_assert(N % NCB == 0);
  static_assert(N % (2 * NR) == 0);

  // assert that inputs are 64-byte aligned
  assert(reinterpret_cast<uintptr_t>(blockA[0]) % 64 == 0);
  assert(reinterpret_cast<uintptr_t>(blockB) % 64 == 0);
  assert(reinterpret_cast<uintptr_t>(blockC[0]) % 64 == 0);

  // Initialize registers
  __m512i c[MR][2];

  for(int mr = 0; mr < MCBActual; mr += MR) {
    int MRActual = std::min(MR, MCBActual - mr);

    if(initBlockWithZero) {
      for(int m = 0; m < MRActual; m++) {
        c[m][0] = _mm512_setzero_si512();
        c[m][1] = _mm512_setzero_si512();
      }
    } else {
      for(int m = 0; m < MRActual; m++) {
        c[m][0] = _mm512_load_si512(blockC[mr + m]);
        c[m][1] = _mm512_load_si512(blockC[mr + m] + NR);
      }
    }

    // Loop over inner dimension in step of row interleave RI
    for(int k = 0; k < KCB; k += RI) {
      // Load 2 blocks of B into registers, in total 2 * 64 = 4 (RI) * 32 = 128 8-bit integers
      const __m512i b0 = _mm512_load_si512(&blockB[k * NCB]);
      const __m512i b1 = _mm512_load_si512(&blockB[k * NCB + RI * NR]);

      for(int m = 0; m < MRActual; m++) {
          // Load 4 elements of A and broadcast to 64 8-bit integers into register a
          const __m512i a = _mm512_broadcastd_epi32(_mm_set1_epi32(*reinterpret_cast<const int32_t*>(&blockA[mr + m][k])));

          c[m][0] = _mm512_dpbusds_epi32(c[m][0], a, b0);
          c[m][1] = _mm512_dpbusds_epi32(c[m][1], a, b1);
      }
    }

    // Store register contents back to memory
    for(int m = 0; m < MRActual; m++) {
      _mm512_store_si512(blockC[mr + m],      c[m][0]);
      _mm512_store_si512(blockC[mr + m] + NR, c[m][1]);
    }
  }
}

/**
 * @brief Computes a block of matrix multiplication using AVX-512 instructions with a fixed number of rows.
 *
 * @tparam IS Instruction set, should be AVX512.
 * @tparam N Number of columns in the block.
 * @tparam K Number of rows in the block.
 * @tparam MRActual Actual number of rows in the block.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 */
template <InstructionSet IS, int N, int K, int MRActual>
inline typename std::enable_if<IS == InstructionSet::AVX512>::type
computeBlock(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC, bool initBlockWithZero = false) {
  constexpr int NCB = BlockingFactors<IS>::NCB;
  constexpr int KCB = BlockingFactors<IS>::KCB;
  constexpr int RI  = BlockingFactors<IS>::RI;
  constexpr int NR  = BlockingFactors<IS>::NR;

  constexpr int MCB = BlockingFactors<IS>::MCB;
  constexpr int MR  = BlockingFactors<IS>::MR;

  static_assert(MCB % MR == 0);
  static_assert(KCB % RI == 0);
  static_assert(K % KCB == 0);
  static_assert(N % NCB == 0);
  static_assert(N % (2 * NR) == 0);

  static_assert(MRActual <= MR);

  // assert that inputs are 64-byte aligned
  assert(reinterpret_cast<uintptr_t>(blockA[0]) % 64 == 0);
  assert(reinterpret_cast<uintptr_t>(blockB) % 64 == 0);
  assert(reinterpret_cast<uintptr_t>(blockC[0]) % 64 == 0);

  // Initialize registers
  __m512i c[MRActual][2];

  if(initBlockWithZero) {
    for(int m = 0; m < MRActual; m++) {
      c[m][0] = _mm512_setzero_si512();
      c[m][1] = _mm512_setzero_si512();
    }
  } else {
    for(int m = 0; m < MRActual; m++) {
      c[m][0] = _mm512_load_si512(blockC[m]);
      c[m][1] = _mm512_load_si512(blockC[m] + NR);
    }
  }

  // Loop over inner dimension in step of row interleave RI
  for(int k = 0; k < KCB; k += RI) {
    const __m512i b0 = _mm512_load_si512(&blockB[k * NCB]);
    const __m512i b1 = _mm512_load_si512(&blockB[k * NCB + RI * NR]);

    for(int m = 0; m < MRActual; m++) {
        // Load 4 elements of A and broadcast to 64 8-bit integers into register a
        // int32_t value into 32-bit register, then broadcast to 64 8-bit integers
        const __m512i a = _mm512_broadcastd_epi32(_mm_set1_epi32(*reinterpret_cast<const int32_t*>(&blockA[m][k])));

        c[m][0] = _mm512_dpbusds_epi32(c[m][0], a, b0);
        c[m][1] = _mm512_dpbusds_epi32(c[m][1], a, b1);
    }
  }

  // Store register contents back to memory
  for(int m = 0; m < MRActual; m++) {
    _mm512_store_si512(blockC[m],      c[m][0]);
    _mm512_store_si512(blockC[m] + NR, c[m][1]);
  }
}

/**
 * @brief Selects the appropriate computeBlock function based on the number of rows.
 *
 * @tparam IS Instruction set, should be AVX512.
 * @tparam N Number of columns in the block.
 * @tparam K Number of rows in the block.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 * @param M Number of rows in the block.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::AVX512>::type
computeBlockSwitch(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC, int M, bool initBlockWithZero = false) {
  switch (M) {
    compute_case(1);
    compute_case(2);
    compute_case(3);
    compute_case(4);
    compute_case(5);
    compute_case(6);
    compute_case(7);
    compute_case(8);
    default:
      computeBlock<IS, N, K>(blockA, blockB, blockC, M, initBlockWithZero);
      break;
  }
}

/**
 * @brief Computes the quantization parameters for matrices A and B.
 *
 * @tparam IS Instruction set, should be AVX512.
 * @tparam N Number of columns in the block.
 * @tparam K Number of rows in the block.
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param quantParams Reference to the quantization parameters structure.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::AVX512>::type
computeQuantizationParams(const float* A, const int8_t* B, QuantizationParams& quantParams) {
  constexpr int VLEN = BlockingFactors<IS>::VLEN;

  // Compute the min and max values of the input via fast AVX2 instructions,
  // assume the length is a multiple of VLEN
  float minA = std::numeric_limits<float>::max();
  float maxA = std::numeric_limits<float>::lowest();

  __m512 src_v;
  __m512 min_v = _mm512_set1_ps(minA);
  __m512 max_v = _mm512_set1_ps(maxA);

  for (std::size_t i = 0; i < K; i += VLEN) {
    src_v = _mm512_load_ps(A + i); // load 8 floats
    min_v = _mm512_min_ps(min_v, src_v); // find min
    max_v = _mm512_max_ps(max_v, src_v); // find max
  }

  maxA = _mm512_reduce_max_ps(max_v); // horizontal max reduction
  minA = _mm512_reduce_min_ps(min_v); // horizontal min reduction

  quantParams.sumA       = 0;

  // Compute the scale and zero point
  quantParams.scaleA     = (maxA - minA) / 255;
  quantParams.zeroPointA = (int32_t)(255 - maxA / quantParams.scaleA);

  // quantization parameters for B, access them from the end of the packed buffer
  // there is n of quantScaleB, quantZeropointB and colOffsetsB, each
  quantParams.packSizeB   = K * N;
  quantParams.scaleB      = (const float*)  (B + quantParams.packSizeB);
  quantParams.zeroPointB  = (const int32_t*)(B + quantParams.packSizeB + N * sizeof(float));
  quantParams.colOffsetsB = (const int32_t*)(B + quantParams.packSizeB + N * sizeof(float) + N * sizeof(int32_t));
}

/**
 * @brief Quantizes a block of matrix A using AVX-512 instructions.
 *
 * @tparam IS Instruction set, should be AVX512.
 * @tparam LEN Length of the block.
 * @param src Pointer to the source matrix.
 * @param dst Pointer to the destination matrix.
 * @param quantParams Reference to the quantization parameters structure.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::AVX512>::type
quantize(const float* src, uint8_t* dst, QuantizationParams& quantParams) {
  constexpr int VLEN = BlockingFactors<IS>::VLEN; // should be 16 for AVX-512

  constexpr float min_val = std::numeric_limits<uint8_t>::min();
  constexpr float max_val = std::numeric_limits<uint8_t>::max();

  __m512 min_val_v = _mm512_set1_ps(min_val);
  __m512 max_val_v = _mm512_set1_ps(max_val);
  __m512 inverse_scale_v = _mm512_set1_ps(1.f / quantParams.scaleA);
  __m512 zero_point_v = _mm512_set1_ps(static_cast<float>(quantParams.zeroPointA));
  __m512i mask = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8, 4, 0);

  assert(LEN % VLEN == 0); // ensure the length is a multiple of VLEN

  __m512i sum_v = _mm512_setzero_si512(); // initialize sum to zero

  for (std::size_t i = 0; i < LEN; i += VLEN) {
    __m512 xv    = _mm512_load_ps(src + i); // load 16 floats
    xv           = _mm512_fmadd_ps(xv, inverse_scale_v, zero_point_v); // multiply and add with zero point
    xv           = _mm512_min_ps(_mm512_max_ps(xv, min_val_v), max_val_v); // clip to [min_val, max_val]
    __m512i xv_i = _mm512_cvtps_epi32(xv); // convert to 32-bit integers

    // Accumulate the 32-bit integers for sum
    sum_v = _mm512_add_epi32(sum_v, xv_i);

    // Compress the 16 32-bit integers into 16 8-bit integers
    xv_i = _mm512_packus_epi32(xv_i, xv_i); // pack 32-bit to 16-bit
    xv_i = _mm512_packus_epi16(xv_i, xv_i); // pack 16-bit to 8-bit
    xv_i = _mm512_permutexvar_epi32(mask, xv_i); // permute for compact storage

    // Store the final 8-bit integers into the destination
    __m128i quant8 = _mm512_castsi512_si128(xv_i); // narrow to 128-bit containing 16 8-bit integers
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), quant8); // store 16 8-bit integers
  }

  quantParams.sumA += _mm512_reduce_add_epi32(sum_v); // reduce the 16 32-bit integers to a single 32-bit integer
}

/**
 * @brief Dequantizes and adds bias to a block of matrix C using AVX-512 instructions.
 *
 * @tparam IS Instruction set, should be AVX512.
 * @tparam LEN Length of the block.
 * @param src Pointer to the source matrix.
 * @param bias Pointer to the bias matrix.
 * @param quantParams Reference to the quantization parameters structure.
 * @param dst Pointer to the destination matrix.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::AVX512>::type
dequantizeAndAdd(const int32_t* src, const float* bias, const QuantizationParams& quantParams, float* dst) {
  constexpr int VLEN = BlockingFactors<IS>::VLEN;

  __m512i quantZeropointA_v = _mm512_set1_epi32(quantParams.zeroPointA);
  __m512i quantSumA_v       = _mm512_set1_epi32(quantParams.sumA);
  __m512 quantScaleA_v      = _mm512_set1_ps(quantParams.scaleA);

  for (int j = 0; j < LEN; j += VLEN) {
    // load xv from cPtrInt32
    __m512i xv                = _mm512_load_si512((__m512i*)(src + j));
    __m512i colOffsetsB_v     = _mm512_load_si512((__m512i*)(quantParams.colOffsetsB + j));
    __m512i quantZeropointB_v = _mm512_load_si512((__m512i*)(quantParams.zeroPointB + j));
    __m512 quantScaleB_v      = _mm512_load_ps(quantParams.scaleB + j);

    // compute xv = xv - quantZeropointA * colOffsetsB
    xv = _mm512_sub_epi32(xv, _mm512_mullo_epi32(quantZeropointA_v, colOffsetsB_v));
    xv = _mm512_sub_epi32(xv, _mm512_mullo_epi32(quantZeropointB_v, quantSumA_v));

    // compute scale * xv
    __m512 scale_v = _mm512_mul_ps(quantScaleA_v, quantScaleB_v);
    __m512 xv_f = _mm512_cvtepi32_ps(xv);
    __m512 result = _mm512_mul_ps(scale_v, xv_f);

    if(bias) {
      __m512 bias_v = _mm512_load_ps(bias + j);
      result = _mm512_add_ps(result, bias_v);
    }

    // store result to cPtr
    _mm512_store_ps(dst + j, result);
  }
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian

#include "mjdgemm_tmp.h"

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Performs GEMM operation using AVX-512 instruction set.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param bias Pointer to the bias matrix.
 * @param C Pointer to the destination matrix.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */
void gemmInt8PackedAVX512(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K) {
  LOG_ONCE(info, "mjdgemm: Using AVX-512 kernels");
  gemmInt8Packed<InstructionSet::AVX512>(A, B, bias, C, M, N, K);
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian
