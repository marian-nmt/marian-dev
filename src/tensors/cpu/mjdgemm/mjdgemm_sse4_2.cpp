#include "mjdgemm.h"
#include "mjdgemm_utils.h"

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Computes a block of matrix multiplication using SSE4.2 instructions.
 *
 * @tparam IS Instruction set.
 * @tparam N Number of columns.
 * @tparam K Number of rows.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 * @param MCBActual Actual number of rows in the block.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::AVX || IS == InstructionSet::SSE4_2>::type
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

  // Ensure inputs are 16-byte aligned for AVX/SSE
  assert(reinterpret_cast<uintptr_t>(blockA[0]) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockB) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockC[0]) % 16 == 0);

  const __m128i constant_1 = _mm_set1_epi16(1);

  #define __LOAD(ptr) \
      _mm_load_si128(reinterpret_cast<const __m128i*>(ptr))

  #define __STORE(ptr, reg) \
      _mm_store_si128(reinterpret_cast<__m128i*>(ptr), reg)

  // There is no _mm512_dpbusds_epi32 equivalent in AVX, so we emulate it here
  // this takes 16 x 8-bit integers in a and b, and fmads into 4 x 32-bit integers in c
  #define __DPBUSDS(cv, av, bv) \
      cv = _mm_add_epi32(cv, _mm_madd_epi16(_mm_maddubs_epi16(av, bv), constant_1))

  // Initialize registers
  __m128i c[MR][8];

  for (int mr = 0; mr < MCBActual; mr += MR) {
    int MRActual = std::min(MR, MCBActual - mr);

    if(initBlockWithZero) {
      for (int m = 0; m < MRActual; m++) {
        c[m][0] = _mm_setzero_si128();
        c[m][1] = _mm_setzero_si128();
        c[m][2] = _mm_setzero_si128();
        c[m][3] = _mm_setzero_si128();
        c[m][4] = _mm_setzero_si128();
        c[m][5] = _mm_setzero_si128();
        c[m][6] = _mm_setzero_si128();
        c[m][7] = _mm_setzero_si128();
      }
    } else {
      for (int m = 0; m < MRActual; m++) {
        c[m][0] = __LOAD(blockC[mr + m]);
        c[m][1] = __LOAD(blockC[mr + m] + 1 * NR);
        c[m][2] = __LOAD(blockC[mr + m] + 2 * NR);
        c[m][3] = __LOAD(blockC[mr + m] + 3 * NR);
        c[m][4] = __LOAD(blockC[mr + m] + 4 * NR);
        c[m][5] = __LOAD(blockC[mr + m] + 5 * NR);
        c[m][6] = __LOAD(blockC[mr + m] + 6 * NR);
        c[m][7] = __LOAD(blockC[mr + m] + 7 * NR);
      }
    }

    for (int k = 0; k < KCB; k += RI) {
      const __m128i b0 = __LOAD(&blockB[k * NCB]);
      const __m128i b1 = __LOAD(&blockB[k * NCB + 1 * RI * NR]);
      const __m128i b2 = __LOAD(&blockB[k * NCB + 2 * RI * NR]);
      const __m128i b3 = __LOAD(&blockB[k * NCB + 3 * RI * NR]);
      const __m128i b4 = __LOAD(&blockB[k * NCB + 4 * RI * NR]);
      const __m128i b5 = __LOAD(&blockB[k * NCB + 5 * RI * NR]);
      const __m128i b6 = __LOAD(&blockB[k * NCB + 6 * RI * NR]);
      const __m128i b7 = __LOAD(&blockB[k * NCB + 7 * RI * NR]);

      for (int m = 0; m < MRActual; m++) {
        int a32i = *reinterpret_cast<const int32_t*>(&blockA[mr + m][k]);
        const __m128i a = _mm_set1_epi32(a32i);

        __DPBUSDS(c[m][0], a, b0);
        __DPBUSDS(c[m][1], a, b1);
        __DPBUSDS(c[m][2], a, b2);
        __DPBUSDS(c[m][3], a, b3);
        __DPBUSDS(c[m][4], a, b4);
        __DPBUSDS(c[m][5], a, b5);
        __DPBUSDS(c[m][6], a, b6);
        __DPBUSDS(c[m][7], a, b7);
      }
    }

    for (int m = 0; m < MRActual; m++) {
      __STORE(blockC[mr + m],          c[m][0]);
      __STORE(blockC[mr + m] + 1 * NR, c[m][1]);
      __STORE(blockC[mr + m] + 2 * NR, c[m][2]);
      __STORE(blockC[mr + m] + 3 * NR, c[m][3]);
      __STORE(blockC[mr + m] + 4 * NR, c[m][4]);
      __STORE(blockC[mr + m] + 5 * NR, c[m][5]);
      __STORE(blockC[mr + m] + 6 * NR, c[m][6]);
      __STORE(blockC[mr + m] + 7 * NR, c[m][7]);
    }
  }

  #undef __LOAD
  #undef __STORE
  #undef __DPBUSDS
}

/**
 * @brief Computes a block of matrix multiplication using NEON instructions with a fixed number of rows.
 *
 * @tparam IS Instruction set.
 * @tparam N Number of columns.
 * @tparam K Number of rows.
 * @tparam MRActual Actual number of rows to process.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 */
template <InstructionSet IS, int N, int K, int MRActual>
inline typename std::enable_if<IS == InstructionSet::AVX || IS == InstructionSet::SSE4_2>::type
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

  // Ensure inputs are 16-byte aligned for AVX/SSE
  assert(reinterpret_cast<uintptr_t>(blockA[0]) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockB) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockC[0]) % 16 == 0);

  const __m128i constant_1 = _mm_set1_epi16(1);

  // Helper macros for 128-bit load and store
  #define __LOAD(ptr) \
      _mm_load_si128(reinterpret_cast<const __m128i*>(ptr))

  #define __STORE(ptr, reg) \
      _mm_store_si128(reinterpret_cast<__m128i*>(ptr), reg)

  // There is no _mm512_dpbusds_epi32 equivalent in AVX, so we emulate it here
  // this takes 16 x 8-bit integers in a and b, and fmads into 4 x 32-bit integers in c
  #define __DPBUSDS(c, a, b) \
      _mm_add_epi32(c, _mm_madd_epi16(_mm_maddubs_epi16(a, b), constant_1))

  // Initialize registers
  __m128i c[MR][8];

  static_assert(MRActual <= MR);

  if(initBlockWithZero) {
    for (int m = 0; m < MRActual; m++) {
      c[m][0] = _mm_setzero_si128();
      c[m][1] = _mm_setzero_si128();
      c[m][2] = _mm_setzero_si128();
      c[m][3] = _mm_setzero_si128();
      c[m][4] = _mm_setzero_si128();
      c[m][5] = _mm_setzero_si128();
      c[m][6] = _mm_setzero_si128();
      c[m][7] = _mm_setzero_si128();
    }
  } else {
    for (int m = 0; m < MRActual; m++) {
      c[m][0] = __LOAD(blockC[m]);
      c[m][1] = __LOAD(blockC[m] + NR);
      c[m][2] = __LOAD(blockC[m] + 2 * NR);
      c[m][3] = __LOAD(blockC[m] + 3 * NR);
      c[m][4] = __LOAD(blockC[m] + 4 * NR);
      c[m][5] = __LOAD(blockC[m] + 5 * NR);
      c[m][6] = __LOAD(blockC[m] + 6 * NR);
      c[m][7] = __LOAD(blockC[m] + 7 * NR);
    }
  }

  for (int k = 0; k < KCB; k += RI) {
    const __m128i b0 = __LOAD(&blockB[k * NCB]);
    const __m128i b1 = __LOAD(&blockB[k * NCB + RI * NR]);
    const __m128i b2 = __LOAD(&blockB[k * NCB + 2 * RI * NR]);
    const __m128i b3 = __LOAD(&blockB[k * NCB + 3 * RI * NR]);
    const __m128i b4 = __LOAD(&blockB[k * NCB + 4 * RI * NR]);
    const __m128i b5 = __LOAD(&blockB[k * NCB + 5 * RI * NR]);
    const __m128i b6 = __LOAD(&blockB[k * NCB + 6 * RI * NR]);
    const __m128i b7 = __LOAD(&blockB[k * NCB + 7 * RI * NR]);

    for (int m = 0; m < MRActual; m++) {
      int a32i = *reinterpret_cast<const int32_t*>(&blockA[m][k]);
      const __m128i a = _mm_set1_epi32(a32i);

      c[m][0] = __DPBUSDS(c[m][0], a, b0);
      c[m][1] = __DPBUSDS(c[m][1], a, b1);
      c[m][2] = __DPBUSDS(c[m][2], a, b2);
      c[m][3] = __DPBUSDS(c[m][3], a, b3);
      c[m][4] = __DPBUSDS(c[m][4], a, b4);
      c[m][5] = __DPBUSDS(c[m][5], a, b5);
      c[m][6] = __DPBUSDS(c[m][6], a, b6);
      c[m][7] = __DPBUSDS(c[m][7], a, b7);
    }
  }

  for (int m = 0; m < MRActual; m++) {
    __STORE(blockC[m], c[m][0]);
    __STORE(blockC[m] + NR, c[m][1]);
    __STORE(blockC[m] + 2 * NR, c[m][2]);
    __STORE(blockC[m] + 3 * NR, c[m][3]);
    __STORE(blockC[m] + 4 * NR, c[m][4]);
    __STORE(blockC[m] + 5 * NR, c[m][5]);
    __STORE(blockC[m] + 6 * NR, c[m][6]);
    __STORE(blockC[m] + 7 * NR, c[m][7]);
  }

  #undef __LOAD
  #undef __STORE
  #undef __DPBUSDS
}

/**
 * @brief Selects the appropriate computeBlock function based on the number of rows.
 *
 * @tparam IS Instruction set.
 * @tparam N Number of columns.
 * @tparam K Number of rows.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 * @param M Number of rows in the block.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::AVX || IS == InstructionSet::SSE4_2>::type
computeBlockSwitch(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC, int M, bool initBlockWithZero = false) {
  switch (M) {
    compute_case(1);
    default:
      computeBlock<IS, N, K>(blockA, blockB, blockC, M, initBlockWithZero);
      break;
  }
}

/**
 * @brief Computes the quantization parameters for matrices A and B.
 *
 * @tparam IS Instruction set.
 * @tparam N Number of columns.
 * @tparam K Number of rows.
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param quantParams Reference to the quantization parameters structure.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::AVX || IS == InstructionSet::SSE4_2>::type
computeQuantizationParams(const float* A, const int8_t* B, QuantizationParams& quantParams) {
  constexpr int VLEN = BlockingFactors<IS>::VLEN;

  // Compute the min and max values of the input via fast AVX2 instructions,
  // assume the length is a multiple of VLEN
  float minA = std::numeric_limits<float>::max();
  float maxA = std::numeric_limits<float>::lowest();

  __m128 min_v = _mm_set1_ps(minA);
  __m128 max_v = _mm_set1_ps(maxA);

  for (std::size_t i = 0; i < K; i += VLEN) {
    __m128 src_v = _mm_load_ps(A + i); // load 8 floats
    min_v = _mm_min_ps(min_v, src_v); // find min
    max_v = _mm_max_ps(max_v, src_v); // find max
  }

  // Horizontal min and max reduction
  alignas(VLEN * sizeof(int32_t)) float min_array[VLEN];
  alignas(VLEN * sizeof(int32_t)) float max_array[VLEN];
  _mm_store_ps(min_array, min_v);
  _mm_store_ps(max_array, max_v);

  for (int i = 0; i < VLEN; ++i) {
    minA = std::min(minA, min_array[i]);
    maxA = std::max(maxA, max_array[i]);
  }

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
 * @brief Quantizes a block of matrix A.
 *
 * @tparam IS Instruction set.
 * @tparam LEN Length of the block.
 * @param src Pointer to the source matrix.
 * @param dst Pointer to the destination matrix.
 * @param quantParams Reference to the quantization parameters structure.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::AVX || IS == InstructionSet::SSE4_2>::type
quantize(const float* src, uint8_t* dst, QuantizationParams& quantParams) {

  constexpr int VLEN = BlockingFactors<IS>::VLEN; // should be 16 for AVX-128

  constexpr float min_val = std::numeric_limits<uint8_t>::min();
  constexpr float max_val = std::numeric_limits<uint8_t>::max();

  __m128 min_val_v = _mm_set1_ps(min_val);
  __m128 max_val_v = _mm_set1_ps(max_val);
  __m128 inverse_scale_v = _mm_set1_ps(1.f / quantParams.scaleA);
  __m128 zero_point_v = _mm_set1_ps(static_cast<float>(quantParams.zeroPointA));

  assert(LEN % VLEN == 0); // ensure the length is a multiple of VLEN

  __m128i sum_v = _mm_setzero_si128(); // initialize sum to zero

  for (std::size_t i = 0; i < LEN; i += VLEN) {
    __m128 xv    = _mm_load_ps(src + i); // load 4 floats
    xv           = _mm_add_ps(_mm_mul_ps(xv, inverse_scale_v), zero_point_v); // multiply and add with zero point
    xv           = _mm_min_ps(_mm_max_ps(xv, min_val_v), max_val_v); // clip to [min_val, max_val]
    __m128i xv_i = _mm_cvtps_epi32(xv); // convert to 4 x 32-bit integers

    // Accumulate the 32-bit integers for sum
    sum_v = _mm_add_epi32(sum_v, xv_i);

    // Compress the 4 32-bit integers into 4 8-bit integers
    xv_i = _mm_packus_epi32(xv_i, xv_i); // pack 32-bit to 16-bit
    xv_i = _mm_packus_epi16(xv_i, xv_i); // pack 16-bit to 8-bit

    // Store the final 8-bit integers into the destination
    _mm_store_ss(reinterpret_cast<float*>(dst + i), _mm_castsi128_ps(xv_i)); // store 4 8-bit integers
  }

  // Horizontal sum reduction across the 16 lanes using SSE instructions directly, like hadd
  sum_v = _mm_hadd_epi32(sum_v, sum_v); // sum the 4 32-bit integers
  sum_v = _mm_hadd_epi32(sum_v, sum_v); // sum the 2 32-bit integers

  // Store the sum to the quantization parameters
  quantParams.sumA += _mm_cvtsi128_si32(sum_v);
}

/**
 * @brief Dequantizes and adds bias to a block of matrix C.
 *
 * @tparam IS Instruction set.
 * @tparam LEN Length of the block.
 * @param src Pointer to the source matrix.
 * @param bias Pointer to the bias matrix.
 * @param quantParams Reference to the quantization parameters structure.
 * @param dst Pointer to the destination matrix.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::AVX || IS == InstructionSet::SSE4_2>::type
dequantizeAndAdd(const int32_t* src, const float* bias, const QuantizationParams& quantParams, float* dst) {

  constexpr int VLEN = BlockingFactors<IS>::VLEN;

  __m128i quantZeropointA_v = _mm_set1_epi32(quantParams.zeroPointA);
  __m128i quantSumA_v       = _mm_set1_epi32(quantParams.sumA);
  __m128 quantScaleA_v      = _mm_set1_ps(quantParams.scaleA);

  for (int j = 0; j < LEN; j += VLEN) {
    // load xv from cPtrInt32
    __m128i xv                = _mm_load_si128((__m128i*)(src + j));
    __m128i colOffsetsB_v     = _mm_load_si128((__m128i*)(quantParams.colOffsetsB + j));
    __m128i quantZeropointB_v = _mm_load_si128((__m128i*)(quantParams.zeroPointB + j));
    __m128 quantScaleB_v      = _mm_load_ps(quantParams.scaleB + j);

    // compute xv = xv - quantZeropointA * colOffsetsB
    xv = _mm_sub_epi32(xv, _mm_mullo_epi32(quantZeropointA_v, colOffsetsB_v));
    xv = _mm_sub_epi32(xv, _mm_mullo_epi32(quantZeropointB_v, quantSumA_v));

    // compute scale * xv
    __m128 scale_v = _mm_mul_ps(quantScaleA_v, quantScaleB_v);
    __m128 xv_f = _mm_cvtepi32_ps(xv);
    __m128 result = _mm_mul_ps(scale_v, xv_f);

    if(bias) {
      __m128 bias_v = _mm_load_ps(bias + j);
      result = _mm_add_ps(result, bias_v);
    }

    // store result to cPtr
    _mm_store_ps(dst + j, result);
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
 * @brief Performs GEMM operation using SSE4.2 instruction set.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param bias Pointer to the bias matrix.
 * @param C Pointer to the destination matrix.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */
void gemmInt8PackedSSE4_2(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K) {
  LOG_ONCE(info, "mjdgemm: Using SSE4_2 kernels");
  gemmInt8Packed<InstructionSet::SSE4_2>(A, B, bias, C, M, N, K);
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian
