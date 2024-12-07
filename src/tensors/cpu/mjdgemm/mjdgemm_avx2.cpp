#include "mjdgemm.h"
#include "mjdgemm_utils.h"

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Computes a block of matrix multiplication using AVX2 instructions.
 *
 * @tparam IS Instruction set, should be AVX2.
 * @tparam N Number of columns in the block.
 * @tparam K Number of rows in the block.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 * @param MCBActual Actual number of rows in the block.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::AVX2>::type
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

  // Ensure inputs are 32-byte aligned for AVX2
  assert(reinterpret_cast<uintptr_t>(blockA[0]) % 32 == 0);
  assert(reinterpret_cast<uintptr_t>(blockB) % 32 == 0);
  assert(reinterpret_cast<uintptr_t>(blockC[0]) % 32 == 0);

  const __m256i constant_1 = _mm256_set1_epi16(1);

  // Initialize registers
  __m256i c[MR][4];

  #define __LOAD(ptr) \
    _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))

  #define __STORE(ptr, reg) \
    _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), reg)

  // There is no _mm512_dpbusds_epi32 equivalent in AVX2, so we emulate it here
  #define __DPBUSDS(c, a, b) \
    _mm256_add_epi32(c, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), constant_1))

  for(int mr = 0; mr < MCBActual; mr += MR) {
    int MRActual = std::min(MR, MCBActual - mr);

    if(initBlockWithZero) {
      for(int m = 0; m < MRActual; m++) {
        c[m][0] = _mm256_setzero_si256();
        c[m][1] = _mm256_setzero_si256();
        c[m][2] = _mm256_setzero_si256();
        c[m][3] = _mm256_setzero_si256();
      }
    } else {
      for(int m = 0; m < MRActual; m++) {
        c[m][0] = __LOAD(blockC[mr + m]);
        c[m][1] = __LOAD(blockC[mr + m] +     NR);
        c[m][2] = __LOAD(blockC[mr + m] + 2 * NR);
        c[m][3] = __LOAD(blockC[mr + m] + 3 * NR);
      }
    }

    // Loop over inner dimension in step of row interleave RI
    for(int k = 0; k < KCB; k += RI) {
      const __m256i b0 = __LOAD(&blockB[k * NCB]);
      const __m256i b1 = __LOAD(&blockB[k * NCB +     RI * NR]);
      const __m256i b2 = __LOAD(&blockB[k * NCB + 2 * RI * NR]);
      const __m256i b3 = __LOAD(&blockB[k * NCB + 3 * RI * NR]);

      for(int m = 0; m < MRActual; m++) {
        // Load 4 elements of A and broadcast to 64 8-bit integers
        // into register a
        const int a32i = *reinterpret_cast<const int32_t*>(&blockA[mr + m][k]);
        const __m256i a = _mm256_set1_epi32(a32i);

        c[m][0] = __DPBUSDS(c[m][0], a, b0);
        c[m][1] = __DPBUSDS(c[m][1], a, b1);
        c[m][2] = __DPBUSDS(c[m][2], a, b2);
        c[m][3] = __DPBUSDS(c[m][3], a, b3);
      }
    }

    // Store register contents back to memory
    for(int m = 0; m < MRActual; m++) {
      __STORE(blockC[mr + m],          c[m][0]);
      __STORE(blockC[mr + m] +     NR, c[m][1]);
      __STORE(blockC[mr + m] + 2 * NR, c[m][2]);
      __STORE(blockC[mr + m] + 3 * NR, c[m][3]);
    }
  }

  #undef __LOAD
  #undef __STORE
  #undef __DPBUSDS
}

/**
 * @brief Computes a block of matrix multiplication using AVX2 instructions with a fixed number of rows.
 *
 * @tparam IS Instruction set, should be AVX2.
 * @tparam N Number of columns in the block.
 * @tparam K Number of rows in the block.
 * @tparam MRActual Actual number of rows in the block.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 */
template <InstructionSet IS, int N, int K, int MRActual>
inline typename std::enable_if<IS == InstructionSet::AVX2>::type
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
  assert(reinterpret_cast<uintptr_t>(blockA) % 32 == 0);
  assert(reinterpret_cast<uintptr_t>(blockB) % 32 == 0);
  assert(reinterpret_cast<uintptr_t>(blockC) % 32 == 0);

  const __m256i constant_1 = _mm256_set1_epi16(1);

  // Initialize registers
  __m256i c[MR][4];

  #define __LOAD(ptr) \
    _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))

  #define __STORE(ptr, reg) \
    _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), reg)

  // There is no _mm512_dpbusds_epi32 equivalent in AVX2, so we emulate it here
  #define __DPBUSDS(c, a, b) \
    _mm256_add_epi32(c, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), constant_1))

  if(initBlockWithZero) {
    for(int m = 0; m < MRActual; m++) {
      c[m][0] = _mm256_setzero_si256();
      c[m][1] = _mm256_setzero_si256();
      c[m][2] = _mm256_setzero_si256();
      c[m][3] = _mm256_setzero_si256();
    }
  } else {
    for(int m = 0; m < MRActual; m++) {
      c[m][0] = __LOAD(blockC[m]);
      c[m][1] = __LOAD(blockC[m] +     NR);
      c[m][2] = __LOAD(blockC[m] + 2 * NR);
      c[m][3] = __LOAD(blockC[m] + 3 * NR);
    }
  }

  // Loop over inner dimension in step of row interleave RI
  for(int k = 0; k < KCB; k += RI) {
    const __m256i b0 = __LOAD(&blockB[k * NCB]);
    const __m256i b1 = __LOAD(&blockB[k * NCB +     RI * NR]);
    const __m256i b2 = __LOAD(&blockB[k * NCB + 2 * RI * NR]);
    const __m256i b3 = __LOAD(&blockB[k * NCB + 3 * RI * NR]);

    for(int m = 0; m < MRActual; m++) {
      // Load 4 elements of A and broadcast to 64 8-bit integers
      // into register a
      const int a32i = *reinterpret_cast<const int32_t*>(&blockA[m][k]);
      const __m256i a = _mm256_set1_epi32(a32i);

      c[m][0] = __DPBUSDS(c[m][0], a, b0);
      c[m][1] = __DPBUSDS(c[m][1], a, b1);
      c[m][2] = __DPBUSDS(c[m][2], a, b2);
      c[m][3] = __DPBUSDS(c[m][3], a, b3);
    }
  }

  // Store register contents back to memory
  for(int m = 0; m < MRActual; m++) {
    __STORE(blockC[m],          c[m][0]);
    __STORE(blockC[m] +     NR, c[m][1]);
    __STORE(blockC[m] + 2 * NR, c[m][2]);
    __STORE(blockC[m] + 3 * NR, c[m][3]);
  }

  #undef __LOAD
  #undef __STORE
  #undef __DPBUSDS
}

/**
 * @brief Selects the appropriate computeBlock function based on the number of rows.
 *
 * @tparam IS Instruction set, should be AVX2.
 * @tparam N Number of columns in the block.
 * @tparam K Number of rows in the block.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 * @param M Number of rows in the block.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::AVX2>::type
computeBlockSwitch(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC, int M, bool initBlockWithZero = false) {
  switch (M) {
    compute_case(1);
    compute_case(2);
    compute_case(3);
    compute_case(4);
    default:
      computeBlock<IS, N, K>(blockA, blockB, blockC, M, initBlockWithZero);
      break;
  }
}

/**
 * @brief Computes the quantization parameters for matrices A and B.
 *
 * @tparam IS Instruction set, should be AVX2.
 * @tparam N Number of columns in the block.
 * @tparam K Number of rows in the block.
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param quantParams Reference to the quantization parameters structure.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::AVX2>::type
computeQuantizationParams(const float* A, const int8_t* B, QuantizationParams& quantParams) {
  constexpr int VLEN = BlockingFactors<IS>::VLEN;

  // Compute the min and max values of the input via fast AVX2 instructions,
  // assume the length is a multiple of VLEN
  float minA = std::numeric_limits<float>::max();
  float maxA = std::numeric_limits<float>::lowest();

  __m256 min_v = _mm256_set1_ps(minA);
  __m256 max_v = _mm256_set1_ps(maxA);

  for (std::size_t i = 0; i < K; i += VLEN) {
    __m256 src_v = _mm256_load_ps(A + i); // load 8 floats
    min_v = _mm256_min_ps(min_v, src_v); // find min
    max_v = _mm256_max_ps(max_v, src_v); // find max
  }

  // Horizontal min and max reduction
  alignas(VLEN * sizeof(int32_t)) float min_array[VLEN];
  alignas(VLEN * sizeof(int32_t)) float max_array[VLEN];
  _mm256_store_ps(min_array, min_v);
  _mm256_store_ps(max_array, max_v);

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
 * @brief Quantizes a block of matrix A using AVX2 instructions.
 *
 * @tparam IS Instruction set, should be AVX2.
 * @tparam LEN Length of the block.
 * @param src Pointer to the source matrix.
 * @param dst Pointer to the destination matrix.
 * @param quantParams Reference to the quantization parameters structure.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::AVX2>::type
quantize(const float* src, uint8_t* dst, QuantizationParams& quantParams) {
  constexpr int VLEN = BlockingFactors<IS>::VLEN;

  constexpr float min_val = std::numeric_limits<uint8_t>::min();
  constexpr float max_val = std::numeric_limits<uint8_t>::max();
  __m256 inverse_scale_v = _mm256_set1_ps(1.f / quantParams.scaleA);

  #define INT8(x) static_cast<int8_t>(x) // helper macro to cast to int8_t to avoid narrowing conversion warning

  __m256i shuffle_mask_v = _mm256_set_epi8(
    INT8(0xff), INT8(0xff), INT8(0xff), INT8(0xff),
    INT8(0xff), INT8(0xff), INT8(0xff), INT8(0xff),
    INT8(0xff), INT8(0xff), INT8(0xff), INT8(0xff),
    INT8(0x0c), INT8(0x08), INT8(0x04), INT8(0x00),
    INT8(0xff), INT8(0xff), INT8(0xff), INT8(0xff),
    INT8(0xff), INT8(0xff), INT8(0xff), INT8(0xff),
    INT8(0xff), INT8(0xff), INT8(0xff), INT8(0xff),
    INT8(0x0c), INT8(0x08), INT8(0x04), INT8(0x00)
  );

  __m256i permute_mask_v = _mm256_set_epi32(
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x04, 0x00
  );


  assert(LEN % VLEN == 0); // ensure the length is a multiple of VLEN
  __m256i sum_v = _mm256_setzero_si256(); // initialize sum to zero

  for (std::size_t i = 0; i < LEN; i += VLEN) {
    __m256 src_v = _mm256_load_ps(src + i); // load 8 floats
    __m256 transformed_v = _mm256_fmadd_ps(src_v, inverse_scale_v, _mm256_set1_ps(static_cast<float>(quantParams.zeroPointA))); // multiply and add with zero point to scale
    __m256 clipped_v = _mm256_min_ps(_mm256_max_ps(transformed_v, _mm256_set1_ps(min_val)), _mm256_set1_ps(max_val)); // clip to [min_val, max_val]
    __m256i rounded_v = _mm256_cvtps_epi32(clipped_v); // convert to 32-bit integers

    // Accumulate the 32-bit integers
    sum_v = _mm256_add_epi32(sum_v, rounded_v);

    // An instruction sequence to save 8 32-bit integers as 8 8-bit integers
    rounded_v = _mm256_shuffle_epi8(rounded_v, shuffle_mask_v); // shuffle 32-bit integers to 8-bit integers
    rounded_v = _mm256_permutevar8x32_epi32(rounded_v, permute_mask_v); // permute 32-bit integers

    __m128i quant8 = _mm256_castsi256_si128(rounded_v);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i), quant8); // store 8 8-bit integers
  }

  // Horizontal sum reduction usin AVX2/AVX instructions
  __m128i sum_lo = _mm256_castsi256_si128(sum_v);
  __m128i sum_hi = _mm256_extracti128_si256(sum_v, 1);
  __m128i sum = _mm_add_epi32(sum_lo, sum_hi);

  // using hadd
  sum = _mm_hadd_epi32(sum, sum); // sum the 4 32-bit integers
  sum = _mm_hadd_epi32(sum, sum); // sum the 2 32-bit integers

  quantParams.sumA += _mm_cvtsi128_si32(sum); // reduce the 8 32-bit integers to a single 32-bit integer
}

/**
 * @brief Dequantizes and adds bias to a block of matrix C using AVX2 instructions.
 *
 * @tparam IS Instruction set, should be AVX2.
 * @tparam LEN Length of the block.
 * @param src Pointer to the source matrix.
 * @param bias Pointer to the bias matrix.
 * @param quantParams Reference to the quantization parameters structure.
 * @param dst Pointer to the destination matrix.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::AVX2>::type
dequantizeAndAdd(const int32_t* src, const float* bias, const QuantizationParams& quantParams, float* dst) {
  constexpr int VLEN = BlockingFactors<IS>::VLEN;

  __m256i quantZeropointA_v = _mm256_set1_epi32(quantParams.zeroPointA);
  __m256i quantSumA_v       = _mm256_set1_epi32(quantParams.sumA);
  __m256 quantScaleA_v      = _mm256_set1_ps(quantParams.scaleA);

  for (int j = 0; j < LEN; j += VLEN) {
    // load xv from cPtrInt32
    __m256i xv                = _mm256_load_si256((__m256i*)(src + j));
    __m256i colOffsetsB_v     = _mm256_load_si256((__m256i*)(quantParams.colOffsetsB + j));
    __m256i quantZeropointB_v = _mm256_load_si256((__m256i*)(quantParams.zeroPointB + j));
    __m256 quantScaleB_v      = _mm256_load_ps(quantParams.scaleB + j);

    // compute xv = xv - quantZeropointA * colOffsetsB
    xv = _mm256_sub_epi32(xv, _mm256_mullo_epi32(quantZeropointA_v, colOffsetsB_v));
    xv = _mm256_sub_epi32(xv, _mm256_mullo_epi32(quantZeropointB_v, quantSumA_v));

    // compute scale * xv
    __m256 scale_v = _mm256_mul_ps(quantScaleA_v, quantScaleB_v);
    __m256 xv_f = _mm256_cvtepi32_ps(xv);
    __m256 result = _mm256_mul_ps(scale_v, xv_f);

    if(bias) {
      __m256 bias_v = _mm256_load_ps(bias + j);
      result = _mm256_add_ps(result, bias_v);
    }

    // store result to cPtr
    _mm256_store_ps(dst + j, result);
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
 * @brief Performs GEMM operation using AVX2 instruction set.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param bias Pointer to the bias matrix.
 * @param C Pointer to the destination matrix.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */
void gemmInt8PackedAVX2(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K) {
  LOG_ONCE(info, "mjdgemm: Using AVX2 kernels");
  gemmInt8Packed<InstructionSet::AVX2>(A, B, bias, C, M, N, K);
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian
