#include "mjdgemm.h"
#include "mjdgemm_utils.h"

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Computes a block of matrix multiplication using NEON instructions.
 *
 * @tparam IS Instruction set, should be NEON.
 * @tparam N Number of columns.
 * @tparam K Number of rows.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 * @param MCBActual Actual number of rows in the block.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::NEON>::type
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

  // Ensure inputs are 16-byte aligned for NEON
  assert(reinterpret_cast<uintptr_t>(blockA[0]) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockB) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockC[0]) % 16 == 0);

  // Initialize registers
  int32x4_t c[MR][8];

  for (int mr = 0; mr < MCBActual; mr += MR) {
    int MRActual = std::min(MR, MCBActual - mr);

    if(initBlockWithZero) {
      const int32x4_t zero = vdupq_n_s32(0);
      for (int m = 0; m < MRActual; m++) {
        c[m][0] = zero; c[m][1] = zero; c[m][2] = zero; c[m][3] = zero;
        c[m][4] = zero; c[m][5] = zero; c[m][6] = zero; c[m][7] = zero;
      }
    } else {
      for (int m = 0; m < MRActual; m++) {
        c[m][0] = vld1q_s32(blockC[mr + m] + 0 * NR);
        c[m][1] = vld1q_s32(blockC[mr + m] + 1 * NR);
        c[m][2] = vld1q_s32(blockC[mr + m] + 2 * NR);
        c[m][3] = vld1q_s32(blockC[mr + m] + 3 * NR);
        c[m][4] = vld1q_s32(blockC[mr + m] + 4 * NR);
        c[m][5] = vld1q_s32(blockC[mr + m] + 5 * NR);
        c[m][6] = vld1q_s32(blockC[mr + m] + 6 * NR);
        c[m][7] = vld1q_s32(blockC[mr + m] + 7 * NR);
      }
    }

    for (int k = 0; k < KCB; k += RI) {
      const int8x16_t b0 = vld1q_s8(&blockB[k * NCB + 0 * RI * NR]);
      const int8x16_t b1 = vld1q_s8(&blockB[k * NCB + 1 * RI * NR]);
      const int8x16_t b2 = vld1q_s8(&blockB[k * NCB + 2 * RI * NR]);
      const int8x16_t b3 = vld1q_s8(&blockB[k * NCB + 3 * RI * NR]);
      const int8x16_t b4 = vld1q_s8(&blockB[k * NCB + 4 * RI * NR]);
      const int8x16_t b5 = vld1q_s8(&blockB[k * NCB + 5 * RI * NR]);
      const int8x16_t b6 = vld1q_s8(&blockB[k * NCB + 6 * RI * NR]);
      const int8x16_t b7 = vld1q_s8(&blockB[k * NCB + 7 * RI * NR]);

      for (int m = 0; m < MRActual; m++) {
        const int32_t aInt32 = *reinterpret_cast<const int32_t*>(blockA[mr + m] + k);
        const int8x16_t a = vreinterpretq_s8_s32(vdupq_n_s32(aInt32));

        c[m][0] = vdotq_s32(c[m][0], a, b0);
        c[m][1] = vdotq_s32(c[m][1], a, b1);
        c[m][2] = vdotq_s32(c[m][2], a, b2);
        c[m][3] = vdotq_s32(c[m][3], a, b3);
        c[m][4] = vdotq_s32(c[m][4], a, b4);
        c[m][5] = vdotq_s32(c[m][5], a, b5);
        c[m][6] = vdotq_s32(c[m][6], a, b6);
        c[m][7] = vdotq_s32(c[m][7], a, b7);
      }
    }

    for (int m = 0; m < MRActual; m++) {
      vst1q_s32(blockC[mr + m] + 0 * NR, c[m][0]);
      vst1q_s32(blockC[mr + m] + 1 * NR, c[m][1]);
      vst1q_s32(blockC[mr + m] + 2 * NR, c[m][2]);
      vst1q_s32(blockC[mr + m] + 3 * NR, c[m][3]);
      vst1q_s32(blockC[mr + m] + 4 * NR, c[m][4]);
      vst1q_s32(blockC[mr + m] + 5 * NR, c[m][5]);
      vst1q_s32(blockC[mr + m] + 6 * NR, c[m][6]);
      vst1q_s32(blockC[mr + m] + 7 * NR, c[m][7]);
    }
  }
}

/**
 * @brief Computes a block of matrix multiplication using NEON instructions with a fixed number of rows.
 *
 * @tparam IS Instruction set, should be NEON.
 * @tparam N Number of columns in the block.
 * @tparam K Number of rows in the block.
 * @tparam MRActual Actual number of rows in the block.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 */
template <InstructionSet IS, int N, int K, int MRActual>
inline typename std::enable_if<IS == InstructionSet::NEON>::type
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

  // Ensure inputs are 16-byte aligned for NEON
  assert(reinterpret_cast<uintptr_t>(blockA[0]) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockB) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockC[0]) % 16 == 0);

  static_assert(MRActual <= MR);

  // Initialize registers
  int32x4_t c[MR][8];

  if(initBlockWithZero) {
    const int32x4_t zero = vdupq_n_s32(0);
    for (int m = 0; m < MRActual; m++) {
      c[m][0] = zero; c[m][1] = zero; c[m][2] = zero; c[m][3] = zero;
      c[m][4] = zero; c[m][5] = zero; c[m][6] = zero; c[m][7] = zero;
    }
  } else {
    for (int m = 0; m < MRActual; m++) {
      c[m][0] = vld1q_s32(blockC[m] + 0 * NR);
      c[m][1] = vld1q_s32(blockC[m] + 1 * NR);
      c[m][2] = vld1q_s32(blockC[m] + 2 * NR);
      c[m][3] = vld1q_s32(blockC[m] + 3 * NR);
      c[m][4] = vld1q_s32(blockC[m] + 4 * NR);
      c[m][5] = vld1q_s32(blockC[m] + 5 * NR);
      c[m][6] = vld1q_s32(blockC[m] + 6 * NR);
      c[m][7] = vld1q_s32(blockC[m] + 7 * NR);
    }
  }

  for (int k = 0; k < KCB; k += RI) {
    const int8x16_t b0 = vld1q_s8(&blockB[k * NCB + 0 * RI * NR]);
    const int8x16_t b1 = vld1q_s8(&blockB[k * NCB + 1 * RI * NR]);
    const int8x16_t b2 = vld1q_s8(&blockB[k * NCB + 2 * RI * NR]);
    const int8x16_t b3 = vld1q_s8(&blockB[k * NCB + 3 * RI * NR]);
    const int8x16_t b4 = vld1q_s8(&blockB[k * NCB + 4 * RI * NR]);
    const int8x16_t b5 = vld1q_s8(&blockB[k * NCB + 5 * RI * NR]);
    const int8x16_t b6 = vld1q_s8(&blockB[k * NCB + 6 * RI * NR]);
    const int8x16_t b7 = vld1q_s8(&blockB[k * NCB + 7 * RI * NR]);

    for (int m = 0; m < MRActual; m++) {
      const int32_t aInt32 = *reinterpret_cast<const int32_t*>(blockA[m] + k);
      const int8x16_t a = vreinterpretq_s8_s32(vdupq_n_s32(aInt32));

      c[m][0] = vdotq_s32(c[m][0], a, b0);
      c[m][1] = vdotq_s32(c[m][1], a, b1);
      c[m][2] = vdotq_s32(c[m][2], a, b2);
      c[m][3] = vdotq_s32(c[m][3], a, b3);
      c[m][4] = vdotq_s32(c[m][4], a, b4);
      c[m][5] = vdotq_s32(c[m][5], a, b5);
      c[m][6] = vdotq_s32(c[m][6], a, b6);
      c[m][7] = vdotq_s32(c[m][7], a, b7);
    }
  }

  for (int m = 0; m < MRActual; m++) {
    vst1q_s32(blockC[m] + 0 * NR, c[m][0]);
    vst1q_s32(blockC[m] + 1 * NR, c[m][1]);
    vst1q_s32(blockC[m] + 2 * NR, c[m][2]);
    vst1q_s32(blockC[m] + 3 * NR, c[m][3]);
    vst1q_s32(blockC[m] + 4 * NR, c[m][4]);
    vst1q_s32(blockC[m] + 5 * NR, c[m][5]);
    vst1q_s32(blockC[m] + 6 * NR, c[m][6]);
    vst1q_s32(blockC[m] + 7 * NR, c[m][7]);
  }

}

/**
 * @brief Selects the appropriate computeBlock function based on the number of rows.
 *
 * @tparam IS Instruction set, should be NEON.
 * @tparam N Number of columns.
 * @tparam K Number of rows.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 * @param M Number of rows in the block.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::NEON>::type
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
 * @tparam IS Instruction set, should be NEON.
 * @tparam N Number of columns.
 * @tparam K Number of rows.
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param quantParams Reference to the quantization parameters structure.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::NEON>::type
computeQuantizationParams(const float* A, const int8_t* B, QuantizationParams& quantParams) {
  constexpr int VLEN = BlockingFactors<IS>::VLEN;

  // Compute the min and max values of the input via fast NEON instructions,
  // assume the length is a multiple of VLEN
  float minA = std::numeric_limits<float>::max();
  float maxA = std::numeric_limits<float>::lowest();

  float32x4_t min_v = vdupq_n_f32(minA);
  float32x4_t max_v = vdupq_n_f32(maxA);

  for(size_t i = 0; i < K; i += VLEN) {
    float32x4_t src_v = vld1q_f32(A + i); // load 4 floats
    min_v = vminq_f32(min_v, src_v); // find min
    max_v = vmaxq_f32(max_v, src_v); // find max
  }

  // Horizontal min and max reduction
  minA = vminvq_f32(min_v);
  maxA = vmaxvq_f32(max_v);

  quantParams.sumA       = 0;
  // Compute the scale and zero point
  quantParams.scaleA     = (maxA - minA) / 255;

  // Note: The zero point is computed differently for NEON compared to AVX/SSE instructions
  // We are actually casting to int8_t here despite using uint8_t in the representation.
  // This is because AVX/SSE has a fused multiply-add instruction that allows multiplies unsigend int8 with
  // signed int8 and adds the result to an int32. NEON does not have this instruction, instead there is
  // a similar instruction that multiplies signed int8 with signed int8 before accumulating the result.

  // Hence we use 127 as the max value for int8_t, and compute the zero point as 127 - maxA / scaleA
  quantParams.zeroPointA = (int32_t)(127 - maxA / quantParams.scaleA);

  // quantization parameters for B, access them from the end of the packed buffer
  // there is n of quantScaleB, quantZeropointB and colOffsetsB, each
  quantParams.packSizeB   = K * N;
  quantParams.scaleB      = (const float*)  (B + quantParams.packSizeB);
  quantParams.zeroPointB  = (const int32_t*)(B + quantParams.packSizeB + N * sizeof(float));
  quantParams.colOffsetsB = (const int32_t*)(B + quantParams.packSizeB + N * sizeof(float) + N * sizeof(int32_t));
}

/**
 * @brief Quantizes a block of matrix A using NEON instructions.
 *
 * @tparam IS Instruction set, should be NEON.
 * @tparam LEN Length of the block.
 * @param src Pointer to the source matrix.
 * @param dst Pointer to the destination matrix.
 * @param quantParams Reference to the quantization parameters structure.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::NEON>::type
quantize(const float* src, uint8_t* dst, QuantizationParams& quantParams) {
  constexpr int VLEN = BlockingFactors<IS>::VLEN; // should be 4 for NEON

  constexpr float min_val = std::numeric_limits<int8_t>::min();
  constexpr float max_val = std::numeric_limits<int8_t>::max();

  const float32x4_t min_val_v = vdupq_n_f32(min_val);
  const float32x4_t max_val_v = vdupq_n_f32(max_val);
  const float32x4_t inverse_scale_v = vdupq_n_f32(1.f / quantParams.scaleA);
  const float32x4_t zero_point_v = vdupq_n_f32(static_cast<float>(quantParams.zeroPointA));

  assert(LEN % VLEN == 0); // ensure the length is a multiple of VLEN

  int32x4_t sum_v = vdupq_n_s32(0); // initialize sum to zero

  for (std::size_t i = 0; i < LEN; i += VLEN) {
    float32x4_t xv = vld1q_f32(src + i); // load 4 floats
    xv = vmlaq_f32(zero_point_v, xv, inverse_scale_v); // multiply and add with zero point
    xv = vmaxq_f32(vminq_f32(xv, max_val_v), min_val_v); // clip to [min_val, max_val]
    int32x4_t xv_i = vcvtq_s32_f32(vrndnq_f32(xv)); // convert to 4 x 32-bit integers

    // Accumulate the 32-bit integers for sum
    sum_v = vaddq_s32(sum_v, xv_i);

    // Compress the 4 32-bit integers into 4 8-bit integers
    const int16x4_t xv_s16 = vmovn_s32(xv_i); // narrow 32-bit to 16-bit

    // narrow 16-bit to 8-bit (this is a signed conversion)
    // Note: Everything here is signed, but we are storing in uint8_t, in the matmul kernel
    // we treat the values as signed again, so everything is consistent.
    const int8x8_t xv_s8 = vmovn_s16(vcombine_s16(xv_s16, xv_s16));

    // Store the final 4 8-bit integers into the destination
    vst1_lane_s32(reinterpret_cast<int32_t*>(dst + i), vreinterpret_s32_s8(xv_s8), 0);
  }

  // Horizontal sum reduction across the 4 lanes
  quantParams.sumA += vaddvq_s32(sum_v);
}

/**
 * @brief Dequantizes and adds bias to a block of matrix C using NEON instructions.
 *
 * @tparam IS Instruction set, should be NEON.
 * @tparam LEN Length of the block.
 * @param src Pointer to the source matrix.
 * @param bias Pointer to the bias matrix.
 * @param quantParams Reference to the quantization parameters structure.
 * @param dst Pointer to the destination matrix.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::NEON>::type
dequantizeAndAdd(const int32_t* src, const float* bias, const QuantizationParams& quantParams, float* dst) {
  constexpr int VLEN = BlockingFactors<IS>::VLEN; // should be 4 for NEON

  const int32x4_t quantZeropointA_v = vdupq_n_s32(quantParams.zeroPointA);
  const int32x4_t quantSumA_v       = vdupq_n_s32(quantParams.sumA);
  const float32x4_t quantScaleA_v   = vdupq_n_f32(quantParams.scaleA);

  for (int j = 0; j < LEN; j += VLEN) {
    // load xv from src
    int32x4_t xv                      = vld1q_s32(src + j);
    const int32x4_t colOffsetsB_v     = vld1q_s32(quantParams.colOffsetsB + j);
    const int32x4_t quantZeropointB_v = vld1q_s32(quantParams.zeroPointB + j);
    const float32x4_t quantScaleB_v   = vld1q_f32(quantParams.scaleB + j);

    // compute xv = xv - quantZeropointA * colOffsetsB
    xv = vsubq_s32(xv, vmulq_s32(quantZeropointA_v, colOffsetsB_v));
    xv = vsubq_s32(xv, vmulq_s32(quantZeropointB_v, quantSumA_v));

    // compute scale * xv
    const float32x4_t scale_v = vmulq_f32(quantScaleA_v, quantScaleB_v);
    const float32x4_t xv_f = vcvtq_f32_s32(xv);

    float32x4_t result = vmulq_f32(scale_v, xv_f);
    if(bias) {
      const float32x4_t bias_v = vld1q_f32(bias + j);
      result = vaddq_f32(result, bias_v);
    }

    // store result to dst
    vst1q_f32(dst + j, result);
  }
}

}  // namespace mjdgemm
}  // namespace cpu
}  // namespace marian

#include "mjdgemm_tmp.h"

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Performs GEMM operation using NEON instruction set.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param bias Pointer to the bias matrix.
 * @param C Pointer to the destination matrix.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */
void gemmInt8PackedNEON(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K) {
  LOG_ONCE(info, "mjdgemm: Using NEON kernels");
  gemmInt8Packed<InstructionSet::NEON>(A, B, bias, C, M, N, K);
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian
