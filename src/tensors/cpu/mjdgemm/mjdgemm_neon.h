#pragma once

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
computeBlock(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC, int MCBActual) {
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
  assert(reinterpret_cast<uintptr_t>(blockA) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockB) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockC) % 16 == 0);

  // Initialize registers
  int32x4_t c[MR][4];

  #define __LOAD_B(var, ptr) \
    const int8x16_t var##_temp = vld1q_s8(ptr); \
    const int16x8x2_t var = {{ \
      vmovl_s8(vget_low_s8(var##_temp)), \
      vmovl_s8(vget_high_s8(var##_temp)) \
    }}

  #define __DPBUSDS(c, a, b) \
    c = vaddq_s32(c, vpaddlq_s16(vpaddq_s16(vmulq_s16(a.val[0], b.val[0]), vmulq_s16(a.val[1], b.val[1]))))

  for (int mr = 0; mr < MCBActual; mr += MR) {
    int MRActual = std::min(MR, MCBActual - mr);

    for(int i = 0; i < 8; i += 4) {
      for (int m = 0; m < MRActual; m++) {
        c[m][0] = vld1q_s32(blockC[mr + m] + (i + 0) * NR);
        c[m][1] = vld1q_s32(blockC[mr + m] + (i + 1) * NR);
        c[m][2] = vld1q_s32(blockC[mr + m] + (i + 2) * NR);
        c[m][3] = vld1q_s32(blockC[mr + m] + (i + 3) * NR);
      }

      for (int k = 0; k < KCB; k += RI) {
        __LOAD_B(b0_16x2, &blockB[k * NCB + (i + 0) * RI * NR]);
        __LOAD_B(b1_16x2, &blockB[k * NCB + (i + 1) * RI * NR]);
        __LOAD_B(b2_16x2, &blockB[k * NCB + (i + 2) * RI * NR]);
        __LOAD_B(b3_16x2, &blockB[k * NCB + (i + 3) * RI * NR]);

        for (int m = 0; m < MRActual; m++) {
          const int32_t aInt32 = *reinterpret_cast<const int32_t*>(blockA[mr + m] + k);
          const int8x16_t tempA = vreinterpretq_s8_s32(vdupq_n_s32(aInt32));

          const int16x8x2_t a = {{
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(tempA))),
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(tempA)))
          }};

          __DPBUSDS(c[m][0], a, b0_16x2);
          __DPBUSDS(c[m][1], a, b1_16x2);
          __DPBUSDS(c[m][2], a, b2_16x2);
          __DPBUSDS(c[m][3], a, b3_16x2);
        }
      }

      for (int m = 0; m < MRActual; m++) {
        vst1q_s32(blockC[mr + m] + (i + 0) * NR, c[m][0]);
        vst1q_s32(blockC[mr + m] + (i + 1) * NR, c[m][1]);
        vst1q_s32(blockC[mr + m] + (i + 2) * NR, c[m][2]);
        vst1q_s32(blockC[mr + m] + (i + 3) * NR, c[m][3]);
      }
    }
  }

  #undef __DPBUSDS
  #undef __LOAD_B
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
computeBlock(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC) {
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
  assert(reinterpret_cast<uintptr_t>(blockA) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockB) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(blockC) % 16 == 0);

  static_assert(MRActual <= MR);

  // Initialize registers
  int32x4_t c[MR][4];

  #define __LOAD_B(var, ptr) \
    const int8x16_t var##_temp = vld1q_s8(ptr); \
    const int16x8x2_t var = {{ \
        vmovl_s8(vget_low_s8(var##_temp)), \
        vmovl_s8(vget_high_s8(var##_temp)) }}

  #define __DPBUSDS(c, a, b) \
    c = vaddq_s32(c, vpaddlq_s16(vpaddq_s16(vmulq_s16(a.val[0], b.val[0]), vmulq_s16(a.val[1], b.val[1]))))

  for(int i = 0; i < 8; i += 4) {
    for (int m = 0; m < MRActual; m++) {
      c[m][0] = vld1q_s32(blockC[m] + (i + 0) * NR);
      c[m][1] = vld1q_s32(blockC[m] + (i + 1) * NR);
      c[m][2] = vld1q_s32(blockC[m] + (i + 2) * NR);
      c[m][3] = vld1q_s32(blockC[m] + (i + 3) * NR);
    }

    for (int k = 0; k < KCB; k += RI) {
      __LOAD_B(b0, &blockB[k * NCB + (i + 0) * RI * NR]);
      __LOAD_B(b1, &blockB[k * NCB + (i + 1) * RI * NR]);
      __LOAD_B(b2, &blockB[k * NCB + (i + 2) * RI * NR]);
      __LOAD_B(b3, &blockB[k * NCB + (i + 3) * RI * NR]);

      for (int m = 0; m < MRActual; m++) {
        const int32_t aInt32 = *reinterpret_cast<const int32_t*>(blockA[m] + k);
        const int8x16_t tempA = vreinterpretq_s8_s32(vdupq_n_s32(aInt32));

        const int16x8x2_t a = {{
          vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(tempA))),
          vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(tempA)))
        }};

        __DPBUSDS(c[m][0], a, b0);
        __DPBUSDS(c[m][1], a, b1);
        __DPBUSDS(c[m][2], a, b2);
        __DPBUSDS(c[m][3], a, b3);
      }
    }

    for (int m = 0; m < MRActual; m++) {
      vst1q_s32(blockC[m] + (i + 0) * NR, c[m][0]);
      vst1q_s32(blockC[m] + (i + 1) * NR, c[m][1]);
      vst1q_s32(blockC[m] + (i + 2) * NR, c[m][2]);
      vst1q_s32(blockC[m] + (i + 3) * NR, c[m][3]);
    }
  }


    #undef __DPBUSDS
    #undef __LOAD_B
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
computeBlockSwitch(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC, int M) {
  switch (M) {
    compute_case(1);
    default:
      computeBlock<IS, N, K>(blockA, blockB, blockC, M);
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

  for (std::size_t i = 0; i < K; i += VLEN) {
    float32x4_t src_v = vld1q_f32(A + i); // load 4 floats
    min_v = vminq_f32(min_v, src_v); // find min
    max_v = vmaxq_f32(max_v, src_v); // find max
  }

  // Horizontal min and max reduction
  alignas(VLEN * sizeof(int32_t)) float min_array[VLEN];
  alignas(VLEN * sizeof(int32_t)) float max_array[VLEN];
  vst1q_f32(min_array, min_v);
  vst1q_f32(max_array, max_v);

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

  constexpr float min_val = std::numeric_limits<uint8_t>::min();
  constexpr float max_val = std::numeric_limits<uint8_t>::max();

  float32x4_t min_val_v = vdupq_n_f32(min_val);
  float32x4_t max_val_v = vdupq_n_f32(max_val);
  float32x4_t inverse_scale_v = vdupq_n_f32(1.f / quantParams.scaleA);
  float32x4_t zero_point_v = vdupq_n_f32(static_cast<float>(quantParams.zeroPointA));

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
    uint16x4_t xv_u16 = vmovn_u32(vreinterpretq_u32_s32(xv_i)); // narrow 32-bit to 16-bit

    // narrow 16-bit to 8-bit
    uint8x8_t xv_u8 = vmovn_u16(vcombine_u16(xv_u16, xv_u16));

    // Store the final 4 8-bit integers into the destination
    vst1_lane_u32(reinterpret_cast<uint32_t*>(dst + i), vreinterpret_u32_u8(xv_u8), 0);
  }

  // Horizontal sum reduction across the 4 lanes
  alignas(VLEN * sizeof(int32_t)) int32_t sum_array[VLEN];
  vst1q_s32(sum_array, sum_v);

  for (int i = 0; i < VLEN; ++i) {
    quantParams.sumA += sum_array[i];
  }
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

  int32x4_t quantZeropointA_v = vdupq_n_s32(quantParams.zeroPointA);
  int32x4_t quantSumA_v       = vdupq_n_s32(quantParams.sumA);
  float32x4_t quantScaleA_v   = vdupq_n_f32(quantParams.scaleA);

  for (int j = 0; j < LEN; j += VLEN) {
    // load xv from src
    int32x4_t xv                = vld1q_s32(src + j);
    int32x4_t colOffsetsB_v     = vld1q_s32(quantParams.colOffsetsB + j);
    int32x4_t quantZeropointB_v = vld1q_s32(quantParams.zeroPointB + j);
    float32x4_t quantScaleB_v   = vld1q_f32(quantParams.scaleB + j);

    // compute xv = xv - quantZeropointA * colOffsetsB
    xv = vsubq_s32(xv, vmulq_s32(quantZeropointA_v, colOffsetsB_v));
    xv = vsubq_s32(xv, vmulq_s32(quantZeropointB_v, quantSumA_v));

    // compute scale * xv
    float32x4_t scale_v = vmulq_f32(quantScaleA_v, quantScaleB_v);
    float32x4_t xv_f = vcvtq_f32_s32(xv);
    float32x4_t result = vmulq_f32(scale_v, xv_f);

    if(bias) {
      float32x4_t bias_v = vld1q_f32(bias + j);
      result = vaddq_f32(result, bias_v);
  }

    // store result to dst
    vst1q_f32(dst + j, result);
  }
}

}  // namespace mjdgemm
}  // namespace cpu
}  // namespace marian
