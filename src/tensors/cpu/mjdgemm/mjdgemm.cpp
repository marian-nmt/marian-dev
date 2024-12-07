#include "mjdgemm.h"
#include "mjdgemm_utils.h"

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Computes a block of the matrix multiplication using the specified instruction set.
 *
 * This function performs matrix multiplication on a block of matrices A and B, storing the result in matrix C.
 * It uses the specified instruction set (IS) to optimize the computation.
 *
 * @tparam IS The instruction set to use for optimization (e.g., AVX2, SSE4.2, etc.).
 * @tparam N The number of columns in matrix B and C.
 * @tparam K The number of columns in matrix A and rows in matrix B.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 * @param MCBActual The actual number of rows in the current block of matrix A and C.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::None>::type
computeBlock(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC, int MCBActual, bool initBlockWithZero = false) {
  constexpr int NCB = BlockingFactors<IS>::NCB;
  constexpr int KCB = BlockingFactors<IS>::KCB;
  constexpr int RI  = BlockingFactors<IS>::RI;
  constexpr int NR  = BlockingFactors<IS>::NR;

  static_assert(KCB % RI == 0);
  static_assert(K % KCB == 0);
  static_assert(N % NCB == 0);
  static_assert(N % (2 * NR) == 0);

  // Initialize registers
  int32_t c[2][NR];

  for(int m = 0; m < MCBActual; m++) {
    // Load 32 (NCB) elements of C into 2 registers c[0] and c[1], 16 each
    if(initBlockWithZero) {
      std::fill(c[0], c[0] + NR, 0);
      std::fill(c[1], c[1] + NR, 0);
    } else {
      std::copy(blockC[m],      blockC[m] +     NR, c[0]);
      std::copy(blockC[m] + NR, blockC[m] + 2 * NR, c[1]);
    }

    // Loop over inner dimension in step of row interleave RI
    for(int k = 0; k < KCB; k += RI) {
      // Load 4 elements of A and broadcast to 64 8-bit integers
      // into register a
      const uint8_t* a = blockA[m] + k;

      // Load 32 * 4 (NCB * RI) elements of B into 2 registers b[0] and b[1]
      const int8_t* b0 = blockB + k * NCB;
      const int8_t* b1 = blockB + k * NCB + RI * NR;

      // Multiply and add in sets of 4 elements and accumulate
      for(int j = 0; j < NR; j++) {
        c[0][j] += a[0] * b0[j * RI + 0]
                 + a[1] * b0[j * RI + 1]
                 + a[2] * b0[j * RI + 2]
                 + a[3] * b0[j * RI + 3];

        c[1][j] += a[0] * b1[j * RI + 0]
                 + a[1] * b1[j * RI + 1]
                 + a[2] * b1[j * RI + 2]
                 + a[3] * b1[j * RI + 3];
      }
    }

    // Store register contents back to memory
    std::copy(c[0], c[0] + NR, blockC[m]);
    std::copy(c[1], c[1] + NR, blockC[m] + NR);
  }
}

template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::None>::type
computeBlockSwitch(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC, int M, bool initBlockWithZero = false) {
  computeBlock<InstructionSet::None, N, K>(blockA, blockB, blockC, M, initBlockWithZero);
}

template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::None>::type
computeQuantizationParams(const float* A, const int8_t* B, QuantizationParams& quantParams) {
  // Compute the min and max values of the input via fast AVX2 instructions,
  // assume the length is a multiple of VLEN
  float minA = std::numeric_limits<float>::max();
  float maxA = std::numeric_limits<float>::lowest();

  // compute minA and maxA
  for (std::size_t i = 0; i < K; i++) {
    minA = std::min(minA, A[i]);
    maxA = std::max(maxA, A[i]);
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
 * @brief Quantizes the given source matrix to the destination matrix using the specified quantization parameters.
 *
 * This function quantizes the given source matrix to the destination matrix using the specified quantization parameters.
 *
 * @tparam IS The instruction set to use for optimization (e.g., AVX2, SSE4.2, etc.).
 * @tparam LEN The length of the source and destination matrices.
 * @param src Pointer to the source matrix.
 * @param dst Pointer to the destination matrix.
 * @param quantParams Reference to the QuantizationParams structure containing the quantization parameters.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::None>::type
quantize(const float* src, uint8_t* dst, QuantizationParams& quantParams) {
  constexpr float min_val = std::numeric_limits<uint8_t>::min();
  constexpr float max_val = std::numeric_limits<uint8_t>::max();

  for (std::size_t i = 0; i < LEN; i++) {
    float transform = src[i] / quantParams.scaleA + quantParams.zeroPointA;
    float clipped = std::min(std::max(transform, min_val), max_val);
    int32_t rounded = (int32_t)std::nearbyint(clipped);

    quantParams.sumA += rounded;

    dst[i] = (uint8_t)rounded;
  }
}

/**
 * @brief Dequantizes the given source matrix and adds the result to the destination matrix.
 *
 * This function dequantizes the given source matrix and adds the result to the destination matrix.
 *
 * @tparam IS The instruction set to use for optimization (e.g., AVX2, SSE4.2, etc.).
 * @tparam LEN The length of the source and destination matrices.
 * @param src Pointer to the source matrix.
 * @param bias Pointer to the bias matrix (optional).
 * @param quantParams Reference to the QuantizationParams structure containing the quantization parameters.
 * @param dst Pointer to the destination matrix.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::None>::type
dequantizeAndAdd(const int32_t* src, const float* bias, const QuantizationParams& quantParams, float* dst) {
  for (int j = 0; j < LEN; j++) {
    int32_t xv = src[j];

    xv = xv - quantParams.zeroPointA * quantParams.colOffsetsB[j];
    xv = xv - quantParams.zeroPointB[j] * quantParams.sumA;

    // compute scale * xv
    float scale = quantParams.scaleA * quantParams.scaleB[j];
    float xv_f = (float)xv;
    float result = scale * xv_f;

    if(bias) {
      result += bias[j];
    }

    dst[j] = result;
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
 * @brief Determines whether to force the use of mjdgemm based on environment variables.
 *
 * @return True if mjdgemm is forced, otherwise false.
 */
bool forceMjdgemm() {
  char* env = std::getenv("MJD_FORCE");
  if(env) {
    std::string force(env);
    if(force == "true" || force == "1" || force == "True" || force == "TRUE"
      || force == "yes" || force == "Yes" || force == "YES"
      || force == "on" || force == "On" || force == "ON") {
      LOG(warn, "mjdgemm: Forced to use mjdgemm");
      return true;
    }
  }
  return false;
}

void gemmInt8PackedAVX512(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K);
void gemmInt8PackedAVX2(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K);
void gemmInt8PackedSSE4_2(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K);
void gemmInt8PackedNEON(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K);

/**
 * @brief Selects and executes the appropriate GEMM implementation based on the highest supported instruction set.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param bias Pointer to the bias matrix.
 * @param C Pointer to the destination matrix.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */
void gemmInt8Packed(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K) {
  static const auto highestInstructionSet = getSupportedInstructionSet();

  switch (highestInstructionSet) {
#if defined(COMPILE_FOR_INTEL)
    case InstructionSet::AVX512_POPCNT:
    case InstructionSet::AVX512:
      gemmInt8PackedAVX512(A, B, bias, C, M, N, K);
      break;
    case InstructionSet::AVX2:
      gemmInt8PackedAVX2(A, B, bias, C, M, N, K);
      break;
    case InstructionSet::AVX:
    case InstructionSet::SSE4_2:
      gemmInt8PackedSSE4_2(A, B, bias, C, M, N, K);
      break;
#elif defined(COMPILE_FOR_ARM)
    case InstructionSet::NEON:
      gemmInt8PackedNEON(A, B, bias, C, M, N, K);
      break;
#endif
    default:
      gemmInt8Packed<InstructionSet::None>(A, B, bias, C, M, N, K);
      break;
  }
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian
