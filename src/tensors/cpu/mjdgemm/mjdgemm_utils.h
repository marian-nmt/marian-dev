#pragma once

#include "common/logging.h"
#include "tensors/cpu/cpu_info.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

namespace marian {
namespace cpu {
namespace mjdgemm {

#define compute_case(M) \
  case M: \
    computeBlock<IS, N, K, M>(blockA, blockB, blockC, initBlockWithZero); \
    break;

/**
 * @brief Struct to hold the blocking factors specific to the instruction set.
 */
template <InstructionSet IS>
struct BlockingFactors {};

template <>
struct BlockingFactors<InstructionSet::AVX512> {
  static constexpr int NCB  = 32;
  static constexpr int KCB  = 256;
  static constexpr int NR   = 16;
  static constexpr int RI   = 4;
  static constexpr int VLEN = 16;

  static constexpr int MR   = 8; // this seems to benefit from tuning
  static constexpr int MCB  = 48;
};

template <>
struct BlockingFactors<InstructionSet::AVX2> {
  static constexpr int NCB = 32;
  static constexpr int KCB = 256;
  static constexpr int RI = 4;
  static constexpr int NR = 8;
  static constexpr int VLEN = 8;

  // test on real AVX2 machine
  static constexpr int MR  = 4;
  static constexpr int MCB = 48;
};

template <>
struct BlockingFactors<InstructionSet::AVX> {
  static constexpr int NCB = 32;
  static constexpr int KCB = 256;
  static constexpr int RI = 4;
  static constexpr int NR = 4;
  static constexpr int VLEN = 4;

  // test on real AVX machine
  static constexpr int MR  = 1;
  static constexpr int MCB = 16;
};

template <>
struct BlockingFactors<InstructionSet::SSE4_2> {
  static constexpr int NCB = 32;
  static constexpr int KCB = 256;
  static constexpr int RI = 4;
  static constexpr int NR = 4;
  static constexpr int VLEN = 4;

  // test on real SSE machine
  static constexpr int MR  = 1;
  static constexpr int MCB = 16;
};

template <>
struct BlockingFactors<InstructionSet::NEON> {
  static constexpr int NCB = 32;
  static constexpr int KCB = 256;
  static constexpr int RI = 4;
  static constexpr int NR = 4;
  static constexpr int VLEN = 4;

  // test on real ARM64 machine
  static constexpr int MR  = 1;
  static constexpr int MCB = 56;
};

template <>
struct BlockingFactors<InstructionSet::None> {
  static constexpr int NCB = 32;
  static constexpr int KCB = 256;
  static constexpr int RI = 4;
  static constexpr int NR = 16;
  static constexpr int VLEN = 16;

  static constexpr int MR = 14;
  static constexpr int MCB = 56;
};

/**
 * @brief Struct for storing the quantization parameters for matrices A and B.
 */
struct QuantizationParams {
  float scaleA{0.f};
  int32_t zeroPointA{0};
  int32_t sumA{0};
  const float* scaleB{nullptr};
  const int32_t* zeroPointB{nullptr};
  const int32_t* colOffsetsB{nullptr};
  size_t packSizeB{0};
};

/**
 * @brief Packs the matrix B into a more cache-friendly format.
 *
 * This function takes a matrix B and packs it into a format that is optimized
 * for cache access patterns during matrix multiplication. The packing is done
 * in blocks of size KCB x NCB, where KCB and NCB are blocking factors specific
 * to the instruction set IS. The elements within each block are interleaved
 * according to the row interleave factor RI.
 *
 * @tparam IS The instruction set to optimize for.
 * @tparam N The number of columns in the matrix B (default is 512).
 * @tparam K The number of rows in the matrix B (default is 512).
 * @param B Pointer to the input matrix B.
 * @param packedB Pointer to the output packed matrix.
 *
 * @note The function assumes that the dimensions of B (N and K) are multiples
 *       of the blocking factors NCB and KCB, respectively. Padding is not
 *       supported.
 */
template <InstructionSet IS, int N=512, int K=512>
void packB(const int8_t* B, int8_t* packedB) {
  constexpr int NCB = BlockingFactors<IS>::NCB;
  constexpr int KCB = BlockingFactors<IS>::KCB;
  constexpr int RI  = BlockingFactors<IS>::RI;

  // Iterate over B in blocks of KCB rows and NCB columns

  // padding is currently not supported
  static_assert(KCB % RI == 0);
  static_assert(K % KCB == 0);
  static_assert(N % NCB == 0);

  int dstIdx = 0;
  // advance to next tile
  for(size_t rowOffset = 0; rowOffset < K; rowOffset += KCB) {
    for(size_t colOffset = 0; colOffset < N; colOffset += NCB) {
      // move through the block ROW_INTERLEAVE rows at a time, this will be 256/4 = 64 iterations
      for(int ir = 0; ir < KCB; ir += RI) { // move in steps of 4, grab a column of 4 elements
      // process the elements of the sub block
        for(int jc = 0; jc < NCB; jc++) {
          for(int jr = 0; jr < RI; jr++) {
            int srcIdx = (rowOffset + ir + jr) * N + (colOffset + jc);
            packedB[dstIdx++] = B[srcIdx]; // copy the element and just increase the index
          }
        }
      }
    }
  }
}

#if defined(COMPILE_FOR_INTEL)

/**
 * @brief Prints the contents of a 128-bit integer register.
 *
 * This function prints the contents of a 128-bit integer register.
 *
 * @tparam T The type of the elements in the register.
 * @param v The 128-bit integer register to print.
 */
template <typename T>
void print128(__m128i v) {
  alignas(16) T out[16 / sizeof(T)];
  _mm_store_si128((__m128i*)out, v);
  for(int i = 0; i < 16 / sizeof(T); i++)
    std::cout << (int32_t)(T)out[i] << " ";
  std::cout << std::endl;
}

/**
 * @brief Prints the contents of a 256-bit integer register.
 *
 * This function prints the contents of a 256-bit integer register.
 *
 * @tparam T The type of the elements in the register.
 * @param v The 256-bit integer register to print.
 */
template <typename T>
void print256(__m256i v) {
  alignas(32) T out[32 / sizeof(T)];
  _mm256_store_si256((__m256i*)out, v);
  for(int i = 0; i < 32 / sizeof(T); i++)
    std::cout << (int32_t)(T)out[i] << " ";
  std::cout << std::endl;
}

/**
 * @brief Prints the contents of a 512-bit integer register.
 *
 * This function prints the contents of a 512-bit integer register.
 *
 * @tparam T The type of the elements in the register.
 * @param v The 512-bit integer register to print.
 */
template <typename T>
void print512(__m512i v) {
  alignas(64) T out[64 / sizeof(T)];
  _mm512_store_si512((__m512i*)out, v);
  for(int i = 0; i < 64 / sizeof(T); i++)
    std::cout << (int32_t)(T)out[i] << " ";
  std::cout << std::endl;
}

#elif defined(COMPILE_FOR_ARM)
/**
 * @brief Prints the contents of a 128-bit integer register of type int8x16_t.
 *
 * @param v The 128-bit integer register to print.
 */
static inline void print128(int8x16_t v) {
  alignas(16) int8_t out[16];
  vst1q_s8(out, v);
  for(int i = 0; i < 16; i++)
    std::cout << (int32_t)out[i] << " ";
  std::cout << std::endl;
}

/**
 * @brief Prints the contents of a 128-bit integer register of type uint8x16_t.
 *
 * @param v The 128-bit integer register to print.
 */
static inline void print128(uint8x16_t v) {
  alignas(16) uint8_t out[16];
  vst1q_u8(out, v);
  for(int i = 0; i < 16; i++)
    std::cout << (int32_t)out[i] << " ";
  std::cout << std::endl;
}

/**
 * @brief Prints the contents of a 128-bit integer register of type int16x8_t.
 *
 * @param v The 128-bit integer register to print.
 */
static inline void print128(int16x8_t v) {
  alignas(16) int16_t out[8];
  vst1q_s16(out, v);
  for(int i = 0; i < 8; i++)
    std::cout << (int32_t)out[i] << " ";
  std::cout << std::endl;
}

/**
 * @brief Prints the contents of a 128-bit integer register of type int32x4_t.
 *
 * @param v The 128-bit integer register to print.
 */
static inline void print128(int32x4_t v) {
  alignas(16) int32_t out[4];
  vst1q_s32(out, v);
  for(int i = 0; i < 4; i++)
    std::cout << (int32_t)out[i] << " ";
  std::cout << std::endl;
}

/**
 * @brief Prints the contents of a 64-bit integer register of type int16x4_t.
 *
 * @param v The 64-bit integer register to print.
 */
static inline void print64(int16x4_t v) {
  for(int i = 0; i < 4; i++)
    std::cout << (int32_t)reinterpret_cast<int16_t*>(&v)[i] << " ";
  std::cout << std::endl;
}

/**
 * @brief Prints the contents of a 64-bit integer register of type uint16x4_t.
 *
 * @param v The 64-bit integer register to print.
 */
static inline void print64(uint16x4_t v) {
  for(int i = 0; i < 4; i++)
    std::cout << (int32_t)reinterpret_cast<uint16_t*>(&v)[i] << " ";
  std::cout << std::endl;
}

/**
 * @brief Prints the contents of a 64-bit integer register of type uint8x8_t.
 *
 * @param v The 64-bit integer register to print.
 */
static inline void print64(uint8x8_t v) {
  for(int i = 0; i < 8; i++)
    std::cout << (int32_t)reinterpret_cast<uint8_t*>(&v)[i] << " ";
  std::cout << std::endl;
}
#endif

/**
 * @brief Naive reference implementation of matrix multiplication.
 *
 * This function performs a naive reference implementation of matrix multiplication.
 * It multiplies matrices A and B, storing the result in matrix C.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param C Pointer to matrix C.
 * @param M The number of rows in matrix A and C.
 * @param N The number of columns in matrix B and C.
 * @param K The number of columns in matrix A and rows in matrix B.
 */
static inline void gemmInt8Naive(const uint8_t* A, const int8_t* B, int32_t* C, int M=1, int N=512, int K=512) {
  // Initialize C to zero
  std::memset(C, 0, M * N * sizeof(int32_t));

  // Simple three-loop matrix multiplication
  for (int i = 0; i < M; ++i) { // rows of A
    for (int j = 0; j < N; ++j) { // columns of B
      for (int k = 0; k < K; ++k) { // columns of A and rows of B
        C[i * N + j] += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
      }
    }
  }
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian