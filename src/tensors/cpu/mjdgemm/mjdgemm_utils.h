#pragma once

#include <cstring>
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <immintrin.h>
#include <emmintrin.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <type_traits>
#include <algorithm>
#include <limits>

#include "common/logging.h"

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Enumeration of supported instruction sets.
 */
enum class InstructionSet : int {
  None = 1,
#if defined(__x86_64__) || defined(_M_X64)
  SSE4_2 = 2,
  AVX = 3,
  AVX2 = 4,
  AVX512 = 5
#elif defined(__aarch64__)
  NEON = 2
#endif
};

#define compute_case(M) \
  case M: \
    computeBlock<IS, N, K, M>(blockA, blockB, blockC); \
    break;

/**
 * @brief Struct to hold the blocking factors specific to the instruction set.
 */
template <InstructionSet IS>
struct BlockingFactors {};

#if defined(__x86_64__) || defined(_M_X64)
template <>
struct BlockingFactors<InstructionSet::AVX512> {
  static constexpr int NCB  = 32;
  static constexpr int KCB  = 256;
  static constexpr int NR   = 16;
  static constexpr int RI   = 4;
  static constexpr int VLEN = 16;

  static constexpr int MCB  = 56;
  static constexpr int MR   = 14;
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
  static constexpr int MCB = 56;
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
#endif

#if defined(__aarch64__)
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
#endif

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

#if defined(__x86_64__) || defined(_M_X64)

#ifdef _MSC_VER
#define CPUID(info, x) __cpuidex(info, x, 0)
#else
#define CPUID(info, x) __cpuid_count(x, 0, info[0], info[1], info[2], info[3])
#endif

/**
 * @brief Checks if AVX-512 is supported on the current CPU.
 *
 * @return True if AVX-512 is supported, false otherwise.
 */
static inline bool isAVX512Supported() {
  int info[4];
  CPUID(info, 7);
  return (info[1] & ((int)1 << 16)) != 0; // Check EBX register (info[1]) for AVX-512 support
}

/**
 * @brief Checks if AVX2 is supported on the current CPU.
 *
 * @return True if AVX2 is supported, false otherwise.
 */
static inline bool isAVX2Supported() {
  int info[4];
  CPUID(info, 7);
  return (info[1] & ((int)1 << 5)) != 0; // Check EBX register (info[1]) for AVX2 support
}

/**
 * @brief Checks if AVX is supported on the current CPU.
 *
 * @return True if AVX is supported, false otherwise.
 */
static inline bool isAVXSupported() {
  int info[4];
  CPUID(info, 1);
  return (info[2] & ((int)1 << 28)) != 0; // Check ECX register (info[2]) for AVX support
}

/**
 * @brief Checks if SSE4.2 is supported on the current CPU.
 *
 * @return True if SSE4.2 is supported, false otherwise.
 */
static inline bool isSSE4_2Supported() {
  int info[4];
  CPUID(info, 1);
  return (info[2] & ((int)1 << 20)) != 0; // Check ECX register (info[2]) for SSE4.2 support
}

#endif // defined(__x86_64__) || defined(_M_X64)

/**
 * @brief Gets the highest supported instruction set on the current CPU.
 *
 * This function checks the highest supported instruction set on the current CPU and returns it.
 * It also allows overriding the detected instruction set via the environment variable MJD_ENABLE_INSTRUCTIONS.
 *
 * @return The highest supported instruction set.
 */
static inline InstructionSet getSupportedInstructionSet() {
  // before testing, check the value of the environment variable
  // MJD_ENABLE_INSTRUCTIONS and return the corresponding instruction set

  char* env = std::getenv("MJD_ENABLE_INSTRUCTIONS");
  InstructionSet requested = InstructionSet::None;

  if(env) {
    std::string instrSet(env);
#if defined(__aarch64__)
    if(instrSet == "NEON")
      requested = InstructionSet::NEON;
#else
    if(instrSet == "AVX512")
      requested = InstructionSet::AVX512;
    else if(instrSet == "AVX2")
      requested = InstructionSet::AVX2;
    else if(instrSet == "AVX")
      requested = InstructionSet::AVX;
    else if(instrSet == "SSE4_2")
      requested = InstructionSet::SSE4_2;
#endif
    else if((instrSet == "None") || (instrSet == "NONE"))
      requested = InstructionSet::None;
    else
      ABORT("Unknown requested instruction set {}", env);

    LOG(warn, "mjdgemm: Requested instructions via MJD_ENABLE_INSTRUCTIONS={}", env);
  }

  InstructionSet highest = InstructionSet::None;

#if defined(__x86_64__) || defined(_M_X64)
  if(isAVX512Supported())
    highest = InstructionSet::AVX512;
  else if(isAVX2Supported())
    highest = InstructionSet::AVX2;
  else if(isAVXSupported())
    highest = InstructionSet::AVX;
  else if(isSSE4_2Supported())
    highest = InstructionSet::SSE4_2;
  else
    highest = InstructionSet::None;
#endif

#if defined(__aarch64__)
  highest = InstructionSet::NEON;
#endif

  ABORT_IF(requested != InstructionSet::None && static_cast<int>(requested) > static_cast<int>(highest),
          "mjdgemm: Instructions selected via MJD_ENABLE_INSTRUCTIONS={} not available on this machine", env);

  return env ? requested : highest;
}

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

#if defined(__x86_64__) || defined(_M_X64)

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

#elif defined(__aarch64__)
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