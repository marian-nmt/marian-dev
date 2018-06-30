#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

#include <cassert>
#include <stdint.h>

namespace intgemm {

/* This macro defines functions that interleave their arguments like
 * inline void Interleave8(__m256i &first, __m256i &second) {
 *   __m256i temp = _mm256_unpacklo_epi8(first, second);
 *   second = _mm256_unpackhi_epi8(first, second);
 *   first = temp;
 * }
 *
 * Example usage:
 *   INTGEMM_INTERLEAVE(__m128i, )
 *   INTGEMM_INTERLEAVE(__m256i, 256)
 *   INTGEMM_INTERLEAVE(__m512i, 512)
 */
#define INTGEMM_INTERLEAVE(type, prefix) \
static inline void Interleave8(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi8(first, second); \
  second = _mm##prefix##_unpackhi_epi8(first, second); \
  first = temp; \
} \
static inline void Interleave16(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi16(first, second); \
  second = _mm##prefix##_unpackhi_epi16(first, second); \
  first = temp; \
} \
static inline void Interleave32(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi32(first, second); \
  second = _mm##prefix##_unpackhi_epi32(first, second); \
  first = temp; \
} \
static inline void Interleave64(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi64(first, second); \
  second = _mm##prefix##_unpackhi_epi64(first, second); \
  first = temp; \
}


template <class Register> static inline Register setzero_si() __attribute__((always_inline));;
#ifdef __SSE2__
INTGEMM_INTERLEAVE(__m128i, )
template <> inline __m128i setzero_si<__m128i>() {
  return _mm_setzero_si128();
}
#endif
#ifdef __AVX2__
INTGEMM_INTERLEAVE(__m256i, 256)
template <> inline __m256i setzero_si<__m256i>() {
  return _mm256_setzero_si256();
}
#endif
#ifdef __AVX512F__
INTGEMM_INTERLEAVE(__m512i, 512)
template <> inline __m512i setzero_si<__m512i>() {
  return _mm512_setzero_si512();
}
#endif

template <class Register> static inline void Swap(Register &a, Register &b) {
  Register tmp = a;
  a = b;
  b = tmp;
}

/* Transpose registers containing 8 packed 16-bit integers.
 * Each 128-bit lane is handled independently.
 */
template <class Register> static inline void Transpose16InLane(Register &r0, Register &r1, Register &r2, Register &r3, Register &r4, Register &r5, Register &r6, Register &r7) {
  // r0: columns 0 1 2 3 4 5 6 7 from row 0
  // r1: columns 0 1 2 3 4 5 6 7 from row 1

  Interleave16(r0, r1);
  Interleave16(r2, r3);
  Interleave16(r4, r5);
  Interleave16(r6, r7);
  // r0: columns 0 0 1 1 2 2 3 3 from rows 0 and 1
  // r1: columns 4 4 5 5 6 6 7 7 from rows 0 and 1
  // r2: columns 0 0 1 1 2 2 3 3 from rows 2 and 3
  // r3: columns 4 4 5 5 6 6 7 7 from rows 2 and 3
  // r4: columns 0 0 1 1 2 2 3 3 from rows 4 and 5
  // r5: columns 4 4 5 5 6 6 7 7 from rows 4 and 5
  // r6: columns 0 0 1 1 2 2 3 3 from rows 6 and 7
  // r7: columns 4 4 5 5 6 6 7 7 from rows 6 and 7

  Interleave32(r0, r2);
  Interleave32(r1, r3);
  Interleave32(r4, r6);
  Interleave32(r5, r7);
  // r0: columns 0 0 0 0 1 1 1 1 from rows 0, 1, 2, and 3
  // r1: columns 4 4 4 4 5 5 5 5 from rows 0, 1, 2, and 3
  // r2: columns 2 2 2 2 3 3 3 3 from rows 0, 1, 2, and 3
  // r3: columns 6 6 6 6 7 7 7 7 from rows 0, 1, 2, and 3
  // r4: columns 0 0 0 0 1 1 1 1 from rows 4, 5, 6, and 7
  // r5: columns 4 4 4 4 5 5 5 5 from rows 4, 5, 6, and 7
  // r6: columns 2 2 2 2 3 3 3 3 from rows 4, 5, 6, and 7
  // r7: columns 6 6 6 6 7 7 7 7 from rows 4, 5, 6, and 7

  Interleave64(r0, r4);
  Interleave64(r1, r5);
  Interleave64(r2, r6);
  Interleave64(r3, r7);
  // r0: columns 0 0 0 0 0 0 0 0 from rows 0 through 7
  // r1: columns 4 4 4 4 4 4 4 4 from rows 0 through 7
  // r2: columns 2 2 2 2 2 2 2 2 from rows 0 through 7
  // r3: columns 6 6 6 6 6 6 6 6 from rows 0 through 7
  // r4: columns 1 1 1 1 1 1 1 1 from rows 0 through 7
  // r5: columns 5 5 5 5 5 5 5 5 from rows 0 through 7
  
  // Empirically gcc is able to remove these movs and just rename the outputs of Interleave64.
  Swap(r1, r4);
  Swap(r3, r6);
}

/* Tranpose registers containing 16 packed 8-bit integers.
 * Each 128-bit lane is handled independently.
 */
template <class Register> static inline void Transpose8InLane(
    Register &r0, Register &r1, Register &r2, Register &r3, Register &r4, Register &r5, Register &r6, Register &r7,
    Register &r8, Register &r9, Register &r10, Register &r11, Register &r12, Register &r13, Register &r14, Register &r15) {
  // Get 8-bit values to 16-bit values so they can travel together.
  Interleave8(r0, r1);
  // r0: columns 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 from rows 0 and 1.
  // r1: columns 8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15 from rows 0 and 1.
  Interleave8(r2, r3);
  // r2: columns 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 from rows 2 and 3.
  Interleave8(r4, r5);
  Interleave8(r6, r7);
  Interleave8(r8, r9);
  Interleave8(r10, r11);
  Interleave8(r12, r13);
  Interleave8(r14, r15);
  Transpose16InLane(r0, r2, r4, r6, r8, r10, r12, r14);
  Transpose16InLane(r1, r3, r5, r7, r9, r11, r13, r15);
  // Permute into correct order.  This is free because the outputs just get pemuted.
  Register tmp;
  tmp = r2;
  r2 = r4;
  r4 = r8;
  r8 = r1;
  r1 = tmp;
  tmp = r3;
  r3 = r6;
  r6 = r12;
  r12 = r9;
  r9 = tmp;
  tmp = r5;
  r5 = r10;
  r10 = tmp;
  tmp = r7;
  r7 = r14;
  r14 = r13;
  r13 = r11;
  r11 = tmp;
}

// PREPARE B: quantize and rearrange.  B is presumed to be constantparameters
// so we can take our time rearranging it in order to save during the multiply.
//
// We presume B starts in row-major order.
//
// In AVX2, a register holds 32 8-bit values or 16 16-bit values and we want
// that many values from the same column in the register.
//
// The multiplier reads 8 rows at a time and we want these reads to be
// contiguous.
//
// Each 8x32 (for 8-bit) or 8x16 (for 16-bit) tile of B is transposed.
// The tiles are stored in column major order.
//
// For AVX2, this matrix shows what index each value of B will be stored at:
//   0  16 ... 240
//   1  17 ... 241
//   2  18 ... 242
//   3  19 ... 243
//   4  20 ... 244
//   5  21 ... 245
//   6  22 ... 246
//   7  23 ... 247
//   8  24 ... 248
//   9  25 ... 249
//  10  26 ... 250
//  11  27 ... 251
//  12  28 ... 252
//  13  29 ... 253
//  14  30 ... 254
//  15  31 ... 255
// 256 272
// 257 273
// ... ...
template <class Quantizer> static inline void PrepareBFor8(const float *input, int8_t *output_shadow, Quantizer q, int rows, int cols) {
  typedef typename Quantizer::Integer Register;
  // Currently all multipliers have a stride of 8 columns.
  const int kColStride = 8;
  assert(cols % kColStride == 0);
  assert(rows % sizeof(Register) == 0);
  assert(reinterpret_cast<uintptr_t>(input) % sizeof(Register) == 0);
  Register *output = reinterpret_cast<Register*>(output_shadow);
  assert(reinterpret_cast<uintptr_t>(output) % sizeof(Register) == 0);

  for (int c = 0; c < cols; c += kColStride) {
    for (int r = 0; r < rows; r += sizeof(Register), output += 8) {
      // Quantize and perform a transpose with height sizeof(Register) and width 8.
      // This isn't quite Transpose8InLane because it's half the number of columns,
      // so each register starts with two rows instead of being one row.
      // The quantizers know to skip a row.
      output[0] = q.ForReshape(input + cols * (r    ) + c, cols);
      output[1] = q.ForReshape(input + cols * (r + 1) + c, cols);
      output[2] = q.ForReshape(input + cols * (r + 4) + c, cols);
      output[3] = q.ForReshape(input + cols * (r + 5) + c, cols);
      output[4] = q.ForReshape(input + cols * (r + 8) + c, cols);
      output[5] = q.ForReshape(input + cols * (r + 9) + c, cols);
      output[6] = q.ForReshape(input + cols * (r + 12) + c, cols);
      output[7] = q.ForReshape(input + cols * (r + 13) + c, cols);
      Interleave8(output[0], output[1]);
      Interleave8(output[2], output[3]);
      Interleave8(output[4], output[5]);
      Interleave8(output[6], output[7]);
      Transpose16InLane(output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7]);
    }
  }
}

template <class Quantizer> static inline void PrepareBFor16(const float *input, int16_t *output_shadow, Quantizer q, int rows, int cols) {
  typedef typename Quantizer::Integer Register;
  assert(cols % 8 == 0);
  assert(rows % (sizeof(Register) / sizeof(int16_t)) == 0);
  assert(reinterpret_cast<uintptr_t>(input) % sizeof(Register) == 0);
  Register *output = reinterpret_cast<Register*>(output_shadow);
  assert(reinterpret_cast<uintptr_t>(output) % sizeof(Register) == 0);

  for (int c = 0; c < cols; c += 8) {
    for (int r = 0; r < rows; r += (sizeof(Register) / sizeof(int16_t)), output += 8) {
      // gcc unrolls this loop and uses registers for output[k]
      for (int k = 0; k < 8; ++k) {
        output[k] = q.ForReshape(input + cols * (r + k) + c, cols);
      }
      Transpose16InLane(output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7]);
    }
  }
}

/* Select columns of B from PrepareB format to PrepareB format.
 */
template <class Register> static inline void SelectColumnsOfB(const Register *input, Register *output, int rows_bytes /* number of bytes in a row */, const std::size_t *cols_begin, const std::size_t *cols_end) {
  // Do columns for multiples of 8.
  int register_rows = rows_bytes / sizeof(Register);
  const std::size_t *cols_end8 = cols_begin + ((cols_end - cols_begin) & ~7);
  const Register *starts[8];
  for (; cols_begin != cols_end8; cols_begin += 8) {
    for (int k = 0; k < 8; ++k) {
      starts[k] = input + (cols_begin[k] & 7) + (cols_begin[k] & ~7) * register_rows;
    }
    for (int r = 0; r < register_rows; ++r) {
      for (int k = 0; k < 8; ++k) {
        *(output++) = *starts[k];
        starts[k] += 8;
      }
    }
  }
}

} // namespace intgemm
