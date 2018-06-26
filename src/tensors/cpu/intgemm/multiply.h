#pragma once

#include "interleave.h"

#include <cassert>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

namespace intgemm {

/* Define a bunch of intrinstics as overloaded functions so they work with
 * templates.
 */
template <class Register> inline Register set1_epi16(int16_t to);
template <class Register> inline Register set1_ps(float to);
template <class Register> inline Register setzero_si();
#ifdef __SSE2__
inline __m128i add_epi32(__m128i first, __m128i second) {
  return _mm_add_epi32(first, second);
}
inline __m128i adds_epi16(__m128i first, __m128i second) {
  return _mm_adds_epi16(first, second);
}
template <> inline __m128i set1_epi16<__m128i>(int16_t to) {
  return _mm_set1_epi16(to);
}
template <> inline __m128 set1_ps<__m128>(float to) {
  return _mm_set1_ps(to);
}
template <> inline __m128i setzero_si<__m128i>() {
  return _mm_setzero_si128();
}
inline __m128i madd_epi16(__m128i first, __m128i second) {
  return _mm_madd_epi16(first, second);
}
inline __m128i maddubs_epi16(__m128i first, __m128i second) {
  return _mm_maddubs_epi16(first, second);
}
inline __m128i sign_epi8(__m128i first, __m128i second) {
  return _mm_sign_epi8(first, second);
}
inline __m128i abs_epi8(__m128i arg) {
  return _mm_abs_epi8(arg);
}

// Complete any reduction, multiply by scaling, and write to memory.
inline void WriteC(float *to, __m128i pack0123, __m128i pack4567, __m128 unquant_reg) {
  // Convert to float, multiply by unquant, and write.
  *reinterpret_cast<__m128*>(to) = _mm_mul_ps(_mm_cvtepi32_ps(pack0123), unquant_reg);
  *reinterpret_cast<__m128*>(to + 4) = _mm_mul_ps(_mm_cvtepi32_ps(pack4567), unquant_reg);
}
#endif
#ifdef __AVX2__
inline __m256i add_epi32(__m256i first, __m256i second) {
  return _mm256_add_epi32(first, second);
}
inline __m256i adds_epi16(__m256i first, __m256i second) {
  return _mm256_adds_epi16(first, second);
}
template <> inline __m256i set1_epi16<__m256i>(int16_t to) {
  return _mm256_set1_epi16(to);
}
template <> inline __m256 set1_ps<__m256>(float to) {
  return _mm256_set1_ps(to);
}
template <> inline __m256i setzero_si<__m256i>() {
  return _mm256_setzero_si256();
}
inline __m256i madd_epi16(__m256i first, __m256i second) {
  return _mm256_madd_epi16(first, second);
}
inline __m256i maddubs_epi16(__m256i first, __m256i second) {
  return _mm256_maddubs_epi16(first, second);
}
inline __m256i sign_epi8(__m256i first, __m256i second) {
  return _mm256_sign_epi8(first, second);
}
inline __m256i abs_epi8(__m256i arg) {
  return _mm256_abs_epi8(arg);
}

inline void WriteC(float *to, __m256i pack0123, __m256i pack4567, __m256 unquant_reg) {
  // This instruction generates 1s 2s 3s 4s 5f 6f 7f 8f
  __m256i rev = _mm256_permute2f128_si256(pack0123, pack4567, 0x21);
  // This instruction generates 1f 2f 3f 4f 5s 6s 7s 8s
  __m256i blended = _mm256_blend_epi32(pack0123, pack4567, 0xf0);
  __m256i total = _mm256_add_epi32(rev, blended);
  // Convert to float, multiply by unquant, and write.
  *reinterpret_cast<__m256*>(to) = _mm256_mul_ps(_mm256_cvtepi32_ps(total), unquant_reg);
}
#endif
#ifdef __AVX512BW__
inline __m512i add_epi32(__m512i first, __m512i second) {
  return _mm512_add_epi32(first, second);
}
template <> inline __m512i set1_epi16<__m512i>(int16_t to) {
  return _mm512_set1_epi16(to);
}
template <> inline __m512 set1_ps<__m512>(float to) {
  return _mm512_set1_ps(to);
}
template <> inline __m512i setzero_si<__m512i>() {
  return _mm512_setzero_si512();
}
inline __m512i madd_epi16(__m512i first, __m512i second) {
  return _mm512_madd_epi16(first, second);
}
inline __m512i maddubs_epi16(__m512i first, __m512i second) {
  return _mm512_maddubs_epi16(first, second);
}
inline __m512i abs_epi8(__m512i arg) {
  return _mm512_abs_epi8(arg);
}

inline void WriteC(float *to, __m512i pack0123, __m512i pack4567, __m256 unquant_reg) {
  // Form [0th 128-bit register of pack0123, 0st 128-bit register of pack4567, 2nd 128-bit register of pack0123, 2nd 128-bit register of pack4567]
  __m512i mix0 = _mm512_mask_permutex_epi64(pack0123, 0xcc, pack4567, (0 << 4) | (1 << 6));
  // Form [1st 128-bit register of pack0123, 1st 128-bit register of pack4567, 3rd 128-bit register of pack0123, 3rd 128-bit register of pack4567]
  __m512i mix1 = _mm512_mask_permutex_epi64(pack4567, 0x33, pack0123, 2 | (3 << 2));
  __m512i added = _mm512_add_epi32(mix0, mix1);
  // Now we have 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7.
  // Fold register over itself.
  __m256i folded = _mm256_add_epi32(_mm512_castsi512_si256(added), _mm512_extracti64x4_epi64(added, 1));
  *reinterpret_cast<__m256*>(to) = _mm256_mul_ps(_mm256_cvtepi32_ps(folded), unquant_reg);
}
#endif

/* Take 4 registers with 32-bit values to be horizontally added.  Reduce them
 * to one register with 32-bit values in the pattern 1 2 3 4 1 2 3 4, leaving
 * the final addition (which crosses 128-bit lanes) to the caller. */
template <class Register> inline Register Pack0123(Register sum0, Register sum1, Register sum2, Register sum3) {
  // 1 2 1 2 1 2 1 2
  Interleave32(sum0, sum1);
  Register pack01 = add_epi32(sum0, sum1);
  // 3 4 3 4 3 4 3 4
  Interleave32(sum2, sum3);
  Register pack23 = add_epi32(sum2, sum3);
  Interleave64(pack01, pack23);
  // 1 2 3 4 1 2 3 4
  return add_epi32(pack01, pack23);
}

// 16-bit multiplier for SSE2, AVX2, and AVX512.
// C = A * B * unquant_mult
//
// This has been substantially revised from Jacob Devlin's SSE code which is:
// Copyright (c) 2017 Microsoft Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// A is a row-major quantized matrix (from PrepareA)
// B is a rearranged quantized matrix (from PrepareB)
// C is output in row-major form.
//
// All of A, B, and C must be in aligned to a multiple of the register size:
// SSE2: 16 bytes
// AVX2: 32 bytes
// AVX512: 64 bytes.
//
// A_rows can be anything non-negative.
// width must be a multiple of the register size.
// B_cols must be a multiple of 8.
template <class Integer, class Float> inline void Multiply16(const int16_t *A, const int16_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) {
  assert(width % (sizeof(Integer) / sizeof(int16_t)) == 0);
  assert(B_cols % 8 == 0);
  assert(reinterpret_cast<uintptr_t>(A) % sizeof(Integer) == 0);
  assert(reinterpret_cast<uintptr_t>(B) % sizeof(Integer) == 0);
  assert(reinterpret_cast<uintptr_t>(C) % sizeof(Integer) == 0);
  const int simd_width = width / (sizeof(Integer) / sizeof(int16_t));
  const Float unquant_reg = set1_ps<Float>(unquant_mult);
  const Integer *B0_col = reinterpret_cast<const Integer *>(B);
  for (int B0_colidx = 0; B0_colidx < B_cols; B0_col += 8 * simd_width, B0_colidx += 8) {
    // Process one row of A at a time.  Doesn't seem to be faster to do multiple rows of A at once.
    for (int A_rowidx = 0; A_rowidx < A_rows; ++A_rowidx) {
      const Integer *A_row = reinterpret_cast<const Integer*>(A + A_rowidx * width);
      // These will be packed 32-bit integers containing sums for each row of B multiplied by the row of A.
      // Iterate over shared (inner) dimension.
      int k = 0;
      Integer a = *(A_row + k);
      Integer sum0 = madd_epi16(a, *(B0_col + k * 8));
      Integer sum1 = madd_epi16(a, *(B0_col + k * 8 + 1));
      Integer sum2 = madd_epi16(a, *(B0_col + k * 8 + 2));
      Integer sum3 = madd_epi16(a, *(B0_col + k * 8 + 3));
      Integer sum4 = madd_epi16(a, *(B0_col + k * 8 + 4));
      Integer sum5 = madd_epi16(a, *(B0_col + k * 8 + 5));
      Integer sum6 = madd_epi16(a, *(B0_col + k * 8 + 6));
      Integer sum7 = madd_epi16(a, *(B0_col + k * 8 + 7));
      for (int k = 1; k < simd_width; ++k) {
        Integer a = *(A_row + k);
        // Multiply 16-bit, horizontally add to packed 32-bit integers.
        Integer mult0 = madd_epi16(a, *(B0_col + k * 8));
        Integer mult1 = madd_epi16(a, *(B0_col + k * 8 + 1));
        Integer mult2 = madd_epi16(a, *(B0_col + k * 8 + 2));
        Integer mult3 = madd_epi16(a, *(B0_col + k * 8 + 3));
        Integer mult4 = madd_epi16(a, *(B0_col + k * 8 + 4));
        Integer mult5 = madd_epi16(a, *(B0_col + k * 8 + 5));
        Integer mult6 = madd_epi16(a, *(B0_col + k * 8 + 6));
        Integer mult7 = madd_epi16(a, *(B0_col + k * 8 + 7));
        // Sum packed 32-bit integers with danger of overflow.  TODO: accumulate in 64-bit every so often.
        sum0 = add_epi32(sum0, mult0);
        sum1 = add_epi32(sum1, mult1);
        sum2 = add_epi32(sum2, mult2);
        sum3 = add_epi32(sum3, mult3);
        sum4 = add_epi32(sum4, mult4);
        sum5 = add_epi32(sum5, mult5);
        sum6 = add_epi32(sum6, mult6);
        sum7 = add_epi32(sum7, mult7);
      }
      // Reduce sums within 128-bit lanes.
      Integer pack0123 = Pack0123(sum0, sum1, sum2, sum3);
      Integer pack4567 = Pack0123(sum4, sum5, sum6, sum7);
      // The specific implementation may need to reduce further.
      WriteC(C + A_rowidx * B_cols + B0_colidx, pack0123, pack4567, unquant_reg);
    }
  }
}

/* This is the C version of the below (for AVX2)
 *void AVX2_8Bit::Multiply(const __m256i *A, const __m256i *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) {
 *  assert(width % 32 == 0);
 *  assert(reinterpret_cast<uintptr_t>(A) % 32 == 0);
 *  assert(reinterpret_cast<uintptr_t>(B) % 32 == 0);
 *  assert(reinterpret_cast<uintptr_t>(C) % 32 == 0);
 *  assert(num_B_rows % 8 == 0);
 *  __m256 unquant_reg = _mm256_set1_ps(unquant_mult);
 *  const int simd_width = width / 32;
 *  int B0_rowidx = 0;
 *  // Go over 8 rows of B at a time. 
 *  for (const __m256i *B0_row = B; B0_rowidx != num_B_rows; B0_row += 8 * simd_width, B0_rowidx += 8) {
 *    // Process one row of A at a time.  Doesn't seem to be faster to do multiple rows of A at once.
 *    for (int A_rowidx = 0; A_rowidx < A_rows; ++A_rowidx) {
 *      const __m256i *A_row = A + A_rowidx * simd_width;
 *      // These will be packed 16-bit integers containing sums for each row of B multiplied by the row of A.
 *      __m256i sum0 = _mm256_setzero_si256();
 *      __m256i sum1 = _mm256_setzero_si256();
 *      __m256i sum2 = _mm256_setzero_si256();
 *      __m256i sum3 = _mm256_setzero_si256();
 *      __m256i sum4 = _mm256_setzero_si256();
 *      __m256i sum5 = _mm256_setzero_si256();
 *      __m256i sum6 = _mm256_setzero_si256();
 *      __m256i sum7 = _mm256_setzero_si256();
 *      // Iterate over shared (inner) dimension.
 *      for (int k = 0; k < simd_width; ++k) {
 *        // Read in 64 8-bit signed integers from A.
 *        __m256i a = *(A_row + k);
 *        // Negate 8-bit values in b if the corresponding a was negative.
 *        // Negation is implemented by subtraction from zero.
 *        __m256i b0 = _mm256_sign_epi8(*(B0_row + k * 8), a);
 *        __m256i b1 = _mm256_sign_epi8(*(B0_row + k * 8 + 1), a);
 *        __m256i b2 = _mm256_sign_epi8(*(B0_row + k * 8 + 2), a);
 *        __m256i b3 = _mm256_sign_epi8(*(B0_row + k * 8 + 3), a);
 *        __m256i b4 = _mm256_sign_epi8(*(B0_row + k * 8 + 4), a);
 *        __m256i b5 = _mm256_sign_epi8(*(B0_row + k * 8 + 5), a);
 *        __m256i b6 = _mm256_sign_epi8(*(B0_row + k * 8 + 6), a);
 *        __m256i b7 = _mm256_sign_epi8(*(B0_row + k * 8 + 7), a);
 *        __m256i a_positive = _mm256_abs_epi8(a);
 *        // Multiply 8-bit unsigned * signed, horizontally add to packed 16-bit integers.
 *        __m256i mult0 = _mm256_maddubs_epi16(a_positive, b0);
 *        __m256i mult1 = _mm256_maddubs_epi16(a_positive, b1);
 *        __m256i mult2 = _mm256_maddubs_epi16(a_positive, b2);
 *        __m256i mult3 = _mm256_maddubs_epi16(a_positive, b3);
 *        __m256i mult4 = _mm256_maddubs_epi16(a_positive, b4);
 *        __m256i mult5 = _mm256_maddubs_epi16(a_positive, b5);
 *        __m256i mult6 = _mm256_maddubs_epi16(a_positive, b6);
 *        __m256i mult7 = _mm256_maddubs_epi16(a_positive, b7);
 *        // Sum packed 16-bit integers with saturation.
 *        // With larger matrices there is a danger of saturating so TODO upcast to 32-bit every so often.
 *        sum0 = _mm256_adds_epi16(mult0, sum0);
 *        sum1 = _mm256_adds_epi16(mult1, sum1);
 *        sum2 = _mm256_adds_epi16(mult2, sum2);
 *        sum3 = _mm256_adds_epi16(mult3, sum3);
 *        sum4 = _mm256_adds_epi16(mult4, sum4);
 *        sum5 = _mm256_adds_epi16(mult5, sum5);
 *        sum6 = _mm256_adds_epi16(mult6, sum6);
 *        sum7 = _mm256_adds_epi16(mult7, sum7);
 *      }
 *      // Write to C.
 *      __m256i combined = Reduce16to32(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
 *      *reinterpret_cast<__m256*>(C + A_rowidx * num_B_rows + B0_rowidx) = _mm256_mul_ps(_mm256_cvtepi32_ps(combined), unquant_reg);
 *    }
 *  }
 *}
*/

/* 8-bit matrix multiply used by AVX and AVX2.
 * These have two peculiar properties:
 * 1. The sign instructions don't exist in AVX512.
 * 2. 16 registers means gcc's register allocation failed so I wrote it in my
 *    own asm.
 * 3. They support 3-argument vpsignb and vpmaddubsw.
 *
 * Fun fact: AVX introduced the three-argument vpsignb and vpmaddubsw but only
 * for 128-bit, despite the primary change in AVX being the addition of
 * 256-bit.  We had to wait for AVX2 to get 256-bit versions of vpsignb and
 * vpmaddubsw.  That's why this code is generic over 128-bit or 256-bit.
 */
struct Multiply8_AVXAVX2 {
  template <class Integer> inline static void Inner(
      Integer a, const Integer *b,
      Integer &sum0, Integer &sum1, Integer &sum2, Integer &sum3,
      Integer &sum4, Integer &sum5, Integer &sum6, Integer &sum7) {
    // Annoyingly the only 8-bit multiply is signed * unsigned (maddubs).
    // So we take the sign bits off of a and apply them each b in a * b.
    //
    // We have only 16 YMM registers but we want to store:
    // 1 for a (or |a|)
    // 8 temporaries for applying sign to each column of B.
    // 8 sums.
    //
    // gcc's register allocator does:
    // 1 for a, do all the sign application, then overwrite with |a|
    // 8 temporaries
    // 7 sums in registers + 1 on the stack
    //
    // But it's possible to complete an operation early, freeing up its
    // temporary register for reuse.  But completing an operation early
    // requires us to have |a| for vpmaddubsw while completing the later
    // operation needs a again to apply sign.
    //
    // So we do two columns, 0 and 1, early.  This allows b0_b6 and b1_b7
    // to be reused by columns 6 and 7, respectively.  And there's enough
    // registers to store both a and |a|.
    //
    // These are the temporary variables used to process each column of b.
    // We let the compiler choose which register number is which, but force
    // it to allocate all registers.
    Integer absa;
    Integer b0_b6, b1_b7, b2, b3, b4, b5;
    // Maybe this will tell gcc that we're accessing 8 registers starting
    // at B_live.  Though I doubt it because we're passing the address as a
    // register.
    typedef struct { Integer x[8]; } B_range;
    asm(
        // Copy the first 6 columns of b to registers.  We assume B has
        // been rearranged so that these 8 columns are consecutive.
        // vpsignb does not take a memory address as its second argument,
        // so this can't be inlined into vsignb.
        "vmovdqa          (%[B]), %[b0_b6]\n"
        "vmovdqa   %c[size](%[B]), %[b1_b7]\n"
        // These multiplies are executed by the assembler, not by the CPU
        // at run time.
        // I would have liked to just initialize b2 etc above but that
        // would make it an input argument "+x" instead of "=&x".  And +x
        // counts as two operands for purposes of gcc's annoying 30-operand
        // limit.
        "vmovdqa 2*%c[size](%[B]), %[b2]\n"
        "vmovdqa 3*%c[size](%[B]), %[b3]\n"
        "vmovdqa 4*%c[size](%[B]), %[b4]\n"
        "vmovdqa 5*%c[size](%[B]), %[b5]\n"
        // Store the absolute value of a in absa.
        "vpabsb  %[a], %[absa]\n"
        // If a byte of a is negative, negate the corresponding byte in
        // b0_b6 etc.
        "vpsignb %[a], %[b0_b6], %[b0_b6]\n"
        "vpsignb %[a], %[b1_b7], %[b1_b7]\n"
        // Multiply signed * unsigned then horizontally add to form packed
        // 16-bit integers:
        // b0[0] * |a|[0] + b0[1] * |a|[1], b0[2] * |a|[2] + b0[3] * |a|[3], ...
        "vpmaddubsw %[b0_b6], %[absa], %[b0_b6]\n"
        "vpmaddubsw %[b1_b7], %[absa], %[b1_b7]\n"
        // vpmaddubsw has latency 5 so work on some other sign bits while
        // we're at it.
        "vpsignb %[a], %[b2], %[b2]\n"
        "vpsignb %[a], %[b3], %[b3]\n"
        "vpsignb %[a], %[b4], %[b4]\n"
        "vpsignb %[a], %[b5], %[b5]\n"
        // Perform a 16-bit add with saturation to accumlate sums.
        "vpaddsw %[b0_b6], %[sum0], %[sum0]\n"
        // Now we can reuse b0_b6 for b6
        "vmovdqa 6*%c[size](%[B]), %[b0_b6]\n"
        "vpaddsw %[b1_b7], %[sum1], %[sum1]\n"
        // Now we can reuse b1_b7 for b7
        "vmovdqa 7*%c[size](%[B]), %[b1_b7]\n"
        // More crunching while the load happens.
        "vpmaddubsw %[b2], %[absa], %[b2]\n"
        "vpmaddubsw %[b3], %[absa], %[b3]\n"
        "vpmaddubsw %[b4], %[absa], %[b4]\n"
        "vpsignb %[a], %[b0_b6], %[b0_b6]\n"
        "vpsignb %[a], %[b1_b7], %[b1_b7]\n"
        "vpmaddubsw %[b5], %[absa], %[b5]\n"
        "vpmaddubsw %[b0_b6], %[absa], %[b0_b6]\n"
        "vpmaddubsw %[b1_b7], %[absa], %[b1_b7]\n"
        "vpaddsw %[b2], %[sum2], %[sum2]\n"
        "vpaddsw %[b3], %[sum3], %[sum3]\n"
        "vpaddsw %[b4], %[sum4], %[sum4]\n"
        "vpaddsw %[b5], %[sum5], %[sum5]\n"
        "vpaddsw %[b0_b6], %[sum6], %[sum6]\n"
        "vpaddsw %[b1_b7], %[sum7], %[sum7]\n"
        : [sum0] "+x" (sum0),
          [sum1] "+x" (sum1),
          [sum2] "+x" (sum2),
          [sum3] "+x" (sum3),
          [sum4] "+x" (sum4),
          [sum5] "+x" (sum5),
          [sum6] "+x" (sum6),
          [sum7] "+x" (sum7),
          [b0_b6] "=&x" (b0_b6),
          [b1_b7] "=&x" (b1_b7),
          [b2] "=&x" (b2),
          [b3] "=&x" (b3),
          [b4] "=&x" (b4),
          [b5] "=&x" (b5),
          [absa] "=&x" (absa)
        : 
          // I would like to use m here but that non-deterministically
          // chooses %(eax) or -256$(eax) and there's no way to add to that
          // memory address:
          // https://gcc.gnu.org/ml/gcc-help/2011-04/msg00518.html
          //
          [B] "r" (reinterpret_cast<const B_range*>(b)),
          [a] "x" (a),
          [size] "i" (sizeof(Integer))
      );
  }
};

// For SSSE3 without AVX
struct Multiply8_C {
  template <class Integer> inline static void Inner(
      Integer a, const Integer *b,
      Integer &sum0, Integer &sum1, Integer &sum2, Integer &sum3,
      Integer &sum4, Integer &sum5, Integer &sum6, Integer &sum7) {
    Integer a_positive = abs_epi8(a);
    sum0 = adds_epi16(sum0, maddubs_epi16(a_positive, sign_epi8(b[0], a)));
    sum1 = adds_epi16(sum1, maddubs_epi16(a_positive, sign_epi8(b[1], a)));
    sum2 = adds_epi16(sum2, maddubs_epi16(a_positive, sign_epi8(b[2], a)));
    sum3 = adds_epi16(sum3, maddubs_epi16(a_positive, sign_epi8(b[3], a)));
    sum4 = adds_epi16(sum4, maddubs_epi16(a_positive, sign_epi8(b[4], a)));
    sum5 = adds_epi16(sum5, maddubs_epi16(a_positive, sign_epi8(b[5], a)));
    sum6 = adds_epi16(sum6, maddubs_epi16(a_positive, sign_epi8(b[6], a)));
    sum7 = adds_epi16(sum7, maddubs_epi16(a_positive, sign_epi8(b[7], a)));
  }
};

template <class Algo, class Integer, class Float> inline void Multiply8_SSE2OrAVX2(const int8_t *A, const int8_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) {
  assert(width % sizeof(Integer) == 0);
  assert(B_cols % 8 == 0);
  assert(reinterpret_cast<uintptr_t>(A) % sizeof(Integer) == 0);
  assert(reinterpret_cast<uintptr_t>(B) % sizeof(Integer) == 0);
  assert(reinterpret_cast<uintptr_t>(C) % sizeof(Integer) == 0);
  Float unquant_reg = set1_ps<Float>(unquant_mult);
  const int simd_width = width / sizeof(Integer);
  const Integer *B0_col = reinterpret_cast<const Integer*>(B);
  // Go over 8 columns of B at a time.
  for (int B0_colidx = 0; B0_colidx != B_cols; B0_col += 8 * simd_width, B0_colidx += 8) {
    // Process one row of A at a time.  Doesn't seem to be faster to do multiple rows of A at once.
    for (int A_rowidx = 0; A_rowidx < A_rows; ++A_rowidx) {
      // Iterate over shared (inner) dimension.
      const Integer *A_live = reinterpret_cast<const Integer *>(A + A_rowidx * width);
      const Integer *A_end = A_live + simd_width;
      const Integer *B_live = B0_col;

      // Rather than initializing as zeros and adding, just initialize the first.
      Integer a = *(A_live++);
      Integer a_positive = abs_epi8(a);
      // These will be packed 16-bit integers containing sums for each column of B multiplied by the row of A.
      Integer sum0 = maddubs_epi16(a_positive, sign_epi8(B_live[0], a));
      Integer sum1 = maddubs_epi16(a_positive, sign_epi8(B_live[1], a));
      Integer sum2 = maddubs_epi16(a_positive, sign_epi8(B_live[2], a));
      Integer sum3 = maddubs_epi16(a_positive, sign_epi8(B_live[3], a));
      Integer sum4 = maddubs_epi16(a_positive, sign_epi8(B_live[4], a));
      Integer sum5 = maddubs_epi16(a_positive, sign_epi8(B_live[5], a));
      Integer sum6 = maddubs_epi16(a_positive, sign_epi8(B_live[6], a));
      Integer sum7 = maddubs_epi16(a_positive, sign_epi8(B_live[7], a));
      B_live += 8;

      // Use A as the loop variable so the add can be done where gcc likes it
      // for branch prediction.
      for (; A_live != A_end; ++A_live, B_live += 8) {
        Algo::Inner(*A_live, B_live, sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
      }
      /* Convert 16-bit to 32-bit and add, not caring what parts are added.
       * Implementations:
       * 1. https://github.com/tesseract-ocr/tesseract/blob/master/src/arch/intsimdmatrixavx2.cpp#L67 under Apache license:
       *   This does a multiply by 1 and horizontal add:
       *    _mm512_madd_epi16(sum, _mm512_set1_epi16(1))
       *   Current fastest.
       *
       * 2. Signed extension and fold halves:
       *    sum = _mm512_add_epi32(
       *      _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum)),
       *      _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum, 1)));
       *
       * 3. Sign extend by abuse of bitshift, then add.
       * sum = _mm512_add_epi32(
       *      _mm512_srai_epi32(_mm512_slli_epi32(sum, 16), 16),
       *      _mm512_srai_epi32(sum, 16));
       */
      Integer ones = set1_epi16<Integer>(1);
      sum0 = madd_epi16(sum0, ones);
      sum1 = madd_epi16(sum1, ones);
      sum2 = madd_epi16(sum2, ones);
      sum3 = madd_epi16(sum3, ones);
      sum4 = madd_epi16(sum4, ones);
      sum5 = madd_epi16(sum5, ones);
      sum6 = madd_epi16(sum6, ones);
      sum7 = madd_epi16(sum7, ones);
      Integer pack0123 = Pack0123(sum0, sum1, sum2, sum3);
      Integer pack4567 = Pack0123(sum4, sum5, sum6, sum7);
      WriteC(C + A_rowidx * B_cols + B0_colidx, pack0123, pack4567, unquant_reg);
    }
  }
}

} // namespace intgemm
