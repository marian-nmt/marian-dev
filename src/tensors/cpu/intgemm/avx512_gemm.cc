#include "avx512_gemm.h"
#include "interleave.h"
#include "multiply.h"

#include <cassert>
#include <cstddef>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

namespace intgemm {

#ifdef __AVX512BW__ // And VL and DQ but they're all on the same CPUs.
namespace {

// Load from memory, multiply, and convert to int32_t.
inline __m512i QuantizerGrab(const float *input, const __m512 quant_mult_reg) {
  // Multiply each by the quantization factor.
  __m512 val = _mm512_mul_ps(*reinterpret_cast<const __m512*>(input), quant_mult_reg);
  // Cast to 32-bit int
  return _mm512_cvtps_epi32(val);
}

} // namespace


// AVX512 has combined collapse and store instructions:
// _mm512_mask_cvtsepi32_storeu_epi16
// _mm512_mask_cvtsepi32_storeu_epi8
// So conversion in memory uses these, but I also implement a wider version for
// rearranging B.
// 
// Convert to 16-bit signed integers.
void AVX512_16bit::Quantize(const float *input, int16_t *output, float quant_mult, int size) {
    assert(size % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(input) % 64 == 0);
    // Fill with the quantization multiplier.
    const __m512 quant_mult_reg = _mm512_set1_ps(quant_mult);
    const float *end = input + size;
    for (; input != end; input += 16, output += 16) {
      // There doesn't seem to be an unmasked version.
      _mm512_mask_cvtsepi32_storeu_epi16(output, 0xffff, QuantizerGrab(input, quant_mult_reg));
    }
}

// Convert to 8-bit signed integers.
void AVX512_8bit::Quantize(const float *input, int8_t *output, float quant_mult, int size) {
  assert(size % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 64 == 0);
  const __m512i neg127 = _mm512_set1_epi32(-127);
  const __m512 quant_mult_reg = _mm512_set1_ps(quant_mult);
  const float *end = input + size;
  for (; input < end; input += 16, output += 16) {
    __m512i asint = QuantizerGrab(input, quant_mult_reg);
    asint = _mm512_max_epi32(asint, neg127);
    // There doesn't seem to be an unmasked version.
    _mm512_mask_cvtsepi32_storeu_epi8(output, 0xffff, asint);
  }
}

namespace {

// For PrepareB we want to read 8 columns at a time.  When converting 32-bit
// floats to 8-bit values, that's 32 bytes of floats.  But AVX512 is 64 bytes
// wide so it reads off the edge of the tile.  We could expand the tile size
// but then the memory written to won't be contiguous anyway so we'd be doing a
// scatter anyway.  Easier to just read the 8 columns we wanted as 256 bits
// concatenate.
inline __m512 Concat(const __m256 first, const __m256 second) {
  // AVX512DQ but that goes with AVX512BW anyway.
  return _mm512_insertf32x8(_mm512_castps256_ps512(first), second, 1);
}

// Like QuantizerGrab, but allows 32-byte halves (i.e. 8 columns) to be controlled independently.
inline __m512i QuantizerGrabHalves(const float *input0, const float *input1, const __m512 quant_mult_reg) {
  __m512 appended = Concat(*reinterpret_cast<const __m256*>(input0), *reinterpret_cast<const __m256*>(input1));
  appended = _mm512_mul_ps(appended, quant_mult_reg);
  return _mm512_cvtps_epi32(appended);
}

// These are only used for reshaping due to the AVX512 instructions
// _mm512_mask_cvtsepi32_storeu_epi16 and _mm512_mask_cvtsepi32_storeu_epi8
// being used for the quantizer.
class QuantizeTile16 {
  public:
    typedef __m512i Integer;

    explicit QuantizeTile16(float mult) : mult_reg_(_mm512_set1_ps(mult)) {}

    inline __m512i ForReshape(const float *input, int cols) {
      __m512i g0 = QuantizerGrabHalves(input, input + 16 * cols, mult_reg_);
      __m512i g1 = QuantizerGrabHalves(input + 8 * cols, input + 24 * cols, mult_reg_);
      __m512i packed = _mm512_packs_epi32(g0, g1);
      // Permute within 256-bit lanes, so same as AVX2
      return _mm512_permutex_epi64(packed, 0xd8 /* 0, 2, 1, 3 */);
    }

  private:
    const __m512 mult_reg_;
};

class QuantizeTile8 {
  public:
    typedef __m512i Integer;

    explicit QuantizeTile8(float mult) : mult_reg_(_mm512_set1_ps(mult)) {}

    inline __m512i ForReshape(const float *input, int cols) {
      // TODO: try alternative: _mm512_cvtsepi32_epi8 ?
			const __m512i neg127 = _mm512_set1_epi8(-127);
			// In reverse order: grabbing the first 32-bit values from each 128-bit register, then the second 32-bit values, etc.
			const __m512i shuffle_param = _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);

			// 32-bit format.
			__m512i g0 = QuantizerGrabHalves(input, input + 2 * cols, mult_reg_);
			__m512i g1 = QuantizerGrabHalves(input + 16 * cols, input + 18 * cols, mult_reg_);
			__m512i g2 = QuantizerGrabHalves(input + 32 * cols, input + 34 * cols, mult_reg_);
			__m512i g3 = QuantizerGrabHalves(input + 48 * cols, input + 50 * cols, mult_reg_);
			// Pack 32-bit to 16-bit.
			__m512i packed0 = _mm512_packs_epi32(g0, g1);
			__m512i packed1 = _mm512_packs_epi32(g2, g3);
			// Pack 16-bit to 8-bit.
			__m512i packed = _mm512_packs_epi16(packed0, packed1);
			// Ban -128.
			packed = _mm512_max_epi8(packed, neg127);
			// 0 1 2 3 16 17 18 19 32 33 34 35 48 49 50 51 4 5 6 7 20 21 22 23 36 37 38 39 52 53 54 55 8 9 10 11 24 25 26 27 40 41 42 43 56 57 58 59 12 13 14 15 28 29 30 31 44 45 46 47 60 61 62 63
			return _mm512_permutexvar_epi32(shuffle_param, packed);
		}

  private:
    const __m512 mult_reg_;
};

} // namespace

void AVX512_16bit::PrepareB(const float *input, int16_t *output, float quant_mult, int rows, int cols) {
  PrepareBFor16(input, output, QuantizeTile16(quant_mult), rows, cols);
}

void AVX512_8bit::PrepareB(const float *input, int8_t *output, float quant_mult, int rows, int cols) {
  PrepareBFor8(input, output, QuantizeTile8(quant_mult), rows, cols);
}

void AVX512_16bit::Multiply(const int16_t *A, const int16_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) {
  // The unquantization is only 256-bit wide because there are 8 results.
  Multiply16<__m512i, __m256> (A, B, C, unquant_mult, A_rows, width, B_cols);
}

// Special AVX512 implementation due to having 32 registers (so I don't have to
// allocate registers manually) and no sign instruction.
void AVX512_8bit::Multiply(const int8_t *A, const int8_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) {
  typedef __m512i Integer;
  typedef __m256 Float; // For quantization we only do 8 at a time.
  // This is copy-paste from Multiply8_SSE2OrAVX2.
  assert(width % sizeof(Integer) == 0);
  assert(B_cols % 8 == 0);
  assert(reinterpret_cast<uintptr_t>(A) % sizeof(Integer) == 0);
  assert(reinterpret_cast<uintptr_t>(B) % sizeof(Integer) == 0);
  assert(reinterpret_cast<uintptr_t>(C) % sizeof(Integer) == 0);
  Float unquant_reg = set1_ps<Float>(unquant_mult);
  const int simd_width = width / sizeof(Integer);
  const Integer *B0_col = reinterpret_cast<const Integer*>(B);
  // Added for AVX512.
  Integer zeros = setzero_si<Integer>();
  // Go over 8 columns of B at a time.
  for (int B0_colidx = 0; B0_colidx != B_cols; B0_col += 8 * simd_width, B0_colidx += 8) {
    // Process one row of A at a time.  Doesn't seem to be faster to do multiple rows of A at once.
    for (int A_rowidx = 0; A_rowidx < A_rows; ++A_rowidx) {
      // Iterate over shared (inner) dimension.
      const Integer *A_live = reinterpret_cast<const Integer *>(A + A_rowidx * width);
      const Integer *A_end = A_live + simd_width;
      const Integer *B_live = B0_col;

      // Do the first iteration to initialize the sums.
      __m512i a = *A_live;
      __mmask64 neg_mask = _mm512_test_epi8_mask(a, _mm512_set1_epi8(-128));
      __m512i a_positive = _mm512_abs_epi8(a);
      // These will be packed 16-bit integers containing sums for each column of B multiplied by the row of A.
      Integer sum0 = maddubs_epi16(a_positive, _mm512_mask_sub_epi8(B_live[0], neg_mask, zeros, B_live[0]));
      Integer sum1 = maddubs_epi16(a_positive, _mm512_mask_sub_epi8(B_live[1], neg_mask, zeros, B_live[1]));
      Integer sum2 = maddubs_epi16(a_positive, _mm512_mask_sub_epi8(B_live[2], neg_mask, zeros, B_live[2]));
      Integer sum3 = maddubs_epi16(a_positive, _mm512_mask_sub_epi8(B_live[3], neg_mask, zeros, B_live[3]));
      Integer sum4 = maddubs_epi16(a_positive, _mm512_mask_sub_epi8(B_live[4], neg_mask, zeros, B_live[4]));
      Integer sum5 = maddubs_epi16(a_positive, _mm512_mask_sub_epi8(B_live[5], neg_mask, zeros, B_live[5]));
      Integer sum6 = maddubs_epi16(a_positive, _mm512_mask_sub_epi8(B_live[6], neg_mask, zeros, B_live[6]));
      Integer sum7 = maddubs_epi16(a_positive, _mm512_mask_sub_epi8(B_live[7], neg_mask, zeros, B_live[7]));

      ++A_live;
      B_live += 8;

      // Use A as the loop variable so the add can be done where gcc likes it
      // for branch prediction.
      for (; A_live != A_end; ++A_live, B_live += 8) {
        // Unique code here: can we do an inline function?
        // Retrieve a.  We will use this as the unsigned part.
        a = *A_live;
        // Retrieve the conveniently consecutive values of B.
        __m512i b0 = *B_live;
        __m512i b1 = *(B_live + 1);
        __m512i b2 = *(B_live + 2);
        __m512i b3 = *(B_live + 3);
        __m512i b4 = *(B_live + 4);
        __m512i b5 = *(B_live + 5);
        __m512i b6 = *(B_live + 6);
        __m512i b7 = *(B_live + 7);

        // Get a mask where a is negative.
        // Didn't seem to make a difference definining sign bits here vs at top
        neg_mask = _mm512_test_epi8_mask(a, _mm512_set1_epi8(-128));
        a_positive = _mm512_abs_epi8(a);

        // Negate by subtracting from zero with a mask.
        b0 = _mm512_mask_sub_epi8(b0, neg_mask, zeros, b0);
        b1 = _mm512_mask_sub_epi8(b1, neg_mask, zeros, b1);
        b2 = _mm512_mask_sub_epi8(b2, neg_mask, zeros, b2);
        b3 = _mm512_mask_sub_epi8(b3, neg_mask, zeros, b3);
        b4 = _mm512_mask_sub_epi8(b4, neg_mask, zeros, b4);
        b5 = _mm512_mask_sub_epi8(b5, neg_mask, zeros, b5);
        b6 = _mm512_mask_sub_epi8(b6, neg_mask, zeros, b6);
        b7 = _mm512_mask_sub_epi8(b7, neg_mask, zeros, b7);
        // The magic 8-bit multiply then horizontal sum into 16-bit.
        b0 = _mm512_maddubs_epi16(a_positive, b0);
        b1 = _mm512_maddubs_epi16(a_positive, b1);
        b2 = _mm512_maddubs_epi16(a_positive, b2);
        b3 = _mm512_maddubs_epi16(a_positive, b3);
        b4 = _mm512_maddubs_epi16(a_positive, b4);
        b5 = _mm512_maddubs_epi16(a_positive, b5);
        b6 = _mm512_maddubs_epi16(a_positive, b6);
        b7 = _mm512_maddubs_epi16(a_positive, b7);
        // Now we have 16-bit results that are the sum of two multiplies.
        // Choosing to approximate and do adds.
        // Perhaps every so often we could accumulate by upcasting.
        sum0 = _mm512_adds_epi16(sum0, b0);
        sum1 = _mm512_adds_epi16(sum1, b1);
        sum2 = _mm512_adds_epi16(sum2, b2);
        sum3 = _mm512_adds_epi16(sum3, b3);
        sum4 = _mm512_adds_epi16(sum4, b4);
        sum5 = _mm512_adds_epi16(sum5, b5);
        sum6 = _mm512_adds_epi16(sum6, b6);
        sum7 = _mm512_adds_epi16(sum7, b7);
        // Unique code ends: can we do an inline function?
      }
      // Upcast to 32-bit and horizontally add.
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

const char *const AVX512_16bit::kName = "16-bit AVX512";
const char *const AVX512_8bit::kName = "8-bit AVX512";

#endif // __AVX512BW__
} // namespace intgemm
