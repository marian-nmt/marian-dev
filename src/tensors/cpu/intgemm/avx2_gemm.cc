#include "avx2_gemm.h"
#include "interleave.h"
#include "multiply.h"

#include <cassert>
#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <stdint.h>

namespace intgemm {

// PREPARE A: just quantization in the same memory order.

namespace {
// Read a vector of floats, multiply them, and cast to 32-bit integer.
inline __m256i QuantizerGrab(const float *input, const __m256 quant_mult_reg) {
  return _mm256_cvtps_epi32(_mm256_mul_ps(*reinterpret_cast<const __m256*>(input), quant_mult_reg));
}

class QuantizeTile16 {
  public:
    typedef __m256i Integer;

    explicit QuantizeTile16(float mult) : mult_(_mm256_set1_ps(mult)) {}

    Integer Consecutive(const float *input) {
      return Tile(input, input + 8);
    }

    Integer ForReshape(const float *input, Index cols) {
      // 8 rows in the first 128-bit register, 8 in the second register.
      return Tile(input, input + 8 * cols);
    }

  private:
    __m256i Tile(const float *input0, const float *input1) {
      __m256i g0 = QuantizerGrab(input0, mult_);
      __m256i g1 = QuantizerGrab(input1, mult_);
      __m256i packed = _mm256_packs_epi32(g0, g1);
      // Reorder the packed values because Intel does 0 1 2 3 8 9 10 11 4 5 6 7 12 13 14 15.
      // Technically this could be removed if the PrepareB did the same reordering internally.
      return _mm256_permute4x64_epi64(packed, 0xd8 /* 0, 2, 1, 3 */);
    }

    const __m256 mult_;
};

} // namespace

// Just quantize everything in order.
void AVX2_16bit::Quantize(const float *input, int16_t *output, float quant_mult, Index size) {
  assert(size % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 32 == 0);
  QuantizeTile16 q(quant_mult);
  const float *end = input + size;
  for (; input != end; input += 16, output += 16) {
    *reinterpret_cast<__m256i*>(output) = q.Consecutive(input);
  }
}

namespace {
/* Read 8 floats at a time from input0, input1, input2, and input3.  Quantize
 * them to 8-bit by multiplying with quant_mult_reg then rounding. Concatenate
 * the result into one register and return it.
 */
class QuantizeTile8 {
  public:
    typedef __m256i Integer;

    explicit QuantizeTile8(float quant_mult) : mult_(_mm256_set1_ps(quant_mult)) {}

    inline __m256i Consecutive(const float *input) {
      return Tile(input, input + 8, input + 16, input + 24);
    }

    inline __m256i ForReshape(const float *input, Index cols) {
      // Put higher rows in the second half of the register.  These will jumble
      // around in the same way then conveniently land in the right place.
      return Tile(input, input + 2 * cols, input + 16 * cols, input + 18 * cols);
    }

  private:
    inline __m256i Tile(const float *input0, const float *input1, const float *input2, const float *input3) {
      // Looking at the assembly, gcc has pulled this outside the loops calling this.
      const __m256i neg127 = _mm256_set1_epi8(-127);
      const __m256i shuffle_param = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
      // Grab 4 registers at a time in 32-bit format.
      __m256i g0 = QuantizerGrab(input0, mult_);
      __m256i g1 = QuantizerGrab(input1, mult_);
      __m256i g2 = QuantizerGrab(input2, mult_);
      __m256i g3 = QuantizerGrab(input3, mult_);
      // Pack 32-bit to 16-bit.
      __m256i packed0 = _mm256_packs_epi32(g0, g1);
      __m256i packed1 = _mm256_packs_epi32(g2, g3);
      // Pack 16-bit to 8-bit.
      __m256i packed = _mm256_packs_epi16(packed0, packed1);
      // Ban -128.
      packed = _mm256_max_epi8(packed, neg127);
      // Currently in 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
      // Or as 32-bit integers 0 2 4 6 1 3 5 7
      // Technically this could be removed so long as the rows are bigger than 16
      // and the values are only used for GEMM.
      return _mm256_permutevar8x32_epi32(packed, shuffle_param);
    }
    
    const __m256 mult_;
};
} // namespace

// Just quantize everything in order.
void AVX2_8bit::Quantize(const float *input, int8_t *output, float quant_mult, Index size) {
  assert(size % 32 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 32 == 0);
  QuantizeTile8 q(quant_mult);
  const float *end = input + size;
  for (; input != end; input += 32, output += 32) {
    *reinterpret_cast<__m256i*>(output) = q.Consecutive(input);
  }
}

void AVX2_16bit::PrepareB(const float *input, int16_t *output, float quant_mult, Index rows, Index cols) {
  PrepareBFor16(input, output, QuantizeTile16(quant_mult), rows, cols);
}

void AVX2_16bit::SelectColumnsB(const int16_t *input, int16_t *output, Index rows, const Index *cols_begin, const Index *cols_end) {
  SelectColumnsOfB((const __m256i*)input, (__m256i*)output, rows * 2, cols_begin, cols_end);
}

void AVX2_8bit::PrepareB(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) {
  PrepareBFor8(input, output, QuantizeTile8(quant_mult), rows, cols);
}

void AVX2_8bit::SelectColumnsB(const int8_t *input, int8_t *output, Index rows, const Index *cols_begin, const Index *cols_end) {
  SelectColumnsOfB((const __m256i*)input, (__m256i*)output, rows, cols_begin, cols_end);
}

void AVX2_16bit::Multiply(const int16_t *A, const int16_t *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols) {
  Multiply16<__m256i, __m256>(A, B, C, unquant_mult, A_rows, width, B_cols);
}

void AVX2_8bit::Multiply(const int8_t *A, const int8_t *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols) {
  Multiply8_SSE2OrAVX2<Multiply8_AVXAVX2, __m256i, __m256>(A, B, C, unquant_mult, A_rows, width, B_cols);
}

const char *const AVX2_16bit::kName = "16-bit AVX2";
const char *const AVX2_8bit::kName = "8-bit AVX2";

float AVX2_MaxAbsolute(const float *begin, const float *end) {
  return MaxAbsoluteBackend<__m256>(begin, end);
}

} // namespace intgemm
