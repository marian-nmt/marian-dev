#include "ssse3_gemm.h"

#include "interleave.h"
#include "multiply.h"

#include <stdint.h>
#include <cassert>
#include <xmmintrin.h>
#include <emmintrin.h>

namespace intgemm {

namespace {
// Same implementation as AVX512, just width.  Grabs 4 32-bit values.
inline __m128i QuantizerGrab(const float *input, const __m128 quant_mult_reg) {
  return _mm_cvtps_epi32(_mm_mul_ps(*reinterpret_cast<const __m128*>(input), quant_mult_reg));
}

class QuantizeTile8 {
  public:
    typedef __m128i Integer;

    explicit QuantizeTile8(float mult) : mult_reg_(_mm_set1_ps(mult)) {}

    inline __m128i ForReshape(const float *input, int cols) {
      // Skip a row.
      return Tile(input, input + 2 * cols);
    }

    inline __m128i Consecutive(const float *input) {
      return Tile(input, input + 8);
    }

  private:
    // Quantize 16xfloat into 16xint8_t
    inline __m128i Tile(const float *input0, const float *input1) {
      const __m128i neg128 = _mm_set1_epi8(-128);
      __m128i g0 = QuantizerGrab(input0, mult_reg_);
      __m128i g1 = QuantizerGrab(input0 + 4, mult_reg_);
      __m128i g2 = QuantizerGrab(input1, mult_reg_);
      __m128i g3 = QuantizerGrab(input1 + 4, mult_reg_);
      __m128i packed0 = _mm_packs_epi32(g0, g1);
      __m128i packed1 = _mm_packs_epi32(g2, g3);
      __m128i packed = _mm_packs_epi16(packed0, packed1);
      /* Ban -128.
       * Don't use the SSE4.1 instruction _mm_max_epi8(packed, neg127).  Instead,
       * use SSE2 instructions _mm_cmpeq_epi8 and _mm_sub_epi8.
       * The first generates 0xff for fields -128.
       * The second subtracts 0xff from -128 which has the effect of converting
       * to -127.
       */
      // packed = _mm_max_epi8(packed, neg127);
      __m128i evils = _mm_cmpeq_epi8(packed, neg128);
      return _mm_sub_epi8(packed, evils);
      // No permute needed.  packs is in order for SSE.
    }

  private:
    const __m128 mult_reg_;
};

} // namespace

void SSSE3_8bit::Quantize(const float *input, int8_t *output, float quant_mult, int size) {
  assert(size % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(output) % 16 == 0);
  QuantizeTile8 q(quant_mult);
  const float *end = input + size;
  for (; input != end; input += 16, output += 16) {
    *reinterpret_cast<__m128i*>(output) = q.Consecutive(input);
  }
}

void SSSE3_8bit::PrepareB(const float *input, int8_t *output, float quant_mult, int rows, int cols) {
  PrepareBFor8(input, output, QuantizeTile8(quant_mult), rows, cols);
}

void SSSE3_8bit::SelectColumnsB(const int8_t *input, int8_t *output, int rows, const std::size_t *cols_begin, const std::size_t *cols_end) {
  SelectColumnsOfB((const __m128i*)input, (__m128i*)output, rows, cols_begin, cols_end);
}

void SSSE3_8bit::Multiply(const int8_t *A, const int8_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) {
  Multiply8_SSE2OrAVX2<Multiply8_C, __m128i, __m128>(A, B, C, unquant_mult, A_rows, width, B_cols);
}

const char *const SSSE3_8bit::kName = "8-bit SSSE3";

} // namespace intgemm
