// This is only 16-bit.  8-bit is in ssse3_gemm.cc since it requires that.
#include "sse2_gemm.h"

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

class QuantizeTile16 {
  public:
    typedef __m128i Integer;

    explicit QuantizeTile16(float mult) : mult_reg_(_mm_set1_ps(mult)) {}

    // Quantize 8xfloat into 8xint16_t
    inline __m128i Consecutive(const float *input) {
      __m128i g0 = QuantizerGrab(input, mult_reg_);
      __m128i g1 = QuantizerGrab(input + 4, mult_reg_);
      return _mm_packs_epi32(g0, g1);
    }

    inline __m128i ForReshape(const float *input, int) {
      return Consecutive(input);
    }

  private:
    const __m128 mult_reg_;
};
} // namespace

/* I also tried an implementation based on _mm_cvtps_pi16 but it was slower:
 * For size 1048576, run 10x in seconds on i7-6700:
 * This code: 0.00228409, 0.00204906
 * With _mm_cvtps_pi16 basis: 0.00391884, 0.00390869
 */
void SSE2_16bit::Quantize(const float *input, int16_t *output, float quant_mult, int size) {
  assert(size % 8 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(output) % 16 == 0);
  QuantizeTile16 q(quant_mult);
  const float *end = input + size;
  for (; input != end; input += 8, output += 8) {
    *reinterpret_cast<__m128i*>(output) = q.Consecutive(input);
  }
}

void SSE2_16bit::PrepareB(const float *input, int16_t *output, float quant_mult, int rows, int cols) {
  PrepareBFor16(input, output, QuantizeTile16(quant_mult), rows, cols);
}

void SSE2_16bit::SelectColumnsB(const int16_t *input, int16_t *output, int rows, const int *cols_begin, const int *cols_end) {
  SelectColumnsOfB((const __m128i*)input, (__m128i*)output, rows * 2, cols_begin, cols_end);
}

void SSE2_16bit::Multiply(const int16_t *A, const int16_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) {
  Multiply16<__m128i, __m128>(A, B, C, unquant_mult, A_rows, width, B_cols);
}

const char *const SSE2_16bit::kName = "16-bit SSE2";

float SSE2_MaxAbsolute(const float *begin, const float *end) {
  return MaxAbsoluteBackend<__m128>(begin, end);
}

} // namespace intgemm
