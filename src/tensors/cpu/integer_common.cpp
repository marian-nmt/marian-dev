#include "integer_common.h"

namespace marian {
namespace cpu {
namespace integer {
// This operates on floats after processing so doesn't care about int8_t vs int16_t.
void AddBias(marian::Tensor C, const marian::Tensor Bias) {
  float* y = C->data();
  const float* x = C->data();
  const float* bias = Bias->data();

  const int m = C->shape().elements() / C->shape()[-1];
  const int n = C->shape()[-1];

  for(int j = 0; j < m; ++j) {
    int i = 0;
#ifdef __AVX512F__
    int n16 = n & ~15;
    for(; i < n16; i += 16) {
      __m512 ai = _mm512_loadu_ps(x + j * n + i);
      __m512 bi = _mm512_loadu_ps(bias + i);
      __m512 yi = _mm512_add_ps(ai, bi);
      _mm512_storeu_ps(y + j * n + i, yi);
    }
#else
    int n4 = (n / 4) * 4;
    for(; i < n4; i += 4) {
      __m128 ai = _mm_loadu_ps(x + j * n + i);
      __m128 bi = _mm_loadu_ps(bias + i);
      __m128 yi = _mm_add_ps(ai, bi);
      _mm_storeu_ps(y + j * n + i, yi);
    }
#endif
    for(; i < n; i++) {
      y[j * n + i] = x[j * n + i] + bias[i];
    }
  }
}

#if defined(__AVX512F__) && (__GNUC__ < 10) && !defined(__clang__)
static inline __m512i _mm512_loadu_epi32(const void * in) {
  __m512 reg = _mm512_loadu_ps(in);
  return *reinterpret_cast<__m512i *>(&reg);
}
static inline __m128i _mm_loadu_epi32(const void * in) {
  __m128 reg = _mm_loadu_ps((const float *)in);
  return *reinterpret_cast<__m128i *>(&reg);
}
#endif

// This is done so we can use dnnl
void JustUnquantise(marian::Tensor C, const float unquant_mult) {
  float* y = C->data();
  const int32_t * x = C->data<int32_t>();

#ifdef __AVX512F__
  const __m512 unquant_mult_reg = _mm512_set1_ps(unquant_mult);
#else
  const __m128 unquant_mult_reg = _mm_set1_ps(unquant_mult);
#endif

  const int m = C->shape().elements() / C->shape()[-1];
  const int n = C->shape()[-1];

  for(int j = 0; j < m; ++j) {
    int i = 0;
#ifdef __AVX512F__
    int n16 = n & ~15;
    for(; i < n16; i += 16) {
      __m512 ai = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_epi32(x + j * n + i)), unquant_mult_reg);
      _mm512_storeu_ps(y + j * n + i, ai);
    }
#else
    int n4 = (n / 4) * 4;
    for(; i < n4; i += 4) {
      __m128 ai = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_epi32(x + j * n + i)), unquant_mult_reg);
      _mm_storeu_ps(y + j * n + i, ai);
    }
#endif
    for(; i < n; i++) {
      y[j * n + i] = unquant_mult*x[j * n + i];
    }
  }
}


void JustUnquantiseRelu(marian::Tensor C, const float unquant_mult) {
  float* y = C->data();
  const int32_t * x = C->data<int32_t>();

#ifdef __AVX512F__
  const __m512 unquant_mult_reg = _mm512_set1_ps(unquant_mult);
  static const auto vconst_zero = _mm512_set1_ps(0.0f);
#else
  const __m128 unquant_mult_reg = _mm_set1_ps(unquant_mult);
  static const auto vconst_zero = _mm_set1_ps(0.0f);
#endif

  const int m = C->shape().elements() / C->shape()[-1];
  const int n = C->shape()[-1];

  for(int j = 0; j < m; ++j) {
    int i = 0;
#ifdef __AVX512F__
    int n16 = n & ~15;
    for(; i < n16; i += 16) {
      __m512 ai = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_epi32(x + j * n + i)), unquant_mult_reg);
      _mm512_storeu_ps(y + j * n + i, _mm512_max_ps(ai, vconst_zero));
    }
#else
    int n4 = (n / 4) * 4;
    for(; i < n4; i += 4) {
      __m128 ai = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_epi32(x + j * n + i)), unquant_mult_reg);
      _mm_storeu_ps(y + j * n + i, _mm_max_ps(ai, vconst_zero));
    }
#endif
    for(; i < n; i++) {
      y[j * n + i] = std::max(unquant_mult*x[j * n + i], 0.0f);
    }
  }
}



// This is done so we can use dnnl
void UnquantiseAndAddBias(marian::Tensor C, const marian::Tensor Bias, const float unquant_mult) {
  float* y = C->data();
  const int32_t * x = C->data<int32_t>();
  const float* bias = Bias->data();
#ifdef __AVX512F__
  const __m512 unquant_mult_reg = _mm512_set1_ps(unquant_mult);
#else
  const __m128 unquant_mult_reg = _mm_set1_ps(unquant_mult);
#endif

  const int m = C->shape().elements() / C->shape()[-1];
  const int n = C->shape()[-1];

  for(int j = 0; j < m; ++j) {
    int i = 0;
#ifdef __AVX512F__
    int n16 = n & ~15;
    for(; i < n16; i += 16) {
      __m512 ai = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_epi32(x + j * n + i)), unquant_mult_reg);
      __m512 bi = _mm512_loadu_ps(bias + i);
      __m512 yi = _mm512_add_ps(ai, bi);
      _mm512_storeu_ps(y + j * n + i, yi);
    }
#else
    int n4 = (n / 4) * 4;
    for(; i < n4; i += 4) {
      __m128 ai = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_epi32(x + j * n + i)), unquant_mult_reg);
      __m128 bi = _mm_loadu_ps(bias + i);
      __m128 yi = _mm_add_ps(ai, bi);
      _mm_storeu_ps(y + j * n + i, yi);
    }
#endif
    for(; i < n; i++) {
      y[j * n + i] = unquant_mult*x[j * n + i] + bias[i];
    }
  }
}

// This is done so we can use
void UnquantiseAndAddBiasAndRelu(marian::Tensor C, const marian::Tensor Bias, const float unquant_mult) {
  float* y = C->data();
  const int32_t * x = C->data<int32_t>();
  const float* bias = Bias->data();
#ifdef __AVX512F__
  const __m512 unquant_mult_reg = _mm512_set1_ps(unquant_mult);
  static const auto vconst_zero = _mm512_set1_ps(0.0f);
#else
  const __m128 unquant_mult_reg = _mm_set1_ps(unquant_mult);
  static const auto vconst_zero = _mm_set1_ps(0.0f);
#endif

  const int m = C->shape().elements() / C->shape()[-1];
  const int n = C->shape()[-1];

  for(int j = 0; j < m; ++j) {
    int i = 0;
#ifdef __AVX512F__
    int n16 = n & ~15;
    for(; i < n16; i += 16) {
      __m512 ai = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_epi32(x + j * n + i)), unquant_mult_reg);
      __m512 bi = _mm512_loadu_ps(bias + i);
      __m512 yi = _mm512_max_ps(_mm512_add_ps(ai, bi), vconst_zero);
      _mm512_storeu_ps(y + j * n + i, yi);
    }
#else
    int n4 = (n / 4) * 4;
    for(; i < n4; i += 4) {
      __m128 ai = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_epi32(x + j * n + i)), unquant_mult_reg);
      __m128 bi = _mm_loadu_ps(bias + i);
      __m128 yi = _mm_max_ps(_mm_add_ps(ai, bi), vonst_zero);
      _mm_storeu_ps(y + j * n + i, yi);
    }
#endif
    for(; i < n; i++) {
      y[j * n + i] = std::max(unquant_mult*x[j * n + i] + bias[i], 0.0f);
    }
  }
}

//template void prepareAndTranspose<intgemm8>;//(io::Item& item, const char * input);
//template void prepareAndTranspose<intgemm16>(io::Item&, const char *);

} //integer
} //cpu
} //marian