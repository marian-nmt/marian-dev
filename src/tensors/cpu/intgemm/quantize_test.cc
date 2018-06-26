#include "avx512_gemm.h"
#include "avx2_gemm.h"
#include "ssse3_gemm.h"
#include "sse2_gemm.h"
#include "aligned.h"
#include "intgemm.h"

#include <cstring>
#include <math.h>

#include <iostream>

namespace intgemm {
namespace {

void QuantizeRef(const float *input, int16_t *output, float quant_mult, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    float value = roundf(input[i] * quant_mult);
    value = std::max(-32768.0f, value);
    value = std::min(32767.0f, value);
    output[i] = value;
  }
}

void QuantizeRef(const float *input, int8_t *output, float quant_mult, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    float value = roundf(input[i] * quant_mult);
    value = std::max(-127.0f, value);
    value = std::min(127.0f, value);
    output[i] = value;
  }
}

template <class I> bool IsOff(float from, I ref, I test) {
  if (ref == test) return false;
  if (ref - test > 1 && test - ref > 1) return true;
  float off_test = fabs((float)test - from);
  float off_ref = fabs((float)ref - from);
  // Allow 0.5 to round either way.
  if (off_test > 0.49 && off_test < 0.51 && off_ref > 0.49 && off_ref < 0.51) return false;
  return true;
}

template <class Backend> bool Test(const float *input_unaligned, float quant_mult, std::size_t size) {
  typedef typename Backend::Integer Integer;
  bool success = true;
  AlignedVector<float> input(size);
  std::memcpy(input.get(), input_unaligned, sizeof(float) * size);

  AlignedVector<Integer> ref(size);
  AlignedVector<Integer> test(size);
  QuantizeRef(input.get(), ref.get(), quant_mult, size);
  Backend::Quantize(input.get(), test.get(), quant_mult, size);
  for (std::size_t i = 0; i < size; ++i) {
    if (IsOff(input.get()[i] * quant_mult, ref.get()[i], test.get()[i])) {
      std::cerr << "Error at " << i << " from " << input.get()[i] << '*' << quant_mult << '=' << (input.get()[i]*quant_mult) << " ref = " <<  ref.get()[i] << " test = " << test.get()[i] << '\n';
      success = false;
    }
  }
  return success;
}

template <class Backend> bool TestMany() {
  bool success = true;
  float input[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  success &= Test<Backend>(input, 1.0, 32);
  success &= Test<Backend>(input, 32.0, 32);
  float corners[32] = {-32769, -32768, -32767, -129, -128, -127, -1, 0, 1, 126, 127, 128, 129, 32766, 32768, 32769, -1.9, -1.5, -1.1, -1, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9, 16056.8};
  success &= Test<Backend>(corners, 1.0, sizeof(corners) / sizeof(float));
  success &= Test<Backend>(corners, -1.0, sizeof(corners) / sizeof(float));
  success &= Test<Backend>(corners, -0.49, sizeof(corners) / sizeof(float));
  return success;
}

} // namespace
} // namespace intgemm

int main() {
  using namespace intgemm;
  bool success = true;
  if (kCPU >= CPU_AVX512BW) {
    success &= TestMany<AVX512_8bit>();
    success &= TestMany<AVX512_16bit>();
  }
  if (kCPU >= CPU_AVX2) {
    success &= TestMany<AVX2_8bit>();
    success &= TestMany<AVX2_16bit>();
  }
  if (kCPU >= CPU_SSSE3) {
    success &= TestMany<SSSE3_8bit>();
  }
  if (kCPU >= CPU_SSE2) {
    success &= TestMany<SSE2_16bit>();
  }
  return success ? 0 : 1;
}
