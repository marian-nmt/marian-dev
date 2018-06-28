// Some compilers don't have AVX512BW support.  Test for them.
#include <immintrin.h>

#include <iostream>

int main() {
  // AVX512F
  __m512i value = _mm512_set1_epi32(1);
  // AVX512BW
  value = _mm512_maddubs_epi16(value, value);

  __m256i value2 = _mm256_set1_epi8(1);
  // AVX512DQ
  value = _mm512_inserti32x8(value, value2, 1);
  return *(int*)&value && __builtin_cpu_supports("avx512f");
}
