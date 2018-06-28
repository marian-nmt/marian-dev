#include "intgemm.h"

#include "cpu_type.h"
#include "sse2_gemm.h"
#include "ssse3_gemm.h"
#include "avx2_gemm.h"
#ifndef INTGEMM_NO_AVX512
#include "avx512_gemm.h"
#endif

namespace intgemm {

UnsupportedCPU::UnsupportedCPU() {}

UnsupportedCPU::~UnsupportedCPU() throw() {}

const char *UnsupportedCPU::what() const throw() {
  return "Integer matrix multiplication has not been efficiently implemented for your CPU.";
}

namespace {

struct Unsupported_16bit {
  static void Quantize(const float *, int16_t *, float, int) {
    throw UnsupportedCPU();
  }
  static void PrepareB(const float *, int16_t *, float, int, int) {
    throw UnsupportedCPU();
  }
  static void SelectColumnsB(const int16_t *input, int16_t *output, int rows, const int *cols_begin, const int *cols_end) {
    throw UnsupportedCPU();
  }
  static void Multiply(const int16_t *, const int16_t *, float *C, float, int, int, int) {
    throw UnsupportedCPU();
  }
  static const char *const kName;
};
const char *const Unsupported_16bit::kName = "16-bit Unsupported";

struct Unsupported_8bit {
  static void Quantize(const float *, int8_t *, float, int) {
    throw UnsupportedCPU();
  }
  static void PrepareB(const float *, int8_t *, float, int, int) {
    throw UnsupportedCPU();
  }
  static void SelectColumnsB(const int8_t *input, int8_t *output, int rows, const int *cols_begin, const int *cols_end) {
    throw UnsupportedCPU();
  }
  static void Multiply(const int8_t *, const int8_t *, float *C, float, int, int, int) {
    throw UnsupportedCPU();
  }
  static const char *const kName;
};
const char *const Unsupported_8bit::kName = "8-bit Unsupported";

/* Returns:
 * avx512 if the CPU supports AVX512F (though really it should be AVX512BW, but
 * cloud providers lie).  TODO: don't catch Knights processors with this.
 *
 * avx2 if the CPU supports AVX2
 * 
 * ssse3 if the CPU supports SSSE3 (this distinction from SSE2 matters for 8-bit)
 * 
 * sse2 if the CPU supports SSE2
 *
 * unsupported otherwise
 */
template <class T> T ChooseCPU(T avx512, T avx2, T ssse3, T sse2, T unsupported) {
  // TODO: don't catch Knights processors here!
#ifndef INTGEMM_NO_AVX512
  if (__builtin_cpu_supports("avx512f")) {
    return avx512;
  }
#endif
  if (__builtin_cpu_supports("avx2")) {
    return avx2;
  } else if (__builtin_cpu_supports("ssse3")) {
    return ssse3;
  } else if (__builtin_cpu_supports("sse2")) {
    return sse2;
  } else {
    return unsupported;
  }
}

#ifdef INTGEMM_NO_AVX512
// These won't ever be called in this capacity, but it does let the code below compile.
typedef Unsupported_16bit AVX512_16bit;
typedef Unsupported_8bit AVX512_8bit;
#endif

} // namespace

void (*Int16::Quantize)(const float *input, int16_t *output, float quant_mult, int size) = ChooseCPU(AVX512_16bit::Quantize, AVX2_16bit::Quantize, SSE2_16bit::Quantize, SSE2_16bit::Quantize, Unsupported_16bit::Quantize);
void (*Int16::PrepareB)(const float *input, int16_t *output, float quant_mult, int rows, int cols) = ChooseCPU(AVX512_16bit::PrepareB, AVX2_16bit::PrepareB, SSE2_16bit::PrepareB, SSE2_16bit::PrepareB, Unsupported_16bit::PrepareB);
void (*Int16::SelectColumnsB)(const int16_t *input, int16_t *output, int rows, const int *cols_begin, const int *cols_end) = ChooseCPU(AVX512_16bit::SelectColumnsB, AVX2_16bit::SelectColumnsB, SSE2_16bit::SelectColumnsB, SSE2_16bit::SelectColumnsB, Unsupported_16bit::SelectColumnsB);
void (*Int16::Multiply)(const int16_t *A, const int16_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) = ChooseCPU(AVX512_16bit::Multiply, AVX2_16bit::Multiply, SSE2_16bit::Multiply, SSE2_16bit::Multiply, Unsupported_16bit::Multiply);
const char *const Int16::kName = ChooseCPU(AVX512_16bit::kName, AVX2_16bit::kName, SSE2_16bit::kName, SSE2_16bit::kName, Unsupported_16bit::kName);

void (*Int8::Quantize)(const float *input, int8_t *output, float quant_mult, int size) = ChooseCPU(AVX512_8bit::Quantize, AVX2_8bit::Quantize, SSSE3_8bit::Quantize, Unsupported_8bit::Quantize, Unsupported_8bit::Quantize);
void (*Int8::PrepareB)(const float *input, int8_t *output, float quant_mult, int rows, int cols) = ChooseCPU(AVX512_8bit::PrepareB, AVX2_8bit::PrepareB, SSSE3_8bit::PrepareB, Unsupported_8bit::PrepareB, Unsupported_8bit::PrepareB);
void (*Int8::SelectColumnsB)(const int8_t *input, int8_t *output, int rows, const int *cols_begin, const int *cols_end) = ChooseCPU(AVX512_8bit::SelectColumnsB, AVX2_8bit::SelectColumnsB, SSSE3_8bit::SelectColumnsB, Unsupported_8bit::SelectColumnsB, Unsupported_8bit::SelectColumnsB);
void (*Int8::Multiply)(const int8_t *A, const int8_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) = ChooseCPU(AVX512_8bit::Multiply, AVX2_8bit::Multiply, SSSE3_8bit::Multiply, Unsupported_8bit::Multiply, Unsupported_8bit::Multiply);
const char *const Int8::kName = ChooseCPU(AVX512_8bit::kName, AVX2_8bit::kName, SSSE3_8bit::kName, Unsupported_8bit::kName, Unsupported_8bit::kName);

const CPUType kCPU = ChooseCPU(CPU_AVX512BW, CPU_AVX2, CPU_SSSE3, CPU_SSE2, CPU_UNSUPPORTED);

} // namespace intgemm
