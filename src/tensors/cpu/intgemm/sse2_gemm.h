#pragma once
#include "cpu_type.h"
#include <stdint.h>
// 8 bit is in ssse3_gemm.h

namespace intgemm {

// This should be pure SSE2 (and below).
struct SSE2_16bit {
  typedef int16_t Integer;

  // Currently A is prepared by quantization but this could theoretically change.
  static inline void PrepareA(const float *input, int16_t *output, float quant_mult, int rows, int cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  static void Quantize(const float *input, int16_t *output, float quant_mult, int size);

  // Tile size for B; B must be a multiple of this block size.
  static const int kBTileRow = 8;
  static const int kBTileCol = 8;

  static void PrepareB(const float *input, int16_t *output, float quant_mult, int rows, int cols);

  static void SelectColumnsB(const int16_t *input, int16_t *output, int rows, const int *cols_begin, const int *cols_end);

  static void Multiply(const int16_t *A, const int16_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols);

  static const char *const kName;

  static const CPUType kUses = CPU_SSE2;
};

// Technically only requires SSE
float SSE2_MaxAbsolute(const float *begin, const float *end);

} // namespace intgemm
