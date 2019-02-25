#pragma once
#include <stdint.h>
#include <cstdint>

#include "types.h"

/* AVX512 implementation.
 * This uses AVX512BW, AVX512DQ, and might use AVX512VL
 * That means it supports mainstream CPUs with AVX512, starting with Skylake
 * Xeons.
 * It does not support any Knights / Xeon Phi processors.
 *
 * All memory must be 64-byte aligned.
 */

namespace intgemm {

struct AVX512_16bit {
  typedef int16_t Integer;

  // Currently A is prepared by quantization but this could theoretically change.
  // rows * cols must be a multiple of 16.
  static inline void PrepareA(const float *input, int16_t *output, float quant_mult, Index rows, Index cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  // Technically output can be unaligned in Quantize.
  // But then it will need to be aligned for Multiply.
  // size must be a multiple of 16.
  static void Quantize(const float *input, int16_t *output, float quant_mult, Index size);

  // Tile size for B; B must be a multiple of this block size.
  static const Index kBTileRow = 32;
  static const Index kBTileCol = 8;

  static void PrepareB(const float *input, int16_t *output, float quant_mult, Index rows, Index cols);

  static void SelectColumnsB(const int16_t *input, int16_t *output, Index rows, const Index *cols_begin, const Index *cols_end);

  static void Multiply(const int16_t *A, const int16_t *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols);

  static const char *const kName;

  static const CPUType kUses = CPU_AVX512BW;
};

struct AVX512_8bit {
  typedef int8_t Integer;

  // Currently A is prepared by quantization but this could theoretically change.
  static inline void PrepareA(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  // Technically output can be unaligned in Quantize.
  // But then it will need to be aligned for Multiply.
  static void Quantize(const float *input, int8_t *output, float quant_mult, Index size);

  // Tile size for B; B must be a multiple of this block size.
  static const Index kBTileRow = 64;
  static const Index kBTileCol = 8;

  static void PrepareB(const float *input, int8_t *output, float quant_mult, Index rows, Index cols);

  static void SelectColumnsB(const int8_t *input, int8_t *output, Index rows, const Index *cols_begin, const Index *cols_end);

  static void Multiply(const int8_t *A, const int8_t *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols);

  static const char *const kName;

  static const CPUType kUses = CPU_AVX512BW;
};

float AVX512_MaxAbsolute(const float *begin_float, const float *end_float);

} // namespace intgemm
