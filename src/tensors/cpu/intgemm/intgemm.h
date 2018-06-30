#pragma once

/* Main interface for integer matrix multiplication.
 *
 * We are computing C = A * B with an optional scaling factor.
 *
 * A is typically activations.
 * Rows a multiple of 1 (no restriction)
 * Columns a multiple of 64 for 8-bit or 32 for 16-bit.
 * Use PrepareA to prepare A for multiplication.  This is meant to be fast.
 *
 * B is typically fixed model parameters.
 * Rows a multiple of 64 for 8-bit or 32 for 16-bit.
 * Columns a multiple of: 8
 * Use PrepareB to prepare B for multiplication.  This is slower, with the
 * intention that it will be prepared once and remembered.
 *
 * C is row major.
 *
 * Once both A and B are prepared, call Multiply.
 *
 * All memory (A, B, and C in float or prepared form) must be 64-byte aligned.
 * It's easy to write code that works on your CPU with lower alignment, but
 * breaks on AVX512.
 *
 * When preparing, you provide a quantization multiplier.  Values will be
 * multiplied by this then rounded to an integer.
 * For 16-bit neural networks, Jacob Devlin recommends 1024.0.
 * For 8-bit, use 127 / largest absolute value.
 *
 * Note that quantization saturates.  However, 16-bit does accumulation in
 * 32-bit which can overflow if you use too big of a multiplier.
 *
 * The multiply routine expects an unquantization multiplier.
 * This should be unquant_mult = 1.0 / (A_quant_mult * B_quant_mult).
 * Where A_quant_mult is what you passed to PrepareA and B_quant_mult is what you
 * passed to PrepareB.
 *
 * Feel free to multiply in a scaling factor to compute C = \lambda A * B by
 * passing unquant_mult = \lambda / (A_quant_mult * B_quant_mult).
 */

// Yes, both headers due to the debacle about int32_t
#include <cstdint>
#include <stdint.h>
#include <exception>

/* Dispatch to functions based on runtime CPUID.  This adds one call-by-variable to each call. */

namespace intgemm {

// This will be thrown if a CPU isn't supported by the routines (16-bit without SSE2 or 8-bit without SSSE3).
class UnsupportedCPU : public std::exception {
  public:
    UnsupportedCPU();

    ~UnsupportedCPU() throw();

    const char *what() const throw();
};

/* 16-bit matrix multiplication. */
struct Int16 {
  typedef int16_t Integer;

  // A's size must be a multiple of 1x32.
  static const int kATileRow = 1;
  static const int kATileCol = 32;
  // B's size must be a multiple of 32x8.
  static const int kBTileRow = 32;
  static const int kBTileCol = 8;

  // Currently A is prepared by quantization but this could theoretically change.
  // A's columns must be a multiple of 8.
  // The number of rows is anything.
  static inline void PrepareA(const float *input, int16_t *output, float quant_mult, int rows, int cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  // Multiply floats by quant_mult then convert to 16-bit integers with saturation.
  // input
  static void (*Quantize)(const float *input, int16_t *output, float quant_mult, int size);

  // Warning: the output of PrepareB depends on the CPU.
  // It will match the Multiply function on the same CPU though.
  static void (*PrepareB)(const float *input, int16_t *output, float quant_mult, int rows, int cols);

  // Select columns from a prepared B matrix.  The number of selected columns must be a multiple of 8. 
  static void (*SelectColumnsB)(const int16_t *input, int16_t *output, int rows, const std::size_t *cols_begin, const std::size_t *cols_end);

  // Multiply C = A * B, presuming A and B have been prepared.
  static void (*Multiply)(const int16_t *A, const int16_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols);

  static const char *const kName;
};

/* 8-bit matrix multiplication */
struct Int8 {
  typedef int8_t Integer;

  // A's size must be a multiple of 1x64.
  static const int kATileRow = 1;
  static const int kATileCol = 64;
  // B's size must be a multiple of 64x8.
  static const int kBTileRow = 64;
  static const int kBTileCol = 8;

  // Currently A is prepared by quantization but this could theoretically change.
  // A's columns must be a multiple of 8.
  // The number of rows is anything.
  static inline void PrepareA(const float *input, int8_t *output, float quant_mult, int rows, int cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  // Multiply floats by quant_mult then convert to 8-bit integers with saturation.
  static void (*Quantize)(const float *input, int8_t *output, float quant_mult, int size);
  
  // Warning: the output of PrepareB depends on the CPU.
  // It will match the Multiply function on the same CPU though.
  static void (*PrepareB)(const float *input, int8_t *output, float quant_mult, int rows, int cols);

  // Select columns from a prepared B matrix.  The number of selected columns must be a multiple of 8. 
  static void (*SelectColumnsB)(const int8_t *input, int8_t *output, int rows, const std::size_t *cols_begin, const std::size_t *cols_end);

  // Multiply C = A * B, presuming A and B have been prepared.
  static void (*Multiply)(const int8_t *A, const int8_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols);
  
  static const char *const kName;
};

// Get the maximum absolute value of an array of floats.  The number of floats must be a multiple of 16 and 64-byte aligned.
extern float (*MaxAbsolute)(const float *begin, const float *end);

} // namespace intgemm
