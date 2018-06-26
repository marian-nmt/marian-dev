#include "intgemm.h"
// This is just for AlignedVector, which helps managed 64-byte aligned memory.
// Feel free to manage memory yourself.
#include "aligned.h" 

#include <cassert>
#include <stdlib.h>
#include <math.h>

int main() {
  const int A_rows = 1;
  // The shared dimension: A's columns and B's rows.
  const int width = 64;
  const int B_cols = 8;

  // This is a simple vector class that allocates memory aligned to 64 bytes.
  // You don't have to use it; just use aligned_alloc and friends directly.
  using intgemm::AlignedVector;
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);

  // Fill with random values in range [-2, 2].
  srand(1);
  for (int i = 0; i < A_rows * width; ++i) {
    A[i] = ((float)rand()/(float)RAND_MAX)*4.0f - 2.0f;
  }
  for (int i = 0; i < width * B_cols; ++i) {
    B[i] = ((float)rand()/(float)RAND_MAX)*4.0f - 2.0f;
  }

  // Compute the top left corner of C as a sanity check.
  float top_left_reference = 0.0;
  for (int w = 0; w < width; ++w) {
    top_left_reference += A[w] * B[w * B_cols];
  }

  // 16-bit multiplication.
  {
    // For 16-bit, Jacob Devlin recommends 1024 so as to not overflow in 32-bit accumulation.
    float quant_mult = 1024.0;
    AlignedVector<int16_t> A_prepared(A_rows * width);
    AlignedVector<int16_t> B_prepared(width * B_cols);
    // Quantize A.
    intgemm::Int16::PrepareA(A.get(), A_prepared.get(), quant_mult, A_rows, width);
    // Quantize and reshape B.
    // Typically you will do this once when parameters are loaded, not every time.
    intgemm::Int16::PrepareB(B.get(), B_prepared.get(), quant_mult, width, B_cols);

    AlignedVector<float> C(A_rows * B_cols);
    // Do the actual multiply.
    intgemm::Int16::Multiply(A_prepared.get(), B_prepared.get(), C.get(), 1.0 / (quant_mult * quant_mult), A_rows, width, B_cols);
    // Sanity check.  C will be row major.
    assert(fabs(C[0] - top_left_reference) < 0.05);
  }

  // 8-bit multiplication.
  {
    // For 8-bit a good quantization multiplier is 127 / largest absolute value..
    float quant_mult = 127.0 / 2.0;
    AlignedVector<int8_t> A_prepared(A_rows * width);
    AlignedVector<int8_t> B_prepared(width * B_cols);
    // Quantize A.
    intgemm::Int8::PrepareA(A.get(), A_prepared.get(), quant_mult, A_rows, width);
    // Quantize and reshape B.
    // Typically you will do this once when parameters are loaded, not every time.
    intgemm::Int8::PrepareB(B.get(), B_prepared.get(), quant_mult, width, B_cols);

    AlignedVector<float> C(A_rows * B_cols);
    // Do the actual multiply.
    intgemm::Int8::Multiply(A_prepared.get(), B_prepared.get(), C.get(), 1.0 / (quant_mult * quant_mult), A_rows, width, B_cols);
    // Sanity check.  C will be row major.
    assert(fabs(C[0] - top_left_reference) < 0.05);
  }
}
