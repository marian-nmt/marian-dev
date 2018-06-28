#include "avx512_gemm.h"
#include "avx2_gemm.h"
#include "ssse3_gemm.h"
#include "sse2_gemm.h"
#include "intgemm.h"
#include "aligned.h"
#include "interleave.h"
#include "stop_watch.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include <iostream>
#include <iomanip>

namespace intgemm {

// Rearrange a tile of simd x unroll entries.
template <class V> void SlowRearrangeTile(const V *from, V *to, int simd, int unroll, int cols) {
  for (int i = 0; i < unroll; ++i) {
    for (int j = 0; j < simd; ++j) {
      to[simd * i + j] = from[cols * j + i];
    }
  }
}

template <class V> void SlowRearrange(const V *from, V *to, int simd, int unroll, int rows, int cols) {
  for (int c = 0; c < cols; c += unroll) {
    for (int r = 0; r < rows; r += simd) {
      SlowRearrangeTile(from + cols * r + c, to, simd, unroll, cols);
      to += unroll * simd;
    }
  }
}

template <class V> void SlowTranspose(const V *from, V *to, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      to[rows * c + r] = from[cols * r + c];
    }
  }
}

void TestTranspose16() {
  AlignedVector<int16_t> input(8 * 8);
  for (int16_t i = 0; i < 64; ++i) {
    input.get()[i] = i;
  }
  AlignedVector<int16_t> ref(8 * 8);
  SlowTranspose(input.get(), ref.get(), 8, 8);

  // Overwrite input.
  __m128i *t = reinterpret_cast<__m128i*>(input.get());
  Transpose16InLane(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

  for (int16_t i = 0; i < 64; ++i) {
    if (ref.get()[i] != input.get()[i]) {
      std::cerr << "16-bit transpose failure at " << i << ": " << ref.get()[i] << " != " << input.get()[i] << '\n';
    }
  }
}

void TestTranspose8() {
  AlignedVector<int8_t> input(16 * 16);
  for (int i = 0; i < 16 * 16; ++i) {
    input.get()[i] = i;
  }
  AlignedVector<int8_t> ref(16 * 16);
  SlowTranspose(input.get(), ref.get(), 16, 16);

  // Overwrite input.
  __m128i *t = reinterpret_cast<__m128i*>(input.get());
  Transpose8InLane(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15]);

  for (int i = 0; i < 16 * 16; ++i) {
    if (ref.get()[i] != input.get()[i]) {
      std::cerr << "8-bit transpose failure at " << i << ": " << (int16_t)ref.get()[i] << " != " << (int16_t)input.get()[i] << '\n';
    }
  }
}

template <class T> void PrintMatrix(const T *mem, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::cout << std::setw(4) << (int64_t) mem[r * cols + c] << ' ';
    }
    std::cout << '\n';
  }
}

template <class Routine> void TestPrepare(int rows = 32, int cols = 16) {
  if (intgemm::kCPU < Routine::kUses) return;
  // Create array.
  AlignedVector<float> input(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    input.get()[i] = //(i > 127) ? (i - 256) : i;
      (float)rand() / (float)RAND_MAX * 256.0 - 127.0;
  }

  typedef typename Routine::Integer Integer;
  // Call Prepare
  AlignedVector<Integer> test(rows * cols);
  Routine::PrepareB(input.get(), test.get(), 1, rows, cols);

  // Compute reference output.
  AlignedVector<Integer> quantized(rows * cols);
  Routine::Quantize(input.get(), quantized.get(), 1, rows * cols);
  AlignedVector<Integer> reference(rows * cols);
  SlowRearrange<Integer>(quantized.get(), reference.get(), Routine::kBTileRow, Routine::kBTileCol, rows, cols);

  if (memcmp(reference.get(), test.get(), rows * cols * sizeof(Integer))) {
    std::cerr << "TestPrepare " << Routine::kName << " Mismatch:\n";
    std::cout << "Quantized Input" << '\n';
    PrintMatrix(quantized.get(), rows, cols);
    std::cerr << "Reference" << '\n';
    PrintMatrix(reference.get(), rows, cols);
    std::cerr << "Routine" << '\n';
    PrintMatrix(test.get(), rows, cols);
  }
}

// Based on https://arxiv.org/abs/1705.01991

// Copyright (c) 2017 Microsoft Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// Compute A*B slowly in floats.
void SlowRefFloat(const float *A, const float *B, float *C, int A_rows, int width, int B_cols) {
  for (int r = 0; r < A_rows; ++r) {
    for (int c = 0; c < B_cols; ++c) {
      float sum = 0.0f;
      for (int k = 0; k < width; ++k) {
        sum += A[r * width + k] * B[k * B_cols + c];
      }
      C[r * B_cols + c] = sum;
    }
  }
}

// Compute A*B slowly from integers.
template <class Integer> void SlowRefInt(const Integer *A, const Integer *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) {
  for (int r = 0; r < A_rows; ++r) {
    for (int c = 0; c < B_cols; ++c) {
      int32_t sum = 0;
      for (int k = 0; k < width; ++k) {
        sum += static_cast<int16_t>(A[r * width + k]) * static_cast<int16_t>(B[k * B_cols + c]);
      }
      C[r * B_cols + c] = sum * unquant_mult;
    }
  }
}


void Compare(const float *float_ref, const float *int_ref, const float *int_test, std::size_t size) {
  float int_sum = 0.0, float_sum = 0.0;
  for (std::size_t i = 0; i < size; ++i) {
    float int_diff = int_ref[i] - int_test[i];
    float float_diff = float_ref[i] - int_test[i];
/*    if (fabs(int_diff) > .1 || fabs(float_diff) > 1) {
      std::cerr << "Inaccurate at " << i << ' ' << float_ref[i] << ' ' << int_ref[i] << ' ' << int_test[i] << '\n';
    }*/
    int_sum += int_diff * int_diff;
    float_sum += float_diff * float_diff;
  }
  std::cout << "Float MSE = " << sqrt(float_sum / size) << "\tInt MSE = " << sqrt(int_sum / size) << std::endl;
}

template <class Routine> void TestMultiply(int A_rows, int width, int B_cols) {
  typedef typename Routine::Integer Integer;
  if (intgemm::kCPU < Routine::kUses) return;
  std::cout << Routine::kName << "\t" << A_rows << '\t' << width << '\t' << B_cols << '\n';

  // Initialize A and B.
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);
  for (int i = 0; i < A_rows * width; i++) {
    A.get()[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
  }
  for (int i = 0; i < width * B_cols; ++i) {
    B.get()[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
  }
  
  float quant_mult = (sizeof(Integer) == 2) ? 1024 : 64;
  float unquant_mult = 1.0/(quant_mult*quant_mult);

  AlignedVector<Integer> A_prep(A_rows * width), B_prep(width * B_cols);
  Routine::PrepareA(A.get(), A_prep.get(), quant_mult, A_rows, width);
  Routine::PrepareB(B.get(), B_prep.get(), quant_mult, width, B_cols);

  AlignedVector<float> test_C(A_rows * B_cols);
  Routine::Multiply(A_prep.get(), B_prep.get(), test_C.get(), unquant_mult, A_rows, width, B_cols);

  AlignedVector<Integer> B_quant(width * B_cols);
  Routine::Quantize(B.get(), B_quant.get(), quant_mult, width * B_cols);
  AlignedVector<float> slowint_C(A_rows * B_cols);
  // Assuming A is just quantization here.
  SlowRefInt(A_prep.get(), B_quant.get(), slowint_C.get(), unquant_mult, A_rows, width, B_cols);

  AlignedVector<float> float_C(A_rows * B_cols);
  SlowRefFloat(A.get(), B.get(), float_C.get(), A_rows, width, B_cols);

  Compare(float_C.get(), slowint_C.get(), test_C.get(), A_rows * B_cols);
}

void TestBoth(int A_rows, int width, int B_cols) {
#ifndef INTGEMM_NO_AVX512
  TestMultiply<AVX512_16bit>(A_rows, width, B_cols);
#endif
  TestMultiply<AVX2_16bit>(A_rows, width, B_cols);
  TestMultiply<SSE2_16bit>(A_rows, width, B_cols);
#ifndef INTGEMM_NO_AVX512
  TestMultiply<AVX512_8bit>(A_rows, width, B_cols);
#endif
  TestMultiply<AVX2_8bit>(A_rows, width, B_cols);
  TestMultiply<SSSE3_8bit>(A_rows, width, B_cols);
}

} // namespace intgemm

// Program takes no input
int main(int argc, char ** argv) {
    std::srand(45678);
    using namespace intgemm;
    if (kCPU >= CPU_SSE2) {
      TestTranspose16();
    }
    if (kCPU >= CPU_SSSE3) {
      TestTranspose8();
    }
#ifndef INTGEMM_NO_AVX512
    TestPrepare<AVX512_8bit>(64, 8);
    TestPrepare<AVX512_8bit>(256, 32);
    TestPrepare<AVX512_16bit>(32, 8);
    TestPrepare<AVX512_16bit>(256, 32);
#endif
    TestPrepare<AVX2_8bit>(64, 32);
    TestPrepare<AVX2_16bit>(64, 32);
    TestPrepare<SSSE3_8bit>(16, 8);
    TestPrepare<SSSE3_8bit>(32, 16);
    TestPrepare<SSSE3_8bit>(32, 32);
    TestPrepare<SSE2_16bit>(8, 8);
    TestPrepare<SSE2_16bit>(32, 32);
    // Top matrix sizes from Marian
    TestBoth(8, 256, 256);
    TestBoth(8, 2048, 256);
    TestBoth(8, 2048, 256);
    TestBoth(320, 256, 256);
    TestBoth(472, 256, 256);
    TestBoth(248, 256, 256);
    TestBoth(200, 256, 256);
    return 0;
}
