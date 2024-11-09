#pragma once

#include <cstdint>

namespace marian {
namespace cpu {

/**
 * @brief Namespace for the mjdgemm implementation.
 *
 * Environment variables MJD_FORCE and MJD_ENABLE_INSTRUCTIONS can be used to force the use of mjdgemm
 * and to select the instruction set to use, respectively. The instruction set can be set to AVX512, AVX2,
 * AVX, SSE4_2, or NEON. Currently, AVX falls back to SSE4_2. E.g. to force the use of mjdgemm with AVX2
 * instructions, set MJD_ENABLE_INSTRUCTIONS=AVX2 and MJD_FORCE=true.
 */
namespace mjdgemm {

/**
 * @brief Determines whether to force the use of mjdgemm based on environment variables.
 *
 * @return True if mjdgemm is forced, otherwise false.
 */
bool forceMjdgemm();

/**
 * @brief Performs a GEMM operation with int8 packed matrices.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param bias Pointer to the bias matrix.
 * @param C Pointer to the destination matrix.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */
void gemmInt8Packed(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K);

} // namespace mjdgemm
} // namespace cpu
} // namespace marian
