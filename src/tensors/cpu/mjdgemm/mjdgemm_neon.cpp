#include "mjdgemm.h"
#include "mjdgemm_utils.h"

#include "mjdgemm_neon.h"
#include "mjdgemm_tmp.h"

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Performs GEMM operation using NEON instruction set.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param bias Pointer to the bias matrix.
 * @param C Pointer to the destination matrix.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */
void gemmInt8PackedNEON(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K) {
  LOG_ONCE(info, "mjdgemm: Using NEON kernels");
  gemmInt8Packed<InstructionSet::NEON>(A, B, bias, C, M, N, K);
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian
