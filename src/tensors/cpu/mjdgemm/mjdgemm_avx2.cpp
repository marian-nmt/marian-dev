#include "mjdgemm.h"
#include "mjdgemm_utils.h"

#if defined(__x86_64__) || defined(_M_X64)
#include "mjdgemm_avx2.h"
#include "mjdgemm_tmp.h"
#endif

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Performs GEMM operation using AVX2 instruction set.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param bias Pointer to the bias matrix.
 * @param C Pointer to the destination matrix.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */
void gemmInt8PackedAVX2(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K) {
#if defined(__x86_64__) || defined(_M_X64)
  LOG_ONCE(info, "mjdgemm: Using AVX2 kernels");
  gemmInt8Packed<InstructionSet::AVX2>(A, B, bias, C, M, N, K);
#else
  ABORT("mjdgemm: AVX2 kernels not available on this machine");
#endif
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian
