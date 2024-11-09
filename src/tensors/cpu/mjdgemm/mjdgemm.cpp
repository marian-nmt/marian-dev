#include "mjdgemm.h"
#include "mjdgemm_utils.h"
#include "mjdgemm_generic.h"
#include "mjdgemm_tmp.h"

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Determines whether to force the use of mjdgemm based on environment variables.
 *
 * @return True if mjdgemm is forced, otherwise false.
 */
bool forceMjdgemm() {
  char* env = std::getenv("MJD_FORCE");
  if(env) {
    std::string force(env);
    if(force == "true" || force == "1" || force == "True" || force == "TRUE"
      || force == "yes" || force == "Yes" || force == "YES"
      || force == "on" || force == "On" || force == "ON") {
      LOG(warn, "mjdgemm: Forced to use mjdgemm");
      return true;
    }
  }
  return false;
}

void gemmInt8PackedAVX512(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K);
void gemmInt8PackedAVX2(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K);
void gemmInt8PackedSSE4_2(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K);
void gemmInt8PackedNEON(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K);

/**
 * @brief Selects and executes the appropriate GEMM implementation based on the highest supported instruction set.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param bias Pointer to the bias matrix.
 * @param C Pointer to the destination matrix.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */
void gemmInt8Packed(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K) {
  static const auto highestInstructionSet = getSupportedInstructionSet();

  switch (highestInstructionSet) {
#if defined(__x86_64__) || defined(_M_X64)
    case InstructionSet::AVX512:
      gemmInt8PackedAVX512(A, B, bias, C, M, N, K);
      break;
    case InstructionSet::AVX2:
      gemmInt8PackedAVX2(A, B, bias, C, M, N, K);
      break;
    case InstructionSet::AVX:
    case InstructionSet::SSE4_2:
      gemmInt8PackedSSE4_2(A, B, bias, C, M, N, K);
      break;
#elif defined(__aarch64__)
    case InstructionSet::NEON:
      gemmInt8PackedNEON(A, B, bias, C, M, N, K);
      break;
#endif
    default:
      gemmInt8Packed<InstructionSet::None>(A, B, bias, C, M, N, K);
      break;
  }
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian
