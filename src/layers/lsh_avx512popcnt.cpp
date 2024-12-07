#include "common/logging.h"
#include "tensors/cpu/cpu_info.h"
#include "lsh.h"

namespace marian {
namespace lsh {

using namespace ::marian::cpu;

// AVX512 Vectorized version of hamming distance computation.
// The implementation uses AVX512 and VPOPCNTDQ ISA available on a subset of x86_64 platforms.
// Using this version requires compile time and runtime checks to ensure feature support.
template <InstructionSet IS, size_t StepsStatic>
inline DistType hamming(ChunkType* queryRow, ChunkType* codeRow) {
  if constexpr (StepsStatic == 2 * (64 / sizeof(ChunkType))) {
    __m512i qr   = _mm512_load_epi64(queryRow);
    __m512i cr   = _mm512_load_epi64(codeRow);
    __m512i dist = _mm512_popcnt_epi64(_mm512_xor_si512(qr, cr));

    qr =   _mm512_load_epi64(queryRow + 8);
    cr =   _mm512_load_epi64(codeRow + 8);
    dist = _mm512_add_epi64(dist, _mm512_popcnt_epi64(_mm512_xor_si512(qr, cr)));

    return (DistType)_mm512_reduce_add_epi64(dist);
  } else if constexpr (StepsStatic == 64 / sizeof(ChunkType)) {
    __m512i qr   = _mm512_load_epi64(queryRow);
    __m512i cr   = _mm512_load_epi64(codeRow);
    __m512i dist = _mm512_popcnt_epi64(_mm512_xor_si512(qr, cr));

    return (DistType)_mm512_reduce_add_epi64(dist);
  } else {
    ABORT("Unsupported number of steps {}", StepsStatic);
  }
}

}  // namespace lsh
}  // namespace marian

#include "layers/lsh_tmp.h"

namespace marian {
namespace lsh {

void hammingTopK_AVX512_POPCNT(const Parameters& parameters, const GatherFn& gather) {
  hammingTopK<InstructionSet::AVX512_POPCNT>(parameters, gather);
}

}  // namespace lsh
}  // namespace marian
