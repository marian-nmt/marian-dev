#include "layers/lsh.h"

#include "common/logging.h"
#include "tensors/cpu/cpu_info.h"

#ifdef _MSC_VER
#define __builtin_popcountl __popcnt64
#define __builtin_popcount __popcnt
#endif

namespace marian {
namespace lsh {

void hammingTopK_AVX512_POPCNT(const Parameters& parameters, const GatherFn& gather);
void hammingTopK_NEON(const Parameters& parameters, const GatherFn& gather);

// Popcount implementation for 64-bit integers, generic version in GCC/Clang
inline DistType popcount_generic(const ChunkType& chunk) {
  switch (sizeof(ChunkType)) {
    case 8 : return (DistType)__builtin_popcountl((uint64_t)chunk);
    case 4 : return (DistType)__builtin_popcount((uint32_t)chunk);
    default: ABORT("Size {} not supported", sizeof(ChunkType));
  }
}

template <::marian::cpu::InstructionSet IS, size_t StepsStatic>
inline std::enable_if_t<IS == ::marian::cpu::InstructionSet::None, DistType>
hamming(ChunkType* queryRow, ChunkType* codeRow) {
  DistType dist = 0;
  for(int i = 0; i < StepsStatic; ++i)
    dist += popcount_generic(queryRow[i] ^ codeRow[i]);
  return dist;
}

}
}

#include "layers/lsh_tmp.h"

namespace marian {
namespace lsh {

void hammingTopK(const Parameters& parameters, const GatherFn& gather) {
  using namespace marian::cpu;

  static const auto highestInstructionSet = getSupportedInstructionSet();
  switch (highestInstructionSet) {
#if defined(COMPILE_FOR_INTEL)
    case InstructionSet::AVX512_POPCNT:
      hammingTopK_AVX512_POPCNT(parameters, gather);
      break;
    case InstructionSet::AVX512:
    case InstructionSet::AVX2:
    case InstructionSet::AVX:
    case InstructionSet::SSE4_2:
    case InstructionSet::None:
      hammingTopK<InstructionSet::None>(parameters, gather);
      break;
#endif
#if defined(COMPILE_FOR_ARM)
    case InstructionSet::NEON:
      hammingTopK_NEON(parameters, gather);
      break;
#endif
    default:
      hammingTopK<InstructionSet::None>(parameters, gather);
      break;
  }
}

}  // namespace lsh
}  // namespace marian
