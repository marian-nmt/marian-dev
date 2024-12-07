#include "layers/lsh.h"

#include "common/logging.h"
#include "tensors/cpu/cpu_info.h"

namespace marian {
namespace lsh {

using namespace ::marian::cpu;

template <InstructionSet IS, size_t StepsStatic>
inline typename std::enable_if<IS == InstructionSet::NEON, DistType>::type
hamming(ChunkType* queryRow, ChunkType* codeRow) {
  // Number of 128-bit vectors to process

  constexpr size_t STEP_SIZE = sizeof(uint8x16_t) / sizeof(ChunkType);
  constexpr size_t STEPS = StepsStatic / STEP_SIZE;
  ABORT_IF(STEP_SIZE == 0, "STEP_SIZE is 0??");

  DistType distance = 0;
  for (size_t i = 0; i < STEPS; i += STEP_SIZE) {
    // Load 16 bytes (128 bits) from queryRow and codeRow
    const uint8x16_t vec_query = vld1q_u8((uint8_t*)&queryRow[i]);
    const uint8x16_t vec_code  = vld1q_u8((uint8_t*)&codeRow[i]);

    // XOR the two vectors
    const uint8x16_t vec_xor = veorq_u8(vec_query, vec_code);

    // Count set bits in each byte
    const uint8x16_t vec_popcnt = vcntq_u8(vec_xor);

    // Sum the counts to get partial distance
    const uint64_t sum = vaddvq_u8(vec_popcnt);

    // Accumulate the distance
    distance += (DistType)sum;
  }
  return distance;
}

}  // namespace lsh
}  // namespace marian

#include "layers/lsh_tmp.h"

namespace marian {
namespace lsh {

void hammingTopK_NEON(const Parameters& parameters, const GatherFn& gather) {
  hammingTopK<InstructionSet::NEON>(parameters, gather);
}

}  // namespace lsh
}  // namespace marian