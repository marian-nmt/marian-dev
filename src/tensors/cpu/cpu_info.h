#pragma once

#include "common/logging.h"

#include <cstdlib>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#define COMPILE_FOR_INTEL
#endif

#if defined(__aarch64__)
#define COMPILE_FOR_ARM
#endif

#if defined(COMPILE_FOR_INTEL)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <immintrin.h>
#include <emmintrin.h>
#endif

#if defined(COMPILE_FOR_ARM)
#include <arm_neon.h>
#endif

namespace marian {
namespace cpu {

/**
 * @brief Enumeration of supported instruction sets.
 */
enum class InstructionSet : int {
  None = 1,
  SSE4_2 = 2,
  AVX = 3,
  AVX2 = 4,
  AVX512 = 5,
  AVX512_POPCNT = 6,
  NEON = 7
};

#if defined(COMPILE_FOR_INTEL)

inline void CPUID(int cpuInfo[4], int function_id, int subfunction_id = 0) {
#ifdef _MSC_VER
    __cpuidex(cpuInfo, function_id, subfunction_id);
#else
    __cpuid_count(function_id, subfunction_id, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#endif
}

/**
 * @brief Checks if AVX-512 with _mm512_popcnt_epi64 is supported on the current CPU.
 *
 * @return True if AVX-512 with _mm512_popcnt_epi64 is supported, false otherwise.
 */
static inline bool isAVX512PopcntSupported() {
  int info[4];
  CPUID(info, 7, 0);
  return (info[2] & ((int)1 << 14)) != 0; // Check ECX register (info[1]) for AVX-512 POPCNT support
}

/**
 * @brief Checks if AVX-512 is supported on the current CPU.
 *
 * @return True if AVX-512 is supported, false otherwise.
 */
static inline bool isAVX512Supported() {
  int info[4];
  CPUID(info, 7);
  return (info[1] & ((int)1 << 16)) != 0; // Check EBX register (info[1]) for AVX-512 support
}

/**
 * @brief Checks if AVX2 is supported on the current CPU.
 *
 * @return True if AVX2 is supported, false otherwise.
 */
static inline bool isAVX2Supported() {
  int info[4];
  CPUID(info, 7);
  return (info[1] & ((int)1 << 5)) != 0; // Check EBX register (info[1]) for AVX2 support
}

/**
 * @brief Checks if AVX is supported on the current CPU.
 *
 * @return True if AVX is supported, false otherwise.
 */
static inline bool isAVXSupported() {
  int info[4];
  CPUID(info, 1);
  return (info[2] & ((int)1 << 28)) != 0; // Check ECX register (info[2]) for AVX support
}

/**
 * @brief Checks if SSE4.2 is supported on the current CPU.
 *
 * @return True if SSE4.2 is supported, false otherwise.
 */
static inline bool isSSE4_2Supported() {
  int info[4];
  CPUID(info, 1);
  return (info[2] & ((int)1 << 20)) != 0; // Check ECX register (info[2]) for SSE4.2 support
}

#endif // COMPILE_FOR_INTEL

/**
 * @brief Gets the highest supported instruction set on the current CPU.
 *
 * This function checks the highest supported instruction set on the current CPU and returns it.
 * It also allows overriding the detected instruction set via the environment variable MJD_ENABLE_INSTRUCTIONS.
 *
 * @return The highest supported instruction set.
 */
static inline InstructionSet getSupportedInstructionSet() {
  // before testing, check the value of the environment variable
  // MJD_ENABLE_INSTRUCTIONS and return the corresponding instruction set

  char* env = std::getenv("MJD_ENABLE_INSTRUCTIONS");
  InstructionSet requested = InstructionSet::None;

  if(env) {
    std::string instrSet(env);
#if defined(COMPILE_FOR_INTEL)
    if(instrSet == "AVX512_POPCNT")
      requested = InstructionSet::AVX512_POPCNT;
    else if(instrSet == "AVX512")
      requested = InstructionSet::AVX512;
    else if(instrSet == "AVX2")
      requested = InstructionSet::AVX2;
    else if(instrSet == "AVX")
      requested = InstructionSet::AVX;
    else if(instrSet == "SSE4_2")
      requested = InstructionSet::SSE4_2;
#else
    if(instrSet == "NEON")
      requested = InstructionSet::NEON;
#endif
    else if((instrSet == "None") || (instrSet == "NONE"))
      requested = InstructionSet::None;
    else
      ABORT("Unknown requested instruction set {}", env);

    LOG(warn, "mjdgemm: Requested instructions via MJD_ENABLE_INSTRUCTIONS={}", env);
  }

  InstructionSet highest = InstructionSet::None;

#if defined(COMPILE_FOR_INTEL)
  if(isAVX512PopcntSupported())
    highest = InstructionSet::AVX512_POPCNT;
  else if(isAVX512Supported())
    highest = InstructionSet::AVX512;
  else if(isAVX2Supported())
    highest = InstructionSet::AVX2;
  else if(isAVXSupported())
    highest = InstructionSet::AVX;
  else if(isSSE4_2Supported())
    highest = InstructionSet::SSE4_2;
  else
    highest = InstructionSet::None;
#endif

#if defined(COMPILE_FOR_ARM)
  highest = InstructionSet::NEON;
#endif

  ABORT_IF(requested != InstructionSet::None && static_cast<int>(requested) > static_cast<int>(highest),
          "mjdgemm: Instructions selected via MJD_ENABLE_INSTRUCTIONS={} not available on this machine", env);

  return env ? requested : highest;
}

}
}