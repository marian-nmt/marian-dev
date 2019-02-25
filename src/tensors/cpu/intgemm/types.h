#pragma once

namespace intgemm {

typedef unsigned int Index;

// If you want to detect the CPU and dispatch yourself, here's what to use:
typedef enum {CPU_AVX512BW = 4, CPU_AVX2 = 3, CPU_SSSE3 = 2, CPU_SSE2 = 1, CPU_UNSUPPORTED} CPUType;

// Running CPU type.  This is defined in intgemm.cc (as the dispatcher).
extern const CPUType kCPU;

} // namespace intgemm
