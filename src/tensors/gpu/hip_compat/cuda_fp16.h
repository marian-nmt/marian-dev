#pragma once
// HIP shim: <cuda_fp16.h> -> HIP fp16. Provides __half, __half2, the
// hsin/hcos/hlog/hexp/hsqrt/hrint/hfloor/hceil/__habs device intrinsics and
// __half_raw used by Marian.
#include <hip/hip_fp16.h>
