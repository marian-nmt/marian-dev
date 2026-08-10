#pragma once
// HIP shim: <nccl.h> -> RCCL, ROCm's collectives library. RCCL keeps the NCCL
// names and semantics, so the communicator source needs no change; only the
// header spelling differs, and the CUDA build never sees this file.
#include <rccl/rccl.h>
