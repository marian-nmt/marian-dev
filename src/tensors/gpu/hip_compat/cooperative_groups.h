#pragma once
// MOAT HIP shim: <cooperative_groups.h> -> HIP cooperative groups. Provides
// the cooperative_groups namespace with this_thread_block/sync/tiled_partition
// and thread_block_tile<N>::shfl_down used by 3rd_party/reduce_all.h.
#include <hip/hip_cooperative_groups.h>
