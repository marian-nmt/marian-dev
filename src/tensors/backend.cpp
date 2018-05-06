#include "tensors/backend.h"

#ifdef CUDA_FOUND
#include "tensors/gpu/backend.h"
#endif

#include "tensors/cpu/backend.h"

namespace marian {

#ifdef CUDA_FOUND
cudaStream_t gpu::Backend::streams[MAX_DEVICES][MAX_DEVICES];
std::mutex gpu::Backend::createStreamMutex;
bool gpu::Backend::streamsCreated;
#endif

Ptr<Backend> BackendByDevice(DeviceId deviceId, size_t seed) {
#ifdef CUDA_FOUND
  if(deviceId.type == DeviceType::gpu)
    return New<gpu::Backend>(deviceId, seed);
  else
#endif
    return New<cpu::Backend>(deviceId, seed);
}
}
