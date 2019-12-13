#pragma once

#include <functional>
#include <random>

#include "common/config.h"
#include "tensors/backend.h"

namespace marian {
namespace cpu {

class Backend : public marian::Backend {
protected:
  bool optimized_{false};
  bool optimized8_{false};

public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {}
  void setDevice() override {}
  void synchronize() override {}

  // for CPU & inference only, sets to use optimized code for inference. Does nothing for GPU.
  void setOptimized(bool optimize) override { optimized_ = optimize; }
  bool isOptimized() override { return optimized_; }

  void setOptimized8(bool optimize) override { optimized8_ = optimize; }
  bool isOptimized8() override { return optimized8_; }
};
}  // namespace cpu
}  // namespace marian
