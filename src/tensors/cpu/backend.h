#pragma once

#include <functional>
#include <random>

#include "common/config.h"
#include "tensors/backend.h"

namespace marian {
namespace cpu {

class Backend : public marian::Backend {
protected:
  bool int16_{false};
  bool int8_{false};
  bool int8shift_{false};

public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {}
  void setDevice() override {}
  void synchronize() override {}

  // for CPU & inference only, sets to use optimized code for inference. Does nothing for GPU.
  void setInt16(bool optimize) override { int16_ = optimize; }
  bool isInt16() override { return int16_; }

  void setInt8(bool optimize) override { int8_ = optimize; }
  bool isInt8() override { return int8_; }

  void setInt8Shift(bool shifted) override { int8shift_ = shifted; }
  bool isInt8Shift() override { return int8shift_ && int8_; }
};
}  // namespace cpu
}  // namespace marian
