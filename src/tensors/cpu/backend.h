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
  bool shifted_{false};
  bool shiftedAll_{false};
  bool dumpMatrices_{false};
  bool alpha_{false};
  bool legacyBatch_{false};
  bool fused_{false};

public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {}
  void setDevice() override {}
  void synchronize() override {}

  // for CPU & inference only, sets to use optimized code for inference. Does nothing for GPU.
  void setInt16(bool optimize) override { int16_ = optimize; }
  bool isInt16() override { return int16_; }

  void setInt8(bool optimize) override { int8_ = optimize; }
  bool isInt8() override { return int8_; }

  void setShifted(bool shifted) override { shifted_ = shifted; }
  bool isShifted() override { return shifted_; }

  void setShiftedAll(bool shiftedAll) override {
    shiftedAll_ = shiftedAll;
    if (shiftedAll_) {
      shifted_ = true;
    }
  }

  bool isShiftedAll() override {
    return shiftedAll_;
  }

  void setDumpQuantMult(bool dump) override {
    dumpMatrices_ = dump;
  }

  bool DumpQuantMult() override {
    return dumpMatrices_;
  }

  void setPrecomputedAlpha(bool alpha) override {
    alpha_ = alpha;
  }
  bool isPrecomputedAlpha() override {
    return alpha_;
  }

  void setLegacyBatchedGemm(bool legacyBatch) override {
    legacyBatch_ = legacyBatch;
  }
  bool isLegacyBatchedGemm() override {
    return legacyBatch_;
  }

  void setTensorCoreGemm(bool tensorCore) override {
    tensorCore;
    LOG_ONCE(info, "setTensorCoreGemm() not supported for CPU.");
  }
  bool useTensorCoreGemm() override {
    return false;
  }
  void setFused(bool fused) override {
    fused;
    LOG_ONCE(info, "setFused() not supported for CPU.");
  }
  bool isFused() override {
    return false;
  }

};
}  // namespace cpu
}  // namespace marian
