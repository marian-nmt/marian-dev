#pragma once

#include "common/definitions.h"
#include "common/options.h"
#include "tensors/rand.h"

namespace marian {

class Backend {
protected:
  DeviceId deviceId_;
  size_t seed_;
  Ptr<RandomGenerator> randomGenerator_;

  // global clipping value for matrix-multiplies, should soon be removed.
  float clipValue_{0.f};

public:
  Backend(DeviceId deviceId, size_t seed)
      : deviceId_(deviceId), seed_(seed), randomGenerator_(createRandomGenerator(seed, deviceId)) {}
  virtual ~Backend() {};
  virtual DeviceId getDeviceId() { return deviceId_; };
  virtual Ptr<RandomGenerator> getRandomGenerator() { return randomGenerator_; }

  // for GPU only, calls cudaSetDevice, does nothing on CPU. Maybe change name.
  virtual void setDevice() = 0;
  virtual void configureDevice(Ptr<Options const> options) = 0;
  virtual void synchronize() = 0;

  virtual void configureIntgemm(Ptr<Options const> options) {
    std::string gemmPrecision = options->get<std::string>("gemm-precision");
    bool dumpQuantMults = options->get<bool>("dump-quantmult");
    if (dumpQuantMults) {
      setOptimized8(true);
      setShifted(true);
      setShiftedAll(true);
      setDumpQuantMult(true);
      //float32, int16, int8, int8shift, int8shiftAlpha, int8shiftAll, int8shiftAlphaAll
    } else if (gemmPrecision == "float32") {
      // Default case, all variables are false. Do nothing
    } else if (gemmPrecision == "int16") {
      setOptimized(true);
    } else if (gemmPrecision == "int8") {
      setOptimized8(true);
    } else if (gemmPrecision == "int8shift") {
      setOptimized8(true);
      setShifted(true);
    } else if (gemmPrecision == "int8shiftAlpha") {
      setOptimized8(true);
      setShifted(true);
      setPrecomputedAlpha(true);
    } else if (gemmPrecision == "int8shiftAll") {
      setOptimized8(true);
      setShifted(true);
      setShiftedAll(true);
    } else if (gemmPrecision == "int8shiftAlphaAll") {
      setOptimized8(true);
      setShifted(true);
      setShiftedAll(true);
      setPrecomputedAlpha(true);
    } else {
      ABORT("Unknown option {} for command line parameter gemm-precision.", gemmPrecision);
    }
  }

  virtual void setClip(float clipValue) { clipValue_ = clipValue; }
  float getClip() { return clipValue_; }

  // for CPU, sets to use optimized code for inference.
  // for GPU, this is invalid. for gpu, isOptimized() function always returns false.
  virtual void setOptimized(bool optimize) = 0;
  virtual bool isOptimized() = 0;
  virtual void setOptimized8(bool optimize) = 0;
  virtual bool isOptimized8() = 0;
  virtual void setShifted(bool shifted) = 0;
  virtual bool isShifted() = 0;
  virtual void setShiftedAll(bool shifted) = 0;
  virtual bool isShiftedAll() = 0;
  virtual void setDumpQuantMult(bool dump) = 0;
  virtual bool DumpQuantMult() = 0;
  virtual void setPrecomputedAlpha(bool alpha) = 0;
  virtual bool isPrecomputedAlpha() = 0;
  virtual void setLegacyBatchedGemm(bool legacyBatch) = 0;
  virtual bool isLegacyBatchedGemm() = 0;
};

Ptr<Backend> BackendByDeviceId(DeviceId deviceId, size_t seed);

}  // namespace marian
