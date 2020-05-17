#pragma once

#include "common/definitions.h"
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
  virtual void synchronize() = 0;

  virtual void setClip(float clipValue) { clipValue_ = clipValue; }
  float getClip() { return clipValue_; }

  // Set the gemm precision
  virtual void setGemmPrecision(std::string gemmPrec) {
    if (gemmPrec == "float32") {
      return; // This is the default precisoin.
    } else if (gemmPrec == "int16") {
      setInt16(true);
    } else if (gemmPrec == "int8") {
      setInt8(true);
    } else if (gemmPrec == "int8shift") {
      setInt8(true);
      setInt8Shift(true);
    } else {
      ABORT("Unsupported GEMM precision type: {}", gemmPrec);
    }
  }

  // for CPU, sets to use optimized (intgemm8/intgemm16) code for matrix multiplication.
  // for GPU, this is invalid. for GPU, all the functions below always returns false and the setters abort.
  virtual void setInt16(bool int16) = 0;
  virtual bool isInt16() = 0;
  virtual void setInt8(bool int8) = 0;
  virtual bool isInt8() = 0;
  virtual void setInt8Shift(bool shifted) = 0;
  virtual bool isInt8Shift() = 0;
};

Ptr<Backend> BackendByDeviceId(DeviceId deviceId, size_t seed);

}  // namespace marian
