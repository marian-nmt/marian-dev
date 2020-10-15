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
  virtual void synchronize() = 0;

  virtual void setClip(float clipValue) { clipValue_ = clipValue; }
  float getClip() { return clipValue_; }

  // Set the gemm precision
  virtual void setGemmPrecision(Ptr<Options const> options) {
    std::string gemmPrecision = options->get<std::string>("gemm-precision");
    bool dumpQuantMults = options->get<bool>("dump-quantmult");
    if (dumpQuantMults) {
      setInt8(true);
      setDumpQuantMult(true);
      if (deviceId_.type == DeviceType::cpu) {
        setShifted(true);
        setShiftedAll(true);
      } else {
        setFused(true); //TensorCores might not be available so we set them separately when dumping quantmults
      }
    }

    //float32, int16, int8, int8shift, int8shiftAlpha, int8shiftAll, int8shiftAlphaAll
    if (gemmPrecision == "float32") {
      return; // This is the default precisoin.
    } else if (gemmPrecision == "int16") {
      setInt16(true);
    } else if (gemmPrecision == "int8") {
      setInt8(true);
    } else if (gemmPrecision == "int8tensor") {
      setInt8(true);
      setTensorCoreGemm(true);
    } else if (gemmPrecision == "int8Alpha") {
      setInt8(true);
      setPrecomputedAlpha(true);
    } else if (gemmPrecision == "int8tensorAlpha") {
      setInt8(true);
      setPrecomputedAlpha(true);
      setTensorCoreGemm(true);
    } else if (gemmPrecision == "int8Fused") {
      setInt8(true);
      setFused(true);
    } else if (gemmPrecision == "int8tensorFused") {
      setInt8(true);
      setTensorCoreGemm(true);
      setFused(true);
    } else if (gemmPrecision == "int8AlphaFused") {
      setInt8(true);
      setPrecomputedAlpha(true);
      setFused(true);
    } else if (gemmPrecision == "int8tensorAlphaFused") {
      setInt8(true);
      setPrecomputedAlpha(true);
      setTensorCoreGemm(true);
      setFused(true);
    } else if (gemmPrecision == "int8shift") {
      setInt8(true);
      setShifted(true);
    } else if (gemmPrecision == "int8shiftAlpha") {
      setInt8(true);
      setShifted(true);
      setPrecomputedAlpha(true);
    } else if (gemmPrecision == "int8shiftAll") {
      setInt8(true);
      setShifted(true);
      setShiftedAll(true);
    } else if (gemmPrecision == "int8shiftAlphaAll") {
      setInt8(true);
      setShifted(true);
      setShiftedAll(true);
      setPrecomputedAlpha(true);
    }else {
      ABORT("Unsupported GEMM precision type: {}", gemmPrecision);
    }
  }

  // for CPU, sets to use optimized code for inference.
  // for GPU, this is invalid. for gpu, isOptimized() function always returns false.
  virtual void setInt16(bool optimize) = 0;
  virtual bool isInt16() = 0;
  virtual void setInt8(bool optimize) = 0;
  virtual bool isInt8() = 0;
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
  virtual void setTensorCoreGemm(bool tensorCore) = 0;
  virtual bool useTensorCoreGemm() = 0;
  virtual void setFused(bool fused) = 0;
  virtual bool isFused() = 0;
};

Ptr<Backend> BackendByDeviceId(DeviceId deviceId, size_t seed);

}  // namespace marian
