#pragma once

#include "common/definitions.h"
#include "tensors/rand.h"
#include <set>

namespace marian {

// GEMM type enum
typedef enum {
  Auto = 0,            // auto tuning between available GEMMs
  Float32 = 1,         // MKL based GEMM, fp32
  FbFp16Packed = 10,   // FBGEMM based fp16 GEMM with packing
  FbInt8Packed = 11,   // FBGEMM based int8 GEMM with packing
  intgemm16packed = 40,      // intgemm8 based packing
  intgemm8packed = 41,       // intgemm16 based packing
  int8tensorcore = 81  // 8bit integer decoding for use with tensor cores
} GemmType;

class Backend {
protected:
  DeviceId deviceId_;
  size_t seed_;
  Ptr<RandomGenerator> randomGenerator_;
  bool shifted_;
  bool shiftedAll_;
  bool precomputedAlpha_;
  bool dumpQuantMult_;
  bool useOneDNNOnly_;
  GemmType gemmType_{GemmType::Float32};

public:
  Backend(DeviceId deviceId, size_t seed)
      : deviceId_(deviceId), seed_(seed), randomGenerator_(createRandomGenerator(seed, deviceId)), shifted_(false), shiftedAll_(false), 
        precomputedAlpha_(false), dumpQuantMult_(false), useOneDNNOnly_(false) {}
  virtual ~Backend() {};
  virtual DeviceId getDeviceId() { return deviceId_; };
  virtual Ptr<RandomGenerator> getRandomGenerator() { return randomGenerator_; }

  // for GPU only, calls cudaSetDevice, does nothing on CPU. Maybe change name.
  virtual void setDevice() = 0;
  virtual void synchronize() = 0;

  // for CPU, sets to use optimized code for inference.
  // for GPU, this is invalid. for gpu, isOptimized() function always returns false.
  virtual void setOptimized(bool optimize) = 0;
  virtual bool isOptimized() = 0;
  // for CPU, selects different GEMM types for the inference.
  // for GPU, there's no gemm type. so, it does nothing.
  virtual void setGemmType(std::string gemmType) = 0;
  virtual GemmType getGemmType() = 0;
  // for CPU, sets quantization range of weight matrices for the inference.
  // for GPU, there's no quantization. so, it does nothing.
  virtual void setQuantizeRange(float range) = 0;
  virtual float getQuantizeRange() = 0;

  void configureIntgemm(std::vector<std::string> intgemmOpts) {
    std::set<std::string> intgemmOptsSet(intgemmOpts.begin(), intgemmOpts.end());
    if (intgemmOptsSet.find("shifted") != intgemmOptsSet.end()) {
      setShifted(true);
    }
    if (intgemmOptsSet.find("all-shifted") != intgemmOptsSet.end()) {
      setShifted(true);
      setShiftedAll(true);
    }
    if (intgemmOptsSet.find("precomputed-alpha") != intgemmOptsSet.end()) {
      setPrecomputedAlpha(true);
    }
    if (intgemmOptsSet.find("dump-quantmult") != intgemmOptsSet.end()) {
      setShifted(true);
      setShiftedAll(true);
      setDumpQuantMult(true);
    }
    if (intgemmOptsSet.find("onednn-only") != intgemmOptsSet.end()) {
      setUseOneDNNOnly(true);
      setShifted(false);
      setShiftedAll(false);
      setDumpQuantMult(false);
    }
  }

  bool isShifted() {
    return shifted_;
  }
  void setShifted(bool shifted) {
    shifted_ = shifted;
    if (deviceId_.type == DeviceType::gpu) {
      LOG(warn, "Shifted has no effect for the GPU backend");
    }
  }
  bool isShiftedAll() {
    return shiftedAll_;
  }
  void setShiftedAll(bool shiftedAll) {
    shiftedAll_ = shiftedAll;
    if (deviceId_.type == DeviceType::gpu) {
      LOG(warn, "Shifted all has no effect for the GPU backend");
    }
  }
  bool isPrecomputedAlpha() {
    return precomputedAlpha_;
  }
  void setPrecomputedAlpha(bool precomputed) {
    precomputedAlpha_ = precomputed;
  }

  void setDumpQuantMult(bool dump) {
    dumpQuantMult_ = dump;
  }

  bool DumpQuantMult() {
    return dumpQuantMult_;
  }

  void setUseOneDNNOnly(bool oneDNNOnly) {
    useOneDNNOnly_ = oneDNNOnly;
    if (deviceId_.type == DeviceType::gpu) {
      LOG(warn, "onednn-only has no effect for the GPU backend");
    }
  }

  bool useOneDNNOnly() {
    return useOneDNNOnly_;
  }

};

Ptr<Backend> BackendByDeviceId(DeviceId deviceId, size_t seed);

}  // namespace marian
