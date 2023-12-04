#pragma once

#include "common/definitions.h"
#include "common/hash.h"
#include "common/logging.h"

#include <random>

namespace marian {

class TensorBase;
typedef IPtr<TensorBase> Tensor;

class RandomGenerator {
protected:
  size_t seed_;

  // hashing device type and id to get a unique seed for each device, e.g. for different samples on different devices
  size_t hashSeed(size_t seed, DeviceId deviceId) {
    // on the first device, use the seed as is. This keeps unit tests etc. working correctly
    // on other devices, hash the seed with the device type and id, so that we get different seeds for different devices
    // this is important for e.g. different samples on different devices
    if(deviceId.no == 0)
      return seed;
    else
      return util::hashArgs(seed, deviceId.type, deviceId.no);
  }

public:
  RandomGenerator(size_t seed, DeviceId deviceId) 
  : seed_(hashSeed(seed, deviceId)) { 
    LOG(debug, "Setting random seed to {} (device {}{})", seed_, deviceId.typeAsString(), deviceId.no);
  }
  virtual ~RandomGenerator() {}
  virtual void uniform(Tensor, float a, float b) = 0;
  virtual void normal(Tensor, float mean, float stddev) = 0;
  virtual size_t seed() { return seed_; }
};

Ptr<RandomGenerator> createRandomGenerator(size_t /*seed*/, DeviceId);

}
