#pragma once

#include <map>
#include <memory>

#include "tensors/tensor.h"
#include "tensors/allocator.h"

namespace marian {

// @TODO: modify computation graph to group all paramters in single matrix
// object.
// This will allow to perform a single large SGD update per batch. Currently
// there
// are as many updates as different parameters.

class ClipperBase {
protected:
  Ptr<Allocator> allocator_;

public:
  virtual float clip(Tensor, float /*costScalingFactor*/ = 1.f) = 0;
  virtual void setAllocator(Ptr<Allocator> allocator) { allocator_ = allocator; }
};

typedef std::shared_ptr<ClipperBase> ClipperPtr;

class Elementwise : public ClipperBase {
public:
  Elementwise(float c = 10.0) : c_(c) {}

  float clip(Tensor t, float costScalingFactor = 1.f) override;

private:
  float c_;
};

class Norm : public ClipperBase {
public:
  Norm(float c = 1.0) : c_(c) {}

  float clip(Tensor t, float costScalingFactor = 1.f) override;

private:
  float c_;
};

template <class Algorithm, typename... Args>
ClipperBasePtr Clipper(Args&&... args) {
  return ClipperBasePtr(new Algorithm(args...));
}
}  // namespace marian
