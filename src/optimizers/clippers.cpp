#include "clippers.h"

#include "functional/functional.h"
#include "tensors/tensor_operators.h"

namespace marian {
void Elementwise::clip(Tensor t, float costScalingFactor) {
  using namespace functional;
  Element(_1 = functional::clip(_1, c_ * costScalingFactor), t);
}

void Norm::clip(Tensor t, float costScalingFactor) {
  using namespace functional;
  float l2Norm = L2Norm(t);
  float clipValue = c_ * costScalingFactor;
  if(l2Norm >= clipValue) {
    Element(_1 = (clipValue / l2Norm) * _1, t);
  }
}
}  // namespace marian
