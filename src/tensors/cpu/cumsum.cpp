#include "tensors/tensor_operators.h"

namespace marian {
namespace cpu {  

// wrap Marian functor to work with thrust
template <class Functor, typename T>
class AccFunctorWrapper {
private:
  Functor functor_;

public:
  AccFunctorWrapper(Functor functor) : functor_(functor) {}
  T operator()(T x, T y) { return (T)functor_((float)x, (float)y); }
};

template <class Functor>
void BatchedScan(Tensor out, const Tensor in, bool reverse, bool exclusive, Functor accOp, float zero) {
  ABORT_IF(!isFloat(in->type()),      "Input should be float type and not {}", in->type());
  ABORT_IF(out->type() != in->type(), "Output should have type {}", in->type());

  int cols = in->shape()[0];
  int rows = in->shape().elements() / cols;

  auto batchedScan = [=](auto inIt, auto outIt) {
    AccFunctorWrapper<Functor, float> accFunctor(accOp);
    
    for(int i = 0; i < rows; ++i) {
      float sum;
      int shift = exclusive ? 1 : 0;

      // handle first element differently based on exclusive flag
      if(exclusive)
        sum = zero;
      else
        sum = inIt[0];
      outIt[0] = sum;
      
      for(int j = 1; j < cols; ++j) {
        sum = accFunctor(sum, inIt[j - shift]);
        outIt[j] = sum;
      }
      
      inIt += cols;
      outIt += cols;
    }
  };

  if(reverse) {
    auto revInIt  = std::make_reverse_iterator(in->data()  + in->size());
    auto revOutIt = std::make_reverse_iterator(out->data() + out->size());
    batchedScan(revInIt, revOutIt);
  } else {
    auto fwdInIt  = in->data();
    auto fwdOutIt = out->data();
    batchedScan(fwdInIt, fwdOutIt);
  }
}

// CPU implementation of logcumsumexp operator for LogCumSumExpNodeOp
void LogCumSumExp(Tensor out, const Tensor in, bool reverse, bool exclusive, bool fast = false) {
  float max = 0;
  if(!fast) {
    // compute max of entire tensor, this is just for stabilization
    // note, if e.g. all values are logprobs, then the max is at most 0 and we can skip this step
    // maybe it should be the default to turn this off?
    max = *std::max_element(in->data(), in->data() + in->size());
  }

  using namespace functional;
  auto functor = log(exp(_1 - max) + exp(_2 - max)) + max;
  auto zero    = -NumericLimits<float>(in->type()).infinity;
  BatchedScan(out, in, reverse, exclusive, functor, zero); 
}

// CPU implementation of cumsum operator for CumSumNodeOp
void CumSum(Tensor out, const Tensor in, bool reverse, bool exclusive) {
  using namespace functional;
  auto functor = _1 + _2;
  BatchedScan(out, in, reverse, exclusive, functor, 0.f);
}

void CumProd(Tensor out, const Tensor in, bool reverse, bool exclusive) {
  using namespace functional;
  auto functor = _1 * _2;
  BatchedScan(out, in, reverse, exclusive, functor, 1.f);
}

} // namespace gpu
} // namespace marian