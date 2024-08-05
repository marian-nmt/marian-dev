#include "tensors/tensor_operators.h"
#include "tensors/gpu/cuda_helpers.h"
#include "tensors/allocator.h"

#include "functional/operators.h"

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/functional.h>

namespace marian {
namespace gpu {  

// small operator to compute the row id of an element in a 2d tensor
class ProjectToRow : public thrust::unary_function<int, int> {
private:
  int cols_;

public:
  ProjectToRow(int cols) : cols_(cols) {}
  HOST_DEVICE int operator()(int i) { return i / cols_; }
};

// create the iterators to group the elements of a 2d tensor by row
auto rowIterators(int rows, int cols) {
    thrust::counting_iterator<int> firstElement(0);
    auto begin = thrust::make_transform_iterator(firstElement,                ProjectToRow(cols));
    auto end   = thrust::make_transform_iterator(firstElement + rows * cols,  ProjectToRow(cols));
    return std::make_pair(begin, end);
};

// create the iterators to group the elements of a 2d tensor by row
auto rowIterators(const Shape& shape) {
  // use last dimension as column size
  int cols = shape[-1];
  // compute number of rows from total number of elements and column size
  int rows = shape.elements() / cols;
  return rowIterators(rows, cols);
}

// wrap Marian functor to work with thrust
template <class Functor, typename T>
class AccFunctorWrapper {
private:
  Functor functor_;

public:
  AccFunctorWrapper(Functor functor) : functor_(functor) {}
  HOST_DEVICE T operator()(T x, T y) { return (T)functor_((float)x, (float)y); }
};

template <typename T, class Functor>
void TypedBatchedScan(Tensor out, const Tensor in, bool reverse, bool exclusive, Functor accOpFunctor, T zero) {
  // use thrust device_ptr to wrap raw pointers
  thrust::device_ptr<const T> inData(in->data<T>());
  thrust::device_ptr<T>       outData(out->data<T>());

  // currently use default stream
  auto exec    = thrust::cuda::par;
  auto equalOp = thrust::equal_to<int>();
  auto accOp   = AccFunctorWrapper<Functor, T>(accOpFunctor);

  auto batchedScan = [=](auto inIt, auto outIt) { 
    // treat each row as as set of keys, only works for last dimension
    const auto range = rowIterators(in->shape());
    auto begin = range.first;
    auto end   = range.second;
    if(exclusive)
      thrust::exclusive_scan_by_key(exec, begin, end, inIt, outIt, zero, equalOp, accOp);
    else
      thrust::inclusive_scan_by_key(exec, begin, end, inIt, outIt, equalOp, accOp);
  };

  if(reverse) {
    auto revInIt  = thrust::make_reverse_iterator(inData  + in->size());
    auto revOutIt = thrust::make_reverse_iterator(outData + out->size());
    batchedScan(revInIt, revOutIt);
  } else {
    auto fwdInIt  = inData;
    auto fwdOutIt = outData;
    batchedScan(fwdInIt, fwdOutIt);
  }
}

template <class Functor>
void BatchedScan(Tensor out, const Tensor in, bool reverse, bool exclusive, Functor functor, float zero) {
  ABORT_IF(!isFloat(in->type()),      "Input should be float type and not {}", in->type());
  ABORT_IF(out->type() != in->type(), "Output should have type {}", in->type());

  if(in->type() == Type::float32) {
    TypedBatchedScan<float>(out, in, reverse, exclusive, functor, zero);
#if COMPILE_FP16
  } else if(in->type() == Type::float16) {
    TypedBatchedScan<__half>(out, in, reverse, exclusive, functor, __float2half(zero));
#endif
  } else {
    ABORT("BatchedScan not implemented for type {}", in->type());
  }
}

template <typename T>
T typedMaxElement(const Tensor in) {
  // use thrust device_ptr to wrap raw pointers
  thrust::device_ptr<const T> inData(in->data<T>());

  // currently use default stream
  auto exec = thrust::cuda::par;

  return *thrust::max_element(exec, inData, inData + in->size());
}

float MaxElement(const Tensor in) {
  ABORT_IF(!isFloat(in->type()), "Input should be float type and not {}", in->type());
  if(in->type() == Type::float32) {
    return typedMaxElement<float>(in);
#if COMPILE_FP16
  } else if(in->type() == Type::float16) {
    return __half2float(typedMaxElement<__half>(in));
#endif
  } else {
    ABORT("MaxElement not implemented for type {}", in->type());
  }
}

void LogCumSumExp(Tensor out, const Tensor in, bool reverse, bool exclusive, bool fast) {
  float max = 0;
  if(!fast) {
    // compute max of entire tensor, this is just for stabilization
    // note, if e.g. all values are logprobs, then the max is at most 0 and we can skip this step
    // maybe it should be the default to turn this off?
    max = MaxElement(in);
  }

  using namespace functional;
  auto functor = log(exp(_1 - max) + exp(_2 - max)) + max;
  auto zero    = -NumericLimits<float>(in->type()).infinity;
  BatchedScan(out, in, reverse, exclusive, functor, zero); 
}

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
