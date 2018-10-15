#pragma once

#include "functional/array.h"
#include "functional/shape.h"
#include "tensors/tensor.h"

namespace marian {
namespace functional {

// general case, do nothing. Usually the number of elements in a tensor is correctly mirrored in the
// shape object. Only special multi-element types like floatX4 (4 floats), floatX16 (16 floats)
// and halfX2 (2 half) require special handling. Similar for multi-element integer types to be added
// later.
template <typename T>
inline marian::Shape adapt(const marian::Shape& shape) {
  return shape;
}

// modify last shape dimension to automatically map to a larger stride. We are moving now by 4 floats
// at once and need to stop earlier. This is a shallow typecast to bascially an array of 4 floats.

template <>
inline marian::Shape adapt<float32x4>(const marian::Shape& shape) {
  ABORT_IF(shape[-1] % 4 != 0,
           "Last dim ({}) is not a multiple of 4 while converting to Tensor<float32x4>",
           shape[-1]);

  marian::Shape x4Shape = shape;
  x4Shape.set(-1, shape[-1] / 4);
  return x4Shape;
}

template <typename T>
struct Tensor {
  T* data_;
  functional::Shape shape_;

  __HD__ Tensor() {}

  __HD__ Tensor(T* ptr, const functional::Shape& shape)
      : data_(ptr), shape_(shape) {}

  __H__ Tensor(marian::Tensor t) : data_(t->data<T>()), shape_(adapt<T>(t->shape())) {}

  __HDI__ T& operator[](size_t i) { return data_[i]; }
  __HDI__ const T& operator[](size_t i) const { return data_[i]; }

  __HDI__ T& operator[](
      const functional::Array<int, functional::Shape::size()>& indices) {
    return data_[shape_.index(indices)];
  }

  __HDI__ const T& operator[](
      const functional::Array<int, functional::Shape::size()>& indices) const {
    return data_[shape_.index(indices)];
  }

  __HDI__ T* data() { return data_; }
  __HDI__ const T* data() const { return data_; }

  __HDI__ Shape& shape() { return shape_; }
  __HDI__ const Shape& shape() const { return shape_; }
};

}  // namespace functional
}  // namespace marian