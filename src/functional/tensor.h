#pragma once

#include "functional/array.h"
#include "functional/shape.h"
#include "tensors/tensor.h"

namespace marian {
namespace functional {

template <typename T>
struct Tensor {
  T* data_;
  functional::Shape shape_;

  __HD__ Tensor() {}

  __HD__ Tensor(T* ptr, const functional::Shape& shape)
      : data_(ptr), shape_(shape) {}

  // @TODO: take into account the change in shape when using float4 or float8
  __H__ Tensor(marian::Tensor t) : data_(t->data<T>()), shape_(t->shape()) {}

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