#pragma once

#include <cstdint>
#include <string>

#include "common/shape.h"

#include "functional/array.h"

namespace marian {

namespace functional {

#define CONST_SHAPE_DIMS 4

/**
 * @brief Represents the size of each dimension in a tensor.
 */

struct Slice {
  static const int END{99999}; // fix

  int begin{0};
  int end{END};
  int stride{1};

  Slice(int b, int e, int s = 1)
  : begin(b), end(e), stride(s) {}

  Slice()
  : begin(0), end(END), stride(1) {}

  Slice(int i)
  : begin(i), end(i + 1), stride(1) {}

  Slice(const std::initializer_list<int>& l) {
    std::vector<int> v(l);
    switch(v.size()) {
      case 0: begin = 0;    end = END;      stride = 1;    break;
      case 1: begin = v[0]; end = v[0] + 1; stride = 1;    break;
      case 2: begin = v[0]; end = v[1];     stride = 1;    break;
      case 3: begin = v[0]; end = v[1];     stride = v[2]; break;
      default:
        ABORT("Too many elements in slice: {}", v.size());
    }
  }
};

const Slice All;

template <const int N>
struct ConstantShape {
  Array<int, N> shape_;
  Array<int, N> stride_;
  Array<int, N> bstride_;
  size_t elements_{1};
  size_t offset_{0};

  __HD__ ConstantShape() {
    shape_.fill(1);
    stride_.fill(1);
    bstride_.fill(0);
  }

  __HD__ ConstantShape(const ConstantShape& shape)
      : shape_(shape.shape_),
        stride_(shape.stride_),
        bstride_(shape.bstride_),
        elements_(shape.elements_),
        offset_(shape.offset_) {}

  template <size_t M>
  __HD__ ConstantShape(const Array<int, M>& shape) {
    ABORT_IF(M > N, "Recompile with CONST_SHAPE_DIMS >= {}", M);

    std::copy(shape.begin(), shape.end(), shape_.begin() + N - M);
    if(N - M)
      std::fill_n(shape_.begin(), N - M, 1);

    updateStrides();
    updateElements();
  }

  __HD__ ConstantShape(const Array<int, N>& shape,
                       const Array<int, N>& stride,
                       size_t offset)
  : shape_(shape), stride_(stride), offset_(offset) {
    updateElements();
  }

  ConstantShape(const marian::Shape& shape) {
    size_t filled = shape.size();

    ABORT_IF(filled > N,
             "Recompile with CONST_SHAPE_DIMS >= " + std::to_string(filled));

    std::copy(shape.begin(), shape.end(), shape_.begin() + N - filled);
    if(N - filled)
      std::fill_n(shape_.begin(), N - filled, 1);

    updateStrides();
    updateElements();
  }

  __HDI__ void updateStrides() {
    stride_[N - 1] = 1;
    bstride_[N - 1] = shape_[N - 1] == 1 ? 0 : stride_[N - 1];

    for(int i = N - 2; i >= 0; --i) {
      stride_[i] = stride_[i + 1] * shape_[i + 1];
      bstride_[i] = shape_[i] == 1 ? 0 : stride_[i];
    }
  }

  __HDI__ void updateElements() {
    elements_ = 1;
    for(int i = 0; i < N; ++i)
      elements_ *= shape_[i];
  }

  __HDI__ void set(int i, int dim) {
    shape_[i] = dim;
    updateStrides();
    updateElements();
  }

  __HDI__ const int& dim(int i) const { return shape_[i]; }

  __HDI__ const int& back() const { return dim(N - 1); }

  __HDI__ const int& operator[](int i) const { return dim(i); }

  __HDI__ const int& stride(int i) const { return stride_[i]; }

  __HDI__ const int& bstride(int i) const { return bstride_[i]; }

  __HDI__ static constexpr size_t size() { return N; }

  __HDI__ int elements() const { return (int)elements_; }

  template <const int K, const int D> struct I {
    __HDI__ static int index(const Array<int, D>& d,
                             const Array<int, D>& stride) {
      return d[K] * stride[K] + I<K-1, D>::index(d, stride);
    }

    __HDI__ static int index(int si,
                             const Array<int, D>& shape,
                             const Array<int, D>& stride) {
        return (si % shape[K]) * stride[K] + I<K-1, D>::index(si / shape[K], shape, stride);
      }
  };

  template <const int D> struct I<0, D> {
    __HDI__ static int index(const Array<int, D>& d,
                             const Array<int, D>& stride) {
      return d[0] * stride[0];
    }

    __HDI__ static int index(int si,
                             const Array<int, D>& shape,
                             const Array<int, D>& stride) {
        return (si % shape[0]) * stride[0];
      }
  };

  __HDI__ int index(const Array<int, N>& d) const {
    return offset_ + I<N-1, N>::index(d, stride_);
  }

  __HDI__ int index(int si) const {
    return offset_ + I<N-1, N>::index(si, shape_, stride_);
  }


  __HDI__ int bindex(const Array<int, N>& d) const {
    int i = 0;
    for(int j = 0; j < N; ++j)
      i += d[j] * bstride_[j];
    return i;
  }

  __HDI__ void dims(int si, Array<int, N>& d) const {
    for(int j = N - 1; j >= 0; --j) {
      d[j] = si % shape_[j];
      si = si / shape_[j];
    }
  }

  // should this check stride
  __HDI__ bool operator==(const ConstantShape& other) const {
    for(int i = 0; i < N; ++i)
      if(shape_[i] != other[i])
        return false;
    return true;
  }

  __HDI__ bool operator!=(const ConstantShape& other) const {
    return !(*this == other);
  }

  std::string toString() const {
    std::stringstream strm;
    strm << "shape=" << (*this)[0];
    for(int i = 1; i < size(); ++i)
      strm << "x" << (*this)[i];
    strm << " size=" << elements();
    return strm.str();
  }

  ConstantShape<N> slice(const Array<Slice, N> slices) {
    Array<int, N> offsets;
    Array<int, N> shape;
    Array<int, N> stride;
    for(int i = 0; i < N; ++i) {
      int beg = slices[i].begin;
      int end = std::min(slices[i].end, shape_[i]);
      int str = slices[i].stride;

      offsets[i] = beg;
      shape[i]   = std::ceil((end - beg) / (float) str);
      stride[i]  = str * stride_[i];
    }

    int offset = index(offsets);
    return ConstantShape<N>(shape, stride, offset);
  }

  friend std::ostream& operator<<(std::ostream& strm, const ConstantShape<N>& shape) {
    strm << shape.toString();
    return strm;
  }
};

typedef ConstantShape<CONST_SHAPE_DIMS> Shape;
}  // namespace functional
}  // namespace marian
