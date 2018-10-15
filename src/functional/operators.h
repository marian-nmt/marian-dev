#pragma once

#include "functional/defs.h"
#include "common/types.h"

namespace marian {
namespace functional {

template <typename T>
struct Ops {
  static __HDI__ T exp(const T& x) { ABORT("Unknown type"); }

  static __HDI__ T add(const T& x, const T& y) { ABORT("Unknown type"); }
  static __HDI__ T sub(const T& x, const T& y) { ABORT("Unknown type"); }
  static __HDI__ T mul(const T& x, const T& y) { ABORT("Unknown type"); }
  static __HDI__ T div(const T& x, const T& y) { ABORT("Unknown type"); }
};

template <>
struct Ops<float> {
  static __HDI__ float exp(const float& x) { return expf(x); }

  static __HDI__ float add(const float& x, const float& y) { return x + y; }
  static __HDI__ float sub(const float& x, const float& y) { return x - y; }
  static __HDI__ float mul(const float& x, const float& y) { return x * y; }
  static __HDI__ float div(const float& x, const float& y) { return x / y; }
};

template <>
struct Ops<float32x4> {
  static inline float32x4 add(const float32x4& x, const float32x4& y) {
    return _mm_add_ps(x, y);
  }

  static inline float32x4 sub(const float32x4& x, const float32x4& y) {
    return _mm_sub_ps(x, y);
  }

  static inline float32x4 mul(const float32x4& x, const float32x4& y) {
    return _mm_mul_ps(x, y);
  }

  static inline float32x4 div(const float32x4& x, const float32x4& y) {
    return _mm_div_ps(x, y);
  }

  static inline float32x4 exp(const float32x4& x) {
    float32x4 ret;
    float* pRet = (float*)&ret;
    const float* ptr = (float*)&x;
    for(int i = 0; i < 4; i++)
    pRet[i] = ::expf(ptr[i]);
    return ret;
  }
};

}
}