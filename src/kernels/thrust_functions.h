#pragma once

#if CUDA_FOUND
#include <cublas_v2.h>
#include <thrust/device_vector.h>

#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#else
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#endif

#include <thrust/functional.h>
#include <cmath>

namespace thrust {
namespace detail {
namespace functional {

template <typename T>
struct unary_exp : public thrust::unary_function<T, T> {
  CUDA_HOST CUDA_DEVICE T operator()(const T &x) const { return std::exp(x); }
};

template <typename Eval>
CUDA_HOST CUDA_DEVICE actor<composite<unary_operator<unary_exp>, actor<Eval>>>
Exp(const actor<Eval> &_1) {
  return compose(unary_operator<unary_exp>(), _1);
}

template <typename T>
struct unary_log : public thrust::unary_function<T, T> {
  CUDA_HOST CUDA_DEVICE T operator()(const T &x) const { return std::log(x); }
};

template <typename Eval>
CUDA_HOST CUDA_DEVICE actor<composite<unary_operator<unary_log>, actor<Eval>>>
Log(const actor<Eval> &_1) {
  return compose(unary_operator<unary_log>(), _1);
}

template <typename T>
struct unary_sigma : public thrust::unary_function<T, T> {
  CUDA_HOST CUDA_DEVICE T operator()(const T &x) const {
    return 1.0 / (1.0 + std::exp(-x));
  }
};

template <typename Eval>
CUDA_HOST CUDA_DEVICE actor<composite<unary_operator<unary_sigma>, actor<Eval>>>
Sigma(const actor<Eval> &_1) {
  return compose(unary_operator<unary_sigma>(), _1);
}

template <typename T>
struct unary_tanh : public thrust::unary_function<T, T> {
  CUDA_HOST CUDA_DEVICE T operator()(const T &x) const { return std::tanh(x); }
};

template <typename Eval>
CUDA_HOST CUDA_DEVICE actor<composite<unary_operator<unary_tanh>, actor<Eval>>>
Tanh(const actor<Eval> &_1) {
  return compose(unary_operator<unary_tanh>(), _1);
}

template <typename T>
struct unary_sqrt : public thrust::unary_function<T, T> {
  CUDA_HOST CUDA_DEVICE T operator()(const T &x) const { return std::sqrt(x); }
};

template <typename Eval>
CUDA_HOST CUDA_DEVICE actor<composite<unary_operator<unary_sqrt>, actor<Eval>>>
Sqrt(const actor<Eval> &_1) {
  return compose(unary_operator<unary_sqrt>(), _1);
}

template <typename T1, typename T2>
CUDA_HOST CUDA_DEVICE
    actor<composite<binary_operator<thrust::maximum>, actor<T1>, actor<T2>>>
    Max(const actor<T1> &_1, const actor<T2> &_2) {
  return compose(
      binary_operator<thrust::maximum>(), make_actor(_1), make_actor(_2));
}

template <typename T>
struct unary_relu : public thrust::unary_function<T, T> {
  CUDA_HOST CUDA_DEVICE T operator()(const T &x) const {
    return x > 0.0f ? x : 0.0f;
  }
};

template <typename Eval>
CUDA_HOST CUDA_DEVICE actor<composite<unary_operator<unary_relu>, actor<Eval>>>
ReLU(const actor<Eval> &_1) {
  return compose(unary_operator<unary_relu>(), _1);
}

template <typename T>
struct unary_reluback : public thrust::unary_function<T, T> {
  CUDA_HOST CUDA_DEVICE T operator()(const T &x) const {
    return x > 0.0f ? 1.0f : 0.0f;
  }
};

template <typename Eval>
CUDA_HOST CUDA_DEVICE
    actor<composite<unary_operator<unary_reluback>, actor<Eval>>>
    ReLUback(const actor<Eval> &_1) {
  return compose(unary_operator<unary_reluback>(), _1);
}

template <typename T>
CUDA_HOST CUDA_DEVICE int sgn(T val) {
  return (float(0) < val) - (val < float(0));
}

template <typename T>
struct binary_clip : public thrust::binary_function<T, T, T> {
  CUDA_HOST CUDA_DEVICE T operator()(const T &x, const T &y) const {
    return std::abs(x) >= y ? sgn(x) * y : x;
  }
};

template <typename T1, typename T2>
CUDA_HOST CUDA_DEVICE actor<composite<binary_operator<binary_clip>,
                                    actor<T1>,
                                    typename as_actor<T2>::type>>
Clip(const actor<T1> &_1, const T2 &_2) {
  return compose(
      binary_operator<binary_clip>(), make_actor(_1), make_actor(_2));
}

template <typename T>
struct binary_prune : public thrust::binary_function<T, T, T> {
  CUDA_HOST CUDA_DEVICE T operator()(const T &x, const T &eps) const {
    return abs(x) >= eps ? x : 0;
  }
};

template <typename T1, typename T2>
CUDA_HOST CUDA_DEVICE actor<composite<binary_operator<binary_prune>,
                                    actor<T1>,
                                    typename as_actor<T2>::type>>
Prune(const actor<T1> &_1, const T2 &_2) {
  return compose(
      binary_operator<binary_prune>(), make_actor(_1), make_actor(_2));
}

template <typename T>
struct binary_pow : public thrust::binary_function<T, T, T> {
  CUDA_HOST CUDA_DEVICE T operator()(const T &x, const T &y) const {
    float tx = x;
    if(y == (int)y && (int)y % 2 == 0)
      tx = std::abs(x);
    return std::pow(tx, y);
  }
};

template <typename T1, typename T2>
CUDA_HOST CUDA_DEVICE actor<composite<binary_operator<binary_pow>,
                                    actor<T1>,
                                    typename as_actor<T2>::type>>
Pow(const actor<T1> &_1, const T2 &_2) {
  return compose(binary_operator<binary_pow>(), make_actor(_1), make_actor(_2));
}
}
}
}
