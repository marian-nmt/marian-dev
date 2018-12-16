#pragma once

#include "common/types.h"
#include <cmath>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

namespace marian {
namespace functional {

// General template, will be used for any type without specializations
// and will fail with an abort message.
template <typename T>
struct Ops {
  static __HDI__ T tanh(const T& x) { ABORT("Unknown type"); }
  static __HDI__ T sin(const T& x)  { ABORT("Unknown type"); }
  static __HDI__ T cos(const T& x)  { ABORT("Unknown type"); }
  static __HDI__ T tan(const T& x)  { ABORT("Unknown type"); }
  static __HDI__ T log(const T& x)  { ABORT("Unknown type"); }
  static __HDI__ T exp(const T& x)  { ABORT("Unknown type"); }
  static __HDI__ T abs(const T& x)  { ABORT("Unknown type"); }
  static __HDI__ T sqrt(const T& x) { ABORT("Unknown type"); }
  static __HDI__ T neg(const T& x)  { ABORT("Unknown type"); }
  static __HDI__ T sgn(const T& x)  { ABORT("Unknown type"); }

  static __HDI__ T add(const T& x, const T& y)  { ABORT("Unknown type"); }
  static __HDI__ T sub(const T& x, const T& y)  { ABORT("Unknown type"); }
  static __HDI__ T mul(const T& x, const T& y)  { ABORT("Unknown type"); }
  static __HDI__ T div(const T& x, const T& y)  { ABORT("Unknown type"); }

  static __HDI__ T max(const T& x, const T& y)  { ABORT("Unknown type"); }
  static __HDI__ T min(const T& x, const T& y)  { ABORT("Unknown type"); }
  static __HDI__ T pow(const T& x, const T& y)  { ABORT("Unknown type"); }

  static __HDI__ T negate(const T& x)  { ABORT("Unknown type"); }
  static __HDI__ T eq(const T& x, const T& y)   { ABORT("Unknown type"); }
  static __HDI__ T neq(const T& x, const T& y)  { ABORT("Unknown type"); }
  static __HDI__ T gt(const T& x, const T& y)   { ABORT("Unknown type"); }
  static __HDI__ T lt(const T& x, const T& y)   { ABORT("Unknown type"); }
  static __HDI__ T geq(const T& x, const T& y)  { ABORT("Unknown type"); }
  static __HDI__ T leq(const T& x, const T& y)  { ABORT("Unknown type"); }
  static __HDI__ T _and(const T& x, const T& y) { ABORT("Unknown type"); } // 'and' is used by gcc
  static __HDI__ T _or(const T& x, const T& y)  { ABORT("Unknown type"); } // 'or' is used by gcc

  // Neural Networks specific functions
  static __HDI__ T sigmoid(const T& x)               { ABORT("Unknown type"); }
  static __HDI__ T logaddexp(const T& x, const T& y) { ABORT("Unknown type"); }
  static __HDI__ T clip(const T& x, const T& y)      { ABORT("Unknown type"); }
  // derivative of Clip, cut-off function
  static __HDI__ T bump(const T& x, const T& y)      { ABORT("Unknown type"); }
  static __HDI__ T relu(const T& x)                  { ABORT("Unknown type"); }
  static __HDI__ T reluBack(const T& x)              { ABORT("Unknown type"); }
  static __HDI__ T prelu(const T& x, const T& y)     { ABORT("Unknown type"); }
  static __HDI__ T preluBack(const T& x, const T& y) { ABORT("Unknown type"); }

  static __HDI__ T if_then_else(const T& x, const T& y, const T& z) { ABORT("Unknown type"); }

  static __HDI__ T sumReduce(const T& x) { ABORT("Unknown type"); }
  static __HDI__ T maxReduce(const T& x) { ABORT("Unknown type"); }
  static __HDI__ T minReduce(const T& x) { ABORT("Unknown type"); }
};

// Specialization for float
template <>
struct Ops<float> {
  typedef float Single;

  static __HDI__ float tanh(const float& x) { return tanhf(x); }
  static __HDI__ float sin(const float& x)  { return sinf(x); }
  static __HDI__ float cos(const float& x)  { return cosf(x); }
  static __HDI__ float tan(const float& x)  { return tanf(x); }
  static __HDI__ float log(const float& x)  { return logf(x); }
  static __HDI__ float exp(const float& x)  { return expf(x); }
  static __HDI__ float abs(const float& x)  { return fabs(x); }
  static __HDI__ float sqrt(const float& x) { return sqrtf(x); }
  static __HDI__ float neg(const float& x)  { return -x; }
  static __HDI__ float sgn(const float& x)  { return (0 < x) - (x < 0); }

  static __HDI__ float add(const float& x, const float& y)  { return x + y; }
  static __HDI__ float sub(const float& x, const float& y)  { return x - y; }
  static __HDI__ float mul(const float& x, const float& y)  { return x * y; }
  static __HDI__ float div(const float& x, const float& y)  { return x / y; }

  static __HDI__ float max(const float& x, const float& y)  { return x < y ? y : x; }
  static __HDI__ float min(const float& x, const float& y)  { return x < y ? x : y; }
  static __HDI__ float pow(const float& x, const float& y)  { return powf(x, y); }


  static __HDI__ float negate(const float& x)  { return !(bool)x; }
  static __HDI__ float eq(const float& x, const float& y)   { return x == y; }
  static __HDI__ float neq(const float& x, const float& y)  { return x != y; }
  static __HDI__ float gt(const float& x, const float& y)   { return x > y; }
  static __HDI__ float lt(const float& x, const float& y)   { return x < y; }
  static __HDI__ float geq(const float& x, const float& y)  { return x >= y; }
  static __HDI__ float leq(const float& x, const float& y)  { return x <= y; }
  static __HDI__ float and_(const float& x, const float& y) { return x && y; } // 'and' is used by gcc
  static __HDI__ float or_(const float& x, const float& y)  { return x || y; } // 'or' is used by gcc

  // Neural Networks specific functions
  static __HDI__ float sigmoid(const float& x) {
    return /*x > 0 ? (1.f / (1.f + exp(-x))) :*/ (exp(x) / (1.f + exp(x)));
  }

  static __HDI__ float logaddexp(const float& x, const float& y) {
    // Note: This may not be ideal for CUDA; cf. CNTK implementation
    return x < y ? (y + log1pf(exp(x - y))) : (x + log1pf(exp(y - x)));
  }

  static __HDI__ float clip(const float& x, const float& y)  { return abs(x) >= y ? sgn(x) * y : x; }
  // derivative of Clip, cut-off function
  static __HDI__ float bump(const float& x, const float& y)  { return abs(x) >= y ? 0.f : 1.f; }

  static __HDI__ float relu(const float& x)     { return x > 0.f ? x : 0.f; }
  static __HDI__ float reluBack(const float& x) { return x > 0.f ? 1.f : 0.f; }

  static __HDI__ float prelu(const float& x, const float& y)     { return x > 0.f ? x : x * y; }
  static __HDI__ float preluBack(const float& x, const float& y) { return x > 0.f ? 1.f : y; }

  static __HDI__ float if_then_else(const float& x, const float& y, const float& z) { return x ? y : z; }

  static __HDI__ float sumReduce(const float& x) { return x; }
  static __HDI__ float maxReduce(const float& x) { return x; }
  static __HDI__ float minReduce(const float& x) { return x; }

};

#ifndef __CUDA_ARCH__

#include "3rd_party/sse_mathfun.h"

// Specialization for float32x8 (=__m128, CPU SSE intrisics)
template <>
struct Ops<float32x4> {
  typedef float Single;

  static inline float32x4 loop4(const std::function<float(const float&)>& f, const float32x4& x) {
    float32x4 out;
    for(int i = 0; i < 4; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i]);
    return out;
  }

  static inline float32x4 loop4(const std::function<float(const float&, const float&)>& f, const float32x4& x, const float32x4& y) {
    float32x4 out;
    for(int i = 0; i < 4; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i], ((const float*)&y)[i]);
    return out;
  }

  static inline float32x4 loop4(const std::function<float(const float&, const float&, const float&)>& f, const float32x4& x, const float32x4& y, const float32x4& z) {
    float32x4 out;
    for(int i = 0; i < 4; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i], ((const float*)&y)[i], ((const float*)&z)[i]);
    return out;
  }

  // @TODO: why is this slow?
  static inline float32x4 tanh(const float32x4& x) { // ( e^x - e^-x )/( e^x + e^-x ) = (e^2x - 1) / (e^2x + 1)
    float32x4 e2x = exp(mul(2.f, x));
    return div(sub(e2x, 1.f), add(e2x, 1.f));
  }

  static inline float32x4 sin(const float32x4& x) { return sin_ps(x); }
  static inline float32x4 cos(const float32x4& x) { return cos_ps(x); }
  static inline float32x4 tan(const float32x4& x) { return div(sin(x), cos(x)); }
  static inline float32x4 log(const float32x4& x) { return log_ps(x); }
  static inline float32x4 exp(const float32x4& x) { return exp_ps(x); }

  // @TODO: get rid of loop4 with proper intrisics
  static inline float32x4 abs(const float32x4& x)  { return loop4(Ops<float>::abs, x); }
  static inline float32x4 sqrt(const float32x4& x) { return _mm_sqrt_ps(x); }
  static inline float32x4 neg(const float32x4& x)  { return sub(0.f, x); }

  // @TODO: get rid of loop4 with proper intrisics
  static inline float32x4 sgn(const float32x4& x)  { return loop4(Ops<float>::sgn, x); }

  static inline float32x4 add(const float32x4& x, const float32x4& y) { return _mm_add_ps(x, y); }
  static inline float32x4 sub(const float32x4& x, const float32x4& y) { return _mm_sub_ps(x, y); }
  static inline float32x4 mul(const float32x4& x, const float32x4& y) { return _mm_mul_ps(x, y); }
  static inline float32x4 div(const float32x4& x, const float32x4& y) { return _mm_div_ps(x, y); }

  static inline float32x4 max(const float32x4& x, const float32x4& y) { return _mm_max_ps(x, y); }
  static inline float32x4 min(const float32x4& x, const float32x4& y) { return _mm_min_ps(x, y); }
  static inline float32x4 pow(const float32x4& x, const float32x4& y) { return exp(mul(y, log(x))); }

  // @TODO: get rid of loop4 with proper intrisics
  static inline float32x4 negate(float32x4& x)  { return loop4(Ops<float>::negate, x); }

  static inline float32x4 eq(const float32x4& x, const float32x4& y)   { return loop4(Ops<float>::eq, x, y); }
  static inline float32x4 neq(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::neq, x, y); }
  static inline float32x4 gt(const float32x4& x, const float32x4& y)   { return loop4(Ops<float>::gt, x, y); }
  static inline float32x4 lt(const float32x4& x, const float32x4& y)   { return loop4(Ops<float>::lt, x, y); }
  static inline float32x4 geq(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::geq, x, y); }
  static inline float32x4 leq(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::leq, x, y); }
  static inline float32x4 and_(const float32x4& x, const float32x4& y) { return loop4(Ops<float>::and_, x, y); } // 'and' is used by gcc
  static inline float32x4 or_(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::or_, x, y); } // 'or' is used by gcc

  // Neural Networks specific functions
  // @TODO: this is unsafe
  static inline float32x4 sigmoid(const float32x4& x) {
    float32x4 e = exp(x);
    return div(e, add(1.f, e));
  }

  // // Neural Networks specific functions
  // static __HDI__ float sigmoid(const float& x) {
  //   return x > 0 ? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
  // }

  static inline float32x4 logaddexp(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::logaddexp, x, y); }

  static inline float32x4 clip(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::clip, x, y); }
  static inline float32x4 bump(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::bump, x, y); }

  static inline float32x4 relu(const float32x4& x)  { return max(0.f, x); }

  static inline float32x4 reluBack(const float32x4& x)  { return loop4(Ops<float>::reluBack, x); }
  static inline float32x4 prelu(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::prelu, x, y); }
  static inline float32x4 preluBack(const float32x4& x, const float32x4& y)  { return loop4(Ops<float>::preluBack, x, y); }

  static inline float32x4 if_then_else(const float32x4& x, const float32x4& y, const float32x4& z) { return loop4(Ops<float>::if_then_else, x, y, z);  }

  static inline Single sumReduce(const float32x4& x) {
    Single sum = 0;
    for(int i = 0; i < 4; ++i)
      sum = Ops<Single>::add(sum, x[i]);
    return sum;
  }

  static inline Single maxReduce(const float32x4& x) {
    Single maxs = x[0];
    for(int i = 1; i < 4; ++i)
      maxs = Ops<Single>::max(maxs, x[i]);
    return maxs;
  }

  static inline Single minReduce(const float32x4& x) {
    Single mins = x[0];
    for(int i = 1; i < 4; ++i)
      mins = Ops<Single>::min(mins, x[i]);
    return mins;
  }


};

#include "3rd_party/avx_mathfun.h"

//*******************************************************************************************
// Specialization for float32x8 (=__m256, CPU AVX intrisics)
template <>
struct Ops<float32x8> {
  typedef float Single;


  static inline float32x8 loop8(const std::function<float(const float&)>& f, const float32x8& x) {
    float32x8 out;
    for(int i = 0; i < 8; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i]);
    return out;
  }

  static inline float32x8 loop8(const std::function<float(const float&, const float&)>& f, const float32x8& x, const float32x8& y) {
    float32x8 out;
    for(int i = 0; i < 8; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i], ((const float*)&y)[i]);
    return out;
  }

  static inline float32x8 loop8(const std::function<float(const float&, const float&, const float&)>& f, const float32x8& x, const float32x8& y, const float32x8& z) {
    float32x8 out;
    for(int i = 0; i < 8; i++)
      ((float*)&out)[i] = f(((const float*)&x)[i], ((const float*)&y)[i], ((const float*)&z)[i]);
    return out;
  }

  static inline float32x8 tanh(const float32x8& x) { // ( e^x - e^-x )/( e^x + e^-x )
    float32x8 e2x = exp(mul(2.f, x));
    return div(sub(e2x, 1.f), add(e2x, 1.f));
  }

  static inline float32x8 sin(const float32x8& x) { return sin256_ps(x); }
  static inline float32x8 cos(const float32x8& x) { return cos256_ps(x); }
  static inline float32x8 tan(const float32x8& x) { return div(sin(x), cos(x)); } // @TODO: use sincos256_ps
  static inline float32x8 log(const float32x8& x) { return log256_ps(x); }
  static inline float32x8 exp(const float32x8& x) { return exp256_ps(x); }

  // @TODO: get rid of loop8 with proper intrisics
  static inline float32x8 abs(const float32x8& x)  { return loop8(Ops<float>::abs, x); }
  static inline float32x8 sqrt(const float32x8& x) { return _mm256_sqrt_ps(x); }
  static inline float32x8 neg(const float32x8& x)  { return sub(0.f, x); }

  // @TODO: get rid of loop8 with proper intrisics
  static inline float32x8 sgn(const float32x8& x)  { return loop8(Ops<float>::sgn, x); }

  static inline float32x8 add(const float32x8& x, const float32x8& y) { return _mm256_add_ps(x, y); }
  static inline float32x8 sub(const float32x8& x, const float32x8& y) { return _mm256_sub_ps(x, y); }
  static inline float32x8 mul(const float32x8& x, const float32x8& y) { return _mm256_mul_ps(x, y); }
  static inline float32x8 div(const float32x8& x, const float32x8& y) { return _mm256_div_ps(x, y); }

  static inline float32x8 max(const float32x8& x, const float32x8& y) { return _mm256_max_ps(x, y); }
  static inline float32x8 min(const float32x8& x, const float32x8& y) { return _mm256_min_ps(x, y); }
  static inline float32x8 pow(const float32x8& x, const float32x8& y) { return exp(mul(y, log(x))); }

  // @TODO: get rid of loop8 with proper intrisics
  static inline float32x8 negate(float32x8& x)  { return loop8(Ops<float>::negate, x); }

  static inline float32x8 eq(const float32x8& x, const float32x8& y)   { return loop8(Ops<float>::eq, x, y); }
  static inline float32x8 neq(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::neq, x, y); }
  static inline float32x8 gt(const float32x8& x, const float32x8& y)   { return loop8(Ops<float>::gt, x, y); }
  static inline float32x8 lt(const float32x8& x, const float32x8& y)   { return loop8(Ops<float>::lt, x, y); }
  static inline float32x8 geq(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::geq, x, y); }
  static inline float32x8 leq(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::leq, x, y); }
  static inline float32x8 and_(const float32x8& x, const float32x8& y) { return loop8(Ops<float>::and_, x, y); } // 'and' is used by gcc
  static inline float32x8 or_(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::or_, x, y); } // 'or' is used by gcc


  // Neural Networks specific functions
  // @TODO: this is unsafe
  static inline float32x8 sigmoid(const float32x8& x) {
    float32x8 e = exp(x);
    return div(e, add(1.f, e));
  }

  static inline float32x8 logaddexp(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::logaddexp, x, y); }

  static inline float32x8 clip(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::clip, x, y); }
  static inline float32x8 bump(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::bump, x, y); }

  static inline float32x8 relu(const float32x8& x)  { return max(0.f, x); }

  static inline float32x8 reluBack(const float32x8& x)  { return loop8(Ops<float>::reluBack, x); }
  static inline float32x8 prelu(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::prelu, x, y); }
  static inline float32x8 preluBack(const float32x8& x, const float32x8& y)  { return loop8(Ops<float>::preluBack, x, y); }

  static inline float32x8 if_then_else(const float32x8& x, const float32x8& y, const float32x8& z) { return loop8(Ops<float>::if_then_else, x, y, z);  }

  static inline Single sumReduce(const float32x8& x) {
    Single sum = 0;
    for(int i = 0; i < 8; ++i)
      sum = Ops<Single>::add(sum, x[i]);
    return sum;
  }

  static inline Single maxReduce(const float32x8& x) {
    Single maxs = x[0];
    for(int i = 1; i < 8; ++i)
      maxs = Ops<Single>::max(maxs, x[i]);
    return maxs;
  }

  static inline Single minReduce(const float32x8& x) {
    Single mins = x[0];
    for(int i = 1; i < 8; ++i)
      mins = Ops<Single>::min(mins, x[i]);
    return mins;
  }
};

#endif

#ifdef __CUDA_ARCH__

// Specialization for __half
template <>
struct Ops<__half> {

  static __DDI__ __half sub(const __half x, const __half y)  { return __hadd(x, y); }
  static __DDI__ __half add(const __half x, const __half y)  { return __hadd(x, y); }
  static __DDI__ __half mul(const __half x, const __half y)  { return __hadd(x, y); }
  static __DDI__ __half div(const __half x, const __half y)  { return __hadd(x, y); }

  static __DDI__ __half tanh(const __half& x) { return __float2half(tanhf(__half2float(x))); }
  static __DDI__ __half sin(const __half& x)  { return 0; /*__hsin(x);*/ }
  static __DDI__ __half cos(const __half& x)  { return 0; /*__hcos(x);*/ }
  static __DDI__ __half tan(const __half& x)  { return __float2half(tanf(__half2float(x))); }
  static __DDI__ __half log(const __half& x)  { return 0; /*__hlog(x);*/ }
  static __DDI__ __half exp(const __half& x)  { return 0; /*__hexp(x);*/ }
  static __DDI__ __half abs(const __half& x)  { return __float2half(fabs(__half2float(x))); }
  static __DDI__ __half sqrt(const __half& x) { return 0; /*__hsqrt(x);*/ }
  static __DDI__ __half neg(const __half& x)  { return __float2half(-__half2float(x)); }
  static __DDI__ __half sgn(const __half& x)  { return __float2half((0 < __half2float(x)) - (__half2float(x) < 0)); }


  static __DDI__ __half max(const __half& x, const __half& y)  { return 0 /* x < y ? y : x*/; }
  static __DDI__ __half min(const __half& x, const __half& y)  { return 0 /*x < y ? x : y*/; }
  static __DDI__ __half pow(const __half& x, const __half& y)  { return 0; /*powf(x, y)*/; }


  static __DDI__ __half negate(const __half& x)  { return !(bool)x; }
  static __DDI__ __half eq(const __half& x, const __half& y)   { return 0 /*x == y*/; }
  static __DDI__ __half neq(const __half& x, const __half& y)  { return 0 /*x != y*/; }
  static __DDI__ __half gt(const __half& x, const __half& y)   { return 0 /*x > y*/; }
  static __DDI__ __half lt(const __half& x, const __half& y)   { return 0 /*x < y*/; }
  static __DDI__ __half geq(const __half& x, const __half& y)  { return 0 /*x >= y*/; }
  static __DDI__ __half leq(const __half& x, const __half& y)  { return 0; /*x <= y*/; }
  static __DDI__ __half and_(const __half& x, const __half& y) { return 0 /*x && y*/; } // 'and' is used by gcc
  static __DDI__ __half or_(const __half& x, const __half& y)  { return 0 /*x || y*/; } // 'or' is used by gcc

  // Neural Networks specific functions
  static __DDI__ __half sigmoid(const __half& x) {
    return /*x > 0 ? (1.f / (1.f + exp(-x))) :*/ 0 ; /*(exp(x) / (__float2half(1.f) + exp(x)))*/;
  }

  static __DDI__ __half log1ph(__half x) {
    return __float2half(log1pf(__half2float(x)));
  }

  static __DDI__ __half logaddexp(const __half& x, const __half& y) {
    // Note: This may not be ideal for CUDA; cf. CNTK implementation
    return 0; /*x < y ? (y + log1ph(exp(x - y))) : (x + log1ph(exp(y - x)))*/;
  }

  static __DDI__ __half clip(const __half& x, const __half& y)  { return 0; /*abs(x) >= y ? sgn(x) * y : x*/; }
  // derivative of Clip, cut-off function
  static __DDI__ __half bump(const __half& x, const __half& y)  { return 0; /* abs(x) >= y ? __float2half(0.f) : __float2half(1.f);*/ }

  static __DDI__ __half relu(const __half& x)     { return  0; }
  static __DDI__ __half reluBack(const __half& x) { return  0; }

  static __DDI__ __half prelu(const __half& x, const __half& y)     { return 0; }
  static __DDI__ __half preluBack(const __half& x, const __half& y) { return 0; }

  static __DDI__ __half if_then_else(const __half& x, const __half& y, const __half& z) { return  0; }

  static __DDI__ __half sumReduce(const __half& x) { return x; }
  static __DDI__ __half maxReduce(const __half& x) { return x; }
  static __DDI__ __half minReduce(const __half& x) { return x; }

};
#endif

//*******************************************************************************************

#include "functional/defs.h"
#include "functional/predicates.h"

UNARY(Tanh,    tanh,       Ops<ElementType>::tanh(x));
UNARY(Sin,     sin,        Ops<ElementType>::sin(x));
UNARY(Cos,     cos,        Ops<ElementType>::cos(x));
UNARY(Tan,     tan,        Ops<ElementType>::tan(x));
UNARY(Log,     log,        Ops<ElementType>::log(x));
UNARY(Exp,     exp,        Ops<ElementType>::exp(x));
UNARY(Abs,     abs,        Ops<ElementType>::abs(x));
UNARY(Sqrt,    sqrt,       Ops<ElementType>::sqrt(x));
UNARY(Neg,     operator-,  Ops<ElementType>::neg(x));
UNARY(Sgn,     sgn,        Ops<ElementType>::sgn(x));

BINARY(Plus,   operator+,  Ops<ElementType>::add(x, y));
BINARY(Minus,  operator-,  Ops<ElementType>::sub(x, y));
BINARY(Mult,   operator*,  Ops<ElementType>::mul(x, y));
BINARY(Div,    operator/,  Ops<ElementType>::div(x, y));
BINARY(Max,    max,        Ops<ElementType>::max(x, y));
BINARY(Min,    min,        Ops<ElementType>::min(x, y));
UNARY(Negate,  operator!,  Ops<ElementType>::negate(x));
BINARY(Eq,     operator==, Ops<ElementType>::eq(x, y));
BINARY(NEq,    operator!=, Ops<ElementType>::neq(x, y));
BINARY(Gt,     operator>,  Ops<ElementType>::gt(x, y));
BINARY(Lt,     operator<,  Ops<ElementType>::lt(x, y));
BINARY(Geq,    operator>=, Ops<ElementType>::geq(x, y));
BINARY(Leq,    operator<=, Ops<ElementType>::leq(x, y));
BINARY(And,    operator&&, Ops<ElementType>::and_(x, y));
BINARY(Or,     operator||, Ops<ElementType>::or_(x, y));
BINARY(Pow,    pow,        Ops<ElementType>::pow(x, y));

TERNARY(IfThenElse, if_then_else, Ops<ElementType>::if_then_else(x, y, z));

// Neural Networks specific functions
BINARY(Clip,       clip,      Ops<ElementType>::clip(x, y));
// derivative of Clip, cut-off function
BINARY(Bump,       bump,      Ops<ElementType>::bump(x, y));

UNARY(Sigmoid,     sigmoid,   Ops<ElementType>::sigmoid(x));
BINARY(LogAddExp,  logaddexp, Ops<ElementType>::logaddexp(x, y));
UNARY(sReLU,       ReLU,      Ops<ElementType>::relu(x));
UNARY(sReLUBack,   ReLUback,  Ops<ElementType>::reluBack(x));
BINARY(sPReLU,     PReLU,     Ops<ElementType>::prelu(x, y));
BINARY(sPReLUBack, PReLUback, Ops<ElementType>::preluBack(x, y));


}
}