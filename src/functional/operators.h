#pragma once

#include "common/types.h"
#include <cmath>

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
};

// Specialization for float
template <>
struct Ops<float> {
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
    return x > 0 ? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
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
};

// Specialization for float32x4 (=__m128, CPU SSE intrisics)
template <>
struct Ops<float32x4> {
  static inline float32x4 add(const float32x4& x, const float32x4& y) { return _mm_add_ps(x, y); }
  static inline float32x4 sub(const float32x4& x, const float32x4& y) { return _mm_sub_ps(x, y); }
  static inline float32x4 mul(const float32x4& x, const float32x4& y) { return _mm_mul_ps(x, y); }
  static inline float32x4 div(const float32x4& x, const float32x4& y) { return _mm_div_ps(x, y); }

  static inline float32x4 exp(const float32x4& x) {
    float32x4 ret;
    float* pRet = (float*)&ret;
    const float* ptr = (float*)&x;
    for(int i = 0; i < 4; i++)
    pRet[i] = ::expf(ptr[i]);
    return ret;
  }
};

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