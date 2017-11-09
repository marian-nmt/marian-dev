#pragma once

#include "functional/operands.h"
#include "functional/predicates.h"
#include "functional/simplification.h"

namespace marian {
  namespace functional {

    template <int N, int K>
    __HDI__ static C<0> deriv(C<N> c, var<K> g) {
      return C<0>();
    }

    template <int N, int K>
    __HDI__ static C<N == K> deriv(var<N> x, var<K> g) {
      return C<N == K>();
    }

    template <int K>
    __HDI__ static C<0> deriv(Capture c, var<K> g) {
      return C<0>();
    }

    #define UNARY_GRAD(name, fderiv)\
    template <class X, int K>\
    __HDI__ static auto deriv(UnaryFunctor<elem::name, X> f, var<K> g)\
      ->decltype(fderiv) {\
      return fderiv;\
    }

    #define BINARY_GRAD(name, fderiv)\
    template <class X, class Y, int K>\
    __HDI__ static auto deriv(BinaryFunctor<elem::name, X, Y> f, var<K> g)\
      ->decltype(fderiv) {\
      return fderiv;\
    }

    UNARY_GRAD(Tanh,   (one - pow(f, two)) * deriv(f.x, g));
    UNARY_GRAD(Log,    deriv(f.x, g) / f);
    UNARY_GRAD(Exp,    f * deriv(f.x, g));
    UNARY_GRAD(Abs,    sgn(f.x) * deriv(f.x, g));
    UNARY_GRAD(Sqrt,   -deriv(f.x, g) / (two * f));
    UNARY_GRAD(Neg,    -deriv(f.x, g));
    UNARY_GRAD(Logit,  f * (one - f) * deriv(f.x, g));
    UNARY_GRAD(Sgn,    zero);

    BINARY_GRAD(Plus,  deriv(f.x, g) + deriv(f.y, g));
    BINARY_GRAD(Minus, deriv(f.x, g) - deriv(f.y, g));
    BINARY_GRAD(Mult,  deriv(f.x, g) * f.y + f.x * deriv(f.y, g));
    BINARY_GRAD(Div,   (deriv(f.x, g) * f.y - f.x * deriv(f.y, g)) / (f.y * f.y));
    BINARY_GRAD(Pow,   (deriv(f.y, g) * log(f.x) + f.y * deriv(f.x, g) / f.x) * f);

    #define TERNARY_GRAD(name, fderiv)\
    template <class X, class Y, class Z, int K>\
    __HDI__ static auto deriv(TernaryFunctor<elem::name, X, Y, Z> f, var<K> g)\
      ->decltype(fderiv) {\
      return fderiv;\
    }

    TERNARY_GRAD(IfThenElse, if_then_else(f.x, deriv(f.y, g), deriv(f.z, g)));
  }

  template <class F, class X>
  auto grad(F f, X x)->decltype(simplify(deriv(f, x))) {
    return simplify(deriv(f, x));
  }
}