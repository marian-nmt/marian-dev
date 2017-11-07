#pragma once

#include "functional/operands.h"
#include "functional/predicates.h"

namespace marian {
  namespace functional {

    template <int N, int K>
    __HDI__ static C<0> grad(C<N> c, var<K> g) {
      return C<0>();
    }

    template <int N, int K>
    __HDI__ static C<N == K> grad(var<N> x, var<K> g) {
      return C<N == K>();
    }

    template <int K>
    __HDI__ static C<0> grad(Capture c, var<K> g) {
      return C<0>();
    }

    #define UNARY_GRAD(name, fgrad)\
    template <class X, int K>\
    __HDI__ static auto grad(UnaryFunctor<elem::name, X> f, var<K> g)\
      ->decltype(fgrad) {\
      return fgrad;\
    }

    #define BINARY_GRAD(name, fgrad)\
    template <class X, class Y, int K>\
    __HDI__ static auto grad(BinaryFunctor<elem::name, X, Y> f, var<K> g)\
      ->decltype(fgrad) {\
      return fgrad;\
    }

    UNARY_GRAD(Tanh,   (one - pow(f, two)) * grad(f.x, g));
    UNARY_GRAD(Log,    grad(f.x, g) / f);
    UNARY_GRAD(Exp,    f * grad(f.x, g));
    UNARY_GRAD(Abs,    sgn(f.x) * grad(f.x, g));
    UNARY_GRAD(Sqrt,   -grad(f.x, g) / (two * f));
    UNARY_GRAD(Neg,    -grad(f.x, g));
    UNARY_GRAD(Logit,  f * (one - f) * grad(f.x, g));
    UNARY_GRAD(Sgn,    zero);

    BINARY_GRAD(Plus,  grad(f.x, g) + grad(f.y, g));
    BINARY_GRAD(Minus, grad(f.x, g) - grad(f.y, g));
    BINARY_GRAD(Mult,  grad(f.x, g) * f.y + f.x * grad(f.y, g));
    BINARY_GRAD(Div,   (grad(f.x, g) * f.y - f.x * grad(f.y, g)) / (f.y * f.y));
    BINARY_GRAD(Pow,   (grad(f.y, g) * log(f.x) + f.y * grad(f.x, g) / f.x) * f);

    #define TERNARY_GRAD(name, fgrad)\
    template <class X, class Y, class Z, int K>\
    __HDI__ static auto grad(TernaryFunctor<elem::name, X, Y, Z> f, var<K> g)\
      ->decltype(fgrad) {\
      return fgrad;\
    }

    TERNARY_GRAD(IfThenElse, if_then_else(f.x, grad(f.y, g), grad(f.z, g)));
  }
}