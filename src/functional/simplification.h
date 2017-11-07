#pragma once

#include "functional/operands.h"
#include "functional/predicates.h"

namespace marian {
  namespace functional {

    template <class F>
    using IsFunctor = typename std::enable_if<F::isFunctor(), bool>::type;

    template <class Y, IsFunctor<Y> = true>
    __HDI__ static Y cut(Plus<C<0>, Y> f) {
      return f.y;
    }

    template <class X, IsFunctor<X> = true>
    __HDI__ static X cut(Plus<X, C<0>> f) {
      return f.x;
    }

    template <int N, int K>
    __HDI__ static C<N + K> cut(Plus<C<N>, C<K>> f) {
      return C<N + K>();
    }

    template <class X, IsFunctor<X> = true>
    __HDI__ static X cut(Minus<X, C<0>> f) {
      return f.x;
    }

    template <int N, int K>
    __HDI__ static C<N - K> cut(Minus<C<N>, C<K>> f) {
      return C<N - K>();
    }

    template <class X, IsFunctor<X> = true>
    __HDI__ static C<0> cut(Minus<X, X> f) {
      return C<0>();
    }

    template <class Y, IsFunctor<Y> = true>
    __HDI__ static Neg<Y> cut(Minus<C<0>, Y> f) {
      return -f.y;
    }

    template <class X>
    __HDI__ static X cut(Neg<Neg<X>> f) {
      return f.x.x;
    }

    template <int N, int K>
    __HDI__ static C<N * K> cut(Mult<C<N>, C<K>> f) {
      return C<N * K>();
    }

    template <int N, int K, class Y>
    __HDI__ static Mult<C<N * K>, Y> cut(Mult<C<N>, Mult<C<K>, Y>> f) {
      return cut(C<N * K>() * f.y.y);
    }

    template <class X, IsFunctor<X> = true>
    __HDI__ static C<0> cut(Mult<X, C<0>> f) {
      return C<0>();
    }

    template <class X, class Y>
    __HDI__ static Mult<X, Y> cut(Mult<Neg<X>, Neg<Y>> f) {
      return cut(f.x.x * f.y.x);
    }

    template <class X, class Y>
    __HDI__ static Neg<Mult<X, Y>> cut(Mult<X, Neg<Y>> f) {
      return -cut(f.x * f.y.x);
    }

    template <class X, class Y>
    __HDI__ static Neg<Mult<X, Y>> cut(Mult<Neg<X>, Y> f) {
      return -cut(f.x.x * f.y);
    }

    template <class Y, IsFunctor<Y> = true>
    __HDI__ static C<0> cut(Mult<C<0>, Y> f) {
      return C<0>();
    }

    template <class X, IsFunctor<X> = true>
    __HDI__ static X cut(Mult<X, C<1>> f) {
      return f.x;
    }

    template <class Y, IsFunctor<Y> = true>
    __HDI__ static Y cut(Mult<C<1>, Y> f) {
      return f.y;
    }

    template <class Y, IsFunctor<Y> = true>
    __HDI__ static C<0> cut(Div<C<0>, Y> f) {
      return C<0>();
    }

    template <class X, IsFunctor<X> = true>
    __HDI__ static X cut(Div<X, C<1>> f) {
      return f.x;
    }

    template <class X, class Y, IsFunctor<Y> = true>
    __HDI__ static auto cut(Mult<X, Div<C<1>, Y>> f)
    ->decltype(cut(f.x / f.y.y)) {
      return cut(f.x / f.y.y);
    }

    template <class X, IsFunctor<X> = true>
    __HDI__ static C<1> cut(Div<X, X> f) {
      return C<1>();
    }

    template <int N>
    __HDI__ static C<0> cut(Div<C<0>, C<N>> f) {
      return C<0>();
    }

    template <class X, IsFunctor<X> = true>
    __HDI__ static C<1> cut(Pow<X, C<0>> f) {
      return C<1>();
    }

    template <class X, IsFunctor<X> = true>
    __HDI__ static X cut(Pow<X, C<1>> f) {
      return f.x;
    }

    __HDI__ static C<1> cut(Exp<C<0>> f) {
      return C<1>();
    }

    template <class X, class Y>
    __HDI__ static Y cut(IfThenElse<X, Y, Y> f) {
      return f.y;
    }

    template <class X>
    __HDI__ static X cut(X x) {
      return x;
    }

    template <class X>
    __HDI__ static X simplify(X x) {
      return x;
    }

    template <class F, class X>
    __HDI__ static auto simplify(UnaryFunctor<F, X> f)
    ->decltype(cut(UnaryFunctor<F, decltype(simplify(f.x))>(simplify(f.x)))) {
      return cut(UnaryFunctor<F, decltype(simplify(f.x))>(simplify(f.x)));
    }

    template <class F, class X, class Y>
    __HDI__ static auto simplify(BinaryFunctor<F, X, Y> f)
    ->decltype(cut(BinaryFunctor<F, decltype(simplify(f.x)), decltype(simplify(f.y))>(simplify(f.x), simplify(f.y)))) {
      return
      cut(BinaryFunctor<F, decltype(simplify(f.x)),
                           decltype(simplify(f.y))>(
            simplify(f.x), simplify(f.y)));
    }

    template <class F, class X, class Y, class Z>
    __HDI__ static auto simplify(TernaryFunctor<F, X, Y, Z> f)
    ->decltype(cut(TernaryFunctor<F, X, decltype(simplify(f.y)), decltype(simplify(f.z))>(f.x, simplify(f.y), simplify(f.z)))) {
      return
      cut(TernaryFunctor<F, X, decltype(simplify(f.y)),
                               decltype(simplify(f.z))>(
            f.x, simplify(f.y), simplify(f.z)));
    }


  }
}