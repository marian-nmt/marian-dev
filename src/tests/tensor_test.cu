#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "marian.h"
#include "functional/functional.h"

namespace marian {
  namespace functional {

    template <int N, int K>
    C<0> grad(C<N> c, var<K> g) {
      return C<0>();
    }

    template <int N, int K>
    C<N == K> grad(var<N> x, var<K> g) {
      return C<N == K>();
    }

    template <int K>
    C<0> grad(Capture c, var<K> g) {
      return C<0>();
    }

    #define UNARY_GRAD(name, fgrad)\
    template <class X, int K>\
    auto grad(UnaryFunctor<elem::name, X> f, var<K> g)\
      ->decltype(fgrad) {\
      return fgrad;\
    }

    #define BINARY_GRAD(name, fgrad)\
    template <class X, class Y, int K>\
    auto grad(BinaryFunctor<elem::name, X, Y> f, var<K> g)\
      ->decltype(fgrad) {\
      return fgrad;\
    }

    UNARY_GRAD(Exp,    f * grad(f.x, g));
    UNARY_GRAD(Tanh,   (one - pow(f, two)) * grad(f.x, g));
    UNARY_GRAD(Logit,  f * (one - f) * grad(f.x, g));
    BINARY_GRAD(Plus,  grad(f.x, g) + grad(f.y, g));
    BINARY_GRAD(Minus, grad(f.x, g) - grad(f.y, g));
    BINARY_GRAD(Mult,  grad(f.x, g) * f.y + f.x * grad(f.y, g));
    BINARY_GRAD(Div,   (grad(f.x, g) * f.y - f.x * grad(f.y, g)) / (f.y * f.y));
    BINARY_GRAD(Pow,   (grad(f.y, g) * log(f.x) + f.y * grad(f.x, g) / f.x) * f);

    /*************************************************************/

    template <class F>
    using IsFunctor = typename std::enable_if<F::isFunctor(), bool>::type;

    template <class Y, IsFunctor<Y> = true>
    Y cut(Plus<C<0>, Y> f) {
      return f.y;
    }

    template <class X, IsFunctor<X> = true>
    X cut(Plus<X, C<0>> f) {
      return f.x;
    }

    template <int N, int K>
    C<N + K> cut(Plus<C<N>, C<K>> f) {
      return C<N + K>();
    }

    template <class X, IsFunctor<X> = true>
    X cut(Minus<X, C<0>> f) {
      return f.x;
    }


    template <int N, int K>
    C<N - K> cut(Minus<C<N>, C<K>> f) {
      return C<N - K>();
    }

    template <class X, IsFunctor<X> = true>
    C<0> cut(Minus<X, X> f) {
      return C<0>();
    }

    template <class Y, IsFunctor<Y> = true>
    Neg<Y> cut(Minus<C<0>, Y> f) {
      return -f.y;
    }

    template <class X>
    X cut(Neg<Neg<X>> f) {
      return f.x.x;
    }

    template <int N, int K>
    C<N * K> cut(Mult<C<N>, C<K>> f) {
      return C<N * K>();
    }

    template <int N, int K, class Y>
    Mult<C<N * K>, Y> cut(Mult<C<N>, Mult<C<K>, Y>> f) {
      return cut(C<N * K>() * f.y.y);
    }

    template <class X, IsFunctor<X> = true>
    C<0> cut(Mult<X, C<0>> f) {
      return C<0>();
    }

    template <class X, class Y>
    Mult<X, Y> cut(Mult<Neg<X>, Neg<Y>> f) {
      return cut(f.x.x * f.y.x);
    }

    template <class X, class Y>
    Neg<Mult<X, Y>> cut(Mult<X, Neg<Y>> f) {
      return -cut(f.x * f.y.x);
    }

    template <class X, class Y>
    Neg<Mult<X, Y>> cut(Mult<Neg<X>, Y> f) {
      return -cut(f.x.x * f.y);
    }

    template <class Y, IsFunctor<Y> = true>
    C<0> cut(Mult<C<0>, Y> f) {
      return C<0>();
    }

    template <class X, IsFunctor<X> = true>
    X cut(Mult<X, C<1>> f) {
      return f.x;
    }

    template <class Y, IsFunctor<Y> = true>
    Y cut(Mult<C<1>, Y> f) {
      return f.y;
    }

    template <class Y, IsFunctor<Y> = true>
    C<0> cut(Div<C<0>, Y> f) {
      return C<0>();
    }

    template <class X, IsFunctor<X> = true>
    X cut(Div<X, C<1>> f) {
      return f.x;
    }

    template <class X, class Y, IsFunctor<Y> = true>
    auto cut(Mult<X, Div<C<1>, Y>> f)
    ->decltype(cut(f.x / f.y.y)) {
      return cut(f.x / f.y.y);
    }

    template <class X, IsFunctor<X> = true>
    C<1> cut(Div<X, X> f) {
      return C<1>();
    }

    template <int N>
    C<0> cut(Div<C<0>, C<N>> f) {
      return C<0>();
    }

    template <class X, IsFunctor<X> = true>
    C<1> cut(Pow<X, C<0>> f) {
      return C<1>();
    }

    template <class X, IsFunctor<X> = true>
    X cut(Pow<X, C<1>> f) {
      return f.x;
    }

    C<1> cut(Exp<C<0>> f) {
      return C<1>();
    }

    template <class X>
    X cut(X x) {
      return x;
    }

    template <class X>
    X simplify(X x) {
      return x;
    }

    template <class F, class X, class Y>
    auto simplify(BinaryFunctor<F, X, Y> f)
    ->decltype(cut(BinaryFunctor<F, decltype(simplify(f.x)), decltype(simplify(f.y))>(simplify(f.x), simplify(f.y)))) {
      return
      cut(BinaryFunctor<F, decltype(simplify(f.x)),
                           decltype(simplify(f.y))>(
            simplify(f.x), simplify(f.y)));
    }

    template <class F, class X>
    auto simplify(UnaryFunctor<F, X> f)
    ->decltype(cut(UnaryFunctor<F, decltype(simplify(f.x))>(simplify(f.x)))) {
      return cut(UnaryFunctor<F, decltype(simplify(f.x))>(simplify(f.x)));
    }
  }
}

int main(int argc, char** argv) {

using namespace marian;
using namespace marian::functional;

var<1> x;
var<2> y;
var<3> z;

auto f = simplify(x * logit(x));

auto df_x = simplify(grad(f, x));
auto df_y = simplify(grad(f, y));
auto df_z = simplify(grad(f, z));

std::cerr << f.to_string() << std::endl;

std::cerr << df_x.to_string() << std::endl;
std::cerr << df_y.to_string() << std::endl;
std::cerr << df_z.to_string() << std::endl;

}
