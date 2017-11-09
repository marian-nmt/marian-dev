#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "marian.h"
#include "functional/functional.h"

namespace marian {
namespace op {

namespace f = marian::functional;

template <class Functor>
struct NodeComposer {
  Functor f;
  std::vector<Expr> children;

  template <int K>
  NodeComposer(Expr x) : f(f::var<K>), children({x}) {}

  NodeComposer(Functor farg, const std::vector<Expr>& args) : f(farg), children(args) {}

  operator Expr() {
    std::cerr << f.to_string() << " " << children.size() << std::endl;
    return Expr();
    //return New<FunctorNode<Functor>>(f, args);
  }
};

template <class F>
NodeComposer<F> compose(F f, const std::vector<Expr> args) {
  return NodeComposer<F>(f, args);
}

auto operator+(Expr x, Expr y)->decltype(compose(f::_1 + f::_2, {})) {
  return compose(f::_1 + f::_2, {x, y});
}

template <class F>
auto operator+(Expr x, NodeComposer<F> c)->decltype(compose(f::var<2>() + c.f, {})) {
  auto children = c.children;
  children.push_back(x);
  return compose(f::var<F::arity + 1>() + c.f, children);
}

template <class F>
auto operator+(NodeComposer<F> c, Expr x)->decltype(compose(c.f + f::var<2>(), {})) {
  auto children = c.children;
  children.push_back(x);
  return compose(c.f + f::var<F::arity + 1>(), children);
}

template <class F>
auto operator+(NodeComposer<A> a, NodeComposer<B> b)->decltype(compose(a.f + shift<A::arity>(b.f), {})) {
  auto children = a.children;
  for(auto c : b.children)
    children.push_back(c);
  return compose(compose(a.f + shift<A::arity + 1>(b.f), children);
}

auto tanh(Expr x)->decltype(compose(f::tanh(f::_1), {})) {
  return compose(f::tanh(f::_1), {x});
}

template <class F>
auto tanh(NodeComposer<F> c)->decltype(compose(f::tanh(c.f), {})) {
  return compose(f::tanh(c.f), c.children);
}

template <int K>
using FExpr = NodeComposer<var<K>>;

}
}

int main(int argc, char** argv) {

using namespace marian;

Expr x, y;

FExpr<1> fx = x;
FExpr<2> fy = y;

Expr z = op::tanh(op::operator+(x, op::tanh(y)));

tanh(x + x + x)

//var<1> x;
//var<2> y;
//var<3> z;
//
//auto f = x * logit(x) + one;
//
//auto df_x1 = deriv(f, x);
//auto df_x2 = grad(f, x);
//
//auto df_y = grad(f, y);
//auto df_z = grad(f, z);
//
//float test;
//std::cerr << f.to_string() << " " << sizeof(f) << " " << sizeof(test) << std::endl;
//
//std::cerr << df_x1.to_string() << " " << sizeof(df_x1) << std::endl;
//std::cerr << df_x2.to_string() << " " << sizeof(df_x2) << std::endl;
//std::cerr << df_y.to_string() << std::endl;
//std::cerr << df_z.to_string() << std::endl;

}
