#pragma once

#include "layers/constructors.h"

namespace marian {

class Motorway {
  public:
    Motorway(const std::string name, size_t depth)
      : name_(name),
        depth_(depth)
    {}

    Expr operator()(Expr x) {
      Expr input = x;
      for (size_t i = 0; i < depth_; ++i) {
        size_t out_dim = x->shape()[1];
        auto gmlp = mlp::mlp(x->graph()).push_back(
            mlp::dense(x->graph())
            ("prefix", name_ + "_d1_" + std::to_string(i))
            ("dim", out_dim)
            ("activation", mlp::act::logit));
        auto g = gmlp->apply(x);

        auto dense2_mlp = mlp::mlp(x->graph()).push_back(
            mlp::dense(x->graph())
            ("prefix", name_ + "_d2_" + std::to_string(i))
            ("dim", out_dim)
            ("activation", mlp::act::linear));
        auto rr = relu(dense2_mlp->apply(x));
        input = (g * rr) + ((1 - g) * input);
      }
      return input;
  }

  protected:
    std::string name_;
    size_t depth_;
};

using Highway = Motorway;

}  // namespace marian
