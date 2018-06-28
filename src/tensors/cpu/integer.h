#pragma once

#include "tensors/cpu/bias.h"
#include "graph/node.h"
#include "tensors/cpu/intgemm/intgemm.h"

namespace marian {
namespace cpu {
namespace integer {

// Prepare A for multiplication.
// Expected template argument: intgemm::Int16 or intgemm::Int8.
template <class Backend> struct PrepareANodeOp : public UnaryNodeOp {
  float quantMult_;

  PrepareANodeOp(Expr a, float clipValue)
    // TODO(emjotde): map from template argument to Type.
  : UnaryNodeOp(a, sizeof(typename Backend::Integer) == 2 ? Type::int16 : Type::int8),
    // TODO(emjotde): just directly expose the quantization multiplier.
    quantMult_{sizeof(typename Backend::Integer) == 2 ? 1024.0f : (127.0f / clipValue)} {}

  NodeOps forwardOps() {
    return { [=] {
      auto c = child(0)->val();
      std::pair<const float *, const float *> minmax = std::minmax_element(c->data(), c->data() + c->shape().elements());
      quantMult_ = 63.0f / std::max<float>(fabs(*minmax.first), fabsf(*minmax.second));
      Backend::PrepareA(
            c->data(),
            val_->data<typename Backend::Integer>(),
            quantMult_,
            // Number of rows
            c->shape().elements() / child(0)->val()->shape()[-1],
            c->shape()[-1]);
    }};
  }

  NodeOps backwardOps() {
    ABORT("Only used for inference");
    return {NodeOp()};
  }

  const std::string type() { return "intPrepareA"; }
};

// Seems exessive to have everything duplicated for PrepareB.
// Expected template argument: intgemm::Int16 or intgemm::Int8.
template <class Backend> struct PrepareBNodeOp : public UnaryNodeOp {
  float quantMult_;

  PrepareBNodeOp(Expr a, float clipValue)
    // TODO(emjotde): map from template argument to Type.
  : UnaryNodeOp(a, sizeof(typename Backend::Integer) == 2 ? Type::int16 : Type::int8),
    // TODO(emjotde): just directly expose the quantization multiplier.
    quantMult_{sizeof(typename Backend::Integer) == 2 ? 1024.0f : (127.0f / clipValue)} {}

  NodeOps forwardOps() {
    return { [=] {
      auto c = child(0)->val();
      std::pair<const float *, const float *> minmax = std::minmax_element(c->data(), c->data() + c->shape().elements());
      quantMult_ = 63.0f / std::max<float>(fabs(*minmax.first), fabsf(*minmax.second));
        Backend::PrepareB(
          c->data(),
          val_->data<typename Backend::Integer>(),
          quantMult_,
          // Number of rows
          c->shape().elements() / c->shape()[-1],
          c->shape()[-1]);
      }};
  }

  NodeOps backwardOps() {
    ABORT("Only used for inference");
    return {NodeOp()};
  }

  const std::string type() { return "intPrepareB"; }
};

template <class Backend> class DotNodeOp : public NaryNodeOp {
private:
  float scalar_;

public:
  DotNodeOp(Expr a, Expr b, float scalar)
      : NaryNodeOp({a, b}, newShape(a, b)),
        scalar_(scalar) {}

  Shape newShape(Expr a, Expr b) {
    auto shapeA = a->shape();
    auto shapeB = b->shape();
    assert(shapeB[-1] % 8 == 0);

    Shape outShape = shapeA;
    outShape.set(-1, shapeB[-1]);
    ABORT_IF(shapeA[-1] != shapeB[-2],
             "matrix product requires dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() {
    return {
      NodeOp(
      typedef typename Backend::Integer Integer;
      Backend::Multiply(
            (const Integer*)child(0)->val()->data(),
            (const Integer*)child(1)->val()->data(),
            val_->data(),
            // TODO(emjotde): please can we just directly expose the quantization multiplier?
            scalar_ / (std::static_pointer_cast<PrepareANodeOp<Integer> >(child(0))->quantMult_ * std::static_pointer_cast<PrepareBNodeOp<Integer> >(child(1))->quantMult_),
            // Number of rows in A
            child(0)->val()->shape().elements() / child(0)->val()->shape()[-1],
            // Shared dimension.
            child(0)->val()->shape()[-1],
            // Number of columns in B
            child(1)->val()->shape()[-1]))
    };
  }

  NodeOps backwardOps() {
    ABORT("Only used for inference");
    return {NodeOp()};
  }

  const std::string type() { return "dotInt"; }
};


template <class Backend> class AffineNodeOp : public NaryNodeOp {
private:
  float scalar_;

public:
  AffineNodeOp(const std::vector<Expr>& nodes,
               float scalar)
      : NaryNodeOp(nodes, newShape(nodes[0], nodes[1])),
        scalar_(scalar) {}

  Shape newShape(Expr a, Expr b) {
    auto shapeA = a->shape();
    auto shapeB = b->shape();
    assert(shapeB[-1] % 8 == 0);

    Shape outShape = shapeA;
    outShape.set(-1, shapeB[-1]);
    ABORT_IF(shapeA[-1] != shapeB[-2],
             "matrix product requires dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() {
    return {
      NodeOp(
          typedef typename Backend::Integer Integer;
          Backend::Multiply(
            (const Integer*)child(0)->val()->data(),
            (const Integer*)child(1)->val()->data(),
            val_->data(),
            // TODO(emjotde): please can we just directly expose the quantization multiplier?
            scalar_ / (std::static_pointer_cast<PrepareANodeOp<Integer> >(child(0))->quantMult_ * std::static_pointer_cast<PrepareBNodeOp<Integer> >(child(1))->quantMult_),
            // Number of rows in A
            child(0)->val()->shape().elements() / child(0)->val()->shape()[-1],
            // Shared dimension.
            child(0)->val()->shape()[-1],
            // Number of columns in B
            child(1)->val()->shape()[-1]);

            AddBias(val_, child(2)->val()))
    };
  }

  NodeOps backwardOps() {
    ABORT("Only used for inference");
    return {NodeOp()};
  }

  const std::string type() { return "affineInt"; }
};
} // namespace integer

namespace int16 {

static inline Expr dot(Expr a, Expr b, float scalar) {
  return Expression<integer::DotNodeOp<intgemm::Int16> >(a, b, scalar);
}

static inline Expr affine(Expr a, Expr b, Expr c, float scalar) {
  std::vector<Expr> nodes = {a, b, c};
  return Expression<integer::AffineNodeOp<intgemm::Int16> >(nodes, scalar);
}

static inline Expr prepareA(Expr a, float clipValue) {
  return Expression<integer::PrepareANodeOp<intgemm::Int16> >(a, clipValue);
}

static inline Expr prepareB(Expr b, float clipValue) {
  return Expression<integer::PrepareBNodeOp<intgemm::Int16> >(b, clipValue);
}

} // namespace int16

namespace int8 {

static inline Expr dot(Expr a, Expr b, float scalar) {
  return Expression<integer::DotNodeOp<intgemm::Int8> >(a, b, scalar);
}

static inline Expr affine(Expr a, Expr b, Expr c, float scalar) {
  std::vector<Expr> nodes = {a, b, c};
  return Expression<integer::AffineNodeOp<intgemm::Int8> >(nodes, scalar);
}

static inline Expr prepareA(Expr a, float clipValue) {
  return Expression<integer::PrepareANodeOp<intgemm::Int8> >(a, clipValue);
}

static inline Expr prepareB(Expr b, float clipValue) {
  return Expression<integer::PrepareBNodeOp<intgemm::Int8> >(b, clipValue);
}

} // namespace int8

}
}
