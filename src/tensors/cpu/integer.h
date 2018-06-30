#pragma once

#include "tensors/cpu/bias.h"
#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "tensors/cpu/intgemm/intgemm.h"

namespace marian {
namespace cpu {
namespace integer {

struct ScaledNodeOp : public UnaryNodeOp {
  float quantMult_;

  explicit ScaledNodeOp(Expr a, Type t) : UnaryNodeOp(a, t) {}
  explicit ScaledNodeOp(Expr a, Shape s, Type t) : UnaryNodeOp(a, s, t) {}

  template <class Integer> void CalculateQuantMult() {
    auto c = child(0)->val();
    if (c->type() != Type::float32) {
      ABORT("Trying to quantize non-float");
    }
    if (sizeof(Integer) == 2) {
      quantMult_ = 1024.0f;
    } else {
      quantMult_ = 127.0f / intgemm::MaxAbsolute(c->data(), c->data() + c->shape().elements());
    }
  }

  // Get number of rows which should really be a method in shape
  int rows() {
    auto c = child(0)->val();
    return c->shape().elements() / c->shape()[-1];
  }

  NodeOps backwardOps() {
    ABORT("Only used for inference");
    return {NodeOp()};
  }
};

// Prepare A for multiplication.
// Expected template argument: intgemm::Int16 or intgemm::Int8.
template <class Backend> struct PrepareANodeOp : public ScaledNodeOp {
  // TODO(emjotde): map from template argument to Type.
  PrepareANodeOp(Expr a, float clipValue) : ScaledNodeOp(a, sizeof(typename Backend::Integer) == 2 ? Type::int16 : Type::int8) {}

  NodeOps forwardOps() {
    return { [=] {
      CalculateQuantMult<typename Backend::Integer>();
      auto c = child(0)->val();
      Backend::PrepareA(
          c->data(),
          val_->data<typename Backend::Integer>(),
          quantMult_,
          rows(),
          c->shape()[-1]);
    }};
  }

  const std::string type() { return "intPrepareA"; }
};

// Seems exessive to have everything duplicated for PrepareB.
// Expected template argument: intgemm::Int16 or intgemm::Int8.
template <class Backend> struct PrepareBNodeOp : public ScaledNodeOp {
  PrepareBNodeOp(Expr a, float clipValue) : ScaledNodeOp(a, sizeof(typename Backend::Integer) == 2 ? Type::int16 : Type::int8) {}

  NodeOps forwardOps() {
    return { [=] {
      CalculateQuantMult<typename Backend::Integer>();
      auto c = child(0)->val();
      Backend::PrepareB(
          c->data(),
          val_->data<typename Backend::Integer>(),
          quantMult_,
          rows(),
          c->shape()[-1]);
    }};
  }

  const std::string type() { return "intPrepareB"; }
};

template <class Backend> class SelectColumnsBNodeOp : public ScaledNodeOp {
  public:
    SelectColumnsBNodeOp(Expr a, const std::vector<size_t> &indices)
      : ScaledNodeOp(
          a,
          newShape(a, indices),
          sizeof(typename Backend::Integer) == 2 ? Type::int16 : Type::int8),
      indices_(indices) {}

    NodeOps forwardOps() {
      return { [=] {
        quantMult_ = std::static_pointer_cast<ScaledNodeOp>(child(0))->quantMult_;
        auto c = child(0)->val();
        Backend::SelectColumnsB(
            (const typename Backend::Integer*)c->data(),
            val_->data<typename Backend::Integer>(),
            rows(),
            &*indices_.begin(),
            &*indices_.end());
      }};
    }

    const std::string type() { return "intSelectColumnsB"; }

  private:
    static Shape newShape(Expr a, const std::vector<size_t>& indices) {
      Shape ret = a->shape();
      ret.set(1, indices.size());
      return ret;
    }

    std::vector<std::size_t> indices_;
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
            scalar_ / (std::static_pointer_cast<ScaledNodeOp>(child(0))->quantMult_ * std::static_pointer_cast<ScaledNodeOp>(child(1))->quantMult_),
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

static inline Expr selectColumnsB(Expr b, const std::vector<size_t> &cols) {
  return Expression<integer::SelectColumnsBNodeOp<intgemm::Int16> >(b, cols);
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

static inline Expr selectColumnsB(Expr b, const std::vector<size_t> &cols) {
  return Expression<integer::SelectColumnsBNodeOp<intgemm::Int8> >(b, cols);
}

} // namespace int8

}
}
