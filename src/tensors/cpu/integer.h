#pragma once

#include "3rd_party/intgemm/intgemm.h"
#include "common/hash.h"
#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "tensors/cpu/bias.h"

namespace marian {
namespace cpu {
namespace integer {

struct ScaledNodeOp : public UnaryNodeOp {
  float quantMult_;

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp()};
  }

  protected:
    ScaledNodeOp(Expr a, Type t) : UnaryNodeOp(a, t) {}
    ScaledNodeOp(Expr a, Shape s, Type t) : UnaryNodeOp(a, s, t) {}

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
};

// Prepare A for multiplication.
// Expected template argument: intgemm::Int16 or intgemm::Int8.
template <class Backend> struct PrepareANodeOp : public ScaledNodeOp {
  // TODO(emjotde): map from template argument to Type.
  PrepareANodeOp(Expr a, float clipValue) : ScaledNodeOp(a, sizeof(typename Backend::Integer) == 2 ? Type::int16 : Type::int8) {}

  NodeOps forwardOps() override {
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

  const std::string type() override { return "intPrepareA"; }
};

// Seems exessive to have everything duplicated for PrepareB.
// Expected template argument: intgemm::Int16 or intgemm::Int8.
template <class Backend> struct PrepareBNodeOp : public ScaledNodeOp {
  PrepareBNodeOp(Expr a, float clipValue) : ScaledNodeOp(a, sizeof(typename Backend::Integer) == 2 ? Type::int16 : Type::int8) {}

  NodeOps forwardOps() override {
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

  const std::string type() override { return "intPrepareB"; }
};

template <class Backend> class SelectColumnsBNodeOp : public ScaledNodeOp {
  public:
    SelectColumnsBNodeOp(Expr a, const std::vector<Word> &indices)
      : ScaledNodeOp(
          a,
          newShape(a, indices),
          sizeof(typename Backend::Integer) == 2 ? Type::int16 : Type::int8),
      indices_(indices) {}

    NodeOps forwardOps() override {
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

    const std::string type() override { return "intSelectColumnsB"; }

    size_t hash() override {
      if (!hash_) {
        hash_ = NaryNodeOp::hash();
        for(auto i : indices_)
          util::hash_combine(hash_, i);
      }
      return hash_;
    }

    bool equal(Expr node) override {
      if(!NaryNodeOp::equal(node)) return false;
      Ptr<SelectColumnsBNodeOp> cnode = std::dynamic_pointer_cast<SelectColumnsBNodeOp>(node);
      if (!cnode) return false;
      return indices_ == cnode->indices_;
    }


  private:
    static Shape newShape(Expr a, const std::vector<Word>& indices) {
      Shape ret = a->shape();
      ret.set(1, indices.size());
      return ret;
    }

    std::vector<Word> indices_;
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

  NodeOps forwardOps() override {
    return {
      NodeOp(
          using Integer = typename Backend::Integer;
          using intgemm::JustUnquantizeC;

          Backend::Multiply(
            (const Integer*)child(0)->val()->data(),
            (const Integer*)child(1)->val()->data(),
            JustUnquantizeC(val_->data(), scalar_ / (std::static_pointer_cast<ScaledNodeOp>(child(0))->quantMult_ * std::static_pointer_cast<ScaledNodeOp>(child(1))->quantMult_)),
            // Number of rows in A
            child(0)->val()->shape().elements() / child(0)->val()->shape()[-1],
            // Shared dimension.
            child(0)->val()->shape()[-1],
            // Number of columns in B
            child(1)->val()->shape()[-1]))
    };
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp()};
  }

  const std::string type() override { return "dotInt"; }
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

  NodeOps forwardOps() override {
    return {
      NodeOp(
          using Integer = typename Backend::Integer;
          using intgemm::JustUnquantizeC;

          Backend::Multiply(
            (const Integer*)child(0)->val()->data(),
            (const Integer*)child(1)->val()->data(),
            JustUnquantizeC(val_->data(), scalar_ / (std::static_pointer_cast<PrepareANodeOp<Integer> >(child(0))->quantMult_ * std::static_pointer_cast<PrepareBNodeOp<Integer> >(child(1))->quantMult_)),
            // Number of rows in A
            child(0)->val()->shape().elements() / child(0)->val()->shape()[-1],
            // Shared dimension.
            child(0)->val()->shape()[-1],
            // Number of columns in B
            child(1)->val()->shape()[-1]);

            AddBias(val_, child(2)->val()))
    };
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp()};
  }

  const std::string type() override { return "affineInt"; }
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

static inline Expr selectColumnsB(Expr b, const std::vector<Word> &cols) {
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

static inline Expr selectColumnsB(Expr b, const std::vector<Word> &cols) {
  return Expression<integer::SelectColumnsBNodeOp<intgemm::Int8> >(b, cols);
}

} // namespace int8

}
}
