#pragma once

#include "3rd_party/intgemm/intgemm.h"
#include "common/hash.h"
#include "graph/node.h"
#include "tensors/cpu/bias.h"

namespace marian {
namespace cpu {
namespace integer {

namespace {

inline int cols(Tensor& tensor) { return tensor->shape()[-1]; }
inline int rows(Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }

template <typename Backend>
constexpr Type TypeFromBackend() { return Type(TypeClass::signed_type + sizeof(typename Backend::Integer)); };

}

struct OnlyForInferenceNodeOp : public NaryNodeOp {
  OnlyForInferenceNodeOp(const std::vector<Expr>& nodes,
                         Shape shape,
                         Type value_type = Type::float32)
      : NaryNodeOp(nodes, shape, value_type) {}

  OnlyForInferenceNodeOp(const std::vector<Expr>& nodes) : NaryNodeOp(nodes) {}

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp()};
  }
};

template <class Backend>
struct QuantizeMultNodeOp : public OnlyForInferenceNodeOp {
  QuantizeMultNodeOp(Expr input) : OnlyForInferenceNodeOp({input}) {}

  NodeOps forwardOps() override {
    return {NodeOp(
      auto input = child(0)->val();
      if (input->type() != Type::float32) {
        ABORT("Trying to quantize non-float");
      }
      if (TypeFromBackend<Backend>() == Type::int16) {
        *val_->data() = 1024.0f;
      } else {
        *val_->data() = 127.0f / intgemm::MaxAbsolute(input->data(), input->data() + input->shape().elements());
      }
    )};
  }

  const std::string type() override { return "intMaxAbs"; }
};

// Prepare A for multiplication.
// Expected template argument: intgemm::Int16 or intgemm::Int8.
template <class Backend> struct PrepareANodeOp : public OnlyForInferenceNodeOp {
  PrepareANodeOp(Expr input, Expr quantize_mult, float clipValue)
      : OnlyForInferenceNodeOp({input, quantize_mult}, input->shape(), TypeFromBackend<Backend>()) {}

  NodeOps forwardOps() override {
    return {NodeOp(
      auto input = child(0)->val();
      auto quantize_mult = child(1)->val();
      Backend::PrepareA(
          input->data(),
          val_->data<typename Backend::Integer>(),
          *quantize_mult->data(),
          rows(input),
          input->shape()[-1]);
    )};
  }

  const std::string type() override { return "intPrepareA"; }
};

// Seems exessive to have everything duplicated for PrepareB.
// Expected template argument: intgemm::Int16 or intgemm::Int8.
template <class Backend> struct PrepareBNodeOp : public OnlyForInferenceNodeOp {
  PrepareBNodeOp(Expr input, Expr quantize_mult, float clipValue)
      : OnlyForInferenceNodeOp({input, quantize_mult}, input->shape(), TypeFromBackend<Backend>()) {}

  NodeOps forwardOps() override {
    return {NodeOp(
      auto input = child(0)->val();
      auto quantize_mult = child(1)->val();
      Backend::PrepareB(
          input->data(),
          val_->data<typename Backend::Integer>(),
          *quantize_mult->data(),
          rows(input),
          input->shape()[-1]);
    )};
  }

  const std::string type() override { return "intPrepareB"; }
};

template <class Backend> class SelectColumnsBNodeOp : public OnlyForInferenceNodeOp {
  public:
    SelectColumnsBNodeOp(Expr input, const std::vector<Word> &indices)
        : OnlyForInferenceNodeOp({input}, newShape(input, indices), TypeFromBackend<Backend>()), indices_(indices) {}

    NodeOps forwardOps() override {
      return {NodeOp(
        auto input = child(0)->val();
        Backend::SelectColumnsB(
            (const typename Backend::Integer*)input->data(),
            val_->data<typename Backend::Integer>(),
            rows(input),
            &*indices_.begin(),
            &*indices_.end());
      )};
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

template <class Backend> class DotNodeOp : public OnlyForInferenceNodeOp {
private:
  float scalar_;

public:
  DotNodeOp(Expr a, Expr b, float scalar)
      : OnlyForInferenceNodeOp({a, b}, newShape(a, b)), scalar_(scalar) {}

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
    return {NodeOp(
      using Integer = typename Backend::Integer;
      using intgemm::JustUnquantizeC;

      auto a = child(0)->val();
      auto b = child(1)->val();
      auto a_scale = *child(0)->child(1)->val()->data();
      auto b_scale = *child(1)->child(1)->val()->data();
      Backend::Multiply(
          (const Integer*)a->data(),
          (const Integer*)b->data(),
          JustUnquantizeC(val_->data(), scalar_ / (a_scale * b_scale)),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));
    )};
  }

  const std::string type() override { return "dotInt"; }
};


template <class Backend> class AffineNodeOp : public OnlyForInferenceNodeOp {
private:
  float scalar_;

public:
  AffineNodeOp(const std::vector<Expr>& nodes,
               float scalar)
      : OnlyForInferenceNodeOp(nodes, newShape(nodes[0], nodes[1])),
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
    return {NodeOp(
      using Integer = typename Backend::Integer;
      using intgemm::JustUnquantizeC;

      auto a = child(0)->val();
      auto b = child(1)->val();
      auto a_scale = *child(0)->child(1)->val()->data();
      auto b_scale = *child(1)->child(1)->val()->data();
      Backend::Multiply(
          (const Integer*)a->data(),
          (const Integer*)b->data(),
          JustUnquantizeC(val_->data(), scalar_ / (a_scale * b_scale)),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));

      AddBias(val_, child(2)->val());
    )};
  }

  const std::string type() override { return "affineInt"; }
};
} // namespace integer

namespace int16 {

static inline Expr dot(Expr a, Expr b, float scalar) {
  return Expression<integer::DotNodeOp<intgemm::Int16>>(a, b, scalar);
}

static inline Expr affine(Expr a, Expr b, Expr c, float scalar) {
  std::vector<Expr> nodes = {a, b, c};
  return Expression<integer::AffineNodeOp<intgemm::Int16>>(nodes, scalar);
}

static inline Expr quantizeMult(Expr a) {
  return Expression<integer::QuantizeMultNodeOp<intgemm::Int16>>(a);
}

static inline Expr prepareA(Expr a, Expr quantize_mult, float clipValue) {
  return Expression<integer::PrepareANodeOp<intgemm::Int16>>(a, quantize_mult, clipValue);
}

static inline Expr prepareB(Expr b, Expr quantize_mult, float clipValue) {
  return Expression<integer::PrepareBNodeOp<intgemm::Int16>>(b, quantize_mult, clipValue);
}

static inline Expr selectColumnsB(Expr b, const std::vector<Word> &cols) {
  return Expression<integer::SelectColumnsBNodeOp<intgemm::Int16>>(b, cols);
}

} // namespace int16

namespace int8 {

static inline Expr dot(Expr a, Expr b, float scalar) {
  return Expression<integer::DotNodeOp<intgemm::Int8>>(a, b, scalar);
}

static inline Expr affine(Expr a, Expr b, Expr c, float scalar) {
  std::vector<Expr> nodes = {a, b, c};
  return Expression<integer::AffineNodeOp<intgemm::Int8>>(nodes, scalar);
}

static inline Expr quantizeMult(Expr a) {
  return Expression<integer::QuantizeMultNodeOp<intgemm::Int8>>(a);
}

static inline Expr prepareA(Expr a, Expr quantize_mult, float clipValue) {
  return Expression<integer::PrepareANodeOp<intgemm::Int8>>(a, quantize_mult, clipValue);
}

static inline Expr prepareB(Expr b, Expr quantize_mult, float clipValue) {
  return Expression<integer::PrepareBNodeOp<intgemm::Int8>>(b, quantize_mult, clipValue);
}

static inline Expr selectColumnsB(Expr b, const std::vector<Word> &cols) {
  return Expression<integer::SelectColumnsBNodeOp<intgemm::Int8>>(b, cols);
}

} // namespace int8

}
}
