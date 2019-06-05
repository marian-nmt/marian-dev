#pragma once

#include "3rd_party/intgemm/intgemm.h"
#include "common/hash.h"
#include "graph/node.h"
#include "tensors/cpu/bias.h"

namespace marian {
namespace cpu {
namespace integer {

namespace { // anonymous namespace

inline int cols(Tensor& tensor) { return tensor->shape()[-1]; }
inline int rows(Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }

template <typename Backend>
using EnableIfBackendIsSupported = typename std::enable_if<
  std::is_same<Backend, intgemm::Int8>::value ||
  std::is_same<Backend, intgemm::Int16>::value,
  void>::type;

template <typename Backend, typename = EnableIfBackendIsSupported<Backend>>
constexpr Type TypeFromBackend() { return Type(TypeClass::signed_type + sizeof(typename Backend::Integer)); };

} // anonymous namespace

class OnlyForInferenceNodeOp : public NaryNodeOp {
public:
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

template <typename Backend, typename = EnableIfBackendIsSupported<Backend>>
class QuantizeMultNodeOp : public OnlyForInferenceNodeOp {
public:
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

namespace { // anonymous namespace

template <typename Integer, typename PrepareMatrixFun>
inline NodeOps prepareMatrixForwardOps(Node* node, PrepareMatrixFun prepare_matrix_fun) {
  return {NodeOp(
    auto input = node->child(0)->val();
    auto quantize_mult = node->child(1)->val();
    prepare_matrix_fun(
        input->data(),
        node->val()->data<Integer>(),
        *quantize_mult->data(),
        rows(input),
        input->shape()[-1]);
  )};
}

} // anonymous namespace

template <typename Backend, typename = EnableIfBackendIsSupported<Backend>>
class PrepareANodeOp : public OnlyForInferenceNodeOp {
public:
  PrepareANodeOp(Expr input, Expr quantize_mult, float clipValue)
      : OnlyForInferenceNodeOp({input, quantize_mult}, input->shape(), TypeFromBackend<Backend>()) {}

  NodeOps forwardOps() override {
    return prepareMatrixForwardOps<typename Backend::Integer>(this, Backend::PrepareA);
  }

  const std::string type() override { return "intPrepareA"; }
};

template <typename Backend, typename = EnableIfBackendIsSupported<Backend>>
class PrepareBNodeOp : public OnlyForInferenceNodeOp {
public:
  PrepareBNodeOp(Expr input, Expr quantize_mult, float clipValue)
      : OnlyForInferenceNodeOp({input, quantize_mult}, input->shape(), TypeFromBackend<Backend>()) {}

  NodeOps forwardOps() override {
    return prepareMatrixForwardOps<typename Backend::Integer>(this, Backend::PrepareB);
  }

  const std::string type() override { return "intPrepareB"; }
};

template <typename Backend, typename = EnableIfBackendIsSupported<Backend>>
class SelectColumnsBNodeOp : public OnlyForInferenceNodeOp {
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

template <typename Backend, typename = EnableIfBackendIsSupported<Backend>>
class DotNodeOp : public OnlyForInferenceNodeOp {
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

  const std::string type() override { return "intDot"; }
};

template <typename Backend, typename = EnableIfBackendIsSupported<Backend>>
class AffineNodeOp : public OnlyForInferenceNodeOp {
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
      using intgemm::BiasAddUnquantizeC;

      auto a = child(0)->val();
      auto b = child(1)->val();
      auto a_scale = *child(0)->child(1)->val()->data();
      auto b_scale = *child(1)->child(1)->val()->data();
      Backend::Multiply(
          (const Integer*)a->data(),
          (const Integer*)b->data(),
          BiasAddUnquantizeC(val_->data(), child(2)->val()->data(), scalar_ / (a_scale * b_scale)),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));

      //AddBias(val_, child(2)->val());
    )};
  }

  const std::string type() override { return "intAffine"; }
};

} // namespace integer

#define API_IMPLEMENTATION(backend)                                                  \
  static inline Expr dot(Expr a, Expr b, float scalar) {                             \
    return Expression<integer::DotNodeOp<backend>>(a, b, scalar);                    \
  }                                                                                  \
  static inline Expr affine(Expr a, Expr b, Expr c, float scalar) {                  \
    std::vector<Expr> nodes = {a, b, c};                                             \
    return Expression<integer::AffineNodeOp<backend>>(nodes, scalar);                \
  }                                                                                  \
  static inline Expr quantizeMult(Expr a) {                                          \
    return Expression<integer::QuantizeMultNodeOp<backend>>(a);                      \
  }                                                                                  \
  static inline Expr prepareA(Expr a, Expr quantizeMult, float clipValue) {          \
    return Expression<integer::PrepareANodeOp<backend>>(a, quantizeMult, clipValue); \
  }                                                                                  \
  static inline Expr prepareB(Expr b, Expr quantizeMult, float clipValue) {          \
    return Expression<integer::PrepareBNodeOp<backend>>(b, quantizeMult, clipValue); \
  }                                                                                  \
  static inline Expr selectColumnsB(Expr b, const std::vector<Word> &cols) {         \
    return Expression<integer::SelectColumnsBNodeOp<backend>>(b, cols);              \
  }


namespace int8 {
API_IMPLEMENTATION(intgemm::Int8)
}

namespace int16 {
API_IMPLEMENTATION(intgemm::Int16)
}

}
}
