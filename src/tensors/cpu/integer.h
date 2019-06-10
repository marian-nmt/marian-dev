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
class QuantMultNodeOp : public OnlyForInferenceNodeOp {
public:
  QuantMultNodeOp(Expr input) : OnlyForInferenceNodeOp({input}) {}

  NodeOps forwardOps() override {
    return {NodeOp(
      auto input = child(0)->val();

      ABORT_IF(input->type() != Type::float32, "Trying to quantize non-float");

      if (TypeFromBackend<Backend>() == Type::int16) {
        *val_->data() = 1024.0f;
      } else {
        *val_->data() = 127.0f / intgemm::MaxAbsolute(input->data(), input->data() + input->shape().elements());
      }
    )};
  }

  const std::string type() override { return "intQuantMult"; }
};

namespace { // anonymous namespace

template <typename Integer, typename PrepareMatrixFun>
inline NodeOps prepareMatrixForwardOps(Node* node, PrepareMatrixFun prepare_matrix_fun) {
  return {NodeOp(
    auto input = node->child(0)->val();
    auto quant_mult = node->child(1)->val();
    prepare_matrix_fun(
        input->data(),
        node->val()->data<Integer>(),
        *quant_mult->data(),
        rows(input),
        input->shape()[-1]);
  )};
}

} // anonymous namespace

template <typename Backend, typename = EnableIfBackendIsSupported<Backend>>
class PrepareANodeOp : public OnlyForInferenceNodeOp {
public:
  PrepareANodeOp(Expr input, Expr quant_mult, float clipValue)
      : OnlyForInferenceNodeOp({input, quant_mult}, input->shape(), TypeFromBackend<Backend>()) {}

  NodeOps forwardOps() override {
    return prepareMatrixForwardOps<typename Backend::Integer>(this, Backend::PrepareA);
  }

  const std::string type() override { return "intPrepareA"; }
};

template <typename Backend, typename = EnableIfBackendIsSupported<Backend>>
class PrepareBNodeOp : public OnlyForInferenceNodeOp {
public:
  PrepareBNodeOp(Expr input, Expr quant_mult, float clipValue)
      : OnlyForInferenceNodeOp({input, quant_mult}, input->shape(), TypeFromBackend<Backend>()) {}

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

/*
*                   +-----------+
*                   |    Dot    |
*                   +-----------+
*                         |
*         +----------+----------+----------+
*         |          |          |          |
*  +-------------+   |   +-------------+   |
*  | Quantized A |   |   | Quantized B |   |
*  +-------------+   |   +-------------+   |
*             +-------------+       +-------------+
*             | QuantMult A |       | QuantMult B |
*             +-------------+       +-------------+
*/
template <typename Backend, typename = EnableIfBackendIsSupported<Backend>>
class DotNodeOp : public OnlyForInferenceNodeOp {
private:
  float scalar_;

public:
  DotNodeOp(Expr a, Expr a_quant_mult, Expr b, Expr b_quant_mult, float scalar)
      : OnlyForInferenceNodeOp({a, a_quant_mult, b, b_quant_mult}, newShape(a, b)), scalar_(scalar) {
    ABORT_IF(children().size() != 4, "expected 4 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "a cannot be null");
    ABORT_IF(child(1) == nullptr, "quantize mult of A cannot be null");
    ABORT_IF(child(2) == nullptr, "quantize mult of B cannot be null");
    ABORT_IF(child(3) == nullptr, "a cannot be null");

    // Check alignment
    assert(child(2)->shape()[-1] % 8 == 0);

    // Check dimmensions
    ABORT_IF(child(0)->shape()[-1] != child(2)->shape()[-2], "matrices cannot be multiplied because there's a dimension mismatch");
  }

  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      using Integer = typename Backend::Integer;
      using intgemm::JustUnquantizeC;

      auto a = child(0)->val();
      auto quant_mult_a = child(1)->val();
      auto b = child(2)->val();
      auto quant_mult_b = child(3)->val();
      Backend::Multiply(
          (const Integer*)a->data(),
          (const Integer*)b->data(),
          JustUnquantizeC(val_->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));
    )};
  }

  const std::string type() override { return "intDot"; }
};

/*
*                         +-----------+
*                         |  Affine   |
*                         +-----------+
*                               |
*         +----------+----------+----------+----------+
*         |          |          |          |          |
*  +-------------+   |   +-------------+   |      +-------+
*  | Quantized A |   |   | Quantized B |   |      | Bias  |
*  +-------------+   |   +-------------+   |      +-------+
*             +-------------+       +-------------+
*             | QuantMult A |       | QuantMult B |
*             +-------------+       +-------------+
*/
template <typename Backend, typename = EnableIfBackendIsSupported<Backend>>
class AffineNodeOp : public OnlyForInferenceNodeOp {
private:
  float scalar_;

public:
  AffineNodeOp(Expr a, Expr a_quant_mult, Expr b, Expr b_quant_mult, Expr bias, float scalar)
      : OnlyForInferenceNodeOp({a, a_quant_mult, b, b_quant_mult, bias}, newShape(a, b, bias)), scalar_(scalar) {
    ABORT_IF(children().size() != 5, "expected 5 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "a cannot be null");
    ABORT_IF(child(1) == nullptr, "quantize mult of A cannot be null");
    ABORT_IF(child(2) == nullptr, "quantize mult of B cannot be null");
    ABORT_IF(child(3) == nullptr, "a cannot be null");
    ABORT_IF(child(4) == nullptr, "bias cannot be null");

    // Check alignment
    assert(child(2)->shape()[-1] % 8 == 0);

    // Check dimmensions
    ABORT_IF(child(0)->shape()[-1] != child(2)->shape()[-2], "matrices cannot be multiplied because there's a dimension mismatch");
    ABORT_IF(child(2)->shape()[-1] != child(4)->shape()[-1], "bias cannot be added because there's a dimension mismatch");
  }

  Shape newShape(Expr a, Expr b, Expr bias) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      using Integer = typename Backend::Integer;
      using intgemm::BiasAddUnquantizeC;

      auto a = child(0)->val();
      auto quant_mult_a = child(1)->val();
      auto b = child(2)->val();
      auto quant_mult_b = child(3)->val();
      auto bias = child(4)->val();
      Backend::Multiply(
          (const Integer*)a->data(),
          (const Integer*)b->data(),
          BiasAddUnquantizeC(val_->data(), bias->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));
    )};
  }

  const std::string type() override { return "intAffine"; }
};

} // namespace integer

#define API_IMPLEMENTATION(backend)                                                                          \
  static inline Expr dot(Expr a, Expr quant_mult_a, Expr b, Expr quant_mult_b, float scalar) {               \
    return Expression<integer::DotNodeOp<backend>>(a, quant_mult_a, b, quant_mult_b, scalar);                \
  }                                                                                                          \
  static inline Expr affine(Expr a, Expr quant_mult_a, Expr b, Expr quant_mult_b, Expr bias, float scalar) { \
    return Expression<integer::AffineNodeOp<backend>>(a, quant_mult_a, b, quant_mult_b, bias, scalar);       \
  }                                                                                                          \
  static inline Expr quantMult(Expr a) {                                                                     \
    return Expression<integer::QuantMultNodeOp<backend>>(a);                                                 \
  }                                                                                                          \
  static inline Expr prepareA(Expr a, Expr quant_mult, float clipValue) {                                    \
    return Expression<integer::PrepareANodeOp<backend>>(a, quant_mult, clipValue);                           \
  }                                                                                                          \
  static inline Expr prepareB(Expr b, Expr quant_mult, float clipValue) {                                    \
    return Expression<integer::PrepareBNodeOp<backend>>(b, quant_mult, clipValue);                           \
  }                                                                                                          \
  static inline Expr selectColumnsB(Expr b, const std::vector<Word> &cols) {                                 \
    return Expression<integer::SelectColumnsBNodeOp<backend>>(b, cols);                                      \
  }


namespace int8 {
API_IMPLEMENTATION(intgemm::Int8)
}

namespace int16 {
API_IMPLEMENTATION(intgemm::Int16)
}

}
}
