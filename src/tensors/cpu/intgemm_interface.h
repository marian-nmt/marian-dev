#pragma once

#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "integer_common.h"

namespace marian {

namespace cpu {
namespace integer {

template<Type vtype>
struct PrepareANodeOp : public NaryNodeOp {
float clipValue_;
float quantMult_;
  PrepareANodeOp(Expr input, Expr quant_mult, float clipValue)
      : NaryNodeOp({input, quant_mult}, input->shape(), vtype), clipValue_{clipValue} {
    ABORT_IF(children().size() != 2, "expected 2 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      quantMult_ = *child(1)->val()->data();
      typedef typename intgemm_<vtype>::type Integer;
      intgemm_<vtype>::width::PrepareA(child(0)->val()->data(), /*input*/
                                    val_->data<Integer>(), /*output*/
                                    *child(1)->val()->data(), /*Quant Mult*/
                                    rows(child(0)->val()),
                                    cols(child(0)->val()));
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmPrepareA"; }
};

template<Type vtype>
struct PrepareBNodeOp : public NaryNodeOp {
float clipValue_;
float quantMult_;

  PrepareBNodeOp(Expr input, Expr quant_mult, float clipValue)
      : NaryNodeOp({input, quant_mult}, input->shape(), intgemm_<vtype>::intgemmType), clipValue_{clipValue} {
    ABORT_IF(children().size() != 2, "expected 2 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of B cannot be null");
    ABORT_IF(input->shape()[-1] %8 != 0, "Columns of matrix: " + input->type() + " must be multiple of 8.");
  }

  NodeOps forwardOps() override {
   return {NodeOp(
      quantMult_ = *child(1)->val()->data();
      typedef typename intgemm_<vtype>::type Integer;
      if (isIntgemm(child(0)->value_type())) {
        val_ = child(0)->val();
      } else {
        intgemm_<vtype>::width::PrepareB(child(0)->val()->data(), /*input*/
                                      val_->data<Integer>(), /*output*/
                                      *child(1)->val()->data(), /*Quant Mult*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()));
      }
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmPrepareB"; }
};

template<Type vtype>
struct SelectColumnsBNodeOp : public UnaryNodeOp {
public:
  float clipValue_;
  float quantMult_;
  SelectColumnsBNodeOp(Expr input, const std::vector<uint_least32_t>  &indices)
      : UnaryNodeOp(input, newShape(input, indices), intgemm_<vtype>::intgemmType), indices_(indices) {
    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "B cannot be null");

    // Check number of selected columns
    ABORT_IF(indices.size() % 8 != 0, "Shortlist selected vocabulary must be a multiple of 8.");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      //We get the quantization multiplier from a PrepareB
      auto bPreppedNode = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(0));
      clipValue_ = bPreppedNode->clipValue_;
      quantMult_ = bPreppedNode->quantMult_;
      auto input = child(0)->val();
      typedef typename intgemm_<vtype>::type Integer;
      intgemm_<vtype>::width::SelectColumnsB(
                    reinterpret_cast<Integer *>(input->data()),
                    val_->data<Integer>(),
                    rows(input),
                    &*indices_.begin(),
                    &*indices_.end());
    )};
  }

  const std::string type() override { return "intgemmSelectColumnsB"; }

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
    auto cnode = std::dynamic_pointer_cast<SelectColumnsBNodeOp<vtype>>(node);
    if (!cnode) return false;
    return indices_ == cnode->indices_;
  }

private:
  static Shape newShape(Expr a, const std::vector<uint_least32_t>& indices) {
    Shape ret = a->shape();
    ret.set(1, indices.size());
    return ret;
  }

  std::vector<uint_least32_t> indices_;
};

template<Type vtype>
struct QuantMultNodeOp : public UnaryNodeOp {
  QuantMultNodeOp(Expr input) : UnaryNodeOp(input, Shape({1}), Type::float32) {}

  NodeOps forwardOps() override {
    return {NodeOp(
      if (vtype == Type::int16) {
        *val_->data() = 1024.0f;
      } else if (child(0)->type() == "intgemmSelectColumnsB") {
        *val_->data() = std::static_pointer_cast<SelectColumnsBNodeOp<vtype> >(child(0))->quantMult_;
      } else if (isIntgemm(child(0)->value_type())) {                    /* So we can template*/
        typedef typename intgemm_<vtype>::type Integer;
        *val_->data() = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(0)->val()->data()) + child(0)->val()->shape().elements()));
      } else {
        *val_->data() = 127.0f / intgemm::MaxAbsolute(child(0)->val()->data(),
                            child(0)->val()->data() + child(0)->val()->shape().elements());
      }
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmQuantMult"; }
};

template<Type vtype>
class DotNodeOp : public NaryNodeOp {
private:
float scalar_;

public:
  DotNodeOp(Expr a, Expr b, float scalar)
      : NaryNodeOp({a, b}, newShape(a, b), Type::float32), scalar_(scalar) {}

  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
          float aQuantMult = std::static_pointer_cast<PrepareANodeOp<vtype> >(child(0))->quantMult_;
          float bQuantMult;
          if (child(1)->type() == "intgemmSelectColumnsB") {
            bQuantMult = std::static_pointer_cast<SelectColumnsBNodeOp<vtype> >(child(1))->quantMult_;
          } else {
            bQuantMult = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(1))->quantMult_;
          }
          float unquant_mult = 1.0f/(aQuantMult*bQuantMult);

          unquant_mult = unquant_mult*scalar_;
          typedef typename intgemm_<vtype>::type Integer;
          intgemm_<vtype>::width::Multiply(reinterpret_cast<Integer *>(child(0)->val()->data()), /*A*/
                                           reinterpret_cast<Integer *>(child(1)->val()->data()), /*B*/
                                           rows(child(0)->val()),
                                           cols(child(0)->val()),
                                           cols(child(1)->val()),
                                           intgemm::callbacks::UnquantizeAndWrite(unquant_mult, val_->data()));
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmDot"; }
};

template<Type vtype>
class AffineNodeOp : public NaryNodeOp {
private:
  float scalar_;

public:
  AffineNodeOp(Expr a, Expr b, Expr Bias, float scalar)
      : NaryNodeOp({a, b, Bias}, newShape(a, b), Type::float32), scalar_(scalar) {}

  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
          float aQuantMult = std::static_pointer_cast<PrepareANodeOp<vtype> >(child(0))->quantMult_;
          float bQuantMult;
          if (child(1)->type() == "intgemmSelectColumnsB") {
            bQuantMult = std::static_pointer_cast<SelectColumnsBNodeOp<vtype> >(child(1))->quantMult_;
          } else {
            bQuantMult = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(1))->quantMult_;
          }
          float unquant_mult = 1.0f/(aQuantMult*bQuantMult);

          unquant_mult = unquant_mult*scalar_;
          typedef typename intgemm_<vtype>::type Integer;
          intgemm_<vtype>::width::Multiply(reinterpret_cast<Integer *>(child(0)->val()->data()), /*A*/
                                           reinterpret_cast<Integer *>(child(1)->val()->data()), /*B*/
                                           rows(child(0)->val()),
                                           cols(child(0)->val()),
                                           cols(child(1)->val()),                                          /*child(2) is bias*/
                                           intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, child(2)->val()->data(), val_->data()));
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmAffine"; }
};

template<Type vtype>
static inline Expr quantMult(Expr a) {
  return Expression<QuantMultNodeOp<vtype> >(a);
}

template<Type vtype>
static inline Expr prepareA(Expr a, Expr quantMult, float clipValue) {
  return Expression<PrepareANodeOp<vtype> >(a, quantMult, clipValue);
}

template<Type vtype>
static inline Expr prepareB(Expr b, Expr quantMult, float clipValue) {
  return Expression<PrepareBNodeOp<vtype> >(b, quantMult, clipValue);
}

template<Type vtype>
static inline Expr selectColumnsB(Expr b, const std::vector<uint_least32_t> &cols) {
  return Expression<SelectColumnsBNodeOp<vtype > >(b, cols);
}

template<Type vtype>
static inline Expr affine(Expr a, Expr b, Expr bias, bool transA, bool transB, float scale, float clipValue=0 /*currently unused*/) {
  Type bElementType = b->value_type();
  auto aQuantMult = quantMult<vtype>(a);
  auto aQuant = prepareA<vtype>(transA ? transpose(a) : a, aQuantMult, scale);
  Expr bQuant;
  auto bQuantMult = quantMult<vtype>(b);
  if (isIntgemm(bElementType)) {
    //This is the case where we already run SelectColumnB or we loaded a prepacked model.
    //We ignore a transpose argument here, because we do not support it.
    ABORT_IF(transB, "Transpose on prepareB not currently supported");
    bQuant = prepareB<vtype>(b, bQuantMult, clipValue);
  } else {
    bQuant = prepareB<vtype>(transB ? transpose(b) : b, bQuantMult, scale);
  }
  if (bias)
    return Expression<AffineNodeOp<vtype> >(aQuant, bQuant, bias, scale);
  else
    return Expression<DotNodeOp<vtype> >(aQuant, bQuant, scale);
}

template<Type vtype>
static inline Expr dot(Expr a, Expr b, bool transA, bool transB, float scale) {
  return affine<vtype>(a, b, nullptr, transA, transB, scale);
}

}  // namespace integer
}  // namespace cpu
}  // namespace marian
