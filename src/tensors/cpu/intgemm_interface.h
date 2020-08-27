#pragma once

#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "integer_common.h"

namespace marian {

namespace cpu {
namespace integer {

#if COMPILE_CPU

/*
 * Prepare an activation matrix into intgemm8/16 format. For now the activation matrix is just quantized.
 * Expr input: The input tensor
 * Expr quant_mult: The quantization multiplier used. Produced by QuantMultNodeOp.
 * float clipValue: Not used for now, just forwarded
 * bool shifted: If the shifted version of intgemm is used (Only available with intgemm8), an offset of 127
 * is added to every single element of the input, effectively making it in the unsigned_int range
 * 
 * the template argument controls whether we're doing 16bit integers or 8bit integers. It can be Type::int8 or Type::int16
 */

template<Type vtype>
struct PrepareANodeOp : public NaryNodeOp {
float clipValue_;
float quantMult_;
bool shifted_;
  PrepareANodeOp(Expr input, Expr quant_mult, float clipValue, bool shifted)
      : NaryNodeOp({input, quant_mult}, input->shape(), vtype), clipValue_(clipValue), shifted_(shifted) {

    if (!shifted) {
      set_name(input->name());
    } else {
      set_name(input->name() + "_shifted");
    }
    setMemoize(false);
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      quantMult_ = *child(1)->val()->data();
      typedef typename intgemm_<vtype>::type Integer;
      if (!shifted_) {
        intgemm_<vtype>::width::PrepareA(child(0)->val()->data(), /*input*/
                                      val_->data<Integer>(), /*output*/
                                      *child(1)->val()->data(), /*Quant Mult*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()));
      } else {
        intgemm::Int8Shift::PrepareA(child(0)->val()->data(), /*input*/
                                      val_->data<int8_t>(), /*output*/
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

  const std::string type() override { return "intgemmPrepareA"; }
};

/*
 * Prepare a parameter matrix into intgemm8/16 format. The activation matrix is quantized and reordered and the reodering
 * depends on the architecture
 * Expr input: The input tensor
 * Expr quant_mult: The quantization multiplier used. Produced by QuantMultNodeOp.
 * float clipValue: Not used for now, just forwarded
 * 
 * the template argument controls whether we're doing 16bit integers or 8bit integers. It can be Type::int8 or Type::int16
 */

template<Type vtype>
struct PrepareBNodeOp : public NaryNodeOp {
float clipValue_;
float quantMult_;

  PrepareBNodeOp(Expr input, Expr quant_mult, float clipValue)
      : NaryNodeOp({input, quant_mult}, input->shape(), intgemm_<vtype>::intgemmType), clipValue_(clipValue) {

    set_name(input->name());
    // Check if arguments are not null
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

/*
 * This is intgemm's equivalent to marian's index_select for the intgemm format. Used for selecting a shortlist from
 * intgemm formatted matrix.
 * Expr input: The input tensor
 * const std::vector<uint_least32_t>  &indices the indecis of the shortlist
 * float clipValue: Not used for now, just forwarded
 * 
 * the template argument controls whether we're doing 16bit integers or 8bit integers. It can be Type::int8 or Type::int16
 */

template<Type vtype>
struct SelectColumnsBNodeOp : public UnaryNodeOp {
public:
  float clipValue_;
  float quantMult_;
  SelectColumnsBNodeOp(Expr input, const std::vector<uint_least32_t>  &indices, float clipValue)
      : UnaryNodeOp(input, newShape(input, indices), intgemm_<vtype>::intgemmType), clipValue_(clipValue), indices_(indices) {

    set_name(input->name());
    setMemoize(false); // SelectColumnsB is memorised in src/layers/generic.cpp:262

    // Check number of selected columns
    ABORT_IF(indices.size() % 8 != 0, "Shortlist selected vocabulary must be a multiple of 8.");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      //We get the quantization multiplier from a PrepareB or directly from the input
      if (child(0)->type() == "intgemmPrepareB") {
        auto bPreppedNode = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(0));
        quantMult_ = bPreppedNode->quantMult_;
      } else {
        typedef typename intgemm_<vtype>::type Integer;
        quantMult_ = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(0)->val()->data()) + child(0)->val()->shape().elements()));
      }
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

  // A custom hasher and comparator are necessary to take the shortlist indecis into account.
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

/*
 * This node calculates the quantization multiplier
 * Expr input: The input tensor
 * bool isA: Whether the matrix is an Activation or parameter
 * 
 * the template argument controls whether we're doing 16bit integers or 8bit integers. It can be Type::int8 or Type::int16
 */
template<Type vtype>
struct QuantMultNodeOp : public UnaryNodeOp {
  bool isA_;
  QuantMultNodeOp(Expr input, bool isA) : UnaryNodeOp(input, Shape({1}), Type::float32), isA_(isA) {
    if (isA_) {
      setMemoize(false);
      set_name(input->name() + "_QuantMultA");
    } else {
      set_name(input->name() + "_QuantMultB");
    }
  }

  /*
   * We have several cases:
   * If we are using 16bit ints, the quantization multiplier is hardcoded to 1024
   * If our input is shortlist (SelectColumnsBNodeOp), it is stored on the node.
   * If our model is prepared intgemm8/intgemm16 format, the qunatization multiplier is stored at the back of the tensor
   * Otherwise, we have the 8bit case, in which case we use 127/MaxAbsolute(input)
   */

#pragma warning( push )
#pragma warning( disable : C4127 ) //VSCODE thinks line 222 is constant conditional expression, which it is only after the template resolution, not before.
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
#pragma warning( pop )
  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override {
    if (isA_)
      return "intgemmQuantMultA";
    else
      return "intgemmQuantMultB";
  }
};


/*
 * This node prepares the bias with the compensation for the intgemm-shifted implementation
 * Expr bias: The input tensor
 * Expr inputB_preppd: The parameter matrix in intgemm8 format.
 * Expr a_quant_mult: The quantization multiplier for the activations matrix.
 * Expr b_quant_mult: The quantization multiplier for the parameter matrix
 */

class PrepareBiasForBNodeOp : public NaryNodeOp {
public:
  PrepareBiasForBNodeOp(Expr bias, Expr inputB_preppd, Expr a_quant_mult, Expr b_quant_mult)
      : NaryNodeOp({bias, inputB_preppd, a_quant_mult, b_quant_mult}, bias->shape(), Type::float32) {

    set_name(bias->name() + "_Prepared");
    setMemoize(false);
  }

  NodeOps forwardOps() override {
    return {NodeOp(
    auto bias = this->child(0)->val();
    auto b = this->child(1)->val();
    auto quant_mult_a = this->child(2)->val();
    auto quant_mult_b = this->child(3)->val();

    float unquant_mult = (-1)*((127.0f / *quant_mult_a->data())*(127.0f / *quant_mult_b->data()))/(127.0f); //Minus one to invert add_ps later on
    intgemm::Int8Shift::PrepareBias((const int8_t *)b->data(), rows(b), cols(b), intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias->data(), val_->data()));
    )};
  }

  const std::string type() override { return "prepareBias"; }
};

/*
 * This computes A*B in intgemm.
 * Expr a: The activation matrix in intgemm format
 * Expr b: The parameter matrix in intgemm fromat
 * 
 * the template argument controls whether we're doing 16bit integers or 8bit integers. It can be Type::int8 or Type::int16
 */

template<Type vtype>
class DotNodeOp : public NaryNodeOp {
private:
float scalar_;

public:
  DotNodeOp(Expr a, Expr b, float scalar)
      : NaryNodeOp({a, b}, newShape(a, b), Type::float32), scalar_(scalar) {
        setMemoize(false); // AFAIK dot is never called with the same matrices
      }

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
          } else if (child(1)->type() == "intgemmPrepareB") {
            bQuantMult = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(1))->quantMult_;
          } else {
            typedef typename intgemm_<vtype>::type Integer;
            bQuantMult = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(1)->val()->data()) + child(1)->val()->shape().elements()));
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

/*
 * This computes A*B + bias in intgemm.
 * Expr a: The activation matrix in intgemm format
 * Expr b: The parameter matrix in intgemm fromat
 * Expr bias: The bias
 * float scalar: AFAIK, not ever used
 * bool shifted: Use the faster shifted intgemm8 multiplier. Requires special versions of the bias and the activation matrix
 * 
 * the template argument controls whether we're doing 16bit integers or 8bit integers. It can be Type::int8 or Type::int16
 */

template<Type vtype>
class AffineNodeOp : public NaryNodeOp {
private:
  float scalar_;
  bool shifted_;

public:
  AffineNodeOp(Expr a, Expr b, Expr Bias, float scalar, bool shifted=false)
      : NaryNodeOp({a, b, Bias}, newShape(a, b), Type::float32), scalar_(scalar), shifted_(shifted) {
        setMemoize(false); // AFAIK affine is never called with the same matrices
      }

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
          } else if (child(1)->type() == "intgemmPrepareB") {
            bQuantMult = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(1))->quantMult_;
          } else {
            typedef typename intgemm_<vtype>::type Integer;
            bQuantMult = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(1)->val()->data()) + child(1)->val()->shape().elements()));
          }
          float unquant_mult = 1.0f/(aQuantMult*bQuantMult);

          unquant_mult = unquant_mult*scalar_;
          typedef typename intgemm_<vtype>::type Integer;
          if (!shifted_) {
            intgemm_<vtype>::width::Multiply(reinterpret_cast<Integer *>(child(0)->val()->data()), /*A*/
                                             reinterpret_cast<Integer *>(child(1)->val()->data()), /*B*/
                                             rows(child(0)->val()),
                                             cols(child(0)->val()),
                                             cols(child(1)->val()),                                          /*child(2) is bias*/
                                             intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, child(2)->val()->data(), val_->data()));
          } else {
            intgemm::Int8Shift::Multiply(reinterpret_cast<int8_t *>(child(0)->val()->data()), /*A*/
                                  reinterpret_cast<int8_t *>(child(1)->val()->data()), /*B*/
                                  rows(child(0)->val()),
                                  cols(child(0)->val()),
                                  cols(child(1)->val()),                                          /*child(2) is bias*/
                                  intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, child(2)->val()->data(), val_->data()));
          }
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmAffine"; }
};


// Wrappers for converting nodes to expression.
template<Type vtype>
static inline Expr quantMult(Expr a, bool isA=false) {
  return Expression<QuantMultNodeOp<vtype> >(a, isA);
}

template<Type vtype>
static inline Expr prepareA(Expr a, Expr quantMult, float clipValue, bool shifted=false) {
  return Expression<PrepareANodeOp<vtype> >(a, quantMult, clipValue, shifted);
}

template<Type vtype>
static inline Expr prepareB(Expr b, Expr quantMult, float clipValue) {
  return Expression<PrepareBNodeOp<vtype> >(b, quantMult, clipValue);
}

template<Type vtype>
static inline Expr selectColumnsB(Expr b, const std::vector<uint_least32_t> &cols, float clipValue) {
  return Expression<SelectColumnsBNodeOp<vtype > >(b, cols, clipValue);
}
#endif

template<Type vtype>
static inline Expr affine(Expr a, Expr b, Expr bias, bool transA, bool transB, float scale, float clipValue=0 /*currently unused*/) {
#if COMPILE_CPU
  clipValue;
  bool shiftedBias = a->graph()->getBackend()->isInt8Shift() && bias; // Use shifted multiplication if we have a enabled it in the options, and we have a bias

  Type bElementType = b->value_type();
  auto aQuantMult = quantMult<vtype>(a, true);
  auto aQuant = prepareA<vtype>(transA ? transpose(a) : a, aQuantMult, scale, shiftedBias);
  Expr bQuantMult = quantMult<vtype>(b);
  Expr bQuant = nullptr;
  if (isIntgemm(bElementType)) {
    // This is the case where we already run SelectColumnB or we loaded a prepacked model.
    // We ignore a transpose argument here, because we do not support it.
    ABORT_IF(transB, "Transpose on prepareB not currently supported");
    bQuant = b;
  } else {
    bQuant = prepareB<vtype>(transB ? transpose(b) : b, bQuantMult, scale);
  }
  if (shiftedBias) {
    // Shifted version of intgemm requires a specially prepared bias.
    bias = Expression<PrepareBiasForBNodeOp>(bias, bQuant, aQuantMult, bQuantMult);
  }

  // IF we don't have a bias, do A*B (AKA dot)
  if (bias)
    return Expression<AffineNodeOp<vtype> >(aQuant, bQuant, bias, scale, shiftedBias);
  else
    return Expression<DotNodeOp<vtype> >(aQuant, bQuant, scale);
#else
  a, b, bias, transA, transB, scale, clipValue;
  ABORT("You need to enable CPU compilation to use this feature. Use cmake .. -DCOMPILE_CPU=ON");
#endif
}

template<Type vtype>
static inline Expr dot(Expr a, Expr b, bool transA, bool transB, float scale) {
  return affine<vtype>(a, b, nullptr, transA, transB, scale, 0 /*currently unused clipValue*/);
}

}  // namespace integer
}  // namespace cpu
}  // namespace marian
