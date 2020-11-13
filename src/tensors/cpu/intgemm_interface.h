#pragma once

#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "integer_common.h"

namespace marian {

namespace cpu {
namespace integer {

#if COMPILE_CPU

template <Type vtype>
static inline float& getQuantMult(marian::Tensor val) {
  ABORT_IF(!isIntgemm(val->type()), "getQuantMult does not work for type {}", val->type());
  typedef typename intgemm_<vtype>::type Integer;
  return *(reinterpret_cast<float*>(val->data<Integer>() + val->shape().elements()));
}

template <Type vtype>
static inline float computeQuantMult(marian::Tensor val) {
  if(sizeOf(vtype) == 1)
    return 127.0f / intgemm::MaxAbsolute(val->data(), val->data() + val->shape().elements());
  else if(sizeOf(vtype) == 2)
    return 1024.0f;
  else
    ABORT("Unhandled type size {}", sizeOf(vtype));
}

/*
 * Prepare an activation matrix into intgemm8/16 format. For now the activation matrix is just quantized.
 * Expr input: The input tensor
 */

template<Type vtype>
static inline Expr prepareA(Expr a) {
  auto nodeOp = [](Expr out, const std::vector<Expr>& nodes) {
    Expr in = nodes[0];
    auto quantMult = computeQuantMult<vtype>(in->val());
    typedef typename intgemm_<vtype>::type Integer;
    intgemm_<vtype>::width::PrepareA(in->val()->data(), /*input*/
                                     out->val()->data<Integer>(), /*output*/
                                     quantMult, /*Quant Mult*/
                                     rows(in->val()),
                                     cols(in->val()));
    getQuantMult<vtype>(out->val()) = quantMult;
  };

  return lambda({a}, a->shape(), vtype, nodeOp);
}
#endif

template<Type vtype>
static inline Expr affineOrDotTyped(Expr a, Expr bQuant, Expr bias, bool transA, bool /*transB*/, float scale) {
#if COMPILE_CPU
  ABORT_IF(!isFloat(a->value_type()), "Intgemm expects type of A to be float32 not {}", a->value_type());
  ABORT_IF(!isIntgemm(bQuant->value_type()), "Intgemm expects type of B to be a variant of intgemm not {}", bQuant->value_type());

  auto aQuant = prepareA<vtype>(transA ? transpose(a) : a); // A should not be quantized yet as seen above, hence quantize here
  
  Shape outShape = aQuant->shape();
  outShape.set(-1, bQuant->shape()[-1]);

  auto dotOrAffineNodeOp = [=](Expr out, const std::vector<Expr>& nodes) {
    Expr aQuant = nodes[0];
    Expr bQuant = nodes[1];
    Expr bias   = nodes.size() > 2 ? nodes[2] : nullptr;

    float aQuantMult = getQuantMult<vtype>(aQuant->val());
    float bQuantMult = getQuantMult<vtype>(bQuant->val());
        
    float unquant_mult = 1.0f / (aQuantMult * bQuantMult);
    unquant_mult = unquant_mult * scale;

    typedef typename intgemm_<vtype>::type Integer;
    if(bias) {
      intgemm_<vtype>::width::Multiply(/*A=*/aQuant->val()->data<Integer>(),
                                       /*B=*/bQuant->val()->data<Integer>(),
                                       rows(aQuant->val()),
                                       cols(aQuant->val()),
                                       cols(bQuant->val()),
                                       intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, /*bias=*/bias->val()->data(), /*output=*/out->val()->data()));
    } else {
      intgemm_<vtype>::width::Multiply(/*A=*/aQuant->val()->data<Integer>(),
                                       /*B=*/bQuant->val()->data<Integer>(),
                                       rows(aQuant->val()),
                                       cols(aQuant->val()),
                                       cols(bQuant->val()),
                                       intgemm::callbacks::UnquantizeAndWrite(unquant_mult, /*output=*/out->val()->data()));
    }
  };

  std::vector<Expr> nodes = {aQuant, bQuant};
  if(bias)
    nodes.push_back(bias);

  return lambda(nodes, outShape, Type::float32, dotOrAffineNodeOp);
#else
  a, b, bias, transA, scale, clipValue;
  ABORT("You need to enable CPU compilation to use this feature. Use cmake .. -DCOMPILE_CPU=ON");
#endif
}

static inline Expr affineOrDot(Expr a, Expr bQuant, Expr bias, bool transA, bool transB, float scale) {
  Type bQuantElementType = bQuant->value_type();
  switch(sizeOf(bQuantElementType)) {
    case 1 :
      return cpu::integer::affineOrDotTyped<Type::intgemm8>(a, bQuant, bias, transA, transB, scale);
    case 2 :
      return cpu::integer::affineOrDotTyped<Type::intgemm16>(a, bQuant, bias, transA, transB, scale);
    default:
      ABORT("Unsupported type {} for Intgemm type??", bQuantElementType);
  }
}

}  // namespace integer
}  // namespace cpu
}  // namespace marian
