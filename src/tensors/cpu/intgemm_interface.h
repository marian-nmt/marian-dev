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
 * bool shifted: whether we use the shifted codepath to deal with unsigned \times signed 
 */
template<Type vtype>
static inline Expr prepareA(Expr a, bool shifted=false, std::string bname="") { // @TODO check if bname is necessary
  auto nodeOp = [shifted, bname](Expr out, const std::vector<Expr>& children) {
    Expr in = children[0];
    auto quantMult = computeQuantMult<vtype>(in->val(), bname + "_quantMultA");
    typedef typename intgemm_<vtype>::type Integer;
    if (shifted)  {
      intgemm::Int8Shift::PrepareA(in->val()->data(), /*input*/
                                      out->val()->data<int8_t>(), /*output*/
                                      quantMult, /*Quant Mult*/
                                      rows(in->val()),
                                      cols(in->val()));
    } else {
      intgemm_<vtype>::width::PrepareA(in->val()->data(), /*input*/
                                      out->val()->data<Integer>(), /*output*/
                                      quantMult, /*Quant Mult*/
                                      rows(in->val()),
                                      cols(in->val()));
    }
    getQuantMult<vtype>(out->val()) = quantMult;
  };

  return lambda({a}, a->shape(), vtype, nodeOp);
}
#endif

// @TODO this is not memoised so we have the longer version further down
/*
#if COMPILE_CPU
template<Type vtype>
static inline Expr prepareB(Expr b) {
  auto nodeOp = [](Expr out, const std::vector<Expr>& children) {
    Expr in = children[0];
    typedef typename intgemm_<vtype>::type Integer;
    if (isIntgemm(in->value_type())) { // @ Does this ever get triggered?
      out->val() = in->val();
    } else {
      auto quantMult = computeQuantMult<vtype>(in->val());
      intgemm_<vtype>::width::PrepareB(in->val()->data(),
                                      out->val()->data<Integer>(),
                                      rows(in->val()),
                                      cols(in->val()));
      getQuantMult<vtype>(out->val()) = quantMult;
    }
  };
  return lambda({b}, b->shape(), vtype, nodeOp);
}
#endif
*/

template<Type vtype>
struct PrepareBNodeOp : public UnaryNodeOp {

  PrepareBNodeOp(Expr input)
      : UnaryNodeOp(input, input->shape(), intgemm_<vtype>::intgemmType){

    set_name(input->name());
    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of B cannot be null");
    ABORT_IF(input->shape()[-1] %8 != 0, "Columns of matrix: " + input->type() + " must be multiple of 8.");
  }

  NodeOps forwardOps() override {
   return {NodeOp(
      typedef typename intgemm_<vtype>::type Integer;
      if (isIntgemm(child(0)->value_type())) {
        val_ = child(0)->val();
      } else {
        auto quantMult = computeQuantMult<vtype>(child(0)->val(), name());
        intgemm_<vtype>::width::PrepareB(child(0)->val()->data(), /*input*/
                                      val_->data<Integer>(), /*output*/
                                      quantMult, /*Quant Mult*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()));
        getQuantMult<vtype>(val_) = quantMult;
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
  SelectColumnsBNodeOp(Expr input, const std::vector<uint_least32_t>  &indices)
      : UnaryNodeOp(input, newShape(input, indices), intgemm_<vtype>::intgemmType), indices_(indices) {

    set_name(input->name());
    setMemoize(false); // Enabling memoization leads to a massive memory leak. Instead use special "midterm" memory.
                       // Still, I don't understand why setMemoize(true) still leaks.
    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "B cannot be null");

    // Check if intgemm
    ABORT_IF(!isIntgemm(input->value_type()), "We need to prepareB before getting the indices here.");

    // Check number of selected columns
    ABORT_IF(indices.size() % 8 != 0, "Shortlist selected vocabulary must be a multiple of 8.");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      //We get the quantization multiplier from a PrepareB or directly from the input
      float quantMult = getQuantMult<vtype>(child(0));
      auto input = child(0)->val();
      typedef typename intgemm_<vtype>::type Integer;
      intgemm_<vtype>::width::SelectColumnsB(
                    reinterpret_cast<Integer *>(input->data()),
                    val_->data<Integer>(),
                    rows(input),
                    &*indices_.begin(),
                    &*indices_.end());
      // Store quant mult on the node
      getQuantMult<vtype>(val_) = quantMult;
      // @TODO Store AQuantMult here as well, if precomputed alphas
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

template<Type vtype> // Without the template marian thinks this is an instrusive ptr, I'm not sure why.
class PrepareBiasForBNodeOp : public NaryNodeOp {
  bool alreadyPrepared_ = false;
public:
  PrepareBiasForBNodeOp(Expr bias, Expr inputB_preppd, Expr inputA_preppd)
      : NaryNodeOp({bias, inputB_preppd, inputA_preppd}, bias->shape(), Type::float32) {

    set_name(bias->name() + "_Prepared");
    if (bias->type() == "cols" && bias->graph()->getBackend()->isPrecomputedAlpha()) {
      ABORT("We shouldn't ever be here");
      alreadyPrepared_ = true;
    } else if (!bias->graph()->getBackend()->isPrecomputedAlpha()){
      setMemoize(false);
    }
  }

  NodeOps forwardOps() override {
    //std::cerr << "TrueBias: " << child(0)->name() << " type: " << child(0)->type() << " bQuantMult: " << this->child(3)->val()->data()[0] <<  " aQuantMult: " << this->child(2)->val()->data()[0] << std::endl;
    //std::cerr << "Bias name and val: " << child(0)->name() << " " << child(0)->val()->data()[0] << std::endl;
    return {NodeOp(
      if (alreadyPrepared_) {
        //God Knows why trying to assign the bias tensor to this node causes a crash, the second time it's referenced
        //even though it's supposed to work fine. We use a memory copy instead.
        ABORT("We shouldn't ever be here.");
        std::memcpy(val_->data(), child(0)->val()->data(), child(0)->shape()[-1]*sizeof(float));
        //val_ = child(0)->val();
      } else {
        auto bias = this->child(0)->val();
        auto b = this->child(1)->val();
        float quant_mult_b = getQuantMult<vtype>(child(1));
        float quant_mult_a = getQuantMult<vtype>(child(2));
        
        float unquant_mult = (-1)*((127.0f / quant_mult_a)*(127.0f / quant_mult_b))/(127.0f); //Minus one to invert add_ps later on
        intgemm::Int8Shift::PrepareBias((const int8_t *)b->data(), rows(b), cols(b), intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias->data(), val_->data()));
      }
      )};
  }

  const std::string type() override { return "prepareBias"; }
};

template<Type vtype> // Without the template marian thinks this is an instrusive ptr, I'm not sure why.
class PrepareFakeBiasForBNodeOp : public NaryNodeOp {
public:
  PrepareFakeBiasForBNodeOp(Expr inputB_preppd, Expr inputA_preppd)
      : NaryNodeOp({inputB_preppd, inputA_preppd}, {1, inputB_preppd->shape()[-1]}, Type::float32) {

    set_name(inputB_preppd->name() + "_FakeBias");
    if (!inputB_preppd->graph()->getBackend()->isPrecomputedAlpha()) {
      setMemoize(false);
    }
  }

  NodeOps forwardOps() override {
    return {NodeOp(
    auto b = this->child(0)->val();
    float quant_mult_b = getQuantMult<vtype>(child(1));
    float quant_mult_a = getQuantMult<vtype>(child(2));

    float unquant_mult = (-1)*((127.0f / quant_mult_a)*(127.0f / quant_mult_b))/(127.0f); //Minus one to invert add_ps later on
    intgemm::Int8Shift::PrepareBias((const int8_t *)b->data(), rows(b), cols(b), intgemm::callbacks::UnquantizeAndWrite(unquant_mult, val_->data()));
    )};
  }

  const std::string type() override { return "prepareFakeBias"; }
};

/*	
 * This computes A*B (+ bias if available) in intgemm.	
 * Expr a: The activation matrix in intgemm format	
 * Expr b: The parameter matrix in intgemm fromat	
 * Expr bias: The bias	
 * bool transA - tranpose input A if true
 * bool transB - unused here (@TODO remove?)
 * float scale - scale the output by `scale`
 * the template argument controls whether we're doing 16bit integers or 8bit integers. 
 * It can be Type::intgemm8 or Type::intgemm16 and all hardware-specific variants	
 */
template<Type vtype>
static inline Expr affineOrDotTyped(Expr a, Expr bQuant, Expr bias, bool transA, bool /*transB*/, float scale) {
#if COMPILE_CPU
  ABORT_IF(!isFloat(a->value_type()), "Intgemm expects type of A to be float32 not {}", a->value_type());
  ABORT_IF(!isIntgemm(bQuant->value_type()), "Intgemm expects type of B to be a variant of intgemm not {}", bQuant->value_type());

  auto aQuant = prepareA<vtype>(transA ? transpose(a) : a); // A should not be quantized yet as seen above, hence quantize here
  
  // determine the output shape m x n for A: m x k and B: k x n
  // since we transpose A beforehand we don't need to take care of transposed shapes here 
  Shape outShape = aQuant->shape();
  outShape.set(-1, bQuant->shape()[-1]);

  // wrap the multiply finctions to be executed in the forward step of a Lambda node
  auto dotOrAffineNodeOp = [=](Expr out, const std::vector<Expr>& children) {
    Expr aQuant = children[0];
    Expr bQuant = children[1];
    Expr bias   = children.size() > 2 ? children[2] : nullptr;

    // when we arrive here, A and B are already quantized, so just get the multipliers
    float aQuantMult = getQuantMult<vtype>(aQuant->val());
    float bQuantMult = getQuantMult<vtype>(bQuant->val());
        
    float unquant_mult = 1.0f / (aQuantMult * bQuantMult);
    unquant_mult = unquant_mult * scale;

    typedef typename intgemm_<vtype>::type Integer;
    if(bias) { // dispatch a multiply with integrated bias addition i.e affine(...)
      intgemm_<vtype>::width::Multiply(/*A=*/aQuant->val()->data<Integer>(),
                                       /*B=*/bQuant->val()->data<Integer>(),
                                       rows(aQuant->val()),
                                       cols(aQuant->val()),
                                       cols(bQuant->val()),
                                       intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, /*bias=*/bias->val()->data(), /*output=*/out->val()->data()));
    } else { // dispatch a multiply without bias addition i.e dot(...)
      intgemm_<vtype>::width::Multiply(/*A=*/aQuant->val()->data<Integer>(),
                                       /*B=*/bQuant->val()->data<Integer>(),
                                       rows(aQuant->val()),
                                       cols(aQuant->val()),
                                       cols(bQuant->val()),
                                       intgemm::callbacks::UnquantizeAndWrite(unquant_mult, /*output=*/out->val()->data()));
    }
  };

  std::vector<Expr> children = {aQuant, bQuant};
  if(bias)
    children.push_back(bias);

  return lambda(children, outShape, Type::float32, dotOrAffineNodeOp); // inference-only Lambda node
#else
  a, bQuant, bias, transA, scale;
  ABORT("You need to enable CPU compilation to use this feature. Use cmake .. -DCOMPILE_CPU=ON");
#endif
}

// Dispatch correct hardware-agnostic or hardware-specific matrix multiplies
static inline Expr affineOrDot(Expr a, Expr bQuant, Expr bias, bool transA, bool transB, float scale) {
  Type bQuantElementType = bQuant->value_type();
  static const bool pass = cpu::integer::passOrAbort(bQuantElementType);
  pass; // We declare this variable as static so that passOrAbort is only ever run once during the initialization.
  switch(bQuantElementType) {
    //case Type::intgemm8 :  // The generic case selects CPU automatically, but we set all the types manually anyways.
    //  return cpu::integer::affineOrDotTyped<Type::intgemm8>(a, bQuant, bias, transA, transB, scale);    
    case Type::intgemm8ssse3 :
      return cpu::integer::affineOrDotTyped<Type::intgemm8ssse3>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm8avx2 :
      return cpu::integer::affineOrDotTyped<Type::intgemm8avx2>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm8avx512 :
      return cpu::integer::affineOrDotTyped<Type::intgemm8avx512>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm8avx512vnni :
      return cpu::integer::affineOrDotTyped<Type::intgemm8avx512vnni>(a, bQuant, bias, transA, transB, scale);
    //case Type::intgemm16 :  // The generic case selects CPU automatically, but we set all the types manually anyways.
    //  return cpu::integer::affineOrDotTyped<Type::intgemm16>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm16sse2 :
      return cpu::integer::affineOrDotTyped<Type::intgemm16sse2>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm16avx2 :
      return cpu::integer::affineOrDotTyped<Type::intgemm16avx2>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm16avx512 :
      return cpu::integer::affineOrDotTyped<Type::intgemm16avx512>(a, bQuant, bias, transA, transB, scale);
    default:
      ABORT("Unsupported type {} for Intgemm type??", bQuantElementType);
  }
}

}  // namespace integer
}  // namespace cpu
}  // namespace marian
