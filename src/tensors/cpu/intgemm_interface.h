#pragma once

#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "integer_common.h"

#include "oneapi/dnnl/dnnl.hpp"

static inline void printDNNLStatus(dnnl::status& status) {
  if (status == dnnl::status::success) {
      std::cout << "DNNL success." << std::endl;
  } else if (status == dnnl::status::out_of_memory ) {
      std::cout << "The operation failed due to an out-of-memory condition." << std::endl;
  } else if (status == dnnl::status::invalid_arguments ) {
      std::cout << "The operation failed because of incorrect function arguments." << std::endl;
  } else if (status == dnnl::status::unimplemented) {
      std::cout << "The operation failed because requested functionality is not implemented." << std::endl;
  } else if (status == dnnl::status::iterator_ends) {
      std::cout << "Primitive iterator passed over last primitive descriptor." << std::endl;
  } else if (status == dnnl::status::runtime_error) {
      std::cout << "Primitive or engine failed on execution." << std::endl;
  } else if (status == dnnl::status::not_required) {
      std::cout << "Queried element is not required for given primitive." << std::endl;
  }
}

static inline dnnl::status my_gemm_s8s8s32(char transa      , /* 'N', whether A is transposed */
                          char transb      , /* transposeB, whether B is transposed */
                          char offsetc     , /* 'F', offsets applied to matrix C, F means same offset to each element, C each column, R each row*/
                          dnnl_dim_t M     , /* M, */
                          dnnl_dim_t N     , /* N, */
                          dnnl_dim_t K     , /* K, */
                          float alpha      , /* 1.0f, scales A and B */
                          const int8_t* A  , /* aQuant->val()->data<int8_t>(), // M*K */
                          dnnl_dim_t lda   , /* lda, leading dimension of A */
                          int8_t ao        , /* ao, offset value for A */
                          const int8_t* B  , /* bQuant->val()->data<int8_t>(), // K*N */
                          dnnl_dim_t ldb   , /* ldb, leading dimension for B */
                          int8_t bo        , /* bo, offset value of B */
                          float beta       , /* 0.0f, scale for matrix C */
                          int32_t* C       , /* out->val()->data<int32_t>(), // M*N */
                          dnnl_dim_t ldc   , /* ldc, leading dimension for C */
                          const int32_t* co /* co.data()); Array of offset values for C, see `offsetc` */
) {
  ABORT_UNLESS(offsetc == 'F' && co[0] == 0, "Offsets for C is not implemented");

  using matmul = dnnl::matmul;
  using dims = dnnl::memory::dims;
  using dt = dnnl::memory::data_type;

  static dnnl::engine eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
  dnnl::matmul matmul_p;

  dims a_dims_strides = transa == 'N' ? dims {DNNL_RUNTIME_DIM_VAL, 1} : dims {1, DNNL_RUNTIME_DIM_VAL};
  dims b_dims_strides = transb == 'N' ? dims {DNNL_RUNTIME_DIM_VAL, 1} : dims {1, DNNL_RUNTIME_DIM_VAL};
  
  dims c_dims_strides = {DNNL_RUNTIME_DIM_VAL, 1};
  dims rt_rt_dims = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
  dims rt_1_dims = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};

  dnnl::memory::desc a_md(rt_rt_dims, dt::s8,  a_dims_strides);
  dnnl::memory::desc b_md(rt_rt_dims, dt::s8,  b_dims_strides);
  dnnl::memory::desc c_md(rt_rt_dims, dt::s32, c_dims_strides);


  dnnl::primitive_attr attr;
  attr.set_output_scales(/* mask */ 0, {DNNL_RUNTIME_F32_VAL});
  if (beta != 0.f) {
      assert(beta == 1.f); // current limitation
      dnnl::post_ops po;
      po.append_sum(beta);
      attr.set_post_ops(po);
  }

  matmul::desc matmul_d(a_md, b_md, c_md);
  matmul::primitive_desc matmul_pd(matmul_d, attr, eng, true);
  if (matmul_pd) matmul_p = matmul(matmul_pd);

  dims a_strides = transa == 'N' ? dims {lda, 1} : dims {1, lda};
  dims b_strides = transb == 'N' ? dims {ldb, 1} : dims {1, ldb};


  dnnl::memory A_m({{M, K}, dt::s8, a_strides}, eng, (void *)A);
  dnnl::memory B_m({{K, N}, dt::s8, b_strides}, eng, (void *)B);
  dnnl::memory C_m({{M, N}, dt::s32, {ldc, 1}}, eng, (void *)C);

  // Prepare oneDNN memory for alpha
  dnnl::memory alpha_m({{1}, dt::f32, {1}}, eng, &alpha);

  dnnl::stream s(eng);
  matmul_p.execute(s, {
    {DNNL_ARG_SRC, A_m},
    {DNNL_ARG_WEIGHTS, B_m},
    {DNNL_ARG_DST, C_m},
    {DNNL_ARG_ATTR_OUTPUT_SCALES, alpha_m}
  });
  s.wait();
  
  return dnnl::status::success;
}

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
static inline Expr prepareA(Expr a, Expr bPreppd, bool shifted=false) { // @TODO check if bname is necessary
  auto nodeOp = [shifted](Expr out, const std::vector<Expr>& children) {
    Expr in = children[0];
    Expr bPreppd = children[1];
    static bool precomputedAlpha = in->graph()->getBackend()->isPrecomputedAlpha();
    float quantMult;
    if (precomputedAlpha) { // If we have precomputed alphas, the quantisation multiplier is saved onto the B node. Else, we don't use it at all
      quantMult = getQuantMultA<vtype>(bPreppd->val());
    } else {
      quantMult = computeQuantMult<vtype>(in->val(), bPreppd->name() + "_QuantMultA");
    }
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

  return lambda({a, bPreppd}, a->shape(), vtype, nodeOp);
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
  bool transpose_;

  PrepareBNodeOp(Expr input, bool transpose)
      : UnaryNodeOp(input, newShape(input, transpose), vtype), transpose_(transpose) {

    set_name(input->name());
    if (!transpose_) {
      //ABORT_IF(input->shape()[-1] %8 != 0, "Columns of matrix: " + input->type() + " must be multiple of 8.");
    } else {
      ABORT_IF((input->shape().elements()/input->shape()[-1]) %8 != 0, "Rows of matrix: " + input->type() + " must be multiple of 8.");
    }
  }

  NodeOps forwardOps() override {
   return {NodeOp(
      static bool precomputedAlpha = child(0)->graph()->getBackend()->isPrecomputedAlpha();
      typedef typename intgemm_<vtype>::type Integer;
      static bool use_oneDNN = child(0)->graph()->getBackend()->useOneDNNOnly();

      if (isIntgemm(child(0)->value_type())) {
        val_ = child(0)->val();
      } else if (use_oneDNN /*&& !transpose_*/) { //@TODO proper codepaths, make sure only the ones that can't do intgemm, go through DNNL
        // Use DNNL in this case, meaning we need prepareA. @TODO maybe try shifted version and also code one that doesn't care about register size
        //ABORT_IF(transpose_, "We haven't implemented DNNL transposed matrices for now.");
        auto quantMult = computeQuantMult<vtype>(child(0)->val(), name());
        intgemm_<vtype>::width::PrepareA(child(0)->val()->data(), /*input*/
                                      val_->data<Integer>(), /*output*/
                                      quantMult, /*Quant Mult*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()));
        getQuantMult<vtype>(val_) = quantMult;
      } else if (!transpose_) {
        auto quantMult = computeQuantMult<vtype>(child(0)->val(), name());
        intgemm_<vtype>::width::PrepareB(child(0)->val()->data(), /*input*/
                                      val_->data<Integer>(), /*output*/
                                      quantMult, /*Quant Mult*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()));
        getQuantMult<vtype>(val_) = quantMult;
      } else {
        auto quantMult = computeQuantMult<vtype>(child(0)->val(), name());
        intgemm_<vtype>::width::PrepareBTransposed(child(0)->val()->data(), /*input*/
                                      val_->data<Integer>(), /*output*/
                                      quantMult,
                                      cols(child(0)->val()), /*Cols and rows need to be swapped*/
                                      rows(child(0)->val())); /*Cols and rows need to be swapped*/
        getQuantMult<vtype>(val_) = quantMult;
      }
      if (precomputedAlpha) { // If we have precomputed alpha we can get them onto B when preparing the model
        std::string aQuantKey = name() + "_QuantMultA";
        //Very Hacky Bit. Unnamed matrix is notpart of the F0 parameter namespace
        if (aQuantKey.at(0) != 'F') {
          aQuantKey = "F0::" + aQuantKey;
        }
        auto map = child(0)->graph()->params()->getMap();
        const auto mapiter = map.find(aQuantKey);
        if (mapiter != map.end()) {
          getQuantMultA<vtype>(val_) = *mapiter->second->val()->data();
        } else {
          ABORT("We did not find an alpha in the model named: {}.", name());
      }
      }
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  static Shape newShape(Expr input, bool transposed) {
    Shape ret = input->shape();
    // In the oneDNN case, handle transposition down the line
    static bool use_oneDNN = input->graph()->getBackend()->useOneDNNOnly();
    if (transposed && !use_oneDNN) {
      ret.set(0, input->shape()[-1]);
      ret.set(1, input->shape()[0]);
    } else {
      ret = input->shape();
    }
    return ret;
  }

  const std::string type() override { return "intgemmPrepareB"; }
};

template<Type vtype>
struct SelectColumnsBNodeOp : public UnaryNodeOp {
public:
  SelectColumnsBNodeOp(Expr input, const std::vector<uint_least32_t>  &indices)
      : UnaryNodeOp(input, newShape(input, indices), vtype), indices_(indices) {

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
      float quantMult = getQuantMult<vtype>(child(0)->val());
      auto input = child(0)->val();
      typedef typename intgemm_<vtype>::type Integer;
      intgemm_<vtype>::width::SelectColumnsB(
                    reinterpret_cast<Integer *>(input->data()),
                    val_->data<Integer>(),
                    rows(input),
                    &*indices_.begin(),
                    &*indices_.end());
      // Store quant mult on the node. It will only be used if precomputedAlphas are turned on
      getQuantMult<vtype>(val_) = quantMult;
      getQuantMultA<vtype>(val_) = getQuantMultA<vtype>(child(0)->val());
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

// Temporary placeholder for QuantMultA for when not using precomputed alphas
template<Type vtype>
struct QuantMultANodeOp : public UnaryNodeOp {
  QuantMultANodeOp(Expr input, std::string& bname) : UnaryNodeOp(input, Shape({1}), Type::float32){
      set_name(input->name() + "_QuantMultB");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
        *val_->data() = getQuantMult<vtype>(child(0));
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override {
      return "intgemmQuantMultA";
  }

};

template<Type vtype> // Without the template marian thinks this is an instrusive ptr, I'm not sure why.
struct PrepareBiasForBNodeOp : public NaryNodeOp {
//private:
//  ENABLE_INTRUSIVE_PTR(PrepareBiasForBNodeOp)
public:
  PrepareBiasForBNodeOp(Expr bias, Expr inputB_preppd, Expr inputA_preppd)
      : NaryNodeOp({bias, inputB_preppd, inputA_preppd}, bias->shape(), Type::float32) {

    set_name(bias->name() + "_Prepared");
    if (bias->type() == "cols" && bias->graph()->getBackend()->isPrecomputedAlpha()) {
      ABORT("We shouldn't ever be here. The bias would have been prepared by prior running select columns b");
    } else if (!bias->graph()->getBackend()->isPrecomputedAlpha()){
      setMemoize(false);
    }
  }

  PrepareBiasForBNodeOp(Expr bias, Expr inputB_preppd)
      : NaryNodeOp({bias, inputB_preppd}, bias->shape(), Type::float32) {

    set_name(bias->name() + "_Prepared");
    if (bias->type() == "cols" && bias->graph()->getBackend()->isPrecomputedAlpha()) {
      ABORT("We shouldn't ever be here. The bias would have been prepared by prior running select columns b");
    } else if (!bias->graph()->getBackend()->isPrecomputedAlpha()){
      ABORT("We can only use this codepath with precomputed alphas, as they are attached to the B node.");
    }
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      auto bias = this->child(0)->val();
      auto b = this->child(1)->val();
      float quant_mult_b = getQuantMult<vtype>(child(1)->val());
      float quant_mult_a;
      if (children().size() == 3) { // Not precomputed alphas, we get the quantMult from the nodeA prepared
        quant_mult_a = getQuantMult<vtype>(child(2)->val());
      } else {
        quant_mult_a = getQuantMultA<vtype>(child(1)->val());
      }
      float unquant_mult = (-1)*((127.0f / quant_mult_a)*(127.0f / quant_mult_b))/(127.0f); //Minus one to invert add_ps later on
      static bool use_oneDNN = child(0)->graph()->getBackend()->useOneDNNOnly();
      if (!use_oneDNN) {
        intgemm::Int8Shift::PrepareBias((const int8_t *)b->data(), rows(b), cols(b), intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias->data(), val_->data()));
      } else {
        static const std::vector<uint8_t> ones(64000, 1); // Large enough static array of ones that we can use
        // DNNL parameters
        const dnnl_dim_t M = 1;
        dnnl_dim_t K = rows(b);
        dnnl_dim_t N = cols(b);

        const int8_t ao = 0;
        const int8_t bo = 0;
        static const std::vector<int32_t> co(1,0);
        ///std::array<int32_t, 1> co = {0}; // This syntax is not allowed due to being in a macro
        auto status = dnnl::gemm_u8s8s32(/*transA*/  'N',
                                        /*transB*/  'N',
                                        /*OffsetC*/ 'F', /* This parameter denotes whether there can be bias adition. Sadly while it technically supports it, it's only int32_t.*/
                                        M,
                                        N,
                                        K,
                                        /*alpha*/ 1.0f,
                                        ones.data(),
                                        /*lda*/ K,
                                        ao,
                                        b->template data<int8_t>(),
                                        /*ldb*/ N,
                                        bo,
                                        /*beta*/ 0.0f,
                                        val_->data<int32_t>(),
                                        /*ldc*/ N,
                                        co.data());

        if (status != dnnl::status::success) {
          printDNNLStatus(status);
          ABORT("PrepareBias gemm didn't run");
        }

        //Unquantise and add bias if necessary
        UnquantiseAndAddBias(val_, bias, unquant_mult);
      }
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
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

  PrepareFakeBiasForBNodeOp(Expr inputB_preppd)
      : NaryNodeOp({inputB_preppd}, {1, inputB_preppd->shape()[-1]}, Type::float32) {

    set_name(inputB_preppd->name() + "_FakeBias");
    if (!inputB_preppd->graph()->getBackend()->isPrecomputedAlpha()) {
      ABORT("We can only use this codepath with precomputed alphas, as they are attached to the B node.");
    }
  }

  NodeOps forwardOps() override {
    return {NodeOp(
    auto b = this->child(0)->val();
    float quant_mult_b = getQuantMult<vtype>(child(0)->val());
    float quant_mult_a;
    if (children().size() == 2) { // Not precomputed alphas
      quant_mult_a = getQuantMult<vtype>(child(1)->val());
    } else {
      quant_mult_a = getQuantMultA<vtype>(child(0)->val());
    }

    float unquant_mult = (-1)*((127.0f / quant_mult_a)*(127.0f / quant_mult_b))/(127.0f); //Minus one to invert add_ps later on
    static bool use_oneDNN = child(0)->graph()->getBackend()->useOneDNNOnly();
    if (!use_oneDNN) {
      intgemm::Int8Shift::PrepareBias((const int8_t *)b->data(), rows(b), cols(b), intgemm::callbacks::UnquantizeAndWrite(unquant_mult, val_->data()));
    } else {
      static const std::vector<uint8_t> ones(64000, 1); // Large enough static array of ones that we can use
        // DNNL parameters
        const dnnl_dim_t M = 1;
        dnnl_dim_t K = rows(b);
        dnnl_dim_t N = cols(b);

        const int8_t ao = 0;
        const int8_t bo = 0;
        static const std::vector<int32_t> co(1,0);
        ///std::array<int32_t, 1> co = {0}; // This syntax is not allowed due to being in a macro
        auto status = dnnl::gemm_u8s8s32(/*transA*/  'N',
                                        /*transB*/  'N',
                                        /*OffsetC*/ 'F', /* This parameter denotes whether there can be bias adition. Sadly while it technically supports it, it's only int32_t.*/
                                        M,
                                        N,
                                        K,
                                        /*alpha*/ 1.0f,
                                        ones.data(),
                                        /*lda*/ K,
                                        ao,
                                        b->template data<int8_t>(),
                                        /*ldb*/ N,
                                        bo,
                                        /*beta*/ 0.0f,
                                        val_->data<int32_t>(),
                                        /*ldc*/ N,
                                        co.data());

        if (status != dnnl::status::success) {
          printDNNLStatus(status);
          ABORT("PrepareBias gemm didn't run");
        }

        JustUnquantise(val_, unquant_mult);
    }
    )};
  }

  const std::string type() override { return "prepareFakeBias"; }
};

static Expr SelectColumnsBTyped(Expr input, const std::vector<uint_least32_t>  &indices) {
  static const Type intgemmType = cpu::integer::getIntgemmType(input->graph()->getBackend()->getGemmType());
  static const bool pass = cpu::integer::passOrAbort(intgemmType);
  pass; // We declare this variable as static so that passOrAbort is only ever run once during the initialization.
  switch(intgemmType) {
    case Type::intgemm8ssse3 :
      return Expression<SelectColumnsBNodeOp<Type::intgemm8ssse3> >(input, indices);
    case Type::intgemm8avx2 :
      return Expression<SelectColumnsBNodeOp<Type::intgemm8avx2> > (input, indices);
    case Type::intgemm8avx512 :
      return Expression<SelectColumnsBNodeOp<Type::intgemm8avx512> >(input, indices);
    case Type::intgemm8avx512vnni :
      return Expression<SelectColumnsBNodeOp<Type::intgemm8avx512vnni> > (input, indices);
    case Type::intgemm16sse2 :
      return Expression<SelectColumnsBNodeOp<Type::intgemm16sse2> >(input, indices);
    case Type::intgemm16avx2 :
      return Expression<SelectColumnsBNodeOp<Type::intgemm16avx2> > (input, indices);
    case Type::intgemm16avx512 :
      return Expression<SelectColumnsBNodeOp<Type::intgemm16avx512> > (input, indices);
    default:
      ABORT("Unsupported type {} for Intgemm type??", intgemmType);
  }
}

static Expr prepareBTyped(Expr input, bool transpose=false) {
  static const Type intgemmType = cpu::integer::getIntgemmType(input->graph()->getBackend()->getGemmType());
  static const bool pass = cpu::integer::passOrAbort(intgemmType);
  pass; // We declare this variable as static so that passOrAbort is only ever run once during the initialization.
  // Get the intgemm type the first time we run into a function, as in the future we will have the same type invocation.
  switch(intgemmType) {
    case Type::intgemm8ssse3 :
      return Expression<PrepareBNodeOp<Type::intgemm8ssse3> >(input, transpose);
    case Type::intgemm8avx2 :
      return Expression<PrepareBNodeOp<Type::intgemm8avx2> > (input, transpose);
    case Type::intgemm8avx512 :
      return Expression<PrepareBNodeOp<Type::intgemm8avx512> >(input, transpose);
    case Type::intgemm8avx512vnni :
      return Expression<PrepareBNodeOp<Type::intgemm8avx512vnni> > (input, transpose);
    case Type::intgemm16sse2 :
      return Expression<PrepareBNodeOp<Type::intgemm16sse2> >(input, transpose);
    case Type::intgemm16avx2 :
      return Expression<PrepareBNodeOp<Type::intgemm16avx2> > (input, transpose);
    case Type::intgemm16avx512 :
      return Expression<PrepareBNodeOp<Type::intgemm16avx512> > (input, transpose);
    default:
      ABORT("Unsupported type {} for Intgemm type??", intgemmType);
  }
}


static Expr PrepareTrueBiasForBTyped(Expr bias, Expr inputB_preppd, Expr inputA_preppd=nullptr) {
  static const Type intgemmType = inputB_preppd->value_type();
  if (inputA_preppd) {
    switch(intgemmType) {
      case Type::intgemm8ssse3 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm8ssse3> >(bias, inputB_preppd, inputA_preppd);
      case Type::intgemm8avx2 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm8avx2> > (bias, inputB_preppd, inputA_preppd);
      case Type::intgemm8avx512 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm8avx512> >(bias, inputB_preppd, inputA_preppd);
      case Type::intgemm8avx512vnni :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm8avx512vnni> > (bias, inputB_preppd, inputA_preppd);
      case Type::intgemm16sse2 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm16sse2> >(bias, inputB_preppd, inputA_preppd);
      case Type::intgemm16avx2 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm16avx2> > (bias, inputB_preppd, inputA_preppd);
      case Type::intgemm16avx512 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm16avx512> > (bias, inputB_preppd, inputA_preppd);
      default:
        ABORT("Unsupported type {} for Intgemm type??", intgemmType);
    }
  } else {
    switch(intgemmType) {
      case Type::intgemm8ssse3 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm8ssse3> >(bias, inputB_preppd);
      case Type::intgemm8avx2 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm8avx2> > (bias, inputB_preppd);
      case Type::intgemm8avx512 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm8avx512> >(bias, inputB_preppd);
      case Type::intgemm8avx512vnni :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm8avx512vnni> > (bias, inputB_preppd);
      case Type::intgemm16sse2 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm16sse2> >(bias, inputB_preppd);
      case Type::intgemm16avx2 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm16avx2> > (bias, inputB_preppd);
      case Type::intgemm16avx512 :
        return Expression<PrepareBiasForBNodeOp<Type::intgemm16avx512> > (bias, inputB_preppd);
      default:
        ABORT("Unsupported type {} for Intgemm type??", intgemmType);
    }
  }
}

static Expr PrepareFakeBiasForBTyped(Expr inputB_preppd, Expr inputA_preppd=nullptr) {
  static const Type intgemmType = inputB_preppd->value_type();
  if (inputA_preppd) {
    switch(intgemmType) {
      case Type::intgemm8ssse3 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm8ssse3> >(inputB_preppd, inputA_preppd);
      case Type::intgemm8avx2 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm8avx2> > (inputB_preppd, inputA_preppd);
      case Type::intgemm8avx512 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm8avx512> >(inputB_preppd, inputA_preppd);
      case Type::intgemm8avx512vnni :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm8avx512vnni> > (inputB_preppd, inputA_preppd);
      case Type::intgemm16sse2 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm16sse2> >(inputB_preppd, inputA_preppd);
      case Type::intgemm16avx2 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm16avx2> > (inputB_preppd, inputA_preppd);
      case Type::intgemm16avx512 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm16avx512> > (inputB_preppd, inputA_preppd);
      default:
        ABORT("Unsupported type {} for Intgemm type??", intgemmType);
    }
  } else {
    switch(intgemmType) {
      case Type::intgemm8ssse3 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm8ssse3> >(inputB_preppd);
      case Type::intgemm8avx2 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm8avx2> > (inputB_preppd);
      case Type::intgemm8avx512 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm8avx512> >(inputB_preppd);
      case Type::intgemm8avx512vnni :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm8avx512vnni> > (inputB_preppd);
      case Type::intgemm16sse2 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm16sse2> >(inputB_preppd);
      case Type::intgemm16avx2 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm16avx2> > (inputB_preppd);
      case Type::intgemm16avx512 :
        return Expression<PrepareFakeBiasForBNodeOp<Type::intgemm16avx512> > (inputB_preppd);
      default:
        ABORT("Unsupported type {} for Intgemm type??", intgemmType);
    }
  }
}

static Expr PrepareBiasForBTyped(Expr bias, Expr inputB_preppd, Expr inputA_preppd=nullptr) {
  static bool precomputedAlpha = inputB_preppd->graph()->getBackend()->isPrecomputedAlpha(); // Detect if we have precomputed alphas or not
  if (precomputedAlpha) {
    inputA_preppd = nullptr; // When we have precomputed alphas we fetch the aQuantMult from B
  }
  if (precomputedAlpha && bias && bias->type() == "cols") {
    return bias; // When we have precomputed alphas the shortlisted bias has already been prepared
  } else if (bias) {
    return PrepareTrueBiasForBTyped(bias, inputB_preppd, inputA_preppd);
  } else {
    return PrepareFakeBiasForBTyped(inputB_preppd, inputA_preppd);
  }
}


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
static inline Expr affineOrDotTyped(Expr a, Expr bQuant, Expr bias, bool transA, bool transB, float scale, bool relu) {
#if COMPILE_CPU
  ABORT_IF(!isFloat(a->value_type()), "Intgemm expects type of A to be float32 not {}", a->value_type());
  ABORT_IF(!isIntgemm(bQuant->value_type()), "Intgemm expects type of B to be a variant of intgemm not {}", bQuant->value_type());

  bool shifted = (a->graph()->getBackend()->isShifted() && bias) || a->graph()->getBackend()->isShiftedAll(); // We use the shifted codepath when we have a bias or shifted-all is enabled
  auto aQuant = prepareA<vtype>(transA ? transpose(a) : a, bQuant, shifted); // A should not be quantized yet as seen above, hence quantize here
  
  static bool use_oneDNN = aQuant->graph()->getBackend()->useOneDNNOnly();
  // determine the output shape m x n for A: m x k and B: k x n
  // since we transpose A beforehand we don't need to take care of transposed shapes here if using intgemm
  // if using DNNL we very much do!
  Shape outShape = aQuant->shape();
  outShape.set(-1, bQuant->shape()[-1]);
  if (transB && use_oneDNN) {
    outShape.set(-1, rows(bQuant->shape()));
  }

  if (shifted) {
    bias = PrepareBiasForBTyped(bias, bQuant, aQuant);
  }

  auto dnnlDotOrAffineNodeOp = [=](Expr out, const std::vector<Expr>& children) {
    Expr aQuant = children[0];
    Expr bQuant = children[1];
    Expr bias   = children.size() > 2 ? children[2] : nullptr;

    // when we arrive here, A and B are already quantized, so just get the multipliers
    float aQuantMult = getQuantMult<vtype>(aQuant->val());
    float bQuantMult = getQuantMult<vtype>(bQuant->val());

    float unquant_mult = 1.0f / (aQuantMult * bQuantMult);
    unquant_mult = unquant_mult * scale;

    // DNNL parameters
    dnnl_dim_t M = rows(aQuant->val());
    dnnl_dim_t K = cols(aQuant->val());
    dnnl_dim_t N = cols(bQuant->val());

    dnnl_dim_t lda = K;
    dnnl_dim_t ldb = N;
    dnnl_dim_t ldc = N;
    if (transB) {
      N = rows(bQuant->val());
      ldc = N;
    }

    static const constexpr int8_t ao = 0;
    static const constexpr int8_t bo = 0;
    static const constexpr std::array<int32_t, 1> co = {0};

    dnnl::status status;
    char transposeB = transB ? 'T' : 'N';

    if (shifted) {
      status = dnnl::gemm_u8s8s32(/*transA*/  'N',
                                   transposeB,
                                   /*OffsetC*/ 'F', /* This parameter denotes whether there can be bias adition. Sadly while it technically supports it, it's only int32_t.*/
                                    M,
                                    N,
                                    K,
                                    /*alpha*/ 1.0f,
                                    aQuant->val()->data<uint8_t>(),
                                    lda,
                                    ao,
                                    bQuant->val()->data<int8_t>(),
                                    ldb,
                                    bo,
                                    /*beta*/ 0.0f,
                                    out->val()->data<int32_t>(),
                                    ldc,
                                    co.data());
    } else {
      //https://oneapi-src.github.io/oneDNN/group_dnnl_api_blas.html?highlight=gemm_s8s8s32#doxid-group-dnnl-api-blas-1ga6bb7da88545097f097bbcd5778787826
      status = my_gemm_s8s8s32(/* char transa       */ 'N',
                                  /* char transb       */ transposeB,
                                  /* char offsetc      */ 'F', /* This parameter denotes whether there can be bias adition. Sadly while it technically supports it, it's only int32_t.*/
                                  /* dnnl_dim_t M      */ M,
                                  /* dnnl_dim_t N      */ N,
                                  /* dnnl_dim_t K      */ K,
                                  /* float alpha       */ 1.0f,
                                  /* const int8_t* A   */ aQuant->val()->data<int8_t>(), // M*K
                                  /* dnnl_dim_t lda    */ lda,
                                  /* int8_t ao         */ ao,
                                  /* const int8_t* B   */ bQuant->val()->data<int8_t>(),
                                  /* dnnl_dim_t ldb    */ ldb,
                                  /* int8_t bo         */ bo,
                                  /* float beta        */ 0.0f,
                                  /* int32_t* C        */ out->val()->data<int32_t>(),
                                  /* dnnl_dim_t ldc    */ ldc,
                                  /* const int32_t* co */ co.data());


    }

    if (status != dnnl::status::success) {
      printDNNLStatus(status);
      ABORT("GEMM failed to run.");
    }

    //Unquantise and add bias if necessary
    if (bias && relu) {
      UnquantiseAndAddBiasAndRelu(out->val(), bias->val(), unquant_mult);
    } else if (bias) {
      UnquantiseAndAddBias(out->val(), bias->val(), unquant_mult);
    } else if (relu) {
      JustUnquantiseRelu(out->val(), unquant_mult);
    } else {
      JustUnquantise(out->val(), unquant_mult);
    }
  };

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
      if (shifted) { // @TODO only architecture agnostic format supported for shift
        if (relu) {
          intgemm::Int8Shift::Multiply(/*A=*/aQuant->val()->data<int8_t>(),
                            /*B=*/bQuant->val()->data<int8_t>(),
                            rows(aQuant->val()),
                            cols(aQuant->val()),
                            cols(bQuant->val()),
                            intgemm::callbacks::UnquantizeAndAddBiasAndWriteRelu(unquant_mult, /*bias=*/bias->val()->data(), /*output=*/out->val()->data()));
        } else {
          intgemm::Int8Shift::Multiply(/*A=*/aQuant->val()->data<int8_t>(),
                                      /*B=*/bQuant->val()->data<int8_t>(),
                                      rows(aQuant->val()),
                                      cols(aQuant->val()),
                                      cols(bQuant->val()),
                                      intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, /*bias=*/bias->val()->data(), /*output=*/out->val()->data()));
        }
      } else {
        if (relu) {
          intgemm_<vtype>::width::Multiply(/*A=*/aQuant->val()->data<Integer>(),
                                          /*B=*/bQuant->val()->data<Integer>(),
                                          rows(aQuant->val()),
                                          cols(aQuant->val()),
                                          cols(bQuant->val()),
                                          intgemm::callbacks::UnquantizeAndAddBiasAndWriteRelu(unquant_mult, /*bias=*/bias->val()->data(), /*output=*/out->val()->data()));
        } else {
          intgemm_<vtype>::width::Multiply(/*A=*/aQuant->val()->data<Integer>(),
                                          /*B=*/bQuant->val()->data<Integer>(),
                                          rows(aQuant->val()),
                                          cols(aQuant->val()),
                                          cols(bQuant->val()),
                                          intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, /*bias=*/bias->val()->data(), /*output=*/out->val()->data()));
        }
      }
    } else { // dispatch a multiply without bias addition i.e dot(...)
      if (relu) {
        intgemm_<vtype>::width::Multiply(/*A=*/aQuant->val()->data<Integer>(),
                                         /*B=*/bQuant->val()->data<Integer>(),
                                         rows(aQuant->val()),
                                         cols(aQuant->val()),
                                         cols(bQuant->val()),
                                         intgemm::callbacks::UnquantizeAndWriteRelu(unquant_mult, /*output=*/out->val()->data()));
      } else {
        intgemm_<vtype>::width::Multiply(/*A=*/aQuant->val()->data<Integer>(),
                                         /*B=*/bQuant->val()->data<Integer>(),
                                         rows(aQuant->val()),
                                         cols(aQuant->val()),
                                         cols(bQuant->val()),
                                         intgemm::callbacks::UnquantizeAndWrite(unquant_mult, /*output=*/out->val()->data()));
      }
    }
  };

  std::vector<Expr> children = {aQuant, bQuant};
  if(bias)
    children.push_back(bias);

  if (use_oneDNN /*&& !transB*/) { //Use DNNL if the inner dimension is not a multiple of 64. @TODO take care of the other case by using shifted-all
    return lambda(children, outShape, Type::float32, dnnlDotOrAffineNodeOp); // inference-only Lambda node
  } else {
    return lambda(children, outShape, Type::float32, dotOrAffineNodeOp); // inference-only Lambda node
  }
#else
  a, bQuant, bias, transA, scale;
  ABORT("You need to enable CPU compilation to use this feature. Use cmake .. -DCOMPILE_CPU=ON");
#endif
}

// Dispatch correct hardware-agnostic or hardware-specific matrix multiplies
static inline Expr affineOrDot(Expr a, Expr bQuant, Expr bias, bool transA, bool transB, float scale, bool relu) {
  Type bQuantElementType = bQuant->value_type();
  static const bool pass = cpu::integer::passOrAbort(bQuantElementType);
  pass; // We declare this variable as static so that passOrAbort is only ever run once during the initialization.
  switch(bQuantElementType) {
    //case Type::intgemm8 :  // The generic case selects CPU automatically, but we set all the types manually anyways.
    //  return cpu::integer::affineOrDotTyped<Type::intgemm8>(a, bQuant, bias, transA, transB, scale);    
    case Type::intgemm8ssse3 :
      return cpu::integer::affineOrDotTyped<Type::intgemm8ssse3>(a, bQuant, bias, transA, transB, scale, relu);
    case Type::intgemm8avx2 :
      return cpu::integer::affineOrDotTyped<Type::intgemm8avx2>(a, bQuant, bias, transA, transB, scale, relu);
    case Type::intgemm8avx512 :
      return cpu::integer::affineOrDotTyped<Type::intgemm8avx512>(a, bQuant, bias, transA, transB, scale, relu);
    case Type::intgemm8avx512vnni :
      return cpu::integer::affineOrDotTyped<Type::intgemm8avx512vnni>(a, bQuant, bias, transA, transB, scale, relu);
    //case Type::intgemm16 :  // The generic case selects CPU automatically, but we set all the types manually anyways.
    //  return cpu::integer::affineOrDotTyped<Type::intgemm16>(a, bQuant, bias, transA, transB, scale);
    case Type::intgemm16sse2 :
      return cpu::integer::affineOrDotTyped<Type::intgemm16sse2>(a, bQuant, bias, transA, transB, scale, relu);
    case Type::intgemm16avx2 :
      return cpu::integer::affineOrDotTyped<Type::intgemm16avx2>(a, bQuant, bias, transA, transB, scale, relu);
    case Type::intgemm16avx512 :
      return cpu::integer::affineOrDotTyped<Type::intgemm16avx512>(a, bQuant, bias, transA, transB, scale, relu);
    default:
      ABORT("Unsupported type {} for Intgemm type??", bQuantElementType);
  }
}

}  // namespace integer
}  // namespace cpu
}  // namespace marian
