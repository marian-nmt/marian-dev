#pragma once

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"
#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/integer_tools.h"

//TMP
//#include "tensors/gpu/uint8tools.h"


namespace marian {

namespace gpu {

namespace integer {

class PreparedContainerNodeOp : public NaryNodeOp {
  public:
  MemoryPiece::PtrType gpuQuantMult;
  PreparedContainerNodeOp(Expr input, Expr quantMult)
    : NaryNodeOp({input, quantMult}, input->shape(), Type::int8) {
      set_name("none");
    }
  NodeOps forwardOps() override {
    return {NodeOp(
      CUDA_CHECK(cudaSetDevice((int)child(0)->val()->getDeviceId().no));
      // Remember the quantised node here
      memCpyDevice(val_->data(), child(0)->val()->data<float>(), child(0)->shape().elements());

      // Put the quantization multiplier on the node
      float * quantMultHolder = child(1)->val()->data<float>();
      gpuQuantMult = graph()->allocator()->alloc<float>(1);
      memCpyDevice(gpuQuantMult->data<float>(), quantMultHolder, 1);
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "PreparedContainer"; }

  ~PreparedContainerNodeOp() {
    graph()->allocator()->free(gpuQuantMult);
  }
};

static const constexpr bool Activation = true;
static const constexpr bool Parameter = false;

template<bool isA_>
struct QuantMultNodeOp : public UnaryNodeOp {
  //bool isA_;
  QuantMultNodeOp(Expr input, std::string& bname) : UnaryNodeOp(input, Shape({1}), Type::float32) {
    if (isA_) {
      setMemoize(false);
      set_name(bname + "_QuantMultA");
    } else {
      set_name(input->name() + "_QuantMultB");
    }
  }

  NodeOps forwardOps() override {
    return {NodeOp(
        CUDA_CHECK(cudaSetDevice((int)child(0)->val()->getDeviceId().no));
        if (child(0)->value_type() == Type::int8) {
          auto  actualExp = std::static_pointer_cast<PreparedContainerNodeOp>(child(0));
          memCpyDevice(val_->data(), actualExp->gpuQuantMult->data<float>(), 1);
        } else {
          auto backend = std::static_pointer_cast<gpu::Backend>(child(0)->val()->getBackend());
          auto cublasHandle = backend->getCublasHandle();

          maxAbsQuantMult(cublasHandle, child(0)->val()->data(), child(0)->val()->shape().elements(), val_->data());
        }
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override {
    if (isA_)
      return "integer8QuantMultA";
    else
      return "integer8QuantMultB";
  }

  /* This is not necessary, but let's leave it just in case
  bool equal(Expr node) override {
    if (isA_) {
      return UnaryNodeOp::equal(node);
    }
    if(hash() == node->hash()) return true;
    return false;
  } */

  //size_t hash() override {
  //  return std::hash<std::string>{}(name());
  //}

};

template<bool isA_>
struct PrepareNodeOp : public NaryNodeOp {
  bool useTensorcores_;
  bool transpose_;
  PrepareNodeOp(Expr input, Expr quant_mult, bool useTensorcores=false, bool transpose=false)
      : NaryNodeOp({input, quant_mult}, input->shape(), Type::int8), useTensorcores_(useTensorcores), transpose_(transpose) {

    set_name(input->name() + "_quantized8bit");
    if (isA_) {
        setMemoize(false);
        useTensorcores_ = false; // We only need the special case for the Parameters, as they need to be quantized AND row-major'd
    }
    if (useTensorcores) {
      set_name(input->name() + "_RowM");
    }
  }

  NodeOps forwardOps() override {
    return {NodeOp(
        CUDA_CHECK(cudaSetDevice((int)child(0)->val()->getDeviceId().no));
        if (child(0)->value_type() == Type::int8) {
          memCpyDevice(val_->data<int8_t>(), child(0)->val()->data<int8_t>(), child(0)->shape().elements());
        } else {
          const float * input = child(0)->val()->data();
          const float * quantMultAddr = child(1)->val()->data();

          if (useTensorcores_ && !isA_ && !transpose_) { /*if the matrix is to be transposed, we don't actually need to do that as we read it in as row major*/
                                                                  /*Cols and rows are inverted, cause cols() will give you the length of the row, which is what we are after*/
            quantizeToRowMajorWrapper(input, val_->data<int8_t>(), cols(child(0)->val()), rows(child(0)->val()), quantMultAddr);
          } else {
            quantize(input, val_->data<int8_t>(), cols(child(0)->val()), rows(child(0)->val()), quantMultAddr);
          }
        }

    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "integer8Quantized"; }
};

class AffineNodeOp : public NaryNodeOp {
private:
  float scalar_;
  bool transA_;
  bool transB_;
  bool useTensorcores_;

public:
  AffineNodeOp(Expr a, Expr b, Expr Bias, Expr deQuantMult, Expr ones, bool transA, bool transB, float scalar, bool useTensorcores)
      : NaryNodeOp({a, b, Bias, deQuantMult, ones},
        newShape(a, b, transA, transB), Type::float32), scalar_(scalar), transA_(transA), transB_(transB), useTensorcores_(useTensorcores) {
        setMemoize(false); // AFAIK affine is never called with the same matrices
      }

  /*Without ones, for fused*/
  AffineNodeOp(Expr a, Expr b, Expr Bias, Expr deQuantMult, bool transA, bool transB, float scalar, bool useTensorcores)
      : NaryNodeOp({a, b, Bias, deQuantMult},
        newShape(a, b, transA, transB), Type::float32), scalar_(scalar), transA_(transA), transB_(transB), useTensorcores_(useTensorcores) {
        setMemoize(false); // AFAIK affine is never called with the same matrices
      }

  Shape newShape(Expr a, Expr b, bool transA, bool transB) {
    auto shapeA = a->shape();
    if(transA) {
      shapeA.set(shapeA.size() - 2, a->shape()[shapeA.size() - 1]);
      shapeA.set(shapeA.size() - 1, a->shape()[shapeA.size() - 2]);
    }

    auto shapeB = b->shape();
    if(transB) {
      shapeB.set(shapeB.size() - 2, b->shape()[shapeB.size() - 1]);
      shapeB.set(shapeB.size() - 1, b->shape()[shapeB.size() - 2]);
    }

    Shape outShape = shapeA;
    outShape.set(outShape.size() - 1, shapeB[shapeB.size() - 1]);
    ABORT_IF(shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2],
             "Matrix product requires inner dimensions to match in {}{} * {}{}", std::string(shapeA), transA, std::string(shapeB), transB);
    return outShape;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      //A and B and swapped here, so are LDA, LDB and transA and transB
      // Perform LDA, LDB, LDC
      Tensor A = child(0)->val();
      Tensor B = child(1)->val();
      Tensor C = val_;
      Tensor bias = child(2)->val();
      
      Tensor deQuantMult = child(3)->val();

      CUDA_CHECK(cudaSetDevice((int)C->getDeviceId().no));
      //float alpha = scalar_;

      int m = A->shape().elements() / A->shape().back();
      int k = A->shape().back();
      if(transA_)
        std::swap(m, k);

      int l = B->shape().elements() / B->shape().back();
      int n = B->shape().back();
      if(transB_)
        std::swap(l, n);

      int lda = A->shape().back();
      int ldb = B->shape().back();
      int ldc = B->shape().back();

      if(transB_)
        ldc = B->shape().elements() / B->shape().back();

      float * gpuOne = std::static_pointer_cast<gpu::Backend>(child(0)->val()->getBackend())->getOneGPU();

      if (useTensorcores_ && !transB_) { //If B is to be transposed we don't need to reverse the dimensions
          ldb = B->shape().elements() / B->shape().back();
      }
      float * beta = nullptr;
      static bool fused = child(0)->graph()->getBackend()->isFused();
      if (fused)
        beta = gpuOne;

      cutlass_igemm_dispatcher(transB_, transA_,
                          n,
                          m,
                          k,
                          deQuantMult->data<float>(),
                          B->data<int8_t>(),
                          ldb,
                          A->data<int8_t>(),
                          lda,
                          beta,
                          C->data<int32_t>(), /*We perform a cast depending on whether its fused or not down the line*/
                          ldc,
                          useTensorcores_,
                          fused,
                          bias->data<float>()); /* Fused Bias GEMM. Only used if beta is not a nullptr and is 1 */

      /*If we are using the unfused codepath, we need to manually unquantize and perform a bias addition*/
      if (!fused) {
        int rowsC = C->shape().elements() / C->shape().back();
        int colsC = C->shape().back();
        dequantize(C->data<int32_t>(), C->data<float>(), rowsC, colsC, deQuantMult->data<float>());
        //Synchronize
        val_->getBackend()->synchronize();

        //Perform bias addition, copied from the master implementation
        Tensor ones = child(4)->val();
        marian::gpu::Prod(val_, ones, bias, false, false, 1.f, 1.f);

      }
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "int8Affine"; }
};

class DotNodeOp : public NaryNodeOp {
private:
  float scalar_;
  bool transA_;
  bool transB_;
  bool useTensorcores_;

public:
  DotNodeOp(Expr a, Expr b, Expr deQuantMult, bool transA, bool transB, float scalar, bool useTensorcores)
      : NaryNodeOp({a, b, deQuantMult},
        newShape(a, b, transA, transB), Type::float32), scalar_(scalar), transA_(transA), transB_(transB), useTensorcores_(useTensorcores) {
        setMemoize(false); // AFAIK affine is never called with the same matrices
      }

  Shape newShape(Expr a, Expr b, bool transA, bool transB) {
    auto shapeA = a->shape();
    if(transA) {
      shapeA.set(shapeA.size() - 2, a->shape()[shapeA.size() - 1]);
      shapeA.set(shapeA.size() - 1, a->shape()[shapeA.size() - 2]);
    }

    auto shapeB = b->shape();
    if(transB) {
      shapeB.set(shapeB.size() - 2, b->shape()[shapeB.size() - 1]);
      shapeB.set(shapeB.size() - 1, b->shape()[shapeB.size() - 2]);
    }

    Shape outShape = shapeA;
    outShape.set(outShape.size() - 1, shapeB[shapeB.size() - 1]);
    ABORT_IF(shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2],
             "Matrix product requires inner dimensions to match in {}{} * {}{}", std::string(shapeA), transA, std::string(shapeB), transB);
    return outShape;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      //A and B and swapped here, so are LDA, LDB and transA and transB
      // Perform LDA, LDB, LDC
      Tensor A = child(0)->val();
      Tensor B = child(1)->val();
      Tensor C = val_;
      
      Tensor deQuantMult = child(2)->val();

      CUDA_CHECK(cudaSetDevice((int)C->getDeviceId().no));

      int m = A->shape().elements() / A->shape().back();
      int k = A->shape().back();
      if(transA_)
        std::swap(m, k);

      int l = B->shape().elements() / B->shape().back();
      int n = B->shape().back();
      if(transB_)
        std::swap(l, n);

      int lda = A->shape().back();
      int ldb = B->shape().back();
      int ldc = B->shape().back();

      if(transB_)
        ldc = B->shape().elements() / B->shape().back();

      float * gpuZero = std::static_pointer_cast<gpu::Backend>(child(0)->val()->getBackend())->getZeroGPU();

      if(useTensorcores_ && !transB_) {
        ldb = B->shape().elements() / B->shape().back();
      }
      float * beta = nullptr;
      static bool fused = child(0)->graph()->getBackend()->isFused();
      if (fused)
        beta = gpuZero;

      cutlass_igemm_dispatcher(transB_, transA_, //@TODO cutlass Check
                        n,
                        m,
                        k,
                        deQuantMult->data<float>(),
                        B->data<int8_t>(),
                        ldb,
                        A->data<int8_t>(),
                        lda,
                        beta,
                        C->data<int32_t>(),
                        ldc,
                        useTensorcores_,
                        fused,
                        nullptr);

      // If we are using the non-fused codepath, we need to unquantize after the fact
      if (!fused) {
        int rowsC = rows(C);
        int colsC = cols(C);
        dequantize(C->data<int32_t>(), C->data<float>(), rowsC, colsC, deQuantMult->data<float>());
        // Synchronize
        val_->getBackend()->synchronize();
      }
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "int8Affine"; }
};

class fetchAlphaFromModelNodeOp : public UnaryNodeOp {
public:
  fetchAlphaFromModelNodeOp(Expr b)
      : UnaryNodeOp(b, Shape({1}), Type::float32) {

    std::string bname = b->name();
    std::string aQuantKey = b->name() + "_QuantMultA";
    //Very Hacky Bit. Unnamed matrix is not part of the F0 parameter namespace
    if (aQuantKey.at(0) != 'F') {
      aQuantKey = "F0::" + aQuantKey;
    }
    set_name(aQuantKey);
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      auto map = child(0)->graph()->params()->getMap();
      const auto mapiter = map.find(name());
      if (mapiter != map.end()) {
        val_ = mapiter->second->val();
      } else {
        ABORT("We did not find an alpha in the model named: {}.", name());
      }
    )};
  }

  bool equal(Expr node) override {
    if(hash() == node->hash()) return true;
    return false;
  }

  size_t hash() override {
    return std::hash<std::string>{}(name());
  }

  const std::string type() override { return "alphaNodeOp"; }
};

class DequantMultNodeOp : public NaryNodeOp {
public:
  DequantMultNodeOp(Expr aQuantMult, Expr bQuantMult)
      : NaryNodeOp({aQuantMult, bQuantMult}, Shape({1}), Type::float32) {
    set_name(aQuantMult->name() + "_" + bQuantMult->name() + "dequantMult");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      float * aQuantMult = child(0)->val()->data<float>();
      float * bQuantMult = child(1)->val()->data<float>();
      getDequantMultWrapper(val_->data<float>(), aQuantMult, bQuantMult);
    )};
  }

  const std::string type() override { return "dequantMultNodeOp"; }
};

static inline Expr affine(Expr A, Expr B, Expr bias, bool transA, bool transB, float scale, float clipValue=0 /*currently unused*/) {
  bool useTensorcores = A->graph()->getBackend()->useTensorCoreGemm();
  ABORT_IF(useTensorcores && transA, "Using tensorcores and transposing the activations is not yet supported!");
  // Quantize to 8bits:
  std::string Bname = B->name();
  Expr AQuantMult = nullptr;
  static bool precomputedAlphas = B->graph()->getBackend()->isPrecomputedAlpha();
  if (precomputedAlphas) {
    AQuantMult = Expression<fetchAlphaFromModelNodeOp>(B);
  } else {
    AQuantMult = Expression<QuantMultNodeOp<Activation> >(A, Bname);
  }
  Expr BQuantMult = Expression<QuantMultNodeOp<Parameter> >(B, Bname);

  Expr AQuantized = Expression<PrepareNodeOp<Activation> >(A, AQuantMult);
  Expr BQuantized = Expression<PrepareNodeOp<Parameter> >(B, BQuantMult, useTensorcores, transB);

  Expr deQuantMult = Expression<DequantMultNodeOp>(AQuantMult, BQuantMult);

  //Perform multiplication KNOWING that A and B are swapped
  static bool fused = A->graph()->getBackend()->isFused();
  if (bias) {                                       // @TODO move it onto BQuantMult or PrepareB, because this is really slow.
    if (fused) {
      return Expression<AffineNodeOp>(AQuantized, BQuantized, bias, deQuantMult, transA, transB, scale, useTensorcores);
    } else {
      int rows = A->shape().elements() / A->shape()[-1]; /*For the unfused codepath, we need this to perform postprocess bias addition*/
      Expr ones = A->graph()->ones({ rows, 1 });
      return Expression<AffineNodeOp>(AQuantized, BQuantized, bias, deQuantMult, ones, transA, transB, scale, useTensorcores);
    }
  } else {
    return Expression<DotNodeOp>(AQuantized, BQuantized, deQuantMult, transA, transB, scale, useTensorcores);
  }
}

static inline Expr dot(Expr a, Expr b, bool transA, bool transB, float scale) {
    // @TODO this will only work for k (cols(a) or rows(b)) % 4 == 0

  return gpu::integer::affine(a, b, nullptr, transA, transB, scale, 0);
}

} // namespace integer
} // namespace gpu
} // namespace marian