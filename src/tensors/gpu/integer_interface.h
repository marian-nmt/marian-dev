#pragma once

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"
#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/integer_tools.h"

//TMP
#include "tensors/gpu/uint8tools.h"


namespace marian {

namespace gpu {

namespace integer {

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
        auto backend = std::static_pointer_cast<gpu::Backend>(child(0)->val()->getBackend());
        auto cublasHandle = backend->getCublasHandle();

        maxAbsQuantMult(cublasHandle, child(0)->val()->data(), child(0)->val()->shape().elements(), val_->data());
        if (!isA_) {
          //std::cerr << " HERE2 " << name() << std::endl; //NOT WORKING YET
        }
        //@TODO syncrhonise device to wait for kernel completion?
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
  PrepareNodeOp(Expr input, Expr quant_mult)
      : NaryNodeOp({input, quant_mult}, input->shape(), Type::int8) {

    set_name(input->name() + "_quantized8bit");
    if (isA_)
        setMemoize(false);
  }

  NodeOps forwardOps() override {
    return {NodeOp(
        CUDA_CHECK(cudaSetDevice((int)child(0)->val()->getDeviceId().no));
        const float * input = child(0)->val()->data();
        const float * quantMultAddr = child(1)->val()->data();

        if (!isA_) {
          //std::cerr << "Preparing: " << name() << std::endl;
        }
        quantize(input, val_->data<int8_t>(), rows(child(0)->val()), cols(child(0)->val()), quantMultAddr);

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

public:
  AffineNodeOp(Expr a, Expr b, Expr Bias, Expr quantMultA, Expr quantMultB, Expr ones, bool transA, bool transB, float scalar)
      : NaryNodeOp({a, b, Bias, quantMultA, quantMultB, ones}, 
        newShape(a, b, transA, transB), Type::float32), scalar_(scalar), transA_(transA), transB_(transB) {
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
      
      Tensor quantMultA = child(3)->val();
      Tensor quantMultB = child(4)->val();

      CUDA_CHECK(cudaSetDevice((int)C->getDeviceId().no));
      float alpha = scalar_;

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

      cutlass_igemm_dispatcher(transB_, transA_,
                        n,
                        m,
                        k,
                        alpha,
                        B->data<int8_t>(),
                        ldb,
                        A->data<int8_t>(),
                        lda,
                        0.0f,
                        C->data<int32_t>(),
                        ldc);

      //Now unquantize... Reusing the same Tensor
      int rowsC = C->shape().elements() / C->shape().back();
      int colsC = C->shape().back();
      dequantize(C->data<int32_t>(), C->data<float>(), rowsC, colsC, quantMultA->data<float>(), quantMultB->data<float>());
      //Synchronize
      val_->getBackend()->synchronize();

      //Perform bias addition, copied from the master implementation
      if (child(2)) {
        Tensor bias = child(2)->val();
        Tensor ones = child(5)->val();
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

public:
  DotNodeOp(Expr a, Expr b, Expr quantMultA, Expr quantMultB, bool transA, bool transB, float scalar)
      : NaryNodeOp({a, b, quantMultA, quantMultB}, 
        newShape(a, b, transA, transB), Type::float32), scalar_(scalar), transA_(transA), transB_(transB) {
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
      
      Tensor quantMultA = child(2)->val();
      Tensor quantMultB = child(3)->val();
      

      CUDA_CHECK(cudaSetDevice((int)C->getDeviceId().no));
      float alpha = scalar_;

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

      cutlass_igemm_dispatcher(transB_, transA_, //@TODO cutlass Check
                        n,
                        m,
                        k,
                        alpha,
                        B->data<int8_t>(),
                        ldb,
                        A->data<int8_t>(),
                        lda,
                        0.0f,
                        C->data<int32_t>(),
                        ldc);
      //Now unquantize... Reusing the same Tensor
      int rowsC = rows(C);
      int colsC = cols(C);

      dequantize(C->data<int32_t>(), C->data<float>(), rowsC, colsC, quantMultA->data<float>(), quantMultB->data<float>());
      //Synchronize
      val_->getBackend()->synchronize();
      //std::cerr << "Af4: " << C->data<float>()[0] << std::endl;

      /*
      //DEEEEBUG
      cublasOperation_t opA = transA_ ? CUBLAS_OP_T : CUBLAS_OP_N;
      cublasOperation_t opB = transB_ ? CUBLAS_OP_T : CUBLAS_OP_N;

      auto backend = std::static_pointer_cast<gpu::Backend>(C->getBackend());
      auto cublasHandle = backend->getCublasHandle();
      //auto computeCapability = backend->getCudaComputeCapability();
      gpuPrinterDispatch(C->data<float>(), 0);
      val_->getBackend()->synchronize();
      std::cerr << "True" << std::endl;
      //setTensorMode(cublasHandle);
      float beta = 0.0f;
      marian::hacky8bit::cublas8bitGemmmEx(cublasHandle,
        opB, 
        opA,
        n, m, k,
        &alpha,
        Bfloat->data<float>(), ldb,
        Afloat->data<float>(), lda,
        &beta,
        C->data<float>(), ldc,
        true);
        gpuPrinterDispatch(C->data<float>(), 0); */
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

static inline Expr affine(Expr A, Expr B, Expr bias, bool transA, bool transB, float scale, float clipValue=0 /*currently unused*/) {
  // Quantize to 8bits:
  std::string Bname = B->name();
  Expr AQuantMult = nullptr;
  static bool precomputedAlphas = B->graph()->getBackend()->isPrecomputedAlpha();
  if (precomputedAlphas) { //Shifting here maybe should check?
    AQuantMult = Expression<fetchAlphaFromModelNodeOp>(B);
  } else {
    AQuantMult = Expression<QuantMultNodeOp<Activation> >(A, Bname);
  }
  Expr BQuantMult = Expression<QuantMultNodeOp<Parameter> >(B, Bname);

  Expr AQuantized = Expression<PrepareNodeOp<Activation> >(A, AQuantMult);
  Expr BQuantized = Expression<PrepareNodeOp<Parameter> >(B, BQuantMult);


  //Perform multiplication KNOWING that A and B are swapped
  
  if (bias) {
    int rows = A->shape().elements() / A->shape()[-1];
    Expr ones = A->graph()->ones({ rows, 1 });
    return Expression<AffineNodeOp>(AQuantized, BQuantized, bias, AQuantMult, BQuantMult, ones, transA, transB, scale);
  } else {
    return Expression<DotNodeOp>(AQuantized, BQuantized, AQuantMult, BQuantMult, transA, transB, scale);
  }
}

static inline Expr dot(Expr a, Expr b, bool transA, bool transB, float scale) {
    // @TODO this will only work for k (cols(a) or rows(b)) % 4 == 0

  return affine(a, b, nullptr, transA, transB, scale, 0);
}

} // namespace integer
} // namespace gpu
} // namespace marian