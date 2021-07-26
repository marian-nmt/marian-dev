#pragma once

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"
#include "graph/node.h"
#include "graph/node_operators_unary.h"
#ifdef CUDA_FOUND
#include "tensors/gpu/backend.h"
#endif
#include "tensors/gpu/integer_tools.h"
#include <fstream>
#include <string>


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
#ifdef CUDA_FOUND
      CUDA_CHECK(cudaSetDevice((int)child(0)->val()->getDeviceId().no));
      // Remember the quantised node here
      memCpyDevice(val_->data(), child(0)->val()->data<float>(), child(0)->shape().elements());

      // Put the quantization multiplier on the node
      float * quantMultHolder = child(1)->val()->data<float>();
      gpuQuantMult = graph()->allocator()->alloc<float>(1);
      memCpyDevice(gpuQuantMult->data<float>(), quantMultHolder, 1);
#endif
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
#ifdef CUDA_FOUND
        CUDA_CHECK(cudaSetDevice((int)child(0)->val()->getDeviceId().no));
        if (child(0)->value_type() == Type::int8) {
          auto  actualExp = std::static_pointer_cast<PreparedContainerNodeOp>(child(0));
          memCpyDevice(val_->data<float>(), reinterpret_cast<float *>(actualExp->gpuQuantMult->data()), 1); // data<float>() fails on GPU only build on some GCC
        } else {
          auto backend = std::static_pointer_cast<gpu::Backend>(child(0)->val()->getBackend());
          auto cublasHandle = backend->getCublasHandle();
          if (child(0)->value_type() == Type::float32) {
            maxAbsQuantMult(cublasHandle,
                            child(0)->val()->data(),
                            child(0)->val()->shape().elements(),
                            val_->data<float>());
          } else if (child(0)->value_type() == Type::float16){
//            maxAbsQuantMultFP16(reinterpret_cast<half *>(child(0)->val()->data()),
//                                child(0)->val()->shape().elements(),
//                                val_->data<float>());

            //@TODO this method is slower than maxAbsQuantMultFP16, but maxAbsQuantMultFP16 need debugging
            maxAbsQuantMultFP16Ref(cublasHandle,
                                    reinterpret_cast<half *>(child(0)->val()->data()),
                                    child(0)->val()->shape().elements(),
                                    val_->data<float>());
          }

          // DumpQuantmults if necessary
          // FP16 is not supported here
          if (child(0)->graph()->getBackend()->DumpQuantMult()) {
            ABORT_IF(child(0)->value_type() == Type::float16, "DumpQuantMult is not supported for FP16");
            MeanStd meanstd = getMeanStd(child(0)->val()->data(), child(0)->val()->shape().elements());
            std::cerr << "Name: " << name() << " MeanAbs: " << meanstd.absMean << " stddevAbs: " << meanstd.absStddev << " Mean: " << meanstd.mean << " stddev: "
            << meanstd.stddev << " MaxAbs: " << getMaxAbs(cublasHandle, child(0)->val()->data(), child(0)->val()->shape().elements()) << std::endl;
          }
        }
#endif
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
#ifdef CUDA_FOUND
        CUDA_CHECK(cudaSetDevice((int)child(0)->val()->getDeviceId().no));
        if (child(0)->value_type() == Type::int8) {
          memCpyDevice(val_->data<int8_t>(), reinterpret_cast<int8_t *>(child(0)->val()->data()), child(0)->shape().elements()); // Using data<int8_t>() fails on some GCC
        } else {
            const float *quantMultAddr = reinterpret_cast<float *>(child(1)->val()->data());
            if(child(0)->value_type() == Type::float32) {
              const float *input = child(0)->val()->data();
              if(useTensorcores_ && !isA_
                 && !transpose_) { /*if the matrix is to be transposed, we don't actually need to do
                                      that as we read it in as row major*/
                /*Cols and rows are inverted, cause cols() will give you the length of the row,
                 * which is what we are after*/
                quantizeToRowMajorWrapper(input,
                                          val_->data<int8_t>(),
                                          cols(child(0)->val()),
                                          rows(child(0)->val()),
                                          quantMultAddr);
              }
              else {
                quantize(input,
                         val_->data<int8_t>(),
                         cols(child(0)->val()),
                         rows(child(0)->val()),
                         quantMultAddr);
              }
            } else if(child(0)->value_type() == Type::float16) {
              const half *input = reinterpret_cast<half *>(child(0)->val()->data());
              if(useTensorcores_ && !isA_
                 && !transpose_) { /*if the matrix is to be transposed, we don't actually need to do
                                       that as we read it in as row major*/
                /*Cols and rows are inverted, cause cols() will give you the length of the row,
                 * which is what we are after*/
                quantizeToRowMajorWrapperFP16(input,
                                          val_->data<int8_t>(),
                                          cols(child(0)->val()),
                                          rows(child(0)->val()),
                                          quantMultAddr);
              } else {
                quantizeFP16(input,
                         val_->data<int8_t>(),
                         cols(child(0)->val()),
                         rows(child(0)->val()),
                         quantMultAddr);
              }
            }
          }
#endif
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
  bool doRelu_;

public:
  AffineNodeOp(Expr a, Expr b, Expr Bias, Expr deQuantMult, Expr ones, bool transA, bool transB, float scalar, bool useTensorcores, bool doRelu)
      : NaryNodeOp({a, b, Bias, deQuantMult, ones},
        newShape(a, b, transA, transB), Bias->value_type()), scalar_(scalar), transA_(transA), transB_(transB), useTensorcores_(useTensorcores), doRelu_(doRelu) {
        setMemoize(false); // AFAIK affine is never called with the same matrices
      }

  /*Without ones, for fused*/
  AffineNodeOp(Expr a, Expr b, Expr Bias, Expr deQuantMult, bool transA, bool transB, float scalar, bool useTensorcores, bool doRelu)
      : NaryNodeOp({a, b, Bias, deQuantMult},
        newShape(a, b, transA, transB), Bias->value_type()), scalar_(scalar), transA_(transA), transB_(transB), useTensorcores_(useTensorcores), doRelu_(doRelu) {
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
#ifdef CUDA_FOUND
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
      if (fused) {
        beta = gpuOne;
      } else if (!fused && doRelu_) {
        ABORT("We can't do fused relu in unfused GEMM."); // Unfused GEMM can't do RELU. ensure that we don't
      }
      if(C->type() == Type::float32) {
            cutlass_igemm_dispatcher(
                transB_,
                transA_,
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
                bias->data<float>(), /* Fused Bias GEMM. Only used if beta is not a nullptr and is 1 */
                doRelu_);           /*Perform fused relu. Only available if fused is also true, otherwise we will produce wrong results.*/

      } else if(C->type() == Type::float16) {
            cutlass_igemm_dispatcher_half(
                transB_,
                transA_,
                n,
                m,
                k,
                deQuantMult->data<float>(),
                B->data<int8_t>(),
                ldb,
                A->data<int8_t>(),
                lda,
                beta,
                C->data<half>(), /*We perform a cast depending on whether its fused or not down the line*/
                ldc,
                useTensorcores_,
                fused,
                bias->data<half>(), /* Fused Bias GEMM. Only used if beta is not a nullptr and is 1 */
                doRelu_);           /*Perform fused relu. Only available if fused is also true, otherwise we will produce wrong results.*/
          }
          /*If we are using the unfused codepath, we need to manually unquantize and perform a bias addition*/
      if (!fused) {
        int rowsC = C->shape().elements() / C->shape().back();
        int colsC = C->shape().back();
        if(C->type() == Type::float32){
          dequantize(
              C->data<int32_t>(), C->data<float>(), rowsC, colsC, deQuantMult->data<float>());
        } else if(C->type() == Type::float16) {
          dequantizeFP16(
              C->data<int32_t>(), C->data<half>(), rowsC, colsC, deQuantMult->data<float>());
        }
        //Synchronize
        val_->getBackend()->synchronize();

        //Perform bias addition, copied from the master implementation
        Tensor ones = child(4)->val();
        marian::gpu::Prod(val_, ones, bias, false, false, 1.f, 1.f);

      }
#endif
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "int8Affine"; }
};
template<Type outputType>
class DotNodeOp : public NaryNodeOp {
private:
  float scalar_;
  bool transA_;
  bool transB_;
  bool useTensorcores_;
  bool doRelu_;

public:
  DotNodeOp(Expr a, Expr b, Expr deQuantMult, bool transA, bool transB, float scalar, bool useTensorcores, bool doRelu)
      : NaryNodeOp({a, b, deQuantMult},
        newShape(a, b, transA, transB), outputType), scalar_(scalar), transA_(transA), transB_(transB), useTensorcores_(useTensorcores), doRelu_(doRelu) {
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
#ifdef CUDA_FOUND
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
      if (fused) {
        beta = gpuZero;
      } else if (!fused && doRelu_) {
        ABORT("We can't do fused relu in unfused GEMM."); // Unfused GEMM can't do RELU. ensure that we don't
      }
      if(C->type() == Type::float32) {
            cutlass_igemm_dispatcher(transB_,
                                     transA_,  //@TODO cutlass Check
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
                                     nullptr,
                                     doRelu_); /*Perform fused relu. Only available if fused is also
                                                  true, otherwise we will produce wrong results.*/
      } else if(C->type() == Type::float16) {
            cutlass_igemm_dispatcher_half(
                transB_,
                transA_,  //@TODO cutlass Check
                n,
                m,
                k,
                deQuantMult->data<float>(),
                B->data<int8_t>(),
                ldb,
                A->data<int8_t>(),
                lda,
                beta,
                C->data<half>(),
                ldc,
                useTensorcores_,
                fused,
                nullptr,
                doRelu_); /*Perform fused relu. Only available if fused is also
                                                  true, otherwise we will produce wrong results.*/
      }
      // If we are using the non-fused codepath, we need to unquantize after the fact
      if (!fused) {
        int rowsC = rows(C);
        int colsC = cols(C);
        if(C->type() == Type::float32) {
          dequantize(
              C->data<int32_t>(), C->data<float>(), rowsC, colsC, deQuantMult->data<float>());
        } else if(C->type() == Type::float16) {
          dequantizeFP16(
              C->data<int32_t>(), C->data<half>(), rowsC, colsC, deQuantMult->data<float>());
        }
        // Synchronize
        val_->getBackend()->synchronize();
      }
#endif
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
        //val_ = mapiter->second->val();
        if(child(0)->value_type() == Type::float32) {
          memCpyDevice(
              val_->data<float>(), reinterpret_cast<float *>(mapiter->second->val()->data()), 1);
        } else if(child(0)->value_type() == Type::float16) {
          memCpyDeviceFP16(
              val_->data<float>(), reinterpret_cast<half *>(mapiter->second->val()->data()));
        }
      } else {
        ABORT("We did not find an alpha in the model named: {}.", name());
      }
    )};
  }
  // Not necessary since we're hashing the expression
  //bool equal(Expr node) override {
  //  if(hash() == node->hash()) return true;
  //  return false;
  //}

  //size_t hash() override {
  //  return std::hash<std::string>{}(name() + type());
  //}

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

static inline Expr affine(Expr A, Expr B, Expr bias, bool transA, bool transB, float scale, float clipValue=0 /*currently unused*/, bool doRelu=false) {
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
  Expr ret = nullptr;
  if (bias) {                                       // @TODO move it onto BQuantMult or PrepareB, because this is really slow.
    if (fused) {
      ret = Expression<AffineNodeOp>(AQuantized, BQuantized, bias, deQuantMult, transA, transB, scale, useTensorcores, doRelu);
    } else {
      int rows = A->shape().elements() / A->shape()[-1]; /*For the unfused codepath, we need this to perform postprocess bias addition*/
      Expr ones = A->graph()->ones({ rows, 1 });
      ret = Expression<AffineNodeOp>(AQuantized, BQuantized, bias, deQuantMult, ones, transA, transB, scale, useTensorcores, false /*Unfused can't do Relu*/);
    }
  } else {
    if (fused) {
      if(A->value_type() == Type::float32)
        ret = Expression<DotNodeOp<Type::float32>>(AQuantized, BQuantized, deQuantMult, transA, transB, scale, useTensorcores, doRelu);
      else if(A->value_type() == Type::float16)
        ret = Expression<DotNodeOp<Type::float16>>(AQuantized, BQuantized, deQuantMult, transA, transB, scale, useTensorcores, doRelu);
    } else {
      if(A->value_type() == Type::float32)
        ret = Expression<DotNodeOp<Type::float32>>(AQuantized, BQuantized, deQuantMult, transA, transB, scale, useTensorcores, false /*Unfused can't do Relu*/);
      else if(A->value_type() == Type::float16)
        ret = Expression<DotNodeOp<Type::float16>>(AQuantized, BQuantized, deQuantMult, transA, transB, scale, useTensorcores, false /*Unfused can't do Relu*/);
    }
  }
  if (!fused && doRelu) { //We can't do RELU if we are not fused, so we need to explicitly perform it as a postprocessing step
    return relu(ret);
  } else {
    return ret;
  }
}

static inline Expr dot(Expr a, Expr b, bool transA, bool transB, float scale, bool doRelu=false) {
  return gpu::integer::affine(a, b, nullptr, transA, transB, scale, 0, doRelu);
}

} // namespace integer
} // namespace gpu
} // namespace marian