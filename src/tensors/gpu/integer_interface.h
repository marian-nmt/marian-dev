#pragma once

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"
#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/integer_tools.h"


namespace marian {

namespace gpu {

namespace integer {

struct QuantMultNodeOp : public UnaryNodeOp {
  bool isA_;
  QuantMultNodeOp(Expr input, bool isA, std::string& bname) : UnaryNodeOp(input, Shape({1}), Type::float32), isA_(isA) {
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

  size_t hash() override {
    return std::hash<std::string>{}(name());
  }

};

struct PrepareNodeOp : public NaryNodeOp {
  PrepareNodeOp(Expr input, Expr quant_mult, bool isA)
      : NaryNodeOp({input, quant_mult}, input->shape(), Type::int8) {

    set_name(input->name() + "_quantized8bit");
    if (isA)
        setMemoize(false);
  }

  NodeOps forwardOps() override {
    return {NodeOp(
        CUDA_CHECK(cudaSetDevice((int)child(0)->val()->getDeviceId().no));
        const float * input = child(0)->val()->data();
        const float * quantMultAddr = child(1)->val()->data();

        quantize(input, val_->data<int8_t>(), rows(child(0)->val()), cols(child(0)->val()), quantMultAddr);

    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "integer8Quantized"; }
};

//static inline Expr dot(Expr a, Expr b, bool transA, bool transB, float scale) {
    // @TODO this will only work for k (cols(a) or rows(b)) % 4 == 0

  //return affine<vtype>(a, b, nullptr, transA, transB, scale, 0 /*currently unused clipValue*/, shiftedBias);
//}

} // namespace integer
} // namespace gpu
} // namespace marian