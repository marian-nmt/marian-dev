#pragma once

#include "graph/node.h"
#include "3rd_party/intgemm/intgemm.h"

namespace marian {

namespace { //Convenient function to get rows and columns of a tensor, shadowed by namespace.
  inline int cols(Tensor& tensor) { return tensor->shape()[-1]; }
  inline int rows(Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }
}

namespace cpu {
namespace int8 {

//@TODO template it for A, B 8bit and 16bit
struct QuantMultNodeOp : public UnaryNodeOp {
  QuantMultNodeOp(Expr input) : UnaryNodeOp(input, Shape({1}), Type::float32) {}

  NodeOps forwardOps() override {
    return {NodeOp(
      *val_->data() = intgemm::MaxAbsolute(child(0)->val()->data(),
                         child(0)->val()->data() + child(0)->val()->shape().elements());
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "quantMult"; }
};

struct PrepareANodeOp : public NaryNodeOp {
float clipValue_;

  PrepareANodeOp(Expr input, Expr quant_mult, float clipValue)
      : NaryNodeOp({input, quant_mult}, input->shape(), Type::int8), clipValue_{clipValue} {
    ABORT_IF(children().size() != 2, "expected 2 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      intgemm::Int8::PrepareA(child(0)->val()->data(), /*input*/
                              val_->data<int8_t>(), /*output*/
                              *child(1)->val()->data(), /*Quant Mult*/
                              rows(child(0)->val()),
                              cols(child(0)->val()));
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "int8PrepareA"; }
};

struct PrepareBNodeOp : public NaryNodeOp {
float clipValue_;

  PrepareBNodeOp(Expr input, Expr quant_mult, float clipValue)
      : NaryNodeOp({input, quant_mult}, input->shape(), Type::int8), clipValue_{clipValue} {
    ABORT_IF(children().size() != 2, "expected 2 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      intgemm::Int8::PrepareB(child(0)->val()->data(), /*input*/
                              val_->data<int8_t>(), /*output*/
                              *child(1)->val()->data(), /*Quant Mult*/
                              rows(child(0)->val()),
                              cols(child(0)->val()));
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "int8PrepareB"; }
};

class DotNodeOp : public NaryNodeOp {
private:
float scalar_;

public:
  DotNodeOp(Expr a, Expr b, Expr aQuantMult, Expr bQuantMult, float scalar)
      : NaryNodeOp({a, b}, newShape(a, b), Type::float32), scalar_(scalar) {

        // Check if arguments are not null
        ABORT_IF(child(0) == nullptr, "A cannot be null");
        ABORT_IF(child(1) == nullptr, "B cannot be null");
        ABORT_IF(child(2) == nullptr, "QuantMult of A cannot be null");
        ABORT_IF(child(3) == nullptr, "QuantMult of B cannot be null");
      }

  Shape newShape(Expr a, Expr b) {
    auto shapeA = a->shape();
    auto shapeB = b->shape();

    // Computing A * B^T
    shapeB.set(-2, b->shape()[-1]);
    shapeB.set(-1, b->shape()[-2]);

    Shape outShape = shapeA;
    outShape.set(-1, shapeB[-1]);
    ABORT_IF(shapeA[-1] != shapeB[-2],
             "matrix product requires dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
          float unquant_mult = scalar_ * (1 / (*child(2)->val()->data() * *child(3)->val()->data()));
          intgemm::Int8::Multiply(child(0)->val()->data<int8_t>(), /*A*/
                                  child(1)->val()->data<int8_t>(), /*B*/
                                   rows(child(0)->val()),
                                   rows(child(1)->val()),
                                   cols(child(1)->val()),
                                   intgemm::callbacks::UnquantizeAndWrite(unquant_mult, val_->data()));
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "dotInt8"; }
};

class AffineNodeOp : public NaryNodeOp {
private:
  float scalar_;

public:
  AffineNodeOp(Expr a, Expr b, Expr aQuantMult, Expr bQuantMult, Expr Bias, float scalar)
      : NaryNodeOp({a, b, aQuantMult, bQuantMult, Bias}, newShape(a, b), Type::float32), scalar_(scalar) {

        // Check if arguments are not null
        ABORT_IF(child(0) == nullptr, "A cannot be null");
        ABORT_IF(child(1) == nullptr, "B cannot be null");
        ABORT_IF(child(2) == nullptr, "QuantMult of A cannot be null");
        ABORT_IF(child(3) == nullptr, "QuantMult of B cannot be null");
        ABORT_IF(child(4) == nullptr, "Bias cannot be null");
      }

  Shape newShape(Expr a, Expr b) {
    auto shapeA = a->shape();
    auto shapeB = b->shape();

    // Computing A * B^T
    shapeB.set(-2, b->shape()[-1]);
    shapeB.set(-1, b->shape()[-2]);

    Shape outShape = shapeA;
    outShape.set(-1, shapeB[-1]);
    ABORT_IF(shapeA[-1] != shapeB[-2],
             "matrix product requires dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
          float unquant_mult = scalar_ * (1 / (*child(2)->val()->data() * *child(3)->val()->data()));
          intgemm::Int8::Multiply(child(0)->val()->data<int8_t>(), /*A*/
                                  child(1)->val()->data<int8_t>(), /*B*/
                                   rows(child(0)->val()),
                                   rows(child(1)->val()),
                                   cols(child(1)->val()),                                          /*child(4) is bias*/
                                   intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, child(4)->val()->data(), val_->data()));
    )};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "affineInt8"; }
};

static inline Expr dot(Expr a, Expr b, Expr aQuantMult, Expr bQuantMult, float scalar) {
  return Expression<cpu::int8::DotNodeOp>(a, b, aQuantMult, bQuantMult, scalar);
}

static inline Expr affine(Expr a, Expr b, Expr aQuantMult, Expr bQuantMult, Expr c, float scalar) {
  return Expression<cpu::int8::AffineNodeOp>(a, b, aQuantMult, bQuantMult, c, scalar);
}

static inline Expr quantMult(Expr a) {
  return Expression<cpu::int8::QuantMultNodeOp>(a);
}

static inline Expr prepareA(Expr a, Expr quantMult, float clipValue) {
  return Expression<cpu::int8::PrepareANodeOp>(a, quantMult, clipValue);
}

static inline Expr prepareB(Expr b, Expr quantMult, float clipValue) {
  return Expression<cpu::int8::PrepareBNodeOp>(b, quantMult, clipValue);
}

}  // namespace int8
}  // namespace cpu
}  // namespace marian
