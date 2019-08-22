#pragma once

#include "3rd_party/intgemm/intgemm.h"
#include "common/hash.h"
#include "graph/node.h"
#include "tensors/cpu/bias.h"
#include "3rd_party/intgemm/aligned.h"
#include <math.h>

namespace marian {
namespace cpu {
namespace integer {

template <Type Type_>
using EnableIfTypeIsSupported = typename std::enable_if<
  std::integral_constant<bool,
    (Type_ == Type::int8) ||
    (Type_ == Type::int16)
  >::value>::type;

namespace { // anonymous namespace

inline int cols(Tensor& tensor) { return tensor->shape()[-1]; }
inline int rows(Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }

template <Type Type_> struct backend_s;
template <> struct backend_s<Type::int8> { using backend = intgemm::Int8; };
template <> struct backend_s<Type::int16> { using backend = intgemm::Int16; };
template <Type Type_> using backend = typename backend_s<Type_>::backend;

} // anonymous namespace

class OnlyForInferenceNodeOp : public NaryNodeOp {
public:
  OnlyForInferenceNodeOp(const std::vector<Expr>& nodes,
                         Shape shape,
                         Type value_type = Type::float32)
      : NaryNodeOp(nodes, shape, value_type) {}

  OnlyForInferenceNodeOp(const std::vector<Expr>& nodes) : NaryNodeOp(nodes) {}

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp()};
  }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class QuantMultNodeOp : public OnlyForInferenceNodeOp {
public:
  std::string matname;
  QuantMultNodeOp(Expr input, std::string name_to_search = "blaaaaaa") : OnlyForInferenceNodeOp({input}, Shape()), matname(name_to_search) {
    ABORT_IF(children().size() != 1, "expected 1 child");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "Input matrix cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      auto input = child(0)->val();

      ABORT_IF(input->type() != Type::float32, "Trying to quantize non-float");

      auto expmap = child(0)->graph()->getNameMap();
      auto mapiter2 = expmap.find(matname);

      if (Type_ == Type::int16) {
        *val_->data() = 1024.0f;
      } else if (mapiter2 == expmap.end()) {
        *val_->data() = 127.0f / intgemm::MaxAbsolute(input->data(), input->data() + input->shape().elements());
      } else {
        //std::cerr << "Artificial quant mult" << std::endl;
        *val_->data() = *(mapiter2->second->val()->data())/2;
      }
    )};
  }

  const std::string type() override { return "intQuantMult"; }
};

namespace { // anonymous namespace

template <Type Type_, typename PrepareMatrixFun>
inline NodeOps prepareMatrixForwardOps(Node* node, PrepareMatrixFun prepare_matrix_fun) {
  return {NodeOp(
    using Integer = typename backend<Type_>::Integer;

    auto input = node->child(0)->val();
    auto quant_mult = node->child(1)->val();
    prepare_matrix_fun(
        input->data(),
        node->val()->data<Integer>(),
        *quant_mult->data(),
        rows(input),
        input->shape()[-1]);
  )};
}

} // anonymous namespace

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class PrepareANodeOp : public OnlyForInferenceNodeOp {
public:
  PrepareANodeOp(Expr input, Expr quant_mult, float clipValue)
      : OnlyForInferenceNodeOp({input, quant_mult}, input->shape(), Type_) {
    ABORT_IF(children().size() != 2, "expected 2 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
  }

  NodeOps forwardOps() override {
    return prepareMatrixForwardOps<Type_>(this, backend<Type_>::PrepareANew);
  }

  const std::string type() override { return "intPrepareA"; }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class PrepareANodeOpOld : public OnlyForInferenceNodeOp {
public:
  PrepareANodeOpOld(Expr input, Expr quant_mult, float clipValue)
      : OnlyForInferenceNodeOp({input, quant_mult}, input->shape(), Type_) {
    ABORT_IF(children().size() != 2, "expected 2 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
  }

  NodeOps forwardOps() override {
    return prepareMatrixForwardOps<Type_>(this, backend<Type_>::PrepareA);
  }

  const std::string type() override { return "intPrepareAold"; }
};

template <Type Type_>
class PrepareBiasForBNodeOp : public OnlyForInferenceNodeOp {
public:
  PrepareBiasForBNodeOp(Expr bias, Expr a, Expr inputB_preppd, Expr a_quant_mult, Expr b_quant_mult)
      : OnlyForInferenceNodeOp({bias, a, inputB_preppd, a_quant_mult, b_quant_mult}, bias->shape(), Type_) {
    ABORT_IF(children().size() != 5, "expected 5 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "Bias cannot be null");
    ABORT_IF(child(1) == nullptr, "A cannot be null");
    ABORT_IF(child(2) == nullptr, "B cannot be null");
    ABORT_IF(child(3) == nullptr, "Quant mult of A cannot be null");
    ABORT_IF(child(4) == nullptr, "Quant mult of B cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
    auto bias = this->child(0)->val();
    auto a = this->child(1)->val();
    auto b = this->child(2)->val();
    auto quant_mult_a = this->child(3)->val();
    auto quant_mult_b = this->child(4)->val();

    float unquant_mult = (-1)*((127.0f / *quant_mult_a->data())*(127.0f / *quant_mult_b->data()))/(127.0f); //Minus one to invert add_ps later on
    intgemm::Int8::PrepareBiasFor8(1, (const int8_t *)b->data(), intgemm::BiasAddUnquantizeC(val_->data(), bias->data(), unquant_mult), 1, cols(a), cols(b));

    /* EXample code for alpha
    auto namedmap = child(0)->graph()->getRevNameMap();
    auto expmap = child(0)->graph()->getNameMap();
    std::string alpha_name;
    auto mapiter = namedmap.find(child(0));
    if (mapiter != namedmap.end()) {
      alpha_name = mapiter->second + "_alpha";
    } else {
      alpha_name = "_alpha";
      std::cerr << "No found EXP name" << std::endl;
    }

    auto mapiter2 = expmap.find(alpha_name);
    if (mapiter2 != expmap.end()) {
      std::cerr << "Alpha actual: " << *quant_mult_a->data() << " alpha stored: " << *(mapiter2->second->val()->data()) << std::endl;
    } else {
      std::cerr << "Exp named: " << alpha_name << " not found" << std::endl;
    }*/
    
    )};
  }

  const std::string type() override { return "prepareBias"; }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class PrepareBNodeOp : public OnlyForInferenceNodeOp {
public:
  PrepareBNodeOp(Expr input, Expr quant_mult, float clipValue)
      : OnlyForInferenceNodeOp({input, quant_mult}, input->shape(), Type_) {
    ABORT_IF(children().size() != 2, "expected 2 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "B cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of B cannot be null");
  }

  NodeOps forwardOps() override {
    return prepareMatrixForwardOps<Type_>(this, backend<Type_>::PrepareB);
  }

  const std::string type() override { return "intPrepareB"; }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class SelectColumnsBNodeOp : public OnlyForInferenceNodeOp {
public:
  SelectColumnsBNodeOp(Expr input, const std::vector<Word> &indices)
      : OnlyForInferenceNodeOp({input}, newShape(input, indices), Type_), indices_(indices) {
    ABORT_IF(children().size() != 1, "expected 1 child");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "B cannot be null");

    // Check number of selected columns
    assert(indices.size() % 8 == 0);
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      using Integer = typename backend<Type_>::Integer;

      auto input = child(0)->val();
      backend<Type_>::SelectColumnsB(
          (const Integer*)input->data(),
          val_->data<Integer>(),
          rows(input),
          &*indices_.begin(),
          &*indices_.end());
    )};
  }

  const std::string type() override { return "intSelectColumnsB"; }

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
    Ptr<SelectColumnsBNodeOp> cnode = std::dynamic_pointer_cast<SelectColumnsBNodeOp>(node);
    if (!cnode) return false;
    return indices_ == cnode->indices_;
  }

private:
  static Shape newShape(Expr a, const std::vector<Word>& indices) {
    Shape ret = a->shape();
    ret.set(1, indices.size());
    return ret;
  }

  std::vector<Word> indices_;
};

/*
*                   +-----------+
*                   |    Dot    |
*                   +-----+-----+
*                         |
*         +----------+----+-----+----------+
*         |          |          |          |
*  +------+------+   |   +------+------+   |
*  | Quantized A |   |   | Quantized B |   |
*  +-------------+   |   +-------------+   |
*             +------+------+       +------+------+
*             | QuantMult A |       | QuantMult B |
*             +-------------+       +-------------+
*/
template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class DotNodeOp : public OnlyForInferenceNodeOp {
private:
  float scalar_;

public:
  DotNodeOp(Expr a, Expr a_quant_mult, Expr b, Expr b_quant_mult, float scalar)
      : OnlyForInferenceNodeOp({a, a_quant_mult, b, b_quant_mult}, newShape(a, b)), scalar_(scalar) {
    ABORT_IF(children().size() != 4, "expected 4 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
    ABORT_IF(child(2) == nullptr, "B cannot be null");
    ABORT_IF(child(3) == nullptr, "Quant mult of B cannot be null");

    // Check alignment
    assert(child(2)->shape()[-1] % 8 == 0);

    // Check dimmensions
    ABORT_IF(child(0)->shape()[-1] != child(2)->shape()[-2], "Matrices cannot be multiplied because there's a dimension mismatch");
  }

  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      using Integer = typename backend<Type_>::Integer;
      using intgemm::JustUnquantizeC;

      auto a = child(0)->val();
      auto quant_mult_a = child(1)->val();
      auto b = child(2)->val();
      auto quant_mult_b = child(3)->val();
      ABORT_IF(true, "We only do multiplication with biases around here");
      backend<Type_>::Multiply(
          (const Integer*)a->data(),
          (const Integer*)b->data(),
          JustUnquantizeC(val_->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));
    )};
  }

  const std::string type() override { return "intDot"; }
};

/*
*                         +-----------+
*                         |  Affine   |
*                         +-----+-----+
*                               |
*         +----------+----------+----------+----------+
*         |          |          |          |          |
*  +------+------+   |   +------+------+   |      +---+---+
*  | Quantized A |   |   | Quantized B |   |      | Bias  |
*  +-------------+   |   +-------------+   |      +-------+
*             +------+------+       +------+------+
*             | QuantMult A |       | QuantMult B |
*             +-------------+       +-------------+
*/
namespace {
template<class Numbah>
void SlowRefFloat2(const Numbah *A, const Numbah *B, Numbah *C, size_t A_rows, size_t width, size_t B_cols, const float *bias) {
  for (size_t r = 0; r < A_rows; ++r) {
    for (size_t c = 0; c < B_cols; ++c) {
      Numbah sum = 0;
      for (size_t k = 0; k < width; ++k) {
        sum += A[r * width + k] * B[k * B_cols + c];
      }
      if (bias) {
        C[r * B_cols + c] = sum + bias[c];
      } else {
        C[r * B_cols + c] = sum;
      }
    }
  }
}

void SaturateMult(const uint8_t *A, int16_t *B, int32_t *C, size_t A_rows, size_t width, size_t B_cols) {
  const int32_t MAXINT16 = 32767;

  //const int32_t MAXINT16 = 9992767;
  for (size_t r = 0; r < A_rows; ++r) {
    for (size_t c = 0; c < B_cols; ++c) {
      int32_t sum = 0;
      int32_t intermediate_sum = 0;
      for (size_t k = 0; k < width; ++k) {
        int32_t num1 = A[r * width + k];
        int32_t num2 = B[k * B_cols + c];
        intermediate_sum += num1*num2;
        if ((k+1)%2 == 0) {
          if (intermediate_sum > MAXINT16) {
            std::cerr << "Saturation: " << intermediate_sum << std::endl;
            intermediate_sum = MAXINT16;
          }
          sum+=intermediate_sum;
          intermediate_sum = 0;
        }
      }
      C[r * B_cols + c] = sum;
    }
  }
}
}
template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class AffineNodeOp : public OnlyForInferenceNodeOp {
private:
  float scalar_;

public:
  AffineNodeOp(Expr a, Expr a_quant_mult, Expr b, Expr b_quant_mult, Expr bias, float scalar)
      : OnlyForInferenceNodeOp({a, a_quant_mult, b, b_quant_mult, bias}, newShape(a, b, bias)), scalar_(scalar) {
    ABORT_IF(children().size() != 5, "expected 5 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
    ABORT_IF(child(2) == nullptr, "B cannot be null");
    ABORT_IF(child(3) == nullptr, "Quant mult of B cannot be null");
    ABORT_IF(child(4) == nullptr, "Bias cannot be null");
    ABORT_IF(scalar_ != 1.0f, "Scalar should be one.");

    // Check alignment
    assert(child(2)->shape()[-1] % 8 == 0);

    // Check dimmensions
    ABORT_IF(child(0)->shape()[-1] != child(2)->shape()[-2], "Matrices cannot be multiplied because there's a dimension mismatch");
    ABORT_IF(child(2)->shape()[-1] != child(4)->shape()[-1], "Bias cannot be added because there's a dimension mismatch");
  }

  Shape newShape(Expr a, Expr b, Expr bias) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      using Integer = typename backend<Type_>::Integer;
      using intgemm::BiasAddUnquantizeC;
      using intgemm::Identity;
      using intgemm::JustUnquantizeC;

      auto a = child(0)->val();
      auto quant_mult_a = child(1)->val();
      auto b = child(2)->val();
      auto quant_mult_b = child(3)->val();
      auto bias = child(4)->val();
      
      backend<Type_>::Multiply8new(
          (const Integer*)a->data(),
          (const Integer*)b->data(),
          BiasAddUnquantizeC(val_->data(), bias->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));

    )};
  }

  const std::string type() override { return "intAffine"; }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
struct ops {
  static inline Expr dot(Expr a, Expr quant_mult_a, Expr b, Expr quant_mult_b, float scalar) {
    return Expression<DotNodeOp<Type_>>(a, quant_mult_a, b, quant_mult_b, scalar);
  }
  static inline Expr affine(Expr a, Expr quant_mult_a, Expr b, Expr quant_mult_b, Expr bias, float scalar) {
    return Expression<AffineNodeOp<Type_>>(a, quant_mult_a, b, quant_mult_b, bias, scalar);
  }
  static inline Expr quantMult(Expr a, Expr bias=nullptr) {
    if (bias) {
      auto namedmap = bias->graph()->getRevNameMap();

      if (namedmap[a] == "") { //Skip b matrix
        std::string alpha_name;
        auto mapiter = namedmap.find(bias);
        if (mapiter != namedmap.end()) {
          alpha_name = mapiter->second + "_alpha";
        } else {
          alpha_name = "_alpha";
        }
        return Expression<QuantMultNodeOp<Type_>>(a, alpha_name);
      } else {
        return Expression<QuantMultNodeOp<Type_>>(a, "BLAAAA");
      }
    }
    return Expression<QuantMultNodeOp<Type_>>(a, "BLAAAA");
  }
  static inline Expr prepareA(Expr a, Expr quant_mult, float clipValue) {
    return Expression<PrepareANodeOp<Type_>>(a, quant_mult, clipValue);
  }
  static inline Expr prepareAOld(Expr a, Expr quant_mult, float clipValue) {
    return Expression<PrepareANodeOpOld<Type_>>(a, quant_mult, clipValue);
  }
  static inline Expr prepareB(Expr b, Expr quant_mult, float clipValue) {
    return Expression<PrepareBNodeOp<Type_>>(b, quant_mult, clipValue);
  }
  static inline Expr PrepareBiasForB(Expr bias, Expr inputA, Expr inputB_preppd, Expr a_quant_mult, Expr b_quant_mult) {
    return Expression<PrepareBiasForBNodeOp<marian::Type::float32>>(bias, inputA, inputB_preppd, a_quant_mult, b_quant_mult);
  }
  static inline Expr selectColumnsB(Expr b, const std::vector<Word> &cols) {
    return Expression<SelectColumnsBNodeOp<Type_>>(b, cols);
  }
};

} // namespace integer

using int8 = integer::ops<Type::int8>;
using int16 = integer::ops<Type::int16>;

}
}
