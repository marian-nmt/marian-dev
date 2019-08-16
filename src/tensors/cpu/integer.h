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
  QuantMultNodeOp(Expr input) : OnlyForInferenceNodeOp({input}, Shape()) {
    ABORT_IF(children().size() != 1, "expected 1 child");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "Input matrix cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      auto input = child(0)->val();

      ABORT_IF(input->type() != Type::float32, "Trying to quantize non-float");

      if (Type_ == Type::int16) {
        *val_->data() = 1024.0f;
      } else {
        *val_->data() = 127.0f / intgemm::MaxAbsolute(input->data(), input->data() + input->shape().elements());
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
  PrepareBiasForBNodeOp(Expr bias, Expr inputB, Expr a_quant_mult)
      : OnlyForInferenceNodeOp({bias, inputB, a_quant_mult}, bias->shape(), Type_) {
    ABORT_IF(children().size() != 3, "expected 2 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "Bias cannot be null");
    ABORT_IF(child(1) == nullptr, "B cannot be null");
    ABORT_IF(child(2) == nullptr, "Quant mult of A cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(

    int rowsB = rows(this->child(1)->val());
    int colsB = cols(this->child(1)->val());

    auto inputB = this->child(1)->val()->data();
    auto bias = this->child(0)->val()->data();

    float alpha = 127/ *this->child(2)->val()->data();

    //copy the bias because we shouldn't modify it in place
    for (int i = 0; i < this->shape()[-1]; i++) {
      this->val()->data()[i] = bias[i];
    }
    static bool first = true;
    if (first) {
      std::cerr << "Alpha: " << alpha << std::endl;
      first = false;
    }

    intgemm::Int8::PrepareBiasFor8(
     inputB,
     this->val()->data(),
     alpha,
     rowsB,
     colsB);
     //std::cout << this->val()->data()[0] << std::endl;
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
  AffineNodeOp(Expr a, Expr a_quant_mult, Expr b, Expr b_quant_mult, Expr bias, float scalar, Expr a_old, Expr bias_old, Expr b_raw)
      : OnlyForInferenceNodeOp({a, a_quant_mult, b, b_quant_mult, bias, a_old, bias_old, b_raw}, newShape(a, b, bias)), scalar_(scalar) {
    ABORT_IF(children().size() != 8, "expected 7 children");

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
    ABORT_IF(child(2) == nullptr, "B cannot be null");
    ABORT_IF(child(3) == nullptr, "Quant mult of B cannot be null");
    ABORT_IF(child(4) == nullptr, "Bias cannot be null");
    ABORT_IF(child(5) == nullptr, "Old A cannot be null");
    ABORT_IF(child(6) == nullptr, "Old bias Cannot be null");
    ABORT_IF(child(7) == nullptr, "Raw B cannot be null");
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
      auto a_old = child(5)->val();
      auto bias_old = child(6)->val();
      auto b_raw = child(7)->val();
      
      /****
       * Haaaaaaaacky
       *
      intgemm::AlignedVector<float> tmpBias(cols(b));
      for (size_t i = 0; i<cols(b); i++) {
        tmpBias[i] = bias->data()[i];
      }
      backend<Type_>::PrepareBiasFor8((const Integer*)b->data(), tmpBias.begin(), *quant_mult_a->data(), rows(b), cols(b));
      ****
       * Haaaaaaaacky
       */
      //static bool first = true;
      //bool written = false;
      intgemm::AlignedVector<int32_t> oldMult16(rows(a)*cols(b));
      intgemm::AlignedVector<int32_t> newMult16(rows(a)*cols(b));

      intgemm::AlignedVector<int16_t> A16(rows(a)*cols(a));
      intgemm::AlignedVector<int16_t> A127(rows(a)*cols(a));
      intgemm::AlignedVector<int16_t> B16(rows(b)*cols(b));

      intgemm::AlignedVector<int16_t> B16_INT_NOREORD(rows(b)*cols(b));
      std::vector<int32_t> B32_INT_NOREORD(rows(b)*cols(b), 0);

 
      intgemm::AVX2_16bit::Quantize(b_raw->data(), B16_INT_NOREORD.begin(), *quant_mult_b->data(), rows(b)*cols(b));


      for (int i = 0; i < B32_INT_NOREORD.size(); i++) {
        B32_INT_NOREORD[i] = B16_INT_NOREORD[i];
      }

      for (int i = 0; i<A16.size(); i++) {
        A16[i] = (int16_t)(((const Integer*)a_old->data())[i]);
        A127[i] = (int16_t)(((const uint8_t*)a->data())[i]);
        if (A16[i] +127 != A127[i]) {
          std::cerr << "ERRROR MISMATCH: " << A16[i] << " " << A127[i] << std::endl;
        }
      }

      intgemm::AVX2_16bit::Multiply(
          A16.begin(),
          B16.begin(),
          //BiasAddUnquantizeC(val_->data(), bias_old->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          //JustUnquantizeC(val_->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          Identity(reinterpret_cast<int32_t*>(oldMult16.begin())),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));

      intgemm::AVX2_16bit::Multiply(
          A127.begin(),
          B16.begin(),
          //BiasAddUnquantizeC(val_->data(), bias_old->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          //JustUnquantizeC(val_->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          Identity(reinterpret_cast<int32_t*>(newMult16.begin())),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));


      intgemm::AlignedVector<int> oldMult(rows(a)*cols(b));
      intgemm::AlignedVector<int> newMult(rows(a)*cols(b));
      backend<Type_>::Multiply(
          (const Integer*)a_old->data(),
          //(const Integer*)a->data(),
          (const Integer*)b->data(),
          //BiasAddUnquantizeC(val_->data(), bias_old->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          //JustUnquantizeC(val_->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          Identity(reinterpret_cast<int32_t*>(oldMult.begin())),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));

      backend<Type_>::Multiply8new(
          //(const Integer*)a_old->data(),
          (const Integer*)a->data(),
          (const Integer*)b->data(),
          //BiasAddUnquantizeC(val_->data(), bias_old->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          //JustUnquantizeC(val_->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          Identity(reinterpret_cast<int32_t*>(newMult.begin())),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));

      std::vector<int> offset(rows(a)*cols(a), 127);
      std::vector<int> b_int(rows(b)*cols(b));
      std::vector<int> b_makeup(rows(a)*cols(b), 0);
      for (size_t i = 0; i<b_int.size(); i++) {
        b_int[i] = ((const Integer*)b->data())[i];
      }
      SlowRefFloat2(&offset[0], &B32_INT_NOREORD[0], &b_makeup[0], rows(a), cols(a), cols(b), nullptr);

      for (size_t i = 0; i < newMult.size(); i++) {
        newMult[i] -= b_makeup[i];
      }

      //16Bit sanity check
      for (size_t i = 0; i < newMult16.size(); i++) {
        newMult16[i] -= b_makeup[i];
      }

      volatile int matches = 0;
      volatile int mismatches = 0;

      
      for (size_t i = 0; i < newMult.size(); i++) {
        if (newMult[i] == oldMult[i]) {
          //std::cerr << "Match: old: " << oldMult[i] << " new: " << newMult[i] << std::endl;
          matches++;
        } else {
          mismatches++;
        }
      }
      if (mismatches > 10) {
        //static int i = 0;
        std::cerr << "NEW MATRICES INCOMING: A: " << rows(a) << "x" << cols(a) << " B: " << rows(b) << "x" << cols(b) << std::endl;
        std::cerr << "Old vs new 8 bit: Matches: " << matches << " mismatches: " << mismatches << std::endl;
        /*std::ofstream file1(std::to_string(i));
        i++;
        file1 << "RowsA: " << rows(a) << " ColsA " << cols(a) << " RowsB " << rows(b) << " ColsB " << cols(b) << std::endl;
        file1 << "QuantMultA: " << *quant_mult_a->data() << " QuantMultB: " << *quant_mult_b->data() << std::endl;
        file1 << "AQuant:" << std::endl;
        for (int i = 0; i < rows(a)*cols(a); i++) {
          file1 << (int32_t)(((const Integer*)a_old->data())[i]) << " ";
        }
        file1 << std::endl << "BRaw:" << std::endl;
        for (int i = 0; i < rows(b)*cols(b); i++) {
          file1 << b_raw->data()[i] << " ";
        }
        file1 << std::endl << "Bias:" << std::endl;
        for (int i = 0; i < bias_old->shape()[-1]; i++) {
          file1 << bias_old->data()[i] << " ";
        }
        file1 << std::endl;
        file1.close();*/
      }
/* 
      intgemm::AlignedVector<int32_t> slowResCwithSat(rows(a)*cols(b));
      SaturateMult((const uint8_t *)a->data(), B16_INT_NOREORD.begin(), slowResCwithSat.begin(), rows(a), cols(a), cols(b));

      //make up for add127
      for (size_t i = 0; i < slowResCwithSat.size(); i++) {
        slowResCwithSat[i] -= b_makeup[i];
      }*/


       /*
      matches = 0;
      mismatches = 0;
      for (size_t i = 0; i < newMult.size(); i++) {
        if (oldMult16[i] == oldMult[i]) {
          //std::cerr << "Mismatch 8-16 old: 8: " << oldMult[i] << " 16: " << oldMult16[i] << std::endl;
          matches++;
        } else {
          mismatches++;
        }
      }
      if (mismatches > 10) {
        std::cerr << "NEW MATRICES INCOMING: A: " << rows(a) << "x" << cols(a) << " B: " << rows(b) << "x" << cols(b) << std::endl;
        std::cerr << "Old 16 bit vs 8 bit: Matches: " << matches << " mismatches: " << mismatches << std::endl;
        for (size_t i =0; i < 10; i++) {
          std::cerr << "Sample: " << oldMult[i] << " " << oldMult16[i] << std::endl;
        }
      }
     
      matches = 0;
      mismatches = 0;
      for (size_t i = 0; i < newMult.size(); i++) {
        if (newMult16[i] == newMult[i]) {
            matches++;
        } else {
          mismatches++;
        }
      }
      std::cerr << "New 16 bit vs 8 bit: Matches: " << matches << " mismatches: " << mismatches << std::endl;
      
      matches = 0;
      mismatches = 0;
      for (size_t i = 0; i < newMult16.size(); i++) {
        if (newMult16[i] == oldMult16[i]) {
          //std::cerr << "Mismatch 16bit: old: " << oldMult16[i] << " new: " << newMult16[i] << std::endl;
          matches++;
        } else {
          mismatches++;
        }
      }
      if (mismatches > 10) {
        //static int i = 0;
        std::cerr << "NEW MATRICES INCOMING: A: " << rows(a) << "x" << cols(a) << " B: " << rows(b) << "x" << cols(b) << std::endl;
        std::cerr << "Old vs new 16 bit: Matches: " << matches << " mismatches: " << mismatches << std::endl;
        //std::ofstream file1((std::to_string(i) + "_16"));
        i++;
        file1 << "RowsA: " << rows(a) << " ColsA " << cols(a) << " RowsB " << rows(b) << " ColsB " << cols(b) << std::endl;
        file1 << "QuantMultA: " << *quant_mult_a->data() << " QuantMultB: " << *quant_mult_b->data() << std::endl;
        file1 << "AQuant:" << std::endl;
        for (int i = 0; i < rows(a)*cols(a); i++) {
          file1 << (int32_t)(((const Integer*)a_old->data())[i]) << " ";
        }
        file1 << std::endl << "BRaw:" << std::endl;
        for (int i = 0; i < rows(b)*cols(b); i++) {
          file1 << b_raw->data()[i] << " ";
        }
        file1 << std::endl << "Bias:" << std::endl;
        for (int i = 0; i < bias_old->shape()[-1]; i++) {
          file1 << bias_old->data()[i] << " ";
        }
        file1 << std::endl;
        file1.close();
      }*/
/*


       backend<Type_>::Multiply(
          (const Integer*)a_old->data(),
          (const Integer*)b->data(),
          BiasAddUnquantizeC(val_->data(), bias_old->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));
      
      for (int i = 0; i< rows(a)*cols(a); i++) {
        if (((const Integer*)a_old->data())[i] + 127 != ((const uint8_t*)a->data())[i]) {
          std::cerr<< "Error at " << i << " old: " << (int)((((const Integer*)a_old->data())[i])) << " new: " << (int)((((const uint8_t*)a->data())[i])) << std::endl;
        }
      }
      std::unique_ptr<float> new_res(new float[rows(a)*cols(b)]);
      for (int i = 0; i<rows(a)*cols(b);i++) {
        new_res.get()[i] = val_->data()[i];
      }
      std::cerr << "Scalar: " << scalar_ << " rows: " << rows(a) << " columns: " << cols(a) << " rows(b) " << rows(b) << std::endl;
      std::cerr << "Quant mult a: " << *quant_mult_a->data() << " quant mult b: " << *quant_mult_b->data() << std::endl;*/
      
        //backend<Type_>::Multiply8new(
        //  (const Integer*)a->data(),
        //  (const Integer*)b->data(),
        //  BiasAddUnquantizeC(val_->data(), bias->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
        //  rows(a),
        //  cols(a), // Shared dimension.
        //  cols(b));
/* 
        backend<Type_>::Multiply(
          (const Integer*)a_old->data(),
          //(const Integer*)a->data(),
          (const Integer*)b->data(),
          BiasAddUnquantizeC(val_->data(), bias_old->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          //JustUnquantizeC(val_->data(), scalar_ / (*quant_mult_a->data() * *quant_mult_b->data())),
          //Identity(reinterpret_cast<int32_t*>(val_->data())),
          rows(a),
          cols(a), // Shared dimension.
          cols(b));
      AddBias(val_, bias_old);
      //Add bias and unqunatize C
      
      //ONEZ: shape: rows(a), cols(a)

      std::vector<float> onez(rows(a)*cols(a), 1.0f);
      std::vector<float> bias_offset(rows(a)*cols(b), 0.0f);

      SlowRefFloat2(&onez[0], b_raw->data(), &bias_offset[0], rows(a), cols(a), cols(b), nullptr);

      //Scale everything by alpha
      for (auto&& num : bias_offset) {
        num = num*(127.0/(*quant_mult_a->data()));
      }
      //Add it to the bias
      std::vector<float> manual_bias_with_offset(cols(b));
      for (size_t i = 0; i<cols(b); i++) {
        manual_bias_with_offset[i] = bias_old->data()[i] - bias_offset[rows(a)*i];
      }

      //Verify the difference between the new way to compute the bias and the old way
      for (size_t i = 0; i<cols(b); i++) {
        if (fabs(manual_bias_with_offset[i] - bias->data()[i]) > 0.00001) {
          //std::cerr << "Biases differ! Assembly: " << bias->data()[i] << " slow CPP: " << manual_bias_with_offset[i] << std::endl;
        }
        bias->data()[i] = manual_bias_with_offset[i];
      }

*/
      for (int i = 0; i < rows(a); i++) {
        for (int j = 0; j < cols(b); j++) {
          float mult_res = (float)(reinterpret_cast<int32_t*>(newMult.begin())[j + i*cols(b)]); // oldMult.begin()
          float unquant_mult = scalar_ / (*quant_mult_a->data() * *quant_mult_b->data());
          val_->data()[j + i*cols(b)] = mult_res*unquant_mult;
        }
      }
      AddBias(val_, bias_old);
      //AddBias(val_, bias);
      /*
      
        float totaldiff = 0;
        for (int i = 0; i < rows(a); i++) {
          for (int j = 0; j < cols(b); j++) {
            float diff = val_->data()[i*(cols(b)) + j] - new_res.get()[i*(cols(b)) + j];
            totaldiff += diff*diff;
          }
        }
        std::cerr << "MSE: " << std::sqrt(totaldiff/(rows(a)*cols(b))) << std::endl;*/
    )};
  }

  const std::string type() override { return "intAffine"; }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
struct ops {
  static inline Expr dot(Expr a, Expr quant_mult_a, Expr b, Expr quant_mult_b, float scalar) {
    return Expression<DotNodeOp<Type_>>(a, quant_mult_a, b, quant_mult_b, scalar);
  }
  static inline Expr affine(Expr a, Expr quant_mult_a, Expr b, Expr quant_mult_b, Expr bias, float scalar, Expr a_old, Expr bias_old, Expr b_raw) {
    return Expression<AffineNodeOp<Type_>>(a, quant_mult_a, b, quant_mult_b, bias, scalar, a_old, bias_old, b_raw);
  }
  static inline Expr quantMult(Expr a) {
    return Expression<QuantMultNodeOp<Type_>>(a);
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
  static inline Expr prepareBiasForB(Expr bias, Expr inputB, Expr a_quant_mult) {
    return Expression<PrepareBiasForBNodeOp<marian::Type::float32>>(bias, inputB, a_quant_mult); //TODO type is wrong
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
