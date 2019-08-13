#pragma once

#include "graph/node.h"
#include "tensors/cpu/sharp/packed_gemm.h"
#include "backend.h"

#if USE_FBGEMM
#include "3rd_party/fbgemm/include/fbgemm/FbgemmFP16.h"
#include "3rd_party/fbgemm/include/fbgemm/Fbgemm.h"
using namespace fbgemm;
#endif  // USE_FBGEMM

namespace marian {
namespace cpu {
namespace variant {

// Enumeration for the Matrix used in pack functions
// A matrix - 0, B matrix - 1
enum class PackMatrix : uint8_t {
  A = 0x00,
  B = 0x01
};

// Pack a matrix into cache utilization efficient way (block format)
// PackMatrix packMat_: the type of packed matrix - A or B matrix
// bool transpose_: transpose
// int nrow_: the number of rows
// int ncol_: the number of columns
// int kernel_ncol_blocks_: the number of column blocks
// int brow_: the number of rows in a block
// int bcol_: the number of columns in a block
// int last_brow_: the number of rows in the last block
// int nbrow_: row index in a block
// int nbcol_: column index in a block
// uint64_t packsize_: the size of the packed matrix
//                    (the number of fp16 elements + padding (1024) + extra temporary memory (256))
struct PackNodeOpFp16 : public UnaryNodeOp {
  PackMatrix packMat_;
  bool transpose_;
  int nrow_;
  int ncol_;
  int kernel_ncol_blocks_;
  int brow_;
  int bcol_;
  int last_brow_;
  int nbrow_;
  int nbcol_;
  uint64_t packsize_;

  PackNodeOpFp16(Expr a, PackMatrix packMat, bool transpose, float clipValue)
      : UnaryNodeOp(a, newShape(a, transpose), Type::uint8),
        packMat_(packMat),
        transpose_(transpose) {
    if(packMat != PackMatrix::B)
      ABORT("Only prepacking of B (weight matrix) is supported");
    if(clipValue != 0)
      ABORT("Clipping is not supported");
    if(!memoize_)
      ABORT("Only constant weight node can be packed");
  }

  NodeOps forwardOps() override {
    return {NodeOp(PackFp16(val_,
                            child(0)->val(),
                            transpose_,
                            nrow_,
                            ncol_,
                            kernel_ncol_blocks_,
                            brow_,
                            bcol_,
                            last_brow_,
                            nbrow_,
                            nbcol_,
                            packsize_))
    };
  }

  NodeOps backwardOps() override {
    ABORT("PackNodeOp only available for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "packMatFp16"; }

  Shape newShape(Expr a, bool transpose) {
#if USE_FBGEMM
    auto shapeMat = a->shape();
    // Should be 2D - weight matrix
    ABORT_IF(shapeMat.size() != 2,
            "Weight Matrix should be 2D");
    PackInfoFp16(shapeMat,
                 transpose,
                 nrow_,
                 ncol_,
                 kernel_ncol_blocks_,
                 brow_,
                 bcol_,
                 last_brow_,
                 nbrow_,
                 nbcol_,
                 packsize_);
    //nrow_ = transpose ? shapeMat[1] : shapeMat[0];
    //ncol_ = transpose ? shapeMat[0] : shapeMat[1];
    //kernel_ncol_blocks_ = 2;
    //brow_ = 512;
    //bcol_ = 8 * kernel_ncol_blocks_;
    //last_brow_ = nrow_ % brow_ == 0 ? brow_ : nrow_ % brow_;
    //nbrow_ = nrow_ % brow_ == 0 ? nrow_ / brow_ : (nrow_ + brow_) / brow_;
    //nbcol_ = ncol_ % bcol_ == 0 ? ncol_ / bcol_ : (ncol_ + bcol_) / bcol_;
    //const int padding = 1024;  // required by sw pipelined kernels
    //const int specialMem = 256;
    //packsize_ = ((nbrow_ * brow_) * (nbcol_ * bcol_)) * sizeof(fbgemm::float16) + padding + specialMem;

    Shape outShape({(int)packsize_});

    return outShape;
#else // USE_FBGEMM
    ABORT("Packed GEMM requires a build with USE_FBGEMM enabled");
    return Shape();
#endif  // USE_FBGEMM
  }
};

// Pack a matrix into cache utilization efficient way (block format)
// PackMatrix packMat_: the type of packed matrix - A or B matrix
// bool transpose_: transpose
// int nrow_: the number of rows
// int ncol_: the number of columns
// int kernel_ncol_blocks_: the number of column blocks
// int brow_: the number of rows in a block
// int bcol_: the number of columns in a block
// int last_brow_: the number of rows in the last block
// int nbrow_: row index in a block
// int nbcol_: column index in a block
// uint64_t packsize_: the size of the packed matrix
//                    (the number of fp16 elements + padding (1024) + extra temporary memory (256))
struct PackNodeOpInt8 : public UnaryNodeOp {
  PackMatrix packMat_;
  bool transpose_;
  int nrow_;
  int ncol_;
  uint64_t packsize_;

  PackNodeOpInt8(Expr a, PackMatrix packMat, bool transpose, float clipValue)
      : UnaryNodeOp(a, newShape(a, transpose), Type::uint8),
        packMat_(packMat),
        transpose_(transpose) {
    if(packMat != PackMatrix::B)
      ABORT("Only prepacking of B (weight matrix) is supported");
    if(clipValue != 0)
      ABORT("Clipping is not supported");
    if(!memoize_)
      ABORT("Only constant weight node can be packed");
  }

  NodeOps forwardOps() override {
    return {NodeOp(PackInt8(val_,
                            child(0)->val(),
                            transpose_,
                            nrow_,
                            ncol_,
                            packsize_))
    };
  }

  NodeOps backwardOps() override {
    ABORT("PackNodeOp only available for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "packMatInt8"; }

  Shape newShape(Expr a, bool transpose) {
#if USE_FBGEMM
    // auto shapeMat = a->shape();
    // // Should be 2D - weight matrix
    // ABORT_IF(shapeMat.size() != 2,
    //         "Weight Matrix should be 2D");
    // nrow_ = transpose ? shapeMat[1] : shapeMat[0];
    // ncol_ = transpose ? shapeMat[0] : shapeMat[1];
    // packsize_ = fbgemm::PackMatrix<fbgemm::PackBMatrix<int8_t>, int8_t>::packedBufferSize(
    //     transpose ? shapeMat[1] : shapeMat[0],
    //     transpose ? shapeMat[0] : shapeMat[1]);
    // // add extra space for storing some other variables specific to B matrix
    // // quantization sacles: 1 per column and float
    // // quantization offset: 1 per column and int32
    // // column offsets: 1 per column and int32
    // packsize_ += ncol_ * (sizeof(float) + sizeof(int32_t) + sizeof(int32_t));
    PackInfoInt8(a->shape(), transpose, nrow_, ncol_, packsize_);
    Shape outShape({(int)packsize_});

    return outShape;
#else // USE_FBGEMM
    ABORT("Packed GEMM requires a build with USE_FBGEMM enabled");
    return Shape();
#endif  // USE_FBGEMM
  }
};

// Affine transform (matrix multiplication) using packed B matrix
// Especially, this GEMM packs and saves weights into fp16.
// Actual computation is done in fp32.
// float scalar_: scalar multiplier
// size_t m_: the number of rows in A and C
// size_t n_: the number of columns in B and C
// size_t k_: the number of columns in A and the number of rows in C
// bool transA_: transpose A
// bool transB_: transpose B
class AffineNodeOpFp16 : public NaryNodeOp {
private:
  float scalar_;
  size_t m_;
  size_t n_;
  size_t k_;
  bool transA_;
  bool transB_;

public:
  AffineNodeOpFp16(const std::vector<Expr>& nodes, Shape bShape, bool transA, bool transB, float scalar)
      : NaryNodeOp(nodes, newShape(nodes[0], bShape, transA), Type::float32),
        scalar_(scalar) {
    transA_ = transA;
    transB_ = transB;
    m_ = nodes[0]->shape().elements() / nodes[0]->shape()[-1];
    k_ = nodes[0]->shape().back();
    if(transA)
      std::swap(m_, k_);

    ABORT_IF(transB, "B should not be transposed for packed GEMM.");
    int nrow = (int) k_;
    // int ncol = transpose ? shape[0] : shape[1];
    // int kernel_ncol_blocks = 2;
    int brow = 512;
    // int bcol = 8 * kernel_ncol_blocks;
    // int last_brow = nrow % brow == 0 ? brow : nrow % brow;
    int nbrow = nrow % brow == 0 ? nrow / brow : (nrow + brow) / brow;
    // nbcol = ncol % bcol == 0 ? ncol / bcol : (ncol + bcol) / bcol;
    // @TODO ncol is not a multiple of 16 (bcol = 8 * 2(kernel_ncol_blocks))
    n_ = (bShape.elements() - PACK16_PADDING - PACK16_SPECIALMEM) / sizeof(fbgemm::float16)
             / nbrow / brow;
    //size_t l = bShape.elements() / bShape[-1];
    //n_ = bShape[-1];
    //if(transB)
    //  std::swap(l, n_);
  }

  Shape newShape(Expr a, Shape bShape, bool transA) {
    auto shapeA = a->shape();
    if(transA) {
      shapeA.set(shapeA.size() - 2, a->shape()[shapeA.size() - 1]);
      shapeA.set(shapeA.size() - 1, a->shape()[shapeA.size() - 2]);
    }

    // @TODO remove transB
    //auto shapeB = bShape;
    //if(transB) {
    //  shapeB.set(shapeB.size() - 2, bShape[shapeB.size() - 1]);
    //  shapeB.set(shapeB.size() - 1, bShape[shapeB.size() - 2]);
    //}

    int nrow = shapeA[shapeA.size() - 1];
    //int ncol = transpose ? shape[0] : shape[1];
    //int kernel_ncol_blocks = 2;
    int brow = 512;
    //int bcol = 8 * kernel_ncol_blocks;
    //int last_brow = nrow % brow == 0 ? brow : nrow % brow;
    int nbrow = nrow % brow == 0 ? nrow / brow : (nrow + brow) / brow;
    //nbcol = ncol % bcol == 0 ? ncol / bcol : (ncol + bcol) / bcol;
    // @TODO ncol is not a multiple of 16 (bcol = 8 * 2(kernel_ncol_blocks))

    auto n = (bShape.elements() - PACK16_PADDING - PACK16_SPECIALMEM) / sizeof(fbgemm::float16)
             / nbrow / brow;
    // std::cout << "!!!!!!!!!!affine out shape: " << n << std::endl;

    Shape outShape = shapeA;
    outShape.set(outShape.size() - 1, n);
//    outShape.set(outShape.size() - 1, shapeB[shapeB.size() - 1]);
    //ABORT_IF(shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2],
    //         "Matrix product requires inner dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() override {
    return {
      NodeOp(GemmPackFp16(val_,
                          child(0)->val(),
                          child(1)->val(),
                          child(2)->val(),
                          m_,
                          n_,
                          k_,
                          transA_,
                          transB_))
    };
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "fp16packed"; }
};

// Affine transform (matrix multiplication) using packed B matrix
// Especially, this gemm performs quantized gemms in 8-bit integers.
// float scalar_: scalar multiplier
// size_t m_: the number of rows in A and C
// size_t n_: the number of columns in B and C
// size_t k_: the number of columns in A and the number of rows in C
// bool transA_: transpose A
// bool transB_: transpose B
class AffineNodeOpInt8 : public NaryNodeOp {
private:
  float scalar_;
  size_t m_;
  size_t n_;
  size_t k_;
  bool transA_;
  bool transB_;

public:
  AffineNodeOpInt8(const std::vector<Expr>& nodes, Shape bShape, bool transA, bool transB, float scalar)
      : NaryNodeOp(nodes, newShape(nodes[0], bShape, transA), Type::float32),
        scalar_(scalar) {
    transA_ = transA;
    transB_ = transB;
    m_ = nodes[0]->shape().elements() / nodes[0]->shape()[-1];
    k_ = nodes[0]->shape().back();
    if(transA)
      std::swap(m_, k_);

    int KCB;
    if (fbgemmHasAvx512Support()) {
      KCB = PackingTraits<int8_t, int32_t, inst_set_t::avx512>::KCB;
    } else {
      // AVX2
      KCB = PackingTraits<int8_t, int32_t, inst_set_t::avx2>::KCB;
    }

    ABORT_IF(transB, "B should not be transposed for packed GEMM.");
    int brow = ((k_ + KCB - 1) / KCB) * KCB;

    // @TODO col should be multiple of NCB (32)
    n_ = bShape.elements() / (brow + sizeof(float) + sizeof(int32_t) + sizeof(int32_t));
    //size_t l = bShape.elements() / bShape[-1];
    //n_ = bShape[-1];
    //if(transB)
    //  std::swap(l, n_);
  }

  Shape newShape(Expr a, Shape bShape, bool transA) {
    auto shapeA = a->shape();
    if(transA) {
      shapeA.set(shapeA.size() - 2, a->shape()[shapeA.size() - 1]);
      shapeA.set(shapeA.size() - 1, a->shape()[shapeA.size() - 2]);
    }

    int KCB;
    if (fbgemmHasAvx512Support()) {
      KCB = PackingTraits<int8_t, int32_t, inst_set_t::avx512>::KCB;
    } else {
      // AVX2
      KCB = PackingTraits<int8_t, int32_t, inst_set_t::avx2>::KCB;
    }

    int brow = ((shapeA[shapeA.size() - 1] + KCB - 1) / KCB) * KCB;

    // @TODO col should be multiple of NCB (32)
    int n = bShape.elements() / (brow + sizeof(float) + sizeof(int32_t) + sizeof(int32_t));
    // std::cout << "affine m_: " << m_ << std::endl;
    // std::cout << "affine KCB: " << KCB << std::endl;
    // std::cout << "affine brow: " << brow << std::endl;
    // std::cout << "affine bShape.elements(): " << bShape.elements() << std::endl;
    // std::cout << "affine out shape: " << n << std::endl;

    Shape outShape = shapeA;
    outShape.set(outShape.size() - 1, n);
    return outShape;
  }

  NodeOps forwardOps() override {
    return {
      NodeOp(GemmPackInt8(val_,
                          child(0)->val(),
                          child(1)->val(),
                          m_,
                          n_,
                          k_,
                          transA_,
                          transB_);
              AddBias(val_, child(2)->val()))
    };
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "int8packed"; }
};

// Dot product (matrix multiplication) using packed B matrix
// Especially, this GEMM packs and saves weights into fp16.
// Actual computation is done in fp32.
// float scalar_: scalar multiplier
// size_t m_: the number of rows in A and C
// size_t n_: the number of columns in B and C
// size_t k_: the number of columns in A and the number of rows in C
// bool transA_: transpose A
// bool transB_: transpose B
class DotNodeOpFp16 : public NaryNodeOp {
private:
  float scalar_;
  size_t m_;
  size_t n_;
  size_t k_;
  bool transA_;
  bool transB_;

public:
  DotNodeOpFp16(const std::vector<Expr>& nodes, Shape bShape, bool transA, bool transB, float scalar)
      : NaryNodeOp(nodes, newShape(nodes[0], bShape, transA), Type::float32),
        scalar_(scalar) {
    transA_ = transA;
    transB_ = transB;
    m_ = nodes[0]->shape().elements() / nodes[0]->shape()[-1];
    k_ = nodes[0]->shape().back();
    if(transA)
      std::swap(m_, k_);

    ABORT_IF(transB, "B should not be transposed for packed GEMM.");
    int nrow = (int) k_;
    // int ncol = transpose ? shape[0] : shape[1];
    // int kernel_ncol_blocks = 2;
    int brow = 512;
    // int bcol = 8 * kernel_ncol_blocks;
    // int last_brow = nrow % brow == 0 ? brow : nrow % brow;
    int nbrow = nrow % brow == 0 ? nrow / brow : (nrow + brow) / brow;
    // nbcol = ncol % bcol == 0 ? ncol / bcol : (ncol + bcol) / bcol;
    // @TODO ncol is not a multiple of 16 (bcol = 8 * 2(kernel_ncol_blocks))
    n_ = (bShape.elements() - PACK16_PADDING - PACK16_SPECIALMEM) / sizeof(fbgemm::float16)
             / nbrow / brow;
    //size_t l = bShape.elements() / bShape[-1];
    //n_ = bShape[-1];
    //if(transB)
    //  std::swap(l, n_);
  }

  Shape newShape(Expr a, Shape bShape, bool transA) {
    auto shapeA = a->shape();
    if(transA) {
      shapeA.set(shapeA.size() - 2, a->shape()[shapeA.size() - 1]);
      shapeA.set(shapeA.size() - 1, a->shape()[shapeA.size() - 2]);
    }

    // @TODO remove transB
    //auto shapeB = bShape;
    //if(transB) {
    //  shapeB.set(shapeB.size() - 2, bShape[shapeB.size() - 1]);
    //  shapeB.set(shapeB.size() - 1, bShape[shapeB.size() - 2]);
    //}

    int nrow = shapeA[shapeA.size() - 1];
    //int ncol = transpose ? shape[0] : shape[1];
    //int kernel_ncol_blocks = 2;
    int brow = 512;
    //int bcol = 8 * kernel_ncol_blocks;
    //int last_brow = nrow % brow == 0 ? brow : nrow % brow;
    int nbrow = nrow % brow == 0 ? nrow / brow : (nrow + brow) / brow;
    //nbcol = ncol % bcol == 0 ? ncol / bcol : (ncol + bcol) / bcol;
    // @TODO ncol is not a multiple of 16 (bcol = 8 * 2(kernel_ncol_blocks))

    auto n = (bShape.elements() - PACK16_PADDING - PACK16_SPECIALMEM) / sizeof(fbgemm::float16)
             / nbrow / brow;
    // std::cout << "!!!!!!!dot out shape: " << n << std::endl;

    Shape outShape = shapeA;
    outShape.set(outShape.size() - 1, n);
//    outShape.set(outShape.size() - 1, shapeB[shapeB.size() - 1]);
    //ABORT_IF(shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2],
    //         "Matrix product requires inner dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() override {
    return {
      NodeOp(GemmPackFp16(val_,
                          child(0)->val(),
                          child(1)->val(),
                          nullptr,
                          m_,
                          n_,
                          k_,
                          transA_,
                          transB_))
    };
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "fp16packeddot"; }
};

// Dot product (matrix multiplication) using packed B matrix
// Especially, this gemm performs quantized gemms in 8-bit integers.
// float scalar_: scalar multiplier
// size_t m_: the number of rows in A and C
// size_t n_: the number of columns in B and C
// size_t k_: the number of columns in A and the number of rows in C
// bool transA_: transpose A
// bool transB_: transpose B
class DotNodeOpInt8 : public NaryNodeOp {
private:
  float scalar_;
  size_t m_;
  size_t n_;
  size_t k_;
  bool transA_;
  bool transB_;

public:
  DotNodeOpInt8(const std::vector<Expr>& nodes, Shape bShape, bool transA, bool transB, float scalar)
      : NaryNodeOp(nodes, newShape(nodes[0], bShape, transA, transB), Type::float32),
        scalar_(scalar) {
    transA_ = transA;
    transB_ = transB;
    m_ = nodes[0]->shape().elements() / nodes[0]->shape()[-1];
    k_ = nodes[0]->shape().back();
    if(transA)
      std::swap(m_, k_);

    int KCB;
    if (fbgemmHasAvx512Support()) {
      KCB = PackingTraits<int8_t, int32_t, inst_set_t::avx512>::KCB;
    } else {
      // AVX2
      KCB = PackingTraits<int8_t, int32_t, inst_set_t::avx2>::KCB;
    }

    ABORT_IF(transB, "B should not be transposed for packed GEMM.");
    int brow = ((k_ + KCB - 1) / KCB) * KCB;

    // @TODO col should be multiple of NCB (32)
    n_ = bShape.elements() / (brow + sizeof(float) + sizeof(int32_t) + sizeof(int32_t));
    //size_t l = bShape.elements() / bShape[-1];
    //n_ = bShape[-1];
    //if(transB)
  }

  Shape newShape(Expr a, Shape bShape, bool transA, bool transB) {
    auto shapeA = a->shape();
    if(transA) {
      shapeA.set(shapeA.size() - 2, a->shape()[shapeA.size() - 1]);
      shapeA.set(shapeA.size() - 1, a->shape()[shapeA.size() - 2]);
    }

    int KCB;
    if (fbgemmHasAvx512Support()) {
      KCB = PackingTraits<int8_t, int32_t, inst_set_t::avx512>::KCB;
    } else {
      // AVX2
      KCB = PackingTraits<int8_t, int32_t, inst_set_t::avx2>::KCB;
    }

    int brow = ((shapeA[shapeA.size() - 1] + KCB - 1) / KCB) * KCB;

    // @TODO col should be multiple of NCB (32)
    int n = bShape.elements() / (brow + sizeof(float) + sizeof(int32_t) + sizeof(int32_t));
    // std::cout << "dot out shape: " << n << std::endl;

    Shape outShape = shapeA;
    outShape.set(outShape.size() - 1, n);
    return outShape;
  }

  NodeOps forwardOps() override {
    return {
      NodeOp(GemmPackInt8(val_,
                          child(0)->val(),
                          child(1)->val(),
                          m_,
                          n_,
                          k_,
                          transA_,
                          transB_))
    };
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "int8packeddot"; }
};

static inline Expr affine(GemmType gemmType, Expr a, Expr b, Shape bShape, Expr c, bool transA, bool transB, float scalar) {
  std::vector<Expr> nodes = {a, b, c};

  if (gemmType == GemmType::FbFp16Packed)
    return Expression<cpu::variant::AffineNodeOpFp16>(nodes, bShape, transA, transB, scalar);
  else if (gemmType == GemmType::FbInt8Packed)
    return Expression<cpu::variant::AffineNodeOpInt8>(nodes, bShape, transA, transB, scalar);
  else {
    ABORT("Only int8 and fp16 are available. {}", gemmType);
    return nullptr;
  }
}

static inline Expr pack(GemmType gemmType, Expr a, PackMatrix packMat, bool transpose, float clipValue) {
  if (gemmType == GemmType::FbFp16Packed)
    return Expression<cpu::variant::PackNodeOpFp16>(a, packMat, transpose, clipValue);
  else if (gemmType == GemmType::FbInt8Packed)
    return Expression<cpu::variant::PackNodeOpInt8>(a, packMat, transpose, clipValue);
  else {
    ABORT("Only int8 and fp16 are available. {}", gemmType);
    return nullptr;
  }
}

static inline Expr dot(GemmType gemmType, Expr a, Expr b, Shape bShape, bool transA, bool transB, float scalar) {
  std::vector<Expr> nodes = {a, b};

  if (gemmType == GemmType::FbFp16Packed)
    return Expression<cpu::variant::DotNodeOpFp16>(nodes, bShape, transA, transB, scalar);
  else if (gemmType == GemmType::FbInt8Packed)
    return Expression<cpu::variant::DotNodeOpInt8>(nodes, bShape, transA, transB, scalar);
  else {
    ABORT("Only int8 and fp16 are available. {}", gemmType);
    return nullptr;
  }
}

}  // namespace variant
}  // namespace cpu
}  // namespace marian
