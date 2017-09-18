#include <memory>
#include <stdexcept>
#include "kernels/sparse.h"

#if MKL_FOUND
namespace marian {

namespace sparse {

namespace cpu {

void multiply(Ptr<CSR_CPU> C, const Ptr<CSR_CPU> A, const Ptr<CSR_CPU> B, bool transA, bool transB) {
  /* MKL does not support sparse result of sparse matrix product. There are
   * libraries which do (e.g. SPARSEKIT) which should be used if/when this
   * implementation is insufficient.
   */

  if (transB) {
    // FIXME: Only not a problem as currently used.
    throw std::runtime_error("operation not currently supported");
  }

  char trans = transA ? 'T' : 'N';

  MKL_INT m = A->rows();
  MKL_INT n = A->cols();
  MKL_INT k = B->cols();

  float* a = A->values();
  MKL_INT* ja = A->colIndices();
  MKL_INT* ia = A->rowIndices();

  float* b = B->values();
  MKL_INT* jb = B->colIndices();
  MKL_INT* ib = B->rowIndices();

  float* c = new float[m*n];
  MKL_INT ldc = trans == 'N' ? m : n;

  mkl_scsrmultd(&trans, &m, &n, &k, a, ja, ia, b, jb, ib, c, &ldc);

  auto c_mem = New<MemoryPiece>(reinterpret_cast<uint8_t*>(c), C->rows()*C->cols() * sizeof(float));
  Tensor C_dense(new TensorCPU(c_mem, { C->rows(), C->cols() }, 0));
  *C = std::move(CSR_CPU(C_dense));
}

// TODO: Avoid duplication with sparse.cu
void LfaForward(Tensor out, Tensor logits, Tensor att, Ptr<CSR_CPU> sparseLf) {
  int batch = att->shape()[0];
  int srcWords = att->shape()[2];
  int trgWords = att->shape()[3];

  std::vector<float> values;
  att->get(values);
  int nonzeros = values.size();
  std::vector<std::tuple<int, int, float>> coo;
  for(size_t i = 0; i < nonzeros; ++i) {
    int r = (i % batch) + (i / (srcWords * batch)) * batch;
    int c = i % (srcWords * batch);
    UTIL_THROW_IF2(r >= trgWords * batch, "Row index too large");
    UTIL_THROW_IF2(c >= srcWords * batch, "Column index too large");
    coo.emplace_back(r, c, values[i]);
  }
  std::sort(coo.begin(), coo.end());
  values.clear();
  values.resize(nonzeros);
  std::vector<int> rowInd(nonzeros);
  std::vector<int> colInd(nonzeros);
  for(int i = 0; i < nonzeros; ++i) {
    rowInd[i] = std::get<0>(coo[i]);
    colInd[i] = std::get<1>(coo[i]);
    values[i] = std::get<2>(coo[i]);
  }

  auto sparseAtt = New<CSR_CPU>(batch * trgWords,
                            batch * srcWords,
                            values,
                            rowInd,
                            colInd,
                            out->getDevice());

  auto sparseLfa
      = New<CSR_CPU>(sparseAtt->rows(), sparseLf->cols(), out->getDevice());
  multiply(sparseLfa, sparseAtt, sparseLf);

  sparseLfa->toTensor(out);
}

static void CollapseAtt(Tensor out_, Tensor in_) {
  int nonzeros = out_->shape().elements();
  int batch = out_->shape()[0];
  int srcWords = out_->shape()[2];

  float* out = out_->data();
  const float* in = in_->data();
  for (int index = 0; index < nonzeros; ++index) {
    int r = (index % batch) + (index / (srcWords * batch)) * batch;
    int c = index % (srcWords * batch);
    float val = in[r * srcWords * batch + c];
    out[index] += val;
  }
}

void LfaBackward(Tensor gradAtt, Tensor adj, Ptr<CSR_CPU> sparseLf) {
  int batch = gradAtt->shape()[0];
  int srcWords = gradAtt->shape()[2];
  int trgWords = gradAtt->shape()[3];
  int nonzeros = gradAtt->shape().elements();

  int dimTrgVoc = adj->shape()[1];

  size_t exSize = batch * srcWords * batch * trgWords;
  float* expandAttGradBuffer = new float[exSize];

  char transa = 'N';

  MKL_INT m = sparseLf->rows();
  MKL_INT n = batch * trgWords;
  MKL_INT k = sparseLf->cols();

  float alpha = 1.f;

  char matdescra[6] = { 0 };
  matdescra[0] = 'G'; // General matrix (vs. e.g. triangular)
  matdescra[3] = 'F'; // using one-based indexing

  float* val = sparseLf->values();
  MKL_INT* indx = sparseLf->colIndices();
  MKL_INT* pntrb = sparseLf->rowIndices();
  MKL_INT* pntre = pntrb + 1;

  float* b = adj->data();
  MKL_INT ldb = dimTrgVoc;

  float beta = 0.f;

  float* c = expandAttGradBuffer;
  MKL_INT ldc = batch * srcWords;

  mkl_scsrmm(&transa, &m, &n, &k, &alpha, matdescra, val, indx, pntrb, pntre, b, &ldb,
      &beta, c, &ldc);

  auto mem = New<MemoryPiece>(reinterpret_cast<uint8_t*>(expandAttGradBuffer), exSize * sizeof(float));
  Tensor expandAttGrad(new TensorCPU(mem, { batch*trgWords, batch*srcWords }, 0));
  CollapseAtt(gradAtt, expandAttGrad);
}

}

}

}
#endif

#if MKL_FOUND || CUDA_FOUND
namespace marian {

namespace sparse {

void multiply(Ptr<CSR> C, const Ptr<CSR> A, const Ptr<CSR> B, bool transA, bool transB) {
  #if MKL_FOUND
  if (C->residency == DEVICE_CPU) {
    cpu::multiply(std::static_pointer_cast<CSR_CPU>(C), std::static_pointer_cast<CSR_CPU>(A),
        std::static_pointer_cast<CSR_CPU>(B), transA, transB);
  }
  #endif

  #if CUDA_FOUND
  if (C->residency == DEVICE_GPU) {
    gpu::multiply(std::static_pointer_cast<CSR_GPU>(C), std::static_pointer_cast<CSR_GPU>(A),
        std::static_pointer_cast<CSR_GPU>(B), transA, transB);
  }
  #endif
}

void LfaForward(Tensor out, Tensor logits, Tensor att, Ptr<CSR> sparseLf) {
  #if MKL_FOUND
  if (out->residency == DEVICE_CPU) {
    cpu::LfaForward(out, logits, att, std::static_pointer_cast<CSR_CPU>(sparseLf));
  }
  #endif

  #if CUDA_FOUND
  if (C->residency == DEVICE_GPU) {
    gpu::LfaForward(out, logits, att, std::static_pointer_cast<CSR_GPU>(sparseLf));
  }
  #endif
}

void LfaBackward(Tensor grad, Tensor adj, Ptr<CSR> sparseLf) {
  #if MKL_FOUND
  if (grad->residency == DEVICE_CPU) {
    cpu::LfaBackward(grad, adj, std::static_pointer_cast<CSR_CPU>(sparseLf));
  }
  #endif

  #if CUDA_FOUND
  if (C->residency == DEVICE_GPU) {
    gpu::LfaBackward(grad, adj, std::static_pointer_cast<CSR_GPU>(sparseLf));
  }
  #endif
}

}

}
#endif
