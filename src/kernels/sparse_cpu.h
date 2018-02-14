/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include <memory>
#include <numeric>

#if MKL_FOUND
#include <mkl.h>

namespace marian {

namespace sparse {

class CSR_CPU : public CSR {
  std::vector<int> rowIndices_;
  std::vector<MKL_INT> colIndices_;
  std::vector<float> values_;

public:
  CSR_CPU(int rows, int cols, size_t device)
    : CSR(rows, cols, device, DEVICE_CPU) {}

  CSR_CPU(int rows, int cols, const std::vector<float>& values,
      const std::vector<int>& rowIndices, const std::vector<int>& colIndices,
      size_t device)
    : CSR(rows, cols, values, rowIndices, colIndices, device, DEVICE_CPU) {
    MKL_INT job[] {
      2, // job[0]: COO -> CSR, with CSR column index sorted in increasing order
      1, // job[1]: CSR representation uses one-based indexing
      0, // job[2]: COO representation uses zero-based indexing
      0, // job[3]: Reserved
      0, // job[4]: Only relevant for CSR -> COO
      0, // job[5]: Populate each of acsr, ja, and ia
    };

    MKL_INT n = rows;
    MKL_INT nnz = nnz_;

    values_.resize(nnz);
    float* acsr = &values_[0];

    colIndices_.resize(nnz);
    MKL_INT* ja = &colIndices_[0];

    rowIndices_.resize(n + 1);
    MKL_INT* ia = &rowIndices_[0];

    float* acoo = const_cast<float*>(&values[0]);
    MKL_INT* rowind = const_cast<MKL_INT*>(&rowIndices[0]);
    MKL_INT* colind = const_cast<MKL_INT*>(&colIndices[0]);

    MKL_INT info; // Only relevant for CSR -> COO
    mkl_scsrcoo(job, &n, acsr, ja, ia, &nnz, acoo, rowind, colind, &info);
  }

  CSR_CPU(Tensor dense)
    : CSR(dense, DEVICE_CPU) {
    int nnz = 0;
    for (int i = 0; i < rows_*cols_; ++i) {
      nnz += dense->data()[i] != 0.f;
    }

    MKL_INT job[] {
      0,   // job[0]: Dense -> CSR
      0,   // job[1]: Dense representation uses zero-based indexing (i.e. row-major)
      1,   // job[2]: CSR representation uses one-based indexing
      2,   // job[3]: adns is not triangular
      nnz, // job[4]: number of non-zero elements
      1    // job[5]: Populate each of acsr, ja, and ia
    };

    MKL_INT m = rows_;
    MKL_INT n = cols_;

    float* adns = dense->data();

    MKL_INT lda = n;

    values_.resize(nnz);
    float* acsr = &values_[0];

    colIndices_.resize(nnz);
    MKL_INT* ja = &colIndices_[0];

    rowIndices_.resize(m + 1);
    MKL_INT* ia = &rowIndices_[0];

    MKL_INT info; // Only relevant for CSR -> Dense
    mkl_sdnscsr(job, &m, &n, adns, &lda, acsr, ja, ia, &info);
  }

  CSR_CPU& operator=(CSR_CPU&& o) {
    nnz_ = o.nnz_;
    rows_ = o.rows_;
    cols_ = o.cols_;
    device_ = o.device_;

    rowIndices_ = std::move(o.rowIndices_);
    colIndices_ = std::move(o.colIndices_);
    values_ = std::move(o.values_);

    return *this;
  }

  void toTensor(Tensor dense) {
    MKL_INT job[] {
      1, // job[0]: CSR -> Dense
      0, // job[1]: Dense representation uses zero-based indexing (i.e. row-major)
      1, // job[2]: CSR representation uses one-based indexing
      2, // job[3]: adns is not triangular
      0, // job[4]: Only relevant to Dense -> CSR
      0  // job[5]: Only relevant to Dense -> CSR
    };

    MKL_INT m = rows_;
    MKL_INT n = cols_;

    float* adns = dense->data();

    MKL_INT lda = n;

    float* acsr = &values_[0];
    MKL_INT* ja = &colIndices_[0];
    MKL_INT* ia = &rowIndices_[0];

    MKL_INT info;
    mkl_sdnscsr(job, &m, &n, adns, &lda, acsr, ja, ia, &info);
  }

  float* values() { return &values_[0]; }
  int* rowIndices() { return &colIndices_[0]; } // n.b. 1-based
  int* colIndices() { return &rowIndices_[0]; } // n.b. 1-based

  std::string debug() {
    size_t size = sizeof(float) * rows() * cols();
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[size]);
    auto mem = New<MemoryPiece>(buffer.get(), size);
    Tensor tensor(new TensorCPU(mem, { rows(), cols() }, device_));
    toTensor(tensor);
    return tensor->debug();
  }
};

namespace cpu {

void multiply(
    Ptr<CSR_CPU>, const Ptr<CSR_CPU>, const Ptr<CSR_CPU>, bool = false, bool = false);

void LfaForward(Tensor out, Tensor logits, Tensor att, Ptr<CSR_CPU> sparseLf);

void LfaBackward(Tensor grad, Tensor adj, Ptr<CSR_CPU> sparseLf);

}

}

}
#endif
