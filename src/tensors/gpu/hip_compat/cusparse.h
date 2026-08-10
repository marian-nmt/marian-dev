#pragma once
// HIP shim: <cusparse.h> -> hipSPARSE. Aliases the generic-API cuSPARSE
// symbols Marian's prod_sparse_cu11.h and backend.h use.
#include <hipsparse/hipsparse.h>
#include <hip/library_types.h> // hipDataType / HIP_R_* used as cudaDataType

using cusparseHandle_t     = hipsparseHandle_t;
using cusparseStatus_t     = hipsparseStatus_t;
using cusparseSpMatDescr_t = hipsparseSpMatDescr_t;
using cusparseDnMatDescr_t = hipsparseDnMatDescr_t;
using cusparseSpMMAlg_t    = hipsparseSpMMAlg_t;

#define CUSPARSE_STATUS_SUCCESS         HIPSPARSE_STATUS_SUCCESS
#define CUSPARSE_INDEX_32I              HIPSPARSE_INDEX_32I
#define CUSPARSE_INDEX_BASE_ZERO        HIPSPARSE_INDEX_BASE_ZERO
#define CUSPARSE_OPERATION_NON_TRANSPOSE HIPSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_OPERATION_TRANSPOSE    HIPSPARSE_OPERATION_TRANSPOSE
#define CUSPARSE_ORDER_COL              HIPSPARSE_ORDER_COL
#define CUSPARSE_ORDER_ROW              HIPSPARSE_ORDER_ROW
#define CUSPARSE_SPMM_ALG_DEFAULT       HIPSPARSE_SPMM_ALG_DEFAULT
#define CUSPARSE_SPMM_CSR_ALG2          HIPSPARSE_SPMM_CSR_ALG2

#define cusparseCreate          hipsparseCreate
#define cusparseDestroy         hipsparseDestroy
#define cusparseCreateCsr       hipsparseCreateCsr
#define cusparseCreateDnMat     hipsparseCreateDnMat
#define cusparseDestroySpMat    hipsparseDestroySpMat
#define cusparseDestroyDnMat    hipsparseDestroyDnMat
#define cusparseSpMM            hipsparseSpMM
#define cusparseSpMM_bufferSize hipsparseSpMM_bufferSize
