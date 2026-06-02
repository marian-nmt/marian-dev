#pragma once
// MOAT HIP shim: <cublasLt.h> -> hipBLASLt. Aliases the cublasLt symbols
// Marian's affine+bias+ReLU epilogue (prod.cpp) uses. The MIN_ALIGNMENT_*_BYTES
// preference attributes do not exist in hipBLASLt and are dropped at the call
// site under USE_HIP, not aliased here.
#include <hipblaslt/hipblaslt.h>

using cublasLtHandle_t              = hipblasLtHandle_t;
using cublasLtMatmulDesc_t          = hipblasLtMatmulDesc_t;
using cublasLtMatrixLayout_t        = hipblasLtMatrixLayout_t;
using cublasLtMatmulPreference_t    = hipblasLtMatmulPreference_t;
using cublasLtMatmulHeuristicResult_t = hipblasLtMatmulHeuristicResult_t;
using cublasLtEpilogue_t            = hipblasLtEpilogue_t;

#define CUBLASLT_EPILOGUE_DEFAULT   HIPBLASLT_EPILOGUE_DEFAULT
#define CUBLASLT_EPILOGUE_BIAS      HIPBLASLT_EPILOGUE_BIAS
#define CUBLASLT_EPILOGUE_RELU_BIAS HIPBLASLT_EPILOGUE_RELU_BIAS

#define CUBLASLT_MATMUL_DESC_TRANSA        HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB        HIPBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_DESC_EPILOGUE      HIPBLASLT_MATMUL_DESC_EPILOGUE
#define CUBLASLT_MATMUL_DESC_BIAS_POINTER  HIPBLASLT_MATMUL_DESC_BIAS_POINTER

#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES

#define cublasLtMatmulDescCreate           hipblasLtMatmulDescCreate
#define cublasLtMatmulDescDestroy          hipblasLtMatmulDescDestroy
#define cublasLtMatmulDescSetAttribute     hipblasLtMatmulDescSetAttribute
#define cublasLtMatrixLayoutCreate         hipblasLtMatrixLayoutCreate
#define cublasLtMatrixLayoutDestroy        hipblasLtMatrixLayoutDestroy
#define cublasLtMatmulPreferenceCreate     hipblasLtMatmulPreferenceCreate
#define cublasLtMatmulPreferenceDestroy    hipblasLtMatmulPreferenceDestroy
#define cublasLtMatmulPreferenceSetAttribute hipblasLtMatmulPreferenceSetAttribute
#define cublasLtMatmulAlgoGetHeuristic     hipblasLtMatmulAlgoGetHeuristic
#define cublasLtMatmul                     hipblasLtMatmul
