#include "tensors/gpu/integer_tools.h"
#include "cutlass/gemm/device/gemm.h"

namespace marian {

namespace gpu {
    
namespace integer {

    /**************************CUTLASS code begins here***********************/
    inline std::string cutlassGetErrorString(cutlass::Status& status) {
        switch (status) {
            case cutlass::Status::kSuccess:
            return "Operation was successful.";
            case cutlass::Status::kErrorMisalignedOperand:
            return "Operands fail alignment requirements.";
            case cutlass::Status::kErrorInvalidLayout:
            return "Layout fails alignment requirement.";
            case cutlass::Status::kErrorInvalidProblem:
            return "Specified problem size is not supported by operator.";
            case cutlass::Status::kErrorNotSupported:
            return "Operation is not supported on current device.";
            case cutlass::Status::kErrorWorkspaceNull:
            return "The given workspace is null when it is required to be non-null";
            case cutlass::Status::kErrorInternal:
            return "An error within CUTLASS occurred.";
            case cutlass::Status::kInvalid:
            return "Status is unspecified.";
        }
        return "Unknown CUTLASS status. Update this section of the code.";
    }
    
    #define CUTLASS_CHECK(expr) do {                                                                        \
        cutlass::Status rc = (expr);                                                                        \
        ABORT_IF(rc != cutlass::Status::kSuccess,                                                           \
                    "Cutlass Error: {} - {}:{}: {}", cutlassGetErrorString(rc), __FILE__, __LINE__, #expr); \
        } while(0)
    
    /*Cutlass matrices*/
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using ColumnMajorT = cutlass::layout::RowMajor; //Transposing in cutlass is done by changing the input from RowMajor to ColumnMajor. Care of the output
    //using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemmTT = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                    ColumnMajorT,  // Layout of A matrix
                                                    int8_t,        // Data-type of B matrix
                                                    ColumnMajorT,  // Layout of B matrix
                                                    int32_t,        // Data-type of C matrix
                                                    ColumnMajor>; // Layout of C matrix

    using CutlassGemmNT = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                    ColumnMajor,  // Layout of A matrix
                                                    int8_t,        // Data-type of B matrix
                                                    ColumnMajorT,  // Layout of B matrix
                                                    int32_t,        // Data-type of C matrix
                                                    ColumnMajor>; // Layout of C matrix

    using CutlassGemmTN = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                    ColumnMajorT,  // Layout of A matrix
                                                    int8_t,        // Data-type of B matrix
                                                    ColumnMajor,  // Layout of B matrix
                                                    int32_t,        // Data-type of C matrix
                                                    ColumnMajor>; // Layout of C matrix

    using CutlassGemmNN = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                    ColumnMajor,  // Layout of A matrix
                                                    int8_t,        // Data-type of B matrix
                                                    ColumnMajor,  // Layout of B matrix
                                                    int32_t,        // Data-type of C matrix
                                                    ColumnMajor>; // Layout of C matrix


    cutlass::Status cutlass_igemm_nn(bool transA, bool transB,
                        int M,
                        int N,
                        int K,
                        float alpha,
                        int8_t const *A,
                        int lda,
                        int8_t const *B,
                        int ldb,
                        float beta,
                        int32_t *C,
                        int ldc) {

        if (!transA && !transB) {
            CutlassGemmNN gemm_operator;
            CutlassGemmNN::Arguments args({M , N, K},  // Gemm Problem dimensions
                {A, lda},    // Tensor-ref for source matrix A
                {B, ldb},    // Tensor-ref for source matrix B
                {C, ldc},    // Tensor-ref for source matrix C
                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                {alpha, beta}); // Scalars used in the Epilogue
            return gemm_operator(args);
        } else if (transA && !transB) {
            CutlassGemmTN gemm_operator;
            CutlassGemmTN::Arguments args({M , N, K},  // Gemm Problem dimensions
                {A, lda},    // Tensor-ref for source matrix A
                {B, ldb},    // Tensor-ref for source matrix B
                {C, ldc},    // Tensor-ref for source matrix C
                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                {alpha, beta}); // Scalars used in the Epilogue
            return gemm_operator(args);
        } else if (!transA && transB) {
            CutlassGemmNT gemm_operator;
            CutlassGemmNT::Arguments args({M , N, K},  // Gemm Problem dimensions
                {A, lda},    // Tensor-ref for source matrix A
                {B, ldb},    // Tensor-ref for source matrix B
                {C, ldc},    // Tensor-ref for source matrix C
                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                {alpha, beta}); // Scalars used in the Epilogue
            return gemm_operator(args);
        } else { // Final case (transA && transB)
            CutlassGemmTT gemm_operator;
            CutlassGemmTT::Arguments args({M , N, K},  // Gemm Problem dimensions
                {A, lda},    // Tensor-ref for source matrix A
                {B, ldb},    // Tensor-ref for source matrix B
                {C, ldc},    // Tensor-ref for source matrix C
                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                {alpha, beta}); // Scalars used in the Epilogue
            return gemm_operator(args);
        }
    }

    /**************************CUTLASS code begins here***********************/

    __global__ void findMaxMinAndQuantMult(const float * input_gpu, int idxMax, int idxMin, float * output) {
        float absMax = abs(input_gpu[idxMax]);
        float absMin = abs(input_gpu[idxMin]);
        if (absMax > absMin) {
            output[0] = 127.0f/absMax;
        } else {
            output[0] = 127.0f/absMin;
        }
    }
    
    //@TODO rewrite with a nice singlePass GPU version that uses shared memory
    void maxAbsQuantMult(cublasHandle_t& handle, const float * input_gpu, size_t items, float * output_gpu) {
        //Get Max Absolute:
        int resMaxIdx;
        CUBLAS_CHECK(cublasIsamax(handle, items, input_gpu, 1, &resMaxIdx));
        int resMinIdx;
        CUBLAS_CHECK(cublasIsamin(handle, items, input_gpu, 1, &resMinIdx));

        findMaxMinAndQuantMult<<<1,1>>>(input_gpu, resMaxIdx - 1, resMinIdx - 1, output_gpu); //FUCK YOU FORTRAN INDEXING
    }

    __global__ void quantize(const float * input, int8_t * output, size_t items, const float * quantMultAddr) {
        const float quantMult = *quantMultAddr; //@TODO ask nvidia if this is the most efficient way to do this here
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < items) {
            output[x] = (int8_t)llrintf((input[x]*quantMult));
        }
    }

    __global__ void dequantize(const int8_t * input, float * output, size_t items, const float * dequantMultAddr) {
        const float dequantMult = *dequantMultAddr; //@TODO ask nvidia if this is the most efficient way to do this here
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < items)
            output[x] = ((float)input[x])*dequantMult;
    }

    void quantize(const float * input, int8_t * output, size_t rows, size_t cols, const float * quantMultAddr) {
        // Make sure we're not running out of threads here.
        int threads = rows;
        int blocks = cols;

        if (threads > 512) {
            std::swap(threads, blocks);
            if (threads > 512) {
                blocks = (int)ceil((threads*blocks)/512);
                threads = 512;
            }
        }
        quantize<<<blocks, threads>>>(input, output, rows*cols, quantMultAddr);
    }

    void dequantize(const int8_t * input, float * output, size_t rows, size_t cols, const float * dequantMultAddr) {
        // Make sure we're not running out of threads here.
        int threads = rows;
        int blocks = cols;

        if (threads > 512) {
            std::swap(threads, blocks);
            if (threads > 512) {
                blocks = (int)ceil((threads*blocks)/512);
                threads = 512;
            }
        }
        dequantize<<<blocks, threads>>>(input, output, rows*cols, dequantMultAddr);
    }

} // namespace integer
} // namespace gpu
} // namespace marian