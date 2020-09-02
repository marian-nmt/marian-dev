#include "tensors/gpu/integer_tools.h"
#include "tensors/gpu/cuda_helpers.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/cutlass.h"

namespace marian {

namespace gpu {
    
namespace integer {

    /**************************CUTLASS code begins here***********************/
    inline const char * cutlassGetErrorString(cutlass::Status& status) {
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
    /*TensorOp matrices*/
    using CutlassGemmTensorOp = cutlass::gemm::device::Gemm<
                                                            int8_t,                            // ElementA
                                                            cutlass::layout::RowMajor,         // LayoutA
                                                            int8_t,                            // ElementB
                                                            cutlass::layout::ColumnMajor,      // LayoutB
                                                            int32_t,                           // ElementOutput
                                                            cutlass::layout::ColumnMajor,      // LayoutOutput
                                                            int32_t,                           // ElementAccumulator
                                                            cutlass::arch::OpClassTensorOp,    // tag indicating Tensor Cores
                                                            cutlass::arch::Sm75                // tag indicating target GPU compute architecture //@TODO this should change, probably
                                                            >;
    /*Non TensorOp matrices*/
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using ColumnMajorT = cutlass::layout::RowMajor; //Transposing in cutlass is done by changing the input from RowMajor to ColumnMajor. Care of the output
    //using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemmTT = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                    ColumnMajorT,  // Layout of A matrix
                                                    int8_t,        // Data-type of B matrix
                                                    ColumnMajorT,  // Layout of B matrix
                                                    int32_t,        // Data-type of C matrix
                                                    ColumnMajor>; // Layout of C matrix
                                                    //int32_t, // Accumulator
                                                    //cutlass::arch::OpClassTensorOp, //TensorOp
                                                    //cutlass::arch::Sm75>;  //SMmodel
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
                        int ldc,
                        bool tensorCore) {
        if (tensorCore) {
            CutlassGemmTensorOp gemm_operator;
            CutlassGemmTensorOp::Arguments args({M , N, K},  // Gemm Problem dimensions
                {A, lda},    // Tensor-ref for source matrix A
                {B, ldb},    // Tensor-ref for source matrix B
                {C, ldc},    // Tensor-ref for source matrix C
                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                {alpha, beta}); // Scalars used in the Epilogue
            return gemm_operator(args);
        } else {
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
    }
    void cutlass_igemm_dispatcher(bool transA, bool transB,
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
        int ldc,
        bool tensorCore) {
            CUTLASS_CHECK(cutlass_igemm_nn(transA, transB,
                M,
                N,
                K,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                tensorCore));
        CUDA_CHECK(cudaGetLastError()); // Sometimes CUTLASS errors manifest as CUDA errors.
    }
    /**************************CUTLASS code ends here***********************/

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
        int i = threadIdx.x;
        __shared__ float share[256]; // Not sure if shared memory is necessary here to take advnatage of globale memory burst
        if (x < items) {
            share[i] = input[x];
            output[x] = (int8_t)llrintf((share[i]*quantMult));
        }
    }

    void quantize(const float * input, int8_t * output, size_t rows, size_t cols, const float * quantMultAddr) {
        // Make sure we're not running out of threads here.
        int threads = 256;
        int blocks = (int)ceil(rows*cols/256);

        quantize<<<blocks, threads>>>(input, output, rows*cols, quantMultAddr);
        CUDA_CHECK(cudaGetLastError()); // Get errors from kernel launches
    }

    __global__ void quantizeToRowMajor(const float * input, int8_t * output, size_t rows, size_t cols, const float * quantMultAddr) {
        const float quantMult = *quantMultAddr; // @TODO ask nvidia if this is the most efficient way to do this here
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        //input is col major, output is row major
        if (row*col < rows*cols) {
            output[cols*row + col] = (int8_t)llrintf((input[rows*col + row]*quantMult));
        }
    }

    void quantizeToRowMajorWrapper(const float * input, int8_t * output, size_t rows, size_t cols, const float * quantMultAddr) {
        // Make sure we're not running out of threads here.

        dim3 dimBlock(32, 32);
        dim3 dimGrid(std::max(cols / dimBlock.x, 1ul), std::max(rows / dimBlock.y, 1ul));
        //dim3 dimGrid(std::max(cols / dimBlock.x, 2ul), std::max(rows / dimBlock.y, 2ul));

        quantizeToRowMajor<<<dimGrid, dimBlock>>>(input, output, rows, cols, quantMultAddr);
        CUDA_CHECK(cudaGetLastError()); // Get errors from kernel launches
    }

    __global__ void dequantize(const int32_t * input, float * output, size_t items, const float * quantMultAaddr, const float * quantMultBaddr) {
        const float aQuantMult = *quantMultAaddr;
        const float bQuantMult = *quantMultBaddr;
        const float dequantMult = 1.0f/(aQuantMult*bQuantMult); //@TODO ask nvidia if this is the most efficient way to do this here
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        //if (x == 0) {
        //    printf("Be4: %d\n", input[x]);
        //}
        if (x < items)
            output[x] = ((float)input[x])*dequantMult;
        //if (x == 0)
        //    printf("Amax %f Bmax %f dequant: %f, after: %f\n", aMaxAbs, bMaxAbs, dequantMult, output[x]);
    }

    void dequantize(const int32_t * input, float * output, size_t rows, size_t cols, const float * quantMultAaddr, const float * quantMultBaddr) {
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
        dequantize<<<blocks, threads>>>(input, output, rows*cols, quantMultAaddr, quantMultBaddr);
        CUDA_CHECK(cudaGetLastError()); // Get errors from kernel launches
    }

    __global__ void gpuPrinter(float * mem, size_t idx) {
        printf("Value at %d idx is %f\n", (int)idx, mem[idx]);
    }

    void gpuPrinterDispatch(float * mem, size_t idx) {
        gpuPrinter<<<1,1>>>(mem, idx);
    }

    __global__ void gpuPrinter(int32_t * mem, size_t idx) {
        printf("Value at %d idx is %d\n", (int)idx, (int)mem[idx]);
    }

    void gpuPrinterDispatch(int32_t * mem, size_t idx) {
        gpuPrinter<<<1,1>>>(mem, idx);
    }

    __global__ void gpuPrinter(int8_t * mem, size_t idx) {
        printf("Value at %d idx is %d\n", (int)idx, (int)mem[idx]);
    }

    void gpuPrinterDispatch(int8_t * mem, size_t idx) {
        gpuPrinter<<<1,1>>>(mem, idx);
    }

    void memCpyDevice(float * dest, float * source, size_t elems) {
        CUDA_CHECK(cudaMemcpy(dest, source, elems*sizeof(float), cudaMemcpyDeviceToDevice));
    }

    void memCpyDevice(int8_t * dest, int8_t * source, size_t elems) {
        CUDA_CHECK(cudaMemcpy(dest, source, elems*sizeof(int8_t), cudaMemcpyDeviceToDevice));
    }
/*
    float * unmanagedGPUAlloc(size_t num) {
        void * tmp;
        CUDA_CHECK(cudaMalloc(&tmp, num*sizeof(float)));
        return (float *)tmp;
    }

    void unmanagedFree(float * in) {
        cudaFree(in);
    }
*/
} // namespace integer
} // namespace gpu
} // namespace marian