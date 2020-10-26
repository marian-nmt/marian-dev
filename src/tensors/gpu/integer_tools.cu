#include "tensors/gpu/integer_tools.h"
#include "tensors/gpu/cuda_helpers.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"

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
    using ElementOutput = float;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;
    /*TensorOp matrices*/
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>;  // <- threadblock tile M = 128, N = 256, K = 64
    // This code section describes tile size a warp will compute
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 64 
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;  // <- MMA Op tile M = 8, N = 8, K = 16


    // This code section describes the epilogue part of the kernel
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
                                                  ElementOutput, // <- data type of output matrix
                                                  128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
                                                  ElementAccumulator, // <- data type of accumulator
                                                  ElementCompute>;  // <- data type for alpha/beta in linear combination function

    using EpilogueOpRelu = cutlass::epilogue::thread::LinearCombinationRelu<
                                                      ElementOutput, // <- data type of output matrix
                                                      128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                                // memory access. For a byte, it's 16
                                                                // elements. This becomes the vector width of
                                                                // math instructions in the epilogue too
                                                      ElementAccumulator, // <- data type of accumulator
                                                      ElementCompute>;  // <- data type for alpha/beta in linear combination function

    using CutlassGemmTensorOp = cutlass::gemm::device::Gemm<int8_t,                            // ElementA
                                                            cutlass::layout::RowMajor,         // LayoutA
                                                            int8_t,                            // ElementB
                                                            cutlass::layout::ColumnMajor,      // LayoutB
                                                            float,                             // ElementOutput
                                                            cutlass::layout::ColumnMajor,      // LayoutOutput
                                                            int32_t,                           // ElementAccumulator
                                                            cutlass::arch::OpClassTensorOp,    // tag indicating Tensor Cores
                                                            cutlass::arch::Sm75,               // tag indicating target GPU compute architecture //@TODO this should change, probably
                                                            ShapeMMAThreadBlock,
                                                            ShapeMMAWarp,
                                                            ShapeMMAOp,
                                                            EpilogueOp>;
    using CutlassGemmTensorOpRelu = cutlass::gemm::device::Gemm<int8_t,                            // ElementA
                                                               cutlass::layout::RowMajor,         // LayoutA
                                                               int8_t,                            // ElementB
                                                               cutlass::layout::ColumnMajor,      // LayoutB
                                                               float,                             // ElementOutput
                                                               cutlass::layout::ColumnMajor,      // LayoutOutput
                                                               int32_t,                           // ElementAccumulator
                                                               cutlass::arch::OpClassTensorOp,    // tag indicating Tensor Cores
                                                               cutlass::arch::Sm75,               // tag indicating target GPU compute architecture //@TODO this should change, probably
                                                               ShapeMMAThreadBlock,
                                                               ShapeMMAWarp,
                                                               ShapeMMAOp,
                                                               EpilogueOpRelu>;
    /*Non TensorOp matrices*/
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;
    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 16>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
    using Epilogue = cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                                  1, /*@TODO should be something different? like 32/64/128?*/
                                                                  ElementAccumulator,
                                                                  ElementCompute>;

    using EpilogueRelu = cutlass::epilogue::thread::LinearCombinationRelu<ElementOutput,
                                                                  1, /*@TODO should be something different? like 32/64/128?*/
                                                                  ElementAccumulator,
                                                                  ElementCompute>;

    using ColumnMajor = cutlass::layout::ColumnMajor;
    using ColumnMajorT = cutlass::layout::RowMajor; //Transposing in cutlass is done by changing the input from RowMajor to ColumnMajor. Care of the output
    //using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemmTT = cutlass::gemm::device::Gemm<int8_t,       // Data-type of A matrix
                                                    ColumnMajorT,   // Layout of A matrix
                                                    int8_t,         // Data-type of B matrix
                                                    ColumnMajorT,   // Layout of B matrix
                                                    float,          // Data-type of C matrix
                                                    ColumnMajor,    // Layout of C matrix
                                                    int32_t,        // Accumulator
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm75,
                                                    ThreadBlockShape,
                                                    WarpShape,
                                                    InstructionShape,
                                                    Epilogue
                                                    >;
    using CutlassGemmNT = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                    ColumnMajor,  // Layout of A matrix
                                                    int8_t,        // Data-type of B matrix
                                                    ColumnMajorT,  // Layout of B matrix
                                                    float,        // Data-type of C matrix
                                                    ColumnMajor, // Layout of C matrix
                                                    int32_t,        // Accumulator
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm75,
                                                    ThreadBlockShape,
                                                    WarpShape,
                                                    InstructionShape,
                                                    Epilogue
                                                    >;

    using CutlassGemmTN = cutlass::gemm::device::Gemm<int8_t,       // Data-type of A matrix
                                                    ColumnMajorT,   // Layout of A matrix
                                                    int8_t,         // Data-type of B matrix
                                                    ColumnMajor,    // Layout of B matrix
                                                    float,          // Data-type of C matrix
                                                    ColumnMajor,    // Layout of C matrix
                                                    int32_t,        // Accumulator
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm75,
                                                    ThreadBlockShape,
                                                    WarpShape,
                                                    InstructionShape,
                                                    Epilogue
                                                    >;

    using CutlassGemmNN = cutlass::gemm::device::Gemm<int8_t,      // Data-type of A matrix
                                                    ColumnMajor,   // Layout of A matrix
                                                    int8_t,        // Data-type of B matrix
                                                    ColumnMajor,   // Layout of B matrix
                                                    float,         // Data-type of C matrix
                                                    ColumnMajor,   // Layout of C matrix
                                                    int32_t,       // Accumulator
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm75,
                                                    ThreadBlockShape,
                                                    WarpShape,
                                                    InstructionShape,
                                                    Epilogue
                                                    >;

    using CutlassGemmTTRelu = cutlass::gemm::device::Gemm<int8_t,       // Data-type of A matrix
                                                    ColumnMajorT,   // Layout of A matrix
                                                    int8_t,         // Data-type of B matrix
                                                    ColumnMajorT,   // Layout of B matrix
                                                    float,          // Data-type of C matrix
                                                    ColumnMajor,    // Layout of C matrix
                                                    int32_t,        // Accumulator
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm75,
                                                    ThreadBlockShape,
                                                    WarpShape,
                                                    InstructionShape,
                                                    EpilogueRelu
                                                    >;
    using CutlassGemmNTRelu = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                    ColumnMajor,  // Layout of A matrix
                                                    int8_t,        // Data-type of B matrix
                                                    ColumnMajorT,  // Layout of B matrix
                                                    float,        // Data-type of C matrix
                                                    ColumnMajor, // Layout of C matrix
                                                    int32_t,        // Accumulator
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm75,
                                                    ThreadBlockShape,
                                                    WarpShape,
                                                    InstructionShape,
                                                    EpilogueRelu
                                                    >;

    using CutlassGemmTNRelu = cutlass::gemm::device::Gemm<int8_t,       // Data-type of A matrix
                                                    ColumnMajorT,   // Layout of A matrix
                                                    int8_t,         // Data-type of B matrix
                                                    ColumnMajor,    // Layout of B matrix
                                                    float,          // Data-type of C matrix
                                                    ColumnMajor,    // Layout of C matrix
                                                    int32_t,        // Accumulator
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm75,
                                                    ThreadBlockShape,
                                                    WarpShape,
                                                    InstructionShape,
                                                    EpilogueRelu
                                                    >;

    using CutlassGemmNNRelu = cutlass::gemm::device::Gemm<int8_t,      // Data-type of A matrix
                                                    ColumnMajor,   // Layout of A matrix
                                                    int8_t,        // Data-type of B matrix
                                                    ColumnMajor,   // Layout of B matrix
                                                    float,         // Data-type of C matrix
                                                    ColumnMajor,   // Layout of C matrix
                                                    int32_t,       // Accumulator
                                                    cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm75,
                                                    ThreadBlockShape,
                                                    WarpShape,
                                                    InstructionShape,
                                                    EpilogueRelu
                                                    >;

    /*Non-Epilogue functions, as they are faster (for now)*/
    using CutlassGemmTensorOpunfused = cutlass::gemm::device::Gemm<int8_t,                         // ElementA
                                                                cutlass::layout::RowMajor,         // LayoutA
                                                                int8_t,                            // ElementB
                                                                cutlass::layout::ColumnMajor,      // LayoutB
                                                                int32_t,                           // ElementOutput
                                                                cutlass::layout::ColumnMajor,      // LayoutOutput
                                                                int32_t,                           // ElementAccumulator
                                                                cutlass::arch::OpClassTensorOp,    // tag indicating Tensor Cores
                                                                cutlass::arch::Sm75>;

    using CutlassGemmTTunfused = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                            ColumnMajorT,   // Layout of A matrix
                                                            int8_t,         // Data-type of B matrix
                                                            ColumnMajorT,   // Layout of B matrix
                                                            int32_t,        // Data-type of C matrix
                                                            ColumnMajor,    // Layout of C matrix
                                                            int32_t>;       // Accumulator

    using CutlassGemmNTunfused = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                            ColumnMajor,    // Layout of A matrix
                                                            int8_t,         // Data-type of B matrix
                                                            ColumnMajorT,   // Layout of B matrix
                                                            int32_t,        // Data-type of C matrix
                                                            ColumnMajor,    // Layout of C matrix
                                                            int32_t>;       // Accumulator

    using CutlassGemmTNunfused = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                            ColumnMajorT,   // Layout of A matrix
                                                            int8_t,         // Data-type of B matrix
                                                            ColumnMajor,    // Layout of B matrix
                                                            int32_t,        // Data-type of C matrix
                                                            ColumnMajor,    // Layout of C matrix
                                                            int32_t>;       // Accumulator

    using CutlassGemmNNunfused = cutlass::gemm::device::Gemm<int8_t,        // Data-type of A matrix
                                                            ColumnMajor,    // Layout of A matrix
                                                            int8_t,         // Data-type of B matrix
                                                            ColumnMajor,    // Layout of B matrix
                                                            int32_t,        // Data-type of C matrix
                                                            ColumnMajor,    // Layout of C matrix
                                                            int32_t>;       // Accumulator

    cutlass::Status cutlass_igemm_nn(bool transA, bool transB,
                        int M,
                        int N,
                        int K,
                        float * alpha,
                        int8_t const *A,
                        int lda,
                        int8_t const *B,
                        int ldb,
                        float * beta,
                        float *C,
                        int ldc,
                        bool tensorCore,
                        bool fused,
                        float * bias,
                        bool doRelu) {
        float * Csrc;
        int ldcSRC;
        if (bias) { /* This is only available for the fused option. Beta needs to be 1? */
            Csrc = bias;
            ldcSRC = 0; /*Having a stride of 0 enables bias broadcast*/
        } else {
            Csrc = C;
            ldcSRC = ldc;
        }
        if (fused) {
            if (doRelu) {
                if (tensorCore) {
                    CutlassGemmTensorOpRelu gemm_operator;
                    CutlassGemmTensorOpRelu::Arguments args({M, N, K},  // Gemm Problem dimensions
                        {A, lda},       // Tensor-ref for source matrix A
                        {B, ldb},       // Tensor-ref for source matrix B
                        {Csrc, ldcSRC}, // Tensor-ref for source matrix C
                        {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                        {alpha, beta}); // Scalars used in the Epilogue
                    return gemm_operator(args);
                } else {
                    if (!transA && !transB) {
                        CutlassGemmNNRelu gemm_operator;
                        CutlassGemmNNRelu::Arguments args({M, N, K},  // Gemm Problem dimensions
                            {A, lda},       // Tensor-ref for source matrix A
                            {B, ldb},       // Tensor-ref for source matrix B
                            {Csrc, ldcSRC}, // Tensor-ref for source matrix C
                            {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                            {alpha, beta}); // Scalars used in the Epilogue
                        return gemm_operator(args);
                    } else if (transA && !transB) {
                        CutlassGemmTNRelu gemm_operator;
                        CutlassGemmTNRelu::Arguments args({M, N, K},  // Gemm Problem dimensions
                            {A, lda},       // Tensor-ref for source matrix A
                            {B, ldb},       // Tensor-ref for source matrix B
                            {Csrc, ldcSRC}, // Tensor-ref for source matrix C
                            {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                            {alpha, beta}); // Scalars used in the Epilogue
                        return gemm_operator(args);
                    } else if (!transA && transB) {
                        CutlassGemmNTRelu gemm_operator;
                        CutlassGemmNTRelu::Arguments args({M, N, K},  // Gemm Problem dimensions
                            {A, lda},       // Tensor-ref for source matrix A
                            {B, ldb},       // Tensor-ref for source matrix B
                            {Csrc, ldcSRC}, // Tensor-ref for source matrix C
                            {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                            {alpha, beta}); // Scalars used in the Epilogue
                        return gemm_operator(args);
                    } else { // Final case (transA && transB)
                        CutlassGemmTTRelu gemm_operator;
                        CutlassGemmTTRelu::Arguments args({M, N, K},  // Gemm Problem dimensions
                            {A, lda},       // Tensor-ref for source matrix A
                            {B, ldb},       // Tensor-ref for source matrix B
                            {Csrc, ldcSRC}, // Tensor-ref for source matrix C
                            {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                            {alpha, beta}); // Scalars used in the Epilogue
                        return gemm_operator(args);
                    }
                }
            } else {
                if (tensorCore) {
                    CutlassGemmTensorOp gemm_operator;
                    CutlassGemmTensorOp::Arguments args({M, N, K},  // Gemm Problem dimensions
                        {A, lda},       // Tensor-ref for source matrix A
                        {B, ldb},       // Tensor-ref for source matrix B
                        {Csrc, ldcSRC}, // Tensor-ref for source matrix C
                        {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                        {alpha, beta}); // Scalars used in the Epilogue
                    return gemm_operator(args);
                } else {
                    if (!transA && !transB) {
                        CutlassGemmNN gemm_operator;
                        CutlassGemmNN::Arguments args({M, N, K},  // Gemm Problem dimensions
                            {A, lda},       // Tensor-ref for source matrix A
                            {B, ldb},       // Tensor-ref for source matrix B
                            {Csrc, ldcSRC}, // Tensor-ref for source matrix C
                            {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                            {alpha, beta}); // Scalars used in the Epilogue
                        return gemm_operator(args);
                    } else if (transA && !transB) {
                        CutlassGemmTN gemm_operator;
                        CutlassGemmTN::Arguments args({M, N, K},  // Gemm Problem dimensions
                            {A, lda},       // Tensor-ref for source matrix A
                            {B, ldb},       // Tensor-ref for source matrix B
                            {Csrc, ldcSRC}, // Tensor-ref for source matrix C
                            {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                            {alpha, beta}); // Scalars used in the Epilogue
                        return gemm_operator(args);
                    } else if (!transA && transB) {
                        CutlassGemmNT gemm_operator;
                        CutlassGemmNT::Arguments args({M, N, K},  // Gemm Problem dimensions
                            {A, lda},       // Tensor-ref for source matrix A
                            {B, ldb},       // Tensor-ref for source matrix B
                            {Csrc, ldcSRC}, // Tensor-ref for source matrix C
                            {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                            {alpha, beta}); // Scalars used in the Epilogue
                        return gemm_operator(args);
                    } else { // Final case (transA && transB)
                        CutlassGemmTT gemm_operator;
                        CutlassGemmTT::Arguments args({M, N, K},  // Gemm Problem dimensions
                            {A, lda},       // Tensor-ref for source matrix A
                            {B, ldb},       // Tensor-ref for source matrix B
                            {Csrc, ldcSRC}, // Tensor-ref for source matrix C
                            {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                            {alpha, beta}); // Scalars used in the Epilogue
                        return gemm_operator(args);
                    }
                }
            }
        } else {
            static const int32_t constexpr alpha_int = 1;
            static const int32_t constexpr beta_int = 0;
            if (tensorCore) {
                CutlassGemmTensorOpunfused gemm_operator;
                CutlassGemmTensorOpunfused::Arguments args({M, N, K},  // Gemm Problem dimensions
                    {A, lda},    // Tensor-ref for source matrix A
                    {B, ldb},    // Tensor-ref for source matrix B
                    {(int32_t *)C, ldc},    // Tensor-ref for source matrix C
                    {(int32_t *)C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                    {alpha_int, beta_int}); // Scalars used in the Epilogue
                return gemm_operator(args);
            } else {
                if (!transA && !transB) {
                    CutlassGemmNNunfused gemm_operator;
                    CutlassGemmNNunfused::Arguments args({M, N, K},  // Gemm Problem dimensions
                        {A, lda},    // Tensor-ref for source matrix A
                        {B, ldb},    // Tensor-ref for source matrix B
                        {(int32_t *)C, ldc},    // Tensor-ref for source matrix C
                        {(int32_t *)C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                        {alpha_int, beta_int}); // Scalars used in the Epilogue
                    return gemm_operator(args);
                } else if (transA && !transB) {
                    CutlassGemmTNunfused gemm_operator;
                    CutlassGemmTNunfused::Arguments args({M, N, K},  // Gemm Problem dimensions
                        {A, lda},    // Tensor-ref for source matrix A
                        {B, ldb},    // Tensor-ref for source matrix B
                        {(int32_t *)C, ldc},    // Tensor-ref for source matrix C
                        {(int32_t *)C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                        {alpha_int, beta_int}); // Scalars used in the Epilogue
                    return gemm_operator(args);
                } else if (!transA && transB) {
                    CutlassGemmNTunfused gemm_operator;
                    CutlassGemmNTunfused::Arguments args({M, N, K},  // Gemm Problem dimensions
                        {A, lda},    // Tensor-ref for source matrix A
                        {B, ldb},    // Tensor-ref for source matrix B
                        {(int32_t *)C, ldc},    // Tensor-ref for source matrix C
                        {(int32_t *)C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                        {alpha_int, beta_int}); // Scalars used in the Epilogue
                    return gemm_operator(args);
                } else { // Final case (transA && transB)
                    CutlassGemmTTunfused gemm_operator;
                    CutlassGemmTTunfused::Arguments args({M, N, K},  // Gemm Problem dimensions
                        {A, lda},    // Tensor-ref for source matrix A
                        {B, ldb},    // Tensor-ref for source matrix B
                        {(int32_t *)C, ldc},    // Tensor-ref for source matrix C
                        {(int32_t *)C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                        {alpha_int, beta_int}); // Scalars used in the Epilogue
                    return gemm_operator(args);
                }
            }
        }
    }
    void cutlass_igemm_dispatcher(bool transA, bool transB,
        int M,
        int N,
        int K,
        float * alpha,
        int8_t const *A,
        int lda,
        int8_t const *B,
        int ldb,
        float * beta,
        int32_t *C,
        int ldc,
        bool tensorCore,
        bool fused,
        float * bias,
        bool doRelu) {
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
                (float *)C,
                ldc,
                tensorCore,
                fused,
                bias,
                doRelu));
        CUDA_CHECK(cudaGetLastError()); // Sometimes CUTLASS errors manifest as CUDA errors.
    }
    /**************************CUTLASS code ends here***********************/
    __global__ void getMaxAbsKernel(const float * input_gpu, int idxMax, int idxMin, float * output) {
        float absMax = abs(input_gpu[idxMax]);
        float absMin = abs(input_gpu[idxMin]);
        if (absMax > absMin) {
            output[0] = absMax;
        } else {
            output[0] = absMin;
        }
    }

    float getMaxAbs(cublasHandle_t& handle, const float * input_gpu, size_t items) {
        // Allocate memory on the GPU
        float * output_gpu;
        CUDA_CHECK(cudaMallocManaged(&output_gpu, sizeof(float)));

        //Get Max Absolute:
        int resMaxIdx;
        CUBLAS_CHECK(cublasIsamax(handle, items, input_gpu, 1, &resMaxIdx));
        int resMinIdx;
        CUBLAS_CHECK(cublasIsamin(handle, items, input_gpu, 1, &resMinIdx));

        getMaxAbsKernel<<<1,1>>>(input_gpu, resMaxIdx - 1, resMinIdx - 1, output_gpu); //FUCK YOU FORTRAN INDEXING
        CUDA_CHECK(cudaDeviceSynchronize()); // We need to synchronise in order to use the managed memory

        float ret = *output_gpu;
        cudaFree(output_gpu);
        return ret;
    }

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
            output[x] = (int8_t)max(-128, min(127, (int)rintf(share[i]*quantMult)));
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

    __global__ void getDequantMult(float * output, float * quantMultAaddr, float * quantMultBaddr) {
        const float aQuantMult = *quantMultAaddr;
        const float bQuantMult = *quantMultBaddr;
        *output = 1.0f/(aQuantMult*bQuantMult);
    }

    void getDequantMultWrapper(float * output, float * quantMultAaddr, float * quantMultBaddr) {
        getDequantMult<<<1,1>>>(output, quantMultAaddr, quantMultBaddr);
    }

    __global__ void dequantize(const int32_t * input, float * output, size_t items, const float * quantMultAaddr, const float * quantMultBaddr) {
        const float aQuantMult = *quantMultAaddr;
        const float bQuantMult = *quantMultBaddr;
        const float dequantMult = 1.0f/(aQuantMult*bQuantMult); //@TODO ask nvidia if this is the most efficient way to do this here
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        int i = threadIdx.x;
        __shared__ int32_t share[256]; // Not sure if shared memory is necessary here to take advnatage of globale memory burst
        if (x < items) {
            share[i] = input[x];
            output[x] = ((float)share[i])*dequantMult;
        }
    }

    void dequantize(const int32_t * input, float * output, size_t rows, size_t cols, const float * quantMultAaddr, const float * quantMultBaddr) {
        // Make sure we're not running out of threads here.
        int threads = 256;
        int blocks = (int)ceil(rows*cols/256);

        dequantize<<<blocks, threads>>>(input, output, rows*cols, quantMultAaddr, quantMultBaddr);
        CUDA_CHECK(cudaGetLastError()); // Get errors from kernel launches
    }

    __global__ void dequantize(const int32_t * input, float * output, size_t items, const float * dequantMultAddr) {
        const float dequantMult = *dequantMultAddr; //@TODO ask nvidia if this is the most efficient way to do this here
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        int i = threadIdx.x;
        __shared__ int32_t share[256]; // Not sure if shared memory is necessary here to take advnatage of global memory burst
        if (x < items) {
            share[i] = input[x];
            output[x] = ((float)share[i])*dequantMult;
        }
    }

    void dequantize(const int32_t * input, float * output, size_t rows, size_t cols, const float * dequantMultAddr) {
        // Make sure we're not running out of threads here.
        int threads = 256;
        int blocks = (int)ceil(rows*cols/256);

        dequantize<<<blocks, threads>>>(input, output, rows*cols, dequantMultAddr);
        CUDA_CHECK(cudaGetLastError()); // Get errors from kernel launches
    }

    __global__ void meanStdkern(float * input, size_t elems, float * mean, float * stddev, float * absMean, float * absStddev, float * normal_sum, float * squares_sum, float * abs_normal_sum) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        float * global_sums[3];

        global_sums[0] = normal_sum;
        global_sums[1] = squares_sum;
        global_sums[2] = abs_normal_sum;

        // Initiate shared memory
        __shared__ float shared_sums[3];
        float * normal_sum_share = &shared_sums[0];
        float * squares_sum_share = &shared_sums[1];
        float * abs_normal_sum_share = &shared_sums[2];
        if (threadIdx.x < 3) {
            shared_sums[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute sums
        if (i < elems) {
            float current = input[i];
            atomicAdd(normal_sum_share, current);
            if (!isfinite(*normal_sum_share)) {
                //printf("nan\\inf detected at line 517, sum %f current %f\n", *normal_sum_share, current);
            }
            atomicAdd(abs_normal_sum_share, fabs(current));
            if (!isfinite(*abs_normal_sum_share)) {
                //printf("nan\\inf detected at line 521, sum %f current %f\n", *abs_normal_sum_share, current);
            }
            atomicAdd(squares_sum_share, current*current);
            if (!isfinite(*squares_sum_share)) {
                //printf("nan\\inf detected at line 525, sum %f current %f\n", *squares_sum_share, current);
            }
        }
        __syncthreads();
        // The first three threads in each block should write to the global_memory
        if (threadIdx.x < 3) {
            atomicAdd(global_sums[threadIdx.x], shared_sums[threadIdx.x]);
            if (!isfinite(*global_sums[threadIdx.x])) {
                //printf("nan\\inf detected at line 533, i is: %d \n", (int)threadIdx.x);
            }
        }
        __syncthreads();

        // Write the results to global memory
        if (i == 0) {
            *mean = (*normal_sum)/elems;
            *stddev = sqrtf(((*squares_sum)/elems) - ((*mean)*(*mean)));
            if (!isfinite(((*mean)*(*mean)))) {
                //printf("nan\\inf detected at line 543\n");
            }
        } else if (i == 1) {
            *absMean = (*abs_normal_sum)/elems;
            *absStddev = sqrtf(((*squares_sum)/elems) - ((*absMean)*(*absMean)));
            if (!isfinite(((*absMean)*(*absMean)))) {
                //printf("nan\\inf detected at line 549\n");
            }
        }
    }

    MeanStd getMeanStd(float * input, size_t elems) {
        MeanStd ret;
        // Allocate GPU memory. Use CudaMallocManaged to avoid copy to CPU memory after
        float * mean;
        float * stddev;
        float * absMean;
        float * absStddev;
        float * normal_sum;
        float * squares_sum;
        float * abs_normal_sum;
        CUDA_CHECK(cudaMallocManaged(&mean, sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&stddev, sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&absMean, sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&absStddev, sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&normal_sum, sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&squares_sum, sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&abs_normal_sum, sizeof(float)));

        *normal_sum = 0.0f;
        *squares_sum = 0.0f;
        *abs_normal_sum = 0.0f;

        // GPU kernel run
        int threads = 256;
        int blocks = (int)ceil(elems/256);
        meanStdkern<<<blocks, threads>>>(input, elems, mean, stddev, absMean, absStddev, normal_sum, squares_sum, abs_normal_sum);
        CUDA_CHECK(cudaDeviceSynchronize()); // Synchronizes GPU and CPU memory

        // copy to the ret object
        ret.mean = *mean;
        ret.stddev = *stddev;
        ret.absMean = *absMean;
        ret.absStddev =  *absStddev;

        // Free the memory
        cudaFree(mean);
        cudaFree(stddev);
        cudaFree(absMean);
        cudaFree(absStddev);
        cudaFree(normal_sum);
        cudaFree(squares_sum);
        cudaFree(abs_normal_sum);

        return ret;
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

    void memCpyHost(float * dest, float * source, size_t elems) {
        CUDA_CHECK(cudaMemcpy(dest, source, elems*sizeof(float), cudaMemcpyDeviceToHost));
    }

    void memCpyHost(int8_t * dest, int8_t * source, size_t elems) {
        CUDA_CHECK(cudaMemcpy(dest, source, elems*sizeof(int8_t), cudaMemcpyDeviceToHost));
    }

    void fieldSetGPU(float * gpuMem, float value) {
        float src = value;
        CUDA_CHECK(cudaMemcpy(gpuMem, &src, 1*sizeof(float), cudaMemcpyHostToDevice));
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