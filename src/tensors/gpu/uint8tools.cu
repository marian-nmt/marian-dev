#include "tensors/gpu/uint8tools.h"
#include "tensors/gpu/cuda_helpers.h"
#include <cmath>
//#include "/home/dheart/uni_stuff/postdoc/cutlass/include/cutlass/gemm/device/gemm.h"

namespace marian {

namespace hacky8bit {
    template<class T>
    void sanityCheck(T * gpumem, size_t num_items, char typechar) {
        T * cpumem = new T[num_items];
        CUDA_CHECK(cudaMemcpy(cpumem, gpumem, num_items*sizeof(T), cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_items; i++) {
            if (cpumem[i] == 127) {
                //fprintf(stderr, "Error at %d, type %c\n", i, typechar);
            }
        }
    }

    static inline int cols(const Tensor& tensor) { return tensor->shape()[-1]; }
    static inline int rows(const Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }
/*
    static void unsetTensorMode(cublasHandle_t cublasHandle) {
        cublasHandle; // fool warnings
      #if CUDA_VERSION >= 9000
        CUBLAS_CHECK(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));
      #endif
      }
        
    static void setTensorMode(cublasHandle_t cublasHandle) {
        cublasHandle; // fool warnings
    #if CUDA_VERSION >= 9000
        static int mode = 0;  // 1: use TC; -1: do not use TC; 0: not set yet
        if (mode == 0) { // multi-thread note: this is sort-of thread-safe, since multiple threads would determine the same value
        const char* var = getenv("ENABLE_CUBLAS_TENSOR_OP_MATH_FP32");
        if (!var)
            var = "1";
        switch(var[0]) {
            case '0': mode = -1; break;
            case '1': mode =  1; break;
            default: ABORT("Invalid ENABLE_CUBLAS_TENSOR_OP_MATH_FP32={}", var);
        }
        if (mode > 0) { // try whether it can be set   --@TODO: check whether this actually works
            CUBLAS_CHECK(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
            cublasMath_t actual = CUBLAS_DEFAULT_MATH;
            cublasGetMathMode(cublasHandle, &actual);
            if (actual != CUBLAS_TENSOR_OP_MATH) {
            LOG(warn, "[gpu] TensorCores requested but not available");
            mode = -1;
            }
        }
        if (mode > 0)
            LOG(info, "[gpu] 16-bit TensorCores enabled for float32 matrix operations");
        }
        CUBLAS_CHECK(cublasSetMathMode(cublasHandle, mode > 0 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
    #endif
    }*/

    __global__ void quantize(const float * input, int8_t * output, size_t items, float quantMult) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < items) {
            output[x] = (int8_t)llrintf((input[x]*quantMult));
            //printf("%d Input: %f, output %d\n", x, input[x], (int)output[x]);
        }
    }
    
    template<class intType>
    __global__ void dequantize(intType * input, float * output, size_t items, float dequantMult) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < items)
            output[x] = ((float)input[x])*dequantMult;
        //if (x == 0 || x == 1 || x ==3 ) {
        //    printf("Id %d  actual %f  mine: %f\n", x, output[x], ((float)input[x])*dequantMult);
        //    output[x] =  ((float)input[x])*dequantMult;
        //}
    }
    
    __global__ void findMaxMin(const float * input_gpu, int idxMax, int idxMin, float * output) {
        float absMax = abs(input_gpu[idxMax]);
        float absMin = abs(input_gpu[idxMin]);
        if (absMax > absMin) {
            output[0] = absMax;
        } else {
            output[0] = absMin;
        }
    }
    
    //@TODO rewrite with a nice singlePass GPU version that uses shared memory
    float maxAbs(cublasHandle_t& handle, const float * input_gpu, size_t items, float * scratchMem) {
        //Get Max Absolute:
        int resMaxIdx;
        CUBLAS_CHECK(cublasIsamax(handle, items, input_gpu, 1, &resMaxIdx));
        int resMinIdx;
        CUBLAS_CHECK(cublasIsamin(handle, items, input_gpu, 1, &resMinIdx));
        float * output_gpu;
        if (scratchMem) {
            output_gpu = scratchMem;
        } else {
            CUDA_CHECK(cudaMalloc(&output_gpu, 1*sizeof(float)));
        }
        findMaxMin<<<1,1>>>(input_gpu, resMaxIdx - 1, resMinIdx - 1, output_gpu); //FUCK YOU FORTRAN INDEXING
        float output;
        CUDA_CHECK(cudaMemcpy(&output, &output_gpu[0], 1*sizeof(float), cudaMemcpyDeviceToHost));
        if (!scratchMem) {
            CUDA_CHECK(cudaFree(output_gpu));
        }
        return output;
    }

    __global__ void findMaxMin( float * input_gpu, int idxMax, int idxMin, float * output) {
        float absMax = abs(input_gpu[idxMax]);
        float absMin = abs(input_gpu[idxMin]);
        if (absMax > absMin) {
            output[0] = absMax;
        } else {
            output[0] = absMin;
        }
    }
    
    //@TODO rewrite with a nice singlePass GPU version that uses shared memory
    float maxAbs(cublasHandle_t& handle, float * input_gpu, size_t items, float * scratchMem) {
        //Get Max Absolute:
        int resMaxIdx;
        CUBLAS_CHECK(cublasIsamax(handle, items, input_gpu, 1, &resMaxIdx));
        int resMinIdx;
        CUBLAS_CHECK(cublasIsamin(handle, items, input_gpu, 1, &resMinIdx));
        float * output_gpu;
        if (scratchMem) {
            output_gpu = scratchMem;
        } else {
            CUDA_CHECK(cudaMalloc(&output_gpu, 1*sizeof(float)));
        }
        findMaxMin<<<1,1>>>(input_gpu, resMaxIdx - 1, resMinIdx - 1, output_gpu); //FUCK YOU FORTRAN INDEXING
        float output;
        CUDA_CHECK(cudaMemcpy(&output, &output_gpu[0], 1*sizeof(float), cudaMemcpyDeviceToHost));
        if (!scratchMem) {
            CUDA_CHECK(cudaFree(output_gpu));
        }
        return output;
    }

    cublasStatus_t cublas8bitGemmm(marian::Tensor& C,
        const marian::Tensor& A,
        const marian::Tensor& B,
        bool transA,
        bool transB,
        float beta,
        float scalar) {
        
        CUDA_CHECK(cudaSetDevice((int)C->getDeviceId().no));
        auto backend = std::static_pointer_cast<gpu::Backend>(C->getBackend());
        //uint8_t * scratch = nullptr; //backend->getScratchMem();

        auto cublasHandle = backend->getCublasHandle();
    
        //First, We need to convert our tensors to 8bit, and use our scratch memory, so we don't do 10000 cudaMallocs.
        //But first, get it to work the stupid way:
        int32_t alpha_int = static_cast<int32_t>(scalar);
        int32_t beta_int = static_cast<int32_t>(beta);
       // fprintf(stderr, "Inside: Alpha is %d, beta is: %d\n", alpha_int, beta_int);
        int8_t* in8bitIntA;
        int8_t* in8bitIntB;
        int32_t * out32bitInt;

        CUDA_CHECK(cudaMalloc(&out32bitInt, C->shape().elements()*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&in8bitIntA, A->shape().elements()*sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&in8bitIntB, B->shape().elements()*sizeof(int8_t)));

        //CUDA_CHECK(cudaMemset(in8bitIntB, 127, B->shape().elements()*sizeof(int8_t)));
        //CUDA_CHECK(cudaMemset(in8bitIntA, 127, A->shape().elements()*sizeof(int8_t)));
        //CUDA_CHECK(cudaMemset(out32bitInt, 127, C->shape().elements()*sizeof(int32_t)));
        //CUDA_CHECK(cudaMemset(C->data<float>(), 127, C->shape().elements()*sizeof(float)));


        //Quantize:
        float aMaxAbs = maxAbs(cublasHandle, A->data<float>(), A->shape().elements(), nullptr/*reinterpret_cast<float *>(&scratch[0])*/);
        float bMaxAbs = maxAbs(cublasHandle, B->data<float>(), B->shape().elements(), nullptr/*reinterpret_cast<float *>(&scratch[4])*/);
        CUDA_CHECK(cudaDeviceSynchronize());
        int rowsA = rows(A);
        int colsA = cols(A);
        int rowsB = rows(B);
        int colsB = cols(B);
        int rowsC = rows(C);
        int colsC = cols(C);
        if (colsA > 512) {
            std::swap(rowsA, colsA);
            if (colsA > 512) {
                fprintf(stderr, "Incompatible sizes: rows %d, cols %d\n", rowsA, colsA);
            }
        }

        if (colsB > 512) {
            std::swap(rowsB, colsB);
            if (colsB > 512) {
                fprintf(stderr, "Incompatible sizes: rows %d, cols %d\n", rowsB, colsB);
            }
        }

        if (colsC > 512) {
            std::swap(rowsC, colsC);
            if (colsC > 512) {
                fprintf(stderr, "Incompatible sizes: rows %d, cols %d\n", rowsC, colsC);
            }
        }
        quantize<<<rowsA, colsA>>>(A->data<float>(), in8bitIntA, A->shape().elements(), 127.0f/aMaxAbs);
        CUDA_CHECK(cudaGetLastError());
        quantize<<<rowsB, colsB>>>(B->data<float>(), in8bitIntB, B->shape().elements(), 127.0f/bMaxAbs);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        sanityCheck(in8bitIntA, A->shape().elements(), 'q');
        sanityCheck(in8bitIntB, B->shape().elements(), 'w');
        /*
        for (int i = 0; i< A->shape().elements(); i++) {
            quantize<<<1, 1>>>(A->data<float>() + i, in8bitIntA, A->shape().elements(), 127.0f/aMaxAbs);
        }

        for (int i = 0; i< B->shape().elements(); i++) {
            quantize<<<1, 1>>>(B->data<float>() + i, in8bitIntB, B->shape().elements(), 127.0f/bMaxAbs);
        }*/


        //Things we do to setup lda, ldb, ldc and then multiply
      
        int m = A->shape().elements() / A->shape().back();
        int k = A->shape().back();
        if(transA)
          std::swap(m, k);
      
        int l = B->shape().elements() / B->shape().back();
        int n = B->shape().back();
        if(transB)
          std::swap(l, n);
      
        int lda = A->shape().back();
        int ldb = B->shape().back();
        int ldc = B->shape().back();
      
        if(transB)
          ldc = B->shape().elements() / B->shape().back();
        
        if (n * m != rows(C)*cols(C)) {
            fprintf(stderr, "n %d, m %d, crows %d, ccols %d\n", n, m, rows(C), cols(C));
        }
        cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        //fprintf(stderr, "m %d k %d l %d n %d lda %d ldb %d ldc %d transa %d transb %d, alpha_i %d, beta_i %d\n", m, k, l, n, lda, ldb, ldc, transA, transB, alpha_int, beta_int);
        //setTensorMode(cublasHandle);
        auto res = cublasGemmEx(cublasHandle, /*1*/
            opB, /*2*/
            opA, /*3*/
            n, /*4*/
            m, /*5*/
            k, /*6*/
            &alpha_int, /*7*/
            in8bitIntB, /*8*/
            CUDA_R_8I, /*9*/
            ldb, /*10*/
            in8bitIntA, /*11*/
            CUDA_R_8I, /*12*/
            lda, /*13*/
            &beta_int, /*14*/
            out32bitInt, /*15*/
            CUDA_R_32I, /*16*/
            ldc, /*17*/
            CUDA_R_32I, /*18*/
            CUBLAS_GEMM_DEFAULT_TENSOR_OP); /*19*/
        CUBLAS_CHECK(res);
        CUDA_CHECK(cudaDeviceSynchronize());
        //unsetTensorMode(cublasHandle);
        //fprintf(stderr, "after\n");
        //Convert back to float into C
        //fprintf(stderr, "C0 %d C1 %d C0 %d C1 %d\n", C->shape()[0], C->shape()[1], C->shape().elements() / C->shape().back(), C->shape().back());
        //fprintf(stderr, "rowsC %d, colsC %d\n", rowsC, colsC);
        if (C->shape().elements() != rowsC*colsC) {
            fprintf(stderr, "rowsC %d, colsC %d\n", rowsC, colsC);
        }
        dequantize<<<rowsC, colsC>>>(out32bitInt, C->data<float>(), C->shape().elements(), (aMaxAbs/127.0f)*(bMaxAbs/127.0f) );
        CUDA_CHECK(cudaGetLastError());
        sanityCheck(C->data<float>(), C->shape().elements(), 'r');
        //for (int i = 0; i< C->shape().elements(); i++) {
        //    dequantize<<<1, 1>>>(out32bitInt +i, C->data<float>() +i, C->shape().elements(), (aMaxAbs/127.0f)*(bMaxAbs/127.0f) );
        //}
        CUDA_CHECK(cudaDeviceSynchronize());
        //Free temporary used memory
        CUDA_CHECK(cudaFree(out32bitInt));
        CUDA_CHECK(cudaFree(in8bitIntA));
        CUDA_CHECK(cudaFree(in8bitIntB));

        return res;
    }

    cublasStatus_t cublas8bitGemmmEx(cublasHandle_t handle,
        cublasOperation_t transa, 
        cublasOperation_t transb,
        int m, int n, int k,
        const float* alpha,
        const float* A, int lda,
        const float* B, int ldb,
        const float* beta,
        float* C, int ldc) {
            
        auto algorithm = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        if (m%4 == 0 && n % 4 == 0 && k % 4 ==0) {
            int BlockA = m;
            int ThreadsA = k;
            int BlockB = k;
            int ThreadsB = n;
            int BlockC = m;
            int ThreadsC = n;

            // Make sure that we have enough threads so that kernel launches don't fail
            if (ThreadsA > 512) {
                std::swap(BlockA, ThreadsA);
                if (ThreadsA > 512) {
                    fprintf(stderr, "Incompatible sizes for A: rows %d, cols %d\n", BlockA, ThreadsA);
                    BlockA = (int)ceil((ThreadsA*BlockA)/512);
                    ThreadsA = 512;
                }
            }
    
            if (ThreadsB > 512) {
                std::swap(BlockB, ThreadsB);
                if (ThreadsB > 512) {
                    fprintf(stderr, "Incompatible sizes for B: rows %d, cols %d\n", BlockB, ThreadsB);
                    BlockB = (int)ceil((ThreadsB*BlockB)/512);
                    ThreadsB = 512;
                }
            }
    
            if (ThreadsC > 512) {
                std::swap(BlockC, ThreadsC);
                if (ThreadsC > 512) {
                    fprintf(stderr, "Incompatible sizes for C: rows %d, cols %d\n", BlockC, ThreadsC);
                    BlockC = (int)ceil((ThreadsC*BlockC)/512);
                    ThreadsC = 512;
                }
            }

            int32_t alpha_int = static_cast<int32_t>(*alpha);
            int32_t beta_int = static_cast<int32_t>(*beta);
            int8_t* in8bitIntA;
            int8_t* in8bitIntB;
            int32_t * out32bitInt;

            CUDA_CHECK(cudaMalloc(&out32bitInt, m*n*sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&in8bitIntA, m*k*sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&in8bitIntB, k*n*sizeof(int8_t)));

            float aMaxAbs = maxAbs(handle, A, m*k, nullptr/*reinterpret_cast<float *>(&scratch[0])*/);
            float bMaxAbs = maxAbs(handle, B, k*n, nullptr/*reinterpret_cast<float *>(&scratch[4])*/);

            quantize<<<BlockA, ThreadsA>>>(A, in8bitIntA, m*k, 127.0f/aMaxAbs);
            quantize<<<BlockB, ThreadsB>>>(B, in8bitIntB, k*n, 127.0f/bMaxAbs);
            CUDA_CHECK(cudaDeviceSynchronize());

            auto res = cublasGemmEx(handle, transa, transb, 
                m, n, k, &alpha_int, 
                in8bitIntA, CUDA_R_8I, lda, 
                in8bitIntB, CUDA_R_8I, ldb, &beta_int, 
                out32bitInt, CUDA_R_32I, ldc,
                CUDA_R_32I, algorithm);
            
            dequantize<<<BlockC, ThreadsC>>>(out32bitInt, C, m*n, (aMaxAbs/127.0f)*(bMaxAbs/127.0f) );
            CUDA_CHECK(cudaDeviceSynchronize());
            //Free temporary used memory
            CUDA_CHECK(cudaFree(out32bitInt));
            CUDA_CHECK(cudaFree(in8bitIntA));
            CUDA_CHECK(cudaFree(in8bitIntB));
            return res;
        } else {
            auto res = cublasGemmEx(handle, transa, transb, 
                m, n, k, alpha, 
                A, CUDA_R_32F, lda, 
                B, CUDA_R_32F, ldb, beta, 
                C, CUDA_R_32F, ldc,
                CUDA_R_32F, algorithm);
            return res;
        }

    }


} // namespace hacky8bit
} //namespace marian