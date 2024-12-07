namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Optimized packed matrix multiplication function.
 *
 * @tparam IS Instruction set.
 * @tparam N Number of columns.
 * @tparam K Number of rows.
 * @param A Pointer to matrix A.
 * @param B Pointer to the packed matrix B.
 * @param bias Pointer to the bias matrix.
 * @param C Pointer to the result matrix C.
 * @param M The number of rows in matrix A and C.
 */
template <InstructionSet IS, int N=512, int K=512>
void gemmInt8Packed(const float* A, const int8_t* B, const float* bias, float* C, int M) {
  constexpr int MCB  = BlockingFactors<IS>::MCB;
  constexpr int NCB  = BlockingFactors<IS>::NCB;
  constexpr int KCB  = BlockingFactors<IS>::KCB;

  constexpr int RI   = BlockingFactors<IS>::RI;
  constexpr int VLEN = BlockingFactors<IS>::VLEN;

  static_assert(KCB % RI == 0);
  static_assert(K % KCB == 0);
  static_assert(N % NCB == 0);

  // aligned memory for quantized A
  alignas(VLEN * sizeof(int32_t)) thread_local uint8_t aQuant[MCB * KCB]; // 56 x 256 bytes ~ 14 KB - scratch space for quantized A

  const uint8_t* aPtrs[MCB];
  float*         cPtrs[MCB];
  int32_t*       cPtrsInt32[MCB];
  QuantizationParams quantParams[MCB];

  for(int rowBlockStart = 0; rowBlockStart < M; rowBlockStart += MCB) {
    int MCBActual = std::min(MCB, M - rowBlockStart);

    for(int m = 0; m < MCBActual; m++) {
      cPtrs[m]          = C + (rowBlockStart + m) * N;
      const float* aPtr = A + (rowBlockStart + m) * K;

      // compute zero point and scale for whole row
      computeQuantizationParams<IS, N, K>(aPtr, B, quantParams[m]);
    }

    for(int innerBlockStart = 0; innerBlockStart < K; innerBlockStart += KCB) {
      const bool initBlockWithZero = innerBlockStart == 0;

      for(int m = 0; m < MCBActual; m++) {
        const float* aPtr   = A + (rowBlockStart + m) * K + innerBlockStart;
        uint8_t* aQuantPtr = aQuant + m * KCB;

        // quantize the input and accumulate the sum in quantParams
        quantize<IS, KCB>(aPtr, aQuantPtr, quantParams[m]);

        aPtrs[m]      = aQuantPtr;
        cPtrsInt32[m] = reinterpret_cast<int32_t*>(cPtrs[m]);
      }

      for(int colBlockStart = 0; colBlockStart < N; colBlockStart += NCB) {
        int offsetB = innerBlockStart * N + KCB * colBlockStart;

        computeBlockSwitch<IS, N, K>(aPtrs, B + offsetB, cPtrsInt32, MCBActual, initBlockWithZero);

        for(int m = 0; m < MCBActual; m++)
          cPtrsInt32[m] += NCB;
      }
    }

    for(int m = 0; m < MCBActual; m++) {
      float* CPtr = C + (rowBlockStart + m) * N;
      dequantizeAndAdd<IS, N>(reinterpret_cast<int32_t*>(CPtr), bias, quantParams[m], CPtr);
    }
  }
}

/**
 * @brief Wrapper function to call gemmInt8Packed with the specified instruction set and matrix dimensions.
 *
 * This function calls gemmInt8Packed with the specified instruction set (IS) and matrix dimensions.
 *
 * @tparam IS The instruction set to use for optimization (e.g., AVX2, SSE4.2, etc.).
 * @param A Pointer to matrix A.
 * @param B Pointer to the packed matrix B.
 * @param bias Pointer to the bias matrix (optional).
 * @param C Pointer to the result matrix C.
 * @param M The number of rows in matrix A and C.
 * @param N The number of columns in matrix B and C.
 * @param K The number of columns in matrix A and rows in matrix B.
 */
template <InstructionSet IS>
void gemmInt8Packed(const float* A, const int8_t* B, const float* bias, float* C, int M, int N, int K) {
  if (N == 512 && K == 512) {
    gemmInt8Packed<IS, 512, 512>(A, B, bias, C, M);
  } else if (N == 1024 && K == 1024) {
    gemmInt8Packed<IS, 1024, 1024>(A, B, bias, C, M);
  } else if (N == 4096 && K == 1024) {
    gemmInt8Packed<IS, 4096, 1024>(A, B, bias, C, M);
  } else if (N == 3072 && K == 1024) {
    gemmInt8Packed<IS, 3072, 1024>(A, B, bias, C, M);
  } else if (N == 1024 && K == 4096) {
    gemmInt8Packed<IS, 1024, 4096>(A, B, bias, C, M);
  } else if (N == 8192 && K == 512) {
    gemmInt8Packed<IS, 8192, 512>(A, B, bias, C, M);
  } else if (N == 512 && K == 8192) {
    gemmInt8Packed<IS, 512, 8192>(A, B, bias, C, M);
  } else if (N == 1024 && K == 512) {
    gemmInt8Packed<IS, 1024, 512>(A, B, bias, C, M);
  } else if (N == 512 && K == 1024) {
    gemmInt8Packed<IS, 512, 1024>(A, B, bias, C, M);
  } else if (N == 32000 && K == 512) {
    gemmInt8Packed<IS, 32000, 512>(A, B, bias, C, M);
  } else {
    ABORT("Unsupported matrix dimensions <{}, {}>", N, K);
  }
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian
