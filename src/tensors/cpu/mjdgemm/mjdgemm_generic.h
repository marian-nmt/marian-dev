#pragma once

#include "mjdgemm_utils.h"

namespace marian {
namespace cpu {
namespace mjdgemm {

/**
 * @brief Computes a block of the matrix multiplication using the specified instruction set.
 *
 * This function performs matrix multiplication on a block of matrices A and B, storing the result in matrix C.
 * It uses the specified instruction set (IS) to optimize the computation.
 *
 * @tparam IS The instruction set to use for optimization (e.g., AVX2, SSE4.2, etc.).
 * @tparam N The number of columns in matrix B and C.
 * @tparam K The number of columns in matrix A and rows in matrix B.
 * @param blockA Pointer to the block of matrix A.
 * @param blockB Pointer to the block of matrix B.
 * @param blockC Pointer to the block of matrix C.
 * @param MCBActual The actual number of rows in the current block of matrix A and C.
 */
template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::None>::type
computeBlock(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC, int MCBActual) {
    constexpr int NCB = BlockingFactors<IS>::NCB;
    constexpr int KCB = BlockingFactors<IS>::KCB;
    constexpr int RI  = BlockingFactors<IS>::RI;
    constexpr int NR  = BlockingFactors<IS>::NR;

    static_assert(KCB % RI == 0);
    static_assert(K % KCB == 0);
    static_assert(N % NCB == 0);
    static_assert(N % (2 * NR) == 0);

    // assert that inputs are 64-byte aligned
    assert(reinterpret_cast<uintptr_t>(blockA) % 64 == 0);
    assert(reinterpret_cast<uintptr_t>(blockB) % 64 == 0);
    assert(reinterpret_cast<uintptr_t>(blockC) % 64 == 0);

    // Initialize registers
    const uint8_t* a;
    const int8_t*  b[2];
    int32_t* c[2];

    for(int m = 0; m < MCBActual; m++) {
        // Load 32 (NCB) elements of C into 2 registers c[0] and c[1], 16 each
        for(int i = 0; i < 2; i++)
            c[i] = blockC[m] + i * NR;

        // Loop over inner dimension in step of row interleave RI
        for(int k = 0; k < KCB; k += RI) {
            // Load 4 elements of A and broadcast to 64 8-bit integers
            // into register a
            a = blockA[m] + k;

            // Load 32 * 4 (NCB * RI) elements of B into 2 registers b[0] and b[1]
            b[0] = blockB + k * NCB;
            b[1] = blockB + k * NCB + RI * NR;

            // Multiply and add in sets of 4 elements and accumulate
            for(int i = 0; i < 2; i++) {
                for(int j = 0; j < NR; j++) {
                    c[i][j] += a[0] * b[i][j * RI + 0]
                             + a[1] * b[i][j * RI + 1]
                             + a[2] * b[i][j * RI + 2]
                             + a[3] * b[i][j * RI + 3];
                }
            }
        }
    }
}

template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::None>::type
computeBlockSwitch(const uint8_t** blockA, const int8_t* blockB, int32_t** blockC, int M) {
  computeBlock<InstructionSet::None, N, K>(blockA, blockB, blockC, M);
}

template <InstructionSet IS, int N, int K>
inline typename std::enable_if<IS == InstructionSet::None>::type
computeQuantizationParams(const float* A, const int8_t* B, QuantizationParams& quantParams) {
    // Compute the min and max values of the input via fast AVX2 instructions,
    // assume the length is a multiple of VLEN
    float minA = std::numeric_limits<float>::max();
    float maxA = std::numeric_limits<float>::lowest();

    // compute minA and maxA
    for (std::size_t i = 0; i < K; i++) {
        minA = std::min(minA, A[i]);
        maxA = std::max(maxA, A[i]);
    }

    quantParams.sumA       = 0;

    // Compute the scale and zero point
    quantParams.scaleA     = (maxA - minA) / 255;
    quantParams.zeroPointA = (int32_t)(255 - maxA / quantParams.scaleA);

    // quantization parameters for B, access them from the end of the packed buffer
    // there is n of quantScaleB, quantZeropointB and colOffsetsB, each
    quantParams.packSizeB   = K * N;
    quantParams.scaleB      = (const float*)  (B + quantParams.packSizeB);
    quantParams.zeroPointB  = (const int32_t*)(B + quantParams.packSizeB + N * sizeof(float));
    quantParams.colOffsetsB = (const int32_t*)(B + quantParams.packSizeB + N * sizeof(float) + N * sizeof(int32_t));
}

/**
 * @brief Quantizes the given source matrix to the destination matrix using the specified quantization parameters.
 *
 * This function quantizes the given source matrix to the destination matrix using the specified quantization parameters.
 *
 * @tparam IS The instruction set to use for optimization (e.g., AVX2, SSE4.2, etc.).
 * @tparam LEN The length of the source and destination matrices.
 * @param src Pointer to the source matrix.
 * @param dst Pointer to the destination matrix.
 * @param quantParams Reference to the QuantizationParams structure containing the quantization parameters.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::None>::type
quantize(const float* src, uint8_t* dst, QuantizationParams& quantParams) {

    constexpr float min_val = std::numeric_limits<uint8_t>::min();
    constexpr float max_val = std::numeric_limits<uint8_t>::max();

    for (std::size_t i = 0; i < LEN; i++) {
        float transform = src[i] / quantParams.scaleA + quantParams.zeroPointA;
        float clipped = std::min(std::max(transform, min_val), max_val);
        int32_t rounded = (int32_t)std::nearbyint(clipped);

        quantParams.sumA += rounded;

        dst[i] = (uint8_t)rounded;
    }
}

/**
 * @brief Dequantizes the given source matrix and adds the result to the destination matrix.
 *
 * This function dequantizes the given source matrix and adds the result to the destination matrix.
 *
 * @tparam IS The instruction set to use for optimization (e.g., AVX2, SSE4.2, etc.).
 * @tparam LEN The length of the source and destination matrices.
 * @param src Pointer to the source matrix.
 * @param bias Pointer to the bias matrix (optional).
 * @param quantParams Reference to the QuantizationParams structure containing the quantization parameters.
 * @param dst Pointer to the destination matrix.
 */
template <InstructionSet IS, int LEN>
inline typename std::enable_if<IS == InstructionSet::None>::type
dequantizeAndAdd(const int32_t* src, const float* bias, const QuantizationParams& quantParams, float* dst) {
    for (int j = 0; j < LEN; j++) {
        int32_t xv = src[j];

        xv = xv - quantParams.zeroPointA * quantParams.colOffsetsB[j];
        xv = xv - quantParams.zeroPointB[j] * quantParams.sumA;

        // compute scale * xv
        float scale = quantParams.scaleA * quantParams.scaleB[j];
        float xv_f = (float)xv;
        float result = scale * xv_f;

        if(bias) {
            result += bias[j];
        }

        dst[j] = result;
    }
}

} // namespace mjdgemm
} // namespace cpu
} // namespace marian
