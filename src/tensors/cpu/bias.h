#pragma once

#include "3rd_party/intgemm/intrinsics.h"
#include "tensors/tensor.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

namespace marian {
namespace cpu {

namespace {
template <class Register> static inline Register loadu_ps(const float* mem_addr);
template <> INTGEMM_SSE2 inline __m128 loadu_ps(const float* mem_addr) {
  return _mm_loadu_ps(mem_addr);
}
template <> INTGEMM_AVX512BW inline __m512 loadu_ps(const float* mem_addr) {
  return _mm512_loadu_ps(mem_addr);
}
}

/* TODO: CPUID dispatch, maybe integrate with intgemm */

// This operates on floats after processing so doesn't care about int8_t vs int16_t.
static void AddBias(marian::Tensor C, const marian::Tensor Bias) {
    float* y = C->data();
    const float* x = C->data();
    const float* bias = Bias->data();
    const int m = C->shape().elements() / C->shape()[-1];
    const int n = C->shape()[-1];

#ifdef __AVX512F__
    using vec_t = __m512;
#else
    using vec_t = __m128;
#endif

    const int step = sizeof(vec_t) / 4; // 4 bytes per float
    const int n_aligned = n & (step - 1);
    for(int j = 0; j < m; ++j) {
        int i = 0;
        for (; i < n_aligned; i += step) {
            auto ai = loadu_ps<vec_t>(x + j * n + i);
            auto bi = loadu_ps<vec_t>(bias + i);
            auto yi = intgemm::add_ps(ai, bi);
            intgemm::storeu_ps(y + j * n + i, yi);
        }
        for (; i < n; ++i) {
            y[j * n + i] = x[j * n + i] + bias[i];
        }
    }
}

}
}
