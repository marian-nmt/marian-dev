#pragma once

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"
#include "common/io_item.h"
#include "3rd_party/intgemm/intgemm.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <cassert>
#include <cstddef>

namespace marian {
namespace cpu {
namespace integer {

//Convenient function to get rows and columns of a tensor, shadowed by namespace.
inline int cols(Tensor& tensor) { return tensor->shape()[-1]; }
inline int rows(Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }

inline int cols(Shape& shape) { return shape[-1]; }
inline int rows(Shape& shape) { return shape.elements() / cols(shape); }

template<Type type> struct intgemm_;
template <> struct intgemm_<Type::int8> {using width = intgemm::Int8;
                                         using type = int8_t;
                                         constexpr static const Type intgemmType = Type::intgemm8;};
template <> struct intgemm_<Type::int16> {using width = intgemm::Int16;
                                          using type = int16_t;
                                          constexpr static const Type intgemmType = Type::intgemm16;};

// This operates on floats after processing so doesn't care about int8_t vs int16_t.
void AddBias(marian::Tensor C, const marian::Tensor Bias);

// For loading architecture agnostic models. We do PrepareAndTranpose, because we already transposed
// in our binary format. Then we copy the quantizationMultiplier information at the end
template<Type vtype>
void prepareAndTransposeB(io::Item& item, const char * input) {
    typedef typename intgemm_<vtype>::type Integer;
    Integer * output_tensor = reinterpret_cast<Integer *>(&(*item.bytes.begin()));
    intgemm_<vtype>::width::PrepareBQuantizedTransposed(reinterpret_cast<const Integer *>(input),
                                               //reinterpret_cast<Integer *>(&(*item.bytes.begin())),
                                               output_tensor,
                                               rows(item.shape),  //Since we only transposed, but didn't update the shape when constructing the binary, 
                                               cols(item.shape)); //rows here returns the columns of the transposed input matrix, and cols -> the rows
    //Copy the quantMult
    float quantMult = *(reinterpret_cast<const float *>(input + item.shape.elements()));
    *(reinterpret_cast<float *>(&(*(output_tensor + item.shape.elements())))) = quantMult;
}

} //integer
} //cpu
} //marian