/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "functional/functional.h"
#include "functional/shape.h"
#include "functional/tensor.h"
#include "functional/tmp.h"
#include "tensors/tensor.h"

namespace marian {

namespace cpu {

template <size_t K, class Functor, class AggFunctor>
void gAggregateGeneric(Functor functor, float aggInit, AggFunctor aggFunctor,
                 const functional::Shape full,
                 functional::Tensor<float> out,
                 functional::Array<functional::Tensor<float>, K> ins,
                 float scale = 1.0) {
  size_t outLength = out.shape().elements();
  bool same = outLength == full.elements();
  for(size_t i = 0; i < K; ++i)
    same = same && outLength == ins[i].shape().elements();

  constexpr size_t N = functional::Shape::size();
  functional::Array<size_t, N> len;
  for(size_t i = 0; i < N; ++i)
    len[i] = full[i] / out.shape()[i];

  functional::Array<size_t, N> dims;
  for(size_t index = 0; index < outLength; ++index) {
    if(same) {
      out[index] = aggFunctor(out[index], functional::apply(functor, ins, index) * scale);
    } else {
      out.shape().dims(index, dims);
      out[index] = aggFunctor(out[index], functional::loops(functor, aggInit, aggFunctor, ins, len, dims) * scale);
    }
  }
}

template <size_t K, class Functor, class AggFunctor>
void gAggregateEqual(Functor functor, AggFunctor aggFunctor,
               functional::Tensor<float> out,
               functional::Array<functional::Tensor<float>, K> ins,
               float scale,
               bool broadcast) {
  size_t length = out.shape().elements();
  functional::Array<size_t, functional::Shape::size()> dims;

  for(size_t index = 0; index < length; ++index) {
    functional::Array<size_t, K> indices;
    indices.fill(index);

    if(broadcast) {
      out.shape().dims(index, dims);
      for(size_t i = 0; i < K; ++i)
        indices[i] = ins[i].shape().bindex(dims);
    }

    out[index] = aggFunctor(out[index], functional::apply(functor, ins, indices) * scale);
  }
}

template <size_t K, class Functor, class AggFunctor>
void gAggregateReduce(Functor functor, float aggInit, AggFunctor aggFunctor,
                const functional::Shape full,
                functional::Tensor<float> out,
                functional::Array<functional::Tensor<float>, K> ins,
                float scale = 1.0) {
  size_t rows = full.elements() / full.back();
  size_t cols = full.back();

  bool same = true;
  for(size_t i = 0; i < K; ++i)
    same = same && ins[i].shape().elements() == full.elements();

  for(size_t j = 0; j < rows; ++j) {
    float colSum = aggInit;
    if(same) {
      for(size_t id = 0; id < cols; ++id)
        colSum = aggFunctor(colSum, functional::apply(functor, ins, j * cols + id));
    } else {
      functional::Array<size_t, functional::Shape::size()> dims;
      for(size_t id = 0; id < cols; ++id) {
        full.dims(j * cols + id, dims);
        functional::Array<size_t, K> indices;
        for(size_t i = 0; i < K; ++i)
          indices[i] = ins[i].shape().bindex(dims);
        colSum = aggFunctor(colSum, functional::apply(functor, ins, indices));
      }
    }
    out[j] = aggFunctor(out[j], colSum * scale);
  }
}

template <class Functor, class AggFunctor, class... Tensors>
void Aggregate(Functor functor, float aggInit, AggFunctor aggFunctor, float scale, marian::Tensor out, Tensors... tensors) {
  auto full = marian::Shape::broadcast({out, tensors...});

  //size_t length = out->shape().elements();

  constexpr size_t K = sizeof...(Tensors);

  functional::Tensor<float> gOut = out;
  functional::Array<functional::Tensor<float>, K> gIns = {tensors...};

  if(full.back() != 1 && out->shape().back() == 1) {
    //size_t m = full.elements() / length;
    //size_t k = full.back();
    cpu::gAggregateReduce(functor, aggInit, aggFunctor, full, gOut, gIns, scale);
  } else if(out->shape() == full) {
    bool broadcast = false;
    for(size_t i = 0; i < K; ++i)
      broadcast = broadcast || gOut.shape() != gIns[i].shape();
    cpu::gAggregateEqual(functor, aggFunctor, gOut, gIns, scale, broadcast);
  } else {
    cpu::gAggregateGeneric(functor, aggInit, aggFunctor, full, gOut, gIns, scale);
  }
}

}  // namespace cpu
}  // namespace marian
