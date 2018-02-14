/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>
  
#include "training/dropper.h"

namespace marian {

template <typename T, typename InputIterator, typename OutputIterator, typename Op>
static void fold(InputIterator begin, InputIterator end, OutputIterator out, Op op, T left) {
  for (; begin != end; ++begin, ++out) {
    left = op(left, *begin);
    *out = left;
  }
}

class GradientDropCPU : public GradientDropBase {
  std::vector<float> feedback;
  std::vector<bool> mask;
  std::vector<int> indices;

  void drop(Tensor t, double rate) {
    int size = t->size();
    if (feedback.size() == 0) {
      feedback.resize(size);
      mask.resize(size);
      indices.resize(size);
    }

    std::fill(mask.begin(), mask.end(), true);

    std::vector<int>::iterator begin = indices.begin(), end = indices.end(), middle;
    std::iota(begin, end, 0);

    float* data = t->data();

    int sortSize = rate * size;
    middle = begin + sortSize;
    std::nth_element(begin, middle-1, end, [&] (int a, int b) {
        return std::abs(data[a] + feedback[a]) > std::abs(data[b] + feedback[b]);
      });

    std::sort(begin, middle);

    int dropBegin = 0;
    for (int i = 0; i < sortSize; ++i) {
      int dropEnd = indices[i];
      for (int j = dropBegin; j < dropEnd; ++j) {
        feedback[j] += data[j];
        data[j] = 0.f;
        mask[j] = false;
      }
    }
  }

  public:
  void dropGraph(Tensor t, SparseTensor destination, double rate) {
    drop(t, rate);

    // FIXME: We should just build the destination as we go.
    std::vector<int> partial_sum;
    fold(mask.begin(), mask.end(), std::back_inserter(partial_sum), std::plus<int>(), 0);

    int denseSize = t->size();
    int sparseSize = partial_sum.back();
    destination->setSize(sparseSize);

    float* dense = t->data(), * sparse = destination->data();
    int* indices = destination->indices();
    for (int iDense = 0; iDense < denseSize; ++iDense) {
      if (!mask[iDense]) {
        continue;
      }

      int iSparse = partial_sum[iDense]-1;
      sparse[iSparse] = dense[iDense];
      indices[iSparse] = iDense;
    }
  }
};

}
