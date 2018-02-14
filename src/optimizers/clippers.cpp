/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "clippers.h"

#include "kernels/tensor_operators.h"
#include "kernels/thrust_functions.h"

namespace marian {

using namespace thrust::placeholders;

void Elementwise::clip(Tensor t) {
  Element(_1 = Clip(_1, c_), t);
}

void Norm::clip(Tensor t) {
  float l2Norm = L2Norm(t);
  if(l2Norm >= c_)
    Element(_1 = (c_ / l2Norm) * _1, t);
}

}
