#pragma once

#include "common/options.h"
#include "functional/functional.h"
#include "graph/expression_graph.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

namespace marian {

class Compresser {
public:
  Compresser(Ptr<Options> options)
      : bit_{options->get<int>("compress-bit")},
        optStep_{options->get<int>("compress-k-means")},
        skipBias_{options->get<bool>("compress-skip-bias")},
        logQuant_{options->get<bool>("compress-log-quantize")} {}

  void compress(Ptr<ExpressionGraph> graph) {
    // reserve tensor for error feedback mechanism
    if(!error_) {
      LOG(info, " EXPERIMENTAL: Applying quantization based compress model to {}-bit", bit_);
      LOG(info, " K-means scale adjustment steps: {}", optStep_);

      int elements = (int)graph->params()->vals()->size();
      errorAlloc_ = New<TensorAllocator>(graph->getBackend());
      errorAlloc_->reserveExact(graph->params()->vals()->memory()->size());
      errorAlloc_->allocate(error_, {1, elements});
    }

    // apply error feedback mechanism
    using namespace functional;
    Element(_1 += _2, graph->params()->vals(), error_);
    error_->copyFrom(graph->params()->vals());

    for(auto p : *graph->params()) {
      // skip biases
      if(!skipBias_ || p->val()->shape()[0] > 1)
        compressImpl(p->val(), bit_, optStep_, logQuant_);
    }

    // get new error
    Element(_1 -= _2, error_, graph->params()->vals());
  }

protected:
  void init(Tensor t);
  void compressImpl(Tensor t, int bit, int opt_step = 0, bool log_quant = false);

  Tensor error_;
  Ptr<TensorAllocator> errorAlloc_;

  int bit_;
  int optStep_;
  bool skipBias_;
  bool logQuant_;

  // temporary Tensor for storing q to calculate optimal S
  Tensor delta_;
  Ptr<TensorAllocator> alloc_;

  // single element Tensor for Reduce swap variable
  Tensor tempVar_;
  Ptr<TensorAllocator> tempAlloc_;
};
}  // namespace marian
