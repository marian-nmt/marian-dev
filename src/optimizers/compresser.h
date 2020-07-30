#pragma once

#include "common/options.h"
#include "graph/expression_graph.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"
#include "tensors/tensor_allocator.h"
#include "functional/functional.h"

namespace marian {

class Compresser {
public:
  Compresser(Ptr<Options> options) 
    : bit_{options->get<int>("compress-bit")},
      opt_step_{options->get<int>("compress-k-means")},
      skip_bias_{options->get<bool>("compress-skip-bias")},
      log_quant_{options->get<bool>("compress-log-quantize")} {     
   }
      
  void compress(Ptr<ExpressionGraph> graph) {
    // reserve tensor for error feedback mechanism
    if (!error) {
      LOG(info, " EXPERIMENTAL: Applying quantization based compress model to {}-bit", bit_);
      LOG(info, " K-means scale adjustment steps: {}", opt_step_);

      int elements = (int) graph->params()->vals()->size();
      error_alloc = New<TensorAllocator>(graph->getBackend());
      error_alloc->reserveExact(graph->params()->vals()->memory()->size());
      error_alloc->allocate(error, {1, elements});
    }

    // apply error feedback mechanism
    using namespace functional;
    Element(_1 += _2, graph->params()->vals(), error);
    error->copyFrom(graph->params()->vals());

    int skip_size = 0;
    for(auto p : *graph->params()){
      // skip biases
      if (skip_bias_ && p->val()->shape()[0] == 1) {
        skip_size += p->val()->size();
          continue;
      }

      compressImpl(p->val(), bit_, opt_step_, log_quant_);
    }

    // get new error
    Element(_1 -= _2, error, graph->params()->vals());
  }


protected:
  void init(Tensor t);
  void compressImpl(Tensor t, int bit, int opt_step = 0, bool log_quant = false); 

  Tensor error;
  Ptr<TensorAllocator> error_alloc;
  
  int bit_;
  int opt_step_;
  bool skip_bias_;
  bool log_quant_;

  // temporary Tensor for storing q to calculate optimal S
  Tensor delta;
  Ptr<TensorAllocator> alloc_;

  // single element Tensor for Reduce swap variable
  Tensor temp_var;
  Ptr<TensorAllocator> temp_alloc;
};
}
