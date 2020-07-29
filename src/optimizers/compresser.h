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
      kMeans_{options->get<int>("compress-k-means")},
      skipBias_{options->get<bool>("compress-skip-bias")},
      logQuant_{options->get<bool>("compress-log-quantize")} {
    }
      
  void compress(Ptr<ExpressionGraph> graph) {
    // reserve tensor for error feedback mechanism
    if (!error) {
      LOG(info, " EXPERIMENTAL: Applying quantization based compress model to {}-bit", bit_);
      LOG(info, " K-means scale adjustment steps: {}", kMeans_);

      int elements = (int) graph->params()->vals()->size();
      errorAlloc = New<TensorAllocator>(graph->getBackend());
      errorAlloc->reserveExact(graph->params()->vals()->memory()->size());
      errorAlloc->allocate(error, {1, elements});
    }

    // apply eror feedback mechanism
    using namespace functional;
    Element(_1 += _2, graph->params()->vals(), error);
    error->copyFrom(graph->params()->vals());

    int skip_size = 0;
    for(auto p : *graph->params()){
      // skip biases
      if (skipBias_ && p->val()->shape()[0] == 1) {
        skip_size += p->val()->size();
          continue;
      }

      compressImpl(p->val(), bit_, kMeans_, logQuant_);
    }

    // get new error
    Element(_1 -= _2, error, graph->params()->vals());
  }


protected:
#ifdef CUDA_FOUND
  void compressImpl(Tensor t, int bit, int kMeanStep = 0, bool logQuant = false); 
#else
  void compressImpl(Tensor t, int bit, int kMeanStep = 0, bool logQuant = false) {
    ABORT("Model compression training requires CUDA");
  }
#endif

  Tensor error;
  Ptr<TensorAllocator> errorAlloc;
  
  int bit_;
  int kMeans_;
  bool skipBias_;
  bool logQuant_;

  // temporary Tensor to calculate optimal S
  Tensor delta;
  Ptr<TensorAllocator> alloc_;
};
}
