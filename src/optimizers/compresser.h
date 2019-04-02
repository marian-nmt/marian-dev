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
      base_{options->get<float>("compress-base")},
      clip_{options->get<float>("compress-clip")},
      interval_{options->get<int>("compress-interval")} {
    }
      
  void compress(Ptr<ExpressionGraph> graph) {
    // reserve tensor for error feedback mechanism
    if (!error) {
      LOG(info, " EXPERIMENTAL: Log-compress model to {}-bit every {} steps", bit_, interval_);

      int elements = (int) graph->params()->vals()->size();
      errorAlloc = New<TensorAllocator>(graph->getBackend());
      errorAlloc->reserveExact(graph->params()->vals()->memory()->size());
      errorAlloc->allocate(error, {1, elements});
    }

    // apply eror feedback mechanism
    using namespace functional;
    Element(_1 += _2, graph->params()->vals(), error);
    error->copyFrom(graph->params()->vals());

    // compress every interval
    if (++step % interval_ == 0)
      for(auto p : *graph->params()){
        compressImpl(p->val(), bit_, base_, clip_);
      }

    // get new error
    Element(_1 -= _2, error, graph->params()->vals());
  
  }

  void compressImpl(Tensor t, int bit, float base, float clip);

protected:
  int step{0};
  Tensor error;
  Ptr<TensorAllocator> errorAlloc;
  int bit_;
  float base_;
  float clip_;
  int interval_;

};
}
