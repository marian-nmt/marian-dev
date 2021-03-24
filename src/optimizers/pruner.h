#pragma once

#include "common/options.h"
#include "functional/functional.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

namespace marian {

  /* pruning implementation */
  static void pruneImpl(Tensor t, int mbSize) {
        
    // TODO: find the actual treshold
    float treshold;

    // currently: 2 step pruning.
    // prune by 0.0001 after 50th update, and prune by 0.001 after 100th update.
    if (mbSize == 50) {
      LOG_ONCE(info, "DO PRUNING first");
      treshold = 0.0001;
    } else if(mbSize == 100) {
      LOG_ONCE(info, "DO PRUNING second");
      treshold = 0.001;
    } else
      return;

    using namespace functional;
    Element(_1 = if_then_else(abs(_1) < treshold, 0, _1), t);
  }

  /* prune the whole graph */
  static void pruneGraph(Ptr<ExpressionGraph> graph, int mbSize) {
    // loop layer by layer
    for(auto p : *graph->params()) {
        pruneImpl(p->val(), mbSize);
    }
  }

  /* basically given the pruned param Tensor, also apply the same pruing to the other tensor 
     this is useful if you want to prune the gradient or moving avg as well..
  */
  static void applyPrune(Tensor t, Tensor b) {
    using namespace functional;
    // if t is 0, then also set b to 0.
    Element(_1 = if_then_else(_2 == 0, 0, _1), b, t);
  }

}

