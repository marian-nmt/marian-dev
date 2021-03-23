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
    float treshold = 0.01;

    // start prunning after 100-th steps.
    if (mbSize > 100) {
      LOG_ONCE(info, "DO PRUNING DUDE");
      using namespace functional;
      Element(_1 = if_then_else(abs(_1) < treshold, 0, _1), t);
    }
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

