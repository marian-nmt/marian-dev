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
  static void pruneImpl(Tensor t, Ptr<ExpressionGraph> graph, int mbNum) {
        
    // TODO: Add gradients to the function
    // We're gonna prune based on magnitude * gradients


    float threshold = 0.0f; // threshold to calculate
    float targetSparsity = 0.9; // sparsity we want to achieve for each layer
    float startSparsity = 0.0; // starting sparsity, probably going to be 0%
    int step = 100; // to prune how frequently (batches)
    int totalSteps = 900; // how many batches to prune for


    // check whether it is time to prune at all (maybe check before that)
    if (mbNum % step != 0) {
      return;
    }

    // calculate the sparsity we have to achieve in this pruning step
    float sparsity = targetSparsity + std::min(0.0f, (startSparsity - targetSparsity) * (1 - step / totalSteps));

    int k = t->size() * sparsity; // calculate k for topk

    // reshape t into a vector??? to get topk from all of it, not just per rowShape({1, rows})
    auto tVec = TensorBase::New(t->memory(), Shape({1, t->shape()[0] * t->shape()[1]}), t->type(), t->getBackend());


    Tensor topKVal, topKInd; // Do I allocate the memory here somehow??? or does topk do it for me
    TopK(topKVal, topKInd, graph->allocator(), tVec, k, 1, true); // calculate TopK value?
    threshold = topKVal->scalar(); // extract topk scalar as a new threshold
    
    LOG(info, "Pruning {} {} {}", mbNum, sparsity, threshold); 
     
    // // currently: 2 step pruning.
    // // prune by 0.0001 after 50th update, and prune by 0.001 after 100th update.
    // if (mbNum == 50) {
      // LOG_ONCE(info, "DO PRUNING first");
      // threshold = 0.0001;
    // } else if(mbNum == 100) {
      // LOG_ONCE(info, "DO PRUNING second");
      // threshold = 0.001;
    // } else
      // return;

    using namespace functional;
    Element(_1 = if_then_else(abs(_1) < threshold, 0, _1), t);
  }

  /* prune the whole graph */
  static void pruneGraph(Ptr<ExpressionGraph> graph, int mbNum) {
    // loop layer by layer
    for(auto p : *graph->params()) {
        pruneImpl(p->val(), graph, mbNum);
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

