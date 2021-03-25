#pragma once

#include "common/options.h"
#include "functional/functional.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

#include <vector>
#include <cmath>
#include <algorithm>

namespace marian {

  /* pruning implementation */
  static void pruneImpl(Tensor t, int mbSize, std::string name = "") {
    if (mbSize  == 0 || mbSize % 50 || mbSize > 450)
      return;
        
    // TODO: find the actual treshold
    float treshold;
    float ratio = 0.1 * (mbSize / 50);
    
    std::vector<float> f;
    t->get(f);
    // get the abs value
    std::transform(f.begin(), f.end(), f.begin(), fabs);
    // sort
    std::sort(f.begin(), f.end());
    int idx = ratio * f.size();
    treshold = f[idx];

    using namespace functional;
    Element(_1 = if_then_else(abs(_1) < treshold, 0, _1), t);
  

    // validation
    int cnt = 0;
    t->get(f);
    for (auto x:f) if (x == 0) cnt++;
    LOG(info, "[{}] prune by {}: treshold: {} || zero count = {}/{}", name, ratio, treshold, cnt, t->size());
  }

  /* prune the whole graph */
  static void pruneGraph(Ptr<ExpressionGraph> graph, int mbSize) {
    // loop layer by layer
    for(auto p : *graph->params()) {
        pruneImpl(p->val(), mbSize, p->name());
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

