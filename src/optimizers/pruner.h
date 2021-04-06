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


class IPruner {
public:
  virtual ~IPruner() {}

  virtual void pruneGraph(Ptr<ExpressionGraph> graph, int batchNum) = 0;

  virtual void maskTensor(Tensor a, Tensor b) = 0;

};

class CoefficientPruner : public IPruner {
protected:
  Ptr<Options> options_;
  
  float threshold_; // threshold to cut parameters off with

  int batchStop_; // for how many batches to prune
  int batchStep_; // how often to prune
  float targetSparsity_; // sparsity level to achieve;
  

public:
  CoefficientPruner(Ptr<Options> options) : options_(options) {
    // TODO load those values from flags
    batchStop_ = 400000;
    batchStep_ = 10000;
    targetSparsity_ = 0.9;
  }

  virtual void calculateThreshold(Tensor t, std::string name, int batchNum) = 0;
  
  virtual void pruneTensor(Tensor t, std::string name, int batchNum) = 0;
  
  void maskTensor(Tensor t, Tensor b) override {
    using namespace functional;
    // if t is 0, then also set b to 0.
    Element(_1 = if_then_else(_2 == 0, 0, _1), b, t);
  }
  
  void pruneGraph(Ptr<ExpressionGraph> graph, int batchNum) override {
    if (batchNum == 0 || batchNum % batchStep_ != 0 || batchNum > batchStop_) {
      LOG_ONCE(info, "Finished pruning at iteration {}...", batchNum);
      return;
    }
    
    // for every tensor in a graph 
    for(auto p : *graph->params()) {
      // do not prune layer normalisation
      if (p->name().find("_ln_") != std::string::npos) { return; }
      pruneTensor(p->val(), p->name(), batchNum);
    }

  }

};

class MagnitudePruner : public CoefficientPruner {
public:

  MagnitudePruner(Ptr<Options> options) : CoefficientPruner(options) {};

  void calculateThreshold(Tensor t, std::string name, int batchNum) override {
    float startSparsity = 0.0f; // TODO: if we modify the algorithm, actually calculate the start sparsity of the tensor/model because sparsity may be non-linear
    float sparsity = targetSparsity_ + std::min(0.0f, (startSparsity - targetSparsity_) * (1 - float(batchNum) / float(batchStop_)));
    int k = t->size() * sparsity; // calculate k for topk
    
    // extract tensor to a vector
    std::vector<float> tVec;
    t->get(tVec);
    // get the abs value
    std::transform(tVec.begin(), tVec.end(), tVec.begin(), fabs); 
    std::sort(tVec.begin(), tVec.end());
    
    threshold_ = tVec[k];


  }

  void pruneTensor(Tensor t, std::string name, int batchNum) override { 
    calculateThreshold(t, name, batchNum);
    
    using namespace functional;
    Element(_1 = if_then_else(abs(_1) < threshold_, 0, _1), t);
    
    int cnt = 0;
    std::vector<float> tVec;
    t->get(tVec);
    for (auto x : tVec) {
      if (x == 0) 
        cnt++;
    } 
    LOG(info, "[{}] threshold: {} || zero count = {}/{}", name, threshold_, cnt, t->size());
  } 

};


  /* pruning implementation */
  static void pruneImpl(Tensor t, Tensor g, Ptr<ExpressionGraph> graph, int mbNum, std::string name = "") {
        
    // TODO: Add gradients to the function
    // We're gonna prune based on magnitude * gradients


    float threshold = 0.0f; // threshold to calculate
    float targetSparsity = 0.1; // sparsity we want to achieve for each layer
    float startSparsity = 0.0; // starting sparsity, probably going to be 0%
    int step = 10; // to prune how frequently (batches)
    int totalSteps = 100; // how many batches to prune for


    // check whether it is time to prune at all (maybe check before that)
    if (mbNum == 0 || mbNum % step != 0 || mbNum > totalSteps) {
      // LOG(info, "EXITING PRUNING BECAUSE IT'S NOT TIME {} {}", mbNum, step);
      return;
    }

    // don't prune layer normalisation???
    if (name.find("_ln_") != std::string::npos) {
      return;
    }

    // calculate the sparsity we have to achieve in this pruning step
    float sparsity = targetSparsity + std::min(0.0f, (startSparsity - targetSparsity) * (1 - float(mbNum) / float(totalSteps)));

    LOG(info, "sparsity {}", sparsity);

    int k = t->size() * sparsity; // calculate k for topk
    
    LOG(info, "tensor size {}", t->size()); 
    LOG(info, "k {}", k);

    // TODO: Do topK with Tensors 
    // // reshape t into a vector??? to get topk from all of it, not just per rowShape({1, rows})
    // auto tVec = TensorBase::New(t->memory(), Shape({1, t->shape()[0] * t->shape()[1]}), t->type(), t->getBackend());
    // Tensor topKVal, topKInd; // Do I allocate the memory here somehow??? or does topk do it for me
    // TopK(topKVal, topKInd, graph->allocator(), tVec, k, 1, true); // calculate TopK value?
    // threshold = topKVal->scalar(); // extract topk scalar as a new threshold
    

    ////////////////////////////////////////////////////////////////////////////
    // Pruning with the simplest C++ sorting?
    ////////////////////////////////////////////////////////////////////////////

    std::vector<float> valVec;
    // std::vector<float> adjVec; // gradients?

    
    t->get(valVec);
    // g->get(adjVec);

    // std::cout << "valVec" << std::endl;

    // for (int i = 1; i <= 15; i++)
      // std::cout << valVec[i] << " ";
    // std::cout << std::endl;

    // std::cout << "adjVec" << std::endl;
    // for (int i = 1; i <= 15; i++)
      // std::cout << adjVec[i] << " ";
    // std::cout << std::endl;


    std::vector<float> scores;
    for (const auto& i: valVec)
      scores.push_back(std::abs(i));
    
    // std::cout << "scores" << std::endl;
    // for (int i = 1; i <= 15; i++)
      // std::cout << scores[i] << " ";
    // std::cout << std::endl;
    
    // std::transform(valVec.begin(), valVec.end(), adjVec.begin(), 
                   // std::back_inserter(scores), std::multiplies<float>());

    std::sort(scores.begin(), scores.end());
    
    threshold = scores[k];


    using namespace functional;
    Element(_1 = if_then_else(abs(_1) < threshold, 0, _1), t);
    
    int cnt = 0;
    t->get(valVec);
    for (auto x : valVec) {
      if (x == 0) 
        cnt++;
    }
    
    LOG(info, "[{}] prune by {}: treshold: {} || zero count = {}/{}", name, sparsity, threshold, cnt, t->size());
  }

  /* prune the whole graph */
  static void pruneGraph(Ptr<ExpressionGraph> graph, int mbNum, Ptr<Options> options) {
    // loop layer by layer
    for(auto p : *graph->params()) {
        pruneImpl(p->val(), p->grad(), graph, mbNum, p->name());
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

