#pragma once

#include "layers/factory.h"
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

// template<typename Function1, typename Function2>
// Function1 combine(Function1 function1, Function2 function2) {
  // return [function1, function2](float v1, float v2) { return function1(function2(v1, v2)); };
// }


// template<typename Function1, typename Function2>
// std::function<
    // float( // rettype
    // Function1, // fn1
    // Function2 // fn2
    // )> combine2(Function1 g, Function2 f) {
  // return [&](float x, float y) -> float { return g(f(x, y)); };
// }


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

  int pruningStart_; // when to start pruning
  int pruningStop_; // when to finish pruning
  int pruningStep_; // how often to prune
  float pruningSparsity_; // sparsity level to achieve;
  

public:
  CoefficientPruner(Ptr<Options> options) : options_(options) {
    // TODO load those values from flags
    pruningStart_ = options_->get<int>("pruning-start");
    pruningStop_ = options_->get<int>("pruning-stop");
    pruningStep_ = options_->get<int>("pruning-step");
    pruningSparsity_ = options_->get<float>("pruning-sparsity");
  }

  virtual void calculateThreshold(Expr p, int batchNum) = 0;
  
  virtual void pruneNode(Expr p, int batchNum) = 0;
  
  void maskTensor(Tensor t, Tensor b) override {
    using namespace functional;
    // if t is 0, then also set b to 0.
    Element(_1 = if_then_else(_2 == 0, 0, _1), b, t);
  }
  
  void pruneGraph(Ptr<ExpressionGraph> graph, int batchNum) override {
    if (batchNum == 0 || batchNum % pruningStep_ != 0 || batchNum > pruningStop_ || batchNum < pruningStart_) {
      // LOG_ONCE(info, "Finished pruning at iteration {}...", batchNum);
      return;
    }
    
    // for every node in a graph 
    for(auto p : *graph->params()) {
      // do not prune layer normalisation
      if (p->name().find("_ln_") != std::string::npos) { continue; }
      if (p->name().find("_b") != std::string::npos) { continue; }
      pruneNode(p, batchNum);
    }
  }

};

class MagnitudePruner : public CoefficientPruner {
public:

  MagnitudePruner(Ptr<Options> options) : CoefficientPruner(options) {};

  void calculateThreshold(Expr p, int batchNum) override {
    float startSparsity = 0.0f; // TODO: if we modify the algorithm, actually calculate the start sparsity of the tensor/model because sparsity may be non-linear
    float sparsity = pruningSparsity_ + std::min(0.0f, (startSparsity - pruningSparsity_) * (1 - float(batchNum) / float(pruningStop_)));
    int k = p->val()->size() * sparsity; // calculate k for topk
    
    // extract tensor to a vector
    std::vector<float> tVec;
    p->val()->get(tVec);
    // get the abs value
    std::transform(tVec.begin(), tVec.end(), tVec.begin(), fabs); 
    std::sort(tVec.begin(), tVec.end());
    
    threshold_ = tVec[k];
  }

  void pruneNode(Expr p, int batchNum) override {
	
    calculateThreshold(p, batchNum);
    
    using namespace functional;
    Element(_1 = if_then_else(abs(_1) < threshold_, 0, _1), p->val());
    
    int cnt = 0;
    std::vector<float> tVec;
    p->val()->get(tVec);
    for (auto x : tVec) {
      if (x == 0) 
        cnt++;
    } 
    LOG(info, "[{}] threshold: {} || zero count = {}/{}", p->name(), threshold_, cnt, p->val()->size());
  } 

};


class MagnitudeGradientPruner : public CoefficientPruner {
public:

  MagnitudeGradientPruner(Ptr<Options> options) : CoefficientPruner(options) {};

  void calculateThreshold(Expr p, int batchNum) override {
    float startSparsity = 0.0f; // TODO: if we modify the algorithm, actually calculate the start sparsity of the tensor/model because sparsity may be non-linear
    float sparsity = pruningSparsity_ + std::min(0.0f, (startSparsity - pruningSparsity_) * (1 - float(batchNum) / float(pruningStop_)));
    int k = p->val()->size() * sparsity; // calculate k for topk
    
    // extract tensor to a vector
    std::vector<float> tVec;
    std::vector<float> gVec;
    p->val()->get(tVec);
    p->grad()->get(gVec);
    // get the abs value

    // auto multiplyAbs2 = [](float v1, float v2){ return fabsf(std::multiplies<float>(v1, v2)); };
    // auto multiplyAbs = combine2(fabsf, std::multiplies<float>()); 
    // std::transform(tVec.begin(), tVec.end(), gVec.begin(), tVec.begin(), multiplyAbs2); //TODO figure out how to do abs and multiplies together with a single std::transform
    std::transform(tVec.begin(), tVec.end(), gVec.begin(), tVec.begin(), std::multiplies<float>());
    std::transform(tVec.begin(), tVec.end(), tVec.begin(), fabs); 
    std::sort(tVec.begin(), tVec.end());
    
    threshold_ = tVec[k];
  }

  void pruneNode(Expr p, int batchNum) override {
	
    calculateThreshold(p, batchNum);
    
    using namespace functional;
    Element(_1 = if_then_else(abs(_1 * _2) < threshold_, 0, _1), p->val(), p->grad());
    
    int cnt = 0;
    std::vector<float> cVec;
    p->val()->get(cVec);
    for (auto x : cVec) {
      if (x == 0) 
        cnt++;
    } 
    LOG(info, "[{}] threshold: {} || zero count = {}/{}", p->name(), threshold_, cnt, p->val()->size());
  } 

};

/* basically given the pruned param Tensor, also apply the same pruing to the other tensor 
     this is useful if you want to prune the gradient or moving avg as well..
*/


class PrunerFactory : public Factory {
public:
  using Factory::Factory;
  PrunerFactory(Ptr<Options> options) : Factory(options) {};

  Ptr<IPruner> construct() {
    std::string pruningType = options_->get<std::string>("pruning-type");

    if (pruningType == "magnitude") {
      LOG_ONCE(info, "Pruning type selected: magnitude");
      return New<MagnitudePruner>(options_);
    } 
    else if (pruningType == "magnitude-gradient") {
      LOG_ONCE(info, "Pruning type selected: magnitude-gradient");
      return New<MagnitudeGradientPruner>(options_);
    }
    else {
      LOG_ONCE(info, "Pruning type selected but not on the list? Returning nullptr, will break");
      return nullptr;
    }
  
  }

};



static void applyPrune(Tensor t, Tensor b) {
  using namespace functional;
  // if t is 0, then also set b to 0.
  Element(_1 = if_then_else(_2 == 0, 0, _1), b, t);
}

}

