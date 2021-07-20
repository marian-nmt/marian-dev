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
  IPruner(Ptr<Options> options) : options_(options) { 
  
    pruningStart_ = options_->get<int>("pruning-start");
    pruningStop_ = options_->get<int>("pruning-stop");
    pruningStep_ = options_->get<int>("pruning-step");
    pruningSparsity_ = options_->get<float>("pruning-sparsity");
  
  }
  
  virtual ~IPruner() {}

  // virtual void pruneGraph(Ptr<ExpressionGraph> graph, int batchNum) = 0;

  // virtual void maskTensor(Tensor a, Tensor b) = 0;
  
  virtual void pruneNode(Expr p, int batchNum, bool rows = false) = 0;
  
  virtual void maskTensor(Tensor t, Tensor b) {
    using namespace functional;
    // if t is 0, then also set b to 0.
    Element(_1 = if_then_else(_2 == 0, 0, _1), b, t);
  }
  
  virtual void pruneGraph(Ptr<ExpressionGraph> graph, int batchNum) {
    if (batchNum == 0 || batchNum % pruningStep_ != 0 || batchNum > pruningStop_ || batchNum < pruningStart_) {
      // LOG_ONCE(info, "Finished pruning at iteration {}...", batchNum);
      return;
    }
    
    // for every node in a graph 
    for(auto p : *graph->params()) {
      // do not prune layer normalisation
      if (p->name().find("_ln_") != std::string::npos) { continue; }
      // do not prune any biases
      if (p->name().find("_b") != std::string::npos) { continue; }
      // do not prune embeddings if said so
      if (options_->get<bool>("pruning-skip-embeddings") && p->name().find("Wemb") != std::string::npos) { continue; }
      // do not prune RNN attention
      if (p->name().find("rnn") != std::string::npos) { continue; }

      bool rows = false;
      if (p->name().find("_Wo") != std::string::npos) { rows = true; }
      if (p->name().find("_W2") != std::string::npos) { rows = true; }

      // if valid to do so, prune that node
      pruneNode(p, batchNum, rows);
    }
  }

protected:
  Ptr<Options> options_;
  
  int pruningStart_; // when to start pruning
  int pruningStop_; // when to finish pruning
  int pruningStep_; // how often to prune
  float pruningSparsity_; // sparsity level to achieve;

};

class CoefficientPruner : public IPruner {
public:
  CoefficientPruner(Ptr<Options> options) : IPruner(options) { }

  virtual void calculateThreshold(Expr p, int batchNum) = 0;

protected:
  float threshold_; // threshold to cut parameters off with

};

class ThresholdPruner : public CoefficientPruner {
public:

  ThresholdPruner(Ptr<Options> options) : CoefficientPruner(options) {};

  void calculateThreshold(Expr p, int batchNum) override {
     
    threshold_ = options_->get<float>("pruning-threshold");
  }

  void pruneNode(Expr p, int batchNum, bool rows = false) override {
	
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
    std::transform(tVec.begin(), tVec.end(), tVec.begin(), fabsf);
    std::sort(tVec.begin(), tVec.end());
    
    threshold_ = tVec[k];
  }

  void pruneNode(Expr p, int batchNum, bool rows = false) override {
	
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
    std::transform(tVec.begin(), tVec.end(), tVec.begin(), fabsf);
    std::sort(tVec.begin(), tVec.end());
    
    threshold_ = tVec[k];
  }

  void pruneNode(Expr p, int batchNum, bool rows = false) override {
	
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


class StructuredPruner : public IPruner {
public:
  StructuredPruner(Ptr<Options> options) : IPruner(options) {
  
    threshold_ = options_->get<float>("pruning-threshold");

  }
                                           
  virtual void calculateMask(Expr p, int batchNum, bool rows = false) = 0;

protected:
  std::vector<float> mask_; // mask of the same size as the Tensor
  float threshold_;
};

class RowcolPruner : public StructuredPruner {
public:
  RowcolPruner(Ptr<Options> options) : StructuredPruner(options) {  }

  void calculateMask(Expr p, int batchNum, bool rows = false) override {
    std::vector<float> vVec;
    std::vector<float> gVec;
    
    p->val()->get(vVec);
    p->grad()->get(gVec);
    
    // multiply vals with grads and do absolute
    // std::transform(tVec.begin(), tVec.end(), gVec.begin(), tVec.begin(), std::multiplies<float>());
    // std::transform(tVec.begin(), tVec.end(), tVec.begin(), fabs); 

    int h = p->shape()[0]; // height
    int w = p->shape()[1]; // width
    
    std::vector<float> thresholds; // we're gonna fill this with scores for each row/col

    // sum columns to create scores for each
    if (!rows) {
      thresholds.resize(w);
      std::cerr << p->name() << " " << p->shape() <<  " vVec size: " << vVec.size() << "gVec size: " << gVec.size() << " thresholds: " << thresholds.size() << std::endl;
      for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
          int e = w * i + j;
          // std::cerr << "e: " << e << "i: " << i << "j: " << j << std::endl;
          thresholds[j] += std::abs(vVec[e] * gVec[e]);
        }
      }
    
      // apply threshold

      std::cerr << "threshold * h " << threshold_ * h << std::endl;

      std::cout << "Scores: ";
      for (auto& t : thresholds) { std::cout << t << " "; }
      std::cout << std::endl; 

      std::transform (thresholds.begin(), thresholds.end(), 
                      thresholds.begin(), 
                      [&](float i) { if (i > threshold_ * h) return 1; else return 0; });
      
      std::cout << "Mask: ";
      for (auto& t : thresholds) { std::cout << t << " "; }
      std::cout << std::endl; 

      // for (int i = 0; i < thresholds.size(); i++) {
        
      // }
      //
      // ABORT_IF(true, "Breaking code uwu");

    }

    else {
      std::cerr << "ROWSSSSSSSSSSSSSSSSSSSSSSSSS" << std::endl;
      thresholds.resize(h);
      std::cerr << p->name() << " " << p->shape() <<  " vVec size: " << vVec.size() << "gVec size: " << gVec.size() << " thresholds: " << thresholds.size() << std::endl;
      for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
          int e = w * i + j;
          // std::cerr << "e: " << e << "i: " << i << "j: " << j << std::endl;
          thresholds[i] += std::abs(vVec[e] * gVec[e]);
        }
      }
    
      // apply threshold

      std::cerr << "threshold * w " << threshold_ * w << std::endl;

      std::cout << "Scores: ";
      for (auto& t : thresholds) { std::cout << t << " "; }
      std::cout << std::endl; 

      std::transform (thresholds.begin(), thresholds.end(), 
                      thresholds.begin(), 
                      [&](float i) { if (i > threshold_ * w) return 1; else return 0; });
      
      std::cout << "Mask: ";
      for (auto& t : thresholds) { std::cout << t << " "; }
      std::cout << std::endl; 


    
    }
        
    // std::sort(tVec.begin(), tVec.end());
    
  }

  void pruneNode(Expr p, int batchNum, bool rows = false) override {
	
    LOG(info, "PRUNING ROWCOL");
    calculateMask(p, batchNum, rows);
    
    // using namespace functional;
    // Element(_1 = if_then_else(abs(_1) < threshold_, 0, _1), p->val());
    
    // int cnt = 0;
    // std::vector<float> tVec;
    // p->val()->get(tVec);
    // for (auto x : tVec) {
      // if (x == 0) 
        // cnt++;
    // } 
    // LOG(info, "[{}] threshold: {} || zero count = {}/{}", p->name(), threshold_, cnt, p->val()->size());
  } 

};


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
    else if (pruningType == "threshold") {
      LOG_ONCE(info, "Pruning type selected: threshold coefficient");
      return New<ThresholdPruner>(options_);
    }
    else if (pruningType == "rowcol") {
      LOG_ONCE(info, "Pruning type selected: rowcol structured");
      return New<RowcolPruner>(options_);
    }
    else {
      LOG_ONCE(info, "Pruning type selected but not on the list? Returning nullptr, will break");
      return nullptr;
    }
  
  }

};


/* basically given the pruned param Tensor, also apply the same pruing to the other tensor 
     this is useful if you want to prune the gradient or moving avg as well..
*/
static void applyPrune(Tensor t, Tensor b) {
  using namespace functional;
  // if t is 0, then also set b to 0.
  Element(_1 = if_then_else(_2 == 0, 0, _1), b, t);
}

}

