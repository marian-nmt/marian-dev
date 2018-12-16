// TODO: move to backend, into graph/
#pragma once

#include "common/config.h"
#include "tensors/tensor.h"

#include <functional>
#include <random>

namespace marian {

class ExpressionGraph;

namespace inits {

class NodeInitializer {
private:
  Ptr<ExpressionGraph> graph_;

public:
  virtual void operator()(Tensor t) = 0;
  void setGraph(Ptr<ExpressionGraph> graph) { graph_ = graph; }
};

Ptr<NodeInitializer> zeros();
Ptr<NodeInitializer> ones();
Ptr<NodeInitializer> fromValue(float v);

Ptr<NodeInitializer> eye(float val = 1.f);
Ptr<NodeInitializer> normal(float mean = 0.f, float stddev = 1.f);
Ptr<NodeInitializer> uniform(float a = 0.f, float b = 1.f);
Ptr<NodeInitializer> bernoulli(float p, float scale = 1.f);

Ptr<NodeInitializer> glorotUniform();
Ptr<NodeInitializer> glorotNormal();

Ptr<NodeInitializer> dropout(float dropoutProbabilty);
Ptr<NodeInitializer> gumbel();
Ptr<NodeInitializer> dummy();

Ptr<NodeInitializer> fromVector(const std::vector<float>& v);
Ptr<NodeInitializer> fromVector(const std::vector<IndexType>& v);
Ptr<NodeInitializer> fromItem(const io::Item& item);
Ptr<NodeInitializer> fromSparseVector(std::pair<std::vector<size_t>, std::vector<float>>& v);
Ptr<NodeInitializer> fromWord2vec(const std::string& file,
                                  int dimVoc,
                                  int dimEmb,
                                  bool normalize = false);

}  // namespace inits

}  // namespace marian
