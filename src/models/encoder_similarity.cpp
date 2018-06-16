#include "encoder_similarity.h"

namespace marian {

EncoderSimilarity::EncoderSimilarity(Ptr<Options> options)
    : options_(options),
      prefix_(options->get<std::string>("prefix", "")),
      inference_(options->get<bool>("inference", false)) {}

std::vector<Ptr<EncoderBase>>& EncoderSimilarity::getEncoders() {
  return encoders_;
}

void EncoderSimilarity::push_back(Ptr<EncoderBase> encoder) {
  encoders_.push_back(encoder);
}

void EncoderSimilarity::load(Ptr<ExpressionGraph> graph,
                  const std::string& name,
                  bool markedReloaded) {
  graph->load(name, markedReloaded && !opt<bool>("ignore-model-config"));
}

void EncoderSimilarity::clear(Ptr<ExpressionGraph> graph) {
  graph->clear();
  for(auto& enc : encoders_)
    enc->clear();
}

Expr EncoderSimilarity::build(Ptr<ExpressionGraph> graph,
                           Ptr<data::CorpusBatch> batch,
                           bool clearGraph) {
  using namespace keywords;

  if(clearGraph)
    clear(graph);

  auto es1 = encoders_[0]->build(graph, batch);
  auto es2 = encoders_[1]->build(graph, batch);

  auto v1 = sum(es1->getContext() * es1->getMask(), axis = -3);
  auto v2 = sum(es2->getContext() * es2->getMask(), axis = -3);

  return scalar_product(v1, v2, axis = -1) /
    (sqrt(scalar_product(v1, v1, axis = -1)) *
     sqrt(scalar_product(v2, v2, axis = -1)));

}

Expr EncoderSimilarity::build(Ptr<ExpressionGraph> graph,
                           Ptr<data::Batch> batch,
                           bool clearGraph) {
  auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);
  return build(graph, corpusBatch, clearGraph);
}

}
