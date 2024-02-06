#include "graph/node_operators_unary.h"
#include "layers_new/attention.h"
#include "layers_new/alibi.h"

namespace marian {
namespace nn {

// Factory function to create attention layers from options
Ptr<AttentionLayer> attentionFromOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options, bool enableCache) {
  // @TODO: currently this does nothing as it isn't set anywhere
  std::string selfAttentionType = options->get<std::string>("transformer-encoder-attention", "default"); // currently only default

  // in the future we might add SingleHead or Additive or LSH-based as in Reformer
  if(selfAttentionType == "default") {
    int numHeads = options->get<int>("transformer-heads");
    int modelDim = options->get<int>("transformer-dim-model", options->get<int>("dim-emb"));

    float attentionDropoutProbability = options->get<float>("transformer-dropout-attention", 0.f);

    return New<MultiHeadAttention>(graph, numHeads, modelDim, modelDim, attentionDropoutProbability, enableCache);
  }
  else {
    ABORT("Unknown transformer encoder attention type: {}", selfAttentionType);
  }
}

// Factory function to create attention mask processors from options
Ptr<MaskProcessor> maskProcessorFromOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options) {
  // currently only default or alibi
  std::string processorType = options->get<std::string>("transformer-attention-mask", "default");
  if(processorType == "default") {
    return New<AttentionMaskProcessor>(graph, options);
  } else if(processorType == "alibi") {
    return New<AlibiAttentionMaskProcessor>(graph, options);
  } else {
    ABORT("Unknown transformer attention mask processor type: {}", processorType);
  }
}

Ptr<DecoderMaskProcessor> selfMaskProcessorFromOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options) {
  auto autoRegType = options->get<std::string>("transformer-decoder-autoreg", "self-attention");
  if(autoRegType == "rnn") {
    // creates a dummy processor that returns an unprocessed mask
    return New<DummyDecoderMaskProcessor>(graph, options);
  } else if(autoRegType == "self-attention") {
    // here we will return modified log masks for self-attention
    std::string processorType = options->get<std::string>("transformer-attention-mask", "default");
    if(processorType == "alibi") {
      return New<AlibiDecoderAttentionMaskProcessor>(graph, options, /*addCausalMask=*/true);
    } else {
      return New<DecoderAttentionMaskProcessor>(graph, options, /*addCausalMask=*/true);
    }
  } else {
    ABORT("Unknown transformer decoder autoregressive type: {}", autoRegType);
  }
}

Ptr<DecoderMaskProcessor> contextDecoderMaskProcessorFromOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options) {
  std::string processorType = options->get<std::string>("transformer-attention-mask", "default");
  if(processorType == "alibi") {
    return New<AlibiDecoderAttentionMaskProcessor>(graph, options, /*addCausalMask=*/false);
  } else {
    return New<DecoderAttentionMaskProcessor>(graph, options, /*addCausalMask=*/false);
  }
}

}  // namespace nn

// specialized faster operator for log-mask computation
class LogMaskNode : public UnaryNodeOp {
private:
  int numHeads_{8};

  Shape newShape(Expr mask, int numHeads) {
    // incoming mask is expected to have shape [dimBatch, 1, 1, dimKeys]
    // see the reshape below in the logMask function
    int dimBatch = mask->shape()[-4];
    int dimKeys  = mask->shape()[-1];
    return { dimBatch, numHeads,       1, dimKeys };
  }

public:
  LogMaskNode(Expr mask, int numHeads)
  : UnaryNodeOp(mask, newShape(mask, numHeads)),
    numHeads_(numHeads)
  {}

  NodeOps forwardOps() override {
    // compared to the multi-operation code this does conversion and broadcasting in one step
    using namespace functional;
    return { NodeOp(Element(_1 = log(_2), val_, child(0)->val())) };
  }

  NodeOps backwardOps() override {
    if(!trainable())
      return { };

    using namespace functional;
    return { NodeOp(Add(_1 / _2, child(0)->grad(), adj_, child(0)->val())) };
  }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, numHeads_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<LogMaskNode>(node);
    if(!cnode)
      return false;
    if(numHeads_ != cnode->numHeads_)
      return false;
    return true;
  }

  const std::string type() override { return "log-mask"; }
};

Expr logMask(Expr mask, int numHeads, bool addCausalMask) {
  // incoming mask has shape [1, dimBatch, dimKeys, 1]
  int dimBatch = mask->shape()[-3];
  int dimKeys  = mask->shape()[-2];
  mask = reshape(mask, {dimBatch, 1, 1, dimKeys});
  auto logMask = Expression<LogMaskNode>(mask, numHeads); // [dimBatch, numHeads, 1, dimKeys]
  logMask = reshape(logMask, {1, dimBatch * numHeads, 1, dimKeys});

  // @TODO: this is needlessly slow, integrate with the above in special kernel
  if(addCausalMask) {
    // add causal mask to logMask
    std::vector<float> vMask(dimKeys * dimKeys, 0.f);
    for(int i = 0; i < dimKeys; ++i)
      for(int j = i + 1; j < dimKeys; ++j)
        vMask[i * dimKeys + j] = -std::numeric_limits<float>::infinity();

    auto triangle = mask->graph()->constant({1, 1, dimKeys, dimKeys}, inits::fromVector(vMask));
    logMask = minimum(logMask, triangle); // [1, dimBatch * numHeads, dimKeys, dimKeys]
  }

  return logMask;
}

}  // namespace marian
