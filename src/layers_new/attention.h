#pragma once

#include "graph/cached_expression.h"
#include "layers_new/neuralnet.h"

namespace marian {
namespace nn {

// Abstract base class for attention mechanisms
class AttentionLayer : public Layer, 
                       public IQuaternaryLayer {
protected:
  using Layer::namedLayers_;
  
public:
  AttentionLayer(Ptr<ExpressionGraph> graph) : Layer(graph) {}
  virtual ~AttentionLayer() = default;
};

class MultiplicativeAttention : public AttentionLayer {
protected:
  using AttentionLayer::namedLayers_;

public:
  Ptr<Dropout> attentionDropout;

  MultiplicativeAttention(Ptr<ExpressionGraph> graph, float dropoutProbability)
   : AttentionLayer(graph) {
    attentionDropout = New<Dropout>(graph, dropoutProbability);
    registerLayer(attentionDropout);
  }

  virtual ~MultiplicativeAttention() = default;

  virtual Expr apply(Expr query, Expr keys, Expr values, Expr logMask = nullptr) const override {
    int dimKeys = keys->shape()[-1];

    // softmax over batched dot product of query and keys (applied over all
    // time steps and batch entries), also add logMask for illegal connections

    // multiplicative attention with flattened softmax
    float scale = 1.0f / std::sqrt((float)dimKeys); // scaling to avoid extreme values due to matrix multiplication
    
    // query, keys and values: [beam depth * batch size, num heads, length, head dim]
    auto z = bdot(query, keys, false, true, scale); // [beam depth, batch size * num heads, max tgt length, max src length]

    // mask out garbage beyond end of sequences
    if(logMask)
      z = z + logMask;

    // take softmax along src sequence axis (-1)
    auto weights = softmax(z); // [beam depth, batch size * num heads, max tgt length, max src length]
    
#if 0 // @TODO: make this work again
    if(saveAttentionWeights)
      collectOneHead(weights, dimBeam);
#endif

    // optional dropout for attention weights
    weights = attentionDropout->apply(weights);

    // apply attention weights to values
    // weights: [beam depth, batch size * num heads, max tgt length, max src length]
    // values:  [beam depth, batch size * num heads, src length, head dim]
    auto output = bdot(weights, values);  // [beam depth, batch size * num heads, max tgt length, split vector dim]
    return output;
  }
};

template <class AttentionType> // Currently only used for MultiplicativeAttention
class MultiHeadAttention : public AttentionType {
protected:
  using AttentionType::namedLayers_;

private:
  IPtr<CachedExpr> cachedKh_; // cached result of key projection
  IPtr<CachedExpr> cachedVh_; // cached result of value projection

public:
  Ptr<Linear> qProj; // query projection layer
  Ptr<Linear> kProj; // key projection layer
  Ptr<Linear> vProj; // value projection layer
  Ptr<Linear> oProj; // output projection layer

  int numHeads;
  int attDim;
  int modelDim;

  MultiHeadAttention(Ptr<ExpressionGraph> graph,
                     int numHeads, 
                     int attDim, 
                     int modelDim,
                     float dropoutProbability)
    : AttentionType(graph, dropoutProbability),
      cachedKh_(new CachedExpr()), 
      cachedVh_(new CachedExpr()),
      numHeads(numHeads), 
      attDim(attDim), 
      modelDim(modelDim) {
    qProj = New<Linear>(graph, attDim);
    registerLayer(qProj);
    kProj = New<Linear>(graph, attDim);
    registerLayer(kProj);
    vProj = New<Linear>(graph, attDim);
    registerLayer(vProj);

    oProj = New<Linear>(graph, modelDim);
    registerLayer(oProj);
  }

  virtual ~MultiHeadAttention() = default;

private:
  // join beam and batch dimension and split model dimension in to heads and head dimension. We also need to transpose to 
  // be able to do an efficient batched matmul.
  Expr splitHeads(Expr input) const {
    int dimSteps = input->shape()[-2];
    int dimBatch = input->shape()[-3];
    int dimBeam  = input->shape()[-4];
    int dimDepth = attDim / numHeads;

    auto output = reshape(input, {dimBeam * dimBatch, dimSteps, numHeads, dimDepth});
    output      = transpose(output, {0, 2, 1, 3}); // [dimBatch*dimBeam, numHeads, dimSteps, dimDepth]
    output      = reshape(output, {dimBeam, dimBatch * numHeads, dimSteps, dimDepth});
    return output;
  }

  // Undoes the effects of the above function by reversing the transposition and reshaping back to original shape
  Expr joinHeads(Expr input) const {
    int dimDepth      = input->shape()[-1];
    int dimSteps      = input->shape()[-2];
    int dimBatchHeads = input->shape()[-3];
    int dimBeam       = input->shape()[-4];
    int dimModel      = numHeads * dimDepth;
    int dimBatch      = dimBatchHeads / numHeads;

    auto output = reshape(input, {dimBeam * dimBatch, numHeads, dimSteps, dimDepth});
    output      = transpose(output, {0, 2, 1, 3});
    output      = reshape(output, {dimBeam, dimBatch, dimSteps, dimModel});
    return output;
  }

public:
  virtual Expr apply(Expr query, Expr keys, Expr values, Expr mask) const override {
    auto qh = splitHeads(qProj->apply(query));

    // @TODO: in original implementation we use shape()->elements(), dunno why
    auto equal = [](Expr a, Expr b) { return a->shape() == b->shape(); };
    
    // these two get conditionally recomputed if their size changes according to criterion above
    auto kh = cachedKh_->apply(keys,   [this](Expr keys)   { 
      return splitHeads(kProj->apply(keys)); 
    }, equal);
    
    auto vh = cachedVh_->apply(values, [this](Expr values) { 
      return splitHeads(vProj->apply(values)); 
    }, equal);

    auto output = AttentionType::apply(qh, kh, vh, mask);

    output = joinHeads(output);
    output = oProj->apply(output);

    return output;
  }

  virtual void clear() override {
    Layer::clear();
    cachedKh_->clear();
    cachedVh_->clear();
  }
};

static Ptr<AttentionLayer> attentionFromOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options) {
  // @TODO: currently this does nothing as it isn't set anywhere
  std::string selfAttentionType = options->get<std::string>("transformer-encoder-attention", "default"); // currently only default

  // in the future we might add SingleHead or Additive or LSH-based as in Reformer
  if(selfAttentionType == "default") {
    int numHeads = options->get<int>("transformer-heads");
    int modelDim = options->get<int>("dim-emb");
    float attentionDropoutProbability = options->get<float>("transformer-dropout-attention", 0.f);

    return New<MultiHeadAttention<MultiplicativeAttention>>(graph, numHeads, modelDim, modelDim, attentionDropoutProbability);
  }
  else {
    ABORT("Unknown transformer encoder attention type: {}", selfAttentionType);
  }
}

} // namespace nn
} // namespace marian
