#pragma once

#include "graph/cached_expression.h"
#include "layers_new/decoder.h"
#include "layers_new/neuralnet.h"

namespace marian {

// specialized operator for faster logMask computation
Expr logMask(Expr mask, int numHeads);

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
    
    // query, keys and values: [dimBeam, dimBatch * numHeads, (dimQuery|dimKeys=dimValues), dimHead]
    auto z = bdot(query, keys, false, true, scale); // [dimBeam, dimBatch * numHeads, dimQuery, dimKeys]

    // mask out garbage beyond end of sequences
    if(logMask)
      z = z + logMask;

    // take softmax along src sequence axis (-1)
    auto weights = softmax(z); // [dimBeam, dimBatch * numHeads, dimQuery, dimKeys]
    
#if 0 // @TODO: make this work again
    if(saveAttentionWeights)
      collectOneHead(weights, dimBeam);
#endif

    // optional dropout for attention weights
    weights = attentionDropout->apply(weights);

    // apply attention weights to values
    // weights: [dimBeam, dimBatch * numHeads, dimQuery, dimKeys]
    // values:  [dimBeam, dimBatch * numHeads,  dimKeys, dimHead]
    auto output = bdot(weights, values);  // [dimBeam, dimBatch * numHeads, dimQuery, dimHead]
    return output;
  }
};

// Base class for multi-head attention
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

protected:
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
  // Apply the multi-head attention to the given query, keys and values
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

    auto output  = AttentionType::apply(qh, kh, vh, mask);

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

// Base class for attention mask processors
// Attention mask processors are used to process a given attention mask before it is used in an attention computation.
struct AttentionMaskProcessor : public LayerWithOptions, public IBinaryLayer, public IBinaryDecoderLayer {
  int numHeads{1};

  AttentionMaskProcessor(Ptr<ExpressionGraph> graph,
                         Ptr<Options> options)
    : LayerWithOptions(graph, options), 
      numHeads(opt<int>("transformer-heads", 1)) {}

  virtual ~AttentionMaskProcessor() = default;
  
  virtual Expr apply(Expr /*query*/, Expr mask) const override {
    if(!mask)
      return nullptr;

    // @TODO eventually remove this branch. For now we keep it for documentation purposes
#if 0
    // LayerAttention expects mask in a different layout
    int dimBatch = mask->shape()[-3];
    int dimKeys  = mask->shape()[-2];

    mask = reshape(mask, {dimBatch, 1, 1, dimKeys}); // [batch size, num heads broadcast=1, max length broadcast=1, max length]

    float maskFactor = std::max(NumericLimits<float>(mask->value_type()).lowest / 2.f, -99999999.f); // to make sure we do not overflow for fp16
    auto logMask = (1 - mask) * maskFactor;
    logMask      = reshape(repeat(logMask, numHeads, -3), {1, dimBatch * numHeads, 1, dimKeys});
    return logMask;
#else
    // shape of mask should be [1, dimBatch, dimKeys, 1]
    // this does all the above work in one step
    return marian::logMask(mask, numHeads); // [1, dimBatch * numHeads, 1, dimKeys]
#endif
  }

  virtual Expr apply(Expr query, Expr mask, Ptr<DecoderState> /*state*/) const override {
    return apply(query, mask);
  }
};

// Factory function to create attention layers from options
Ptr<AttentionLayer> attentionFromOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options);

// Factory function to create attention mask processors from options
Ptr<AttentionMaskProcessor> attentionMaskProcessorFromOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options);

} // namespace nn
} // namespace marian
