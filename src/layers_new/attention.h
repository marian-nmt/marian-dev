#pragma once

#include "graph/cached_expression.h"
#include "layers_new/decoder.h"
#include "layers_new/neuralnet.h"

namespace marian {

/**
 * Specialized operator for faster logMask computation
 */
Expr logMask(Expr mask, int numHeads, bool addCausalMask);

namespace nn {

/**
 * Abstract base class for attention mechanisms
 */
class AttentionLayer : public Layer,
                       public IQuaternaryLayer {
protected:
  using Layer::namedLayers_;

public:
  AttentionLayer(Ptr<ExpressionGraph> graph) : Layer(graph) {}
  virtual ~AttentionLayer() = default;
};

/**
 * Base class for attention layers that collect attention weights
 */
class AttentionCollector {
private:
  mutable std::vector<Expr> alignments_; // @TODO: rename to something more accurate

public:
  bool saveAttentionWeights{false};
  int numHeads{8};

  AttentionCollector(bool saveAttentionWeights, int numHeads = 8)
    : saveAttentionWeights(saveAttentionWeights), numHeads(numHeads) {}

  void collectOneHead(Expr weights) const {
    // weights: [dimBeam, dimBatch * numHeads, dimQuery|1, dimKeys]

    int dimBeam       = weights->shape()[-4];
    int dimBatchHeads = weights->shape()[-3];
    int dimQuery      = weights->shape()[-2]; // (max) length of trg sequence, or 1 in decoding
    int dimKeys       = weights->shape()[-1]; // (max) length of src sequence

    int dimBatch = dimBatchHeads / numHeads;

    weights = reshape(weights, {dimBeam * dimBatch, numHeads, dimQuery, dimKeys});
    auto head0 = slice(weights, -3, 0); // [dimBeam * dimBatch, 1, dimQuery, dimKeys]

    // reshape and transpose to match the format guided_alignment expects
    head0 = reshape(head0, {dimBeam, dimBatch, dimQuery, dimKeys});
    head0 = transpose(head0, {0, 3, 1, 2}); // [beam depth, dimKeys, dimBatch, dimQuery|1]

    // save only last alignment set. For training this will be all alignments,
    // for translation only the last one. Also split alignments by target words.
    // @TODO: make splitting obsolete
    // @TODO: why is this even here?
    alignments_.clear();
    for(int i = 0; i < dimQuery; ++i) { // loop over all trg positions. In decoding, there is only one.
      alignments_.push_back(slice(head0, -1, i)); // [tgt index][beam depth, max src length, batch size, 1] P(src pos|trg pos, beam index, batch index)
    }
  }

  const std::vector<Expr>& getAlignments() const {
    return alignments_;
  }

  void clear() {
    alignments_.clear();
  }
};

/**
 * Base class for multiplicative attention layers (can collect attention weights)
 */
class MultiplicativeAttention : public AttentionLayer, public AttentionCollector {
protected:
  using AttentionLayer::namedLayers_;

public:
  Ptr<Dropout> attentionDropout;

  MultiplicativeAttention(Ptr<ExpressionGraph> graph, float dropoutProbability, bool saveAttentionWeights = false)
   : AttentionLayer(graph), AttentionCollector(saveAttentionWeights) {
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

    if(saveAttentionWeights) {
      collectOneHead(weights);
    }

    // optional dropout for attention weights
    weights = attentionDropout->apply(weights);

    // apply attention weights to values
    // weights: [dimBeam, dimBatch * numHeads, dimQuery, dimKeys]
    // values:  [dimBeam, dimBatch * numHeads,  dimKeys, dimHead]
    auto output = bdot(weights, values);  // [dimBeam, dimBatch * numHeads, dimQuery, dimHead]
    return output;
  }

  virtual void clear() override {
    AttentionLayer::clear();
    AttentionCollector::clear();
  }
};

/**
 * Extended multiplicative attention layer with multiple heads
 * and separate query, key and value projections, as well as
 * an output projection.
 */
class MultiHeadAttention : public MultiplicativeAttention {
protected:
  using MultiplicativeAttention::namedLayers_;
  using AttentionCollector::saveAttentionWeights;

private:
  bool enableCache_{false};
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
                     float dropoutProbability,
                     bool enableCache = false)
    : MultiplicativeAttention(graph, dropoutProbability),
      enableCache_(enableCache),
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
    // @TODO: implement custom bdot to avoid splitHeads/joinHeads
    // @TODO: explore FlashAttention-like cpu implementation
    auto qh = splitHeads(qProj->apply(query));

    // if enabledCache_ is true, we cache the results of the key and value projections
    // otherwise equal is always false and the key and value projections are recomputed
    Expr kh, vh;
    if(enableCache_) {
      // @TODO: in original implementation we use shape()->elements(), dunno why
      auto equal = [](Expr a, Expr b) { return a->shape() == b->shape(); };
      // these two get conditionally recomputed if their size changes according to criterion above
      kh = cachedKh_->apply(keys,   [this](Expr keys)   { return splitHeads(kProj->apply(keys)); }, equal);
      vh = cachedVh_->apply(values, [this](Expr values) { return splitHeads(vProj->apply(values)); }, equal);
    } else {
      kh = splitHeads(kProj->apply(keys));
      vh = splitHeads(vProj->apply(values));
    }

    auto output  = MultiplicativeAttention::apply(qh, kh, vh, mask);

    // @TODO: combine joinHeads and apply in one matrix multiplication via striding
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

/**
 * Base class for mask processors.
 */
struct MaskProcessor : public LayerWithOptions, public IBinaryLayer {
  IPtr<CachedExpr> cachedMask_;

  MaskProcessor(Ptr<ExpressionGraph> graph,
                Ptr<Options> options)
    : LayerWithOptions(graph, options),
    cachedMask_(new CachedExpr()) {}

  virtual ~MaskProcessor() = default;

  void clear() override {
    LayerWithOptions::clear();
    cachedMask_->clear();
  }
};

/**
 * Base class for decoder mask processors.
 */
struct DecoderMaskProcessor : public LayerWithOptions, public IBinaryDecoderLayer {
  bool addCausalMask{false};
  IPtr<CachedExpr> cachedMask_;

  DecoderMaskProcessor(Ptr<ExpressionGraph> graph,
                       Ptr<Options> options,
                       bool addCausalMask = false)
    : LayerWithOptions(graph, options),
      addCausalMask(addCausalMask),
      cachedMask_(new CachedExpr()) {}

  virtual ~DecoderMaskProcessor() = default;

  void clear() override {
    LayerWithOptions::clear();
    cachedMask_->clear();
  }
};

/**
 * Attention mask processors are used to process a given attention mask
 * before it is used in an attention computation.
 */
struct AttentionMaskProcessor : public MaskProcessor {
  int numHeads{1};

  AttentionMaskProcessor(Ptr<ExpressionGraph> graph,
                         Ptr<Options> options)
    : MaskProcessor(graph, options),
      numHeads(opt<int>("transformer-heads", 1)) {}

  virtual ~AttentionMaskProcessor() = default;

  virtual Expr apply(Expr /*query*/, Expr mask) const override {
    if(!mask)
      return nullptr;

    // shape of input `mask` should be [1, dimBatch, dimKeys, 1]
    // output shape will be // [1, dimBatch * numHeads, 1, dimKeys] if addCausalMask is false
    // or [1, dimBatch * numHeads, dimKeys, dimKeys] if addCausalMask is true
    auto processMask = [this](Expr mask) { return marian::logMask(mask, numHeads, /*addCausalMask=*/false); };

    // recompute the mask if input mask changes (different memory address), otherwise return cached version
    auto equal       = [](Expr a, Expr b) { return a == b; };

    // recompute the mask if the shape changes, otherwise return cached version
    return cachedMask_->apply(mask, processMask, equal);
  }
};

/**
 * Base class for decoder attention mask processors. Attention mask processors are used to
 * process a given attention mask before it is used in an attention computation.
 * Decoder attention mask processors can take advantage of information from the decoder state.
 */
struct DecoderAttentionMaskProcessor : public DecoderMaskProcessor {
  int numHeads{1};

  DecoderAttentionMaskProcessor(Ptr<ExpressionGraph> graph,
                                Ptr<Options> options,
                                bool addCausalMask = false)
    : DecoderMaskProcessor(graph, options, addCausalMask),
      numHeads(opt<int>("transformer-heads", 1)) {}

  virtual ~DecoderAttentionMaskProcessor() = default;

  virtual void initState(Ptr<DecoderState> /*state*/) const override {}

  virtual Expr apply(Expr query, Expr mask, Ptr<DecoderState> /*state*/) const override {
     if(!mask)
      return nullptr;

    // shape of input `mask` should be [1, dimBatch, dimKeys, 1]
    // output shape will be // [1, dimBatch * numHeads, 1, dimKeys] if addCausalMask is false
    // or [1, dimBatch * numHeads, dimKeys, dimKeys] if addCausalMask is true
    auto processMask = [this](Expr mask) { return marian::logMask(mask, numHeads, addCausalMask); };

    // recompute the mask if input mask changes (different memory address), otherwise return cached version
    auto equal       = [](Expr a, Expr b) { return a == b; };

    // recompute the mask if the shape changes, otherwise return cached version
    return cachedMask_->apply(mask, processMask, equal);
  }
};

/**
 * Dummy decoder mask processor that returns the unprocessed mask, used for RNN autoregressive decoding
 */
struct DummyDecoderMaskProcessor : public DecoderMaskProcessor {
  DummyDecoderMaskProcessor(Ptr<ExpressionGraph> graph,
                            Ptr<Options> options)
    : DecoderMaskProcessor(graph, options, /*addCausalMask=*/false) {}

  virtual ~DummyDecoderMaskProcessor() = default;

  virtual void initState(Ptr<DecoderState> /*state*/) const override {}

  virtual Expr apply(Expr /*query*/, Expr mask, Ptr<DecoderState> /*state*/) const override {
    return mask;
  }
};

/**
 * Factory function to create attention layers from options
 */
Ptr<AttentionLayer> attentionFromOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options, bool enableCache = false);

/**
 * Factory function to create mask processors from options
 */
Ptr<MaskProcessor>        maskProcessorFromOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options);
Ptr<DecoderMaskProcessor> selfMaskProcessorFromOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options);
Ptr<DecoderMaskProcessor> contextDecoderMaskProcessorFromOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options);

} // namespace nn
} // namespace marian
