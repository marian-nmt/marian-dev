#pragma once

#include "layers_new/transformer.h"

#include "models/encoder.h"
#include "layers/constructors.h"

namespace marian {
namespace models {

class BleurtTypeEmbeddingLayer : public nn::LayerWithOptions {
public:
  Expr embeddings;

  BleurtTypeEmbeddingLayer(Ptr<ExpressionGraph> graph, Ptr<Options> options) 
  : LayerWithOptions(graph, options) {}

  virtual ~BleurtTypeEmbeddingLayer() = default;
  
  Expr apply(Ptr<data::SubBatch> subBatch) const {
    int dimEmb   = opt<int>("dim-emb");
    int dimTypes = opt<int>("bert-type-vocab-size", 2);

    // Embedding layer initialization should depend only on embedding size, hence fanIn=false
    auto initFunc = inits::glorotUniform(/*fanIn=*/false, /*fanOut=*/true); // -> embedding vectors have roughly unit length
    registerParameterLazy(embeddings, Shape({dimTypes, dimEmb}), initFunc);

    const auto& words = subBatch->data();
    const auto vocab = subBatch->vocab();
    
    // Get word id of special symbols
    Word sepId   = vocab->getEosId();

    int dimBatch = (int)subBatch->batchSize();
    int dimTime  = (int)subBatch->batchWidth();
    const size_t maxSentPos = dimTypes;

    // create indices for BERT sentence embeddings A and B
    std::vector<IndexType> sentenceIndices(dimBatch * dimTime, 0); // each word is either in sentence A or B
    std::vector<IndexType> sentPos(dimBatch, 0); // initialize each batch entry with being A [0]
    for(int i = 0; i < dimTime; ++i) {   // advance word-wise
      for(int j = 0; j < dimBatch; ++j) { // scan batch-wise
        int k = i * dimBatch + j;
        sentenceIndices[k] = sentPos[j]; // set to current sentence position for batch entry, max position 1.
        if(words[k] == sepId && sentPos[j] < maxSentPos) { // if current word is a separator and not beyond range
          sentPos[j]++;                   // then increase sentence position for batch entry (to B [1])
        }
      }
    }

    return reshape(rows(embeddings, sentenceIndices), {dimTime, dimBatch, dimEmb});
  }
};

struct BleurtEncoder final : public nn::TransformerEncoder {
  Ptr<nn::Linear> eProj;

  BleurtEncoder(Ptr<ExpressionGraph> graph, 
               Ptr<Options> options) 
    : TransformerEncoder(graph, options) {
    
    eProj = New<nn::Linear>(graph, opt<int>("transformer-dim-model"));
    registerLayer(eProj);

    for(auto norm : allLayers<nn::LayerNorm>())
      norm->eps = 1e-12f; // hard-coded as in original BLEURT model
  }

  Expr apply(Expr input, Expr mask) const override {
    auto output = marian::nn::swapTimeBatch(input); // [beam depth=1, batch size, max length, vector dim]
    mask = marian::nn::swapTimeBatch(mask);   // [beam depth=1, batch size, max length, vector dim=1]
  
    // apply positional embeddings to contextual input
    output = positionEmbedding->apply(output);

    // apply dropout or layer-norm to embeddings if required
    output = preprocessor->apply(output);
    
    // scale from 256 to 1152
    output = eProj->apply(output);
    
    // traverse the layers, use the same mask for each
    for(auto layer : *layers)
      output = layer->apply(output, mask); 

    return output;
  }
};

// Wrapper for backwards compatibility that uses current encoder/decoder framework
struct BleurtBatchEncoder final : public nn::LayerWithOptions, 
                                  public nn::IEmbeddingLayer,  // TransformerBatchEncoder is an IEmbeddingLayer that produces contextual embeddings
                                  public EncoderBase {         // @TODO: should all encoders be IEmbeddingLayer?
  Ptr<BleurtTypeEmbeddingLayer> typeEmbedding;
  Ptr<BleurtEncoder> encoder;
  
  BleurtBatchEncoder(Ptr<ExpressionGraph> graph, 
                    Ptr<Options> options)
    : LayerWithOptions(graph, options),
      EncoderBase(graph, options)
  {
    typeEmbedding = New<BleurtTypeEmbeddingLayer>(graph, options);
    registerLayer(typeEmbedding);

    encoder = New<BleurtEncoder>(graph, options);
    registerLayer(encoder);
  }

  // @TODO: subBatch should be of type Expr
  virtual std::tuple<Expr/*embeddings*/, Expr/*mask*/> apply(Ptr<data::SubBatch> subBatch) const override {
    auto embeddingLayer = getEmbeddingLayer(EncoderBase::opt<bool>("ulr", false));
    const auto& [batchEmbeddings, batchMask] = embeddingLayer->apply(subBatch);
    
#if 1
    auto typeEmbeddings = typeEmbedding->apply(subBatch);
    auto embeddings = batchEmbeddings + typeEmbeddings;
#else
    auto embeddings = batchEmbeddings;
#endif

    auto batchContext = encoder->apply(embeddings, batchMask); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
    return std::make_tuple(batchContext, batchMask);
  }

  virtual Expr apply(const Words& words, const Shape& shape) const override final {
    return applyIndices(toWordIndexVector(words), shape);
  }

  // alternative from indices directly
  virtual Expr applyIndices(const std::vector<WordIndex>& wordIndices, const Shape& shape) const override final {
    auto embeddingLayer = getEmbeddingLayer(EncoderBase::opt<bool>("ulr", false));
    Expr batchEmbedding = embeddingLayer->applyIndices(wordIndices, shape);
    auto batchContext = encoder->apply(batchEmbedding, /*mask=*/nullptr); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
    return batchContext;
  }

  // @TODO: currently here for backwards compat, should be replaced with apply()
  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::CorpusBatch> batch) override {
#if 1
    // @TODO: this should be removed, currently hack to init graph. Should happen in graph groups and constructors
    EncoderBase::graph_ = graph;
    setGraph(graph);
    // This makes sure that the graph passed into the model during construction and now evaluation are identical.
    // A good check to have for catching weird situations early. 
    ABORT_IF(this->graph() != graph, "Graph used for construction and graph parameter do not match");
#endif

    // @TODO: this needs to convert to a BERT-batch
    
    const auto& [batchEmbedding, batchMask] = apply((*batch)[batchIndex_]);
    return New<EncoderState>(batchEmbedding, batchMask, batch);
  }

  virtual void clear() override {
    Layer::clear();
  }
};

class BleurtPooler final : public nn::LayerWithOptions, 
                           public PoolerBase {
private:
  Ptr<nn::Sequential> layers;
  std::mt19937 rng{(uint32_t)Config::seed};

public:
  BleurtPooler(Ptr<ExpressionGraph> graph, Ptr<Options> options)
  : LayerWithOptions(graph, options),
    PoolerBase(graph, options) {
    
    float dropoutProb = 0.f;
    layers = New<nn::Sequential>(
      graph,
      New<nn::Linear>(graph, LayerWithOptions::opt<int>("transformer-dim-model")), // @TODO: get rid of amibuigity
      New<nn::Tanh>(graph),
      New<nn::Dropout>(graph, dropoutProb),
      New<nn::Linear>(graph, 1)
    );
    
    registerLayer(layers);
  }

  std::vector<Expr> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
#if 1
    // @TODO: this should be removed, currently hack to init graph. Should happen in graph groups and constructors
    PoolerBase::graph_ = graph;
    setGraph(graph);
    // This makes sure that the graph passed into the model during construction and now evaluation are identical.
    // A good check to have for catching weird situations early. 
    ABORT_IF(this->graph() != graph, "Graph used for construction and graph parameter do not match");
#endif

    auto modelType = LayerWithOptions::opt<std::string>("type");
    
    auto emb = slice(encoderStates[0]->getContext(), -2, 0);
    emb = marian::cast(emb, Type::float32);
    
    Expr output;
    if(LayerWithOptions::opt<int>("usage") == (int)models::usage::evaluating) {
      output = layers->apply(emb);
      int dimBatch = output->shape()[-3];
      output = reshape(output, {dimBatch, 1, 1});
      return { output };
    } else {
      ABORT("Usage other than evaluating not implemented");  
    }
  }

  void clear() override {}
};

} // namespace models
} // namespace marian

