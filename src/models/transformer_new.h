#pragma once

#include "layers_new/transformer.h"

#include "models/encoder.h"
#include "models/decoder.h"
#include "models/states.h"
#include "layers/constructors.h"

namespace marian {

// Wrapper for backwards compatibility that uses current encoder/decoder framework
struct TransformerBatchEncoder : public nn::LayerWithOptions, 
                                 public nn::IEmbeddingLayer,  // TransformerBatchEncoder is an IEmbeddingLayer that produces contextual embeddings
                                 public EncoderBase {         // @TODO: should all encoders be IEmbeddingLayer?
  Ptr<nn::TransformerEncoder> encoder;

  TransformerBatchEncoder(Ptr<ExpressionGraph> graph, 
                          Ptr<Options> options)
    : LayerWithOptions(graph, options),
      EncoderBase(graph, options)
  {
    encoder = New<nn::TransformerEncoder>(graph, options);
    registerLayer(encoder);
  }

  // @TODO: subBatch should be of type Expr
  virtual std::tuple<Expr/*embeddings*/, Expr/*mask*/> apply(Ptr<data::SubBatch> subBatch) const override {
    // @TODO: this is still using the bad old interface
    auto embeddingLayer = getEmbeddingLayer(EncoderBase::opt<bool>("ulr", false));
    const auto& [batchEmbedding, batchMask] = embeddingLayer->apply(subBatch);
    auto batchContext = encoder->apply(batchEmbedding, batchMask); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
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
    
    const auto& [batchEmbedding, batchMask] = apply((*batch)[batchIndex_]);
    return New<EncoderState>(batchEmbedding, batchMask, batch);
  }

  virtual void clear() override {
    Layer::clear();
  }
};

// Wrapper for backwards compatibility that uses current encoder/decoder framework
class TransformerBatchDecoder : public nn::LayerWithOptions, 
                                public DecoderBase {

  Ptr<nn::TransformerDecoder> decoder;
  Ptr<mlp::Output> output_; 

  void lazyCreateOutputLayer()
  {
    using db = DecoderBase;

    if(output_) // create it lazily
      return;

    int dimTrgVoc = db::opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    auto outputFactory = mlp::OutputFactory(
        "prefix", prefix_ + "_ff_logit_out",
        "dim", dimTrgVoc,
        "vocab", db::opt<std::vector<std::string>>("vocabs")[batchIndex_], // for factored outputs
        "output-omit-bias", db::opt<bool>("output-omit-bias", false),
        "output-approx-knn", db::opt<std::vector<int>>("output-approx-knn", {}),
        "lemma-dim-emb", db::opt<int>("lemma-dim-emb", 0),
        "lemma-dependency", db::opt<std::string>("lemma-dependency", ""), // for factored outputs
        "factors-combine", db::opt<std::string>("factors-combine", "")); // for factored outputs

    if(db::opt<bool>("tied-embeddings") || db::opt<bool>("tied-embeddings-all"))
      outputFactory.tieTransposed(db::opt<bool>("tied-embeddings-all") || db::opt<bool>("tied-embeddings-src") ? "Wemb" : prefix_ + "_Wemb");

    output_ = std::dynamic_pointer_cast<mlp::Output>(outputFactory.construct(graph())); // (construct() returns only the underlying interface)
  }

public:
  TransformerBatchDecoder(Ptr<ExpressionGraph> graph, Ptr<Options> options) 
  : LayerWithOptions(graph, options), DecoderBase(graph, options) {
    
    decoder = New<nn::TransformerDecoder>(graph, options);
    registerLayer(decoder);

  }

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch,
                                       std::vector<Ptr<EncoderState>>& encStates) override {

#if 1
    // @TODO: this should be removed, currently hack to init graph. Should happen in graph groups and constructors
    DecoderBase::graph_ = graph;
    setGraph(graph);
    // This makes sure that the graph passed into the model during construction and now evaluation are identical.
    // A good check to have for catching weird situations early. 
    ABORT_IF(this->graph() != graph, "Graph used for construction and graph parameter do not match");
#endif

    std::string layerType = DecoderBase::opt<std::string>("transformer-decoder-autoreg", "self-attention");
    if (layerType == "rnn") {
      int dimBatch = (int)batch->size();
      int dim = DecoderBase::opt<int>("dim-emb");

      auto start = graph->constant({1, 1, dimBatch, dim}, inits::zeros());
      rnn::States startStates(DecoderBase::opt<size_t>("dec-depth"), {start, start});

      // don't use TransformerState for RNN layers
      return New<DecoderState>(startStates, Logits(), encStates, batch, /*isBatchMajor=*/false);
    }
    else {
      rnn::States startStates;
      return New<DecoderState>(startStates, Logits(), encStates, batch, /*isBatchMajor=*/true);
    }
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) override {
#if 1 // Sanity check for as long as we mix legacy code and new code
    ABORT_IF(this->graph() != graph, "Graph used for construction and graph parameter do not match");
#endif

    lazyCreateOutputLayer();
    return step(state);
  }

  Ptr<DecoderState> step(Ptr<DecoderState> state) {
    auto embeddings  = state->getTargetHistoryEmbeddings(); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]
    auto decoderMask = state->getTargetMask();              // [max length, batch size, 1]  --this is a hypothesis

    //************************************************************************//

    auto encoderContext = state->getEncoderStates()[0]->getContext(); // encoder output
    auto encoderMask    = state->getEncoderStates()[0]->getMask(); // note: may differ from Encoder self-attention mask in that additional positions are banned for cross-attention
    
    // Convert old style decoder state to new decoder state
    size_t position = state->getPosition();
    auto nnState = New<nn::DecoderStateList>(position);
    for(auto& layerState : state->getStates())
      nnState->as<nn::DecoderStateList>()->append(New<nn::DecoderStateItem>(layerState.cell, position));

    auto decoderContext = decoder->apply(embeddings, decoderMask, encoderContext, encoderMask, nnState);

    // final feed-forward layer (output)
    if(shortlist_)
      output_->setShortlist(shortlist_);
    auto logits = output_->applyAsLogits(decoderContext); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab or shortlist dim]
    
    // Convert new style decoder state to old decoder state
    // @TODO: This is such a mess!
    rnn::States decoderStates;
    for(auto layerState : *nnState->as<nn::DecoderStateList>()) {
      auto cellState = layerState->as<nn::DecoderStateItem>()->get();
      decoderStates.push_back(rnn::State({ cellState, cellState }));
    }
    // return unnormalized(!) probabilities
    auto nextState = New<DecoderState>(decoderStates, logits, state->getEncoderStates(), state->getBatch(), state->isBatchMajor());
    nextState->setPosition(state->getPosition() + 1);

    return nextState;
  }

  // helper function for guided alignment
  // @TODO: const vector<> seems wrong. Either make it non-const or a const& (more efficient but dangerous)
  virtual const std::vector<Expr> getAlignments(int /*i*/ = 0) override {
    ABORT("Not implemented");
    return {};
  }

  virtual void clear() override {
    Layer::clear();
    if (output_)
      output_->clear();
  }
};

} // namespace marian

#if 0 // ignore me. To-be-removed once fully functional.

static void testme() {
  using namespace marian;
  using namespace nn;
  
  auto options = New<Options>(
    "enc-depth", 12, 
    "transformer-heads", 8, 
    "dim-emb", 512, 
    "transformer-ffn-depth", 2,
    "transformer-dim-ffn",   2048, 
    "transformer-dropout",   0.1,
    "transformer-dropout-attention", 0.0,
    "transformer-postprocess", "dan",
    "transformer-ffn-activation", "relu",
    "transformer-train-position-embeddings", false,
    "transformer-depth-scaling", true,
    "max-length", 256);

  Config::seed = 1234;

  auto graph = New<ExpressionGraph>(/*inference=*/true);
  graph->setDevice(CPU0);
  graph->reserveWorkspaceMB(1000);

  auto input = graph->constant({10, 1, 512}, inits::glorotUniform()); // [length, batch, dim]
  auto mask  = graph->constant({10, 1,   1}, inits::ones());          // [length, batch,   1]

  auto encoder = New<TransformerEncoder>(graph, options);
  encoder->setName("TransformerEncoder");
  encoder->setEvalMode();
  
  auto context = encoder->apply(input, mask);

  std::cerr << encoder->layerInfo(/*includeChildren=*/true) << std::endl;

  debug(context);
  
  graph->forward();
  graph->save("test.npz");
}

#endif
