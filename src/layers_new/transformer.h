#pragma once

#include "layers_new/attention.h"
#include "layers_new/decoder.h"
#include "layers_new/embeddings.h"
#include "layers_new/neuralnet.h"
#include "layers_new/rnn.h"

#include <cmath>

namespace marian {
namespace nn {

/**
 * This groups the typical transformer pre/post-processing steps in to a class.
 * Currently these are usually dropout, layer normalization and skip connections.
 * A transformer block will usually apply one of them.
 */
class TransformerPrePostProcessor final : public Layer, public IBinaryLayer {
public:
  Ptr<Dropout> dropout;
  Ptr<Norm> norm;
  std::string actionDesc;

  TransformerPrePostProcessor(Ptr<ExpressionGraph> graph,
                              const std::string& actionDesc,
                              float dropoutProbablity)
    : Layer(graph),
      actionDesc(actionDesc)
  {
    for(char a : actionDesc) {
      if(a == 'd') {
        ABORT_IF(dropout, "Dropout layer already initialized? Did you specify 'd' more than once?");
        dropout = New<Dropout>(graph, dropoutProbablity);
        registerLayer(dropout);
      } else if(a == 'n') {
        ABORT_IF(norm, "Norm layer already initialized? Did you specify 'n' or 'r' more than once?");
        norm = New<LayerNorm>(graph);
        registerLayer(norm);
      } else if(a == 'r') {
        ABORT_IF(norm, "Norm layer already initialized? Did you specify 'n' or 'r' more than once?");
        norm = New<RMSNorm>(graph);
        registerLayer(norm);
      }
    }
  }

  virtual ~TransformerPrePostProcessor() = default;

  Expr apply(Expr input, Expr previous = nullptr) const override {
    Expr output = input;
    for(char action : actionDesc) {
      if(action == 'd')
        output = dropout->apply(output);
      else if(action == 'a' && previous)
        output = output + previous;
      else if(action == 'a' && !previous)
        ABORT("Action 'a' (add skip connection) specified but no previous input given");
      else if(action == 'n' || action == 'r')
        output = norm->apply(output);
      else
        ABORT("Action '{}' in '{}' unknown", action, actionDesc);
    }
    return output;
  }
};

/**
 * This is a transformer self-attention block without state. The default configuration will
 * use a multi-head multiplicative self-attention layer, followed by dropout, the skip
 * connection and layer normalization (dan) in the post-processor. The pre-processor does
 * nothing in the default configuration. See TransformerDecoderSelfAttentionBlock for a
 * version that can be used in the decoder with state.
 */
class TransformerSelfAttentionBlock final : public LayerWithOptions, public IBinaryLayer {
public:
  Ptr<TransformerPrePostProcessor> preprocessor;
  Ptr<MaskProcessor> selfMaskProcessor;
  Ptr<AttentionLayer> selfAttention;
  Ptr<TransformerPrePostProcessor> postprocessor;

  TransformerSelfAttentionBlock(Ptr<ExpressionGraph> graph,
                                Ptr<Options> options,
                                Ptr<MaskProcessor> selfMaskProcessorInit = nullptr)
    : LayerWithOptions(graph, options),
      selfMaskProcessor(selfMaskProcessorInit)
  {
    preprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-preprocess", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(preprocessor);

    if(!selfMaskProcessor) {
      selfMaskProcessor = maskProcessorFromOptions(graph, options);
      registerLayer(selfMaskProcessor);
    }

    selfAttention = attentionFromOptions(graph, options);
    registerLayer(selfAttention);

    postprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-postprocess", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(postprocessor);
  }

  Expr apply(Expr input, Expr inputMask = nullptr) const override {
    auto output  = preprocessor->apply(input);                            // optional preprocessing
    auto logMask = selfMaskProcessor->apply(output, inputMask);           // mask out attention to padding symbols
    output       = selfAttention->apply(output, output, output, logMask); // self attention, @TODO: make this a IBinaryLayer rather than IQuaternaryLayer
    output       = postprocessor->apply(output, input);                   // optional postprocessing, optional skip connection
    return output;
  }
};

/**
 * This is a typical transformer filter (1-dimensional convolution) block. The default configuration will
 * use scale up to a larger dimension, apply a ReLU activation and scale down again, followed by dropout,
 * the skip connection and layer normalization (dan) in the post-processor. The pre-processor does
 * nothing in the default configuration.
 */
class TransformerFilterBlock final : public LayerWithOptions, public IUnaryLayer {
public:
  Ptr<TransformerPrePostProcessor> preprocessor;
  Ptr<Sequential> layers;
  Ptr<TransformerPrePostProcessor> postprocessor;
  bool isDecoder{false};

  TransformerFilterBlock(Ptr<ExpressionGraph> graph,
                         Ptr<Options> options,
                         bool isDecoder = false)
    : LayerWithOptions(graph, options), isDecoder(isDecoder)
  {
    preprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-preprocess", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(preprocessor);

    int modelDim = opt<int>("transformer-dim-model", opt<int>("dim-emb"));
    int ffnDim   = opt<int>("transformer-dim-ffn");
    if(isDecoder && opt<int>("transformer-decoder-dim-ffn") != 0)
      ffnDim = opt<int>("transformer-decoder-dim-ffn");

    int depth    = opt<int>("transformer-ffn-depth", 2);
    if(isDecoder && opt<int>("transformer-decoder-ffn-depth") != 0)
      depth = opt<int>("transformer-decoder-ffn-depth");

    auto actName = opt<std::string>("transformer-ffn-activation", "relu");
    float ffnDropoutProbability = opt<float>("transformer-dropout-ffn", 0.f);

    ABORT_IF(depth < 1, "Filter depth {} is smaller than 1", depth);

    // assemble filter of given depth
    layers = New<Sequential>(graph);
    registerLayer(layers);

    if(actName == "relu") {
      layers->append(New<LinearReluDropout>(graph, ffnDim, ffnDropoutProbability));
    } else {
      layers->append(New<Linear>(graph, ffnDim));
      layers->append(activationLayerByName(graph, actName));
      layers->append(New<Dropout>(graph, ffnDropoutProbability));
    }
    for(int i = 1; i < depth-1; ++i) {
      if(actName == "relu") {
        layers->append(New<LinearReluDropout>(graph, ffnDim, ffnDropoutProbability));
      } else {
        layers->append(New<Linear>(graph, ffnDim));
        layers->append(activationLayerByName(graph, actName));
        layers->append(New<Dropout>(graph, ffnDropoutProbability));
      }
    }
    layers->append(New<Linear>(graph, modelDim));

    postprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-postprocess", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(postprocessor);
  }

  Expr apply(Expr input) const override {
    Expr output = preprocessor->apply(input);          // optional preprocessing
    output      = layers->apply(output);               // main FFN
    output      = postprocessor->apply(output, input); // optional postprocessing, optional skip connection
    return output;
  }
};

/**
 * A full transformer encoder layer consists of a self-attention block followed by
 * a filter block. Skip connections etc. are handled inside the blocks, see above.
 */
class TransformerEncoderLayer final : public LayerWithOptions, public IBinaryLayer {
public:
  Ptr<TransformerSelfAttentionBlock> selfAttentionBlock;
  Ptr<TransformerFilterBlock> filterBlock;

  TransformerEncoderLayer(Ptr<ExpressionGraph> graph,
                          Ptr<Options> options,
                          Ptr<MaskProcessor> selfMaskProcessorInit = nullptr)
    : LayerWithOptions(graph, options)
  {
    selfAttentionBlock = New<TransformerSelfAttentionBlock>(graph, options, selfMaskProcessorInit);
    registerLayer(selfAttentionBlock);

    filterBlock = New<TransformerFilterBlock>(graph, options);
    registerLayer(filterBlock);
  }

  Expr apply(Expr input, Expr mask = nullptr) const override {
    Expr output = selfAttentionBlock->apply(input, mask);
    output      = filterBlock->apply(output);

    checkpoint(output); // A full transformer block is a good point for gradient checkpointing (currently manual)

    return output;
  }
};

/**
 * A full transformer encoder stack. Before applying multiple transformer layers (depth of the encoder), we
 * add positional embeddings and apply post-processing actions to the combined embeddings. Due to backward-compatiblity
 * with RNN models and for easier beam-search we transpose batch and time dimensions on input and output.
 * @TODO: get rid of these transposes.
 */
class TransformerEncoder : public LayerWithOptions, public IBinaryLayer {
public:
  Ptr<PositionEmbeddingLayer> positionEmbedding;
  Ptr<TransformerPrePostProcessor> preprocessor;
  Ptr<LayerList> layers;
  Ptr<TransformerPrePostProcessor> postprocessor;

protected: // @TODO: should this be public?
   // collect hidden states as we step through the layers
  mutable bool keepHiddenStates{false};
  mutable std::vector<Expr> hiddenStates;
  // apply this function to hidden states before collecting them
  mutable std::function<Expr(Expr)> hiddenTransformFn = [](Expr x) { return x; };

public:
  TransformerEncoder(Ptr<ExpressionGraph> graph,
                     Ptr<Options> options)
    : LayerWithOptions(graph, options) {
    if(!opt<bool>("transformer-disable-position-embeddings", false)) {
      positionEmbedding = positionEmbeddingFromOptions(graph, options, /*positionAxis=*/-2);
      registerLayer(positionEmbedding);
    }

    preprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-postprocess-emb", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(preprocessor);

    layers = New<LayerList>(graph);
    registerLayer(layers);

    Ptr<MaskProcessor> selfMaskProcessor; // this will be initialized in the first encoder layer
    for(int i = 0; i < opt<int>("enc-depth"); ++i) {
      auto transformerEncoderLayer = New<TransformerEncoderLayer>(graph, options, selfMaskProcessor);
      layers->append(transformerEncoderLayer);

      if(!selfMaskProcessor)
        selfMaskProcessor = transformerEncoderLayer->selfAttentionBlock->selfMaskProcessor;

      // example of changing linear layer init functions burried deep in the model
      if(opt<bool>("transformer-depth-scaling", false))
        for(auto linear : transformerEncoderLayer->allLayers<Linear>())
          linear->init = inits::glorotUniform(true, true, /*scale=*/ 1.f / std::sqrt((float)i + 1));

      if(opt<bool>("transformer-no-bias", false))
        for(auto linear : transformerEncoderLayer->allLayers<Linear>())
          linear->useBias = false;

      if(opt<bool>("transformer-no-affine", false)) {
        for(auto norm : transformerEncoderLayer->allLayers<Norm>()) {
          norm->useScale = false;
          norm->useBias  = false;
        }
      }
    }

    postprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-postprocess-top", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(postprocessor);
  }

  virtual ~TransformerEncoder() = default;

  Expr apply(Expr input, Expr inputMask = nullptr) const override {
    // first and last operations (see at the bottom of this function) switch the time and batch
    // dimensions. This order is more natural for the transformer, but more difficult to handle
    // during beam search or when using RNNs. Hence the input/output transpositions here.

    // @TODO: still worth to review this whole transpose business across the tool. In the
    // decoder state, Frank added information about batchMajor/timeMajor orientation. If we
    // do that everywhere we can detect inconsistencies automatically.
    // reorganize batch and timestep
    auto output = swapTimeBatch(input); // [1, dimBatch, dimSrcWords, dimModel]
    if(inputMask)
      inputMask = swapTimeBatch(inputMask); // [1, dimBatch, dimSrcWords, 1]

    // apply positional embeddings to contextual input
    if(positionEmbedding)
      output = positionEmbedding->apply(output);
    else
      output = std::sqrt((float)output->shape()[-1]) * output;

    // handle for skip connection at top
    auto prevOutput = output;

    // apply dropout or layer-norm to embeddings if required
    output = preprocessor->apply(output);

    // traverse the layers, use the same mask for each
    for(auto layer : *layers) {
      if(keepHiddenStates) // note, with pre-norm, the hidden states will not be normed here.
        hiddenStates.push_back(hiddenTransformFn(output));
      output = layer->apply(output, inputMask);
    }

    // apply final postprocessor if required, e.g. final layer-norm for pre-norm or final skip connection
    output = postprocessor->apply(output, prevOutput);
    if(keepHiddenStates)
      hiddenStates.push_back(hiddenTransformFn(output));

    // restore organization of batch and time steps. This is currently required
    // to make RNN-based decoders and beam search work with this. We are looking
    // into making this more natural.

    // @TODO: it might be worth to make this optional when the input goes into a
    // transformer decoder which now has to undo that again -- or even better
    // detect idempotent transposes during a process similar to auto-batching.
    // Or as other toolkits do it, make the transformer order the default and only transpose for RNNs.
    output = swapTimeBatch(output); // [beam depth=1, max length, batch size, vector dim]
    return output;
  }

  virtual void clear() override {
    LayerWithOptions::clear();
    hiddenStates.clear();
  }
};

/**
 * This is a typical transformer cross-attention block. The default configuration will
 * use a multi-head multiplicative cross-attention layer, followed by dropout, the skip
 * connection and layer normalization (dan) in the post-processor. The pre-processor does
 * nothing in the default configuration.
 */
class TransformerDecoderCrossAttentionBlock final : public LayerWithOptions, public ITernaryDecoderLayer {
public:
  Ptr<TransformerPrePostProcessor> preprocessor;
  Ptr<DecoderMaskProcessor> contextMaskProcessor;
  Ptr<AttentionLayer> crossAttention;
  Ptr<TransformerPrePostProcessor> postprocessor;

  TransformerDecoderCrossAttentionBlock(Ptr<ExpressionGraph> graph,
                                        Ptr<Options> options,
                                        Ptr<DecoderMaskProcessor> contextMaskProcessorInit = nullptr)
    : LayerWithOptions(graph, options),
      contextMaskProcessor(contextMaskProcessorInit)
  {
    preprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-preprocess", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(preprocessor);

    if(!contextMaskProcessor) {
      contextMaskProcessor = contextDecoderMaskProcessorFromOptions(graph, options);
      registerLayer(contextMaskProcessor);
    }

    // @TODO: factory to support different attention flavors?
    // for cross-attention, we cache the projected keys and values since they come from
    // the encoder and are static during decoding unless the batch size changes.
    crossAttention = attentionFromOptions(graph, options, /*enableCache=*/true);
    registerLayer(crossAttention);

    postprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-postprocess", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(postprocessor);
  }

  void initState(Ptr<DecoderState> state) const override {}

  Expr apply(Expr input, Expr context, Expr contextMask, Ptr<DecoderState> state) const override {
    auto output  = preprocessor->apply(input); // optional preprocessing
    auto logMask = contextMaskProcessor->apply(output, contextMask, state);
    output       = crossAttention->apply(output, context, context, logMask); // cross attention, @TODO: make this a ITernaryLayer rather than IQuaternaryLayer
    output       = postprocessor->apply(output, input);                      // optional postprocessing, optional skip connection
    return output;
  }
};

/**
 * Base class for transformer auto-regressive blocks. These are blocks that can be used in the decoder
 * and that take the previous step's output as input. Currently this is either a self-attention block
 * or an RNN block.
 */
class TransformerDecoderAutoRegressiveBlock : public LayerWithOptions, public IBinaryDecoderLayer {
public:
  Ptr<TransformerPrePostProcessor> preprocessor;
  Ptr<DecoderMaskProcessor> selfMaskProcessor;
  Ptr<TransformerPrePostProcessor> postprocessor;

  TransformerDecoderAutoRegressiveBlock(Ptr<ExpressionGraph> graph,
                                        Ptr<Options> options,
                                        Ptr<DecoderMaskProcessor> selfMaskProcessorInit = nullptr)
    : LayerWithOptions(graph, options),
      selfMaskProcessor(selfMaskProcessorInit)
  {
    preprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-preprocess", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(preprocessor);

    if(!selfMaskProcessor) {
      selfMaskProcessor = selfMaskProcessorFromOptions(graph, options);
      registerLayer(selfMaskProcessor);
    }

    postprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-postprocess", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(postprocessor);
  }

  virtual ~TransformerDecoderAutoRegressiveBlock() = default;

  using IBinaryDecoderLayer::initState;
  using IBinaryDecoderLayer::apply;
};

/**
 * This is a typical transformer self-attention block. The default configuration will
 * use a multi-head multiplicative self-attention layer, followed by dropout, the skip
 * connection and layer normalization (dan) in the post-processor. The pre-processor does
 * nothing in the default configuration.
 */
class TransformerDecoderSelfAttentionBlock final : public TransformerDecoderAutoRegressiveBlock {
public:
  Ptr<AttentionLayer> selfAttention;

  using TransformerDecoderAutoRegressiveBlock::preprocessor;
  using TransformerDecoderAutoRegressiveBlock::selfMaskProcessor;
  using TransformerDecoderAutoRegressiveBlock::postprocessor;

  TransformerDecoderSelfAttentionBlock(Ptr<ExpressionGraph> graph,
                                       Ptr<Options> options,
                                       Ptr<DecoderMaskProcessor> selfMaskProcessorInit = nullptr)
    : TransformerDecoderAutoRegressiveBlock(graph, options, selfMaskProcessorInit)
  {
    // no caching of keys and values for self-attention since they change at each step
    selfAttention = attentionFromOptions(graph, options, /*enableCache=*/false);
    registerLayer(selfAttention);
  }

  void initState(Ptr<DecoderState> state) const override {
    state->setPosition(0);
  }

  Expr apply(Expr input, Expr inputMask, Ptr<DecoderState> state) const override {
    auto output = preprocessor->apply(input);           // optional preprocessing

    // Here we extend the state with the keys and values from the previous step.
    auto query      = output;
    auto keysValues = output;
    if(state->getPosition() > 0) {
      auto kvHistory = state->as<DecoderStateItem>()->get(); // [dimBeam, dimBatch, dimHistory, dimModel]
      keysValues     = concatenate({kvHistory, keysValues}, /*axis=*/-2); // [dimBeam, dimBatch, dimHistory + 1, dimModel]
    }
    state->as<DecoderStateItem>()->set(keysValues);

    auto logMask = selfMaskProcessor->apply(query, inputMask, state);
    output       = selfAttention->apply(query, keysValues, keysValues, logMask);
    output       = postprocessor->apply(output, input);  // optional postprocessing, optional skip connection
    return output;
  }
};

/**
 * This is a transformer RNN block that can be used as a replacement for the self-attention
 * block in the decoder.
 */
class TransformerDecoderRNNBlock final : public TransformerDecoderAutoRegressiveBlock {
public:
  Ptr<RNN<SSRU>> rnn; // @TODO: support other RNN types like LSTM or GRU

  using TransformerDecoderAutoRegressiveBlock::preprocessor;
  using TransformerDecoderAutoRegressiveBlock::postprocessor;

  TransformerDecoderRNNBlock(Ptr<ExpressionGraph> graph,
                             Ptr<Options> options,
                             Ptr<DecoderMaskProcessor> selfMaskProcessorInit = nullptr)
    : TransformerDecoderAutoRegressiveBlock(graph, options, selfMaskProcessorInit)
  {
    // @TODO: factory to support different attention flavors?
    int modelDim = opt<int>("transformer-dim-model", opt<int>("dim-emb"));
    rnn = New<RNN<SSRU>>(graph, modelDim, opt<bool>("transformer-rnn-projection", false));
    registerLayer(rnn);
  }

  void initState(Ptr<DecoderState> state) const override {
    rnn->as<IBinaryDecoderLayer>()->initState(state);
  }

  Expr apply(Expr input, Expr inputMask, Ptr<DecoderState> state) const override {
    auto output = preprocessor->apply(input);           // optional preprocessing
    output      = rnn->apply(output, inputMask, state); // rnn application with state extension
    output      = postprocessor->apply(output, input);  // optional postprocessing, optional skip connection
    return output;
  }
};

/**
 * A full transformer (LM) decoder layer consists of a self-attention block followed by
 * a filter block. Skip connections etc. are handled inside the blocks, see above.
 */
class TransformerDecoderLayer : public LayerWithOptions, public IBinaryDecoderLayer {
public:
  Ptr<TransformerDecoderAutoRegressiveBlock> autoRegressiveBlock;
  Ptr<TransformerFilterBlock> filterBlock;

  TransformerDecoderLayer(Ptr<ExpressionGraph> graph,
                          Ptr<Options> options,
                          Ptr<DecoderMaskProcessor> selfMaskProcessorInit = nullptr)
    : LayerWithOptions(graph, options)
  {
    auto autoRegressionType = opt<std::string>("transformer-decoder-autoreg", "self-attention");
    if(autoRegressionType == "self-attention") {
      autoRegressiveBlock = New<TransformerDecoderSelfAttentionBlock>(graph, options, selfMaskProcessorInit);
    } else if(autoRegressionType == "rnn") {
      autoRegressiveBlock = New<TransformerDecoderRNNBlock>(graph, options, selfMaskProcessorInit);
    } else {
      ABORT("Unknown auto-regression block type {}", autoRegressionType);
    }
    registerLayer(autoRegressiveBlock);

    filterBlock = New<TransformerFilterBlock>(graph, options, /*isDecoder=*/true);
    registerLayer(filterBlock);
  }

  void initState(Ptr<DecoderState> state) const override {
    autoRegressiveBlock->as<IBinaryDecoderLayer>()->initState(state);
  }

  Expr apply(Expr input, Expr inputMask, Ptr<DecoderState> state) const override {
    Expr output = autoRegressiveBlock->apply(input, inputMask, state);
    output      = filterBlock->apply(output);

    checkpoint(output); // A full transformer block is a good point for gradient checkpointing (currently manual)
    return output;
  }
};

/**
 * A transformer (S2S) decoder layer consists of a self-attention block followed by
 * cross-attention block and a filter block. Skip connections etc. are handled inside
 * the blocks. We inherit from TransformerDecoderLayer and add the cross-attention block.
 * * @TODO: get rid of IQuaternaryDecoderLayer and use IBinaryDecoderLayer instead
 */
class TransformerDecoderLayerWithCrossAttention : public TransformerDecoderLayer, public IQuaternaryDecoderLayer {
public:
  Ptr<TransformerDecoderCrossAttentionBlock> crossAttentionBlock;
  using TransformerDecoderLayer::autoRegressiveBlock;
  using TransformerDecoderLayer::filterBlock;

  TransformerDecoderLayerWithCrossAttention(Ptr<ExpressionGraph> graph,
                                            Ptr<Options> options,
                                            Ptr<DecoderMaskProcessor> selfMaskProcessorInit = nullptr,
                                            Ptr<DecoderMaskProcessor> contextMaskProcessorInit = nullptr)
    : TransformerDecoderLayer(graph, options, selfMaskProcessorInit)
  {
    crossAttentionBlock = New<TransformerDecoderCrossAttentionBlock>(graph, options, contextMaskProcessorInit);
    registerLayer(crossAttentionBlock);
  }

  void initState(Ptr<DecoderState> state) const override {
    TransformerDecoderLayer::initState(state);
  }

  Expr apply(Expr input, Expr inputMask, Expr context, Expr contextMask, Ptr<DecoderState> state) const override {
    Expr output  = autoRegressiveBlock->apply(input, inputMask, state);
    output       = crossAttentionBlock->apply(output, context, contextMask, state);
    output       = filterBlock->apply(output);

    checkpoint(output); // A full transformer block is a good point for gradient checkpointing (currently manual)
    return output;
  }

private:
  // @TODO: once we have correct decoder states we can change the interface to IBinaryDecoderLayer and remove this
  // this is a dummy implementation to satisfy the interface, it should never be called
  Expr apply(Expr input, Expr inputMask, Ptr<DecoderState> state) const override {
    ABORT("This should never be called");
  }
};

/**
 * A full transformer decoder stack. Before applying multiple transformer layers (depth of the decoder), we
 * add positional embeddings and apply post-processing actions to the combined embeddings. Due to backward-compatiblity
 * with RNN models and for easier beam-search we transpose batch and time dimensions on input and output.
 * @TODO: get rid of these transposes.
 */
class TransformerDecoder final : public LayerWithOptions, public IBinaryDecoderLayer {
private:
  Ptr<AttentionCollector> attentionCollector_;

public:
  Ptr<PositionEmbeddingLayer> positionEmbedding;
  Ptr<TransformerPrePostProcessor> preprocessor;
  Ptr<TransformerPrePostProcessor> postprocessor;
  Ptr<LayerList> layers;

  TransformerDecoder(Ptr<ExpressionGraph> graph,
                     Ptr<Options> options)
    : LayerWithOptions(graph, options)
  {
    if(!opt<bool>("transformer-disable-position-embeddings", false)) {
      positionEmbedding = positionEmbeddingFromOptions(graph, options, /*positionAxis=*/-2);
      registerLayer(positionEmbedding);
    }

    preprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-postprocess-emb", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(preprocessor);

    postprocessor = New<TransformerPrePostProcessor>(
      graph,
      opt<std::string>("transformer-postprocess-top", ""),
      opt<float>("transformer-dropout", 0.f));
    registerLayer(postprocessor);

    size_t decDepth = opt<size_t>("dec-depth");
    std::vector<size_t> tiedLayers = opt<std::vector<size_t>>("transformer-tied-layers", std::vector<size_t>());
    ABORT_IF(!tiedLayers.empty() && tiedLayers.size() != decDepth,
             "Specified layer tying for {} layers, but decoder has {} layers",
             tiedLayers.size(),
             decDepth);
    // shift to base-0 indexing
    for(auto& layerNo : tiedLayers)
      layerNo = layerNo - 1;

    layers = New<LayerList>(graph);
    registerLayer(layers);

    Ptr<DecoderMaskProcessor> selfMaskProcessor;    // this will be initialized in the first decoder layer
    Ptr<DecoderMaskProcessor> contextMaskProcessor; // this will be initialized in the first decoder layer
    for(size_t i = 0; i < decDepth; ++i) {
      if(tiedLayers.empty() || tiedLayers[i] == i) { // not tied or tied to itself, so needs to be created first
        auto transformerDecoderLayer = New<TransformerDecoderLayerWithCrossAttention>(graph, options, selfMaskProcessor, contextMaskProcessor);
        layers->append(transformerDecoderLayer);

        if(!selfMaskProcessor)
          selfMaskProcessor    = transformerDecoderLayer->autoRegressiveBlock->selfMaskProcessor;
        if(!contextMaskProcessor)
          contextMaskProcessor = transformerDecoderLayer->crossAttentionBlock->contextMaskProcessor;

      } else {
        ABORT_IF(tiedLayers[i] > i, "Cannot tie to layer above this layer??");
        layers->append(layers->at(tiedLayers[i])); // repeat layer to tie weights
      }

      auto currentLayer = layers->at(i)->as<TransformerDecoderLayerWithCrossAttention>();

      // example of changing linear layer init functions burried deep in the model
      if(opt<bool>("transformer-depth-scaling", false)) {
        auto autoRegLayerRNN = currentLayer->autoRegressiveBlock->as<TransformerDecoderRNNBlock>();
        if(autoRegLayerRNN)
          autoRegLayerRNN->rnn->oProj->init = inits::glorotUniform(true, true, /*scale=*/ 1.f / std::sqrt((float)i + 1));

        auto autoRegLayerSA = currentLayer->autoRegressiveBlock->as<TransformerDecoderSelfAttentionBlock>();
        if(autoRegLayerSA)
          for(auto linear : autoRegLayerSA->allLayers<Linear>())
            linear->init = inits::glorotUniform(true, true, /*scale=*/ 1.f / std::sqrt((float)i + 1));

        for(auto linear : currentLayer->crossAttentionBlock->allLayers<Linear>())
          linear->init = inits::glorotUniform(true, true, /*scale=*/ 1.f / std::sqrt((float)i + 1));

        for(auto linear : currentLayer->filterBlock->allLayers<Linear>())
          linear->init = inits::glorotUniform(true, true, /*scale=*/ 1.f / std::sqrt((float)i + 1));
      }

      if(opt<bool>("transformer-no-bias", false))
        for(auto linear : currentLayer->allLayers<Linear>())
          linear->useBias = false;

      if(opt<bool>("transformer-no-affine", false)) {
        for(auto norm : currentLayer->allLayers<Norm>()) {
          norm->useScale = false;
          norm->useBias = false;
        }
      }

      if(opt<std::string>("guided-alignment", "none") != "none" || options_->hasAndNotEmpty("alignment")) {
        std::string gaStr = opt<std::string>("transformer-guided-alignment-layer", "last");

        size_t attLayer = decDepth - 1;
        if(gaStr != "last")
          attLayer = std::stoull(gaStr) - 1;

        ABORT_IF(attLayer >= decDepth, "Chosen layer for guided attention ({}) larger than number of layers ({})", attLayer + 1, decDepth);

        if(i == attLayer) {
          attentionCollector_ = currentLayer->crossAttentionBlock->crossAttention->as<nn::AttentionCollector>();
          attentionCollector_->saveAttentionWeights = true;              // @TODO: ugly
          attentionCollector_->numHeads = opt<int>("transformer-heads"); // @TODO: ugly
        }
      }
    }
  }

  void initState(Ptr<DecoderState> state) const override {
    ABORT("Remove this abort once this is actually used in the decoder");
    size_t positiion = 0;
    state->setPosition(positiion);
    for(auto layer : *layers) {
      Ptr<DecoderStateItem> layerState = New<DecoderStateItem>(positiion);
      layer->as<TransformerDecoderLayer>()->initState(layerState);
      state->as<DecoderStateList>()->append(layerState);
    }
  }

  Expr apply(Expr input, Expr inputMask, Ptr<DecoderState> state) const override {
    // first and last operations (see at the bottom of this function) switch the time and batch
    // dimensions. This order is more natural for the transformer, but more difficult to handle
    // during beam search or when using RNNs. Hence the input/output transpositions here.
    Expr output = swapTimeBatch(input); // [beam depth=1, batch size, max length, vector dim]

    // set current target token position during decoding or training. At training
    // this should be 0. During translation the current length of the translation.
    // Used for position embeddings and creating new decoder states.
    int startPos = (int)state->getPosition();

    if(inputMask)
      inputMask = swapTimeBatch(inputMask); // [dimBeam=1, dimBatch, dimTrgWords, dimModel=1]

    Expr context     = state->as<EncoderContext>()->getContext();
    Expr contextMask = state->as<EncoderContext>()->getContextMask();

    // @TODO: get rid of this
    context = swapTimeBatch(context); // [dimBeam=1, dimBatch, dimSrcWords, dimModel]
    if(contextMask)
      contextMask = swapTimeBatch(contextMask);  // [dimBeam=1, dimBatch, dimSrcWords, dimModel=1]

    // apply positional embeddings to contextual input
    if(positionEmbedding)
      output = positionEmbedding->apply(output, startPos);
    else
      output = std::sqrt((float)output->shape()[-1]) * output;

    // handle for skip connection at top
    auto prevOutput = output;

    // apply dropout or layer-norm to embeddings if required
    output = preprocessor->apply(output);

    // get an iterator to per-layer states
    auto layerStateIt = state->as<nn::DecoderStateList>()->begin();
    // traverse the layers, use the same mask for each
    for(auto layer : *layers) {
      // @TODO: can we put logmask computation inside this layer? Then we can reduce the number of arguments here
      // and use only the decoder state to provide context and mask.
      output = layer->as<TransformerDecoderLayerWithCrossAttention>()->apply(output, inputMask, context, contextMask, /*in/out=*/*layerStateIt++);
    }

    // apply final postprocessor if requred, e.g. final layer-norm for pre-norm or final skip connection
    output = postprocessor->apply(output, prevOutput);

    // restore organization of batch and time steps. This is currently required
    // to make RNN-based decoders and beam search work with this. We are looking
    // into making this more natural.
    // @TODO: it might be worth to make this optional when the input goes into a
    // transformer decoder which now has to undo that again -- or even better
    // detect idempotent transposes during a process similar to auto-batching.
    // Or as other toolkits do it, make the transformer order the default and only transpose for RNNs.
    output = swapTimeBatch(output); // [beam depth=1, max length, batch size, vector dim]
    return output;
  }

  std::vector<Expr> getAlignments() {
    if(attentionCollector_)
      return attentionCollector_->getAlignments();
    else
      return {};
  }

  virtual void clear() override {
    LayerWithOptions::clear();
  }

};

} // namespace nn
} // namespace marian
