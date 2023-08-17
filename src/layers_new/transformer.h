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
struct TransformerPrePostProcessor final : public Layer, public IBinaryLayer {
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
 * This is a typical transformer self-attention block. The default configuration will
 * use a multi-head multiplicative self-attention layer, followed by dropout, the skip
 * connection and layer normalization (dan) in the post-processor. The pre-processor does
 * nothing in the default configuration.
 */
class TransformerSelfAttentionBlock final : public LayerWithOptions, public IBinaryLayer {
public:
  Ptr<TransformerPrePostProcessor> preprocessor;
  Ptr<AttentionLayer> selfAttention;
  Ptr<TransformerPrePostProcessor> postprocessor;

  TransformerSelfAttentionBlock(Ptr<ExpressionGraph> graph, 
                                Ptr<Options> options)
    : LayerWithOptions(graph, options)
  {
    preprocessor = New<TransformerPrePostProcessor>(
      graph, 
      opt<std::string>("transformer-preprocess", ""),  
      opt<float>("transformer-dropout", 0.f));
    registerLayer(preprocessor);

    // @TODO: factory to support different attention flavors?
    selfAttention = attentionFromOptions(graph, options);
    registerLayer(selfAttention);

    postprocessor = New<TransformerPrePostProcessor>(
      graph, 
      opt<std::string>("transformer-postprocess", ""), 
      opt<float>("transformer-dropout", 0.f));
    registerLayer(postprocessor);
  }

  Expr apply(Expr input, Expr mask = nullptr) const override {
    auto output = preprocessor->apply(input);                          // optional preprocessing
    output      = selfAttention->apply(output, output, output, mask);  // self attention, @TODO: make this a IBinaryLayer rather than IQuaternaryLayer
    output      = postprocessor->apply(output, input);                 // optional postprocessing, optional skip connection
    return output;
  }
};

/** 
 * This is a typical transformer filter (1-dimensional convolution) block. The default configuration will
 * use scale up to a larger dimension, apply a ReLU activation and scale down again, followed by dropout, 
 * the skip connection and layer normalization (dan) in the post-processor. The pre-processor does
 * nothing in the default configuration.
 */
struct TransformerFilterBlock final : public LayerWithOptions, public IUnaryLayer {
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
struct TransformerEncoderLayer final : public LayerWithOptions, public IBinaryLayer {
  Ptr<TransformerSelfAttentionBlock> selfAttentionBlock;
  Ptr<TransformerFilterBlock> filterBlock;

  TransformerEncoderLayer(Ptr<ExpressionGraph> graph, 
                          Ptr<Options> options)
    : LayerWithOptions(graph, options)
  {
    selfAttentionBlock = New<TransformerSelfAttentionBlock>(graph, options);
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
struct TransformerEncoder : public LayerWithOptions, public IBinaryLayer {
  Ptr<PositionEmbeddingLayer> positionEmbedding;
  Ptr<TransformerPrePostProcessor> preprocessor;
  Ptr<LayerList> layers;
  Ptr<TransformerPrePostProcessor> postprocessor;

  TransformerEncoder(Ptr<ExpressionGraph> graph, 
                     Ptr<Options> options)
    : LayerWithOptions(graph, options)
  {
    positionEmbedding = positionEmbeddingFromOptions(graph, options, /*positionAxis=*/-2);
    registerLayer(positionEmbedding);

    preprocessor = New<TransformerPrePostProcessor>(
      graph, 
      opt<std::string>("transformer-postprocess-emb", ""),  
      opt<float>("transformer-dropout", 0.f));
    registerLayer(preprocessor);

    layers = New<LayerList>(graph);
    registerLayer(layers);
    for(int i = 0; i < opt<int>("enc-depth"); ++i) {
      auto transformerEncoderLayer = New<TransformerEncoderLayer>(graph, options);
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
      layers->append(transformerEncoderLayer);
    }

    postprocessor = New<TransformerPrePostProcessor>(
      graph, 
      opt<std::string>("transformer-postprocess-top", ""),  
      opt<float>("transformer-dropout", 0.f));
    registerLayer(postprocessor);
  }

  virtual ~TransformerEncoder() = default;

  Expr apply(Expr input, Expr mask = nullptr) const override {
    // first and last operations (see at the bottom of this function) switch the time and batch
    // dimensions. This order is more natural for the transformer, but more difficult to handle
    // during beam search or when using RNNs. Hence the input/output transpositions here.

    // @TODO: still worth to review this whole transpose business across the tool. In the 
    // decoder state, Frank added information about batchMajor/timeMajor orientation. If we 
    // do that everywhere we can detect inconsistencies automatically. 
    // reorganize batch and timestep
    auto output = swapTimeBatch(input); // [beam depth=1, batch size, max length, vector dim]
    if(mask) {
      mask = swapTimeBatch(mask);   // [beam depth=1, batch size, max length, vector dim=1]
      mask = transposedLogMask(mask, opt<int>("transformer-heads"));
    }

    // apply positional embeddings to contextual input
    output = positionEmbedding->apply(output);

    // handle for skip connection at top
    auto prevOutput = output;

    // apply dropout or layer-norm to embeddings if required
    output = preprocessor->apply(output);

    // traverse the layers, use the same mask for each
    for(auto layer : *layers)
      output = layer->apply(output, mask);

    // apply final postprocessor if required, e.g. final layer-norm for pre-norm or final skip connection
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
};

/** 
 * This is a typical transformer cross-attention block. The default configuration will
 * use a multi-head multiplicative cross-attention layer, followed by dropout, the skip
 * connection and layer normalization (dan) in the post-processor. The pre-processor does
 * nothing in the default configuration.
 */
class TransformerCrossAttentionBlock final : public LayerWithOptions, public ITernaryLayer {
public:
  Ptr<TransformerPrePostProcessor> preprocessor;
  Ptr<AttentionLayer> crossAttention;
  Ptr<TransformerPrePostProcessor> postprocessor;

  TransformerCrossAttentionBlock(Ptr<ExpressionGraph> graph, 
                                 Ptr<Options> options)
    : LayerWithOptions(graph, options)
  {
    preprocessor = New<TransformerPrePostProcessor>(
      graph, 
      opt<std::string>("transformer-preprocess", ""),  
      opt<float>("transformer-dropout", 0.f));
    registerLayer(preprocessor);

    // @TODO: factory to support different attention flavors?
    crossAttention = attentionFromOptions(graph, options);
    registerLayer(crossAttention);

    postprocessor = New<TransformerPrePostProcessor>(
      graph, 
      opt<std::string>("transformer-postprocess", ""), 
      opt<float>("transformer-dropout", 0.f));
    registerLayer(postprocessor);
  }

  Expr apply(Expr input, Expr context, Expr contextMask = nullptr) const override {
    auto output = preprocessor->apply(input);                                   // optional preprocessing
    output      = crossAttention->apply(output, context, context, contextMask); // cross attention, @TODO: make this a ITernaryLayer rather than IQuaternaryLayer
    output      = postprocessor->apply(output, input);                          // optional postprocessing, optional skip connection
    return output;
  }
};

#if 1

class TransformerAutoRegressiveBlock : public LayerWithOptions, public IBinaryDecoderLayer {
public:
  TransformerAutoRegressiveBlock(Ptr<ExpressionGraph> graph, 
                                 Ptr<Options> options)
    : LayerWithOptions(graph, options) {}
  
  virtual ~TransformerAutoRegressiveBlock() = default;

  using IBinaryDecoderLayer::apply;
};

/** 
 * This is a transformer RNN block. 
 */
class TransformerRNNBlock final : public TransformerAutoRegressiveBlock {
public:
  Ptr<TransformerPrePostProcessor> preprocessor;
  Ptr<RNN<SSRU>> rnn;
  Ptr<TransformerPrePostProcessor> postprocessor;

  TransformerRNNBlock(Ptr<ExpressionGraph> graph, 
                      Ptr<Options> options)
    : TransformerAutoRegressiveBlock(graph, options)
  {
    preprocessor = New<TransformerPrePostProcessor>(
      graph, 
      opt<std::string>("transformer-preprocess", ""),  
      opt<float>("transformer-dropout", 0.f));
    registerLayer(preprocessor);

    // @TODO: factory to support different attention flavors?
    int modelDim = opt<int>("transformer-dim-model", opt<int>("dim-emb"));
    rnn = New<RNN<SSRU>>(graph, modelDim, opt<bool>("transformer-rnn-projection", false));
    registerLayer(rnn);

    postprocessor = New<TransformerPrePostProcessor>(
      graph, 
      opt<std::string>("transformer-postprocess", ""), 
      opt<float>("transformer-dropout", 0.f));
    registerLayer(postprocessor);
  }

  Expr apply(Expr input, Expr inputMask, Ptr<DecoderState> state) const override {
    auto output = preprocessor->apply(input);           // optional preprocessing
    output      = rnn->apply(output, inputMask, state); // rnn application with state extension
    output      = postprocessor->apply(output, input);  // optional postprocessing, optional skip connection
    return output;
  }
};

/** 
 * A full transformer decoder layer consists of a self-attention block followed by
 * cross-attention block and a filter block. Skip connections etc. are handled inside 
 * the blocks, see above.
 * 
 * For the self-attention block we need a special mask, usually a triangle mask that
 * prohibits to look into the future. 
 * @TODO: should the triangle mask be constructed locally here? Would make sense, but expensive 
 * for many layers. 
 */
struct TransformerDecoderLayer final : public LayerWithOptions, public IQuaternaryDecoderLayer {
  Ptr<TransformerAutoRegressiveBlock> autoRegressiveBlock;
  Ptr<TransformerCrossAttentionBlock> crossAttentionBlock;
  Ptr<TransformerFilterBlock> filterBlock;

  TransformerDecoderLayer(Ptr<ExpressionGraph> graph, 
                          Ptr<Options> options)
    : LayerWithOptions(graph, options)
  {
    auto autoRegressionType = opt<std::string>("transformer-decoder-autoreg", "self-attention");
    if(autoRegressionType == "self-attention") {
      ABORT("Auto-regression block type {} not yet implemented", autoRegressionType);
    } else if(autoRegressionType == "rnn") {
      autoRegressiveBlock = New<TransformerRNNBlock>(graph, options);
    } else {
      ABORT("Unknown auto-regression block type {}", autoRegressionType);
    }
    registerLayer(autoRegressiveBlock);
  
    crossAttentionBlock = New<TransformerCrossAttentionBlock>(graph, options);
    registerLayer(crossAttentionBlock);
    
    filterBlock = New<TransformerFilterBlock>(graph, options, /*isDecoder=*/true);
    registerLayer(filterBlock);
  }

  Expr apply(Expr input, Expr inputMask, Expr context, Expr contextMask, Ptr<DecoderState> state) const override {
    Expr output = autoRegressiveBlock->apply(input, inputMask, state);
    output      = crossAttentionBlock->apply(output, context, contextMask);
    output      = filterBlock->apply(output);

    checkpoint(output); // A full transformer block is a good point for gradient checkpointing (currently manual)    
    return output;
  }
};

/**
 * A full transformer decoder stack. Before applying multiple transformer layers (depth of the decoder), we 
 * add positional embeddings and apply post-processing actions to the combined embeddings. Due to backward-compatiblity
 * with RNN models and for easier beam-search we transpose batch and time dimensions on input and output. 
 * @TODO: get rid of these transposes.
 */
struct TransformerDecoder final : public LayerWithOptions, public IQuaternaryDecoderLayer {
  Ptr<PositionEmbeddingLayer> positionEmbedding;
  Ptr<TransformerPrePostProcessor> preprocessor;
  Ptr<LayerList> layers;
  Ptr<TransformerPrePostProcessor> postprocessor;
  
  TransformerDecoder(Ptr<ExpressionGraph> graph, 
                     Ptr<Options> options)
    : LayerWithOptions(graph, options)
  {
    positionEmbedding = positionEmbeddingFromOptions(graph, options, /*positionAxis=*/-2);
    registerLayer(positionEmbedding);

    preprocessor = New<TransformerPrePostProcessor>(
      graph, 
      opt<std::string>("transformer-postprocess-emb", ""),  
      opt<float>("transformer-dropout", 0.f));
    registerLayer(preprocessor);

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
    for(size_t i = 0; i < decDepth; ++i) {
      if(tiedLayers.empty() || tiedLayers[i] == i) { // not tied or tied to itself, so needs to be created first
        auto transformerDecoderLayer = New<TransformerDecoderLayer>(graph, options);
        layers->append(transformerDecoderLayer);
      } else {
        ABORT_IF(tiedLayers[i] > i, "Cannot tie to layer above this layer??");
        layers->append(layers->at(tiedLayers[i])); // repeat layer to tie weights
      }

      auto currentLayer = layers->at(i)->as<TransformerDecoderLayer>();
      // example of changing linear layer init functions burried deep in the model 
      if(opt<bool>("transformer-depth-scaling", false)) {
        auto autoRegLayer = currentLayer->autoRegressiveBlock->as<TransformerRNNBlock>();
        autoRegLayer->rnn->oProj->init = inits::glorotUniform(true, true, /*scale=*/ 1.f / std::sqrt((float)i + 1));

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
    }

    postprocessor = New<TransformerPrePostProcessor>(
      graph, 
      opt<std::string>("transformer-postprocess-top", ""),  
      opt<float>("transformer-dropout", 0.f));
    registerLayer(postprocessor);
  }

  Expr apply(Expr input, Expr inputMask, Expr context, Expr contextMask, Ptr<DecoderState> state) const override {
    // first and last operations (see at the bottom of this function) switch the time and batch
    // dimensions. This order is more natural for the transformer, but more difficult to handle
    // during beam search or when using RNNs. Hence the input/output transpositions here.
    Expr output = swapTimeBatch(input); // [beam depth=1, batch size, max length, vector dim]
    context = swapTimeBatch(context); 

    // @TODO: write function prepareMasks();
    // @TODO: create triangle mask here and combine with inputMask
    LOG_ONCE(info, "Don't forget the triangle mask if required!");
    if(inputMask) {
      inputMask = swapTimeBatch(inputMask);   // [beam depth=1, batch size, max length, vector dim=1]
    }

    if(contextMask) {
      contextMask = swapTimeBatch(contextMask);    // [beam depth=1, max length, batch size, vector dim=1]
      contextMask = transposedLogMask(contextMask, opt<int>("transformer-heads")); // [beam broadcast=1, batch size * num heads, max length broadcast=1, max length]
    }
    
    // apply positional embeddings to contextual input @TODO: remove need for conversion to int
    output = positionEmbedding->apply(output, (int)state->getPosition());
    
    // handle for skip connection at top
    auto prevOutput = output;

    // apply dropout or layer-norm to embeddings if required
    output = preprocessor->apply(output);

    // get an iterator to per-layer states
    auto layerStateIt = state->as<nn::DecoderStateList>()->begin();
    // traverse the layers, use the same mask for each
    for(auto layer : *layers)
      output = layer->as<TransformerDecoderLayer>()->apply(output, inputMask, context, contextMask, /*in/out=*/*layerStateIt++);

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
};
#endif

} // namespace nn
} // namespace marian
