#pragma once

#include "marian.h"

namespace marian {

class DecoderFrantic : public DecoderBase {
private:
  Ptr<rnn::RNN> rnn_;

  Ptr<rnn::RNN> constructDecoderRNN(Ptr<ExpressionGraph> graph,
                                    Ptr<DecoderState> state) {
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
    auto rnn = rnn::rnn(graph)                                     //
        ("type", opt<std::string>("dec-cell"))                     //
        ("dimInput", opt<int>("dim-emb"))                          //
        ("dimState", opt<int>("dim-rnn"))                          //
        ("dropout", dropoutRnn)                                    //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("nematus-normalization",
         options_->has("original-type")
             && opt<std::string>("original-type") == "nematus")  //
        ("skip", opt<bool>("skip"));

    size_t decoderBaseDepth = 2;
    // setting up conditional (transitional) cell
    auto baseCell = rnn::stacked_cell(graph);
    for(int i = 1; i <= decoderBaseDepth; ++i) {
      bool transition = (i > 2);
      auto paramPrefix = prefix_ + "_cell" + std::to_string(i);
      baseCell.push_back(rnn::cell(graph)         //
                         ("prefix", paramPrefix)  //
                         ("final", i > 1)         //
                         ("transition", transition));
      if(i == 1) {
        for(int k = 0; k < state->getEncoderStates().size(); ++k) {
          auto attPrefix = prefix_;
          if(state->getEncoderStates().size() > 1)
            attPrefix += "_att" + std::to_string(k + 1);

          auto encState = state->getEncoderStates()[k];

          baseCell.push_back(rnn::attention(graph)  //
                             ("prefix", attPrefix)  //
                                 .set_state(encState));
        }
      }
    }
    // Add cell to RNN (first layer)
    rnn.push_back(baseCell);
    return rnn.construct();
  }

public:
  DecoderFrantic(Ptr<Options> options) : DecoderBase(options) {}

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>>& encStates) {
    using namespace keywords;

    std::vector<Expr> meanContexts;
    for(auto& encState : encStates) {
      // average the source context weighted by the batch mask
      // this will remove padded zeros from the average
      meanContexts.push_back(weighted_average(
          encState->getContext(), encState->getMask(), axis = -3));
    }

    Expr start;
    if(!meanContexts.empty()) {
      // apply single layer network to mean to map into decoder space
      auto mlp = mlp::mlp(graph).push_back(
          mlp::dense(graph)                                          //
          ("prefix", prefix_ + "_ff_state")                          //
          ("dim", opt<int>("dim-rnn"))                               //
          ("activation", mlp::act::tanh)                             //
          ("layer-normalization", opt<bool>("layer-normalization"))  //
          ("nematus-normalization",
           options_->has("original-type")
               && opt<std::string>("original-type") == "nematus")  //
          );
      start = mlp->apply(meanContexts);
    } else {
      int dimBatch = batch->size();
      int dimRnn = opt<int>("dim-rnn");

      start = graph->constant({dimBatch, dimRnn}, init = inits::zeros);
    }

    rnn::States startStates(opt<size_t>("dec-depth"), {start, start});
    return New<DecoderState>(startStates, nullptr, encStates);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    auto embeddings = state->getTargetEmbeddings();

    // dropout target words
    float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
    if(dropoutTrg) {
      int trgWords = embeddings->shape()[-3];
      auto trgWordDrop = graph->dropout(dropoutTrg, {trgWords, 1, 1});
      embeddings = dropout(embeddings, mask = trgWordDrop);
    }

    if(!rnn_)
      rnn_ = constructDecoderRNN(graph, state);

    // apply RNN to embeddings, initialized with encoder context mapped into
    // decoder space
    auto decoderContext = rnn_->transduce(embeddings, state->getStates());

    int dimState = decoderContext->shape()[-1];
    int dimFrantic = 768;
    int decoderDepth = opt<int>("dec-depth") - 1;
    
    auto Wgru = graph->param(prefix_ + "_gru2frantic_W", {dimState, dimFrantic},
                             init = inits::glorot_uniform);
    auto bgru = graph->param(prefix_ + "_gru2frantic_b", {1, dimFrantic},
                             init = inits::zeros);  
    auto frantic = relu(affine(decoderContext, Wgru, bgru));
    auto franticPrev = frantic;
    for(int i = 1; i <= decoderDepth; ++i) {
      auto W = graph->param(prefix_ + "_frantic_W" + std::to_string(i), {dimFrantic, dimFrantic},
                            init = inits::glorot_uniform);
      auto b = graph->param(prefix_ + "_frantic_b" + std::to_string(i), {1, dimFrantic},
                            init = inits::zeros);
      if(i % 2 == 0) {
        frantic = relu(affine(frantic, W, b) + franticPrev);
        franticPrev = frantic;
      }
      else {
        frantic = relu(affine(frantic, W, b));
      }
    }
    
    // retrieve the last state per layer. They are required during translation
    // in order to continue decoding for the next word
    rnn::States decoderStates = rnn_->lastCellStates();
    
    std::vector<Expr> alignedContexts;
    for(int k = 0; k < state->getEncoderStates().size(); ++k) {
      // retrieve all the aligned contexts computed by the attention mechanism
      auto att = rnn_->at(0)
                     ->as<rnn::StackedCell>()
                     ->at(k + 1)
                     ->as<rnn::Attention>();
      alignedContexts.push_back(att->getContext());
    }

    Expr alignedContext;
    if(alignedContexts.size() > 1)
      alignedContext = concatenate(alignedContexts, axis = -1);
    else if(alignedContexts.size() == 1)
      alignedContext = alignedContexts[0];

      
    // construct deep output multi-layer network layer-wise
    auto layer1 = mlp::dense(graph)                                //
        ("prefix", prefix_ + "_ff_logit_l1")                       //
        ("dim", opt<int>("dim-emb"))                               //
        ("activation", mlp::act::tanh)                             //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("nematus-normalization",
         options_->has("original-type")
             && opt<std::string>("original-type") == "nematus");

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    auto layer2 = mlp::dense(graph)           //
        ("prefix", prefix_ + "_ff_logit_l2")  //
        ("dim", dimTrgVoc);

    if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all")) {
      std::string tiedPrefix = prefix_ + "_Wemb";
      if(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src"))
        tiedPrefix = "Wemb";
      layer2.tie_transposed("W", tiedPrefix);
    }

    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto output = mlp::mlp(graph)         //
                      .push_back(layer1)  //
                      .push_back(layer2);

    Expr logits;
    if(alignedContext)
      logits = output->apply(embeddings, frantic, alignedContext);
    else
      logits = output->apply(embeddings, frantic);
      
    // return unormalized(!) probabilities
    return New<DecoderState>(decoderStates, logits, state->getEncoderStates());
  }

  // helper function for guided alignment
  virtual const std::vector<Expr> getAlignments(int i = 0) {
    auto att
        = rnn_->at(0)->as<rnn::StackedCell>()->at(i + 1)->as<rnn::Attention>();
    return att->getAlignments();
  }

  void clear() { rnn_ = nullptr; }
};
}
