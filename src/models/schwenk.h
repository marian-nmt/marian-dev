#pragma once

#include "models/s2s.h"
#include "layers/convolution.h"

namespace marian {

class DecoderSchwenk : public DecoderBase {
private:
  Ptr<rnn::RNN> rnn_;
  Expr tiedOutputWeights_;

Ptr<rnn::RNN> constructDecoderRNN(Ptr<ExpressionGraph> graph,
                                  Ptr<DecoderState> state) {
  float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
    // std::cerr << "AAAA: " << __LINE__ << std::endl;
  auto rnn = rnn::rnn(graph)
             ("type", opt<std::string>("dec-cell"))
             ("dimInput", opt<std::vector<int>>("dim-emb").back())
             ("dimState", opt<std::vector<int>>("dim-rnn").back())
             ("dropout", dropoutRnn)
             ("layer-normalization", opt<bool>("layer-normalization"))
             ("skip", opt<bool>("skip"));

  size_t decoderLayers = opt<size_t>("dec-depth");
  size_t decoderBaseDepth = opt<size_t>("dec-cell-base-depth");
  size_t decoderHighDepth = opt<size_t>("dec-cell-high-depth");

  // setting up conditional (transitional) cell
  auto baseCell = rnn::stacked_cell(graph);
  for(size_t i = 1; i <= decoderBaseDepth; ++i) {
    auto paramPrefix = prefix_ + "_cell" + std::to_string(i);
    baseCell.push_back(rnn::cell(graph)
                       ("prefix", paramPrefix)
                       ("final", i > 1));
  }
  // Add cell to RNN (first layer)
  rnn.push_back(baseCell);

  // Add more cells to RNN (stacked RNN)
  for(size_t i = 2; i <= decoderLayers; ++i) {
    // deep transition
    auto highCell = rnn::stacked_cell(graph);

    for(size_t j = 1; j <= decoderHighDepth; j++) {
      auto paramPrefix = prefix_ + "_l" + std::to_string(i) + "_cell" + std::to_string(j);
      highCell.push_back(rnn::cell(graph)
                         ("prefix", paramPrefix));
    }

    // @TODO:
    // dec-high-context : none repeat conditional conditional-repeat
    // conditional and conditional-repeat require dec-cell-high-depth > 1

    // Repeat attention output as input for each layer
    //if(opt<std::string>("dec-high-context") == "repeat") {
    //  highCell.add_input(
    //    [](Ptr<rnn::RNN> rnn) {
    //      return rnn->at(0)->as<rnn::StackedCell>()
    //                ->at(1)->as<rnn::Attention>()
    //                ->getContext();
    //    }
    //  );
    //  highCell("dimInputExtra", 2 * opt<int>("dim-rnn"));
    //}

    // Add cell to RNN (more layers)
    rnn.push_back(highCell);
  }

  return rnn.construct();
}

public:
  template <class... Args>
  DecoderSchwenk(Ptr<Config> options, Args... args)
      : DecoderBase(options, args...)
  {}

  Expr convert2NCHW(Expr x) {
    std::vector<size_t> newIndeces;
    int batchDim = x->shape()[0];
    int sentenceDim = x->shape()[2];

    for (int b = 0; b < batchDim; ++b) {
      for (int t = 0; t < sentenceDim; ++t) {
        newIndeces.push_back((t * batchDim) + b);
      }
    }

    Shape shape({batchDim, 1, sentenceDim, x->shape()[1]});
    return  reshape(rows(x, newIndeces), shape);
  }

  virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
    using namespace keywords;

    // average the source context weighted by the batch mask
    // this will remove padded zeros from the average
    auto meanContext = max_pooling2(convert2NCHW(encState->getContext()),
                                    convert2NCHW(encState->getMask()),
                                    encState->getContext()->shape()[2],
                                    false);

    auto graph = meanContext->graph();

    // apply single layer network to mean to map into decoder space
    // std::cerr << "AAAA: " << __LINE__ << std::endl;
    auto mlp = mlp::mlp(graph)
               .push_back(mlp::dense(graph)
                          ("prefix", prefix_ + "_ff_state")
                          ("dim", opt<std::vector<int>>("dim-rnn").back())
                          ("activation", mlp::act::tanh)
                          ("layer-normalization", opt<bool>("layer-normalization")));
    auto start = mlp->apply(meanContext);

    rnn::States startStates(opt<size_t>("dec-depth"), {start, start});
    return New<DecoderState>(startStates, nullptr, encState);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    auto embeddings = state->getTargetEmbeddings();

    // dropout target words
    float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
    if(dropoutTrg) {
      int trgWords = embeddings->shape()[2];
      auto trgWordDrop = graph->dropout(dropoutTrg, {1, 1, trgWords});
      embeddings = dropout(embeddings, mask = trgWordDrop);
    }

    if(!rnn_) {
      rnn_ = constructDecoderRNN(graph, state);
    }

    // apply RNN to embeddings, initialized with encoder context mapped into
    // decoder space
    auto decoderContext = rnn_->transduce(embeddings, state->getStates());

    // retrieve the last state per layer. They are required during translation
    // in order to continue decoding for the next word
    rnn::States decoderStates = rnn_->lastCellStates();

    // construct deep output multi-layer network layer-wise
    auto layer1 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l1")
                  ("dim", opt<std::vector<int>>("dim-emb").back())
                  ("activation", mlp::act::tanh)
                  ("layer-normalization", opt<bool>("layer-normalization"));

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs").back();

    auto layer2 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l2")
                  ("dim", dimTrgVoc);
    if(opt<bool>("tied-embeddings"))
      layer2.tie_transposed("W", prefix_ + "_Wemb");

    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto logits = mlp::mlp(graph)
                  .push_back(layer1)
                  .push_back(layer2)
                  ->apply(embeddings, decoderContext);

    // return unormalized(!) probabilities
    return New<DecoderState>(decoderStates, logits, state->getEncoderState());
  }

  // helper function for guided alignment
  virtual const std::vector<Expr> getAlignments() {
    auto att = rnn_->at(0)->as<rnn::StackedCell>()->at(1)->as<rnn::Attention>();
    return att->getAlignments();
  }
};

typedef EncoderDecoder<EncoderS2S, DecoderSchwenk> Schwenk;


}
