#pragma once

#include "marian.h"

namespace marian {

class EncoderFrantic : public EncoderBase {
public:
  Expr applyEncoderRNN(Ptr<ExpressionGraph> graph,
                       Expr embeddings,
                       Expr mask) {
    
    using namespace keywords;
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");

    auto rnnFw = rnn::rnn(graph)                                   //
        ("type", opt<std::string>("enc-cell"))                     //
        ("direction", rnn::dir::forward)                           //
        ("dimInput", embeddings->shape()[-1])                      //
        //("dimState", opt<int>("dim-rnn"))                        //
        ("dimState", 512)                                          // hard-coded!
        ("dropout", dropoutRnn)                                    //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("skip", opt<bool>("skip"));

    auto rnnBw = rnnFw.clone()
        ("direction", rnn::dir::forward);
        
    for(int i = 1; i <= opt<int>("enc-depth"); ++i) {
      rnnFw.push_back(rnn::cell(graph)
                      ("prefix", prefix_ + "_bi_ltr_l" + std::to_string(i)));
      rnnBw.push_back(rnn::cell(graph)
                      ("prefix", prefix_ + "_bi_rtl_l" + std::to_string(i)));
    }
    
    return concatenate({rnnFw->transduce(embeddings, mask),
                        rnnBw->transduce(embeddings, mask)},
                       axis = -1);
  }

  Expr buildSourceEmbeddings(Ptr<ExpressionGraph> graph) {
    // create source embeddings
    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];
    int dimEmb = opt<int>("dim-emb");

    auto embFactory = embedding(graph)  //
        ("dimVocab", dimVoc)            //
        ("dimEmb", dimEmb);

    if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
      embFactory("prefix", "Wemb");
    else
      embFactory("prefix", prefix_ + "_Wemb");

    if(options_->has("embedding-fix-src"))
      embFactory("fixed", opt<bool>("embedding-fix-src"));

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      embFactory                              //
          ("embFile", embFiles[batchIndex_])  //
          ("normalization", opt<bool>("embedding-normalization"));
    }

    return embFactory.construct();
  }

  EncoderFrantic(Ptr<Options> options) : EncoderBase(options) {}

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::CorpusBatch> batch) {
    auto embeddings = buildSourceEmbeddings(graph);

    using namespace keywords;
    // select embeddings that occur in the batch
    Expr batchEmbeddings, batchMask;
    std::tie(batchEmbeddings, batchMask)
        = EncoderBase::lookup(embeddings, batch);

    // apply dropout over source words
    float dropProb = inference_ ? 0 : opt<float>("dropout-src");
    if(dropProb) {
      int srcWords = batchEmbeddings->shape()[-3];
      auto dropMask = graph->dropout(dropProb, {srcWords, 1, 1});
      batchEmbeddings = dropout(batchEmbeddings, mask = dropMask);
    }

    auto context = applyEncoderRNN(graph, batchEmbeddings, batchMask);
    
    auto ffKeys = mlp::mlp(graph)                                //
      ("prefix", prefix_ + "_ff_keys")                           //
      ("dim", 512)                                               // hard-coded!
      ("activation", mlp::act::tanh)                             //
      ("layer-normalization", opt<bool>("layer-normalization"))  //
      .push_back(mlp::dense(graph));
        
    auto ffValues = ffKeys.clone()
      ("prefix", prefix_ + "_ff_values");
  
    auto keys = ffKeys->apply(context);
    auto values = ffValues->apply(context);

    auto state = New<EncoderState>(context, batchMask, batch);
    state->setKeys(keys);
    state->setValues(values);
    
    return state;
  }

  void clear() {}
};

class DecoderFrantic : public DecoderBase {
private:
  Ptr<rnn::RNN> rnn_;

  Ptr<rnn::RNN> constructDecoderRNN(Ptr<ExpressionGraph> graph,
                                    Ptr<DecoderState> state) {
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
    
    auto rnn = rnn::rnn(graph)                                     //
        ("type", opt<std::string>("dec-cell"))                     //
        ("dimInput", opt<int>("dim-emb"))                          //
        ("dimState", 1024)                                         // hard-coded!
        ("dropout", dropoutRnn)                                    //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("skip", opt<bool>("skip"));

    // setting up conditional cell
    auto condCell = rnn::stacked_cell(graph)                 //
        .push_back(rnn::cell(graph)                          //
                   ("prefix", prefix_ + "_cell1"))           //
        .push_back(rnn::attention(graph)                     //
                   ("prefix", prefix_ + "_att")              //
                   .set_state(state->getEncoderStates()[0])) // hard-coded!
        .push_back(rnn::cell(graph)                          //
                   ("prefix", prefix_ + "_cell2"));          //
    
    rnn.push_back(condCell);
    
    return rnn.construct();
  }

public:
  DecoderFrantic(Ptr<Options> options) : DecoderBase(options) {}

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>>& encStates) {
    using namespace keywords;

    int lastIdx = encStates[0]->getContext()->shape()[-3] - 1;
    auto lastContext = marian::step(encStates[0]->getContext(), lastIdx, -3); // hard-coded!
    
    auto mlp = mlp::mlp(graph)
      .push_back(mlp::dense(graph)                                  //
        ("prefix", prefix_ + "_ff_state")                           //
        ("dim", 1024)                                               // hard-coded!
        ("layer-normalization", opt<bool>("layer-normalization"))); //
    
    auto start = mlp->apply(lastContext);
  
    rnn::States startStates(1, {start, start}); // hard-coded!
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
    int decoderDepth = 6; // hard-coded, 6 instead of 8
    
    auto Wgru = graph->param(prefix_ + "_rnn2frantic_W", {dimState, dimFrantic},
                             init = inits::glorot_uniform);
    auto bgru = graph->param(prefix_ + "_rnn2frantic_b", {1, dimFrantic},
                             init = inits::zeros);  
    auto frantic = relu(affine(decoderContext, Wgru, bgru));
    
    auto franticPrev = frantic;
    for(int i = 1; i <= decoderDepth; ++i) {
      auto W = graph->param(prefix_ + "_frantic_W" + std::to_string(i),
                            {dimFrantic, dimFrantic},
                            init = inits::glorot_uniform);
      auto b = graph->param(prefix_ + "_frantic_b" + std::to_string(i),
                            {1, dimFrantic},
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
    
    auto att = rnn_->at(0)->as<rnn::StackedCell>()->at(1)->as<rnn::Attention>(); // hard-coded
    auto alignedContext = att->getContext();
      
    // construct deep output multi-layer network layer-wise
    auto layer1 = mlp::dense(graph)                                //
        ("prefix", prefix_ + "_ff_logit_l1")                       //
        ("dim", opt<int>("dim-emb"))                               //
        ("activation", mlp::act::tanh)                             //
        ("layer-normalization", opt<bool>("layer-normalization"));

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

    auto logits = output->apply(frantic, alignedContext);
      
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
