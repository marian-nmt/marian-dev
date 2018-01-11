#pragma once

#include "models/s2s.h"
#include "layers/convolution.h"

namespace marian {

class CharS2SEncoder : public EncoderS2S {
public:
  CharS2SEncoder(Ptr<Options> options) : EncoderS2S(options) {}

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::CorpusBatch> batch) {
    using namespace keywords;
    int dimEmb = opt<int>("dim-emb");
    auto convSizes = options_->get<std::vector<int>>("char-conv-filters-num");
    auto convWidths = options_->get<std::vector<int>>("char-conv-filters-widths");
    int stride = opt<int>("char-stride");
    int highwayNum = opt<int>("char-highway");
    int embDim = opt<int>("char-dim-src-emb", opt<int>("dim-emb"));
    int rnnDim = opt<int>("char-dim-src-rnn", opt<int>("dim-rnn"));
    bool useSelu = opt<bool>("char-use-selu");

    auto embeddings = buildSourceEmbeddings(graph, embDim);

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

    auto conved = CharConvPooling(
        prefix_ + "conv_pooling",
        dimEmb,
        convWidths,
        convSizes,
        stride,
        useSelu)
      (batchEmbeddings, batchMask);


    auto inHighway = conved;
    for (int i = 0; i < highwayNum; ++i) {
      int highwayDim = inHighway->shape()[-1];
      std::string tgate_name = prefix_ +"_" + std::to_string(i) + "_highway_d1";
      Expr tgate_W = graph->param(tgate_name + "_W",
                            {highwayDim, highwayDim},
                            keywords::init = inits::glorot_uniform);

      Expr tgate_b = graph->param(tgate_name + "_b",
                            {1, highwayDim},
                            keywords::init = inits::from_value(-3));

      Expr tgate = affine(inHighway, tgate_W, tgate_b);

      std::string hgate_name = prefix_ +"_" + std::to_string(i) + "_highway_d2";
      Expr hgate_W = graph->param(hgate_name + "_W",
                            {highwayDim, highwayDim},
                            keywords::init = inits::glorot_uniform);

      Expr hgate_b = graph->param(hgate_name + "_b",
                            {1, highwayDim},
                            keywords::init = inits::zeros);

      Expr hgate = relu(affine(inHighway, hgate_W, hgate_b));

      inHighway = highway(hgate, inHighway, tgate);
    }

    float charDropout = inference_ ? 0 : opt<float>("char-dropout");
    if(charDropout) {
      if (useSelu) {
        conved = dropout_selu(conved, dropout_prob = charDropout);
      } else {
        conved = dropout(conved, dropout_prob = charDropout);
      }
    }

    Expr stridedMask = getStridedMask(graph, batch, stride);
    Expr context = applyEncoderRNN(
        graph, inHighway, stridedMask, opt<std::string>("enc-type"), rnnDim);

    return New<EncoderState>(context, stridedMask, batch);
  }

protected:
  Expr applyEncoderRNN(Ptr<ExpressionGraph> graph,
                       Expr embeddings,
                       Expr mask,
                       std::string type,
                       int rnnDim) {
    int first, second;
    if(type == "bidirectional" || type == "alternating") {
      // build two separate stacks, concatenate top output
      first = opt<int>("enc-depth");
      second = 0;
    } else {
      // build 1-layer bidirectional stack, concatenate,
      // build n-1 layer unidirectional stack
      first = 1;
      second = opt<int>("enc-depth") - first;
    }

    auto forward = type == "alternating" ? rnn::dir::alternating_forward
                                         : rnn::dir::forward;

    auto backward = type == "alternating" ? rnn::dir::alternating_backward
                                          : rnn::dir::backward;

    using namespace keywords;
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");

    auto rnnFw = rnn::rnn(graph)                                   //
        ("type", opt<std::string>("enc-cell"))                     //
        ("direction", forward)                                     //
        ("dimInput", embeddings->shape()[-1])                      //
        ("dimState", rnnDim)                          //
        ("dropout", dropoutRnn)                                    //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("skip", opt<bool>("skip"));

    for(int i = 1; i <= first; ++i) {
      auto stacked = rnn::stacked_cell(graph);
      for(int j = 1; j <= opt<int>("enc-cell-depth"); ++j) {
        std::string paramPrefix = prefix_ + "_bi";
        if(i > 1)
          paramPrefix += "_l" + std::to_string(i);
        if(i > 1 || j > 1)
          paramPrefix += "_cell" + std::to_string(j);
        bool transition = (j > 1);

        stacked.push_back(rnn::cell(graph)         //
                          ("prefix", paramPrefix)  //
                          ("transition", transition));
      }
      rnnFw.push_back(stacked);
    }

    auto rnnBw = rnn::rnn(graph)                                   //
        ("type", opt<std::string>("enc-cell"))                     //
        ("direction", backward)                                    //
        ("dimInput", embeddings->shape()[-1])                      //
        ("dimState", rnnDim)                          //
        ("dropout", dropoutRnn)                                    //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("skip", opt<bool>("skip"));

    for(int i = 1; i <= first; ++i) {
      auto stacked = rnn::stacked_cell(graph);
      for(int j = 1; j <= opt<int>("enc-cell-depth"); ++j) {
        std::string paramPrefix = prefix_ + "_bi_r";
        if(i > 1)
          paramPrefix += "_l" + std::to_string(i);
        if(i > 1 || j > 1)
          paramPrefix += "_cell" + std::to_string(j);
        bool transition = (j > 1);

        stacked.push_back(rnn::cell(graph)         //
                          ("prefix", paramPrefix)  //
                          ("transition", transition));
      }
      rnnBw.push_back(stacked);
    }

    auto context = concatenate({rnnFw->transduce(embeddings, mask),
                                rnnBw->transduce(embeddings, mask)},
                               axis = -1);

    if(second > 0) {
      // add more layers (unidirectional) by transducing the output of the
      // previous bidirectional RNN through multiple layers

      // construct RNN first
      auto rnnUni = rnn::rnn(graph)                                  //
          ("type", opt<std::string>("enc-cell"))                     //
          ("dimInput", 2 * rnnDim)                      //
          ("dimState", rnnDim)                          //
          ("dropout", dropoutRnn)                                    //
          ("layer-normalization", opt<bool>("layer-normalization"))  //
          ("skip", opt<bool>("skip"));

      for(int i = first + 1; i <= second + first; ++i) {
        auto stacked = rnn::stacked_cell(graph);
        for(int j = 1; j <= opt<int>("enc-cell-depth"); ++j) {
          std::string paramPrefix = prefix_ + "_l" + std::to_string(i) + "_cell"
                                    + std::to_string(j);
          stacked.push_back(rnn::cell(graph)("prefix", paramPrefix));
        }
        rnnUni.push_back(stacked);
      }

      // transduce context to new context
      context = rnnUni->transduce(context);
    }
    return context;
  }

  virtual Expr buildSourceEmbeddings(Ptr<ExpressionGraph> graph, int dimEmb) {
    // create source embeddings
    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

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

  Expr getStridedMask(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch,
                      int stride) {
    auto subBatch = (*batch)[batchIndex_];

    int dimBatch = subBatch->batchSize();

    std::vector<float> strided;
    for (size_t wordIdx = 0; wordIdx < subBatch->mask().size(); wordIdx += stride * dimBatch) {
      for (size_t j = wordIdx; j < wordIdx + dimBatch; ++j) {
        strided.push_back(subBatch->mask()[j]);
      }
    }
    int dimWords = strided.size() / dimBatch;
    auto stridedMask = graph->constant({dimWords, dimBatch, 1},
                                       keywords::init = inits::from_vector(strided));
    return stridedMask;
  }
};

}
