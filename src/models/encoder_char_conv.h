#pragma once

#include "models/encdec.h"
#include "models/s2s.h"
#include "layers/convolution.h"
#include "layers/highway.h"

namespace marian {

class EncoderCharConv : public EncoderBase {
public:
  EncoderCharConv(Ptr<Options> options)
      : EncoderBase(options) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch) {
    using namespace keywords;

    int dimVoc = opt<int>("dim-vocabs");
    int dimEmb = opt<int>("dim-emb");

    auto embFactory = embedding(graph)
                      ("prefix", prefix_ + "_Wemb")
                      ("dimVocab", dimVoc)
                      ("dimEmb", dimEmb);

    if(options_->has("embedding-fix-src"))
      embFactory
        ("fixed", opt<bool>("embedding-fix-src"));

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      embFactory
        ("embFile", embFiles[batchIndex_])
        ("normalization", opt<bool>("embedding-normalization"));
    }

    auto embeddings = embFactory.construct();

    if (options_->has("normalize-src-vocab")) {
      if (opt<bool>("normalize-src-vocab")) {
        auto gamma = graph->param("src_vocab_gamma",
                              {1, dimVoc},
                              keywords::init = inits::from_value(1.0));
        auto beta = graph->param("src_vocab_beta",
                              {1, dimVoc},
                              keywords::init = inits::zeros);
        embeddings = transpose(layer_norm(transpose(embeddings), gamma, beta));
      }
    }

    Expr batchEmbeddings, batchMask;
    std::tie(batchEmbeddings, batchMask)
      = EncoderBase::lookup(embeddings, batch);

    float dropProb = inference_ ? 0 : opt<float>("dropout-src");
    if(dropProb) {
      int srcWords = batchEmbeddings->shape()[2];
      auto dropMask = graph->dropout(dropProb, {1, 1, srcWords});
      batchEmbeddings = dropout(batchEmbeddings, mask = dropMask);
    }

    std::vector<int> convWidths({1, 2, 3, 4, 5, 6, 7, 8});
    std::vector<int> convSizes({200, 200, 200, 200, 200, 200, 200, 200});
    int stride = 5;

    auto convolution = MultiConvolution("multi_conv", dimEmb, convWidths, convSizes)
      (batchEmbeddings, batchMask);
    auto highway = Highway("highway", 4)(convolution);
    Expr stridedMask = getStridedMask(graph, batch, stride);
    Expr context = applyEncoderRNN(graph, highway, stridedMask);

    return New<EncoderState>(context, stridedMask, batch);
  }

  void clear() { }

protected:
  Expr applyEncoderRNN(Ptr<ExpressionGraph> graph,
                       Expr embeddings, Expr mask) {
    using namespace keywords;
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");

    size_t embDim = embeddings->shape()[1];

    auto rnnFw = rnn::rnn(graph)
                 ("type", opt<std::string>("enc-cell"))
                 ("direction", rnn::dir::forward)
                 ("dimInput", embDim)
                 ("dimState", opt<int>("dim-rnn"))
                 ("dropout", dropoutRnn)
                 ("layer-normalization", opt<bool>("layer-normalization"))
                 ("skip", opt<bool>("skip"));

    auto stacked = rnn::stacked_cell(graph);
    stacked.push_back(rnn::cell(graph)("prefix", prefix_ + "_bi"));
    rnnFw.push_back(stacked);

    auto rnnBw = rnn::rnn(graph)
                 ("type", opt<std::string>("enc-cell"))
                 ("direction", rnn::dir::backward)
                 ("dimInput", embDim)
                 ("dimState", opt<int>("dim-rnn"))
                 ("dropout", dropoutRnn)
                 ("layer-normalization", opt<bool>("layer-normalization"))
                 ("skip", opt<bool>("skip"));

    auto stackedBack = rnn::stacked_cell(graph);
    stackedBack.push_back(rnn::cell(graph)("prefix", prefix_ + "_bi_r"));
    rnnBw.push_back(stackedBack);

    auto context = concatenate({rnnFw->transduce(embeddings, mask),
                                rnnBw->transduce(embeddings, mask)},
                                axis=1);
    return context;
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
    auto stridedMask = graph->constant({dimBatch, 1, dimWords},
                                       keywords::init = inits::from_vector(strided));
    return stridedMask;
  }

};

}
