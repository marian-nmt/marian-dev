#pragma once

#include "models/encdec.h"
#include "models/s2s.h"
#include "layers/convolution.h"
#include "layers/highway.h"

namespace marian {

/***********************************************************
 *
Implementation of a model from ''Fully Character-Level Neural
Machine Translation without Explicit Segmentation'' by Lee et all.

There are four parameters you can set:
 * conv-char-widths (type: std::vector<int>)
   The widths of convolutions. The encoder consists of many
   convolution layers which can have different widths.
   The default value is "{1, 2, 3, 4, 5, 6, 7, 8}". That means
   there are 8 convolution layers with widths from 1 to 8.

 * conv-char-filters (type: std::vector<int>)
   The numbers of convolution fitlers of each width.
   The default value is "{200, 200, 250, 250, 300, 300, 300, 300}".
   That means there are 200 filters of width 1, 200 filters of width 2
   and so on.

  * conv-char-stride (type: int)
    The stride of pooling. The default value is 5

  * conv-char-highway (type: int)
    Number of highway layers before RNN. The default value is 4.

The default values are proposed by the authors of the paper.

***********************************************************/


class EncoderCharConv : public EncoderBase {
public:
  EncoderCharConv(Ptr<Options> options)
      : EncoderBase(options) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch) {
    using namespace keywords;

    int dimVoc = opt<std::vector<int>>("dim-vocabs").front();
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

    std::vector<int> convWidths;
    if (options_->has("conv-char-widths")) {
      convWidths = options_->get<std::vector<int>>("conv-char-widths");
    } else {
      convWidths = {1, 2, 3, 4, 5, 6, 7, 8};
    }

    std::vector<int> convSizes;
    if (options_->has("conv-char-filters")) {
      convSizes = options_->get<std::vector<int>>("conv-char-filters");
    } else {
      convSizes = {200, 200, 250, 250, 300, 300, 300, 300};
    }

    int stride;
    if (options_->has("conv-char-stride")) {
      stride = options_->get<int>("conv-char-stride");
    } else {
      stride = 5;
    }

    int highwayNum;
    if (options_->has("conv-char-highway")) {
      highwayNum = options_->get<int>("conv-char-highway");
    } else {
      highwayNum = 4;
    }

    auto convolution = MultiConvolution("multi_conv", dimEmb, convWidths, convSizes, stride)
      (batchEmbeddings, batchMask);
    auto highway = Highway("highway", highwayNum)(convolution);
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
