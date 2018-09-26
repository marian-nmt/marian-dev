#pragma once

#include "marian.h"
#include "states.h"

#include "data/shortlist.h"
#include "layers/generic.h"

namespace marian {

class DecoderBase {
protected:
  Ptr<Options> options_;
  std::string prefix_{"decoder"};
  bool inference_{false};
  size_t batchIndex_{1};

  Ptr<data::Shortlist> shortlist_;
  std::vector<size_t> vmap_;

public:
  DecoderBase(Ptr<Options> options)
      : options_(options),
        prefix_(options->get<std::string>("prefix", "decoder")),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", 1)) {
          
    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];
    if(!options_->get<std::string>("vmap", "").empty()) {
      vmap_.resize(dimVoc);
      for(size_t i = 0; i < vmap_.size(); ++i)
        vmap_[i] = i;

      InputFileStream vmapFile(options_->get<std::string>("vmap"));
      size_t from, to;
      while(vmapFile >> from >> to) {
        vmap_[from] = to;
        std::cerr << from << " -> " << to << std::endl;
      }
    }

  }

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph>,
                                       Ptr<data::CorpusBatch> batch,
                                       std::vector<Ptr<EncoderState>>&)
      = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph>, Ptr<DecoderState>) = 0;
  
  Expr vmap(Expr chosenEmbeddings, Expr srcEmbeddings, const std::vector<size_t>& indices) const {
    if(!vmap_.empty()) {
      std::vector<size_t> vmapped(indices.size());
      for(size_t i = 0; i < vmapped.size(); ++i)
        vmapped[i] = vmap_[indices[i]];

      auto vmapEmbeddings = rows(srcEmbeddings, vmapped);
      chosenEmbeddings = (chosenEmbeddings + vmapEmbeddings) / 2.f;
    }
    return chosenEmbeddings;
  }

  virtual void embeddingsFromBatch(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state,
                                   Ptr<data::CorpusBatch> batch) {
    using namespace keywords;

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];
    int dimEmb = opt<int>("dim-emb");

    auto yEmbFactory = embedding(graph)  //
        ("dimVocab", dimVoc)             //
        ("dimEmb", dimEmb);

    if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
      yEmbFactory("prefix", "Wemb");
    else
      yEmbFactory("prefix", prefix_ + "_Wemb");

    if(options_->has("embedding-fix-trg"))
      yEmbFactory("fixed", opt<bool>("embedding-fix-trg"));

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      yEmbFactory("embFile", embFiles[batchIndex_])  //
          ("normalization", opt<bool>("embedding-normalization"));
    }

    auto yEmb = yEmbFactory.construct();

    auto subBatch = (*batch)[batchIndex_];
    int dimBatch = (int)subBatch->batchSize();
    int dimWords = (int)subBatch->batchWidth();

    auto chosenEmbeddings = rows(yEmb, subBatch->data());
    chosenEmbeddings = vmap(chosenEmbeddings, yEmb, subBatch->data());

    auto y
        = reshape(chosenEmbeddings, {dimWords, dimBatch, opt<int>("dim-emb")});

    auto yMask = graph->constant({dimWords, dimBatch, 1},
                                 inits::from_vector(subBatch->mask()));

    Expr yData;
    if(shortlist_) {
      yData = graph->constant({(int)shortlist_->mappedIndices().size(), 1},
                              inits::from_vector(shortlist_->mappedIndices()));
    } else {
      yData = graph->constant({(int)subBatch->data().size(), 1},
                              inits::from_vector(subBatch->data()));
    }

    auto yShifted = shift(y, {1, 0, 0});

    state->setTargetEmbeddings(yShifted);
    state->setTargetMask(yMask);
    state->setTargetIndices(yData);
  }

  virtual void embeddingsFromPrediction(Ptr<ExpressionGraph> graph,
                                        Ptr<DecoderState> state,
                                        const std::vector<size_t>& embIdx,
                                        int dimBatch,
                                        int dimBeam) {
    using namespace keywords;

    int dimTrgEmb = opt<int>("dim-emb");
    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    // embeddings are loaded from model during translation, no fixing required
    auto yEmbFactory = embedding(graph)  //
        ("dimVocab", dimTrgVoc)          //
        ("dimEmb", dimTrgEmb);

    if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
      yEmbFactory("prefix", "Wemb");
    else
      yEmbFactory("prefix", prefix_ + "_Wemb");

    auto yEmb = yEmbFactory.construct();

    Expr selectedEmbs;
    if(embIdx.empty()) {
      selectedEmbs = graph->constant({1, 1, dimBatch, dimTrgEmb}, inits::zeros);
    } else {
      selectedEmbs = rows(yEmb, embIdx);
      selectedEmbs = vmap(selectedEmbs, yEmb, embIdx);
      selectedEmbs = reshape(selectedEmbs, {dimBeam, 1, dimBatch, dimTrgEmb});
    }
    state->setTargetEmbeddings(selectedEmbs);
  }

  virtual const std::vector<Expr> getAlignments(int /*i*/ = 0) { return {}; };

  virtual Ptr<data::Shortlist> getShortlist() { return shortlist_; }
  virtual void setShortlist(Ptr<data::Shortlist> shortlist) {
    shortlist_ = shortlist;
  }

  template <typename T>
  T opt(const std::string& key) const {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string& key, const T& def) {
    return options_->get<T>(key, def);
  }

  virtual void clear() = 0;
};

}  // namespace marian
