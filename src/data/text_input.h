#pragma once

#include "data/iterator_facade.h"
#include "data/corpus.h"

namespace marian {
namespace data {

class TextInput;

class TextIterator : public IteratorFacade<TextIterator, SentenceTuple const> {
public:
  TextIterator();
  explicit TextIterator(TextInput& corpus);

private:
  void increment() override;

  bool equal(TextIterator const& other) const override;

  const SentenceTuple& dereference() const override;

  TextInput* corpus_;

  long long int pos_;
  SentenceTuple tup_;
};

class TextInput : public DatasetBase<SentenceTuple, TextIterator, CorpusBatch> {
protected:
  std::vector<UPtr<std::istringstream>> files_;
  std::vector<Ptr<Vocab>> vocabs_;

  size_t pos_{0};

  size_t maxLength_{0};
  bool maxLengthCrop_{false};
  bool rightLeft_{false};

  // copied from corpus.h - TODO: refactor or unify code between Corpus and TextInput
  bool prependZero_{false};
  bool joinFields_{false};      // if true when given a TSV file or multiple inputs, join them together into a single sentence tuple,
                                // the already present </s> separator will demark the fields (mostly used for BLEURT and COMET-KIWI)
  bool insertSeparator_{false}; // when joining fields with joinFields_, additionally use this separator (mostly used for COMET-KIWI)

public:
  TextInput(std::vector<std::string> inputs, std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options);
  virtual ~TextInput() {}

  SentenceTuple next() override;

  void shuffle() override {}
  void reset() override {}

  iterator begin() override { return iterator(*this); }
  iterator end() override { return iterator(); }

  // TODO: There are half dozen functions called toBatch(), which are very
  // similar. Factor them.
  batch_ptr toBatch(const std::vector<SentenceTuple>& batchVector) override {
    size_t batchSize = batchVector.size();

    std::vector<size_t> sentenceIds;

    std::vector<int> maxDims;
    for(auto& ex : batchVector) {
      if(maxDims.size() < ex.size())
        maxDims.resize(ex.size(), 0);
      for(size_t i = 0; i < ex.size(); ++i) {
        if(ex[i].size() > (size_t)maxDims[i])
          maxDims[i] = (int)ex[i].size();
      }
      sentenceIds.push_back(ex.getId());
    }

    std::vector<Ptr<SubBatch>> subBatches;
    for(size_t j = 0; j < maxDims.size(); ++j) {
      subBatches.emplace_back(New<SubBatch>(batchSize, maxDims[j], vocabs_[j]));
    }

    std::vector<size_t> words(maxDims.size(), 0);
    for(size_t i = 0; i < batchSize; ++i) {
      for(size_t j = 0; j < maxDims.size(); ++j) {
        for(size_t k = 0; k < batchVector[i][j].size(); ++k) {
          subBatches[j]->data()[k * batchSize + i] = batchVector[i][j][k];
          subBatches[j]->mask()[k * batchSize + i] = 1.f;
          words[j]++;
        }
      }
    }

    for(size_t j = 0; j < maxDims.size(); ++j)
      subBatches[j]->setWords(words[j]);

    auto batch = batch_ptr(new batch_type(subBatches));
    batch->setSentenceIds(sentenceIds);

    return batch;
  }

  void prepare() override {}

  SentenceTuple encode(std::vector<std::string>& row, size_t id) {
    ABORT_IF(row.size() != vocabs_.size(), "Number of fields does not match number of vocabs");
    // fill up the sentence tuple with source and/or target sentences
    SentenceTupleImpl tup(id);

    // copied and adapted from corpus.cpp - @TODO: refactor or unify code between Corpus and TextInput
    for(size_t batchIndex = 0; batchIndex < row.size(); ++batchIndex) {
      std::string& field = row[batchIndex];
      Words words = vocabs_[batchIndex]->encode(field, /*addEOS =*/true, inference_);
      ABORT_IF(words.empty(), "Empty input sequences are presently untested");

      // This handles adding starts symbols for COMET (<s>) and BERT/BLEURT ([CLS])
      bool prepend = prependZero_ && (!joinFields_ || (joinFields_ && batchIndex == 0));
      if(prepend)
        words.insert(words.begin(), Word::fromWordIndex(0));

      bool prependSep = insertSeparator_ && joinFields_ && batchIndex > 0;
      if(prependSep)
        words.insert(words.begin(), vocabs_[batchIndex]->getSepId());

      // if fields are joined and the current sentence is not the first one, we need to make sure that
      // the current sentence is not longer than the maximum length minus the length of the previous sentence
      // (minus 1 for the separator <eos> token or 2 if we also add a separator <sep> token)
      size_t localMaxLength = maxLength_;
      if(joinFields_ && !tup.empty())
        localMaxLength = std::max(1 + (int)prependSep, (int)maxLength_ - (int)tup.back().size());

      // if the current sentence is longer than the maximum length, we need to crop it
      if(maxLengthCrop_ && words.size() > localMaxLength) {
        words.resize(localMaxLength);
        words.back() = vocabs_[batchIndex]->getEosId();
      }

      // if true, the words are reversed
      if(rightLeft_)
        std::reverse(words.begin(), words.end() - 1);

      // if true, the numeric indices get joined with the previous sentence, <eos> acts as a separator here
      if(joinFields_) {
        size_t currLength = tup.empty() ? 0 : tup.back().size();
        // if the current sentence would exceed the maximum length we don't add any more fields
        if(currLength + words.size() < maxLength_)
          tup.appendToBack(words);
      } else {
        tup.pushBack(words);
      }
    }
    return SentenceTuple(tup);
  }

};
}  // namespace data
}  // namespace marian
