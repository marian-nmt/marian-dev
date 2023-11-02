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

  SentenceTuple encode(std::vector<std::string>& row, long long int id){
    ABORT_IF(row.size() != vocabs_.size(), "Number of fields doenst match number of vocabs");
    // fill up the sentence tuple with source and/or target sentences
    SentenceTupleImpl tup(id);
    for(size_t i = 0; i < row.size(); ++i) {
      std::string field = row[i];
      Words words = vocabs_[i]->encode(field, /*addEOS=*/true, /*inference=*/inference_);
      if(this->maxLengthCrop_ && words.size() > this->maxLength_) {
        words.resize(maxLength_);
        words.back() = vocabs_.back()->getEosId();  // note: this will not work with class-labels
      }
      ABORT_IF(words.empty(), "No words (not even EOS) found in string??");
      tup.pushBack(words);
    }
    return SentenceTuple(tup);
  }

};
}  // namespace data
}  // namespace marian
