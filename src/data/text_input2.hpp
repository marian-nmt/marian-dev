#pragma once

#include <functional>
#include "data/iterator_facade.h"
#include "data/corpus.h"

// Original TextInput and TextInputIterator has cyclic dependency: TextInput<->TextIterator, makes it complicated to subclass
// Here I rewrite them to with dependence in one dir: TextInput2 -> TextIterator2 -> IteratorFacade

namespace marian::data {

  using RowEncoder = std::function<SentenceTuple(const std::vector<std::string>&, long long int)>;

  class TextIterator2 : public IteratorFacade<TextIterator2, SentenceTuple const> {

  protected:
    SentenceTuple next_;
    RowEncoder encoder_  {nullptr};

    bool ended_{ true };
    long long int pos_{ -1 };


  public:
    TextIterator2() {}

    TextIterator2(RowEncoder encoder) :
      encoder_(encoder)
    {}

    bool equal(TextIterator2 const& other) const override {
      // two ended iterators are equal or two iterators at the same position
      return (this->ended_ && other.ended_) || false;
    }

    virtual const SentenceTuple& dereference() const override {
      return next_;
    };

    virtual void increment() override const = 0;

  };


  template <class Iterator>
  class TextInput2 : public DatasetBase<SentenceTuple, Iterator, CorpusBatch> {

  protected:
    Ptr<Iterator> iterator_;
    std::vector<Ptr<Vocab>> vocabs_;
    size_t maxLength_{ 0 };
    bool maxLengthCrop_{ false };

  public:
    // we have to redefine them here, because they are not auto inherited
    // https://stackoverflow.com/a/1643190/1506477
    typedef CorpusBatch batch_type;
    typedef Ptr<CorpusBatch> batch_ptr;

    TextInput2(Ptr<Iterator> iterator, std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options)
      : DatasetBase<SentenceTuple, Iterator, CorpusBatch>({}, options),
      iterator_{ iterator },
      vocabs_{ vocabs },
      maxLength_ {options->get<size_t>("max-length")},
      maxLengthCrop_ {options->get<bool>("max-length-crop")}
    {}


    virtual ~TextInput2() {}

    void shuffle() override {}
    void reset() override {}

    Iterator begin() override { return *iterator_; }
    Iterator end() override { return Iterator(); }


    SentenceTuple next() override {
      return iterator_->dereference();
    }

    virtual SentenceTuple encode(const std::vector<std::string>& row, long long int id) {
      // get index of the current sentence
      //size_t curId = pos_++;

      // fill up the sentence tuple with source and/or target sentences
      SentenceTupleImpl tup(id);

      for (size_t i = 0; i < row.size(); ++i) {
        std::string line = row[i];
        Words words = vocabs_[i]->encode(line, /*addEOS=*/true, /*inference=*/this->inference_);
        if (this->maxLengthCrop_ && words.size() > this->maxLength_) {
          words.resize(maxLength_);
          words.back() = vocabs_.back()->getEosId();  // note: this will not work with class-labels
        }

        ABORT_IF(words.empty(), "No words (not even EOS) found in string??");
        ABORT_IF(tup.size() != i, "Previous tuple elements are missing.");
        tup.pushBack(words);
      }

      if (tup.size() == vocabs_.size()) // check if each input file provided an example
        return SentenceTuple(tup);
      else if (tup.size() == 0) // if no file provided examples we are done
        return SentenceTupleImpl(); // return an empty tuple if above test does not pass();
      else // neither all nor none => we have at least on missing entry
        ABORT("There are missing entries in the text tuples.");
    }

    // TODO: There are half dozen functions called toBatch(), which are very
    // similar. Factor them.
    batch_ptr toBatch(const std::vector<SentenceTuple>& batchVector) override {
      size_t batchSize = batchVector.size();

      std::vector<size_t> sentenceIds;

      std::vector<int> maxDims;
      for (auto& ex : batchVector) {
        if (maxDims.size() < ex.size())
          maxDims.resize(ex.size(), 0);
        for (size_t i = 0; i < ex.size(); ++i) {
          if (ex[i].size() > (size_t)maxDims[i])
            maxDims[i] = (int)ex[i].size();
        }
        sentenceIds.push_back(ex.getId());
      }

      std::vector<Ptr<SubBatch>> subBatches;
      for (size_t j = 0; j < maxDims.size(); ++j) {
        subBatches.emplace_back(New<SubBatch>(batchSize, maxDims[j], vocabs_[j]));
      }

      std::vector<size_t> words(maxDims.size(), 0);
      for (size_t i = 0; i < batchSize; ++i) {
        for (size_t j = 0; j < maxDims.size(); ++j) {
          for (size_t k = 0; k < batchVector[i][j].size(); ++k) {
            subBatches[j]->data()[k * batchSize + i] = batchVector[i][j][k];
            subBatches[j]->mask()[k * batchSize + i] = 1.f;
            words[j]++;
          }
        }
      }

      for (size_t j = 0; j < maxDims.size(); ++j)
        subBatches[j]->setWords(words[j]);

      auto batch = batch_ptr(new batch_type(subBatches));
      batch->setSentenceIds(sentenceIds);

      return batch;
    }

    void prepare() override {}
  };


}  // namespace marian::data

