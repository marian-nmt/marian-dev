// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once

// "Corpus" that gets its content from a queue.

#include "data/iterator_facade.h"
#include "data/corpus.h"
#include "server/queue.h"
#include <stdint.h>
#include <vector>
#include <atomic>

namespace marian {
namespace data {

class QueuedInput;

class QueuedInputIterator : public IteratorFacade<QueuedInputIterator, SentenceTuple const> {
public:
  QueuedInputIterator();
  explicit QueuedInputIterator(QueuedInput& corpus);

private:
  void increment() override;
  bool equal(QueuedInputIterator const& other) const override;
  SentenceTuple const& dereference() const override;
  QueuedInput* corpus_;
  long long int pos_;
  SentenceTuple tup_;
};

// Each TranslationJob represents a single sentence tuple!
class QueuedInput
  : public DatasetBase<SentenceTuple, QueuedInputIterator, CorpusBatch>
{
public:
  typedef std::pair<uint64_t, std::vector<std::string>> TranslationJob;
  typedef server::Queue<Ptr<TranslationJob>> JobQueue;
  typedef QueuedInputIterator Iterator;

private:
  std::vector<Ptr<Vocab const>> vocabs_;
  JobQueue job_queue_;
  int timeout_; // queue pop timeout (in milliseconds)
  std::atomic_ullong job_ctr_{0};
public:
  typedef SentenceTuple Sample;

  QueuedInput(std::vector<Ptr<Vocab const>> const& vocabs,
              Ptr<Options> options);

  Sample next() override { return next(false); }
  Sample next(bool starts_batch);
  // starts_batch == true => use longer timeout for first in batch

  QueuedInput::batch_ptr toBatch(const std::vector<Sample>& batchVector) override;
  iterator begin() override { return iterator(*this); }
  iterator end() override { return iterator(); }

  // push translation job, return job ID
  // Note that the vector accommodates factored input,
  // not multiple sentences at once.
  uint64_t push(std::vector<std::string> const& src);

  void shuffle() override {}
  void reset() override {}
  void prepare() override {}
  size_t size() const { return job_queue_.size(); }
};
}  // namespace data
}  // namespace marian
