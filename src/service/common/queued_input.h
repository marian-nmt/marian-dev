#pragma once
#include "data/iterator_facade.h"
#include "data/corpus.h"
#include "service/common/queue.h"
#include "service/common/translation_job.h"
#include <stdint.h>
#include <vector>
#include <atomic>
#include <sys/time.h>

namespace marian {
namespace data {

class QueuedInput;

class QueuedInputIterator
  : public IteratorFacade<QueuedInputIterator, SentenceTuple const> {
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
  typedef server::Queue<Ptr<server::Job>> JobQueue;
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
  Sample next(bool starts_batch); // if true, use longer timeout

  QueuedInput::batch_ptr toBatch(const std::vector<Sample>& batchVector) override;
  iterator begin() override { return iterator(*this); }
  iterator end() override { return iterator(); }

  bool push(Ptr<server::Job> job);
  void shuffle() override {}
  void reset() override {}
  void prepare() override {}
  size_t size() const { return job_queue_.size(); }
};

}}  // namespace marian::data
