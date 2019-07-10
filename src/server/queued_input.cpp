#include "data/text_input.h"
#include "common/utils.h"
#include "queued_input.h"

namespace marian {
namespace data {

QueuedInputIterator::QueuedInputIterator() : pos_(-1), tup_(0) {}
QueuedInputIterator::QueuedInputIterator(QueuedInput& corpus) : corpus_(&corpus), pos_(0), tup_(corpus_->next(true)) {}

void QueuedInputIterator::increment() {
  tup_ = corpus_->next();
  pos_++;
}

bool QueuedInputIterator::equal(QueuedInputIterator const& other) const {
  return this->pos_ == other.pos_ || (this->tup_.empty() && other.tup_.empty());
}

const SentenceTuple& QueuedInputIterator::dereference() const {
  return tup_;
}

QueuedInput::
QueuedInput(std::vector<Ptr<Vocab const>> const& vocabs, Ptr<Options> options)
  : DatasetBase(options), vocabs_(vocabs),
    timeout_(options ? options->get<int>("queue-timeout",100) : 100)
{ }

// QueuedInput is mainly used for inference in the server mode, not
// for training, so skipping too long or ill-formed inputs is not
// necessary here
SentenceTuple QueuedInput::next(bool starts_batch) {
  // Use a longer timeout when starting a batch, because if there's no
  // input in that case, no one is waiting for a reponse, so response
  // latency isn't an issue.
  // LOG(info, "timeout is {}", starts_batch ? 1000 : timeout_);
  auto timeout = std::chrono::milliseconds(starts_batch ? 1000 : timeout_);

  Ptr<TranslationJob> job;
  JobQueue::STATUS_CODE success;
  do {
    success = job_queue_.pop(job,timeout);
    // LOG(info, "job queue status code: {} [{} batch]", success, (starts_batch ? "new" : "old"));
    if (success == JobQueue::SUCCESS) {
      // fill up the sentence tuple with source and/or target sentences
      SentenceTuple tup(job->first); // job ID should be unique
      std::vector<std::string> const& snt = job->second;
      LOG(info, "QueuedInput is shipping job {}", job->first);
      for(size_t i = 0; i < snt.size(); ++i) {
        std::istringstream buf(snt[i]);
        io::InputFileStream dummy(buf);
        std::string line;
        if(io::getline(dummy, line)) {
          // second, third parameter below: addEOS, inference mode==true?
          Words words = vocabs_[i]->encode(line,true,inference_);
          if(words.empty())
            words.push_back(DEFAULT_EOS_ID);
          tup.push_back(words);
        }
      }
      // check if each input file provided an example
      // if(tup.size() == vocabs_.size())
      return tup;
    }
  } while(starts_batch and success == JobQueue::EMPTY);
  return SentenceTuple(0);
}

// TODO: There are half dozen functions called toBatch(), which are very
// similar. Factor them.
// Why is this even a member function?
QueuedInput::batch_ptr
QueuedInput::
toBatch(const std::vector<Sample>& batchVector) {
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

  auto batch = QueuedInput::batch_ptr(new batch_type(subBatches));
  batch->setSentenceIds(sentenceIds);

  return batch;
}

uint64_t
QueuedInput::
push(std::vector<std::string> const& src) {
  // push a new item for translation
  uint64_t jid = ++job_ctr_;
  std::chrono::milliseconds timeout(5000);
  auto job = New<TranslationJob>(jid,src);
  auto status = job_queue_.push(job,timeout);
  if (status == JobQueue::SUCCESS) return jid;
  return 0; // failed
}

}  // namespace data
}  // namespace marian
