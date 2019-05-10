// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/shortlist.h"
#include "data/text_input.h"

#include "3rd_party/threadpool.h"
#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/output_printer.h"

#include "models/model_task.h"
#include "translator/scorers.h"

#include "translation_worker.h"
#include "logging.h"
#include "queue.h"
#include "queued_input.h"

namespace marian {


#include <string>

extern Logger logger;

namespace marian {

template<class Search>
class TranslationService;

template<class Search, typename ResponseHandler>
class TranslationWorker
{
private:
  DeviceId device_;
  QueuedInput& job_queue_;
  ResponseHandler& callback_;
  Ptr<Options> options_;
  Ptr<ExpressionGraph> graph_;
  std::vector<Ptr<Scorer>> scorers_;

public:
  TranslationWorker(DeviceId const device,
                    QueuedInput& job_queue,
                    ResponseHandler& callback,
                    Options options)
    : device_(device)
    , job_queue_(job_queue)
    , callback_(callback)
    , options_(options)
  { }

  void init() {
    bool optimize = service.options_->get<bool>("optimize");
    graph_ = New<ExpressionGraph>(true, optimize);
    graph_->setDevice(device);
    graph_->getBackend()->setClip(service.options_->get<float>("clip-gemm"));
    graph_->reserveWorkspackeMB(service.options_->get<size_t>("workspace"));

    // Do we need a separate set of scorers for each worker, or can they
    // be shared? Do we ever need to alter their state?
    scorers_ = createScorers(service_.options_);
    for (auto s: scorers_) s->init(graph_);
  }

  void run()
  {
    while(True) {
      data::BatchGenerator<data::QueuedInput> bg(job_queue, options_);

    auto tOptions = New<Options>();
    tOptions->merge(options_);
    // are these ever changed, or can we move this to service_ and do it just once?

    bg.prepare(false);
    size_t i = 0;
    for (auto b: bg) // there should be only one batch to begin with ...
      {
        auto search = New<Search>(tOptions, scorers_,
                                  service.trgVocab_->getEosId(),
                                  service.trgVocab_->getUnkId());
        auto histories = search->search(graph_, b);
        bool reverse = service_.options_->get<bool>("right-left");
        for (auto h: histories)
          {
            auto result = history->Top();
            auto words = std::get<0>(result);
            if(reverse) std::reverse(words.begin(), words.end());
            batch_[i++]->translation.set_value(service_.trgVocab_->decode(words));
          }
      }
    batch_.clear();
    items_in_batch = 0;
  }

  void run(JobQueue& Q)
  {
    std::chrono::milliseconds timeout(100);
    // timeout: max wait on Q for more input, after that, translate
    // partial batch instead of waiting for more input to fill it up
    // to do: make this configurable

    JobQueue::STATUS_CODE status;
    Ptr<TranslationJob> job;
    // single line of input to be processed, plus a promise for the result

    while (true)
      {
        status = Q.pop(job);
        while (status == JobQueue::SUCCESS && batch_.size() < batch_.capacity())
          {
            batch_.push_back(job);
            status = Q.pop(job, timeout);
          }
        if (batch_.size()) process_batch();
        if (status == JobQueue::CLOSED)
          break;
      }
  } // end of function run
};

template<class Search>
class TranslationService
{
  Ptr<Config> options_;
  Ptr<Vocab> trgVocab_;
  std::vector<Ptr<Vocab> > srcVocabs_;
  std::vector<DeviceId> devices_;
  friend TranslationWorker<Search>;
public:
  TranslationService(Ptr<config> options)
    : options_(options)
  {
    options->set("inference", true);
    auto vpaths = options_->get<std::vector<std::string>>("vocabs");
    auto vdims  = options_->get<std::vector<int>>("dim-vocabs");

    auto topt = New<Options>();
    topt->merge(options_);

    assert(vpaths.size());
    srcVocabs.resize(vpaths.size()-1);
    for(size_t i = 0; i < srcVocabs.size(); ++i) {
      (srcVocabs_[i] = New<Vocab>(topt, i))->load(vpaths[i],vdims[i]);
    }
    (trgVocab_ = New<Vocab>(topt, vpaths.size() - 1))->load(vpaths.back());

    // to do: add workers

};
} // end of namespace marian
